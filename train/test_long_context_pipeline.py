"""長系列拡張および Late Chunking / InSeNT パイプラインの統合テスト。

YaRN による局所窓保護 RoPE スケーリング、Late Chunking 事後プーリング、
並列条項判定、Top-k Cross-Attention、InSeNT 対照損失、
および LongContextTrainer によるマルチタスクステップ損失計算の統合動作を検証する。
"""

import torch
from transformers import ModernBertConfig, ModernBertModel

from models.decision_head import JevDecisionModel
from models.long_context import (
    LateChunkingPooling,
    ParallelClauseNoulScanner,
    TopKChunkCrossAttention,
    apply_yarn_to_modernbert,
)
from training.long_context_config import LongContextConfig
from training.long_context_trainer import LongContextTrainer


def test_long_context_pipeline_e2e() -> None:
    """YaRN、Late Chunking、および InSeNT を統合した End-to-End パイプラインを検証する。"""
    hidden_size = 64
    num_heads = 4
    num_layers = 3
    num_options = 4

    config = ModernBertConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_hidden_layers=num_layers,
        intermediate_size=128,
        max_position_embeddings=8192,
        local_attention=128,
        layer_types=["full_attention", "sliding_attention", "sliding_attention"],
        rope_parameters={
            "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
    )

    backbone = ModernBertModel(config)
    model = JevDecisionModel(backbone=backbone, mlp_hidden_size=64)

    sliding_freq = backbone.rotary_emb.sliding_attention_inv_freq
    assert isinstance(sliding_freq, torch.Tensor)
    orig_sliding_freq = sliding_freq.clone()
    apply_yarn_to_modernbert(backbone, target_max_position=16384)

    # 局所層が不変であることを再確認
    assert torch.equal(sliding_freq, orig_sliding_freq)

    # 2. LongContextTrainer の初期化
    long_config = LongContextConfig(
        use_yarn=True,
        original_max_position=8192,
        target_max_position=16384,
        pooling_strategy="mean",
        top_k_chunks=3,
        use_insent=True,
        lambda_seq=0.2,
        lambda_insent=0.15,
        insent_temperature=0.05,
    )
    trainer = LongContextTrainer(config=long_config, model=model)

    # 3. ダミーバッチ入力の作成
    batch_size = 2
    seq_len = 80
    input_ids = torch.randint(10, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    op_indices = torch.tensor([[10, 20, 30, 40], [15, 25, 35, 45]])
    op_mask = torch.ones((batch_size, num_options), dtype=torch.bool)
    labels = torch.tensor([1, 2], dtype=torch.long)

    # 各サンプルの条項スパン (各サンプル 3 チャンク)
    chunk_spans = [
        [(0, 25), (26, 50), (51, 79)],
        [(0, 20), (21, 45), (46, 79)],
    ]
    target_chunk_indices = torch.tensor([0, 1], dtype=torch.long)

    # 4. ステップ損失計算の実行
    total_loss, metrics = trainer.compute_step_loss(
        input_ids=input_ids,
        attention_mask=attention_mask,
        op_indices=op_indices,
        op_mask=op_mask,
        labels=labels,
        chunk_spans=chunk_spans,
        target_chunk_indices=target_chunk_indices,
    )

    # 検証
    assert total_loss.item() > 0.0
    assert torch.isfinite(total_loss)
    assert "total_loss" in metrics
    assert "loss_choice" in metrics
    assert "loss_insent" in metrics
    assert "loss_seq" in metrics
    assert "loss_batch" in metrics
    assert metrics["loss_insent"] > 0.0

    # 勾配逆伝播の検証
    total_loss.backward()
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_topk_and_parallel_clause_pipeline() -> None:
    """集約チャンク表現に対する Top-k Cross-Attention および並列条項 Noul スキャンを検証する。"""
    hidden_size = 32
    batch_size = 2
    seq_len = 100
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # 1. Late Chunking Pooling
    chunk_spans = [
        [(0, 19), (20, 39), (40, 59), (60, 79), (80, 99)],
        [(0, 29), (30, 59), (60, 89)],
    ]
    pooler = LateChunkingPooling(hidden_size=hidden_size, pooling_strategy="attention")
    v_chunks, chunk_mask = pooler(hidden_states, chunk_spans=chunk_spans)

    assert v_chunks.shape == (batch_size, 5, hidden_size)
    assert chunk_mask.shape == (batch_size, 5)

    # 2. 並列条項 Noul スキャン (一括条項判定)
    scanner = ParallelClauseNoulScanner(hidden_size=hidden_size)
    logits, probs = scanner(v_chunks, chunk_mask=chunk_mask)

    assert logits.shape == (batch_size, 5)
    assert probs.shape == (batch_size, 5)
    assert (probs[1, 3:] == 0.0).all()  # サンプル1の無効チャンクは 0.0

    # 3. Top-k Cross-Attention (上位 2 チャンクを選択して集約)
    topk_attn = TopKChunkCrossAttention(
        hidden_size=hidden_size,
        top_k=2,
        num_heads=2,
    )
    query_states = torch.randn(batch_size, 10, hidden_size)
    context_query = topk_attn(
        query_states=query_states,
        chunk_states=v_chunks,
        chunk_mask=chunk_mask,
    )

    assert context_query.shape == (batch_size, 10, hidden_size)
    assert torch.isfinite(context_query).all()
