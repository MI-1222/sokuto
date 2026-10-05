"""長系列拡張および Late Chunking 事後プーリング層の単体テスト。

YaRN による選択的 RoPE スケーリング (局所窓保護)、Late Chunking プーリング (Mean / Attention)、
Top-k Cross-Attention、および並列条項 Noul スキャナーの動作・数理的整合性を検証する。
"""

import math

import torch
from transformers import ModernBertConfig, ModernBertModel

from models.long_context import (
    DEFAULT_MASK_VALUE,
    LateChunkingPooling,
    ParallelClauseNoulScanner,
    TopKChunkCrossAttention,
    YaRNScaledRotaryEmbedding,
    apply_yarn_to_modernbert,
)


def test_yarn_scaled_rotary_embedding_math() -> None:
    """YaRN RoPE スケーリングの数理特性 (高周波保護と低周波補間) を検証する。"""
    dim = 64
    original_max = 8192
    target_max = 16384
    scale = 2.0

    yarn = YaRNScaledRotaryEmbedding(
        dim=dim,
        original_max_position=original_max,
        target_max_position=target_max,
        base=160000.0,
        alpha=1.0,
        beta=32.0,
    )

    # 温度補正係数の検証
    expected_mscale = 0.1 * math.log(scale) + 1.0
    assert abs(yarn.mscale - expected_mscale) < 1e-6

    # 出力テンソル形状と有限性の検証
    seq_len = 1024
    device = torch.device("cpu")
    cos, sin = yarn(seq_len=seq_len, device=device)

    assert cos.shape == (seq_len, dim)
    assert sin.shape == (seq_len, dim)
    assert torch.isfinite(cos).all()
    assert torch.isfinite(sin).all()

    # 最高周波数 (pos=0) は未補間 (gamma=1) であるため元の周波数と一致するはず
    original_inv_freq_0 = 1.0
    yarn_inv_freq = yarn.inv_freq
    assert isinstance(yarn_inv_freq, torch.Tensor)
    assert abs(yarn_inv_freq[0].item() - original_inv_freq_0) < 1e-5

    # 最低周波数 (pos=dim-2) は低周波帯域 (gamma=0) で 1/scale 倍に補間されるはず
    base = 160000.0
    pos_last = (dim - 2) / dim
    orig_inv_freq_last = 1.0 / (base**pos_last)
    expected_yarn_last = orig_inv_freq_last / scale
    assert abs(yarn_inv_freq[-1].item() - expected_yarn_last) < 1e-5


def test_apply_yarn_to_modernbert_sliding_window_protection() -> None:
    """ModernBERT の大域層のみに YaRN が適用され、局所スライディング窓層が固定保護されることを検証する。"""
    config = ModernBertConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=3,
        intermediate_size=128,
        max_position_embeddings=8192,
        local_attention=128,
        layer_types=["full_attention", "sliding_attention", "sliding_attention"],
        rope_parameters={
            "full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
    )
    model = ModernBertModel(config)
    rotary_emb = model.rotary_emb

    sliding_freq = rotary_emb.sliding_attention_inv_freq
    full_freq = rotary_emb.full_attention_inv_freq
    assert isinstance(sliding_freq, torch.Tensor)
    assert isinstance(full_freq, torch.Tensor)

    # 適用前の局所層および大域層の周波数を記録
    orig_sliding_inv_freq = sliding_freq.clone()
    orig_full_inv_freq = full_freq.clone()

    # YaRN スケーリングを適用 (8192 -> 16384)
    apply_yarn_to_modernbert(model, target_max_position=16384)

    # 1. 局所層の周波数は一切変更されていないこと (最重要要件)
    assert torch.equal(sliding_freq, orig_sliding_inv_freq)

    # 2. 大域層の周波数は YaRN スケーリングされて変更されていること
    assert not torch.equal(full_freq, orig_full_inv_freq)

    # 3. 大域層の attention_scaling に温度補正係数が設定されていること
    expected_mscale = 0.1 * math.log(2.0) + 1.0
    scaling_attr = rotary_emb.full_attention_attention_scaling
    assert isinstance(scaling_attr, (float, int, torch.Tensor))
    actual_scaling = float(
        scaling_attr.item() if isinstance(scaling_attr, torch.Tensor) else scaling_attr
    )
    assert abs(actual_scaling - expected_mscale) < 1e-5

    # 4. config の max_position_embeddings が更新されていること
    assert model.config.max_position_embeddings == 16384


def test_late_chunking_pooling_mean() -> None:
    """LateChunkingPooling の Mean プーリングの集約計算とマスク生成を検証する。"""
    hidden_size = 16
    pooler = LateChunkingPooling(hidden_size=hidden_size, pooling_strategy="mean")

    batch_size = 2
    seq_len = 50
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)

    # サンプル0: 2チャンク, サンプル1: 3チャンク
    chunk_spans = [
        [(0, 9), (10, 29)],
        [(0, 19), (20, 39), (40, 49)],
    ]

    pooled, chunk_mask = pooler(hidden_states, chunk_spans=chunk_spans)

    # 形状の検証: 最大チャンク数は 3
    assert pooled.shape == (batch_size, 3, hidden_size)
    assert chunk_mask.shape == (batch_size, 3)

    # マスクの検証
    assert chunk_mask[0, 0].item() is True
    assert chunk_mask[0, 1].item() is True
    assert chunk_mask[0, 2].item() is False  # サンプル0の3番目は無効
    assert chunk_mask[1, :].all().item() is True  # サンプル1は全3チャンク有効

    # 平均値の手計算一致検証
    expected_b0_c0 = hidden_states[0, 0:10, :].mean(dim=0)
    assert torch.allclose(pooled[0, 0], expected_b0_c0, atol=1e-5)

    expected_b1_c2 = hidden_states[1, 40:50, :].mean(dim=0)
    assert torch.allclose(pooled[1, 2], expected_b1_c2, atol=1e-5)


def test_late_chunking_pooling_attention() -> None:
    """LateChunkingPooling の Attention-Weighted プーリングと逆伝播を検証する。"""
    hidden_size = 16
    pooler = LateChunkingPooling(hidden_size=hidden_size, pooling_strategy="attention")

    hidden_states = torch.randn(2, 40, hidden_size, requires_grad=True)
    chunk_spans = [
        [(0, 15), (16, 35)],
        [(5, 25)],
    ]

    pooled, chunk_mask = pooler(hidden_states, chunk_spans=chunk_spans)

    assert pooled.shape == (2, 2, hidden_size)
    assert chunk_mask[0, 0].item() is True
    assert chunk_mask[0, 1].item() is True
    assert chunk_mask[1, 0].item() is True
    assert chunk_mask[1, 1].item() is False

    # 勾配の逆伝播を検証
    loss = pooled.sum()
    loss.backward()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()


def test_late_chunking_pooling_tensor_inputs() -> None:
    """テンソル形式の境界入力に対する LateChunkingPooling の動作を検証する。"""
    hidden_size = 16
    pooler = LateChunkingPooling(hidden_size=hidden_size, pooling_strategy="mean")

    hidden_states = torch.randn(2, 30, hidden_size)
    chunk_starts = torch.tensor([[0, 10], [5, 15]], dtype=torch.long)
    chunk_ends = torch.tensor([[9, 25], [14, 29]], dtype=torch.long)
    chunk_mask = torch.tensor([[True, True], [True, False]], dtype=torch.bool)

    pooled, out_mask = pooler(
        hidden_states=hidden_states,
        chunk_starts=chunk_starts,
        chunk_ends=chunk_ends,
        chunk_mask=chunk_mask,
    )

    assert pooled.shape == (2, 2, hidden_size)
    assert torch.equal(out_mask, chunk_mask)
    expected = hidden_states[0, 0:10, :].mean(dim=0)
    assert torch.allclose(pooled[0, 0], expected, atol=1e-5)

    # Attention プーリングでのテンソル入力テスト。
    pooler_attn = LateChunkingPooling(
        hidden_size=hidden_size, pooling_strategy="attention"
    )
    pooled_attn, out_mask_attn = pooler_attn(
        hidden_states=hidden_states,
        chunk_starts=chunk_starts,
        chunk_ends=chunk_ends,
        chunk_mask=chunk_mask,
    )
    assert pooled_attn.shape == (2, 2, hidden_size)
    assert torch.equal(out_mask_attn, chunk_mask)
    assert torch.isfinite(pooled_attn).all()


def test_top_k_chunk_cross_attention() -> None:
    """Top-k チャンク交差アテンションの抽出およびアテンション集約を検証する。"""
    hidden_size = 32
    top_k = 3
    num_chunks = 10
    batch_size = 2
    query_len = 8

    topk_attn = TopKChunkCrossAttention(
        hidden_size=hidden_size,
        top_k=top_k,
        num_heads=4,
    )

    query_states = torch.randn(batch_size, query_len, hidden_size)
    chunk_states = torch.randn(batch_size, num_chunks, hidden_size)
    chunk_mask = torch.ones(batch_size, num_chunks, dtype=torch.bool)
    chunk_mask[0, 8:] = False  # サンプル0の末尾2チャンクをマスク

    out = topk_attn(
        query_states=query_states,
        chunk_states=chunk_states,
        chunk_mask=chunk_mask,
    )

    # 出力形状は質問系列と一致するはず
    assert out.shape == (batch_size, query_len, hidden_size)
    assert torch.isfinite(out).all()


def test_parallel_clause_noul_scanner() -> None:
    """全条項並列 Noul スキャナーの一括判定とマスク処理を検証する。"""
    hidden_size = 32
    scanner = ParallelClauseNoulScanner(hidden_size=hidden_size)

    batch_size = 2
    num_chunks = 6
    chunk_states = torch.randn(batch_size, num_chunks, hidden_size)
    chunk_mask = torch.ones(batch_size, num_chunks, dtype=torch.bool)
    chunk_mask[0, 4:] = False  # サンプル0の4番目以降を無効化

    logits, probs = scanner(chunk_states=chunk_states, chunk_mask=chunk_mask)

    assert logits.shape == (batch_size, num_chunks)
    assert probs.shape == (batch_size, num_chunks)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # マスク領域の検証: ロジットは DEFAULT_MASK_VALUE、確率は 0.0
    assert (logits[0, 4:] == DEFAULT_MASK_VALUE).all()
    assert (probs[0, 4:] == 0.0).all()
    # 有効領域の確率は正の値であること
    assert (probs[0, :4] > 0.0).all()
