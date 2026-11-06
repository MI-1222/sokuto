"""Dual-Sequence 2セッション分割型アーキテクチャ単体およびエクスポート検証テストスイート。

Cross-Attention、Context-Aware Gating、StateEncoder、QueryEvaluator、および
ONNX デュアルグラフエクスポートと ONNX Runtime パリティ (Atol <= 1e-4) を網羅的に検証する。
"""

from pathlib import Path
from typing import cast

import onnx
import onnxruntime as ort
import pytest
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedTokenizerFast

from contract import (
    TENSOR_ATTENTION_MASK,
    TENSOR_INPUT_IDS,
    TENSOR_LOGITS,
    TENSOR_OP_INDICES,
    TENSOR_QUERY_IDS,
    TENSOR_QUERY_MASK,
    TENSOR_STATE_HIDDEN_STATES,
    TENSOR_STATE_MASK,
    TOKEN_OPTION_MARKER,
)
from export.export_dual import export_dual_models
from models.decision_head import DEFAULT_MASK_VALUE
from models.dual_seq import (
    AnchorPoolingLayer,
    ContextAwareGating,
    CrossAttentionBlock,
    CrossAttentionStack,
    DualSeqDecisionModel,
    SokutoQueryEvaluator,
    SokutoStateEncoder,
)


@pytest.fixture
def lightweight_dual_models(
    tmp_path: Path,
) -> tuple[SokutoStateEncoder, SokutoQueryEvaluator, PreTrainedTokenizerFast]:
    """テスト用の軽量バックボーンを持つ StateEncoder と QueryEvaluator を準備するフィクスチャ。

    Args:
        tmp_path (Path): pytest 提供の一時ディレクトリ。

    Returns:
        tuple[SokutoStateEncoder, SokutoQueryEvaluator, PreTrainedTokenizerFast]: 軽量モデル群とトークナイザー。
    """
    config = AutoConfig.from_pretrained("answerdotai/ModernBERT-base")
    config.num_hidden_layers = 2
    if hasattr(config, "layer_types") and isinstance(config.layer_types, list):
        config.layer_types = config.layer_types[:2]
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_attention_heads = 4
    config._attn_implementation = "eager"

    raw_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    assert raw_tokenizer is not None
    tokenizer = cast(PreTrainedTokenizerFast, raw_tokenizer)
    if TOKEN_OPTION_MARKER not in tokenizer.get_vocab():
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [TOKEN_OPTION_MARKER]}
        )

    backbone = AutoModel.from_config(config)
    backbone.resize_token_embeddings(len(tokenizer))

    state_encoder = SokutoStateEncoder(backbone=backbone)
    query_evaluator = SokutoQueryEvaluator(
        backbone=backbone,
        hidden_size=64,
        num_cross_heads=4,
        num_cross_layers=2,
        num_sab_heads=4,
        mlp_hidden_size=32,
    )

    state_encoder.eval()
    state_encoder.requires_grad_(False)
    query_evaluator.eval()
    query_evaluator.requires_grad_(False)

    return state_encoder, query_evaluator, tokenizer


def test_cross_attention_block_and_stack() -> None:
    """CrossAttentionBlock および CrossAttentionStack のフォワードと勾配伝播を検証する。"""
    batch_size = 2
    q_len = 8
    s_len = 16
    hidden_size = 32

    block = CrossAttentionBlock(
        hidden_size=hidden_size,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
        eps=1e-5,
    )
    stack = CrossAttentionStack(
        hidden_size=hidden_size,
        num_layers=2,
        num_heads=4,
        ffn_dim=64,
        dropout=0.0,
        eps=1e-5,
    )

    q = torch.randn(batch_size, q_len, hidden_size, requires_grad=True)
    kv = torch.randn(batch_size, s_len, hidden_size, requires_grad=True)
    mask = torch.ones(batch_size, s_len, dtype=torch.long)
    mask[:, 10:] = 0

    out_block = block(q, kv, key_mask=mask)
    assert out_block.shape == (batch_size, q_len, hidden_size)
    assert not torch.isnan(out_block).any()

    out_stack = stack(q, kv, key_mask=mask)
    assert out_stack.shape == (batch_size, q_len, hidden_size)
    assert not torch.isnan(out_stack).any()

    loss = out_stack.sum()
    loss.backward()
    assert q.grad is not None
    assert kv.grad is not None


def test_context_aware_gating() -> None:
    """ContextAwareGating の動的融合出力を検証する。"""
    batch_size = 2
    seq_len = 6
    hidden_size = 32

    gating = ContextAwareGating(hidden_size=hidden_size)
    q_h = torch.randn(batch_size, seq_len, hidden_size)
    c_h = torch.randn(batch_size, seq_len, hidden_size)

    fused = gating(q_h, c_h)
    assert fused.shape == (batch_size, seq_len, hidden_size)
    assert not torch.isnan(fused).any()


def test_anchor_pooling_layer() -> None:
    """AnchorPoolingLayer の大域平均射影注入を検証する。"""
    batch_size = 2
    q_len = 8
    s_len = 16
    hidden_size = 32

    layer = AnchorPoolingLayer(hidden_size=hidden_size)
    q_h = torch.randn(batch_size, q_len, hidden_size)
    s_h = torch.randn(batch_size, s_len, hidden_size)
    s_mask = torch.ones(batch_size, s_len, dtype=torch.long)
    s_mask[:, 12:] = 0

    out = layer(q_h, s_h, state_mask=s_mask)
    assert out.shape == (batch_size, q_len, hidden_size)
    assert not torch.isnan(out).any()


def test_dual_seq_e2e_forward(
    lightweight_dual_models: tuple[
        SokutoStateEncoder, SokutoQueryEvaluator, PreTrainedTokenizerFast
    ],
) -> None:
    """StateEncoder と QueryEvaluator の連動フォワードおよびパディング処理を検証する。"""
    state_encoder, query_evaluator, _ = lightweight_dual_models

    batch_size = 2
    state_seq_len = 32
    query_seq_len = 16
    num_options = 3

    state_ids = torch.randint(0, 100, (batch_size, state_seq_len), dtype=torch.long)
    state_mask = torch.ones((batch_size, state_seq_len), dtype=torch.long)
    state_mask[1, 24:] = 0

    query_ids = torch.randint(0, 100, (batch_size, query_seq_len), dtype=torch.long)
    query_mask = torch.ones((batch_size, query_seq_len), dtype=torch.long)

    op_indices = torch.tensor(
        [
            [4, 8, 12],
            [3, 7, -1],
        ],
        dtype=torch.long,
    )

    state_hidden = state_encoder(input_ids=state_ids, attention_mask=state_mask)
    assert state_hidden.shape == (batch_size, state_seq_len, 64)

    logits = query_evaluator(
        query_ids=query_ids,
        query_mask=query_mask,
        op_indices=op_indices,
        state_hidden_states=state_hidden,
        state_mask=state_mask,
    )
    assert logits.shape == (batch_size, num_options)
    assert not torch.isnan(logits).any()

    # パディングスロット (-1) が -1e4 でマスクされていることの確認
    assert logits[1, 2].item() == pytest.approx(DEFAULT_MASK_VALUE, rel=1e-3)

    # 統合モデルラッパーの動作確認
    e2e_model = DualSeqDecisionModel(
        state_encoder=state_encoder,
        query_evaluator=query_evaluator,
    )
    e2e_logits = e2e_model(
        state_ids=state_ids,
        state_mask=state_mask,
        query_ids=query_ids,
        query_mask=query_mask,
        op_indices=op_indices,
    )
    torch.testing.assert_close(logits, e2e_logits)


def test_export_dual_models_and_parity(
    lightweight_dual_models: tuple[
        SokutoStateEncoder, SokutoQueryEvaluator, PreTrainedTokenizerFast
    ],
    tmp_path: Path,
) -> None:
    """StateEncoder および QueryEvaluator の ONNX エクスポートと ORT 数値パリティを検証する。"""
    state_encoder, query_evaluator, _ = lightweight_dual_models
    out_dir = tmp_path / "dual_export"

    state_onnx, query_onnx = export_dual_models(
        state_encoder=state_encoder,
        query_evaluator=query_evaluator,
        output_dir=out_dir,
        opset_version=18,
        dummy_batch_size=1,
        dummy_state_seq_len=24,
        dummy_query_seq_len=16,
        dummy_num_options=3,
        verify_parity=True,
        atol=1e-4,
    )

    assert state_onnx.exists()
    assert query_onnx.exists()

    # 1. StateEncoder トポロジー・契約検査
    state_proto = onnx.load_model(str(state_onnx))
    state_inputs = [inp.name for inp in state_proto.graph.input]
    assert TENSOR_INPUT_IDS in state_inputs
    assert TENSOR_ATTENTION_MASK in state_inputs
    state_outputs = [out.name for out in state_proto.graph.output]
    assert TENSOR_STATE_HIDDEN_STATES in state_outputs

    # 2. QueryEvaluator トポロジー・契約検査
    query_proto = onnx.load_model(str(query_onnx))
    query_inputs = [inp.name for inp in query_proto.graph.input]
    assert TENSOR_QUERY_IDS in query_inputs
    assert TENSOR_QUERY_MASK in query_inputs
    assert TENSOR_OP_INDICES in query_inputs
    assert TENSOR_STATE_HIDDEN_STATES in query_inputs
    assert TENSOR_STATE_MASK in query_inputs
    query_outputs = [out.name for out in query_proto.graph.output]
    assert TENSOR_LOGITS in query_outputs

    # 3. 動的軸での推論パリティ検証 (可変バッチ: 2, State長: 40, Query長: 20, 候補数: 4)
    test_batch = 2
    test_state_len = 40
    test_query_len = 20

    test_state_ids = torch.randint(
        0, 100, (test_batch, test_state_len), dtype=torch.long
    )
    test_state_mask = torch.ones((test_batch, test_state_len), dtype=torch.long)
    test_query_ids = torch.randint(
        0, 100, (test_batch, test_query_len), dtype=torch.long
    )
    test_query_mask = torch.ones((test_batch, test_query_len), dtype=torch.long)
    test_op_indices = torch.tensor(
        [
            [2, 6, 10, 14],
            [3, 7, 11, -1],
        ],
        dtype=torch.long,
    )

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    sess_state = ort.InferenceSession(str(state_onnx), sess_options=opts)
    sess_query = ort.InferenceSession(str(query_onnx), sess_options=opts)

    # PyTorch Eager 実行
    with torch.no_grad():
        pt_state_hidden = state_encoder(test_state_ids, test_state_mask)
        pt_logits = query_evaluator(
            test_query_ids,
            test_query_mask,
            test_op_indices,
            pt_state_hidden,
            test_state_mask,
        )

    # ONNX Runtime 実行 (StateEncoder)
    ort_state_out = sess_state.run(
        None,
        {
            TENSOR_INPUT_IDS: test_state_ids.numpy(),
            TENSOR_ATTENTION_MASK: test_state_mask.numpy(),
        },
    )[0]
    torch.testing.assert_close(
        pt_state_hidden,
        torch.from_numpy(ort_state_out),
        atol=1e-4,
        rtol=1e-4,
    )

    # ONNX Runtime 実行 (QueryEvaluator)
    ort_query_out = sess_query.run(
        None,
        {
            TENSOR_QUERY_IDS: test_query_ids.numpy(),
            TENSOR_QUERY_MASK: test_query_mask.numpy(),
            TENSOR_OP_INDICES: test_op_indices.numpy(),
            TENSOR_STATE_HIDDEN_STATES: ort_state_out,
            TENSOR_STATE_MASK: test_state_mask.numpy(),
        },
    )[0]
    torch.testing.assert_close(
        pt_logits,
        torch.from_numpy(ort_query_out),
        atol=1e-4,
        rtol=1e-4,
    )
