"""Dual-Sequence 2セッション分割型 ONNX エクスポートモジュール。

長文 State エンコーダ (`state_encoder.onnx`) と質問評価エンジン (`query_evaluator.onnx`) を
独立した 2 つの ONNX 計算グラフとして出力し、動的軸の適合性および
PyTorch Eager 実行との数値パリティ (Atol <= 1e-4) を検証する。
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor

from contract import (
    TENSOR_ATTENTION_MASK,
    TENSOR_INPUT_IDS,
    TENSOR_OP_INDICES,
    TENSOR_QUERY_IDS,
    TENSOR_QUERY_MASK,
    TENSOR_STATE_HIDDEN_STATES,
    TENSOR_STATE_MASK,
)
from models.backbone import DEFAULT_BACKBONE_MODEL_ID, prepare_backbone_and_tokenizer
from models.dual_seq import SokutoQueryEvaluator, SokutoStateEncoder

logger = logging.getLogger(__name__)

DEFAULT_DUAL_OPSET_VERSION: int = 18
"""Dual-Sequence エクスポートで推奨される標準 ONNX Opset バージョン。"""


def export_dual_models(
    state_encoder: SokutoStateEncoder,
    query_evaluator: SokutoQueryEvaluator,
    output_dir: Path | str,
    opset_version: int = DEFAULT_DUAL_OPSET_VERSION,
    dummy_batch_size: int = 1,
    dummy_state_seq_len: int = 32,
    dummy_query_seq_len: int = 16,
    dummy_num_options: int = 3,
    verbose: bool = False,
    verify_parity: bool = True,
    atol: float = 1e-4,
) -> tuple[Path, Path]:
    """StateEncoder と QueryEvaluator を 2 つの独立した ONNX グラフとしてエクスポートする。

    内部処理手順:
    1. 両モデルを評価モード (`eval()`) に設定し勾配追跡を無効化する。
    2. バックボーンのアテンション実装を `eager` に設定して未サポートカーネルを回避する。
    3. `state_encoder.onnx` 用のダミー入力を生成し、エクスポートおよびトポロジー検証を実行する。
    4. `query_evaluator.onnx` 用のダミー入力を生成し、エクスポートおよびトポロジー検証を実行する。
    5. `verify_parity=True` の場合、ONNX Runtime で推論を実行し、Eager 実行との絶対誤差が `atol` 以下であることを確認する。

    Args:
        state_encoder (SokutoStateEncoder): エクスポート対象の State エンコーダ。
        query_evaluator (SokutoQueryEvaluator): エクスポート対象の Query 評価エンジン。
        output_dir (Path | str): 出力先ディレクトリ。
        opset_version (int): ONNX Opset バージョン (デフォルト: 18)。
        dummy_batch_size (int): トレーシングに使用するバッチサイズ。
        dummy_state_seq_len (int): トレーシングに使用する State 系列長。
        dummy_query_seq_len (int): トレーシングに使用する Query 系列長。
        dummy_num_options (int): トレーシングに使用する候補数。
        verbose (bool): 詳細ログ出力フラグ。
        verify_parity (bool): エクスポート後に ONNX Runtime パリティ検証を行うかどうか。
        atol (float): 許容最大絶対誤差 (デフォルト: 1e-4)。

    Returns:
        tuple[Path, Path]: 保存された (state_encoder.onnx, query_evaluator.onnx) のパス。

    Raises:
        ValueError: トポロジー検査またはパリティ検証に失敗した場合。
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_onnx_path = out_dir / "state_encoder.onnx"
    query_onnx_path = out_dir / "query_evaluator.onnx"

    # 1. 評価モード化および勾配追跡停止
    state_encoder.eval()
    state_encoder.requires_grad_(False)
    query_evaluator.eval()
    query_evaluator.requires_grad_(False)

    # 2. ModernBERT のアテンション展開設定
    for model_obj in (state_encoder, query_evaluator):
        if hasattr(model_obj.backbone, "config"):
            model_obj.backbone.config._attn_implementation = "eager"
            model_obj.backbone.config.attn_implementation = "eager"

    vocab_size = getattr(state_encoder.backbone.config, "vocab_size", 50368)

    # 3. StateEncoder ダミー入力作成
    dummy_state_input_ids: Tensor = torch.randint(
        0,
        min(vocab_size, 1000),
        (dummy_batch_size, dummy_state_seq_len),
        dtype=torch.long,
    )
    dummy_state_attention_mask: Tensor = torch.ones(
        (dummy_batch_size, dummy_state_seq_len),
        dtype=torch.long,
    )

    logger.info(
        "StateEncoder ONNX エクスポートを開始します (出力先: %s, opset: %d)...",
        state_onnx_path,
        opset_version,
    )

    torch.onnx.export(
        state_encoder,
        (dummy_state_input_ids, dummy_state_attention_mask),
        str(state_onnx_path),
        input_names=state_encoder.get_input_names(),
        output_names=state_encoder.get_output_names(),
        dynamic_axes=state_encoder.get_dynamic_axes(),
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
        verbose=verbose,
    )

    # StateEncoder トポロジー検証
    state_model_proto = onnx.load(str(state_onnx_path))
    onnx.checker.check_model(state_model_proto)
    logger.info("StateEncoder ONNX トポロジー検査に合格しました。")

    # 4. QueryEvaluator ダミー入力作成
    dummy_query_ids: Tensor = torch.randint(
        0,
        min(vocab_size, 1000),
        (dummy_batch_size, dummy_query_seq_len),
        dtype=torch.long,
    )
    dummy_query_mask: Tensor = torch.ones(
        (dummy_batch_size, dummy_query_seq_len),
        dtype=torch.long,
    )

    step = max(1, dummy_query_seq_len // (dummy_num_options + 1))
    op_pos_list = [step * (i + 1) for i in range(dummy_num_options)]
    dummy_op_indices: Tensor = torch.tensor(
        [op_pos_list] * dummy_batch_size,
        dtype=torch.long,
    )

    # StateEncoder の出力または同形状のダミー隠れ状態
    with torch.no_grad():
        dummy_state_hidden: Tensor = state_encoder(
            dummy_state_input_ids, dummy_state_attention_mask
        )

    logger.info(
        "QueryEvaluator ONNX エクスポートを開始します (出力先: %s, opset: %d)...",
        query_onnx_path,
        opset_version,
    )

    torch.onnx.export(
        query_evaluator,
        (
            dummy_query_ids,
            dummy_query_mask,
            dummy_op_indices,
            dummy_state_hidden,
            dummy_state_attention_mask,
        ),
        str(query_onnx_path),
        input_names=query_evaluator.get_input_names(),
        output_names=query_evaluator.get_output_names(),
        dynamic_axes=query_evaluator.get_dynamic_axes(),
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
        verbose=verbose,
    )

    # QueryEvaluator トポロジー検証
    query_model_proto = onnx.load(str(query_onnx_path))
    onnx.checker.check_model(query_model_proto)
    logger.info("QueryEvaluator ONNX トポロジー検査に合格しました。")

    # 5. 数値パリティ検証
    if verify_parity:
        logger.info("ONNX Runtime パリティ検証を開始します (許容 Atol: %e)...", atol)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3

        sess_state = ort.InferenceSession(
            str(state_onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        sess_query = ort.InferenceSession(
            str(query_onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        # StateEncoder パリティ検証
        ort_state_inputs = {
            TENSOR_INPUT_IDS: dummy_state_input_ids.cpu().numpy(),
            TENSOR_ATTENTION_MASK: dummy_state_attention_mask.cpu().numpy(),
        }
        ort_state_hidden = sess_state.run(None, ort_state_inputs)[0]
        state_max_diff = float(
            np.max(np.abs(dummy_state_hidden.cpu().numpy() - ort_state_hidden))
        )
        logger.info("StateEncoder 最大絶対誤差: %e", state_max_diff)
        if state_max_diff > atol:
            raise ValueError(
                f"StateEncoder パリティ検証失敗: 最大誤差 {state_max_diff:.6e} が許容値 {atol:.6e} を超過しました。"
            )

        # QueryEvaluator パリティ検証
        with torch.no_grad():
            pt_logits: Tensor = query_evaluator(
                dummy_query_ids,
                dummy_query_mask,
                dummy_op_indices,
                dummy_state_hidden,
                dummy_state_attention_mask,
            )

        ort_query_inputs = {
            TENSOR_QUERY_IDS: dummy_query_ids.cpu().numpy(),
            TENSOR_QUERY_MASK: dummy_query_mask.cpu().numpy(),
            TENSOR_OP_INDICES: dummy_op_indices.cpu().numpy(),
            TENSOR_STATE_HIDDEN_STATES: ort_state_hidden,
            TENSOR_STATE_MASK: dummy_state_attention_mask.cpu().numpy(),
        }
        ort_logits = sess_query.run(None, ort_query_inputs)[0]
        query_max_diff = float(np.max(np.abs(pt_logits.cpu().numpy() - ort_logits)))
        logger.info("QueryEvaluator 最大絶対誤差: %e", query_max_diff)
        if query_max_diff > atol:
            raise ValueError(
                f"QueryEvaluator パリティ検証失敗: 最大誤差 {query_max_diff:.6e} が許容値 {atol:.6e} を超過しました。"
            )

        logger.info(
            "Dual-Sequence パリティ検証に完全合格しました (State 誤差: %e, Query 誤差: %e)。",
            state_max_diff,
            query_max_diff,
        )

    return state_onnx_path, query_onnx_path


def main() -> None:
    """コマンドラインから Dual-Sequence モデルのエクスポートを実行する。"""
    parser = argparse.ArgumentParser(
        description="Dual-Sequence (StateEncoder / QueryEvaluator) ONNX エクスポートツール。"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_BACKBONE_MODEL_ID,
        help="バックボーンモデル識別子。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/export/dual_seq",
        help="ONNX 出力先ディレクトリ。",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=DEFAULT_DUAL_OPSET_VERSION,
        help="ONNX Opset バージョン。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細エクスポートログを表示する。",
    )
    parser.add_argument(
        "--skip-parity",
        action="store_true",
        help="ONNX Runtime パリティ検証をスキップする。",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="パリティ許容最大絶対誤差。",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("バックボーンモデルをロードしています: %s...", args.model_id)
    backbone, _, _ = prepare_backbone_and_tokenizer(args.model_id)
    hidden_size = getattr(backbone.config, "hidden_size", 768)

    state_encoder = SokutoStateEncoder(backbone=backbone)
    query_evaluator = SokutoQueryEvaluator(
        backbone=backbone,
        hidden_size=hidden_size,
    )

    export_dual_models(
        state_encoder=state_encoder,
        query_evaluator=query_evaluator,
        output_dir=args.output_dir,
        opset_version=args.opset,
        verbose=args.verbose,
        verify_parity=not args.skip_parity,
        atol=args.atol,
    )
    logger.info("エクスポートが正常に完了しました。")


if __name__ == "__main__":
    main()
