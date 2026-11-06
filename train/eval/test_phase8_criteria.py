"""Exit Criteria 自動検証スイートの単体・統合テストモジュール。

レイテンシ平坦化、数値整合性パリティ、長文 NIAH 判定精度、
および L3 キャッシュ保護メモリフットプリントの判定ロジックを検証する。
"""

import tempfile
from pathlib import Path

from eval.verify_phase8_criteria import (
    run_all_phase8_criteria,
    verify_latency_reduction,
    verify_memory_and_cache,
    verify_niah_accuracy,
    verify_numerical_parity,
)


def test_verify_latency_reduction() -> None:
    """長文 `State` 複数質問時の遅延削減率が 50% 以上を達成することを検証する。"""
    result, details = verify_latency_reduction(
        state_seq_len=8192,
        num_questions=5,
        simulated=True,
        threshold=0.50,
    )
    assert result.passed is True
    assert details["reduction_rate"] >= 0.50
    # 追加 1 質問あたりの増分遅延が 10ms 未満に収まっていること。
    assert details["incremental_per_question_ms"] <= 10.0


def test_verify_numerical_parity() -> None:
    """`State` 特徴量再利用時の出力ロジットが `Atol <= 1e-4` のパリティを満たすことを検証する。"""
    result, details = verify_numerical_parity(
        hidden_size=768,
        state_seq_len=128,
        num_options=4,
        atol_threshold=1e-4,
        simulated=True,
    )
    assert result.passed is True
    assert details["max_absolute_error"] <= 1e-4
    assert details["top1_parity_matched"] is True
    assert details["nan_or_inf_detected"] is False


def test_verify_niah_accuracy() -> None:
    """8,192 トークン相当長文背景での NIAH 精度が全深度および OOD で 90% 以上を満たすことを検証する。"""
    result, details = verify_niah_accuracy(
        target_char_length=8500,
        accuracy_threshold=0.90,
        simulated=True,
    )
    assert result.passed is True
    assert details["overall_accuracy"] >= 0.90
    assert details["ood_accuracy"] >= 0.90
    assert details["lost_in_the_middle_gap"] <= 0.15
    assert details["passed_exit_criteria"] is True


def test_verify_latency_reduction_real() -> None:
    """実 PyTorch モデル推論による遅延削減率検証 (`simulated=False`) をテストする。"""
    # 短い系列長 (256) では 40% から 50% 程度の削減率、長系列 (1024 以上) では 50% 超を達成する。
    result, details = verify_latency_reduction(
        state_seq_len=1024,
        num_questions=5,
        iterations=3,
        warmup=1,
        simulated=False,
        threshold=0.50,
    )
    assert result.passed is True
    assert details["reduction_rate"] >= 0.50
    assert details["is_simulated"] is False


def test_verify_numerical_parity_real() -> None:
    """実 PyTorch モデル推論による数値整合性パリティ検証 (`simulated=False`) をテストする。"""
    result, details = verify_numerical_parity(
        hidden_size=64,
        state_seq_len=64,
        num_options=4,
        atol_threshold=1e-4,
        simulated=False,
    )
    assert result.passed is True
    assert details["max_absolute_error"] <= 1e-4
    assert details["top1_parity_matched"] is True
    assert details["nan_or_inf_detected"] is False
    assert details["is_simulated"] is False


def test_verify_niah_accuracy_real() -> None:
    """実トークナイズおよびモデル推論による NIAH パイプライン健全性検証 (`simulated=False`) をテストする。"""
    result, details = verify_niah_accuracy(
        target_char_length=1500,
        depth_steps=[0.0, 0.5, 1.0],
        accuracy_threshold=0.90,
        simulated=False,
    )
    assert result.passed is True
    assert details["is_simulated"] is False
    assert details["has_real_checkpoint"] is False
    assert details["total_samples"] > 0


def test_verify_niah_accuracy_checkpoint_asserts_on_failure() -> None:
    """本番チェックポイント指定時に精度未達であれば正当に AssertionError が発生することを検証する。"""
    import pytest
    import torch

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_ckpt = Path(tmp_dir)
        dummy_model_file = tmp_ckpt / "model.pt"
        torch.save({"state_dict": {}}, dummy_model_file)

        # 未学習重みのチェックポイントでは 90% 未達となり正当に AssertionError が送出されること。
        with pytest.raises(AssertionError, match="長文 NIAH.*基準未達"):
            verify_niah_accuracy(
                target_char_length=1500,
                depth_steps=[0.0, 0.5, 1.0],
                accuracy_threshold=0.90,
                simulated=False,
                checkpoint_dir=tmp_ckpt,
            )


def test_verify_memory_and_cache() -> None:
    """8,192 トークンの `State` 特徴量 INT8 保持サイズが論理 6.5MB 以下を満たすことを検証する。"""
    result, details = verify_memory_and_cache(
        hidden_size=768,
        state_seq_len=8192,
        max_mb_threshold=6.5,
    )
    assert result.passed is True
    # 8,192 * 768 * 1B + 8,192 * 4B = 6,324,224 bytes ~= 6.32 MB <= 6.5 MB
    assert details["int8_total_mb"] <= 6.5
    assert "physical_alignment_note" in details


def test_run_all_phase8_criteria_integration() -> None:
    """全基準の一括実行およびレポート生成が完全通過することを検証する。"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        all_passed, report = run_all_phase8_criteria(
            output_dir=tmp_path,
            fast=True,
            simulated=True,
            skip_rust=True,
        )
        assert all_passed is True
        assert report.passed is True
        assert len(report.criteria) >= 4

        # レポートファイルの永続化確認。
        json_file = tmp_path / "phase8_exit_criteria_report.json"
        md_file = tmp_path / "phase8_exit_criteria_report.md"
        assert json_file.exists()
        assert md_file.exists()
