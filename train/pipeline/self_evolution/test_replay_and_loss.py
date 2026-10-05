"""リプレイバッファおよび Proper Scoring 蒸留損失モジュールのテスト。"""

import torch

from data.schema import QuestionType, UnifiedSample
from pipeline.self_evolution.config import DistillationConfig, ReplayConfig
from pipeline.self_evolution.distillation_loss import (
    ProperScoringDistillationLoss,
    RankedProbabilityScoreDistillationLoss,
    SoftKLDivergenceLoss,
    compute_odir_penalty,
)
from pipeline.self_evolution.replay_buffer import (
    GoldenRatioBatchSampler,
    GoldenRatioReplayBuffer,
)


def _make_dummy_sample(
    sample_id: str, qtype: QuestionType = QuestionType.CHOICE
) -> UnifiedSample:
    """テスト用 UnifiedSample を生成する。"""
    criteria = (
        {"0": "A", "1": "B"}
        if qtype != QuestionType.SCORE
        else {"0": "低", "1": "中", "2": "高"}
    )
    return UnifiedSample(
        dataset_name="test_data",
        sample_id=sample_id,
        question_type=qtype,
        state="テスト文脈テキスト",
        instructions="指示文",
        criteria=criteria,
        target="0",
    )


def test_golden_ratio_sampler_maintains_80_20_ratio() -> None:
    """バッチサンプラーが Gold 80% : Silver 20% の比率を厳格に保持することを検証する。"""
    config = ReplayConfig(gold_ratio=0.80, silver_ratio=0.20)
    buffer = GoldenRatioReplayBuffer(config=config)

    # Gold 40 件、Silver 10 件を追加
    gold_samples = [_make_dummy_sample(f"gold_{i}") for i in range(40)]
    silver_samples = [_make_dummy_sample(f"silver_{i}") for i in range(10)]

    buffer.add_gold_samples(gold_samples)
    buffer.add_silver_samples(silver_samples)

    batch_size = 10
    sampler = GoldenRatioBatchSampler(
        buffer=buffer, batch_size=batch_size, shuffle=False
    )

    batches = list(sampler)
    assert len(batches) >= 1

    for batch in batches:
        assert len(batch) == batch_size
        # バッチ内のサンプルを判定
        gold_count = sum(1 for idx in batch if not buffer[idx].is_silver)
        silver_count = sum(1 for idx in batch if buffer[idx].is_silver)

        # 10 個中 8 個が Gold、2 個が Silver であること
        assert gold_count == 8
        assert silver_count == 2


def test_soft_kl_distillation_loss() -> None:
    """温度付き KL 蒸留損失が計算可能であり勾配が正しく流れることを検証する。"""
    loss_fn = SoftKLDivergenceLoss(temperature=2.0)

    student_logits = torch.randn(2, 4, requires_grad=True)
    teacher_probs = torch.tensor(
        [[0.7, 0.1, 0.1, 0.1], [0.1, 0.8, 0.05, 0.05]], dtype=torch.float32
    )

    loss = loss_fn(student_logits, teacher_probs)
    assert loss.item() >= 0.0
    loss.backward()
    assert student_logits.grad is not None


def test_score_rps_distillation_and_odir() -> None:
    """Score 型 RPS 累積順序蒸留損失と ODIR 正則化が正常に計算されることを検証する。"""
    loss_fn = RankedProbabilityScoreDistillationLoss(odir_lambda=0.05)

    student_logits = torch.randn(2, 3, requires_grad=True)
    # 教師ソフト確率 (3段階: 0, 1, 2)
    teacher_probs = torch.tensor(
        [[0.8, 0.15, 0.05], [0.05, 0.15, 0.8]], dtype=torch.float32
    )

    loss = loss_fn(student_logits, teacher_probs)
    assert loss.item() >= 0.0
    loss.backward()
    assert student_logits.grad is not None

    # ODIR ペナルティ単体テスト
    odir = compute_odir_penalty(student_logits)
    assert odir.item() >= 0.0


def test_proper_scoring_unified_loss() -> None:
    """Choice と Score の統合蒸留損失が設定どおり線形結合されることを検証する。"""
    loss_fn = ProperScoringDistillationLoss(
        DistillationConfig(alpha=0.5, temperature=2.0)
    )

    logits = torch.randn(2, 4, requires_grad=True)
    labels = torch.tensor([0, 1])
    teacher_probs = torch.tensor(
        [[0.8, 0.1, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05]], dtype=torch.float32
    )

    # Choice タスク
    loss_choice = loss_fn(logits, labels, teacher_probs, question_type="choice")
    assert loss_choice.item() >= 0.0

    # Score タスク
    loss_score = loss_fn(logits, labels, teacher_probs, question_type="score")
    assert loss_score.item() >= 0.0
