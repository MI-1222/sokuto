"""蒸留トレーナーおよび品質ゲートモジュールのテスト。"""

import torch
from torch import nn

from data.schema import QuestionType, UnifiedSample
from pipeline.self_evolution.config import (
    DifferentialPrivacyConfig,
    DistillationConfig,
    QualityGateConfig,
    ReplayConfig,
)
from pipeline.self_evolution.log_store import HardSampleRecord
from pipeline.self_evolution.quality_gate import RegressionQualityGate
from pipeline.self_evolution.replay_buffer import GoldenRatioReplayBuffer
from pipeline.self_evolution.trainer import SelfEvolutionDistillationTrainer


class DummyModernBertModel(nn.Module):
    """ModernBERT の層構造を模倣したテスト用ダミーモデル。"""

    def __init__(self, num_layers: int = 12) -> None:
        super().__init__()
        # 下位層と上位層のトランスフォーマーレイヤー
        self.layers = nn.ModuleList([nn.Linear(32, 32) for _ in range(num_layers)])
        self.choice_head = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return self.choice_head(h)


def test_lower_layers_frozen() -> None:
    """下位 8 層のパラメータが Freeze され requires_grad が False になることを検証する。"""
    model = DummyModernBertModel(num_layers=12)
    distill_config = DistillationConfig(freeze_layers=8)
    _trainer = SelfEvolutionDistillationTrainer(
        model=model, distill_config=distill_config
    )

    # 0〜7 層は Freeze されていること
    for i in range(8):
        for param in model.layers[i].parameters():
            assert param.requires_grad is False

    # 8〜11 層およびヘッドは学習対象 (requires_grad == True) であること
    for i in range(8, 12):
        for param in model.layers[i].parameters():
            assert param.requires_grad is True

    for param in model.choice_head.parameters():
        assert param.requires_grad is True


def test_dp_sgd_privacy_budget_tracking() -> None:
    """DP-SGD 実行時にプライバシー予算 epsilon が正常に追跡され 3.0 以下に収まることを検証する。"""
    model = DummyModernBertModel(num_layers=4)
    distill_config = DistillationConfig(freeze_layers=1, num_epochs=1, batch_size=4)
    dp_config = DifferentialPrivacyConfig(
        enabled=True, target_epsilon=3.0, noise_multiplier=1.0
    )

    trainer = SelfEvolutionDistillationTrainer(
        model=model, distill_config=distill_config, dp_config=dp_config
    )

    buffer = GoldenRatioReplayBuffer(ReplayConfig(gold_ratio=0.75, silver_ratio=0.25))
    dummy_sample = UnifiedSample(
        dataset_name="d",
        sample_id="s",
        question_type=QuestionType.CHOICE,
        state="t",
        instructions="i",
        criteria={"0": "A", "1": "B"},
        target="0",
    )
    buffer.add_gold_samples([dummy_sample] * 6)
    buffer.add_silver_samples([dummy_sample] * 2)

    history = trainer.fit(buffer)

    assert history.final_epsilon > 0.0
    assert history.final_epsilon <= 3.0
    assert history.total_steps >= 1


def test_quality_gate_approved_and_rejected() -> None:
    """品質ゲートが回帰精度低下や ECE 超過を検知して正しく承認/却下を判定することを検証する。"""
    gate = RegressionQualityGate(
        QualityGateConfig(
            max_regression_rate=0.00,
            max_ece=0.06,
            min_escalation_reduction_rate=0.30,
        )
    )

    hard_records = [
        HardSampleRecord(
            sample_id=f"h_{i}",
            question_type="choice",
            state="s",
            instructions="i",
            criteria={"0": "A", "1": "B"},
            escalation_reason="r",
            system1_confidence=0.4,
        )
        for i in range(10)
    ]
    resolved_ids = {f"h_{i}" for i in range(4)}  # 40% 削減 (>= 30% 合格)

    # 1. 正常合格ケース (劣化なし, ECE 2.5%, ε 2.1)
    base_preds = [True] * 60
    new_preds = [True] * 60
    report_ok = gate.evaluate_checkpoint(
        baseline_contrast_preds=base_preds,
        new_contrast_preds=new_preds,
        hard_records=hard_records,
        resolved_sample_ids=resolved_ids,
        ece=0.025,
        privacy_epsilon=2.1,
    )
    assert report_ok.approved is True
    assert report_ok.regression_rate == 0.00
    assert len(report_ok.failure_reasons) == 0

    # 2. 回帰劣化ケース (ベースライン 100% に対し新モデル 95% -> 低下 5%)
    bad_preds = [True] * 57 + [False] * 3
    report_regress = gate.evaluate_checkpoint(
        baseline_contrast_preds=base_preds,
        new_contrast_preds=bad_preds,
        hard_records=hard_records,
        resolved_sample_ids=resolved_ids,
        ece=0.025,
        privacy_epsilon=2.1,
    )
    assert report_regress.approved is False
    assert any("回帰精度低下" in r for r in report_regress.failure_reasons)

    # 3. ECE 悪化ケース (ECE 7.2% > 6.0%)
    report_bad_ece = gate.evaluate_checkpoint(
        baseline_contrast_preds=base_preds,
        new_contrast_preds=new_preds,
        hard_records=hard_records,
        resolved_sample_ids=resolved_ids,
        ece=0.072,
        privacy_epsilon=2.1,
    )
    assert report_bad_ece.approved is False
    assert any("ECE" in r for r in report_bad_ece.failure_reasons)


class DummyTokenizer:
    """tokenize_sample で要求されるインターフェースを満たすダミートークナイザー。"""

    pad_token_id = 0
    cls_token_id = 2
    sep_token_id = 3
    bos_token_id = None
    eos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens: list[int] = []
        parts = text.split("[OP]")
        for i, part in enumerate(parts):
            if i > 0:
                tokens.append(1)  # [OP] トークン ID
            for w in part.split():
                tokens.append(abs(hash(w)) % 50 + 4)
        if not tokens:
            tokens = [4]
        if add_special_tokens:
            return [2] + tokens + [3]
        return tokens


class DummyJevDecisionModel(nn.Module):
    """(input_ids, attention_mask, op_indices) を受け取る実モデル模倣クラス。"""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(16, 16)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        op_indices: torch.Tensor,
    ) -> torch.Tensor:
        b, num_ops = op_indices.shape
        # 各 [OP] に対応するダミーロジット
        return torch.randn(b, num_ops, requires_grad=True)


def test_real_model_collate_integration() -> None:
    """tokenizer と pad_jev_collate_fn を介した実モデルバッチテンソル実行を検証する。"""
    model = DummyJevDecisionModel()
    tokenizer = DummyTokenizer()
    distill_config = DistillationConfig(freeze_layers=0, num_epochs=1, batch_size=2)

    trainer = SelfEvolutionDistillationTrainer(
        model=model,
        distill_config=distill_config,
        tokenizer=tokenizer,
        op_token_id=1,
    )

    buffer = GoldenRatioReplayBuffer(ReplayConfig(gold_ratio=0.5, silver_ratio=0.5))
    s1 = UnifiedSample(
        dataset_name="d1",
        sample_id="s1",
        question_type=QuestionType.CHOICE,
        state="文脈A",
        instructions="指示A",
        criteria={"0": "選択肢1", "1": "選択肢2"},
        target="0",
    )
    s2 = UnifiedSample(
        dataset_name="d2",
        sample_id="s2",
        question_type=QuestionType.CHOICE,
        state="文脈B",
        instructions="指示B",
        criteria={"0": "選択肢1", "1": "選択肢2"},
        target="1",
    )
    buffer.add_gold_samples([s1])
    buffer.add_silver_samples([s2], soft_labels_list=[{"0": 0.2, "1": 0.8}])

    loss = trainer.train_epoch(buffer, batch_size=2)
    assert loss >= 0.0
