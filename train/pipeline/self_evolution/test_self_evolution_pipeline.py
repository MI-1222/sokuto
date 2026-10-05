"""自己進化蒸留パイプライン全体のエンドツーエンド統合テスト。"""

import asyncio
from pathlib import Path

import torch
from torch import nn

from data.schema import QuestionType, UnifiedSample
from pipeline.self_evolution.config import (
    DifferentialPrivacyConfig,
    DistillationConfig,
    QualityGateConfig,
    ReplayConfig,
    SelfEvolutionConfig,
    VerifierConfig,
)
from pipeline.self_evolution.log_store import HardSampleRecord
from pipeline.self_evolution.runner import SelfEvolutionPipelineRunner
from pipeline.self_evolution.verifier import MockLLMClient


class MockIntegratedModel(nn.Module):
    """パイプライン統合テスト用モデル。"""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(8)])
        self.choice_head = nn.Linear(16, 2)
        self.score_head = nn.Linear(16, 3)
        self.noul_head = nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return self.choice_head(h)


def test_full_self_evolution_pipeline_e2e(tmp_path: Path) -> None:
    """自己進化パイプラインの全ステージが連携して正常終了し、レポートが生成されることを検証する。"""

    async def _run() -> None:
        temp_dir = tmp_path / "self_evolution_run"

        config = SelfEvolutionConfig(
            verifier=VerifierConfig(strict_match_required=True),
            replay=ReplayConfig(gold_ratio=0.80, silver_ratio=0.20),
            distillation=DistillationConfig(
                num_epochs=1,
                batch_size=5,
                freeze_layers=4,
            ),
            dp=DifferentialPrivacyConfig(enabled=True, target_epsilon=3.0),
            gate=QualityGateConfig(
                max_regression_rate=0.00,
                max_ece=0.06,
                min_escalation_reduction_rate=0.30,
            ),
            output_dir=temp_dir,
        )

        llm_client = MockLLMClient(default_label="1")
        runner = SelfEvolutionPipelineRunner(config=config, llm_client=llm_client)

        # 1. 難例ログの準備 (個人情報を含む難例)
        raw_hard_records = [
            HardSampleRecord(
                sample_id=f"hard_sample_{i}",
                question_type="choice",
                state=f"鈴木一郎様は契約書第{i}条に基づき返金を申請した。口座番号は 1234-5678-9012 である。",
                instructions="返金申請が適格か判定せよ。",
                criteria={"0": "不適格", "1": "適格"},
                escalation_reason="Pareto Margin 不足 (0.05 < 0.15)",
                system1_confidence=0.52,
                top_margin=0.05,
                normalized_entropy=0.78,
                free_energy=-0.8,
            )
            for i in range(5)
        ]

        # 2. 本番認定済み Gold Standard の準備
        gold_samples = [
            UnifiedSample(
                dataset_name="enterprise_gold",
                sample_id=f"gold_{i}",
                question_type=QuestionType.CHOICE,
                state=f"通常トランザクション文脈 {i}",
                instructions="適格性判定",
                criteria={"0": "不適格", "1": "適格"},
                target="0",
            )
            for i in range(20)
        ]

        # 3. テストモデルの初期化
        model = MockIntegratedModel()

        # 4. 回帰テスト用予測列 (60ペアの Contrast Sets 相当)
        baseline_preds = [True] * 60
        new_preds = [True] * 60
        resolved_ids = {"hard_sample_0", "hard_sample_1"}  # 2/5 = 40% 削減 (>= 30%)

        # パイプライン実行
        report = await runner.run_pipeline(
            raw_hard_records=raw_hard_records,
            gold_samples=gold_samples,
            model=model,
            baseline_contrast_preds=baseline_preds,
            new_contrast_preds=new_preds,
            resolved_sample_ids=resolved_ids,
            ece=0.024,
        )

        # 検証
        assert report.approved is True
        assert report.regression_rate == 0.00
        assert report.ece == 0.024
        assert report.escalation_reduction_rate >= 0.30
        assert report.privacy_epsilon <= 3.0

        # 生成レポートファイルの確認
        report_json_path = temp_dir / "quality_gate_report.json"
        report_md_path = temp_dir / "quality_gate_report.md"
        assert report_json_path.exists()
        assert report_md_path.exists()

        md_content = report_md_path.read_text(encoding="utf-8")
        assert "合格 (APPROVED)" in md_content
        assert "**既存回帰精度低下率**: `0.0000`" in md_content

    asyncio.run(_run())
