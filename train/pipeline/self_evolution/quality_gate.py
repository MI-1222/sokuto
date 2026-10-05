"""回帰テスト品質ゲート & CI 自動判定モジュール。

自己進化再学習によって生成された新世代チェックポイントに対し、
コアテストスイート (Contrast Sets 60ペア等) を用いた回帰検証を実行する。
既存タスクの精度低下 0.00% (劣化ゼロ)、ECE <= 6.0%、プライバシー予算 epsilon <= 3.0、
および難例エスカレーション削減率 >= 30.0% を厳格に判定し、本番昇格またはロールバックを自律判定する。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pipeline.self_evolution.config import QualityGateConfig
from pipeline.self_evolution.log_store import HardSampleRecord


@dataclass
class QualityGateReport:
    """品質ゲート判定結果および診断レポート。

    Attributes:
        approved (bool): 本番デプロイ承認フラグ。
        regression_rate (float): 既存タスクに対する正解率の低下幅 (0.00 以下が合格)。
        baseline_accuracy (float): ベースラインモデルの正解率。
        new_accuracy (float): 新世代モデルの正解率。
        ece (float): 新世代モデルの Expected Calibration Error (ECE)。
        escalation_reduction_rate (float): 難例に対するエスカレーション削減率。
        privacy_epsilon (float): 消費された最終プライバシー予算 $\\epsilon$。
        failure_reasons (list[str]): ゲート不合格となった理由のリスト。
        diagnostics (dict[str, Any]): 各種サブテストの詳細診断情報。
    """

    approved: bool
    regression_rate: float
    baseline_accuracy: float
    new_accuracy: float
    ece: float
    escalation_reduction_rate: float
    privacy_epsilon: float
    failure_reasons: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書オブジェクトへ変換する。"""
        return asdict(self)

    def to_markdown(self) -> str:
        """人間可読な Markdown レポートを生成する。"""
        status_badge = (
            "✅ **合格 (APPROVED)**" if self.approved else "❌ **不合格 (REJECTED)**"
        )
        lines = [
            f"# 自己進化品質ゲート検証レポート: {status_badge}",
            "",
            "## 1. 定量判定サマリー",
            f"- **本番昇格判定**: {'承認' if self.approved else '却下 (ロールバック)'}",
            f"- **既存回帰精度低下率**: `{self.regression_rate:.4f}` (許容上限: `<= 0.0000`)",
            f"- **既存正解率**: `{self.baseline_accuracy:.4f}` -> `{self.new_accuracy:.4f}`",
            f"- **確率較正度 (ECE)**: `{self.ece:.4f}` (許容上限: `<= 0.0600`)",
            f"- **難例エスカレーション削減率**: `{self.escalation_reduction_rate * 100:.2f}%` (目標: `>= 30.0%`)",
            f"- **差分プライバシー予算 (ε)**: `{self.privacy_epsilon:.2f}` (保証上限: `<= 3.00`)",
            "",
        ]

        if self.failure_reasons:
            lines.append("## 2. ゲート不合格理由")
            for reason in self.failure_reasons:
                lines.append(f"- ⚠️ {reason}")
            lines.append("")

        lines.append("## 3. 診断詳細")
        lines.append(
            f"```json\n{json.dumps(self.diagnostics, ensure_ascii=False, indent=2)}\n```"
        )
        return "\n".join(lines)


class RegressionQualityGate:
    """既存精度劣化ゼロおよび較正健全性を検証する品質ゲートクラス。"""

    def __init__(self, config: QualityGateConfig | None = None) -> None:
        """品質ゲートを初期化する。

        Args:
            config (QualityGateConfig | None): ゲート設定。
        """
        self.config = config or QualityGateConfig()

    def evaluate_contrast_sets(
        self,
        baseline_preds: list[bool],
        new_preds: list[bool],
    ) -> tuple[float, float, float]:
        """コアテストスイート (対照ペア) における回帰精度を算出する。

        Args:
            baseline_preds (list[bool]): ベースラインモデルの正誤リスト。
            new_preds (list[bool]): 新世代モデルの正誤リスト。

        Returns:
            tuple[float, float, float]: (ベースライン正解率, 新世代正解率, 回帰低下率)。
        """
        if not baseline_preds or not new_preds:
            return 1.0, 1.0, 0.0

        n = len(baseline_preds)
        base_acc = sum(baseline_preds) / n
        new_acc = sum(new_preds) / n

        # 回帰低下率: ベースラインよりどれだけ下がったか (負の値は精度向上)
        regression_rate = max(base_acc - new_acc, 0.0)
        return base_acc, new_acc, regression_rate

    def evaluate_escalation_reduction(
        self,
        hard_records: list[HardSampleRecord],
        resolved_sample_ids: set[str],
    ) -> float:
        """難例ログのうち、新世代モデルによって System 1 内部で解決可能になった割合を算出する。

        Args:
            hard_records (list[HardSampleRecord]): 収集されたエスカレーション難例レコード列。
            resolved_sample_ids (set[str]): 新モデル推論で高確信度・正解判定できたサンプルID。

        Returns:
            float: エスカレーション削減率 (0.0 〜 1.0)。
        """
        if not hard_records:
            return 1.0

        total = len(hard_records)
        resolved = sum(1 for r in hard_records if r.sample_id in resolved_sample_ids)
        return resolved / total

    def evaluate_checkpoint(
        self,
        baseline_contrast_preds: list[bool],
        new_contrast_preds: list[bool],
        hard_records: list[HardSampleRecord],
        resolved_sample_ids: set[str],
        ece: float,
        privacy_epsilon: float,
        output_dir: str | Path | None = None,
    ) -> QualityGateReport:
        """新世代チェックポイントの全指標を総合判定し、承認レポートを生成する。

        Args:
            baseline_contrast_preds (list[bool]): 既存モデルの Contrast 正誤リスト。
            new_contrast_preds (list[bool]): 新世代モデルの Contrast 正誤リスト。
            hard_records (list[HardSampleRecord]): 難例レコード列。
            resolved_sample_ids (set[str]): 解決済み難例ID。
            ece (float): 新世代モデルの ECE。
            privacy_epsilon (float): 消費された DP-SGD プライバシー予算。
            output_dir (str | Path | None): レポート出力先ディレクトリ。

        Returns:
            QualityGateReport: 判定レポート。
        """
        base_acc, new_acc, reg_rate = self.evaluate_contrast_sets(
            baseline_contrast_preds, new_contrast_preds
        )
        reduction_rate = self.evaluate_escalation_reduction(
            hard_records, resolved_sample_ids
        )

        failure_reasons: list[str] = []

        # 1. 回帰低下率チェック (目標: 0.00%)
        if reg_rate > self.config.max_regression_rate:
            failure_reasons.append(
                f"回帰精度低下が検知されました: 低下幅 {reg_rate:.4f} > 許容上限 {self.config.max_regression_rate:.4f}。"
            )

        # 2. 確率較正 ECE チェック (目標: <= 6.0%)
        if ece > self.config.max_ece:
            failure_reasons.append(
                f"確率較正度 (ECE) が悪化しています: ECE {ece:.4f} > 許容上限 {self.config.max_ece:.4f}。"
            )

        # 3. エスカレーション削減率チェック (目標: >= 30.0%)
        if reduction_rate < self.config.min_escalation_reduction_rate:
            failure_reasons.append(
                f"難例エスカレーション削減率が目標未達です: {reduction_rate * 100:.2f}% < 目標 {self.config.min_escalation_reduction_rate * 100:.2f}%。"
            )

        # 4. プライバシー予算チェック (目標: <= 3.0)
        if privacy_epsilon > 3.0:
            failure_reasons.append(
                f"差分プライバシー予算 (ε) が超過しています: ε {privacy_epsilon:.2f} > 上限 3.00。"
            )

        approved = len(failure_reasons) == 0

        report = QualityGateReport(
            approved=approved,
            regression_rate=reg_rate,
            baseline_accuracy=base_acc,
            new_accuracy=new_acc,
            ece=ece,
            escalation_reduction_rate=reduction_rate,
            privacy_epsilon=privacy_epsilon,
            failure_reasons=failure_reasons,
            diagnostics={
                "num_contrast_pairs": len(baseline_contrast_preds),
                "num_hard_samples": len(hard_records),
                "num_resolved_samples": len(resolved_sample_ids),
                "config": {
                    "max_regression_rate": self.config.max_regression_rate,
                    "max_ece": self.config.max_ece,
                    "min_escalation_reduction_rate": self.config.min_escalation_reduction_rate,
                },
            },
        )

        if output_dir:
            out_p = Path(output_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            report_json = out_p / "quality_gate_report.json"
            report_md = out_p / "quality_gate_report.md"

            report_json.write_text(
                json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            report_md.write_text(report.to_markdown(), encoding="utf-8")

        return report
