"""自己進化蒸留パイプライン統合オーケストレーターモジュール。

難例ログ抽出、HMAC 疑似匿名化、Dual-LLM 検証、CoT DAG 射影、80:20 リプレイバッファ、
Proper Scoring 蒸留学習、および回帰品質ゲート判定の全ステージをエンドツーエンドで統制する。
"""

from __future__ import annotations

from pathlib import Path

from torch import nn

from data.schema import UnifiedSample
from pipeline.self_evolution.anonymizer import EntityConsistentAnonymizer
from pipeline.self_evolution.config import SelfEvolutionConfig
from pipeline.self_evolution.cot_distiller import CoTDistiller
from pipeline.self_evolution.log_store import HardSampleRecord, HardSampleStore
from pipeline.self_evolution.quality_gate import (
    QualityGateReport,
    RegressionQualityGate,
)
from pipeline.self_evolution.replay_buffer import GoldenRatioReplayBuffer
from pipeline.self_evolution.trainer import SelfEvolutionDistillationTrainer
from pipeline.self_evolution.verifier import DualLLMVerifier, LLMClientProtocol


class SelfEvolutionPipelineRunner:
    """自己進化蒸留パイプラインのエンドツーエンド実行クラス。"""

    def __init__(
        self,
        config: SelfEvolutionConfig | None = None,
        llm_client: LLMClientProtocol | None = None,
    ) -> None:
        """パイプラインランナーを初期化する。

        Args:
            config (SelfEvolutionConfig | None): パイプライン設定。
            llm_client (LLMClientProtocol | None): LLM 検証クライアント (任意)。
        """
        self.config = config or SelfEvolutionConfig()
        self.anonymizer = EntityConsistentAnonymizer(config=self.config.anonymizer)
        self.verifier = DualLLMVerifier(config=self.config.verifier, client=llm_client)
        self.distiller = CoTDistiller()
        self.gate = RegressionQualityGate(config=self.config.gate)

    async def run_pipeline(
        self,
        raw_hard_records: list[HardSampleRecord],
        gold_samples: list[UnifiedSample],
        model: nn.Module,
        baseline_contrast_preds: list[bool],
        new_contrast_preds: list[bool],
        resolved_sample_ids: set[str],
        ece: float = 0.0261,
    ) -> QualityGateReport:
        """自己進化パイプラインを全ステージ実行する。

        Args:
            raw_hard_records (list[HardSampleRecord]): 収集された生のエスカレーション難例ログ。
            gold_samples (list[UnifiedSample]): 本番認定済みの過去データ (Gold Standard)。
            model (nn.Module): 学習対象の sokuto 決定モデル。
            baseline_contrast_preds (list[bool]): 既存モデルの Contrast 正誤列。
            new_contrast_preds (list[bool]): 新世代モデルの Contrast 正誤列。
            resolved_sample_ids (set[str]): 新モデル推論で解決された難例サンプルID。
            ece (float): 新世代モデルの ECE。

        Returns:
            QualityGateReport: 品質ゲート判定レポート。
        """
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: 難例インジェスト & 疑似匿名化
        store = HardSampleStore(
            store_path=out_dir / "hard_samples.jsonl",
            anonymizer=self.anonymizer,
        )
        for rec in raw_hard_records:
            store.add_record(rec, auto_anonymize=True)

        # 不確実性フィルタリング
        filtered_records = store.filter_by_uncertainty()

        # Stage 2: Dual-LLM ブラインドクロスチェック
        silver_samples, _isolated_samples = await self.verifier.verify_batch(
            filtered_records
        )

        # Stage 3: CoT 前提抽出 & DAG 射影
        silver_unified_samples: list[UnifiedSample] = []
        silver_soft_labels: list[dict[str, float]] = []

        for s_sample in silver_samples:
            dag_proj = self.distiller.project_to_dag(s_sample)
            sub_samples = self.distiller.to_unified_samples(dag_proj, s_sample.state)
            silver_unified_samples.extend(sub_samples)
            for sub_s in sub_samples:
                if sub_s.question_type.value == "noul":
                    target_val = sub_s.target.lower()
                    soft = (
                        {"true": 0.9, "false": 0.1}
                        if target_val == "true"
                        else {"true": 0.1, "false": 0.9}
                    )
                    silver_soft_labels.append(soft)
                else:
                    silver_soft_labels.append(s_sample.soft_labels)

        # Stage 4: 80:20 黄金比リプレイバッファの編成
        replay_buffer = GoldenRatioReplayBuffer(config=self.config.replay)
        replay_buffer.add_gold_samples(gold_samples)
        replay_buffer.add_silver_samples(
            silver_unified_samples, soft_labels_list=silver_soft_labels
        )

        # Stage 5: 下位層 Freeze + DP-SGD 蒸留学習
        trainer = SelfEvolutionDistillationTrainer(
            model=model,
            distill_config=self.config.distillation,
            dp_config=self.config.dp,
        )
        training_history = trainer.fit(
            replay_buffer=replay_buffer,
            output_dir=out_dir,
        )

        # Stage 6: 回帰テスト品質ゲート判定
        report = self.gate.evaluate_checkpoint(
            baseline_contrast_preds=baseline_contrast_preds,
            new_contrast_preds=new_contrast_preds,
            hard_records=filtered_records,
            resolved_sample_ids=resolved_sample_ids,
            ece=ece,
            privacy_epsilon=training_history.final_epsilon,
            output_dir=out_dir,
        )

        return report
