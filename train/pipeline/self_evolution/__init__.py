"""CoT 自己進化蒸留パイプライン & モデル崩壊防止パッケージ。

エスカレーション難例ログの蓄積と HMAC エンティティ一貫疑似匿名化、
Dual-LLM (Claude 3.7 / GPT-4o) ブラインドクロスチェック、
CoT 前提条件の中間 DAG 射影蒸留、
80:20 黄金比リプレイサンプラー、
Proper Scoring Rules 蒸留損失 (Soft-KL / Score-RPS / ODIR)、
下位層 Freeze + DP-SGD 対応蒸留トレーナー、
およびコアテストスイート回帰品質ゲートを提供する。
"""

from pipeline.self_evolution.anonymizer import (
    EntityConsistentAnonymizer,
    EntitySpan,
)
from pipeline.self_evolution.clients import FrontierLLMClient
from pipeline.self_evolution.config import (
    AnonymizerConfig,
    DifferentialPrivacyConfig,
    DistillationConfig,
    QualityGateConfig,
    ReplayConfig,
    SelfEvolutionConfig,
    VerifierConfig,
)
from pipeline.self_evolution.cot_distiller import (
    CoTDistiller,
    DistilledDAGProjection,
    MicroDecision,
)
from pipeline.self_evolution.distillation_loss import (
    ProperScoringDistillationLoss,
    RankedProbabilityScoreDistillationLoss,
    SoftKLDivergenceLoss,
    compute_odir_penalty,
)
from pipeline.self_evolution.log_store import (
    HardSampleRecord,
    HardSampleStore,
)
from pipeline.self_evolution.quality_gate import (
    QualityGateReport,
    RegressionQualityGate,
)
from pipeline.self_evolution.replay_buffer import (
    DistillationSample,
    GoldenRatioBatchSampler,
    GoldenRatioReplayBuffer,
)
from pipeline.self_evolution.runner import (
    SelfEvolutionPipelineRunner,
)
from pipeline.self_evolution.trainer import (
    PrivacyBudgetTracker,
    SelfEvolutionDistillationTrainer,
    TrainingHistory,
)
from pipeline.self_evolution.verifier import (
    DualLLMVerifier,
    IsolatedSample,
    LLMClientProtocol,
    LLMVerificationResponse,
    MockLLMClient,
    SilverSample,
)

__all__ = [
    "AnonymizerConfig",
    "CoTDistiller",
    "DifferentialPrivacyConfig",
    "DistillationConfig",
    "DistillationSample",
    "DistilledDAGProjection",
    "DualLLMVerifier",
    "EntityConsistentAnonymizer",
    "EntitySpan",
    "FrontierLLMClient",
    "GoldenRatioBatchSampler",
    "GoldenRatioReplayBuffer",
    "HardSampleRecord",
    "HardSampleStore",
    "IsolatedSample",
    "LLMClientProtocol",
    "LLMVerificationResponse",
    "MicroDecision",
    "MockLLMClient",
    "PrivacyBudgetTracker",
    "ProperScoringDistillationLoss",
    "QualityGateConfig",
    "QualityGateReport",
    "RankedProbabilityScoreDistillationLoss",
    "RegressionQualityGate",
    "ReplayConfig",
    "SelfEvolutionConfig",
    "SelfEvolutionDistillationTrainer",
    "SelfEvolutionPipelineRunner",
    "SilverSample",
    "SoftKLDivergenceLoss",
    "TrainingHistory",
    "VerifierConfig",
    "compute_odir_penalty",
]
