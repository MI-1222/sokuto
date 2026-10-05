"""Jev SFT 学習パイプラインパッケージ。

設定、マスク付き損失関数、多面的評価指標、および Accelerate ベースの Trainer を提供する。
"""

from training.config import SFTConfig
from training.hierarchical_config import HierarchicalSFTConfig
from training.hierarchical_loss import (
    HierarchicalConsistencyLoss,
    HierarchicalMultiTaskLoss,
)
from training.hierarchical_trainer import HierarchicalSFTTrainer
from training.insent_loss import InSeNTLoss
from training.long_context_config import LongContextConfig
from training.long_context_trainer import LongContextTrainer
from training.loss import (
    AsymmetricBCELoss,
    EarthMoverDistanceLoss,
    InfoNCEContrastiveLoss,
    JevMultiTaskLoss,
    LabelSmoothedFocalLoss,
    MaskedCrossEntropyLoss,
    compute_masked_loss,
)
from training.metrics import MetricsTracker
from training.rlcd_config import RLCDConfig
from training.rlcd_loss import (
    RLCDLoss,
    compute_entropy,
    compute_group_advantages,
    compute_masked_kl_divergence,
    sample_perturbed_logits,
)
from training.rlcd_trainer import (
    RLCDTrainer,
    compute_expected_calibration_error,
)
from training.scoring import (
    ProperScoringEngine,
    ProperScoringEvaluator,
    ProperScoringLoss,
    compute_bounded_log_score,
    compute_brier_reward,
    compute_brier_score,
    compute_composite_scores,
    compute_ranked_probability_score,
    compute_rps_loss,
    compute_rps_reward,
    compute_spherical_score,
    get_normalized_probabilities,
)
from training.scoring_config import ScoringConfig
from training.trainer import SFTDataset, SFTTrainer, sft_collate_fn

__all__ = [
    "AsymmetricBCELoss",
    "EarthMoverDistanceLoss",
    "HierarchicalConsistencyLoss",
    "HierarchicalMultiTaskLoss",
    "HierarchicalSFTConfig",
    "HierarchicalSFTTrainer",
    "InSeNTLoss",
    "InfoNCEContrastiveLoss",
    "JevMultiTaskLoss",
    "LabelSmoothedFocalLoss",
    "LongContextConfig",
    "LongContextTrainer",
    "MaskedCrossEntropyLoss",
    "MetricsTracker",
    "ProperScoringEngine",
    "ProperScoringEvaluator",
    "ProperScoringLoss",
    "RLCDConfig",
    "RLCDLoss",
    "RLCDTrainer",
    "SFTConfig",
    "SFTDataset",
    "SFTTrainer",
    "ScoringConfig",
    "compute_bounded_log_score",
    "compute_brier_reward",
    "compute_brier_score",
    "compute_composite_scores",
    "compute_entropy",
    "compute_expected_calibration_error",
    "compute_group_advantages",
    "compute_masked_kl_divergence",
    "compute_masked_loss",
    "compute_ranked_probability_score",
    "compute_rps_loss",
    "compute_rps_reward",
    "compute_spherical_score",
    "get_normalized_probabilities",
    "sample_perturbed_logits",
    "sft_collate_fn",
]
