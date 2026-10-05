"""Proper Scoring Rules 蒸留損失モジュール。

Choice (温度付き KL ダイバージェンス + CE)、Score (Ranked Probability Score: RPS 累積順序蒸留 + ODIR 正則化)、
および Noul (二値ソフト蒸留) を統合し、ワンホットへの過剰適合による確率較正 (ECE) の崩壊を防止する。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pipeline.self_evolution.config import DistillationConfig

DEFAULT_MASK_VALUE = -1e4


def compute_odir_penalty(logits: Tensor, op_mask: Tensor | None = None) -> Tensor:
    """過信抑制のための非対角情報正則化 (Off-Diagonal Information Regularization: ODIR) を算出する。

    ロジットの非対角共分散を抑制し、候補間の過剰な相関・過信増幅を防ぐ。

    Args:
        logits (Tensor): ロジットテンソル `[batch_size, num_classes]`。
        op_mask (Tensor | None): 有効候補マスク `[batch_size, num_classes]`。

    Returns:
        Tensor: スカラー正則化ペナルティ。
    """
    if op_mask is not None:
        valid_counts = op_mask.sum(dim=-1, keepdim=True).clamp(min=2)
        # マスクされた箇所のロジットを平均値で置換して中心化
        masked_logits = logits.masked_fill(~op_mask, 0.0)
        mean = masked_logits.sum(dim=-1, keepdim=True) / valid_counts
        centered = (logits - mean).masked_fill(~op_mask, 0.0)
    else:
        centered = logits - logits.mean(dim=-1, keepdim=True)

    # 共分散行列の算出: C = (centered^T * centered) / B
    batch_size = logits.size(0)
    cov = torch.matmul(centered.transpose(0, 1), centered) / max(batch_size, 1)

    # 非対角成分の二乗和
    eye = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    off_diag = cov * (1.0 - eye)
    return off_diag.pow(2).mean()


class SoftKLDivergenceLoss(nn.Module):
    """温度スケーリング付き Kullback-Leibler ダイバージェンス蒸留損失層。

    数理仕様:
    $$\\mathcal{L}_{\\text{KL}} = \\tau^2 \\sum_{k} q_k^\\tau \\left( \\ln q_k^\\tau - \\ln p_k^\\tau \\right)$$

    Attributes:
        temperature (float): 蒸留温度 $\\tau$。
        mask_value (float): パディング無効候補に適用する負の無限大代替値。
    """

    def __init__(
        self, temperature: float = 2.0, mask_value: float = DEFAULT_MASK_VALUE
    ) -> None:
        """KL 蒸留損失層を初期化する。

        Args:
            temperature (float): 蒸留温度 $\\tau$。
            mask_value (float): 無効候補マスク値。
        """
        super().__init__()
        self.temperature = temperature
        self.mask_value = mask_value

    def forward(
        self,
        student_logits: Tensor,
        teacher_probs: Tensor,
        op_mask: Tensor | None = None,
    ) -> Tensor:
        """温度付き KL 蒸留損失を算出する。

        Args:
            student_logits (Tensor): 生徒モデルの未正規化ロジット `[batch_size, num_classes]`。
            teacher_probs (Tensor): 教師モデルのソフト確率分布 `[batch_size, num_classes]`。
            op_mask (Tensor | None): 有効候補マスク `[batch_size, num_classes]`。

        Returns:
            Tensor: スカラー損失テンソル。
        """
        tau = self.temperature

        if op_mask is not None:
            masked_student = student_logits.masked_fill(~op_mask, self.mask_value)
            masked_teacher = teacher_probs.masked_fill(~op_mask, 0.0)
            # 教師確率の正規化
            teacher_sum = masked_teacher.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            norm_teacher = masked_teacher / teacher_sum
        else:
            masked_student = student_logits
            norm_teacher = teacher_probs

        student_log_probs = F.log_softmax(masked_student / tau, dim=-1)

        # 教師確率の温度スケーリング
        teacher_log_probs = torch.log(norm_teacher.clamp(min=1e-8))
        scaled_teacher_probs = F.softmax(teacher_log_probs / tau, dim=-1)

        # KL Divergence: sum(q * (log(q) - log(p)))
        kl_loss = F.kl_div(
            student_log_probs,
            scaled_teacher_probs,
            reduction="batchmean",
        )
        return (tau**2) * kl_loss


class RankedProbabilityScoreDistillationLoss(nn.Module):
    """Score (順序尺度) 向け Ranked Probability Score (RPS) ソフト蒸留損失層。

    教師と生徒の累積分布関数 (CDF) 間の二乗距離を最小化し、
    順序尺度の幾何構造を完全に保持した確率分布蒸留を実現する。

    数理仕様:
    $$P_m = \\sum_{k=1}^m p_k, \\quad Q_m = \\sum_{k=1}^m q_k$$
    $$\\mathcal{L}_{\\text{Score\\_RPS}} = \\frac{1}{K-1} \\sum_{m=1}^{K-1} (P_m - Q_m)^2 + \\lambda_{\\text{ODIR}} \\Omega_{\\text{ODIR}}(z)$$

    Attributes:
        odir_lambda (float): ODIR 正則化係数 $\\lambda_{\\text{ODIR}}$。
        mask_value (float): 無効候補マスク値。
    """

    def __init__(
        self, odir_lambda: float = 0.01, mask_value: float = DEFAULT_MASK_VALUE
    ) -> None:
        """RPS 蒸留損失層を初期化する。

        Args:
            odir_lambda (float): ODIR 正則化係数。
            mask_value (float): 無効候補マスク値。
        """
        super().__init__()
        self.odir_lambda = odir_lambda
        self.mask_value = mask_value

    def forward(
        self,
        student_logits: Tensor,
        teacher_probs: Tensor,
        op_mask: Tensor | None = None,
    ) -> Tensor:
        """RPS 累積順序蒸留損失を算出する。

        Args:
            student_logits (Tensor): 生徒モデルのロジット `[batch_size, num_levels]`。
            teacher_probs (Tensor): 教師モデルのソフト確率分布 `[batch_size, num_levels]`。
            op_mask (Tensor | None): 有効段階マスク `[batch_size, num_levels]`。

        Returns:
            Tensor: スカラー損失テンソル。
        """
        if op_mask is not None:
            masked_logits = student_logits.masked_fill(~op_mask, self.mask_value)
            masked_teacher = teacher_probs.masked_fill(~op_mask, 0.0)
            teacher_sum = masked_teacher.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            norm_teacher = masked_teacher / teacher_sum
            k_levels = op_mask.sum(dim=-1).float().clamp(min=2)
        else:
            masked_logits = student_logits
            norm_teacher = teacher_probs
            k_levels = torch.full(
                (student_logits.size(0),),
                float(student_logits.size(1)),
                device=student_logits.device,
            )

        student_probs = F.softmax(masked_logits, dim=-1)

        # 累積分布関数 (CDF) の計算: P_m, Q_m
        cdf_student = torch.cumsum(student_probs, dim=-1)
        cdf_teacher = torch.cumsum(norm_teacher, dim=-1)

        # 最終段階 (CDF=1.0) を除外した差分二乗和
        cdf_diff_sq = (cdf_student[..., :-1] - cdf_teacher[..., :-1]).pow(2)

        if op_mask is not None:
            # マスクされた無効段階の差分を除外
            valid_cdf_mask = op_mask[..., :-1]
            cdf_diff_sq = cdf_diff_sq.masked_fill(~valid_cdf_mask, 0.0)

        # サンプルごとに 1 / (K - 1) で正規化
        sample_rps = cdf_diff_sq.sum(dim=-1) / (k_levels - 1.0)
        mean_rps = sample_rps.mean()

        # ODIR 正則化の付加
        if self.odir_lambda > 0.0:
            odir_loss = compute_odir_penalty(student_logits, op_mask)
            return mean_rps + self.odir_lambda * odir_loss

        return mean_rps


class ProperScoringDistillationLoss(nn.Module):
    """全プリミティブ (Choice, Score, Noul) 統合型 Proper Scoring Rules 蒸留損失層。

    ハードラベルと教師ソフト確率分布をブレンドし、
    タスク幾何構造に応じた最適な蒸留損失を動的に適用する。
    """

    def __init__(self, config: DistillationConfig | None = None) -> None:
        """統合蒸留損失層を初期化する。

        Args:
            config (DistillationConfig | None): 蒸留設定。
        """
        super().__init__()
        self.config = config or DistillationConfig()
        self.kl_loss_fn = SoftKLDivergenceLoss(temperature=self.config.temperature)
        self.rps_loss_fn = RankedProbabilityScoreDistillationLoss(
            odir_lambda=self.config.odir_lambda
        )

    def forward(
        self,
        student_logits: Tensor,
        labels: Tensor,
        teacher_probs: Tensor | None = None,
        question_type: str = "choice",
        op_mask: Tensor | None = None,
    ) -> Tensor:
        """タスク種別に応じた複合蒸留損失を算出する。

        Args:
            student_logits (Tensor): 生徒ロジット `[batch_size, num_classes]`。
            labels (Tensor): 正解インデックス `[batch_size]`。
            teacher_probs (Tensor | None): 教師ソフト確率 `[batch_size, num_classes]`。
            question_type (str): タスク種別 ('choice', 'score', 'noul')。
            op_mask (Tensor | None): 有効候補マスク `[batch_size, num_classes]`。

        Returns:
            Tensor: スカラー損失テンソル。
        """
        alpha = self.config.alpha
        qtype = question_type.lower()

        # 教師ソフトラベルが存在しない場合は通常のハード損失を算出
        if teacher_probs is None or alpha <= 0.0:
            masked_logits = (
                student_logits.masked_fill(~op_mask, DEFAULT_MASK_VALUE)
                if op_mask is not None
                else student_logits
            )
            return F.cross_entropy(masked_logits, labels)

        # 1. ハードクロスエントロピー損失
        masked_logits = (
            student_logits.masked_fill(~op_mask, DEFAULT_MASK_VALUE)
            if op_mask is not None
            else student_logits
        )
        hard_loss = F.cross_entropy(masked_logits, labels)

        # 2. タスク別ソフト蒸留損失
        if qtype == "score":
            soft_loss = self.rps_loss_fn(
                student_logits=student_logits,
                teacher_probs=teacher_probs,
                op_mask=op_mask,
            )
            return (
                1.0 - alpha
            ) * hard_loss + alpha * self.config.score_rps_weight * soft_loss

        # Choice および Noul: 温度付き KL 蒸留
        soft_loss = self.kl_loss_fn(
            student_logits=student_logits,
            teacher_probs=teacher_probs,
            op_mask=op_mask,
        )
        return (1.0 - alpha) * hard_loss + alpha * soft_loss
