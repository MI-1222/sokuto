"""数理的複合損失エンジンモジュール。

可変候補数に対応するバッチにおいて、Choice (Label Smoothing + Focal),
Score (EMD / Wasserstein 順序尺度損失), Noul (非対称重み付き BCE),
および補助対照損失 (InfoNCE) を統合し、幾何構造に合致した動的損失計算と
パディング領域への確率漏洩の完全遮断を実現する。
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from data.schema import QuestionType
from training.insent_loss import InSeNTLoss

DEFAULT_MASK_VALUE = -1e4
"""無効候補に適用する負の無限大代替値。半精度アンダーフローや NaN を防止する。"""


def compute_masked_loss(
    logits: Tensor,
    labels: Tensor,
    op_mask: Tensor,
    mask_value: float = DEFAULT_MASK_VALUE,
) -> Tensor:
    """有効候補マスクを適用してクロスエントロピー損失を算出する。

    数理仕様:
    $$z_i^{\\text{masked}} = \\begin{cases} z_i & (\\text{op\\_mask}_i = \\text{True}) \\\\ \\text{mask\\_value} & (\\text{op\\_mask}_i = \\text{False}) \\end{cases}$$
    $$\\mathcal{L} = -\\log \\left( \\frac{\\exp(z_{\\text{label}}^{\\text{masked}})}{\\sum_k \\exp(z_k^{\\text{masked}})} \\right)$$

    Args:
        logits (Tensor): 未マスクのロジットテンソル `[batch_size, num_options]`。
        labels (Tensor): 正解候補インデックス `[batch_size]`。
        op_mask (Tensor): 有効候補マスク `[batch_size, num_options]` (有効: True, 無効: False)。
        mask_value (float): 無効候補に代入する値 (デフォルト: -1e4)。

    Returns:
        Tensor: スカラー損失テンソル。
    """
    masked_logits = logits.masked_fill(~op_mask, mask_value)
    return F.cross_entropy(masked_logits, labels)


class MaskedCrossEntropyLoss(nn.Module):
    """パディング候補を自動遮断するクロスエントロピー損失層。

    Attributes:
        mask_value (float): 無効候補をマスクする負の値。
    """

    def __init__(self, mask_value: float = DEFAULT_MASK_VALUE) -> None:
        """損失層を初期化する。

        Args:
            mask_value (float): 無効候補をマスクする負の値。
        """
        super().__init__()
        self.mask_value = mask_value

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        op_mask: Tensor,
    ) -> Tensor:
        """有効候補マスクを適用して損失を算出する。

        Args:
            logits (Tensor): ロジットテンソル `[batch_size, num_options]`。
            labels (Tensor): 正解インデックス `[batch_size]`。
            op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。

        Returns:
            Tensor: スカラー損失テンソル。
        """
        return compute_masked_loss(
            logits=logits,
            labels=labels,
            op_mask=op_mask,
            mask_value=self.mask_value,
        )


class LabelSmoothedFocalLoss(nn.Module):
    """可変候補数に対応した動的ラベル平滑化および多クラス Focal Loss 層。

    数理仕様:
    正解ラベル $y$ に対し、有効候補数 $K$ に応じた平滑化ターゲットを生成する：
    $$\\tilde{y}_k = (1 - \\epsilon) \\cdot \\mathbb{I}(k = y) + \\frac{\\epsilon}{K}$$
    予測対数確率 $\\ln p_k = \\text{log\\_softmax}(z)_k$ に対し、Focal 変調係数 $(1 - p_k)^\\gamma$ を適用する：
    $$\\mathcal{L} = - \\sum_{k=0}^{K-1} \\tilde{y}_k (1 - p_k)^\\gamma \\ln(p_k)$$

    Attributes:
        label_smoothing (float): 平滑化係数 $\\epsilon$。
        focal_gamma (float): Focal 変調係数 $\\gamma$。0.0 の場合は通常の平滑化クロスエントロピー。
    """

    def __init__(
        self,
        label_smoothing: float = 0.05,
        focal_gamma: float = 0.0,
    ) -> None:
        """ラベル平滑化 Focal Loss を初期化する。

        Args:
            label_smoothing (float): 平滑化係数。
            focal_gamma (float): Focal 変調係数。
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma

    def forward(self, logits: Tensor, label: Tensor) -> Tensor:
        """単一サンプルの有効候補スライスに対する損失を算出する。

        Args:
            logits (Tensor): 有効候補のロジット `[num_valid_options]` または `[1, num_valid_options]`。
            label (Tensor): 正解インデックススカラー。

        Returns:
            Tensor: スカラー損失テンソル。
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        num_classes = logits.size(-1)
        if num_classes <= 1:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        target_idx = label.view(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # 動的ラベル平滑化ターゲットの構築
        smooth_val = self.label_smoothing / num_classes
        targets = torch.full_like(logits, smooth_val)
        targets.scatter_add_(
            1,
            target_idx.unsqueeze(1),
            torch.full_like(
                target_idx.unsqueeze(1),
                1.0 - self.label_smoothing,
                dtype=logits.dtype,
            ),
        )

        if self.focal_gamma > 0.0:
            probs = torch.exp(log_probs)
            focal_weights = torch.pow(1.0 - probs, self.focal_gamma)
            loss = -torch.sum(targets * focal_weights * log_probs, dim=-1)
        else:
            loss = -torch.sum(targets * log_probs, dim=-1)

        return loss.mean()


class EarthMoverDistanceLoss(nn.Module):
    """順序尺度空間での累積分布関数 (CDF) 差分に基づく Earth Mover's Distance (EMD) 損失層。

    数理仕様:
    予測確率分布 $p = \\text{Softmax}(z)$ と正解 One-hot ベクトル $q$ の累積分布を算出する：
    $$P_k = \\sum_{j=0}^k p_j, \\quad Q_k = \\sum_{j=0}^k q_j \\quad (k = 0, \\dots, K-2)$$
    確率の総和は常に 1.0 であるため、最後の要素 $k=K-1$ は必ず $P_{K-1} - Q_{K-1} = 0$ となり除外する。
    1次元 Wasserstein 距離として以下のノルム総和を計算する：
    $$\\mathcal{L} = \\sum_{k=0}^{K-2} |P_k - Q_k|^r$$
    $r=2$ は二乗 Wasserstein 距離、$r=1$ は Smooth L1 (Huber) 形式により勾配振動を抑制する。

    Attributes:
        power (int): べき数 (1: Smooth L1 / Huber, 2: 二乗)。
        beta (float): Smooth L1 適用時の閾値パラメータ。
    """

    def __init__(self, power: int = 2, beta: float = 0.1) -> None:
        """EMD 損失層を初期化する。

        Args:
            power (int): 距離のべき数 (1 または 2)。
            beta (float): Smooth L1 適用時の閾値パラメータ。
        """
        super().__init__()
        self.power = power
        self.beta = beta

    def forward(self, logits: Tensor, label: Tensor) -> Tensor:
        """単一サンプルの有効候補スライスに対する EMD 損失を算出する。

        Args:
            logits (Tensor): 有効候補のロジット `[num_valid_options]` または `[1, num_valid_options]`。
            label (Tensor): 正解インデックススカラー。

        Returns:
            Tensor: スカラー損失テンソル。
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        num_classes = logits.size(-1)
        if num_classes <= 1:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        target_idx = label.view(-1)
        probs = F.softmax(logits, dim=-1)
        target_onehot = F.one_hot(target_idx, num_classes=num_classes).to(logits.dtype)

        # 累積分布関数 (CDF) の算出 (末尾要素は累積和 1.0 同士で差が 0 のため除外)
        cdf_pred = torch.cumsum(probs, dim=-1)[:, :-1]
        cdf_true = torch.cumsum(target_onehot, dim=-1)[:, :-1]

        if self.power == 1:
            loss = F.smooth_l1_loss(cdf_pred, cdf_true, beta=self.beta, reduction="sum")
        else:
            loss = torch.sum(torch.pow(cdf_pred - cdf_true, 2))

        return loss


class RankedProbabilityScoreLoss(nn.Module):
    """Score (順序尺度) 向け Ranked Probability Score (RPS) 損失層。

    累積分布関数 (CDF) 間の差の二乗平均 (離散型 Wasserstein-1 距離) を計算する。
    重大インシデントにおける判定逆転 (例: 段階 5 を段階 1 と誤認) を二乗オーダーで強く抑止する。

    数理仕様:
    予測確率ベクトル $p = \\text{Softmax}(z) \\in \\mathbb{R}^K$、正解 One-hot $q \\in \\mathbb{R}^K$。
    累積分布関数 $P_m = \\sum_{j=0}^m p_j, \\quad Q_m = \\sum_{j=0}^m q_j \\quad (m = 0, \\dots, K-2)$。
    $$\\mathcal{L}_{\\text{RPS}} = \\frac{1}{K-1} \\sum_{m=0}^{K-2} (P_m - Q_m)^2$$
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """RPS 損失層を初期化する。

        Args:
            eps (float): 数値安定性のための微小値。
        """
        super().__init__()
        self.eps = eps

    def forward(
        self,
        logits_or_probs: Tensor,
        label: Tensor,
        is_probs: bool = False,
    ) -> Tensor:
        """単一サンプルの有効候補スライスに対する RPS 損失を算出する。

        Args:
            logits_or_probs (Tensor): ロジットまたは確率テンソル `[num_valid_options]` または `[1, num_valid_options]`。
            label (Tensor): 正解インデックススカラー。
            is_probs (bool): 入力がすでに確率分布であるかどうか。

        Returns:
            Tensor: スカラー損失テンソル。
        """
        if logits_or_probs.dim() == 1:
            logits_or_probs = logits_or_probs.unsqueeze(0)
        num_classes = logits_or_probs.size(-1)
        if num_classes <= 1:
            return torch.tensor(
                0.0, device=logits_or_probs.device, dtype=logits_or_probs.dtype
            )

        target_idx = label.view(-1)
        if is_probs:
            probs = logits_or_probs
        else:
            probs = F.softmax(logits_or_probs, dim=-1)

        target_onehot = F.one_hot(target_idx, num_classes=num_classes).to(
            logits_or_probs.dtype
        )

        cdf_pred = torch.cumsum(probs, dim=-1)[:, :-1]
        cdf_true = torch.cumsum(target_onehot, dim=-1)[:, :-1]

        # 最終ランクを除外した二乗誤差平均
        rps = torch.sum(torch.pow(cdf_pred - cdf_true, 2), dim=-1) / (num_classes - 1)
        return rps.mean()


class AsymmetricBCELoss(nn.Module):
    """Noul 型向けの非対称重み付き二値クロスエントロピー損失層。

    数理仕様:
    `formatter.py` の規約 (`["true", "false"]` で True=0, False=1) に基づき、
    ロジット差分 $\\Delta z = z_0 - z_1 = z_{\\text{true}} - z_{\\text{false}}$ を入力とする。
    正解ターゲットは True (label 0) のとき 1.0、False (label 1) のとき 0.0 とする：
    $$y_{\\text{binary}} = (y == 0).\\text{float}()$$
    $$\\mathcal{L} = - \\beta \\cdot y_{\\text{binary}} \\ln \\sigma(\\Delta z) - (1 - y_{\\text{binary}}) \\ln (1 - \\sigma(\\Delta z))$$

    Attributes:
        pos_weight (float): 正例 (True) に対する重み係数 $\\beta$。
    """

    def __init__(self, pos_weight: float = 1.0) -> None:
        """非対称 BCE 損失層を初期化する。

        Args:
            pos_weight (float): 正例の重み係数。
        """
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: Tensor, label: Tensor) -> Tensor:
        """2候補スライスに対する二値クロスエントロピー損失を算出する。

        Args:
            logits (Tensor): 2候補のロジット `[2]` または `[1, 2]`。
            label (Tensor): 正解インデックススカラー (0: True, 1: False)。

        Returns:
            Tensor: スカラー損失テンソル。
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # True (0) と False (1) のロジット差分
        delta_z = logits[:, 0] - logits[:, 1]
        target_idx = label.view(-1)
        # label 0 (True) -> 1.0, label 1 (False) -> 0.0
        target_float = (target_idx == 0).to(logits.dtype)

        pos_weight_tensor = None
        if self.pos_weight != 1.0:
            pos_weight_tensor = torch.tensor(
                [self.pos_weight], device=logits.device, dtype=logits.dtype
            )

        return F.binary_cross_entropy_with_logits(
            delta_z,
            target_float,
            pos_weight=pos_weight_tensor,
        )


class AsymmetricLoss(nn.Module):
    """Noul (真偽判定) 向け非対称損失層 (Asymmetric Loss: ASL)。

    容易な負例をハードマージンで切り捨て、不均衡データでの正例の学習を安定化する。

    数理仕様:
    $$p = \\sigma(\\Delta z)$$
    $$p_m = \\max(p - m, 0)$$
    $$\\mathcal{L}_{\\text{ASL}} = - y (1 - p)^{\\gamma_+} \\ln(p + \\epsilon) - (1 - y) (p_m)^{\\gamma_-} \\ln(1 - p_m + \\epsilon)$$
    ここで $y \\in \\{0.0, 1.0\\}$ は正例フラグ ($0$ 番目の候補が正解のとき $1.0$)。

    Attributes:
        gamma_neg (float): 負例の変調指数 $\\gamma_-$ (デフォルト: 4.0)。
        gamma_pos (float): 正例の変調指数 $\\gamma_+$ (デフォルト: 1.0)。
        clip_margin (float): 負例のハードマージン $m$ (デフォルト: 0.05)。
        eps (float): 数値安定化微小値。
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip_margin: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        """非対称損失層を初期化する。

        Args:
            gamma_neg (float): 負例の変調指数。
            gamma_pos (float): 正例の変調指数。
            clip_margin (float): 負例のハードマージン。
            eps (float): 微小値。
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip_margin = clip_margin
        self.eps = eps

    def forward(self, logits: Tensor, label: Tensor) -> Tensor:
        """2候補スライスに対する非対称損失を算出する。

        Args:
            logits (Tensor): 2候補のロジット `[2]` または `[1, 2]`。
            label (Tensor): 正解インデックススカラー (0: True, 1: False)。

        Returns:
            Tensor: スカラー損失テンソル。
        """
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        delta_z = logits[:, 0] - logits[:, 1]
        target_idx = label.view(-1)
        target_float = (target_idx == 0).to(logits.dtype)

        probs = torch.sigmoid(delta_z)

        # 正例損失 (Focal 形式)
        loss_pos = (
            target_float
            * torch.log(probs.clamp(min=self.eps))
            * torch.pow(1.0 - probs, self.gamma_pos)
        )

        # 負例損失 (マージンシフトによる容易な負例のハードカット)
        probs_neg = (probs - self.clip_margin).clamp(min=0.0)
        loss_neg = (
            (1.0 - target_float)
            * torch.log((1.0 - probs_neg).clamp(min=self.eps))
            * torch.pow(probs_neg, self.gamma_neg)
        )

        return -torch.mean(loss_pos + loss_neg)


class SymmetryRegularizationLoss(nn.Module):
    """候補順序の置換に対する対称性正則化損失層 ($\\mathcal{L}_{\\text{sym}}$)。

    元の入力に対する予測確率分布と、候補順序をシャッフルした入力に対する
    予測確率分布の置換結果との間の KL ダイバージェンスを計算し、位置バイアスを解消する。

    数理仕様:
    $$\\mathcal{L}_{\\text{sym}} = D_{\\text{KL}}(\\pi(\\text{Softmax}(z)) \\parallel \\text{Softmax}(z_\\pi))$$
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """対称性正則化損失層を初期化する。

        Args:
            eps (float): 数値安定化微小値。
        """
        super().__init__()
        self.eps = eps

    def forward(
        self,
        orig_logits: Tensor,
        perm_logits: Tensor,
        perm_indices: Tensor,
        op_mask: Tensor,
    ) -> Tensor:
        """元のロジットと置換ロジット間の対称性 KL 損失を算出する。

        Args:
            orig_logits (Tensor): 元の順序でのロジット `[batch_size, num_options]`。
            perm_logits (Tensor): 置換順序でのロジット `[batch_size, num_options]`。
            perm_indices (Tensor): 置換順序インデックス `[batch_size, num_options]`。
            op_mask (Tensor): 有効候補マスク `[batch_size, num_options]`。

        Returns:
            Tensor: スカラー対称性正則化損失。
        """
        orig_probs = F.softmax(orig_logits.masked_fill(~op_mask, -1e4), dim=-1)
        perm_probs = F.softmax(perm_logits.masked_fill(~op_mask, -1e4), dim=-1)

        # orig_probs を perm_indices で並び替えて perm_probs と比較
        permuted_orig_probs = torch.gather(orig_probs, dim=1, index=perm_indices)

        # 有効要素のみを対象に KL ダイバージェンスを計算
        kl_div = permuted_orig_probs * (
            torch.log(permuted_orig_probs.clamp(min=self.eps))
            - torch.log(perm_probs.clamp(min=self.eps))
        )
        masked_kl = kl_div * op_mask.to(kl_div.dtype)
        return masked_kl.sum(dim=-1).mean()


class InfoNCEContrastiveLoss(nn.Module):
    """State 文脈ベクトルと候補マーカーベクトル間の InfoNCE 類似度対照損失層。

    数理仕様:
    文脈ベクトル $h_{\\text{State}} \\in \\mathbb{R}^D$ と各候補マーカーベクトル $h_{\\text{op}, k} \\in \\mathbb{R}^D$
    の $L_2$ 正規化コサイン類似度を行列化し、正解インデックス $y$ に対するクロスエントロピーを算出する：
    $$\\mathcal{L} = - \\ln \\frac{\\exp(\\text{sim}(h_{\\text{State}}, h_{\\text{op}, y}) / \\tau_c)}{\\sum_{k=0}^{K-1} \\exp(\\text{sim}(h_{\\text{State}}, h_{\\text{op}, k}) / \\tau_c)}$$

    Attributes:
        temperature (float): スケーリング温度パラメータ $\\tau_c$。
    """

    def __init__(self, temperature: float = 0.07) -> None:
        """対照損失層を初期化する。

        Args:
            temperature (float): 温度パラメータ。
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        state_repr: Tensor,
        option_repr: Tensor,
        label: Tensor,
    ) -> Tensor:
        """単一サンプルの特徴量ベクトルに対する対照損失を算出する。

        Args:
            state_repr (Tensor): 文脈ベクトル `[hidden_size]` または `[1, hidden_size]`。
            option_repr (Tensor): 有効候補ベクトル `[num_valid_options, hidden_size]`。
            label (Tensor): 正解候補インデックススカラー。

        Returns:
            Tensor: スカラー損失テンソル。
        """
        if state_repr.dim() == 1:
            state_repr = state_repr.unsqueeze(0)
        if option_repr.size(0) <= 1:
            return torch.tensor(0.0, device=state_repr.device, dtype=state_repr.dtype)

        # L2 正規化 (ゼロ除算防止の eps 付き)
        state_norm = F.normalize(state_repr, p=2, dim=-1, eps=1e-8)
        option_norm = F.normalize(option_repr, p=2, dim=-1, eps=1e-8)

        # コサイン類似度: [1, num_valid_options]
        sim = torch.matmul(state_norm, option_norm.transpose(0, 1)) / self.temperature
        target_idx = label.view(-1)

        return F.cross_entropy(sim, target_idx)


class JevMultiTaskLoss(nn.Module):
    """Jev アーキテクチャ用 数理的複合損失エンジン。

    Choice (動的平滑化 + Focal), Score (RPS / EMD), Noul (ASL / 非対称 BCE),
    および補助対照損失 (InfoNCE) を統合し、異種タスク混在バッチに対して
    サンプル単位で幾何損失を動的ディスパッチして集約する。

    Attributes:
        choice_loss_fn (LabelSmoothedFocalLoss): Choice 用損失層。
        score_loss_fn (nn.Module): Score 用損失層 (RPS または EMD)。
        noul_loss_fn (nn.Module): Noul 用損失層 (ASL または AsymmetricBCE)。
        contrastive_loss_fn (InfoNCEContrastiveLoss): 補助対照損失層。
        contrastive_weight (float): 対照損失の結合係数 $\\lambda_c$。
        choice_weight (float): Choice 損失の重み係数。
        score_weight (float): Score 損失の重み係数。
        noul_weight (float): Noul 損失の重み係数。
    """

    def __init__(
        self,
        label_smoothing: float = 0.05,
        focal_gamma: float = 0.0,
        score_loss_type: str = "rps",
        score_loss_power: int = 2,
        noul_loss_type: str = "asl",
        noul_pos_weight: float = 1.0,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.07,
        choice_weight: float = 1.0,
        score_weight: float = 1.0,
        noul_weight: float = 1.0,
        insent_weight: float = 0.0,
        insent_temperature: float = 0.05,
        insent_lambda_seq: float = 0.2,
    ) -> None:
        """複合損失エンジンを初期化する。

        Args:
            label_smoothing (float): Choice 型のラベル平滑化係数。
            focal_gamma (float): Choice 型の Focal Loss 変調係数。
            score_loss_type (str): Score 型損失種別 ('rps' または 'emd')。
            score_loss_power (int): Score 型 EMD 損失のべき数 (1: Smooth L1, 2: 二乗)。
            noul_loss_type (str): Noul 型損失種別 ('asl' または 'bce')。
            noul_pos_weight (float): Noul 型 BCE 適用時の正例重み係数。
            contrastive_weight (float): 補助対照損失の重み係数 (0.0 で無効化)。
            contrastive_temperature (float): 補助対照損失の温度パラメータ。
            choice_weight (float): Choice 型損失の重み係数。
            score_weight (float): Score 型損失の重み係数。
            noul_weight (float): Noul 型損失の重み係数。
            insent_weight (float): InSeNT 対照損失の重み係数 (0.0 で無効化, 推奨: 0.15)。
            insent_temperature (float): InSeNT 対照損失の温度パラメータ。
            insent_lambda_seq (float): InSeNT の In-Sequence 損失比率。
        """
        super().__init__()
        self.choice_loss_fn = LabelSmoothedFocalLoss(
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
        )

        if score_loss_type.lower() == "emd":
            self.score_loss_fn: nn.Module = EarthMoverDistanceLoss(
                power=score_loss_power
            )
        else:
            self.score_loss_fn = RankedProbabilityScoreLoss()

        if noul_loss_type.lower() == "bce":
            self.noul_loss_fn: nn.Module = AsymmetricBCELoss(pos_weight=noul_pos_weight)
        else:
            self.noul_loss_fn = AsymmetricLoss()

        self.contrastive_loss_fn = InfoNCEContrastiveLoss(
            temperature=contrastive_temperature
        )
        self.insent_loss_fn = InSeNTLoss(
            temperature=insent_temperature,
            lambda_seq=insent_lambda_seq,
        )

        self.contrastive_weight = contrastive_weight
        self.choice_weight = choice_weight
        self.score_weight = score_weight
        self.noul_weight = noul_weight
        self.insent_weight = insent_weight

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        op_mask: Tensor,
        question_types: list[str | QuestionType] | None = None,
        state_repr: Tensor | None = None,
        option_repr: Tensor | None = None,
        query_embeddings: Tensor | None = None,
        chunk_embeddings: Tensor | None = None,
        target_chunk_indices: Tensor | None = None,
        chunk_mask: Tensor | None = None,
        return_dict: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, float]]:
        """異種タスク混在バッチに対する複合数理損失を算出する。

        内部ロジック:
        1. 各サンプル $i$ の有効候補数 $K_i$ を `op_mask` から取得し、スライス抽出する。
        2. タスク種別 (`choice`, `score`, `noul`) に応じて各サブ損失を計算する。
        3. `contrastive_weight > 0` かつ特徴量が提供されている場合、有効候補マーカーとの InfoNCE 類似度損失を算出する。
        4. タスクごとに平均損失を求め、タスク重みで合算する (ゼロ除算ガード適用)。
        5. `insent_weight > 0` かつチャンク情報が提供されている場合、InSeNT 対照損失を加算する。
        6. `return_dict=True` の場合はメトリクス内訳辞書を併せて返却する。

        Args:
            logits (Tensor): ロジットテンソル `[batch_size, max_options]`。
            labels (Tensor): 正解インデックス `[batch_size]`。
            op_mask (Tensor): 有効候補マスク `[batch_size, max_options]`。
            question_types (list[str | QuestionType] | None): 各サンプルの質問種別リスト。未指定時は全て Choice として処理。
            state_repr (Tensor | None): 文脈表現ベクトル `[batch_size, hidden_size]`。
            option_repr (Tensor | None): 候補マーカーベクトル `[batch_size, max_options, hidden_size]`。
            query_embeddings (Tensor | None): 質問表現ベクトル `[batch_size, hidden_size]` (InSeNT 用)。
            chunk_embeddings (Tensor | None): 集約チャンク表現 `[batch_size, num_chunks, hidden_size]`。
            target_chunk_indices (Tensor | None): 各サンプルの正解条項インデックス `[batch_size]`。
            chunk_mask (Tensor | None): 有効チャンクマスク `[batch_size, num_chunks]`。
            return_dict (bool): メトリクス辞書を同時に返却するかどうか。


        Returns:
            Tensor | tuple[Tensor, dict[str, float]]:
                - 通常時: スカラー総損失テンソル。
                - 辞書返却時: (total_loss, loss_dict) のタプル。
        """

        batch_size = logits.size(0)
        device = logits.device

        choice_losses: list[Tensor] = []
        score_losses: list[Tensor] = []
        noul_losses: list[Tensor] = []
        contrast_losses: list[Tensor] = []

        should_compute_contrast = (
            self.contrastive_weight > 0.0
            and state_repr is not None
            and option_repr is not None
        )

        for i in range(batch_size):
            # 有効候補数の取得とスライシング (パディング完全遮断)
            num_valid = int(op_mask[i].sum().item())
            if num_valid == 0:
                continue

            l_i = logits[i, :num_valid]
            target_i = labels[i]

            # タスク種別の正規化
            if question_types is not None:
                q_type = question_types[i]
                q_type_str = (
                    q_type.value
                    if isinstance(q_type, QuestionType)
                    else str(q_type).lower()
                )
            else:
                q_type_str = QuestionType.CHOICE.value

            if q_type_str == QuestionType.SCORE.value or q_type_str == "score":
                loss_i = self.score_loss_fn(l_i, target_i)
                score_losses.append(loss_i)
            elif q_type_str == QuestionType.NOUL.value or q_type_str == "noul":
                loss_i = self.noul_loss_fn(l_i, target_i)
                noul_losses.append(loss_i)
            else:
                # デフォルトは Choice
                loss_i = self.choice_loss_fn(l_i, target_i)
                choice_losses.append(loss_i)

            # 補助対照損失の計算
            if should_compute_contrast:
                assert state_repr is not None
                assert option_repr is not None
                s_rep_i = state_repr[i]
                op_rep_i = option_repr[i, :num_valid]
                c_loss_i = self.contrastive_loss_fn(s_rep_i, op_rep_i, target_i)
                contrast_losses.append(c_loss_i)

        # タスク別平均損失の算出 (ゼロ除算防止)
        mean_choice = (
            torch.stack(choice_losses).mean()
            if choice_losses
            else torch.tensor(0.0, device=device)
        )
        mean_score = (
            torch.stack(score_losses).mean()
            if score_losses
            else torch.tensor(0.0, device=device)
        )
        mean_noul = (
            torch.stack(noul_losses).mean()
            if noul_losses
            else torch.tensor(0.0, device=device)
        )
        mean_contrast = (
            torch.stack(contrast_losses).mean()
            if contrast_losses
            else torch.tensor(0.0, device=device)
        )

        # タスク損失の加重合成
        total_task_weight = 0.0
        weighted_task_loss = torch.tensor(0.0, device=device)

        if choice_losses:
            weighted_task_loss = weighted_task_loss + self.choice_weight * mean_choice
            total_task_weight += self.choice_weight
        if score_losses:
            weighted_task_loss = weighted_task_loss + self.score_weight * mean_score
            total_task_weight += self.score_weight
        if noul_losses:
            weighted_task_loss = weighted_task_loss + self.noul_weight * mean_noul
            total_task_weight += self.noul_weight

        if total_task_weight > 0.0:
            task_loss = weighted_task_loss / total_task_weight
        else:
            task_loss = torch.tensor(0.0, device=device)

        # 補助対照損失の加算
        total_loss = task_loss
        if contrast_losses and self.contrastive_weight > 0.0:
            total_loss = total_loss + self.contrastive_weight * mean_contrast

        # InSeNT 対照損失の加算 (長系列 & Late Chunking 時)
        insent_loss_val = 0.0
        if (
            self.insent_weight > 0.0
            and chunk_embeddings is not None
            and target_chunk_indices is not None
        ):
            # 質問側表現: query_embeddings または state_repr または option_repr の平均。
            q_emb = (
                query_embeddings
                if query_embeddings is not None
                else (
                    state_repr
                    if state_repr is not None
                    else (option_repr.mean(dim=1) if option_repr is not None else None)
                )
            )
            if q_emb is None:
                raise ValueError(
                    "InSeNT 損失の計算には質問側の表現 (query_embeddings, state_repr, または option_repr) が必須です。"
                    "chunk_embeddings への自己対照フォールバックは数理的に推奨されません。"
                )

            insent_loss = self.insent_loss_fn(
                query_embeddings=q_emb,
                chunk_embeddings=chunk_embeddings,
                target_chunk_indices=target_chunk_indices,
                chunk_mask=chunk_mask,
            )
            total_loss = total_loss + self.insent_weight * insent_loss
            insent_loss_val = float(insent_loss.detach().item())

        if not return_dict:
            return total_loss

        loss_dict: dict[str, float] = {
            "loss": float(total_loss.detach().item()),
            "loss_choice": float(mean_choice.detach().item()) if choice_losses else 0.0,
            "loss_score": float(mean_score.detach().item()) if score_losses else 0.0,
            "loss_noul": float(mean_noul.detach().item()) if noul_losses else 0.0,
            "loss_contrast": float(mean_contrast.detach().item())
            if contrast_losses
            else 0.0,
            "loss_insent": insent_loss_val,
        }

        return total_loss, loss_dict
