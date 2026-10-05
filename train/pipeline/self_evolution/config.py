"""自己進化蒸留パイプライン設定モジュール。

難例ログ抽出、HMAC 疑似匿名化、Dual-LLM 検証、80:20 リプレイバッファ、
Proper Scoring Rules 蒸留、DP-SGD、および品質ゲートのハイパーパラメータを定義する。
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AnonymizerConfig:
    """HMAC エンティティ一貫疑似匿名化設定。

    Attributes:
        salt (str): セッションまたはリクエスト単位の HMAC-SHA256 ソルト文字列。
        mask_person (bool): 人名エンティティの仮名化を行うか。
        mask_org (bool): 組織名・企業名エンティティの仮名化を行うか。
        mask_phone (bool): 電話番号の仮名化を行うか。
        mask_financial (bool): 口座・カード番号の仮名化を行うか。
        mask_email (bool): メールアドレスの仮名化を行うか。
        mask_date (bool): 日付表現の正規化・仮名化を行うか。
    """

    salt: str = "sokuto-self-evolution-secret-salt"
    mask_person: bool = True
    mask_org: bool = True
    mask_phone: bool = True
    mask_financial: bool = True
    mask_email: bool = True
    mask_date: bool = False


@dataclass(frozen=True)
class VerifierConfig:
    """Dual-LLM ブラインドクロスチェック検証設定。

    Attributes:
        primary_model (str): 第 1 検証モデル識別子 (例: 'claude-3-7-sonnet-20250219')。
        secondary_model (str): 第 2 検証モデル識別子 (例: 'gpt-4o')。
        strict_match_required (bool): 離散ラベルの完全一致を必須とするか。
        temperature (float): 検証推論時のサンプリング温度 (決定論的評価のため 0.0 推奨)。
        max_tokens (int): 最大生成トークン数。
        blind_mode (bool): System 1 の事前判定を完全隠蔽するブラインドプロンプトを強制するか。
    """

    primary_model: str = "claude-3-7-sonnet-20250219"
    secondary_model: str = "gpt-4o"
    strict_match_required: bool = True
    temperature: float = 0.0
    max_tokens: int = 1500
    blind_mode: bool = True


@dataclass(frozen=True)
class ReplayConfig:
    """80:20 黄金比リプレイバッファ設定。

    Attributes:
        gold_ratio (float): バッチ内の本番認定済み教師データ比率 (デフォルト: 0.80)。
        silver_ratio (float): バッチ内の新規難例データ比率 (デフォルト: 0.20)。
        buffer_capacity (int): リプレイバッファの最大保持サンプル数。
        reservoir_sampling (bool): リザーバサンプリングによる均一更新を行うか。
    """

    gold_ratio: float = 0.80
    silver_ratio: float = 0.20
    buffer_capacity: int = 10000
    reservoir_sampling: bool = True

    def __post_init__(self) -> None:
        """比率の合計が 1.0 に近似することを検証する。"""
        total = self.gold_ratio + self.silver_ratio
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"Gold 比率 ({self.gold_ratio}) と Silver 比率 ({self.silver_ratio}) の合計は 1.0 である必要があります。"
            )


@dataclass(frozen=True)
class DistillationConfig:
    """Proper Scoring Rules 蒸留損失および最適化設定。

    Attributes:
        temperature (float): ソフトラベル蒸留における温度パラメータ $\\tau$。
        alpha (float): ハード損失とソフト蒸留損失の線形補間比率 $\\alpha$ (0.0: ハードのみ, 1.0: ソフトのみ)。
        score_rps_weight (float): Score タスクにおける RPS 損失の重み係数。
        odir_lambda (float): Score タスクにおける過信抑制 ODIR 正則化係数 $\\lambda_{\\text{ODIR}}$。
        freeze_layers (int): バックボーンで固定 (Freeze) する下位トランスフォーマー層数。
        learning_rate (float): ファインチューニング学習率。
        weight_decay (float): 重み減衰係数。
        batch_size (int): 学習バッチサイズ。
        num_epochs (int): 学習エポック数。
    """

    temperature: float = 2.0
    alpha: float = 0.5
    score_rps_weight: float = 1.0
    odir_lambda: float = 0.01
    freeze_layers: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    num_epochs: int = 3


@dataclass(frozen=True)
class DifferentialPrivacyConfig:
    """差分プライバシー (DP-SGD) 設定。

    Attributes:
        enabled (bool): DP-SGD を有効化するか。
        max_grad_norm (float): サンプル単位勾配の L2 クリッピング閾値 $C$。
        noise_multiplier (float): ガウスノイズ倍率 $\\sigma$。
        target_epsilon (float): 目標プライバシー予算 $\\epsilon$。
        target_delta (float): 目標許容破壊確率 $\\delta$。
    """

    enabled: bool = True
    max_grad_norm: float = 1.0
    noise_multiplier: float = 0.8
    target_epsilon: float = 3.0
    target_delta: float = 1e-5


@dataclass(frozen=True)
class QualityGateConfig:
    """回帰テスト品質ゲート設定。

    Attributes:
        max_regression_rate (float): 既存タスクに対する許容精度低下率 (0.00: 劣化ゼロを厳格要求)。
        max_ece (float): 許容される最大 Expected Calibration Error (ECE)。
        min_escalation_reduction_rate (float): 難例に対する最小エスカレーション削減率 (目標 $\\ge 30.0\\%$)。
        contrast_suite_path (str | None): コアテストスイートの Contrast Sets パス。
    """

    max_regression_rate: float = 0.00
    max_ece: float = 0.06
    min_escalation_reduction_rate: float = 0.30
    contrast_suite_path: str | None = None


@dataclass(frozen=True)
class SelfEvolutionConfig:
    """自己進化蒸留パイプライン全体の統合設定。

    Attributes:
        anonymizer: 疑似匿名化設定。
        verifier: Dual-LLM 検証設定。
        replay: リプレイバッファ設定。
        distillation: 蒸留損失・学習設定。
        dp: 差分プライバシー設定。
        gate: 品質ゲート設定。
        output_dir: チェックポイントおよび検証レポートの出力先ディレクトリ。
    """

    anonymizer: AnonymizerConfig = field(default_factory=AnonymizerConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    dp: DifferentialPrivacyConfig = field(default_factory=DifferentialPrivacyConfig)
    gate: QualityGateConfig = field(default_factory=QualityGateConfig)
    output_dir: Path = Path("runs/self_evolution")
