"""SFT 学習設定モジュール。

教師あり指示学習(SFT)のハイパーパラメータおよび入出力設定を定義し、
YAML および JSON による相互シリアライズと厳密な再現性を担保する。
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from contract import MAX_SEQUENCE_LENGTH
from models.backbone import DEFAULT_BACKBONE_MODEL_ID

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """SFT 学習ループの包括的設定 dataclass。

    Attributes:
        model_name_or_path (str): バックボーンモデル識別子またはローカルパス。
        mlp_hidden_size (int | None): デシジョンヘッドの MLP 中間次元数。None の場合は隠れ層次元と同一。
        max_sequence_length (int): トークナイズ時の最大許容系列長。
        dataset_names (list[str]): 学習に使用するデータセット名のリスト。
        negative_ratio (float): 合成ネガティブサンプルの混入比率。
        max_samples_per_dataset (int | None): データセットごとの最大取得サンプル数。データバランシングや高速検証に使用。
        batch_size (int): デバイスあたりのミニバッチサイズ。
        gradient_accumulation_steps (int): 勾配累積ステップ数。
        learning_rate_backbone (float): 事前学習済みバックボーン Transformer レイヤーの学習率。
        learning_rate_embed (float): [OP] を含む入力埋め込み層の学習率。
        learning_rate_head (float): デシジョンヘッド (および OptionGatherLayer) の学習率。
        weight_decay (float): オプティマイザの重み減衰率。
        num_epochs (int): 学習エポック数。
        warmup_ratio (float): ウォームアップステップの比率。
        mixed_precision (str): 混合精度モード ('no', 'fp16', 'bf16')。
        max_grad_norm (float): 勾配クリッピングの最大ノルム。
        early_stopping_patience (int | None): 早期終了の許容エポック数。None の場合は早期終了なし。
        eval_metric (str): 最良チェックポイントおよび早期終了の判定に使用するメトリクス名 ('composite_metric', 'choice_accuracy', 'accuracy', 'loss' 等)。
        evaluate_position_bias (bool): 検証フェーズで Choice 型の選択肢順序シャッフル不変性 (位置バイアス) を自動評価するかどうか。
        seed (int): 乱数シード。
        output_dir (str): 成果物およびログのベース出力ディレクトリ。
        label_smoothing (float): Choice 型のラベル平滑化係数。
        focal_gamma (float): Choice 型の Focal Loss 変調係数。0.0 で無効化。
        score_loss_power (int): Score 型 EMD 損失のべき数 (1: Smooth L1 / L1, 2: 二乗 Wasserstein)。
        noul_pos_weight (float): Noul 型 BCE 損失における正例 (True) の重み係数。
        contrastive_weight (float): 補助対照損失 (InfoNCE) の重み係数。0.0 で無効化。
        contrastive_temperature (float): 補助対照損失の温度パラメータ。
        choice_weight (float): マルチタスク損失における Choice 型の損失重み。
        score_weight (float): マルチタスク損失における Score 型の損失重み。
        noul_weight (float): マルチタスク損失における Noul 型の損失重み。
    """

    model_name_or_path: str = DEFAULT_BACKBONE_MODEL_ID
    mlp_hidden_size: int | None = None
    max_sequence_length: int = MAX_SEQUENCE_LENGTH
    gradient_checkpointing: bool = False

    dataset_names: list[str] = field(
        default_factory=lambda: [
            "jglue_marc_ja",
            "jglue_jnli",
            "jglue_jsts",
            "jglue_jcommonsenseqa",
        ]
    )
    negative_ratio: float = 0.15
    max_samples_per_dataset: int | None = 5000

    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    learning_rate_backbone: float = 2e-5
    learning_rate_embed: float = 5e-5
    learning_rate_head: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    mixed_precision: str = "no"
    max_grad_norm: float = 1.0
    early_stopping_patience: int | None = 3
    eval_metric: str = "composite_metric"
    evaluate_position_bias: bool = True

    # JevMultiTaskLoss 関連ハイパーパラメータ
    label_smoothing: float = 0.05
    focal_gamma: float = 0.0
    score_loss_power: int = 2
    noul_pos_weight: float = 1.0
    contrastive_weight: float = 0.0
    contrastive_temperature: float = 0.07
    choice_weight: float = 1.0
    score_weight: float = 1.0
    noul_weight: float = 1.0

    # 長系列拡張 & Late Chunking / InSeNT パラメータ
    use_yarn: bool = False
    original_max_position: int = 8192
    target_max_position: int = 16384
    yarn_alpha: float = 1.0
    yarn_beta: float = 32.0
    pooling_strategy: str = "mean"
    top_k_chunks: int = 5
    insent_weight: float = 0.0
    insent_temperature: float = 0.05
    insent_lambda_seq: float = 0.2

    seed: int = 42
    output_dir: str = "runs/sft"

    def to_dict(self) -> dict[str, Any]:
        """設定を辞書型に変換する。

        Returns:
            dict[str, Any]: 全フィールドの辞書表現。
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SFTConfig":
        """辞書型データから設定インスタンスを復元する。

        Args:
            data (dict[str, Any]): 設定パラメータ辞書。

        Returns:
            SFTConfig: 復元された設定インスタンス。
        """
        known_keys = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in known_keys}
        return cls(**filtered_data)

    def to_yaml(self) -> str:
        """設定を YAML 文字列として出力する。

        Returns:
            str: 整形された YAML 文字列。
        """
        return yaml.dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def save_yaml(self, path: Path | str) -> Path:
        """設定を YAML ファイルとして保存する。

        Args:
            path (Path | str): 保存先ファイルパス。

        Returns:
            Path: 保存されたファイルパス。
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(self.to_yaml(), encoding="utf-8")
        logger.info("SFT 設定を YAML に保存しました: %s。", save_path)
        return save_path

    @classmethod
    def from_yaml(cls, path_or_content: Path | str) -> "SFTConfig":
        """YAML ファイルパスまたは YAML 文字列から設定を読み込む。

        Args:
            path_or_content (Path | str): ファイルパスまたは YAML 文字列。

        Returns:
            SFTConfig: 復元された設定インスタンス。
        """
        path = Path(path_or_content)
        if path.is_file():
            content = path.read_text(encoding="utf-8")
        else:
            content = str(path_or_content)
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            raise TypeError(f"YAML のパース結果が辞書型ではありません: {type(data)}。")
        return cls.from_dict(data)

    def to_json(self, indent: int = 2) -> str:
        """設定を JSON 文字列として出力する。

        Args:
            indent (int): インデント幅。

        Returns:
            str: 整形された JSON 文字列。
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, path: Path | str) -> Path:
        """設定を JSON ファイルとして保存する。

        Args:
            path (Path | str): 保存先ファイルパス。

        Returns:
            Path: 保存されたファイルパス。
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(self.to_json(), encoding="utf-8")
        logger.info("SFT 設定を JSON に保存しました: %s。", save_path)
        return save_path

    @classmethod
    def from_json(cls, path_or_content: Path | str) -> "SFTConfig":
        """JSON ファイルパスまたは JSON 文字列から設定を読み込む。

        Args:
            path_or_content (Path | str): ファイルパスまたは JSON 文字列。

        Returns:
            SFTConfig: 復元された設定インスタンス。
        """
        path = Path(path_or_content)
        if path.is_file():
            content = path.read_text(encoding="utf-8")
        else:
            content = str(path_or_content)
        data = json.loads(content)
        if not isinstance(data, dict):
            raise TypeError(f"JSON のパース結果が辞書型ではありません: {type(data)}。")
        return cls.from_dict(data)

    def save(self, path: Path | str) -> Path:
        """拡張子に応じて YAML または JSON で設定を自動保存する。

        Args:
            path (Path | str): 保存先ファイルパス。

        Returns:
            Path: 保存されたファイルパス。
        """
        save_path = Path(path)
        if save_path.suffix.lower() in [".yaml", ".yml"]:
            return self.save_yaml(save_path)
        return self.save_json(save_path)
