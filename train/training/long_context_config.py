"""長系列拡張および Late Chunking / InSeNT 学習設定モジュール。

YaRN による局所窓保護 RoPE スケーリング、Late Chunking プーリング、
および In-Sequence & In-Batch 対照損失 (InSeNT) のハイパーパラメータを定義する。
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class LongContextConfig:
    """長系列拡張および Late Chunking / InSeNT パイプラインの設定 dataclass。

    Attributes:
        use_yarn (bool): YaRN による RoPE スケーリングを有効化するかどうか。
        original_max_position (int): 事前学習時の最大系列長 (ModernBERT は 8,192)。
        target_max_position (int): 拡張後の目標系列長 (例: 16,384)。
        yarn_alpha (float): YaRN 低周波補間の開始しきい値。
        yarn_beta (float): YaRN 高周波保護の完了しきい値。
        global_rope_theta (float): 大域注意層の RoPE 基底値 (160,000.0)。
        local_rope_theta (float): 局所スライディング窓層の RoPE 基底値 (10,000.0, 固定保護)。
        pooling_strategy (Literal["mean", "attention"]): Late Chunking のプーリング方式。
        top_k_chunks (int): Top-k Cross-Attention で選択する上位チャンク数。
        use_insent (bool): InSeNT 対照損失をマルチタスク学習に導入するかどうか。
        lambda_seq (float): In-Sequence 損失 L_seq のブレンド比率 (1 - lambda_seq が L_batch)。
        lambda_insent (float): マルチタスク総合損失における InSeNT 損失の加重係数 (推奨: 0.15)。
        insent_temperature (float): InSeNT コサイン類似度 Softmax の温度パラメータ。
        gradient_checkpointing (bool): 長系列訓練時の VRAM 節約のための勾配チェックポインティング有効化。
    """

    # YaRN RoPE スケーリング設定
    use_yarn: bool = True
    original_max_position: int = 8192
    target_max_position: int = 16384
    yarn_alpha: float = 1.0
    yarn_beta: float = 32.0
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 10000.0

    # Late Chunking プーリング設定
    pooling_strategy: Literal["mean", "attention"] = "mean"
    top_k_chunks: int = 5

    # InSeNT 対照損失設定
    use_insent: bool = True
    lambda_seq: float = 0.2
    lambda_insent: float = 0.15
    insent_temperature: float = 0.05

    # メモリ最適化
    gradient_checkpointing: bool = True

    def to_dict(self) -> dict[str, Any]:
        """設定を辞書形式に変換する。

        Returns:
            dict[str, Any]: 設定辞書。
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LongContextConfig":
        """辞書から設定オブジェクトを生成する。未知のキーは無視する。

        Args:
            data (dict[str, Any]): 設定辞書。

        Returns:
            LongContextConfig: 生成された設定インスタンス。
        """
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_yaml(self, path: str | Path) -> None:
        """設定を YAML ファイルとして保存する。

        Args:
            path (str | Path): 保存先パス。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.to_dict(), f, default_flow_style=False, allow_unicode=True
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "LongContextConfig":
        """YAML ファイルから設定を読み込む。

        Args:
            path (str | Path): 読み込み元 YAML パス。

        Returns:
            LongContextConfig: 生成された設定インスタンス。
        """
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_json(self, path: str | Path) -> None:
        """設定を JSON ファイルとして保存する。

        Args:
            path (str | Path): 保存先パス。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str | Path) -> "LongContextConfig":
        """JSON ファイルから設定を読み込む。

        Args:
            path (str | Path): 読み込み元 JSON パス。

        Returns:
            LongContextConfig: 生成された設定インスタンス。
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
