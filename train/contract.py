"""成果物引き渡し契約(Artifact Contract)モジュール。

Python 学習・エクスポート側と Rust ランタイム推論エンジン側で合意された、
ONNX モデルのテンソル名・形状仕様およびキャリブレーション設定(`calibration.json`)を定義する。
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

TENSOR_INPUT_IDS: str = "input_ids"
"""入力トークンID列テンソル名。"""

TENSOR_ATTENTION_MASK: str = "attention_mask"
"""アテンションマスクテンソル名。"""

TENSOR_OP_INDICES: str = "op_indices"
"""オプションマーカー位置インデックステンソル名。"""

TENSOR_LOGITS: str = "logits"
"""決定ヘッド出力ロジットテンソル名。"""

TENSOR_STATE_HIDDEN_STATES: str = "state_hidden_states"
"""State エンコーダが出力し、Query 評価器が参照する最終層隠れ状態テンソル名。"""

TENSOR_STATE_MASK: str = "state_mask"
"""State 系列の有効トークンマスクテンソル名。"""

TENSOR_QUERY_IDS: str = "query_ids"
"""質問系列の入力トークンID列テンソル名。"""

TENSOR_QUERY_MASK: str = "query_mask"
"""質問系列のアテンションマスクテンソル名。"""

TOKEN_OPTION_MARKER: str = "[OP]"
"""各候補の先頭に付与される決定アンカー特殊トークン。"""

MAX_SEQUENCE_LENGTH: int = 8192
"""許容される最大入力トークン系列長。"""

MAX_NUM_OPTIONS: int = 255
"""単一質問あたりの最大候補数。"""

MIN_NUM_OPTIONS: int = 1
"""単一質問あたりの最小候補数。"""

DEFAULT_HIGH_CONFIDENCE_THRESHOLD: float = 0.70
"""自動実行 (AutoExecute) と判定するための高確信度下限閾値。"""

DEFAULT_LOW_CONFIDENCE_THRESHOLD: float = 0.35
"""確認・二次検証要求 (ConfirmOrEscalate) と判定するための中確信度下限閾値。"""

DEFAULT_TOP_MARGIN_THRESHOLD: float = 0.15
"""上位2候補の最小確率マージン閾値。"""

DEFAULT_ENERGY_THRESHOLD: float = -1.0
"""OOD 判定用のデフォルト正規化自由エネルギー閾値。"""


def compute_normalized_entropy(probabilities: list[float] | tuple[float, ...]) -> float:
    """確率分布から候補数 K に非依存な正規化シャノンエントロピー H_norm を算出する。

    数理仕様:
    $$H_{\\text{norm}}(p) = \\begin{cases} 0.0 & (K = 1) \\\\ \\frac{-\\sum_{k=1}^K p_k \\ln p_k}{\\ln K} & (K \\ge 2) \\end{cases}$$

    Args:
        probabilities (list[float] | tuple[float, ...]): 各候補の確率値シーケンス。

    Returns:
        float: [0.0, 1.0] にクランプされた正規化エントロピー。

    Raises:
        ValueError: 配列が空、または負数や非有限値が含まれている場合。
    """
    import math

    k = len(probabilities)
    if k == 0:
        raise ValueError("確率分布配列が空です。")
    if k == 1:
        p0 = probabilities[0]
        if not math.isfinite(p0) or p0 < 0.0:
            raise ValueError(f"不正な確率値が含まれています: {p0}。")
        return 0.0

    entropy = 0.0
    for p in probabilities:
        if not math.isfinite(p) or p < 0.0:
            raise ValueError(f"不正な確率値が含まれています: {p}。")
        if p > 0.0:
            entropy -= p * math.log(p)

    max_entropy = math.log(k)
    if max_entropy <= 0.0:
        return 0.0
    h_norm = entropy / max_entropy
    return max(0.0, min(1.0, float(h_norm)))


def compute_top_margin(probabilities: list[float] | tuple[float, ...]) -> float:
    """確率分布から上位2候補の確率差 (Top-Margin) M(p) を算出する。

    数理仕様:
    $$M(p) = \\begin{cases} 1.0 & (K = 1) \\\\ p_{(1)} - p_{(2)} & (K \\ge 2) \\end{cases}$$

    Args:
        probabilities (list[float] | tuple[float, ...]): 各候補の確率値シーケンス。

    Returns:
        float: [0.0, 1.0] にクランプされた Top-Margin。

    Raises:
        ValueError: 配列が空、または負数や非有限値が含まれている場合。
    """
    import math

    k = len(probabilities)
    if k == 0:
        raise ValueError("確率分布配列が空です。")
    if k == 1:
        p0 = probabilities[0]
        if not math.isfinite(p0) or p0 < 0.0:
            raise ValueError(f"不正な確率値が含まれています: {p0}。")
        return 1.0

    max1 = -math.inf
    max2 = -math.inf
    for p in probabilities:
        if not math.isfinite(p) or p < 0.0:
            raise ValueError(f"不正な確率値が含まれています: {p}。")
        if p > max1:
            max2 = max1
            max1 = p
        elif p > max2:
            max2 = p

    margin = max1 - max2
    return max(0.0, min(1.0, float(margin)))


def compute_composite_confidence(
    probabilities: list[float] | tuple[float, ...],
) -> float:
    """正規化エントロピーと Top-Margin を統合した複合確信度スコア S_confidence を算出する。

    数理仕様:
    $$S_{\\text{confidence}}(p) = (1.0 - H_{\\text{norm}}(p)) \\times M(p)$$

    Args:
        probabilities (list[float] | tuple[float, ...]): 各候補の確率値シーケンス。

    Returns:
        float: [0.0, 1.0] にクランプされた複合確信度スコア。

    Raises:
        ValueError: 配列が空、または負数や非有限値が含まれている場合。
    """
    import math

    k = len(probabilities)
    if k == 0:
        raise ValueError("確率分布配列が空です。")
    if k == 1:
        p0 = probabilities[0]
        if not math.isfinite(p0) or p0 < 0.0:
            raise ValueError(f"不正な確率値が含まれています: {p0}。")
        return 1.0

    h_norm = compute_normalized_entropy(probabilities)
    margin = compute_top_margin(probabilities)
    score = (1.0 - h_norm) * margin
    return max(0.0, min(1.0, float(score)))


@dataclass
class GatingThresholds:
    """ゲーティング判定用閾値設定。

    Attributes:
        high_threshold (float): 高確信度下限閾値 (自動実行境界)。
        low_threshold (float): 中確信度下限閾値 (確認要求境界)。
        top_margin_threshold (float): 上位2候補確率マージン閾値。
        ood_enabled (bool): OOD 安全弁を有効化するかどうか。
        energy_threshold (float): OOD 判定用の正規化自由エネルギー閾値。
        energy_temperature (float | None): OOD 自由エネルギー算出専用の温度パラメータ。
        loose_energy_threshold (float | None): ハイブリッド OOD 判定用の緩和正規化自由エネルギー閾値。
        ood_min_confidence (float | None): ハイブリッド OOD 判定用の最小確信度閾値。
        ood_min_margin (float | None): ハイブリッド OOD 判定用の最小マージン閾値。
    """

    high_threshold: float = DEFAULT_HIGH_CONFIDENCE_THRESHOLD
    low_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD
    top_margin_threshold: float = DEFAULT_TOP_MARGIN_THRESHOLD
    ood_enabled: bool = True
    energy_threshold: float = DEFAULT_ENERGY_THRESHOLD
    energy_temperature: float | None = None
    loose_energy_threshold: float | None = None
    ood_min_confidence: float | None = None
    ood_min_margin: float | None = None


@dataclass
class TemperatureMap:
    """質問プリミティブ別・候補数バケット別の最適温度パラメータマップ。

    Attributes:
        choice (dict[str, float]): Choice 型の候補数バケット別温度テーブル。
        score (dict[str, float]): Score 型の段階数バケット別温度テーブル。
        noul (float): Noul 型の二値判定温度係数。
    """

    choice: dict[str, float] = field(
        default_factory=lambda: {
            "2": 1.05,
            "3-5": 1.12,
            "6-10": 1.20,
            "11+": 1.35,
        }
    )
    score: dict[str, float] = field(
        default_factory=lambda: {
            "2-5": 1.00,
            "6-10": 1.08,
        }
    )
    noul: float = 1.0


@dataclass
class CalibrationConfig:
    """成果物引き渡し用 `calibration.json` のデータ構造。

    Attributes:
        version (str): キャリブレーションスキーマのバージョン。
        default_temperature (float): バケットに合致しない場合に使用するフォールバック温度。
        temperature_map (TemperatureMap): プリミティブ別の温度テーブル。
        gating_thresholds (GatingThresholds): 確信度ゲーティング閾値設定。
    """

    version: str = "1.0"
    default_temperature: float = 1.0
    temperature_map: TemperatureMap = field(default_factory=TemperatureMap)
    gating_thresholds: GatingThresholds = field(default_factory=GatingThresholds)

    def to_dict(self) -> dict[str, Any]:
        """辞書オブジェクトへ変換する。

        Returns:
            dict[str, Any]: シリアライズ可能な辞書表現。
        """
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """JSON 文字列へ変換する。

        Args:
            indent (int): インデント幅。

        Returns:
            str: 整形された JSON 文字列。
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: Path | str) -> None:
        """ファイルへ保存する。

        Args:
            path (Path | str): 出力先ファイルパス。
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.to_json() + "\n")

    @classmethod
    def load(cls, path: Path | str) -> "CalibrationConfig":
        """JSON ファイルから設定を読み込む。

        Args:
            path (Path | str): 読み込み元ファイルパス。

        Returns:
            CalibrationConfig: 読み込まれたキャリブレーション設定。
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        temp_map = TemperatureMap(
            choice=data.get("temperature_map", {}).get("choice", {}),
            score=data.get("temperature_map", {}).get("score", {}),
            noul=data.get("temperature_map", {}).get("noul", 1.0),
        )
        raw_thresholds = data.get("gating_thresholds", {})
        gating_thresholds = GatingThresholds(
            high_threshold=raw_thresholds.get(
                "high_threshold", DEFAULT_HIGH_CONFIDENCE_THRESHOLD
            ),
            low_threshold=raw_thresholds.get(
                "low_threshold", DEFAULT_LOW_CONFIDENCE_THRESHOLD
            ),
            top_margin_threshold=raw_thresholds.get(
                "top_margin_threshold", DEFAULT_TOP_MARGIN_THRESHOLD
            ),
        )
        return cls(
            version=data.get("version", "1.0"),
            default_temperature=data.get("default_temperature", 1.0),
            temperature_map=temp_map,
            gating_thresholds=gating_thresholds,
        )
