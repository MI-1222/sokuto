"""不確実性メタデータおよび型安全ラッパーモジュール。

sokuto の判定結果に含まれる較正済み確信度、確率分布、
ヘルムホルツ自由エネルギー、正規化エントロピー等の意思決定メタデータを
Python の型空間へ安全にバインドするためのコンテナを提供する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, get_args

from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import core_schema


class SokutoMeta(BaseModel):
    """sokuto 意思決定の不確実性・キャリブレーション統計情報コンテナ。

    Attributes:
        confidence (float): 較正済み確信度 (0.0 <= conf <= 1.0)。
        probabilities (dict[str, float] | None): 全候補または全段階の確率分布マップ。
        free_energy (float | None): ヘルムホルツ自由エネルギー (OOD 検知指標)。
        normalized_entropy (float | None): 正規化シャノンエントロピー (0.0 <= H <= 1.0)。
        top_margin (float | None): 上位2候補の確率差 (Top-Margin)。
        route (str | None): ゲーティング判定ルート ('auto_execute', 'escalate', 等)。
        effective_confidence (float): Noul 型等の尖り度を反映した実効確信度。
    """

    model_config = ConfigDict(frozen=True)

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="較正済み確信度。",
    )
    probabilities: dict[str, float] | None = Field(
        default=None,
        description="全候補の確率分布マップ。",
    )
    free_energy: float | None = Field(
        default=None,
        description="ヘルムホルツ自由エネルギー。",
    )
    normalized_entropy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="正規化シャノンエントロピー。",
    )
    top_margin: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="上位2候補の確率差。",
    )
    route: str | None = Field(
        default=None,
        description="ルーティング判定結果種別。",
    )
    effective_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="実効確信度。",
    )

    @classmethod
    def from_answer_dict(cls, answer: dict[str, Any]) -> SokutoMeta:
        """sokuto-server の Answer 辞書からメタデータを生成する。

        Args:
            answer (dict[str, Any]): sokuto-server から返却された Answer 辞書。

        Returns:
            SokutoMeta: 生成されたメタデータオブジェクト。
        """
        conf = float(answer.get("confidence") or 0.0)
        noul_val = answer.get("noul")
        if noul_val is not None and "confidence" not in answer:
            # Noul 型は真実確率 P(true) から尖り度 |2P - 1.0| を実効確信度として算出する。
            p = float(noul_val)
            eff_conf = max(0.0, min(1.0, abs(2.0 * p - 1.0)))
            conf = eff_conf
        else:
            eff_conf = conf

        gating = answer.get("gating") or {}
        probabilities = answer.get("probabilities")
        if probabilities is not None:
            probabilities = {str(k): float(v) for k, v in probabilities.items()}

        free_energy = gating.get("free_energy")
        if free_energy is None:
            free_energy = gating.get("energy")

        norm_entropy = gating.get("normalized_entropy")
        if norm_entropy is None:
            norm_entropy = gating.get("entropy")

        top_margin = gating.get("top_margin")
        if top_margin is None:
            top_margin = gating.get("margin")

        route = gating.get("route")

        return cls(
            confidence=conf,
            probabilities=probabilities,
            free_energy=float(free_energy) if free_energy is not None else None,
            normalized_entropy=(float(norm_entropy) if norm_entropy is not None else None),
            top_margin=float(top_margin) if top_margin is not None else None,
            route=str(route) if route is not None else None,
            effective_confidence=eff_conf,
        )


@dataclass(frozen=True)
class Uncertain[T]:
    """判定値と決定不確実性メタデータを内包する型安全ラッパー。

    呼び出し側コードにおいて、確定値 `.value` に直接アクセスしつつ、
    `.meta` 経由で信頼性やエスカレーションの要否を検査可能にする。

    Attributes:
        value (T): 推論された確定値 (str, int, bool など)。
        meta (SokutoMeta): 判定不確実性メタデータ。
    """

    value: T
    meta: SokutoMeta

    def __repr__(self) -> str:
        """文字列表現を生成する。"""
        return f"Uncertain(value={self.value!r}, conf={self.meta.confidence:.3f})"

    def is_confident(self, threshold: float = 0.70) -> bool:
        """実効確信度が指定閾値以上であるかを判定する。

        Args:
            threshold (float): 判定閾値。

        Returns:
            bool: 確信度が閾値以上であれば True。
        """
        return self.meta.effective_confidence >= threshold

    def is_ood(self, energy_threshold: float = -1.0) -> bool:
        """自由エネルギーに基づき OOD (未定義ドメイン) であるかを判定する。

        Args:
            energy_threshold (float): OOD 判定閾値。

        Returns:
            bool: 自由エネルギーが閾値を超過していれば True。
        """
        if self.meta.free_energy is None:
            return False
        return self.meta.free_energy > energy_threshold

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: Any,
    ) -> core_schema.CoreSchema:
        """Pydantic v2 向けの CoreSchema を生成する。

        arbitrary_types_allowed の設定なしに通常の Pydantic モデル内で
        Uncertain[T] を透過的に利用可能にし、シリアライズもサポートする。

        Args:
            source_type (Any): 型注釈情報 (例: Uncertain[bool])。
            handler (Any): Pydantic のスキーマハンドラ。

        Returns:
            core_schema.CoreSchema: 構築された Pydantic コアスキーマ。
        """
        args = get_args(source_type)
        value_schema = handler(args[0]) if args else core_schema.any_schema()

        meta_schema = handler(SokutoMeta)

        def validate_from_value(value: Any) -> Uncertain[Any]:
            if isinstance(value, Uncertain):
                return value
            if isinstance(value, dict) and "value" in value and "meta" in value:
                raw_meta = value["meta"]
                meta_obj = (
                    raw_meta
                    if isinstance(raw_meta, SokutoMeta)
                    else SokutoMeta.model_validate(raw_meta)
                )
                return Uncertain(value=value["value"], meta=meta_obj)
            raise ValueError(
                f"Uncertain インスタンスまたは {{'value': ..., 'meta': ...}} 辞書が必要です。取得: {type(value)}"
            )

        from_instance_schema = core_schema.chain_schema(
            [
                core_schema.no_info_plain_validator_function(validate_from_value),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=from_instance_schema,
            python_schema=from_instance_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda u: {
                    "value": u.value,
                    "meta": u.meta.model_dump(),
                },
                info_arg=False,
                return_schema=core_schema.typed_dict_schema(
                    {
                        "value": core_schema.typed_dict_field(value_schema),
                        "meta": core_schema.typed_dict_field(meta_schema),
                    }
                ),
            ),
        )


class UncertainDecisionError(Exception):
    """不確実性が高く、確定的な決定を下せない場合に送出される例外。

    System 2 (上位 LLM) へのエスカレーションや人手レビューへの分岐シグナルとして機能する。

    Attributes:
        field_name (str): 判定に失敗したフィールド名。
        meta (SokutoMeta): 判定時のメタデータ。
        reason (str): エスカレーション理由。
    """

    def __init__(self, field_name: str, meta: SokutoMeta, reason: str) -> None:
        """例外オブジェクトを初期化する。

        Args:
            field_name (str): フィールド識別子。
            meta (SokutoMeta): 不確実性メタデータ。
            reason (str): 例外の発生理由。
        """
        super().__init__(
            f"フィールド '{field_name}' の不確実性が高いためエスカレーションが必要です: {reason} (確信度: {meta.confidence:.3f})"
        )
        self.field_name = field_name
        self.meta = meta
        self.reason = reason
