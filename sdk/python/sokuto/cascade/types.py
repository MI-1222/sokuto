"""カスケード統合 SDK 向け共通型定義モジュール。

System 1 (sokuto) と System 2 (フロンティア LLM) のハイブリッド実行における
3軸ゲーティング判定結果、サーキットブレーカー状態、対比プロンプト定義、
および統合実行結果コンテナを提供する。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DecisionRoute(StrEnum):
    """ゲーティング判定に基づく意思決定ルート種別。"""

    AUTO_EXECUTE = "auto_execute"
    CONFIRM_OR_ESCALATE = "confirm_or_escalate"
    FALLBACK = "fallback"


class CascadeSource(StrEnum):
    """判定を出力した実行主体。"""

    SYSTEM1 = "system1"
    SYSTEM2 = "system2"


class GatingConfig(BaseModel):
    """3軸 Pareto 最適ゲーティングの閾値設定。

    Attributes:
        margin_threshold (float): Top-Margin 閾値。上位2候補の確率差がこの値未満のときエスカレーションする。
        entropy_threshold (float): 正規化シャノンエントロピー上限。この値を超過したときエスカレーションする。
        energy_threshold (float): ヘルムホルツ自由エネルギー上限。この値を超過したとき OOD とみなしエスカレーションする。
        score_var_threshold (float): Score 型判定の正規化分散上限。
        bimodality_threshold (float): Score 型判定の双峰性係数限界。二極対立を検知したときエスカレーションする。
        energy_temperature (float | None): 自由エネルギー計算時に使用する温度。
    """

    model_config = ConfigDict(frozen=True)

    margin_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Top-Margin 判定閾値。",
    )
    entropy_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="正規化シャノンエントロピー上限値。",
    )
    energy_threshold: float = Field(
        default=-1.0,
        description="正規化ヘルムホルツ自由エネルギー上限値 (OOD 安全弁)。",
    )
    score_var_threshold: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Score 型における許容正規化分散上限値。",
    )
    bimodality_threshold: float = Field(
        default=0.555,
        ge=0.0,
        le=1.0,
        description="Score 型における双峰性係数 (二極化) 限界値。",
    )
    energy_temperature: float | None = Field(
        default=None,
        gt=0.0,
        description="自由エネルギー計算用温度パラメータ。",
    )


class GatingDecision(BaseModel):
    """3軸 Pareto ゲーティングの判定結果および診断メトリクス。

    Attributes:
        route (DecisionRoute): 確定されたルーティング種別。
        escalate (bool): System 2 への委任が必要であるかどうかの真偽値。
        confidence (float): 判定に用いられた較正済み確信度。
        top_margin (float | None): 上位2候補の確率差。
        normalized_entropy (float | None): 確率分布の正規化シャノンエントロピー。
        free_energy (float | None): 算出されたヘルムホルツ自由エネルギー。
        is_ood (bool): 未知カテゴリ・分布外 (OOD) として検知されたか。
        reason (str): ルーティング判定の根拠説明文。
        top_candidates (list[tuple[str, float]]): 確率上位候補のリスト。
    """

    model_config = ConfigDict(frozen=True)

    route: DecisionRoute = Field(description="判定されたルート。")
    escalate: bool = Field(description="エスカレーション要否フラグ。")
    confidence: float = Field(ge=0.0, le=1.0, description="実効確信度。")
    top_margin: float | None = Field(default=None, description="上位2候補差分。")
    normalized_entropy: float | None = Field(default=None, description="正規化エントロピー。")
    free_energy: float | None = Field(default=None, description="自由エネルギー。")
    is_ood: bool = Field(default=False, description="OOD 検知フラグ。")
    reason: str = Field(description="判定理由。")
    top_candidates: list[tuple[str, float]] = Field(
        default_factory=list,
        description="確率上位候補と確率のタプル列。",
    )


class TriagePrompt(BaseModel):
    """アンカリング思考汚染防止ニュートラル対比型プロンプト。

    Attributes:
        system_prompt (str): システムプロンプト。中立性と事前確率無視を義務付ける。
        user_prompt (str): ユーザー入力プロンプト。候補 A と候補 B を対等に提示する。
        structured_schema (dict[str, Any]): System 2 に要求する構造化出力スキーマ。
    """

    model_config = ConfigDict(frozen=True)

    system_prompt: str = Field(description="中立検証を促すシステムプロンプト。")
    user_prompt: str = Field(description="対照形式のユーザープロンプト。")
    structured_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema 仕様。",
    )


class CascadeResult[T](BaseModel):
    """カスケード推論の最終確定結果および監査証跡コンテナ。

    Attributes:
        decision (T): 確定判定結果 (型安全な値または Pydantic モデル)。
        source (CascadeSource): 判定主体 ('system1' または 'system2')。
        gating (GatingDecision): 判定時の 3 軸ゲーティング詳細。
        latency_ms (float): 全体所要時間 (ミリ秒)。
        escalation_reason (str | None): System 2 への委任理由。
        system2_thinking (str | None): System 2 が出力した思考連鎖 (CoT)。
        model_name (str | None): 推論を担当したモデル識別子。
        raw_s1_response (dict[str, Any] | None): System 1 の生レスポンス。
        raw_s2_response (dict[str, Any] | None): System 2 の生レスポンス。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    decision: T = Field(description="確定された判定値。")
    source: CascadeSource = Field(description="判定元システム。")
    gating: GatingDecision = Field(description="ゲーティング評価結果。")
    latency_ms: float = Field(default=0.0, description="処理所要時間 (ms)。")
    escalation_reason: str | None = Field(default=None, description="委任根拠理由。")
    system2_thinking: str | None = Field(default=None, description="System 2 の推論過程。")
    model_name: str | None = Field(default=None, description="モデル識別子。")
    raw_s1_response: dict[str, Any] | None = Field(default=None, description="S1 生データ。")
    raw_s2_response: dict[str, Any] | None = Field(default=None, description="S2 生データ。")
