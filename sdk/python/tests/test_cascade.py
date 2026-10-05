"""System 1 / System 2 カスケード統合 SDK ユニットテスト。

3軸 Pareto 最適ゲーティング、アンカリング防止対比型 Triage プロンプト、
サーキットブレーカーのフェイルオープン、および統合クライアントの動作を検証する。
"""

from __future__ import annotations

from typing import Literal
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, Field

from sokuto.cascade import (
    CascadeSource,
    CircuitBreaker,
    CircuitState,
    DecisionRoute,
    GatingDecision,
    MockSystem2Provider,
    SokutoCascadeClient,
    SokutoCascadeClientSync,
    calculate_entropy,
    calculate_top_margin,
    compute_free_energy,
    evaluate_gating,
    synthesize_triage_prompt,
)


def test_gating_choice_auto_execute() -> None:
    """高確信度な Choice 分布で AutoExecute が判定されることを検証する。"""
    # 候補Aが突出 (Margin = 0.85 - 0.10 = 0.75 >= 0.15, Entropy は低い)
    probs = {"A": 0.85, "B": 0.10, "C": 0.05}
    free_e = -2.5

    decision = evaluate_gating(
        probabilities=probs,
        free_energy=free_e,
        question_type="choice",
    )

    assert decision.route == DecisionRoute.AUTO_EXECUTE
    assert decision.escalate is False
    assert decision.is_ood is False
    assert decision.top_margin == pytest.approx(0.75)
    assert decision.top_candidates[0] == ("A", 0.85)
    assert decision.top_candidates[1] == ("B", 0.10)


def test_gating_choice_margin_collapse_escalate() -> None:
    """上位2候補が僅差で競合する場合にエスカレーションされることを検証する。"""
    # 僅差拮抗 (Margin = 0.46 - 0.44 = 0.02 < 0.15)
    probs = {"Option1": 0.46, "Option2": 0.44, "Option3": 0.10}
    free_e = -1.8

    decision = evaluate_gating(
        probabilities=probs,
        free_energy=free_e,
        question_type="choice",
    )

    assert decision.route == DecisionRoute.CONFIRM_OR_ESCALATE
    assert decision.escalate is True
    assert decision.top_margin == pytest.approx(0.02)
    assert "Top-Margin 拮抗" in decision.reason


def test_gating_choice_entropy_dispersion_escalate() -> None:
    """確率が全体に拡散している場合にエスカレーションされることを検証する。"""
    # ほぼ一様分布 (K=4, 各0.25 -> H_norm = 1.0 > 0.65)
    probs = [0.25, 0.25, 0.25, 0.25]
    decision = evaluate_gating(probabilities=probs, question_type="choice")

    assert decision.route == DecisionRoute.CONFIRM_OR_ESCALATE
    assert decision.escalate is True
    assert decision.normalized_entropy == pytest.approx(1.0)
    assert "エントロピー拡散" in decision.reason


def test_gating_choice_ood_fallback() -> None:
    """自由エネルギーが閾値を超過した場合に OOD として Fallback へ短絡することを検証する。"""
    probs = {"A": 0.90, "B": 0.10}
    # 自由エネルギーが閾値 (-1.0) を超過
    free_e = 0.5

    decision = evaluate_gating(
        probabilities=probs,
        free_energy=free_e,
        question_type="choice",
    )

    assert decision.route == DecisionRoute.FALLBACK
    assert decision.escalate is True
    assert decision.is_ood is True
    assert "OOD 検知" in decision.reason


def test_gating_noul_primitive() -> None:
    """Noul (真偽値) プリミティブにおけるマージンとエントロピー評価を検証する。"""
    # 高確信度 True (P=0.95 -> Margin = 0.90 >= 0.15)
    d_confident = evaluate_gating(noul_probability=0.95, question_type="noul")
    assert d_confident.route == DecisionRoute.AUTO_EXECUTE
    assert d_confident.escalate is False
    assert d_confident.top_margin == pytest.approx(0.90)

    # 拮抗 (P=0.52 -> Margin = 0.04 < 0.15)
    d_ambiguous = evaluate_gating(noul_probability=0.52, question_type="noul")
    assert d_ambiguous.route == DecisionRoute.CONFIRM_OR_ESCALATE
    assert d_ambiguous.escalate is True


def test_gating_score_primitive_and_bimodality() -> None:
    """Score (順序尺度) プリミティブにおける単峰性充足と双峰性 (二極対立) 排除を検証する。"""
    # 単峰・高確信度分布
    p_unimodal = [0.05, 0.10, 0.70, 0.10, 0.05]
    d_unimodal = evaluate_gating(probabilities=p_unimodal, question_type="score")
    assert d_unimodal.route == DecisionRoute.AUTO_EXECUTE
    assert d_unimodal.escalate is False

    # 双峰性 (両端対立) 分布
    p_bimodal = [0.48, 0.02, 0.00, 0.02, 0.48]
    d_bimodal = evaluate_gating(probabilities=p_bimodal, question_type="score")
    assert d_bimodal.route == DecisionRoute.CONFIRM_OR_ESCALATE
    assert d_bimodal.escalate is True
    assert "双峰性" in d_bimodal.reason


def test_gating_singularities() -> None:
    """K=1 やゼロ確率などの特異点でゼロ除算や NaN が生じないことを検証する。"""
    # K=1
    h_raw, h_norm = calculate_entropy([1.0])
    assert h_raw == 0.0
    assert h_norm == 0.0

    margin = calculate_top_margin([1.0])
    assert margin == 1.0

    # ゼロ確率含む
    _, h_norm_zero = calculate_entropy([1.0, 0.0, 0.0])
    assert h_norm_zero == 0.0

    # 自由エネルギー K=1
    fe_single = compute_free_energy([3.5], temperature=1.0)
    assert fe_single == pytest.approx(-3.5)


def test_synthesize_triage_prompt_anchoring_protection() -> None:
    """対比型 Triage プロンプトが単一候補を漏洩せず、Top-1 vs Top-2 の対立構造になっていることを検証する。"""
    gating = GatingDecision(
        route=DecisionRoute.CONFIRM_OR_ESCALATE,
        escalate=True,
        confidence=0.52,
        top_margin=0.04,
        normalized_entropy=0.68,
        free_energy=-1.5,
        is_ood=False,
        reason="僅差拮抗",
        top_candidates=[("approve", 0.52), ("reject", 0.48)],
    )

    prompt = synthesize_triage_prompt(
        state="ユーザーの本人確認書類画像と申請情報",
        instruction="申請を承認するか否認するか判定せよ。",
        gating=gating,
        criteria={"approve": "要件完全充足", "reject": "書類不備"},
    )

    # 1. 単一推奨クラスの漏洩がないこと
    assert "sokuto は approve と判定しました" not in prompt.user_prompt
    assert "確信度: 0.52" not in prompt.user_prompt
    assert (
        "事前の確率" not in prompt.user_prompt
        or "無視" in prompt.user_prompt
        or "迎合せず" in prompt.system_prompt
    )

    # 2. 候補 A と 候補 B が対等に提示されていること
    assert "- 候補 A: `approve`" in prompt.user_prompt
    assert "- 候補 B: `reject`" in prompt.user_prompt

    # 3. 批判的ステップ推論の指示
    assert "<thinking>" in prompt.user_prompt
    assert "中立" in prompt.system_prompt


def test_synthesize_triage_prompt_ood_warning() -> None:
    """OOD 検知時に 'none_of_the_above' の警告・選択肢が付与されることを検証する。"""
    gating = GatingDecision(
        route=DecisionRoute.FALLBACK,
        escalate=True,
        confidence=0.30,
        top_margin=0.02,
        free_energy=0.8,
        is_ood=True,
        reason="自由エネルギー超過",
        top_candidates=[("plan_a", 0.30), ("plan_b", 0.28)],
    )

    prompt = synthesize_triage_prompt(
        state="完全に未知の入力文脈",
        instruction="プランを選定せよ。",
        gating=gating,
        allow_none_of_the_above=True,
    )

    assert "【未知ドメイン警告】" in prompt.user_prompt
    assert "none_of_the_above" in prompt.user_prompt
    assert "none_of_the_above" in prompt.structured_schema["properties"]["decision"]["enum"]


def test_circuit_breaker_transitions() -> None:
    """サーキットブレーカーの CLOSED -> OPEN -> HALF_OPEN -> CLOSED 遷移を検証する。"""
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.05, success_threshold=2)

    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute() is True

    # 失敗を蓄積
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED

    # 3回目の失敗で OPEN
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.can_execute() is False

    # クールダウン待機
    import time

    time.sleep(0.06)

    # クールダウン後に自動で HALF_OPEN
    assert cb.state == CircuitState.HALF_OPEN
    assert cb.can_execute() is True

    # HALF_OPEN で 1 回成功
    cb.record_success()
    assert cb.state == CircuitState.HALF_OPEN

    # 2回目の成功で CLOSED に復帰
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute() is True


@pytest.mark.anyio
async def test_cascade_client_system1_auto_execute() -> None:
    """System 1 が高確信度で判定した場合、System 2 を呼ばずに即時返却することを検証する。"""
    mock_s2 = MockSystem2Provider()

    mock_client = AsyncMock()
    # サーバーからの高確信度レスポンス
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "answers": {
            "q1": {
                "choice": "credit_card",
                "probabilities": {"credit_card": 0.92, "bank_transfer": 0.08},
                "gating": {"energy": -2.1},
            }
        },
        "model_name": "sokuto-v1",
    }
    mock_client.post.return_value = mock_resp

    cascade_client = SokutoCascadeClient(
        system2_provider=mock_s2,
        http_client=mock_client,
    )

    result = await cascade_client.predict_question(
        instruction="決済手段を選択せよ。",
        criteria={"credit_card": "クレカ", "bank_transfer": "振込"},
        state="顧客がクレジットカード番号を入力した。",
        question_id="q1",
    )

    assert result.source == CascadeSource.SYSTEM1
    assert result.decision == "credit_card"
    assert result.gating.route == DecisionRoute.AUTO_EXECUTE
    assert mock_s2.call_count == 0  # System 2 は呼ばれない！


@pytest.mark.anyio
async def test_cascade_client_escalation_to_system2() -> None:
    """System 1 で僅差拮抗が発生した場合、System 2 へ中立対比プロンプトでエスカレーションされることを検証する。"""
    mock_s2 = MockSystem2Provider(
        default_decision="bank_transfer", default_thinking="振込口座への言及を確認。"
    )

    mock_client = AsyncMock()
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    # 僅差拮抗 (0.51 vs 0.49 -> Margin 0.02)
    mock_resp.json.return_value = {
        "answers": {
            "q1": {
                "choice": "credit_card",
                "probabilities": {"credit_card": 0.51, "bank_transfer": 0.49},
                "gating": {"energy": -1.5},
            }
        },
    }
    mock_client.post.return_value = mock_resp

    cascade_client = SokutoCascadeClient(
        system2_provider=mock_s2,
        http_client=mock_client,
    )

    result = await cascade_client.predict_question(
        instruction="決済手段を選択せよ。",
        criteria={"credit_card": "クレカ", "bank_transfer": "振込"},
        state="カードと振込の両方について言及がある文脈。",
        question_id="q1",
    )

    assert result.source == CascadeSource.SYSTEM2
    assert result.decision == "bank_transfer"
    assert result.gating.escalate is True
    assert result.system2_thinking == "振込口座への言及を確認。"
    assert mock_s2.call_count == 1  # System 2 が正しく呼び出された


@pytest.mark.anyio
async def test_cascade_client_circuit_breaker_fail_open() -> None:
    """System 1 が通信断またはサーキットオープン時に System 2 へフェイルオープンすることを検証する。"""
    mock_s2 = MockSystem2Provider(default_decision="fail_open_decision")

    mock_client = AsyncMock()
    # タイムアウト例外
    mock_client.post.side_effect = TimeoutError("Connection timed out")

    cb = CircuitBreaker(failure_threshold=1)
    cascade_client = SokutoCascadeClient(
        system2_provider=mock_s2,
        http_client=mock_client,
        circuit_breaker=cb,
    )

    result = await cascade_client.predict_question(
        instruction="障害時のテスト",
        criteria=["a", "b"],
        state="テストデータ",
        question_id="q1",
    )

    # 例外がスローされず、System 2 で救済される (Fail-Open)
    assert result.source == CascadeSource.SYSTEM2
    assert result.decision == "fail_open_decision"
    assert cb.state == CircuitState.OPEN
    assert mock_s2.call_count == 1


class CustomerClassification(BaseModel):
    category: Literal["vip", "standard", "fraud"] = Field(description="顧客分類。")
    is_trusted: bool = Field(description="信頼フラグ。")


@pytest.mark.anyio
async def test_cascade_client_execute_model_pydantic() -> None:
    """Pydantic v2 モデルに対するカスケード推論 (execute_model) を検証する。"""
    mock_s2 = MockSystem2Provider(
        callback=lambda prompt: {"category": "standard", "is_trusted": True}
    )

    mock_client = AsyncMock()
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    # System 1 が高確信度で全フィールドを回答
    mock_resp.json.return_value = {
        "answers": {
            "req.category": {
                "choice": "vip",
                "probabilities": {"vip": 0.95, "standard": 0.04, "fraud": 0.01},
                "gating": {"energy": -2.5},
            },
            "req.is_trusted": {
                "noul": 0.98,
                "gating": {"confidence": 0.98},
            },
        },
    }
    mock_client.post.return_value = mock_resp

    cascade_client = SokutoCascadeClient(
        system2_provider=mock_s2,
        http_client=mock_client,
    )

    result = await cascade_client.execute_model(
        CustomerClassification,
        state={"spending": 1000000, "history_years": 5},
        prefix="req",
    )

    assert result.source == CascadeSource.SYSTEM1
    assert isinstance(result.decision, CustomerClassification)
    assert result.decision.category == "vip"
    assert result.decision.is_trusted is True
    assert mock_s2.call_count == 0


def test_cascade_client_sync_wrapper() -> None:
    """同期ラッパークライアント SokutoCascadeClientSync の動作を検証する。"""
    mock_s2 = MockSystem2Provider(default_decision="sync_decision")

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "answers": {
                "q1": {
                    "choice": "sync_choice",
                    "probabilities": {"sync_choice": 0.96, "other": 0.04},
                    "gating": {"energy": -2.0},
                }
            }
        }
        mock_post.return_value = mock_resp

        with SokutoCascadeClientSync(system2_provider=mock_s2) as client:
            result = client.predict_question(
                instruction="同期テスト",
                criteria=["sync_choice", "other"],
                state="文脈",
                question_id="q1",
            )
            assert result.source == CascadeSource.SYSTEM1
            assert result.decision == "sync_choice"
