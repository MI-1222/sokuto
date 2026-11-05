"""Python SDK バッチ推論および透過キャッシュ統合テスト。

predict_batch、execute_models、および部分カスケード (Partial Cascade) を網羅的に検証する。
"""

from typing import Literal

import httpx
import pytest
from pydantic import BaseModel, Field

from sokuto.cascade.client import SokutoCascadeClient
from sokuto.cascade.providers import MockSystem2Provider
from sokuto.schema.client import SokutoClient


class ApprovalDecision(BaseModel):
    """申請承認判定モデル。"""

    decision: Literal["承認", "却下", "差戻し"] = Field(description="申請の最終判定。")


class RiskAssessment(BaseModel):
    """リスク評価モデル。"""

    risk_level: Literal["低", "中", "高"] = Field(description="想定リスク水準。")


def test_predict_batch_and_state_cache() -> None:
    """predict_batch による一括推論とヘッダーハッシュ透過連携を検証する。"""

    def mock_transport(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/systemone/state-batch"
        assert request.headers.get("x-state-hash") == "hash_abc123"

        return httpx.Response(
            status_code=200,
            json={
                "answers": {
                    "q1": {"choice": "承認", "confidence": 0.96},
                    "q2": {"choice": "低", "confidence": 0.91},
                },
                "usage": {"prompt_tokens": 45, "completion_tokens": 0, "total_tokens": 45},
                "cache_hit": True,
            },
            headers={
                "X-State-Hash": "hash_abc123",
                "X-Cache-Status": "HIT",
            },
        )

    mock_client = httpx.Client(
        base_url="http://localhost:8080",
        transport=httpx.MockTransport(mock_transport),
    )
    client = SokutoClient(http_client=mock_client)
    client.last_state_hash = "hash_abc123"

    res = client.predict_batch(
        state="申請内容テキスト",
        questions={
            "q1": {"instructions": "承認可否", "type": "choice"},
            "q2": {"instructions": "リスク水準", "type": "choice"},
        },
    )

    assert res["cache_hit"] is True
    assert res["cache_status"] == "HIT"
    assert res["state_hash"] == "hash_abc123"
    assert res["answers"]["q1"]["choice"] == "承認"


def test_execute_models_batch() -> None:
    """execute_models による複数 Pydantic モデルの一括復元を検証する。"""

    def mock_transport(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/systemone/state-batch"

        return httpx.Response(
            status_code=200,
            json={
                "answers": {
                    "m0_.decision": {
                        "choice": "承認",
                        "confidence": 0.95,
                        "probabilities": {"承認": 0.95, "却下": 0.03, "差戻し": 0.02},
                    },
                    "m1_.risk_level": {
                        "choice": "低",
                        "confidence": 0.92,
                        "probabilities": {"低": 0.92, "中": 0.06, "高": 0.02},
                    },
                },
                "usage": {"prompt_tokens": 60, "completion_tokens": 0, "total_tokens": 60},
                "cache_hit": False,
            },
            headers={
                "X-State-Hash": "new_state_789",
                "X-Cache-Status": "MISS",
            },
        )

    mock_client = httpx.Client(
        base_url="http://localhost:8080",
        transport=httpx.MockTransport(mock_transport),
    )
    client = SokutoClient(http_client=mock_client)

    results = client.execute_models(
        models=[ApprovalDecision, RiskAssessment],
        state="契約締結前の審査コンテキスト",
    )

    assert len(results) == 2
    app_dec = results[0]
    risk_dec = results[1]

    assert isinstance(app_dec, ApprovalDecision)
    assert app_dec.decision == "承認"

    assert isinstance(risk_dec, RiskAssessment)
    assert risk_dec.risk_level == "低"
    assert client.get_last_state_hash() == "new_state_789"


def test_enable_state_offloading_fallback() -> None:
    """State 本文省略とキャッシュミス時の自動再送フォールバックを検証する。"""
    call_count = 0

    def mock_transport(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        import orjson

        data = orjson.loads(request.content)
        if call_count == 1:
            assert data["state"] == ""
            assert request.headers.get("X-State-Hash") == "stale_hash"
            return httpx.Response(
                status_code=400,
                json={"error": {"code": "state_cache_miss", "message": "Cache Miss"}},
            )

        assert data["state"] == "復元すべきState本文"
        assert "x-state-hash" not in request.headers and "X-State-Hash" not in request.headers
        return httpx.Response(
            status_code=200,
            json={
                "answers": {
                    "decision": {
                        "choice": "承認",
                        "confidence": 0.97,
                        "probabilities": {"承認": 0.97, "却下": 0.03},
                    }
                },
                "usage": {"prompt_tokens": 40, "completion_tokens": 0, "total_tokens": 40},
            },
            headers={"X-State-Hash": "recovered_hash", "X-Cache-Status": "MISS"},
        )

    mock_client = httpx.Client(
        base_url="http://localhost:8080",
        transport=httpx.MockTransport(mock_transport),
    )
    client = SokutoClient(http_client=mock_client)
    client.last_state_hash = "stale_hash"

    result = client.predict(
        ApprovalDecision,
        state="復元すべきState本文",
        enable_state_offloading=True,
    )

    assert call_count == 2
    assert result.decision == "承認"
    assert client.get_last_state_hash() == "recovered_hash"


@pytest.mark.anyio
async def test_cascade_partial_escalation() -> None:
    """predict_batch_questions における高確信度の即時返却と低確信度の部分エスカレーションを検証する。"""

    def mock_transport(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/systemone/state-batch"
        return httpx.Response(
            status_code=200,
            json={
                "answers": {
                    "q_high": {
                        "choice": "承認",
                        "confidence": 0.99,
                        "probabilities": {"承認": 0.99, "却下": 0.01},
                        "gating": {"route": "auto_execute", "confidence": 0.99, "energy": -2.5},
                    },
                    "q_low": {
                        "choice": "高",
                        "confidence": 0.40,
                        "probabilities": {"高": 0.40, "中": 0.38, "低": 0.22},
                        "gating": {
                            "route": "confirm_or_escalate",
                            "confidence": 0.40,
                            "energy": 0.5,
                        },
                    },
                },
                "usage": {"prompt_tokens": 70, "completion_tokens": 0, "total_tokens": 70},
                "cache_hit": True,
            },
            headers={"X-State-Hash": "casc_hash", "X-Cache-Status": "HIT"},
        )

    mock_async_client = httpx.AsyncClient(
        base_url="http://localhost:8080",
        transport=httpx.MockTransport(mock_transport),
    )
    cascade_client = SokutoCascadeClient(
        system2_provider=MockSystem2Provider(
            default_decision="System 2 救済: 中リスク",
            default_thinking="詳細調査により中リスクと判断",
        ),
        http_client=mock_async_client,
    )

    batch_res = await cascade_client.predict_batch_questions(
        questions={
            "q_high": {"instruction": "承認可否", "criteria": {"承認": "可", "却下": "否"}},
            "q_low": {"instruction": "リスク評価", "criteria": {"高": "3", "中": "2", "低": "1"}},
        },
        state="審査対象の取引詳細",
    )

    assert "q_high" in batch_res
    assert "q_low" in batch_res

    # q_high: System 1 で即時確定
    assert batch_res["q_high"].source.value == "system1"
    assert batch_res["q_high"].decision == "承認"

    # q_low: System 2 へ部分エスカレーション
    assert batch_res["q_low"].source.value == "system2"
    assert batch_res["q_low"].decision == "System 2 救済: 中リスク"
