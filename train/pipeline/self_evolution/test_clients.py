"""フロンティア LLM 非同期クライアントのテストモジュール。"""

import asyncio
import json

import httpx

from pipeline.self_evolution.clients import FrontierLLMClient


def test_anthropic_client_call() -> None:
    """Claude 向け Messages API への非同期リクエストとレスポンス抽出を検証する。"""

    def _handler(request: httpx.Request) -> httpx.Response:
        assert "api.anthropic.com" in str(request.url)
        assert request.headers["x-api-key"] == "test-anthropic-key"
        body = json.loads(request.read())
        assert body["model"] == "claude-3-7-sonnet"

        resp_data = {
            "content": [
                {
                    "type": "text",
                    "text": "<premises>\n- [TRUE] 前提1\n</premises>\nDECISION: 0",
                }
            ]
        }
        return httpx.Response(200, json=resp_data)

    async def _run() -> None:
        client = FrontierLLMClient(anthropic_api_key="test-anthropic-key")
        # transport の差し替え
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_handler)
        ) as mock_http:
            res = await client._call_anthropic(
                client=mock_http,
                model_name="claude-3-7-sonnet",
                system_prompt="システムプロンプト",
                user_prompt="ユーザープロンプト",
                temperature=0.0,
                max_tokens=1000,
            )
            assert (
                res.decision_label
                == "<premises>\n- [TRUE] 前提1\n</premises>\nDECISION: 0"
            )
            assert "DECISION: 0" in res.thinking_process

    asyncio.run(_run())


def test_openai_client_call() -> None:
    """GPT-4o 向け Chat Completions API への非同期リクエストとレスポンス抽出を検証する。"""

    def _handler(request: httpx.Request) -> httpx.Response:
        assert "api.openai.com" in str(request.url)
        assert request.headers["authorization"] == "Bearer test-openai-key"
        body = json.loads(request.read())
        assert body["model"] == "gpt-4o"

        resp_data = {
            "choices": [
                {
                    "message": {
                        "content": "<premises>\n- [TRUE] 前提1\n</premises>\nDECISION: 1"
                    }
                }
            ]
        }
        return httpx.Response(200, json=resp_data)

    async def _run() -> None:
        client = FrontierLLMClient(openai_api_key="test-openai-key")
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_handler)
        ) as mock_http:
            res = await client._call_openai(
                client=mock_http,
                model_name="gpt-4o",
                system_prompt="システム",
                user_prompt="ユーザー",
                temperature=0.0,
                max_tokens=1000,
            )
            assert "DECISION: 1" in res.decision_label

    asyncio.run(_run())
