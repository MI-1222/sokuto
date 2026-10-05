"""フロンティア LLM 実非同期推論クライアントモジュール。

Claude 3.7 Sonnet (Anthropic API) および GPT-4o (OpenAI API) への非同期通信、
並行度制御 Semaphore、指数バックオフ付きリトライ、および構造化レスポンス抽出を提供する。
"""

from __future__ import annotations

import asyncio
import os

import httpx

from pipeline.self_evolution.verifier import (
    LLMClientProtocol,
    LLMVerificationResponse,
)


class FrontierLLMClient(LLMClientProtocol):
    """Anthropic および OpenAI 向け汎用非同期 LLM クライアント。"""

    def __init__(
        self,
        anthropic_api_key: str | None = None,
        openai_api_key: str | None = None,
        max_concurrency: int = 5,
        max_retries: int = 3,
        initial_retry_delay: float = 1.0,
        timeout: float = 60.0,
    ) -> None:
        """フロンティア LLM クライアントを初期化する。

        Args:
            anthropic_api_key (str | None): Anthropic API キー (未指定時は環境変数から取得)。
            openai_api_key (str | None): OpenAI API キー (未指定時は環境変数から取得)。
            max_concurrency (int): 最大並行リクエスト数。
            max_retries (int): 失敗時の最大リトライ回数。
            initial_retry_delay (float): 初回リトライ待機秒数 (指数バックオフ)。
            timeout (float): HTTP タイムアウト秒数。
        """
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.timeout = timeout

    async def _call_anthropic(
        self,
        client: httpx.AsyncClient,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMVerificationResponse:
        """Anthropic Messages API を呼び出す。"""
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }

        resp = await client.post(
            url, headers=headers, json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()

        # レスポンスからテキストを抽出
        content_blocks = data.get("content", [])
        text_content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")

        return LLMVerificationResponse(
            model_name=model_name,
            decision_label=text_content,
            thinking_process=text_content,
            raw_response=data,
        )

    async def _call_openai(
        self,
        client: httpx.AsyncClient,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMVerificationResponse:
        """OpenAI Chat Completions API を呼び出す。"""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "content-type": "application/json",
        }
        payload = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        resp = await client.post(
            url, headers=headers, json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices", [])
        text_content = ""
        if choices:
            text_content = choices[0].get("message", {}).get("content", "")

        return LLMVerificationResponse(
            model_name=model_name,
            decision_label=text_content,
            thinking_process=text_content,
            raw_response=data,
        )

    async def complete(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1500,
    ) -> LLMVerificationResponse:
        """モデル名に応じて適切な API へ推論リクエストを送信する。

        指数バックオフによる自動リトライおよびセマフォ並行度制御を行う。

        Args:
            model_name (str): モデル識別子 ('claude-3-7-sonnet...' または 'gpt-4o' 等)。
            system_prompt (str): システムプロンプト。
            user_prompt (str): ユーザープロンプト。
            temperature (float): サンプリング温度。
            max_tokens (int): 最大生成トークン数。

        Returns:
            LLMVerificationResponse: 推論レスポンス。
        """
        async with self.semaphore, httpx.AsyncClient() as client:
            last_err: Exception | None = None
            delay = self.initial_retry_delay

            for attempt in range(self.max_retries):
                try:
                    if "claude" in model_name.lower():
                        return await self._call_anthropic(
                            client=client,
                            model_name=model_name,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                    # デフォルトは OpenAI 互換 API
                    return await self._call_openai(
                        client=client,
                        model_name=model_name,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    last_err = e
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(delay)
                        delay *= 2.0
                    else:
                        break

            raise RuntimeError(
                f"LLM API 呼び出しが {self.max_retries} 回失敗しました: {last_err}"
            ) from last_err
