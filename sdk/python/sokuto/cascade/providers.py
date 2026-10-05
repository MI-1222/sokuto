"""System 2 (フロンティア自己回帰 LLM) プロバイダー抽象化モジュール。

特定ベンダーの SDK (OpenAI, Anthropic 等) への直接密結合を排し、
Strategy パターンによる差し替え可能なエスカレーション実行インターフェースを提供する。
"""

from __future__ import annotations

import inspect
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .types import TriagePrompt


class System2Response(BaseModel):
    """System 2 からの実行結果および推論メタデータ。

    Attributes:
        decision (Any): 確定された最終判定ラベルまたは構造化辞書。
        thinking (str | None): 思考連鎖 (CoT) のテキスト。
        raw_output (str | None): LLM からの未加工生出力。
        confidence (float | None): 自己申告確信度。
        usage (dict[str, int] | None): トークン消費統計。
    """

    model_config = ConfigDict(frozen=True)

    decision: Any = Field(description="確定決定結果。")
    thinking: str | None = Field(default=None, description="思考過程。")
    raw_output: str | None = Field(default=None, description="生出力文字列。")
    confidence: float | None = Field(default=None, description="自己確信度。")
    usage: dict[str, int] | None = Field(default=None, description="トークン使用量。")


class System2Provider(ABC):
    """System 2 実行プロバイダーの抽象基底クラス。"""

    @abstractmethod
    async def call(self, prompt: TriagePrompt) -> System2Response:
        """エスカレーションプロンプトを受け取り、判定結果を返却する。

        Args:
            prompt (TriagePrompt): 対比型エスカレーションプロンプト。

        Returns:
            System2Response: System 2 の確定判定とメタデータ。
        """


class MockSystem2Provider(System2Provider):
    """テストおよび検証用モックプロバイダー。

    Attributes:
        default_decision (Any): 返却する既定決定値。
        default_thinking (str): 返却する既定思考テキスト。
        call_count (int): 呼び出し回数カウンタ。
    """

    def __init__(
        self,
        default_decision: Any = "candidate_a",
        default_thinking: str = "モック思考連鎖による検証。",
        callback: Callable[[TriagePrompt], Any] | None = None,
    ) -> None:
        """モックプロバイダーを初期化する。

        Args:
            default_decision (Any): デフォルトの決定値。
            default_thinking (str): デフォルトの思考ログ。
            callback (Callable[[TriagePrompt], Any] | None): 動的決定用コールバック。
        """
        self.default_decision = default_decision
        self.default_thinking = default_thinking
        self.callback = callback
        self.call_count = 0
        self.last_prompt: TriagePrompt | None = None

    async def call(self, prompt: TriagePrompt) -> System2Response:
        """モック判定を返却する。"""
        self.call_count += 1
        self.last_prompt = prompt

        if self.callback is not None:
            ret = self.callback(prompt)
            if inspect.iscoroutine(ret):
                ret = await ret
            if isinstance(ret, System2Response):
                return ret
            return System2Response(
                decision=ret,
                thinking=self.default_thinking,
                raw_output=str(ret),
                confidence=0.95,
            )

        # プロンプトの候補リストに合わせたフォールバック
        decision = self.default_decision
        return System2Response(
            decision=decision,
            thinking=self.default_thinking,
            raw_output=json.dumps({"thinking": self.default_thinking, "decision": decision}),
            confidence=0.95,
            usage={"prompt_tokens": 120, "completion_tokens": 45},
        )


class CallableSystem2Provider(System2Provider):
    """任意の関数またはコルーチンをラップするプロバイダー。"""

    def __init__(
        self,
        func: Callable[[TriagePrompt], Coroutine[Any, Any, Any] | Any],
    ) -> None:
        """プロバイダーを初期化する。

        Args:
            func (Callable): プロンプトを受け取り結果を返す同期/非同期関数。
        """
        self.func = func

    async def call(self, prompt: TriagePrompt) -> System2Response:
        """ラップされた関数を実行して結果を返却する。"""
        res = self.func(prompt)
        if inspect.iscoroutine(res):
            res = await res

        if isinstance(res, System2Response):
            return res

        thinking = None
        decision = res
        if isinstance(res, dict):
            decision = res.get("decision", res)
            thinking = res.get("thinking")

        return System2Response(
            decision=decision,
            thinking=thinking,
            raw_output=str(res),
            confidence=0.90,
        )


class GenericHttpSystem2Provider(System2Provider):
    """OpenAI 互換エンドポイント (chat/completions) と通信する汎用 HTTP プロバイダー。"""

    def __init__(
        self,
        endpoint_url: str,
        api_key: str = "",
        model: str = "gpt-4o",
        timeout: float = 30.0,
    ) -> None:
        """HTTP プロバイダーを初期化する。

        Args:
            endpoint_url (str): API エンドポイント URL。
            api_key (str): 認証トークン。
            model (str): モデル名。
            timeout (float): タイムアウト秒数。
        """
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    async def call(self, prompt: TriagePrompt) -> System2Response:
        """OpenAI 互換エンドポイントへ JSON リクエストを送信し、構造化結果を抽出する。"""
        import httpx

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": prompt.user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.endpoint_url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        raw_content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        # JSON パースの試行
        try:
            parsed = json.loads(raw_content)
            thinking = parsed.get("thinking")
            decision = parsed.get("decision")
            confidence = parsed.get("confidence")
        except Exception:
            # 正規表現による思考タグと決定のフォールバック抽出
            match = re.search(r"<thinking>(.*?)</thinking>", raw_content, re.DOTALL)
            thinking = match.group(1).strip() if match else None
            decision = raw_content.strip()
            confidence = None

        return System2Response(
            decision=decision,
            thinking=thinking,
            raw_output=raw_content,
            confidence=confidence,
            usage=usage,
        )
