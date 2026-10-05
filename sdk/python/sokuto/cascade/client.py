"""System 1 / System 2 カスケード統合統括クライアントモジュール。

超低遅延・高確信度のローカル非自己回帰判断エンジン (sokuto / System 1) と、
推論連鎖に優れるフロンティア自己回帰 LLM (Claude, GPT 等 / System 2) を協調させ、
定常トラフィックの 80% 以上をミリ秒で完結させつつ境界事例を高精度に救済する。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import inspect
import json
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from ..schema.compiler import CompiledTemplate, compile_schema
from ..schema.transpiler import SchemaTranspiler
from .circuit_breaker import CircuitBreaker
from .gating import evaluate_gating
from .providers import MockSystem2Provider, System2Provider
from .triage import synthesize_triage_prompt
from .types import (
    CascadeResult,
    CascadeSource,
    DecisionRoute,
    GatingConfig,
    GatingDecision,
)

M = TypeVar("M", bound=BaseModel)


class SokutoCascadeClient:
    """非同期 System 1 / System 2 カスケード統合クライアント。

    Attributes:
        sokuto_url (str): sokuto-server のベース URL。
        system2_provider (System2Provider): エスカレーション先フロンティア LLM プロバイダー。
        gating_config (GatingConfig): 3軸 Pareto 最適ゲーティング設定。
        timeout (float): sokuto 呼び出しのハードタイムアウト秒数 (デフォルト: 50ms)。
        circuit_breaker (CircuitBreaker): 障害遮断およびフェイルオープン制御器。
    """

    def __init__(
        self,
        sokuto_url: str = "http://localhost:8080",
        system2_provider: System2Provider | None = None,
        gating_config: GatingConfig | None = None,
        timeout: float = 0.05,
        circuit_breaker: CircuitBreaker | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """カスケードクライアントを初期化する。

        Args:
            sokuto_url (str): sokuto サーバーの URL。
            system2_provider (System2Provider | None): System 2 プロバイダー (未指定時はモック)。
            gating_config (GatingConfig | None): ゲーティング閾値設定。
            timeout (float): sokuto 呼び出しタイムアウト (秒)。
            circuit_breaker (CircuitBreaker | None): サーキットブレーカーインスタンス。
            http_client (httpx.AsyncClient | None): 既存の非同期 HTTP クライアント。
        """
        self.sokuto_url = sokuto_url.rstrip("/")
        self.system2_provider = system2_provider or MockSystem2Provider()
        self.gating_config = gating_config or GatingConfig()
        self.timeout = timeout
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self._owns_http_client = http_client is None
        self._client = (
            http_client
            if http_client is not None
            else httpx.AsyncClient(base_url=self.sokuto_url, timeout=self.timeout)
        )

    async def close(self) -> None:
        """HTTP リソースを解放する。"""
        if self._owns_http_client:
            await self._client.aclose()

    async def __aenter__(self) -> SokutoCascadeClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.close()

    async def predict_question(
        self,
        instruction: str,
        criteria: dict[str, str] | list[str],
        state: Any,
        question_type: str = "choice",
        question_id: str = "q1",
        context_reducer: Callable[[Any], str] | None = None,
    ) -> CascadeResult[Any]:
        """単一の質問定義に対してカスケード判定を実行する。

        Args:
            instruction (str): 判断タスク指示文。
            criteria (dict[str, str] | list[str]): 候補定義。
            state (Any): 根拠文脈データ。
            question_type (str): 質問種別 ('choice', 'score', 'noul')。
            question_id (str): 質問識別子。

        Returns:
            CascadeResult[Any]: 確定判定結果とメタデータ。
        """
        start_time = time.perf_counter()

        # サーキットブレーカーの判定
        if not self.circuit_breaker.can_execute():
            # フェイルオープン: System 2 へ直接転送
            return await self._escalate_to_system2(
                instruction=instruction,
                criteria=criteria,
                state=state,
                gating=GatingDecision(
                    route=DecisionRoute.FALLBACK,
                    escalate=True,
                    confidence=0.0,
                    reason="サーキットブレーカー遮断中 (Fail-Open 直行)。",
                    top_candidates=[],
                ),
                start_time=start_time,
                reason="circuit_breaker_open",
                context_reducer=context_reducer,
            )

        # System 1 (sokuto) の呼び出し
        s1_result: dict[str, Any] | None = None
        try:
            req_payload = {
                "state": state,
                "questions": {
                    question_id: {
                        "instructions": instruction,
                        "criteria": criteria,
                        "type": question_type,
                    }
                },
            }
            resp = await self._client.post(
                "/v1/systemone",
                json=req_payload,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                json_data = resp.json()
                if inspect.iscoroutine(json_data):
                    s1_result = await json_data
                else:
                    s1_result = json_data
                self.circuit_breaker.record_success()
            else:
                self.circuit_breaker.record_failure()
        except Exception:
            self.circuit_breaker.record_failure()

        # サーバー障害時は System 2 へフェイルオープン
        if s1_result is None:
            return await self._escalate_to_system2(
                instruction=instruction,
                criteria=criteria,
                state=state,
                gating=GatingDecision(
                    route=DecisionRoute.FALLBACK,
                    escalate=True,
                    confidence=0.0,
                    reason="System 1 通信タイムアウトまたはサーバーエラー (Fail-Open)。",
                    top_candidates=[],
                ),
                start_time=start_time,
                reason="system1_timeout_or_error",
                context_reducer=context_reducer,
            )

        # レスポンスの解析
        answers = s1_result.get("answers", {})
        answer_data = answers.get(question_id, {})
        probs = answer_data.get("probabilities")
        noul_p = answer_data.get("noul")
        server_gating = answer_data.get("gating")

        # 3軸ゲーティング評価
        gating = evaluate_gating(
            probabilities=probs,
            question_type=question_type,
            config=self.gating_config,
            noul_probability=noul_p,
            server_gating=server_gating,
        )

        # 充足時は System 1 確定返却
        if not gating.escalate and gating.route == DecisionRoute.AUTO_EXECUTE:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            decision = (
                answer_data.get("choice")
                or answer_data.get("score")
                or answer_data.get("noul")
                or (gating.top_candidates[0][0] if gating.top_candidates else None)
            )
            return CascadeResult(
                decision=decision,
                source=CascadeSource.SYSTEM1,
                gating=gating,
                latency_ms=elapsed_ms,
                model_name=s1_result.get("model_name"),
                raw_s1_response=s1_result,
            )

        # 不確実性検知による System 2 エスカレーション
        return await self._escalate_to_system2(
            instruction=instruction,
            criteria=criteria,
            state=state,
            gating=gating,
            start_time=start_time,
            reason=gating.reason,
            raw_s1_response=s1_result,
            context_reducer=context_reducer,
        )

    async def execute_model(
        self,
        model_cls: type[M],
        state: Any,
        prefix: str | None = None,
        model_name: str | None = None,
        auto_hierarchical: bool | None = None,
    ) -> CascadeResult[M]:
        """Pydantic v2 モデルを受け取り、型安全なカスケード推論を実行する。

        7.2 で構築された SchemaTranspiler と連携し、System 1 で解決した場合は
        0.15ms 以下のゼロアロケーション復元を実行し、エスカレーション時は
        型スキーマに基づく System 2 構造化抽出を行って同一の型 M を返却する。

        Args:
            model_cls (type[M]): 対象 Pydantic モデルクラス。
            state (Any): 入力文脈 (state)。
            prefix (str | None): 質問キープレフィックス。
            model_name (str | None): モデル名。
            auto_hierarchical (bool | None): 階層 DAG 有効化。

        Returns:
            CascadeResult[M]: 復元された Pydantic モデルインスタンスと監査メタデータ。
        """
        start_time = time.perf_counter()
        template: CompiledTemplate[M] = compile_schema(
            model_cls,
            prefix=prefix,
            model_name=model_name,
            auto_hierarchical=auto_hierarchical,
        )
        transpiler = template.transpiler

        # サーキットブレーカー検査
        if not self.circuit_breaker.can_execute():
            return await self._escalate_model_to_system2(
                model_cls=model_cls,
                state=state,
                transpiler=transpiler,
                gating=GatingDecision(
                    route=DecisionRoute.FALLBACK,
                    escalate=True,
                    confidence=0.0,
                    reason="サーキットブレーカー遮断中 (Fail-Open 直行)。",
                    top_candidates=[],
                ),
                start_time=start_time,
                reason="circuit_breaker_open",
            )

        # System 1 への事前コンパイルバイト送信
        s1_result: dict[str, Any] | None = None
        try:
            body_bytes = template.build_request_bytes(state)
            headers = {"Content-Type": "application/json"}
            resp = await self._client.post(
                "/v1/systemone",
                content=body_bytes,
                headers=headers,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                json_data = resp.json()
                if inspect.iscoroutine(json_data):
                    s1_result = await json_data
                else:
                    s1_result = json_data
                self.circuit_breaker.record_success()
            else:
                self.circuit_breaker.record_failure()
        except Exception:
            self.circuit_breaker.record_failure()

        if s1_result is None:
            return await self._escalate_model_to_system2(
                model_cls=model_cls,
                state=state,
                transpiler=transpiler,
                gating=GatingDecision(
                    route=DecisionRoute.FALLBACK,
                    escalate=True,
                    confidence=0.0,
                    reason="System 1 通信タイムアウトまたはサーバー障害 (Fail-Open)。",
                    top_candidates=[],
                ),
                start_time=start_time,
                reason="system1_timeout_or_error",
            )

        # モデル全質問に対する集約ゲーティング判定
        answers = s1_result.get("answers", {})
        overall_escalate = False
        worst_gating: GatingDecision | None = None

        for q_id, q_spec in transpiler.questions.items():
            ans = answers.get(q_id, {})
            q_type = q_spec.get("question_type") or q_spec.get("type", "choice")
            probs = ans.get("probabilities")
            noul_val = ans.get("noul")
            server_g = ans.get("gating")

            g = evaluate_gating(
                probabilities=probs,
                question_type=q_type,
                config=self.gating_config,
                noul_probability=noul_val,
                server_gating=server_g,
            )

            if g.escalate:
                overall_escalate = True
                worst_gating = g
                break
            elif worst_gating is None or g.confidence < worst_gating.confidence:
                worst_gating = g

        final_gating = worst_gating or GatingDecision(
            route=DecisionRoute.AUTO_EXECUTE,
            escalate=False,
            confidence=1.0,
            reason="全フィールド充足。",
        )

        # System 1 で全条件クリアした場合の即座復元
        if not overall_escalate:
            instance = transpiler.deserialize_response(s1_result, fail_on_uncertainty=False)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return CascadeResult(
                decision=instance,
                source=CascadeSource.SYSTEM1,
                gating=final_gating,
                latency_ms=elapsed_ms,
                model_name=s1_result.get("model_name"),
                raw_s1_response=s1_result,
            )

        # エスカレーションが必要な場合
        return await self._escalate_model_to_system2(
            model_cls=model_cls,
            state=state,
            transpiler=transpiler,
            gating=final_gating,
            start_time=start_time,
            reason=final_gating.reason,
            raw_s1_response=s1_result,
        )

    async def _escalate_to_system2(
        self,
        instruction: str,
        criteria: dict[str, str] | list[str],
        state: Any,
        gating: GatingDecision,
        start_time: float,
        reason: str,
        raw_s1_response: dict[str, Any] | None = None,
        context_reducer: Callable[[Any], str] | None = None,
    ) -> CascadeResult[Any]:
        """対比型プロンプトを合成して System 2 を実行する。"""
        triage_prompt = synthesize_triage_prompt(
            state=state,
            instruction=instruction,
            gating=gating,
            criteria=criteria,
            context_reducer=context_reducer,
        )

        s2_response = await self.system2_provider.call(triage_prompt)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        decision_val = s2_response.decision
        if isinstance(decision_val, dict):
            decision_val = decision_val.get("decision", decision_val)

        return CascadeResult(
            decision=decision_val,
            source=CascadeSource.SYSTEM2,
            gating=gating,
            latency_ms=elapsed_ms,
            escalation_reason=reason,
            system2_thinking=s2_response.thinking,
            raw_s1_response=raw_s1_response,
            raw_s2_response=s2_response.model_dump(),
        )

    async def _escalate_model_to_system2(
        self,
        model_cls: type[M],
        state: Any,
        transpiler: SchemaTranspiler,
        gating: GatingDecision,
        start_time: float,
        reason: str,
        raw_s1_response: dict[str, Any] | None = None,
    ) -> CascadeResult[M]:
        """Pydantic モデル構造を強制する Structured Outputs で System 2 を実行する。"""
        json_schema = model_cls.model_json_schema()
        state_str = (
            json.dumps(state, ensure_ascii=False, indent=2)
            if isinstance(state, dict)
            else str(state)
        )

        system_prompt = (
            "あなたは厳密かつ中立な構造化情報抽出エージェントです。\n"
            "機械学習モデルの事前判定が不確実なため、提示された事実 (State) のみから客観的に判断してください。\n"
            "要求された JSON スキーマに厳密に合致する JSON オブジェクトを出力してください。"
        )
        user_prompt = (
            f"以下の事実 (State) に基づき、指定スキーマに従って判定・抽出を行ってください。\n\n"
            f"### 入力文脈 (State)\n```text\n{state_str}\n```\n\n"
            f"### 出力スキーマ\n```json\n{json.dumps(json_schema, ensure_ascii=False, indent=2)}\n```"
        )

        triage_prompt = synthesize_triage_prompt(
            state=state,
            instruction="Pydantic モデルの各フィールドを中立に抽出してください。",
            gating=gating,
        )
        # スキーマを Pydantic モデルスキーマで上書き
        triage_prompt = triage_prompt.model_copy(
            update={
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "structured_schema": json_schema,
            }
        )

        s2_response = await self.system2_provider.call(triage_prompt)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Pydantic モデルとしてのパース
        parsed_data = s2_response.decision
        if isinstance(parsed_data, str):
            with contextlib.suppress(Exception):
                parsed_data = json.loads(parsed_data)

        if isinstance(parsed_data, dict):
            instance = model_cls.model_validate(parsed_data)
        elif isinstance(parsed_data, model_cls):
            instance = parsed_data
        else:
            instance = model_cls.model_validate_json(s2_response.raw_output or "{}")

        return CascadeResult(
            decision=instance,
            source=CascadeSource.SYSTEM2,
            gating=gating,
            latency_ms=elapsed_ms,
            escalation_reason=reason,
            system2_thinking=s2_response.thinking,
            raw_s1_response=raw_s1_response,
            raw_s2_response=s2_response.model_dump(),
        )


_SYNC_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_SYNC_LOCK = threading.Lock()


def _get_sync_executor() -> concurrent.futures.ThreadPoolExecutor:
    """同期実行用の永続スレッドプールを取得する。"""
    global _SYNC_EXECUTOR
    if _SYNC_EXECUTOR is None:
        with _SYNC_LOCK:
            if _SYNC_EXECUTOR is None:
                _SYNC_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                    max_workers=4,
                    thread_name_prefix="sokuto_sync_pool",
                )
    return _SYNC_EXECUTOR


def _run_sync[R](coro: Any) -> R:
    """既存のイベントループ有無を判定し、同期的にコルーチンを実行する。"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        executor = _get_sync_executor()
        return executor.submit(lambda: asyncio.run(coro)).result()
    return asyncio.run(coro)


class SokutoCascadeClientSync:
    """SokutoCascadeClient の同期ラッパークライアント。"""

    def __init__(self, **kwargs: Any) -> None:
        """非同期クライアントを同期ラップする。"""
        self._async_client = SokutoCascadeClient(**kwargs)

    def close(self) -> None:
        """リソースを解放する。"""
        _run_sync(self._async_client.close())

    def __enter__(self) -> SokutoCascadeClientSync:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()

    def predict_question(self, *args: Any, **kwargs: Any) -> CascadeResult[Any]:
        """同期的に単一質問のカスケード判定を実行する。"""
        return _run_sync(self._async_client.predict_question(*args, **kwargs))

    def execute_model(self, *args: Any, **kwargs: Any) -> CascadeResult[Any]:
        """同期的に Pydantic モデルのカスケード判定を実行する。"""
        return _run_sync(self._async_client.execute_model(*args, **kwargs))
