"""sokuto-server HTTP 通信および透過的スキーマ駆動推論クライアントモジュール。

Pydantic モデル定義と動的コンテキスト (state) を受け取り、
事前コンパイルされたバイトテンプレートを用いて sokuto-server と通信し、
確定結果と不確実性メタデータを内包した型安全なモデルインスタンスを復元する。
"""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, Self, TypeVar

import httpx
import orjson
from pydantic import BaseModel

from .compiler import CompiledTemplate, compile_schema

if TYPE_CHECKING:
    from .dag_synthesizer import DagSynthesizer

M = TypeVar("M", bound=BaseModel)


class SokutoClientError(Exception):
    """sokuto-server との通信またはサーバー側での推論処理に失敗した場合の例外。"""


class SokutoClient:
    """sokuto-server 向け同期推論クライアント。

    Attributes:
        base_url (str): sokuto サーバーのベース URL (例: "http://localhost:8080")。
        timeout (float): HTTP タイムアウト秒数。既定値は 10.0 秒。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 10.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        """クライアントを初期化する。

        Args:
            base_url (str): サーバーのベース URL。
            timeout (float): タイムアウト秒数。
            http_client (httpx.Client | None): 既存の HTTP クライアントインスタンス。
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.last_state_hash: str | None = None
        self._client = (
            http_client
            if http_client is not None
            else httpx.Client(base_url=self.base_url, timeout=self.timeout)
        )

    def get_last_state_hash(self) -> str | None:
        """直近にサーバーから受信した State キャッシュキーを取得する。"""
        return self.last_state_hash

    def clear_state_hash(self) -> None:
        """記憶している State キャッシュキーをクリアする。"""
        self.last_state_hash = None

    def close(self) -> None:
        """HTTP クライアントのリソースを解放する。"""
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        self.close()

    def predict(
        self,
        model_cls: type[M],
        state: Any,
        prefix: str | None = None,
        model_name: str | None = None,
        auto_hierarchical: bool | None = None,
        fail_on_uncertainty: bool = False,
        min_confidence: float = 0.0,
        enable_state_offloading: bool = False,
        state_hash: str | None = None,
    ) -> M:
        """Pydantic モデルに基づいて単一フォワードパス推論を実行し、型安全なインスタンスを復元する。

        Args:
            model_cls (type[M]): 期待される出力 Pydantic モデルクラス。
            state (Any): 判断の材料となるコンテキスト (文字列、辞書など)。
            prefix (str | None): 質問キープレフィックス。
            model_name (str | None): sokuto モデル識別子。
            auto_hierarchical (bool | None): 粗密階層推論の自動有効化フラグ。
            fail_on_uncertainty (bool): 不確実性が高い場合に例外を送出するかどうか。
            min_confidence (float): 許容最小確信度。
            enable_state_offloading (bool): State 本文の省略 (キャッシュ再利用) を有効化するかどうか。
            state_hash (str | None): クライアント指定の State キャッシュキー。

        Returns:
            M: 復元された Pydantic モデルインスタンス。

        Raises:
            SokutoClientError: サーバーエラーやネットワークエラーが発生した場合。
            UncertainDecisionError: 確信度が不足した場合。
        """
        template: CompiledTemplate = compile_schema(
            model_cls,
            prefix=prefix,
            model_name=model_name,
            auto_hierarchical=auto_hierarchical,
        )

        effective_hash = state_hash or self.last_state_hash
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if effective_hash:
            headers["X-State-Hash"] = effective_hash

        sent_state = "" if (enable_state_offloading and effective_hash) else state
        request_bytes = template.build_request_bytes(sent_state)

        try:
            response = self._client.post(
                "/v1/systemone",
                content=request_bytes,
                headers=headers,
            )
            if (
                response.status_code == 400
                and enable_state_offloading
                and sent_state == ""
                and state != ""
            ):
                fallback_bytes = template.build_request_bytes(state)
                headers.pop("X-State-Hash", None)
                response = self._client.post(
                    "/v1/systemone",
                    content=fallback_bytes,
                    headers=headers,
                )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise SokutoClientError(
                f"sokuto-server (/v1/systemone) への推論リクエストに失敗しました: {e}"
            ) from e

        rec_hash = response.headers.get("x-state-hash") or response.headers.get("X-State-Hash")
        if rec_hash:
            self.last_state_hash = rec_hash

        return template.deserialize(
            response.content,
            fail_on_uncertainty=fail_on_uncertainty,
            min_confidence=min_confidence,
        )

    def predict_batch(
        self,
        state: Any,
        questions: dict[str, Any],
        model_name: str | None = None,
        auto_hierarchical: bool | None = None,
        gating: dict[str, Any] | None = None,
        enable_state_offloading: bool = False,
        state_hash: str | None = None,
    ) -> dict[str, Any]:
        """単一 State に対する複数質問を一括送信し、判定結果マップを取得する。

        Args:
            state (Any): 判断コンテキスト。
            questions (dict[str, Any]): 質問定義マップ。
            model_name (str | None): モデル名。
            auto_hierarchical (bool | None): 階層探索フラグ。
            gating (dict[str, Any] | None): ゲーティング設定。
            enable_state_offloading (bool): State 本文の省略最適化。
            state_hash (str | None): キャッシュキー。

        Returns:
            dict[str, Any]: StateBatchResponse 準拠の辞書。
        """
        effective_hash = state_hash or self.last_state_hash
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if effective_hash:
            headers["X-State-Hash"] = effective_hash

        sent_state = "" if (enable_state_offloading and effective_hash) else state
        payload: dict[str, Any] = {
            "state": sent_state,
            "questions": questions,
        }
        if model_name is not None:
            payload["model"] = model_name
        if auto_hierarchical is not None:
            payload["auto_hierarchical"] = auto_hierarchical
        if gating is not None:
            payload["gating"] = gating
        if effective_hash is not None:
            payload["state_hash"] = effective_hash

        req_bytes = orjson.dumps(payload)

        try:
            response = self._client.post(
                "/v1/systemone/state-batch",
                content=req_bytes,
                headers=headers,
            )
            if (
                response.status_code == 400
                and enable_state_offloading
                and sent_state == ""
                and state != ""
            ):
                payload["state"] = state
                payload.pop("state_hash", None)
                headers.pop("X-State-Hash", None)
                response = self._client.post(
                    "/v1/systemone/state-batch",
                    content=orjson.dumps(payload),
                    headers=headers,
                )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise SokutoClientError(
                f"sokuto-server (/v1/systemone/state-batch) へのバッチ推論に失敗しました: {e}"
            ) from e

        rec_hash = response.headers.get("x-state-hash") or response.headers.get("X-State-Hash")
        if rec_hash:
            self.last_state_hash = rec_hash

        result_dict: dict[str, Any] = orjson.loads(response.content)
        cache_status = (
            response.headers.get("x-cache-status")
            or response.headers.get("X-Cache-Status")
            or "BYPASS"
        )
        result_dict["cache_status"] = cache_status
        if rec_hash:
            result_dict["state_hash"] = rec_hash

        return result_dict

    def execute_models(
        self,
        models: list[type[BaseModel]],
        state: Any,
        prefix_list: list[str] | None = None,
        model_name: str | None = None,
        auto_hierarchical: bool | None = None,
        fail_on_uncertainty: bool = False,
        min_confidence: float = 0.0,
        enable_state_offloading: bool = False,
    ) -> list[BaseModel]:
        """同一 State に対して複数の異なる Pydantic モデルを一括解決する。

        Args:
            models (list[type[BaseModel]]): 解決対象モデルクラスのリスト。
            state (Any): 共通コンテキストデータ。
            prefix_list (list[str] | None): 各モデルのプレフィックス一覧。
            model_name (str | None): モデル識別子。
            auto_hierarchical (bool | None): 粗密階層フラグ。
            fail_on_uncertainty (bool): 不確実性例外フラグ。
            min_confidence (float): 最小確信度。
            enable_state_offloading (bool): State 本文省略フラグ。

        Returns:
            list[BaseModel]: 復元された各モデルインスタンスのリスト。
        """
        templates: list[CompiledTemplate] = []
        merged_questions: dict[str, Any] = {}

        for i, model_cls in enumerate(models):
            pfx = prefix_list[i] if prefix_list and i < len(prefix_list) else f"m{i}_"
            t = compile_schema(
                model_cls,
                prefix=pfx,
                model_name=model_name,
                auto_hierarchical=auto_hierarchical,
            )
            templates.append(t)
            merged_questions.update(t.cached_questions)

        batch_result = self.predict_batch(
            state=state,
            questions=merged_questions,
            model_name=model_name,
            auto_hierarchical=auto_hierarchical,
            enable_state_offloading=enable_state_offloading,
        )

        resp_bytes = orjson.dumps(batch_result)
        restored: list[BaseModel] = []
        for t in templates:
            inst = t.deserialize(
                resp_bytes,
                fail_on_uncertainty=fail_on_uncertainty,
                min_confidence=min_confidence,
            )
            restored.append(inst)

        return restored

    def execute_dag(
        self,
        dag_definition: dict[str, Any],
        state: Any,
        timeout_ms: int | None = None,
        synthesizer: DagSynthesizer | None = None,
    ) -> Any:
        """Phase 6 のインプロセス DAG を実行する。

        synthesizer が指定された場合、DAG の実行結果から対応する
        Pydantic サブモデルインスタンスを自動復元して返却する。

        Args:
            dag_definition (dict[str, Any]): DagDefinition スキーマ準拠の辞書。
            state (Any): 判断コンテキスト。
            timeout_ms (int | None): クライアント側指定のタイムアウト (ミリ秒)。
            synthesizer (DagSynthesizer | None): DAG 自動合成器インスタンス。

        Returns:
            Any: synthesizer 指定時は復元された BaseModel、未指定時は DagResponse 辞書。

        Raises:
            SokutoClientError: 実行に失敗した場合。
        """
        payload: dict[str, Any] = {
            "state": state,
            "dag": dag_definition,
        }
        if timeout_ms is not None:
            payload["timeout_ms"] = timeout_ms

        req_bytes = orjson.dumps(payload)

        try:
            response = self._client.post(
                "/v1/systemone/dag",
                content=req_bytes,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise SokutoClientError(
                f"sokuto-server (/v1/systemone/dag) の実行に失敗しました: {e}"
            ) from e

        res_dict = orjson.loads(response.content)
        if synthesizer is not None:
            return synthesizer.deserialize_dag_response(res_dict)
        return res_dict


class AsyncSokutoClient:
    """sokuto-server 向け非同期推論クライアント。

    Attributes:
        base_url (str): サーバーベース URL。
        timeout (float): タイムアウト秒数。
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 10.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """非同期クライアントを初期化する。

        Args:
            base_url (str): サーバーベース URL。
            timeout (float): タイムアウト秒数。
            http_client (httpx.AsyncClient | None): 既存の非同期 HTTP クライアント。
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.last_state_hash: str | None = None
        self._client = (
            http_client
            if http_client is not None
            else httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        )

    def get_last_state_hash(self) -> str | None:
        """直近にサーバーから受信した State キャッシュキーを取得する。"""
        return self.last_state_hash

    def clear_state_hash(self) -> None:
        """記憶している State キャッシュキーをクリアする。"""
        self.last_state_hash = None

    async def close(self) -> None:
        """HTTP クライアントのリソースを解放する。"""
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        await self.close()

    async def predict(
        self,
        model_cls: type[M],
        state: Any,
        prefix: str | None = None,
        model_name: str | None = None,
        auto_hierarchical: bool | None = None,
        fail_on_uncertainty: bool = False,
        min_confidence: float = 0.0,
        enable_state_offloading: bool = False,
        state_hash: str | None = None,
    ) -> M:
        """非同期で単一フォワードパス推論を実行し、型安全なインスタンスを復元する。

        Args:
            model_cls (type[M]): 期待される出力 Pydantic モデルクラス。
            state (Any): コンテキスト。
            prefix (str | None): 質問キープレフィックス。
            model_name (str | None): sokuto モデル識別子。
            auto_hierarchical (bool | None): 階層ルーティング設定。
            fail_on_uncertainty (bool): 不確実性判定フラグ。
            min_confidence (float): 最小確信度。
            enable_state_offloading (bool): State 本文の省略 (キャッシュ再利用) を有効化するかどうか。
            state_hash (str | None): クライアント指定の State キャッシュキー。

        Returns:
            M: 復元された Pydantic モデルインスタンス。
        """
        template: CompiledTemplate = compile_schema(
            model_cls,
            prefix=prefix,
            model_name=model_name,
            auto_hierarchical=auto_hierarchical,
        )

        effective_hash = state_hash or self.last_state_hash
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if effective_hash:
            headers["X-State-Hash"] = effective_hash

        sent_state = "" if (enable_state_offloading and effective_hash) else state
        request_bytes = template.build_request_bytes(sent_state)

        try:
            response = await self._client.post(
                "/v1/systemone",
                content=request_bytes,
                headers=headers,
            )
            if (
                response.status_code == 400
                and enable_state_offloading
                and sent_state == ""
                and state != ""
            ):
                fallback_bytes = template.build_request_bytes(state)
                headers.pop("X-State-Hash", None)
                response = await self._client.post(
                    "/v1/systemone",
                    content=fallback_bytes,
                    headers=headers,
                )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise SokutoClientError(
                f"sokuto-server (/v1/systemone) への非同期推論リクエストに失敗しました: {e}"
            ) from e

        rec_hash = response.headers.get("x-state-hash") or response.headers.get("X-State-Hash")
        if rec_hash:
            self.last_state_hash = rec_hash

        return template.deserialize(
            response.content,
            fail_on_uncertainty=fail_on_uncertainty,
            min_confidence=min_confidence,
        )

    async def predict_batch(
        self,
        state: Any,
        questions: dict[str, Any],
        model_name: str | None = None,
        auto_hierarchical: bool | None = None,
        gating: dict[str, Any] | None = None,
        enable_state_offloading: bool = False,
        state_hash: str | None = None,
    ) -> dict[str, Any]:
        """単一 State に対する複数質問を非同期一括送信し、判定結果マップを取得する。

        Args:
            state (Any): 判断コンテキスト。
            questions (dict[str, Any]): 質問定義マップ。
            model_name (str | None): モデル名。
            auto_hierarchical (bool | None): 階層探索フラグ。
            gating (dict[str, Any] | None): ゲーティング設定。
            enable_state_offloading (bool): State 本文の省略最適化。
            state_hash (str | None): キャッシュキー。

        Returns:
            dict[str, Any]: StateBatchResponse 準拠の辞書。
        """
        effective_hash = state_hash or self.last_state_hash
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if effective_hash:
            headers["X-State-Hash"] = effective_hash

        sent_state = "" if (enable_state_offloading and effective_hash) else state
        payload: dict[str, Any] = {
            "state": sent_state,
            "questions": questions,
        }
        if model_name is not None:
            payload["model"] = model_name
        if auto_hierarchical is not None:
            payload["auto_hierarchical"] = auto_hierarchical
        if gating is not None:
            payload["gating"] = gating
        if effective_hash is not None:
            payload["state_hash"] = effective_hash

        req_bytes = orjson.dumps(payload)

        try:
            response = await self._client.post(
                "/v1/systemone/state-batch",
                content=req_bytes,
                headers=headers,
            )
            if (
                response.status_code == 400
                and enable_state_offloading
                and sent_state == ""
                and state != ""
            ):
                payload["state"] = state
                payload.pop("state_hash", None)
                headers.pop("X-State-Hash", None)
                response = await self._client.post(
                    "/v1/systemone/state-batch",
                    content=orjson.dumps(payload),
                    headers=headers,
                )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise SokutoClientError(
                f"sokuto-server (/v1/systemone/state-batch) への非同期バッチ推論に失敗しました: {e}"
            ) from e

        rec_hash = response.headers.get("x-state-hash") or response.headers.get("X-State-Hash")
        if rec_hash:
            self.last_state_hash = rec_hash

        result_dict: dict[str, Any] = orjson.loads(response.content)
        cache_status = (
            response.headers.get("x-cache-status")
            or response.headers.get("X-Cache-Status")
            or "BYPASS"
        )
        result_dict["cache_status"] = cache_status
        if rec_hash:
            result_dict["state_hash"] = rec_hash

        return result_dict

    async def execute_models(
        self,
        models: list[type[BaseModel]],
        state: Any,
        prefix_list: list[str] | None = None,
        model_name: str | None = None,
        auto_hierarchical: bool | None = None,
        fail_on_uncertainty: bool = False,
        min_confidence: float = 0.0,
        enable_state_offloading: bool = False,
    ) -> list[BaseModel]:
        """同一 State に対して複数の異なる Pydantic モデルを非同期で一括解決する。

        Args:
            models (list[type[BaseModel]]): 解決対象モデルクラスのリスト。
            state (Any): 共通コンテキストデータ。
            prefix_list (list[str] | None): 各モデルのプレフィックス一覧。
            model_name (str | None): モデル識別子。
            auto_hierarchical (bool | None): 粗密階層フラグ。
            fail_on_uncertainty (bool): 不確実性例外フラグ。
            min_confidence (float): 最小確信度。
            enable_state_offloading (bool): State 本文省略フラグ。

        Returns:
            list[BaseModel]: 復元された各モデルインスタンスのリスト。
        """
        templates: list[CompiledTemplate] = []
        merged_questions: dict[str, Any] = {}

        for i, model_cls in enumerate(models):
            pfx = prefix_list[i] if prefix_list and i < len(prefix_list) else f"m{i}_"
            t = compile_schema(
                model_cls,
                prefix=pfx,
                model_name=model_name,
                auto_hierarchical=auto_hierarchical,
            )
            templates.append(t)
            merged_questions.update(t.cached_questions)

        batch_result = await self.predict_batch(
            state=state,
            questions=merged_questions,
            model_name=model_name,
            auto_hierarchical=auto_hierarchical,
            enable_state_offloading=enable_state_offloading,
        )

        resp_bytes = orjson.dumps(batch_result)
        restored: list[BaseModel] = []
        for t in templates:
            inst = t.deserialize(
                resp_bytes,
                fail_on_uncertainty=fail_on_uncertainty,
                min_confidence=min_confidence,
            )
            restored.append(inst)

        return restored

    async def execute_dag(
        self,
        dag_definition: dict[str, Any],
        state: Any,
        timeout_ms: int | None = None,
        synthesizer: DagSynthesizer | None = None,
    ) -> Any:
        """Phase 6 のインプロセス DAG を非同期実行する。

        synthesizer が指定された場合、DAG の実行結果から対応する
        Pydantic サブモデルインスタンスを自動復元して返却する。

        Args:
            dag_definition (dict[str, Any]): DagDefinition スキーマ準拠辞書。
            state (Any): コンテキスト。
            timeout_ms (int | None): タイムアウト (ミリ秒)。
            synthesizer (DagSynthesizer | None): DAG 自動合成器インスタンス。

        Returns:
            Any: synthesizer 指定時は復元された BaseModel、未指定時は DagResponse 辞書。
        """
        payload: dict[str, Any] = {
            "state": state,
            "dag": dag_definition,
        }
        if timeout_ms is not None:
            payload["timeout_ms"] = timeout_ms

        req_bytes = orjson.dumps(payload)

        try:
            response = await self._client.post(
                "/v1/systemone/dag",
                content=req_bytes,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise SokutoClientError(
                f"sokuto-server (/v1/systemone/dag) の非同期実行に失敗しました: {e}"
            ) from e

        res_dict = orjson.loads(response.content)
        if synthesizer is not None:
            return synthesizer.deserialize_dag_response(res_dict)
        return res_dict
