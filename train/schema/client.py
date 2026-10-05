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
        self._client = (
            http_client
            if http_client is not None
            else httpx.Client(base_url=self.base_url, timeout=self.timeout)
        )

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

        request_bytes = template.build_request_bytes(state)

        try:
            response = self._client.post(
                "/v1/systemone",
                content=request_bytes,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise SokutoClientError(
                f"sokuto-server (/v1/systemone) への推論リクエストに失敗しました: {e}"
            ) from e

        return template.deserialize(
            response.content,
            fail_on_uncertainty=fail_on_uncertainty,
            min_confidence=min_confidence,
        )

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
        self._client = (
            http_client
            if http_client is not None
            else httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        )

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

        Returns:
            M: 復元された Pydantic モデルインスタンス。
        """
        template: CompiledTemplate = compile_schema(
            model_cls,
            prefix=prefix,
            model_name=model_name,
            auto_hierarchical=auto_hierarchical,
        )

        request_bytes = template.build_request_bytes(state)

        try:
            response = await self._client.post(
                "/v1/systemone",
                content=request_bytes,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise SokutoClientError(
                f"sokuto-server (/v1/systemone) への非同期推論リクエストに失敗しました: {e}"
            ) from e

        return template.deserialize(
            response.content,
            fail_on_uncertainty=fail_on_uncertainty,
            min_confidence=min_confidence,
        )

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
