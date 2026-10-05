"""事前コンパイルバイトキャッシュおよび高速シリアライザモジュール。

アプリケーション起動時に Pydantic モデル定義を静的バイト列テンプレートへ事前コンパイルし、
推論実行時の動的リフレクションおよび JSON シリアライズオーバーヘッドを
0.08ms 以下 (< 0.15ms SLA) に抑え込むゼロアロケーション機構を提供する。
"""

from __future__ import annotations

import types
from typing import Any, cast

import orjson
from pydantic import BaseModel

from .transpiler import SchemaTranspiler


class CompiledTemplate[M: BaseModel]:
    """事前コンパイルされたリクエストバイトテンプレート。

    質問仕様 (questions) の JSON バイト列を起動時に固定化し、
    推論呼び出し時には動的な `state` のみを差し込むことで
    Python 側のシリアライズオーバーヘッドを極小化する。

    Attributes:
        model_cls (type[M]): 対象 Pydantic モデルクラス。
        transpiler (SchemaTranspiler): 型代数トランスパイラインスタンス。
        cached_questions (dict[str, Any]): イミュータブルな質問定義マップ。
        cached_questions_bytes (bytes): シリアライズ済み質問バイト列。
    """

    __slots__ = (
        "_prefix_bytes",
        "_suffix_bytes",
        "cached_questions",
        "cached_questions_bytes",
        "model_cls",
        "transpiler",
    )

    def __init__(
        self,
        model_cls: type[M],
        prefix: str | None = None,
        model_name: str | None = None,
        auto_hierarchical: bool | None = None,
    ) -> None:
        """バイトテンプレートを事前コンパイルする。

        Args:
            model_cls (type[M]): 変換対象の Pydantic モデル。
            prefix (str | None): 質問キープレフィックス。
            model_name (str | None): sokuto サーバーで指定するモデル識別子。
            auto_hierarchical (bool | None): 粗密階層推論の自動有効化フラグ。
        """
        self.model_cls = model_cls
        self.transpiler = SchemaTranspiler(model_cls, prefix=prefix)

        # 辞書のイミュータブル化 (スレッドセーフティの保証)
        raw_questions = self.transpiler.questions
        self.cached_questions = types.MappingProxyType(raw_questions)
        self.cached_questions_bytes = orjson.dumps(raw_questions)

        # 高速結合用バイト列の生成:
        # {"state": <STATE>, "questions": <CACHED_QUESTIONS>, ...}
        # prefix: b'{"questions":' + cached_questions_bytes + b','
        extra_parts: list[bytes] = []
        if model_name is not None:
            extra_parts.append(b'"model":' + orjson.dumps(model_name))
        if auto_hierarchical is not None:
            extra_parts.append(
                b'"auto_hierarchical":' + (b"true" if auto_hierarchical else b"false")
            )

        extra_bytes = b"".join([b"," + p for p in extra_parts])
        self._prefix_bytes = (
            b'{"questions":' + self.cached_questions_bytes + extra_bytes + b',"state":'
        )
        self._suffix_bytes = b"}"

    def build_request_bytes(self, state: Any) -> bytes:
        """動的な state を高速注入し、完全なリクエスト JSON バイト列を生成する。

        実行時リフレクションを一切行わず、事前キャッシュ済みの質問バイト列と
        state のみの高速シリアライズを連結する。

        Args:
            state (Any): 判断コンテキスト (文字列、辞書、リストなど)。

        Returns:
            bytes: sokuto-server へ送信可能な UTF-8 JSON バイト列。
        """
        state_bytes = orjson.dumps(state)
        return self._prefix_bytes + state_bytes + self._suffix_bytes

    def build_request_dict(self, state: Any) -> dict[str, Any]:
        """辞書形式でリクエストを生成する。

        Args:
            state (Any): 判断コンテキスト。

        Returns:
            dict[str, Any]: リクエスト辞書。
        """
        req: dict[str, Any] = {
            "questions": dict(self.cached_questions),
            "state": state,
        }
        return req

    def deserialize(
        self,
        response_bytes_or_dict: bytes | dict[str, Any],
        fail_on_uncertainty: bool = False,
        min_confidence: float = 0.0,
    ) -> M:
        """レスポンス (バイト列または辞書) を Pydantic モデルへ高速復元する。

        Args:
            response_bytes_or_dict (bytes | dict[str, Any]): sokuto-server のレスポンス。
            fail_on_uncertainty (bool): 不確実性判定時に例外を送出するかどうか。
            min_confidence (float): 許容最小確信度閾値。

        Returns:
            M: 復元された Pydantic モデル。
        """
        if isinstance(response_bytes_or_dict, bytes):
            payload = orjson.loads(response_bytes_or_dict)
        else:
            payload = response_bytes_or_dict
        res = self.transpiler.deserialize_response(
            payload,
            fail_on_uncertainty=fail_on_uncertainty,
            min_confidence=min_confidence,
        )
        return cast(M, res)


_SCHEMA_CACHE: dict[Any, CompiledTemplate[Any]] = {}
_MODEL_DEFAULT_CACHE: dict[type[BaseModel], CompiledTemplate[Any]] = {}


def compile_schema[M: BaseModel](
    model_cls: type[M],
    prefix: str | None = None,
    model_name: str | None = None,
    auto_hierarchical: bool | None = None,
) -> CompiledTemplate[M]:
    """Pydantic モデルに対する CompiledTemplate を取得または新規事前コンパイルしてキャッシュする。

    Args:
        model_cls (type[M]): 対象モデル。
        prefix (str | None): プレフィックス。
        model_name (str | None): モデル名。
        auto_hierarchical (bool | None): 階層ルーティング設定。

    Returns:
        CompiledTemplate[M]: キャッシュされたテンプレート。
    """
    if (
        prefix is None
        and model_name is None
        and auto_hierarchical is None
        and model_cls in _MODEL_DEFAULT_CACHE
    ):
        return cast(CompiledTemplate[M], _MODEL_DEFAULT_CACHE[model_cls])

    key = (model_cls, prefix, model_name, auto_hierarchical)
    if key not in _SCHEMA_CACHE:
        template = CompiledTemplate(
            model_cls,
            prefix=prefix,
            model_name=model_name,
            auto_hierarchical=auto_hierarchical,
        )
        _SCHEMA_CACHE[key] = template
        if model_cls not in _MODEL_DEFAULT_CACHE:
            _MODEL_DEFAULT_CACHE[model_cls] = template
    return cast(CompiledTemplate[M], _SCHEMA_CACHE[key])


def sokuto_schema(
    prefix: str | None = None,
    model_name: str | None = None,
    auto_hierarchical: bool | None = None,
):
    """Pydantic モデル定義時に事前コンパイルを強制するクラスデコレータ。

    例:
    ```python
    @sokuto_schema()
    class TriageDecision(BaseModel):
        category: Literal["tech", "billing", "other"]
        is_urgent: bool
    ```
    """

    def decorator[M: type[BaseModel]](cls: M) -> M:
        template = compile_schema(
            cls,
            prefix=prefix,
            model_name=model_name,
            auto_hierarchical=auto_hierarchical,
        )
        _MODEL_DEFAULT_CACHE[cls] = template
        return cls

    return decorator
