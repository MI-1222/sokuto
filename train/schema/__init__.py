"""Pydantic v2 スキーマ駆動推論基盤パッケージ。

Python の標準的な Pydantic v2 モデル定義から sokuto-server への
型安全な代数射影、事前コンパイルバイトキャッシュ、Discriminated Union DAG 自動合成、
および不確実性メタデータを保持した高速復元を提供する。
"""

from .client import AsyncSokutoClient, SokutoClient, SokutoClientError
from .compiler import CompiledTemplate, compile_schema, sokuto_schema
from .dag_synthesizer import DagSynthesisError, DagSynthesizer
from .transpiler import (
    FieldMappingError,
    InvalidScoreRangeError,
    SchemaTranspiler,
    clean_key,
)
from .wrapper import SokutoMeta, Uncertain, UncertainDecisionError

__all__ = [
    "AsyncSokutoClient",
    "CompiledTemplate",
    "DagSynthesisError",
    "DagSynthesizer",
    "FieldMappingError",
    "InvalidScoreRangeError",
    "SchemaTranspiler",
    "SokutoClient",
    "SokutoClientError",
    "SokutoMeta",
    "Uncertain",
    "UncertainDecisionError",
    "clean_key",
    "compile_schema",
    "sokuto_schema",
]
