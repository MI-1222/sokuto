"""Pydantic v2 スキーマから sokuto プリミティブへの型代数トランスパイラモジュール。

Pydantic の型注釈 (Literal, Enum, Annotated[int, ...], bool, Uncertain[T]) を解析し、
sokuto の質問仕様 (Choice, Score, Noul) への決定論的代数射影および
推論結果の型安全な復元を行う。
"""

from __future__ import annotations

import contextlib
import enum
import inspect
import re
import warnings
from typing import (
    Annotated,
    Any,
    Literal,
    get_args,
    get_origin,
)

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from .wrapper import SokutoMeta, Uncertain, UncertainDecisionError


def clean_key(key: str) -> str:
    """スネークケースまたはケバブケースの識別子を自然言語の表記へ展開する。

    Args:
        key (str): 変換対象のキー文字列。

    Returns:
        str: 読みやすく整形された説明文。
    """
    cleaned = re.sub(r"[_\-]+", " ", str(key)).strip()
    return cleaned.capitalize() if cleaned else str(key)


class FieldMappingError(TypeError):
    """sokuto プリミティブへ射影できない型が指定された場合の例外。"""


class InvalidScoreRangeError(ValueError):
    """Score プリミティブの段階数制約 (2〜10 段階) に違反した場合の例外。"""


class SchemaTranspiler:
    """Pydantic v2 の BaseModel 定義を sokuto 質問スキーマへ射影・復元するトランスパイラ。

    Attributes:
        model_cls (type[BaseModel]): 変換対象の Pydantic モデルクラス。
        prefix (str): 質問キーに付与するプレフィックス (名前空間分離用)。
    """

    def __init__(
        self,
        model_cls: type[BaseModel],
        prefix: str | None = None,
    ) -> None:
        """トランスパイラを初期化し、スキーマの構文解析を実行する。

        Args:
            model_cls (type[BaseModel]): 対象の Pydantic モデルクラス。
            prefix (str | None): 質問キーのプレフィックス。None の場合はクラス名を使用。

        Raises:
            TypeError: サポート外の型が含まれている場合。
            ValueError: 制約条件が不正な場合。
        """
        if not (inspect.isclass(model_cls) and issubclass(model_cls, BaseModel)):
            raise TypeError(
                f"model_cls は Pydantic の BaseModel のサブクラスである必要があります: {model_cls}"
            )
        self.model_cls = model_cls
        self.prefix = prefix if prefix is not None else model_cls.__name__
        self._field_meta_types: dict[str, bool] = {}  # field_name -> is_uncertain
        self._field_base_types: dict[str, Any] = {}  # field_name -> base_type
        self._compiled_questions: dict[str, dict[str, Any]] = self._compile_questions()

    @property
    def questions(self) -> dict[str, dict[str, Any]]:
        """コンパイル済みの質問マップ (キー: 完全修飾名) を取得する。"""
        return self._compiled_questions

    def _qualify_key(self, field_name: str) -> str:
        """フィールド名から完全修飾された質問識別子を生成する。"""
        return f"{self.prefix}.{field_name}"

    def _unqualify_key(self, qualified_key: str) -> str:
        """完全修飾識別子からフィールド名を取り出す。"""
        if qualified_key.startswith(f"{self.prefix}."):
            return qualified_key[len(self.prefix) + 1 :]
        return qualified_key

    def _unwrap_type(self, annotation: Any) -> tuple[Any, bool]:
        """Uncertain[T] や Annotated などのラッパーを剥がして基底型と Uncertain フラグを判定する。

        Args:
            annotation (Any): 型注釈。

        Returns:
            tuple[Any, bool]: (基底型, Uncertain[T] であったかどうか)。
        """
        origin = get_origin(annotation)
        if origin is Uncertain:
            args = get_args(annotation)
            base_type = args[0] if args else Any
            return base_type, True
        return annotation, False

    def _compile_questions(self) -> dict[str, dict[str, Any]]:
        """全モデルフィールドを走査して sokuto 質問辞書を生成する。"""
        questions: dict[str, dict[str, Any]] = {}
        for field_name, field_info in self.model_cls.model_fields.items():
            raw_annotation = field_info.annotation
            base_type, is_uncertain = self._unwrap_type(raw_annotation)
            self._field_meta_types[field_name] = is_uncertain
            self._field_base_types[field_name] = base_type

            q_def = self._map_field_to_question(field_name, base_type, field_info)
            qualified_key = self._qualify_key(field_name)
            questions[qualified_key] = q_def
        return questions

    def _map_field_to_question(
        self,
        name: str,
        annotation: Any,
        info: FieldInfo,
    ) -> dict[str, Any]:
        """単一フィールドの型定義を sokuto Question 仕様へ射影する。

        Args:
            name (str): フィールド名。
            annotation (Any): アンラップ済みの型注釈。
            info (FieldInfo): Pydantic のフィールドメタデータ。

        Returns:
            dict[str, Any]: sokuto Question 辞書。

        Raises:
            FieldMappingError: 射影不可能な型の場合。
            InvalidScoreRangeError: Score の段階数制約に違反した場合。
        """
        origin = get_origin(annotation)
        args = get_args(annotation)

        # 指示文の決定: Field(description=...) があれば優先、なければ自動生成
        instructions = info.description or f"Determine the appropriate value for {name}."

        # 1. bool -> Noul
        if annotation is bool:
            return {
                "type": "noul",
                "instructions": instructions,
                "criteria": None,
            }

        # 2. Literal[...] -> Choice
        if origin is Literal:
            criteria = self._extract_literal_criteria(args, info)
            return {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }

        # 3. Enum -> Choice
        if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
            criteria = self._extract_enum_criteria(annotation, info)
            return {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }

        # 4. Annotated[int, Field(ge=..., le=...)] または制約付き int -> Score
        if (
            annotation is int
            or origin is Annotated
            or (origin is not None and issubclass(origin, int))
        ):
            score_q = self._try_map_score(name, annotation, info, instructions)
            if score_q is not None:
                return score_q

        # サポート外の型に対する Fast-Fail
        raise FieldMappingError(
            f"フィールド '{name}' の型 '{annotation}' は sokuto プリミティブへ直接トランスパイルできません。"
            "サポートされている型は bool (Noul), Literal/Enum (Choice), "
            "Annotated[int, Field(ge=A, le=B)] (Score; 2〜10段階), およびそれらを包む Uncertain[T] です。"
        )

    def _extract_literal_criteria(
        self,
        args: tuple[Any, ...],
        info: FieldInfo,
    ) -> dict[str, str]:
        """Literal の引数群から Choice 用の Criteria マップを生成する。"""
        if not args:
            raise FieldMappingError("Literal には最低 1 つの選択肢が必要です。")

        # json_schema_extra に各選択肢の説明文があるか探索
        extra = info.json_schema_extra
        desc_map: dict[str, str] = {}
        if isinstance(extra, dict) and "descriptions" in extra:
            desc_map = extra["descriptions"]

        criteria: dict[str, str] = {}
        for arg in args:
            str_arg = str(arg)
            if str_arg in desc_map:
                criteria[str_arg] = desc_map[str_arg]
            else:
                criteria[str_arg] = clean_key(str_arg)

        # 全てが未定義（自動展開されたもの）かつ Field(description) もない場合は警告
        if not desc_map and not info.description:
            warnings.warn(
                f"Literal 候補 {args} に説明文 (Criteria) が指定されていません。"
                "分類精度を最大化するために Field(description=...) または json_schema_extra['descriptions'] の指定を推奨します。",
                UserWarning,
                stacklevel=3,
            )
        return criteria

    def _extract_enum_criteria(
        self,
        enum_cls: type[enum.Enum],
        info: FieldInfo,
    ) -> dict[str, str]:
        """Enum クラスから Choice 用の Criteria マップを生成する。"""
        criteria: dict[str, str] = {}
        has_custom_desc = False

        for member in enum_cls:
            val = member.value
            # タプル形式 ("KEY", "説明文") の場合
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                criteria[member.name] = str(val[1])
                has_custom_desc = True
            elif isinstance(val, str) and val != member.name:
                criteria[member.name] = val
                has_custom_desc = True
            elif member.__doc__ and member.__doc__ != enum.Enum.__doc__:
                criteria[member.name] = member.__doc__.strip()
                has_custom_desc = True
            else:
                criteria[member.name] = clean_key(member.name)

        if not has_custom_desc and not info.description:
            warnings.warn(
                f"Enum '{enum_cls.__name__}' に説明文 (Criteria) が指定されていません。"
                "分類精度を維持するために Enum の docstring またはタプル値形式の指定を推奨します。",
                UserWarning,
                stacklevel=3,
            )
        return criteria

    def _try_map_score(
        self,
        name: str,
        annotation: Any,
        info: FieldInfo,
        instructions: str,
    ) -> dict[str, Any] | None:
        """Annotated や FieldInfo のメタデータから ge/le 制約を抽出し、Score 質問を生成する。"""
        ge: int | None = None
        le: int | None = None

        metadata_items: list[Any] = list(getattr(info, "metadata", []))

        # annotation が Annotated の場合、その引数からもメタデータおよび FieldInfo を走査
        origin = get_origin(annotation)
        if origin is Annotated:
            for arg in get_args(annotation)[1:]:
                metadata_items.append(arg)
                if isinstance(arg, FieldInfo):
                    metadata_items.extend(getattr(arg, "metadata", []))

        for meta in metadata_items:
            if hasattr(meta, "ge") and meta.ge is not None:
                ge = int(meta.ge)
            if hasattr(meta, "le") and meta.le is not None:
                le = int(meta.le)

        if ge is None or le is None:
            # 制約が不足している int 型はトランスパイル不可
            raise FieldMappingError(
                f"整数型フィールド '{name}' には ge と le の範囲制約が必要です "
                "(例: Annotated[int, Field(ge=1, le=5)])。"
            )

        step_count = le - ge + 1
        if step_count < 2 or step_count > 10:
            raise InvalidScoreRangeError(
                f"フィールド '{name}' の段階数 {step_count} (ge={ge}, le={le}) は "
                "sokuto の Score プリミティブ制約 (2〜10 段階) に違反しています。"
                "11 段階以上の評価が必要な場合は、段階の粗視化または Choice へのバケット分割を検討してください。"
            )

        # カスタムラベル定義が json_schema_extra にあれば使用
        criteria: list[str] = []
        extra = info.json_schema_extra
        if (
            isinstance(extra, dict)
            and "labels" in extra
            and isinstance(extra["labels"], list)
            and len(extra["labels"]) == step_count
        ):
            criteria = [str(x) for x in extra["labels"]]

        if not criteria:
            criteria = [f"Grade {step}" for step in range(ge, le + 1)]

        return {
            "type": "score",
            "instructions": instructions,
            "criteria": criteria,
        }

    def deserialize_response(
        self,
        response_payload: dict[str, Any],
        fail_on_uncertainty: bool = False,
        min_confidence: float = 0.0,
    ) -> BaseModel:
        """sokuto-server のレスポンスペイロードから Pydantic モデルインスタンスを復元する。

        命令的バリデータ (@model_validator, @field_validator) はこの段階で
        クライアント側プロセスにおいて自動的に実行される。

        Args:
            response_payload (dict[str, Any]): sokuto-server (SystemOneResponse) の JSON ペイロード。
            fail_on_uncertainty (bool): 不確実性が高い場合に例外を送出するかどうか。
            min_confidence (float): 許容する最小確信度閾値。

        Returns:
            BaseModel: 復元・検証された Pydantic モデルインスタンス。

        Raises:
            UncertainDecisionError: 確信度が不足し fail_on_uncertainty が有効な場合。
            pydantic.ValidationError: モデルのバリデーションに失敗した場合。
        """
        answers = response_payload.get("answers", {})
        raw_data: dict[str, Any] = {}

        for field_name, field_info in self.model_cls.model_fields.items():
            qualified_key = self._qualify_key(field_name)
            # 修飾名または素のフィールド名で探索
            ans = answers.get(qualified_key) or answers.get(field_name)
            if ans is None:
                # デフォルト値が存在する場合はスキップ
                if not field_info.is_required():
                    continue
                raise KeyError(
                    f"レスポンス内にフィールド '{field_name}' (キー: {qualified_key}) の回答が含まれていません。"
                )

            sokuto_meta = SokutoMeta.from_answer_dict(ans)

            # 確信度チェック
            if (
                fail_on_uncertainty
                and min_confidence > 0.0
                and sokuto_meta.effective_confidence < min_confidence
            ):
                raise UncertainDecisionError(
                    field_name=field_name,
                    meta=sokuto_meta,
                    reason=f"実効確信度 ({sokuto_meta.effective_confidence:.3f}) が必要下限 ({min_confidence:.3f}) を下回りました。",
                )

            # 確定値の取り出し
            val: Any = None
            if ans.get("choice") is not None:
                val = ans["choice"]
                # Enum フィールドの場合、メンバー名または値から Enum メンバーへ復元
                base_type = self._field_base_types.get(field_name)
                if inspect.isclass(base_type) and issubclass(base_type, enum.Enum):
                    if isinstance(val, str) and val in base_type.__members__:
                        val = base_type[val]
                    else:
                        with contextlib.suppress(ValueError):
                            val = base_type(val)
            elif ans.get("score") is not None:
                val = round(ans["score"])
            elif ans.get("noul") is not None:
                val = bool(ans["noul"] >= 0.5)
            else:
                raise ValueError(
                    f"回答辞書に choice, score, noul のいずれも含まれていません: {ans}"
                )

            # Uncertain[T] 型かどうかに応じて格納
            is_uncertain = self._field_meta_types.get(field_name, False)
            if is_uncertain:
                raw_data[field_name] = Uncertain(value=val, meta=sokuto_meta)
            else:
                raw_data[field_name] = val

        # クライアント側での命令的バリデーション実行
        return self.model_cls.model_validate(raw_data)
