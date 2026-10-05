"""Discriminated Union および階層依存モデルから Phase 6 インプロセス DAG への自動合成モジュール。

Pydantic の Field(discriminator=...) や Literal 分岐を静的解析し、
Phase 6 互換の宣言的 DagDefinition (JSON スキーマ) を決定論的に自動生成する。
"""

from __future__ import annotations

import inspect
import types
from typing import (
    Annotated,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from .transpiler import SchemaTranspiler


class DagSynthesisError(ValueError):
    """DAG 自動合成処理において構造破綻や循環参照が検出された場合の例外。"""


class DagSynthesizer:
    """判別共用体 (Discriminated Union) から Phase 6 インプロセス DAG を合成するエンジン。

    Attributes:
        dag_id (str): 合成される DAG の識別子。
        timeout_ms (int): DAG の全体タイムアウト時間 (ミリ秒)。既定値は 100ms。
    """

    def __init__(self, dag_id: str = "synthesized_dag", timeout_ms: int = 100) -> None:
        """DAG 合成エンジンを初期化する。

        Args:
            dag_id (str): DAG 識別子。
            timeout_ms (int): タイムアウト制限値 (ミリ秒)。
        """
        self.dag_id = dag_id
        self.timeout_ms = timeout_ms
        self._union_type: Any | None = None
        self._branch_model_map: dict[str, type[BaseModel]] = {}
        self._discriminator_field: str | None = None

    def synthesize_from_discriminated_union(
        self,
        union_type: Any,
        discriminator_field: str | None = None,
    ) -> dict[str, Any]:
        """Discriminated Union 型定義から Phase 6 互換の DAG 定義辞書を自動生成する。

        例:
        ```python
        class ModelA(BaseModel):
            status: Literal["approved"]
            credit_score: Annotated[int, Field(ge=1, le=5)]

        class ModelB(BaseModel):
            status: Literal["rejected"]
            reject_reason: Literal["fraud", "policy", "other"]

        union_type = Annotated[Union[ModelA, ModelB], Field(discriminator="status")]
        dag = synthesizer.synthesize_from_discriminated_union(union_type)
        ```

        Args:
            union_type (Any): Annotated[Union[...], Field(discriminator=...)] または Union[...] 型。
            discriminator_field (str | None): 判別キーフィールド名。型アノテーションから取得できない場合に指定。

        Returns:
            dict[str, Any]: Phase 6 の DagDefinition スキーマに準拠した辞書。

        Raises:
            DagSynthesisError: 判別キーが見つからない場合や循環参照が存在する場合。
        """
        sub_models, disc_field = self._extract_union_models(
            union_type, discriminator_field
        )
        if not sub_models:
            raise DagSynthesisError(
                "Discriminated Union に含まれるサブモデルが存在しません。"
            )

        # 循環参照検査 (モデルクラスの ID を確認)
        visited_ids: set[int] = set()
        for m in sub_models:
            mid = id(m)
            if mid in visited_ids:
                raise DagSynthesisError(
                    f"モデル '{m.__name__}' で循環参照が検出されました。"
                )
            visited_ids.add(mid)

        # 1. 判別キーの全選択肢と Criteria の収集
        choice_criteria: dict[str, str] = {}
        branch_model_map: dict[str, type[BaseModel]] = {}

        for model_cls in sub_models:
            if disc_field not in model_cls.model_fields:
                raise DagSynthesisError(
                    f"サブモデル '{model_cls.__name__}' に判別キー '{disc_field}' が定義されていません。"
                )
            disc_info = model_cls.model_fields[disc_field]
            disc_annotation = disc_info.annotation
            origin = get_origin(disc_annotation)
            args = get_args(disc_annotation)

            if origin is not Literal or not args:
                raise DagSynthesisError(
                    f"サブモデル '{model_cls.__name__}' の判別キー '{disc_field}' は Literal 型である必要があります。"
                )

            for literal_val in args:
                literal_str = str(literal_val)
                desc = (
                    disc_info.description
                    or model_cls.__doc__
                    or f"Model {model_cls.__name__}"
                )
                choice_criteria[literal_str] = desc.strip()
                branch_model_map[literal_str] = model_cls

        # 2. ルート推論ノード (Discriminator 判定) の作成
        entry_node_id = f"eval_{disc_field}"
        nodes: dict[str, Any] = {}

        root_question = {
            "type": "choice",
            "instructions": f"Determine the {disc_field} classification.",
            "criteria": choice_criteria,
        }

        # Discriminator の条件分岐エッジ
        conditions = []
        for lit_val in branch_model_map:
            target_branch_id = f"node_{lit_val}"
            conditions.append(
                {
                    "field": "answer",
                    "op": "eq",
                    "value": lit_val,
                    "target_node": target_branch_id,
                }
            )

        nodes[entry_node_id] = {
            "node_type": "inference",
            "question": root_question,
            "conditions": conditions,
        }

        # 3. 各ブランチ配下の推論ノード群の生成
        for lit_val, model_cls in branch_model_map.items():
            transpiler = SchemaTranspiler(
                model_cls, prefix=f"{model_cls.__name__}_{lit_val}"
            )
            # 判別キー以外のフィールドを抽出
            child_fields = [
                (fn, fi)
                for fn, fi in model_cls.model_fields.items()
                if fn != disc_field
            ]

            if not child_fields:
                # 追加入力不要な場合は終端ノード
                target_node_id = f"node_{lit_val}"
                nodes[target_node_id] = {
                    "node_type": "branch",
                    "conditions": [],
                    "reason": f"Completed determination for {lit_val}",
                }
            elif len(child_fields) == 1:
                # 単一フィールドの場合は直接推論ノード
                fn, _ = child_fields[0]
                q_key = transpiler._qualify_key(fn)
                q_def = transpiler.questions[q_key]
                target_node_id = f"node_{lit_val}"
                nodes[target_node_id] = {
                    "node_type": "inference",
                    "question": q_def,
                    "conditions": [],
                }
            else:
                # 複数フィールドがある場合は Parallel または直列チェーンノードを作成
                # Phase 6 の Parallel ノードを活用
                branch_entry_id = f"node_{lit_val}"
                sub_parallel_node_ids = []

                for fn, _ in child_fields:
                    sub_node_id = f"{lit_val}_{fn}"
                    q_key = transpiler._qualify_key(fn)
                    q_def = transpiler.questions[q_key]
                    nodes[sub_node_id] = {
                        "node_type": "inference",
                        "question": q_def,
                        "conditions": [],
                    }
                    sub_parallel_node_ids.append(sub_node_id)

                join_node_id = f"join_{lit_val}"
                nodes[join_node_id] = {
                    "node_type": "branch",
                    "conditions": [],
                    "reason": f"Join parallel evaluation for {lit_val}",
                }

                nodes[branch_entry_id] = {
                    "node_type": "parallel",
                    "parallel_nodes": sub_parallel_node_ids,
                    "join_node": join_node_id,
                    "conditions": [],
                }

        # 4. DagDefinition スキーマ辞書の構築
        dag_dict = {
            "dag_id": self.dag_id,
            "timeout_ms": self.timeout_ms,
            "entry_node": entry_node_id,
            "nodes": nodes,
        }

        self._union_type = union_type
        self._discriminator_field = disc_field
        self._branch_model_map = branch_model_map

        return dag_dict

    def deserialize_dag_response(
        self,
        dag_response: dict[str, Any],
        union_type: Any | None = None,
    ) -> BaseModel:
        """DAG 実行結果 (DagResponse) から対応する Pydantic サブモデルインスタンスを復元する。

        実行パスおよび各ステップの Answer 辞書を走査し、
        採択された判別ブランチに対応するサブモデルのフィールドを特定して
        型安全にインスタンス化する。

        Args:
            dag_response (dict[str, Any]): sokuto-server (/v1/systemone/dag) のレスポンス辞書。
            union_type (Any | None): 対象の判別共用体型。None の場合は直近の合成型を使用。

        Returns:
            BaseModel: 復元された Pydantic サブモデルインスタンス。

        Raises:
            DagSynthesisError: エスカレーション終了時、またはモデル復元に失敗した場合。
        """
        if union_type is not None and union_type != self._union_type:
            # 新たな型が渡された場合は再解析
            sub_models, disc_field = self._extract_union_models(
                union_type, self._discriminator_field
            )
            branch_model_map = {}
            for m in sub_models:
                info = m.model_fields[disc_field]
                args = get_args(info.annotation)
                for a in args:
                    branch_model_map[str(a)] = m
        else:
            disc_field = self._discriminator_field
            branch_model_map = self._branch_model_map

        if not disc_field or not branch_model_map:
            raise DagSynthesisError(
                "DAG の判別モデル情報が初期化されていません。先に synthesize_from_discriminated_union を実行するか union_type を指定してください。"
            )

        status = dag_response.get("status")
        if status == "escalated":
            escalation = dag_response.get("escalation") or {}
            raise DagSynthesisError(
                f"DAG 実行がエスカレーション終了しました: {escalation}"
            )

        steps = dag_response.get("steps", {})

        # 1. 判別キーの値を取得
        entry_node_id = f"eval_{disc_field}"
        entry_step = steps.get(entry_node_id, {})
        entry_ans = entry_step.get("answer", {})
        disc_val = entry_ans.get("choice")

        if disc_val is None:
            raise DagSynthesisError(
                f"エントリノード '{entry_node_id}' から判別キー '{disc_field}' の判定値を取得できませんでした。"
            )

        target_model = branch_model_map.get(str(disc_val))
        if target_model is None:
            raise DagSynthesisError(
                f"判別値 '{disc_val}' に対応するサブモデルが登録されていません。"
            )

        # 2. 各フィールドの判定結果を steps から集約
        answers_for_model: dict[str, Any] = {}
        answers_for_model[disc_field] = {"choice": disc_val}

        child_fields = [
            (fn, fi) for fn, fi in target_model.model_fields.items() if fn != disc_field
        ]

        if len(child_fields) == 1:
            fn, _ = child_fields[0]
            step_node_id = f"node_{disc_val}"
            step_obj = steps.get(step_node_id, {})
            if "answer" in step_obj:
                answers_for_model[fn] = step_obj["answer"]
        elif len(child_fields) > 1:
            for fn, _ in child_fields:
                step_node_id = f"{disc_val}_{fn}"
                step_obj = steps.get(step_node_id, {})
                if "answer" in step_obj:
                    answers_for_model[fn] = step_obj["answer"]

        # SchemaTranspiler を用いてデシリアライズ
        transpiler = SchemaTranspiler(
            target_model, prefix=f"{target_model.__name__}_{disc_val}"
        )
        fake_response = {"answers": answers_for_model}
        return transpiler.deserialize_response(fake_response)

    def _extract_union_models(
        self,
        union_type: Any,
        explicit_discriminator: str | None,
    ) -> tuple[list[type[BaseModel]], str]:
        """Union 型からサブモデルクラスのリストと判別キー名を抽出する。"""
        disc_field = explicit_discriminator

        if isinstance(union_type, (list, tuple)):
            args = tuple(union_type)
            origin = Union
        else:
            origin = get_origin(union_type)
            args = get_args(union_type)

            # Annotated[Union[...], Field(discriminator="...")] の展開
            if origin is Annotated:
                base_type = args[0]
                for metadata in args[1:]:
                    if hasattr(metadata, "discriminator") and metadata.discriminator:
                        disc_field = metadata.discriminator
                    elif (
                        isinstance(metadata, FieldInfo)
                        and metadata.discriminator is not None
                    ):
                        disc_field = str(metadata.discriminator)
                origin = get_origin(base_type)
                args = get_args(base_type)

            # Union / UnionType の確認
            if origin is not Union and origin is not types.UnionType:
                raise DagSynthesisError(
                    f"型 '{union_type}' は Union または Annotated[Union, ...] である必要があります。"
                )

        models: list[type[BaseModel]] = []
        for arg in args:
            if inspect.isclass(arg) and issubclass(arg, BaseModel):
                models.append(arg)
            else:
                raise DagSynthesisError(
                    f"Union の構成要素 '{arg}' は Pydantic の BaseModel である必要があります。"
                )

        if disc_field is None:
            # 各モデルに共通する単一の Literal フィールドを自動探索
            common_literal_fields: set[str] = set()
            for i, m in enumerate(models):
                lit_fields = {
                    fn
                    for fn, fi in m.model_fields.items()
                    if get_origin(fi.annotation) is Literal
                }
                if i == 0:
                    common_literal_fields = lit_fields
                else:
                    common_literal_fields &= lit_fields

            if len(common_literal_fields) == 1:
                disc_field = next(iter(common_literal_fields))
            else:
                raise DagSynthesisError(
                    "判別キー (discriminator) を特定できませんでした。Field(discriminator='...') を明示してください。"
                )

        return models, disc_field
