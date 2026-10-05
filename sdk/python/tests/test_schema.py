"""Pydantic v2 スキーマ駆動推論基盤の網羅的テストスイート。

代数射影、Fast-Fail 検出、Discriminated Union DAG 合成、
事前コンパイルバイトキャッシュ、命令的バリデータ境界、
不確実性ラッパー、および 0.15ms SLA マイクロベンチマークを検証する。
"""

from __future__ import annotations

import enum
import time
from typing import Annotated, Literal

import httpx
import orjson
import pytest
from pydantic import BaseModel, Field, field_validator, model_validator

from sokuto.schema import (
    AsyncSokutoClient,
    DagSynthesisError,
    DagSynthesizer,
    FieldMappingError,
    InvalidScoreRangeError,
    SchemaTranspiler,
    SokutoClient,
    SokutoMeta,
    Uncertain,
    UncertainDecisionError,
    compile_schema,
    sokuto_schema,
)


class PriorityEnum(enum.Enum):
    """タスク優先度列挙型。"""

    LOW = "低優先度"
    MEDIUM = "通常優先度"
    HIGH = "緊急対応要"


class SimpleDecision(BaseModel):
    """単純な基本型モデル。"""

    is_verified: bool = Field(description="本人確認が完了しているか。")
    priority: PriorityEnum = Field(description="案件の優先度。")
    risk_level: Annotated[int, Field(ge=1, le=5, description="リスク評価 (1〜5)。")]
    action: Literal["approve", "review", "reject"] = Field(description="適用すべきアクション。")


class UncertainDecision(BaseModel):
    """不確実性ラッパーを含むモデル。"""

    is_fraud: Uncertain[bool] = Field(description="不正検知フラグ。")
    severity: Uncertain[Annotated[int, Field(ge=1, le=4, description="重要度レベル。")]]
    target_queue: Uncertain[Literal["support", "security", "ops"]] = Field(
        description="ルーティング先キュー。",
        json_schema_extra={
            "descriptions": {
                "support": "サポート窓口",
                "security": "セキュリティチーム",
                "ops": "運用チーム",
            }
        },
    )


class ImperativeValidationModel(BaseModel):
    """命令的バリデータを持つモデル。"""

    is_urgent: bool
    score: Annotated[int, Field(ge=1, le=5)]

    @field_validator("score")
    @classmethod
    def validate_score_custom(cls, v: int) -> int:
        if v == 3:
            raise ValueError("スコア 3 はビジネスルール上受け入れられません。")
        return v

    @model_validator(mode="after")
    def check_consistency(self) -> ImperativeValidationModel:
        if self.is_urgent and self.score < 4:
            raise ValueError("緊急時はスコア 4 以上でなければなりません。")
        return self


class TenFieldCompositeModel(BaseModel):
    """10 フィールド構成の複合意思決定モデル (SLA ベンチマーク用)。"""

    f1_bool: bool = Field(description="フラグ1。")
    f2_bool: bool = Field(description="フラグ2。")
    f3_literal: Literal["A", "B", "C"] = Field(
        description="選択肢3。",
        json_schema_extra={"descriptions": {"A": "区分A", "B": "区分B", "C": "区分C"}},
    )
    f4_literal: Literal["X", "Y"] = Field(
        description="選択肢4。",
        json_schema_extra={"descriptions": {"X": "種別X", "Y": "種別Y"}},
    )
    f5_enum: PriorityEnum = Field(description="優先度5。")
    f6_score: Annotated[int, Field(ge=1, le=5, description="スコア6。")]
    f7_score: Annotated[int, Field(ge=1, le=3, description="スコア7。")]
    f8_uncertain_bool: Uncertain[bool] = Field(description="不確実フラグ8。")
    f9_uncertain_literal: Uncertain[Literal["opt1", "opt2", "opt3"]] = Field(
        description="不確実選択肢9。",
        json_schema_extra={"descriptions": {"opt1": "候補1", "opt2": "候補2", "opt3": "候補3"}},
    )
    f10_uncertain_score: Uncertain[
        Annotated[int, Field(ge=1, le=4, description="不確実スコア10。")]
    ]


# --- 1. 型代数射影 & Fast-Fail テスト ---


def test_algebraic_transpilation_primitives() -> None:
    """基本型が sokuto の Question 仕様へ正確に射影されることを検証する。"""
    transpiler = SchemaTranspiler(SimpleDecision, prefix="test")
    q = transpiler.questions

    # bool -> Noul
    assert "test.is_verified" in q
    assert q["test.is_verified"]["type"] == "noul"
    assert q["test.is_verified"]["criteria"] is None
    assert q["test.is_verified"]["instructions"] == "本人確認が完了しているか。"

    # Enum -> Choice
    assert "test.priority" in q
    assert q["test.priority"]["type"] == "choice"
    assert q["test.priority"]["criteria"] == {
        "LOW": "低優先度",
        "MEDIUM": "通常優先度",
        "HIGH": "緊急対応要",
    }

    # Constrained int -> Score (2〜10段階)
    assert "test.risk_level" in q
    assert q["test.risk_level"]["type"] == "score"
    assert q["test.risk_level"]["criteria"] == [
        "Grade 1",
        "Grade 2",
        "Grade 3",
        "Grade 4",
        "Grade 5",
    ]

    # Literal -> Choice
    assert "test.action" in q
    assert q["test.action"]["type"] == "choice"
    assert "approve" in q["test.action"]["criteria"]


def test_unsupported_types_fast_fail() -> None:
    """未対応の型が指定された場合、起動時コンパイルで即座に拒絶されることを検証する。"""

    class InvalidStringModel(BaseModel):
        text: str  # 非 Literal の文字列

    with pytest.raises(FieldMappingError) as exc_info:
        SchemaTranspiler(InvalidStringModel)
    assert "sokuto プリミティブへ直接トランスパイルできません" in str(exc_info.value)

    class InvalidUnconstrainedIntModel(BaseModel):
        count: int  # 制約なし int

    with pytest.raises(FieldMappingError) as exc_info:
        SchemaTranspiler(InvalidUnconstrainedIntModel)
    assert "ge と le の範囲制約が必要です" in str(exc_info.value)


def test_invalid_score_range_fast_fail() -> None:
    """Score の 2〜10 段階制約に違反した場合に Fast-Fail することを検証する。"""

    class ZeroStepScoreModel(BaseModel):
        val: Annotated[int, Field(ge=5, le=5)]  # 1段階 (不足)

    with pytest.raises(InvalidScoreRangeError) as exc_info:
        SchemaTranspiler(ZeroStepScoreModel)
    assert "sokuto の Score プリミティブ制約 (2〜10 段階) に違反しています" in str(exc_info.value)

    class OverStepScoreModel(BaseModel):
        score_100: Annotated[int, Field(ge=0, le=100)]  # 101段階 (超過)

    with pytest.raises(InvalidScoreRangeError) as exc_info:
        SchemaTranspiler(OverStepScoreModel)
    assert "sokuto の Score プリミティブ制約 (2〜10 段階) に違反しています" in str(exc_info.value)


# --- 2. Discriminated Union DAG 自動合成テスト ---


class ApproveSubModel(BaseModel):
    status: Literal["approved"] = Field(description="承認ステータス。")
    credit_limit: Annotated[int, Field(ge=1, le=5, description="与信限度額グレード。")]


class RejectSubModel(BaseModel):
    status: Literal["rejected"] = Field(description="却下ステータス。")
    reject_reason: Literal["fraud", "policy", "low_income"] = Field(description="却下事由。")


class CircularModelA(BaseModel):
    status: Literal["a"]


def test_dag_synthesizer_from_discriminated_union() -> None:
    """Discriminated Union から Phase 6 互換 DAG 定義が自動生成されることを検証する。"""
    union_type = Annotated[ApproveSubModel | RejectSubModel, Field(discriminator="status")]

    synthesizer = DagSynthesizer(dag_id="credit_dag", timeout_ms=80)
    dag = synthesizer.synthesize_from_discriminated_union(union_type)

    assert dag["dag_id"] == "credit_dag"
    assert dag["timeout_ms"] == 80
    assert dag["entry_node"] == "eval_status"

    nodes = dag["nodes"]
    assert "eval_status" in nodes
    entry_node = nodes["eval_status"]
    assert entry_node["node_type"] == "inference"
    assert entry_node["question"]["type"] == "choice"

    # 条件付き遷移エッジの検証
    conditions = entry_node["conditions"]
    assert len(conditions) == 2
    cond_map = {c["value"]: c["target_node"] for c in conditions}
    assert cond_map["approved"] == "node_approved"
    assert cond_map["rejected"] == "node_rejected"

    # 派生先ノードの検証
    assert "node_approved" in nodes
    assert nodes["node_approved"]["question"]["type"] == "score"

    assert "node_rejected" in nodes
    assert nodes["node_rejected"]["question"]["type"] == "choice"


def test_dag_synthesizer_cycle_detection() -> None:
    """循環参照が検出された場合に DagSynthesisError を投げることを検証する。"""
    synthesizer = DagSynthesizer()
    # 同一モデルの重複シーケンスを指定して循環参照を検出
    invalid_union = [CircularModelA, CircularModelA]
    with pytest.raises(DagSynthesisError) as exc_info:
        synthesizer.synthesize_from_discriminated_union(invalid_union, discriminator_field="status")
    assert "循環参照が検出されました" in str(exc_info.value)


def test_dag_synthesizer_deserialize_dag_response() -> None:
    """DAG 実行レスポンスから採択ブランチのサブモデルインスタンスが型安全に復元されることを検証する。"""
    union_type = Annotated[ApproveSubModel | RejectSubModel, Field(discriminator="status")]
    synthesizer = DagSynthesizer(dag_id="credit_dag")
    _ = synthesizer.synthesize_from_discriminated_union(union_type)

    # 承認ブランチのモックレスポンス
    approved_dag_response = {
        "dag_id": "credit_dag",
        "status": "completed",
        "execution_path": ["eval_status", "node_approved"],
        "steps": {
            "eval_status": {
                "node_id": "eval_status",
                "answer": {"choice": "approved", "confidence": 0.95},
            },
            "node_approved": {
                "node_id": "node_approved",
                "answer": {"score": 4.0, "confidence": 0.90},
            },
        },
    }

    result_model = synthesizer.deserialize_dag_response(approved_dag_response)
    assert isinstance(result_model, ApproveSubModel)
    assert result_model.status == "approved"
    assert result_model.credit_limit == 4

    # 却下ブランチのモックレスポンス
    rejected_dag_response = {
        "dag_id": "credit_dag",
        "status": "completed",
        "execution_path": ["eval_status", "node_rejected"],
        "steps": {
            "eval_status": {
                "node_id": "eval_status",
                "answer": {"choice": "rejected", "confidence": 0.92},
            },
            "node_rejected": {
                "node_id": "node_rejected",
                "answer": {"choice": "fraud", "confidence": 0.88},
            },
        },
    }

    reject_model = synthesizer.deserialize_dag_response(rejected_dag_response)
    assert isinstance(reject_model, RejectSubModel)
    assert reject_model.status == "rejected"
    assert reject_model.reject_reason == "fraud"


# --- 3. 事前コンパイルバイトキャッシュ & 高速シリアライザテスト ---


def test_compiled_template_bytes_generation() -> None:
    """CompiledTemplate によるリクエストバイト列が正しく整形されることを検証する。"""
    template = compile_schema(SimpleDecision, prefix="req", auto_hierarchical=True)
    req_bytes = template.build_request_bytes(state="ユーザーがログインを試行")

    parsed = orjson.loads(req_bytes)
    assert parsed["state"] == "ユーザーがログインを試行"
    assert parsed["auto_hierarchical"] is True
    assert "questions" in parsed
    assert "req.is_verified" in parsed["questions"]


def test_sokuto_schema_decorator() -> None:
    """@sokuto_schema デコレータによりクラス定義時に事前コンパイルされることを検証する。"""

    @sokuto_schema(prefix="deco")
    class DecoratedModel(BaseModel):
        flag: bool

    template = compile_schema(DecoratedModel)
    assert template.transpiler.prefix == "deco"
    assert "deco.flag" in template.cached_questions


# --- 4. 型安全復元 & 不確実性ラッパーテスト ---


def test_deserialize_simple_decision() -> None:
    """モックレスポンスから SimpleDecision が完全かつ型安全に復元されることを検証する。"""
    template = compile_schema(SimpleDecision, prefix="test")

    mock_response = {
        "answers": {
            "test.is_verified": {"noul": 0.88},
            "test.priority": {
                "choice": "HIGH",
                "confidence": 0.92,
                "probabilities": {"LOW": 0.02, "MEDIUM": 0.06, "HIGH": 0.92},
            },
            "test.risk_level": {
                "score": 4.12,
                "confidence": 0.85,
            },
            "test.action": {
                "choice": "approve",
                "confidence": 0.94,
            },
        }
    }

    result = template.deserialize(mock_response)
    assert isinstance(result, SimpleDecision)
    assert result.is_verified is True
    assert result.priority == PriorityEnum.HIGH
    assert result.risk_level == 4
    assert result.action == "approve"


def test_deserialize_uncertain_decision() -> None:
    """Uncertain[T] を含むモデルがメタデータを保持して復元されることを検証する。"""
    template = compile_schema(UncertainDecision, prefix="unc")

    mock_response = {
        "answers": {
            "unc.is_fraud": {
                "noul": 0.95,
                "gating": {"energy": -1.8, "route": "auto_execute"},
            },
            "unc.severity": {
                "score": 3.0,
                "confidence": 0.82,
                "gating": {"energy": -1.2},
            },
            "unc.target_queue": {
                "choice": "security",
                "confidence": 0.89,
                "probabilities": {
                    "support": 0.05,
                    "security": 0.89,
                    "ops": 0.06,
                },
                "gating": {"top_margin": 0.83, "entropy": 0.15},
            },
        }
    }

    result = template.deserialize(mock_response)
    assert isinstance(result, UncertainDecision)

    # is_fraud: Uncertain[bool]
    assert isinstance(result.is_fraud, Uncertain)
    assert result.is_fraud.value is True
    assert result.is_fraud.meta.effective_confidence == pytest.approx(
        0.90, abs=1e-3
    )  # |2*0.95 - 1| = 0.90
    assert result.is_fraud.meta.free_energy == -1.8
    assert result.is_fraud.is_confident(0.70) is True
    assert result.is_fraud.is_ood(-1.0) is False

    # severity: Uncertain[int]
    assert result.severity.value == 3
    assert result.severity.meta.confidence == 0.82

    # target_queue: Uncertain[str]
    assert result.target_queue.value == "security"
    assert result.target_queue.meta.top_margin == 0.83


def test_uncertain_pydantic_core_schema() -> None:
    """Uncertain[T] が Pydantic v2 の core_schema に統合され、

    model_dump や辞書からの model_validate が arbitrary_types_allowed なしで動作することを検証する。
    """
    meta = SokutoMeta(
        confidence=0.92,
        effective_confidence=0.92,
        free_energy=-1.5,
    )
    unc = Uncertain(value=True, meta=meta)

    # 1. 直接インスタンスを渡して検証
    model1 = UncertainDecision(
        is_fraud=unc,
        severity=Uncertain(value=2, meta=meta),
        target_queue=Uncertain(value="ops", meta=meta),
    )
    assert model1.is_fraud.value is True

    # 2. model_dump() によるシリアライズ検証
    dumped = model1.model_dump()
    assert isinstance(dumped["is_fraud"], dict)
    assert dumped["is_fraud"]["value"] is True
    assert dumped["is_fraud"]["meta"]["confidence"] == 0.92

    # 3. 辞書からの model_validate による復元検証
    reconstructed = UncertainDecision.model_validate(dumped)
    assert isinstance(reconstructed.is_fraud, Uncertain)
    assert reconstructed.is_fraud.value is True
    assert reconstructed.is_fraud.meta.confidence == 0.92


def test_uncertain_decision_fail_fast() -> None:
    """確信度不足時に UncertainDecisionError が正しく送出されることを検証する。"""
    template = compile_schema(SimpleDecision, prefix="test")

    mock_low_conf_response = {
        "answers": {
            "test.is_verified": {"noul": 0.52},  # 尖り度 |2*0.52 - 1| = 0.04
            "test.priority": {"choice": "LOW", "confidence": 0.40},
            "test.risk_level": {"score": 1.0, "confidence": 0.50},
            "test.action": {"choice": "reject", "confidence": 0.45},
        }
    }

    with pytest.raises(UncertainDecisionError) as exc_info:
        template.deserialize(
            mock_low_conf_response,
            fail_on_uncertainty=True,
            min_confidence=0.70,
        )
    assert "不確実性が高いためエスカレーションが必要です" in str(exc_info.value)


# --- 5. 命令的バリデータ境界テスト ---


def test_imperative_validators_client_boundary() -> None:
    """クライアント側の @field_validator / @model_validator が復元時に実行されることを検証する。"""
    template = compile_schema(ImperativeValidationModel, prefix="val")

    # 1. フィールドバリデータ違反 (score == 3)
    mock_invalid_score = {
        "answers": {
            "val.is_urgent": {"noul": 0.1},
            "val.score": {"score": 3.0},
        }
    }
    with pytest.raises(ValueError) as exc_info:
        template.deserialize(mock_invalid_score)
    assert "スコア 3 はビジネスルール上受け入れられません" in str(exc_info.value)

    # 2. モデルバリデータ違反 (is_urgent=True かつ score < 4)
    mock_inconsistent = {
        "answers": {
            "val.is_urgent": {"noul": 0.9},
            "val.score": {"score": 2.0},
        }
    }
    with pytest.raises(ValueError) as exc_info:
        template.deserialize(mock_inconsistent)
    assert "緊急時はスコア 4 以上でなければなりません" in str(exc_info.value)

    # 3. 正常系
    mock_valid = {
        "answers": {
            "val.is_urgent": {"noul": 0.9},
            "val.score": {"score": 4.8},  # round -> 5
        }
    }
    instance = template.deserialize(mock_valid)
    assert instance.is_urgent is True
    assert instance.score == 5


# --- 6. SokutoClient 通信モックテスト ---


def test_sokuto_client_mock() -> None:
    """SokutoClient の同期リクエストが正常に送受信・復元されることを検証する。"""

    def handler(request: httpx.Request) -> httpx.Response:
        req_data = orjson.loads(request.content)
        assert "questions" in req_data
        assert req_data["state"] == "テストコンテキスト"

        resp_data = {
            "answers": {
                "test.is_verified": {"noul": 0.95},
                "test.priority": {"choice": "HIGH", "confidence": 0.90},
                "test.risk_level": {"score": 2.0, "confidence": 0.88},
                "test.action": {"choice": "approve", "confidence": 0.95},
            },
            "usage": {"prompt_tokens": 42, "completion_tokens": 0},
        }
        return httpx.Response(200, content=orjson.dumps(resp_data))

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, base_url="http://mock-server")

    with SokutoClient(base_url="http://mock-server", http_client=http_client) as client:
        result = client.predict(SimpleDecision, state="テストコンテキスト", prefix="test")
        assert result.is_verified is True
        assert result.priority == PriorityEnum.HIGH


def test_async_sokuto_client_mock() -> None:
    """AsyncSokutoClient の非同期リクエストが正常に動作することを検証する。"""
    import asyncio

    def handler(request: httpx.Request) -> httpx.Response:
        resp_data = {
            "answers": {
                "test.is_verified": {"noul": 0.99},
                "test.priority": {"choice": "LOW", "confidence": 0.90},
                "test.risk_level": {"score": 1.0, "confidence": 0.95},
                "test.action": {"choice": "review", "confidence": 0.85},
            },
            "usage": {"prompt_tokens": 30, "completion_tokens": 0},
        }
        return httpx.Response(200, content=orjson.dumps(resp_data))

    async def _run() -> None:
        transport = httpx.MockTransport(handler)
        async_http = httpx.AsyncClient(transport=transport, base_url="http://mock-server")

        async with AsyncSokutoClient(
            base_url="http://mock-server", http_client=async_http
        ) as client:
            result = await client.predict(SimpleDecision, state="非同期コンテキスト", prefix="test")
            assert result.is_verified is True
            assert result.action == "review"

    asyncio.run(_run())


def test_sokuto_client_execute_dag_with_synthesizer() -> None:
    """SokutoClient.execute_dag で synthesizer を指定した場合に、

    サブモデルインスタンスへ自動復元されることを検証する。
    """
    union_type = Annotated[ApproveSubModel | RejectSubModel, Field(discriminator="status")]
    synthesizer = DagSynthesizer(dag_id="credit_dag")
    dag_def = synthesizer.synthesize_from_discriminated_union(union_type)

    def handler(request: httpx.Request) -> httpx.Response:
        req_data = orjson.loads(request.content)
        assert "dag" in req_data
        assert req_data["state"] == "DAGコンテキスト"

        resp_data = {
            "dag_id": "credit_dag",
            "status": "completed",
            "execution_path": ["eval_status", "node_approved"],
            "steps": {
                "eval_status": {
                    "node_id": "eval_status",
                    "answer": {"choice": "approved", "confidence": 0.98},
                },
                "node_approved": {
                    "node_id": "node_approved",
                    "answer": {"score": 5.0, "confidence": 0.95},
                },
            },
            "usage": {"prompt_tokens": 50, "completion_tokens": 0},
        }
        return httpx.Response(200, content=orjson.dumps(resp_data))

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport, base_url="http://mock-server")

    with SokutoClient(base_url="http://mock-server", http_client=http_client) as client:
        # synthesizer を指定して実行
        result = client.execute_dag(
            dag_definition=dag_def,
            state="DAGコンテキスト",
            synthesizer=synthesizer,
        )
        assert isinstance(result, ApproveSubModel)
        assert result.status == "approved"
        assert result.credit_limit == 5


# --- 7. マイクロベンチマーク (SLA <= 0.15ms) & 100% 型整合性検証 ---


def test_microbenchmark_and_type_integrity() -> None:
    """10 フィールド複合モデルにおいて、事前コンパイル後の

    シリアライズ＋デシリアライズ遅延が 0.15ms 以下であることを検証する (10,000 回ループ)。
    また、型整合性率 100.0% (型エラー 0) を検証する。
    """
    template = compile_schema(TenFieldCompositeModel, prefix="bench")

    # モックレスポンスの準備
    mock_response = {
        "answers": {
            "bench.f1_bool": {"noul": 0.8},
            "bench.f2_bool": {"noul": 0.2},
            "bench.f3_literal": {"choice": "B", "confidence": 0.9},
            "bench.f4_literal": {"choice": "Y", "confidence": 0.85},
            "bench.f5_enum": {"choice": "HIGH", "confidence": 0.95},
            "bench.f6_score": {"score": 4.0, "confidence": 0.88},
            "bench.f7_score": {"score": 2.0, "confidence": 0.92},
            "bench.f8_uncertain_bool": {"noul": 0.9, "gating": {"energy": -1.5}},
            "bench.f9_uncertain_literal": {
                "choice": "opt2",
                "confidence": 0.87,
                "gating": {"margin": 0.4},
            },
            "bench.f10_uncertain_score": {
                "score": 3.0,
                "confidence": 0.80,
                "gating": {"entropy": 0.2},
            },
        }
    }
    resp_bytes = orjson.dumps(mock_response)
    state_input = "会員ランクゴールド、直近ログインあり、決済エラー頻度大"

    # ウォームアップ (100 回)
    for _ in range(100):
        _ = template.build_request_bytes(state_input)
        res = template.deserialize(resp_bytes)
        assert res.f3_literal == "B"

    # 計測 (10,000 回)
    iterations = 10_000
    latencies: list[float] = []

    start_total = time.perf_counter()
    for _ in range(iterations):
        t0 = time.perf_counter()
        _req_b = template.build_request_bytes(state_input)
        _res = template.deserialize(resp_bytes)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms 単位
    total_elapsed_ms = (time.perf_counter() - start_total) * 1000.0

    avg_latency_ms = total_elapsed_ms / iterations
    latencies.sort()
    p99_latency_ms = latencies[int(iterations * 0.99)]

    print(
        f"\n[Microbenchmark] 10-Field Composite Model (10,000 iterations): "
        f"Avg = {avg_latency_ms:.4f} ms, P99 = {p99_latency_ms:.4f} ms"
    )

    # Exit Criteria: トランスパイル＋検証遅延 <= 0.15ms
    assert avg_latency_ms <= 0.15, f"平均遅延 {avg_latency_ms:.4f}ms が SLA 0.15ms を超過しました。"
    assert p99_latency_ms <= 0.25, f"P99 遅延 {p99_latency_ms:.4f}ms が許容上限を超過しました。"
