//! # OpenAPI スキーマ整合性・回帰テストスイート
//!
//! Serde によるシリアライズ/デシリアライズの実測結果と、
//! `utoipa` から自動導出された OpenAPI 3.1 仕様書のプロパティ整合性を検証する。

use indexmap::IndexMap;
use sokuto_core::schema::{
    Answer, Criteria, Question, QuestionType, SystemOneRequest, SystemOneResponse, Usage,
};
use sokuto_server::openapi::generate_openapi_spec;

#[test]
fn test_openapi_spec_generation_and_components() {
    let spec = generate_openapi_spec();

    // 仕様書の基本メタデータ検証
    assert_eq!(spec.info.title, "Sokuto HTTP API");
    assert_eq!(spec.info.version, env!("CARGO_PKG_VERSION"));

    // コンポーネントスキーマの登録確認
    let components = spec.components.expect("components が定義されていること。");
    let schemas = components.schemas;

    let required_schemas = [
        "SystemOneRequest",
        "SystemOneResponse",
        "Question",
        "QuestionType",
        "Criteria",
        "Answer",
        "Usage",
        "DecisionRoute",
        "GatingConfig",
        "GatingMetadata",
        "CandidateProbability",
        "EscalationContext",
        "SystemRoutingSummary",
        "ErrorResponse",
        "ErrorDetail",
        "HealthStatusResponse",
        "DagRequest",
        "DagResponse",
        "DagDefinitionDto",
        "DagNodeDto",
        "DagNodeTypeDto",
        "NodeConditionDto",
        "NodeConditionOpDto",
        "DagStepResultDto",
        "DagEscalationDetailDto",
        "DagUsageDto",
    ];

    for schema_name in required_schemas {
        assert!(
            schemas.contains_key(schema_name),
            "スキーマ '{schema_name}' が OpenAPI 仕様書に含まれていません。"
        );
    }

    assert!(
        spec.paths.paths.contains_key("/v1/systemone/dag"),
        "エンドポイント '/v1/systemone/dag' が OpenAPI 仕様書に含まれていません。"
    );
}

#[test]
fn test_decision_route_enum_casing_parity() {
    let spec = generate_openapi_spec();
    let spec_json = serde_json::to_value(&spec).expect("JSON シリアライズに成功すること。");

    let route_schema = &spec_json["components"]["schemas"]["DecisionRoute"];
    let enum_values = route_schema["enum"]
        .as_array()
        .expect("enum 定義が存在すること。");

    let string_values: Vec<&str> = enum_values.iter().filter_map(|v| v.as_str()).collect();

    // serde(rename_all = "snake_case") と完全一致すること
    assert_eq!(
        string_values,
        vec!["auto_execute", "confirm_or_escalate", "fallback"]
    );

    use sokuto_core::gating::DecisionRoute;
    assert_eq!(
        serde_json::to_string(&DecisionRoute::AutoExecute).unwrap(),
        "\"auto_execute\""
    );
    assert_eq!(
        serde_json::to_string(&DecisionRoute::ConfirmOrEscalate).unwrap(),
        "\"confirm_or_escalate\""
    );
    assert_eq!(
        serde_json::to_string(&DecisionRoute::Fallback).unwrap(),
        "\"fallback\""
    );
}

#[test]
fn test_question_type_enum_casing_parity() {
    let spec = generate_openapi_spec();
    let spec_json = serde_json::to_value(&spec).expect("JSON シリアライズに成功すること。");

    // QuestionType の enum 定義を抽出
    let question_type_schema = &spec_json["components"]["schemas"]["QuestionType"];
    let enum_values = question_type_schema["enum"]
        .as_array()
        .expect("enum 定義が存在すること。");

    let string_values: Vec<&str> = enum_values.iter().filter_map(|v| v.as_str()).collect();

    // serde(rename_all = "lowercase") と完全一致すること
    assert_eq!(string_values, vec!["choice", "score", "noul"]);

    // Serde 側の実測値
    let serde_choice = serde_json::to_string(&QuestionType::Choice).unwrap();
    let serde_score = serde_json::to_string(&QuestionType::Score).unwrap();
    let serde_noul = serde_json::to_string(&QuestionType::Noul).unwrap();

    assert_eq!(serde_choice, "\"choice\"");
    assert_eq!(serde_score, "\"score\"");
    assert_eq!(serde_noul, "\"noul\"");
}

#[test]
fn test_question_field_rename_parity() {
    let spec = generate_openapi_spec();
    let spec_json = serde_json::to_value(&spec).expect("JSON シリアライズに成功すること。");

    let question_schema = &spec_json["components"]["schemas"]["Question"];
    let properties = &question_schema["properties"];

    // `question_type` ではなく `type` としてエクスポートされていること
    assert!(
        properties.get("type").is_some(),
        "Question スキーマに 'type' プロパティが存在すること。"
    );
    assert!(
        properties.get("question_type").is_none(),
        "内部フィールド名 'question_type' は露出していないこと。"
    );

    // Serde シリアライズ結果との突合
    let mut criteria_map = IndexMap::new();
    criteria_map.insert("a".to_string(), "オプション A".to_string());
    let question = Question::new_choice("指示文", criteria_map);

    let json_val = serde_json::to_value(&question).unwrap();
    assert!(json_val.get("type").is_some());
    assert!(json_val.get("question_type").is_none());
}

#[test]
fn test_systemone_request_roundtrip_and_schema_parity() {
    let spec = generate_openapi_spec();
    let spec_json = serde_json::to_value(&spec).expect("JSON シリアライズに成功すること。");

    let req_schema = &spec_json["components"]["schemas"]["SystemOneRequest"];
    let properties = &req_schema["properties"];

    // 必須および主要フィールドの確認
    assert!(properties.get("state").is_some());
    assert!(properties.get("questions").is_some());
    assert!(properties.get("model").is_some());

    // 実リクエストペイロードとの互換性
    let mut questions = IndexMap::new();
    questions.insert("q1".to_string(), Question::new_noul("言明文"));

    let req = SystemOneRequest::new("共通コンテキスト", questions);
    let req_json = serde_json::to_value(&req).unwrap();

    assert_eq!(req_json["state"], "共通コンテキスト");
    assert!(req_json["questions"]["q1"].is_object());
    assert_eq!(req_json["questions"]["q1"]["type"], "noul");
}

#[test]
fn test_systemone_response_schema_parity() {
    let spec = generate_openapi_spec();
    let spec_json = serde_json::to_value(&spec).expect("JSON シリアライズに成功すること。");

    let resp_schema = &spec_json["components"]["schemas"]["SystemOneResponse"];
    let properties = &resp_schema["properties"];

    assert!(properties.get("answers").is_some());
    assert!(properties.get("usage").is_some());

    let mut answers = IndexMap::new();
    answers.insert(
        "q1".to_string(),
        Answer::choice("opt_a", IndexMap::new(), 0.95),
    );

    let resp = SystemOneResponse::new(answers, Usage::new(42));
    let resp_json = serde_json::to_value(&resp).unwrap();

    assert!(resp_json["answers"]["q1"].is_object());
    assert_eq!(resp_json["answers"]["q1"]["choice"], "opt_a");
    assert_eq!(resp_json["answers"]["q1"]["confidence"], 0.95);
    assert_eq!(resp_json["usage"]["prompt_tokens"], 42);
    assert_eq!(resp_json["usage"]["completion_tokens"], 0);
    assert_eq!(resp_json["usage"]["total_tokens"], 42);
}

#[test]
fn test_answer_flat_schema_parity() {
    let spec = generate_openapi_spec();
    let spec_json = serde_json::to_value(&spec).expect("JSON シリアライズに成功すること。");

    let schemas = &spec_json["components"]["schemas"];
    let answer_schema = &schemas["Answer"];
    assert!(
        answer_schema.is_object(),
        "Answer スキーマがオブジェクト定義として存在すること。"
    );

    let properties = &answer_schema["properties"];

    // 内部直和型フィールド payload が露出していないことの検証
    assert!(
        properties.get("payload").is_none(),
        "Answer スキーマに内部フィールド 'payload' が露出してはいけません。"
    );
    assert!(
        !schemas.as_object().unwrap().contains_key("AnswerPayload"),
        "内部実装 enum 'AnswerPayload' は OpenAPI コンポーネントに露出してはいけません。"
    );

    // フラット化された公開フィールド群の存在検証
    let expected_fields = [
        "choice",
        "score",
        "noul",
        "probabilities",
        "confidence",
        "gating",
    ];
    for field in expected_fields {
        assert!(
            properties.get(field).is_some(),
            "Answer スキーマにフラットフィールド '{field}' が定義されていること。"
        );
    }

    // 実際のシリアライズ結果との整合性検証 (Choice, Score, Noul)
    let choice_ans = Answer::choice("cand_1", IndexMap::new(), 0.9);
    let choice_val = serde_json::to_value(&choice_ans).unwrap();
    for key in choice_val.as_object().unwrap().keys() {
        assert!(
            properties.get(key).is_some(),
            "Choice JSON のキー '{key}' が OpenAPI スキーマプロパティに含まれていること。"
        );
    }

    let score_ans = Answer::score(3.5, IndexMap::new(), 0.85);
    let score_val = serde_json::to_value(&score_ans).unwrap();
    for key in score_val.as_object().unwrap().keys() {
        assert!(
            properties.get(key).is_some(),
            "Score JSON のキー '{key}' が OpenAPI スキーマプロパティに含まれていること。"
        );
    }

    let noul_ans = Answer::noul(0.75);
    let noul_val = serde_json::to_value(&noul_ans).unwrap();
    for key in noul_val.as_object().unwrap().keys() {
        assert!(
            properties.get(key).is_some(),
            "Noul JSON のキー '{key}' が OpenAPI スキーマプロパティに含まれていること。"
        );
    }
}

#[test]
fn test_criteria_untagged_schema_compatibility() {
    let spec = generate_openapi_spec();
    let spec_json = serde_json::to_value(&spec).expect("JSON シリアライズに成功すること。");

    let criteria_schema = &spec_json["components"]["schemas"]["Criteria"];
    assert!(
        criteria_schema.is_object(),
        "Criteria スキーマがオブジェクト定義として存在すること。"
    );

    // 空オブジェクトに縮退せず、oneOf としてバリアントが定義されていることを厳密に検証
    let one_of = criteria_schema["oneOf"]
        .as_array()
        .expect("Criteria は oneOf 配列として定義されている必要があります。");
    assert_eq!(
        one_of.len(),
        3,
        "Criteria は Map, List, None の 3 バリアントを持つ必要があります。"
    );

    // 1. Choice Map (object)
    assert!(one_of.iter().any(|s| s["type"] == "object"));
    // 2. Score List (array)
    assert!(one_of.iter().any(|s| s["type"] == "array"));
    // 3. Noul None (null)
    assert!(one_of.iter().any(|s| s["type"] == "null"));

    // Map 形式
    let mut map = IndexMap::new();
    map.insert("k1".to_string(), "説明 1".to_string());
    let c_map = Criteria::Map(map);
    let json_map = serde_json::to_value(&c_map).unwrap();
    assert!(json_map.is_object());

    // List 形式
    let c_list = Criteria::List(vec!["段階1".to_string(), "段階2".to_string()]);
    let json_list = serde_json::to_value(&c_list).unwrap();
    assert!(json_list.is_array());

    // None 形式
    let c_none = Criteria::None;
    let json_none = serde_json::to_value(&c_none).unwrap();
    assert!(json_none.is_null());
}

#[test]
fn test_gating_metadata_schema_and_serialization_parity() {
    let spec = generate_openapi_spec();
    let spec_json = serde_json::to_value(&spec).expect("JSON シリアライズに成功すること。");

    let meta_schema = &spec_json["components"]["schemas"]["GatingMetadata"];
    let properties = &meta_schema["properties"];

    // 必須および新設プロパティの存在確認
    assert!(properties.get("route").is_some());
    assert!(properties.get("confidence").is_some());
    assert!(properties.get("entropy").is_some());
    assert!(properties.get("margin").is_some());
    assert!(properties.get("energy").is_some());
    assert!(properties.get("is_ood").is_some());
    assert!(properties.get("reason").is_some());
    assert!(properties.get("escalation").is_some());

    use sokuto_core::gating::{
        CandidateProbability, DecisionRoute, EscalationContext, GatingMetadata,
    };

    let meta = GatingMetadata {
        route: DecisionRoute::ConfirmOrEscalate,
        confidence: 0.65,
        entropy: Some(0.25),
        margin: Some(0.10),
        energy: Some(-1.8),
        is_ood: false,
        reason: "テスト降格理由".to_string(),
        escalation: Some(EscalationContext {
            top_candidates: vec![
                CandidateProbability {
                    candidate: "opt_a".to_string(),
                    probability: 0.55,
                },
                CandidateProbability {
                    candidate: "opt_b".to_string(),
                    probability: 0.45,
                },
            ],
            margin: Some(0.10),
            uncertainty_reason: "僅差競合".to_string(),
            prompt_template: Some("<context>文脈</context>".to_string()),
        }),
    };

    let json_val = serde_json::to_value(&meta).unwrap();
    assert_eq!(json_val["route"], "confirm_or_escalate");
    assert_eq!(json_val["confidence"], 0.65);
    assert_eq!(json_val["entropy"], 0.25);
    assert_eq!(json_val["margin"], 0.10);
    assert_eq!(json_val["energy"], -1.8);
    assert_eq!(json_val["is_ood"], false);
    assert_eq!(json_val["reason"], "テスト降格理由");

    let esc_val = &json_val["escalation"];
    assert!(esc_val.is_object());
    assert_eq!(esc_val["margin"], 0.10);
    assert_eq!(esc_val["top_candidates"][0]["candidate"], "opt_a");
    assert_eq!(esc_val["top_candidates"][0]["probability"], 0.55);
}

#[test]
fn test_system_routing_summary_schema_parity() {
    let spec = generate_openapi_spec();
    let spec_json = serde_json::to_value(&spec).expect("JSON シリアライズに成功すること。");

    let summary_schema = &spec_json["components"]["schemas"]["SystemRoutingSummary"];
    let properties = &summary_schema["properties"];

    assert!(properties.get("aggregate_route").is_some());
    assert!(properties.get("auto_execute_count").is_some());
    assert!(properties.get("confirm_count").is_some());
    assert!(properties.get("fallback_count").is_some());
    assert!(properties.get("escalation_needed").is_some());

    use sokuto_core::gating::{DecisionRoute, SystemRoutingSummary};

    let summary = SystemRoutingSummary {
        aggregate_route: DecisionRoute::Fallback,
        auto_execute_count: 2,
        confirm_count: 1,
        fallback_count: 1,
        escalation_needed: true,
    };

    let json_val = serde_json::to_value(&summary).unwrap();
    assert_eq!(json_val["aggregate_route"], "fallback");
    assert_eq!(json_val["auto_execute_count"], 2);
    assert_eq!(json_val["confirm_count"], 1);
    assert_eq!(json_val["fallback_count"], 1);
    assert_eq!(json_val["escalation_needed"], true);
}

#[test]
fn test_dag_schema_and_roundtrip_parity() {
    use sokuto_server::schema::dag::{
        DagDefinitionDto, DagNodeDto, DagNodeTypeDto, DagRequest, DagResponse, DagUsageDto,
        NodeConditionDto, NodeConditionOpDto,
    };

    let spec = generate_openapi_spec();
    let spec_json = serde_json::to_value(&spec).expect("JSON シリアライズに成功すること。");

    let dag_req_schema = &spec_json["components"]["schemas"]["DagRequest"];
    assert!(dag_req_schema["properties"].get("dag").is_some());
    assert!(dag_req_schema["properties"].get("state").is_some());

    let dag_res_schema = &spec_json["components"]["schemas"]["DagResponse"];
    assert!(dag_res_schema["properties"].get("status").is_some());
    assert!(dag_res_schema["properties"].get("execution_path").is_some());
    assert!(dag_res_schema["properties"].get("usage").is_some());

    // JSON ラウンドトリップの検証
    let mut nodes = indexmap::IndexMap::new();
    nodes.insert(
        "step1".to_string(),
        DagNodeDto {
            node_type: DagNodeTypeDto::Inference,
            question: Some(Question::new_noul("テスト質問")),
            conditions: vec![NodeConditionDto {
                field: "answer".to_string(),
                op: NodeConditionOpDto::Eq,
                value: Some(serde_json::Value::Bool(true)),
                target_node: "escalate".to_string(),
            }],
            parallel_nodes: Vec::new(),
            join_node: None,
            reason: None,
        },
    );
    nodes.insert(
        "escalate".to_string(),
        DagNodeDto {
            node_type: DagNodeTypeDto::Escalate,
            question: None,
            conditions: Vec::new(),
            parallel_nodes: Vec::new(),
            join_node: None,
            reason: Some("不確実性によるエスカレーション".to_string()),
        },
    );

    let dag_req = DagRequest {
        state: serde_json::json!("テストコンテキスト"),
        dag: DagDefinitionDto {
            dag_id: "test_dag_1".to_string(),
            timeout_ms: 100,
            entry_node: "step1".to_string(),
            nodes,
        },
        timeout_ms: Some(150),
    };

    let serialized = serde_json::to_string(&dag_req).expect("シリアライズ成功すること。");
    let deserialized: DagRequest =
        serde_json::from_str(&serialized).expect("デシリアライズ成功すること。");
    assert_eq!(deserialized.dag.dag_id, "test_dag_1");
    assert_eq!(deserialized.timeout_ms, Some(150));

    let dag_res = DagResponse {
        dag_id: "test_dag_1".to_string(),
        status: "completed".to_string(),
        final_node: Some("step1".to_string()),
        final_decision: None,
        execution_path: vec!["step1".to_string()],
        steps: indexmap::IndexMap::new(),
        escalation: None,
        usage: DagUsageDto {
            total_duration_ms: 12.5,
            steps_executed: 1,
            completion_tokens: 0,
        },
    };

    let serialized_res = serde_json::to_string(&dag_res).expect("シリアライズ成功すること。");
    let deserialized_res: DagResponse =
        serde_json::from_str(&serialized_res).expect("デシリアライズ成功すること。");
    assert_eq!(deserialized_res.status, "completed");
    assert_eq!(deserialized_res.usage.completion_tokens, 0);
}
