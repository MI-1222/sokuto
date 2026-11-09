//! # DAG 実行および透過的粗密ルーティング統合テスト
//!
//! `POST /v1/systemone/dag` および `POST /v1/systemone` (透過モード) の
//! エンドツーエンド動作、HTTP ステータスコード、およびメトリクス整合性を検証する。

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use indexmap::IndexMap;
use serde_json::Value;
use sokuto_core::schema::{Criteria, Question, QuestionType, SystemOneRequest, SystemOneResponse};
use sokuto_runtime::hierarchical::mapping::HierarchicalMapping;
use sokuto_server::create_router;
use sokuto_server::schema::dag::{
    DagDefinitionDto, DagNodeDto, DagNodeTypeDto, DagRequest, DagResponse, NodeConditionDto,
    NodeConditionOpDto,
};
use tower::ServiceExt;

#[allow(dead_code)]
mod common;

#[tokio::test]
async fn test_dag_endpoint_completed_execution() {
    let state = match common::init_benchmark_app_state(32, 2) {
        Some(s) => s,
        None => {
            eprintln!("モデルファイルが存在しないためスキップします。");
            return;
        }
    };

    let app = create_router(state, None);

    // 2 ステップ (推論 -> 推論) の DAG
    let mut nodes = IndexMap::new();
    nodes.insert(
        "step1".to_string(),
        DagNodeDto {
            node_type: DagNodeTypeDto::Inference,
            question: Some(Question::new_noul("これは電化製品の注文ですか？")),
            conditions: vec![
                NodeConditionDto {
                    field: "answer".to_string(),
                    op: NodeConditionOpDto::Eq,
                    value: Some(Value::Bool(true)),
                    target_node: "step2".to_string(),
                },
                NodeConditionDto {
                    field: "answer".to_string(),
                    op: NodeConditionOpDto::Eq,
                    value: Some(Value::Bool(false)),
                    target_node: "escalate".to_string(),
                },
            ],
            parallel_nodes: Vec::new(),
            join_node: None,
            reason: None,
        },
    );
    nodes.insert(
        "step2".to_string(),
        DagNodeDto {
            node_type: DagNodeTypeDto::Inference,
            question: Some(Question::new_score(
                "緊急度を評価してください。",
                vec!["低".to_string(), "中".to_string(), "高".to_string()],
            )),
            conditions: Vec::new(),
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
            reason: Some("電化製品以外の注文のためエスカレーション".to_string()),
        },
    );

    let dag_req = DagRequest {
        state: serde_json::json!(
            "ノートパソコンのバッテリーが過熱して煙が出ています。早急に交換してください。"
        ),
        dag: DagDefinitionDto {
            dag_id: "defect_triage_dag".to_string(),
            timeout_ms: 500,
            entry_node: "step1".to_string(),
            nodes,
        },
        timeout_ms: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/systemone/dag")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&dag_req).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let dag_res: DagResponse =
        serde_json::from_slice(&body_bytes).expect("有効な DagResponse であること。");

    assert_eq!(dag_res.dag_id, "defect_triage_dag");
    assert_eq!(dag_res.status, "completed");
    assert!(dag_res.execution_path.len() >= 2);
    assert_eq!(dag_res.execution_path[0], "step1");
    assert_eq!(dag_res.execution_path[1], "step2");
    assert!(dag_res.final_decision.is_some());
    assert_eq!(dag_res.usage.completion_tokens, 0);
    assert!(dag_res.usage.total_duration_ms > 0.0);
}

#[tokio::test]
async fn test_dag_endpoint_escalated_execution() {
    let state = match common::init_benchmark_app_state(32, 2) {
        Some(s) => s,
        None => {
            eprintln!("モデルファイルが存在しないためスキップします。");
            return;
        }
    };

    let app = create_router(state, None);

    // 最初からエスカレーションに直行する条件
    let mut nodes = IndexMap::new();
    nodes.insert(
        "entry".to_string(),
        DagNodeDto {
            node_type: DagNodeTypeDto::Inference,
            question: Some(Question::new_noul("これはセール品ですか？")),
            conditions: vec![
                NodeConditionDto {
                    field: "answer".to_string(),
                    op: NodeConditionOpDto::Eq,
                    value: Some(Value::Bool(true)),
                    target_node: "escalate".to_string(),
                },
                NodeConditionDto {
                    field: "answer".to_string(),
                    op: NodeConditionOpDto::Eq,
                    value: Some(Value::Bool(false)),
                    target_node: "escalate".to_string(),
                },
            ],
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
            reason: Some("不確実性またはルールによるエスカレーション。".to_string()),
        },
    );

    let dag_req = DagRequest {
        state: serde_json::json!("昨日購入した商品について質問です。"),
        dag: DagDefinitionDto {
            dag_id: "escalation_test_dag".to_string(),
            timeout_ms: 500,
            entry_node: "entry".to_string(),
            nodes,
        },
        timeout_ms: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/systemone/dag")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&dag_req).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // エスカレーション時も正常終了 (HTTP 200 OK)
    assert_eq!(response.status(), StatusCode::OK);

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let dag_res: DagResponse =
        serde_json::from_slice(&body_bytes).expect("有効な DagResponse であること。");

    assert_eq!(dag_res.status, "escalated");
    assert_eq!(dag_res.final_node, Some("escalate".to_string()));
    assert!(dag_res.escalation.is_some());
    assert_eq!(dag_res.usage.completion_tokens, 0);
}

#[tokio::test]
async fn test_dag_endpoint_validation_error_bad_request() {
    let state = match common::init_benchmark_app_state(32, 2) {
        Some(s) => s,
        None => {
            eprintln!("モデルファイルが存在しないためスキップします。");
            return;
        }
    };

    let app = create_router(state, None);

    // 未定義の target_node を参照する不正な DAG
    let mut nodes = IndexMap::new();
    nodes.insert(
        "entry".to_string(),
        DagNodeDto {
            node_type: DagNodeTypeDto::Inference,
            question: Some(Question::new_noul("質問")),
            conditions: vec![NodeConditionDto {
                field: "answer".to_string(),
                op: NodeConditionOpDto::Eq,
                value: Some(Value::Bool(true)),
                target_node: "undefined_node".to_string(),
            }],
            parallel_nodes: Vec::new(),
            join_node: None,
            reason: None,
        },
    );

    let dag_req = DagRequest {
        state: serde_json::json!("テスト"),
        dag: DagDefinitionDto {
            dag_id: "invalid_dag".to_string(),
            timeout_ms: 100,
            entry_node: "entry".to_string(),
            nodes,
        },
        timeout_ms: None,
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/systemone/dag")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&dag_req).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let err_json: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(err_json["error"]["code"], "dag_validation_error");
}

#[tokio::test]
async fn test_dag_endpoint_timeout_gateway_timeout() {
    let state = match common::init_benchmark_app_state(32, 2) {
        Some(s) => s,
        None => {
            eprintln!("モデルファイルが存在しないためスキップします。");
            return;
        }
    };

    let app = create_router(state, None);

    let mut nodes = IndexMap::new();
    nodes.insert(
        "entry".to_string(),
        DagNodeDto {
            node_type: DagNodeTypeDto::Inference,
            question: Some(Question::new_noul("タイムアウトテスト質問")),
            conditions: Vec::new(),
            parallel_nodes: Vec::new(),
            join_node: None,
            reason: None,
        },
    );

    let dag_req = DagRequest {
        state: serde_json::json!("長いコンテキスト文章。".repeat(20)),
        dag: DagDefinitionDto {
            dag_id: "timeout_dag".to_string(),
            timeout_ms: 0, // 0ms のため即座にタイムアウト
            entry_node: "entry".to_string(),
            nodes,
        },
        timeout_ms: Some(0),
    };

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/systemone/dag")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&dag_req).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let err_json: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(err_json["error"]["code"], "dag_timeout");
}

#[tokio::test]
async fn test_systemone_transparent_hierarchical_routing_integration() {
    let state = match common::init_benchmark_app_state(32, 2) {
        Some(s) => s,
        None => {
            eprintln!("モデルファイルが存在しないためスキップします。");
            return;
        }
    };

    // 粗密階層マッピングを構築
    // 大分類 2個、細分類 各2個 (計4個)
    let mut builder = HierarchicalMapping::builder("test_hierarchical_routing");
    builder.add_category(
        "billing",
        "請求・支払い・返金",
        vec![
            ("refund", "返金申請の手続き"),
            ("invoice", "請求書・領収書の発行"),
        ],
    );
    builder.add_category(
        "technical",
        "システム不具合・技術トラブル",
        vec![
            ("login_issue", "ログインできない・認証エラー"),
            ("bug_report", "画面表示崩れやシステム不具合の報告"),
        ],
    );
    let mapping = builder.build().expect("マッピング構築成功すること。");

    let state = std::sync::Arc::new(
        (*state)
            .clone()
            .with_hierarchical_mapping(std::sync::Arc::new(mapping)),
    );
    let app = create_router(state, None);

    // auto_hierarchical: true を指定して透過的粗密推論を発動
    let mut criteria_map = IndexMap::new();
    criteria_map.insert("refund".to_string(), "返金申請の手続き".to_string());
    criteria_map.insert("invoice".to_string(), "請求書・領収書の発行".to_string());
    criteria_map.insert(
        "login_issue".to_string(),
        "ログインできない・認証エラー".to_string(),
    );
    criteria_map.insert(
        "bug_report".to_string(),
        "画面表示崩れやシステム不具合の報告".to_string(),
    );

    let mut questions = IndexMap::new();
    questions.insert(
        "q_topic".to_string(),
        Question {
            question_type: QuestionType::Choice,
            instructions: "お問い合わせ内容のカテゴリを選択してください。".to_string(),
            criteria: Some(Criteria::Map(criteria_map)),
        },
    );

    let mut req =
        SystemOneRequest::new("二重請求が発生しているため返金をお願いします。", questions);
    req.auto_hierarchical = Some(true);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/systemone")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&req).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let sys1_res: SystemOneResponse =
        serde_json::from_slice(&body_bytes).expect("有効な SystemOneResponse であること。");

    let answer = sys1_res
        .answers
        .get("q_topic")
        .expect("回答が存在すること。");
    assert!(answer.is_choice());
    let probs = answer.probabilities().expect("確率分布が存在すること。");
    assert_eq!(probs.len(), 4);
    // 確率の合計値がほぼ 1.0 であること
    let sum_prob: f64 = probs.values().sum();
    assert!((sum_prob - 1.0).abs() < 1e-4);
    assert_eq!(sys1_res.usage.completion_tokens, 0);
}

#[tokio::test]
async fn test_systemone_transparent_hierarchical_routing_fallback_on_unrelated_choices() {
    let state = match common::init_benchmark_app_state(32, 2) {
        Some(s) => s,
        None => {
            eprintln!("モデルファイルが存在しないためスキップします。");
            return;
        }
    };

    // 粗密階層マッピングを構築 (billing / technical のみ)
    let mut builder = HierarchicalMapping::builder("test_hierarchical_routing");
    builder.add_category(
        "billing",
        "請求・支払い・返金",
        vec![("refund", "返金申請の手続き")],
    );
    let mapping = builder.build().expect("マッピング構築成功すること。");

    let state = std::sync::Arc::new(
        (*state)
            .clone()
            .with_hierarchical_mapping(std::sync::Arc::new(mapping)),
    );
    let app = create_router(state, None);

    // auto_hierarchical: true を指定するが、質問はマッピングとは全く無関係な低基数の業務質問
    let mut criteria_map = IndexMap::new();
    criteria_map.insert("priority_low".to_string(), "低優先度".to_string());
    criteria_map.insert("priority_high".to_string(), "高優先度".to_string());

    let mut questions = IndexMap::new();
    questions.insert(
        "q_priority".to_string(),
        Question {
            question_type: QuestionType::Choice,
            instructions: "優先度を選択してください。".to_string(),
            criteria: Some(Criteria::Map(criteria_map)),
        },
    );

    let mut req = SystemOneRequest::new("急ぎの要件です。", questions);
    req.auto_hierarchical = Some(true);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/systemone")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&req).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    // マッピング未適合のため通常推論へ安全フォールバックし 200 OK
    assert_eq!(response.status(), StatusCode::OK);

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let sys1_res: SystemOneResponse =
        serde_json::from_slice(&body_bytes).expect("有効な SystemOneResponse であること。");

    let answer = sys1_res
        .answers
        .get("q_priority")
        .expect("回答が存在すること。");
    assert!(answer.is_choice());
    assert_eq!(sys1_res.usage.completion_tokens, 0);
}

#[tokio::test]
async fn test_dag_endpoint_timeout_clamped_to_max() {
    use sokuto_server::handlers::MAX_DAG_TIMEOUT_MS;

    let state = match common::init_benchmark_app_state(32, 2) {
        Some(s) => s,
        None => {
            eprintln!("モデルファイルが存在しないためスキップします。");
            return;
        }
    };

    let app = create_router(state, None);

    let mut nodes = IndexMap::new();
    nodes.insert(
        "step1".to_string(),
        DagNodeDto {
            node_type: DagNodeTypeDto::Inference,
            question: Some(Question::new_noul("クランプ検証質問")),
            conditions: Vec::new(),
            parallel_nodes: Vec::new(),
            join_node: None,
            reason: None,
        },
    );

    // 悪意のある過大なタイムアウト (例: 100秒) を指定
    let dag_req = DagRequest {
        state: serde_json::json!("テストコンテキスト。"),
        dag: DagDefinitionDto {
            dag_id: "clamp_test_dag".to_string(),
            timeout_ms: 100_000,
            entry_node: "step1".to_string(),
            nodes,
        },
        timeout_ms: Some(100_000),
    };

    assert_eq!(MAX_DAG_TIMEOUT_MS, 5000);

    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/systemone/dag")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&dag_req).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let dag_res: DagResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(dag_res.status, "completed");
}
