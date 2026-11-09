//! # Axum HTTP API サーバー統合テストスイート
//!
//! `tower::ServiceExt::oneshot` によるソケット非依存のインメモリ高速統合テスト。
//! ヘルスチェック、Readiness、Prometheus メトリクス、Swagger UI、
//! および `POST /v1/systemone` の推論正常系・異常系バリデーションを網羅的に検証する。

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use indexmap::IndexMap;
use serde_json::Value;
use sokuto_core::contract::calibration::CalibrationConfig;
use sokuto_core::schema::{Criteria, Question, SystemOneRequest, SystemOneResponse};
use sokuto_runtime::engine::{CoarseToFineConfig, InferenceEngine, SessionConfig};
use sokuto_runtime::tokenizer::JevTokenizer;
use sokuto_server::create_router;
use sokuto_server::error::ErrorResponse;
use sokuto_server::metrics::setup_metrics_recorder;
use sokuto_server::state::AppState;
use tower::ServiceExt;

/// ワークスペースのルートディレクトリを取得する。
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ワークスペースルートの解決に失敗しました。")
        .to_path_buf()
}

/// 配布モデルディレクトリを取得する。
fn default_model_dir() -> PathBuf {
    workspace_root().join("models").join("default")
}

/// テスト用 `AppState` の初期化ヘルパー。
fn init_test_app_state(max_questions: usize) -> Option<Arc<AppState>> {
    let model_dir = default_model_dir();
    let onnx_path = model_dir.join("model.onnx");
    let tok_path = model_dir.join("tokenizer.json");
    let calib_path = model_dir.join("calibration.json");

    if !onnx_path.exists() || !tok_path.exists() {
        return None;
    }

    let session_config = SessionConfig {
        pool_size: 1,
        ..Default::default()
    };

    let engine = Arc::new(InferenceEngine::new(&onnx_path, session_config).ok()?);
    let tokenizer = Arc::new(JevTokenizer::from_file(&tok_path).ok()?);
    let calib_config = if calib_path.exists() {
        std::fs::read_to_string(&calib_path)
            .ok()
            .and_then(|s| CalibrationConfig::from_json_str(&s).ok())
            .unwrap_or_default()
    } else {
        CalibrationConfig::default()
    };

    let state = AppState::with_options(
        engine,
        tokenizer,
        Arc::new(calib_config),
        CoarseToFineConfig::default(),
        max_questions,
        16,
    );

    Some(Arc::new(state))
}

#[tokio::test]
async fn test_health_and_readiness_endpoints() {
    let state = match init_test_app_state(128) {
        Some(s) => s,
        None => return,
    };

    let app = create_router(state, None);

    // 1. GET /health (Liveness)
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/health")
                .method("GET")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(json["status"], "ok");

    // 2. GET /ready (Readiness)
    let res = app
        .oneshot(
            Request::builder()
                .uri("/ready")
                .method("GET")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let json: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(json["status"], "ready");
}

#[tokio::test]
async fn test_metrics_and_openapi_endpoints() {
    let state = match init_test_app_state(128) {
        Some(s) => s,
        None => return,
    };

    let prometheus_handle = setup_metrics_recorder().ok();
    let app = create_router(state, prometheus_handle);

    // 1. GET /metrics
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .method("GET")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);

    // 2. GET /api-docs/openapi.json
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api-docs/openapi.json")
                .method("GET")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);
    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let spec: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert!(spec["paths"]["/v1/systemone"].is_object());

    // 3. GET /swagger-ui/
    #[cfg(feature = "swagger")]
    {
        let res = app
            .oneshot(
                Request::builder()
                    .uri("/swagger-ui/")
                    .method("GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(res.status(), StatusCode::OK);
    }
}

#[tokio::test]
async fn test_systemone_inference_mixed_primitives() {
    let state = match init_test_app_state(128) {
        Some(s) => s,
        None => return,
    };

    let app = create_router(state, None);

    // 質問群の構築 (Choice, Score, Noul 混載)
    let mut questions = IndexMap::new();

    let mut choice_criteria = IndexMap::new();
    choice_criteria.insert(
        "banking".to_string(),
        "銀行振込・口座関連の問い合わせ".to_string(),
    );
    choice_criteria.insert(
        "card_issue".to_string(),
        "クレジットカード発行・紛失".to_string(),
    );
    choice_criteria.insert(
        "other".to_string(),
        "その他の一般的な問い合わせ".to_string(),
    );

    questions.insert(
        "q_category".to_string(),
        Question::new_choice("問い合わせカテゴリを選択してください。", choice_criteria),
    );

    questions.insert(
        "q_urgency".to_string(),
        Question::new_score(
            "対応の緊急度を1〜5で評価してください。",
            vec![
                "極めて低い".to_string(),
                "低い".to_string(),
                "普通".to_string(),
                "高い".to_string(),
                "至急対応が必要".to_string(),
            ],
        ),
    );

    questions.insert(
        "q_fraud".to_string(),
        Question::new_noul("この問い合わせには不正利用の疑いがある。"),
    );

    let request_payload = SystemOneRequest::new(
        "急ぎでクレジットカードの利用を停止してください！財布を紛失してしまいました。",
        questions,
    );

    let req_body = serde_json::to_vec(&request_payload).unwrap();

    let res = app
        .oneshot(
            Request::builder()
                .uri("/v1/systemone")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(req_body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::OK);

    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let response: SystemOneResponse = serde_json::from_slice(&body_bytes)
        .expect("SystemOneResponse のデシリアライズに成功すること。");

    // 全質問の結果が返却されていること
    assert_eq!(response.answers.len(), 3);

    // 1. Choice 質問の検証
    let ans_choice = response.answers.get("q_category").unwrap();
    assert!(ans_choice.is_choice());
    assert!(ans_choice.probabilities().is_some());
    assert!(ans_choice.confidence().is_some());
    let probs = ans_choice.probabilities().unwrap();
    assert_eq!(probs.len(), 3);

    // 2. Score 質問の検証
    let ans_score = response.answers.get("q_urgency").unwrap();
    assert!(ans_score.is_score());
    let score_val = ans_score.as_score().unwrap();
    assert!((1.0..=5.0).contains(&score_val));
    assert!(ans_score.confidence().is_some());

    // 3. Noul 質問の検証
    let ans_noul = response.answers.get("q_fraud").unwrap();
    assert!(ans_noul.is_noul());
    let noul_prob = ans_noul.as_noul().unwrap();
    assert!((0.0..=1.0).contains(&noul_prob));

    // 4. トークン使用量の検証
    assert!(response.usage.prompt_tokens > 0);
    assert_eq!(response.usage.completion_tokens, 0);
    assert_eq!(response.usage.total_tokens, response.usage.prompt_tokens);
}

#[tokio::test]
async fn test_systemone_validation_errors() {
    let state = match init_test_app_state(128) {
        Some(s) => s,
        None => return,
    };

    let app = create_router(state, None);

    // 1. 空の questions リクエスト (400 Bad Request)
    let empty_payload = SystemOneRequest::new("テスト", IndexMap::new());
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/systemone")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&empty_payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let err_resp: ErrorResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(err_resp.error.error_type, "invalid_request_error");
    assert_eq!(err_resp.error.code, "empty_questions");

    // 2. Choice 型で候補数が 0 (400 Bad Request)
    let mut questions = IndexMap::new();
    questions.insert(
        "q_bad".to_string(),
        Question {
            question_type: sokuto_core::schema::QuestionType::Choice,
            instructions: "指示文".to_string(),
            criteria: Some(Criteria::Map(IndexMap::new())),
        },
    );

    let bad_choice_payload = SystemOneRequest::new("テスト", questions);
    let res = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/systemone")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&bad_choice_payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let err_resp: ErrorResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(err_resp.error.code, "invalid_choice_count");
}

#[tokio::test]
async fn test_systemone_max_questions_payload_too_large() {
    // 最大質問数を 2 に制限した AppState
    let state = match init_test_app_state(2) {
        Some(s) => s,
        None => return,
    };

    let app = create_router(state, None);

    // 3 個の質問を含むリクエスト
    let mut questions = IndexMap::new();
    questions.insert("q1".to_string(), Question::new_noul("質問 1"));
    questions.insert("q2".to_string(), Question::new_noul("質問 2"));
    questions.insert("q3".to_string(), Question::new_noul("質問 3"));

    let payload = SystemOneRequest::new("テスト", questions);
    let res = app
        .oneshot(
            Request::builder()
                .uri("/v1/systemone")
                .method("POST")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(res.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let body_bytes = res.into_body().collect().await.unwrap().to_bytes();
    let err_resp: ErrorResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(err_resp.error.code, "payload_too_large");
}
