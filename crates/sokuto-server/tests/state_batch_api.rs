//! # 一括バッチ質問および透過キャッシュ統合テストスイート
//!
//! `POST /v1/systemone` における `X-State-Hash` ヘッダーおよび透過キャッシュプロトコル、
//! `POST /v1/systemone/state-batch` における単一 State に対する複数異種質問の一括評価、
//! ならびに State-Offloading のキャッシュミス時バリデーションを検証する。

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use indexmap::IndexMap;
use serde_json::Value;
use sokuto_core::contract::calibration::CalibrationConfig;
use sokuto_core::schema::{
    Question, StateBatchRequest, StateBatchResponse, SystemOneRequest, SystemOneResponse,
};
use sokuto_runtime::engine::{
    CacheCompressionMode, CoarseToFineConfig, InferenceEngine, SessionConfig, StateCacheManager,
};
use sokuto_runtime::tokenizer::JevTokenizer;
use sokuto_server::create_router;
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
fn init_test_app_state() -> Option<Arc<AppState>> {
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

    let cache_manager = Arc::new(StateCacheManager::new(
        64 * 1024 * 1024,
        600,
        CacheCompressionMode::Fp16,
    ));

    let mut state = AppState::with_options(
        engine,
        tokenizer,
        Arc::new(calib_config),
        CoarseToFineConfig::default(),
        128,
        16,
    );
    state.cache_manager = Some(cache_manager);

    Some(Arc::new(state))
}

#[tokio::test]
async fn test_systemone_transparent_cache_headers() {
    let state = match init_test_app_state() {
        Some(s) => s,
        None => {
            eprintln!("スキップ: モデルまたはトークナイザーが存在しません。");
            return;
        }
    };

    let app = create_router(state, None);

    let mut questions = IndexMap::new();
    questions.insert(
        "q1".to_string(),
        Question::new_choice(
            "この文章の感情を判定してください。",
            [
                ("positive".to_string(), "肯定的".to_string()),
                ("negative".to_string(), "否定的".to_string()),
            ]
            .into_iter()
            .collect(),
        ),
    );

    let req_payload = SystemOneRequest::new("とても素晴らしい商品でした！", questions);
    let req_json = serde_json::to_vec(&req_payload).unwrap();

    let request = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("Content-Type", "application/json")
        .body(Body::from(req_json))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // 透過キャッシュヘッダーの検証
    let headers = response.headers();
    assert!(headers.contains_key("X-State-Hash"));
    assert!(headers.contains_key("X-Cache-Status"));

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let res: SystemOneResponse = serde_json::from_slice(&body_bytes).unwrap();
    assert!(res.answers.contains_key("q1"));
}

#[tokio::test]
async fn test_state_offloading_miss_returns_bad_request() {
    let state = match init_test_app_state() {
        Some(s) => s,
        None => {
            eprintln!("スキップ: モデルまたはトークナイザーが存在しません。");
            return;
        }
    };

    let app = create_router(state, None);

    let mut questions = IndexMap::new();
    questions.insert(
        "q1".to_string(),
        Question::new_choice(
            "質問",
            [
                ("a".to_string(), "A".to_string()),
                ("b".to_string(), "B".to_string()),
            ]
            .into_iter()
            .collect(),
        ),
    );

    // State を空文字にし、未登録のハッシュのみ指定
    let req_payload = SystemOneRequest::new("", questions).with_state_hash("0123456789abcdef");
    let req_json = serde_json::to_vec(&req_payload).unwrap();

    let request = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("Content-Type", "application/json")
        .header("X-State-Hash", "0123456789abcdef")
        .body(Body::from(req_json))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body_json: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(body_json["error"]["code"], "state_cache_miss");
}

#[tokio::test]
async fn test_state_batch_endpoint() {
    let state = match init_test_app_state() {
        Some(s) => s,
        None => {
            eprintln!("スキップ: モデルまたはトークナイザーが存在しません。");
            return;
        }
    };

    let app = create_router(state, None);

    let mut questions = IndexMap::new();
    questions.insert(
        "sentiment".to_string(),
        Question::new_choice(
            "感情を判定してください。",
            [
                ("pos".to_string(), "肯定的".to_string()),
                ("neg".to_string(), "否定的".to_string()),
            ]
            .into_iter()
            .collect(),
        ),
    );
    questions.insert(
        "factual".to_string(),
        Question::new_noul("事実に反する記述が含まれますか？"),
    );

    let batch_req = StateBatchRequest::new(
        "本製品は大変品質が高く満足しています。配送も迅速でした。",
        questions,
    );
    let req_json = serde_json::to_vec(&batch_req).unwrap();

    let request = Request::builder()
        .method("POST")
        .uri("/v1/systemone/state-batch")
        .header("Content-Type", "application/json")
        .body(Body::from(req_json))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let headers = response.headers();
    assert!(headers.contains_key("X-State-Hash"));
    assert!(headers.contains_key("X-Cache-Status"));

    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let res: StateBatchResponse = serde_json::from_slice(&body_bytes).unwrap();

    assert!(res.answers.contains_key("sentiment"));
    assert!(res.answers.contains_key("factual"));
    assert!(res.usage.prompt_tokens > 0);
}
