//! # 確信度ゲーティング (3系統ルーティング) 統合テストスイート
//!
//! HTTP API (`POST /v1/systemone`) 経由での実効確信度に基づく 3 系統ルーティング
//! (`auto_execute`, `confirm_or_escalate`, `fallback`)、Margin ガード、
//! レスポンスメタデータ、System 2 エスカレーションペイロード、および Prometheus 監視を包括的に検証する。

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use indexmap::IndexMap;
use serde_json::Value;
use sokuto_core::contract::calibration::CalibrationConfig;
use sokuto_core::gating::{DecisionRoute, GatingConfig, GatingThresholds};
use sokuto_core::schema::{Criteria, Question, SystemOneRequest, SystemOneResponse};
use sokuto_runtime::engine::{CoarseToFineConfig, InferenceEngine, SessionConfig};
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

    Some(Arc::new(AppState::with_options(
        engine,
        tokenizer,
        Arc::new(calib_config),
        CoarseToFineConfig::default(),
        128,
        16,
    )))
}

/// 高確信度リクエストによる `auto_execute` ルーティングの検証。
#[tokio::test]
async fn test_gating_auto_execute_routing() {
    let state = match init_test_app_state() {
        Some(s) => s,
        None => return,
    };
    let app = create_router(state, None);

    let mut questions = IndexMap::new();
    let mut criteria = IndexMap::new();
    criteria.insert("refund".to_string(), "返金・注文キャンセル".to_string());
    criteria.insert("inquiry".to_string(), "商品の仕様確認".to_string());
    criteria.insert("praise".to_string(), "サービスへの感謝".to_string());

    questions.insert(
        "intent".to_string(),
        Question {
            question_type: sokuto_core::QuestionType::Choice,
            instructions: "問い合わせの意図を分類せよ。".to_string(),
            criteria: Some(Criteria::Map(criteria)),
        },
    );

    // テスト用ダミーモデルの実効確信度 (約0.0034) に基づき、AutoExecute 領域 (>= 0.002) を指定
    let auto_gating = GatingConfig {
        enabled: true,
        thresholds: GatingThresholds {
            high_threshold: 0.002,
            low_threshold: 0.001,
            top_margin_threshold: 0.0,
            ood_enabled: false,
            energy_threshold: 100.0,
            ..GatingThresholds::default()
        },
    };

    let req_payload = SystemOneRequest::new(
        "注文した商品が届かないため、即座に返金して注文をキャンセルしてください。",
        questions,
    )
    .with_gating(auto_gating);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&req_payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let resp_obj: SystemOneResponse = serde_json::from_slice(&body_bytes).unwrap();

    // 回答レベルのゲーティング検証
    let ans = resp_obj.answers.get("intent").unwrap();
    assert!(ans.gating.is_some());
    let gating = ans.gating.as_ref().unwrap();

    // 確信度とルーティングの検証
    assert_eq!(gating.route, DecisionRoute::AutoExecute);
    assert!(gating.confidence >= 0.002);
    assert!(gating.escalation.is_none());
    assert!(gating.reason.contains("自動実行"));

    // リクエスト全体の集約ルーティング検証
    assert!(resp_obj.routing.is_some());
    let summary = resp_obj.routing.as_ref().unwrap();
    assert_eq!(summary.aggregate_route, DecisionRoute::AutoExecute);
    assert_eq!(summary.auto_execute_count, 1);
    assert_eq!(summary.confirm_count, 0);
    assert_eq!(summary.fallback_count, 0);
    assert!(!summary.escalation_needed);
}

/// 競合・閾値制御による `confirm_or_escalate` と System 2 エスカレーションペイロードの検証。
#[tokio::test]
async fn test_gating_confirm_or_escalate_with_system2_context() {
    let state = match init_test_app_state() {
        Some(s) => s,
        None => return,
    };
    let app = create_router(state, None);

    let mut questions = IndexMap::new();
    let mut criteria = IndexMap::new();
    criteria.insert("general_complaint".to_string(), "苦情全般".to_string());
    criteria.insert("service_feedback".to_string(), "フィードバック".to_string());

    questions.insert(
        "category".to_string(),
        Question {
            question_type: sokuto_core::QuestionType::Choice,
            instructions: "内容のカテゴリを分類せよ。".to_string(),
            criteria: Some(Criteria::Map(criteria)),
        },
    );

    // テストモデルの確信度 (~0.9968) が確認要求境界 [0.00005, 0.998) に入るよう閾値を設定
    let strict_gating = GatingConfig {
        enabled: true,
        thresholds: GatingThresholds {
            high_threshold: 0.998,
            low_threshold: 0.00005,
            top_margin_threshold: 0.90,
            ood_enabled: false,
            energy_threshold: 100.0,
            ..GatingThresholds::default()
        },
    };

    let mut req_payload = SystemOneRequest::new(
        "サービスの対応について少し気になる点があったので連絡しました。",
        questions,
    );
    req_payload = req_payload.with_gating(strict_gating);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&req_payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let resp_obj: SystemOneResponse = serde_json::from_slice(&body_bytes).unwrap();

    let ans = resp_obj.answers.get("category").unwrap();
    let gating = ans.gating.as_ref().unwrap();

    assert_eq!(gating.route, DecisionRoute::ConfirmOrEscalate);
    assert!(gating.escalation.is_some());

    let esc = gating.escalation.as_ref().unwrap();
    assert!(!esc.top_candidates.is_empty());
    assert!(esc.prompt_template.is_some());
    let prompt = esc.prompt_template.as_ref().unwrap();
    assert!(prompt.contains("System 1 の判定では候補"));
    assert!(prompt.contains("Chain-of-Thought"));

    // 集約サマリーの検証
    let summary = resp_obj.routing.as_ref().unwrap();
    assert_eq!(summary.aggregate_route, DecisionRoute::ConfirmOrEscalate);
    assert_eq!(summary.confirm_count, 1);
    assert!(summary.escalation_needed);
}

/// 低確信度時の `fallback` ルーティングの検証。
#[tokio::test]
async fn test_gating_fallback_routing() {
    let state = match init_test_app_state() {
        Some(s) => s,
        None => return,
    };
    let app = create_router(state, None);

    let mut questions = IndexMap::new();
    let mut criteria = IndexMap::new();
    criteria.insert("opt1".to_string(), "選択肢1".to_string());
    criteria.insert("opt2".to_string(), "選択肢2".to_string());

    questions.insert(
        "ambiguous".to_string(),
        Question {
            question_type: sokuto_core::QuestionType::Choice,
            instructions: "曖昧な文脈を分類せよ。".to_string(),
            criteria: Some(Criteria::Map(criteria)),
        },
    );

    // デフォルト閾値 (high=0.85, low=0.50) ではダミーモデルの確信度は確実に Fallback (< 0.50) となる
    let req_payload =
        SystemOneRequest::new("文脈テキスト", questions).with_gating(GatingConfig::default());

    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&req_payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let resp_obj: SystemOneResponse = serde_json::from_slice(&body_bytes).unwrap();

    let ans = resp_obj.answers.get("ambiguous").unwrap();
    let gating = ans.gating.as_ref().unwrap();

    assert_eq!(gating.route, DecisionRoute::Fallback);
    assert!(gating.reason.contains("安全弁フォールバック"));

    let summary = resp_obj.routing.as_ref().unwrap();
    assert_eq!(summary.aggregate_route, DecisionRoute::Fallback);
    assert_eq!(summary.fallback_count, 1);
    assert!(summary.escalation_needed);
}

/// ゲーティング無効化オプトアウト時の検証。
#[tokio::test]
async fn test_gating_disabled_opt_out() {
    let state = match init_test_app_state() {
        Some(s) => s,
        None => return,
    };
    let app = create_router(state, None);

    let mut questions = IndexMap::new();
    questions.insert(
        "is_urgent".to_string(),
        Question::new_noul("緊急案件であるか。"),
    );

    let disabled_gating = GatingConfig {
        enabled: false,
        ..Default::default()
    };

    let req_payload =
        SystemOneRequest::new("通常のお問い合わせです。", questions).with_gating(disabled_gating);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&req_payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json_val: Value = serde_json::from_slice(&body_bytes).unwrap();

    // ゲーティング無効時は routing および gating フィールドが JSON 上に一切出現しないこと
    assert!(json_val.get("routing").is_none());
    let answers = json_val.get("answers").unwrap().as_object().unwrap();
    let ans = answers.get("is_urgent").unwrap();
    assert!(ans.get("gating").is_none());
}

/// ゲーティングメトリクスが Prometheus レコーダーに記録されることを検証。
#[tokio::test]
async fn test_gating_metrics_recorded() {
    let state = match init_test_app_state() {
        Some(s) => s,
        None => return,
    };
    let prometheus_handle = sokuto_server::metrics::setup_metrics_recorder().ok();
    let app = create_router(state, prometheus_handle.clone());

    let mut questions = IndexMap::new();
    questions.insert("q_test".to_string(), Question::new_noul("真偽テスト言明。"));

    let req_payload = SystemOneRequest::new("テストコンテキスト。", questions)
        .with_gating(GatingConfig::default());
    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&req_payload).unwrap()))
        .unwrap();

    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    if let Some(handle) = prometheus_handle {
        let metrics_text = handle.render();
        assert!(metrics_text.contains("sokuto_gating_routes_total"));
    }
}

/// Gating 部分指定 (GatingPatch) 時にベース設定が維持され正常適用されること、
/// および逆転閾値のパッチ指定時に HTTP 400 で拒否されることの検証。
#[tokio::test]
async fn test_gating_partial_patch_application_and_rejection() {
    let state = match init_test_app_state() {
        Some(s) => s,
        None => return,
    };
    let app = create_router(state, None);

    let mut questions = IndexMap::new();
    let mut criteria = IndexMap::new();
    criteria.insert("refund".to_string(), "返金・注文キャンセル".to_string());
    criteria.insert("inquiry".to_string(), "商品の仕様確認".to_string());
    questions.insert(
        "intent".to_string(),
        Question {
            question_type: sokuto_core::QuestionType::Choice,
            instructions: "問い合わせ分類。".to_string(),
            criteria: Some(Criteria::Map(criteria)),
        },
    );

    // 1. enabled のみ true にし、閾値はベース設定を維持する部分パッチ
    let partial_patch_payload = serde_json::json!({
        "state": "返金してください。",
        "questions": questions,
        "gating": {
            "enabled": true
        }
    });

    let req_ok = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&partial_patch_payload).unwrap(),
        ))
        .unwrap();

    let resp_ok = app.clone().oneshot(req_ok).await.unwrap();
    assert_eq!(resp_ok.status(), StatusCode::OK);

    // 2. 逆転閾値 (high < low) の部分パッチ指定時に HTTP 400 で拒否されること
    let invalid_patch_payload = serde_json::json!({
        "state": "返金してください。",
        "questions": questions,
        "gating": {
            "enabled": true,
            "thresholds": {
                "high_threshold": 0.20,
                "low_threshold": 0.80
            }
        }
    });

    let req_err = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&invalid_patch_payload).unwrap(),
        ))
        .unwrap();

    let resp_err = app.oneshot(req_err).await.unwrap();
    assert_eq!(resp_err.status(), StatusCode::BAD_REQUEST);
}

/// 空 State オフロード判定時にガードレールで勝手なタイムスタンプ追記が行われず、
/// オフロード経路(キャッシュミス検証)へ直行することの検証。
#[tokio::test]
async fn test_empty_state_offload_bypass_guardrail() {
    let state = match init_test_app_state() {
        Some(s) => s,
        None => return,
    };
    let app = create_router(state, None);

    let mut questions = IndexMap::new();
    questions.insert("q1".to_string(), Question::new_noul("テスト質問。"));

    // 空文字列 State と X-State-Hash ヘッダーを指定
    let empty_state_payload = serde_json::json!({
        "state": "",
        "questions": questions
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .header("X-State-Hash", "0123456789abcdef")
        .body(Body::from(
            serde_json::to_vec(&empty_state_payload).unwrap(),
        ))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    // dual_engine 未構成またはキャッシュミスのため、400 Bad Request (state_cache_miss) が返却されること
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let json_val: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(
        json_val["error"]["code"], "state_cache_miss",
        "空 State が保持されてキャッシュミスエラーとなること。"
    );
}
