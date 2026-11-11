//! # Python-Rust E2E 数値・決定・契約パリティ結合テストスイート
//!
//! ロードマップ 3.4「E2E パリティテストおよび API 結合検証」に基づき、
//! Python 参照パイプライン (`train/`) が算出したゴールデンフィクスチャ (`e2e_parity_fixtures.json`) と、
//! Rust 本番 HTTP API サーバー (`POST /v1/systemone`) の推論結果をインプロセスで突き合わせ、
//! 以下の全項目を機械的に検証する。
//!
//! 1. **決定一致性**: Choice の採択キー、Score の期待値、Noul の真実確率値が完全一致すること。
//! 2. **数値パリティ**: 各候補の較正後確率分布および実効確信度 $S_{\text{confidence}}$ が $\text{Atol} \le 10^{-4}$ で一致すること。
//! 3. **3系統ゲーティング**: `AutoExecute`, `ConfirmOrEscalate`, `Fallback` のルーティング判定が完全一致すること。
//! 4. **エスカレーション契約 & Zero-Allocation**: `AutoExecute` 時のエスカレーション構造体非生成 (`None`)、
//!    および非 AutoExecute 時の `<context>` カプセル化付き CoT プロンプト生成を検証すること。
//! 5. **最悪値集約ルール**: 複数質問リクエストにおける `aggregate_route` の保守的論理結合を検証すること。

use std::path::PathBuf;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::Value;
use sokuto_core::contract::calibration::CalibrationConfig;
use sokuto_core::schema::SystemOneResponse;
use sokuto_runtime::engine::{CoarseToFineConfig, InferenceEngine, SessionConfig};
use sokuto_runtime::tokenizer::JevTokenizer;
use sokuto_server::create_router;
use sokuto_server::state::AppState;
use tower::ServiceExt;

/// 許容最大絶対誤差 (Atol)。
const PARITY_ATOL: f64 = 1.0e-4;

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

/// ゴールデンフィクスチャ JSON ファイルのパスを取得する。
fn fixtures_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("e2e_parity_fixtures.json")
}

/// テスト用 `AppState` の初期化ヘルパー。
///
/// モデルファイルが存在しない場合は `None` を返却し、テストを安全にスキップする。
fn init_test_app_state() -> Option<Arc<AppState>> {
    let model_dir = default_model_dir();
    let onnx_path = model_dir.join("model.onnx");
    let tok_path = model_dir.join("tokenizer.json");
    let calib_path = model_dir.join("calibration.json");

    if !onnx_path.exists() || !tok_path.exists() {
        eprintln!(
            "モデルファイルが存在しないためスキップします: ONNX={:?}, Tok={:?}",
            onnx_path, tok_path
        );
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

    let mut guardrail_config = sokuto_server::guardrails::GuardrailConfig::default();
    guardrail_config.temporal.attach_reference_time = false;
    guardrail_config.temporal.enable_inline_normalization = false;

    let state = AppState::with_guardrails(
        engine,
        tokenizer,
        Arc::new(calib_config.clone()),
        CoarseToFineConfig::default(),
        128,
        16,
        guardrail_config,
    )
    .with_gating_config(calib_config.gating_config());

    Some(Arc::new(state))
}

/// ゴールデンフィクスチャ JSON を読み込みパースする。
fn load_fixtures() -> Option<Value> {
    let path = fixtures_path();
    if !path.exists() {
        eprintln!("フィクスチャファイルが存在しません: {:?}", path);
        return None;
    }
    let content = std::fs::read_to_string(&path).expect("フィクスチャの読み込みに失敗しました。");
    let val: Value =
        serde_json::from_str(&content).expect("フィクスチャ JSON のパースに失敗しました。");
    Some(val)
}

/// 全ゴールデンテストケースに対する Python 参照出力と Rust API レスポンスの完全パリティ検証。
#[tokio::test]
async fn test_e2e_parity_all_golden_cases() {
    let state = match init_test_app_state() {
        Some(s) => s,
        None => return,
    };
    let fixtures = match load_fixtures() {
        Some(f) => f,
        None => return,
    };

    let cases = fixtures["cases"]
        .as_array()
        .expect("cases 配列が存在すること。");
    assert!(!cases.is_empty(), "フィクスチャケースが 0 件です。");

    let app = create_router(state, None);

    for case in cases {
        let case_id = case["id"].as_str().unwrap_or("unknown");
        let state_val = &case["state"];
        let questions_val = &case["questions"];
        let gating_cfg_val = &case["gating_config"];
        let expected = &case["expected"];

        // リクエストペイロードの構築
        let mut req_body_map = serde_json::Map::new();
        req_body_map.insert("state".to_string(), state_val.clone());
        req_body_map.insert("questions".to_string(), questions_val.clone());
        if !gating_cfg_val.is_null() {
            req_body_map.insert("gating".to_string(), gating_cfg_val.clone());
        }
        let req_json = Value::Object(req_body_map);

        // HTTP POST /v1/systemone のインプロセス実行
        let req = Request::builder()
            .method("POST")
            .uri("/v1/systemone")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&req_json).unwrap()))
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "ケース '{case_id}' で 200 OK 以外のステータスコードが返却されました。"
        );

        let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let actual_resp: SystemOneResponse =
            serde_json::from_slice(&body_bytes).unwrap_or_else(|_| {
                panic!("ケース '{case_id}' のレスポンスデシリアライズに失敗しました。")
            });

        // 各質問の期待値検証
        let expected_answers = &expected["answers"];
        for (q_id, expected_ans) in expected_answers.as_object().unwrap() {
            let actual_ans = actual_resp.answers.get(q_id).unwrap_or_else(|| {
                panic!("ケース '{case_id}': 回答 '{q_id}' が存在しません。");
            });

            // 1. Choice 型の採択キー検証
            if let Some(expected_choice) = expected_ans["choice"].as_str() {
                assert_eq!(
                    actual_ans.as_choice(),
                    Some(expected_choice),
                    "ケース '{case_id}' ({q_id}): Choice 採択キーが不一致です。"
                );
            }

            // 2. Score 型の加重平均値検証
            if let Some(expected_score) = expected_ans["score"].as_f64() {
                let actual_score = actual_ans.as_score().unwrap_or_else(|| {
                    panic!("ケース '{case_id}' ({q_id}): score が None です。");
                });
                let diff = (actual_score - expected_score).abs();
                assert!(
                    diff <= PARITY_ATOL,
                    "ケース '{case_id}' ({q_id}): Score 値が許容誤差を超加しています (実測={}, 期待={}, 差分={diff})。",
                    actual_score,
                    expected_score
                );
            }

            // 3. Noul 型の真実確率値検証
            if let Some(expected_noul) = expected_ans["noul"].as_f64() {
                let actual_noul = actual_ans
                    .as_noul()
                    .unwrap_or_else(|| panic!("ケース '{case_id}' ({q_id}): noul が None です。"));
                let diff = (actual_noul - expected_noul).abs();
                assert!(
                    diff <= PARITY_ATOL,
                    "ケース '{case_id}' ({q_id}): Noul 値が許容誤差を超加しています (実測={}, 期待={}, 差分={diff})。",
                    actual_noul,
                    expected_noul
                );
            }

            // 4. 各候補の確率分布検証
            if let Some(expected_probs) = expected_ans["probabilities"].as_object() {
                let actual_probs = actual_ans.probabilities().unwrap_or_else(|| {
                    panic!("ケース '{case_id}' ({q_id}): probabilities が None です。");
                });
                for (opt_key, expected_prob_val) in expected_probs {
                    let exp_p = expected_prob_val.as_f64().unwrap();
                    let act_p = *actual_probs.get(opt_key).unwrap_or_else(|| {
                        panic!(
                            "ケース '{case_id}' ({q_id}): 候補キー '{opt_key}' が見つかりません。"
                        );
                    });
                    let diff = (act_p - exp_p).abs();
                    assert!(
                        diff <= PARITY_ATOL,
                        "ケース '{case_id}' ({q_id}): 候補 '{opt_key}' の確率値誤差が超過しています (実測={act_p}, 期待={exp_p}, 差分={diff})。",
                    );
                }
            }

            // 5. 確信度スコアの検証
            if let Some(expected_conf) = expected_ans["confidence"].as_f64() {
                let actual_conf = actual_ans.confidence().unwrap_or_else(|| {
                    panic!("ケース '{case_id}' ({q_id}): confidence が None です。");
                });
                let diff = (actual_conf - expected_conf).abs();
                assert!(
                    diff <= PARITY_ATOL,
                    "ケース '{case_id}' ({q_id}): 確信度スコア誤差が超過しています (実測={actual_conf}, 期待={expected_conf}, 差分={diff})。",
                );
            }

            // 6. Gating ルーティング検証
            let expected_gating = &expected_ans["gating"];
            let expected_route = expected_gating["route"].as_str().unwrap();
            let actual_gating = actual_ans.gating.as_ref().unwrap_or_else(|| {
                panic!("ケース '{case_id}' ({q_id}): gating メタデータが None です。");
            });
            assert_eq!(
                actual_gating.route.as_str(),
                expected_route,
                "ケース '{case_id}' ({q_id}): Gating ルーティング種別が不一致です。"
            );

            // 7. Zero-Allocation & エスカレーション契約検証
            if expected_route == "auto_execute" {
                assert!(
                    actual_gating.escalation.is_none(),
                    "ケース '{case_id}' ({q_id}): auto_execute 時は escalation が None である必要があります。"
                );
            } else {
                let esc = actual_gating.escalation.as_ref().unwrap_or_else(|| {
                    panic!(
                        "ケース '{case_id}' ({q_id}): {expected_route} 時は escalation が必須です。"
                    );
                });
                let prompt = esc.prompt_template.as_ref().unwrap_or_else(|| {
                    panic!("ケース '{case_id}' ({q_id}): prompt_template が生成されていません。");
                });
                assert!(
                    prompt.contains("<context>"),
                    "ケース '{case_id}' ({q_id}): エスカレーションプロンプトに <context> タグが含まれていません。"
                );
                assert!(
                    prompt.contains("</context>"),
                    "ケース '{case_id}' ({q_id}): エスカレーションプロンプトに </context> タグが含まれていません。"
                );
            }
        }

        // 8. リクエスト全体の集約ルーティングサマリー検証
        let expected_agg_route = expected["routing"]["aggregate_route"].as_str().unwrap();
        let actual_routing = actual_resp.routing.as_ref().unwrap_or_else(|| {
            panic!("ケース '{case_id}': routing サマリーが存在しません。");
        });
        assert_eq!(
            actual_routing.aggregate_route.as_str(),
            expected_agg_route,
            "ケース '{case_id}': 集約ルーティング種別が不一致です。"
        );
    }
}

/// 実務再現シナリオ A (EC・ハードウェアトラブル: Choice + Score + Noul 混在) の詳細結合検証。
#[tokio::test]
async fn test_e2e_scenario_ec_hardware_defect_details() {
    let state = match init_test_app_state() {
        Some(s) => s,
        None => return,
    };
    let fixtures = match load_fixtures() {
        Some(f) => f,
        None => return,
    };

    let cases = fixtures["cases"].as_array().unwrap();
    let ec_case = cases
        .iter()
        .find(|c| c["id"] == "scenario_ec_hardware_defect")
        .expect("scenario_ec_hardware_defect ケースが存在すること。");

    let app = create_router(state, None);

    let mut req_body_map = serde_json::Map::new();
    req_body_map.insert("state".to_string(), ec_case["state"].clone());
    req_body_map.insert("questions".to_string(), ec_case["questions"].clone());

    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&req_body_map).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let actual: SystemOneResponse = serde_json::from_slice(&body_bytes).unwrap();

    // 部署判定: tech_support が採択されること
    let dept = actual.answers.get("department").unwrap();
    assert_eq!(dept.as_choice(), Some("tech_support"));
    let dept_probs = dept.probabilities().unwrap();
    assert!(dept_probs["tech_support"] > 0.50);

    // 緊急度判定: Score が 1.0〜5.0 の範囲内であること
    let urgency = actual.answers.get("urgency_score").unwrap();
    let score_val = urgency.as_score().unwrap();
    assert!((1.0..=5.0).contains(&score_val));

    // ハードウェア初期不良判定: Noul 真実確率が [0.0, 1.0] に収まり、実効確信度が高水準であること
    let defect = actual.answers.get("is_hardware_defect").unwrap();
    let p_defect = defect.as_noul().unwrap();
    assert!((0.0..=1.0).contains(&p_defect));
    let eff_conf = defect.effective_confidence();
    assert!(eff_conf > 0.80);

    // Usage トークン数が 0 より大きいこと、completion_tokens が 0 であること
    assert!(actual.usage.prompt_tokens > 0);
    assert_eq!(actual.usage.completion_tokens, 0);
    assert_eq!(actual.usage.total_tokens, actual.usage.prompt_tokens);
}

/// 実務再現シナリオ B (決済インシデント E-403) の詳細結合検証。
#[tokio::test]
async fn test_e2e_scenario_payment_incident_e403_details() {
    let state = match init_test_app_state() {
        Some(s) => s,
        None => return,
    };
    let fixtures = match load_fixtures() {
        Some(f) => f,
        None => return,
    };

    let cases = fixtures["cases"].as_array().unwrap();
    let payment_case = cases
        .iter()
        .find(|c| c["id"] == "scenario_payment_incident_e403")
        .expect("scenario_payment_incident_e403 ケースが存在すること。");

    let app = create_router(state, None);

    let mut req_body_map = serde_json::Map::new();
    req_body_map.insert("state".to_string(), payment_case["state"].clone());
    req_body_map.insert("questions".to_string(), payment_case["questions"].clone());

    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&req_body_map).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let actual: SystemOneResponse = serde_json::from_slice(&body_bytes).unwrap();

    let action = actual.answers.get("action").unwrap();
    assert!(action.is_choice());
    assert!(action.gating.is_some());
    let gating_meta = action.gating.as_ref().unwrap();
    assert_eq!(
        gating_meta.route,
        sokuto_core::gating::DecisionRoute::Fallback
    );
    // Fallback 時は安全弁としてエスカレーションプロンプトが生成されていること
    let esc = gating_meta
        .escalation
        .as_ref()
        .expect("Fallback 時は escalation が必須です。");
    let prompt = esc
        .prompt_template
        .as_ref()
        .expect("prompt_template が必須です。");
    assert!(prompt.contains("<context>"));
    assert!(prompt.contains("</context>"));
}

/// 実務再現シナリオ C (不正検知スクリーニング) の詳細結合検証。
#[tokio::test]
async fn test_e2e_scenario_fraud_detection_vip_details() {
    let state = match init_test_app_state() {
        Some(s) => s,
        None => return,
    };
    let fixtures = match load_fixtures() {
        Some(f) => f,
        None => return,
    };

    let cases = fixtures["cases"].as_array().unwrap();
    let fraud_case = cases
        .iter()
        .find(|c| c["id"] == "scenario_fraud_detection_vip")
        .expect("scenario_fraud_detection_vip ケースが存在すること。");

    let app = create_router(state, None);

    let mut req_body_map = serde_json::Map::new();
    req_body_map.insert("state".to_string(), fraud_case["state"].clone());
    req_body_map.insert("questions".to_string(), fraud_case["questions"].clone());

    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&req_body_map).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let actual: SystemOneResponse = serde_json::from_slice(&body_bytes).unwrap();

    let fraud = actual.answers.get("fraud_risk").unwrap();
    let p_true = fraud.as_noul().unwrap();
    // VIP かつリスクスコア低のため、不正確率は低水準 (< 0.50) であること
    assert!(
        p_true < 0.50,
        "VIP注文の不正確率は 0.50 未満である必要があります: {p_true}"
    );
}

/// 大規模候補数 (Choice K=16) の推論および全候補確率復元の検証。
#[tokio::test]
async fn test_e2e_large_scale_k16_reconstruction() {
    let state = match init_test_app_state() {
        Some(s) => s,
        None => return,
    };
    let fixtures = match load_fixtures() {
        Some(f) => f,
        None => return,
    };

    let cases = fixtures["cases"].as_array().unwrap();
    let k16_case = cases
        .iter()
        .find(|c| c["id"] == "choice_k16_large_scale")
        .expect("choice_k16_large_scale ケースが存在すること。");

    let app = create_router(state, None);

    let mut req_body_map = serde_json::Map::new();
    req_body_map.insert("state".to_string(), k16_case["state"].clone());
    req_body_map.insert("questions".to_string(), k16_case["questions"].clone());

    let req = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&req_body_map).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body_bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let actual: SystemOneResponse = serde_json::from_slice(&body_bytes).unwrap();

    let pref = actual.answers.get("prefecture").unwrap();
    assert!(pref.is_choice());
    let probs = pref.probabilities().unwrap();
    assert_eq!(probs.len(), 16, "全16候補の確率が復元されていること。");

    let sum_probs: f64 = probs.values().sum();
    assert!(
        (sum_probs - 1.0).abs() <= 1e-3,
        "確率の総和が 1.0 付近であること: {sum_probs}"
    );
}

/// Gating 境界値ケース (強制 AutoExecute および強制 Fallback) の動作検証。
#[tokio::test]
async fn test_e2e_forced_gating_boundary_cases() {
    let state = match init_test_app_state() {
        Some(s) => s,
        None => return,
    };
    let fixtures = match load_fixtures() {
        Some(f) => f,
        None => return,
    };

    let cases = fixtures["cases"].as_array().unwrap();

    let app = create_router(state, None);

    // 1. 低閾値による強制 AutoExecute
    let auto_case = cases
        .iter()
        .find(|c| c["id"] == "gating_boundary_forced_auto")
        .unwrap();

    let mut auto_req_map = serde_json::Map::new();
    auto_req_map.insert("state".to_string(), auto_case["state"].clone());
    auto_req_map.insert("questions".to_string(), auto_case["questions"].clone());
    auto_req_map.insert("gating".to_string(), auto_case["gating_config"].clone());

    let req_auto = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&auto_req_map).unwrap()))
        .unwrap();

    let resp_auto = app.clone().oneshot(req_auto).await.unwrap();
    assert_eq!(resp_auto.status(), StatusCode::OK);
    let bytes_auto = resp_auto.into_body().collect().await.unwrap().to_bytes();
    let actual_auto: SystemOneResponse = serde_json::from_slice(&bytes_auto).unwrap();
    let plan_ans = actual_auto.answers.get("plan").unwrap();
    assert_eq!(
        plan_ans.gating.as_ref().unwrap().route,
        sokuto_core::gating::DecisionRoute::AutoExecute
    );
    assert!(plan_ans.gating.as_ref().unwrap().escalation.is_none());

    // 2. 超高閾値による強制 Fallback
    let fallback_case = cases
        .iter()
        .find(|c| c["id"] == "gating_boundary_forced_fallback")
        .unwrap();

    let mut fallback_req_map = serde_json::Map::new();
    fallback_req_map.insert("state".to_string(), fallback_case["state"].clone());
    fallback_req_map.insert("questions".to_string(), fallback_case["questions"].clone());
    fallback_req_map.insert("gating".to_string(), fallback_case["gating_config"].clone());

    let req_fb = Request::builder()
        .method("POST")
        .uri("/v1/systemone")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&fallback_req_map).unwrap()))
        .unwrap();

    let resp_fb = app.oneshot(req_fb).await.unwrap();
    assert_eq!(resp_fb.status(), StatusCode::OK);
    let bytes_fb = resp_fb.into_body().collect().await.unwrap().to_bytes();
    let actual_fb: SystemOneResponse = serde_json::from_slice(&bytes_fb).unwrap();
    let amb_ans = actual_fb.answers.get("ambiguity").unwrap();
    assert_eq!(
        amb_ans.gating.as_ref().unwrap().route,
        sokuto_core::gating::DecisionRoute::Fallback
    );
    assert!(amb_ans.gating.as_ref().unwrap().escalation.is_some());
}
