//! # 粗密二段階階層ルーティング統合テスト
//!
//! `CoarseToFineRouter`, `HierarchicalMapping`, `HierarchicalRouterConfig`,
//! および `InferenceEngine::evaluate_question_hierarchical` の振る舞いを検証する。

use std::path::PathBuf;

use sokuto_core::contract::calibration::CalibrationConfig;
use sokuto_core::gating::DecisionRoute;
use sokuto_runtime::engine::{InferenceEngine, SessionConfig};
use sokuto_runtime::hierarchical::{HierarchicalMapping, HierarchicalRouterConfig};
use sokuto_runtime::tokenizer::JevTokenizer;

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

/// トークナイザーの初期化ヘルパー。
fn init_default_tokenizer() -> Option<JevTokenizer> {
    let tok_path = default_model_dir().join("tokenizer.json");
    if tok_path.exists() {
        JevTokenizer::from_file(&tok_path).ok()
    } else {
        None
    }
}

/// 較正設定のロードヘルパー。
fn init_default_calibration() -> CalibrationConfig {
    let calib_path = default_model_dir().join("calibration.json");
    if calib_path.exists() {
        std::fs::read_to_string(&calib_path)
            .ok()
            .and_then(|s| CalibrationConfig::from_json_str(&s).ok())
            .unwrap_or_default()
    } else {
        CalibrationConfig::default()
    }
}

#[test]
fn test_hierarchical_router_config_validation() {
    let config = HierarchicalRouterConfig::default();
    assert!(config.validate().is_ok());

    // 異常値
    let mut invalid = config.clone();
    invalid.beam_margin_threshold = 1.5;
    assert!(invalid.validate().is_err());

    let mut invalid2 = config.clone();
    invalid2.escalate_confidence_threshold = -0.1;
    assert!(invalid2.validate().is_err());
}

#[test]
fn test_hierarchical_routing_with_real_model() {
    let model_path = default_model_dir().join("model.onnx");
    if !model_path.exists() {
        // モデルファイルが存在しないテスト環境ではスキップ
        return;
    }

    let tokenizer = match init_default_tokenizer() {
        Some(t) => t,
        None => return,
    };

    let calib_config = init_default_calibration();
    let session_config = SessionConfig::default();
    let engine = match InferenceEngine::new(&model_path, session_config) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mapping = HierarchicalMapping::banking77();
    let config = HierarchicalRouterConfig::default();

    let state = "カードを落として紛失してしまったので、今すぐ利用を停止して新しいカードを再発行したいです。";
    let instructions = "お客様のお問い合わせ内容に最も適合するカテゴリを選択してください。";

    let result = engine
        .evaluate_question_hierarchical(
            &tokenizer,
            state,
            instructions,
            &mapping,
            &calib_config,
            config,
            None,
        )
        .expect("階層推論の実行に成功する。");

    let answer = result.answer();
    assert!(answer.is_choice());

    // 確率分布の総数が全 77 候補であること
    let probs = answer.probabilities().unwrap();
    assert_eq!(probs.len(), 77);

    // 確率総和が 1.0 であること
    let sum: f64 = probs.values().sum();
    assert!((sum - 1.0).abs() < 1e-4);

    // トレース情報の検証
    let trace = result.trace();
    assert!(!trace.top1_coarse.is_empty());
    assert!(trace.top1_coarse_prob > 0.0);
    assert!(trace.evaluated_fine_count > 0);

    // カード関連の大分類(card_ops または account_sec)が選ばれていること
    assert!(
        trace.top1_coarse == "card_ops" || trace.top1_coarse == "account_sec",
        "想定される大分類が選ばれていること: {}",
        trace.top1_coarse
    );

    // Answer への消費変換が可能なこと
    let converted_answer = result.into_answer();
    assert!(converted_answer.is_choice());
}

#[test]
fn test_hierarchical_routing_early_exit() {
    let model_path = default_model_dir().join("model.onnx");
    if !model_path.exists() {
        return;
    }

    let tokenizer = match init_default_tokenizer() {
        Some(t) => t,
        None => return,
    };

    let calib_config = init_default_calibration();
    let session_config = SessionConfig::default();
    let engine = match InferenceEngine::new(&model_path, session_config) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mapping = HierarchicalMapping::banking77();
    // 意図的に極めて高いエスカレーション閾値を設定
    let config = HierarchicalRouterConfig {
        escalate_confidence_threshold: 0.99,
        ..Default::default()
    };

    let state = "こんにちは。";
    let instructions = "お客様のお問い合わせ内容を選択してください。";

    let result = engine
        .evaluate_question_hierarchical(
            &tokenizer,
            state,
            instructions,
            &mapping,
            &calib_config,
            config,
            None,
        )
        .expect("Early-Exit 階層推論に成功する。");

    let trace = result.trace();
    assert!(trace.early_exit_triggered);
    assert_eq!(trace.evaluated_fine_count, 0);

    let answer = result.answer();
    assert_eq!(answer.route(), Some(DecisionRoute::ConfirmOrEscalate));
}

#[test]
fn test_hierarchical_routing_with_dirichlet_calibrator() {
    let model_path = default_model_dir().join("model.onnx");
    if !model_path.exists() {
        return;
    }

    let tokenizer = match init_default_tokenizer() {
        Some(t) => t,
        None => return,
    };

    let calib_config = init_default_calibration();
    let session_config = SessionConfig::default();
    let engine = match InferenceEngine::new(&model_path, session_config) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mapping = HierarchicalMapping::banking77();
    let num_coarse = mapping.total_coarse_count();

    // 恒等変換 Dirichlet 較正器を作成
    let calibrator = sokuto_runtime::hierarchical::DirichletCalibrator::identity(num_coarse);
    let config = HierarchicalRouterConfig::default().with_coarse_calibrator(calibrator);

    let state = "暗証番号を忘れてしまい、ATMでロックがかかってしまいました。";
    let instructions = "お客様のお問い合わせ内容に最も適合するカテゴリを選択してください。";

    let result = engine
        .evaluate_question_hierarchical(
            &tokenizer,
            state,
            instructions,
            &mapping,
            &calib_config,
            config,
            None,
        )
        .expect("Dirichlet 較正付き階層推論の実行に成功する。");

    let answer = result.answer();
    assert!(answer.is_choice());
    assert_eq!(answer.probabilities().unwrap().len(), 77);

    // クラス数不一致のエラーハンドリング検証 (大分類 7 に対し クラス数 3 の較正器)
    let mismatched_calibrator = sokuto_runtime::hierarchical::DirichletCalibrator::identity(3);
    let invalid_config =
        HierarchicalRouterConfig::default().with_coarse_calibrator(mismatched_calibrator);

    let err = engine.evaluate_question_hierarchical(
        &tokenizer,
        state,
        instructions,
        &mapping,
        &calib_config,
        invalid_config,
        None,
    );
    assert!(err.is_err());
}

#[test]
fn test_evaluate_batch_questions_hierarchical() {
    let model_path = default_model_dir().join("model.onnx");
    if !model_path.exists() {
        return;
    }

    let tokenizer = match init_default_tokenizer() {
        Some(t) => t,
        None => return,
    };

    let calib_config = init_default_calibration();
    let session_config = SessionConfig::default();
    let engine = match InferenceEngine::new(&model_path, session_config) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mapping = HierarchicalMapping::banking77();
    let config = HierarchicalRouterConfig::default();

    let mut questions = indexmap::IndexMap::new();
    questions.insert(
        "q1".to_string(),
        sokuto_core::schema::Question {
            question_type: sokuto_core::schema::QuestionType::Choice,
            instructions: "カードの紛失手続きについてカテゴリを選んでください。".to_string(),
            criteria: None,
        },
    );
    questions.insert(
        "q2".to_string(),
        sokuto_core::schema::Question {
            question_type: sokuto_core::schema::QuestionType::Choice,
            instructions: "海外送金の手数料についてカテゴリを選んでください。".to_string(),
            criteria: None,
        },
    );

    let state = "カードを落としたので止めてください。また海外へ送金したいです。";
    let batch_results = engine
        .evaluate_batch_questions_hierarchical(
            &tokenizer,
            state,
            &questions,
            &mapping,
            &calib_config,
            config,
            None,
        )
        .expect("バッチ階層推論の実行に成功する。");

    assert_eq!(batch_results.len(), 2);
    assert!(batch_results.get("q1").unwrap().answer().is_choice());
    assert!(batch_results.get("q2").unwrap().answer().is_choice());
}

#[test]
fn test_hierarchical_routing_new_dual_session_engine() {
    let state_model_path = default_model_dir().join("state_encoder.onnx");
    let query_model_path = default_model_dir().join("query_evaluator.onnx");
    if !state_model_path.exists() || !query_model_path.exists() {
        return;
    }

    let tokenizer = match init_default_tokenizer() {
        Some(t) => t,
        None => return,
    };

    let calib_config = init_default_calibration();
    let session_config = SessionConfig::default();
    let cache_manager = std::sync::Arc::new(sokuto_runtime::engine::StateCacheManager::new(
        64 * 1024 * 1024,
        600,
        sokuto_runtime::engine::CacheCompressionMode::Fp16,
    ));

    let dual_engine = match sokuto_runtime::engine::DualSessionEngine::new(
        &state_model_path,
        &query_model_path,
        session_config,
        cache_manager,
    ) {
        Ok(e) => e,
        Err(_) => return,
    };

    let state = "カードを落としたので止めてください。";
    let (s_ids, s_mask) = tokenizer
        .encode_state_tensor(state)
        .expect("Stateのエンコードに成功すること。");
    let cached_state = dual_engine
        .encode_and_cache_state(state, &s_ids, &s_mask, [1, s_ids.len()])
        .expect("Stateのキャッシュに成功すること。");

    let mapping = HierarchicalMapping::banking77();
    let config = HierarchicalRouterConfig::default();

    let router = sokuto_runtime::hierarchical::CoarseToFineRouter::new_dual(
        &dual_engine,
        &cached_state,
        &tokenizer,
        &mapping,
        &calib_config,
        config,
    )
    .expect("デュアルエンジンのルーター初期化に成功すること。");

    let result = router
        .route(
            state,
            "カードの紛失手続きについてカテゴリを選んでください。",
            None,
        )
        .expect("デュアルセッションによる階層推論に成功すること。");

    let answer = result.answer();
    assert!(answer.is_choice());
    assert!(result.trace().evaluated_fine_count > 0);
}
