//! # ONNX Runtime 推論セッション統合テストスイート
//!
//! セッション初期化、入出力テンソル契約検証、単一フォワードパス推論、
//! トークナイザーから決定数理までの一気通貫パイプライン、および異常入力防壁を検証する。

use std::path::PathBuf;

use indexmap::IndexMap;
use sokuto_core::contract::model_spec::ModelInputDimensions;
use sokuto_core::decision::evaluate_question;
use sokuto_core::schema::{Criteria, Question, QuestionType, SystemOneResponse};
use sokuto_runtime::engine::{ExecutionProvider, InferenceEngine, SessionConfig};
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

#[test]
fn test_session_init_non_existent_file() {
    let dummy_path = PathBuf::from("/non/existent/model.onnx");
    let result = InferenceEngine::new(&dummy_path, SessionConfig::default());
    assert!(result.is_err());
    let err_msg = result.err().unwrap().to_string();
    assert!(err_msg.contains("モデルファイルが存在しません"));
}

#[test]
fn test_session_init_and_contract_verification() {
    let model_dir = default_model_dir();
    let model_path = model_dir.join("model.onnx");
    if !model_path.exists() {
        eprintln!(
            "スキップ: model.onnx が存在しません (パス: {:?})。",
            model_path
        );
        return;
    }

    let config = SessionConfig::cpu_only();
    let engine = InferenceEngine::new(&model_path, config).expect("モデルロードに失敗しました。");

    assert_eq!(engine.active_provider(), ExecutionProvider::CPU);
    assert!(engine.input_names().contains(&"input_ids".to_string()));
    assert!(engine.input_names().contains(&"attention_mask".to_string()));
    assert!(engine.input_names().contains(&"op_indices".to_string()));
    assert!(engine.output_names().contains(&"logits".to_string()));
}

#[test]
fn test_forward_raw_and_buffer_reuse() {
    let model_dir = default_model_dir();
    let model_path = model_dir.join("model.onnx");
    if !model_path.exists() {
        eprintln!("スキップ: model.onnx が存在しません。");
        return;
    }

    let engine =
        InferenceEngine::new(&model_path, SessionConfig::cpu_only()).expect("モデルロード失敗。");

    let batch_size = 1;
    let seq_len = 16;
    let num_options = 3;
    let dims = ModelInputDimensions::new(batch_size, seq_len, num_options).unwrap();

    let input_ids = vec![
        101i64, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 102,
    ];
    let attention_mask = vec![1i64; seq_len];
    let op_indices = vec![2i64, 5, 8];

    // forward_raw の検証
    let logits = engine
        .forward_raw(&input_ids, &attention_mask, &op_indices, dims)
        .expect("forward_raw に失敗しました。");

    assert_eq!(logits.len(), 1);
    assert_eq!(logits[0].len(), num_options);
    for val in &logits[0] {
        assert!(
            val.is_finite(),
            "ロジットに非有限値が含まれています: {val}。"
        );
    }

    // forward_raw_into (バッファ再利用) の検証
    let mut out_buf = vec![0.0f64; num_options];
    engine
        .forward_raw_into(&input_ids, &attention_mask, &op_indices, dims, &mut out_buf)
        .expect("forward_raw_into に失敗しました。");

    for (a, b) in logits[0].iter().zip(out_buf.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "バッファ出力と通常出力が一致しません: {a} vs {b}。"
        );
    }
}

#[test]
fn test_end_to_end_question_inference_and_decision() {
    let model_dir = default_model_dir();
    let model_path = model_dir.join("model.onnx");
    let tokenizer_path = model_dir.join("tokenizer.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        eprintln!("スキップ: model.onnx または tokenizer.json が存在しません。");
        return;
    }

    let tokenizer = JevTokenizer::from_file(&tokenizer_path).expect("トークナイザーロード失敗。");
    let engine =
        InferenceEngine::new(&model_path, SessionConfig::cpu_only()).expect("モデルロード失敗。");

    // 1. Choice 型質問のエンドツーエンド推論
    let mut criteria_map = IndexMap::new();
    criteria_map.insert(
        "card".to_string(),
        "クレジットカード関連の問い合わせ".to_string(),
    );
    criteria_map.insert("loan".to_string(), "各種ローン関連の相談".to_string());
    criteria_map.insert("other".to_string(), "その他の照会事項".to_string());

    let question = Question {
        question_type: QuestionType::Choice,
        instructions: "ユーザーの意図を最も適切なカテゴリに分類せよ。".to_string(),
        criteria: Some(Criteria::Map(criteria_map)),
    };

    let state = "カードを紛失してしまったのですが、利用停止手続きを行いたいです。";
    let tokenized = tokenizer
        .encode_question(state, &question)
        .expect("トークナイズ失敗。");

    assert_eq!(tokenized.op_indices.len(), 3);
    assert_eq!(tokenized.option_keys, vec!["card", "loan", "other"]);

    let logits = engine.forward_question(&tokenized).expect("推論実行失敗。");
    assert_eq!(logits.len(), 3);

    let calibration_path = model_dir.join("calibration.json");
    let calib_config = if calibration_path.exists() {
        std::fs::read_to_string(&calibration_path)
            .ok()
            .and_then(|s| {
                sokuto_core::contract::calibration::CalibrationConfig::from_json_str(&s).ok()
            })
            .unwrap_or_default()
    } else {
        sokuto_core::contract::calibration::CalibrationConfig::default()
    };

    let answer =
        evaluate_question(&question, &logits, &calib_config).expect("決定数理評価に失敗しました。");

    assert!(answer.is_choice());
    let selected = answer.selected().unwrap();
    assert!(["card", "loan", "other"].contains(&selected));
    assert!(answer.confidence().unwrap() >= 0.0 && answer.confidence().unwrap() <= 1.0);
    assert_eq!(answer.probabilities().unwrap().len(), 3);

    // 2. SystemOneResponse への統合
    let mut answers = IndexMap::new();
    answers.insert("q_choice_01".to_string(), answer);

    let response = SystemOneResponse::new(
        answers,
        sokuto_core::schema::Usage::new(tokenized.input_ids.len()),
    );
    let json = serde_json::to_string(&response).expect("シリアライズ失敗。");
    assert!(json.contains("q_choice_01"));
    assert!(json.contains("probabilities"));

    // 3. Score 型質問のエンドツーエンド推論
    let score_question = Question {
        question_type: QuestionType::Score,
        instructions: "カスタマーサポートの対応品質を評価せよ。".to_string(),
        criteria: Some(Criteria::List(vec![
            "極めて不満".to_string(),
            "不満".to_string(),
            "普通".to_string(),
            "満足".to_string(),
            "大変満足".to_string(),
        ])),
    };
    let score_tokenized = tokenizer
        .encode_question(
            "丁寧かつ迅速に対応していただき、大変助かりました。",
            &score_question,
        )
        .expect("Score トークナイズ失敗。");
    let score_logits = engine
        .forward_question(&score_tokenized)
        .expect("Score 推論失敗。");
    assert_eq!(score_logits.len(), 5);

    let score_answer = evaluate_question(&score_question, &score_logits, &calib_config)
        .expect("Score 決定数理評価失敗。");
    assert!(score_answer.is_score());
    let score_val = score_answer.as_score().unwrap();
    assert!((0.0..=4.0).contains(&score_val));
    assert!((0.0..=1.0).contains(&score_answer.confidence().unwrap()));

    // 4. Noul 型質問のエンドツーエンド推論
    let noul_question = Question {
        question_type: QuestionType::Noul,
        instructions: "この文書は機密情報に該当するか判定せよ。".to_string(),
        criteria: None,
    };
    let noul_tokenized = tokenizer
        .encode_question("本資料は社外秘であり複製を禁じます。", &noul_question)
        .expect("Noul トークナイズ失敗。");
    let noul_logits = engine
        .forward_question(&noul_tokenized)
        .expect("Noul 推論失敗。");
    assert_eq!(noul_logits.len(), 2);

    let noul_answer = evaluate_question(&noul_question, &noul_logits, &calib_config)
        .expect("Noul 決定数理評価失敗。");
    assert!(noul_answer.is_noul());
    let noul_prob = noul_answer.as_noul().unwrap();
    assert!((0.0..=1.0).contains(&noul_prob));
    let eff_conf = noul_answer.effective_confidence().unwrap();
    assert!((0.0..=1.0).contains(&eff_conf));
}

#[test]
fn test_guard_against_out_of_bounds_op_indices() {
    let model_dir = default_model_dir();
    let model_path = model_dir.join("model.onnx");
    if !model_path.exists() {
        return;
    }

    let engine =
        InferenceEngine::new(&model_path, SessionConfig::cpu_only()).expect("モデルロード失敗。");

    let batch_size = 1;
    let seq_len = 16;
    let num_options = 2;
    let dims = ModelInputDimensions::new(batch_size, seq_len, num_options).unwrap();

    let input_ids = vec![0i64; seq_len];
    let attention_mask = vec![1i64; seq_len];

    // 系列長 16 に対してインデックス 16 (範囲外 0..16) を指定
    let invalid_op_indices = vec![5i64, 16];

    let result = engine.forward_raw(&input_ids, &attention_mask, &invalid_op_indices, dims);
    assert!(
        result.is_err(),
        "範囲外の op_indices が拒絶されませんでした。"
    );
    let err_str = result.err().unwrap().to_string();
    assert!(
        err_str.contains("範囲外"),
        "期待されるエラーメッセージではありません: {err_str}。"
    );
}

#[test]
fn test_guard_against_shape_mismatches() {
    let model_dir = default_model_dir();
    let model_path = model_dir.join("model.onnx");
    if !model_path.exists() {
        return;
    }

    let engine =
        InferenceEngine::new(&model_path, SessionConfig::cpu_only()).expect("モデルロード失敗。");

    let dims = ModelInputDimensions::new(1, 16, 2).unwrap();
    let input_ids = vec![0i64; 16];
    let attention_mask = vec![1i64; 15]; // 長さ不一致 (15 != 16)
    let op_indices = vec![2i64, 4];

    let result = engine.forward_raw(&input_ids, &attention_mask, &op_indices, dims);
    assert!(result.is_err(), "形状不一致が拒絶されませんでした。");
}

#[test]
fn test_concurrent_inference_with_session_pool() {
    use std::sync::Arc;
    use std::thread;

    let model_dir = default_model_dir();
    let model_path = model_dir.join("model.onnx");
    let tokenizer_path = model_dir.join("tokenizer.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        return;
    }

    let tokenizer =
        Arc::new(JevTokenizer::from_file(&tokenizer_path).expect("トークナイザーロード失敗。"));
    let config = SessionConfig::cpu_only().with_pool_size(2);
    let engine = Arc::new(InferenceEngine::new(&model_path, config).expect("モデルロード失敗。"));

    assert_eq!(engine.pool_size(), 2);

    let mut handles = Vec::new();
    for thread_id in 0..4 {
        let engine_clone = Arc::clone(&engine);
        let tokenizer_clone = Arc::clone(&tokenizer);

        let handle = thread::spawn(move || {
            let question = Question::new_noul(format!("スレッド {thread_id} の言明妥当性テスト。"));
            let tokenized = tokenizer_clone
                .encode_question("並行推論テスト用の入力テキストです。", &question)
                .expect("トークナイズ失敗。");
            let logits = engine_clone
                .forward_question(&tokenized)
                .expect("並行推論実行失敗。");
            assert_eq!(logits.len(), 2);
            assert!(logits[0].is_finite());
            assert!(logits[1].is_finite());
        });
        handles.push(handle);
    }

    for handle in handles {
        handle
            .join()
            .expect("スレッド実行でパニックが発生しました。");
    }
}

#[test]
fn test_optimized_model_cache_and_tier_presets() {
    let model_dir = default_model_dir();
    let model_path = model_dir.join("model.onnx");
    if !model_path.exists() {
        return;
    }

    let cache_dir = std::env::temp_dir().join(format!("sokuto_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&cache_dir);
    let cache_path = cache_dir.join("optimized_model.onnx");

    let config = SessionConfig::for_tier1().with_optimized_model_path(&cache_path);

    let engine = InferenceEngine::new(&model_path, config).expect("Tier 1 セッション初期化失敗。");
    assert_eq!(engine.pool_size(), 1);

    // 最適化モデルファイルが生成されたことを確認
    assert!(
        cache_path.exists(),
        "最適化モデルファイルが出力されていること。"
    );

    // クリーンアップ
    let _ = std::fs::remove_file(&cache_path);
    let _ = std::fs::remove_dir(&cache_dir);
}
