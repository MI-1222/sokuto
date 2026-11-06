//! # プレフィックス共有および単一フォワードパス並列推論統合テスト
//!
//! 複数質問に対するプレフィックス共有バッチトークナイズ、
//! 質問間アテンション干渉 (Context Rot) の完全遮断 (単一推論とのロジット完全一致パリティ)、
//! 候補数不揃い時の Gather 層安全性、State 優先トランケーションの質問間独立性、
//! ならびに決定論的プリミティブ解決の一気通貫パイプラインを検証する。

use std::path::PathBuf;

use indexmap::IndexMap;
use sokuto_core::contract::{CalibrationConfig, OP_INDEX_PADDING};
use sokuto_core::schema::Question;
use sokuto_runtime::engine::{BatchScratchpad, InferenceEngine, SessionConfig};
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
fn test_batch_encoding_and_padding_integrity() {
    let model_dir = default_model_dir();
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        eprintln!("スキップ: tokenizer.json が存在しません。");
        return;
    }

    let tokenizer = JevTokenizer::from_file(&tokenizer_path).expect("トークナイザーのロード失敗。");

    let mut questions = IndexMap::new();

    // 質問 1: Choice 2 択
    let mut choice_map = IndexMap::new();
    choice_map.insert("positive".to_string(), "肯定的・良好".to_string());
    choice_map.insert("negative".to_string(), "否定的・悪化".to_string());
    questions.insert(
        "sentiment".to_string(),
        Question::new_choice("感情分類を行ってください。", choice_map),
    );

    // 質問 2: Score 5 段階
    let score_labels = vec![
        "全く同意しない".to_string(),
        "あまり同意しない".to_string(),
        "どちらともいえない".to_string(),
        "やや同意する".to_string(),
        "強く同意する".to_string(),
    ];
    questions.insert(
        "satisfaction".to_string(),
        Question::new_score("顧客満足度を 5 段階で評価してください。", score_labels),
    );

    // 質問 3: Noul (真偽 2 択)
    questions.insert(
        "is_urgent".to_string(),
        Question::new_noul("この問い合わせは緊急対応が必要ですか？"),
    );

    let state = "ユーザーからのフィードバック: 注文した商品がまだ届かず困っています。至急確認してください。";
    let batch = tokenizer
        .encode_batch_questions(state, &questions)
        .expect("バッチエンコード失敗。");

    assert_eq!(batch.dims.batch_size, 3);
    assert_eq!(batch.dims.num_options, 5); // 最大候補数は Score の 5
    assert_eq!(batch.candidate_counts, vec![2, 5, 2]);
    assert_eq!(
        batch.question_keys,
        vec!["sentiment", "satisfaction", "is_urgent"]
    );

    let l_max = batch.dims.sequence_length;
    let k_max = batch.dims.num_options;

    // 質問 1 (sentiment): 2 択なので候補インデックスの後半 3 スロットは -1 (パディング値)
    let q1_op_slice = &batch.op_indices[0..k_max];
    assert_ne!(q1_op_slice[0], OP_INDEX_PADDING);
    assert_ne!(q1_op_slice[1], OP_INDEX_PADDING);
    assert_eq!(q1_op_slice[2], OP_INDEX_PADDING);
    assert_eq!(q1_op_slice[3], OP_INDEX_PADDING);
    assert_eq!(q1_op_slice[4], OP_INDEX_PADDING);

    // 質問 2 (satisfaction): 5 択なので全スロット非パディング
    let q2_op_slice = &batch.op_indices[k_max..k_max * 2];
    for &idx in q2_op_slice {
        assert_ne!(idx, OP_INDEX_PADDING);
    }

    // 質問 3 (is_urgent): Noul は真・偽の 2 択なので後半 3 スロットは -1
    let q3_op_slice = &batch.op_indices[k_max * 2..k_max * 3];
    assert_ne!(q3_op_slice[0], OP_INDEX_PADDING);
    assert_ne!(q3_op_slice[1], OP_INDEX_PADDING);
    assert_eq!(q3_op_slice[2], OP_INDEX_PADDING);
    assert_eq!(q3_op_slice[3], OP_INDEX_PADDING);
    assert_eq!(q3_op_slice[4], OP_INDEX_PADDING);

    // 各行のアテンションマスク検証
    for i in 0..3 {
        let mask = &batch.attention_mask[i * l_max..(i + 1) * l_max];
        let valid_len = mask.iter().filter(|&&m| m == 1).count();
        assert!(valid_len > 0);
        // 有効トークン以降は厳密に 0
        for &m in &mask[valid_len..] {
            assert_eq!(m, 0);
        }
    }
}

#[test]
fn test_batch_inference_parity_and_context_rot_isolation() {
    let model_dir = default_model_dir();
    let model_path = model_dir.join("model.onnx");
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !model_path.exists() || !tokenizer_path.exists() {
        eprintln!("スキップ: モデルまたはトークナイザーが存在しません。");
        return;
    }

    let tokenizer = JevTokenizer::from_file(&tokenizer_path).expect("トークナイザーロード失敗。");
    let engine =
        InferenceEngine::new(&model_path, SessionConfig::cpu_only()).expect("モデルロード失敗。");

    let state = "システム監視ログ: サーバー CPU 使用率が 98% に達し、応答遅延が発生しています。";

    let mut questions = IndexMap::new();

    let mut choice_map = IndexMap::new();
    choice_map.insert("low".to_string(), "低負荷".to_string());
    choice_map.insert("medium".to_string(), "中負荷".to_string());
    choice_map.insert("high".to_string(), "高負荷".to_string());
    questions.insert(
        "cpu_load".to_string(),
        Question::new_choice("サーバーの負荷状態を選択してください。", choice_map),
    );

    let score_labels = vec![
        "正常".to_string(),
        "軽微".to_string(),
        "警戒".to_string(),
        "重大".to_string(),
    ];
    questions.insert(
        "alert_level".to_string(),
        Question::new_score(
            "インシデント深刻度を 4 段階で判定してください。",
            score_labels,
        ),
    );

    questions.insert(
        "requires_reboot".to_string(),
        Question::new_noul("サーバーの即時再起動が必要ですか？"),
    );

    // 1. 各質問を個別に単一推論 (Single Forward Pass x 3)
    let mut single_logits: IndexMap<String, Vec<f64>> = IndexMap::new();
    for (q_key, q) in &questions {
        let tokenized = tokenizer
            .encode_question(state, q)
            .expect("単一エンコード失敗。");
        let dims = sokuto_core::contract::model_spec::ModelInputDimensions::new(
            1,
            tokenized.input_ids.len(),
            tokenized.op_indices.len(),
        )
        .unwrap();

        let logits = engine
            .forward_raw(
                &tokenized.input_ids,
                &tokenized.attention_mask,
                &tokenized.op_indices,
                dims,
            )
            .expect("単一推論失敗。");
        single_logits.insert(q_key.clone(), logits[0].clone());
    }

    // 2. 複数質問を一括バッチ推論 (Single Forward Pass x 1)
    let batch_logits = engine
        .forward_batch_questions(&tokenizer, state, &questions)
        .expect("バッチ推論失敗。");

    assert_eq!(batch_logits.len(), 3);

    // 3. 単一推論とバッチ推論のロジット完全一致 (パリティ) を検証
    for (q_key, expected_logits) in &single_logits {
        let actual_logits = &batch_logits[q_key];
        assert_eq!(
            expected_logits.len(),
            actual_logits.len(),
            "質問 '{}' のロジット要素数が一致すること。",
            q_key
        );

        for (k, (&expected, &actual)) in expected_logits.iter().zip(actual_logits).enumerate() {
            let diff = (expected - actual).abs();
            assert!(
                diff < 1e-5,
                "質問 '{q_key}' の候補インデックス {k} においてロジットが一致しません: 単一={expected}, バッチ={actual}, 差分={diff}。"
            );
        }
    }
}

#[test]
fn test_gather_safety_with_disparate_candidate_counts() {
    let model_dir = default_model_dir();
    let model_path = model_dir.join("model.onnx");
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !model_path.exists() || !tokenizer_path.exists() {
        eprintln!("スキップ: モデルまたはトークナイザーが存在しません。");
        return;
    }

    let tokenizer = JevTokenizer::from_file(&tokenizer_path).expect("トークナイザーロード失敗。");
    let engine =
        InferenceEngine::new(&model_path, SessionConfig::cpu_only()).expect("モデルロード失敗。");

    let state = "顧客サポート問い合わせ: 決済エラーが発生しました。";

    let mut questions = IndexMap::new();

    // 質問 1: 2 択
    let mut map_2 = IndexMap::new();
    map_2.insert("yes".to_string(), "はい".to_string());
    map_2.insert("no".to_string(), "いいえ".to_string());
    questions.insert(
        "q_small".to_string(),
        Question::new_choice("決済は成功しましたか？", map_2),
    );

    // 質問 2: 7 択 (候補数差が大きい)
    let mut map_7 = IndexMap::new();
    for i in 1..=7 {
        map_7.insert(format!("opt_{i}"), format!("エラーカテゴリ {i}"));
    }
    questions.insert(
        "q_large".to_string(),
        Question::new_choice("エラー種別を選択してください。", map_7),
    );

    let batch_logits = engine
        .forward_batch_questions(&tokenizer, state, &questions)
        .expect("候補数不揃いバッチの推論がクラッシュせずに完了すること。");

    assert_eq!(batch_logits["q_small"].len(), 2);
    assert_eq!(batch_logits["q_large"].len(), 7);

    for val in &batch_logits["q_small"] {
        assert!(val.is_finite());
    }
    for val in &batch_logits["q_large"] {
        assert!(val.is_finite());
    }
}

#[test]
fn test_state_priority_truncation_independence_in_batch() {
    let model_dir = default_model_dir();
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        eprintln!("スキップ: tokenizer.json が存在しません。");
        return;
    }

    let tokenizer = JevTokenizer::from_file(&tokenizer_path).expect("トークナイザーロード失敗。");

    // 意図的に極端に長い State を用意
    let long_state = "共通の長い文脈情報です。".repeat(50);

    let mut questions = IndexMap::new();

    // 質問 A: 極めて短い指示文と選択肢
    let mut map_a = IndexMap::new();
    map_a.insert("1".to_string(), "A".to_string());
    map_a.insert("2".to_string(), "B".to_string());
    questions.insert(
        "short_q".to_string(),
        Question::new_choice("短い指示。", map_a),
    );

    // 質問 B: やや長めの指示文と選択肢 (固定長が constrained_max_len 未満に収まるように設計)
    let mut map_b = IndexMap::new();
    for i in 1..=4 {
        map_b.insert(format!("k_{i}"), format!("候補項目 {i}"));
    }
    questions.insert(
        "long_q".to_string(),
        Question::new_choice(
            "この質問は State 優先トランケーションの独立性を検証するための指示文です。",
            map_b,
        ),
    );

    // 最大系列長を 128 に制限して強制的に State をトランケートさせる
    let constrained_max_len = 128;
    let batch = tokenizer
        .encode_batch_questions_with_max_len(&long_state, &questions, constrained_max_len)
        .expect("制限付きバッチエンコードに成功すること。");

    assert_eq!(batch.dims.batch_size, 2);
    assert!(batch.dims.sequence_length <= constrained_max_len);

    // どちらの質問でもすべての [OP] マーカーが脱落せず保持されていることを検証
    assert_eq!(batch.candidate_counts, vec![2, 4]);
}

#[test]
fn test_evaluate_batch_questions_and_scratchpad_parity() {
    let model_dir = default_model_dir();
    let model_path = model_dir.join("model.onnx");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let calib_path = model_dir.join("calibration.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        eprintln!("スキップ: モデルまたはトークナイザーが存在しません。");
        return;
    }

    let tokenizer = JevTokenizer::from_file(&tokenizer_path).expect("トークナイザーロード失敗。");
    let engine =
        InferenceEngine::new(&model_path, SessionConfig::cpu_only()).expect("モデルロード失敗。");

    let calib_config = if calib_path.exists() {
        let content = std::fs::read_to_string(&calib_path).expect("較正ファイル読み込み失敗。");
        CalibrationConfig::from_json_str(&content).expect("較正設定ロード失敗。")
    } else {
        CalibrationConfig::default()
    };

    let state =
        "注文番号 #12345: 配達予定日を過ぎていますが荷物が届きません。再配達をお願いします。";

    let mut questions = IndexMap::new();
    let mut category_map = IndexMap::new();
    category_map.insert("delivery".to_string(), "配送関連".to_string());
    category_map.insert("payment".to_string(), "決済関連".to_string());
    category_map.insert("product".to_string(), "製品仕様".to_string());
    questions.insert(
        "category".to_string(),
        Question::new_choice("問い合わせのカテゴリを分類してください。", category_map),
    );

    questions.insert(
        "urgency".to_string(),
        Question::new_score(
            "対応優先度を評価してください。",
            vec!["低".to_string(), "中".to_string(), "高".to_string()],
        ),
    );

    questions.insert(
        "resolved".to_string(),
        Question::new_noul("この問題は既に解決済みですか？"),
    );

    // 通常の evaluate_batch_questions
    let answers = engine
        .evaluate_batch_questions(&tokenizer, state, &questions, &calib_config)
        .expect("一括決定評価に成功すること。");

    assert_eq!(answers.len(), 3);

    // Choice の検証
    let cat_ans = &answers["category"];
    assert!(cat_ans.choice.is_some());
    assert!(cat_ans.confidence.unwrap() >= 0.0 && cat_ans.confidence.unwrap() <= 1.0);
    assert_eq!(cat_ans.probabilities.as_ref().unwrap().len(), 3);

    // Score の検証 (3 段階: レベル 0..=2)
    let urg_ans = &answers["urgency"];
    assert!(urg_ans.score.is_some());
    assert!(urg_ans.score.unwrap() >= 0.0 && urg_ans.score.unwrap() <= 2.0);

    // Noul の検証
    let res_ans = &answers["resolved"];
    assert!(res_ans.noul.is_some());
    assert!(res_ans.noul.unwrap() >= 0.0 && res_ans.noul.unwrap() <= 1.0);

    // スクラッチパッド再利用版とのパリティ検証
    let mut scratchpad = BatchScratchpad::default();
    let answers_scratch = engine
        .evaluate_batch_questions_with_scratchpad(
            &tokenizer,
            state,
            &questions,
            &calib_config,
            &mut scratchpad,
        )
        .expect("スクラッチパッド一括決定評価に成功すること。");

    assert_eq!(answers, answers_scratch);
}

#[test]
fn test_chunked_batch_inference_parity() {
    let model_dir = default_model_dir();
    let model_path = model_dir.join("model.onnx");
    let tokenizer_path = model_dir.join("tokenizer.json");
    let calib_path = model_dir.join("calibration.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        eprintln!("スキップ: モデルまたはトークナイザーが存在しません。");
        return;
    }

    let tokenizer = JevTokenizer::from_file(&tokenizer_path).expect("トークナイザーロード失敗。");
    let engine =
        InferenceEngine::new(&model_path, SessionConfig::cpu_only()).expect("モデルロード失敗。");

    let calib_config = if calib_path.exists() {
        let content = std::fs::read_to_string(&calib_path).expect("較正ファイル読み込み失敗。");
        CalibrationConfig::from_json_str(&content).expect("較正設定ロード失敗。")
    } else {
        CalibrationConfig::default()
    };

    let state = "サーバーアクセスログ: 多数の 500 エラーが発生しています。";

    // 5 つの質問を用意
    let mut questions = IndexMap::new();
    for i in 1..=5 {
        let mut map = IndexMap::new();
        map.insert("opt1".to_string(), "選択肢 1".to_string());
        map.insert("opt2".to_string(), "選択肢 2".to_string());
        questions.insert(
            format!("q_{i}"),
            Question::new_choice(format!("質問 {i} の指示文です。"), map),
        );
    }

    // 通常の全件一括バッチ推論
    let full_batch_answers = engine
        .evaluate_batch_questions(&tokenizer, state, &questions, &calib_config)
        .expect("全件一括推論成功。");

    // チャンクサイズ 2 でのマイクロバッチ推論 (計 3 回のマイクロバッチ: 2 + 2 + 1)
    let chunked_answers = engine
        .evaluate_batch_questions_chunked(&tokenizer, state, &questions, &calib_config, 2)
        .expect("チャンク分割推論成功。");

    assert_eq!(full_batch_answers.len(), 5);
    assert_eq!(chunked_answers.len(), 5);

    // 全質問の結果が完全一致することを検証
    for (key, expected_ans) in &full_batch_answers {
        let actual_ans = &chunked_answers[key];
        assert_eq!(expected_ans, actual_ans);
    }
}
