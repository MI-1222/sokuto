//! # 粗密 2 段階探索 (Coarse-to-Fine) 統合テストスイート
//!
//! 大規模候補数 ($K > 30$) に対する自動トリガー、ネガティブ保護 (Pinning)、
//! 決定論的タイブレーク、語彙スコアラーと埋め込みキャッシュ、確率空間復元、
//! ならびに実 ONNX モデルによる単一およびバッチ推論を網羅的に検証する。

use std::path::PathBuf;

use indexmap::IndexMap;
use sokuto_core::contract::calibration::CalibrationConfig;
use sokuto_core::schema::Question;
use sokuto_runtime::engine::{
    CandidateEmbeddingCache, CoarseScorer, CoarseToFineConfig, EmbeddingCoarseScorer,
    InferenceEngine, LexicalCoarseScorer, SessionConfig, filter_top_candidates,
    reconstruct_probabilities,
};
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

/// Banking77 風の大規模候補マップ (77候補) を生成するヘルパー。
fn generate_banking77_candidates() -> IndexMap<String, String> {
    let mut map = IndexMap::with_capacity(77);
    let sample_categories = [
        ("card_lost", "カードの紛失・盗難・利用停止手続き"),
        ("card_arrival", "申し込んだカードの配送状況や未着の確認"),
        ("card_pin_change", "暗証番号の再設定や変更手続き"),
        ("transfer_fee", "他行宛て振込手数料の確認や免除条件"),
        ("transfer_failed", "振込手続きの失敗・エラー原因の確認"),
        ("foreign_exchange", "外貨預金や為替レートの確認・両替"),
        ("atm_error", "ATM での現金出金トラブルやカード吸い込み"),
        ("balance_inquiry", "普通預金口座の残高照会・入出金明細"),
        ("direct_debit", "口座引き落とし・自動口座振替の登録解除"),
        (
            "loan_application",
            "マイカーローン・カードローンの新規借入申込",
        ),
        ("loan_repayment", "借入金の一括返済・繰り上げ返済の手続き"),
        ("address_change", "引越し等に伴う登録住所・電話番号の変更"),
        ("identity_verification", "本人確認書類の提出・再認証手続き"),
        (
            "app_login_issue",
            "スマートフォンアプリのログイン不可・生体認証エラー",
        ),
        ("apple_pay_setup", "Apple Pay / Google Pay の連携設定方法"),
    ];

    for (k, d) in sample_categories {
        map.insert(k.to_string(), d.to_string());
    }

    // 残りの候補を生成して合計 76 件の実候補を作成
    for i in 15..76 {
        map.insert(
            format!("banking_intent_{i:02}"),
            format!("銀行取引カテゴリ業務インテント番号 {i}"),
        );
    }

    // 77 件目に「該当なし (none/other)」ネガティブ候補を配置
    map.insert(
        "other".to_string(),
        "上記のいずれにも該当しないその他のお問い合わせ".to_string(),
    );

    map
}

#[test]
fn test_negative_preservation_when_similarity_is_zero() {
    let tokenizer = match init_default_tokenizer() {
        Some(t) => t,
        None => {
            eprintln!("スキップ: tokenizer.json が存在しません。");
            return;
        }
    };

    let mut candidates = IndexMap::new();
    // 40 候補を生成
    for i in 0..39 {
        candidates.insert(
            format!("opt_{i:02}"),
            format!("一般的な金融口座の取引インテント {i}"),
        );
    }
    // スコアが極小になる無関係なネガティブ候補
    candidates.insert(
        "none_of_the_above".to_string(),
        "該当なし / サポート外の問い合わせ".to_string(),
    );

    let config = CoarseToFineConfig {
        threshold: 30,
        top_m: 16,
        preserve_negative: true,
        negative_keys: vec!["none_of_the_above".to_string()],
        ..Default::default()
    };

    let scorer = LexicalCoarseScorer::new(&tokenizer);
    let state = "口座残高と昨日の引き落とし履歴を確認したいです。";
    let filtered = filter_top_candidates(state, &candidates, &config, &scorer)
        .expect("フィルタリングに失敗しました。");

    assert_eq!(filtered.selected.len(), 16);
    assert_eq!(filtered.total_candidates, 40);

    // ネガティブ候補がスコアに関わらず確実に保護されていること
    assert!(
        filtered.selected.contains_key("none_of_the_above"),
        "ネガティブ候補 'none_of_the_above' が保護されていません。"
    );
}

#[test]
fn test_deterministic_tie_breaking_order() {
    let tokenizer = match init_default_tokenizer() {
        Some(t) => t,
        None => return,
    };

    // 説明文が完全に同一でスコアがタイになる候補群
    let mut candidates = IndexMap::new();
    for i in 0..50 {
        candidates.insert(
            format!("tie_candidate_{i:02}"),
            "同一の候補説明テキスト".to_string(),
        );
    }

    let config = CoarseToFineConfig {
        threshold: 30,
        top_m: 20,
        preserve_negative: false,
        ..Default::default()
    };

    let scorer = LexicalCoarseScorer::new(&tokenizer);
    let query = "照会テストクエリ";

    // 複数回実行して完全に同一の結果が得られるか検証
    let run1 = filter_top_candidates(query, &candidates, &config, &scorer).unwrap();
    let run2 = filter_top_candidates(query, &candidates, &config, &scorer).unwrap();

    let keys1: Vec<&String> = run1.selected.keys().collect();
    let keys2: Vec<&String> = run2.selected.keys().collect();

    assert_eq!(keys1, keys2);
    // 先頭優先 (先勝ち) で 0..20 が選ばれていること
    for (i, key) in keys1.iter().enumerate().take(20) {
        assert_eq!(*key, &format!("tie_candidate_{i:02}"));
    }
}

#[test]
fn test_threshold_boundary_bypass_vs_trigger() {
    let tokenizer = match init_default_tokenizer() {
        Some(t) => t,
        None => return,
    };

    let config = CoarseToFineConfig {
        threshold: 30,
        top_m: 20,
        ..Default::default()
    };

    let scorer = LexicalCoarseScorer::new(&tokenizer);

    // 1. K = 30 の場合: should_trigger は false
    assert!(!config.should_trigger(30));

    // 2. K = 31 の場合: should_trigger は true
    assert!(config.should_trigger(31));

    let mut cand_31 = IndexMap::new();
    for i in 0..31 {
        cand_31.insert(format!("opt_{i:02}"), format!("説明文 {i}"));
    }

    let filtered = filter_top_candidates("クエリ", &cand_31, &config, &scorer).unwrap();
    assert_eq!(filtered.selected.len(), 20);
    assert_eq!(filtered.excluded_keys.len(), 11);
}

#[test]
fn test_reconstruct_probabilities_full_keys() {
    let mut original_keys = Vec::new();
    for i in 0..77 {
        original_keys.push(format!("cat_{i:02}"));
    }

    let mut sub_probabilities = IndexMap::new();
    sub_probabilities.insert("cat_01".to_string(), 0.65);
    sub_probabilities.insert("cat_05".to_string(), 0.25);
    sub_probabilities.insert("cat_10".to_string(), 0.10);

    let full_probs = reconstruct_probabilities(&original_keys, &sub_probabilities);

    assert_eq!(full_probs.len(), 77);
    assert_eq!(full_probs.get("cat_01"), Some(&0.65));
    assert_eq!(full_probs.get("cat_05"), Some(&0.25));
    assert_eq!(full_probs.get("cat_10"), Some(&0.10));
    assert_eq!(full_probs.get("cat_00"), Some(&0.0));
    assert_eq!(full_probs.get("cat_76"), Some(&0.0));

    // 確率の総和が 1.0 であること
    let sum: f64 = full_probs.values().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn test_lexical_coarse_scorer_keyword_relevance() {
    let tokenizer = match init_default_tokenizer() {
        Some(t) => t,
        None => return,
    };

    let mut candidates = IndexMap::new();
    candidates.insert("apple".to_string(), "赤いリンゴと果物".to_string());
    candidates.insert("car".to_string(), "自動車と高速道路の運転".to_string());
    candidates.insert(
        "computer".to_string(),
        "パソコンとプログラミング言語".to_string(),
    );

    let scorer = LexicalCoarseScorer::new(&tokenizer);
    let cand_pairs: Vec<(&str, &str)> = candidates
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let scores = scorer
        .score_candidates("プログラミングのコードを書く", &cand_pairs)
        .unwrap();

    // computer が最高スコアであること
    assert!(scores[2] > scores[0]);
    assert!(scores[2] > scores[1]);
}

#[test]
fn test_candidate_embedding_cache_and_scorer() {
    let mut cache = CandidateEmbeddingCache::new();
    // 3 次元の L2 正規化ベクトル
    cache.insert("finance", vec![1.0, 0.0, 0.0]).unwrap();
    cache.insert("tech", vec![0.0, 1.0, 0.0]).unwrap();
    cache.insert("health", vec![0.0, 0.0, 1.0]).unwrap();

    let query_vector = vec![0.1, 0.99, 0.0]; // tech に最も近い
    let scorer = EmbeddingCoarseScorer::new(&cache, &query_vector);

    let mut candidates = IndexMap::new();
    candidates.insert("finance".to_string(), "金融".to_string());
    candidates.insert("tech".to_string(), "IT技術".to_string());
    candidates.insert("health".to_string(), "健康医療".to_string());

    let mut cand_iter = candidates.iter().map(|(k, v)| (k.as_str(), v.as_str()));
    let scores = scorer
        .score_candidates_iter("クエリ", &mut cand_iter)
        .unwrap();
    assert!(scores[1] > scores[0]);
    assert!(scores[1] > scores[2]);
}

#[test]
fn test_end_to_end_coarse_to_fine_with_onnx_model() {
    let model_dir = default_model_dir();
    let model_path = model_dir.join("model.onnx");
    if !model_path.exists() {
        eprintln!("スキップ: model.onnx が存在しません。");
        return;
    }

    let tokenizer = match init_default_tokenizer() {
        Some(t) => t,
        None => return,
    };

    let engine = InferenceEngine::new(&model_path, SessionConfig::cpu_only())
        .expect("推論エンジンの初期化に失敗しました。");

    let calib_config = init_default_calibration();
    let coarse_config = CoarseToFineConfig {
        threshold: 30,
        top_m: 16,
        preserve_negative: true,
        ..Default::default()
    };

    // 77 候補の質問を作成
    let candidates = generate_banking77_candidates();
    assert_eq!(candidates.len(), 77);

    let question = Question::new_choice(
        "ユーザーの問い合わせ意図に最も合致する銀行業務カテゴリを選択してください。",
        candidates,
    );

    let state = "クレジットカードを外出先で無くしてしまいました。至急利用を止めたいです。";

    // Coarse-to-Fine 推論を実行
    let answer = engine
        .evaluate_question_coarse_to_fine(
            &tokenizer,
            state,
            &question,
            &calib_config,
            &coarse_config,
        )
        .expect("Coarse-to-Fine 推論に失敗しました。");

    // 検証
    assert!(answer.is_choice());
    let selected = answer.as_choice().unwrap();
    eprintln!("選択された候補: {selected}");

    // 確信度が 0.0〜1.0 の範囲内であること
    let conf = answer.confidence().unwrap();
    assert!((0.0..=1.0).contains(&conf));

    // 全 77 候補のキーが probabilities に含まれていること
    let probs = answer.probabilities().unwrap();
    assert_eq!(probs.len(), 77);
    assert!(probs.contains_key("other"));
    assert!(probs.contains_key(selected));
    assert!(probs.get(selected).unwrap() > &0.0);

    // 足切りされた候補の確率は 0.0 であること

    let zero_count = probs.values().filter(|&&p| p == 0.0).count();
    // 77 件中、Fine 段階に選ばれたのは最大 16 件なので、少なくとも 61 件は 0.0
    assert!(zero_count >= 77 - 16);
}

#[test]
fn test_batch_coarse_to_fine_mixed_questions() {
    let model_dir = default_model_dir();
    let model_path = model_dir.join("model.onnx");
    if !model_path.exists() {
        eprintln!("スキップ: model.onnx が存在しません。");
        return;
    }

    let tokenizer = match init_default_tokenizer() {
        Some(t) => t,
        None => return,
    };

    let engine = InferenceEngine::new(&model_path, SessionConfig::cpu_only())
        .expect("推論エンジンの初期化に失敗しました。");

    let calib_config = init_default_calibration();
    let coarse_config = CoarseToFineConfig {
        threshold: 30,
        top_m: 16,
        preserve_negative: true,
        ..Default::default()
    };

    let mut questions = IndexMap::new();

    // 1. K = 77 の大規模 Choice 質問
    let banking_candidates = generate_banking77_candidates();
    questions.insert(
        "q_large_choice".to_string(),
        Question::new_choice(
            "銀行業務の問い合わせカテゴリを選択してください。",
            banking_candidates,
        ),
    );

    // 2. K = 3 の通常 Choice 質問 (閾値以下、Coarse 適用外)
    let mut small_candidates = IndexMap::new();
    small_candidates.insert("urgent".to_string(), "緊急対応が必要".to_string());
    small_candidates.insert("normal".to_string(), "通常対応".to_string());
    small_candidates.insert("low".to_string(), "低優先度".to_string());
    questions.insert(
        "q_small_choice".to_string(),
        Question::new_choice("緊急度を選択してください。", small_candidates),
    );

    // 3. Score 質問 (5段階評価、Coarse 適用外)
    questions.insert(
        "q_score".to_string(),
        Question::new_score(
            "ユーザーの不満度を評価してください。",
            vec![
                "全く不満はない".to_string(),
                "やや不満".to_string(),
                "普通".to_string(),
                "不満である".to_string(),
                "極めて強い不満・憤り".to_string(),
            ],
        ),
    );

    // 4. Noul 質問 (真偽判定、Coarse 適用外)
    questions.insert(
        "q_noul".to_string(),
        Question::new_noul("この問い合わせはカード紛失に関するものである。"),
    );

    let state = "カードを盗まれてしまい、今すぐ利用を停止したいです！至急対応してください！";

    // バッチ粗密推論を実行
    let answers = engine
        .evaluate_batch_questions_coarse_to_fine(
            &tokenizer,
            state,
            &questions,
            &calib_config,
            &coarse_config,
        )
        .expect("バッチ Coarse-to-Fine 推論に失敗しました。");

    assert_eq!(answers.len(), 4);

    // q_large_choice: 77 候補の確率マップが復元され、候補が選ばれていること
    let ans_large = answers.get("q_large_choice").unwrap();
    assert!(ans_large.is_choice());
    assert_eq!(ans_large.probabilities().unwrap().len(), 77);

    // q_small_choice: 3 候補の確率マップ、候補が選ばれていること
    let ans_small = answers.get("q_small_choice").unwrap();
    assert!(ans_small.is_choice());
    assert_eq!(ans_small.probabilities().unwrap().len(), 3);

    // q_score: 期待スコアが算出されていること
    let ans_score = answers.get("q_score").unwrap();
    assert!(ans_score.is_score());

    // q_noul: 真実確率が算出されていること
    let ans_noul = answers.get("q_noul").unwrap();
    assert!(ans_noul.is_noul());
    assert!((0.0..=1.0).contains(&ans_noul.as_noul().unwrap()));
}

#[test]
fn test_excessive_negative_candidates_strict_top_m_bound() {
    let tokenizer = match init_default_tokenizer() {
        Some(t) => t,
        None => return,
    };

    let mut candidates = IndexMap::new();
    // 35 件中 25 件がネガティブ候補
    for i in 0..25 {
        candidates.insert(format!("none_{i:02}"), format!("該当なし {i}"));
    }
    for i in 0..10 {
        candidates.insert(format!("valid_{i:02}"), format!("通常の銀行業務 {i}"));
    }

    let config = CoarseToFineConfig {
        threshold: 30,
        top_m: 16,
        preserve_negative: true,
        negative_keys: (0..25).map(|i| format!("none_{i:02}")).collect(),
        ..Default::default()
    };

    let scorer = LexicalCoarseScorer::new(&tokenizer);
    let filtered = filter_top_candidates("クエリ", &candidates, &config, &scorer).unwrap();

    // ネガティブ候補が 25 件あっても、厳密に top_m (16) 件に収まっていること
    assert_eq!(filtered.selected.len(), 16);
    assert_eq!(filtered.excluded_keys.len(), 35 - 16);
}

#[test]
fn test_lexical_coarse_scorer_desc_priority_and_no_key_norm_inflation() {
    let tokenizer = match init_default_tokenizer() {
        Some(t) => t,
        None => return,
    };

    let scorer = LexicalCoarseScorer::new(&tokenizer);

    // 1. 長大な英語内部キーを持つが説明文は同一の候補
    let candidates = [
        (
            "internal_uuid_bank_account_balance_inquiry_001_very_long_id",
            "口座残高の確認",
        ),
        ("b", "口座残高の確認"),
        ("fallback_key_atm", ""), // 空説明文: キー名がフォールバック
    ];

    let scores = scorer
        .score_candidates("口座の残高を確認したい", &candidates)
        .expect("スコアリングに成功。");

    // 長い英語キーを持っても説明文のみでエンコードされるため、スコアが等しいこと
    assert!(
        (scores[0] - scores[1]).abs() < 1e-6,
        "英語内部キーの混入による分母ノルム膨張が発生していないこと。"
    );
    assert!(scores[0] > 0.3, "妥当なコサイン類似度が得られること。");

    // 2. 空説明文のフォールバック動作
    let fallback_scores = scorer
        .score_candidates("atm", &candidates)
        .expect("スコアリングに成功。");
    assert!(
        fallback_scores[2] > 0.0,
        "説明文が空の場合はキー名が照合対象としてフォールバックされること。"
    );
}

#[test]
fn test_general_inquiry_not_pinned_by_default() {
    let tokenizer = match init_default_tokenizer() {
        Some(t) => t,
        None => return,
    };

    let mut candidates = IndexMap::new();
    // 35 件の候補を作成
    for i in 0..33 {
        candidates.insert(format!("cat_{i:02}"), format!("業務カテゴリ {i}"));
    }
    // 一般的な問い合わせカテゴリ(独立カテゴリとして扱われるべきもの)
    candidates.insert(
        "general_inquiry".to_string(),
        "その他のお問い合わせ".to_string(),
    );
    // 明示的なメタ除外候補
    candidates.insert("none".to_string(), "該当なし".to_string());

    let config = CoarseToFineConfig {
        threshold: 30,
        top_m: 5,
        preserve_negative: true,
        ..Default::default()
    };

    let scorer = LexicalCoarseScorer::new(&tokenizer);
    // クエリは特定の業務カテゴリに合致
    let filtered = filter_top_candidates("業務カテゴリ 01 の手続き", &candidates, &config, &scorer)
        .expect("フィルタリングに成功。");

    // "none" はデフォルト設定でネガティブ保護される
    assert!(filtered.selected.contains_key("none"));

    // "general_inquiry" ("その他のお問い合わせ") はデフォルトではネガティブ候補として保護されず、
    // 類似度が低ければ Top-5 に残らないこと(不当な Pinning が発生しないこと)
    assert!(
        !filtered.selected.contains_key("general_inquiry"),
        "その他のお問い合わせがネガティブ候補として不当に Pinning されないこと。"
    );
}
