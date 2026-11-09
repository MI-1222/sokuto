//! # ランタイム統合ゲーティングテストスイート
//!
//! 推論エンジン (`InferenceEngine`) における、単一質問推論・プレフィックス共有バッチ推論・
//! 粗密 2 段階探索推論を通じた実効確信度ゲーティング (3 系統ルーティング) の統合動作を検証する。

use std::path::PathBuf;

use indexmap::IndexMap;
use sokuto_core::contract::calibration::{CalibrationConfig, GatingThresholds};
use sokuto_core::gating::{DecisionRoute, GatingConfig};
use sokuto_core::schema::Question;
use sokuto_runtime::engine::{CoarseToFineConfig, InferenceEngine, SessionConfig};
use sokuto_runtime::tokenizer::JevTokenizer;

/// ワークスペースのルートディレクトリを取得する。
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ワークスペースルートの解決に失敗した。")
        .to_path_buf()
}

/// 配布モデルディレクトリを取得する。
fn default_model_dir() -> PathBuf {
    workspace_root().join("models").join("default")
}

/// テスト用エンジンおよびトークナイザーの準備ヘルパー。
fn init_test_engine() -> Option<(InferenceEngine, JevTokenizer, CalibrationConfig)> {
    let model_dir = default_model_dir();
    let onnx_path = model_dir.join("model.onnx");
    let tok_path = model_dir.join("tokenizer.json");
    let calib_path = model_dir.join("calibration.json");

    if !onnx_path.exists() || !tok_path.exists() {
        eprintln!("スキップ: 配布モデルファイルが存在しない。");
        return None;
    }

    let session_config = SessionConfig {
        pool_size: 1,
        ..Default::default()
    };

    let engine = InferenceEngine::new(&onnx_path, session_config).ok()?;
    let tokenizer = JevTokenizer::from_file(&tok_path).ok()?;
    let calib_config = if calib_path.exists() {
        std::fs::read_to_string(&calib_path)
            .ok()
            .and_then(|s| CalibrationConfig::from_json_str(&s).ok())
            .unwrap_or_default()
    } else {
        CalibrationConfig::default()
    };

    Some((engine, tokenizer, calib_config))
}

/// 単一質問推論におけるゲーティング透過適用の検証。
#[test]
fn test_single_question_gating() {
    let (engine, tokenizer, mut calib_config) = match init_test_engine() {
        Some(ctx) => ctx,
        None => return,
    };

    let mut choice_map = IndexMap::new();
    choice_map.insert("positive".to_string(), "肯定的・良好".to_string());
    choice_map.insert("negative".to_string(), "否定的・悪化".to_string());
    let question = Question::new_choice("感情分類を行ってください。", choice_map);
    let coarse_config = CoarseToFineConfig::default();

    // 1. calib_config のデフォルトゲーティング設定による透過適用
    calib_config.gating_thresholds = GatingThresholds {
        high_threshold: 0.70,
        low_threshold: 0.35,
        top_margin_threshold: 0.15,
        ..Default::default()
    };
    let answer = engine
        .evaluate_question_coarse_to_fine(
            &tokenizer,
            "本サービスは非常に使いやすく満足しています。",
            &question,
            &calib_config,
            &coarse_config,
        )
        .expect("推論に成功する。");

    assert!(answer.gating.is_some());
    let gating = answer.gating.as_ref().unwrap();
    assert!(gating.confidence > 0.0);
    assert!(gating.entropy.is_some());
    assert!(gating.margin.is_some());

    // 2. 明示的な GatingConfig による上書き (AutoExecute の誘導)
    let auto_gating = GatingConfig {
        enabled: true,
        thresholds: GatingThresholds {
            high_threshold: 0.001,
            low_threshold: 0.0005,
            top_margin_threshold: 0.0,
            ood_enabled: false,
            energy_threshold: 100.0,
            ..GatingThresholds::default()
        },
    };
    let answer_auto = engine
        .evaluate_question_coarse_to_fine_with_gating(
            &tokenizer,
            "本サービスは非常に使いやすく満足しています。",
            &question,
            &calib_config,
            &coarse_config,
            Some(&auto_gating),
        )
        .expect("推論に成功する。");

    let gating_auto = answer_auto.gating.as_ref().unwrap();
    assert_eq!(gating_auto.route, DecisionRoute::AutoExecute);
    assert!(gating_auto.escalation.is_none());

    // 3. GatingConfig 無効化 (enabled: false)
    let disabled_gating = GatingConfig {
        enabled: false,
        ..Default::default()
    };
    let answer_disabled = engine
        .evaluate_question_coarse_to_fine_with_gating(
            &tokenizer,
            "本サービスは非常に使いやすく満足しています。",
            &question,
            &calib_config,
            &coarse_config,
            Some(&disabled_gating),
        )
        .expect("推論に成功する。");

    assert!(answer_disabled.gating.is_none());
}

/// プレフィックス共有バッチ推論およびチャンキング推論におけるゲーティング検証。
#[test]
fn test_batch_and_chunked_gating() {
    let (engine, tokenizer, calib_config) = match init_test_engine() {
        Some(ctx) => ctx,
        None => return,
    };

    let mut questions = IndexMap::new();

    let mut choice_map = IndexMap::new();
    choice_map.insert("bug".to_string(), "不具合報告".to_string());
    choice_map.insert("feature".to_string(), "機能要望".to_string());
    questions.insert(
        "type".to_string(),
        Question::new_choice("問い合わせの種類を選択してください。", choice_map),
    );

    questions.insert(
        "is_urgent".to_string(),
        Question::new_noul("緊急度が高い案件であるか判定してください。"),
    );

    // 確信度境界を厳格にして ConfirmOrEscalate を誘導
    let strict_gating = GatingConfig {
        enabled: true,
        thresholds: GatingThresholds {
            high_threshold: 0.9999,
            low_threshold: 0.00001,
            top_margin_threshold: 0.9999,
            ood_enabled: false,
            energy_threshold: 100.0,
            ..GatingThresholds::default()
        },
    };

    // 通常バッチ推論でのゲーティング
    let answers = engine
        .evaluate_batch_questions_with_gating(
            &tokenizer,
            "ログインボタンを押しても画面が遷移しません。早急に対応願います。",
            &questions,
            &calib_config,
            Some(&strict_gating),
        )
        .expect("バッチ推論に成功する。");

    assert_eq!(answers.len(), 2);
    for (qid, answer) in &answers {
        assert!(
            answer.gating.is_some(),
            "質問 {} にゲーティングが付与される。",
            qid
        );
        let gating = answer.gating.as_ref().unwrap();
        assert_eq!(
            gating.route,
            DecisionRoute::ConfirmOrEscalate,
            "質問 {} は確認境界に達する。",
            qid
        );
        assert!(gating.escalation.is_some());
        let esc = gating.escalation.as_ref().unwrap();
        assert!(esc.prompt_template.is_some());
        let prompt = esc.prompt_template.as_ref().unwrap();
        assert!(prompt.contains("<context>"));
        assert!(prompt.contains("ログインボタンを押しても画面が遷移しません"));
        assert!(prompt.contains("思考連鎖 (Chain-of-Thought)"));
    }

    // チャンク推論 (chunk_size = 1) でのゲーティング
    let chunked_answers = engine
        .evaluate_batch_questions_chunked_with_gating(
            &tokenizer,
            "ログインボタンを押しても画面が遷移しません。早急に対応願います。",
            &questions,
            &calib_config,
            1,
            Some(&strict_gating),
        )
        .expect("チャンク推論に成功する。");

    assert_eq!(chunked_answers.len(), 2);
    for (qid, answer) in &chunked_answers {
        assert!(
            answer.gating.is_some(),
            "質問 {} にゲーティングが付与される。",
            qid
        );
        let gating = answer.gating.as_ref().unwrap();
        assert_eq!(gating.route, DecisionRoute::ConfirmOrEscalate);
    }
}

/// 粗密 2 段階探索推論におけるゲーティングおよび確率復元検証。
#[test]
fn test_coarse_to_fine_gating_and_reconstruction() {
    let (engine, tokenizer, calib_config) = match init_test_engine() {
        Some(ctx) => ctx,
        None => return,
    };

    let mut questions = IndexMap::new();

    // 10 候補の Choice 質問を作成 (Coarse スクリーニングの対象)
    let mut criteria = IndexMap::new();
    for i in 1..=10 {
        criteria.insert(format!("cat_{}", i), format!("カテゴリ説明 {}", i));
    }
    questions.insert(
        "large_choice".to_string(),
        Question::new_choice("多数のカテゴリから最適なものを選択せよ。", criteria),
    );

    let coarse_config = CoarseToFineConfig {
        threshold: 5,
        top_m: 3,
        preserve_negative: false,
        negative_keys: Vec::new(),
    };

    let gating_config = GatingConfig {
        enabled: true,
        thresholds: GatingThresholds {
            high_threshold: 0.70,
            low_threshold: 0.35,
            top_margin_threshold: 0.15,
            ood_enabled: true,
            energy_threshold: -1.0,
            ..GatingThresholds::default()
        },
    };

    let answers = engine
        .evaluate_batch_questions_coarse_to_fine_chunked_with_gating(
            &tokenizer,
            "カテゴリに関する問い合わせテキストです。",
            &questions,
            &calib_config,
            &coarse_config,
            2,
            Some(&gating_config),
        )
        .expect("粗密探索推論に成功する。");

    let ans = answers.get("large_choice").expect("回答が存在する。");
    assert!(ans.gating.is_some());
    let gating = ans.gating.as_ref().unwrap();

    // 確率マップが全 10 候補に復元されていることを検証
    let probs = ans.probabilities().expect("確率マップが存在する。");
    assert_eq!(probs.len(), 10, "10 候補すべての確率が復元されている。");

    // ゲーティングの確信度・エントロピー・マージンが正しく計算されていることを検証
    assert!(gating.confidence >= 0.0 && gating.confidence <= 1.0);
    assert!(gating.entropy.is_some());
    assert!(gating.margin.is_some());

    match gating.route {
        DecisionRoute::AutoExecute => {
            assert!(
                gating.escalation.is_none(),
                "AutoExecute ではエスカレーション生成が抑制される。"
            );
        }
        DecisionRoute::ConfirmOrEscalate => {
            assert!(
                gating.escalation.is_some(),
                "ConfirmOrEscalate ではエスカレーション文脈が生成される。"
            );
        }
        DecisionRoute::Fallback => {
            assert!(gating.reason.contains("フォールバック"));
        }
    }
}

#[test]
fn test_runtime_energy_ood_circuit_breaker() {
    let (engine, tokenizer, calib_config) = match init_test_engine() {
        Some(t) => t,
        None => return,
    };

    let mut choice_map = IndexMap::new();
    choice_map.insert("positive".to_string(), "肯定的・改善".to_string());
    choice_map.insert("neutral".to_string(), "中立・変化なし".to_string());
    choice_map.insert("negative".to_string(), "否定的・悪化".to_string());
    let question = Question::new_choice("感情分類を行ってください。", choice_map);
    let coarse_config = CoarseToFineConfig::default();

    // 1. エネルギー閾値を厳しく設定し、意図的に OOD 安全弁を発動させるケース
    // 正規化エネルギー閾値を -50.0 (非常に低い値) に設定すると、通常の推論ロジットでも energy > threshold となり OOD 検知される
    let strict_ood_config = GatingConfig {
        enabled: true,
        thresholds: GatingThresholds {
            high_threshold: 0.10, // 確信度条件は緩い
            low_threshold: 0.05,
            top_margin_threshold: 0.0,
            ood_enabled: true,
            energy_threshold: -50.0, // 極めて厳しいエネルギー閾値 -> 強制 OOD
            ..GatingThresholds::default()
        },
    };

    let answer_ood = engine
        .evaluate_question_coarse_to_fine_with_gating(
            &tokenizer,
            "本サービスは非常に使いやすく満足しています。",
            &question,
            &calib_config,
            &coarse_config,
            Some(&strict_ood_config),
        )
        .expect("推論に成功する。");

    let gating_ood = answer_ood.gating.expect("gating メタデータが存在する。");
    assert_eq!(
        gating_ood.route,
        DecisionRoute::Fallback,
        "OOD 安全弁発動により Fallback へ短絡降格される。"
    );
    assert!(gating_ood.is_ood, "is_ood フラグが true である。");
    assert!(
        gating_ood.energy.is_some(),
        "energy 値が算出・記録されている。"
    );

    let prompt = gating_ood
        .escalation
        .as_ref()
        .expect("エスカレーション情報が存在する。")
        .prompt_template
        .as_ref()
        .expect("プロンプトテンプレートが存在する。");
    assert!(
        prompt.contains("None of the above / 該当なし"),
        "OOD 特化の思考誘導が含まれる。"
    );
    assert!(
        prompt.contains("未定義カテゴリ"),
        "未定義カテゴリに関する指示が含まれる。"
    );

    // 2. ood_enabled = false で OOD 安全弁を無効化したケース
    let disabled_ood_config = GatingConfig {
        enabled: true,
        thresholds: GatingThresholds {
            high_threshold: 0.10,
            low_threshold: 0.05,
            top_margin_threshold: 0.0,
            ood_enabled: false,
            energy_threshold: -50.0,
            ..GatingThresholds::default()
        },
    };

    let answer_disabled = engine
        .evaluate_question_coarse_to_fine_with_gating(
            &tokenizer,
            "本サービスは非常に使いやすく満足しています。",
            &question,
            &calib_config,
            &coarse_config,
            Some(&disabled_ood_config),
        )
        .expect("推論に成功する。");

    let gating_disabled = answer_disabled
        .gating
        .expect("gating メタデータが存在する。");
    assert_ne!(
        gating_disabled.route,
        DecisionRoute::Fallback,
        "OOD 安全弁が無効化されているため Fallback への強制降格は発生しない。"
    );
    assert!(!gating_disabled.is_ood, "is_ood フラグは false である。");
}
