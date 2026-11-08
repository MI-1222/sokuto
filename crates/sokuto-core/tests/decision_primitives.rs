//! # 3大決定プリミティブ数理統合テスト
//!
//! Choice(離散選択)、Score(順序尺度評価)、Noul(真偽確率)の数理計算、
//! キャリブレーション温度適用、境界値・極限値処理、および Jev レスポンス生成の総合検証を行う。

use indexmap::IndexMap;
use serde_json::json;
use sokuto_core::{
    CalibrationConfig, CoreError, Criteria, Question, SystemOneResponse, Usage, evaluate_choice,
    evaluate_noul, evaluate_question, evaluate_score,
};

/// Choice プリミティブの総合評価テスト。
#[test]
fn test_choice_comprehensive() {
    let mut criteria_map = IndexMap::new();
    criteria_map.insert("support".to_string(), "サポート問い合わせ".to_string());
    criteria_map.insert("billing".to_string(), "請求・支払い関連".to_string());
    criteria_map.insert("sales".to_string(), "営業・導入相談".to_string());
    criteria_map.insert("other".to_string(), "その他".to_string());
    let criteria = Criteria::Map(criteria_map);

    // 1. billing に明確に強いロジット
    let logits = vec![1.0, 6.0, 0.5, -2.0];
    let ans = evaluate_choice(&logits, &criteria, 1.0).unwrap();

    assert_eq!(ans.choice.as_deref(), Some("billing"));
    assert!(ans.score.is_none());
    assert!(ans.noul.is_none());

    let probs = ans.probabilities.as_ref().unwrap();
    assert_eq!(probs.len(), 4);
    assert!(probs["billing"] > 0.90);
    assert!(probs["billing"] + probs["support"] + probs["sales"] + probs["other"] > 0.9999);

    let conf = ans.confidence.unwrap();
    assert!(conf > 0.70 && conf <= 1.0);

    // 2. 完全一様分布(全ロジット同値): 確信度は 0.0、かつタイブレークで先頭候補が選ばれる
    let uniform_logits = vec![2.0, 2.0, 2.0, 2.0];
    let ans_uniform = evaluate_choice(&uniform_logits, &criteria, 1.0).unwrap();
    assert_eq!(ans_uniform.choice.as_deref(), Some("support"));
    assert!(ans_uniform.confidence.unwrap() < 1e-6);

    // 3. 単一候補 (K=1) のエッジケース: 確信度 1.0、確率 1.0
    let mut single_map = IndexMap::new();
    single_map.insert("only".to_string(), "唯一の選択肢".to_string());
    let criteria_single = Criteria::Map(single_map);
    let ans_single = evaluate_choice(&[5.0], &criteria_single, 1.0).unwrap();
    assert_eq!(ans_single.choice.as_deref(), Some("only"));
    assert_eq!(ans_single.confidence, Some(1.0));
    assert_eq!(ans_single.probabilities.unwrap()["only"], 1.0);

    // K=1 特異点における非有限値 (NaN, Inf, -Inf) のエラー検出。
    assert!(matches!(
        evaluate_choice(&[f64::NAN], &criteria_single, 1.0),
        Err(CoreError::MathError { .. })
    ));
    assert!(matches!(
        evaluate_choice(&[f64::INFINITY], &criteria_single, 1.0),
        Err(CoreError::MathError { .. })
    ));
    assert!(matches!(
        evaluate_choice(&[f64::NEG_INFINITY], &criteria_single, 1.0),
        Err(CoreError::MathError { .. })
    ));
}

/// Score プリミティブの総合評価テスト。
#[test]
fn test_score_comprehensive() {
    let levels = vec![
        "非常に不満".to_string(), // k=0
        "不満".to_string(),       // k=1
        "普通".to_string(),       // k=2
        "満足".to_string(),       // k=3
        "大変満足".to_string(),   // k=4
    ];
    let criteria = Criteria::List(levels);

    // 1. レベル 3 (満足) にピークがある場合
    let logits = vec![-1.0, 0.0, 2.0, 5.0, 1.0];
    let ans = evaluate_score(&logits, &criteria, 1.0).unwrap();

    let score = ans.score.unwrap();
    // 期待値は 2.5 〜 3.5 の間に収まる
    assert!(score > 2.5 && score < 3.5);

    let probs = ans.probabilities.as_ref().unwrap();
    assert_eq!(probs.len(), 5);
    assert!(probs["満足"] > 0.70);

    let conf = ans.confidence.unwrap();
    assert!(conf > 0.50 && conf <= 1.0);

    // 2. 二峰性両極端分裂 [0.5, 0.0, 0.0, 0.0, 0.5] (k=0 と k=4)
    // 最大可能分散 V_max = (5-1)^2 / 4 = 4.0
    // Var = (0 - 2.0)^2 * 0.5 + (4 - 2.0)^2 * 0.5 = 4.0
    // 確信度 C_var = 1.0 - 4.0 / 4.0 = 0.0
    let bimodal_logits = vec![10.0, -100.0, -100.0, -100.0, 10.0];
    let ans_bimodal = evaluate_score(&bimodal_logits, &criteria, 1.0).unwrap();
    assert!((ans_bimodal.score.unwrap() - 2.0).abs() < 1e-3);
    assert!(ans_bimodal.confidence.unwrap() < 1e-3);

    // 3. 単一レベル集中 (ワンホット): 確信度 1.0
    let one_hot_logits = vec![-100.0, -100.0, 20.0, -100.0, -100.0];
    let ans_one_hot = evaluate_score(&one_hot_logits, &criteria, 1.0).unwrap();
    assert!((ans_one_hot.score.unwrap() - 2.0).abs() < 1e-3);
    assert!((ans_one_hot.confidence.unwrap() - 1.0).abs() < 1e-3);
}

/// Noul プリミティブの総合評価テスト。
#[test]
fn test_noul_comprehensive() {
    // 1. 明確に真である場合
    let ans_true = evaluate_noul(&[5.0, 0.0], 1.0).unwrap();
    assert!(ans_true.noul.unwrap() > 0.99);
    assert!(ans_true.choice.is_none());
    assert!(ans_true.score.is_none());
    assert!(ans_true.probabilities.is_none());
    assert!(ans_true.confidence.is_none());

    // 2. 明確に偽である場合
    let ans_false = evaluate_noul(&[-2.0, 3.0], 1.0).unwrap();
    assert!(ans_false.noul.unwrap() < 0.01);

    // 3. 五分五分(真偽ロジット同値)
    let ans_neutral = evaluate_noul(&[1.5, 1.5], 1.0).unwrap();
    assert!((ans_neutral.noul.unwrap() - 0.5).abs() < 1e-6);

    // 4. 極端なロジット差でもオーバーフロー・アンダーフローせず動作
    let ans_extreme = evaluate_noul(&[5000.0, -5000.0], 1.0).unwrap();
    assert!((ans_extreme.noul.unwrap() - 1.0).abs() < 1e-6);
}

/// CalibrationConfig を介した温度自動適用と複数質問の一括レスポンス生成テスト。
#[test]
fn test_evaluate_question_and_systemone_response() {
    let calibration_json = r#"{
        "version": "1.0",
        "default_temperature": 1.0,
        "temperature_map": {
            "choice": {
                "2": 1.05,
                "3-5": 1.15,
                "6+": 1.25
            },
            "score": {
                "2-5": 1.08,
                "6-10": 1.18
            },
            "noul": 0.95
        }
    }"#;
    let config = CalibrationConfig::from_json_str(calibration_json).unwrap();

    // 質問 1: Choice (3候補 -> choice バケット "3-5" の温度 1.15 が適用される)
    let mut category_map = IndexMap::new();
    category_map.insert("bug".to_string(), "バグ報告".to_string());
    category_map.insert("feature".to_string(), "機能要望".to_string());
    category_map.insert("question".to_string(), "質問".to_string());
    let q_choice = Question::new_choice("問い合わせを分類せよ。", category_map);
    let logits_choice = vec![3.0, 1.0, 0.0];
    let ans_choice = evaluate_question(&q_choice, &logits_choice, &config).unwrap();
    assert_eq!(ans_choice.choice.as_deref(), Some("bug"));

    // 質問 2: Score (3段階 -> score バケット "2-5" の温度 1.08 が適用される)
    let q_score = Question::new_score(
        "緊急度を判定せよ。",
        vec!["低".to_string(), "中".to_string(), "高".to_string()],
    );
    let logits_score = vec![0.0, 2.0, 4.0];
    let ans_score = evaluate_question(&q_score, &logits_score, &config).unwrap();
    assert!(ans_score.score.unwrap() > 1.5);

    // 質問 3: Noul (温度 0.95 が適用される)
    let q_noul = Question::new_noul("有料プランのユーザーか。");
    let logits_noul = vec![2.0, 0.0];
    let ans_noul = evaluate_question(&q_noul, &logits_noul, &config).unwrap();
    assert!(ans_noul.noul.unwrap() > 0.85);

    // レスポンス全体の構築
    let mut answers = IndexMap::new();
    answers.insert("category".to_string(), ans_choice);
    answers.insert("urgency".to_string(), ans_score);
    answers.insert("is_paid".to_string(), ans_noul);

    let usage = Usage::new(128);
    let response = SystemOneResponse::new(answers, usage);

    // JSON シリアライズとスキーマ整合性検証
    let res_json = serde_json::to_value(&response).unwrap();
    assert_eq!(res_json["usage"]["completion_tokens"], json!(0));
    assert_eq!(res_json["usage"]["prompt_tokens"], json!(128));
    assert_eq!(res_json["answers"]["category"]["choice"], json!("bug"));
    assert!(res_json["answers"]["urgency"]["score"].as_f64().unwrap() > 1.5);
    assert!(res_json["answers"]["is_paid"]["noul"].as_f64().unwrap() > 0.85);
}

/// 作業用バッファを活用した Zero-Allocation API の動作検証。
#[test]
fn test_zero_allocation_buffers() {
    use sokuto_core::{evaluate_choice_with_buf, evaluate_score_with_buf};

    // 1. Choice with buf
    let mut criteria_map = IndexMap::new();
    criteria_map.insert("a".to_string(), "A".to_string());
    criteria_map.insert("b".to_string(), "B".to_string());
    let mut buf = [0.0; 2];
    let ans_choice =
        evaluate_choice_with_buf(&[2.0, 1.0], &Criteria::Map(criteria_map), 1.0, &mut buf).unwrap();
    assert_eq!(ans_choice.choice.as_deref(), Some("a"));

    // 2. Score with buf
    let criteria_list = Criteria::List(vec!["0".to_string(), "1".to_string(), "2".to_string()]);
    let mut buf_score = [0.0; 3];
    let ans_score =
        evaluate_score_with_buf(&[0.0, 1.0, 3.0], &criteria_list, 1.0, &mut buf_score).unwrap();
    assert!(ans_score.score.unwrap() > 1.0);
}

/// ゲーティング処理に向けた effective_confidence の動作検証。
#[test]
fn test_effective_confidence_for_gating() {
    // Choice: 保持している confidence がそのまま返却される
    let mut probs = IndexMap::new();
    probs.insert("opt".to_string(), 1.0);
    let ans_choice = sokuto_core::Answer::choice("opt", probs, 0.92);
    assert_eq!(ans_choice.effective_confidence(), Some(0.92));

    // Noul: confidence は None だが、effective_confidence() では |2P - 1| が透過的に得られる
    let ans_noul_high = sokuto_core::Answer::noul(0.95);
    assert!(ans_noul_high.confidence.is_none());
    assert!((ans_noul_high.effective_confidence().unwrap() - 0.90).abs() < 1e-6);

    let ans_noul_mid = sokuto_core::Answer::noul(0.50);
    assert!((ans_noul_mid.effective_confidence().unwrap() - 0.0).abs() < 1e-6);
}
