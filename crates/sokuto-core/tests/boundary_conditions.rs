//! # 境界値・極限値統合単体テスト (`boundary_conditions.rs`)
//!
//! ロードマップ 2.3「境界値・極限値単体テスト」に基づき、
//! 生ロジットやリクエストが極端な値・異常値を取った場合でも、推論エンジンが
//! パニックせず、数学的に正確な解または型安全な `CoreError` を返却することを保証する。

use indexmap::IndexMap;
use sokuto_core::contract::CalibrationConfig;
use sokuto_core::decision::{
    evaluate_choice, evaluate_choice_with_buf, evaluate_noul, evaluate_question, evaluate_score,
    evaluate_score_with_buf,
};
use sokuto_core::error::CoreError;
use sokuto_core::schema::{Criteria, Question};

/// 確率分布および確信度の極限状態を検証するテストモジュール。
mod distribution_extremes {
    use super::*;

    #[test]
    fn test_choice_one_hot_extreme() {
        // 単一候補のみロジットが圧倒的に大きいワンホット極限。
        // 期待値: 採択候補の確率が厳密に 1.0、確信度が厳密に 1.0。
        let mut map = IndexMap::new();
        map.insert("A".to_string(), "選択肢A".to_string());
        map.insert("B".to_string(), "選択肢B".to_string());
        map.insert("C".to_string(), "選択肢C".to_string());
        let criteria = Criteria::Map(map);

        let logits = vec![100.0, -100.0, -100.0];
        let answer = evaluate_choice(&logits, &criteria, 1.0).unwrap();

        assert_eq!(answer.choice.as_deref(), Some("A"));
        let probs = answer.probabilities.as_ref().unwrap();
        assert!((probs.get("A").copied().unwrap() - 1.0).abs() < 1e-6);
        assert!(probs.get("B").copied().unwrap() < 1e-6);
        assert!(probs.get("C").copied().unwrap() < 1e-6);

        let conf = answer.confidence.unwrap();
        assert!((conf - 1.0).abs() < 1e-6);
        assert_eq!(answer.effective_confidence(), Some(1.0));

        // 完全アンダーフロー(10000.0)の極限ケース
        let extreme_logits = vec![10000.0, -10000.0, -10000.0];
        let ans_extreme = evaluate_choice(&extreme_logits, &criteria, 1.0).unwrap();
        let probs_ext = ans_extreme.probabilities.unwrap();
        assert_eq!(probs_ext.get("A").copied(), Some(1.0));
        assert_eq!(probs_ext.get("B").copied(), Some(0.0));
        assert_eq!(probs_ext.get("C").copied(), Some(0.0));
        assert_eq!(ans_extreme.confidence, Some(1.0));
    }

    #[test]
    fn test_choice_uniform_extreme() {
        // 全候補のロジットが同一の完全一様分布極限。
        // 期待値: 全確率が 1/K、エントロピー最大により確信度が厳密に 0.0。
        let mut map = IndexMap::new();
        map.insert("opt1".to_string(), "オプション1".to_string());
        map.insert("opt2".to_string(), "オプション2".to_string());
        map.insert("opt3".to_string(), "オプション3".to_string());
        map.insert("opt4".to_string(), "オプション4".to_string());
        let criteria = Criteria::Map(map);

        let logits = vec![5.0, 5.0, 5.0, 5.0];
        let answer = evaluate_choice(&logits, &criteria, 1.0).unwrap();

        // 決定論的タイブレークにより先頭候補が採択される。
        assert_eq!(answer.choice.as_deref(), Some("opt1"));

        let probs = answer.probabilities.as_ref().unwrap();
        for val in probs.values() {
            assert!((val - 0.25).abs() < 1e-6);
        }

        let conf = answer.confidence.unwrap();
        assert!(conf < 1e-6);
        assert!((0.0..=1.0).contains(&conf));
        assert!(answer.effective_confidence().unwrap() < 1e-6);
    }

    #[test]
    fn test_score_extremes_and_bimodal_variance() {
        let criteria = Criteria::List(vec![
            "極小".to_string(),
            "低".to_string(),
            "中".to_string(),
            "高".to_string(),
        ]); // M = 4, レベル 0, 1, 2, 3

        // 1. ワンホット極限: 単一レベル(k=2)に集中。
        let logits_one_hot = vec![-100.0, -100.0, 100.0, -100.0];
        let ans_one_hot = evaluate_score(&logits_one_hot, &criteria, 1.0).unwrap();
        assert!((ans_one_hot.score.unwrap() - 2.0).abs() < 1e-6);
        assert!((ans_one_hot.confidence.unwrap() - 1.0).abs() < 1e-6);

        // 2. 二峰性両極端分裂極限: レベル 0 とレベル 3 のみに二分。
        // 理論最大分散 V_max = (4-1)^2 / 4 = 2.25。
        // 確信度 C_var = 1.0 - 2.25 / 2.25 = 0.0。
        let logits_bimodal = vec![100.0, -100.0, -100.0, 100.0];
        let ans_bimodal = evaluate_score(&logits_bimodal, &criteria, 1.0).unwrap();
        assert!((ans_bimodal.score.unwrap() - 1.5).abs() < 1e-6);
        assert!(ans_bimodal.confidence.unwrap() < 1e-6);
        assert_eq!(ans_bimodal.effective_confidence(), Some(0.0));

        // 3. 隣接段階分裂: レベル 1 とレベル 2 のみに二分。
        // 分散 Var = 0.25, 確信度 C_var = 1.0 - 0.25 / 2.25 ≈ 0.888889。
        let logits_adjacent = vec![-100.0, 100.0, 100.0, -100.0];
        let ans_adjacent = evaluate_score(&logits_adjacent, &criteria, 1.0).unwrap();
        assert!((ans_adjacent.score.unwrap() - 1.5).abs() < 1e-6);
        let conf_adj = ans_adjacent.confidence.unwrap();
        assert!((conf_adj - (8.0 / 9.0)).abs() < 1e-4);
        assert!(conf_adj > 0.85);
    }

    #[test]
    fn test_noul_extremes() {
        // 1. 真偽等確率(完全な迷い)。
        let ans_tie = evaluate_noul(&[2.5, 2.5], 1.0).unwrap();
        let prob_tie = ans_tie.noul.unwrap();
        assert!((prob_tie - 0.5).abs() < 1e-6);
        assert!(ans_tie.confidence.is_none());
        assert!((ans_tie.effective_confidence().unwrap() - 0.0).abs() < 1e-6);

        // 2. 真方向への絶対的確信。
        let ans_true = evaluate_noul(&[100.0, -100.0], 1.0).unwrap();
        assert!((ans_true.noul.unwrap() - 1.0).abs() < 1e-6);
        assert!((ans_true.effective_confidence().unwrap() - 1.0).abs() < 1e-6);

        // 3. 偽方向への絶対的確信。
        let ans_false = evaluate_noul(&[-100.0, 100.0], 1.0).unwrap();
        assert!((ans_false.noul.unwrap() - 0.0).abs() < 1e-6);
        assert!((ans_false.effective_confidence().unwrap() - 1.0).abs() < 1e-6);
    }
}

/// 候補数・段階数等の構造的境界値を検証するテストモジュール。
mod structural_bounds {
    use super::*;

    #[test]
    fn test_choice_single_candidate_singularity() {
        // Choice 特異点: K = 1。
        // ショートサーキットにより、指数計算および ln(1)=0 のゼロ除算をスキップして即座に 1.0 が返る。
        let mut map = IndexMap::new();
        map.insert("only".to_string(), "単一候補".to_string());
        let criteria = Criteria::Map(map);

        let logits = vec![42.0];
        let answer = evaluate_choice(&logits, &criteria, 1.0).unwrap();

        assert_eq!(answer.choice.as_deref(), Some("only"));
        assert_eq!(answer.confidence, Some(1.0));
        let probs = answer.probabilities.unwrap();
        assert_eq!(probs.len(), 1);
        assert_eq!(probs.get("only").copied(), Some(1.0));

        // K=1 特異点における非有限値 (NaN, ±Inf) のバリデーション。
        assert!(matches!(
            evaluate_choice(&[f64::NAN], &criteria, 1.0),
            Err(CoreError::MathError { .. })
        ));
        assert!(matches!(
            evaluate_choice(&[f64::INFINITY], &criteria, 1.0),
            Err(CoreError::MathError { .. })
        ));
        assert!(matches!(
            evaluate_choice(&[f64::NEG_INFINITY], &criteria, 1.0),
            Err(CoreError::MathError { .. })
        ));
    }

    #[test]
    fn test_choice_count_boundaries() {
        // K = 2 (最小分岐境界)
        let mut map2 = IndexMap::new();
        map2.insert("a".to_string(), "A".to_string());
        map2.insert("b".to_string(), "B".to_string());
        let ans2 = evaluate_choice(&[1.0, 2.0], &Criteria::Map(map2), 1.0);
        assert!(ans2.is_ok());

        // K = 255 (仕様上限境界)
        let mut map255 = IndexMap::new();
        let mut logits255 = Vec::with_capacity(255);
        for i in 0..255 {
            map255.insert(format!("opt_{i}"), format!("説明 {i}"));
            logits255.push(i as f64 * 0.01);
        }
        let ans255 = evaluate_choice(&logits255, &Criteria::Map(map255), 1.0).unwrap();
        assert_eq!(ans255.choice.as_deref(), Some("opt_254"));
        let sum: f64 = ans255.probabilities.unwrap().values().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // K = 0 (異常系)
        let empty_map = IndexMap::new();
        let err0 = evaluate_choice(&[], &Criteria::Map(empty_map), 1.0);
        assert!(matches!(
            err0,
            Err(CoreError::InvalidChoiceCount { count: 0, .. })
        ));

        // K = 256 (上限超過異常系)
        let mut map256 = IndexMap::new();
        let mut logits256 = Vec::with_capacity(256);
        for i in 0..256 {
            map256.insert(format!("k_{i}"), format!("V_{i}"));
            logits256.push(1.0);
        }
        let err256 = evaluate_choice(&logits256, &Criteria::Map(map256), 1.0);
        assert!(matches!(
            err256,
            Err(CoreError::InvalidChoiceCount { count: 256, .. })
        ));
    }

    #[test]
    fn test_score_level_boundaries() {
        // M = 2 (下限境界)
        let crit2 = Criteria::List(vec!["低".to_string(), "高".to_string()]);
        let ans2 = evaluate_score(&[1.0, 2.0], &crit2, 1.0);
        assert!(ans2.is_ok());

        // M = 10 (上限境界)
        let crit10 = Criteria::List((0..10).map(|i| format!("Lv{i}")).collect());
        let logits10 = vec![0.5; 10];
        let ans10 = evaluate_score(&logits10, &crit10, 1.0);
        assert!(ans10.is_ok());

        // M = 1 (異常系下限未満)
        let crit1 = Criteria::List(vec!["単一".to_string()]);
        let err1 = evaluate_score(&[1.0], &crit1, 1.0);
        assert!(matches!(
            err1,
            Err(CoreError::InvalidScoreLevelCount { count: 1, .. })
        ));

        // M = 11 (異常系上限超過)
        let crit11 = Criteria::List((0..11).map(|i| format!("Lv{i}")).collect());
        let logits11 = vec![0.5; 11];
        let err11 = evaluate_score(&logits11, &crit11, 1.0);
        assert!(matches!(
            err11,
            Err(CoreError::InvalidScoreLevelCount { count: 11, .. })
        ));
    }

    #[test]
    fn test_noul_length_boundaries() {
        // 長さ 2 (正常)
        assert!(evaluate_noul(&[1.0, 0.0], 1.0).is_ok());

        // 長さ 0 (異常系)
        assert!(matches!(
            evaluate_noul(&[], 1.0),
            Err(CoreError::MathError { .. })
        ));

        // 長さ 1 (異常系)
        assert!(matches!(
            evaluate_noul(&[1.0], 1.0),
            Err(CoreError::MathError { .. })
        ));
    }

    #[test]
    fn test_buffer_length_mismatch() {
        // 作業バッファ長とロジット長の不一致ガードを検証する。
        let mut map = IndexMap::new();
        map.insert("a".to_string(), "A".to_string());
        map.insert("b".to_string(), "B".to_string());
        let criteria_choice = Criteria::Map(map);

        let logits = vec![1.0, 2.0];
        let mut short_buf = vec![0.0; 1];
        let mut long_buf = vec![0.0; 3];

        assert!(matches!(
            evaluate_choice_with_buf(&logits, &criteria_choice, 1.0, &mut short_buf),
            Err(CoreError::MathError { .. })
        ));
        assert!(matches!(
            evaluate_choice_with_buf(&logits, &criteria_choice, 1.0, &mut long_buf),
            Err(CoreError::MathError { .. })
        ));

        let criteria_score = Criteria::List(vec!["A".to_string(), "B".to_string()]);
        assert!(matches!(
            evaluate_score_with_buf(&logits, &criteria_score, 1.0, &mut short_buf),
            Err(CoreError::MathError { .. })
        ));
        assert!(matches!(
            evaluate_score_with_buf(&logits, &criteria_score, 1.0, &mut long_buf),
            Err(CoreError::MathError { .. })
        ));
    }
}

/// 浮動小数点異常値・温度極限値に対する堅牢性を検証するテストモジュール。
mod non_finite_and_extreme_guards {
    use super::*;

    #[test]
    fn test_non_finite_logits_rejection() {
        let mut map = IndexMap::new();
        map.insert("a".to_string(), "A".to_string());
        map.insert("b".to_string(), "B".to_string());
        let crit_choice = Criteria::Map(map);
        let crit_score = Criteria::List(vec!["A".to_string(), "B".to_string()]);

        let bad_logits_cases = [
            [f64::NAN, 1.0],
            [1.0, f64::NAN],
            [f64::INFINITY, 1.0],
            [1.0, f64::INFINITY],
            [f64::NEG_INFINITY, 1.0],
            [1.0, f64::NEG_INFINITY],
        ];

        for (i, case) in bad_logits_cases.iter().enumerate() {
            assert!(
                matches!(
                    evaluate_choice(case, &crit_choice, 1.0),
                    Err(CoreError::MathError { .. })
                ),
                "Choice ケース {i} で非有限ロジットがすり抜けました。"
            );
            assert!(
                matches!(
                    evaluate_score(case, &crit_score, 1.0),
                    Err(CoreError::MathError { .. })
                ),
                "Score ケース {i} で非有限ロジットがすり抜けました。"
            );
            assert!(
                matches!(evaluate_noul(case, 1.0), Err(CoreError::MathError { .. })),
                "Noul ケース {i} で非有限ロジットがすり抜けました。"
            );
        }
    }

    #[test]
    fn test_temperature_extremes_and_guards() {
        let mut map = IndexMap::new();
        map.insert("a".to_string(), "A".to_string());
        map.insert("b".to_string(), "B".to_string());
        let criteria = Criteria::Map(map);
        let logits = vec![2.0, 1.0];

        // 異常な温度: 0.0, 負数, NaN, Inf はすべてエラーとなる。
        let bad_temps = [0.0, -1.0, -0.001, f64::NAN, f64::INFINITY];
        for (i, &temp) in bad_temps.iter().enumerate() {
            assert!(
                matches!(
                    evaluate_choice(&logits, &criteria, temp),
                    Err(CoreError::MathError { .. })
                ),
                "異常温度ケース {i} ({temp}) がすり抜けました。"
            );
        }

        // 極小温度: 1e-4 および 1e-9 (有効下限クランプによりオーバーフローせず最大値 One-hot 化する)。
        let ans_micro = evaluate_choice(&logits, &criteria, 1e-9).unwrap();
        assert_eq!(ans_micro.choice.as_deref(), Some("a"));
        assert!((ans_micro.confidence.unwrap() - 1.0).abs() < 1e-6);

        // 高温極限: 1000.0 (一様分布へ平滑化され確信度が極小化する)。
        let ans_high = evaluate_choice(&logits, &criteria, 1000.0).unwrap();
        assert!(ans_high.confidence.unwrap() < 1e-3);
    }
}

/// 決定論的タイブレークおよび API スキーマ整合性を検証するテストモジュール。
mod deterministic_and_contract {
    use super::*;

    #[test]
    fn test_deterministic_tie_breaking() {
        let mut map = IndexMap::new();
        map.insert("first".to_string(), "先頭候補".to_string());
        map.insert("second".to_string(), "2番目候補".to_string());
        map.insert("third".to_string(), "3番目候補".to_string());
        let criteria = Criteria::Map(map);

        // 1番目と2番目が同率最大ロジットの場合、確定的に先頭インデックスが採択される。
        let tied_logits = vec![5.0, 5.0, 1.0];
        for _ in 0..10 {
            let ans = evaluate_choice(&tied_logits, &criteria, 1.0).unwrap();
            assert_eq!(
                ans.choice.as_deref(),
                Some("first"),
                "同率タイブレークで先頭候補が採択されませんでした。"
            );
        }

        // 全候補が同率の場合でも先頭が確定的に採択される。
        let all_tied = vec![3.0, 3.0, 3.0];
        let ans_all = evaluate_choice(&all_tied, &criteria, 1.0).unwrap();
        assert_eq!(ans_all.choice.as_deref(), Some("first"));
    }

    #[test]
    fn test_score_duplicate_and_empty_label_fallback() {
        // 重複ラベルや空文字ラベルが存在する場合、キー衝突を防ぐためインデックス文字列へフォールバックされる。
        let logits = vec![1.0, 2.0, 3.0];

        // 1. 完全重複
        let crit_dup_all = Criteria::List(vec![
            "同じ".to_string(),
            "同じ".to_string(),
            "同じ".to_string(),
        ]);
        let ans_dup_all = evaluate_score(&logits, &crit_dup_all, 1.0).unwrap();
        let probs_all = ans_dup_all.probabilities.unwrap();
        assert_eq!(probs_all.len(), 3);
        assert!(probs_all.contains_key("0"));
        assert!(probs_all.contains_key("1"));
        assert!(probs_all.contains_key("2"));

        // 2. 部分重複
        let crit_dup_part = Criteria::List(vec![
            "Low".to_string(),
            "Mid".to_string(),
            "Low".to_string(),
        ]);
        let ans_dup_part = evaluate_score(&logits, &crit_dup_part, 1.0).unwrap();
        let probs_part = ans_dup_part.probabilities.unwrap();
        assert_eq!(probs_part.len(), 3);
        assert!(probs_part.contains_key("0"));
        assert!(probs_part.contains_key("1"));
        assert!(probs_part.contains_key("2"));

        // 3. 空文字混入
        let crit_empty =
            Criteria::List(vec!["".to_string(), "Mid".to_string(), "High".to_string()]);
        let ans_empty = evaluate_score(&logits, &crit_empty, 1.0).unwrap();
        let probs_empty = ans_empty.probabilities.unwrap();
        assert_eq!(probs_empty.len(), 3);
        assert!(probs_empty.contains_key("0"));
        assert!(probs_empty.contains_key("1"));
        assert!(probs_empty.contains_key("2"));

        // 4. 重複も空文字もない正常系では元のラベルがそのまま使用される。
        let crit_normal = Criteria::List(vec![
            "Low".to_string(),
            "Mid".to_string(),
            "High".to_string(),
        ]);
        let ans_normal = evaluate_score(&logits, &crit_normal, 1.0).unwrap();
        let probs_normal = ans_normal.probabilities.unwrap();
        assert_eq!(probs_normal.len(), 3);
        assert!(probs_normal.contains_key("Low"));
        assert!(probs_normal.contains_key("Mid"));
        assert!(probs_normal.contains_key("High"));
    }

    #[test]
    fn test_evaluate_question_dispatch_and_effective_confidence() {
        let config = CalibrationConfig::default();

        // 1. Choice
        let q_choice = Question::new_choice(
            "好きな色は？",
            IndexMap::from([
                ("red".to_string(), "赤".to_string()),
                ("blue".to_string(), "青".to_string()),
            ]),
        );
        let ans_choice = evaluate_question(&q_choice, &[10.0, 0.0], &config).unwrap();
        assert_eq!(ans_choice.choice.as_deref(), Some("red"));
        assert!(ans_choice.effective_confidence().unwrap() > 0.99);

        // 2. Score
        let q_score = Question::new_score(
            "満足度",
            vec!["不満".to_string(), "普通".to_string(), "満足".to_string()],
        );
        let ans_score = evaluate_question(&q_score, &[0.0, 0.0, 10.0], &config).unwrap();
        assert!((ans_score.score.unwrap() - 2.0).abs() < 1e-3);
        assert!(ans_score.effective_confidence().unwrap() > 0.99);

        // 3. Noul
        let q_noul = Question::new_noul("空は青いか？");
        let ans_noul = evaluate_question(&q_noul, &[5.0, 0.0], &config).unwrap();
        assert!(ans_noul.noul.unwrap() > 0.99);
        assert!(ans_noul.effective_confidence().unwrap() > 0.98);
    }
}
