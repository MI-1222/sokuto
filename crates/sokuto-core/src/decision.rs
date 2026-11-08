//! # 決定プリミティブ解決モジュール
//!
//! 推論エンジンが出力した生ロジット列とキャリブレーション設定から、
//! Jev 互換の判定結果構造体 (`Answer`) を決定論的に導出する。

use indexmap::IndexMap;

use crate::contract::CalibrationConfig;
use crate::error::{CoreError, Result};
use crate::math::{
    argmax, composite_confidence, expected_score, normalized_variance_confidence, noul_probability,
    softmax_into,
};
use crate::schema::{Answer, Criteria, Question, QuestionType};

/// 作業用バッファを活用し、Choice プリミティブの生ロジットから採択候補、確率分布、および複合確信度を算出する。
///
/// # 概要
/// - 呼び出し側から作業用バッファスライス (`probs_buf`) を受け取ることで、推論バッチ処理における
///   メモリヒープ確保を最小化する。
/// - 候補キーの順序付きマップ (`Criteria::Map`) とロジット列を 1 対 1 で結合する。
/// - 候補数 $K=1$ の特異点では、ショートサーキットにより確信度 1.0・確率 1.0 を即座に返却する。
/// - 最大確率を与える候補の採択には決定論的タイブレーク規則が適用され、同率最大時は先頭インデックスが採択される。
/// - 確信度には、候補数 $K$ に非依存な正規化エントロピーと Top-Margin を統合した
///   複合確信度スコア $S_{\text{confidence}} = (1.0 - H_{\text{norm}}) \times M(p)$ を採用する。
///
/// # 引数
/// - `logits`: 未正規化の生ロジットスライス(`f64`)。
/// - `criteria`: 候補識別子と説明文のマップを保持する評価基準。
/// - `temperature`: 適用する較正温度パラメータ $\tau$。
/// - `probs_buf`: 確率分布書き込み用の作業バッファスライス(`logits` と同長が必要)。
///
/// # 戻り値
/// - 成功時は Choice 判定が格納された `Answer`、入力不正時は `CoreError`。
pub fn evaluate_choice_with_buf(
    logits: &[f64],
    criteria: &Criteria,
    temperature: f64,
    probs_buf: &mut [f64],
) -> Result<Answer> {
    let map = match criteria {
        Criteria::Map(m) => m,
        Criteria::List(_) => {
            return Err(CoreError::InvalidCriteriaType {
                question_type: "choice".to_string(),
                expected: "Map (オブジェクト)",
                actual: "List (配列)",
            });
        }
        Criteria::None => {
            return Err(CoreError::MissingCriteria {
                question_type: "choice".to_string(),
            });
        }
    };

    let k = map.len();
    if k == 0 || k > 255 {
        return Err(CoreError::InvalidChoiceCount { count: k });
    }

    if logits.len() != k {
        return Err(CoreError::MathError {
            message: format!(
                "ロジットの要素数({logits_len})が候補数({k})と一致しません。",
                logits_len = logits.len()
            ),
        });
    }

    // 候補数 1 の特異点処理。
    if k == 1 {
        if !logits[0].is_finite() {
            return Err(CoreError::MathError {
                message: "ロジットに非有限値(NaNまたは無限大)が含まれています。".to_string(),
            });
        }
        let (first_key, _) = map
            .get_index(0)
            .expect("k == 1 なので必ず先頭要素が存在する。");
        let mut probabilities = IndexMap::with_capacity(1);
        probabilities.insert(first_key.clone(), 1.0);
        return Ok(Answer::choice(first_key.clone(), probabilities, 1.0));
    }

    // ゼロアロケーション版ソフトマックスの実行。
    softmax_into(logits, probs_buf, temperature)?;

    // 決定論的タイブレーク付き最大値インデックスの取得。
    let best_idx = argmax(probs_buf)?;
    let (selected_key, _) = map
        .get_index(best_idx)
        .expect("best_idx は 0..k の範囲内であることが保証されている。");

    // 候補数非依存の複合確信度スコア S_confidence = (1 - H_norm) * M(p) の算出。
    let confidence = composite_confidence(probs_buf)?;

    // 候補識別子と確率値のマッピング構築。
    let mut probabilities = IndexMap::with_capacity(k);
    for (idx, (key, _)) in map.iter().enumerate() {
        probabilities.insert(key.clone(), probs_buf[idx]);
    }

    Ok(Answer::choice(
        selected_key.clone(),
        probabilities,
        confidence,
    ))
}

/// Choice プリミティブの生ロジットから採択候補、確率分布、および正規化確信度を算出する。
///
/// # 概要
/// - 内部で必要なバッファを確保した上で [`evaluate_choice_with_buf`] を呼び出す。
pub fn evaluate_choice(logits: &[f64], criteria: &Criteria, temperature: f64) -> Result<Answer> {
    let mut probs_buf = vec![0.0; logits.len()];
    evaluate_choice_with_buf(logits, criteria, temperature, &mut probs_buf)
}

/// 作業用バッファを活用し、Score プリミティブの生ロジットから加重平均スコア実数値、確率分布、および分散確信度を算出する。
///
/// # 概要
/// - 呼び出し側から作業用バッファスライス (`probs_buf`) を受け取ることでヒープ確保を最小化する。
/// - 順序尺度評価配列 (`Criteria::List`) とロジット列を結合し、$M \in [2, 10]$ 段階の評価を行う。
/// - 加重平均期待値 $\hat{s} = \sum_{k=0}^{M-1} k \cdot p_k$ を連続値スコアとして算出する。
/// - 確信度には順序尺度のバイモーダル(両極端割れ)と低分散(隣接割れ)を適切に分離する
///   正規化分散確信度 $C_{\text{var}} = 1.0 - \frac{4 \cdot \text{Var}\[S\]}{(M-1)^2}$ を採用する。
/// - 確率マップのキーには、評価段階ラベルに重複がない場合はそのラベル名を用い、
///   重複または空文字が存在する場合はインデックス文字列 (`"0"`, `"1"`, ...) を用いる。
///   $M \le 10$ であるため、重複検査はヒープ確保を行わずにスタック上の二重走査で実施する。
///
/// # 引数
/// - `logits`: 未正規化の生ロジットスライス(`f64`)。
/// - `criteria`: 順序付けられた評価段階基準の配列。
/// - `temperature`: 適用する較正温度パラメータ $\tau$。
/// - `probs_buf`: 確率分布書き込み用の作業バッファスライス(`logits` と同長が必要)。
///
/// # 戻り値
/// - 成功時は Score 判定が格納された `Answer`、入力不正時は `CoreError`。
pub fn evaluate_score_with_buf(
    logits: &[f64],
    criteria: &Criteria,
    temperature: f64,
    probs_buf: &mut [f64],
) -> Result<Answer> {
    let list = match criteria {
        Criteria::List(l) => l,
        Criteria::Map(_) => {
            return Err(CoreError::InvalidCriteriaType {
                question_type: "score".to_string(),
                expected: "List (配列)",
                actual: "Map (オブジェクト)",
            });
        }
        Criteria::None => {
            return Err(CoreError::MissingCriteria {
                question_type: "score".to_string(),
            });
        }
    };

    let m = list.len();
    if !(2..=10).contains(&m) {
        return Err(CoreError::InvalidScoreLevelCount { count: m });
    }

    if logits.len() != m {
        return Err(CoreError::MathError {
            message: format!(
                "ロジットの要素数({logits_len})が段階数({m})と一致しません。",
                logits_len = logits.len()
            ),
        });
    }

    // ゼロアロケーション版ソフトマックス確率分布の算出。
    softmax_into(logits, probs_buf, temperature)?;

    // 加重平均連続値スコアの算出。
    let score_val = expected_score(probs_buf)?;

    // 順序尺度分散確信度の算出。
    let confidence = normalized_variance_confidence(probs_buf)?;

    // 確率マップキーの一意性判定(M <= 10 のため HashSet を使わず二重走査でアロケーションゼロ判定)。
    let is_unique_and_non_empty = list
        .iter()
        .enumerate()
        .all(|(i, label)| !label.is_empty() && list[i + 1..].iter().all(|other| label != other));

    let mut probabilities = IndexMap::with_capacity(m);
    for (k, &prob) in probs_buf.iter().enumerate() {
        let key = if is_unique_and_non_empty {
            list[k].clone()
        } else {
            k.to_string()
        };
        probabilities.insert(key, prob);
    }

    Ok(Answer::score(score_val, probabilities, confidence))
}

/// Score プリミティブの生ロジットから加重平均連続値スコア、確率分布、および分散確信度を算出する。
///
/// # 概要
/// - 内部で必要なバッファを確保した上で [`evaluate_score_with_buf`] を呼び出す。
pub fn evaluate_score(logits: &[f64], criteria: &Criteria, temperature: f64) -> Result<Answer> {
    let mut probs_buf = vec![0.0; logits.len()];
    evaluate_score_with_buf(logits, criteria, temperature, &mut probs_buf)
}

/// Noul プリミティブの生ロジットから真偽確率判定結果を算出する。
///
/// # 概要
/// - 入力ロジットの先頭 2 要素 $(z_{\text{true}}, z_{\text{false}})$ の差分から、
///   数値安定シグモイド関数により言明が真である確率 $P(\text{true}) \in [0.0, 1.0]$ を直接算出する。
/// - Jev 公式 API 契約に従い、Noul 型の判定結果では `choice`, `score`, `probabilities`, `confidence`
///   をすべて `None` とし、`noul: Some(P(true))` のみを格納した `Answer` を返却する。
///
/// # 引数
/// - `logits`: 生ロジットスライス(`f64`)。先頭 2 要素がそれぞれ真・偽に対応する。
/// - `temperature`: 適用する較正温度パラメータ $\tau$。
///
/// # 戻り値
/// - 成功時は Noul 判定が格納された `Answer`、要素数不足等の場合は `CoreError`。
pub fn evaluate_noul(logits: &[f64], temperature: f64) -> Result<Answer> {
    if logits.len() < 2 {
        return Err(CoreError::MathError {
            message: format!(
                "Noul 判定には真・偽に対応する 2 要素以上のロジットが必要です(指定数: {})。",
                logits.len()
            ),
        });
    }

    let prob_true = noul_probability(logits[0], logits[1], temperature)?;
    Ok(Answer::noul(prob_true))
}

/// 作業用バッファを活用し、質問定義、生ロジット、および較正設定から型安全に判定結果 (`Answer`) を導出する。
pub fn evaluate_question_with_buf(
    question: &Question,
    logits: &[f64],
    config: &CalibrationConfig,
    probs_buf: &mut [f64],
) -> Result<Answer> {
    // 質問定義のスキーマ検証を実施する。
    question.validate("evaluate_question")?;

    match question.question_type {
        QuestionType::Choice => {
            let criteria =
                question
                    .criteria
                    .as_ref()
                    .ok_or_else(|| CoreError::MissingCriteria {
                        question_type: "choice".to_string(),
                    })?;
            let k = criteria.len();
            let temperature = config.get_temperature(QuestionType::Choice, k);
            evaluate_choice_with_buf(logits, criteria, temperature, probs_buf)
        }
        QuestionType::Score => {
            let criteria =
                question
                    .criteria
                    .as_ref()
                    .ok_or_else(|| CoreError::MissingCriteria {
                        question_type: "score".to_string(),
                    })?;
            let m = criteria.len();
            let temperature = config.get_temperature(QuestionType::Score, m);
            evaluate_score_with_buf(logits, criteria, temperature, probs_buf)
        }
        QuestionType::Noul => {
            let temperature = config.get_temperature(QuestionType::Noul, 2);
            evaluate_noul(logits, temperature)
        }
    }
}

/// 単一の質問定義、生ロジット、および較正設定から型安全に判定結果 (`Answer`) を導出する。
///
/// # 概要
/// - 質問タイプ (`QuestionType`) および候補数・段階数に基づいて `CalibrationConfig` から
///   最適な較正温度 $\tau^*$ を自動解決する。
/// - 質問プリミティブに応じた評価関数へディスパッチする。
///
/// # 引数
/// - `question`: 評価対象の質問定義。
/// - `logits`: 推論エンジンから出力された生ロジットスライス。
/// - `config`: 最適温度マップを保持するキャリブレーション設定。
///
/// # 戻り値
/// - 成功時は算出された判定結果 `Answer`、異常時は `CoreError`。
pub fn evaluate_question(
    question: &Question,
    logits: &[f64],
    config: &CalibrationConfig,
) -> Result<Answer> {
    let mut probs_buf = vec![0.0; logits.len()];
    evaluate_question_with_buf(question, logits, config, &mut probs_buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_choice_basic() {
        let mut criteria_map = IndexMap::new();
        criteria_map.insert("cat".to_string(), "猫".to_string());
        criteria_map.insert("dog".to_string(), "犬".to_string());
        criteria_map.insert("bird".to_string(), "鳥".to_string());
        let criteria = Criteria::Map(criteria_map);

        let logits = vec![1.0, 5.0, 2.0];
        let answer = evaluate_choice(&logits, &criteria, 1.0).unwrap();

        assert_eq!(answer.choice.as_deref(), Some("dog"));
        assert!(answer.score.is_none());
        assert!(answer.noul.is_none());

        let probs = answer.probabilities.unwrap();
        assert_eq!(probs.len(), 3);
        assert!(probs["dog"] > probs["bird"]);
        assert!(probs["bird"] > probs["cat"]);

        let conf = answer.confidence.unwrap();
        assert!(conf > 0.0 && conf <= 1.0);
    }

    #[test]
    fn test_evaluate_choice_single_candidate() {
        let mut criteria_map = IndexMap::new();
        criteria_map.insert("only_one".to_string(), "唯一の候補".to_string());
        let criteria = Criteria::Map(criteria_map);

        let logits = vec![std::f64::consts::PI];
        let answer = evaluate_choice(&logits, &criteria, 1.0).unwrap();

        assert_eq!(answer.choice.as_deref(), Some("only_one"));
        assert_eq!(answer.confidence, Some(1.0));
        let probs = answer.probabilities.unwrap();
        assert_eq!(probs["only_one"], 1.0);
    }

    #[test]
    fn test_evaluate_choice_tied_logits() {
        // 同率最大値の場合、先頭候補が確定的に選ばれるタイブレークの検証。
        let mut criteria_map = IndexMap::new();
        criteria_map.insert("opt_a".to_string(), "候補A".to_string());
        criteria_map.insert("opt_b".to_string(), "候補B".to_string());
        criteria_map.insert("opt_c".to_string(), "候補C".to_string());
        let criteria = Criteria::Map(criteria_map);

        let logits = vec![4.0, 4.0, 1.0];
        let answer = evaluate_choice(&logits, &criteria, 1.0).unwrap();

        assert_eq!(answer.choice.as_deref(), Some("opt_a"));
    }

    #[test]
    fn test_evaluate_choice_errors() {
        let mut criteria_map = IndexMap::new();
        criteria_map.insert("a".to_string(), "A".to_string());
        let criteria = Criteria::Map(criteria_map);

        // ロジット数と候補数の不一致
        let logits = vec![1.0, 2.0];
        assert!(evaluate_choice(&logits, &criteria, 1.0).is_err());

        // 不正な criteria 種別 (List や None)
        assert!(evaluate_choice(&logits, &Criteria::List(vec!["A".to_string()]), 1.0).is_err());
        assert!(evaluate_choice(&logits, &Criteria::None, 1.0).is_err());

        // K=1 特異点における非有限値(NaN, Inf, -Inf)のエラー検出。
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
    fn test_evaluate_score_basic() {
        let criteria = Criteria::List(vec!["低".to_string(), "中".to_string(), "高".to_string()]);

        // レベル 2 (高) に高いロジット
        let logits = vec![0.0, 1.0, 4.0];
        let answer = evaluate_score(&logits, &criteria, 1.0).unwrap();

        assert!(answer.choice.is_none());
        assert!(answer.noul.is_none());

        let score = answer.score.unwrap();
        assert!(score > 1.5 && score <= 2.0);

        let probs = answer.probabilities.unwrap();
        assert_eq!(probs.len(), 3);
        assert!(probs["高"] > probs["中"]);

        let conf = answer.confidence.unwrap();
        assert!(conf > 0.0 && conf <= 1.0);
    }

    #[test]
    fn test_evaluate_score_duplicate_labels() {
        // ラベルに重複がある場合はインデックス文字列 ("0", "1", "2") がキーとなる検証。
        let criteria = Criteria::List(vec![
            "同じ".to_string(),
            "同じ".to_string(),
            "同じ".to_string(),
        ]);

        let logits = vec![1.0, 2.0, 3.0];
        let answer = evaluate_score(&logits, &criteria, 1.0).unwrap();
        let probs = answer.probabilities.unwrap();

        assert_eq!(probs.len(), 3);
        assert!(probs.contains_key("0"));
        assert!(probs.contains_key("1"));
        assert!(probs.contains_key("2"));
    }

    #[test]
    fn test_evaluate_score_errors() {
        // 段階数 2 未満
        let criteria_small = Criteria::List(vec!["単一".to_string()]);
        assert!(evaluate_score(&[1.0], &criteria_small, 1.0).is_err());

        // 段階数 10 超過
        let criteria_large = Criteria::List((0..11).map(|i| i.to_string()).collect());
        let logits_11 = vec![0.0; 11];
        assert!(evaluate_score(&logits_11, &criteria_large, 1.0).is_err());

        // 長さ不一致
        let criteria_3 = Criteria::List(vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        assert!(evaluate_score(&[1.0, 2.0], &criteria_3, 1.0).is_err());
    }

    #[test]
    fn test_evaluate_noul_basic() {
        let logits = vec![3.0, 0.0];
        let answer = evaluate_noul(&logits, 1.0).unwrap();

        assert!(answer.choice.is_none());
        assert!(answer.score.is_none());
        assert!(answer.probabilities.is_none());
        assert!(answer.confidence.is_none());

        let prob = answer.noul.unwrap();
        assert!(prob > 0.90 && prob <= 1.0);

        // 異常系: ロジット数が 2 未満
        assert!(evaluate_noul(&[1.0], 1.0).is_err());
    }

    #[test]
    fn test_evaluate_question_dispatch() {
        let config = CalibrationConfig::default();

        // 1. Choice のディスパッチ検証
        let mut c_map = IndexMap::new();
        c_map.insert("yes".to_string(), "はい".to_string());
        c_map.insert("no".to_string(), "いいえ".to_string());
        let q_choice = Question::new_choice("選択肢を選べ。", c_map);
        let ans_choice = evaluate_question(&q_choice, &[2.0, 0.5], &config).unwrap();
        assert_eq!(ans_choice.choice.as_deref(), Some("yes"));

        // 2. Score のディスパッチ検証
        let q_score = Question::new_score(
            "評価せよ。",
            vec!["1".to_string(), "2".to_string(), "3".to_string()],
        );
        let ans_score = evaluate_question(&q_score, &[0.0, 2.0, 4.0], &config).unwrap();
        assert!(ans_score.score.unwrap() > 1.0);

        // 3. Noul のディスパッチ検証
        let q_noul = Question::new_noul("真偽を判定せよ。");
        let ans_noul = evaluate_question(&q_noul, &[2.5, 0.0], &config).unwrap();
        assert!(ans_noul.noul.unwrap() > 0.85);
    }
}
