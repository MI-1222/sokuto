//! # 決定プリミティブ解決モジュール
//!
//! 推論エンジンが出力した生ロジット列とキャリブレーション設定から、
//! Jev 互換の判定結果構造体 (`Answer`) を決定論的に導出する。

use indexmap::IndexMap;

use crate::contract::CalibrationConfig;
use crate::error::{CoreError, Result};
use crate::math::{
    composite_confidence_and_argmax, normalized_variance_confidence_from_variance,
    noul_probability_with_bias, score_moments_iter, softmax_into,
};
use crate::schema::{Answer, Criteria, Question, QuestionType};

/// 失敗結果に対して質問識別子 (`question_id`) が指定されている場合にエラーコンテキストを付与する。
fn attach_question_id<T>(res: Result<T>, question_id: Option<&str>) -> Result<T> {
    match (res, question_id) {
        (Err(err), Some(qid)) => Err(err.with_question_id(qid)),
        (other, _) => other,
    }
}

/// 検証済みの候補マップ (`map`) を用いて Choice プリミティブの生ロジットから判定結果を算出する内部ヘルパー。
fn evaluate_choice_map_with_buf(
    logits: &[f64],
    map: &IndexMap<String, String>,
    temperature: f64,
    probs_buf: &mut [f64],
) -> Result<Answer> {
    let k = map.len();
    if logits.len() != k {
        return Err(CoreError::MathError {
            question_id: None,
            message: format!(
                "ロジットの要素数({logits_len})が候補数({k})と一致しません。",
                logits_len = logits.len()
            ),
        });
    }

    // ゼロアロケーション版ソフトマックスの実行(出力バッファ長検証を含む)。
    softmax_into(logits, probs_buf, temperature)?;

    // 決定論的タイブレーク付き最大値インデックスおよび複合確信度スコアの単一パス算出。
    let (confidence, best_idx) = composite_confidence_and_argmax(probs_buf)?;
    let (selected_key, _) = map
        .get_index(best_idx)
        .expect("best_idx は 0..k の範囲内であることが保証されている。");

    // 候補識別子と確率値のマッピング構築。
    let probabilities = map.keys().cloned().zip(probs_buf.iter().copied()).collect();

    Ok(Answer::choice(
        selected_key.clone(),
        probabilities,
        confidence,
    ))
}

/// 作業用バッファを活用し、Choice プリミティブの生ロジットから採択候補、確率分布、および複合確信度を算出する。
///
/// # 概要
/// - 呼び出し側から作業用バッファスライス (`probs_buf`) を受け取ることで、推論バッチ処理における
///   メモリヒープ確保を最小化する。
/// - 候補キーの順序付きマップ (`Criteria::Map`) とロジット列を 1 対 1 で結合する。
/// - 候補数 $K=1$ の特異点では、[`softmax_into`] および [`composite_confidence_and_argmax`] の特異点処理により確率 1.0・確信度 1.0 が算出される。
/// - 最大確率を与える候補の採択には決定論的タイブレーク規則が適用され、同率最大時は先頭インデックスが採択される。
/// - 確信度には、候補数 $K$ に非依存な正規化エントロピーと Top-Margin を統合した
///   複合確信度スコア $S_{\text{confidence}} = (1.0 - H_{\text{norm}}) \times M(p)$ を採用する。
///
/// # 引数
/// - `logits`: 未正規化の生ロジットスライス(`f64`)。
/// - `criteria`: 候補識別子と説明文のマップを保持する評価基準。
/// - `temperature`: 適用する較正温度パラメータ $\tau$。
/// - `probs_buf`: 確率分布書き込み用の作業バッファスライス(`logits` と同長が必要)。
/// - `question_id`: 質問識別子 (任意)。エラー発生時に診断メッセージへ反映される。
///
/// # 戻り値
/// - 成功時は Choice 判定が格納された `Answer`、入力不正時は `CoreError`。
pub fn evaluate_choice_with_buf(
    logits: &[f64],
    criteria: &Criteria,
    temperature: f64,
    probs_buf: &mut [f64],
    question_id: Option<&str>,
) -> Result<Answer> {
    let res = criteria
        .validate_choice_map(None)
        .and_then(|map| evaluate_choice_map_with_buf(logits, map, temperature, probs_buf));
    attach_question_id(res, question_id)
}

/// Choice プリミティブの生ロジットから採択候補、確率分布、および正規化確信度を算出する。
///
/// # 概要
/// - 内部で必要なバッファを確保した上で [`evaluate_choice_with_buf`] を呼び出す。
pub fn evaluate_choice(logits: &[f64], criteria: &Criteria, temperature: f64) -> Result<Answer> {
    let mut probs_buf = vec![0.0; logits.len()];
    evaluate_choice_with_buf(logits, criteria, temperature, &mut probs_buf, None)
}

/// 検証済みの段階配列 (`list`) を用いて Score プリミティブの生ロジットから判定結果を算出する内部ヘルパー。
fn evaluate_score_list_with_buf(
    logits: &[f64],
    list: &[String],
    temperature: f64,
    probs_buf: &mut [f64],
) -> Result<Answer> {
    let m = list.len();
    if logits.len() != m {
        return Err(CoreError::MathError {
            question_id: None,
            message: format!(
                "ロジットの要素数({logits_len})が段階数({m})と一致しません。",
                logits_len = logits.len()
            ),
        });
    }

    // ゼロアロケーション版ソフトマックス確率分布の算出(出力バッファ長検証を含む)。
    softmax_into(logits, probs_buf, temperature)?;

    // 単一パス走査による加重平均期待値(スコア)および事後分散の算出。
    let (m_observed, score_val, var) = score_moments_iter(probs_buf.iter())?;

    // 順序尺度分散確信度の算出。
    let confidence = normalized_variance_confidence_from_variance(m_observed, var);

    // 確率マップキーの一意性判定(M <= 10 のため HashSet を使わず二重走査でアロケーションゼロ判定)。
    let is_unique_and_non_empty = list
        .iter()
        .enumerate()
        .all(|(i, label)| !label.is_empty() && list[i + 1..].iter().all(|other| label != other));

    let probabilities = if is_unique_and_non_empty {
        list.iter()
            .cloned()
            .zip(probs_buf.iter().copied())
            .collect()
    } else {
        probs_buf
            .iter()
            .copied()
            .enumerate()
            .map(|(k, prob)| (k.to_string(), prob))
            .collect()
    };

    Ok(Answer::score(score_val, probabilities, confidence))
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
/// - 同一確率バッファに対する加重平均期待値と事後分散の算出は [`score_moments_iter`] による単一パス走査に集約し、二重走査を排除する。
///
/// # 引数
/// - `logits`: 未正規化の生ロジットスライス(`f64`)。
/// - `criteria`: 順序付けられた評価段階基準の配列。
/// - `temperature`: 適用する較正温度パラメータ $\tau$。
/// - `probs_buf`: 確率分布書き込み用の作業バッファスライス(`logits` と同長が必要)。
/// - `question_id`: 質問識別子 (任意)。エラー発生時に診断メッセージへ反映される。
///
/// # 戻り値
/// - 成功時は Score 判定が格納された `Answer`、入力不正時は `CoreError`。
pub fn evaluate_score_with_buf(
    logits: &[f64],
    criteria: &Criteria,
    temperature: f64,
    probs_buf: &mut [f64],
    question_id: Option<&str>,
) -> Result<Answer> {
    let res = criteria
        .validate_score_list(None)
        .and_then(|list| evaluate_score_list_with_buf(logits, list, temperature, probs_buf));
    attach_question_id(res, question_id)
}

/// Score プリミティブの生ロジットから加重平均連続値スコア、確率分布、および分散確信度を算出する。
///
/// # 概要
/// - 内部で必要なバッファを確保した上で [`evaluate_score_with_buf`] を呼び出す。
pub fn evaluate_score(logits: &[f64], criteria: &Criteria, temperature: f64) -> Result<Answer> {
    let mut probs_buf = vec![0.0; logits.len()];
    evaluate_score_with_buf(logits, criteria, temperature, &mut probs_buf, None)
}

/// Noul プリミティブの生ロジットからバイアス補正項付きで真偽確率判定結果を算出する。
///
/// # 概要
/// - 入力ロジットの 2 要素 $(z_{\text{true}}, z_{\text{false}})$ の差分から、バイアス補正項 `bias` を差し引いた上で、
///   数値安定シグモイド関数により言明が真である確率 $P(\text{true}) \in [0.0, 1.0]$ を直接算出する。
/// - Jev 公式 API 契約に従い、Noul 型の判定結果では `choice`, `score`, `probabilities`, `confidence`
///   をすべて `None` とし、`noul: Some(P(true))` のみを格納した `Answer` を返却する。
/// - Choice や Score と同様に、入力ロジット要素数が 2 と一致しない場合は `CoreError::MathError` として拒否する。
///
/// # 引数
/// - `logits`: 生ロジットスライス(`f64`)。真・偽に対応する厳密に 2 要素が必要。
/// - `temperature`: 適用する較正温度パラメータ $\tau$。
/// - `bias`: 肯定過信バイアス補正オフセット $b$。有限実数。
///
/// # 戻り値
/// - 成功時は Noul 判定が格納された `Answer`、要素数不一致等の場合は `CoreError`。
pub fn evaluate_noul_with_bias(logits: &[f64], temperature: f64, bias: f64) -> Result<Answer> {
    if logits.len() != 2 {
        return Err(CoreError::MathError {
            question_id: None,
            message: format!(
                "Noul 判定には真・偽に対応する厳密に 2 要素のロジットが必要です(指定数: {})。",
                logits.len()
            ),
        });
    }

    let prob_true = noul_probability_with_bias(logits[0], logits[1], temperature, bias)?;
    Ok(Answer::noul(prob_true))
}

/// Noul プリミティブの生ロジットから真偽確率判定結果を算出する。
///
/// # 概要
/// - 内部でバイアス補正 `0.0` を指定して [`evaluate_noul_with_bias`] を呼び出す。
#[inline]
pub fn evaluate_noul(logits: &[f64], temperature: f64) -> Result<Answer> {
    evaluate_noul_with_bias(logits, temperature, 0.0)
}

/// 作業用バッファを活用し、質問定義、生ロジット、および較正設定から型安全に判定結果 (`Answer`) を導出する。
///
/// # 引数
/// - `question`: 評価対象の質問定義。
/// - `logits`: 推論エンジンから出力された生ロジットスライス。
/// - `config`: 最適温度マップを保持するキャリブレーション設定。
/// - `probs_buf`: 確率値書き込み用の作業バッファスライス (`logits` と同長が必要)。
/// - `question_id`: 質問識別子 (任意)。エラー発生時に診断メッセージへ反映される。
pub fn evaluate_question_with_buf(
    question: &Question,
    logits: &[f64],
    config: &CalibrationConfig,
    probs_buf: &mut [f64],
    question_id: Option<&str>,
) -> Result<Answer> {
    let res = match question.question_type {
        QuestionType::Choice => question.validate_choice_map(None).and_then(|map| {
            let temperature = config.get_temperature(QuestionType::Choice, map.len());
            evaluate_choice_map_with_buf(logits, map, temperature, probs_buf)
        }),
        QuestionType::Score => question.validate_score_list(None).and_then(|list| {
            let temperature = config.get_temperature(QuestionType::Score, list.len());
            evaluate_score_list_with_buf(logits, list, temperature, probs_buf)
        }),
        QuestionType::Noul => {
            let temperature = config.get_temperature(QuestionType::Noul, 2);
            evaluate_noul_with_bias(logits, temperature, config.get_noul_bias())
        }
    };

    attach_question_id(res, question_id)
}

/// 単一の質問定義、生ロジット、および較正設定から型安全に判定結果 (`Answer`) を導出する。
///
/// # 概要
/// - 質問タイプ (`QuestionType`) および候補数・段階数に基づいて `CalibrationConfig` から
///   最適な較正温度 $\tau^*$ を自動解決する。
/// - 質問プリミティブに応じた評価関数へディスパッチする。
/// - エラー診断時に質問識別子を付与したい場合は [`evaluate_question_with_id`] を使用する。
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
    evaluate_question_with_id(None, question, logits, config)
}

/// 質問識別子を指定し、質問定義、生ロジット、および較正設定から型安全に判定結果 (`Answer`) を導出する。
///
/// エラー発生時に指定された `question_id` がエラーコンテキストへ反映される。
pub fn evaluate_question_with_id(
    question_id: Option<&str>,
    question: &Question,
    logits: &[f64],
    config: &CalibrationConfig,
) -> Result<Answer> {
    let mut probs_buf = vec![0.0; logits.len()];
    evaluate_question_with_buf(question, logits, config, &mut probs_buf, question_id)
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

        assert_eq!(answer.as_choice(), Some("dog"));
        assert!(answer.as_score().is_none());
        assert!(answer.as_noul().is_none());
        assert!(answer.is_choice());

        let probs = answer.probabilities().unwrap();
        assert_eq!(probs.len(), 3);
        assert!(probs["dog"] > probs["bird"]);
        assert!(probs["bird"] > probs["cat"]);

        let conf = answer.confidence().unwrap();
        assert!(conf > 0.0 && conf <= 1.0);
    }

    #[test]
    fn test_evaluate_choice_single_candidate() {
        let mut criteria_map = IndexMap::new();
        criteria_map.insert("only_one".to_string(), "唯一の候補".to_string());
        let criteria = Criteria::Map(criteria_map);

        let logits = vec![std::f64::consts::PI];
        let answer = evaluate_choice(&logits, &criteria, 1.0).unwrap();

        assert_eq!(answer.as_choice(), Some("only_one"));
        assert_eq!(answer.confidence(), Some(1.0));
        let probs = answer.probabilities().unwrap();
        assert_eq!(probs["only_one"], 1.0);

        // K=1 特異点において作業バッファ probs_buf が 1.0 に更新されることを検証する。
        let mut probs_buf = vec![0.0];
        let answer_buf =
            evaluate_choice_with_buf(&logits, &criteria, 1.0, &mut probs_buf, None).unwrap();
        assert_eq!(answer_buf.as_choice(), Some("only_one"));
        assert_eq!(probs_buf[0], 1.0);

        // バッファ長不一致の検証。
        let mut empty_buf = vec![];
        assert!(evaluate_choice_with_buf(&logits, &criteria, 1.0, &mut empty_buf, None).is_err());
        let mut long_buf = vec![0.0, 0.0];
        assert!(evaluate_choice_with_buf(&logits, &criteria, 1.0, &mut long_buf, None).is_err());

        // 温度異常の検証。
        let mut buf = vec![0.0];
        assert!(evaluate_choice_with_buf(&logits, &criteria, 0.0, &mut buf, None).is_err());
        assert!(evaluate_choice_with_buf(&logits, &criteria, -1.0, &mut buf, None).is_err());
        assert!(evaluate_choice_with_buf(&logits, &criteria, f64::NAN, &mut buf, None).is_err());
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

        assert_eq!(answer.as_choice(), Some("opt_a"));
    }

    #[test]
    fn test_evaluate_choice_errors() {
        let mut criteria_map = IndexMap::new();
        criteria_map.insert("a".to_string(), "A".to_string());
        let criteria = Criteria::Map(criteria_map);

        // ロジット数と候補数の不一致
        let logits = vec![1.0, 2.0];
        assert!(evaluate_choice(&logits, &criteria, 1.0).is_err());

        // 不正な criteria 種別 (List)
        assert!(evaluate_choice(&logits, &Criteria::List(vec!["A".to_string()]), 1.0).is_err());

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

        assert!(answer.as_choice().is_none());
        assert!(answer.as_noul().is_none());
        assert!(answer.is_score());

        let score = answer.as_score().unwrap();
        assert!(score > 1.5 && score <= 2.0);

        let probs = answer.probabilities().unwrap();
        assert_eq!(probs.len(), 3);
        assert!(probs["高"] > probs["中"]);

        let conf = answer.confidence().unwrap();
        assert!(conf > 0.0 && conf <= 1.0);
    }

    #[test]
    fn test_evaluate_score_duplicate_labels() {
        // ラベルに重複がある場合は DuplicateCriteriaLabel エラーとなる検証。
        let criteria = Criteria::List(vec![
            "同じ".to_string(),
            "同じ".to_string(),
            "同じ".to_string(),
        ]);

        let logits = vec![1.0, 2.0, 3.0];
        let err = evaluate_score(&logits, &criteria, 1.0).unwrap_err();
        assert!(matches!(
            err,
            CoreError::DuplicateCriteriaLabel { label, .. } if label == "同じ"
        ));
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

        assert!(answer.as_choice().is_none());
        assert!(answer.as_score().is_none());
        assert!(answer.probabilities().is_none());
        assert!(answer.confidence().is_none());
        assert!(answer.is_noul());

        let prob = answer.as_noul().unwrap();
        assert!(prob > 0.90 && prob <= 1.0);

        // evaluate_noul_with_bias の検証
        let answer_zero_bias = evaluate_noul_with_bias(&logits, 1.0, 0.0).unwrap();
        assert_eq!(answer.as_noul(), answer_zero_bias.as_noul());

        let answer_biased = evaluate_noul_with_bias(&logits, 1.0, 1.0).unwrap();
        assert!(answer_biased.as_noul().unwrap() < prob);

        // 異常系: ロジット数が 2 以外 (不足・過剰)
        assert!(matches!(
            evaluate_noul(&[], 1.0),
            Err(CoreError::MathError { .. })
        ));
        assert!(matches!(
            evaluate_noul(&[1.0], 1.0),
            Err(CoreError::MathError { .. })
        ));
        assert!(matches!(
            evaluate_noul(&[1.0, 0.0, 0.5], 1.0),
            Err(CoreError::MathError { .. })
        ));
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
        assert_eq!(ans_choice.as_choice(), Some("yes"));

        // 2. Score のディスパッチ検証
        let q_score = Question::new_score(
            "評価せよ。",
            vec!["1".to_string(), "2".to_string(), "3".to_string()],
        );
        let ans_score = evaluate_question(&q_score, &[0.0, 2.0, 4.0], &config).unwrap();
        assert!(ans_score.as_score().unwrap() > 1.0);

        // 3. Noul のディスパッチ検証
        let q_noul = Question::new_noul("真偽を判定せよ。");
        let ans_noul = evaluate_question(&q_noul, &[2.5, 0.0], &config).unwrap();
        assert!(ans_noul.as_noul().unwrap() > 0.85);

        // 4. Noul でバイアス補正が設定されている場合の検証
        let mut custom_calib = CalibrationConfig::default();
        custom_calib.temperature_map.noul_bias = 2.0;
        let ans_noul_biased = evaluate_question(&q_noul, &[2.5, 0.0], &custom_calib).unwrap();
        // バイアス 2.0 が引かれるため、実効ロジット差は 2.5 - 2.0 = 0.5 となり確率は低下する
        assert!(ans_noul_biased.as_noul().unwrap() < ans_noul.as_noul().unwrap());
    }

    #[test]
    fn test_evaluate_question_error_preserves_question_id() {
        let config = CalibrationConfig::default();

        // 候補数 0 の不正な Choice 質問定義を作成
        let invalid_q = Question {
            question_type: QuestionType::Choice,
            instructions: "無効な指示".to_string(),
            criteria: Some(Criteria::Map(IndexMap::new())),
        };

        // 1. question_id を指定した場合: エラーに指定した question_id が保持されること
        let err_with_id =
            evaluate_question_with_id(Some("intent"), &invalid_q, &[1.0], &config).unwrap_err();
        match &err_with_id {
            CoreError::InvalidChoiceCount { question_id, count } => {
                assert_eq!(question_id.as_deref(), Some("intent"));
                assert_eq!(*count, 0);
            }
            other => panic!("予期しないエラー型です: {other:?}"),
        }
        let err_msg = err_with_id.to_string();
        assert!(
            err_msg.contains("質問 'intent': "),
            "エラーメッセージに質問識別子が含まれていること: {err_msg}"
        );
        assert!(
            !err_msg.contains("evaluate_question"),
            "旧ダミープレフィックスが含まれていないこと: {err_msg}"
        );

        // 2. question_id が None の場合: question_id は None となり余分なプレフィックスが付与されないこと
        let err_no_id = evaluate_question(&invalid_q, &[1.0], &config).unwrap_err();
        match &err_no_id {
            CoreError::InvalidChoiceCount { question_id, count } => {
                assert_eq!(question_id, &None);
                assert_eq!(*count, 0);
            }
            other => panic!("予期しないエラー型です: {other:?}"),
        }
        let err_no_id_msg = err_no_id.to_string();
        assert!(
            !err_no_id_msg.contains("evaluate_question"),
            "旧ダミープレフィックスが含まれていないこと: {err_no_id_msg}"
        );
        assert!(
            !err_no_id_msg.starts_with("質問 '"),
            "識別子なしの場合は質問プレフィックスが付与されないこと: {err_no_id_msg}"
        );

        // 3. evaluate_question_with_buf でも question_id が正しく伝搬すること
        let mut buf = [0.0; 2];
        let err_buf =
            evaluate_question_with_buf(&invalid_q, &[1.0], &config, &mut buf, Some("urgency"))
                .unwrap_err();
        assert!(err_buf.to_string().contains("質問 'urgency': "));

        // 4. ロジット数不一致時の MathError にも question_id が伝播すること
        let mut valid_c_map = IndexMap::new();
        valid_c_map.insert("a".to_string(), "A".to_string());
        valid_c_map.insert("b".to_string(), "B".to_string());
        let valid_q = Question::new_choice("選択せよ。", valid_c_map);
        let mut mismatch_buf = [0.0; 3];
        let err_logits_mismatch = evaluate_question_with_buf(
            &valid_q,
            &[1.0, 2.0, 3.0],
            &config,
            &mut mismatch_buf,
            Some("category"),
        )
        .unwrap_err();
        match &err_logits_mismatch {
            CoreError::MathError {
                question_id,
                message,
            } => {
                assert_eq!(question_id.as_deref(), Some("category"));
                assert!(message.contains("ロジットの要素数(3)が候補数(2)と一致しません"));
                assert!(!message.starts_with("質問 '"));
            }
            other => panic!("MathError が返る必要があります: {other:?}"),
        }
        let err_logits_msg = err_logits_mismatch.to_string();
        assert!(
            err_logits_msg.starts_with("質問 'category': 数値計算エラーが発生しました: "),
            "エラーメッセージの先頭に質問識別子と定型句が並ぶこと: {err_logits_msg}"
        );
        assert_eq!(err_logits_msg.matches("質問 'category': ").count(), 1);

        // 5. Score 型の段階数不一致時にも question_id が伝播すること
        let invalid_score_q = Question::new_score("評価せよ。", vec!["単一".to_string()]);
        let mut score_buf = [0.0; 1];
        let err_score = evaluate_question_with_buf(
            &invalid_score_q,
            &[1.0],
            &config,
            &mut score_buf,
            Some("rating"),
        )
        .unwrap_err();
        assert!(err_score.to_string().contains("質問 'rating': "));
        assert!(matches!(
            err_score,
            CoreError::InvalidScoreLevelCount { .. }
        ));

        // 6. evaluate_choice_with_buf / evaluate_score_with_buf / evaluate_question_with_buf における
        //    softmax_into 失敗時にも外周の attach_question_id で question_id が 1 回のみ付与されること
        let valid_choice_criteria = valid_q.criteria.as_ref().unwrap();
        let mut choice_buf = [0.0; 2];
        let err_choice_nan = evaluate_choice_with_buf(
            &[f64::NAN, 1.0],
            valid_choice_criteria,
            1.0,
            &mut choice_buf,
            Some("choice_nan"),
        )
        .unwrap_err();
        assert_eq!(
            err_choice_nan
                .to_string()
                .matches("質問 'choice_nan': ")
                .count(),
            1
        );

        let valid_score_criteria = Criteria::List(vec!["1".to_string(), "2".to_string()]);
        let mut score_nan_buf = [0.0; 2];
        let err_score_nan = evaluate_score_with_buf(
            &[1.0, f64::NAN],
            &valid_score_criteria,
            1.0,
            &mut score_nan_buf,
            Some("score_nan"),
        )
        .unwrap_err();
        assert_eq!(
            err_score_nan
                .to_string()
                .matches("質問 'score_nan': ")
                .count(),
            1
        );

        let err_q_nan = evaluate_question_with_buf(
            &valid_q,
            &[f64::NAN, 1.0],
            &config,
            &mut choice_buf,
            Some("q_nan"),
        )
        .unwrap_err();
        assert_eq!(err_q_nan.to_string().matches("質問 'q_nan': ").count(), 1);
    }
}
