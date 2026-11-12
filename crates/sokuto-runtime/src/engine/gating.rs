//! # 推論エンジン向け確信度ゲーティング統合モジュール
//!
//! モデル成果物の較正情報 (`CalibrationConfig.gating_thresholds`) または
//! リクエスト指定の `GatingConfig` を用いて、推論直後の判定結果 (`Answer`) に対して
//! 透過的に 3 系統ルーティング (`AutoExecute` / `ConfirmOrEscalate` / `Fallback`) を確定・付与する。

use indexmap::IndexMap;
use sokuto_core::contract::CalibrationConfig;
pub use sokuto_core::gating::DecisionRoute;
use sokuto_core::gating::{GatingConfig, SystemRoutingSummary, evaluate_response_routing};
use sokuto_core::schema::{Answer, Question};

/// リクエスト固有設定とモデル較正設定から実効的なゲーティング設定をカスケード解決する。
///
/// # 解決優先順位
/// 1. `gating_config`: リクエストで明示された設定。
/// 2. `calib_config.gating_thresholds`: モデル成果物同梱の閾値設定。
///
/// # 引数
/// - `gating_config`: リクエストで指定された設定 (任意)。
/// - `calib_config`: モデルのキャリブレーション設定。
///
/// # 戻り値
/// 解決された `GatingConfig`。
pub fn resolve_gating_config(
    gating_config: Option<&GatingConfig>,
    calib_config: &CalibrationConfig,
) -> GatingConfig {
    match gating_config {
        Some(cfg) => cfg.clone(),
        None => calib_config.gating_config(),
    }
}

use crate::engine::escalation::build_rich_escalation_prompt;
use crate::error::{Result, RuntimeError};
use sokuto_core::decision::evaluate_question_with_buf;
use sokuto_core::gating::evaluate_answer_gating_opts;
use sokuto_core::math::free_energy;
use sokuto_core::schema::{Criteria, QuestionType};

/// Choice 型の質問とロジット配列から、正規化ヘルムホルツ自由エネルギーを算出する。
///
/// モデル較正設定の `energy_temperature` (または候補数バケット温度 $T = \text{calib\_config.get\_temperature(QuestionType::Choice, K)}$) を用い、
/// $E_{\text{norm}}(x) = -T \ln \sum_{i=1}^K \exp(z_i / T) + T \ln K$ を算出する。
/// Choice 型以外 (Score 型、Noul 型) またはロジットが空の場合は `None` を返す。
///
/// # 引数
/// - `question`: 質問定義。
/// - `logits`: 決定ロジットスライス。
/// - `calib_config`: キャリブレーション設定。
///
/// # 戻り値
/// 正規化ヘルムホルツ自由エネルギー値 (任意)。
pub fn calculate_choice_energy(
    question: &Question,
    logits: &[f64],
    calib_config: &CalibrationConfig,
) -> Option<f64> {
    calculate_choice_energy_with_config(question, logits, calib_config, None)
}

/// Choice 型の質問とロジット配列、およびリクエスト固有のゲーティング設定から、正規化ヘルムホルツ自由エネルギーを算出する。
///
/// 粗密探索 (Coarse-to-Fine) 等で $M$ 個に枝刈りされたロジット列と全 $K$ 候補を含む質問定義が渡された場合、
/// 母集団数 $K$ ($K \ge M$) を用いて較正温度を解決し、除外候補を確率 0 (ロジット $-\infty$) とみなした
/// 全候補空間基準の正規化自由エネルギー $E_{\text{norm}}(x; K) = -T \ln \sum_{m=1}^M \exp(z_m / T) + T \ln K$ を算出する。
///
/// # 温度解決の優先順位
/// 1. `gating_config.energy_temperature`: リクエストで明示されたエネルギー温度。
/// 2. `calib_config.gating_thresholds.energy_temperature`: モデル成果物で事前較正されたエネルギー専用温度。
/// 3. `calib_config.get_temperature(QuestionType::Choice, K)`: 候補数バケットに基づく Softmax 較正温度。
///
/// # 引数
/// - `question`: 質問定義。
/// - `logits`: 決定ロジットスライス。
/// - `calib_config`: キャリブレーション設定。
/// - `gating_config`: リクエスト固有のゲーティング設定 (任意)。
///
/// # 戻り値
/// 正規化ヘルムホルツ自由エネルギー値 (任意)。
pub fn calculate_choice_energy_with_config(
    question: &Question,
    logits: &[f64],
    calib_config: &CalibrationConfig,
    gating_config: Option<&GatingConfig>,
) -> Option<f64> {
    if question.question_type != QuestionType::Choice || logits.is_empty() {
        return None;
    }
    let k = question
        .criteria
        .as_ref()
        .and_then(Criteria::as_map)
        .map_or_else(|| logits.len(), |m| m.len().max(logits.len()));
    let temp = gating_config
        .and_then(|cfg| cfg.thresholds.energy_temperature)
        .or(calib_config.gating_thresholds.energy_temperature)
        .unwrap_or_else(|| calib_config.get_temperature(QuestionType::Choice, k));

    if k <= 1 {
        free_energy(logits, temp).ok()
    } else {
        let raw_energy = free_energy(logits, temp).ok()?;
        let effective_tau = temp.max(1e-4);
        let shift = effective_tau * (k as f64).ln();
        Some(raw_energy + shift)
    }
}

/// 単一の判定結果 (`Answer`) に対し、エネルギー値と解決されたゲーティング設定を適用する。
///
/// 設定の `enabled` が `true` の場合、`evaluate_answer_gating_with_energy` を適用して
/// `answer.gating` に監査メタデータを付与する。無効 (`enabled == false`) の場合は何もしない。
/// Choice 型かつ `is_ood == true` の場合、または確信度不足によるエスカレーション時は、
/// 文脈や診断情報を含む高度な CoT エスカレーションプロンプトを構築して格納する。
///
/// # 引数
/// - `answer`: 判定結果 (可変参照)。
/// - `energy`: 正規化ヘルムホルツ自由エネルギー値 (任意)。
/// - `state`: 文脈テキスト (プロンプト構築用、任意)。
/// - `question_id`: 質問識別子 (プロンプト構築用、任意)。
/// - `question`: 質問定義 (任意)。
/// - `gating_config`: 明示的なゲーティング設定 (任意)。
/// - `calib_config`: モデルのキャリブレーション設定。
pub fn apply_gating_to_answer_with_energy(
    answer: &mut Answer,
    energy: Option<f64>,
    state: Option<&str>,
    question_id: Option<&str>,
    question: Option<&Question>,
    gating_config: Option<&GatingConfig>,
    calib_config: &CalibrationConfig,
) {
    let config = resolve_gating_config(gating_config, calib_config);
    if config.enabled {
        // 上位層で rich_escalation_prompt に置換されるため、コア層でのデフォルトプロンプト生成(二重アロケーション)を抑止する。
        let mut meta =
            evaluate_answer_gating_opts(answer, question_id, question, &config, energy, false);
        if meta.route != DecisionRoute::AutoExecute {
            let rich_prompt = meta.escalation.as_ref().map(|esc| {
                build_rich_escalation_prompt(
                    state,
                    question_id,
                    question,
                    &meta,
                    &esc.top_candidates,
                )
            });
            if let (Some(esc), Some(prompt)) = (&mut meta.escalation, rich_prompt) {
                esc.prompt_template = Some(prompt);
            }
        }
        answer.gating = Some(meta);
    }
}

/// 単一の判定結果 (`Answer`) とロジットスライスからエネルギーを自動算出し、ゲーティング設定を適用する。
///
/// # 引数
/// - `answer`: 判定結果 (可変参照)。
/// - `logits`: 決定ロジットスライス (任意)。
/// - `state`: 文脈テキスト (任意)。
/// - `question_id`: 質問識別子 (任意)。
/// - `question`: 質問定義 (任意)。
/// - `gating_config`: 明示的なゲーティング設定 (任意)。
/// - `calib_config`: モデルのキャリブレーション設定。
pub fn apply_gating_to_answer_with_logits(
    answer: &mut Answer,
    logits: Option<&[f64]>,
    state: Option<&str>,
    question_id: Option<&str>,
    question: Option<&Question>,
    gating_config: Option<&GatingConfig>,
    calib_config: &CalibrationConfig,
) {
    let energy = match (question, logits) {
        (Some(q), Some(lg)) => {
            calculate_choice_energy_with_config(q, lg, calib_config, gating_config)
        }
        _ => None,
    };
    apply_gating_to_answer_with_energy(
        answer,
        energy,
        state,
        question_id,
        question,
        gating_config,
        calib_config,
    );
}

/// 単一質問の決定数理評価からロジット由来エネルギー算出・確信度ゲーティングまでを一貫実行する (バッファ再利用版)。
///
/// 事前確保された作業バッファ `probs_buf` を再利用しつつ、`evaluate_question_with_buf` による
/// 決定プリミティブ解決と [`apply_gating_to_answer_with_logits`] によるエネルギー算出・ゲーティング適用を一括で行う。
///
/// # 引数
/// - `question`: 評価対象の質問定義。
/// - `logits`: 推論エンジンから出力された生ロジットスライス。
/// - `probs_buf`: 確率値書き込み用の作業バッファスライス (`logits` と同長が必要)。
/// - `state`: 文脈テキスト (エスカレーションプロンプト構築用、任意)。
/// - `question_id`: 質問識別子 (エラー診断およびプロンプト構築用、任意)。
/// - `gating_config`: 明示的なゲーティング設定 (任意)。
/// - `calib_config`: モデルのキャリブレーション設定。
pub fn evaluate_and_gate_answer_with_buf(
    question: &Question,
    logits: &[f64],
    probs_buf: &mut [f64],
    state: Option<&str>,
    question_id: Option<&str>,
    gating_config: Option<&GatingConfig>,
    calib_config: &CalibrationConfig,
) -> Result<Answer> {
    let mut answer =
        evaluate_question_with_buf(question, logits, calib_config, probs_buf, question_id)
            .map_err(RuntimeError::Core)?;
    apply_gating_to_answer_with_logits(
        &mut answer,
        Some(logits),
        state,
        question_id,
        Some(question),
        gating_config,
        calib_config,
    );
    Ok(answer)
}

/// 単一質問の決定数理評価からロジット由来エネルギー算出・確信度ゲーティングまでを一貫実行する。
///
/// 内部で確率計算用バッファを確保し、[`evaluate_and_gate_answer_with_buf`] へ委譲する。
///
/// # 引数
/// - `question`: 評価対象の質問定義。
/// - `logits`: 推論エンジンから出力された生ロジットスライス。
/// - `state`: 文脈テキスト (エスカレーションプロンプト構築用、任意)。
/// - `question_id`: 質問識別子 (エラー診断およびプロンプト構築用、任意)。
/// - `gating_config`: 明示的なゲーティング設定 (任意)。
/// - `calib_config`: モデルのキャリブレーション設定。
pub fn evaluate_and_gate_answer(
    question: &Question,
    logits: &[f64],
    state: Option<&str>,
    question_id: Option<&str>,
    gating_config: Option<&GatingConfig>,
    calib_config: &CalibrationConfig,
) -> Result<Answer> {
    let mut probs_buf = vec![0.0f64; logits.len()];
    evaluate_and_gate_answer_with_buf(
        question,
        logits,
        &mut probs_buf,
        state,
        question_id,
        gating_config,
        calib_config,
    )
}

/// 単一の判定結果 (`Answer`) に対し、解決されたゲーティング設定を適用する。
///
/// エネルギー値は `None` として処理されるため、OOD チェックを伴わない標準の確信度ゲーティングが行われる。
/// ロジットに基づく OOD 安全弁を有効化する場合は [`apply_gating_to_answer_with_energy`] または
/// [`apply_gating_to_answer_with_logits`] を使用する。
///
/// # 引数
/// - `answer`: 判定結果 (可変参照)。
/// - `state`: 文脈テキスト (プロンプト構築用、任意)。
/// - `question_id`: 質問識別子 (プロンプト構築用、任意)。
/// - `question`: 質問定義 (任意)。
/// - `gating_config`: 明示的なゲーティング設定 (任意)。
/// - `calib_config`: モデルのキャリブレーション設定。
pub fn apply_gating_to_answer(
    answer: &mut Answer,
    state: Option<&str>,
    question_id: Option<&str>,
    question: Option<&Question>,
    gating_config: Option<&GatingConfig>,
    calib_config: &CalibrationConfig,
) {
    apply_gating_to_answer_with_energy(
        answer,
        None,
        state,
        question_id,
        question,
        gating_config,
        calib_config,
    );
}

/// 複数質問の判定結果マップに対し、ロジット配列を用いたエネルギー付きゲーティング判定を一括適用する。
///
/// 各回答にゲーティングメタデータ (および OOD 診断) を付与し、有効な場合は `SystemRoutingSummary` を返却する。
///
/// # 引数
/// - `answers`: 判定結果マップ (可変参照)。
/// - `logits_map`: 各質問のロジット配列マップ。
/// - `state`: 共通の文脈テキスト (任意)。
/// - `questions`: 質問定義マップ。
/// - `gating_config`: 明示的なゲーティング設定 (任意)。
/// - `calib_config`: モデルのキャリブレーション設定。
///
/// # 戻り値
/// ゲーティングが有効な場合は `Some(SystemRoutingSummary)`、無効な場合は `None`。
pub fn apply_gating_to_answers_with_logits(
    answers: &mut IndexMap<String, Answer>,
    logits_map: &IndexMap<String, Vec<f64>>,
    state: Option<&str>,
    questions: &IndexMap<String, Question>,
    gating_config: Option<&GatingConfig>,
    calib_config: &CalibrationConfig,
) -> Option<SystemRoutingSummary> {
    let config = resolve_gating_config(gating_config, calib_config);
    if !config.enabled {
        return None;
    }

    for (qid, answer) in answers.iter_mut() {
        let q_def = questions.get(qid);
        let q_logits = logits_map.get(qid).map(|v| v.as_slice());
        apply_gating_to_answer_with_logits(
            answer,
            q_logits,
            state,
            Some(qid),
            q_def,
            gating_config,
            calib_config,
        );
    }

    Some(evaluate_response_routing(answers))
}

/// 複数質問の判定結果マップに対し、ゲーティング判定を一括適用し集約サマリーを算出する。
///
/// 各回答にゲーティングメタデータを付与し、有効な場合は `SystemRoutingSummary` を返却する。
///
/// # 引数
/// - `answers`: 判定結果マップ (可変参照)。
/// - `state`: 共通の文脈テキスト (任意)。
/// - `questions`: 質問定義マップ。
/// - `gating_config`: 明示的なゲーティング設定 (任意)。
/// - `calib_config`: モデルのキャリブレーション設定。
///
/// # 戻り値
/// ゲーティングが有効な場合は `Some(SystemRoutingSummary)`、無効な場合は `None`。
pub fn apply_gating_to_answers(
    answers: &mut IndexMap<String, Answer>,
    state: Option<&str>,
    questions: &IndexMap<String, Question>,
    gating_config: Option<&GatingConfig>,
    calib_config: &CalibrationConfig,
) -> Option<SystemRoutingSummary> {
    let config = resolve_gating_config(gating_config, calib_config);
    if !config.enabled {
        return None;
    }

    for (qid, answer) in answers.iter_mut() {
        let q_def = questions.get(qid);
        apply_gating_to_answer(answer, state, Some(qid), q_def, gating_config, calib_config);
    }

    Some(evaluate_response_routing(answers))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sokuto_core::contract::calibration::GatingThresholds;
    use sokuto_core::math::normalized_free_energy;

    #[test]
    fn test_resolve_gating_config_cascade() {
        let calib = CalibrationConfig {
            gating_thresholds: GatingThresholds {
                high_threshold: 0.72,
                low_threshold: 0.38,
                top_margin_threshold: 0.18,
                ..Default::default()
            },
            ..Default::default()
        };

        // 1. 指定なしの場合は calib_config から導出
        let resolved_default = resolve_gating_config(None, &calib);
        assert!(resolved_default.enabled);
        assert!((resolved_default.thresholds.high_threshold - 0.72).abs() < 1e-9);

        // 2. 明示指定された場合はそれを優先
        let explicit = GatingConfig {
            enabled: false,
            thresholds: GatingThresholds {
                high_threshold: 0.90,
                low_threshold: 0.40,
                top_margin_threshold: 0.20,
                ood_enabled: true,
                energy_threshold: -1.0,
                ..GatingThresholds::default()
            },
        };
        let resolved_explicit = resolve_gating_config(Some(&explicit), &calib);
        assert!(!resolved_explicit.enabled);
        assert!((resolved_explicit.thresholds.high_threshold - 0.90).abs() < 1e-9);
    }

    #[test]
    fn test_apply_gating_to_answer() {
        let calib = CalibrationConfig::default();
        let mut probs = IndexMap::new();
        probs.insert("opt_a".to_string(), 0.95);
        probs.insert("opt_b".to_string(), 0.05);

        let mut ans = Answer::choice("opt_a", probs, 0.90);
        assert!(ans.gating.is_none());

        apply_gating_to_answer(&mut ans, Some("文脈"), Some("q1"), None, None, &calib);
        assert!(ans.gating.is_some());
        assert_eq!(
            ans.gating.as_ref().unwrap().route,
            DecisionRoute::AutoExecute
        );
        // AutoExecute 時はプロンプト生成がスキップされ、escalation は None のまま
        assert!(ans.gating.as_ref().unwrap().escalation.is_none());
    }

    #[test]
    fn test_apply_gating_to_answer_disabled() {
        let calib = CalibrationConfig::default();
        let disabled_cfg = GatingConfig {
            enabled: false,
            ..GatingConfig::default()
        };

        let mut probs = IndexMap::new();
        probs.insert("opt_a".to_string(), 0.95);
        probs.insert("opt_b".to_string(), 0.05);

        let mut ans = Answer::choice("opt_a", probs, 0.90);
        apply_gating_to_answer(
            &mut ans,
            None,
            Some("q1"),
            None,
            Some(&disabled_cfg),
            &calib,
        );
        assert!(ans.gating.is_none());
    }

    #[test]
    fn test_apply_gating_to_answer_escalation_rich_prompt() {
        let calib = CalibrationConfig::default();
        let mut criteria = IndexMap::new();
        criteria.insert("opt_a".to_string(), "オプションAの詳細説明".to_string());
        criteria.insert("opt_b".to_string(), "オプションBの詳細説明".to_string());
        let q = Question::new_choice("指示文", criteria);

        let mut probs = IndexMap::new();
        probs.insert("opt_a".to_string(), 0.51);
        probs.insert("opt_b".to_string(), 0.49);

        let mut ans = Answer::choice("opt_a", probs, 0.51);
        apply_gating_to_answer(
            &mut ans,
            Some("ユーザー問い合わせ文脈"),
            Some("q_test"),
            Some(&q),
            None,
            &calib,
        );

        assert!(ans.gating.is_some());
        let meta = ans.gating.as_ref().unwrap();
        assert_eq!(meta.route, DecisionRoute::ConfirmOrEscalate);
        assert!(meta.escalation.is_some());

        let prompt = meta
            .escalation
            .as_ref()
            .unwrap()
            .prompt_template
            .as_ref()
            .unwrap();

        assert!(prompt.contains("<context>"));
        assert!(prompt.contains("ユーザー問い合わせ文脈"));
        assert!(prompt.contains("</context>"));
        assert!(prompt.contains("オプションAの詳細説明"));
        assert!(prompt.contains("System 1 の判定では候補"));
        assert!(prompt.contains("思考連鎖 (Chain-of-Thought)"));
        assert!(prompt.contains("```json"));
    }

    #[test]
    fn test_calculate_choice_energy() {
        let calib = CalibrationConfig::default();
        let mut criteria = IndexMap::new();
        criteria.insert("a".to_string(), "説明A".to_string());
        criteria.insert("b".to_string(), "説明B".to_string());
        let q = Question::new_choice("指示", criteria);

        // ID 的ロジット (顕著な差)
        let id_logits = [5.0, -2.0];
        let energy_id = calculate_choice_energy(&q, &id_logits, &calib).unwrap();
        // OOD 的ロジット (低確信度・平坦)
        let ood_logits = [0.1, 0.1];
        let energy_ood = calculate_choice_energy(&q, &ood_logits, &calib).unwrap();

        // 顕著な差がある ID は低エネルギー、平坦な OOD は高エネルギー
        assert!(energy_id < energy_ood);
    }

    #[test]
    fn test_apply_gating_to_answer_with_energy_ood() {
        let calib = CalibrationConfig::default();
        let mut criteria = IndexMap::new();
        criteria.insert("opt_a".to_string(), "説明A".to_string());
        criteria.insert("opt_b".to_string(), "説明B".to_string());
        let q = Question::new_choice("指示文", criteria);

        let mut probs = IndexMap::new();
        probs.insert("opt_a".to_string(), 0.95);
        probs.insert("opt_b".to_string(), 0.05);

        // 確信度は高くても、エネルギーが閾値 (-1.0) を超えて OOD (平坦/低信頼性) と検知された場合
        let mut ans = Answer::choice("opt_a", probs, 0.95);
        apply_gating_to_answer_with_energy(
            &mut ans,
            Some(-0.2), // > -1.0
            Some("未定義カテゴリの文脈"),
            Some("q_ood"),
            Some(&q),
            None,
            &calib,
        );

        assert!(ans.gating.is_some());
        let meta = ans.gating.as_ref().unwrap();
        assert_eq!(meta.route, DecisionRoute::Fallback);
        assert!(meta.is_ood);
        assert_eq!(meta.energy, Some(-0.2));

        // エスカレーションプロンプトに OOD 警告が含まれることを検証
        let esc = meta.escalation.as_ref().unwrap();
        let prompt = esc.prompt_template.as_ref().unwrap();
        assert!(prompt.contains("None of the above / 該当なし"));
        assert!(prompt.contains("未定義カテゴリ"));
    }

    #[test]
    fn test_apply_gating_to_answer_with_logits() {
        let calib = CalibrationConfig::default();
        let mut criteria = IndexMap::new();
        criteria.insert("opt_a".to_string(), "説明A".to_string());
        criteria.insert("opt_b".to_string(), "説明B".to_string());
        let q = Question::new_choice("指示文", criteria);

        let mut probs = IndexMap::new();
        probs.insert("opt_a".to_string(), 0.95);
        probs.insert("opt_b".to_string(), 0.05);

        let mut ans = Answer::choice("opt_a", probs, 0.95);
        // ID 的な明確なロジット -> 低エネルギー -> AutoExecute 通過
        let id_logits = [5.0, -2.0];
        apply_gating_to_answer_with_logits(
            &mut ans,
            Some(&id_logits),
            Some("通常文脈"),
            Some("q_id"),
            Some(&q),
            None,
            &calib,
        );

        let meta = ans.gating.as_ref().unwrap();
        assert_eq!(meta.route, DecisionRoute::AutoExecute);
        assert!(!meta.is_ood);
        assert!(meta.energy.is_some());
    }

    #[test]
    fn test_calculate_choice_energy_with_config_prefers_request_temperature() {
        let calib = CalibrationConfig {
            gating_thresholds: GatingThresholds {
                energy_temperature: Some(1.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let mut criteria = IndexMap::new();
        criteria.insert("a".to_string(), "説明A".to_string());
        criteria.insert("b".to_string(), "説明B".to_string());
        let q = Question::new_choice("指示", criteria);
        let logits = [3.0, -1.0];

        let default_energy = calculate_choice_energy(&q, &logits, &calib).unwrap();
        let gating_cfg = GatingConfig {
            enabled: true,
            thresholds: GatingThresholds {
                energy_temperature: Some(4.0),
                ..Default::default()
            },
        };
        let custom_energy =
            calculate_choice_energy_with_config(&q, &logits, &calib, Some(&gating_cfg)).unwrap();
        let expected_energy = normalized_free_energy(&logits, 4.0).unwrap();

        assert!((custom_energy - expected_energy).abs() < 1e-12);
        assert!((custom_energy - default_energy).abs() > 1e-3);
    }

    #[test]
    fn test_calculate_choice_energy_with_coarse_pruning_population() {
        let calib = CalibrationConfig::default();
        let mut criteria = IndexMap::new();
        criteria.insert("a".to_string(), "説明A".to_string());
        criteria.insert("b".to_string(), "説明B".to_string());
        criteria.insert("c".to_string(), "説明C".to_string());
        criteria.insert("d".to_string(), "説明D".to_string());
        let q_all = Question::new_choice("指示", criteria);

        // 粗密探索で 4 候補中上位 2 候補のみ推論した場合
        let sub_logits = [4.0, -1.0];
        let energy = calculate_choice_energy(&q_all, &sub_logits, &calib).unwrap();

        // 生エネルギー -T ln(exp(4) + exp(-1)) に 全候補数 K=4 のシフト +T ln(4) が加算されること
        let raw_energy = free_energy(&sub_logits, 1.0).unwrap();
        let expected_energy = raw_energy + (4.0f64).ln();
        assert!((energy - expected_energy).abs() < 1e-12);

        // 誤って 2 候補の質問を渡した場合のエネルギーより高くなる (OOD 側に正しく評価される) ことを確認
        let mut sub_criteria = IndexMap::new();
        sub_criteria.insert("a".to_string(), "説明A".to_string());
        sub_criteria.insert("b".to_string(), "説明B".to_string());
        let q_sub = Question::new_choice("指示", sub_criteria);
        let sub_energy = calculate_choice_energy(&q_sub, &sub_logits, &calib).unwrap();

        // 差分は ln(4) - ln(2) = ln(2)
        assert!((energy - sub_energy - (2.0f64).ln()).abs() < 1e-12);
    }

    #[test]
    fn test_evaluate_and_gate_answer_pipeline() {
        let calib = CalibrationConfig::default();
        let gating_cfg = GatingConfig {
            enabled: true,
            thresholds: GatingThresholds {
                high_threshold: 0.90,
                low_threshold: 0.20,
                top_margin_threshold: 0.80,
                ood_enabled: true,
                energy_threshold: 0.0,
                energy_temperature: None,
                ..GatingThresholds::default()
            },
        };

        let mut criteria = IndexMap::new();
        criteria.insert("a".to_string(), "候補A".to_string());
        criteria.insert("b".to_string(), "候補B".to_string());
        let q = Question::new_choice("選択せよ", criteria);

        // 僅差ロジットは ConfirmOrEscalate とエスカレーションプロンプト・エネルギーを付与すること。
        let ambiguous_logits = [2.5, 0.5];
        let ans = evaluate_and_gate_answer(
            &q,
            &ambiguous_logits,
            Some("入力文脈"),
            Some("q_ambig"),
            Some(&gating_cfg),
            &calib,
        )
        .unwrap();
        assert_eq!(ans.as_choice(), Some("a"));
        let meta = ans.gating.as_ref().unwrap();
        assert_eq!(meta.route, DecisionRoute::ConfirmOrEscalate);
        assert!(meta.energy.is_some());
        assert!(meta.escalation.is_some());

        // バッファ再利用版でも同一結果が得られ、エラー時は question_id が維持されること。
        let mut probs_buf = [0.0f64; 2];
        let ans_buf = evaluate_and_gate_answer_with_buf(
            &q,
            &ambiguous_logits,
            &mut probs_buf,
            Some("入力文脈"),
            Some("q_ambig"),
            Some(&gating_cfg),
            &calib,
        )
        .unwrap();
        assert_eq!(ans_buf.as_choice(), Some("a"));
        assert_eq!(
            ans_buf.gating.as_ref().unwrap().energy,
            ans.gating.as_ref().unwrap().energy
        );

        let err = evaluate_and_gate_answer_with_buf(
            &q,
            &[1.0],
            &mut [0.0],
            Some("入力文脈"),
            Some("q_mismatch"),
            Some(&gating_cfg),
            &calib,
        )
        .unwrap_err();
        assert!(err.to_string().contains("q_mismatch"));
    }
}
