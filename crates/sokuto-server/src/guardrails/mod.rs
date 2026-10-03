//! # 前処理ガードレール統括パイプラインモジュール
//!
//! リクエスト受信直後から推論エンジン投入までの間に、以下の一連のパイプラインを適用する。
//!
//! - 物理的 OOM 防壁 (Limits / Fast-Fail)
//! - 特殊トークン偽装・プロンプトインジェクション防壁 (Sanitizer)
//! - コンテキスト縮約・ノイズ除去 (Compactor / Anti-Context Rot)
//! - 相対日時絶対正規化 (Temporal Grounding)
//! - 算術・数え上げ事前集計 (Arithmetic Annotation)

pub mod arithmetic;
pub mod attention;
pub mod bimodal;
pub mod compactor;
pub mod fast_fail;
pub mod limits;
pub mod multitask;
pub mod nli;
pub mod proxy;
pub mod sanitization;
pub mod streaming;
pub mod temporal;
pub mod window;

use std::sync::Arc;

use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use serde_json::Value;
use sokuto_core::schema::{Answer, Criteria, Question, QuestionType, SystemOneRequest};

use crate::error::ServerError;
use crate::guardrails::arithmetic::{ArithmeticConfig, compute_arithmetic_summary};
pub use crate::guardrails::attention::{
    ATTN_MASK_ALLOW, ATTN_MASK_BLOCK, AttentionIsolationConfig, SegmentType,
    build_flat_attention_mask, build_segment_isolated_attention_mask,
};
use crate::guardrails::bimodal::{BimodalConfig, BimodalDetector, verify_and_adjust_bimodal};
use crate::guardrails::compactor::{CompactorConfig, compact_text};
pub use crate::guardrails::fast_fail::{
    FastFailConfig, FastFailScanner, GuardrailAction, GuardrailViolation,
};
use crate::guardrails::limits::{LimitsConfig, validate_limits};
pub use crate::guardrails::multitask::{
    MARKER_INJECTION, MARKER_PII, MARKER_TOXICITY, MultiTaskScanResult, MultiTaskScannerConfig,
    build_multitask_prompt, compute_free_energy, evaluate_multitask_logits, sigmoid,
};
pub use crate::guardrails::nli::{
    HallucinationVerificationResult, HallucinationVerifierConfig, NormalizedEntity,
    check_entity_mismatch, compute_synonym_coverage, extract_normalized_entities, verify_claim_nli,
};
pub use crate::guardrails::proxy::{GuardrailEngine, inline_guardrail_middleware};
use crate::guardrails::sanitization::{
    SanitizerConfig, normalize_noul_instruction, sanitize_score_criteria_text, sanitize_text,
};
pub use crate::guardrails::streaming::{
    SENTENCE_DELIMITERS, SentenceChunkBuffer, SpeculativeStreamInterceptor,
    StreamingVerifierConfig, extract_content_from_sse_line, intercept_sse_stream,
};
use crate::guardrails::temporal::{
    TemporalConfig, append_reference_time_metadata, normalize_temporal,
};
pub use crate::guardrails::window::{
    WindowConfig, WindowExtractionResult, estimate_tokens, extract_head_tail_window,
};

/// ガードレール全体の統合設定構造体。
#[derive(Debug, Clone, PartialEq)]
pub struct GuardrailConfig {
    /// ガードレール全体の有効化フラグ。
    pub enabled: bool,
    /// 物理リソース制限設定。
    pub limits: LimitsConfig,
    /// 決定論的 Fast-Fail 禁止語設定。
    pub fast_fail: FastFailConfig,
    /// 境界縮約窓設定。
    pub window: WindowConfig,
    /// 構造的分離アテンション設定。
    pub attention: AttentionIsolationConfig,
    /// マルチタスク動的マーカー並列スキャン設定。
    pub multitask: MultiTaskScannerConfig,
    /// 出力ハルシネーション検証設定。
    pub hallucination: HallucinationVerifierConfig,
    /// 投機的ストリーミング検証設定。
    pub streaming: StreamingVerifierConfig,
    /// サニタイズ設定。
    pub sanitizer: SanitizerConfig,
    /// コンテキスト縮約設定。
    pub compactor: CompactorConfig,
    /// 相対日時正規化設定。
    pub temporal: TemporalConfig,
    /// 算術事前集計設定。
    pub arithmetic: ArithmeticConfig,
    /// 二峰性 (バイモーダル) 分布検出設定。
    pub bimodal: BimodalConfig,
}

impl Default for GuardrailConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            limits: LimitsConfig::default(),
            fast_fail: FastFailConfig::default(),
            window: WindowConfig::default(),
            attention: AttentionIsolationConfig::default(),
            multitask: MultiTaskScannerConfig::default(),
            hallucination: HallucinationVerifierConfig::default(),
            streaming: StreamingVerifierConfig::default(),
            sanitizer: SanitizerConfig::default(),
            compactor: CompactorConfig::default(),
            temporal: TemporalConfig::default(),
            arithmetic: ArithmeticConfig::default(),
            bimodal: BimodalConfig::default(),
        }
    }
}

/// 前処理ガードレール実行結果レポート。
#[derive(Debug, Clone, Default)]
pub struct PreprocessReport {
    /// 事前見積もりされたトークン数。
    pub estimated_tokens: usize,
    /// 算術集計ブロックが付加されたか。
    pub added_arithmetic_summary: bool,
    /// 基準時刻メタデータが付加されたか。
    pub added_reference_time: bool,
    /// 境界縮約窓が適用されたか。
    pub window_truncated: bool,
}

/// 前処理および後処理ガードレールパイプライン。
#[derive(Clone)]
pub struct GuardrailPipeline {
    config: Arc<GuardrailConfig>,
    bimodal_detector: BimodalDetector,
    fast_fail_scanner: Arc<FastFailScanner>,
}

impl GuardrailPipeline {
    /// 新規 `GuardrailPipeline` を生成する。
    pub fn new(config: GuardrailConfig) -> Self {
        let bimodal_detector = BimodalDetector::new(config.bimodal.clone());
        let fast_fail_scanner = Arc::new(FastFailScanner::new(config.fast_fail.clone()));
        Self {
            config: Arc::new(config),
            bimodal_detector,
            fast_fail_scanner,
        }
    }

    /// 設定への参照を取得する。
    pub fn config(&self) -> &GuardrailConfig {
        &self.config
    }

    /// リクエストを検査および正規化する。
    ///
    /// # 引数
    /// - `req`: 可変推論リクエスト参照。
    /// - `ref_time`: 基準日時 (省略時は現在日時 `Utc::now()`)。
    ///
    /// # 戻り値
    /// 成功時は前処理レポートを返却し、物理上限超過時は `ServerError::PayloadTooLarge` を返却する。
    pub fn process(
        &self,
        req: &mut SystemOneRequest,
        ref_time: Option<DateTime<Utc>>,
    ) -> Result<PreprocessReport, ServerError> {
        if !self.config.enabled {
            return Ok(PreprocessReport::default());
        }

        let now = ref_time.unwrap_or_else(Utc::now);
        let mut report = PreprocessReport::default();

        // 0. 決定論的 Fast-Fail 走査 (Aho-Corasick, <0.1ms)
        if let Value::String(ref s) = req.state
            && let Err(violation) = self.fast_fail_scanner.scan(s)
        {
            return Err(ServerError::GuardrailViolation(violation.rule));
        }
        for question in req.questions.values() {
            if let Err(violation) = self.fast_fail_scanner.scan(&question.instructions) {
                return Err(ServerError::GuardrailViolation(violation.rule));
            }
        }

        // 1. 物理 OOM 防壁 (L1/L2 Fast-Fail)
        let (_, estimated_tokens) = validate_limits(req, &self.config.limits)?;
        report.estimated_tokens = estimated_tokens;

        // 2. State の正規化 (文字列への展開と前処理)
        let mut raw_state = match &req.state {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };

        // 境界縮約窓の適用 (長文に対する Early-Exit 窓走査)
        let window_res = extract_head_tail_window(&raw_state, &self.config.window);
        if window_res.was_truncated {
            raw_state = window_res.window_text.into_owned();
            report.window_truncated = true;
        }

        // 算術集計サマリーの事前計算 (元の Value から集計)
        let arithmetic_summary = compute_arithmetic_summary(&req.state, &self.config.arithmetic);

        // State 文字列のサニタイズ、縮約、日時正規化
        let state_sanitized = sanitize_text(&raw_state, &self.config.sanitizer);
        let state_compacted = compact_text(&state_sanitized, &self.config.compactor);
        let state_temporal = normalize_temporal(&state_compacted, now, &self.config.temporal);

        raw_state = state_temporal.into_owned();

        // 算術集計サマリーの付加
        if let Some(summary) = arithmetic_summary {
            raw_state.push_str(&summary);
            report.added_arithmetic_summary = true;
        }

        // 基準時刻メタデータの付加
        if self.config.temporal.attach_reference_time {
            append_reference_time_metadata(
                &mut raw_state,
                now,
                self.config.temporal.timezone_offset_hours,
            );
            report.added_reference_time = true;
        }

        // 正規化後の State を req に再格納
        req.state = Value::String(raw_state);

        // 3. 各 Question (Instructions, Criteria) の正規化
        let mut normalized_questions = IndexMap::with_capacity(req.questions.len());

        for (q_key, question) in &req.questions {
            // Instructions の正規化
            let inst_sanitized = sanitize_text(&question.instructions, &self.config.sanitizer);
            let inst_compacted = compact_text(&inst_sanitized, &self.config.compactor);
            let inst_temporal = normalize_temporal(&inst_compacted, now, &self.config.temporal);
            let mut normalized_inst = inst_temporal.into_owned();

            // Noul 型言明の正準プロンプトテンプレート自動ラッピング
            if question.question_type == QuestionType::Noul
                && self.config.sanitizer.normalize_noul_instructions
            {
                normalized_inst = normalize_noul_instruction(&normalized_inst).into_owned();
            }

            // Criteria の正規化
            let normalized_criteria = match &question.criteria {
                Some(Criteria::Map(map)) => {
                    let mut new_map = IndexMap::with_capacity(map.len());
                    for (k, v) in map {
                        let k_sanitized = sanitize_text(k, &self.config.sanitizer);
                        let v_sanitized = sanitize_text(v, &self.config.sanitizer);
                        let v_compacted = compact_text(&v_sanitized, &self.config.compactor);
                        new_map.insert(k_sanitized.into_owned(), v_compacted.into_owned());
                    }
                    Some(Criteria::Map(new_map))
                }
                Some(Criteria::List(list)) => {
                    let mut new_list = Vec::with_capacity(list.len());
                    for item in list {
                        let item_sanitized = sanitize_text(item, &self.config.sanitizer);
                        let item_compacted = compact_text(&item_sanitized, &self.config.compactor);
                        let mut item_final = item_compacted.into_owned();

                        // Score 型 Criteria の数字・記号プレフィックス自動サニタイズ (プロンプト用)
                        if question.question_type == QuestionType::Score
                            && self.config.sanitizer.sanitize_score_prefixes
                        {
                            item_final = sanitize_score_criteria_text(&item_final).into_owned();
                        }

                        new_list.push(item_final);
                    }
                    Some(Criteria::List(new_list))
                }
                Some(Criteria::None) => Some(Criteria::None),
                None => None,
            };

            let new_question = Question {
                question_type: question.question_type,
                instructions: normalized_inst,
                criteria: normalized_criteria,
            };

            normalized_questions.insert(q_key.clone(), new_question);
        }

        req.questions = normalized_questions;

        Ok(report)
    }

    /// 推論完了後のレスポンス (Answer 群) に対し、後処理ガードレールを適用する。
    ///
    /// # 処理内容
    /// 1. **二峰性 (バイモーダル) 分布検出**: Score 型の回答において意見二極化を検知した場合、Gating を `ConfirmOrEscalate` へ強制降格する。
    /// 2. **Score 型レスポンスキーの非破壊復元**: 前処理サニタイズで剥離されたプレフィックスを持つ元のラベル名に `probabilities` のマップキーを復元し、クライアント API 破壊を防止する。
    pub fn post_process(
        &self,
        answers: &mut IndexMap<String, Answer>,
        original_questions: &IndexMap<String, Question>,
    ) {
        if !self.config.enabled {
            return;
        }

        // 1. 二峰性 (バイモーダル) 分布検出と Gating 降格
        verify_and_adjust_bimodal(answers, original_questions, &self.bimodal_detector);

        // 2. Score 型レスポンスキーの復元
        if self.config.sanitizer.sanitize_score_prefixes {
            for (qid, answer) in answers.iter_mut() {
                let Some(orig_q) = original_questions.get(qid) else {
                    continue;
                };

                if orig_q.question_type != QuestionType::Score {
                    continue;
                }

                let Some(Criteria::List(orig_list)) = &orig_q.criteria else {
                    continue;
                };

                let Some(ref mut probs) = answer.probabilities else {
                    continue;
                };

                // 段階数が一致している場合、元のラベルキーでマップを再構築する
                if probs.len() == orig_list.len() {
                    let mut restored_probs = IndexMap::with_capacity(orig_list.len());
                    for (k, orig_label) in orig_list.iter().enumerate() {
                        if let Some((_, &p)) = probs.get_index(k) {
                            restored_probs.insert(orig_label.clone(), p);
                        }
                    }
                    *probs = restored_probs;
                }
            }
        }
    }
}

impl Default for GuardrailPipeline {
    fn default() -> Self {
        Self::new(GuardrailConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use serde_json::json;

    #[test]
    fn test_pipeline_end_to_end() {
        let pipeline = GuardrailPipeline::default();
        let ref_time = Utc.with_ymd_and_hms(2026, 9, 22, 10, 0, 0).unwrap();

        let mut questions = IndexMap::new();
        questions.insert(
            "q1".to_string(),
            Question::new_choice(
                "3日前の注文 [OP] を確認せよ。".to_string(),
                IndexMap::from([("yes".to_string(), "はい [OP]".to_string())]),
            ),
        );

        let mut req = SystemOneRequest::new(
            json!({
                "items": [
                    {"name": "apple", "amount": 100},
                    {"name": "banana", "amount": 200}
                ]
            }),
            questions,
        );

        let res = pipeline.process(&mut req, Some(ref_time));
        assert!(res.is_ok());

        let state_str = req.state.as_str().unwrap();
        // 特殊トークンは含まれていないが、算術集計と基準時刻が付与されていること
        assert!(state_str.contains("items_count=2"));
        assert!(state_str.contains("total_amount=300"));
        assert!(state_str.contains("[Reference Time: 2026-09-22T10:00:00Z]"));

        // Question 側の [OP] がサニタイズされ、相対日時が絶対化されていること
        let q1 = req.questions.get("q1").unwrap();
        assert!(q1.instructions.contains("3日前 (2026-09-19)"));
        assert!(q1.instructions.contains("[ OP ]"));

        let criteria_map = q1.criteria.as_ref().unwrap().as_map().unwrap();
        assert_eq!(criteria_map.get("yes").unwrap(), "はい [ OP ]");
    }

    #[test]
    fn test_pipeline_score_and_noul_guardrails() {
        let pipeline = GuardrailPipeline::default();

        let mut questions = IndexMap::new();
        // 1. Score 型質問 (数字記号プレフィックス付き)
        questions.insert(
            "q_score".to_string(),
            Question::new_score(
                "インシデントの緊急度を判定せよ。".to_string(),
                vec![
                    "1: 軽微".to_string(),
                    "2. 中度".to_string(),
                    "(3) 重大".to_string(),
                    "第4段階: 致命的".to_string(),
                ],
            ),
        );
        // 2. Noul 型質問 (簡潔な疑問文)
        questions.insert(
            "q_noul".to_string(),
            Question::new_noul("初期不良か？".to_string()),
        );

        let orig_questions = questions.clone();
        let mut req = SystemOneRequest::new("システムログ本文".to_string(), questions);

        // 前処理の実行
        let res = pipeline.process(&mut req, None);
        assert!(res.is_ok());

        // Score 型の Criteria がサニタイズされていること (プロンプト用)
        let q_score = req.questions.get("q_score").unwrap();
        match &q_score.criteria {
            Some(Criteria::List(list)) => {
                assert_eq!(list, &["軽微", "中度", "重大", "致命的"]);
            }
            _ => panic!("Criteria::List が必要です。"),
        }

        // Noul 型の Instructions が正準テンプレートに正規化されていること
        let q_noul = req.questions.get("q_noul").unwrap();
        assert_eq!(
            q_noul.instructions,
            "前提テキストの情報のみに基づいて、言明「初期不良」が真実であるか評価せよ。"
        );

        // 推論結果のシミュレーション (サニタイズされたラベルで Answer が生成されたと仮定)
        let mut answers = IndexMap::new();
        let mut score_probs = IndexMap::new();
        score_probs.insert("軽微".to_string(), 0.1);
        score_probs.insert("中度".to_string(), 0.7);
        score_probs.insert("重大".to_string(), 0.15);
        score_probs.insert("致命的".to_string(), 0.05);

        answers.insert(
            "q_score".to_string(),
            Answer::score(1.15, score_probs, 0.85),
        );

        // 後処理の適用
        pipeline.post_process(&mut answers, &orig_questions);

        // Score 型の probabilities キーがクライアント送信時の元ラベルに復元されていること
        let ans_score = answers.get("q_score").unwrap();
        let probs = ans_score.probabilities.as_ref().unwrap();
        assert_eq!(probs.get("1: 軽微"), Some(&0.1));
        assert_eq!(probs.get("2. 中度"), Some(&0.7));
        assert_eq!(probs.get("(3) 重大"), Some(&0.15));
        assert_eq!(probs.get("第4段階: 致命的"), Some(&0.05));
    }
}
