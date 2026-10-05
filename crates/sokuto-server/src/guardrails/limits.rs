//! # OOM 防壁・リソース制限モジュール
//!
//! リクエストの質問数、文字数、および推定トークン数を推論前に事前検証し、
//! メモリ枯渇 (OOM) や GPU/CPU リソースの過負荷を未然に遮断する Fast-Fail 機構を提供する。

use serde_json::Value;
use sokuto_core::schema::SystemOneRequest;

use crate::error::ServerError;

/// デフォルトのリクエストあたり最大質問数。
pub const DEFAULT_MAX_QUESTIONS: usize = 128;

/// デフォルトの State 最大許容文字数 (約 64KB)。
pub const DEFAULT_MAX_STATE_CHARS: usize = 65_536;

/// デフォルトの質問あたり最大許容文字数 (約 8KB)。
pub const DEFAULT_MAX_QUESTION_CHARS: usize = 8_192;

/// デフォルトの推定総トークン数上限 (約 16,384 トークン)。
pub const DEFAULT_MAX_ESTIMATED_TOKENS: usize = 16_384;

/// デフォルトのリクエストボディ最大許容バイト数 (2MB)。
pub const DEFAULT_MAX_BODY_BYTES: usize = 2 * 1024 * 1024;

/// リソース制限ガードレールの設定構造体。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LimitsConfig {
    /// 1 リクエストで許容される最大質問数。
    pub max_questions: usize,
    /// State 文字列の最大許容文字数。
    pub max_state_chars: usize,
    /// 1 質問あたりの最大許容文字数 (Instructions + Criteria)。
    pub max_question_chars: usize,
    /// リクエスト全体の最大推定トークン数。
    pub max_estimated_tokens: usize,
    /// リクエストボディ全体の最大許容バイト数。
    pub max_body_bytes: usize,
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_questions: DEFAULT_MAX_QUESTIONS,
            max_state_chars: DEFAULT_MAX_STATE_CHARS,
            max_question_chars: DEFAULT_MAX_QUESTION_CHARS,
            max_estimated_tokens: DEFAULT_MAX_ESTIMATED_TOKENS,
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
        }
    }
}

/// 文字列からトークン数を高速に安全側 (過大評価) で事前見積もりする。
///
/// UTF-8 バイト数および文字種に基づき、ゼロアロケーションで上限トークン数を算出する。
/// 英語 (1 語 1〜2 トークン) および日本語 (1 文字 1〜2 トークン) の双対特性を吸収し、
/// 最悪ケースとして `chars.len()` をベースとした安全側概算を行う。
#[inline]
pub fn estimate_tokens_fast(text: &str) -> usize {
    // 空文字は 0 トークン
    if text.is_empty() {
        return 0;
    }
    // 日本語混じりでも 1 文字あたり最大 2 トークン程度に収まる安全側バウンド
    let char_count = text.chars().count();
    // 最低でも 1 トークン、文字数とバイト数から過大推定 (保守的バジェット)
    char_count.max(text.len() / 3).max(1)
}

/// リクエスト全体の文字数および質問数を検証する。
///
/// # 引数
/// - `req`: Jev 推論リクエスト。
/// - `config`: リソース制限設定。
///
/// # 戻り値
/// 検証成功時は `(推定総文字数, 推定総トークン数)` のタプルを返却し、上限超過時は `ServerError::PayloadTooLarge` を返却する。
pub fn validate_limits(
    req: &SystemOneRequest,
    config: &LimitsConfig,
) -> Result<(usize, usize), ServerError> {
    // 1. 質問数上限チェック (L1 Fast-Fail)
    let question_count = req.questions.len();
    if question_count > config.max_questions {
        return Err(ServerError::PayloadTooLarge(format!(
            "リクエストに含まれる質問数 ({question_count}) が許容上限 ({}) を超過しています。",
            config.max_questions
        )));
    }

    // 2. State の文字数チェック (L1 Fast-Fail)
    let state_chars = match &req.state {
        Value::String(s) => s.chars().count(),
        other => other.to_string().chars().count(),
    };
    if state_chars > config.max_state_chars {
        return Err(ServerError::PayloadTooLarge(format!(
            "State の文字数 ({state_chars}) が許容上限 ({}) を超過しています。",
            config.max_state_chars
        )));
    }

    // 3. 各質問の文字数および全体のトークン見積もり (L2 チェック)
    let mut total_chars = state_chars;
    let mut total_estimated_tokens = match &req.state {
        Value::String(s) => estimate_tokens_fast(s),
        other => estimate_tokens_fast(&other.to_string()),
    };

    for (q_key, question) in &req.questions {
        let mut q_chars = question.instructions.chars().count();
        let mut q_tokens = estimate_tokens_fast(&question.instructions);

        if let Some(ref criteria) = question.criteria {
            if let Some(map) = criteria.as_map() {
                for (k, v) in map {
                    q_chars += k.chars().count() + v.chars().count();
                    q_tokens += estimate_tokens_fast(k) + estimate_tokens_fast(v);
                }
            } else if let Some(list) = criteria.as_list() {
                for item in list {
                    q_chars += item.chars().count();
                    q_tokens += estimate_tokens_fast(item);
                }
            }
        }

        if q_chars > config.max_question_chars {
            return Err(ServerError::PayloadTooLarge(format!(
                "質問 '{q_key}' の文字数 ({q_chars}) が許容上限 ({}) を超過しています。",
                config.max_question_chars
            )));
        }

        total_chars += q_chars;
        total_estimated_tokens += q_tokens.max(1);
    }

    if total_estimated_tokens > config.max_estimated_tokens {
        return Err(ServerError::PayloadTooLarge(format!(
            "リクエスト全体の推定トークン数 ({total_estimated_tokens}) が許容上限 ({}) を超過しています。",
            config.max_estimated_tokens
        )));
    }

    Ok((total_chars, total_estimated_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use sokuto_core::schema::Question;

    #[test]
    fn test_validate_limits_normal() {
        let mut questions = IndexMap::new();
        questions.insert(
            "q1".to_string(),
            Question::new_choice("test".to_string(), IndexMap::new()),
        );
        let req = SystemOneRequest::new(Value::String("hello world".to_string()), questions);
        let config = LimitsConfig::default();

        let res = validate_limits(&req, &config);
        assert!(res.is_ok());
    }

    #[test]
    fn test_validate_limits_exceed_questions() {
        let mut questions = IndexMap::new();
        for i in 0..10 {
            questions.insert(
                format!("q{i}"),
                Question::new_choice("test".to_string(), IndexMap::new()),
            );
        }
        let req = SystemOneRequest::new(Value::String("hello".to_string()), questions);
        let config = LimitsConfig {
            max_questions: 5,
            ..Default::default()
        };

        let res = validate_limits(&req, &config);
        assert!(matches!(res, Err(ServerError::PayloadTooLarge(_))));
    }

    #[test]
    fn test_validate_limits_exceed_state_chars() {
        let req = SystemOneRequest::new(
            Value::String("A".repeat(100)),
            IndexMap::from([(
                "q1".to_string(),
                Question::new_choice("test".to_string(), IndexMap::new()),
            )]),
        );
        let config = LimitsConfig {
            max_state_chars: 50,
            ..Default::default()
        };

        let res = validate_limits(&req, &config);
        assert!(matches!(res, Err(ServerError::PayloadTooLarge(_))));
    }
}
