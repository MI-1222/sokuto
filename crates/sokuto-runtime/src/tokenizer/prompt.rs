//! # プロンプト書式化モジュール
//!
//! Jev 互換の質問仕様 (`Question`) から、Python 学習パイプラインと
//! 完全一致するプロンプトパーツおよび候補キー列を構築する。

use sokuto_core::contract::model_spec::{
    MAX_NUM_OPTIONS, MAX_SCORE_LEVELS, MIN_NUM_OPTIONS, MIN_SCORE_LEVELS, TOKEN_OPTION_MARKER,
};
use sokuto_core::schema::{Criteria, Question, QuestionType};

use crate::error::{Result, RuntimeError};

/// 固定プレフィックス文字列。
pub const PREFIX_STATE: &str = "State: ";

/// フォーマット済みのプロンプトパーツ構造体。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormattedPrompt {
    /// プレフィックス文字列 (`"State: "`)。
    pub prefix: &'static str,
    /// サフィックス文字列 (`"\nInstructions: ...\nCriteria: [OP] ..."` 等)。
    pub suffix: String,
    /// プロンプト出現順に整列した候補識別子リスト。
    pub option_keys: Vec<String>,
}

/// 質問仕様からフォーマット済みプロンプトパーツを生成する。
///
/// # 挙動仕様
/// - **Choice 型**: `Criteria::Map` の要素順序を保持し、各記述の先頭に `[OP]` を付与して半角スペースで結合する。
/// - **Score 型**: `Criteria::List` の段階順序を保持し、各段階の先頭に `[OP]` を付与して半角スペースで結合する。候補キーは `"0"`, `"1"`, ... とする。
/// - **Noul 型**: 暗黙の真偽2候補 `"[OP] 真 (True) [OP] 偽 (False)"` を展開し、候補キーは `["true", "false"]` とする。
///
/// # エラー
/// 質問の整合性検証(候補数範囲、対応する Criteria 形式)に違反した場合は `RuntimeError` を返却する。
pub fn format_prompt(question: &Question) -> Result<FormattedPrompt> {
    match question.question_type {
        QuestionType::Choice => {
            let criteria = question.criteria.as_ref().ok_or_else(|| {
                RuntimeError::InvalidQuestion(
                    "Choice 型には Criteria::Map が必須です。".to_string(),
                )
            })?;

            let map = match criteria {
                Criteria::Map(map) => map,
                _ => {
                    return Err(RuntimeError::InvalidQuestion(
                        "Choice 型には Criteria::Map が指定される必要があります。".to_string(),
                    ));
                }
            };

            let count = map.len();
            if !(MIN_NUM_OPTIONS..=MAX_NUM_OPTIONS).contains(&count) {
                return Err(RuntimeError::InvalidQuestion(format!(
                    "Choice 型の候補数は {MIN_NUM_OPTIONS}〜{MAX_NUM_OPTIONS} 件である必要がありますが、{count} 件指定されました。"
                )));
            }

            let mut option_keys = Vec::with_capacity(count);
            let mut criteria_parts = Vec::with_capacity(count);

            for (key, desc) in map {
                option_keys.push(key.clone());
                criteria_parts.push(format!("{TOKEN_OPTION_MARKER} {desc}"));
            }

            let criteria_text = criteria_parts.join(" ");
            let suffix = format!(
                "\nInstructions: {}\nCriteria: {}",
                question.instructions, criteria_text
            );

            Ok(FormattedPrompt {
                prefix: PREFIX_STATE,
                suffix,
                option_keys,
            })
        }
        QuestionType::Score => {
            let criteria = question.criteria.as_ref().ok_or_else(|| {
                RuntimeError::InvalidQuestion(
                    "Score 型には Criteria::List が必須です。".to_string(),
                )
            })?;

            let list = match criteria {
                Criteria::List(list) => list,
                _ => {
                    return Err(RuntimeError::InvalidQuestion(
                        "Score 型には Criteria::List が指定される必要があります。".to_string(),
                    ));
                }
            };

            let count = list.len();
            if !(MIN_SCORE_LEVELS..=MAX_SCORE_LEVELS).contains(&count) {
                return Err(RuntimeError::InvalidQuestion(format!(
                    "Score 型の段階数は {MIN_SCORE_LEVELS}〜{MAX_SCORE_LEVELS} 段階である必要がありますが、{count} 段階指定されました。"
                )));
            }

            let mut option_keys = Vec::with_capacity(count);
            let mut criteria_parts = Vec::with_capacity(count);

            for (i, desc) in list.iter().enumerate() {
                option_keys.push(i.to_string());
                criteria_parts.push(format!("{TOKEN_OPTION_MARKER} {desc}"));
            }

            let criteria_text = criteria_parts.join(" ");
            let suffix = format!(
                "\nInstructions: {}\nCriteria: {}",
                question.instructions, criteria_text
            );

            Ok(FormattedPrompt {
                prefix: PREFIX_STATE,
                suffix,
                option_keys,
            })
        }
        QuestionType::Noul => {
            let option_keys = vec!["true".to_string(), "false".to_string()];
            let criteria_text =
                format!("{TOKEN_OPTION_MARKER} 真 (True) {TOKEN_OPTION_MARKER} 偽 (False)");
            let suffix = format!(
                "\nInstructions: {}\nCriteria: {}",
                question.instructions, criteria_text
            );

            Ok(FormattedPrompt {
                prefix: PREFIX_STATE,
                suffix,
                option_keys,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;

    #[test]
    fn test_format_prompt_choice() {
        let mut criteria = IndexMap::new();
        criteria.insert("activate".to_string(), "カードの有効化".to_string());
        criteria.insert("status".to_string(), "配送状況の確認".to_string());

        let question = Question::new_choice("意図を分類せよ。", criteria);
        let formatted = format_prompt(&question).unwrap();

        assert_eq!(formatted.prefix, "State: ");
        assert_eq!(formatted.option_keys, vec!["activate", "status"]);
        assert!(formatted.suffix.contains("Instructions: 意図を分類せよ。"));
        assert!(
            formatted
                .suffix
                .contains("Criteria: [OP] カードの有効化 [OP] 配送状況の確認")
        );
    }

    #[test]
    fn test_format_prompt_score() {
        let criteria = vec![
            "極めて不満".to_string(),
            "不満".to_string(),
            "満足".to_string(),
            "大変満足".to_string(),
        ];
        let question = Question::new_score("顧客満足度を評価せよ。", criteria);
        let formatted = format_prompt(&question).unwrap();

        assert_eq!(formatted.prefix, "State: ");
        assert_eq!(formatted.option_keys, vec!["0", "1", "2", "3"]);
        assert!(
            formatted
                .suffix
                .contains("Instructions: 顧客満足度を評価せよ。")
        );
        assert!(
            formatted
                .suffix
                .contains("Criteria: [OP] 極めて不満 [OP] 不満 [OP] 満足 [OP] 大変満足")
        );
    }

    #[test]
    fn test_format_prompt_noul() {
        let question = Question::new_noul("この言明は妥当であるか判定せよ。");
        let formatted = format_prompt(&question).unwrap();

        assert_eq!(formatted.prefix, "State: ");
        assert_eq!(formatted.option_keys, vec!["true", "false"]);
        assert!(
            formatted
                .suffix
                .contains("Instructions: この言明は妥当であるか判定せよ。")
        );
        assert!(
            formatted
                .suffix
                .contains("Criteria: [OP] 真 (True) [OP] 偽 (False)")
        );
    }

    #[test]
    fn test_format_prompt_invalid_choice_empty() {
        let question = Question::new_choice("空の候補", IndexMap::new());
        assert!(format_prompt(&question).is_err());
    }

    #[test]
    fn test_format_prompt_invalid_score_count() {
        let question = Question::new_score("1段階のみ", vec!["評価".to_string()]);
        assert!(format_prompt(&question).is_err());
    }
}
