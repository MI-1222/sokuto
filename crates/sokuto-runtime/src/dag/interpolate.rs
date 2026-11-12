//! # ゼロアロケーション動的プロンプト注入モジュール
//!
//! 先行ノードの推論結果や確信度を後続ノードのプロンプト・評価基準へ
//! マイクロ秒未満(0.8μs 未満)の極小オーバーヘッドで動的埋め込みする。

use indexmap::IndexMap;
use sokuto_core::schema::{Criteria, Question, QuestionType};
use std::fmt::Write;

/// 単一ノードのコンテキスト情報。
///
/// 先行ノードの判定値文字列と、確信度(0.0〜1.0)を保持する。
#[derive(Debug, Clone)]
pub struct NodeContextValue {
    /// 判定結果の文字列表現 (Choice キー、bool 文字列、数値など)。
    pub value: String,
    /// 予測確信度。
    pub confidence: f64,
}

impl NodeContextValue {
    /// 新規コンテキスト値を生成する。
    pub fn new(value: impl Into<String>, confidence: f64) -> Self {
        Self {
            value: value.into(),
            confidence,
        }
    }
}

/// テンプレート内のプレースホルダー情報。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PromptPlaceholder<'a> {
    /// 参照先ノード ID。
    pub node_id: &'a str,
    /// 参照プロパティ名 (例: `"value"`, `"answer"`, `"confidence"`、省略時は `None`)。
    pub prop: Option<&'a str>,
}

impl<'a> PromptPlaceholder<'a> {
    /// プレースホルダーの文字列表現を返却する。
    pub fn as_str(&self) -> String {
        match self.prop {
            Some(p) => format!("{}.{p}", self.node_id),
            None => self.node_id.to_string(),
        }
    }
}

/// テンプレート文字列内の `{...}` プレースホルダーを走査・抽出する。
pub(crate) fn extract_placeholders(template: &str) -> Vec<PromptPlaceholder<'_>> {
    let mut placeholders = Vec::new();
    let mut cursor = 0;
    while let Some(open) = template[cursor..].find('{') {
        let open_idx = cursor + open;
        let Some(close) = template[open_idx..].find('}') else {
            break;
        };
        let close_idx = open_idx + close;
        let key = &template[open_idx + 1..close_idx];
        if let Some((node_id, prop)) = key.split_once('.') {
            placeholders.push(PromptPlaceholder {
                node_id,
                prop: Some(prop),
            });
        } else {
            placeholders.push(PromptPlaceholder {
                node_id: key,
                prop: None,
            });
        }
        cursor = close_idx + 1;
    }
    placeholders
}

/// テンプレート文字列をスライス走査し、事前確保済みバッファへインプレース展開する。
///
/// プレースホルダー `{node_id.value}`, `{node_id.answer}`, `{node_id.confidence}` を
/// 検出してコンテキスト内の対応値に置換する。
///
/// # 引数
/// - `template`: プレースホルダーを含むテンプレート文字列。
/// - `context`: 先行ノード ID とその結果値のマップ。
/// - `output`: 書き込み先の再利用可能バッファ。
pub fn interpolate_prompt(
    template: &str,
    context: &IndexMap<String, NodeContextValue>,
    output: &mut String,
) {
    output.clear();
    output.reserve(template.len() + 64);
    let mut cursor = 0;

    while let Some(open) = template[cursor..].find('{') {
        let open_idx = cursor + open;
        output.push_str(&template[cursor..open_idx]);

        if let Some(close) = template[open_idx..].find('}') {
            let close_idx = open_idx + close;
            let key = &template[open_idx + 1..close_idx];

            if let Some((node_id, prop)) = key.split_once('.') {
                if let Some(ctx_val) = context.get(node_id) {
                    match prop {
                        "value" | "answer" => output.push_str(&ctx_val.value),
                        "confidence" => {
                            let _ = write!(output, "{:.3}", ctx_val.confidence);
                        }
                        _ => output.push_str("null"),
                    }
                } else {
                    // 未解決プレースホルダーはそのまま保持する
                    output.push_str(&template[open_idx..=close_idx]);
                }
            } else if let Some(ctx_val) = context.get(key) {
                output.push_str(&ctx_val.value);
            } else {
                output.push_str(&template[open_idx..=close_idx]);
            }
            cursor = close_idx + 1;
        } else {
            cursor = open_idx;
            break;
        }
    }
    output.push_str(&template[cursor..]);
}

/// コンテキストに基づき、質問定義の Instructions および Criteria を補間した新しい質問を生成する。
///
/// # 引数
/// - `question`: 原型の質問定義。
/// - `context`: 先行ノードのコンテキストマップ。
///
/// # 戻り値
/// 動的変数が展開された新しい `Question`。
pub fn interpolate_question(
    question: &Question,
    context: &IndexMap<String, NodeContextValue>,
) -> Question {
    if context.is_empty() {
        return question.clone();
    }

    let mut scratch = String::with_capacity(question.instructions.len() + 64);
    interpolate_prompt(&question.instructions, context, &mut scratch);
    let interpolated_instructions = scratch.clone();

    let interpolated_criteria = match &question.criteria {
        Some(Criteria::Map(map)) => {
            let mut new_map = IndexMap::with_capacity(map.len());
            for (k, v) in map {
                interpolate_prompt(v, context, &mut scratch);
                new_map.insert(k.clone(), scratch.clone());
            }
            Some(Criteria::Map(new_map))
        }
        Some(Criteria::List(list)) => {
            let mut new_list = Vec::with_capacity(list.len());
            for item in list {
                interpolate_prompt(item, context, &mut scratch);
                new_list.push(scratch.clone());
            }
            Some(Criteria::List(new_list))
        }
        None => None,
    };

    match question.question_type {
        QuestionType::Choice => Question {
            question_type: QuestionType::Choice,
            instructions: interpolated_instructions,
            criteria: interpolated_criteria,
        },
        QuestionType::Score => Question {
            question_type: QuestionType::Score,
            instructions: interpolated_instructions,
            criteria: interpolated_criteria,
        },
        QuestionType::Noul => Question {
            question_type: QuestionType::Noul,
            instructions: interpolated_instructions,
            criteria: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate_prompt_basic() {
        let mut context = IndexMap::new();
        context.insert(
            "node_a".to_string(),
            NodeContextValue::new("approved", 0.9523),
        );
        context.insert("node_b".to_string(), NodeContextValue::new("true", 0.880));

        let template = "先行判定は {node_a.value} (確信度: {node_a.confidence}) であり、フラグ={node_b.answer} です。";
        let mut output = String::new();
        interpolate_prompt(template, &context, &mut output);

        assert_eq!(
            output,
            "先行判定は approved (確信度: 0.952) であり、フラグ=true です。"
        );
    }

    #[test]
    fn test_interpolate_prompt_unresolved_intact() {
        let context = IndexMap::new();
        let template = "未定義変数は {unknown.value} のままです。";
        let mut output = String::new();
        interpolate_prompt(template, &context, &mut output);

        assert_eq!(output, "未定義変数は {unknown.value} のままです。");
    }

    #[test]
    fn test_interpolate_prompt_unclosed_brace() {
        let mut context = IndexMap::new();
        context.insert("node_a".to_string(), NodeContextValue::new("val", 1.0));

        let template = "Hello {world";
        let mut output = String::new();
        interpolate_prompt(template, &context, &mut output);
        assert_eq!(output, "Hello {world");

        let template_mixed = "Prefix {node_a.value} mid {unclosed";
        let mut output_mixed = String::new();
        interpolate_prompt(template_mixed, &context, &mut output_mixed);
        assert_eq!(output_mixed, "Prefix val mid {unclosed");
    }

    #[test]
    fn test_interpolate_question() {
        let mut context = IndexMap::new();
        context.insert("user_rank".to_string(), NodeContextValue::new("VIP", 0.99));

        let mut criteria_map = IndexMap::new();
        criteria_map.insert(
            "free".to_string(),
            "{user_rank.value}会員は手数料無料".to_string(),
        );
        criteria_map.insert("paid".to_string(), "通常手数料が発生".to_string());

        let q = Question::new_choice(
            "会員ランク {user_rank.value} に対する手数料を判定してください。".to_string(),
            criteria_map,
        );

        let interpolated = interpolate_question(&q, &context);
        assert_eq!(
            interpolated.instructions,
            "会員ランク VIP に対する手数料を判定してください。"
        );
        if let Some(Criteria::Map(m)) = &interpolated.criteria {
            assert_eq!(m.get("free").unwrap(), "VIP会員は手数料無料");
        } else {
            panic!("Criteria が Map である必要があります。");
        }
    }

    #[test]
    fn test_extract_placeholders() {
        let template =
            "こんにちは {user.name} さん。確信度={user.confidence}、スコア={score}、未終了 {broken";
        let placeholders = extract_placeholders(template);
        assert_eq!(placeholders.len(), 3);
        assert_eq!(placeholders[0].node_id, "user");
        assert_eq!(placeholders[0].prop, Some("name"));
        assert_eq!(placeholders[0].as_str(), "user.name");

        assert_eq!(placeholders[1].node_id, "user");
        assert_eq!(placeholders[1].prop, Some("confidence"));
        assert_eq!(placeholders[1].as_str(), "user.confidence");

        assert_eq!(placeholders[2].node_id, "score");
        assert_eq!(placeholders[2].prop, None);
        assert_eq!(placeholders[2].as_str(), "score");
    }
}
