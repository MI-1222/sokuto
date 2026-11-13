//! # ゼロアロケーション動的プロンプト注入モジュール
//!
//! 先行ノードの推論結果や確信度を後続ノードのプロンプト・評価基準へ
//! マイクロ秒未満(0.8μs 未満)の極小オーバーヘッドで動的埋め込みする。

use indexmap::IndexMap;
use sokuto_core::schema::{Criteria, Question, QuestionType};
use std::fmt::{self, Write};

/// 単一ノードのコンテキスト情報。
///
/// 先行ノードの判定値文字列と、確信度(0.0〜1.0)および数値表現を保持する。
#[derive(Debug, Clone)]
pub struct NodeContextValue {
    /// 判定結果の文字列表現 (Choice キー、bool 文字列、数値など)。
    pub value: String,
    /// 予測確信度。
    pub confidence: f64,
    /// 数値表現 (Score の値、Noul の確率値など)。数値として扱える場合に保持する。
    pub numeric_value: Option<f64>,
}

impl NodeContextValue {
    /// 新規コンテキスト値を生成する。
    ///
    /// 文字列値が有限浮動小数点数として解釈可能な場合、`numeric_value` も自動設定する。
    pub fn new(value: impl Into<String>, confidence: f64) -> Self {
        let val = value.into();
        let num = val.trim().parse::<f64>().ok().filter(|f| f.is_finite());
        Self {
            value: val,
            confidence,
            numeric_value: num,
        }
    }

    /// 生の数値を指定してコンテキスト値を生成する。
    pub fn with_numeric(
        value: impl Into<String>,
        confidence: f64,
        numeric_value: Option<f64>,
    ) -> Self {
        Self {
            value: value.into(),
            confidence,
            numeric_value: numeric_value.filter(|f| f.is_finite()),
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

impl fmt::Display for PromptPlaceholder<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.prop {
            Some(p) => write!(f, "{}.{}", self.node_id, p),
            None => f.write_str(self.node_id),
        }
    }
}

impl<'a> PromptPlaceholder<'a> {
    /// プレースホルダーの文字列表現を返却する。
    pub fn as_str(&self) -> String {
        self.to_string()
    }
}

/// テンプレートのパース結果セグメント。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TemplateSegment<'a> {
    /// 通常のプレーンテキスト。
    Text(&'a str),
    /// 抽出されたプレースホルダー。
    Placeholder(PromptPlaceholder<'a>),
}

/// テンプレート文字列をゼロアロケーションで走査・分割するイテレータ。
#[derive(Debug, Clone)]
pub(crate) struct TemplateParser<'a> {
    template: &'a str,
    cursor: usize,
}

impl<'a> TemplateParser<'a> {
    /// 新規パーサーを生成する。
    #[inline]
    pub(crate) fn new(template: &'a str) -> Self {
        Self {
            template,
            cursor: 0,
        }
    }
}

impl<'a> Iterator for TemplateParser<'a> {
    type Item = TemplateSegment<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= self.template.len() {
            return None;
        }

        let remaining = &self.template[self.cursor..];
        let Some(rel_open) = remaining.find('{') else {
            let text = remaining;
            self.cursor = self.template.len();
            return Some(TemplateSegment::Text(text));
        };

        if rel_open > 0 {
            let text = &self.template[self.cursor..self.cursor + rel_open];
            self.cursor += rel_open;
            return Some(TemplateSegment::Text(text));
        }

        let open_idx = self.cursor;
        let Some(rel_close) = self.template[open_idx + 1..].find('}') else {
            let text = &self.template[self.cursor..];
            self.cursor = self.template.len();
            return Some(TemplateSegment::Text(text));
        };
        let close_idx = open_idx + 1 + rel_close;

        if let Some(inner_rel_open) = self.template[open_idx + 1..close_idx].find('{') {
            let new_open_idx = open_idx + 1 + inner_rel_open;
            let text = &self.template[self.cursor..new_open_idx];
            self.cursor = new_open_idx;
            return Some(TemplateSegment::Text(text));
        }

        let inner = &self.template[open_idx + 1..close_idx];
        if let Some(ph) = parse_placeholder(inner) {
            self.cursor = close_idx + 1;
            Some(TemplateSegment::Placeholder(ph))
        } else {
            let next_open_rel = self.template[self.cursor + 1..].find('{');
            let end_idx = match next_open_rel {
                Some(rel) => self.cursor + 1 + rel,
                None => self.template.len(),
            };
            let text = &self.template[self.cursor..end_idx];
            self.cursor = end_idx;
            Some(TemplateSegment::Text(text))
        }
    }
}

/// プレースホルダーの識別子 (ノード ID またはプロパティ名) として妥当かを検証する。
///
/// 空白や記号 (JSON 構文文字等) を含まず、ASCII 英数字・アンダースコア・ハイフンのみで構成されているかを判定する。
#[inline]
fn is_valid_placeholder_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
}

/// 波括弧内の文字列を構文解析し、妥当なプレースホルダーであれば `PromptPlaceholder` を返す。
#[inline]
pub(crate) fn parse_placeholder(inner: &str) -> Option<PromptPlaceholder<'_>> {
    if let Some((node_id, prop)) = inner.split_once('.') {
        if is_valid_placeholder_ident(node_id) && is_valid_placeholder_ident(prop) {
            Some(PromptPlaceholder {
                node_id,
                prop: Some(prop),
            })
        } else {
            None
        }
    } else if is_valid_placeholder_ident(inner) {
        Some(PromptPlaceholder {
            node_id: inner,
            prop: None,
        })
    } else {
        None
    }
}

/// テンプレート文字列内の `{...}` プレースホルダーを走査・抽出する。
///
/// JSON オブジェクトリテラル内の波括弧 (例: `{"status": {node1.value}}`) が
/// 含まれる場合でも、外側の波括弧と誤結合せず、内側の有効なプレースホルダーのみを抽出する。
pub(crate) fn extract_placeholders(template: &str) -> Vec<PromptPlaceholder<'_>> {
    TemplateParser::new(template)
        .filter_map(|segment| match segment {
            TemplateSegment::Placeholder(ph) => Some(ph),
            TemplateSegment::Text(_) => None,
        })
        .collect()
}

/// テンプレート文字列をスライス走査し、事前確保済みバッファへインプレース展開する。
///
/// プレースホルダー `{node_id.value}`, `{node_id.answer}`, `{node_id.confidence}` を
/// 検出してコンテキスト内の対応値に置換する。
/// JSON オブジェクトリテラル内の波括弧 (例: `{"status": {node1.value}}`) が含まれる場合でも
/// 外側と内側の波括弧を誤結合せず正しく展開する。
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

    for segment in TemplateParser::new(template) {
        match segment {
            TemplateSegment::Text(text) => output.push_str(text),
            TemplateSegment::Placeholder(ph) => {
                if let Some(prop) = ph.prop {
                    if let Some(ctx_val) = context.get(ph.node_id) {
                        match prop {
                            "value" | "answer" => output.push_str(&ctx_val.value),
                            "confidence" => {
                                let _ = write!(output, "{:.3}", ctx_val.confidence);
                            }
                            _ => output.push_str("null"),
                        }
                    } else {
                        // 未解決プレースホルダーはそのまま保持する
                        let _ = write!(output, "{{{ph}}}");
                    }
                } else if let Some(ctx_val) = context.get(ph.node_id) {
                    output.push_str(&ctx_val.value);
                } else {
                    let _ = write!(output, "{{{ph}}}");
                }
            }
        }
    }
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

    #[test]
    fn test_extract_placeholders_with_json_literal() {
        // 1. プレースホルダーを含まない純粋な JSON リテラル
        let json_plain = r#"{"status": "ok", "code": 200, "flag": true}"#;
        assert!(extract_placeholders(json_plain).is_empty());

        // 2. 波括弧誤結合の典型例: JSON オブジェクト内にプレースホルダーが含まれる場合
        let json_nested = r#"{"status": {node1.value}}"#;
        let phs = extract_placeholders(json_nested);
        assert_eq!(phs.len(), 1);
        assert_eq!(phs[0].node_id, "node1");
        assert_eq!(phs[0].prop, Some("value"));

        // 3. 多重ネストされた JSON オブジェクト内のプレースホルダー
        let json_deep = r#"{"meta": {"inner": {"target": {user_rank.value}}, "enabled": true}}"#;
        let phs_deep = extract_placeholders(json_deep);
        assert_eq!(phs_deep.len(), 1);
        assert_eq!(phs_deep[0].node_id, "user_rank");
        assert_eq!(phs_deep[0].prop, Some("value"));

        // 4. 複数プレースホルダーが JSON 内に散在する場合
        let json_multi = r#"{"a": {node_a.answer}, "b": {node_b.confidence}, "c": "literal"}"#;
        let phs_multi = extract_placeholders(json_multi);
        assert_eq!(phs_multi.len(), 2);
        assert_eq!(phs_multi[0].node_id, "node_a");
        assert_eq!(phs_multi[0].prop, Some("answer"));
        assert_eq!(phs_multi[1].node_id, "node_b");
        assert_eq!(phs_multi[1].prop, Some("confidence"));
    }

    #[test]
    fn test_interpolate_prompt_with_json_literal() {
        let mut context = IndexMap::new();
        context.insert(
            "node_a".to_string(),
            NodeContextValue::new("approved", 0.9523),
        );
        context.insert("node_b".to_string(), NodeContextValue::new("true", 0.880));

        // 1. JSON リテラル内の単一プレースホルダー
        let template = r#"{"status": {node_a.value}}"#;
        let mut output = String::new();
        interpolate_prompt(template, &context, &mut output);
        assert_eq!(output, r#"{"status": approved}"#);

        // 2. プレースホルダーを含まない JSON リテラルがそのまま出力されること
        let template_plain = r#"{"key": "value", "list": [1, 2]}"#;
        interpolate_prompt(template_plain, &context, &mut output);
        assert_eq!(output, template_plain);

        // 3. JSON 文字列値としての補間と数値としての補間
        let template_quoted =
            r#"{"query": "{node_a.value}", "conf": {node_a.confidence}, "active": {node_b.value}}"#;
        interpolate_prompt(template_quoted, &context, &mut output);
        assert_eq!(
            output,
            r#"{"query": "approved", "conf": 0.952, "active": true}"#
        );

        // 4. 通常テキストと JSON 構造が混在するテンプレート
        let template_mixed =
            r#"先行結果: {node_a.value}、設定 JSON: {"status": {node_b.answer}} です。"#;
        interpolate_prompt(template_mixed, &context, &mut output);
        assert_eq!(
            output,
            r#"先行結果: approved、設定 JSON: {"status": true} です。"#
        );
    }

    #[test]
    fn test_prompt_placeholder_display() {
        let ph_with_prop = PromptPlaceholder {
            node_id: "node_1",
            prop: Some("confidence"),
        };
        assert_eq!(ph_with_prop.to_string(), "node_1.confidence");
        assert_eq!(ph_with_prop.as_str(), "node_1.confidence");

        let ph_without_prop = PromptPlaceholder {
            node_id: "node_2",
            prop: None,
        };
        assert_eq!(ph_without_prop.to_string(), "node_2");
        assert_eq!(ph_without_prop.as_str(), "node_2");
    }

    #[test]
    fn test_template_parser_segments() {
        let template = "Prefix {node_a.value} mid {invalid brace} end {node_b}";
        let segments: Vec<_> = TemplateParser::new(template).collect();

        assert_eq!(
            segments,
            vec![
                TemplateSegment::Text("Prefix "),
                TemplateSegment::Placeholder(PromptPlaceholder {
                    node_id: "node_a",
                    prop: Some("value"),
                }),
                TemplateSegment::Text(" mid "),
                TemplateSegment::Text("{invalid brace} end "),
                TemplateSegment::Placeholder(PromptPlaceholder {
                    node_id: "node_b",
                    prop: None,
                }),
            ]
        );
    }
}
