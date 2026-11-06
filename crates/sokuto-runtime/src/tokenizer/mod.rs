//! # 高速トークナイザー統合モジュール
//!
//! Hugging Face `tokenizers` クレートをラップし、`tokenizer.json` のロード、
//! 特殊トークンおよび `[OP]` マーカーの検証、State 優先トランケーション (State-Priority Truncation)、
//! ならびに ONNX Runtime 入力用テンソルバッファの高速生成を行う。

pub mod batch;
pub mod prompt;

pub use batch::BatchTokenizedQuestions;

use std::path::Path;

use sokuto_core::contract::model_spec::{MAX_SEQUENCE_LENGTH, TOKEN_OPTION_MARKER};
use sokuto_core::schema::Question;

use crate::error::{Result, RuntimeError};
use prompt::format_prompt;

/// 単一の質問に対するトークナイズ結果構造体。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizedQuestion {
    /// 入力トークン列テンソル用データ (`int64[sequence_length]`)。
    pub input_ids: Vec<i64>,
    /// アテンションマスクテンソル用データ (`int64[sequence_length]`)。
    pub attention_mask: Vec<i64>,
    /// 候補出現位置インデックステンソル用データ (`int64[num_options]`)。
    pub op_indices: Vec<i64>,
    /// プロンプト出現順に整列した候補識別子キー列。
    pub option_keys: Vec<String>,
}

/// Jev システム用高速トークナイザー。
#[derive(Debug, Clone)]
pub struct JevTokenizer {
    /// 内部の Hugging Face トークナイザーインスタンス。
    inner: tokenizers::Tokenizer,
    /// `[OP]` オプションマーカートークンの ID。
    op_token_id: u32,
    /// パディングトークンの ID。
    pad_token_id: Option<u32>,
    /// 先頭特殊トークン ID 列 (`[CLS]` または `<s>` 等)。
    leading_special_tokens: Vec<u32>,
    /// 末尾特殊トークン ID 列 (`[SEP]` または `</s>` 等)。
    trailing_special_tokens: Vec<u32>,
    /// 固定プレフィックス `"State: "` のキャッシュ済みトークン ID 列。
    state_prefix_tokens: Vec<u32>,
    /// 許容最大入力系列長。
    max_sequence_length: usize,
}

impl JevTokenizer {
    /// トークナイザー定義ファイル (`tokenizer.json`) からインスタンスを初期化する。
    ///
    /// 最大系列長にはデフォルトの `MAX_SEQUENCE_LENGTH` (8192) が適用される。
    ///
    /// # エラー
    /// ファイルのロード失敗、`[OP]` トークン不在、または `[OP]` トークン分割テストに失敗した場合にエラーを返す。
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_file_with_max_length(path, MAX_SEQUENCE_LENGTH)
    }

    /// 最大系列長を指定してトークナイザー定義ファイルからインスタンスを初期化する。
    ///
    /// # エラー
    /// ファイルのロード失敗、`[OP]` トークン不在、または `[OP]` トークン分割テストに失敗した場合にエラーを返す。
    pub fn from_file_with_max_length(
        path: impl AsRef<Path>,
        max_sequence_length: usize,
    ) -> Result<Self> {
        let path_ref = path.as_ref();
        let inner = tokenizers::Tokenizer::from_file(path_ref).map_err(|e| {
            RuntimeError::TokenizerLoadError(format!(
                "ファイル '{}' のロードに失敗しました: {e}",
                path_ref.display()
            ))
        })?;

        Self::from_tokenizer_with_max_length(inner, max_sequence_length)
    }

    /// 既存の `tokenizers::Tokenizer` インスタンスから初期化する。
    pub fn from_tokenizer_with_max_length(
        inner: tokenizers::Tokenizer,
        max_sequence_length: usize,
    ) -> Result<Self> {
        // [OP] マーカーの ID 取得
        let op_token_id =
            inner
                .token_to_id(TOKEN_OPTION_MARKER)
                .ok_or(RuntimeError::MissingSpecialToken {
                    token: TOKEN_OPTION_MARKER,
                })?;

        // [OP] トークンが単一トークンとしてエンコードされるかの分割テスト検証
        let op_encoding = inner
            .encode(TOKEN_OPTION_MARKER, false)
            .map_err(|e| RuntimeError::TokenizerEncodeError(e.to_string()))?;
        let op_ids = op_encoding.get_ids();
        if op_ids != [op_token_id] {
            return Err(RuntimeError::InvalidOptionMarkerEncoding {
                expected: op_token_id,
                actual: op_ids.to_vec(),
            });
        }

        // 先頭特殊トークン (CLS / BOS) の同定。
        // ModernBERT-ja は `<cls>` を使用し、BERT系は `[CLS]`、RoBERTa/Gemma等は `<s>` / `[BOS]` を使用する。
        let leading_special_tokens: Vec<u32> = inner
            .token_to_id("<cls>")
            .or_else(|| inner.token_to_id("[CLS]"))
            .or_else(|| inner.token_to_id("<s>"))
            .or_else(|| inner.token_to_id("[BOS]"))
            .map(|id| vec![id])
            .unwrap_or_default();

        // 末尾特殊トークン (SEP / EOS) の同定。
        // ModernBERT-ja は `<sep>` を使用し、BERT系は `[SEP]`、RoBERTa/Gemma等は `</s>` / `[EOS]` を使用する。
        let trailing_special_tokens: Vec<u32> = inner
            .token_to_id("<sep>")
            .or_else(|| inner.token_to_id("[SEP]"))
            .or_else(|| inner.token_to_id("</s>"))
            .or_else(|| inner.token_to_id("[EOS]"))
            .map(|id| vec![id])
            .unwrap_or_default();

        // パディングトークン (PAD) の同定。
        let pad_token_id: Option<u32> = inner
            .token_to_id("<pad>")
            .or_else(|| inner.token_to_id("[PAD]"));

        // 固定 Prefix ("State: ") のトークン列を事前エンコード・キャッシュ
        let state_prefix_encoding = inner
            .encode(prompt::PREFIX_STATE, false)
            .map_err(|e| RuntimeError::TokenizerEncodeError(e.to_string()))?;
        let state_prefix_tokens = state_prefix_encoding.get_ids().to_vec();

        Ok(Self {
            inner,
            op_token_id,
            pad_token_id,
            leading_special_tokens,
            trailing_special_tokens,
            state_prefix_tokens,
            max_sequence_length,
        })
    }

    /// `[OP]` マーカーのトークン ID を取得する。
    pub fn op_token_id(&self) -> u32 {
        self.op_token_id
    }

    /// パディングトークン ID を取得する。
    pub fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }

    /// 先頭特殊トークン ID 列を取得する。
    pub fn leading_special_tokens(&self) -> &[u32] {
        &self.leading_special_tokens
    }

    /// 末尾特殊トークン ID 列を取得する。
    pub fn trailing_special_tokens(&self) -> &[u32] {
        &self.trailing_special_tokens
    }

    /// 固定プレフィックス `"State: "` のキャッシュ済みトークン ID 列を取得する。
    pub fn state_prefix_tokens(&self) -> &[u32] {
        &self.state_prefix_tokens
    }

    /// 設定されている最大系列長を取得する。
    pub fn max_sequence_length(&self) -> usize {
        self.max_sequence_length
    }

    /// 内部の `tokenizers::Tokenizer` への参照を取得する。
    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }

    /// 文字列を特殊トークン付加なしでエンコードする。
    fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| RuntimeError::TokenizerEncodeError(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// State 文字列を事前にトークナイズする (プレフィックス共有用)。
    pub fn encode_state(&self, state: &str) -> Result<Vec<u32>> {
        self.encode_text(state)
    }

    /// 単一の State と Question から ONNX 入力用バッファを生成する。
    pub fn encode_question(&self, state: &str, question: &Question) -> Result<TokenizedQuestion> {
        self.encode_question_with_max_len(state, question, self.max_sequence_length)
    }

    /// 系列長制約を明示指定して、単一の State と Question をエンコードする。
    pub fn encode_question_with_max_len(
        &self,
        state: &str,
        question: &Question,
        max_length: usize,
    ) -> Result<TokenizedQuestion> {
        let state_ids = self.encode_state(state)?;
        self.encode_with_pretokenized_state(&state_ids, question, max_length)
    }

    /// 事前エンコード済み State トークン列を用いてエンコードする (State 優先トランケーション適用)。
    ///
    /// # State 優先トランケーションの不変条件
    /// 文末の Criteria (すべての `[OP]` マーカー群) および Instructions を絶対に保護し、
    /// 系列長が上限を超える場合は State のみ末尾からスライスして切り詰める。
    ///
    /// # エラー
    /// - 固定部 (特殊トークン + Prefix + Instructions + Criteria) が `max_length` 以上の場合は `PromptExceedsMaxLength`。
    /// - 検出された `[OP]` マーカー数と候補数が不一致の場合は `OptionMarkerCountMismatch`。
    pub fn encode_with_pretokenized_state(
        &self,
        state_ids: &[u32],
        question: &Question,
        max_length: usize,
    ) -> Result<TokenizedQuestion> {
        let formatted = format_prompt(question)?;

        // キャッシュされた固定 Prefix ("State: ") のトークン列を再利用
        let prefix_ids = &self.state_prefix_tokens;
        let suffix_ids = self.encode_text(&formatted.suffix)?;

        let fixed_len = self.leading_special_tokens.len()
            + prefix_ids.len()
            + suffix_ids.len()
            + self.trailing_special_tokens.len();

        if fixed_len >= max_length {
            return Err(RuntimeError::PromptExceedsMaxLength {
                fixed_len,
                max_len: max_length,
            });
        }

        // State を収容可能な残余スロット数を算出
        let allowed_state_len = max_length - fixed_len;
        let truncated_state_ids = if state_ids.len() > allowed_state_len {
            &state_ids[..allowed_state_len]
        } else {
            state_ids
        };

        let total_len = fixed_len + truncated_state_ids.len();
        let mut input_ids = Vec::with_capacity(total_len);

        // トークン列の結合: [CLS] + Prefix + Truncated_State + Suffix + [SEP]
        for &id in &self.leading_special_tokens {
            input_ids.push(id as i64);
        }
        for &id in prefix_ids {
            input_ids.push(id as i64);
        }
        for &id in truncated_state_ids {
            input_ids.push(id as i64);
        }
        for &id in &suffix_ids {
            input_ids.push(id as i64);
        }
        for &id in &self.trailing_special_tokens {
            input_ids.push(id as i64);
        }

        let attention_mask = vec![1i64; total_len];

        // [OP] マーカー出現インデックスの高速抽出
        let op_token_id_i64 = self.op_token_id as i64;
        let mut op_indices = Vec::with_capacity(formatted.option_keys.len());

        for (idx, &token_id) in input_ids.iter().enumerate() {
            if token_id == op_token_id_i64 {
                op_indices.push(idx as i64);
            }
        }

        // 候補数との完全一致検証
        if op_indices.len() != formatted.option_keys.len() {
            return Err(RuntimeError::OptionMarkerCountMismatch {
                detected: op_indices.len(),
                expected: formatted.option_keys.len(),
            });
        }

        Ok(TokenizedQuestion {
            input_ids,
            attention_mask,
            op_indices,
            option_keys: formatted.option_keys,
        })
    }

    /// 単一の質問定義から `QueryEvaluator` 入力用トークン列および `[OP]` マーカー位置を生成する。
    ///
    /// # 戻り値
    /// - `TokenizedQuestion`: Query 部分の `input_ids`、`attention_mask`、`op_indices`、`option_keys`。
    pub fn encode_query(&self, question: &Question) -> Result<TokenizedQuestion> {
        self.encode_query_with_max_len(question, self.max_sequence_length)
    }

    /// 系列長制約を指定して、単一の質問定義から `QueryEvaluator` 入力用トークン列を生成する。
    pub fn encode_query_with_max_len(
        &self,
        question: &Question,
        max_length: usize,
    ) -> Result<TokenizedQuestion> {
        let formatted = format_prompt(question)?;
        let suffix_ids = self.encode_text(&formatted.suffix)?;

        let mut query_tokens = Vec::with_capacity(
            self.leading_special_tokens.len()
                + suffix_ids.len()
                + self.trailing_special_tokens.len(),
        );
        query_tokens.extend_from_slice(&self.leading_special_tokens);
        query_tokens.extend_from_slice(&suffix_ids);
        query_tokens.extend_from_slice(&self.trailing_special_tokens);

        if query_tokens.len() > max_length {
            return Err(RuntimeError::PromptExceedsMaxLength {
                fixed_len: query_tokens.len(),
                max_len: max_length,
            });
        }

        let seq_len = query_tokens.len();
        let input_ids: Vec<i64> = query_tokens.iter().map(|&id| id as i64).collect();
        let attention_mask = vec![1i64; seq_len];

        let mut op_indices = Vec::with_capacity(formatted.option_keys.len());
        let op_token_id_i64 = self.op_token_id as i64;
        for (idx, &token_id) in input_ids.iter().enumerate() {
            if token_id == op_token_id_i64 {
                op_indices.push(idx as i64);
            }
        }

        if op_indices.len() != formatted.option_keys.len() {
            return Err(RuntimeError::OptionMarkerCountMismatch {
                detected: op_indices.len(),
                expected: formatted.option_keys.len(),
            });
        }

        Ok(TokenizedQuestion {
            input_ids,
            attention_mask,
            op_indices,
            option_keys: formatted.option_keys,
        })
    }

    /// `State` 文字列を `StateEncoder` 入力用トークン列 (`input_ids`, `attention_mask`) としてエンコードする。
    pub fn encode_state_tensor(&self, state: &str) -> Result<(Vec<i64>, Vec<i64>)> {
        let text_tokens = self.encode_state(state)?;
        let mut tokens = Vec::with_capacity(
            self.leading_special_tokens.len()
                + self.state_prefix_tokens.len()
                + text_tokens.len()
                + self.trailing_special_tokens.len(),
        );
        tokens.extend_from_slice(&self.leading_special_tokens);
        tokens.extend_from_slice(&self.state_prefix_tokens);
        tokens.extend_from_slice(&text_tokens);
        tokens.extend_from_slice(&self.trailing_special_tokens);

        if tokens.len() > self.max_sequence_length {
            tokens.truncate(self.max_sequence_length);
        }

        let seq_len = tokens.len();
        let input_ids: Vec<i64> = tokens.iter().map(|&t| t as i64).collect();
        let attention_mask = vec![1i64; seq_len];
        Ok((input_ids, attention_mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;
    use std::path::PathBuf;

    /// ワークスペース内の default tokenizer.json のパスを取得するヘルパー。
    fn default_tokenizer_path() -> Option<PathBuf> {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = manifest_dir
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("models")
            .join("default")
            .join("tokenizer.json");

        if !path.exists() {
            if std::env::var("CI").is_ok() {
                panic!(
                    "CI 環境で必須トークナイザーファイルが見つかりません: {:?}",
                    path
                );
            }
            eprintln!("スキップ: tokenizer.json が見つかりません: {:?}", path);
            return None;
        }

        Some(path)
    }

    #[test]
    fn test_load_default_tokenizer() {
        let Some(path) = default_tokenizer_path() else {
            return;
        };

        let tokenizer = JevTokenizer::from_file(&path).unwrap();
        assert_eq!(tokenizer.max_sequence_length(), 8192);
        assert!(!tokenizer.leading_special_tokens().is_empty());
        assert!(!tokenizer.trailing_special_tokens().is_empty());
        assert!(!tokenizer.state_prefix_tokens().is_empty());
        assert!(tokenizer.pad_token_id().is_some());

        // [OP] トークン ID が正しく解決されていること
        let op_id = tokenizer.op_token_id();
        assert!(op_id > 0);
    }

    #[test]
    fn test_tokenize_choice_question() {
        let Some(path) = default_tokenizer_path() else {
            return;
        };

        let tokenizer = JevTokenizer::from_file(&path).unwrap();

        let mut criteria = IndexMap::new();
        criteria.insert(
            "card_arrival".to_string(),
            "Card delivery status".to_string(),
        );
        criteria.insert("lost_card".to_string(), "Reporting lost card".to_string());
        criteria.insert("pin_reset".to_string(), "Resetting PIN code".to_string());

        let question = Question::new_choice("意図を分類せよ。", criteria);
        let state = "I lost my card yesterday.";

        let tokenized = tokenizer.encode_question(state, &question).unwrap();

        assert_eq!(tokenized.input_ids.len(), tokenized.attention_mask.len());
        assert_eq!(
            tokenized.option_keys,
            vec!["card_arrival", "lost_card", "pin_reset"]
        );
        assert_eq!(tokenized.op_indices.len(), 3);

        // 各 [OP] 位置のトークン ID が op_token_id と一致すること
        let op_id_i64 = tokenizer.op_token_id() as i64;
        for &idx in &tokenized.op_indices {
            assert_eq!(tokenized.input_ids[idx as usize], op_id_i64);
        }
    }

    #[test]
    fn test_tokenize_noul_question() {
        let Some(path) = default_tokenizer_path() else {
            return;
        };

        let tokenizer = JevTokenizer::from_file(&path).unwrap();
        let question = Question::new_noul("言明「空は青い」の真偽を判定せよ。");
        let state = "The sky is blue.";

        let tokenized = tokenizer.encode_question(state, &question).unwrap();

        assert_eq!(tokenized.option_keys, vec!["true", "false"]);
        assert_eq!(tokenized.op_indices.len(), 2);

        let op_id_i64 = tokenizer.op_token_id() as i64;
        for &idx in &tokenized.op_indices {
            assert_eq!(tokenized.input_ids[idx as usize], op_id_i64);
        }
    }

    #[test]
    fn test_state_priority_truncation() {
        let Some(path) = default_tokenizer_path() else {
            return;
        };

        let tokenizer = JevTokenizer::from_file(&path).unwrap();

        let mut criteria = IndexMap::new();
        criteria.insert("opt_a".to_string(), "First option description".to_string());
        criteria.insert("opt_b".to_string(), "Second option description".to_string());
        criteria.insert("opt_c".to_string(), "Third option description".to_string());
        criteria.insert("opt_d".to_string(), "Fourth option description".to_string());

        let question = Question::new_choice("分類せよ。", criteria);

        // 非常に長い State テキスト
        let long_state =
            "This is an extremely long transaction log with hundreds of details. ".repeat(50);

        // 系列長上限を狭い 128 に設定
        let max_len = 128;
        let tokenized = tokenizer
            .encode_question_with_max_len(&long_state, &question, max_len)
            .unwrap();

        // 系列長が上限以下に収まっていること
        assert!(tokenized.input_ids.len() <= max_len);

        // 【重要】State が切り詰められても、4候補すべての [OP] マーカーが完全に保持されていること
        assert_eq!(tokenized.op_indices.len(), 4);
        assert_eq!(tokenized.option_keys.len(), 4);

        let op_id_i64 = tokenizer.op_token_id() as i64;
        for &idx in &tokenized.op_indices {
            assert_eq!(tokenized.input_ids[idx as usize], op_id_i64);
        }
    }

    #[test]
    fn test_prompt_exceeds_max_length_error() {
        let Some(path) = default_tokenizer_path() else {
            return;
        };

        let tokenizer = JevTokenizer::from_file(&path).unwrap();

        let mut criteria = IndexMap::new();
        for i in 0..10 {
            criteria.insert(format!("opt_{i}"), format!("詳細な候補説明その{i}"));
        }
        let question = Question::new_choice("分類せよ。", criteria);

        // 固定部より小さい極端な max_len (例えば 10) を指定
        let res = tokenizer.encode_question_with_max_len("短い状態", &question, 10);
        assert!(matches!(
            res,
            Err(RuntimeError::PromptExceedsMaxLength { .. })
        ));
    }
}
