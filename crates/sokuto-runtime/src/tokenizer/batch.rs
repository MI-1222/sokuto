//! # 複数質問バッチトークナイズモジュール
//!
//! 単一の State に対する複数質問のプレフィックス共有エンコード、
//! 可変長系列および可変長候補数の動的パディング、
//! ならびに ONNX Runtime 入力テンソル平坦化バッファの生成を提供する。

use indexmap::IndexMap;
use sokuto_core::contract::model_spec::{ModelInputDimensions, OP_INDEX_PADDING};
use sokuto_core::schema::Question;

use crate::error::{Result, RuntimeError};
use crate::tokenizer::JevTokenizer;

/// 複数質問を一括パッキングしたバッチトークナイズ結果構造体。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchTokenizedQuestions {
    /// 平坦化された入力トークン列テンソル用データ (`[batch_size * sequence_length]`)。
    pub input_ids: Vec<i64>,
    /// 平坦化されたアテンションマスクテンソル用データ (`[batch_size * sequence_length]`)。
    pub attention_mask: Vec<i64>,
    /// 平坦化された候補出現位置インデックステンソル用データ (`[batch_size * num_options]`)。
    /// パディング領域には未選択スロットを表す -1 (`OP_INDEX_PADDING`) が設定される。
    pub op_indices: Vec<i64>,
    /// バッチテンソルの次元情報 (`batch_size`, `sequence_length`, `num_options`)。
    pub dims: ModelInputDimensions,
    /// バッチ内に含まれる質問識別子キーの整列リスト。
    pub question_keys: Vec<String>,
    /// 各質問の本来の有効候補数 $K_i$ のリスト。
    pub candidate_counts: Vec<usize>,
    /// 各質問のプロンプト出現順候補識別子キー列のリスト。
    pub option_keys_per_q: Vec<Vec<String>>,
}

impl JevTokenizer {
    /// 共通 State と複数質問群からプレフィックス共有バッチトークナイズ結果を生成する。
    ///
    /// 最大系列長にはインスタンスに設定されている `max_sequence_length` が適用される。
    ///
    /// # 引数
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問識別子と質問定義のマップ。
    ///
    /// # 戻り値
    /// パディングおよび平坦化済みの `BatchTokenizedQuestions`。
    ///
    /// # エラー
    /// - 質問群が空の場合は `InvalidQuestion`。
    /// - 系列長超過またはマーカー不整合の場合は `RuntimeError`。
    pub fn encode_batch_questions(
        &self,
        state: &str,
        questions: &IndexMap<String, Question>,
    ) -> Result<BatchTokenizedQuestions> {
        self.encode_batch_questions_with_max_len(state, questions, self.max_sequence_length)
    }

    /// 系列長制約を明示指定して、共通 State と複数質問群を一括エンコードする。
    ///
    /// # 処理フロー
    /// 1. `state` を `encode_state` により 1 度だけトークナイズし、トークン ID 列を共有する。
    /// 2. 各質問に対して独立した State 優先トランケーションを適用し、個別トークナイズを実行する。
    /// 3. バッチ内の最大系列長 $L_{\\max}$ および最大候補数 $K_{\\max}$ を算出する。
    /// 4. パディングトークンおよび未選択スロット (-1) を用いて平坦化バッファを構築する。
    ///
    /// # 引数
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問識別子と質問定義のマップ。
    /// - `max_length`: 許容最大入力系列長。
    ///
    /// # エラー
    /// - 質問群が空の場合は `InvalidQuestion`。
    /// - テンソル形状が不正な場合は `ModelContractViolation`。
    pub fn encode_batch_questions_with_max_len(
        &self,
        state: &str,
        questions: &IndexMap<String, Question>,
        max_length: usize,
    ) -> Result<BatchTokenizedQuestions> {
        let state_ids = self.encode_state(state)?;
        self.encode_batch_with_pretokenized_state(&state_ids, questions, max_length)
    }

    /// 事前エンコード済み State トークン列を用いて複数質問群を一括バッチエンコードする。
    ///
    /// サーバー層でのマイクロバッチ分割(チャンキング)時に、同一 State の再トークナイズを完全に防ぐ。
    ///
    /// # 引数
    /// - `state_ids`: 事前トークナイズ済み State トークン ID 列。
    /// - `questions`: 質問識別子と質問定義のマップ。
    /// - `max_length`: 許容最大入力系列長。
    pub fn encode_batch_with_pretokenized_state(
        &self,
        state_ids: &[u32],
        questions: &IndexMap<String, Question>,
        max_length: usize,
    ) -> Result<BatchTokenizedQuestions> {
        let batch_size = questions.len();
        if batch_size == 0 {
            return Err(RuntimeError::InvalidQuestion(
                "質問群が空のためバッチを構築できません。".to_string(),
            ));
        }

        let mut question_keys = Vec::with_capacity(batch_size);
        let mut tokenized_list = Vec::with_capacity(batch_size);
        let mut candidate_counts = Vec::with_capacity(batch_size);
        let mut option_keys_per_q = Vec::with_capacity(batch_size);

        let mut max_seq_len = 0usize;
        let mut max_num_options = 0usize;

        for (q_key, question) in questions {
            let tokenized = self.encode_with_pretokenized_state(state_ids, question, max_length)?;
            let seq_len = tokenized.input_ids.len();
            let num_options = tokenized.op_indices.len();

            if seq_len > max_seq_len {
                max_seq_len = seq_len;
            }
            if num_options > max_num_options {
                max_num_options = num_options;
            }

            question_keys.push(q_key.clone());
            candidate_counts.push(num_options);
            option_keys_per_q.push(tokenized.option_keys.clone());
            tokenized_list.push(tokenized);
        }

        let dims = ModelInputDimensions::new(batch_size, max_seq_len, max_num_options)
            .map_err(|e| RuntimeError::ModelContractViolation(e.to_string()))?;

        let pad_id = self.pad_token_id.unwrap_or(0) as i64;
        let total_tokens = batch_size * max_seq_len;
        let total_options = batch_size * max_num_options;

        let mut input_ids = vec![pad_id; total_tokens];
        let mut attention_mask = vec![0i64; total_tokens];
        // パディング候補スロットは未選択スロットとして -1 で埋める。
        let mut op_indices = vec![OP_INDEX_PADDING; total_options];

        for (i, tokenized) in tokenized_list.into_iter().enumerate() {
            let token_offset = i * max_seq_len;
            let seq_len = tokenized.input_ids.len();
            input_ids[token_offset..token_offset + seq_len].copy_from_slice(&tokenized.input_ids);
            attention_mask[token_offset..token_offset + seq_len]
                .copy_from_slice(&tokenized.attention_mask);

            let op_offset = i * max_num_options;
            let num_ops = tokenized.op_indices.len();
            op_indices[op_offset..op_offset + num_ops].copy_from_slice(&tokenized.op_indices);
        }

        Ok(BatchTokenizedQuestions {
            input_ids,
            attention_mask,
            op_indices,
            dims,
            question_keys,
            candidate_counts,
            option_keys_per_q,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn default_tokenizer_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .join("models")
            .join("default")
            .join("tokenizer.json")
    }

    #[test]
    fn test_encode_batch_questions_empty_error() {
        let tokenizer_path = default_tokenizer_path();
        if !tokenizer_path.exists() {
            return;
        }
        let tokenizer = JevTokenizer::from_file(&tokenizer_path).unwrap();
        let empty_questions = IndexMap::new();

        let result = tokenizer.encode_batch_questions("Context", &empty_questions);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RuntimeError::InvalidQuestion(_)
        ));
    }

    #[test]
    fn test_encode_batch_questions_padding_and_alignment() {
        let tokenizer_path = default_tokenizer_path();
        if !tokenizer_path.exists() {
            return;
        }
        let tokenizer = JevTokenizer::from_file(&tokenizer_path).unwrap();

        let mut questions = IndexMap::new();
        // 質問 1: 2 択
        let mut map_a = IndexMap::new();
        map_a.insert("opt1".to_string(), "選択肢 1".to_string());
        map_a.insert("opt2".to_string(), "選択肢 2".to_string());
        questions.insert(
            "q1".to_string(),
            Question::new_choice("質問 1 の指示文。", map_a),
        );

        // 質問 2: 4 択 (長い指示文)
        let mut map_b = IndexMap::new();
        map_b.insert("a".to_string(), "候補 A".to_string());
        map_b.insert("b".to_string(), "候補 B".to_string());
        map_b.insert("c".to_string(), "候補 C".to_string());
        map_b.insert("d".to_string(), "候補 D".to_string());
        questions.insert(
            "q2".to_string(),
            Question::new_choice(
                "質問 2 の詳細かつ非常に長い指示文です。文脈に合わせて最適な項目を選択してください。",
                map_b,
            ),
        );

        let batch = tokenizer
            .encode_batch_questions("共通のコンテキスト情報です。", &questions)
            .expect("バッチエンコードに成功すること。");

        assert_eq!(batch.dims.batch_size, 2);
        assert_eq!(batch.dims.num_options, 4);
        assert_eq!(batch.candidate_counts, vec![2, 4]);
        assert_eq!(batch.question_keys, vec!["q1", "q2"]);

        let l_max = batch.dims.sequence_length;
        let k_max = batch.dims.num_options;

        // 行 1 (q1) の検証: 2 択なので候補スロット 2, 3 はパディング (-1)
        assert_ne!(batch.op_indices[0], OP_INDEX_PADDING);
        assert_ne!(batch.op_indices[1], OP_INDEX_PADDING);
        assert_eq!(batch.op_indices[2], OP_INDEX_PADDING);
        assert_eq!(batch.op_indices[3], OP_INDEX_PADDING);

        // 系列長の末尾がパディングされていることの検証 (q1 は q2 より短い)
        let q1_tokens = &batch.input_ids[0..l_max];
        let q1_mask = &batch.attention_mask[0..l_max];
        let valid_q1_len = q1_mask.iter().filter(|&&m| m == 1).count();
        assert!(valid_q1_len < l_max);
        // パディング領域のマスクが 0 であり、pad_token_id が埋められていること
        let pad_id = tokenizer.pad_token_id().unwrap_or(0) as i64;
        for &m in &q1_mask[valid_q1_len..] {
            assert_eq!(m, 0);
        }
        for &token in &q1_tokens[valid_q1_len..] {
            assert_eq!(token, pad_id);
        }

        // 行 2 (q2) の検証: 4 択すべて有効
        for k in 0..4 {
            assert_ne!(batch.op_indices[k_max + k], OP_INDEX_PADDING);
        }
    }
}
