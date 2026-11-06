//! # ONNX モデル入出力テンソル仕様および契約定義モジュール
//!
//! Python 学習・エクスポート側と Rust ランタイム推論エンジン側で
//! 厳密に共有される入出力テンソル名、形状制約、および特殊トークン仕様を定義する。

use crate::error::{CoreError, Result};

/// 入力トークン列テンソル名 (`int64[batch_size, sequence_length]`)。
pub const TENSOR_INPUT_IDS: &str = "input_ids";

/// アテンションマスクテンソル名 (`int64[batch_size, sequence_length]`)。
pub const TENSOR_ATTENTION_MASK: &str = "attention_mask";

/// オプションマーカー位置インデックステンソル名 (`int64[batch_size, num_options]`)。
pub const TENSOR_OP_INDICES: &str = "op_indices";

/// 候補インデックステンソルにおけるパディング（未選択スロット）ダミー値。
pub const OP_INDEX_PADDING: i64 = -1;

/// 決定ヘッド出力ロジットテンソル名 (`float32[batch_size, num_options]`)。
pub const TENSOR_LOGITS: &str = "logits";

/// `State` エンコーダが出力し、`Query` 評価器が参照する最終層隠れ状態テンソル名 (`float32[batch_size, state_seq_len, hidden_size]`)。
pub const TENSOR_STATE_HIDDEN_STATES: &str = "state_hidden_states";

/// `State` 系列の有効トークンマスクテンソル名 (`int64[batch_size, state_seq_len]`)。
pub const TENSOR_STATE_MASK: &str = "state_mask";

/// 質問系列の入力トークンID列テンソル名 (`int64[batch_size, query_seq_len]`)。
pub const TENSOR_QUERY_IDS: &str = "query_ids";

/// 質問系列のアテンションマスクテンソル名 (`int64[batch_size, query_seq_len]`)。
pub const TENSOR_QUERY_MASK: &str = "query_mask";

/// 各候補の先頭に付与されるオプションマーカートークン文字列。
pub const TOKEN_OPTION_MARKER: &str = "[OP]";

/// システムで許容される最大入力トークン系列長。
pub const MAX_SEQUENCE_LENGTH: usize = 8192;

/// システムで許容される単一質問あたりの最大候補数(Choice 型上限)。
pub const MAX_NUM_OPTIONS: usize = 255;

/// 最小候補数。
pub const MIN_NUM_OPTIONS: usize = 1;

/// 検証済みのモデル入力次元情報。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelInputDimensions {
    /// バッチサイズ(質問並列数)。
    pub batch_size: usize,
    /// 入力系列長(パディング含む)。
    pub sequence_length: usize,
    /// 評価候補数。
    pub num_options: usize,
}

impl ModelInputDimensions {
    /// 新規次元情報を構築し、仕様制約を満たしているか検証する。
    pub fn new(batch_size: usize, sequence_length: usize, num_options: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(CoreError::MathError {
                message: "バッチサイズは 1 以上である必要があります。".to_string(),
            });
        }
        if sequence_length == 0 || sequence_length > MAX_SEQUENCE_LENGTH {
            return Err(CoreError::MathError {
                message: format!(
                    "入力系列長は 1〜{MAX_SEQUENCE_LENGTH} の範囲内である必要がありますが、{sequence_length} が指定されました。"
                ),
            });
        }
        if !(MIN_NUM_OPTIONS..=MAX_NUM_OPTIONS).contains(&num_options) {
            return Err(CoreError::InvalidChoiceCount { count: num_options });
        }

        Ok(Self {
            batch_size,
            sequence_length,
            num_options,
        })
    }

    /// 各入力テンソルの形状スライスを検証し、一致する次元情報を返却する。
    pub fn validate_tensor_shapes(
        input_ids_shape: &[usize],
        attention_mask_shape: &[usize],
        op_indices_shape: &[usize],
    ) -> Result<Self> {
        if input_ids_shape.len() != 2 {
            return Err(CoreError::MathError {
                message: format!(
                    "{TENSOR_INPUT_IDS} は 2 次元テンソル [batch, seq_len] である必要があります: {input_ids_shape:?}。"
                ),
            });
        }
        if attention_mask_shape != input_ids_shape {
            return Err(CoreError::MathError {
                message: format!(
                    "{TENSOR_ATTENTION_MASK} の形状 {attention_mask_shape:?} は {TENSOR_INPUT_IDS} の形状 {input_ids_shape:?} と一致する必要があります。"
                ),
            });
        }
        if op_indices_shape.len() != 2 {
            return Err(CoreError::MathError {
                message: format!(
                    "{TENSOR_OP_INDICES} は 2 次元テンソル [batch, num_options] である必要があります: {op_indices_shape:?}。"
                ),
            });
        }
        if op_indices_shape[0] != input_ids_shape[0] {
            return Err(CoreError::MathError {
                message: format!(
                    "{TENSOR_OP_INDICES} のバッチサイズ {} は {TENSOR_INPUT_IDS} のバッチサイズ {} と一致する必要があります。",
                    op_indices_shape[0], input_ids_shape[0]
                ),
            });
        }

        Self::new(input_ids_shape[0], input_ids_shape[1], op_indices_shape[1])
    }

    /// 出力 logits テンソルの形状が入力次元と整合しているか検証する。
    pub fn validate_output_shape(&self, logits_shape: &[usize]) -> Result<()> {
        if logits_shape.len() != 2 {
            return Err(CoreError::MathError {
                message: format!(
                    "{TENSOR_LOGITS} は 2 次元テンソル [batch, num_options] である必要があります: {logits_shape:?}。"
                ),
            });
        }
        if logits_shape[0] != self.batch_size || logits_shape[1] != self.num_options {
            return Err(CoreError::MathError {
                message: format!(
                    "{TENSOR_LOGITS} の形状 {logits_shape:?} は期待される形状 [{}, {}] と一致しません。",
                    self.batch_size, self.num_options
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_shape_validation_success() {
        let input_ids = [2, 128];
        let attention_mask = [2, 128];
        let op_indices = [2, 4];

        let dims =
            ModelInputDimensions::validate_tensor_shapes(&input_ids, &attention_mask, &op_indices)
                .unwrap();

        assert_eq!(dims.batch_size, 2);
        assert_eq!(dims.sequence_length, 128);
        assert_eq!(dims.num_options, 4);

        // 出力 logits [2, 4] の検証成功
        assert!(dims.validate_output_shape(&[2, 4]).is_ok());

        // 出力 logits の形状不一致エラー
        assert!(dims.validate_output_shape(&[2, 5]).is_err());
    }

    #[test]
    fn test_tensor_shape_validation_mismatch() {
        // attention_mask の系列長不一致
        let res = ModelInputDimensions::validate_tensor_shapes(&[1, 64], &[1, 32], &[1, 3]);
        assert!(res.is_err());

        // バッチサイズ不一致
        let res2 = ModelInputDimensions::validate_tensor_shapes(&[2, 64], &[2, 64], &[1, 3]);
        assert!(res2.is_err());
    }
}
