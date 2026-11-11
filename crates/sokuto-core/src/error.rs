//! # エラー型モジュール
//!
//! `sokuto-core` クレート内で発生するドメインエラーおよび結果型を定義する。

use thiserror::Error;

use crate::schema::QuestionType;

/// 質問識別子(任意)に基づき、エラーメッセージ用のプレフィックス文字列を生成する。
fn question_prefix(id: &Option<String>) -> String {
    match id {
        Some(qid) => format!("質問 '{qid}': "),
        None => String::new(),
    }
}

/// `sokuto-core` の標準結果型。
pub type Result<T> = std::result::Result<T, CoreError>;

/// コアモジュールで発生するエラーの列挙型。
#[derive(Debug, Error, PartialEq)]
pub enum CoreError {
    /// 選択肢の数が制限範囲外である場合のエラー(Choice 型は 1〜255 個)。
    #[error(
        "{prefix}Choice 型の候補数は 1〜255 個である必要がありますが、{count} 個指定されました。",
        prefix = question_prefix(question_id)
    )]
    InvalidChoiceCount {
        /// 対象の質問識別子(任意)。
        question_id: Option<String>,
        /// 指定された候補数。
        count: usize,
    },

    /// スコア評価尺度の段階数が範囲外である場合のエラー(Score 型は 2〜10 段階)。
    #[error(
        "{prefix}Score 型の評価尺度数は 2〜10 段階である必要がありますが、{count} 個指定されました。",
        prefix = question_prefix(question_id)
    )]
    InvalidScoreLevelCount {
        /// 対象の質問識別子(任意)。
        question_id: Option<String>,
        /// 指定された段階数。
        count: usize,
    },

    /// 質問タイプに対する評価基準(Criteria)の形式が不一致である場合のエラー。
    #[error(
        "{prefix}{question_type} 型に対して不正な Criteria 形式が指定されました。期待値: {expected}、実際: {actual}。",
        prefix = question_prefix(question_id)
    )]
    InvalidCriteriaType {
        /// 対象の質問識別子(任意)。
        question_id: Option<String>,
        /// 対象の質問タイプ。
        question_type: QuestionType,
        /// 期待される形式名。
        expected: &'static str,
        /// 実際に指定された形式名。
        actual: &'static str,
    },

    /// 必須の評価基準(Criteria)が欠落している場合のエラー。
    #[error(
        "{prefix}{question_type} 型には Criteria の指定が必須です。",
        prefix = question_prefix(question_id)
    )]
    MissingCriteria {
        /// 対象の質問識別子(任意)。
        question_id: Option<String>,
        /// 対象の質問タイプ。
        question_type: QuestionType,
    },

    /// 質問リストが空である場合のエラー。
    #[error("リクエストに含まれる questions が空です。1 件以上の質問を指定してください。")]
    EmptyQuestions,

    /// 確率分布の計算やエントロピー算出における数値計算エラー。
    #[error(
        "{prefix}数値計算エラーが発生しました: {message}。",
        prefix = question_prefix(question_id)
    )]
    MathError {
        /// 対象の質問識別子(任意)。
        question_id: Option<String>,
        /// エラー詳細メッセージ。
        message: String,
    },

    /// シリアライズまたはデシリアライズ失敗エラー。
    #[error("シリアライズ/デシリアライズに失敗しました: {message}。")]
    SerializationError {
        /// エラー詳細メッセージ。
        message: String,
    },

    /// 確信度ゲーティング設定のバリデーションエラー。
    #[error("ゲーティング設定が不正です: {message}。")]
    InvalidGatingConfig {
        /// エラー詳細メッセージ。
        message: String,
    },

    /// バッチサイズが 0 または仕様制約外である場合のエラー。
    #[error("バッチサイズは 1 以上である必要がありますが、{batch_size} が指定されました。")]
    InvalidBatchSize {
        /// 指定されたバッチサイズ。
        batch_size: usize,
    },

    /// 入力系列長が仕様制約外である場合のエラー。
    #[error(
        "入力系列長は {min}〜{max} の範囲内である必要がありますが、{length} が指定されました。"
    )]
    InvalidSequenceLength {
        /// 指定された系列長。
        length: usize,
        /// 許容される最小系列長。
        min: usize,
        /// 許容される最大系列長。
        max: usize,
    },

    /// モデルテンソルにおけるオプションマーカー数([OP]数)が仕様制約外である場合のエラー。
    #[error(
        "モデルオプション数は {min}〜{max} の範囲内である必要がありますが、{count} が指定されました。"
    )]
    InvalidOptionCount {
        /// 指定されたオプション数。
        count: usize,
        /// 許容される最小オプション数。
        min: usize,
        /// 許容される最大オプション数。
        max: usize,
    },

    /// テンソルの形状(shape)が期待される仕様と不一致である場合のエラー。
    #[error("テンソル '{tensor}' の形状が不正です。期待値: {expected:?}、実際: {actual:?}。")]
    InvalidTensorShape {
        /// 対象のテンソル名。
        tensor: &'static str,
        /// 期待される形状。
        expected: Vec<usize>,
        /// 実際に渡された形状。
        actual: Vec<usize>,
    },

    /// テンソル間で特定の次元サイズ(バッチサイズなど)や次元数が不一致である場合のエラー。
    #[error(
        "テンソル '{tensor}' の {dimension} が不一致です。期待値: {expected}、実際: {actual}。"
    )]
    DimensionMismatch {
        /// 対象のテンソル名。
        tensor: &'static str,
        /// 不一致となった次元または属性名(例: "batch_size", "rank")。
        dimension: &'static str,
        /// 期待される値。
        expected: usize,
        /// 実際の値。
        actual: usize,
    },
}

impl CoreError {
    /// エラーに質問識別子 (question_id) のコンテキストを付与または上書きする。
    pub fn with_question_id(mut self, qid: impl Into<String>) -> Self {
        let qid_str = Some(qid.into());
        match &mut self {
            Self::InvalidChoiceCount { question_id, .. }
            | Self::InvalidScoreLevelCount { question_id, .. }
            | Self::InvalidCriteriaType { question_id, .. }
            | Self::MissingCriteria { question_id, .. }
            | Self::MathError { question_id, .. } => {
                *question_id = qid_str;
            }
            _ => {}
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_error_structured_question_id_and_no_duplicate_prefix() {
        let err = CoreError::MathError {
            question_id: Some("category".to_string()),
            message: "ロジット長が不正です。".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "質問 'category': 数値計算エラーが発生しました: ロジット長が不正です。。"
        );

        // with_question_id を複数回適用してもプレフィックスが重複せず、question_id のみが更新されること。
        let updated = err.with_question_id("category").with_question_id("urgency");
        match &updated {
            CoreError::MathError {
                question_id,
                message,
            } => {
                assert_eq!(question_id.as_deref(), Some("urgency"));
                assert_eq!(message, "ロジット長が不正です。");
            }
            other => panic!("MathError が返る必要があります: {other:?}"),
        }
        assert_eq!(
            updated.to_string(),
            "質問 'urgency': 数値計算エラーが発生しました: ロジット長が不正です。。"
        );
    }
}
