//! # ランタイムエラー型定義モジュール
//!
//! トークナイザー処理および推論パイプラインで発生するエラーを定義する。

use thiserror::Error;

/// ランタイム推論エンジンおよびトークナイザーで発生するエラー型。
#[derive(Debug, Error)]
pub enum RuntimeError {
    /// トークナイザー定義ファイルの読み込み失敗。
    #[error("トークナイザーファイルの読み込みに失敗しました: {0}。")]
    TokenizerLoadError(String),

    /// 必須特殊トークンが未定義。
    #[error("必須特殊トークン '{token}' がトークナイザー内に存在しません。")]
    MissingSpecialToken {
        /// 未定義の特殊トークン文字列。
        token: &'static str,
    },

    /// オプションマーカートークン \[OP\] が単一トークンとして分割されていない。
    #[error(
        "オプションマーカー [OP] の分割テストに失敗しました: 期待ID={expected}, 実際={actual:?}。"
    )]
    InvalidOptionMarkerEncoding {
        /// 期待される単一トークンID。
        expected: u32,
        /// 実際にエンコードされたトークンID列。
        actual: Vec<u32>,
    },

    /// トークナイズ処理時の内部エラー。
    #[error("トークナイズ処理中にエラーが発生しました: {0}。")]
    TokenizerEncodeError(String),

    /// プロンプトの固定部が最大許容系列長を超過している。
    #[error(
        "プロンプト固定長 ({fixed_len}) が最大系列長 ({max_len}) を超過しており、State を収容できません。"
    )]
    PromptExceedsMaxLength {
        /// 固定部分の合計トークン数。
        fixed_len: usize,
        /// 許容最大系列長。
        max_len: usize,
    },

    /// 検出された \[OP\] マーカー数と質問候補数の不一致。
    #[error("検出された [OP] マーカー数 ({detected}) が候補数 ({expected}) と一致しません。")]
    OptionMarkerCountMismatch {
        /// 検出されたマーカー数。
        detected: usize,
        /// 期待される候補数。
        expected: usize,
    },

    /// 不正な質問スキーマまたは評価基準。
    #[error("質問仕様が不正です: {0}。")]
    InvalidQuestion(String),

    /// ONNX Runtime 内部エラー。
    #[error("ONNX Runtime 内部エラー: {0}。")]
    Ort(#[from] ort::Error),

    /// ONNX Runtime セッション構築設定エラー。
    #[error("ONNX Runtime 設定エラー: {0}。")]
    OrtConfig(String),

    /// モデルの入出力契約違反(テンソル名や次元数の不整合)。
    #[error("モデル入出力契約違反: {0}。")]
    ModelContractViolation(String),

    /// テンソルデータが不正(系列長超過インデックスや NaN/Inf など)。
    #[error("テンソルデータが不正です: {0}。")]
    InvalidTensorData(String),

    /// Execution Provider の初期化エラー。
    #[error("Execution Provider の初期化に失敗しました: {0}。")]
    ExecutionProviderError(String),

    /// キャッシュ未登録または期限切れ。
    #[error("キャッシュキー {0} に対応する特徴量が存在しません。")]
    CacheMiss(u64),

    /// キャッシュと入力テンソルの次元不整合。
    #[error(
        "キャッシュ特徴量と入力テンソルの系列長が不一致です: 期待長={expected}, 実際={actual}。"
    )]
    CacheDimensionMismatch {
        /// 期待される系列長。
        expected: usize,
        /// 実際の系列長。
        actual: usize,
    },

    /// コア数理・型スキーマクレート由来のエラー。
    #[error("コアエラー: {0}。")]
    Core(#[from] sokuto_core::error::CoreError),

    /// DAG 静的検証エラー。
    #[error("DAG 静的検証エラー: {0}。")]
    DagValidation(#[from] crate::dag::validator::DagValidationError),
}

/// ランタイム処理の標準結果型。
pub type Result<T> = std::result::Result<T, RuntimeError>;
