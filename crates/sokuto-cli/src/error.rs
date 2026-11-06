//! # CLI エラー定義モジュール
//!
//! Sokuto CLI の各サブコマンドおよび構成処理で発生するエラーを集約・型安全に管理する。

use thiserror::Error;

/// Sokuto CLI で発生するエラー型。
#[derive(Debug, Error)]
pub enum CliError {
    /// モデルディレクトリまたは必須ファイルの検証エラー。
    #[error("モデルディレクトリ不正: {0}")]
    InvalidModelDir(String),

    /// 未知または非対応の Execution Provider 指定。
    #[error("未知の Execution Provider: {0}")]
    UnknownProvider(String),

    /// ヘルスチェックエンドポイントの監視・検証失敗。
    #[error("ヘルスチェック失敗: {0}")]
    HealthcheckFailed(String),

    /// 量子化スクリプトまたは成果物処理の失敗。
    #[error("量子化処理失敗: {0}")]
    Quantize(String),

    /// ベンチマーク実行中のエラー。
    #[error("ベンチマーク実行失敗: {0}")]
    Benchmark(String),

    /// 設定値またはコマンドライン引数の不正。
    #[error("設定不正: {0}")]
    Config(String),

    /// Sokuto ランタイムエンジン内部で発生したエラー。
    #[error(transparent)]
    Runtime(#[from] sokuto_runtime::error::RuntimeError),

    /// コア契約・プリミティブ処理で発生したエラー。
    #[error(transparent)]
    Core(#[from] sokuto_core::error::CoreError),

    /// HTTP サーバー内部または起動処理で発生したエラー。
    #[error("サーバーエラー: {0}")]
    Server(String),

    /// I/O 処理で発生したエラー。
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// 非同期タスクの結合 (join) 失敗。
    #[error(transparent)]
    TaskJoin(#[from] tokio::task::JoinError),

    /// JSON シリアライズ/デシリアライズ失敗。
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

impl From<crate::commands::model_dir::ModelDirError> for CliError {
    fn from(err: crate::commands::model_dir::ModelDirError) -> Self {
        Self::InvalidModelDir(err.to_string())
    }
}
