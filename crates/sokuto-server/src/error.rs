//! # エラーハンドリングモジュール
//!
//! HTTP レスポンスへのマッピングおよび Jev 互換 JSON エラーペイロードを定義する。

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use sokuto_core::error::CoreError;
use sokuto_runtime::error::RuntimeError;
use utoipa::ToSchema;

/// Jev 互換のエラー詳細情報。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct ErrorDetail {
    /// エラー内容の人間向け説明文。
    pub message: String,
    /// エラーの種別分類(`invalid_request_error`, `internal_server_error` 等)。
    #[serde(rename = "type")]
    pub error_type: String,
    /// 固有のエラー識別コード。
    pub code: String,
}

/// Jev 互換エラーレスポンス構造体。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct ErrorResponse {
    /// エラー詳細ペイロード。
    pub error: ErrorDetail,
}

impl ErrorResponse {
    /// 新規エラーレスポンスを生成する。
    pub fn new(
        message: impl Into<String>,
        error_type: impl Into<String>,
        code: impl Into<String>,
    ) -> Self {
        Self {
            error: ErrorDetail {
                message: message.into(),
                error_type: error_type.into(),
                code: code.into(),
            },
        }
    }
}

/// サーバー内部で発生する各種エラーの統括型。
#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    /// 不正なリクエスト構文またはパラメータ。
    #[error("不正なリクエスト: {0}")]
    BadRequest(String),

    /// Core スキーマの整合性検証エラー。
    #[error("スキーマ検証エラー: {0}")]
    ValidationError(#[from] CoreError),

    /// 質問数またはペイロードサイズの上限超過。
    #[error("リソース上限超過: {0}")]
    PayloadTooLarge(String),

    /// ランタイム推論エンジン障害。
    #[error("推論エンジン障害: {0}")]
    Runtime(#[from] RuntimeError),

    /// DAG 実行エラー。
    #[error("DAG 実行エラー: {0}")]
    Dag(#[from] sokuto_runtime::dag::DagError),

    /// ガードレール違反 (プロンプトインジェクション、禁止語、ハルシネーション等)。
    #[error("ガードレール違反: {0}")]
    GuardrailViolation(String),

    /// サーバー内部の不整合・障害。
    #[error("サーバー内部エラー: {0}")]
    Internal(String),
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, error_type, code, message) = match self {
            ServerError::BadRequest(msg) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "bad_request",
                msg,
            ),
            ServerError::ValidationError(core_err) => {
                let code = match &core_err {
                    CoreError::EmptyQuestions => "empty_questions",
                    CoreError::MissingCriteria { .. } => "missing_criteria",
                    CoreError::InvalidCriteriaType { .. } => "invalid_criteria_type",
                    CoreError::InvalidChoiceCount { .. } => "invalid_choice_count",
                    CoreError::InvalidScoreLevelCount { .. } => "invalid_score_level_count",
                    _ => "schema_validation_error",
                };
                (
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    code,
                    core_err.to_string(),
                )
            }
            ServerError::PayloadTooLarge(msg) => (
                StatusCode::PAYLOAD_TOO_LARGE,
                "invalid_request_error",
                "payload_too_large",
                msg,
            ),
            ServerError::Runtime(runtime_err) => {
                tracing::error!("ランタイム推論エラーが発生しました: {:?}", runtime_err);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_server_error",
                    "inference_engine_error",
                    format!("推論エンジンでエラーが発生しました: {runtime_err}"),
                )
            }
            ServerError::Dag(dag_err) => {
                use sokuto_runtime::dag::DagError;
                match dag_err {
                    DagError::Validation(v) => (
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        "dag_validation_error",
                        v.to_string(),
                    ),
                    DagError::Pool(p) => (
                        StatusCode::SERVICE_UNAVAILABLE,
                        "service_unavailable",
                        "pool_error",
                        p.to_string(),
                    ),
                    DagError::Timeout(ms) => (
                        StatusCode::GATEWAY_TIMEOUT,
                        "gateway_timeout",
                        "dag_timeout",
                        format!("DAG 実行がタイムアウトしました ({ms}ms)。"),
                    ),
                    DagError::StepLimitExceeded(max) => (
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        "step_limit_exceeded",
                        format!("DAG 実行ステップ数が上限 ({max}) を超過しました。"),
                    ),
                    DagError::NodeNotFound(node) => (
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        "node_not_found",
                        format!("ノード '{node}' が見つかりません。"),
                    ),
                    DagError::Runtime(r) => {
                        tracing::error!("DAG 推論ランタイムエラー: {:?}", r);
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "internal_server_error",
                            "inference_engine_error",
                            format!("DAG 推論エンジンでエラーが発生しました: {r}"),
                        )
                    }
                    DagError::ExecutionFailed(msg) => {
                        tracing::error!("DAG 実行失敗: {}", msg);
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "internal_server_error",
                            "dag_execution_failed",
                            msg,
                        )
                    }
                }
            }
            ServerError::GuardrailViolation(msg) => {
                tracing::warn!("ガードレール違反によりリクエストを遮断しました: {}", msg);
                (
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "guardrail_violation",
                    msg,
                )
            }
            ServerError::Internal(msg) => {
                tracing::error!("サーバー内部エラーが発生しました: {}", msg);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal_server_error",
                    "internal_server_error",
                    msg,
                )
            }
        };

        let body = ErrorResponse::new(message, error_type, code);
        (status, axum::Json(body)).into_response()
    }
}
