//! # DAG 実行エンドポイントモジュール
//!
//! `POST /v1/systemone/dag` に対する宣言的 DAG リクエスト検証、
//! インプロセスオーケストレーション実行、および実行経路・決定結果の返却を提供する。

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use sokuto_runtime::dag::DagError;
use sokuto_runtime::dag::schema::{DagDefinition, DagNodeType};

use crate::error::{ErrorResponse, ServerError};
use crate::metrics::{
    record_dag_execution, record_dag_node_execution, record_http_duration, record_http_request,
};
use crate::schema::dag::{DagRequest, DagResponse};
use crate::state::AppState;

/// DAG 実行タイムアウトのサーバー側ハード上限 (ミリ秒)。
///
/// クライアントからの過大なタイムアウト指定によるセッションプールの長期占有や
/// リソース枯渇 (DoS 脆弱性) を防止する。既定値は 5,000ms (5秒)。
pub const MAX_DAG_TIMEOUT_MS: u64 = 5000;

/// 宣言的マイクロ決定グラフ (DAG) 推論実行エンドポイント。
///
/// 反事実ルールや多段階の条件付き決定グラフを非自己回帰型モデルで順次実行し、
/// 辿った実行経路、各ステップの所要時間、および最終決定結果を一括返却する。
/// エスカレーションノードに到達した場合は `status: "escalated"` として正常終了 (200 OK) する。
#[utoipa::path(
    post,
    path = "/v1/systemone/dag",
    tag = "Inference",
    request_body = DagRequest,
    responses(
        (status = 200, description = "DAG 実行成功 (完了またはエスカレーション)", body = DagResponse),
        (status = 400, description = "DAG 定義スキーマまたは整合性検証違反", body = ErrorResponse),
        (status = 504, description = "DAG 実行タイムアウト", body = ErrorResponse),
        (status = 503, description = "セッションプール枯渇", body = ErrorResponse),
        (status = 500, description = "推論エンジンまたはサーバー内部障害", body = ErrorResponse)
    )
)]
pub async fn dag_handler(
    State(state): State<Arc<AppState>>,
    axum::Json(req): axum::Json<DagRequest>,
) -> Result<impl IntoResponse, ServerError> {
    let start_time = Instant::now();

    // 1. State テキストの正規化
    let state_text = match &req.state {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };

    // 2. DagDefinition DTO からランタイム型への変換 (ハードタイムアウト上限クランプ適用)
    let mut dag_def: DagDefinition = match req.dag.try_into() {
        Ok(def) => def,
        Err(err) => {
            record_http_request("/v1/systemone/dag", 400);
            record_http_duration("/v1/systemone/dag", start_time.elapsed().as_secs_f64());
            return Err(ServerError::Dag(DagError::Validation(err)));
        }
    };
    let requested_timeout = req.timeout_ms.unwrap_or(dag_def.timeout_ms);
    dag_def.timeout_ms = requested_timeout.min(MAX_DAG_TIMEOUT_MS);

    let dag_id = dag_def.dag_id.clone();

    // 3. インプロセス DAG 実行器の呼び出し
    let result = match state.dag_executor.execute(&state_text, &dag_def).await {
        Ok(res) => res,
        Err(err) => {
            let status_code = match &err {
                DagError::Validation(_)
                | DagError::StepLimitExceeded(_)
                | DagError::NodeNotFound(_) => 400,
                DagError::Timeout(_) => 504,
                _ => 500,
            };
            record_http_request("/v1/systemone/dag", status_code);
            record_http_duration("/v1/systemone/dag", start_time.elapsed().as_secs_f64());
            return Err(ServerError::Dag(err));
        }
    };

    // 4. メトリクス記録
    let status_str = if result.escalated {
        "escalated"
    } else {
        "completed"
    };
    record_dag_execution(&dag_id, status_str, result.total_duration_ms / 1000.0);

    for (node_id, step) in &result.steps {
        let node_type_str = match step.node_type {
            DagNodeType::Inference => "inference",
            DagNodeType::Branch => "branch",
            DagNodeType::Parallel => "parallel",
            DagNodeType::Escalate => "escalate",
        };
        record_dag_node_execution(&dag_id, node_id, node_type_str, step.duration_ms / 1000.0);
    }

    record_http_request("/v1/systemone/dag", 200);
    record_http_duration("/v1/systemone/dag", start_time.elapsed().as_secs_f64());

    // 5. レスポンス構築
    let response: DagResponse = result.into();
    Ok((StatusCode::OK, axum::Json(response)))
}
