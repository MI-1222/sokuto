//! # Jev 互換推論エンドポイントモジュール
//!
//! `POST /v1/systemone` に対するリクエスト検証、推論サービス呼び出し、
//! および型付き判定結果のシリアライズを提供する。

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use sokuto_core::schema::{SystemOneRequest, SystemOneResponse};

use crate::error::{ErrorResponse, ServerError};
use crate::metrics::{record_http_duration, record_http_request};
use crate::service::InferenceService;
use crate::state::AppState;

/// TypeSafe AI Jev 互換の非自己回帰型判断推論エンドポイント。
///
/// 共通 State と複数質問群を受け取り、単一フォワードパスおよび
/// 決定プリミティブ数理に基づく型付き確率決定を一括返却する。
#[utoipa::path(
    post,
    path = "/v1/systemone",
    tag = "Inference",
    request_body = SystemOneRequest,
    responses(
        (status = 200, description = "推論および決定解決成功", body = SystemOneResponse),
        (status = 400, description = "スキーマまたはパラメータ検証違反", body = ErrorResponse),
        (status = 413, description = "質問数またはペイロード上限超過", body = ErrorResponse),
        (status = 500, description = "推論エンジンまたはサーバー内部障害", body = ErrorResponse)
    )
)]
pub async fn system_one_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<SystemOneRequest>,
) -> Result<impl IntoResponse, ServerError> {
    let start_time = Instant::now();

    req.validate()?;

    let client_hash_str = headers
        .get("x-state-hash")
        .or_else(|| headers.get("X-State-Hash"))
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .or_else(|| req.state_hash.clone());

    let inference_service = InferenceService::new(state);
    let output = inference_service.execute(req, client_hash_str).await?;

    let resp_headers = output.to_headers();
    let mut response = SystemOneResponse::new(output.answers, output.usage);
    if let Some(summary) = output.routing_summary {
        response = response.with_routing(summary);
    }

    let total_duration = start_time.elapsed().as_secs_f64();
    record_http_duration("/v1/systemone", total_duration);
    record_http_request("/v1/systemone", 200);

    Ok((resp_headers, axum::Json(response)))
}
