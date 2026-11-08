//! # 一括バッチ質問エンドポイントモジュール
//!
//! 単一の長文 `State` に対して複数の異種質問 (`Choice`/`Score`/`Noul`) を一括投入し、
//! `State` エンコードを 1 度きりに抑えてミリ秒級で並列判定結果を返却する。

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use sokuto_core::schema::{StateBatchRequest, StateBatchResponse, SystemOneRequest};

use crate::error::{ErrorResponse, ServerError};
use crate::metrics::{record_http_duration, record_http_request};
use crate::service::InferenceService;
use crate::state::AppState;

/// 単一 `State` に対する一括バッチ質問推論エンドポイント。
///
/// 共通 `State` を 1 回のみエンコード (またはキャッシュ取得) し、
/// 全質問をプレフィックス共有バッチ化して一括評価する。
#[utoipa::path(
    post,
    path = "/v1/systemone/state-batch",
    tag = "Inference",
    request_body = StateBatchRequest,
    responses(
        (status = 200, description = "バッチ推論および決定解決成功", body = StateBatchResponse),
        (status = 400, description = "スキーマまたはパラメータ検証違反", body = ErrorResponse),
        (status = 413, description = "質問数またはペイロード上限超過", body = ErrorResponse),
        (status = 500, description = "推論エンジンまたはサーバー内部障害", body = ErrorResponse)
    )
)]
pub async fn state_batch_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(req): axum::Json<StateBatchRequest>,
) -> Result<impl IntoResponse, ServerError> {
    let start_time = Instant::now();

    req.validate()?;

    let client_hash_str = headers
        .get("x-state-hash")
        .or_else(|| headers.get("X-State-Hash"))
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .or_else(|| req.state_hash.clone());

    let s1_req = SystemOneRequest::from(req);

    let inference_service = InferenceService::new(state);
    let output = inference_service.execute(s1_req, client_hash_str).await?;

    let resp_headers = output.to_headers();
    let hex_key_opt = output.state_key.map(|k| format!("{:016x}", k));

    let mut response = StateBatchResponse::new(output.answers, output.usage, output.cache_hit);
    if let Some(summary) = output.routing_summary {
        response = response.with_routing(summary);
    }
    if let Some(ref hex_key) = hex_key_opt {
        response = response.with_state_hash(hex_key);
    }

    let total_duration = start_time.elapsed().as_secs_f64();
    record_http_duration("/v1/systemone/state-batch", total_duration);
    record_http_request("/v1/systemone/state-batch", 200);

    Ok((resp_headers, axum::Json(response)))
}
