//! # sokuto-server
//!
//! Axum をベースとした TypeSafe AI Jev 互換 HTTP API サーバーを提供するクレート。
//!
//! ## 概要
//! - `POST /v1/systemone` 推論エンドポイントの実装。
//! - OpenAPI 3.1 仕様書のコンパイル時自動導出と Swagger UI (`GET /swagger-ui`) の統合。
//! - 運用監視 (`GET /health`, `GET /ready`, `GET /metrics`)。
//! - Tokio 非同期ワーカースレッドのストールを防止するブロッキング分離。

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Extension;
use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use metrics_exporter_prometheus::PrometheusHandle;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

pub use sokuto_core as core;
pub use sokuto_runtime as runtime;

pub mod error;
pub mod guardrails;
pub mod handlers;
pub mod metrics;
pub mod openapi;
pub mod schema;
pub mod state;

pub use error::{ErrorResponse, ServerError};
pub use guardrails::{GuardrailConfig, GuardrailPipeline};
pub use openapi::{export_openapi_json, generate_openapi_spec};
pub use schema::dag::{DagRequest, DagResponse};
pub use state::AppState;

/// サーバーのバージョン情報を取得する。
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Jev HTTP API サーバーのルーターを構築する。
///
/// ソケットバインドを行わずに `Router` を返却するため、
/// `tower::ServiceExt::oneshot` 等を用いたインメモリ統合テストが可能である。
///
/// # 引数
/// - `state`: アプリケーション共有状態。
/// - `prometheus_handle`: Prometheus メトリクスレンダリングハンドル(任意)。
pub fn create_router(state: Arc<AppState>, prometheus_handle: Option<PrometheusHandle>) -> Router {
    let api_routes = Router::new()
        .route("/v1/systemone", post(handlers::system_one_handler))
        .route(
            "/v1/systemone/state-batch",
            post(handlers::state_batch_handler),
        )
        .route("/v1/systemone/dag", post(handlers::dag_handler))
        .route("/health", get(handlers::health_handler))
        .route("/ready", get(handlers::ready_handler))
        .route("/metrics", get(handlers::metrics_handler))
        .layer(Extension(prometheus_handle))
        .with_state(state);

    let router = Router::new()
        .merge(api_routes)
        // 巨大リクエストによる OOM を防止するボディ制限(10MB)
        .layer(DefaultBodyLimit::max(10 * 1024 * 1024))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http());

    // Swagger UI のマウント(feature = "swagger" 有効時のみ実体展開)
    openapi::mount_swagger_ui(router)
}

/// 指定されたアドレスで HTTP サーバーを起動する。
///
/// グレースフルシャットダウンシグナル(Ctrl+C)を受信するまでイベントループを継続する。
///
/// # 引数
/// - `addr`: バインドするソケットアドレス。
/// - `state`: アプリケーション共有状態。
/// - `prometheus_handle`: Prometheus レコーダーハンドル。
pub async fn run_server(
    addr: SocketAddr,
    state: Arc<AppState>,
    prometheus_handle: Option<PrometheusHandle>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let app = create_router(state, prometheus_handle);

    tracing::info!("HTTP サーバーを起動します (アドレス: http://{})...", addr);
    tracing::info!("Swagger UI: http://{}/swagger-ui", addr);
    tracing::info!("OpenAPI JSON: http://{}/api-docs/openapi.json", addr);
    tracing::info!("メトリクス: http://{}/metrics", addr);
    tracing::info!("ヘルスチェック: http://{addr}/health, http://{addr}/ready");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("HTTP サーバーが正常に終了しました。");
    Ok(())
}

/// グレースフルシャットダウンシグナル待機用フューチャー。
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Ctrl+C シグナルの捕捉に失敗しました。");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("SIGTERM シグナルの捕捉に失敗しました。")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("シャットダウンシグナル (Ctrl+C) を受信しました。");
        },
        _ = terminate => {
            tracing::info!("シャットダウンシグナル (SIGTERM) を受信しました。");
        },
    }
}
