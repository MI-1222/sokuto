//! # sokuto-server スタンドアロン実行バイナリ
//!
//! Jev 互換 HTTP API サーバーを起動するためのエントリーポイント。

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use sokuto_core::contract::calibration::CalibrationConfig;
use sokuto_runtime::engine::{InferenceEngine, SessionConfig};
use sokuto_runtime::tokenizer::JevTokenizer;
use sokuto_server::metrics::setup_metrics_recorder;
use sokuto_server::state::AppState;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // 構造化ログトレーシングの初期化
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sokuto_server=info,tower_http=info".into()),
        )
        .init();

    tracing::info!(
        "sokuto-server v{} の起動準備を開始します...",
        sokuto_server::version()
    );

    // Prometheus レコーダーのセットアップ
    let prometheus_handle = match setup_metrics_recorder() {
        Ok(handle) => {
            tracing::info!("Prometheus メトリクスレコーダーを初期化しました。");
            Some(handle)
        }
        Err(err) => {
            tracing::warn!("Prometheus メトリクス初期化をスキップしました: {err}");
            None
        }
    };

    // モデルディレクトリの解決
    let model_dir = std::env::var("SOKUTO_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("models/default"));

    tracing::info!("モデル格納ディレクトリ: {}", model_dir.display());

    let onnx_path = model_dir.join("model.onnx");
    let tok_path = model_dir.join("tokenizer.json");
    let calib_path = model_dir.join("calibration.json");

    let pool_size = std::env::var("SOKUTO_POOL_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(2);

    let intra_threads = std::env::var("SOKUTO_INTRA_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    let inter_threads = std::env::var("SOKUTO_INTER_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .or(Some(1));

    let session_config = SessionConfig {
        pool_size,
        intra_threads,
        inter_threads,
        ..Default::default()
    };

    tracing::info!("ONNX Runtime セッションプールを構築中 (プール数: {pool_size})...");
    let engine = Arc::new(InferenceEngine::new(&onnx_path, session_config.clone())?);

    tracing::info!("高速トークナイザーを読み込み中...");
    let tokenizer = Arc::new(JevTokenizer::from_file(&tok_path)?);

    let calib_config = if calib_path.exists() {
        tracing::info!("事後較正パラメータを読み込み中: {}", calib_path.display());
        let calib_json = std::fs::read_to_string(&calib_path)?;
        Arc::new(CalibrationConfig::from_json_str(&calib_json)?)
    } else {
        tracing::warn!("calibration.json が見つからないため、デフォルト較正値を使用します。");
        Arc::new(CalibrationConfig::default())
    };

    let state = AppState::new(engine, tokenizer, calib_config)
        .init_dual_engine_from_dir(&model_dir, session_config);

    let state = Arc::new(state);

    let host = std::env::var("SOKUTO_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("SOKUTO_PORT")
        .ok()
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(3000);

    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    sokuto_server::run_server(addr, state, prometheus_handle).await?;

    Ok(())
}
