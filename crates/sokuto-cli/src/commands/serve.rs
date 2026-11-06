//! # HTTP API サーバー起動コマンド
//!
//! TypeSafe AI Jev 互換 HTTP API サーバーを初期化し、
//! 指定されたモデルと Execution Provider でエンドポイントを提供する。

use std::path::PathBuf;
use std::sync::Arc;

pub use crate::commands::model_dir::{
    ModelArtifactPaths, ModelDirError, load_calibration_config, validate_model_dir,
};
use crate::commands::net::resolve_single_socket_addr;
pub use crate::commands::session_config::{ExecutionProviderArg, SessionArgs, SessionConfigArgs};
use crate::error::CliError;
use sokuto_runtime::engine::InferenceEngine;
use sokuto_runtime::tokenizer::JevTokenizer;
use sokuto_server::metrics::setup_metrics_recorder;
use sokuto_server::state::AppState;

/// サーバー起動オプション構造体。
#[derive(Debug, Clone, clap::Args, PartialEq, Eq)]
pub struct ServeArgs {
    /// バインド先ホストアドレス。
    #[arg(short = 'H', long, env = "SOKUTO_HOST", default_value = "0.0.0.0")]
    pub host: String,

    /// バインド先ポート番号。
    #[arg(short, long, env = "SOKUTO_PORT", default_value_t = 3000)]
    pub port: u16,

    /// モデルファイル格納ディレクトリ。
    #[arg(
        short,
        long,
        env = "SOKUTO_MODEL_DIR",
        default_value = "models/default"
    )]
    pub model_dir: PathBuf,

    /// ONNX Runtime セッション構成引数。
    #[command(flatten)]
    pub session: SessionArgs,
}

/// HTTP サーバー起動処理を実行する。
///
/// # 引数
/// - `args`: サーバー起動構成パラメータ。
pub async fn run_serve(args: ServeArgs) -> Result<(), CliError> {
    // 1. モデルディレクトリの検証 (Fast-Fail)
    validate_model_dir(&args.model_dir)?;

    // 2. セッション設定の構築
    let session_config = args.session.build_config();

    let artifact_paths = ModelArtifactPaths::resolve(&args.model_dir);

    tracing::info!(
        "sokuto サーバーを起動します (モデル: {}, アドレス: {}:{}, セッションプール数: {})...",
        args.model_dir.display(),
        args.host,
        args.port,
        args.session.pool_size
    );

    // 3. Prometheus レコーダーの初期化
    let prometheus_handle = match setup_metrics_recorder() {
        Ok(handle) => Some(handle),
        Err(err) => {
            tracing::warn!("Prometheus メトリクス初期化をスキップしました: {err}");
            None
        }
    };

    // 4. 推論エンジンとトークナイザーのロード
    let engine = Arc::new(InferenceEngine::new(
        &artifact_paths.model_onnx,
        session_config.clone(),
    )?);
    let tokenizer = Arc::new(JevTokenizer::from_file(&artifact_paths.tokenizer_json)?);
    let calib_config = Arc::new(load_calibration_config(&args.model_dir)?);

    let state = AppState::new(engine, tokenizer, calib_config)
        .load_hierarchical_mapping_from_dir(&args.model_dir)
        .init_dual_engine_from_dir(&args.model_dir, session_config);

    let state = Arc::new(state);
    let addr = resolve_single_socket_addr(&args.host, args.port)
        .await
        .map_err(|err| CliError::Config(err.to_string()))?;

    sokuto_server::run_server(addr, state, prometheus_handle)
        .await
        .map_err(|e| CliError::Server(e.to_string()))?;

    Ok(())
}
