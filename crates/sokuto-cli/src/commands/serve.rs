//! # HTTP API サーバー起動コマンド
//!
//! TypeSafe AI Jev 互換 HTTP API サーバーを初期化し、
//! 指定されたモデルと Execution Provider でエンドポイントを提供する。

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

pub use crate::commands::model_dir::{
    ModelArtifactPaths, ModelDirError, load_calibration_config, validate_model_dir,
};
use crate::commands::session_config::{ExecutionProviderArg, SessionConfigBuilder};
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

    /// セッションプールサイズ。
    #[arg(long, env = "SOKUTO_POOL_SIZE", default_value_t = 2)]
    pub pool_size: usize,

    /// 優先 Execution Provider。
    #[arg(long, env = "SOKUTO_PROVIDER")]
    pub provider: Option<ExecutionProviderArg>,

    /// 単一オペレータ内の並列スレッド数 (intra-op)。
    #[arg(long, env = "SOKUTO_INTRA_THREADS")]
    pub intra_threads: Option<usize>,

    /// 複数オペレータ間の並列スレッド数 (inter-op)。
    #[arg(long, env = "SOKUTO_INTER_THREADS")]
    pub inter_threads: Option<usize>,
}

/// HTTP サーバー起動処理を実行する。
///
/// # 引数
/// - `args`: サーバー起動構成パラメータ。
pub async fn run_serve(args: ServeArgs) -> Result<(), CliError> {
    // 1. モデルディレクトリの検証 (Fast-Fail)
    validate_model_dir(&args.model_dir)?;

    // 2. セッション設定の構築
    let session_config = SessionConfigBuilder::new()
        .pool_size(args.pool_size)
        .intra_threads(args.intra_threads)
        .inter_threads(args.inter_threads)
        .provider_opt(args.provider)
        .build();

    let artifact_paths = ModelArtifactPaths::resolve(&args.model_dir);

    tracing::info!(
        "sokuto サーバーを起動します (モデル: {}, アドレス: {}:{}, セッションプール数: {})...",
        args.model_dir.display(),
        args.host,
        args.port,
        args.pool_size
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
    let addr = resolve_socket_addr(&args.host, args.port).await?;

    sokuto_server::run_server(addr, state, prometheus_handle)
        .await
        .map_err(|e| CliError::Server(e.to_string()))?;

    Ok(())
}

/// ホスト名または IP アドレスとポート番号からソケットアドレスを非同期解決する。
///
/// IPv6 アドレスが角括弧で囲まれている場合 (`[::1]`) も除去して正規化し、
/// 非同期名前解決またはパースを実行する。
///
/// # 引数
/// - `host`: ホスト名または IP アドレス (例: "127.0.0.1", "localhost", "::1", "[::1]")。
/// - `port`: ポート番号。
pub async fn resolve_socket_addr(host: &str, port: u16) -> Result<SocketAddr, CliError> {
    let normalized_host = host.trim_start_matches('[').trim_end_matches(']');
    let mut addrs = tokio::net::lookup_host((normalized_host, port))
        .await
        .map_err(|err| {
            CliError::Config(format!(
                "ホスト名 '{host}:{port}' の名前解決に失敗しました: {err}。"
            ))
        })?;

    addrs.next().ok_or_else(|| {
        CliError::Config(format!(
            "ホスト名 '{host}:{port}' に対応するソケットアドレスが見つかりませんでした。"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_resolve_socket_addr_ipv4() {
        let addr = resolve_socket_addr("127.0.0.1", 8080)
            .await
            .expect("IPv4 アドレスの解決に失敗。");
        assert_eq!(addr, "127.0.0.1:8080".parse::<SocketAddr>().unwrap());
    }

    #[tokio::test]
    async fn test_resolve_socket_addr_localhost() {
        let addr = resolve_socket_addr("localhost", 8080)
            .await
            .expect("localhost の解決に失敗。");
        assert_eq!(addr.port(), 8080);
        assert!(addr.ip().is_loopback());
    }

    #[tokio::test]
    async fn test_resolve_socket_addr_ipv6() {
        let addr = resolve_socket_addr("::1", 8080)
            .await
            .expect("IPv6 アドレス (角括弧なし) の解決に失敗。");
        assert_eq!(addr, "[::1]:8080".parse::<SocketAddr>().unwrap());
    }

    #[tokio::test]
    async fn test_resolve_socket_addr_ipv6_bracketed() {
        let addr = resolve_socket_addr("[::1]", 8080)
            .await
            .expect("IPv6 アドレス (角括弧あり) の解決に失敗。");
        assert_eq!(addr, "[::1]:8080".parse::<SocketAddr>().unwrap());
    }
}
