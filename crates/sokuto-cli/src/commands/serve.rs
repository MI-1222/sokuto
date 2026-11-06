//! # HTTP API サーバー起動コマンド
//!
//! TypeSafe AI Jev 互換 HTTP API サーバーを初期化し、
//! 指定されたモデルと Execution Provider でエンドポイントを提供する。

use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use sokuto_core::contract::calibration::CalibrationConfig;
use sokuto_runtime::engine::{
    CacheCompressionMode, DualSessionEngine, ExecutionProvider, InferenceEngine, SessionConfig,
    StateCacheManager,
};
use sokuto_runtime::tokenizer::JevTokenizer;
use sokuto_server::metrics::setup_metrics_recorder;
use sokuto_server::state::AppState;

/// サーバー起動オプション構造体。
#[derive(Debug, Clone)]
pub struct ServeArgs {
    /// バインド先ホストアドレス。
    pub host: String,
    /// バインド先ポート番号。
    pub port: u16,
    /// モデルファイル格納ディレクトリ。
    pub model_dir: PathBuf,
    /// セッションプールサイズ。
    pub pool_size: usize,
    /// 優先 Execution Provider (auto, coreml, cuda, tensorrt, cpu)。
    pub provider: Option<String>,
    /// 単一オペレータ内の並列スレッド数 (intra-op)。
    pub intra_threads: Option<usize>,
    /// 複数オペレータ間の並列スレッド数 (inter-op)。
    pub inter_threads: Option<usize>,
}

/// 文字列から Execution Provider をパースする。
pub fn parse_provider(name: &str) -> Result<ExecutionProvider, String> {
    match name.to_lowercase().as_str() {
        "auto" => Ok(ExecutionProvider::Auto),
        "coreml" => Ok(ExecutionProvider::CoreML),
        "cuda" => Ok(ExecutionProvider::CUDA),
        "tensorrt" => Ok(ExecutionProvider::TensorRT),
        "cpu" => Ok(ExecutionProvider::CPU),
        other => Err(format!(
            "未知の Execution Provider です: '{other}'。利用可能: auto, coreml, cuda, tensorrt, cpu。"
        )),
    }
}

/// モデルディレクトリの完全性を検証する。
pub fn validate_model_dir(dir: &Path) -> Result<(), String> {
    if !dir.exists() {
        return Err(format!(
            "モデルディレクトリが存在しません: {}。",
            dir.display()
        ));
    }

    let onnx_file = dir.join("model.onnx");
    if !onnx_file.exists() {
        return Err(format!(
            "必須モデルファイルが見つかりません: {}。",
            onnx_file.display()
        ));
    }

    let tok_file = dir.join("tokenizer.json");
    if !tok_file.exists() {
        return Err(format!(
            "必須トークナイザー定義ファイルが見つかりません: {}。",
            tok_file.display()
        ));
    }

    Ok(())
}

/// HTTP サーバー起動処理を実行する。
///
/// # 引数
/// - `args`: サーバー起動構成パラメータ。
pub async fn run_serve(args: ServeArgs) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sokuto_server=info,tower_http=info".into()),
        )
        .try_init();

    // 1. モデルディレクトリの検証 (Fast-Fail)
    if let Err(err) = validate_model_dir(&args.model_dir) {
        tracing::error!("{err}");
        return Err(err.into());
    }

    // 2. セッション設定の構築
    let mut session_config = SessionConfig {
        pool_size: args.pool_size.max(1),
        intra_threads: args.intra_threads,
        inter_threads: args.inter_threads.or(Some(1)),
        ..Default::default()
    };

    if let Some(ref provider_str) = args.provider {
        let provider = parse_provider(provider_str)?;
        session_config.preferred_providers = match provider {
            ExecutionProvider::Auto => SessionConfig::default().preferred_providers,
            other => vec![other, ExecutionProvider::CPU],
        };
    }

    let onnx_path = args.model_dir.join("model.onnx");
    let tok_path = args.model_dir.join("tokenizer.json");
    let calib_path = args.model_dir.join("calibration.json");

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
    let engine = Arc::new(InferenceEngine::new(&onnx_path, session_config.clone())?);
    let tokenizer = Arc::new(JevTokenizer::from_file(&tok_path)?);

    let calib_config = if calib_path.exists() {
        let calib_json = fs::read_to_string(&calib_path)?;
        tracing::info!("較正温度設定をロードしました: {}", calib_path.display());
        Arc::new(CalibrationConfig::from_json_str(&calib_json)?)
    } else {
        tracing::info!("較正設定ファイルが存在しないため、デフォルト温度設定を使用します。");
        Arc::new(CalibrationConfig::default())
    };

    let mut state = AppState::new(engine, tokenizer, calib_config)
        .load_hierarchical_mapping_from_dir(&args.model_dir);

    // デュアルセッションモデル (state_encoder.onnx, query_evaluator.onnx) の検出と初期化
    let state_model_path = args.model_dir.join("state_encoder.onnx");
    let query_model_path = args.model_dir.join("query_evaluator.onnx");
    if state_model_path.exists() && query_model_path.exists() {
        tracing::info!(
            "2 セッション分割型モデルを検出しました: State={}, Query={}",
            state_model_path.display(),
            query_model_path.display()
        );

        let cache_max_bytes = std::env::var("SOKUTO_STATE_CACHE_MAX_BYTES")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(512 * 1024 * 1024);

        let cache_ttl_secs = std::env::var("SOKUTO_STATE_CACHE_TTL_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(600);

        let compression_mode = match std::env::var("SOKUTO_STATE_CACHE_COMPRESSION")
            .unwrap_or_else(|_| "fp16".to_string())
            .to_lowercase()
            .as_str()
        {
            "none" | "fp32" => CacheCompressionMode::None,
            "int8" => CacheCompressionMode::Int8Symmetric,
            _ => CacheCompressionMode::Fp16,
        };

        let cache_manager = Arc::new(StateCacheManager::new(
            cache_max_bytes,
            cache_ttl_secs,
            compression_mode,
        ));

        match DualSessionEngine::new(
            &state_model_path,
            &query_model_path,
            session_config,
            Arc::clone(&cache_manager),
        ) {
            Ok(dual_engine) => {
                tracing::info!(
                    "DualSessionEngine を初期化しました (キャッシュ上限: {}MB, 圧縮: {:?})。",
                    cache_max_bytes / (1024 * 1024),
                    compression_mode
                );
                state = state.with_dual_engine(Arc::new(dual_engine), cache_manager);
            }
            Err(err) => {
                tracing::warn!(
                    "DualSessionEngine の構築に失敗したため、単一モデルモードで運用します: {err}"
                );
            }
        }
    }

    let state = Arc::new(state);
    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;

    sokuto_server::run_server(addr, state, prometheus_handle).await?;

    Ok(())
}
