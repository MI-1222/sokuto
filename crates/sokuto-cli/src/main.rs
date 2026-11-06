//! # sokuto-cli
//!
//! Sokuto のコマンドライン運用・管理・性能検証ツール。
//!
//! ## 概要
//! - HTTP API サーバーの起動 (`serve`)。
//! - インプロセス推論レイテンシ・スループット測定 (`benchmark`)。
//! - ONNX INT8 PTQ (Post-Training Quantization) 量子化 (`quantize`)。
//! - OpenAPI 3.1 仕様書の静的エクスポート (`export-openapi`)。

use clap::Parser;
use sokuto_cli::commands::benchmark::run_benchmark;
use sokuto_cli::commands::export_openapi::run_export_openapi;
use sokuto_cli::commands::healthcheck::run_healthcheck;
use sokuto_cli::commands::quantize::run_quantize;
use sokuto_cli::commands::serve::run_serve;
use sokuto_cli::{Cli, Commands};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// 構造化ログトレーシング (`tracing_subscriber`) を初期化する。
///
/// 環境変数 `RUST_LOG` が設定されている場合はそのディレクティブを優先し、
/// 未指定の場合は sokuto 関連クレートおよび HTTP サーバーの標準ログレベル (info) を適用する。
fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "sokuto=info,sokuto_cli=info,sokuto_core=info,sokuto_runtime=info,sokuto_server=info,tower_http=info".into()
            }),
        )
        .try_init();
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    init_tracing();

    let cli = Cli::parse();

    match cli.command {
        Commands::ExportOpenapi { output } => {
            run_export_openapi(output)?;
        }
        Commands::Healthcheck(args) => {
            run_healthcheck(args)?;
        }
        Commands::Serve(args) => {
            run_serve(args).await?;
        }
        Commands::Benchmark(args) => {
            // CPU バウンドな計算ループをワーカースレッドのブロッキングから分離
            tokio::task::spawn_blocking(move || run_benchmark(args)).await??;
        }
        Commands::Quantize(args) => {
            tokio::task::spawn_blocking(move || run_quantize(args)).await??;
        }
    }

    Ok(())
}
