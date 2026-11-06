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
use sokuto_cli::commands::benchmark::{BenchmarkArgs, run_benchmark};
use sokuto_cli::commands::export_openapi::run_export_openapi;
use sokuto_cli::commands::healthcheck::{HealthcheckArgs, run_healthcheck};
use sokuto_cli::commands::quantize::{QuantizeArgs, run_quantize};
use sokuto_cli::commands::serve::{ServeArgs, run_serve};
use sokuto_cli::{Cli, Commands};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::ExportOpenapi { output } => {
            run_export_openapi(output)?;
        }
        Commands::Healthcheck { url, timeout_secs } => {
            let args = HealthcheckArgs { url, timeout_secs };
            run_healthcheck(args)?;
        }
        Commands::Serve {
            host,
            port,
            model_dir,
            pool_size,
            provider,
            intra_threads,
            inter_threads,
        } => {
            let args = ServeArgs {
                host,
                port,
                model_dir,
                pool_size,
                provider,
                intra_threads,
                inter_threads,
            };
            run_serve(args).await?;
        }
        Commands::Benchmark {
            model_dir,
            iterations,
            warmup,
            provider,
            scenario,
            pool_size,
            intra_threads,
            inter_threads,
            no_mem_arena,
            json,
        } => {
            let args = BenchmarkArgs {
                model_dir,
                iterations,
                warmup,
                provider,
                scenario,
                pool_size,
                intra_threads,
                inter_threads,
                enable_mem_arena: !no_mem_arena,
                json,
            };
            // CPU バウンドな計算ループをワーカースレッドのブロッキングから分離
            tokio::task::spawn_blocking(move || run_benchmark(args)).await??;
        }
        Commands::Quantize {
            model_dir,
            output_dir,
            per_channel,
            verify,
            script_path,
        } => {
            let args = QuantizeArgs {
                model_dir,
                output_dir,
                per_channel,
                verify,
                script_path,
            };
            tokio::task::spawn_blocking(move || run_quantize(args)).await??;
        }
    }

    Ok(())
}
