//! # sokuto-cli ライブラリクレート
//!
//! Sokuto CLI のコマンド実装および構成パラメータを提供する。

pub mod commands;

pub use commands::benchmark::BenchmarkArgs;
pub use commands::healthcheck::HealthcheckArgs;
pub use commands::quantize::QuantizeArgs;
pub use commands::serve::ServeArgs;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

/// Sokuto CLI 管理ツールのコマンドライン引数定義。
#[derive(Debug, Parser)]
#[command(
    name = "sokuto",
    version,
    about = "TypeSafe AI Jev 互換の非自己回帰型判断特化モデル CLI ツール。"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

/// 利用可能なサブコマンド一覧。
#[derive(Debug, Subcommand, PartialEq, Eq)]
pub enum Commands {
    /// OpenAPI 3.1 仕様書を JSON 形式で出力する。
    ExportOpenapi {
        /// 出力先ファイルパス (省略時は標準出力)。
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Jev 互換 HTTP API サーバーを起動する。
    Serve(ServeArgs),

    /// サーバーエンドポイントの死活監視・準備状態確認を実行する。
    Healthcheck(HealthcheckArgs),

    /// インプロセスでの推論レイテンシおよびスループットを計測する。
    Benchmark(BenchmarkArgs),

    /// FP32 ONNX モデルに対して動的 INT8 PTQ 量子化を実行する。
    Quantize(QuantizeArgs),
}
