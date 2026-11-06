//! # CLI 引数パース統合テスト
//!
//! `clap` を用いた各サブコマンドの引数パース、
//! デフォルト値設定、およびオプション指定の正当性を検証する。

use std::path::PathBuf;

use clap::Parser;
use sokuto_cli::{Cli, Commands};

#[test]
fn test_parse_export_openapi_default() {
    let args = ["sokuto", "export-openapi"];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
        Commands::ExportOpenapi { output } => {
            assert_eq!(output, None);
        }
        _ => panic!("ExportOpenapi コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_export_openapi_with_output() {
    let args = ["sokuto", "export-openapi", "--output", "openapi.json"];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
        Commands::ExportOpenapi { output } => {
            assert_eq!(output, Some(PathBuf::from("openapi.json")));
        }
        _ => panic!("ExportOpenapi コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_serve_defaults() {
    let args = ["sokuto", "serve"];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
        Commands::Serve {
            host,
            port,
            model_dir,
            pool_size,
            provider,
            intra_threads,
            inter_threads,
        } => {
            assert_eq!(host, "0.0.0.0");
            assert_eq!(port, 3000);
            assert_eq!(model_dir, PathBuf::from("models/default"));
            assert_eq!(pool_size, 2);
            assert_eq!(provider, None);
            assert_eq!(intra_threads, None);
            assert_eq!(inter_threads, None);
        }
        _ => panic!("Serve コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_serve_custom_options() {
    let args = [
        "sokuto",
        "serve",
        "-H",
        "127.0.0.1",
        "-p",
        "8080",
        "-m",
        "custom_models",
        "--pool-size",
        "4",
        "--provider",
        "cpu",
        "--intra-threads",
        "4",
        "--inter-threads",
        "2",
    ];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
        Commands::Serve {
            host,
            port,
            model_dir,
            pool_size,
            provider,
            intra_threads,
            inter_threads,
        } => {
            assert_eq!(host, "127.0.0.1");
            assert_eq!(port, 8080);
            assert_eq!(model_dir, PathBuf::from("custom_models"));
            assert_eq!(pool_size, 4);
            assert_eq!(provider, Some("cpu".to_string()));
            assert_eq!(intra_threads, Some(4));
            assert_eq!(inter_threads, Some(2));
        }
        _ => panic!("Serve コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_benchmark_defaults() {
    let args = ["sokuto", "benchmark"];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
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
            assert_eq!(model_dir, PathBuf::from("models/default"));
            assert_eq!(iterations, 50);
            assert_eq!(warmup, 10);
            assert_eq!(provider, None);
            assert_eq!(scenario, "all");
            assert_eq!(pool_size, 1);
            assert_eq!(intra_threads, None);
            assert_eq!(inter_threads, None);
            assert!(!no_mem_arena);
            assert!(!json);
        }
        _ => panic!("Benchmark コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_benchmark_custom_options() {
    let args = [
        "sokuto",
        "benchmark",
        "-m",
        "bench_model",
        "-n",
        "100",
        "-w",
        "20",
        "--provider",
        "coreml",
        "-s",
        "batch",
        "--pool-size",
        "4",
        "--intra-threads",
        "2",
        "--inter-threads",
        "1",
        "--no-mem-arena",
        "--json",
    ];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
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
            assert_eq!(model_dir, PathBuf::from("bench_model"));
            assert_eq!(iterations, 100);
            assert_eq!(warmup, 20);
            assert_eq!(provider, Some("coreml".to_string()));
            assert_eq!(scenario, "batch");
            assert_eq!(pool_size, 4);
            assert_eq!(intra_threads, Some(2));
            assert_eq!(inter_threads, Some(1));
            assert!(no_mem_arena);
            assert!(json);
        }
        _ => panic!("Benchmark コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_quantize_defaults() {
    let args = ["sokuto", "quantize"];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
        Commands::Quantize {
            model_dir,
            output_dir,
            per_channel,
            verify,
            script_path,
        } => {
            assert_eq!(model_dir, PathBuf::from("models/default"));
            assert_eq!(output_dir, PathBuf::from("models/quantized"));
            assert!(per_channel);
            assert!(verify);
            assert_eq!(script_path, None);
        }
        _ => panic!("Quantize コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_quantize_custom_options() {
    let args = [
        "sokuto",
        "quantize",
        "--model-dir",
        "input_dir",
        "--output-dir",
        "output_dir",
        "--per-channel=false",
        "--verify=false",
        "--script-path",
        "custom/quantize.py",
    ];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
        Commands::Quantize {
            model_dir,
            output_dir,
            per_channel,
            verify,
            script_path,
        } => {
            assert_eq!(model_dir, PathBuf::from("input_dir"));
            assert_eq!(output_dir, PathBuf::from("output_dir"));
            assert!(!per_channel);
            assert!(!verify);
            assert_eq!(script_path, Some(PathBuf::from("custom/quantize.py")));
        }
        _ => panic!("Quantize コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_healthcheck_defaults() {
    let args = ["sokuto", "healthcheck"];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
        Commands::Healthcheck { url, timeout_secs } => {
            assert_eq!(url, "http://127.0.0.1:3000/ready");
            assert_eq!(timeout_secs, 3);
        }
        _ => panic!("Healthcheck コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_healthcheck_custom() {
    let args = [
        "sokuto",
        "healthcheck",
        "-u",
        "http://localhost:8080/health",
        "-t",
        "5",
    ];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
        Commands::Healthcheck { url, timeout_secs } => {
            assert_eq!(url, "http://localhost:8080/health");
            assert_eq!(timeout_secs, 5);
        }
        _ => panic!("Healthcheck コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_invalid_command() {
    let args = ["sokuto", "unknown-command"];
    let result = Cli::try_parse_from(args);
    assert!(
        result.is_err(),
        "未定義のコマンドはパースエラーとなること。"
    );
}
