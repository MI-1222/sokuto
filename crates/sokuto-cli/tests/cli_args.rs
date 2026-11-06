//! # CLI 引数パース統合テスト
//!
//! `clap` を用いた各サブコマンドの引数パース、
//! デフォルト値設定、およびオプション指定の正当性を検証する。

use std::path::PathBuf;

use clap::Parser;
use sokuto_cli::{BenchmarkScenario, Cli, Commands};

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
        Commands::Serve(args) => {
            assert_eq!(args.host, "0.0.0.0");
            assert_eq!(args.port, 3000);
            assert_eq!(args.model_dir, PathBuf::from("models/default"));
            assert_eq!(args.pool_size, 2);
            assert_eq!(args.provider, None);
            assert_eq!(args.intra_threads, None);
            assert_eq!(args.inter_threads, None);
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
        Commands::Serve(args) => {
            assert_eq!(args.host, "127.0.0.1");
            assert_eq!(args.port, 8080);
            assert_eq!(args.model_dir, PathBuf::from("custom_models"));
            assert_eq!(args.pool_size, 4);
            assert_eq!(args.provider, Some("cpu".to_string()));
            assert_eq!(args.intra_threads, Some(4));
            assert_eq!(args.inter_threads, Some(2));
        }
        _ => panic!("Serve コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_benchmark_defaults() {
    let args = ["sokuto", "benchmark"];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
        Commands::Benchmark(args) => {
            assert_eq!(args.model_dir, PathBuf::from("models/default"));
            assert_eq!(args.iterations, 50);
            assert_eq!(args.warmup, 10);
            assert_eq!(args.provider, None);
            assert_eq!(args.scenario, BenchmarkScenario::All);
            assert_eq!(args.pool_size, 1);
            assert_eq!(args.intra_threads, None);
            assert_eq!(args.inter_threads, None);
            assert!(!args.no_mem_arena);
            assert!(!args.json);
        }
        _ => panic!("Benchmark コマンドのパースに失敗。"),
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
        Commands::Benchmark(args) => {
            assert_eq!(args.model_dir, PathBuf::from("bench_model"));
            assert_eq!(args.iterations, 100);
            assert_eq!(args.warmup, 20);
            assert_eq!(args.provider, Some("coreml".to_string()));
            assert_eq!(args.scenario, BenchmarkScenario::Batch);
            assert_eq!(args.pool_size, 4);
            assert_eq!(args.intra_threads, Some(2));
            assert_eq!(args.inter_threads, Some(1));
            assert!(args.no_mem_arena);
            assert!(args.json);
        }
        _ => panic!("Benchmark コマンドのパースに失敗。"),
    }
}

#[test]
fn test_parse_benchmark_unknown_scenario_fails() {
    let args = ["sokuto", "benchmark", "--scenario", "unknown_scenario"];
    let result = Cli::try_parse_from(args);
    assert!(
        result.is_err(),
        "未知のシナリオ指定時にパースエラーで拒否されることに成功。"
    );
}

#[test]
fn test_parse_quantize_defaults() {
    let args = ["sokuto", "quantize"];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
        Commands::Quantize(args) => {
            assert_eq!(args.model_dir, PathBuf::from("models/default"));
            assert_eq!(args.output_dir, PathBuf::from("models/quantized"));
            assert!(args.per_channel);
            assert!(args.verify);
            assert_eq!(args.script_path, None);
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
        Commands::Quantize(args) => {
            assert_eq!(args.model_dir, PathBuf::from("input_dir"));
            assert_eq!(args.output_dir, PathBuf::from("output_dir"));
            assert!(!args.per_channel);
            assert!(!args.verify);
            assert_eq!(args.script_path, Some(PathBuf::from("custom/quantize.py")));
        }
        _ => panic!("Quantize コマンドがパースされること。"),
    }
}

#[test]
fn test_parse_healthcheck_defaults() {
    let args = ["sokuto", "healthcheck"];
    let cli = Cli::try_parse_from(args).expect("引数パースに成功すること。");

    match cli.command {
        Commands::Healthcheck(args) => {
            assert_eq!(args.url, "http://127.0.0.1:3000/ready");
            assert_eq!(args.timeout_secs, 3);
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
        Commands::Healthcheck(args) => {
            assert_eq!(args.url, "http://localhost:8080/health");
            assert_eq!(args.timeout_secs, 5);
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
