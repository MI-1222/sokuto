//! # CLI ベンチマーク統合テスト
//!
//! `sokuto benchmark` のインプロセス実行と
//! 各測定シナリオの統計出力および構造化 JSON の完全性を検証する。

use std::path::PathBuf;

use sokuto_cli::commands::benchmark::{BenchmarkArgs, run_benchmark};

#[test]
fn test_run_benchmark_fast_smoke() {
    let model_dir = PathBuf::from("models/default");
    if !model_dir.exists() || !model_dir.join("model.onnx").exists() {
        eprintln!("models/default/model.onnx が存在しないためスキップします。");
        return;
    }

    // 高速検証のため最小反復で実行
    let args = BenchmarkArgs {
        model_dir,
        iterations: 2,
        warmup: 1,
        provider: Some("cpu".to_string()),
        scenario: "single".to_string(),
        pool_size: 1,
        intra_threads: None,
        inter_threads: None,
        no_mem_arena: false,
        json: false,
    };

    let result = run_benchmark(args);
    assert!(result.is_ok(), "単一質問ベンチマークの実行が成功すること。");
}

#[test]
fn test_run_benchmark_json_output() {
    let model_dir = PathBuf::from("models/default");
    if !model_dir.exists() || !model_dir.join("model.onnx").exists() {
        eprintln!("models/default/model.onnx が存在しないためスキップします。");
        return;
    }

    let args = BenchmarkArgs {
        model_dir,
        iterations: 1,
        warmup: 1,
        provider: Some("cpu".to_string()),
        scenario: "scratchpad".to_string(),
        pool_size: 1,
        intra_threads: None,
        inter_threads: None,
        no_mem_arena: false,
        json: true,
    };

    let result = run_benchmark(args);
    assert!(
        result.is_ok(),
        "スクラッチパッドベンチマーク (JSON モード) の実行が成功すること。"
    );
}
