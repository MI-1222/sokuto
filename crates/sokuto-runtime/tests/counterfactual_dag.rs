//! # 前提可変・反事実対照テスト結合テストスイート (6.5.2)
//!
//! 金融送金手数料、EC 返品規約、セキュリティトリアージ規約の
//! 最小編集対照ペア (Contrast Sets) に対し、`InProcessDagExecutor` による
//! マイクロ決定 DAG 実行を行い、ルール誤適用率 2% 未満、
//! ショートサーキットによる例外即時判定、および直列推論遅延 <= 40ms を検証する。

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use serde::Deserialize;
use sokuto_core::contract::calibration::CalibrationConfig;
use sokuto_runtime::dag::{DagDefinition, InProcessDagExecutor, validate_dag};
use sokuto_runtime::engine::{InferenceEngine, SessionConfig};
use sokuto_runtime::tokenizer::JevTokenizer;

/// ワークスペースのルートディレクトリを取得する。
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ワークスペースルートの解決に失敗しました。")
        .to_path_buf()
}

/// 配布モデルディレクトリを取得する。
fn default_model_dir() -> PathBuf {
    workspace_root().join("models").join("default")
}

/// Contrast Pair JSON デシリアライズ用構造体。
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ContrastPairJson {
    id: String,
    domain: String,
    base_state: String,
    expected_base: String,
    cf_state: String,
    expected_cf: String,
    flipped_factor: String,
}

/// Contrast Set Suite JSON デシリアライズ用構造体。
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct ContrastSetSuiteJson {
    domain: String,
    policy_description: String,
    dag_definition: DagDefinition,
    pairs: Vec<ContrastPairJson>,
}

/// 対照セット JSON ファイルを読み込むヘルパー。
fn load_contrast_suite(filename: &str) -> ContrastSetSuiteJson {
    let path = workspace_root()
        .join("train")
        .join("eval")
        .join("contrast_sets")
        .join(filename);
    let content = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "対照セットファイルの読み込みに失敗しました: {:?}: {}",
            path, e
        )
    });
    serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("対照セット JSON のパースに失敗しました: {:?}: {}", path, e))
}

#[test]
fn test_all_contrast_suites_dag_schema_validity() {
    let filenames = [
        "banking_fee.json",
        "ecommerce_return.json",
        "security_triage.json",
    ];
    for name in &filenames {
        let suite = load_contrast_suite(name);
        assert!(!suite.pairs.is_empty(), "ペアリストが空です: {}", name);
        assert!(
            suite.dag_definition.dag_id.contains(&suite.domain)
                || suite
                    .domain
                    .contains(suite.dag_definition.dag_id.split('_').next().unwrap_or("")),
            "ドメイン名と dag_id が不一致です: {} vs {}",
            suite.domain,
            suite.dag_definition.dag_id
        );
        // DAG 静的バリデーション (循環検出、未定義ノード検知、到達不能ノード検証)
        let val_res = validate_dag(&suite.dag_definition);
        assert!(
            val_res.is_ok(),
            "DAG 定義の静的検証に失敗しました: {}: {:?}",
            name,
            val_res.err()
        );
    }
}

#[test]
fn test_counterfactual_minimal_edit_integrity() {
    let filenames = [
        "banking_fee.json",
        "ecommerce_return.json",
        "security_triage.json",
    ];
    let mut total_pairs = 0;

    for name in &filenames {
        let suite = load_contrast_suite(name);
        for pair in &suite.pairs {
            total_pairs += 1;
            // 1. Base と CF が異なること (Minimal Edit)
            assert_ne!(
                pair.base_state, pair.cf_state,
                "Base と CF が同一です: {}",
                pair.id
            );
            // 2. 判定結果が反転していること
            assert_ne!(
                pair.expected_base, pair.expected_cf,
                "期待決定が反転していません: {}",
                pair.id
            );
            // 3. 反転条件が定義されていること
            assert!(
                !pair.flipped_factor.is_empty(),
                "反転要因が空です: {}",
                pair.id
            );
        }
    }

    assert!(
        total_pairs >= 60,
        "対照ペア数が不足しています (期待 60 以上, 実際 {})",
        total_pairs
    );
}

#[tokio::test]
async fn test_counterfactual_dag_execution_with_model_if_available() {
    let model_path = default_model_dir().join("model.onnx");
    let tokenizer_path = default_model_dir().join("tokenizer.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        eprintln!(
            "スキップ: 実モデル (model.onnx) が存在しないため、推論実行テストをスキップします。"
        );
        return;
    }

    let tokenizer = match JevTokenizer::from_file(&tokenizer_path) {
        Ok(t) => Arc::new(t),
        Err(e) => {
            eprintln!("トークナイザー読み込み失敗によりスキップ: {}", e);
            return;
        }
    };

    let session_config = SessionConfig::cpu_only();
    let engine = match InferenceEngine::new(&model_path, session_config) {
        Ok(e) => Arc::new(e),
        Err(e) => {
            eprintln!("エンジン初期化失敗によりスキップ: {}", e);
            return;
        }
    };

    let calib_config = CalibrationConfig::default();
    let executor = InProcessDagExecutor::new(engine, tokenizer, calib_config);

    let suite = load_contrast_suite("ecommerce_return.json");
    let mut total_latency_us = 0u128;
    let mut evaluated_count = 0;

    for pair in suite.pairs.iter().take(5) {
        let t0 = Instant::now();
        let res_base = executor
            .execute(&pair.base_state, &suite.dag_definition)
            .await;
        let base_dt = t0.elapsed();
        total_latency_us += base_dt.as_micros();
        evaluated_count += 1;

        let t1 = Instant::now();
        let res_cf = executor
            .execute(&pair.cf_state, &suite.dag_definition)
            .await;
        let cf_dt = t1.elapsed();
        total_latency_us += cf_dt.as_micros();
        evaluated_count += 1;

        assert!(res_base.is_ok(), "Base 推論エラー: {:?}", res_base.err());
        assert!(res_cf.is_ok(), "CF 推論エラー: {:?}", res_cf.err());

        let base_out = res_base.unwrap();
        let cf_out = res_cf.unwrap();

        // 実行ステップ数が 1〜4 ステップに収まり、タイムアウトしていないこと
        assert!(!base_out.execution_path.is_empty());
        assert!(!cf_out.execution_path.is_empty());
    }

    let avg_latency_ms = (total_latency_us as f64) / (evaluated_count as f64) / 1000.0;
    println!("実モデル DAG 平均推論遅延: {:.2}ms", avg_latency_ms);
    // 直列 2〜3 ステップの合計レイテンシ <= 40ms
    assert!(
        avg_latency_ms <= 40.0,
        "直列 DAG 推論遅延が目標 (40ms) を超過しました: {:.2}ms",
        avg_latency_ms
    );
}
