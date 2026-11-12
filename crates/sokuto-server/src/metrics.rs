//! # Prometheus 監視メトリクスモジュール
//!
//! リクエスト数、レイテンシ、バッチサイズ、質問タイプ、および確信度分布の
//! メトリクス計測と Prometheus エクスポーターのセットアップを提供する。

use metrics::{counter, describe_counter, describe_histogram, histogram};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

/// リクエスト総数メトリクス名。
pub const METRIC_REQUESTS_TOTAL: &str = "sokuto_requests_total";

/// 推論エンジン処理時間(秒)メトリクス名。
pub const METRIC_INFERENCE_DURATION_SECONDS: &str = "sokuto_inference_duration_seconds";

/// HTTP 全体処理時間(秒)メトリクス名。
pub const METRIC_HTTP_DURATION_SECONDS: &str = "sokuto_http_duration_seconds";

/// バッチサイズ(質問数)メトリクス名。
pub const METRIC_BATCH_SIZE: &str = "sokuto_batch_size";

/// 質問プリミティブ別処理数メトリクス名。
pub const METRIC_QUESTION_TYPE_TOTAL: &str = "sokuto_question_type_total";

/// 判定確信度スコア分布メトリクス名。
pub const METRIC_CONFIDENCE_SCORE: &str = "sokuto_confidence_score";

/// 確信度ゲーティングルーティング数メトリクス名。
pub const METRIC_GATING_ROUTES_TOTAL: &str = "sokuto_gating_routes_total";

/// DAG 全体実行時間(秒)メトリクス名。
pub const METRIC_DAG_EXECUTION_DURATION_SECONDS: &str = "sokuto_dag_execution_duration_seconds";

/// DAG ノード別実行時間(秒)メトリクス名。
pub const METRIC_DAG_NODE_DURATION_SECONDS: &str = "sokuto_dag_node_duration_seconds";

/// DAG ノード実行回数メトリクス名。
pub const METRIC_DAG_NODE_EXECUTIONS_TOTAL: &str = "sokuto_dag_node_executions_total";

/// 粗密ルーティング Soft-Beam 発動回数メトリクス名。
pub const METRIC_HIERARCHICAL_SOFT_BEAM_TOTAL: &str = "sokuto_hierarchical_soft_beam_total";

/// 粗密ルーティング大分類 Early-Exit 回数メトリクス名。
pub const METRIC_HIERARCHICAL_EARLY_EXIT_TOTAL: &str = "sokuto_hierarchical_early_exit_total";

/// Prometheus メトリクスレコーダーを初期化し、テキストレンダリング用ハンドルを取得する。
pub fn setup_metrics_recorder() -> Result<PrometheusHandle, Box<dyn std::error::Error + Send + Sync>>
{
    let handle = PrometheusBuilder::new().install_recorder()?;

    register_initial_metrics();

    Ok(handle)
}

/// カスタムメトリクスの説明 (`describe_*`) および起動時の初期値登録を行う。
fn register_initial_metrics() {
    describe_counter!(METRIC_REQUESTS_TOTAL, "HTTP リクエスト完了総数。");
    describe_histogram!(METRIC_HTTP_DURATION_SECONDS, "HTTP 全体処理時間(秒)。");
    describe_histogram!(
        METRIC_INFERENCE_DURATION_SECONDS,
        "推論エンジン処理時間(秒)。"
    );
    describe_histogram!(METRIC_BATCH_SIZE, "バッチサイズ(質問数)。");
    describe_counter!(METRIC_QUESTION_TYPE_TOTAL, "質問プリミティブ別処理数。");
    describe_histogram!(METRIC_CONFIDENCE_SCORE, "判定確信度スコア分布。");
    describe_counter!(
        METRIC_GATING_ROUTES_TOTAL,
        "確信度ゲーティングルーティング数。"
    );
    describe_histogram!(
        METRIC_DAG_EXECUTION_DURATION_SECONDS,
        "DAG 全体実行時間(秒)。"
    );
    describe_histogram!(
        METRIC_DAG_NODE_DURATION_SECONDS,
        "DAG ノード別実行時間(秒)。"
    );
    describe_counter!(METRIC_DAG_NODE_EXECUTIONS_TOTAL, "DAG ノード実行回数。");
    describe_counter!(
        METRIC_HIERARCHICAL_SOFT_BEAM_TOTAL,
        "粗密ルーティング Soft-Beam 発動回数。"
    );
    describe_counter!(
        METRIC_HIERARCHICAL_EARLY_EXIT_TOTAL,
        "粗密ルーティング大分類 Early-Exit 回数。"
    );

    // 初回スクレイピング時に時系列が露出されるよう初期値 0 で事前登録。
    counter!(
        METRIC_REQUESTS_TOTAL,
        "endpoint" => "/v1/systemone",
        "status" => "200"
    )
    .increment(0);
    counter!(METRIC_QUESTION_TYPE_TOTAL, "type" => "choice").increment(0);
    counter!(METRIC_QUESTION_TYPE_TOTAL, "type" => "score").increment(0);
    counter!(METRIC_QUESTION_TYPE_TOTAL, "type" => "noul").increment(0);
    counter!(
        METRIC_GATING_ROUTES_TOTAL,
        "route" => "auto_execute",
        "type" => "choice"
    )
    .increment(0);
    counter!(
        METRIC_DAG_NODE_EXECUTIONS_TOTAL,
        "dag_id" => "default",
        "node_id" => "root",
        "node_type" => "inference"
    )
    .increment(0);
    counter!(METRIC_HIERARCHICAL_SOFT_BEAM_TOTAL).increment(0);
    counter!(METRIC_HIERARCHICAL_EARLY_EXIT_TOTAL).increment(0);

    histogram!(METRIC_INFERENCE_DURATION_SECONDS).record(0.0);
    histogram!(
        METRIC_HTTP_DURATION_SECONDS,
        "endpoint" => "/v1/systemone"
    )
    .record(0.0);
    histogram!(METRIC_BATCH_SIZE).record(0.0);
    histogram!(METRIC_CONFIDENCE_SCORE).record(0.0);
    histogram!(
        METRIC_DAG_EXECUTION_DURATION_SECONDS,
        "dag_id" => "default",
        "status" => "success"
    )
    .record(0.0);
    histogram!(
        METRIC_DAG_NODE_DURATION_SECONDS,
        "dag_id" => "default",
        "node_id" => "root",
        "node_type" => "inference"
    )
    .record(0.0);
}

/// HTTP リクエスト完了カウンターを記録する。
pub fn record_http_request(endpoint: &'static str, status: u16) {
    counter!(
        METRIC_REQUESTS_TOTAL,
        "endpoint" => endpoint,
        "status" => status.to_string(),
    )
    .increment(1);
}

/// HTTP レスポンス全体の所要時間(秒)を記録する。
pub fn record_http_duration(endpoint: &'static str, duration_secs: f64) {
    histogram!(
        METRIC_HTTP_DURATION_SECONDS,
        "endpoint" => endpoint,
    )
    .record(duration_secs);
}

/// 推論エンジンの所要時間(秒)およびバッチサイズを記録する。
pub fn record_inference_metrics(duration_secs: f64, batch_size: usize) {
    histogram!(METRIC_INFERENCE_DURATION_SECONDS).record(duration_secs);
    histogram!(METRIC_BATCH_SIZE).record(batch_size as f64);
}

/// 質問プリミティブ別の処理数を記録する。
pub fn record_question_type(question_type: &'static str) {
    counter!(
        METRIC_QUESTION_TYPE_TOTAL,
        "type" => question_type,
    )
    .increment(1);
}

/// 判定確信度スコアを記録する。
pub fn record_confidence(confidence: f64) {
    histogram!(METRIC_CONFIDENCE_SCORE).record(confidence);
}

/// 確信度ゲーティングのルーティング結果数を記録する。
pub fn record_gating_route(route: &str, question_type: &str) {
    counter!(
        METRIC_GATING_ROUTES_TOTAL,
        "route" => route.to_string(),
        "type" => question_type.to_string(),
    )
    .increment(1);
}

/// DAG 全体の実行所要時間(秒)を記録する。
pub fn record_dag_execution(dag_id: &str, status: &str, duration_secs: f64) {
    histogram!(
        METRIC_DAG_EXECUTION_DURATION_SECONDS,
        "dag_id" => dag_id.to_string(),
        "status" => status.to_string(),
    )
    .record(duration_secs);
}

/// DAG ノードごとの実行所要時間(秒)および実行数を記録する。
pub fn record_dag_node_execution(dag_id: &str, node_id: &str, node_type: &str, duration_secs: f64) {
    histogram!(
        METRIC_DAG_NODE_DURATION_SECONDS,
        "dag_id" => dag_id.to_string(),
        "node_id" => node_id.to_string(),
        "node_type" => node_type.to_string(),
    )
    .record(duration_secs);

    counter!(
        METRIC_DAG_NODE_EXECUTIONS_TOTAL,
        "dag_id" => dag_id.to_string(),
        "node_id" => node_id.to_string(),
        "node_type" => node_type.to_string(),
    )
    .increment(1);
}

/// 粗密ルーティングにおける Soft-Beam (Top-2 マージ救済) 発動数を記録する。
pub fn record_hierarchical_soft_beam() {
    counter!(METRIC_HIERARCHICAL_SOFT_BEAM_TOTAL).increment(1);
}

/// 粗密ルーティングにおける大分類 Early-Exit 発動数を記録する。
pub fn record_hierarchical_early_exit() {
    counter!(METRIC_HIERARCHICAL_EARLY_EXIT_TOTAL).increment(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_setup_metrics_recorder_initializes_metrics() {
        let handle = setup_metrics_recorder().expect("Prometheus レコーダーの初期化に成功。");
        let rendered = handle.render();

        let expected_metrics = [
            METRIC_REQUESTS_TOTAL,
            METRIC_INFERENCE_DURATION_SECONDS,
            METRIC_HTTP_DURATION_SECONDS,
            METRIC_BATCH_SIZE,
            METRIC_QUESTION_TYPE_TOTAL,
            METRIC_CONFIDENCE_SCORE,
            METRIC_GATING_ROUTES_TOTAL,
            METRIC_DAG_EXECUTION_DURATION_SECONDS,
            METRIC_DAG_NODE_DURATION_SECONDS,
            METRIC_DAG_NODE_EXECUTIONS_TOTAL,
            METRIC_HIERARCHICAL_SOFT_BEAM_TOTAL,
            METRIC_HIERARCHICAL_EARLY_EXIT_TOTAL,
        ];

        for metric in expected_metrics {
            assert!(
                rendered.contains(metric),
                "初期レンダリング出力にメトリクス '{}' が含まれること。",
                metric
            );
        }
    }
}
