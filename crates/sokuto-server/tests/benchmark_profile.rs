//! # Pillar 1: ランタイム内部レイテンシ・プロファイリング検証スイート
//!
//! 前処理ガードレール、トークナイズ、ONNX forward (ホットパス)、決定数理解決の
//! 各区間レイテンシをナノ秒精度でプロファイリングし、4つの代表シナリオにおける
//! パーセンタイル分布および CPU 環境での p50 < 300ms 達成を検証する。

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::{BenchmarkStats, ScenarioBuilder, init_benchmark_app_state};
use sokuto_core::gating::{evaluate_answer_gating, evaluate_response_routing};
use sokuto_core::schema::SystemOneRequest;

/// 区間別所要時間の計測レコード。
#[derive(Debug, Default, Clone)]
struct BreakdownProfile {
    /// 前処理ガードレール所要時間。
    pub guardrail: Duration,
    /// トークナイズ所要時間。
    pub tokenize: Duration,
    /// ONNX Runtime 推論所要時間。
    pub onnx_forward: Duration,
    /// 決定数理解決・ゲーティング所要時間。
    pub decision_math: Duration,
    /// 合計所要時間。
    pub total: Duration,
}

/// 単一リクエストの内部区間プロファイル計測関数。
fn profile_single_request(
    state: &Arc<sokuto_server::state::AppState>,
    mut req: SystemOneRequest,
) -> (BreakdownProfile, usize) {
    let t_total_start = Instant::now();

    // 1. スキーマ検証
    req.validate().expect("スキーマ検証に失敗しました。");

    // 2. 前処理ガードレール
    let t_guard_start = Instant::now();
    state
        .guardrail_pipeline
        .process(&mut req, None)
        .expect("ガードレール処理に失敗しました。");
    let guardrail = t_guard_start.elapsed();

    let question_count = req.questions.len();
    let state_text = match &req.state {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    };

    // 3. トークナイズ
    let t_tok_start = Instant::now();
    let _state_tokens = state
        .tokenizer
        .encode_state(&state_text)
        .map(|toks| toks.len())
        .unwrap_or(0);
    let tokenize = t_tok_start.elapsed();

    // 4. ONNX Runtime 推論
    let t_onnx_start = Instant::now();
    let mut answers = state
        .engine
        .evaluate_batch_questions_coarse_to_fine_chunked(
            &state.tokenizer,
            &state_text,
            &req.questions,
            &state.calib_config,
            &state.coarse_config,
            state.chunk_size,
        )
        .expect("推論エンジン実行に失敗しました。");
    let onnx_forward = t_onnx_start.elapsed();

    // 5. 決定数理解決 & 確信度ゲーティング
    let t_math_start = Instant::now();
    let gating_config = match req.gating.as_ref() {
        Some(patch) => state.gating_config.apply_patch(patch).unwrap(),
        None => state.gating_config.clone(),
    };
    if gating_config.enabled {
        for (qid, answer) in answers.iter_mut() {
            let q_def = req.questions.get(qid);
            let meta = evaluate_answer_gating(answer, Some(qid), q_def, &gating_config);
            answer.gating = Some(meta);
        }
        let _summary = evaluate_response_routing(&answers);
    }
    let decision_math = t_math_start.elapsed();

    let total = t_total_start.elapsed();

    (
        BreakdownProfile {
            guardrail,
            tokenize,
            onnx_forward,
            decision_math,
            total,
        },
        question_count,
    )
}

#[tokio::test]
async fn test_pillar1_latency_profiling_and_breakdown() {
    let state = match init_benchmark_app_state(128, 1) {
        Some(s) => s,
        None => {
            eprintln!("モデルファイルが見つからないため、ベンチマークをスキップします。");
            return;
        }
    };

    println!("\n================================================================================");
    println!(" [Pillar 1] ランタイム内部レイテンシ・プロファイリング & 区間ブレークダウン検証");
    println!("================================================================================\n");

    // 環境変数 SOKUTO_BENCH_FULL=1 が設定されている場合はフル試行、通常テスト時は軽量試行とする。
    let is_full_bench = std::env::var("SOKUTO_BENCH_FULL")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    // 1. ウォームアップ実行 (コールドスタートの排除)
    let warmup_count = if is_full_bench { 20 } else { 3 };
    println!("[1/5] ウォームアップを {} 回実行中...", warmup_count);
    let warmup_req = ScenarioBuilder::build_lightweight();
    for _ in 0..warmup_count {
        let _ = profile_single_request(&state, warmup_req.clone());
    }
    println!("ウォームアップ完了。キャッシュおよびアリーナ確保が定常化しました。\n");

    // 測定シナリオ定義
    let scenarios = vec![
        (
            "Scenario A: Lightweight (Choice x 1)",
            ScenarioBuilder::build_lightweight(),
            if is_full_bench { 30 } else { 5 },
        ),
        (
            "Scenario B: Prefix-Sharing (Mixed x 8)",
            ScenarioBuilder::build_prefix_sharing_batch(),
            if is_full_bench { 20 } else { 3 },
        ),
        (
            "Scenario C: Coarse-to-Fine (77 Options x 1)",
            ScenarioBuilder::build_coarse_to_fine_77(),
            if is_full_bench { 20 } else { 3 },
        ),
        (
            "Scenario D: Stress Heavy (State 2000tok, 16 Questions)",
            ScenarioBuilder::build_stress_heavy(),
            if is_full_bench { 10 } else { 2 },
        ),
    ];

    let mut overall_stats = Vec::new();

    for (name, req_template, iterations) in scenarios {
        let mut total_durations = Vec::with_capacity(iterations);
        let mut guard_durations = Vec::with_capacity(iterations);
        let mut tok_durations = Vec::with_capacity(iterations);
        let mut onnx_durations = Vec::with_capacity(iterations);
        let mut math_durations = Vec::with_capacity(iterations);

        let q_count = req_template.questions.len();
        let loop_start = Instant::now();

        for _ in 0..iterations {
            let (profile, _) = profile_single_request(&state, req_template.clone());
            total_durations.push(profile.total);
            guard_durations.push(profile.guardrail);
            tok_durations.push(profile.tokenize);
            onnx_durations.push(profile.onnx_forward);
            math_durations.push(profile.decision_math);
        }

        let loop_elapsed = loop_start.elapsed();
        let stats = BenchmarkStats::calculate(name, total_durations.clone(), q_count, loop_elapsed);
        overall_stats.push(stats.clone());

        // 平均内訳の算出
        let avg_guard: f64 = guard_durations
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / iterations as f64;
        let avg_tok: f64 = tok_durations
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / iterations as f64;
        let avg_onnx: f64 = onnx_durations
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / iterations as f64;
        let avg_math: f64 = math_durations
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / iterations as f64;
        let avg_total: f64 = total_durations
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / iterations as f64;

        println!("### {}", name);
        println!("- 試行回数: {} 回, 質問数/Req: {} 問", iterations, q_count);
        println!(
            "- レイテンシ: p50 = {:.2} ms, p95 = {:.2} ms, p99 = {:.2} ms, Mean = {:.2} ms",
            stats.p50.as_secs_f64() * 1000.0,
            stats.p95.as_secs_f64() * 1000.0,
            stats.p99.as_secs_f64() * 1000.0,
            stats.mean.as_secs_f64() * 1000.0,
        );
        println!(
            "- スループット: RPS = {:.2} req/s, DPS = {:.2} decisions/s",
            stats.rps, stats.dps
        );
        println!("- 内部区間内訳 (平均):");
        println!(
            "  1. ガードレール前処理: {:>6.2} ms ({:>5.1}%)",
            avg_guard,
            (avg_guard / avg_total) * 100.0
        );
        println!(
            "  2. トークナイズ      : {:>6.2} ms ({:>5.1}%)",
            avg_tok,
            (avg_tok / avg_total) * 100.0
        );
        println!(
            "  3. ONNX 推論 (ホット): {:>6.2} ms ({:>5.1}%)",
            avg_onnx,
            (avg_onnx / avg_total) * 100.0
        );
        println!(
            "  4. 決定数理・ゲーティング: {:>6.2} ms ({:>5.1}%)",
            avg_math,
            (avg_math / avg_total) * 100.0
        );
        println!();
    }

    println!("#### [総合サマリーテーブル]");
    println!(
        "| シナリオ | サンプル数 | 最小 | p50 (中央値) | p95 | p99 | RPS | DPS (スループット) |"
    );
    println!("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |");
    for s in &overall_stats {
        println!("{}", s.to_markdown_row());
    }
    println!();

    // シナリオ A (Lightweight) の p50 が CPU 目標 (< 300ms) を達成していることを検証
    let scenario_a = &overall_stats[0];
    let p50_millis = scenario_a.p50.as_secs_f64() * 1000.0;
    println!(
        "[判定] シナリオ A 単一即応 p50: {:.2} ms (目標 < 300.0 ms)",
        p50_millis
    );
    assert!(
        p50_millis < 300.0,
        "CPU環境での単一質問 p50 ({:.2} ms) が目標基準 300ms を超過しました。",
        p50_millis
    );
}
