//! # 推論レイテンシ・スループット測定ベンチマークコマンド
//!
//! ネットワークオーバーヘッドを排除したインプロセスエンジン直接駆動により、
//! 単一質問、並列バッチ、大規模候補探索 (Coarse-to-Fine)、およびスクラッチパッド再利用時の
//! 純粋な計算レイテンシとスループットを精密に計測する。

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use sokuto_core::contract::calibration::CalibrationConfig;
use sokuto_core::decision::evaluate_question;
use sokuto_core::schema::Question;
use sokuto_runtime::engine::coarse::CoarseToFineConfig;
use sokuto_runtime::engine::{BatchScratchpad, InferenceEngine};
use sokuto_runtime::tokenizer::JevTokenizer;

use crate::commands::serve::validate_model_dir;
use crate::commands::session_config::SessionConfigBuilder;

/// ベンチマーク実行構成パラメータ。
#[derive(Debug, Clone)]
pub struct BenchmarkArgs {
    /// モデルファイル格納ディレクトリ。
    pub model_dir: PathBuf,
    /// 測定反復回数。
    pub iterations: usize,
    /// ウォームアップ反復回数。
    pub warmup: usize,
    /// 優先 Execution Provider。
    pub provider: Option<String>,
    /// 実行対象シナリオ ("all", "single", "batch", "coarse", "scratchpad")。
    pub scenario: String,
    /// セッションプールサイズ。
    pub pool_size: usize,
    /// 単一オペレータ内の並列スレッド数 (intra-op)。
    pub intra_threads: Option<usize>,
    /// 複数オペレータ間の並列スレッド数 (inter-op)。
    pub inter_threads: Option<usize>,
    /// CPU メモリアリーナを有効化するかどうか。
    pub enable_mem_arena: bool,
    /// CI 連携用 JSON 形式出力フラグ。
    pub json: bool,
}

/// 単一シナリオの計測統計サマリー。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioStats {
    /// シナリオ識別名。
    pub name: String,
    /// 評価質問数。
    pub total_decisions: usize,
    /// 実行反復回数。
    pub iterations: usize,
    /// 最小レイテンシ (ミリ秒)。
    pub min_ms: f64,
    /// 平均レイテンシ (ミリ秒)。
    pub mean_ms: f64,
    /// 最大レイテンシ (ミリ秒)。
    pub max_ms: f64,
    /// 50 パーセンタイル (p50) レイテンシ (ミリ秒)。
    pub p50_ms: f64,
    /// 90 パーセンタイル (p90) レイテンシ (ミリ秒)。
    pub p90_ms: f64,
    /// 95 パーセンタイル (p95) レイテンシ (ミリ秒)。
    pub p95_ms: f64,
    /// 99 パーセンタイル (p99) レイテンシ (ミリ秒)。
    pub p99_ms: f64,
    /// スループット (decisions / 秒)。
    pub throughput_dps: f64,
}

/// ベンチマーク全体の実行結果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuiteResult {
    /// 使用した Execution Provider 名。
    pub active_provider: String,
    /// モデルディレクトリ。
    pub model_dir: String,
    /// 各シナリオの計測結果一覧。
    pub scenarios: Vec<ScenarioStats>,
}

/// 時間測定値配列から統計量を算出する。
fn compute_stats(name: &str, decisions_per_run: usize, durations_ms: &mut [f64]) -> ScenarioStats {
    durations_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = durations_ms.len();

    let min_ms = durations_ms.first().copied().unwrap_or(0.0);
    let max_ms = durations_ms.last().copied().unwrap_or(0.0);
    let sum: f64 = durations_ms.iter().sum();
    let mean_ms = if n > 0 { sum / (n as f64) } else { 0.0 };

    let p50_ms = percentile(durations_ms, 50.0);
    let p90_ms = percentile(durations_ms, 90.0);
    let p95_ms = percentile(durations_ms, 95.0);
    let p99_ms = percentile(durations_ms, 99.0);

    let throughput_dps = if mean_ms > 0.0 {
        (decisions_per_run as f64) / (mean_ms / 1000.0)
    } else {
        0.0
    };

    ScenarioStats {
        name: name.to_string(),
        total_decisions: decisions_per_run * n,
        iterations: n,
        min_ms,
        mean_ms,
        max_ms,
        p50_ms,
        p90_ms,
        p95_ms,
        p99_ms,
        throughput_dps,
    }
}

/// パーセンタイル値を算出する。
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * ((sorted.len() - 1) as f64)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// 指定されたシナリオ操作をウォームアップ実行および本測定し、統計量を算出する。
fn measure_scenario<F, T>(
    name: &str,
    decisions_per_run: usize,
    warmup: usize,
    iterations: usize,
    mut op: F,
) -> Result<ScenarioStats, Box<dyn std::error::Error + Send + Sync>>
where
    F: FnMut() -> Result<T, Box<dyn std::error::Error + Send + Sync>>,
{
    for _ in 0..warmup {
        let _ = op()?;
    }
    let mut durations = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = op()?;
        durations.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(compute_stats(name, decisions_per_run, &mut durations))
}

/// ベンチマークコマンドを実行する。
pub fn run_benchmark(args: BenchmarkArgs) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // 1. モデルディレクトリ検証
    if let Err(err) = validate_model_dir(&args.model_dir) {
        eprintln!("エラー: {err}");
        return Err(err.into());
    }

    // 2. セッション設定の構築
    let session_config = SessionConfigBuilder::new()
        .pool_size(args.pool_size)
        .intra_threads(args.intra_threads)
        .inter_threads(args.inter_threads)
        .provider_opt(args.provider.as_deref())
        .enable_mem_arena(args.enable_mem_arena)
        .build()?;

    let onnx_path = args.model_dir.join("model.onnx");
    let tok_path = args.model_dir.join("tokenizer.json");
    let calib_path = args.model_dir.join("calibration.json");

    eprintln!(
        "推論エンジンおよびトークナイザーをロード中 (パス: {})...",
        args.model_dir.display()
    );

    let engine = InferenceEngine::new(&onnx_path, session_config)?;
    let tokenizer = JevTokenizer::from_file(&tok_path)?;

    let calib_config = if calib_path.exists() {
        let calib_json = fs::read_to_string(&calib_path)?;
        CalibrationConfig::from_json_str(&calib_json)?
    } else {
        CalibrationConfig::default()
    };

    let active_provider = engine.active_provider().name().to_string();
    eprintln!("アクティブ Execution Provider: {active_provider}");
    eprintln!(
        "測定設定: 反復回数={}, ウォームアップ={}, シナリオ={}",
        args.iterations, args.warmup, args.scenario
    );

    let mut suite_results = BenchmarkSuiteResult {
        active_provider: active_provider.clone(),
        model_dir: args.model_dir.display().to_string(),
        scenarios: Vec::new(),
    };

    let scenario_filter = args.scenario.to_lowercase();
    let should_run_single = scenario_filter == "all" || scenario_filter == "single";
    let should_run_batch = scenario_filter == "all" || scenario_filter == "batch";
    let should_run_coarse = scenario_filter == "all" || scenario_filter == "coarse";
    let should_run_scratch = scenario_filter == "all" || scenario_filter == "scratchpad";

    // --- シナリオ 1〜3: Single Decision (Choice, Score, Noul) ---
    if should_run_single {
        eprintln!("単一質問シナリオ (Single Choice, Score, Noul) を測定中...");

        // 1. Single Choice (K=3)
        let state_choice =
            "顧客から注文キャンセルのリクエストを受信しました。注文ステータスは発送準備中です。";
        let mut criteria_choice = IndexMap::new();
        criteria_choice.insert(
            "cancel_order".to_string(),
            "注文を直ちにキャンセルする".to_string(),
        );
        criteria_choice.insert(
            "contact_support".to_string(),
            "サポート窓口へ転送する".to_string(),
        );
        criteria_choice.insert("ignore".to_string(), "特に対処せず保留する".to_string());
        let question_choice =
            Question::new_choice("適切な対応を選択してください。", criteria_choice);
        let tokenized_choice = tokenizer.encode_question(state_choice, &question_choice)?;

        // 生推論ループ (事前トークナイズ入力に対する純粋なフォワードパス)
        suite_results.scenarios.push(measure_scenario(
            "Single Choice (K=3, Raw)",
            1,
            args.warmup,
            args.iterations,
            || {
                engine
                    .forward_question(&tokenized_choice)
                    .map_err(Into::into)
            },
        )?);

        // E2E パイプライン (トークナイズ + 推論 + 決定プリミティブ数理解決)
        suite_results.scenarios.push(measure_scenario(
            "Single Choice (K=3, E2E)",
            1,
            args.warmup,
            args.iterations,
            || {
                let tok = tokenizer.encode_question(state_choice, &question_choice)?;
                let logits = engine.forward_question(&tok)?;
                let _ = evaluate_question(&question_choice, &logits, &calib_config)?;
                Ok(())
            },
        )?);

        // 2. Single Score (K=5)
        let state_score =
            "過去 5 分間に不正なパスワード試行が同一 IP アドレスから 200 回検知されました。";
        let criteria_score = vec![
            "極めて安全 (影響なし)".to_string(),
            "低リスク (要注視)".to_string(),
            "中リスク (調査推奨)".to_string(),
            "高リスク (要即時対応)".to_string(),
            "致命的インシデント (全遮断)".to_string(),
        ];
        let question_score = Question::new_score(
            "セキュリティ脅威の深刻度を段階評価してください。",
            criteria_score,
        );
        let tokenized_score = tokenizer.encode_question(state_score, &question_score)?;

        suite_results.scenarios.push(measure_scenario(
            "Single Score (K=5, Raw)",
            1,
            args.warmup,
            args.iterations,
            || {
                engine
                    .forward_question(&tokenized_score)
                    .map_err(Into::into)
            },
        )?);

        suite_results.scenarios.push(measure_scenario(
            "Single Score (K=5, E2E)",
            1,
            args.warmup,
            args.iterations,
            || {
                let tok = tokenizer.encode_question(state_score, &question_score)?;
                let logits = engine.forward_question(&tok)?;
                let _ = evaluate_question(&question_score, &logits, &calib_config)?;
                Ok(())
            },
        )?);

        // 3. Single Noul (真偽判定)
        let state_noul = "サーバーの CPU 使用率が 98%、空きメモリが 50MB 未満です。";
        let question_noul = Question::new_noul("直ちにアラートを発報すべき緊急事態ですか？");
        let tokenized_noul = tokenizer.encode_question(state_noul, &question_noul)?;

        suite_results.scenarios.push(measure_scenario(
            "Single Noul (Binary, Raw)",
            1,
            args.warmup,
            args.iterations,
            || engine.forward_question(&tokenized_noul).map_err(Into::into),
        )?);

        suite_results.scenarios.push(measure_scenario(
            "Single Noul (Binary, E2E)",
            1,
            args.warmup,
            args.iterations,
            || {
                let tok = tokenizer.encode_question(state_noul, &question_noul)?;
                let logits = engine.forward_question(&tok)?;
                let _ = evaluate_question(&question_noul, &logits, &calib_config)?;
                Ok(())
            },
        )?);
    }

    // --- シナリオ 4: Batch Scaling (4, 8, 16 Questions) ---
    if should_run_batch {
        eprintln!("並列バッチシナリオ (Batch Scaling: Q=4, 8, 16) を測定中...");
        let state = "ユーザーが新規アカウント登録を行い、プロファイル情報を入力しました。";

        for &batch_size in &[4, 8, 16] {
            let mut questions = IndexMap::with_capacity(batch_size);
            for i in 0..batch_size {
                let mut c = IndexMap::new();
                c.insert(format!("approve_{i}"), "承認する".to_string());
                c.insert(format!("review_{i}"), "審査に回す".to_string());
                c.insert(format!("reject_{i}"), "却下する".to_string());
                questions.insert(
                    format!("q_{i}"),
                    Question::new_choice(format!("質問項目 {i} の判定を行ってください。"), c),
                );
            }
            let batch = tokenizer.encode_batch_questions(state, &questions)?;

            // 生推論ループ (事前トークナイズ入力に対する純粋なバッチフォワードパス)
            suite_results.scenarios.push(measure_scenario(
                &format!("Batch Scaling (Q={batch_size}, Raw)"),
                batch_size,
                args.warmup,
                args.iterations,
                || engine.forward_batch_tokenized(&batch).map_err(Into::into),
            )?);

            // E2E パイプライン (トークナイズ + バッチ推論 + 各質問の決定数理解決・ゲーティング)
            suite_results.scenarios.push(measure_scenario(
                &format!("Batch Scaling (Q={batch_size}, E2E)"),
                batch_size,
                args.warmup,
                args.iterations,
                || {
                    engine
                        .evaluate_batch_questions(&tokenizer, state, &questions, &calib_config)
                        .map_err(Into::into)
                },
            )?);
        }
    }

    // --- シナリオ 5: Coarse-to-Fine (K=77 大規模候補探索対照) ---
    if should_run_coarse {
        eprintln!("高基数候補探索シナリオ (Flat vs Coarse-to-Fine: K=77) を測定中...");
        let state =
            "ATM から現金を引き出そうとしたところ、カードが取り込まれて返却されませんでした。";
        let mut large_criteria = IndexMap::with_capacity(77);
        for i in 0..77 {
            large_criteria.insert(
                format!("category_{i:02}"),
                format!("バンキング意図カテゴリ {i:02} に関する手続き"),
            );
        }
        let question = Question::new_choice("問い合わせの意図を分類してください。", large_criteria);
        let coarse_config = CoarseToFineConfig {
            threshold: 30,
            top_m: 16,
            ..Default::default()
        };

        // 通常 Flat 評価 (K=77 の全候補に対する E2E パイプライン)
        suite_results.scenarios.push(measure_scenario(
            "Flat Choice (K=77, E2E)",
            1,
            args.warmup,
            args.iterations,
            || {
                let tok = tokenizer.encode_question(state, &question)?;
                let logits = engine.forward_question(&tok)?;
                let _ = evaluate_question(&question, &logits, &calib_config)?;
                Ok(())
            },
        )?);

        // Coarse-to-Fine 評価 (粗密2段階探索による枝刈り E2E パイプライン)
        suite_results.scenarios.push(measure_scenario(
            "Coarse-to-Fine (K=77, E2E)",
            1,
            args.warmup,
            args.iterations,
            || {
                engine
                    .evaluate_question_coarse_to_fine(
                        &tokenizer,
                        state,
                        &question,
                        &calib_config,
                        &coarse_config,
                    )
                    .map_err(Into::into)
            },
        )?);
    }

    // --- シナリオ 6: Scratchpad 再利用によるアロケーション抑制効果対照 ---
    if should_run_scratch {
        eprintln!("スクラッチパッド再利用シナリオ (Standard vs Scratchpad: Q=8) を測定中...");
        let state = "注文履歴の確認と配送日時の変更手続きを受け付けました。";
        let mut questions = IndexMap::with_capacity(8);
        for i in 0..8 {
            let mut c = IndexMap::new();
            c.insert("allow".to_string(), "変更を許可する".to_string());
            c.insert("deny".to_string(), "変更を拒否する".to_string());
            questions.insert(
                format!("check_{i}"),
                Question::new_choice("変更可否を判定してください。", c),
            );
        }

        // 通常バッチ評価 (毎回新規アロケーションを行う E2E パイプライン)
        suite_results.scenarios.push(measure_scenario(
            "Batch Standard (Q=8, E2E)",
            8,
            args.warmup,
            args.iterations,
            || {
                engine
                    .evaluate_batch_questions(&tokenizer, state, &questions, &calib_config)
                    .map_err(Into::into)
            },
        )?);

        // スクラッチパッド再利用バッチ評価 (アロケーションを抑える E2E パイプライン)
        let mut scratchpad = BatchScratchpad::default();
        suite_results.scenarios.push(measure_scenario(
            "Batch Scratchpad (Q=8, E2E)",
            8,
            args.warmup,
            args.iterations,
            || {
                engine
                    .evaluate_batch_questions_with_scratchpad(
                        &tokenizer,
                        state,
                        &questions,
                        &calib_config,
                        &mut scratchpad,
                    )
                    .map_err(Into::into)
            },
        )?);
    }

    // 4. 結果の出力
    if args.json {
        let json_output = serde_json::to_string_pretty(&suite_results)?;
        println!("{json_output}");
    } else {
        render_table(&suite_results);
    }

    Ok(())
}

/// 計測結果を人間向けの整形テーブルとして標準出力に描画する。
fn render_table(results: &BenchmarkSuiteResult) {
    println!();
    println!(
        "================================================================================================================="
    );
    println!(
        " Sokuto 推論性能ベンチマーク (EP: {}, モデル: {})",
        results.active_provider, results.model_dir
    );
    println!(
        "================================================================================================================="
    );
    println!(
        "{:<32} | {:>6} | {:>8} | {:>8} | {:>8} | {:>8} | {:>8} | {:>14}",
        "シナリオ", "反復", "Min(ms)", "Mean(ms)", "P50(ms)", "P90(ms)", "P99(ms)", "Throughput"
    );
    println!(
        "-----------------------------------------------------------------------------------------------------------------"
    );

    for s in &results.scenarios {
        let throughput_str = format!("{:.1} dec/s", s.throughput_dps);
        println!(
            "{:<32} | {:>6} | {:>8.2} | {:>8.2} | {:>8.2} | {:>8.2} | {:>8.2} | {:>14}",
            s.name, s.iterations, s.min_ms, s.mean_ms, s.p50_ms, s.p90_ms, s.p99_ms, throughput_str
        );
    }
    println!(
        "================================================================================================================="
    );
    println!();
}
