//! # インプロセス DAG 実行器モジュール
//!
//! 非同期グラフ巡回、ゼロコピー文脈注入、および CPU 集約的推論の
//! `tokio::task::spawn_blocking` オフロードによる超低遅延オーケストレーションを提供する。

use std::sync::Arc;
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use sokuto_core::contract::calibration::CalibrationConfig;
use sokuto_core::schema::{Answer, AnswerPayload, Question};

use crate::dag::interpolate::{NodeContextValue, interpolate_question};
use crate::dag::pool::PoolError;
use crate::dag::schema::{
    CompiledCondition, CompiledDag, CompiledDagNode, DagDefinition, DagExecutionResult,
    DagNodeType, NodeCondition, StepExecutionResult,
};
use crate::dag::validator::{DagValidationError, compile_dag};
use crate::engine::InferenceEngine;
use crate::error::RuntimeError;
use crate::tokenizer::JevTokenizer;

/// DAG 実行時に発生するエラー型。
#[derive(Debug, thiserror::Error)]
pub enum DagError {
    /// 静的検証エラー。
    #[error("DAG 検証エラー: {0}。")]
    Validation(#[from] DagValidationError),

    /// セッションプール借用エラー。
    #[error("セッションプールエラー: {0}。")]
    Pool(#[from] PoolError),

    /// DAG 全体実行のタイムアウト。
    #[error("DAG 実行がタイムアウトしました ({0}ms)。")]
    Timeout(u64),

    /// グラフ巡回のステップ数上限超過 (無限ループ動的防壁)。
    #[error("DAG 実行ステップ数が上限 ({0}) を超過しました。")]
    StepLimitExceeded(usize),

    /// 指定ノードが存在しない。
    #[error("ノード '{0}' が見つかりません。")]
    NodeNotFound(String),

    /// ランタイム推論エンジン由来のエラー。
    #[error("推論ランタイムエラー: {0}。")]
    Runtime(#[from] RuntimeError),

    /// 並行タスクジョインエラーなど。
    #[error("DAG 実行失敗: {0}。")]
    ExecutionFailed(String),
}

/// インプロセス DAG 実行エンジン。
///
/// 単一プロンプトでの複合推論を避け、軽量モデルによる多段判定と
/// Rust 側での決定論的ブール論理・制御フローをミリ秒台で実行する。
pub struct InProcessDagExecutor {
    engine: Arc<InferenceEngine>,
    tokenizer: Arc<JevTokenizer>,
    calib_config: CalibrationConfig,
    max_steps: usize,
}

impl InProcessDagExecutor {
    /// 新規 DAG 実行エンジンを構築する。
    ///
    /// # 引数
    /// - `engine`: ONNX 推論エンジン。
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `calib_config`: 較正温度設定。
    pub fn new(
        engine: Arc<InferenceEngine>,
        tokenizer: Arc<JevTokenizer>,
        calib_config: CalibrationConfig,
    ) -> Self {
        Self {
            engine,
            tokenizer,
            calib_config,
            max_steps: 50,
        }
    }

    /// 実行ステップ数の安全上限を設定する。既定値は 50。
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// DAG を実行し、実行結果を返却する。
    ///
    /// 実行前に静的検証を行い、DAG 全体を `dag.timeout_ms` の制限時間でラップして実行する。
    ///
    /// # 引数
    /// - `state`: 入力コンテキスト文。
    /// - `dag`: 宣言的 DAG 定義。
    pub async fn execute(
        &self,
        state: &str,
        dag: &DagDefinition,
    ) -> Result<DagExecutionResult, DagError> {
        // 1. 静的検証 & 事前コンパイル (Fast-Fail)
        let compiled_dag = compile_dag(dag)?;
        self.execute_compiled(state, &compiled_dag).await
    }

    /// 事前コンパイル済みの DAG を実行し、実行結果を返却する。
    pub async fn execute_compiled(
        &self,
        state: &str,
        dag: &CompiledDag,
    ) -> Result<DagExecutionResult, DagError> {
        let timeout_duration = Duration::from_millis(dag.timeout_ms);
        let start_time = Instant::now();

        // 2. タイムアウト監視付きでステップループを実行
        let exec_result = tokio::time::timeout(timeout_duration, self.run_dag(state, dag)).await;

        match exec_result {
            Ok(Ok(mut res)) => {
                res.total_duration_ms = start_time.elapsed().as_secs_f64() * 1000.0;
                Ok(res)
            }
            Ok(Err(err)) => Err(err),
            Err(_) => Err(DagError::Timeout(dag.timeout_ms)),
        }
    }

    /// DAG ステップ実行ループ本体。
    async fn run_dag(
        &self,
        initial_state: &str,
        dag: &CompiledDag,
    ) -> Result<DagExecutionResult, DagError> {
        let mut context: IndexMap<String, NodeContextValue> = IndexMap::new();
        let mut steps: IndexMap<String, StepExecutionResult> = IndexMap::new();
        let mut execution_path: Vec<String> = Vec::new();

        // 外部入力である state は変数補間を行わず、Arc<str> で各非同期タスクへゼロコピー共有する
        let state: Arc<str> = Arc::from(initial_state);

        let mut curr_node_id = dag.entry_node.clone();
        let mut step_count = 0;
        let mut final_answer: Option<Answer> = None;
        let mut last_inference_node_id: Option<String> = None;
        let mut escalated = false;
        let mut escalation_reason = None;
        let mut escalation_node_id = None;
        let mut escalation_prompt = None;
        let mut escalation_gating = None;

        loop {
            if step_count >= self.max_steps {
                return Err(DagError::StepLimitExceeded(self.max_steps));
            }
            step_count += 1;
            let node = dag
                .nodes
                .get(&curr_node_id)
                .ok_or_else(|| DagError::NodeNotFound(curr_node_id.clone()))?;

            execution_path.push(curr_node_id.clone());
            let step_start = Instant::now();

            match node {
                CompiledDagNode::Inference {
                    question,
                    conditions,
                    reason,
                } => {
                    let handle = self.spawn_inference_task(Arc::clone(&state), question, &context);
                    let answer = Self::await_inference_task(handle, "スレッド結合失敗").await?;

                    // コンテキストに結果を反映
                    let ctx_val = answer_to_context_value(&answer);
                    context.insert(curr_node_id.clone(), ctx_val);
                    final_answer = Some(answer.clone());
                    last_inference_node_id = Some(curr_node_id.clone());

                    // 条件評価
                    let next_node =
                        evaluate_compiled_conditions(conditions, Some(&answer), &context);
                    let duration_ms = step_start.elapsed().as_secs_f64() * 1000.0;

                    steps.insert(
                        curr_node_id.clone(),
                        StepExecutionResult {
                            node_id: curr_node_id.clone(),
                            node_type: DagNodeType::Inference,
                            answer: Some(answer),
                            duration_ms,
                            next_node: next_node.clone(),
                            reason: reason.clone(),
                        },
                    );

                    match next_node {
                        Some(nxt) => curr_node_id = nxt,
                        None => break,
                    }
                }
                CompiledDagNode::Branch { conditions, reason } => {
                    let next_node =
                        evaluate_compiled_conditions(conditions, final_answer.as_ref(), &context);
                    let duration_ms = step_start.elapsed().as_secs_f64() * 1000.0;

                    steps.insert(
                        curr_node_id.clone(),
                        StepExecutionResult {
                            node_id: curr_node_id.clone(),
                            node_type: DagNodeType::Branch,
                            answer: None,
                            duration_ms,
                            next_node: next_node.clone(),
                            reason: reason.clone(),
                        },
                    );

                    match next_node {
                        Some(nxt) => curr_node_id = nxt,
                        None => break,
                    }
                }
                CompiledDagNode::Parallel {
                    parallel_nodes,
                    join_node,
                    reason,
                } => {
                    let mut tasks = Vec::with_capacity(parallel_nodes.len());

                    for p_node_id in parallel_nodes {
                        let p_node = dag
                            .nodes
                            .get(p_node_id)
                            .ok_or_else(|| DagError::NodeNotFound(p_node_id.clone()))?;

                        let raw_question = p_node.question().ok_or_else(|| {
                            DagError::ExecutionFailed(format!(
                                "並行ノード '{p_node_id}' に question が存在しません。"
                            ))
                        })?;

                        let handle =
                            self.spawn_inference_task(Arc::clone(&state), raw_question, &context);
                        let p_id = p_node_id.clone();
                        tasks.push(async move {
                            let ans =
                                Self::await_inference_task(handle, "並行タスク結合失敗").await?;
                            Ok::<_, DagError>((p_id, ans))
                        });
                    }

                    // 並行実行を待機
                    let results = futures_util::future::join_all(tasks).await;
                    let mut parallel_answers = Vec::with_capacity(results.len());
                    for res in results {
                        parallel_answers.push(res?);
                    }

                    let duration_ms = step_start.elapsed().as_secs_f64() * 1000.0;
                    let next_node = Some(join_node.clone());
                    record_parallel_results(
                        parallel_answers,
                        next_node.as_deref(),
                        duration_ms,
                        &mut context,
                        &mut steps,
                        &mut final_answer,
                    );

                    steps.insert(
                        curr_node_id.clone(),
                        StepExecutionResult {
                            node_id: curr_node_id.clone(),
                            node_type: DagNodeType::Parallel,
                            answer: None,
                            duration_ms,
                            next_node: next_node.clone(),
                            reason: reason.clone(),
                        },
                    );

                    match next_node {
                        Some(nxt) => curr_node_id = nxt,
                        None => break,
                    }
                }
                CompiledDagNode::Escalate { reason } => {
                    let duration_ms = step_start.elapsed().as_secs_f64() * 1000.0;
                    escalated = true;
                    escalation_reason = reason.clone();

                    // エスカレーションをトリガーした直前ノードの特定(Inference -> Branch -> Escalate 等の遷移でも直近の推論ノードを捕捉)
                    let trigger_node_id = last_inference_node_id
                        .clone()
                        .or_else(|| execution_path.iter().rev().nth(1).cloned());
                    escalation_node_id = trigger_node_id.clone();

                    // 最後の Answer から Gating メタデータを抽出
                    escalation_gating = final_answer.as_ref().and_then(|a| a.gating.clone());

                    // System 2 向けの Guided CoT プロンプト案の導出
                    if let Some(ref g) = escalation_gating
                        && let Some(ref esc_ctx) = g.escalation
                        && let Some(ref tmpl) = esc_ctx.prompt_template
                    {
                        escalation_prompt = Some(tmpl.clone());
                    } else if let Some(ref trig_id) = trigger_node_id
                        && let Some(trig_node) = dag.nodes.get(trig_id)
                        && let Some(q) = trig_node.question()
                        && let Some(ref ans) = final_answer
                        && let Some(ref g) = escalation_gating
                    {
                        use sokuto_core::gating::CandidateProbability;
                        let candidates: Vec<CandidateProbability> = ans
                            .probabilities()
                            .map(|p| {
                                let mut list: Vec<_> = p
                                    .iter()
                                    .map(|(k, &v)| CandidateProbability {
                                        candidate: k.clone(),
                                        probability: v,
                                    })
                                    .collect();
                                list.sort_by(|a, b| b.probability.total_cmp(&a.probability));
                                list
                            })
                            .unwrap_or_default();

                        use crate::engine::escalation::build_rich_escalation_prompt;
                        escalation_prompt = Some(build_rich_escalation_prompt(
                            Some(initial_state),
                            Some(trig_id.as_str()),
                            Some(q),
                            g,
                            &candidates,
                        ));
                    } else {
                        escalation_prompt = reason.clone();
                    }

                    steps.insert(
                        curr_node_id.clone(),
                        StepExecutionResult {
                            node_id: curr_node_id.clone(),
                            node_type: DagNodeType::Escalate,
                            answer: None,
                            duration_ms,
                            next_node: None,
                            reason: reason.clone(),
                        },
                    );
                    break;
                }
            }
        }

        Ok(DagExecutionResult {
            dag_id: dag.dag_id.clone(),
            execution_path,
            steps,
            total_duration_ms: 0.0,
            escalated,
            escalation_reason,
            escalation_node_id,
            escalation_prompt,
            escalation_gating,
            final_answer,
        })
    }

    /// 単一質問に対する推論タスクをバックグラウンドスレッドへオフロードして生成する。
    ///
    /// 質問内のプレースホルダーを変数補間した上で、
    /// `tokio::task::spawn_blocking` により推論エンジンを呼び出すタスクを生成する。
    /// 外部入力テキスト (`state`) は、内部判定値や確信度の漏洩を防ぐため補間対象から除外し、
    /// そのまま推論エンジンに渡す。
    ///
    /// # 引数
    /// - `state`: 初期入力テキスト。
    /// - `raw_question`: 未補間の質問仕様。
    /// - `context`: 現在のノード実行コンテキスト。
    fn spawn_inference_task(
        &self,
        state: Arc<str>,
        raw_question: &Question,
        context: &IndexMap<String, NodeContextValue>,
    ) -> tokio::task::JoinHandle<Result<Answer, RuntimeError>> {
        let interpolated_question = interpolate_question(raw_question, context);

        let engine = self.engine.clone();
        let tokenizer = self.tokenizer.clone();
        let calib_config = self.calib_config.clone();

        tokio::task::spawn_blocking(move || {
            engine.predict_single_question(
                &tokenizer,
                &state,
                &interpolated_question,
                &calib_config,
            )
        })
    }

    /// 単一推論タスクの終了を待機し、Answer または DagError を返却する。
    ///
    /// # 引数
    /// - `handle`: 推論タスクの JoinHandle。
    /// - `error_context`: スレッド結合失敗時のエラー文脈文字列。
    async fn await_inference_task(
        handle: tokio::task::JoinHandle<Result<Answer, RuntimeError>>,
        error_context: &str,
    ) -> Result<Answer, DagError> {
        handle
            .await
            .map_err(|e| DagError::ExecutionFailed(format!("{error_context}: {e}。")))?
            .map_err(DagError::Runtime)
    }
}

/// Answer をコンテキスト値に変換する内部ヘルパー。
fn answer_to_context_value(answer: &Answer) -> NodeContextValue {
    let conf = answer.effective_confidence();
    match &answer.payload {
        AnswerPayload::Choice { selected, .. } => NodeContextValue::new(selected.clone(), conf),
        AnswerPayload::Score { value, .. } => NodeContextValue::new(format!("{value}"), conf),
        AnswerPayload::Noul { probability } => {
            NodeContextValue::new(if *probability >= 0.5 { "true" } else { "false" }, conf)
        }
    }
}

/// 並行実行されたタスク群の判定結果を `context`・`steps` へ反映し、最後のタスク結果を `final_answer` に設定する。
///
/// `futures_util::future::join_all` は入力タスク列 (`parallel_nodes`) の順序を保持するため、
/// 最後のタスクの `Answer` を代表値として `final_answer` を更新し、後続の `Branch` ノードによる
/// `field: "answer"` / `"confidence"` / `"route"` 参照および最終 `DagExecutionResult.final_answer` との整合を保つ。
fn record_parallel_results(
    parallel_answers: Vec<(String, Answer)>,
    join_node: Option<&str>,
    duration_ms: f64,
    context: &mut IndexMap<String, NodeContextValue>,
    steps: &mut IndexMap<String, StepExecutionResult>,
    final_answer: &mut Option<Answer>,
) {
    let total = parallel_answers.len();
    let next_node = join_node.map(str::to_string);

    for (idx, (p_id, ans)) in parallel_answers.into_iter().enumerate() {
        let ctx_val = answer_to_context_value(&ans);
        context.insert(p_id.clone(), ctx_val);

        if idx + 1 == total {
            *final_answer = Some(ans.clone());
        }

        steps.insert(
            p_id.clone(),
            StepExecutionResult {
                node_id: p_id,
                node_type: DagNodeType::Inference,
                answer: Some(ans),
                duration_ms,
                next_node: next_node.clone(),
                reason: None,
            },
        );
    }
}

/// コンパイル済み条件リストを順次評価し、最初に適合した条件の `target_node` を返却する (Short-circuit)。
#[inline]
pub fn evaluate_compiled_conditions(
    conditions: &[CompiledCondition],
    answer: Option<&Answer>,
    context: &IndexMap<String, NodeContextValue>,
) -> Option<String> {
    for cond in conditions {
        if cond.matches(answer, context) {
            return Some(cond.target_node.clone());
        }
    }
    None
}

/// 条件リストを順次評価し、最初に適合した条件の `target_node` を返却する (Short-circuit)。
pub fn evaluate_conditions(
    conditions: &[NodeCondition],
    answer: Option<&Answer>,
    context: &IndexMap<String, NodeContextValue>,
) -> Option<String> {
    for cond in conditions {
        if let Some(compiled) = CompiledCondition::from_node_condition(cond)
            && compiled.matches(answer, context)
        {
            return Some(cond.target_node.clone());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::schema::{ConditionField, ContextProp, DagNode, NodeConditionOp};
    use sokuto_core::gating::DecisionRoute;

    #[test]
    fn test_evaluate_conditions_confidence() {
        let answer = Answer::choice("transfer", IndexMap::new(), 0.62);
        let context = IndexMap::new();

        let conditions = vec![
            NodeCondition {
                field: "confidence".to_string(),
                op: NodeConditionOp::ConfidenceLt,
                value: Some(serde_json::json!(0.65)),
                target_node: "escalate_low_confidence".to_string(),
            },
            NodeCondition {
                field: "answer".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!("transfer")),
                target_node: "transfer_flow".to_string(),
            },
        ];

        let target = evaluate_conditions(&conditions, Some(&answer), &context);
        assert_eq!(target, Some("escalate_low_confidence".to_string()));
    }

    #[test]
    fn test_evaluate_conditions_noul() {
        let answer = Answer::noul(0.02);
        let context = IndexMap::new();

        let conditions = vec![
            NodeCondition {
                field: "answer".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!(true)),
                target_node: "reject_flow".to_string(),
            },
            NodeCondition {
                field: "answer".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!(false)),
                target_node: "continue_flow".to_string(),
            },
        ];

        let target = evaluate_conditions(&conditions, Some(&answer), &context);
        assert_eq!(target, Some("continue_flow".to_string()));
    }

    #[test]
    fn test_evaluate_conditions_context_bool_and_number() {
        let mut context = IndexMap::new();
        context.insert(
            "fraud_check".to_string(),
            NodeContextValue::new("true", 0.99),
        );
        context.insert("risk_score".to_string(), NodeContextValue::new("3", 0.85));

        let conditions = vec![
            NodeCondition {
                field: "fraud_check.value".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!(true)),
                target_node: "escalate_fraud".to_string(),
            },
            NodeCondition {
                field: "risk_score.value".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!(3)),
                target_node: "high_risk_flow".to_string(),
            },
        ];

        let target = evaluate_conditions(&conditions, None, &context);
        assert_eq!(target, Some("escalate_fraud".to_string()));
    }

    #[test]
    fn test_matches_context_property_number_variants() {
        let mut context = IndexMap::new();
        context.insert("risk_score".to_string(), NodeContextValue::new("3", 0.85));
        context.insert("delta".to_string(), NodeContextValue::new("-12.5", 0.90));
        context.insert(
            "big_id".to_string(),
            NodeContextValue::new("18446744073709551615", 0.99),
        );
        context.insert("label".to_string(), NodeContextValue::new("transfer", 0.95));

        assert_eq!(
            evaluate_conditions(
                &[NodeCondition {
                    field: "risk_score.value".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!(3)),
                    target_node: "hit_int".to_string(),
                }],
                None,
                &context,
            ),
            Some("hit_int".to_string())
        );

        assert_eq!(
            evaluate_conditions(
                &[NodeCondition {
                    field: "delta.value".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!(-12.5)),
                    target_node: "hit_float".to_string(),
                }],
                None,
                &context,
            ),
            Some("hit_float".to_string())
        );

        assert_eq!(
            evaluate_conditions(
                &[NodeCondition {
                    field: "big_id.value".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!(u64::MAX)),
                    target_node: "hit_u64".to_string(),
                }],
                None,
                &context,
            ),
            Some("hit_u64".to_string())
        );

        // 文字列長が一部一致する場合 (prefix 一致のみ等) に誤って Eq にならないこと
        assert_eq!(
            evaluate_conditions(
                &[NodeCondition {
                    field: "risk_score.value".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!(30)),
                    target_node: "miss_prefix".to_string(),
                }],
                None,
                &context,
            ),
            None
        );

        // 非数値コンテキスト値に対する Neq が従来どおり true となること
        assert_eq!(
            evaluate_conditions(
                &[NodeCondition {
                    field: "label.value".to_string(),
                    op: NodeConditionOp::Neq,
                    value: Some(serde_json::json!(3)),
                    target_node: "not_three".to_string(),
                }],
                None,
                &context,
            ),
            Some("not_three".to_string())
        );
    }

    #[test]
    fn test_evaluate_conditions_score_and_numeric_string_values() {
        let context = IndexMap::new();
        let score_three = Answer::score(3.0, IndexMap::new(), 0.90);

        // 1. 文字列数値 ("3", " 3.0 ") が正しくパースされ、数値リテラル同様に判定されること
        assert_eq!(
            evaluate_conditions(
                &[NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!("3")),
                    target_node: "exact_three".to_string(),
                }],
                Some(&score_three),
                &context,
            ),
            Some("exact_three".to_string())
        );

        assert_eq!(
            evaluate_conditions(
                &[NodeCondition {
                    field: "value".to_string(),
                    op: NodeConditionOp::Gte,
                    value: Some(serde_json::json!(" 3.0 ")),
                    target_node: "gte_three".to_string(),
                }],
                Some(&score_three),
                &context,
            ),
            Some("gte_three".to_string())
        );

        // 2. "4.0" が 0.0 にフォールバックして score_val > 0.0 と誤判定されないこと
        assert_eq!(
            evaluate_conditions(
                &[NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Gt,
                    value: Some(serde_json::json!("4.0")),
                    target_node: "gt_four".to_string(),
                }],
                Some(&score_three),
                &context,
            ),
            None
        );

        // 3. 非数値文字列・NaN・None が 0.0 にフォールバックして誤マッチしないこと
        let score_zero = Answer::score(0.0, IndexMap::new(), 0.90);
        for invalid_val in [
            Some(serde_json::json!("not_a_number")),
            Some(serde_json::json!("NaN")),
            None,
        ] {
            assert_eq!(
                evaluate_conditions(
                    &[NodeCondition {
                        field: "answer".to_string(),
                        op: NodeConditionOp::Gt,
                        value: invalid_val.clone(),
                        target_node: "should_not_match".to_string(),
                    }],
                    Some(&score_three),
                    &context,
                ),
                None
            );
            assert_eq!(
                evaluate_conditions(
                    &[NodeCondition {
                        field: "answer".to_string(),
                        op: NodeConditionOp::Eq,
                        value: invalid_val,
                        target_node: "should_not_match_zero".to_string(),
                    }],
                    Some(&score_zero),
                    &context,
                ),
                None
            );
        }

        // 4. Noul 確率判定および confidence 判定でも文字列数値が解釈されること
        let noul_high = Answer::noul(0.82);
        assert_eq!(
            evaluate_conditions(
                &[NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Gte,
                    value: Some(serde_json::json!("0.80")),
                    target_node: "noul_pass".to_string(),
                }],
                Some(&noul_high),
                &context,
            ),
            Some("noul_pass".to_string())
        );
        assert_eq!(
            evaluate_conditions(
                &[NodeCondition {
                    field: "confidence".to_string(),
                    op: NodeConditionOp::ConfidenceGte,
                    value: Some(serde_json::json!("0.85")),
                    target_node: "conf_pass".to_string(),
                }],
                Some(&score_three),
                &context,
            ),
            Some("conf_pass".to_string())
        );
    }

    #[test]
    fn test_condition_field_parse_and_route_in_context_variants() {
        assert_eq!(
            ConditionField::parse("confidence"),
            ConditionField::Confidence
        );
        assert_eq!(ConditionField::parse("route"), ConditionField::Route);
        assert_eq!(ConditionField::parse("answer"), ConditionField::Answer);
        assert_eq!(ConditionField::parse("value"), ConditionField::Answer);
        assert_eq!(
            ConditionField::parse("node_a.value"),
            ConditionField::ContextProperty {
                node_id: "node_a",
                prop: ContextProp::Value,
            }
        );
        assert_eq!(
            ConditionField::parse("node_a.confidence"),
            ConditionField::ContextProperty {
                node_id: "node_a",
                prop: ContextProp::Confidence,
            }
        );
        assert_eq!(
            ConditionField::parse("node_a.unknown_prop"),
            ConditionField::Unknown
        );
        assert_eq!(
            ConditionField::parse("unsupported"),
            ConditionField::Unknown
        );

        // route / In / context.confidence / Unknown の各分岐評価
        let mut context = IndexMap::new();
        context.insert(
            "prev_node".to_string(),
            NodeContextValue::new("card_lost", 0.72),
        );

        let gating = sokuto_core::gating::GatingMetadata {
            route: DecisionRoute::ConfirmOrEscalate,
            confidence: 0.55,
            entropy: None,
            margin: None,
            energy: None,
            is_ood: false,
            reason: "test".to_string(),
            escalation: None,
        };
        let ans = Answer::choice("card_lost", IndexMap::new(), 0.55).with_gating(gating);

        let conditions = vec![
            NodeCondition {
                field: "unknown_field".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!("anything")),
                target_node: "never".to_string(),
            },
            NodeCondition {
                field: "prev_node.confidence".to_string(),
                op: NodeConditionOp::Gte,
                value: Some(serde_json::json!(0.80)),
                target_node: "high_prev_conf".to_string(),
            },
            NodeCondition {
                field: "route".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!("confirm_or_escalate")),
                target_node: "escalate_node".to_string(),
            },
            NodeCondition {
                field: "answer".to_string(),
                op: NodeConditionOp::In,
                value: Some(serde_json::json!(["card_lost", "card_stolen"])),
                target_node: "card_issue".to_string(),
            },
        ];

        assert_eq!(
            evaluate_conditions(&conditions, Some(&ans), &context),
            Some("escalate_node".to_string())
        );

        // route をスキップした後は Choice の In がマッチする
        assert_eq!(
            evaluate_conditions(&conditions[3..], Some(&ans), &context),
            Some("card_issue".to_string())
        );
    }

    #[test]
    fn test_matches_route_does_not_default_to_auto_execute_when_missing() {
        let context = IndexMap::new();
        let route_conditions = vec![
            NodeCondition {
                field: "route".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!("auto_execute")),
                target_node: "auto_node".to_string(),
            },
            NodeCondition {
                field: "route".to_string(),
                op: NodeConditionOp::Neq,
                value: Some(serde_json::json!("auto_execute")),
                target_node: "non_auto_node".to_string(),
            },
            NodeCondition {
                field: "route".to_string(),
                op: NodeConditionOp::Neq,
                value: Some(serde_json::json!("fallback")),
                target_node: "non_fallback_node".to_string(),
            },
        ];

        // 1. answer が None (未推論状態) の場合、Eq/Neq ともに "auto_execute" へフォールバックせず false となること。
        assert_eq!(evaluate_conditions(&route_conditions, None, &context), None);

        // 2. answer.gating が None (ゲーティング無効時) の場合も同様に false となること。
        let ungated_answer = Answer::choice("transfer", IndexMap::new(), 0.30);
        assert_eq!(
            evaluate_conditions(&route_conditions, Some(&ungated_answer), &context),
            None
        );

        // 3. gating が付与されている場合は通常どおり評価されること。
        let auto_gating = sokuto_core::gating::GatingMetadata {
            route: DecisionRoute::AutoExecute,
            confidence: 0.95,
            entropy: None,
            margin: None,
            energy: None,
            is_ood: false,
            reason: "high_conf".to_string(),
            escalation: None,
        };
        let auto_answer =
            Answer::choice("transfer", IndexMap::new(), 0.95).with_gating(auto_gating);
        assert_eq!(
            evaluate_conditions(&route_conditions, Some(&auto_answer), &context),
            Some("auto_node".to_string())
        );

        let fallback_gating = sokuto_core::gating::GatingMetadata {
            route: DecisionRoute::Fallback,
            confidence: 0.20,
            entropy: None,
            margin: None,
            energy: None,
            is_ood: true,
            reason: "ood".to_string(),
            escalation: None,
        };
        let fallback_answer =
            Answer::choice("transfer", IndexMap::new(), 0.20).with_gating(fallback_gating);
        assert_eq!(
            evaluate_conditions(&route_conditions, Some(&fallback_answer), &context),
            Some("non_auto_node".to_string())
        );

        // 4. 比較値が非文字列または None の場合は Neq でも誤マッチしないこと。
        let invalid_value_conds = vec![NodeCondition {
            field: "route".to_string(),
            op: NodeConditionOp::Neq,
            value: None,
            target_node: "should_not_match".to_string(),
        }];
        assert_eq!(
            evaluate_conditions(&invalid_value_conds, Some(&auto_answer), &context),
            None
        );
    }

    #[test]
    fn test_record_parallel_results_updates_final_answer_and_context() {
        let mut context = IndexMap::new();
        let mut steps = IndexMap::new();

        // Parallel 通過前の古い final_answer が残っているケースも模擬する
        let stale_answer = Answer::noul(0.05);
        let mut final_answer = Some(stale_answer);

        let task_a_ans = Answer::noul(0.92);
        let task_b_ans = Answer::choice("priority_high", IndexMap::new(), 0.88);

        record_parallel_results(
            vec![
                ("fraud_check".to_string(), task_a_ans),
                ("priority_eval".to_string(), task_b_ans.clone()),
            ],
            Some("branch_after_parallel"),
            1.25,
            &mut context,
            &mut steps,
            &mut final_answer,
        );

        // 1. final_answer が最後の並行タスク (priority_eval) の結果で上書きされていること
        assert_eq!(
            final_answer.as_ref().and_then(|a| a.as_choice()),
            Some("priority_high")
        );
        assert_eq!(final_answer.as_ref(), Some(&task_b_ans));

        // 2. 両タスクの結果が context と steps に格納されていること
        assert_eq!(
            context.get("fraud_check").map(|c| c.value.as_str()),
            Some("true")
        );
        assert_eq!(
            context.get("priority_eval").map(|c| c.value.as_str()),
            Some("priority_high")
        );
        assert_eq!(steps.len(), 2);
        assert_eq!(
            steps["fraud_check"].next_node.as_deref(),
            Some("branch_after_parallel")
        );
        assert_eq!(
            steps["priority_eval"].next_node.as_deref(),
            Some("branch_after_parallel")
        );

        // 3. 後続 Branch ノードでの field: "answer" / "fraud_check.value" 評価が正しく動作すること
        let branch_conditions = vec![
            NodeCondition {
                field: "answer".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!("priority_high")),
                target_node: "handle_priority_high".to_string(),
            },
            NodeCondition {
                field: "answer".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!(false)),
                target_node: "stale_branch_should_not_trigger".to_string(),
            },
        ];
        assert_eq!(
            evaluate_conditions(&branch_conditions, final_answer.as_ref(), &context),
            Some("handle_priority_high".to_string())
        );

        let context_branch_conditions = vec![NodeCondition {
            field: "fraud_check.value".to_string(),
            op: NodeConditionOp::Eq,
            value: Some(serde_json::json!(true)),
            target_node: "escalate_fraud".to_string(),
        }];
        assert_eq!(
            evaluate_conditions(&context_branch_conditions, final_answer.as_ref(), &context),
            Some("escalate_fraud".to_string())
        );

        // 4. 空結果時は既存の final_answer を維持すること
        let mut unchanged = Some(Answer::noul(0.80));
        record_parallel_results(
            Vec::new(),
            Some("join"),
            0.1,
            &mut context,
            &mut steps,
            &mut unchanged,
        );
        assert_eq!(unchanged.as_ref().and_then(|a| a.as_noul()), Some(0.80));
    }

    /// 指定ステップ数を持つ直列 DAG (Inference / Branch / Escalate) を生成するテスト用ヘルパー。
    fn build_linear_control_dag(num_steps: usize, terminate_with_escalate: bool) -> DagDefinition {
        use crate::dag::schema::DagNode;
        use sokuto_core::schema::Question;

        let mut nodes = IndexMap::new();
        for i in 0..num_steps {
            let is_last = i + 1 == num_steps;
            let node_id = format!("step_{i}");
            let node = if is_last && terminate_with_escalate {
                DagNode::Escalate {
                    reason: Some("テスト終端エスカレーション。".to_string()),
                }
            } else if is_last {
                DagNode::Branch {
                    conditions: Vec::new(),
                    reason: None,
                }
            } else if i == 0 {
                DagNode::Inference {
                    question: Question::new_noul("テストステップ判定".to_string()),
                    conditions: vec![NodeCondition {
                        field: "confidence".to_string(),
                        op: NodeConditionOp::ConfidenceGte,
                        value: Some(serde_json::json!(0.0)),
                        target_node: format!("step_{}", i + 1),
                    }],
                    reason: None,
                }
            } else {
                DagNode::Branch {
                    conditions: vec![NodeCondition {
                        field: "confidence".to_string(),
                        op: NodeConditionOp::ConfidenceGte,
                        value: Some(serde_json::json!(0.0)),
                        target_node: format!("step_{}", i + 1),
                    }],
                    reason: None,
                }
            };
            nodes.insert(node_id, node);
        }

        DagDefinition {
            dag_id: "max_steps_boundary_dag".to_string(),
            timeout_ms: 500,
            entry_node: "step_0".to_string(),
            nodes,
        }
    }

    #[tokio::test]
    async fn test_run_dag_max_steps_exact_boundary_succeeds() {
        let engine = Arc::new(InferenceEngine::empty_for_test());
        let tokenizer = Arc::new(JevTokenizer::empty_for_test());
        let calib = CalibrationConfig::default();

        // 1. max_steps = 1 の単一 Branch ノード DAG (next_node == None で正常終了)
        let executor_1 =
            InProcessDagExecutor::new(engine.clone(), tokenizer.clone(), calib.clone())
                .with_max_steps(1);
        let dag_single_branch = build_linear_control_dag(1, false);
        let res_branch = executor_1
            .execute("state", &dag_single_branch)
            .await
            .expect("max_steps = 1 で単一 Branch ノードの DAG は正常終了すること。");
        assert_eq!(res_branch.execution_path, vec!["step_0"]);
        assert!(!res_branch.escalated);

        // 2. max_steps = 1 の単一 Escalate ノード DAG (Escalate の break で正常終了)
        let dag_single_escalate = build_linear_control_dag(1, true);
        let res_escalate = executor_1
            .execute("state", &dag_single_escalate)
            .await
            .expect("max_steps = 1 で単一 Escalate ノードの DAG は正常終了すること。");
        assert_eq!(res_escalate.execution_path, vec!["step_0"]);
        assert!(res_escalate.escalated);

        // 3. max_steps = 2 でちょうど 2 ステップ目に終端する DAG も正常終了すること
        let executor_2 = InProcessDagExecutor::new(engine, tokenizer, calib).with_max_steps(2);
        let dag_two_steps = build_linear_control_dag(2, true);
        let res_two = executor_2
            .execute("state", &dag_two_steps)
            .await
            .expect("ちょうど max_steps ステップ目で正常終了する DAG は成功すること。");
        assert_eq!(res_two.execution_path, vec!["step_0", "step_1"]);
        assert!(res_two.escalated);
    }

    #[tokio::test]
    async fn test_run_dag_max_steps_exceeded_fails() {
        let engine = Arc::new(InferenceEngine::empty_for_test());
        let tokenizer = Arc::new(JevTokenizer::empty_for_test());
        let calib = CalibrationConfig::default();

        // 1. max_steps = 1 に対して 2 ステップ以上必要な DAG は上限超過エラーになること
        let executor_1 =
            InProcessDagExecutor::new(engine.clone(), tokenizer.clone(), calib.clone())
                .with_max_steps(1);
        let dag_two_steps = build_linear_control_dag(2, false);
        let err_1 = executor_1
            .execute("state", &dag_two_steps)
            .await
            .expect_err("max_steps = 1 を超えて 2 ステップ目へ遷移する場合はエラーになること。");
        assert!(matches!(err_1, DagError::StepLimitExceeded(1)));

        // 2. max_steps = 2 に対して 3 ステップ必要な DAG は上限超過エラーになること
        let executor_2 =
            InProcessDagExecutor::new(engine.clone(), tokenizer.clone(), calib.clone())
                .with_max_steps(2);
        let dag_three_steps = build_linear_control_dag(3, false);
        let err_2 = executor_2
            .execute("state", &dag_three_steps)
            .await
            .expect_err("max_steps = 2 を超えて 3 ステップ目へ遷移する場合はエラーになること。");
        assert!(matches!(err_2, DagError::StepLimitExceeded(2)));

        // 3. max_steps = 0 は 1 ステップ目に入る前に上限超過エラーになること
        let executor_0 = InProcessDagExecutor::new(engine, tokenizer, calib).with_max_steps(0);
        let dag_one_step = build_linear_control_dag(1, false);
        let err_0 = executor_0
            .execute("state", &dag_one_step)
            .await
            .expect_err("max_steps = 0 は 1 ステップも実行できずエラーになること。");
        assert!(matches!(err_0, DagError::StepLimitExceeded(0)));
    }

    #[test]
    fn test_matches_choice_and_context_rejects_missing_or_incompatible_values() {
        let mut context = IndexMap::new();
        context.insert("label".to_string(), NodeContextValue::new("transfer", 0.95));
        context.insert("empty_label".to_string(), NodeContextValue::new("", 0.95));

        let choice_transfer = Answer::choice("transfer", IndexMap::new(), 0.95);
        let choice_empty = Answer::choice("", IndexMap::new(), 0.95);

        let incompatible_choice_values = vec![
            None,
            Some(serde_json::Value::Null),
            Some(serde_json::json!(123)),
            Some(serde_json::json!(true)),
            Some(serde_json::json!(["transfer"])),
            Some(serde_json::json!({"key": "transfer"})),
        ];

        // 1. Choice の Eq / Neq は欠損値・非文字列に対して常に false となること
        for val in incompatible_choice_values {
            for op in [NodeConditionOp::Eq, NodeConditionOp::Neq] {
                let conds = vec![NodeCondition {
                    field: "answer".to_string(),
                    op,
                    value: val.clone(),
                    target_node: "should_not_match".to_string(),
                }];
                assert_eq!(
                    evaluate_conditions(&conds, Some(&choice_transfer), &context),
                    None,
                    "Choice(transfer) に対する非互換値 {val:?} ({op:?}) が誤マッチしてはならない。"
                );
                assert_eq!(
                    evaluate_conditions(&conds, Some(&choice_empty), &context),
                    None,
                    "Choice('') に対する非互換値 {val:?} ({op:?}) が誤マッチしてはならない。"
                );
            }
        }

        // 通常の文字列 Eq / Neq は期待どおり動作すること
        assert_eq!(
            evaluate_conditions(
                &[NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Neq,
                    value: Some(serde_json::json!("card")),
                    target_node: "not_card".to_string(),
                }],
                Some(&choice_transfer),
                &context,
            ),
            Some("not_card".to_string())
        );

        // 2. ContextProp::Value の Eq / Neq は None / Null / Array / Object に対して常に false となること
        let incompatible_context_values = vec![
            None,
            Some(serde_json::Value::Null),
            Some(serde_json::json!(["transfer"])),
            Some(serde_json::json!({"key": "transfer"})),
        ];
        for val in incompatible_context_values {
            for field in ["label.value", "empty_label.value"] {
                for op in [NodeConditionOp::Eq, NodeConditionOp::Neq] {
                    let conds = vec![NodeCondition {
                        field: field.to_string(),
                        op,
                        value: val.clone(),
                        target_node: "should_not_match".to_string(),
                    }];
                    assert_eq!(
                        evaluate_conditions(&conds, None, &context),
                        None,
                        "{field} に対する非互換値 {val:?} ({op:?}) が誤マッチしてはならない。"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_spawn_inference_task_keeps_initial_state_uninterpolated() {
        let engine = Arc::new(InferenceEngine::empty_for_test());
        let tokenizer = Arc::new(JevTokenizer::empty_for_test());
        let calib = CalibrationConfig::default();
        let executor = InProcessDagExecutor::new(engine, tokenizer, calib);

        let mut context = IndexMap::new();
        context.insert(
            "node_a".to_string(),
            NodeContextValue::new("secret_value", 0.99),
        );

        // initial_state 内にプレースホルダー文字列が含まれる場合でも、補間されずそのまま推論タスクへ渡されること
        let initial_state: Arc<str> =
            Arc::from("User input containing {node_a.value} and {node_a.confidence}");
        let raw_question = Question::new_noul("テスト判定".to_string());

        let handle =
            executor.spawn_inference_task(Arc::clone(&initial_state), &raw_question, &context);

        // empty_for_test でも空ロジットから Answer が生成されるため、
        // initial_state 内のプレースホルダーが展開されずに生テキストのまま維持されていることを検証する
        let answer = InProcessDagExecutor::await_inference_task(handle, "推論タスク結合失敗")
            .await
            .expect("推論タスクの実行待機に成功。");

        let prompt_template = answer
            .gating
            .as_ref()
            .and_then(|g| g.escalation.as_ref())
            .and_then(|e| e.prompt_template.as_deref())
            .expect("エスカレーションプロンプトテンプレートの取得に成功。");

        assert!(
            prompt_template.contains("{node_a.value}"),
            "initial_state 内のプレースホルダーが補間されず維持されている必要があります。"
        );
        assert!(
            prompt_template.contains("{node_a.confidence}"),
            "initial_state 内のプレースホルダーが補間されず維持されている必要があります。"
        );
        assert!(
            !prompt_template.contains("secret_value"),
            "context 内の値が initial_state 内へ漏洩・展開されてはなりません。"
        );
    }

    #[tokio::test]
    async fn test_escalate_via_branch_preserves_last_inference_node_id_and_prompt() {
        let engine = Arc::new(InferenceEngine::empty_for_test());
        let tokenizer = Arc::new(JevTokenizer::empty_for_test());
        let calib = CalibrationConfig::default();
        let executor = InProcessDagExecutor::new(engine, tokenizer, calib);

        let mut nodes = IndexMap::new();
        // 1. 推論ノード
        nodes.insert(
            "step_infer".to_string(),
            DagNode::inference(
                Question::new_noul("テスト質問".to_string()),
                vec![NodeCondition {
                    field: "confidence".to_string(),
                    op: NodeConditionOp::ConfidenceGte,
                    value: Some(serde_json::json!(0.0)),
                    target_node: "step_branch".to_string(),
                }],
            ),
        );

        // 2. Branch ノード
        nodes.insert(
            "step_branch".to_string(),
            DagNode::branch(vec![NodeCondition {
                field: "confidence".to_string(),
                op: NodeConditionOp::ConfidenceGte,
                value: Some(serde_json::json!(0.0)),
                target_node: "step_escalate".to_string(),
            }]),
        );

        // 3. Escalate ノード
        nodes.insert(
            "step_escalate".to_string(),
            DagNode::escalate(Some("ポリシー違反によるエスカレーション。")),
        );

        let dag = DagDefinition {
            dag_id: "test_branch_escalate_dag".to_string(),
            timeout_ms: 1000,
            entry_node: "step_infer".to_string(),
            nodes,
        };

        let result = executor
            .execute("テスト入力", &dag)
            .await
            .expect("DAG 実行に成功。");

        assert!(result.escalated);
        assert_eq!(
            result.execution_path,
            vec!["step_infer", "step_branch", "step_escalate"]
        );
        assert_eq!(
            result.escalation_node_id.as_deref(),
            Some("step_infer"),
            "直前ノードが Branch であっても、最後に実行された Inference ノードの ID が記録される必要があります。"
        );
        assert!(
            result.escalation_prompt.is_some(),
            "System 2 向け Guided CoT プロンプト案が生成される必要があります。"
        );
    }
}
