//! # インプロセス DAG 実行器モジュール
//!
//! 非同期グラフ巡回、ゼロコピー文脈注入、および CPU 集約的推論の
//! `tokio::task::spawn_blocking` オフロードによる超低遅延オーケストレーションを提供する。

use std::sync::Arc;
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use sokuto_core::contract::calibration::CalibrationConfig;
use sokuto_core::gating::DecisionRoute;
use sokuto_core::schema::{Answer, AnswerPayload};

use crate::dag::interpolate::{NodeContextValue, interpolate_prompt, interpolate_question};
use crate::dag::pool::PoolError;
use crate::dag::schema::{
    DagDefinition, DagExecutionResult, DagNodeType, NodeCondition, NodeConditionOp,
    StepExecutionResult,
};
use crate::dag::validator::{DagValidationError, validate_dag};
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
        // 1. 静的検証 (Fast-Fail)
        validate_dag(dag)?;

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
        dag: &DagDefinition,
    ) -> Result<DagExecutionResult, DagError> {
        let mut context: IndexMap<String, NodeContextValue> = IndexMap::new();
        let mut steps: IndexMap<String, StepExecutionResult> = IndexMap::new();
        let mut execution_path: Vec<String> = Vec::new();

        // スクラッチバッファをループ外で事前確保し、各ステップで再利用する
        let mut scratch_state = String::with_capacity(initial_state.len() + 64);

        let mut curr_node_id = dag.entry_node.clone();
        let mut step_count = 0;
        let mut final_answer: Option<Answer> = None;
        let mut escalated = false;
        let mut escalation_reason = None;
        let mut escalation_node_id = None;
        let mut escalation_prompt = None;
        let mut escalation_gating = None;

        while step_count < self.max_steps {
            step_count += 1;
            let node = dag
                .nodes
                .get(&curr_node_id)
                .ok_or_else(|| DagError::NodeNotFound(curr_node_id.clone()))?;

            execution_path.push(curr_node_id.clone());
            let step_start = Instant::now();

            match node.node_type {
                DagNodeType::Inference => {
                    let raw_question = node
                        .question
                        .as_ref()
                        .ok_or_else(|| DagError::NodeNotFound(curr_node_id.clone()))?;

                    // ゼロアロケーション動的コンテキスト補間 (事前確保バッファの再利用)
                    interpolate_prompt(initial_state, &context, &mut scratch_state);
                    let interpolated_question = interpolate_question(raw_question, &context);

                    // 推論処理を専用ブロッキングスレッドへオフロード
                    let engine = self.engine.clone();
                    let tokenizer = self.tokenizer.clone();
                    let calib_config = self.calib_config.clone();
                    let current_state = scratch_state.clone();

                    let answer = tokio::task::spawn_blocking(move || {
                        engine.predict_single_question(
                            &tokenizer,
                            &current_state,
                            &interpolated_question,
                            &calib_config,
                        )
                    })
                    .await
                    .map_err(|e| DagError::ExecutionFailed(format!("スレッド結合失敗: {e}。")))?
                    .map_err(DagError::Runtime)?;

                    // コンテキストに結果を反映
                    let ctx_val = answer_to_context_value(&answer);
                    context.insert(curr_node_id.clone(), ctx_val);
                    final_answer = Some(answer.clone());

                    // 条件評価
                    let next_node = evaluate_conditions(&node.conditions, Some(&answer), &context);
                    let duration_ms = step_start.elapsed().as_secs_f64() * 1000.0;

                    steps.insert(
                        curr_node_id.clone(),
                        StepExecutionResult {
                            node_id: curr_node_id.clone(),
                            node_type: DagNodeType::Inference,
                            answer: Some(answer),
                            duration_ms,
                            next_node: next_node.clone(),
                            reason: node.reason.clone(),
                        },
                    );

                    match next_node {
                        Some(nxt) => curr_node_id = nxt,
                        None => break,
                    }
                }
                DagNodeType::Branch => {
                    let next_node =
                        evaluate_conditions(&node.conditions, final_answer.as_ref(), &context);
                    let duration_ms = step_start.elapsed().as_secs_f64() * 1000.0;

                    steps.insert(
                        curr_node_id.clone(),
                        StepExecutionResult {
                            node_id: curr_node_id.clone(),
                            node_type: DagNodeType::Branch,
                            answer: None,
                            duration_ms,
                            next_node: next_node.clone(),
                            reason: node.reason.clone(),
                        },
                    );

                    match next_node {
                        Some(nxt) => curr_node_id = nxt,
                        None => break,
                    }
                }
                DagNodeType::Parallel => {
                    let mut tasks = Vec::with_capacity(node.parallel_nodes.len());

                    for p_node_id in &node.parallel_nodes {
                        let p_node = dag
                            .nodes
                            .get(p_node_id)
                            .ok_or_else(|| DagError::NodeNotFound(p_node_id.clone()))?;

                        let raw_question = p_node.question.as_ref().ok_or_else(|| {
                            DagError::ExecutionFailed(format!(
                                "並行ノード '{p_node_id}' に question が存在しません。"
                            ))
                        })?;

                        // 事前確保バッファを活用してプロンプトを展開
                        interpolate_prompt(initial_state, &context, &mut scratch_state);
                        let interpolated_question = interpolate_question(raw_question, &context);

                        let engine = self.engine.clone();
                        let tokenizer = self.tokenizer.clone();
                        let calib_config = self.calib_config.clone();
                        let node_id_owned = p_node_id.clone();
                        let state_owned = scratch_state.clone();

                        let handle = tokio::task::spawn_blocking(move || {
                            let ans = engine.predict_single_question(
                                &tokenizer,
                                &state_owned,
                                &interpolated_question,
                                &calib_config,
                            );
                            (node_id_owned, ans)
                        });
                        tasks.push(handle);
                    }

                    // 並行実行を待機
                    let results = futures_util::future::join_all(tasks).await;

                    for res in results {
                        let (p_id, ans_res) = res.map_err(|e| {
                            DagError::ExecutionFailed(format!("並行タスク結合失敗: {e}。"))
                        })?;
                        let ans = ans_res.map_err(DagError::Runtime)?;
                        let ctx_val = answer_to_context_value(&ans);
                        context.insert(p_id.clone(), ctx_val);

                        steps.insert(
                            p_id.clone(),
                            StepExecutionResult {
                                node_id: p_id,
                                node_type: DagNodeType::Inference,
                                answer: Some(ans),
                                duration_ms: step_start.elapsed().as_secs_f64() * 1000.0,
                                next_node: node.join_node.clone(),
                                reason: None,
                            },
                        );
                    }

                    let duration_ms = step_start.elapsed().as_secs_f64() * 1000.0;
                    let next_node = node.join_node.clone();

                    steps.insert(
                        curr_node_id.clone(),
                        StepExecutionResult {
                            node_id: curr_node_id.clone(),
                            node_type: DagNodeType::Parallel,
                            answer: None,
                            duration_ms,
                            next_node: next_node.clone(),
                            reason: node.reason.clone(),
                        },
                    );

                    match next_node {
                        Some(nxt) => curr_node_id = nxt,
                        None => break,
                    }
                }
                DagNodeType::Escalate => {
                    let duration_ms = step_start.elapsed().as_secs_f64() * 1000.0;
                    escalated = true;
                    escalation_reason = node.reason.clone();

                    // エスカレーションをトリガーした直前ノードの特定
                    let trigger_node_id = execution_path.iter().rev().nth(1).cloned();
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
                        && let Some(ref q) = trig_node.question
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
                                list.sort_by(|a, b| {
                                    b.probability
                                        .partial_cmp(&a.probability)
                                        .unwrap_or(std::cmp::Ordering::Equal)
                                });
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
                        escalation_prompt = node.reason.clone();
                    }

                    steps.insert(
                        curr_node_id.clone(),
                        StepExecutionResult {
                            node_id: curr_node_id.clone(),
                            node_type: DagNodeType::Escalate,
                            answer: None,
                            duration_ms,
                            next_node: None,
                            reason: node.reason.clone(),
                        },
                    );
                    break;
                }
            }
        }

        if step_count >= self.max_steps {
            return Err(DagError::StepLimitExceeded(self.max_steps));
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

/// 条件リストを順次評価し、最初に適合した条件の `target_node` を返却する (Short-circuit)。
pub fn evaluate_conditions(
    conditions: &[NodeCondition],
    answer: Option<&Answer>,
    context: &IndexMap<String, NodeContextValue>,
) -> Option<String> {
    for cond in conditions {
        if matches_condition(cond, answer, context) {
            return Some(cond.target_node.clone());
        }
    }
    None
}

/// 条件値 (`serde_json::Value`) を有限な `f64` として解釈する。
///
/// JSON 数値リテラルに加えて、`"3"` や `"3.0"` のような数値文字列もパースして返却する。
fn parse_condition_f64(value: Option<&serde_json::Value>) -> Option<f64> {
    let val = value?;
    val.as_f64()
        .or_else(|| val.as_str().and_then(|s| s.trim().parse::<f64>().ok()))
        .filter(|n| n.is_finite())
}

/// 2 つの有限浮動小数点数を許容誤差 `1e-6` 付きで比較する。
fn compare_f64_with_op(lhs: f64, op: NodeConditionOp, rhs: f64) -> bool {
    match op {
        NodeConditionOp::Eq => (lhs - rhs).abs() < 1e-6,
        NodeConditionOp::Neq => (lhs - rhs).abs() >= 1e-6,
        NodeConditionOp::Gt => lhs > rhs + 1e-6,
        NodeConditionOp::Gte => lhs >= rhs - 1e-6,
        NodeConditionOp::Lt => lhs < rhs - 1e-6,
        NodeConditionOp::Lte => lhs <= rhs + 1e-6,
        _ => false,
    }
}

/// 単一条件の適合判定を行う。
fn matches_condition(
    cond: &NodeCondition,
    answer: Option<&Answer>,
    context: &IndexMap<String, NodeContextValue>,
) -> bool {
    let field = cond.field.as_str();

    // 1. confidence フィールドの比較
    if field == "confidence" {
        let conf = answer.map(Answer::effective_confidence).unwrap_or(0.0);
        let Some(target_val) = parse_condition_f64(cond.value.as_ref()) else {
            return false;
        };
        let normalized_op = match cond.op {
            NodeConditionOp::ConfidenceGte => NodeConditionOp::Gte,
            NodeConditionOp::ConfidenceLt => NodeConditionOp::Lt,
            other => other,
        };
        return compare_f64_with_op(conf, normalized_op, target_val);
    }

    // 2. route (ゲーティングルート) フィールドの比較
    if field == "route" {
        let route_str = answer
            .and_then(|a| a.gating.as_ref())
            .map(|g| match g.route {
                DecisionRoute::AutoExecute => "auto_execute",
                DecisionRoute::ConfirmOrEscalate => "confirm_or_escalate",
                DecisionRoute::Fallback => "fallback",
            })
            .unwrap_or("auto_execute");

        let target_str = cond.value.as_ref().and_then(|v| v.as_str()).unwrap_or("");
        return match cond.op {
            NodeConditionOp::Eq => route_str == target_str,
            NodeConditionOp::Neq => route_str != target_str,
            _ => false,
        };
    }

    // 3. answer / value フィールドの比較
    if (field == "answer" || field == "value")
        && let Some(ans) = answer
    {
        // Noul (言明真実確率値 f64: 0.0〜1.0) 型判定
        if let Some(noul_val) = ans.as_noul() {
            // bool リテラルまたは文字列による判定 (true: >= 0.5, false: < 0.5)
            let target_bool = match &cond.value {
                Some(serde_json::Value::Bool(b)) => Some(*b),
                Some(serde_json::Value::String(s)) => match s.as_str() {
                    "true" => Some(true),
                    "false" => Some(false),
                    _ => None,
                },
                _ => None,
            };
            if let Some(tb) = target_bool {
                let is_true = noul_val >= 0.5;
                return match cond.op {
                    NodeConditionOp::Eq => is_true == tb,
                    NodeConditionOp::Neq => is_true != tb,
                    _ => false,
                };
            }

            // 数値による直接判定 (確率値そのものの閾値比較)
            if let Some(target_num) = parse_condition_f64(cond.value.as_ref()) {
                return compare_f64_with_op(noul_val, cond.op, target_num);
            }
        }

        // Choice (文字列) 型判定
        if let Some(choice_val) = ans.as_choice() {
            match cond.op {
                NodeConditionOp::Eq => {
                    let target_str = cond.value.as_ref().and_then(|v| v.as_str()).unwrap_or("");
                    return choice_val == target_str;
                }
                NodeConditionOp::Neq => {
                    let target_str = cond.value.as_ref().and_then(|v| v.as_str()).unwrap_or("");
                    return choice_val != target_str;
                }
                NodeConditionOp::In => {
                    if let Some(serde_json::Value::Array(arr)) = &cond.value {
                        return arr
                            .iter()
                            .any(|item| item.as_str().is_some_and(|s| s == choice_val));
                    }
                }
                _ => {}
            }
        }

        // Score (数値) 型判定
        if let Some(score_val) = ans.as_score() {
            return parse_condition_f64(cond.value.as_ref())
                .is_some_and(|target_num| compare_f64_with_op(score_val, cond.op, target_num));
        }
    }

    // 4. コンテキスト内の先行ノード ID 直接参照 (例: `node_id.value`)
    if let Some((node_id, prop)) = field.split_once('.')
        && let Some(ctx_val) = context.get(node_id)
    {
        match prop {
            "value" | "answer" => {
                let num_str;
                let target_str = match &cond.value {
                    Some(serde_json::Value::String(s)) => s.as_str(),
                    Some(serde_json::Value::Bool(true)) => "true",
                    Some(serde_json::Value::Bool(false)) => "false",
                    Some(serde_json::Value::Number(n)) => {
                        num_str = n.to_string();
                        num_str.as_str()
                    }
                    _ => "",
                };
                return match cond.op {
                    NodeConditionOp::Eq => ctx_val.value == target_str,
                    NodeConditionOp::Neq => ctx_val.value != target_str,
                    _ => false,
                };
            }
            "confidence" => {
                let Some(target_num) = parse_condition_f64(cond.value.as_ref()) else {
                    return false;
                };
                return match cond.op {
                    NodeConditionOp::ConfidenceGte | NodeConditionOp::Gte => {
                        ctx_val.confidence >= target_num - 1e-6
                    }
                    NodeConditionOp::ConfidenceLt | NodeConditionOp::Lt => {
                        ctx_val.confidence < target_num - 1e-6
                    }
                    _ => false,
                };
            }
            _ => {}
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
