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
                    let mut parallel_answers = Vec::with_capacity(results.len());
                    for res in results {
                        let (p_id, ans_res) = res.map_err(|e| {
                            DagError::ExecutionFailed(format!("並行タスク結合失敗: {e}。"))
                        })?;
                        let ans = ans_res.map_err(DagError::Runtime)?;
                        parallel_answers.push((p_id, ans));
                    }

                    let duration_ms = step_start.elapsed().as_secs_f64() * 1000.0;
                    let next_node = node.join_node.clone();
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

/// コンテキストノード参照 (`node_id.prop`) で参照可能なプロパティ種別。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContextProp {
    /// 先行ノードの回答値 (`value` / `answer`)。
    Value,
    /// 先行ノードの確信度 (`confidence`)。
    Confidence,
}

/// `NodeCondition.field` 文字列をパースした型安全な評価対象フィールド。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConditionField<'a> {
    /// 現在のノードの実効確信度 (`confidence`)。
    Confidence,
    /// 現在のノードのゲーティングルート (`route`)。
    Route,
    /// 現在のノードの判定結果値 (`answer` / `value`)。
    Answer,
    /// 先行ノードの出力コンテキスト参照 (`<node_id>.<prop>`)。
    ContextProperty { node_id: &'a str, prop: ContextProp },
    /// 未定義または構文不正なフィールド名。
    Unknown,
}

impl<'a> ConditionField<'a> {
    /// フィールド名文字列を `ConditionField` にパースする。
    fn parse(field: &'a str) -> Self {
        match field {
            "confidence" => Self::Confidence,
            "route" => Self::Route,
            "answer" | "value" => Self::Answer,
            _ => match field.split_once('.') {
                Some((node_id, "value" | "answer")) => Self::ContextProperty {
                    node_id,
                    prop: ContextProp::Value,
                },
                Some((node_id, "confidence")) => Self::ContextProperty {
                    node_id,
                    prop: ContextProp::Confidence,
                },
                _ => Self::Unknown,
            },
        }
    }
}

/// 単一条件の適合判定を行う。
fn matches_condition(
    cond: &NodeCondition,
    answer: Option<&Answer>,
    context: &IndexMap<String, NodeContextValue>,
) -> bool {
    match ConditionField::parse(&cond.field) {
        ConditionField::Confidence => matches_confidence(cond, answer),
        ConditionField::Route => matches_route(cond, answer),
        ConditionField::Answer => answer.is_some_and(|ans| matches_answer_value(cond, ans)),
        ConditionField::ContextProperty { node_id, prop } => context
            .get(node_id)
            .is_some_and(|ctx_val| matches_context_property(cond, ctx_val, prop)),
        ConditionField::Unknown => false,
    }
}

/// `confidence` フィールドに対する閾値比較を評価する。
fn matches_confidence(cond: &NodeCondition, answer: Option<&Answer>) -> bool {
    let conf = answer.map(Answer::effective_confidence).unwrap_or(0.0);
    let Some(target_val) = parse_condition_f64(cond.value.as_ref()) else {
        return false;
    };
    let normalized_op = match cond.op {
        NodeConditionOp::ConfidenceGte => NodeConditionOp::Gte,
        NodeConditionOp::ConfidenceLt => NodeConditionOp::Lt,
        other => other,
    };
    compare_f64_with_op(conf, normalized_op, target_val)
}

/// `route` (ゲーティングルート) フィールドの一致比較を評価する。
///
/// `answer` または `answer.gating` が `None` の場合 (未推論状態またはゲーティング無効時)、
/// および比較対象値が文字列でない場合は `false` を返す。
fn matches_route(cond: &NodeCondition, answer: Option<&Answer>) -> bool {
    let Some(gating) = answer.and_then(|a| a.gating.as_ref()) else {
        return false;
    };
    let Some(target_str) = cond.value.as_ref().and_then(|v| v.as_str()) else {
        return false;
    };
    let route_str = match gating.route {
        DecisionRoute::AutoExecute => "auto_execute",
        DecisionRoute::ConfirmOrEscalate => "confirm_or_escalate",
        DecisionRoute::Fallback => "fallback",
    };

    match cond.op {
        NodeConditionOp::Eq => route_str == target_str,
        NodeConditionOp::Neq => route_str != target_str,
        _ => false,
    }
}

/// `answer` / `value` フィールドに対する型別 (Noul / Choice / Score) 条件評価を行う。
fn matches_answer_value(cond: &NodeCondition, ans: &Answer) -> bool {
    if let Some(noul_val) = ans.as_noul() {
        return matches_noul_value(cond, noul_val);
    }
    if let Some(choice_val) = ans.as_choice() {
        return matches_choice_value(cond, choice_val);
    }
    if let Some(score_val) = ans.as_score() {
        return parse_condition_f64(cond.value.as_ref())
            .is_some_and(|target_num| compare_f64_with_op(score_val, cond.op, target_num));
    }
    false
}

/// Noul 判定の真偽確率に対する真偽値または数値閾値比較を評価する。
fn matches_noul_value(cond: &NodeCondition, noul_val: f64) -> bool {
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

    parse_condition_f64(cond.value.as_ref())
        .is_some_and(|target_num| compare_f64_with_op(noul_val, cond.op, target_num))
}

/// Choice 判定の選択キーに対する文字列比較および集合包含 (`In`) 比較を評価する。
fn matches_choice_value(cond: &NodeCondition, choice_val: &str) -> bool {
    match cond.op {
        NodeConditionOp::Eq => {
            let target_str = cond.value.as_ref().and_then(|v| v.as_str()).unwrap_or("");
            choice_val == target_str
        }
        NodeConditionOp::Neq => {
            let target_str = cond.value.as_ref().and_then(|v| v.as_str()).unwrap_or("");
            choice_val != target_str
        }
        NodeConditionOp::In => cond
            .value
            .as_ref()
            .and_then(|v| v.as_array())
            .is_some_and(|arr| {
                arr.iter()
                    .any(|item| item.as_str().is_some_and(|s| s == choice_val))
            }),
        _ => false,
    }
}

/// 先行ノードのコンテキストプロパティ (`value` / `confidence`) を評価する。
fn matches_context_property(
    cond: &NodeCondition,
    ctx_val: &NodeContextValue,
    prop: ContextProp,
) -> bool {
    match prop {
        ContextProp::Value => {
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
            match cond.op {
                NodeConditionOp::Eq => ctx_val.value == target_str,
                NodeConditionOp::Neq => ctx_val.value != target_str,
                _ => false,
            }
        }
        ContextProp::Confidence => {
            let Some(target_num) = parse_condition_f64(cond.value.as_ref()) else {
                return false;
            };
            match cond.op {
                NodeConditionOp::ConfidenceGte | NodeConditionOp::Gte => {
                    ctx_val.confidence >= target_num - 1e-6
                }
                NodeConditionOp::ConfidenceLt | NodeConditionOp::Lt => {
                    ctx_val.confidence < target_num - 1e-6
                }
                _ => false,
            }
        }
    }
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
}
