//! # DAG 静的検証モジュール
//!
//! 実行前の DAG 定義に対し、循環参照 (閉路)、未定義ノード参照、
//! 型契約違反、到達不能ノードを Fast-Fail で静的検証する。

use std::collections::{HashMap, HashSet, VecDeque};

use indexmap::IndexMap;
use sokuto_core::schema::{Criteria, Question, QuestionType};

use crate::dag::interpolate::extract_placeholders;
use crate::dag::schema::{
    CompiledAnswerPredicate, CompiledCondition, CompiledDag, CompiledDagNode, ConditionField,
    ConditionPredicate, ContextProp, DagDefinition, DagNode, DagNodeType, NodeCondition,
    NodeConditionOp,
};

/// DAG 静的検証で発生するエラー型。
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DagValidationError {
    /// ノードマップが空。
    #[error("DAG 内にノードが 1 つも定義されていません。")]
    EmptyNodes,

    /// エントリノードが未定義。
    #[error("エントリノード '{0}' が nodes マップ内に存在しません。")]
    EntryNodeNotFound(String),

    /// 参照先ノードが未定義。
    #[error("ノード '{from_node}' から未定義のターゲットノード '{target_node}' を参照しています。")]
    TargetNodeNotFound {
        /// 参照元ノード ID。
        from_node: String,
        /// 未定義の参照先ノード ID。
        target_node: String,
    },

    /// 有向グラフに循環 (サイクル) が検出された。
    #[error("DAG 内に循環参照 (閉路) が検出されました: {0}。")]
    CycleDetected(String),

    /// ノード種別に対する必須プロパティの欠落。
    #[error("ノード '{node_id}' ({node_type:?}) の定義が不正です: {reason}。")]
    InvalidNodeSpec {
        /// 不正なノード ID。
        node_id: String,
        /// ノード種別。
        node_type: DagNodeType,
        /// 理由説明文。
        reason: String,
    },

    /// エントリノードから到達不能な孤立ノードが存在する。
    #[error("エントリノードから到達不能な孤立ノードが検出されました: {0:?}。")]
    UnreachableNodes(Vec<String>),

    /// 条件式のフィールド名・参照先・演算子と値の型が不正。
    #[error("ノード '{node_id}' の条件式 (field='{field}') が不正です: {reason}。")]
    InvalidCondition {
        /// 条件式を持つノード ID。
        node_id: String,
        /// 条件式フィールド名。
        field: String,
        /// 理由説明文。
        reason: String,
    },

    /// プロンプト補間変数 `{node_id.prop}` の参照先またはプロパティ名が不正。
    #[error("ノード '{node_id}' のプロンプト補間変数 '{{{placeholder}}}' が不正です: {reason}。")]
    InvalidInterpolation {
        /// 補間変数を含むノード ID。
        node_id: String,
        /// 不正なプレースホルダー文字列。
        placeholder: String,
        /// 理由説明文。
        reason: String,
    },
}

/// DAG 定義の整合性と無閉路性を静的に検証する。
///
/// # 検査項目
/// DAG の静的検証を行い、整合性を確認する。
pub fn validate_dag(dag: &DagDefinition) -> Result<(), DagValidationError> {
    compile_dag(dag).map(|_| ())
}

/// DAG を静的検証し、実行用プラン `CompiledDag` へ事前コンパイルする。
///
/// 以下の整合性検証を実行する:
/// 1. `nodes` が 1 つ以上のノードを含むこと。
/// 2. `entry_node` が `nodes` に実在すること。
/// 3. 各ノードのプロパティ検証 (推論ノードの質問、並行ノードの合流先など)。
/// 4. すべての遷移先 (`target_node`, `parallel_nodes`, `join_node`) が実在すること。
/// 5. DFS による有向グラフの閉路 (サイクル) 検出。
/// 6. `entry_node` から全ノードへの到達性確認。
/// 7. 各条件式のフィールド・上流参照・型整合性検証および型付き条件への事前コンパイル。
/// 8. 全 Inference ノードのプロンプト補間変数 (`{node_id.prop}`) の参照先実在性・プロパティ名・上流先行性検証。
pub fn compile_dag(dag: &DagDefinition) -> Result<CompiledDag, DagValidationError> {
    if dag.nodes.is_empty() {
        return Err(DagValidationError::EmptyNodes);
    }

    if !dag.nodes.contains_key(&dag.entry_node) {
        return Err(DagValidationError::EntryNodeNotFound(
            dag.entry_node.clone(),
        ));
    }

    // 1. ノード種別整合性および参照先実在チェック
    for (node_id, node) in &dag.nodes {
        match node {
            DagNode::Inference { .. } => {
                // question は型レベルで必須定義される。
            }
            DagNode::Branch { .. } => {
                // conditions が空の場合は終端ノードとして扱われ、次ノードなしで正常終了する。
            }
            DagNode::Parallel {
                parallel_nodes,
                join_node,
                ..
            } => {
                if parallel_nodes.is_empty() {
                    return Err(DagValidationError::InvalidNodeSpec {
                        node_id: node_id.clone(),
                        node_type: node.node_type(),
                        reason: "Parallel ノードには 1 つ以上の parallel_nodes が必要です。"
                            .to_string(),
                    });
                }
                if !dag.nodes.contains_key(join_node) {
                    return Err(DagValidationError::TargetNodeNotFound {
                        from_node: node_id.clone(),
                        target_node: join_node.clone(),
                    });
                }
                for p_node_id in parallel_nodes {
                    let p_node = dag.nodes.get(p_node_id).ok_or_else(|| {
                        DagValidationError::TargetNodeNotFound {
                            from_node: node_id.clone(),
                            target_node: p_node_id.clone(),
                        }
                    })?;
                    if p_node.node_type() != DagNodeType::Inference {
                        return Err(DagValidationError::InvalidNodeSpec {
                            node_id: p_node_id.clone(),
                            node_type: p_node.node_type(),
                            reason: "Parallel ノード内のタスクノードは Inference 型である必要があります。".to_string(),
                        });
                    }
                    if !p_node.conditions().is_empty() {
                        return Err(DagValidationError::InvalidNodeSpec {
                            node_id: p_node_id.clone(),
                            node_type: p_node.node_type(),
                            reason: "Parallel ノード内のタスクノードは独自の conditions を持てません(親ノードの join_node へ合流するため)。".to_string(),
                        });
                    }
                }
            }
            DagNode::Escalate { .. } => {
                // 終端ノードであり、遷移条件は型レベルで持たない。
            }
        }

        for cond in node.conditions() {
            if !dag.nodes.contains_key(&cond.target_node) {
                return Err(DagValidationError::TargetNodeNotFound {
                    from_node: node_id.clone(),
                    target_node: cond.target_node.clone(),
                });
            }
        }
    }

    // 2. 有向隣接リストの構築
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for (id, node) in &dag.nodes {
        let mut neighbors = Vec::new();
        match node {
            DagNode::Inference { conditions, .. } | DagNode::Branch { conditions, .. } => {
                for cond in conditions {
                    neighbors.push(cond.target_node.as_str());
                }
            }
            DagNode::Parallel {
                parallel_nodes,
                join_node,
                ..
            } => {
                for p in parallel_nodes {
                    neighbors.push(p.as_str());
                }
                for p in parallel_nodes {
                    // 並行ノード群から合流ノードへのエッジも考慮
                    adj.entry(p.as_str()).or_default().push(join_node.as_str());
                }
            }
            DagNode::Escalate { .. } => {}
        }
        adj.entry(id.as_str()).or_default().extend(neighbors);
    }

    // 3. DFS によるサイクル検出 (0: 未訪問, 1: 訪問中, 2: 完了)
    let mut state: HashMap<&str, u8> = HashMap::new();
    let mut path_stack = Vec::new();

    fn dfs<'a>(
        u: &'a str,
        adj: &HashMap<&str, Vec<&'a str>>,
        state: &mut HashMap<&'a str, u8>,
        path_stack: &mut Vec<&'a str>,
    ) -> Result<(), DagValidationError> {
        state.insert(u, 1);
        path_stack.push(u);

        if let Some(neighbors) = adj.get(u) {
            for &v in neighbors {
                let v_state = state.get(v).copied().unwrap_or(0);
                if v_state == 1 {
                    // 訪問中ノードへの遷移 = サイクル検出
                    let cycle_start = path_stack.iter().position(|&x| x == v).unwrap_or(0);
                    let mut cycle_path = path_stack[cycle_start..].to_vec();
                    cycle_path.push(v);
                    return Err(DagValidationError::CycleDetected(cycle_path.join(" -> ")));
                } else if v_state == 0 {
                    dfs(v, adj, state, path_stack)?;
                }
            }
        }

        path_stack.pop();
        state.insert(u, 2);
        Ok(())
    }

    for node_id in dag.nodes.keys() {
        let node_id_str = node_id.as_str();
        if state.get(node_id_str).copied().unwrap_or(0) == 0 {
            dfs(node_id_str, &adj, &mut state, &mut path_stack)?;
        }
    }

    // 4. 到達性チェック (entry_node から到達可能か)
    let mut reachable = HashSet::new();
    let mut queue = vec![dag.entry_node.as_str()];
    reachable.insert(dag.entry_node.as_str());

    while let Some(curr) = queue.pop() {
        if let Some(neighbors) = adj.get(curr) {
            for &nxt in neighbors {
                if reachable.insert(nxt) {
                    queue.push(nxt);
                }
            }
        }
    }

    let unreachable: Vec<String> = dag
        .nodes
        .keys()
        .filter(|k| !reachable.contains(k.as_str()))
        .cloned()
        .collect();

    if !unreachable.is_empty() {
        return Err(DagValidationError::UnreachableNodes(unreachable));
    }

    // 5. 逆隣接リストおよび先祖関係 (上流先行ノード集合) マップの構築
    let mut rev_adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for (&u, neighbors) in &adj {
        for &v in neighbors {
            rev_adj.entry(v).or_default().push(u);
        }
    }

    let mut ancestors: HashMap<&str, HashSet<&str>> = HashMap::new();
    for node_id in dag.nodes.keys().map(String::as_str) {
        let mut anc_set = HashSet::new();
        let mut q = vec![node_id];
        while let Some(curr) = q.pop() {
            if let Some(preds) = rev_adj.get(curr) {
                for &pred in preds {
                    if anc_set.insert(pred) {
                        q.push(pred);
                    }
                }
            }
        }
        ancestors.insert(node_id, anc_set);
    }

    // 6. 条件式の静的検証と型解決コンパイル (フィールド妥当性・先祖参照・型整合性)
    let mut compiled_conditions = validate_node_conditions(dag, &ancestors, &rev_adj)?;

    // 7. プロンプト補間変数の静的検証 (参照先実在性・プロパティ名・上流先行性)
    validate_prompt_interpolations(dag, &ancestors)?;

    // 8. CompiledDagNode へのマッピング
    let mut compiled_nodes = IndexMap::with_capacity(dag.nodes.len());
    for (node_id, node) in &dag.nodes {
        let conds = compiled_conditions.swap_remove(node_id).unwrap_or_default();
        let compiled_node = match node {
            DagNode::Inference {
                question, reason, ..
            } => CompiledDagNode::Inference {
                question: question.clone(),
                conditions: conds,
                reason: reason.clone(),
            },
            DagNode::Branch { reason, .. } => CompiledDagNode::Branch {
                conditions: conds,
                reason: reason.clone(),
            },
            DagNode::Parallel {
                parallel_nodes,
                join_node,
                reason,
            } => CompiledDagNode::Parallel {
                parallel_nodes: parallel_nodes.clone(),
                join_node: join_node.clone(),
                reason: reason.clone(),
            },
            DagNode::Escalate { reason } => CompiledDagNode::Escalate {
                reason: reason.clone(),
            },
        };
        compiled_nodes.insert(node_id.clone(), compiled_node);
    }

    Ok(CompiledDag {
        dag_id: dag.dag_id.clone(),
        timeout_ms: dag.timeout_ms,
        entry_node: dag.entry_node.clone(),
        nodes: compiled_nodes,
    })
}

/// 条件式を評価する対象の上流推論ノード群を探索・解決する。
///
/// `Inference` ノードの場合は自身の `question` を返し、
/// `Branch` ノードの場合は直近の上流 `Inference` ノード群の質問仕様を幅優先探索で解決する。
///
/// # 検証ルール
/// 1. 直近の上流 `Inference` ノードが存在しない場合はエラー。
/// 2. 上流に同一の並行ノード (`Parallel`) に由来する複数タスクが存在する場合、暗黙のフィールド参照は曖昧となるため、
///    `node_id.value` (または `node_id.confidence`) 形式での明示指定を要求するエラーを返す。
fn resolve_upstream_inference_nodes<'a>(
    node_id: &'a str,
    node: &'a DagNode,
    field: &str,
    dag: &'a DagDefinition,
    rev_adj: &HashMap<&str, Vec<&str>>,
) -> Result<IndexMap<&'a str, &'a Question>, DagValidationError> {
    if let Some(q) = node.question() {
        let mut map = IndexMap::new();
        map.insert(node_id, q);
        return Ok(map);
    }

    let mut visited = HashSet::new();
    let mut q = VecDeque::new();
    if let Some(preds) = rev_adj.get(node_id) {
        for &pred in preds {
            if visited.insert(pred) {
                q.push_back(pred);
            }
        }
    }

    let mut upstream_nodes: IndexMap<&'a str, &'a Question> = IndexMap::new();
    while let Some(curr) = q.pop_front() {
        if let Some((curr_key, curr_node)) = dag.nodes.get_key_value(curr) {
            if let Some(question) = curr_node.question() {
                upstream_nodes.insert(curr_key.as_str(), question);
                // 直近の Inference ノードに到達したため、この経路の更なる上流探索は打ち切る。
            } else if let Some(preds) = rev_adj.get(curr) {
                for &pred in preds {
                    if visited.insert(pred) {
                        q.push_back(pred);
                    }
                }
            }
        }
    }

    if upstream_nodes.is_empty() {
        return Err(DagValidationError::InvalidCondition {
            node_id: node_id.to_string(),
            field: field.to_string(),
            reason: format!(
                "上流に推論結果を出力する Inference ノードが存在しないため {field} フィールドを評価できません。"
            ),
        });
    }

    // 並行ノード (Parallel) 由来の複数タスクが同時に合流しているかチェック。
    for p_node in dag.nodes.values() {
        if let DagNode::Parallel { parallel_nodes, .. } = p_node
            && parallel_nodes.len() > 1
        {
            let matched_count = parallel_nodes
                .iter()
                .filter(|id| upstream_nodes.contains_key(id.as_str()))
                .count();
            if matched_count > 1 {
                let example_field = if field == "confidence" {
                    "node_id.confidence"
                } else {
                    "node_id.value"
                };
                return Err(DagValidationError::InvalidCondition {
                    node_id: node_id.to_string(),
                    field: field.to_string(),
                    reason: format!(
                        "並行ノードからの合流ノードでは複数の推論結果が存在するため、暗黙の '{field}' フィールドは使用できません。'{example_field}' 形式で評価対象ノードを明示的に指定してください。"
                    ),
                });
            }
        }
    }

    Ok(upstream_nodes)
}

/// 条件式を評価する対象の上流推論ノード群の質問仕様を探索・解決し、質問タイプの一致を検証する。
///
/// # 検証ルール
/// 1. `resolve_upstream_inference_nodes` による上流存在性・並行曖昧性検証。
/// 2. 合流する複数の直近上流 `Inference` ノード間で質問タイプ (`question_type`) が不一致の場合はエラー。
fn resolve_upstream_questions<'a>(
    node_id: &'a str,
    node: &'a DagNode,
    field: &str,
    dag: &'a DagDefinition,
    rev_adj: &HashMap<&str, Vec<&str>>,
) -> Result<IndexMap<&'a str, &'a Question>, DagValidationError> {
    let upstream_nodes = resolve_upstream_inference_nodes(node_id, node, field, dag, rev_adj)?;

    // 合流する全上流 Inference ノード間で質問タイプ (question_type) が一致しているか検証。
    let first_type = upstream_nodes.values().next().unwrap().question_type;
    let all_same_type = upstream_nodes
        .values()
        .all(|q| q.question_type == first_type);
    if !all_same_type {
        let type_summary = upstream_nodes
            .iter()
            .map(|(id, q)| format!("'{id}'={:?}", q.question_type))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(DagValidationError::InvalidCondition {
            node_id: node_id.to_string(),
            field: field.to_string(),
            reason: format!(
                "合流する上流推論ノード間で質問タイプ (question_type) が一致していません ({type_summary})。'node_id.value' 形式で明示的に参照するか、合流する全ノードの質問タイプを統一してください。"
            ),
        });
    }

    Ok(upstream_nodes)
}

/// 各ノードの条件式リスト (`conditions`) に対し、フィールド妥当性・先祖参照・型整合性を静的検証し、型付き条件へ事前コンパイルする。
fn validate_node_conditions(
    dag: &DagDefinition,
    ancestors: &HashMap<&str, HashSet<&str>>,
    rev_adj: &HashMap<&str, Vec<&str>>,
) -> Result<IndexMap<String, Vec<CompiledCondition>>, DagValidationError> {
    let mut compiled_map = IndexMap::with_capacity(dag.nodes.len());
    for (node_id, node) in &dag.nodes {
        let mut compiled_list = Vec::with_capacity(node.conditions().len());
        for cond in node.conditions() {
            let compiled = validate_single_condition(node_id, node, cond, dag, ancestors, rev_adj)?;
            compiled_list.push(compiled);
        }
        compiled_map.insert(node_id.clone(), compiled_list);
    }

    Ok(compiled_map)
}

/// 全 Inference ノードの Instructions および Criteria におけるプロンプト補間変数 (`{node_id.prop}`) を静的検証する。
///
/// 外部入力である `initial_state` は静的定義に含まれず、意図しない変数展開や内部情報漏洩を防止するため
/// 実行時にも補間対象から除外される。そのため本検証の対象はノード定義内の `question.instructions` および `question.criteria` のみである。
fn validate_prompt_interpolations(
    dag: &DagDefinition,
    ancestors: &HashMap<&str, HashSet<&str>>,
) -> Result<(), DagValidationError> {
    for (node_id, node) in &dag.nodes {
        let Some(question) = node.question() else {
            continue;
        };

        // instructions 内の補間変数検証
        validate_template_placeholders(node_id, &question.instructions, dag, ancestors)?;

        // criteria 内の補間変数検証
        match &question.criteria {
            Some(Criteria::Map(map)) => {
                for desc in map.values() {
                    validate_template_placeholders(node_id, desc, dag, ancestors)?;
                }
            }
            Some(Criteria::List(list)) => {
                for item in list {
                    validate_template_placeholders(node_id, item, dag, ancestors)?;
                }
            }
            None => {}
        }
    }
    Ok(())
}

fn validate_template_placeholders(
    node_id: &str,
    template: &str,
    dag: &DagDefinition,
    ancestors: &HashMap<&str, HashSet<&str>>,
) -> Result<(), DagValidationError> {
    let placeholders = extract_placeholders(template);
    for ph in placeholders {
        let placeholder = ph.as_str();

        if let Some(prop) = ph.prop
            && !matches!(prop, "value" | "answer" | "confidence")
        {
            return Err(DagValidationError::InvalidInterpolation {
                node_id: node_id.to_string(),
                placeholder,
                reason: format!(
                    "不正なプロパティ名 '{prop}' です。'value', 'answer', 'confidence' のいずれかを指定してください。"
                ),
            });
        }

        let Some(ref_node) = dag.nodes.get(ph.node_id) else {
            return Err(DagValidationError::InvalidInterpolation {
                node_id: node_id.to_string(),
                placeholder,
                reason: format!(
                    "参照先ノード '{}' が nodes マップ内に存在しません。",
                    ph.node_id
                ),
            });
        };

        if ref_node.node_type() != DagNodeType::Inference {
            return Err(DagValidationError::InvalidInterpolation {
                node_id: node_id.to_string(),
                placeholder,
                reason: format!(
                    "参照先ノード '{}' ({:?}) は推論結果を出力する Inference ノードである必要があります。",
                    ph.node_id,
                    ref_node.node_type()
                ),
            });
        }

        if ph.node_id == node_id {
            return Err(DagValidationError::InvalidInterpolation {
                node_id: node_id.to_string(),
                placeholder,
                reason: format!("ノード '{node_id}' から自身を自己参照補間することはできません。"),
            });
        }

        if !ancestors
            .get(node_id)
            .is_some_and(|anc| anc.contains(ph.node_id))
        {
            return Err(DagValidationError::InvalidInterpolation {
                node_id: node_id.to_string(),
                placeholder,
                reason: format!(
                    "参照先ノード '{}' は現在のノード '{node_id}' の上流 (先行実行パス) に存在する必要があります。",
                    ph.node_id
                ),
            });
        }
    }
    Ok(())
}

fn validate_single_condition(
    node_id: &str,
    node: &DagNode,
    cond: &NodeCondition,
    dag: &DagDefinition,
    ancestors: &HashMap<&str, HashSet<&str>>,
    rev_adj: &HashMap<&str, Vec<&str>>,
) -> Result<CompiledCondition, DagValidationError> {
    let field = ConditionField::parse(&cond.field);

    let predicate = match field {
        ConditionField::Unknown => {
            return Err(DagValidationError::InvalidCondition {
                node_id: node_id.to_string(),
                field: cond.field.clone(),
                reason: "未定義または構文不正なフィールド名です。".to_string(),
            });
        }
        ConditionField::ContextProperty {
            node_id: ref_id,
            prop,
        } => {
            let Some(ref_node) = dag.nodes.get(ref_id) else {
                return Err(DagValidationError::InvalidCondition {
                    node_id: node_id.to_string(),
                    field: cond.field.clone(),
                    reason: format!("参照先ノード '{ref_id}' が nodes マップ内に存在しません。"),
                });
            };
            if ref_node.node_type() != DagNodeType::Inference {
                return Err(DagValidationError::InvalidCondition {
                    node_id: node_id.to_string(),
                    field: cond.field.clone(),
                    reason: format!(
                        "参照先ノード '{ref_id}' ({:?}) は推論結果を出力する Inference ノードである必要があります。",
                        ref_node.node_type()
                    ),
                });
            }
            if !ancestors
                .get(node_id)
                .is_some_and(|anc| anc.contains(ref_id))
            {
                return Err(DagValidationError::InvalidCondition {
                    node_id: node_id.to_string(),
                    field: cond.field.clone(),
                    reason: format!(
                        "参照先ノード '{ref_id}' は現在のノード '{node_id}' の上流 (先行実行パス) に存在する必要があります。"
                    ),
                });
            }
            match prop {
                ContextProp::Confidence => {
                    if cond.op == NodeConditionOp::In {
                        return Err(DagValidationError::InvalidCondition {
                            node_id: node_id.to_string(),
                            field: cond.field.clone(),
                            reason: "確信度プロパティに対して 'in' 演算子は使用できません。"
                                .to_string(),
                        });
                    }
                    let Some(threshold) = cond.value_as_f64() else {
                        return Err(DagValidationError::InvalidCondition {
                            node_id: node_id.to_string(),
                            field: cond.field.clone(),
                            reason: "確信度プロパティには有限数値 (または数値文字列) を指定する必要があります。".to_string(),
                        });
                    };
                    let op = match cond.op {
                        NodeConditionOp::ConfidenceGte | NodeConditionOp::Gte => {
                            NodeConditionOp::Gte
                        }
                        NodeConditionOp::ConfidenceLt | NodeConditionOp::Lt => NodeConditionOp::Lt,
                        other => other,
                    };
                    ConditionPredicate::ContextConfidence {
                        node_id: ref_id.to_string(),
                        op,
                        threshold,
                    }
                }
                ContextProp::Value => {
                    if !matches!(cond.op, NodeConditionOp::Eq | NodeConditionOp::Neq) {
                        return Err(DagValidationError::InvalidCondition {
                            node_id: node_id.to_string(),
                            field: cond.field.clone(),
                            reason: "コンテキスト値プロパティに対しては等価比較 (eq / neq) のみ使用できます。".to_string(),
                        });
                    }
                    let is_scalar = match cond.value.as_ref() {
                        Some(serde_json::Value::String(_)) | Some(serde_json::Value::Bool(_)) => {
                            true
                        }
                        Some(serde_json::Value::Number(n)) => {
                            n.as_f64().is_some_and(|f| f.is_finite())
                        }
                        _ => false,
                    };
                    if !is_scalar {
                        return Err(DagValidationError::InvalidCondition {
                            node_id: node_id.to_string(),
                            field: cond.field.clone(),
                            reason: "コンテキスト値プロパティにはスカラー値 (文字列、真偽値、有限数値) を指定する必要があります。".to_string(),
                        });
                    }
                    let expected = match cond.value.as_ref().unwrap() {
                        serde_json::Value::String(s) => s.clone(),
                        serde_json::Value::Bool(b) => b.to_string(),
                        serde_json::Value::Number(n) => n.to_string(),
                        _ => unreachable!(),
                    };
                    ConditionPredicate::ContextValue {
                        node_id: ref_id.to_string(),
                        op: cond.op,
                        expected,
                    }
                }
            }
        }
        ConditionField::Confidence => {
            resolve_upstream_inference_nodes(node_id, node, &cond.field, dag, rev_adj)?;
            if cond.op == NodeConditionOp::In {
                return Err(DagValidationError::InvalidCondition {
                    node_id: node_id.to_string(),
                    field: cond.field.clone(),
                    reason: "confidence フィールドに対して 'in' 演算子は使用できません。"
                        .to_string(),
                });
            }
            let Some(threshold) = cond.value_as_f64() else {
                return Err(DagValidationError::InvalidCondition {
                    node_id: node_id.to_string(),
                    field: cond.field.clone(),
                    reason: "confidence フィールドには有限数値 (または数値文字列) を指定する必要があります。".to_string(),
                });
            };
            let op = match cond.op {
                NodeConditionOp::ConfidenceGte | NodeConditionOp::Gte => NodeConditionOp::Gte,
                NodeConditionOp::ConfidenceLt | NodeConditionOp::Lt => NodeConditionOp::Lt,
                other => other,
            };
            ConditionPredicate::Confidence { op, threshold }
        }
        ConditionField::Route => {
            resolve_upstream_inference_nodes(node_id, node, &cond.field, dag, rev_adj)?;
            if !matches!(cond.op, NodeConditionOp::Eq | NodeConditionOp::Neq) {
                return Err(DagValidationError::InvalidCondition {
                    node_id: node_id.to_string(),
                    field: cond.field.clone(),
                    reason: "route フィールドに対しては等価比較 (eq / neq) のみ使用できます。"
                        .to_string(),
                });
            }
            let is_valid_route = match cond.value.as_ref().and_then(|v| v.as_str()) {
                Some(s) => matches!(s, "auto_execute" | "confirm_or_escalate" | "fallback"),
                None => false,
            };
            if !is_valid_route {
                return Err(DagValidationError::InvalidCondition {
                    node_id: node_id.to_string(),
                    field: cond.field.clone(),
                    reason: "route フィールドには 'auto_execute', 'confirm_or_escalate', 'fallback' のいずれかの文字列を指定する必要があります。".to_string(),
                });
            }
            let expected = cond.value.as_ref().unwrap().as_str().unwrap().to_string();
            ConditionPredicate::Route {
                op: cond.op,
                expected,
            }
        }
        ConditionField::Answer => {
            let upstream_nodes =
                resolve_upstream_questions(node_id, node, &cond.field, dag, rev_adj)?;
            let q_type = upstream_nodes.values().next().map(|q| q.question_type);

            match q_type {
                Some(QuestionType::Choice) => {
                    if matches!(
                        cond.op,
                        NodeConditionOp::Gt
                            | NodeConditionOp::Gte
                            | NodeConditionOp::Lt
                            | NodeConditionOp::Lte
                            | NodeConditionOp::ConfidenceGte
                            | NodeConditionOp::ConfidenceLt
                    ) {
                        return Err(DagValidationError::InvalidCondition {
                            node_id: node_id.to_string(),
                            field: cond.field.clone(),
                            reason: "大小比較演算子は Choice 型判定に対して無効です。".to_string(),
                        });
                    }
                    match cond.op {
                        NodeConditionOp::Eq | NodeConditionOp::Neq => {
                            let Some(s) = cond.value.as_ref().and_then(|v| v.as_str()) else {
                                return Err(DagValidationError::InvalidCondition {
                                    node_id: node_id.to_string(),
                                    field: cond.field.clone(),
                                    reason:
                                        "Choice 型の等価比較には文字列を指定する必要があります。"
                                            .to_string(),
                                });
                            };
                            for (up_id, q) in &upstream_nodes {
                                if let Some(Criteria::Map(ref map)) = q.criteria
                                    && !map.contains_key(s)
                                {
                                    return Err(DagValidationError::InvalidCondition {
                                        node_id: node_id.to_string(),
                                        field: cond.field.clone(),
                                        reason: format!(
                                            "Choice 質問 '{up_id}' の criteria 定義に候補キー '{s}' が存在しません。"
                                        ),
                                    });
                                }
                            }
                        }
                        NodeConditionOp::In => {
                            let Some(arr) = cond.value.as_ref().and_then(|v| v.as_array()) else {
                                return Err(DagValidationError::InvalidCondition {
                                    node_id: node_id.to_string(),
                                    field: cond.field.clone(),
                                    reason:
                                        "Choice 型の 'in' 比較には配列を指定する必要があります。"
                                            .to_string(),
                                });
                            };
                            if arr.is_empty() {
                                return Err(DagValidationError::InvalidCondition {
                                    node_id: node_id.to_string(),
                                    field: cond.field.clone(),
                                    reason: "'in' 演算子には 1 つ以上の要素を含む配列を指定する必要があります。".to_string(),
                                });
                            }
                            for item in arr {
                                let Some(s) = item.as_str() else {
                                    return Err(DagValidationError::InvalidCondition {
                                        node_id: node_id.to_string(),
                                        field: cond.field.clone(),
                                        reason: "'in' 演算子の配列要素はすべて文字列である必要があります。".to_string(),
                                    });
                                };
                                for (up_id, q) in &upstream_nodes {
                                    if let Some(Criteria::Map(ref map)) = q.criteria
                                        && !map.contains_key(s)
                                    {
                                        return Err(DagValidationError::InvalidCondition {
                                            node_id: node_id.to_string(),
                                            field: cond.field.clone(),
                                            reason: format!(
                                                "Choice 質問 '{up_id}' の criteria 定義に候補キー '{s}' が存在しません。"
                                            ),
                                        });
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Some(QuestionType::Score) => {
                    if matches!(
                        cond.op,
                        NodeConditionOp::In
                            | NodeConditionOp::ConfidenceGte
                            | NodeConditionOp::ConfidenceLt
                    ) {
                        return Err(DagValidationError::InvalidCondition {
                            node_id: node_id.to_string(),
                            field: cond.field.clone(),
                            reason: "Score 型判定に対して 'in' 演算子または確信度専用演算子は使用できません。".to_string(),
                        });
                    }
                    if cond.value_as_f64().is_none() {
                        return Err(DagValidationError::InvalidCondition {
                            node_id: node_id.to_string(),
                            field: cond.field.clone(),
                            reason: "Score 型判定には有限数値 (または数値文字列) を指定する必要があります。".to_string(),
                        });
                    }
                }
                Some(QuestionType::Noul) => match cond.op {
                    NodeConditionOp::Eq | NodeConditionOp::Neq => {
                        let is_valid = match cond.value.as_ref() {
                            Some(serde_json::Value::Bool(_)) => true,
                            Some(serde_json::Value::String(s)) => {
                                matches!(s.as_str(), "true" | "false")
                                    || s.trim().parse::<f64>().is_ok_and(|f| f.is_finite())
                            }
                            Some(serde_json::Value::Number(n)) => {
                                n.as_f64().is_some_and(|f| f.is_finite())
                            }
                            _ => false,
                        };
                        if !is_valid {
                            return Err(DagValidationError::InvalidCondition {
                                    node_id: node_id.to_string(),
                                    field: cond.field.clone(),
                                    reason: "Noul 型の等価比較には真偽値、'true'/'false' 文字列、または有限数値を指定する必要があります。".to_string(),
                                });
                        }
                    }
                    NodeConditionOp::Gt
                    | NodeConditionOp::Gte
                    | NodeConditionOp::Lt
                    | NodeConditionOp::Lte => {
                        if cond.value_as_f64().is_none() {
                            return Err(DagValidationError::InvalidCondition {
                                node_id: node_id.to_string(),
                                field: cond.field.clone(),
                                reason: "Noul 型の大小比較には有限数値を指定する必要があります。"
                                    .to_string(),
                            });
                        }
                    }
                    _ => {
                        return Err(DagValidationError::InvalidCondition {
                            node_id: node_id.to_string(),
                            field: cond.field.clone(),
                            reason: "Noul 型判定に対して不正な演算子です。".to_string(),
                        });
                    }
                },
                None => {}
            }

            let ans_pred = if cond.op == NodeConditionOp::In {
                let arr = cond.value.as_ref().unwrap().as_array().unwrap();
                let candidates = arr
                    .iter()
                    .map(|item| item.as_str().unwrap().to_string())
                    .collect();
                CompiledAnswerPredicate::In { candidates }
            } else if let Some(b) = cond.value.as_ref().and_then(|v| v.as_bool()) {
                CompiledAnswerPredicate::Bool {
                    op: cond.op,
                    expected: b,
                }
            } else if let Some(n) = cond.value.as_ref().and_then(|v| v.as_f64()) {
                CompiledAnswerPredicate::Numeric {
                    op: cond.op,
                    threshold: n,
                }
            } else if let Some(s) = cond.value.as_ref().and_then(|v| v.as_str()) {
                let as_bool = match s {
                    "true" => Some(true),
                    "false" => Some(false),
                    _ => None,
                };
                let as_f64 = s.trim().parse::<f64>().ok().filter(|f| f.is_finite());
                CompiledAnswerPredicate::String {
                    op: cond.op,
                    expected: s.to_string(),
                    as_f64,
                    as_bool,
                }
            } else {
                unreachable!()
            };

            ConditionPredicate::Answer(ans_pred)
        }
    };

    Ok(CompiledCondition {
        predicate,
        target_node: cond.target_node.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::schema::{DagNode, NodeCondition, NodeConditionOp};
    use indexmap::IndexMap;
    use sokuto_core::schema::Question;

    #[test]
    fn test_valid_dag() {
        let mut nodes = IndexMap::new();
        nodes.insert(
            "node1".to_string(),
            DagNode::inference(
                Question::new_noul("テスト？".to_string()),
                vec![NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::Value::Bool(true)),
                    target_node: "node2".to_string(),
                }],
            ),
        );
        nodes.insert("node2".to_string(), DagNode::escalate(Some("終了")));

        let dag = DagDefinition {
            dag_id: "valid_dag".to_string(),
            timeout_ms: 100,
            entry_node: "node1".to_string(),
            nodes,
        };

        assert!(validate_dag(&dag).is_ok());
    }

    #[test]
    fn test_cycle_detection() {
        let mut nodes = IndexMap::new();
        nodes.insert(
            "node1".to_string(),
            DagNode::inference(
                Question::new_noul("Q1".to_string()),
                vec![NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::Value::Bool(true)),
                    target_node: "node2".to_string(),
                }],
            ),
        );
        nodes.insert(
            "node2".to_string(),
            DagNode::inference(
                Question::new_noul("Q2".to_string()),
                vec![NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::Value::Bool(true)),
                    target_node: "node1".to_string(), // 閉路
                }],
            ),
        );

        let dag = DagDefinition {
            dag_id: "cyclic_dag".to_string(),
            timeout_ms: 100,
            entry_node: "node1".to_string(),
            nodes,
        };

        let res = validate_dag(&dag);
        assert!(matches!(res, Err(DagValidationError::CycleDetected(_))));
    }

    #[test]
    fn test_target_not_found() {
        let mut nodes = IndexMap::new();
        nodes.insert(
            "node1".to_string(),
            DagNode::inference(
                Question::new_noul("Q1".to_string()),
                vec![NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::Value::Bool(true)),
                    target_node: "non_existent".to_string(),
                }],
            ),
        );

        let dag = DagDefinition {
            dag_id: "broken_dag".to_string(),
            timeout_ms: 100,
            entry_node: "node1".to_string(),
            nodes,
        };

        let res = validate_dag(&dag);
        assert!(matches!(
            res,
            Err(DagValidationError::TargetNodeNotFound { .. })
        ));
    }

    #[test]
    fn test_unreachable_nodes_rejected() {
        let mut nodes = IndexMap::new();
        nodes.insert("entry".to_string(), DagNode::branch(Vec::new()));
        nodes.insert("orphan".to_string(), DagNode::escalate(Some("孤立ノード")));

        let dag = DagDefinition {
            dag_id: "unreachable_dag".to_string(),
            timeout_ms: 100,
            entry_node: "entry".to_string(),
            nodes,
        };

        let res = validate_dag(&dag);
        assert_eq!(
            res,
            Err(DagValidationError::UnreachableNodes(vec![
                "orphan".to_string()
            ]))
        );
    }

    #[test]
    fn test_parallel_dag_reachability_valid() {
        let mut nodes = IndexMap::new();
        nodes.insert(
            "parallel_entry".to_string(),
            DagNode::parallel(vec!["p1".to_string(), "p2".to_string()], "join"),
        );
        for id in ["p1", "p2"] {
            nodes.insert(
                id.to_string(),
                DagNode::inference(Question::new_noul(format!("質問 {id}")), Vec::new()),
            );
        }
        nodes.insert("join".to_string(), DagNode::branch(Vec::new()));

        let dag = DagDefinition {
            dag_id: "parallel_valid_dag".to_string(),
            timeout_ms: 100,
            entry_node: "parallel_entry".to_string(),
            nodes,
        };

        assert!(validate_dag(&dag).is_ok());
    }

    #[test]
    fn test_validate_dag_rejects_unknown_condition_field() {
        let mut nodes = IndexMap::new();
        nodes.insert(
            "node1".to_string(),
            DagNode::inference(
                Question::new_noul("テスト？".to_string()),
                vec![NodeCondition {
                    field: "unknown_field".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!(true)),
                    target_node: "node2".to_string(),
                }],
            ),
        );
        nodes.insert(
            "node2".to_string(),
            DagNode::escalate(Option::<String>::None),
        );

        let dag = DagDefinition {
            dag_id: "unknown_field_dag".to_string(),
            timeout_ms: 100,
            entry_node: "node1".to_string(),
            nodes,
        };

        let err = validate_dag(&dag).expect_err("未知フィールドは拒否されること。");
        assert!(matches!(
            err,
            DagValidationError::InvalidCondition {
                ref node_id,
                ref field,
                ..
            } if node_id == "node1" && field == "unknown_field"
        ));
    }

    #[test]
    fn test_validate_dag_rejects_undefined_or_non_upstream_context_property() {
        let mut nodes = IndexMap::new();
        // node1 -> node2 -> node3
        // node1 が node2 (下流) の出力を参照しようとするケース
        nodes.insert(
            "node1".to_string(),
            DagNode::inference(
                Question::new_noul("Q1".to_string()),
                vec![NodeCondition {
                    field: "node2.value".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!("true")),
                    target_node: "node2".to_string(),
                }],
            ),
        );
        nodes.insert(
            "node2".to_string(),
            DagNode::inference(
                Question::new_noul("Q2".to_string()),
                vec![NodeCondition {
                    field: "ghost.value".to_string(), // 未定義ノード
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!("true")),
                    target_node: "node3".to_string(),
                }],
            ),
        );
        nodes.insert(
            "node3".to_string(),
            DagNode::escalate(Option::<String>::None),
        );

        let dag = DagDefinition {
            dag_id: "context_ref_dag".to_string(),
            timeout_ms: 100,
            entry_node: "node1".to_string(),
            nodes,
        };

        // node1 が下流 node2 を参照しているためエラー
        let err1 = validate_dag(&dag).expect_err("下流ノード参照は拒否されること。");
        assert!(matches!(
            err1,
            DagValidationError::InvalidCondition {
                ref node_id,
                ref field,
                ..
            } if node_id == "node1" && field == "node2.value"
        ));
    }

    #[test]
    fn test_validate_dag_rejects_condition_type_inconsistency() {
        // 1. Confidence に対する In や非数値
        let mut nodes = IndexMap::new();
        nodes.insert(
            "n1".to_string(),
            DagNode::inference(
                Question::new_noul("Q1".to_string()),
                vec![NodeCondition {
                    field: "confidence".to_string(),
                    op: NodeConditionOp::In,
                    value: Some(serde_json::json!([0.8])),
                    target_node: "n2".to_string(),
                }],
            ),
        );
        nodes.insert("n2".to_string(), DagNode::escalate(Option::<String>::None));

        let dag = DagDefinition {
            dag_id: "bad_conf_dag".to_string(),
            timeout_ms: 100,
            entry_node: "n1".to_string(),
            nodes,
        };
        assert!(matches!(
            validate_dag(&dag),
            Err(DagValidationError::InvalidCondition { .. })
        ));

        // 2. Choice に対する大小比較
        let mut criteria = IndexMap::new();
        criteria.insert("opt1".to_string(), "選択肢1".to_string());
        let mut nodes_choice = IndexMap::new();
        nodes_choice.insert(
            "c1".to_string(),
            DagNode::inference(
                Question::new_choice("質問".to_string(), criteria),
                vec![NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Gt,
                    value: Some(serde_json::json!(1.0)),
                    target_node: "c2".to_string(),
                }],
            ),
        );
        nodes_choice.insert("c2".to_string(), DagNode::escalate(Option::<String>::None));
        let dag_choice = DagDefinition {
            dag_id: "bad_choice_dag".to_string(),
            timeout_ms: 100,
            entry_node: "c1".to_string(),
            nodes: nodes_choice,
        };
        assert!(matches!(
            validate_dag(&dag_choice),
            Err(DagValidationError::InvalidCondition { .. })
        ));

        // 3. Route に対する不正なルート文字列
        let mut nodes_route = IndexMap::new();
        nodes_route.insert(
            "r1".to_string(),
            DagNode::inference(
                Question::new_noul("Q".to_string()),
                vec![NodeCondition {
                    field: "route".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!("invalid_route_str")),
                    target_node: "r2".to_string(),
                }],
            ),
        );
        nodes_route.insert("r2".to_string(), DagNode::escalate(Option::<String>::None));
        let dag_route = DagDefinition {
            dag_id: "bad_route_dag".to_string(),
            timeout_ms: 100,
            entry_node: "r1".to_string(),
            nodes: nodes_route,
        };
        assert!(matches!(
            validate_dag(&dag_route),
            Err(DagValidationError::InvalidCondition { .. })
        ));
    }

    #[test]
    fn test_validate_dag_valid_interpolations() {
        let mut nodes = IndexMap::new();
        nodes.insert(
            "node1".to_string(),
            DagNode::inference(
                Question::new_noul("最初の質問？".to_string()),
                vec![NodeCondition {
                    field: "confidence".to_string(),
                    op: NodeConditionOp::Gte,
                    value: Some(serde_json::json!(0.5)),
                    target_node: "node2".to_string(),
                }],
            ),
        );

        let mut criteria = IndexMap::new();
        criteria.insert(
            "opt_a".to_string(),
            "先行判定 {node1.value} (確信度: {node1.confidence}) に基づく".to_string(),
        );
        nodes.insert(
            "node2".to_string(),
            DagNode::inference(
                Question::new_choice("先行値={node1.answer} を考慮した質問".to_string(), criteria),
                Vec::new(),
            ),
        );

        let dag = DagDefinition {
            dag_id: "valid_interp_dag".to_string(),
            timeout_ms: 100,
            entry_node: "node1".to_string(),
            nodes,
        };
        assert!(validate_dag(&dag).is_ok());
    }

    #[test]
    fn test_validate_dag_rejects_invalid_interpolations() {
        // 1. 未定義ノード補間
        let mut nodes1 = IndexMap::new();
        nodes1.insert(
            "node1".to_string(),
            DagNode::inference(
                Question::new_noul("参照={ghost.value}".to_string()),
                Vec::new(),
            ),
        );
        let dag1 = DagDefinition {
            dag_id: "ghost_dag".to_string(),
            timeout_ms: 100,
            entry_node: "node1".to_string(),
            nodes: nodes1,
        };
        let err1 = validate_dag(&dag1).expect_err("未定義ノード補間は拒否されること。");
        assert!(matches!(
            err1,
            DagValidationError::InvalidInterpolation {
                ref node_id,
                ref placeholder,
                ..
            } if node_id == "node1" && placeholder == "ghost.value"
        ));

        // 2. 自己参照補間
        let mut nodes2 = IndexMap::new();
        nodes2.insert(
            "self_node".to_string(),
            DagNode::inference(
                Question::new_noul("自己={self_node.value}".to_string()),
                Vec::new(),
            ),
        );
        let dag2 = DagDefinition {
            dag_id: "self_interp_dag".to_string(),
            timeout_ms: 100,
            entry_node: "self_node".to_string(),
            nodes: nodes2,
        };
        let err2 = validate_dag(&dag2).expect_err("自己参照補間は拒否されること。");
        assert!(matches!(
            err2,
            DagValidationError::InvalidInterpolation {
                ref node_id,
                ref placeholder,
                ..
            } if node_id == "self_node" && placeholder == "self_node.value"
        ));

        // 3. 不正プロパティ名補間
        let mut nodes3 = IndexMap::new();
        nodes3.insert(
            "n1".to_string(),
            DagNode::inference(
                Question::new_noul("Q1".to_string()),
                vec![NodeCondition {
                    field: "route".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!("auto_execute")),
                    target_node: "n2".to_string(),
                }],
            ),
        );
        nodes3.insert(
            "n2".to_string(),
            DagNode::inference(
                Question::new_noul("不正プロパティ={n1.bogus}".to_string()),
                Vec::new(),
            ),
        );
        let dag3 = DagDefinition {
            dag_id: "bad_prop_dag".to_string(),
            timeout_ms: 100,
            entry_node: "n1".to_string(),
            nodes: nodes3,
        };
        let err3 = validate_dag(&dag3).expect_err("不正プロパティ名補間は拒否されること。");
        assert!(matches!(
            err3,
            DagValidationError::InvalidInterpolation {
                ref node_id,
                ref placeholder,
                ..
            } if node_id == "n2" && placeholder == "n1.bogus"
        ));

        // 4. 下流ノード参照補間
        let mut nodes4 = IndexMap::new();
        nodes4.insert(
            "first".to_string(),
            DagNode::inference(
                Question::new_noul("下流参照={second.value}".to_string()),
                vec![NodeCondition {
                    field: "route".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!("auto_execute")),
                    target_node: "second".to_string(),
                }],
            ),
        );
        nodes4.insert(
            "second".to_string(),
            DagNode::inference(Question::new_noul("Q2".to_string()), Vec::new()),
        );
        let dag4 = DagDefinition {
            dag_id: "downstream_dag".to_string(),
            timeout_ms: 100,
            entry_node: "first".to_string(),
            nodes: nodes4,
        };
        let err4 = validate_dag(&dag4).expect_err("下流ノード補間は拒否されること。");
        assert!(matches!(
            err4,
            DagValidationError::InvalidInterpolation {
                ref node_id,
                ref placeholder,
                ..
            } if node_id == "first" && placeholder == "second.value"
        ));
    }

    #[test]
    fn test_parallel_join_rejects_implicit_answer_condition() {
        let mut nodes = IndexMap::new();
        nodes.insert(
            "par_entry".to_string(),
            DagNode::parallel(
                vec!["noul_task".to_string(), "choice_task".to_string()],
                "join_branch",
            ),
        );
        nodes.insert(
            "noul_task".to_string(),
            DagNode::inference(Question::new_noul("真偽判定".to_string()), Vec::new()),
        );
        let mut choice_criteria = IndexMap::new();
        choice_criteria.insert("opt_a".to_string(), "選択肢A".to_string());
        nodes.insert(
            "choice_task".to_string(),
            DagNode::inference(
                Question::new_choice("選択肢判定".to_string(), choice_criteria),
                Vec::new(),
            ),
        );
        nodes.insert(
            "join_branch".to_string(),
            DagNode::branch(vec![NodeCondition {
                field: "answer".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!("opt_a")),
                target_node: "term".to_string(),
            }]),
        );
        nodes.insert("term".to_string(), DagNode::escalate(Some("終端")));

        let dag = DagDefinition {
            dag_id: "parallel_implicit_answer_dag".to_string(),
            timeout_ms: 100,
            entry_node: "par_entry".to_string(),
            nodes,
        };

        let err = validate_dag(&dag)
            .expect_err("並行ノードからの合流における暗黙 answer 参照は拒否されること。");
        assert!(matches!(
            err,
            DagValidationError::InvalidCondition {
                ref node_id,
                ref field,
                ..
            } if node_id == "join_branch" && field == "answer"
        ));
    }

    #[test]
    fn test_parallel_join_allows_explicit_context_property() {
        let mut nodes = IndexMap::new();
        nodes.insert(
            "par_entry".to_string(),
            DagNode::parallel(
                vec!["noul_task".to_string(), "choice_task".to_string()],
                "join_branch",
            ),
        );
        nodes.insert(
            "noul_task".to_string(),
            DagNode::inference(Question::new_noul("真偽判定".to_string()), Vec::new()),
        );
        let mut choice_criteria = IndexMap::new();
        choice_criteria.insert("opt_a".to_string(), "選択肢A".to_string());
        nodes.insert(
            "choice_task".to_string(),
            DagNode::inference(
                Question::new_choice("選択肢判定".to_string(), choice_criteria),
                Vec::new(),
            ),
        );
        nodes.insert(
            "join_branch".to_string(),
            DagNode::branch(vec![NodeCondition {
                field: "choice_task.value".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!("opt_a")),
                target_node: "term".to_string(),
            }]),
        );
        nodes.insert("term".to_string(), DagNode::escalate(Some("終端")));

        let dag = DagDefinition {
            dag_id: "parallel_explicit_prop_dag".to_string(),
            timeout_ms: 100,
            entry_node: "par_entry".to_string(),
            nodes,
        };

        assert!(
            validate_dag(&dag).is_ok(),
            "明示的な context プロパティ参照はバリデーションに合格すること。"
        );
    }

    #[test]
    fn test_branch_join_condition_type_consistency() {
        // 1. 異種型合流 (Noul と Choice) での暗黙 answer 参照拒否
        let mut nodes_hetero = IndexMap::new();
        nodes_hetero.insert(
            "root".to_string(),
            DagNode::inference(
                Question::new_noul("ルート推論".to_string()),
                vec![
                    NodeCondition {
                        field: "confidence".to_string(),
                        op: NodeConditionOp::ConfidenceGte,
                        value: Some(serde_json::json!(0.5)),
                        target_node: "path_noul".to_string(),
                    },
                    NodeCondition {
                        field: "confidence".to_string(),
                        op: NodeConditionOp::ConfidenceLt,
                        value: Some(serde_json::json!(0.5)),
                        target_node: "path_choice".to_string(),
                    },
                ],
            ),
        );
        nodes_hetero.insert(
            "path_noul".to_string(),
            DagNode::inference(
                Question::new_noul("真偽".to_string()),
                vec![NodeCondition {
                    field: "confidence".to_string(),
                    op: NodeConditionOp::ConfidenceGte,
                    value: Some(serde_json::json!(0.0)),
                    target_node: "join_node".to_string(),
                }],
            ),
        );
        let mut criteria = IndexMap::new();
        criteria.insert("yes".to_string(), "はい".to_string());
        nodes_hetero.insert(
            "path_choice".to_string(),
            DagNode::inference(
                Question::new_choice("選択".to_string(), criteria),
                vec![NodeCondition {
                    field: "confidence".to_string(),
                    op: NodeConditionOp::ConfidenceGte,
                    value: Some(serde_json::json!(0.0)),
                    target_node: "join_node".to_string(),
                }],
            ),
        );
        nodes_hetero.insert(
            "join_node".to_string(),
            DagNode::branch(vec![NodeCondition {
                field: "answer".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!("yes")),
                target_node: "term".to_string(),
            }]),
        );
        nodes_hetero.insert("term".to_string(), DagNode::escalate(Some("終端")));

        let dag_hetero = DagDefinition {
            dag_id: "hetero_join_dag".to_string(),
            timeout_ms: 100,
            entry_node: "root".to_string(),
            nodes: nodes_hetero,
        };
        let err = validate_dag(&dag_hetero).expect_err("異種型の合流は拒否されること。");
        assert!(matches!(
            err,
            DagValidationError::InvalidCondition {
                ref node_id,
                ref field,
                ..
            } if node_id == "join_node" && field == "answer"
        ));

        // 2. 同種型合流 (Choice と Choice) で全 criteria に存在するキーの検証合格
        let mut nodes_homo = IndexMap::new();
        nodes_homo.insert(
            "root".to_string(),
            DagNode::inference(
                Question::new_noul("ルート推論".to_string()),
                vec![
                    NodeCondition {
                        field: "confidence".to_string(),
                        op: NodeConditionOp::ConfidenceGte,
                        value: Some(serde_json::json!(0.5)),
                        target_node: "choice_a".to_string(),
                    },
                    NodeCondition {
                        field: "confidence".to_string(),
                        op: NodeConditionOp::ConfidenceLt,
                        value: Some(serde_json::json!(0.5)),
                        target_node: "choice_b".to_string(),
                    },
                ],
            ),
        );
        let mut crit_a = IndexMap::new();
        crit_a.insert("approved".to_string(), "承認".to_string());
        crit_a.insert("rejected".to_string(), "却下".to_string());
        nodes_homo.insert(
            "choice_a".to_string(),
            DagNode::inference(
                Question::new_choice("A判定".to_string(), crit_a),
                vec![NodeCondition {
                    field: "confidence".to_string(),
                    op: NodeConditionOp::ConfidenceGte,
                    value: Some(serde_json::json!(0.0)),
                    target_node: "join_node".to_string(),
                }],
            ),
        );
        let mut crit_b = IndexMap::new();
        crit_b.insert("approved".to_string(), "承認".to_string());
        crit_b.insert("escalated".to_string(), "上位判断".to_string());
        nodes_homo.insert(
            "choice_b".to_string(),
            DagNode::inference(
                Question::new_choice("B判定".to_string(), crit_b),
                vec![NodeCondition {
                    field: "confidence".to_string(),
                    op: NodeConditionOp::ConfidenceGte,
                    value: Some(serde_json::json!(0.0)),
                    target_node: "join_node".to_string(),
                }],
            ),
        );
        nodes_homo.insert(
            "join_node".to_string(),
            DagNode::branch(vec![NodeCondition {
                field: "answer".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!("approved")),
                target_node: "term".to_string(),
            }]),
        );
        nodes_homo.insert("term".to_string(), DagNode::escalate(Some("終端")));

        let dag_homo = DagDefinition {
            dag_id: "homo_join_dag".to_string(),
            timeout_ms: 100,
            entry_node: "root".to_string(),
            nodes: nodes_homo,
        };
        assert!(
            validate_dag(&dag_homo).is_ok(),
            "全上流 Choice ノードの criteria に存在する共通キーの評価は合格すること。"
        );

        // 3. 片方の上流 Choice ノードにしか存在しないキー ('rejected') は拒否されること
        let mut nodes_homo_missing = dag_homo.nodes.clone();
        *nodes_homo_missing.get_mut("join_node").unwrap() = DagNode::branch(vec![NodeCondition {
            field: "answer".to_string(),
            op: NodeConditionOp::Eq,
            value: Some(serde_json::json!("rejected")),
            target_node: "term".to_string(),
        }]);
        let dag_missing_key = DagDefinition {
            dag_id: "homo_missing_key_dag".to_string(),
            timeout_ms: 100,
            entry_node: "root".to_string(),
            nodes: nodes_homo_missing,
        };
        let err_missing = validate_dag(&dag_missing_key)
            .expect_err("片方の上流に存在しないキーの評価は拒否されること。");
        assert!(matches!(
            err_missing,
            DagValidationError::InvalidCondition {
                ref node_id,
                ref field,
                ..
            } if node_id == "join_node" && field == "answer"
        ));
    }

    #[test]
    fn test_parallel_invalid_specs() {
        // 1. parallel_nodes が空の場合は拒否
        let mut nodes_empty = IndexMap::new();
        nodes_empty.insert(
            "parallel".to_string(),
            DagNode::parallel(Vec::new(), "join"),
        );
        nodes_empty.insert("join".to_string(), DagNode::escalate(Some("合流")));

        let dag_empty = DagDefinition {
            dag_id: "parallel_empty".to_string(),
            timeout_ms: 100,
            entry_node: "parallel".to_string(),
            nodes: nodes_empty,
        };
        let err_empty =
            validate_dag(&dag_empty).expect_err("parallel_nodes が空の場合はエラーとなること。");
        assert!(matches!(
            err_empty,
            DagValidationError::InvalidNodeSpec {
                ref node_id,
                node_type: DagNodeType::Parallel,
                ..
            } if node_id == "parallel"
        ));

        // 2. Parallel ノード内のタスクが Inference でない (Branch) 場合は拒否
        let mut nodes_not_inf = IndexMap::new();
        nodes_not_inf.insert(
            "parallel".to_string(),
            DagNode::parallel(vec!["p_branch".to_string()], "join"),
        );
        nodes_not_inf.insert("p_branch".to_string(), DagNode::branch(Vec::new()));
        nodes_not_inf.insert("join".to_string(), DagNode::escalate(Some("合流")));

        let dag_not_inf = DagDefinition {
            dag_id: "parallel_not_inf".to_string(),
            timeout_ms: 100,
            entry_node: "parallel".to_string(),
            nodes: nodes_not_inf,
        };
        let err_not_inf = validate_dag(&dag_not_inf)
            .expect_err("並行タスクが Inference 以外の場合はエラーとなること。");
        assert!(matches!(
            err_not_inf,
            DagValidationError::InvalidNodeSpec {
                ref node_id,
                node_type: DagNodeType::Branch,
                ..
            } if node_id == "p_branch"
        ));

        // 3. Parallel ノード内のタスクが独自の conditions を持っている場合は拒否
        let mut nodes_task_cond = IndexMap::new();
        nodes_task_cond.insert(
            "parallel".to_string(),
            DagNode::parallel(vec!["p_task".to_string()], "join"),
        );
        nodes_task_cond.insert(
            "p_task".to_string(),
            DagNode::inference(
                Question::new_noul("並行タスク".to_string()),
                vec![NodeCondition {
                    field: "confidence".to_string(),
                    op: NodeConditionOp::ConfidenceGte,
                    value: Some(serde_json::json!(0.5)),
                    target_node: "join".to_string(),
                }],
            ),
        );
        nodes_task_cond.insert("join".to_string(), DagNode::escalate(Some("合流")));

        let dag_task_cond = DagDefinition {
            dag_id: "parallel_task_with_cond".to_string(),
            timeout_ms: 100,
            entry_node: "parallel".to_string(),
            nodes: nodes_task_cond,
        };
        let err_task_cond = validate_dag(&dag_task_cond)
            .expect_err("並行タスクが独自の conditions を持っている場合はエラーとなること。");
        assert!(matches!(
            err_task_cond,
            DagValidationError::InvalidNodeSpec {
                ref node_id,
                node_type: DagNodeType::Inference,
                ..
            } if node_id == "p_task"
        ));
    }

    #[test]
    fn test_branch_without_upstream_inference_rejects_confidence_and_route() {
        // 1. entry_node が Branch で field: "confidence" を評価しようとする場合は拒否
        let mut nodes_conf = IndexMap::new();
        nodes_conf.insert(
            "entry_branch".to_string(),
            DagNode::branch(vec![NodeCondition {
                field: "confidence".to_string(),
                op: NodeConditionOp::ConfidenceGte,
                value: Some(serde_json::json!(0.8)),
                target_node: "term".to_string(),
            }]),
        );
        nodes_conf.insert("term".to_string(), DagNode::escalate(Some("終端")));

        let dag_conf = DagDefinition {
            dag_id: "branch_entry_conf".to_string(),
            timeout_ms: 100,
            entry_node: "entry_branch".to_string(),
            nodes: nodes_conf,
        };
        let err_conf = validate_dag(&dag_conf)
            .expect_err("上流推論ノード無しの confidence 評価は拒否されること。");
        assert!(matches!(
            err_conf,
            DagValidationError::InvalidCondition {
                ref node_id,
                ref field,
                ..
            } if node_id == "entry_branch" && field == "confidence"
        ));

        // 2. entry_node が Branch で field: "route" を評価しようとする場合は拒否
        let mut nodes_route = IndexMap::new();
        nodes_route.insert(
            "entry_branch".to_string(),
            DagNode::branch(vec![NodeCondition {
                field: "route".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!("auto_execute")),
                target_node: "term".to_string(),
            }]),
        );
        nodes_route.insert("term".to_string(), DagNode::escalate(Some("終端")));

        let dag_route = DagDefinition {
            dag_id: "branch_entry_route".to_string(),
            timeout_ms: 100,
            entry_node: "entry_branch".to_string(),
            nodes: nodes_route,
        };
        let err_route = validate_dag(&dag_route)
            .expect_err("上流推論ノード無しの route 評価は拒否されること。");
        assert!(matches!(
            err_route,
            DagValidationError::InvalidCondition {
                ref node_id,
                ref field,
                ..
            } if node_id == "entry_branch" && field == "route"
        ));
    }

    #[test]
    fn test_branch_with_upstream_inference_allows_confidence_and_route() {
        let mut nodes = IndexMap::new();
        nodes.insert(
            "root".to_string(),
            DagNode::inference(
                Question::new_noul("推論？".to_string()),
                vec![NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::Value::Bool(true)),
                    target_node: "branch".to_string(),
                }],
            ),
        );
        nodes.insert(
            "branch".to_string(),
            DagNode::branch(vec![
                NodeCondition {
                    field: "route".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!("auto_execute")),
                    target_node: "term".to_string(),
                },
                NodeCondition {
                    field: "confidence".to_string(),
                    op: NodeConditionOp::ConfidenceGte,
                    value: Some(serde_json::json!(0.7)),
                    target_node: "term".to_string(),
                },
            ]),
        );
        nodes.insert("term".to_string(), DagNode::escalate(Some("終端")));

        let dag = DagDefinition {
            dag_id: "branch_with_upstream_inference".to_string(),
            timeout_ms: 100,
            entry_node: "root".to_string(),
            nodes,
        };
        assert!(
            validate_dag(&dag).is_ok(),
            "上流に Inference ノードが存在する Branch ノードでの confidence / route 評価は合格すること。"
        );
    }
}
