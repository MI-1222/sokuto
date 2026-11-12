//! # DAG 静的検証モジュール
//!
//! 実行前の DAG 定義に対し、循環参照 (閉路)、未定義ノード参照、
//! 型契約違反、到達不能ノードを Fast-Fail で静的検証する。

use std::collections::{HashMap, HashSet};

use sokuto_core::schema::{Criteria, Question, QuestionType};

use crate::dag::executor::{ConditionField, ContextProp};
use crate::dag::schema::{DagDefinition, DagNode, DagNodeType, NodeCondition, NodeConditionOp};

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
}

/// DAG 定義の整合性と無閉路性を静的に検証する。
///
/// # 検査項目
/// 1. `nodes` が 1 つ以上のノードを含むこと。
/// 2. `entry_node` が `nodes` に実在すること。
/// 3. 各ノードのプロパティ検証 (推論ノードの質問、並行ノードの合流先など)。
/// 4. すべての遷移先 (`target_node`, `parallel_nodes`, `join_node`) が実在すること。
/// 5. DFS による有向グラフの閉路 (サイクル) 検出。
/// 6. `entry_node` から全ノードへの到達性確認。
/// 7. 各条件式のフィールド・上流参照・型整合性検証。
pub fn validate_dag(dag: &DagDefinition) -> Result<(), DagValidationError> {
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
        match node.node_type {
            DagNodeType::Inference => {
                if node.question.is_none() {
                    return Err(DagValidationError::InvalidNodeSpec {
                        node_id: node_id.clone(),
                        node_type: node.node_type,
                        reason: "Inference ノードには question の定義が必須です。".to_string(),
                    });
                }
            }
            DagNodeType::Branch => {
                // conditions が空の場合は終端ノードとして扱われ、次ノードなしで正常終了する。
            }
            DagNodeType::Parallel => {
                if node.parallel_nodes.is_empty() {
                    return Err(DagValidationError::InvalidNodeSpec {
                        node_id: node_id.clone(),
                        node_type: node.node_type,
                        reason: "Parallel ノードには 1 つ以上の parallel_nodes が必要です。"
                            .to_string(),
                    });
                }
                let join_node =
                    node.join_node
                        .as_ref()
                        .ok_or_else(|| DagValidationError::InvalidNodeSpec {
                            node_id: node_id.clone(),
                            node_type: node.node_type,
                            reason: "Parallel ノードには join_node の定義が必須です。".to_string(),
                        })?;
                if !dag.nodes.contains_key(join_node) {
                    return Err(DagValidationError::TargetNodeNotFound {
                        from_node: node_id.clone(),
                        target_node: join_node.clone(),
                    });
                }
                for p_node_id in &node.parallel_nodes {
                    let p_node = dag.nodes.get(p_node_id).ok_or_else(|| {
                        DagValidationError::TargetNodeNotFound {
                            from_node: node_id.clone(),
                            target_node: p_node_id.clone(),
                        }
                    })?;
                    if p_node.node_type != DagNodeType::Inference {
                        return Err(DagValidationError::InvalidNodeSpec {
                            node_id: p_node_id.clone(),
                            node_type: p_node.node_type,
                            reason: "Parallel ノード内のタスクノードは Inference 型である必要があります。".to_string(),
                        });
                    }
                    if !p_node.conditions.is_empty() {
                        return Err(DagValidationError::InvalidNodeSpec {
                            node_id: p_node_id.clone(),
                            node_type: p_node.node_type,
                            reason: "Parallel ノード内のタスクノードは独自の conditions を持てません(親ノードの join_node へ合流するため)。".to_string(),
                        });
                    }
                    if p_node.question.is_none() {
                        return Err(DagValidationError::InvalidNodeSpec {
                            node_id: p_node_id.clone(),
                            node_type: p_node.node_type,
                            reason:
                                "Parallel ノード内のタスクノードには question の定義が必須です。"
                                    .to_string(),
                        });
                    }
                }
            }
            DagNodeType::Escalate => {}
        }

        for cond in &node.conditions {
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
        match node.node_type {
            DagNodeType::Inference | DagNodeType::Branch => {
                for cond in &node.conditions {
                    neighbors.push(cond.target_node.as_str());
                }
            }
            DagNodeType::Parallel => {
                for p in &node.parallel_nodes {
                    neighbors.push(p.as_str());
                }
                if let Some(join) = &node.join_node {
                    for p in &node.parallel_nodes {
                        // 並行ノード群から合流ノードへのエッジも考慮
                        adj.entry(p.as_str()).or_default().push(join.as_str());
                    }
                }
            }
            DagNodeType::Escalate => {}
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

    // 5. 条件式の静的検証 (フィールド妥当性・先祖参照・型整合性)
    validate_node_conditions(dag, &adj)?;

    Ok(())
}

/// 条件値 (`serde_json::Value`) が有限な `f64` (または数値文字列) として解釈可能か判定する。
fn is_finite_number_value(val: Option<&serde_json::Value>) -> bool {
    let Some(v) = val else { return false };
    if let Some(n) = v.as_f64() {
        return n.is_finite();
    }
    if let Some(s) = v.as_str() {
        return s.trim().parse::<f64>().is_ok_and(|f| f.is_finite());
    }
    false
}

/// 条件式を評価する対象の質問仕様を解決する。
///
/// `Inference` ノードの場合は自身の `question` を返し、
/// `Branch` ノードの場合は直近の上流 `Inference` ノード群の質問仕様を探索・解決する。
fn resolve_target_question<'a>(
    node_id: &'a str,
    node: &'a DagNode,
    dag: &'a DagDefinition,
    rev_adj: &HashMap<&str, Vec<&str>>,
) -> Result<Option<&'a Question>, DagValidationError> {
    if let Some(ref q) = node.question {
        return Ok(Some(q));
    }

    let mut visited = HashSet::new();
    let mut q = Vec::new();
    if let Some(preds) = rev_adj.get(node_id) {
        for &pred in preds {
            q.push(pred);
            visited.insert(pred);
        }
    }

    let mut upstream_questions = Vec::new();
    while let Some(curr) = q.pop() {
        if let Some(curr_node) = dag.nodes.get(curr) {
            if curr_node.node_type == DagNodeType::Inference {
                if let Some(ref question) = curr_node.question {
                    upstream_questions.push(question);
                }
            } else if let Some(preds) = rev_adj.get(curr) {
                for &pred in preds {
                    if visited.insert(pred) {
                        q.push(pred);
                    }
                }
            }
        }
    }

    if upstream_questions.is_empty() {
        return Err(DagValidationError::InvalidCondition {
            node_id: node_id.to_string(),
            field: "answer".to_string(),
            reason: "上流に推論結果を出力する Inference ノードが存在しないため answer フィールドを評価できません。".to_string(),
        });
    }

    Ok(Some(upstream_questions[0]))
}

/// 各ノードの条件式リスト (`conditions`) に対し、フィールド妥当性・先祖参照・型整合性を静的検証する。
fn validate_node_conditions(
    dag: &DagDefinition,
    adj: &HashMap<&str, Vec<&str>>,
) -> Result<(), DagValidationError> {
    // 逆隣接リストの構築
    let mut rev_adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for (&u, neighbors) in adj {
        for &v in neighbors {
            rev_adj.entry(v).or_default().push(u);
        }
    }

    // 先祖関係 (上流先行ノード集合) マップの構築
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

    // 各ノードの各条件式を静的検査
    for (node_id, node) in &dag.nodes {
        for cond in &node.conditions {
            validate_single_condition(node_id, node, cond, dag, &ancestors, &rev_adj)?;
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
) -> Result<(), DagValidationError> {
    let field = ConditionField::parse(&cond.field);

    match field {
        ConditionField::Unknown => Err(DagValidationError::InvalidCondition {
            node_id: node_id.to_string(),
            field: cond.field.clone(),
            reason: "未定義または構文不正なフィールド名です。".to_string(),
        }),
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
            if ref_node.node_type != DagNodeType::Inference {
                return Err(DagValidationError::InvalidCondition {
                    node_id: node_id.to_string(),
                    field: cond.field.clone(),
                    reason: format!(
                        "参照先ノード '{ref_id}' ({:?}) は推論結果を出力する Inference ノードである必要があります。",
                        ref_node.node_type
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
                    if !is_finite_number_value(cond.value.as_ref()) {
                        return Err(DagValidationError::InvalidCondition {
                            node_id: node_id.to_string(),
                            field: cond.field.clone(),
                            reason: "確信度プロパティには有限数値 (または数値文字列) を指定する必要があります。".to_string(),
                        });
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
                }
            }
            Ok(())
        }
        ConditionField::Confidence => {
            if cond.op == NodeConditionOp::In {
                return Err(DagValidationError::InvalidCondition {
                    node_id: node_id.to_string(),
                    field: cond.field.clone(),
                    reason: "confidence フィールドに対して 'in' 演算子は使用できません。"
                        .to_string(),
                });
            }
            if !is_finite_number_value(cond.value.as_ref()) {
                return Err(DagValidationError::InvalidCondition {
                    node_id: node_id.to_string(),
                    field: cond.field.clone(),
                    reason: "confidence フィールドには有限数値 (または数値文字列) を指定する必要があります。".to_string(),
                });
            }
            Ok(())
        }
        ConditionField::Route => {
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
            Ok(())
        }
        ConditionField::Answer => {
            let target_q = resolve_target_question(node_id, node, dag, rev_adj)?;
            let q_type = target_q.map(|q| q.question_type);

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
                            if let Some(q) = target_q
                                && let Some(Criteria::Map(ref map)) = q.criteria
                                && !map.contains_key(s)
                            {
                                return Err(DagValidationError::InvalidCondition {
                                    node_id: node_id.to_string(),
                                    field: cond.field.clone(),
                                    reason: format!(
                                        "Choice 質問の criteria 定義に候補キー '{s}' が存在しません。"
                                    ),
                                });
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
                                if let Some(q) = target_q
                                    && let Some(Criteria::Map(ref map)) = q.criteria
                                    && !map.contains_key(s)
                                {
                                    return Err(DagValidationError::InvalidCondition {
                                        node_id: node_id.to_string(),
                                        field: cond.field.clone(),
                                        reason: format!(
                                            "Choice 質問の criteria 定義に候補キー '{s}' が存在しません。"
                                        ),
                                    });
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
                    if !is_finite_number_value(cond.value.as_ref()) {
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
                        if !is_finite_number_value(cond.value.as_ref()) {
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
            Ok(())
        }
    }
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
            DagNode {
                node_type: DagNodeType::Inference,
                question: Some(Question::new_noul("テスト？".to_string())),
                conditions: vec![NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::Value::Bool(true)),
                    target_node: "node2".to_string(),
                }],
                parallel_nodes: Vec::new(),
                join_node: None,
                reason: None,
            },
        );
        nodes.insert(
            "node2".to_string(),
            DagNode {
                node_type: DagNodeType::Escalate,
                question: None,
                conditions: Vec::new(),
                parallel_nodes: Vec::new(),
                join_node: None,
                reason: Some("終了".to_string()),
            },
        );

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
            DagNode {
                node_type: DagNodeType::Inference,
                question: Some(Question::new_noul("Q1".to_string())),
                conditions: vec![NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::Value::Bool(true)),
                    target_node: "node2".to_string(),
                }],
                parallel_nodes: Vec::new(),
                join_node: None,
                reason: None,
            },
        );
        nodes.insert(
            "node2".to_string(),
            DagNode {
                node_type: DagNodeType::Inference,
                question: Some(Question::new_noul("Q2".to_string())),
                conditions: vec![NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::Value::Bool(true)),
                    target_node: "node1".to_string(), // 閉路
                }],
                parallel_nodes: Vec::new(),
                join_node: None,
                reason: None,
            },
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
            DagNode {
                node_type: DagNodeType::Inference,
                question: Some(Question::new_noul("Q1".to_string())),
                conditions: vec![NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::Value::Bool(true)),
                    target_node: "non_existent".to_string(),
                }],
                parallel_nodes: Vec::new(),
                join_node: None,
                reason: None,
            },
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
        nodes.insert(
            "entry".to_string(),
            DagNode {
                node_type: DagNodeType::Branch,
                question: None,
                conditions: Vec::new(),
                parallel_nodes: Vec::new(),
                join_node: None,
                reason: Some("終端".to_string()),
            },
        );
        nodes.insert(
            "orphan".to_string(),
            DagNode {
                node_type: DagNodeType::Escalate,
                question: None,
                conditions: Vec::new(),
                parallel_nodes: Vec::new(),
                join_node: None,
                reason: Some("孤立ノード".to_string()),
            },
        );

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
            DagNode {
                node_type: DagNodeType::Parallel,
                question: None,
                conditions: Vec::new(),
                parallel_nodes: vec!["p1".to_string(), "p2".to_string()],
                join_node: Some("join".to_string()),
                reason: None,
            },
        );
        for id in ["p1", "p2"] {
            nodes.insert(
                id.to_string(),
                DagNode {
                    node_type: DagNodeType::Inference,
                    question: Some(Question::new_noul(format!("質問 {id}"))),
                    conditions: Vec::new(),
                    parallel_nodes: Vec::new(),
                    join_node: None,
                    reason: None,
                },
            );
        }
        nodes.insert(
            "join".to_string(),
            DagNode {
                node_type: DagNodeType::Branch,
                question: None,
                conditions: Vec::new(),
                parallel_nodes: Vec::new(),
                join_node: None,
                reason: Some("合流終端".to_string()),
            },
        );

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
            DagNode {
                node_type: DagNodeType::Inference,
                question: Some(Question::new_noul("テスト？".to_string())),
                conditions: vec![NodeCondition {
                    field: "unknown_field".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!(true)),
                    target_node: "node2".to_string(),
                }],
                parallel_nodes: Vec::new(),
                join_node: None,
                reason: None,
            },
        );
        nodes.insert(
            "node2".to_string(),
            DagNode {
                node_type: DagNodeType::Escalate,
                question: None,
                conditions: Vec::new(),
                parallel_nodes: Vec::new(),
                join_node: None,
                reason: None,
            },
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
            DagNode {
                node_type: DagNodeType::Inference,
                question: Some(Question::new_noul("Q1".to_string())),
                conditions: vec![NodeCondition {
                    field: "node2.value".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!("true")),
                    target_node: "node2".to_string(),
                }],
                parallel_nodes: Vec::new(),
                join_node: None,
                reason: None,
            },
        );
        nodes.insert(
            "node2".to_string(),
            DagNode {
                node_type: DagNodeType::Inference,
                question: Some(Question::new_noul("Q2".to_string())),
                conditions: vec![NodeCondition {
                    field: "ghost.value".to_string(), // 未定義ノード
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!("true")),
                    target_node: "node3".to_string(),
                }],
                parallel_nodes: Vec::new(),
                join_node: None,
                reason: None,
            },
        );
        nodes.insert(
            "node3".to_string(),
            DagNode {
                node_type: DagNodeType::Escalate,
                question: None,
                conditions: Vec::new(),
                parallel_nodes: Vec::new(),
                join_node: None,
                reason: None,
            },
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
            DagNode {
                node_type: DagNodeType::Inference,
                question: Some(Question::new_noul("Q1".to_string())),
                conditions: vec![NodeCondition {
                    field: "confidence".to_string(),
                    op: NodeConditionOp::In,
                    value: Some(serde_json::json!([0.8])),
                    target_node: "n2".to_string(),
                }],
                parallel_nodes: Vec::new(),
                join_node: None,
                reason: None,
            },
        );
        nodes.insert(
            "n2".to_string(),
            DagNode {
                node_type: DagNodeType::Escalate,
                question: None,
                conditions: Vec::new(),
                parallel_nodes: Vec::new(),
                join_node: None,
                reason: None,
            },
        );

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
            DagNode {
                node_type: DagNodeType::Inference,
                question: Some(Question::new_choice("質問".to_string(), criteria)),
                conditions: vec![NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Gt,
                    value: Some(serde_json::json!(1.0)),
                    target_node: "c2".to_string(),
                }],
                parallel_nodes: Vec::new(),
                join_node: None,
                reason: None,
            },
        );
        nodes_choice.insert(
            "c2".to_string(),
            DagNode {
                node_type: DagNodeType::Escalate,
                question: None,
                conditions: Vec::new(),
                parallel_nodes: Vec::new(),
                join_node: None,
                reason: None,
            },
        );
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
            DagNode {
                node_type: DagNodeType::Inference,
                question: Some(Question::new_noul("Q".to_string())),
                conditions: vec![NodeCondition {
                    field: "route".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!("invalid_route_str")),
                    target_node: "r2".to_string(),
                }],
                parallel_nodes: Vec::new(),
                join_node: None,
                reason: None,
            },
        );
        nodes_route.insert(
            "r2".to_string(),
            DagNode {
                node_type: DagNodeType::Escalate,
                question: None,
                conditions: Vec::new(),
                parallel_nodes: Vec::new(),
                join_node: None,
                reason: None,
            },
        );
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
}
