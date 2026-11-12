//! # DAG 静的検証モジュール
//!
//! 実行前の DAG 定義に対し、循環参照 (閉路)、未定義ノード参照、
//! 型契約違反、到達不能ノードを Fast-Fail で静的検証する。

use crate::dag::schema::{DagDefinition, DagNodeType};
use std::collections::{HashMap, HashSet};

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

    Ok(())
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
}
