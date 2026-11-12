//! # DAG 実行エンドポイント向け DTO スキーマモジュール
//!
//! 非同期グラフ巡回、反事実ルール、および多段階条件分岐判定を処理する
//! 宣言的 DAG 定義のリクエスト・レスポンス構造体を定義する。

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use sokuto_core::gating::GatingMetadata;
use sokuto_core::schema::{Answer, Question};
use sokuto_runtime::dag::DagValidationError;
use sokuto_runtime::dag::schema::{
    DagDefinition, DagExecutionResult, DagNode, DagNodeType, NodeCondition, NodeConditionOp,
    StepExecutionResult,
};
use utoipa::ToSchema;

/// DAG 実行リクエスト構造体。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct DagRequest {
    /// 判断の材料となる非構造化コンテキストデータ(文字列、オブジェクト、配列など)。
    #[schema(value_type = Object)]
    pub state: serde_json::Value,

    /// 実行対象の宣言的 DAG 定義。
    pub dag: DagDefinitionDto,

    /// クライアント指定のハードタイムアウト(ミリ秒、任意)。DAG 定義内の timeout_ms より優先される。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

/// 宣言的 DAG 定義 DTO 構造体。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct DagDefinitionDto {
    /// DAG の一意識別子。
    pub dag_id: String,

    /// 全体タイムアウト時間(ミリ秒)。
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,

    /// 実行開始ノードの識別子。
    pub entry_node: String,

    /// DAG を構成するノードマップ。
    pub nodes: IndexMap<String, DagNodeDto>,
}

fn default_timeout_ms() -> u64 {
    100
}

/// DAG ノード種別 DTO。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum DagNodeTypeDto {
    /// 単一質問のモデル推論を実行するノード。
    Inference,
    /// 先行ノードの判定値・確信度のみに基づいて遷移先を決定する純粋な条件分岐ノード。
    Branch,
    /// 独立した複数の質問を並行実行し合流するノード。
    Parallel,
    /// 確信度不足や例外検知時に上位 LLM または人手へ委ねるエスカレーションノード。
    Escalate,
}

/// DAG を構成する個々のノード定義 DTO。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct DagNodeDto {
    /// ノードの種別。
    pub node_type: DagNodeTypeDto,

    /// 推論ノード (`Inference`) で実行する質問仕様。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question: Option<Question>,

    /// ノード完了後の条件付き遷移規則リスト。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditions: Vec<NodeConditionDto>,

    /// 並行実行対象となるノード ID 一覧 (`Parallel` ノード用)。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parallel_nodes: Vec<String>,

    /// 並行実行完了後の合流先ノード ID (`Parallel` ノード用)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub join_node: Option<String>,

    /// エスカレーションまたは分岐の理由説明文。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// 条件付き遷移規則 DTO。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct NodeConditionDto {
    /// 評価対象フィールド (`"answer"`, `"confidence"`, `"route"` など)。
    pub field: String,

    /// 比較演算子。
    pub op: NodeConditionOpDto,

    /// 比較対象の値。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Option<Object>)]
    pub value: Option<serde_json::Value>,

    /// 条件合致時の遷移先ノード ID。
    pub target_node: String,
}

/// 条件評価演算子 DTO。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum NodeConditionOpDto {
    /// 等価 (`==`)。
    Eq,
    /// 不等価 (`!=`)。
    Neq,
    /// より大きい (`>`)。
    Gt,
    /// 以上 (`>=`)。
    Gte,
    /// より小さい (`<`)。
    Lt,
    /// 以下 (`<=`)。
    Lte,
    /// 配列に含まれる (`in`)。
    In,
    /// 確信度が指定値以上。
    ConfidenceGte,
    /// 確信度が指定値未満。
    ConfidenceLt,
}

/// DAG 実行レスポンス構造体。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct DagResponse {
    /// 実行された DAG の識別子。
    pub dag_id: String,

    /// 実行終了ステータス (`"completed"` または `"escalated"`)。
    pub status: String,

    /// 最終到達ノード ID。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_node: Option<String>,

    /// 最終推論ノードの決定結果(推論ノードを通過した場合)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_decision: Option<Answer>,

    /// 辿ったノード ID の実行経路。
    pub execution_path: Vec<String>,

    /// 各ノードの実行詳細マップ(ノード ID -> 実行記録)。
    pub steps: IndexMap<String, DagStepResultDto>,

    /// エスカレーション発生時の診断情報(該当する場合)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub escalation: Option<DagEscalationDetailDto>,

    /// リソース消費サマリー(`completion_tokens: 0` を含む)。
    pub usage: DagUsageDto,
}

/// 単一ステップの実行結果 DTO。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct DagStepResultDto {
    /// 実行ノード ID。
    pub node_id: String,

    /// ノード種別。
    pub node_type: DagNodeTypeDto,

    /// 推論ノードの場合の判定結果。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<Answer>,

    /// ステップ実行所要時間(ミリ秒)。
    pub duration_ms: f64,

    /// 次に遷移したノード ID。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_node: Option<String>,

    /// ノードの理由または補足メッセージ。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// エスカレーション発生時の詳細診断情報 DTO。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct DagEscalationDetailDto {
    /// エスカレーション理由。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,

    /// エスカレーションをトリガーしたノード ID。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,

    /// 上位 LLM (System 2) 向け Guided CoT プロンプト案。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,

    /// ゲーティング詳細メタデータ。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gating: Option<GatingMetadata>,
}

/// DAG 実行のリソース消費サマリー DTO。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
pub struct DagUsageDto {
    /// DAG 全体の実行所要時間(ミリ秒)。
    pub total_duration_ms: f64,

    /// 実行されたステップ数。
    pub steps_executed: usize,

    /// 自己回帰生成トークン数(常に 0)。
    pub completion_tokens: usize,
}

// ---------------------------------------------------------------------------
// sokuto-runtime 型との相互変換 (From / Into) 実装
// ---------------------------------------------------------------------------

impl From<DagNodeTypeDto> for DagNodeType {
    fn from(dto: DagNodeTypeDto) -> Self {
        match dto {
            DagNodeTypeDto::Inference => DagNodeType::Inference,
            DagNodeTypeDto::Branch => DagNodeType::Branch,
            DagNodeTypeDto::Parallel => DagNodeType::Parallel,
            DagNodeTypeDto::Escalate => DagNodeType::Escalate,
        }
    }
}

impl From<DagNodeType> for DagNodeTypeDto {
    fn from(kind: DagNodeType) -> Self {
        match kind {
            DagNodeType::Inference => DagNodeTypeDto::Inference,
            DagNodeType::Branch => DagNodeTypeDto::Branch,
            DagNodeType::Parallel => DagNodeTypeDto::Parallel,
            DagNodeType::Escalate => DagNodeTypeDto::Escalate,
        }
    }
}

impl From<NodeConditionOpDto> for NodeConditionOp {
    fn from(dto: NodeConditionOpDto) -> Self {
        match dto {
            NodeConditionOpDto::Eq => NodeConditionOp::Eq,
            NodeConditionOpDto::Neq => NodeConditionOp::Neq,
            NodeConditionOpDto::Gt => NodeConditionOp::Gt,
            NodeConditionOpDto::Gte => NodeConditionOp::Gte,
            NodeConditionOpDto::Lt => NodeConditionOp::Lt,
            NodeConditionOpDto::Lte => NodeConditionOp::Lte,
            NodeConditionOpDto::In => NodeConditionOp::In,
            NodeConditionOpDto::ConfidenceGte => NodeConditionOp::ConfidenceGte,
            NodeConditionOpDto::ConfidenceLt => NodeConditionOp::ConfidenceLt,
        }
    }
}

impl From<NodeConditionOp> for NodeConditionOpDto {
    fn from(op: NodeConditionOp) -> Self {
        match op {
            NodeConditionOp::Eq => NodeConditionOpDto::Eq,
            NodeConditionOp::Neq => NodeConditionOpDto::Neq,
            NodeConditionOp::Gt => NodeConditionOpDto::Gt,
            NodeConditionOp::Gte => NodeConditionOpDto::Gte,
            NodeConditionOp::Lt => NodeConditionOpDto::Lt,
            NodeConditionOp::Lte => NodeConditionOpDto::Lte,
            NodeConditionOp::In => NodeConditionOpDto::In,
            NodeConditionOp::ConfidenceGte => NodeConditionOpDto::ConfidenceGte,
            NodeConditionOp::ConfidenceLt => NodeConditionOpDto::ConfidenceLt,
        }
    }
}

impl From<NodeConditionDto> for NodeCondition {
    fn from(dto: NodeConditionDto) -> Self {
        Self {
            field: dto.field,
            op: dto.op.into(),
            value: dto.value,
            target_node: dto.target_node,
        }
    }
}

impl From<NodeCondition> for NodeConditionDto {
    fn from(c: NodeCondition) -> Self {
        Self {
            field: c.field,
            op: c.op.into(),
            value: c.value,
            target_node: c.target_node,
        }
    }
}

fn convert_node_dto(node_id: &str, dto: DagNodeDto) -> Result<DagNode, DagValidationError> {
    let node_type: DagNodeType = dto.node_type.into();
    match node_type {
        DagNodeType::Inference => {
            let question = dto
                .question
                .ok_or_else(|| DagValidationError::InvalidNodeSpec {
                    node_id: node_id.to_string(),
                    node_type,
                    reason: "Inference ノードには question の定義が必須です。".to_string(),
                })?;
            Ok(DagNode::Inference {
                question,
                conditions: dto.conditions.into_iter().map(Into::into).collect(),
                reason: dto.reason,
            })
        }
        DagNodeType::Branch => Ok(DagNode::Branch {
            conditions: dto.conditions.into_iter().map(Into::into).collect(),
            reason: dto.reason,
        }),
        DagNodeType::Parallel => {
            if !dto.conditions.is_empty() {
                return Err(DagValidationError::InvalidNodeSpec {
                    node_id: node_id.to_string(),
                    node_type,
                    reason: "Parallel ノードは独自の conditions を持てません。".to_string(),
                });
            }
            let join_node = dto
                .join_node
                .ok_or_else(|| DagValidationError::InvalidNodeSpec {
                    node_id: node_id.to_string(),
                    node_type,
                    reason: "Parallel ノードには join_node の定義が必須です。".to_string(),
                })?;
            Ok(DagNode::Parallel {
                parallel_nodes: dto.parallel_nodes,
                join_node,
                reason: dto.reason,
            })
        }
        DagNodeType::Escalate => {
            if !dto.conditions.is_empty() {
                return Err(DagValidationError::InvalidNodeSpec {
                    node_id: node_id.to_string(),
                    node_type,
                    reason: "Escalate ノードは終端ノードであり、conditions を持てません。"
                        .to_string(),
                });
            }
            Ok(DagNode::Escalate { reason: dto.reason })
        }
    }
}

impl TryFrom<DagNodeDto> for DagNode {
    type Error = DagValidationError;

    fn try_from(dto: DagNodeDto) -> Result<Self, Self::Error> {
        convert_node_dto("", dto)
    }
}

impl From<DagNode> for DagNodeDto {
    fn from(node: DagNode) -> Self {
        match node {
            DagNode::Inference {
                question,
                conditions,
                reason,
            } => Self {
                node_type: DagNodeTypeDto::Inference,
                question: Some(question),
                conditions: conditions.into_iter().map(Into::into).collect(),
                parallel_nodes: Vec::new(),
                join_node: None,
                reason,
            },
            DagNode::Branch { conditions, reason } => Self {
                node_type: DagNodeTypeDto::Branch,
                question: None,
                conditions: conditions.into_iter().map(Into::into).collect(),
                parallel_nodes: Vec::new(),
                join_node: None,
                reason,
            },
            DagNode::Parallel {
                parallel_nodes,
                join_node,
                reason,
            } => Self {
                node_type: DagNodeTypeDto::Parallel,
                question: None,
                conditions: Vec::new(),
                parallel_nodes,
                join_node: Some(join_node),
                reason,
            },
            DagNode::Escalate { reason } => Self {
                node_type: DagNodeTypeDto::Escalate,
                question: None,
                conditions: Vec::new(),
                parallel_nodes: Vec::new(),
                join_node: None,
                reason,
            },
        }
    }
}

impl TryFrom<DagDefinitionDto> for DagDefinition {
    type Error = DagValidationError;

    fn try_from(dto: DagDefinitionDto) -> Result<Self, Self::Error> {
        let mut nodes = IndexMap::with_capacity(dto.nodes.len());
        for (k, v) in dto.nodes {
            let node = convert_node_dto(&k, v)?;
            nodes.insert(k, node);
        }
        Ok(Self {
            dag_id: dto.dag_id,
            timeout_ms: dto.timeout_ms,
            entry_node: dto.entry_node,
            nodes,
        })
    }
}

impl From<DagDefinition> for DagDefinitionDto {
    fn from(dag: DagDefinition) -> Self {
        let mut nodes = IndexMap::with_capacity(dag.nodes.len());
        for (k, v) in dag.nodes {
            nodes.insert(k, v.into());
        }
        Self {
            dag_id: dag.dag_id,
            timeout_ms: dag.timeout_ms,
            entry_node: dag.entry_node,
            nodes,
        }
    }
}

impl From<StepExecutionResult> for DagStepResultDto {
    fn from(step: StepExecutionResult) -> Self {
        Self {
            node_id: step.node_id,
            node_type: step.node_type.into(),
            answer: step.answer,
            duration_ms: step.duration_ms,
            next_node: step.next_node,
            reason: step.reason,
        }
    }
}

impl From<DagExecutionResult> for DagResponse {
    fn from(res: DagExecutionResult) -> Self {
        let status = if res.escalated {
            "escalated".to_string()
        } else {
            "completed".to_string()
        };

        let final_node = res.execution_path.last().cloned();
        let steps_executed = res.steps.len();

        let mut steps = IndexMap::with_capacity(res.steps.len());
        for (k, v) in res.steps {
            steps.insert(k, v.into());
        }

        let escalation = if res.escalated {
            Some(DagEscalationDetailDto {
                reason: res.escalation_reason,
                node_id: res.escalation_node_id,
                prompt: res.escalation_prompt,
                gating: res.escalation_gating,
            })
        } else {
            None
        };

        Self {
            dag_id: res.dag_id,
            status,
            final_node,
            final_decision: res.final_answer,
            execution_path: res.execution_path,
            steps,
            escalation,
            usage: DagUsageDto {
                total_duration_ms: res.total_duration_ms,
                steps_executed,
                completion_tokens: 0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_node_dto_inference_missing_question_fails() {
        let dto = DagNodeDto {
            node_type: DagNodeTypeDto::Inference,
            question: None,
            conditions: vec![],
            parallel_nodes: vec![],
            join_node: None,
            reason: None,
        };
        let res = DagNode::try_from(dto);
        assert!(matches!(
            res,
            Err(DagValidationError::InvalidNodeSpec { .. })
        ));
    }

    #[test]
    fn test_convert_node_dto_parallel_missing_join_node_fails() {
        let dto = DagNodeDto {
            node_type: DagNodeTypeDto::Parallel,
            question: None,
            conditions: vec![],
            parallel_nodes: vec!["task1".to_string()],
            join_node: None,
            reason: None,
        };
        let res = DagNode::try_from(dto);
        assert!(matches!(
            res,
            Err(DagValidationError::InvalidNodeSpec { .. })
        ));
    }

    #[test]
    fn test_convert_node_dto_parallel_with_conditions_fails() {
        let dto = DagNodeDto {
            node_type: DagNodeTypeDto::Parallel,
            question: None,
            conditions: vec![NodeConditionDto {
                field: "confidence".to_string(),
                op: NodeConditionOpDto::ConfidenceGte,
                value: Some(serde_json::json!(0.5)),
                target_node: "join".to_string(),
            }],
            parallel_nodes: vec!["task1".to_string()],
            join_node: Some("join".to_string()),
            reason: None,
        };
        let res = DagNode::try_from(dto);
        assert!(matches!(
            res,
            Err(DagValidationError::InvalidNodeSpec { .. })
        ));
    }

    #[test]
    fn test_convert_node_dto_escalate_with_conditions_fails() {
        let dto = DagNodeDto {
            node_type: DagNodeTypeDto::Escalate,
            question: None,
            conditions: vec![NodeConditionDto {
                field: "confidence".to_string(),
                op: NodeConditionOpDto::ConfidenceGte,
                value: Some(serde_json::json!(0.5)),
                target_node: "root".to_string(),
            }],
            parallel_nodes: vec![],
            join_node: None,
            reason: Some("終端".to_string()),
        };
        let res = DagNode::try_from(dto);
        assert!(matches!(
            res,
            Err(DagValidationError::InvalidNodeSpec { .. })
        ));
    }
}
