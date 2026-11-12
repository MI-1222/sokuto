//! # 宣言的 DAG 定義スキーマモジュール
//!
//! 非自己回帰型判断エンジンにおける複合ルールや反事実判定を有向非巡回グラフとして
//! 表現するためのデータ構造、ノード種別、および条件判定規則を定義する。

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use sokuto_core::schema::{Answer, Question};

/// 宣言的 DAG 定義構造体。
///
/// 複数の推論ノード、分岐ノード、並行ノード、エスカレーションノードから構成され、
/// 各種ルールの決定論的オーケストレーションを定義する。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagDefinition {
    /// DAG の一意識別子。
    pub dag_id: String,
    /// 全体タイムアウト時間 (ミリ秒)。既定値は 100ms。
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    /// 実行開始ノードの識別子。
    pub entry_node: String,
    /// DAG を構成するノードマップ。
    pub nodes: IndexMap<String, DagNode>,
}

fn default_timeout_ms() -> u64 {
    100
}

/// DAG ノード種別。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DagNodeType {
    /// 単一質問のモデル推論を実行するノード。
    Inference,
    /// 先行ノードの判定値・確信度のみに基づいて遷移先を決定する純粋な条件分岐ノード。
    Branch,
    /// 独立した複数の質問を並行実行し合流するノード。
    Parallel,
    /// 確信度不足や例外検知時に上位 LLM または人手へ委ねるエスカレーションノード。
    Escalate,
}

/// DAG を構成する個々のノード定義。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagNode {
    /// ノードの種別。
    pub node_type: DagNodeType,
    /// 推論ノード (`Inference`) で実行する質問仕様。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question: Option<Question>,
    /// ノード完了後の条件付き遷移規則リスト。先頭から順に評価される。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conditions: Vec<NodeCondition>,
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

/// 条件付き遷移規則。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCondition {
    /// 評価対象フィールド (`"answer"`, `"confidence"`, `"route"` など)。
    pub field: String,
    /// 比較演算子。
    pub op: NodeConditionOp,
    /// 比較対象の値。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<serde_json::Value>,
    /// 条件合致時の遷移先ノード ID。
    pub target_node: String,
}

impl NodeCondition {
    /// 自身の `value` フィールドを有限な `f64` として解釈する。
    pub fn value_as_f64(&self) -> Option<f64> {
        parse_condition_f64(self.value.as_ref())
    }
}

/// 条件値 (`serde_json::Value`) を有限な `f64` として解釈する。
///
/// JSON 数値リテラルに加えて、`"3"` や `"3.0"` のような数値文字列もパースして返却する。
/// 有限な数値でない場合 (`NaN`, `Infinity`, パース失敗など) は `None` を返す。
pub fn parse_condition_f64(value: Option<&serde_json::Value>) -> Option<f64> {
    let val = value?;
    val.as_f64()
        .or_else(|| val.as_str().and_then(|s| s.trim().parse::<f64>().ok()))
        .filter(|n| n.is_finite())
}

/// 条件評価演算子。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeConditionOp {
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

impl NodeConditionOp {
    /// 浮動小数点数比較で用いる許容誤差イプシロン。
    const FLOAT_EPSILON: f64 = 1e-6;

    /// 2 つの有限浮動小数点数を演算子のセマンティクスに従って許容誤差 (`1e-6`) 付きで比較する。
    ///
    /// 数値比較演算子 (`Eq`, `Neq`, `Gt`, `Gte`, `Lt`, `Lte`) のみを評価対象とし、
    /// 集合演算子 (`In`) および確信度専用エイリアス (`ConfidenceGte`, `ConfidenceLt`) に対しては `false` を返す。
    ///
    /// # 引数
    /// - `lhs`: 左辺値 (評価対象の実測値)。
    /// - `rhs`: 右辺値 (条件の閾値)。
    pub fn eval_f64(&self, lhs: f64, rhs: f64) -> bool {
        match self {
            Self::Eq => (lhs - rhs).abs() < Self::FLOAT_EPSILON,
            Self::Neq => (lhs - rhs).abs() >= Self::FLOAT_EPSILON,
            Self::Gt => lhs > rhs + Self::FLOAT_EPSILON,
            Self::Gte => lhs >= rhs - Self::FLOAT_EPSILON,
            Self::Lt => lhs < rhs - Self::FLOAT_EPSILON,
            Self::Lte => lhs <= rhs + Self::FLOAT_EPSILON,
            Self::In | Self::ConfidenceGte | Self::ConfidenceLt => false,
        }
    }
}

/// 単一ステップの実行結果記録。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepExecutionResult {
    /// 実行ノード ID。
    pub node_id: String,
    /// ノード種別。
    pub node_type: DagNodeType,
    /// 推論ノードの場合の判定結果。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub answer: Option<Answer>,
    /// ステップ実行所要時間 (ミリ秒)。
    pub duration_ms: f64,
    /// 次に遷移したノード ID。終端の場合は `None`。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_node: Option<String>,
    /// ノードの理由または補足メッセージ。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// DAG 全体の実行結果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagExecutionResult {
    /// 実行された DAG の識別子。
    pub dag_id: String,
    /// 辿ったノード ID の実行経路。
    pub execution_path: Vec<String>,
    /// 各ステップの詳細実行結果マップ。
    pub steps: IndexMap<String, StepExecutionResult>,
    /// DAG 全体の実行所要時間 (ミリ秒)。
    pub total_duration_ms: f64,
    /// エスカレーションノードに到達して終了したか否か。
    pub escalated: bool,
    /// エスカレーション理由 (該当する場合)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub escalation_reason: Option<String>,
    /// エスカレーションをトリガーした直前ノード ID。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub escalation_node_id: Option<String>,
    /// 上位 LLM (System 2) 向けの診断 Guided CoT プロンプト案。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub escalation_prompt: Option<String>,
    /// エスカレーション要因となった確信度ゲーティング詳細メタデータ。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub escalation_gating: Option<sokuto_core::gating::GatingMetadata>,
    /// 最終決定 Answer (最後の推論ノードの結果)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_answer: Option<Answer>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_dag_definition() {
        let json_data = r#"{
            "dag_id": "test_dag",
            "timeout_ms": 50,
            "entry_node": "step1",
            "nodes": {
                "step1": {
                    "node_type": "inference",
                    "question": {
                        "type": "noul",
                        "instructions": "セール品ですか？"
                    },
                    "conditions": [
                        { "field": "answer", "op": "eq", "value": true, "target_node": "escalate" },
                        { "field": "answer", "op": "eq", "value": false, "target_node": "step2" }
                    ]
                },
                "step2": {
                    "node_type": "branch",
                    "conditions": []
                },
                "escalate": {
                    "node_type": "escalate",
                    "reason": "セール品のため返品不可。"
                }
            }
        }"#;

        let dag: DagDefinition = serde_json::from_str(json_data).expect("パース成功すること。");
        assert_eq!(dag.dag_id, "test_dag");
        assert_eq!(dag.timeout_ms, 50);
        assert_eq!(dag.entry_node, "step1");
        assert_eq!(dag.nodes.len(), 3);
        assert_eq!(dag.nodes["step1"].node_type, DagNodeType::Inference);
        assert_eq!(dag.nodes["escalate"].node_type, DagNodeType::Escalate);
    }

    #[test]
    fn test_node_condition_op_eval_f64() {
        assert!(NodeConditionOp::Eq.eval_f64(1.0, 1.0 + 5e-7));
        assert!(!NodeConditionOp::Eq.eval_f64(1.0, 1.0 + 2e-6));
        assert!(NodeConditionOp::Neq.eval_f64(1.0, 1.0 + 2e-6));
        assert!(NodeConditionOp::Gt.eval_f64(1.0 + 2e-6, 1.0));
        assert!(!NodeConditionOp::Gt.eval_f64(1.0, 1.0));
        assert!(NodeConditionOp::Gte.eval_f64(1.0 - 5e-7, 1.0));
        assert!(NodeConditionOp::Lt.eval_f64(1.0 - 2e-6, 1.0));
        assert!(NodeConditionOp::Lte.eval_f64(1.0 + 5e-7, 1.0));
        assert!(!NodeConditionOp::In.eval_f64(1.0, 1.0));
        assert!(!NodeConditionOp::ConfidenceGte.eval_f64(1.0, 0.5));
        assert!(!NodeConditionOp::ConfidenceLt.eval_f64(0.2, 0.5));
    }

    #[test]
    fn test_parse_condition_f64() {
        let num_val = serde_json::json!(42.5);
        assert_eq!(parse_condition_f64(Some(&num_val)), Some(42.5));

        let str_val = serde_json::json!("  -12.5  ");
        assert_eq!(parse_condition_f64(Some(&str_val)), Some(-12.5));

        let invalid_str = serde_json::json!("not_a_number");
        assert_eq!(parse_condition_f64(Some(&invalid_str)), None);

        let bool_val = serde_json::json!(true);
        assert_eq!(parse_condition_f64(Some(&bool_val)), None);

        assert_eq!(parse_condition_f64(None), None);
    }
}
