//! # 宣言的 DAG 定義スキーマモジュール
//!
//! 非自己回帰型判断エンジンにおける複合ルールや反事実判定を有向非巡回グラフとして
//! 表現するためのデータ構造、ノード種別、および条件判定規則を定義する。

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use sokuto_core::gating::DecisionRoute;
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

impl DagDefinition {
    /// DAG 定義を静的検証し、実行用プラン `CompiledDag` へ事前コンパイルする。
    pub fn compile(&self) -> Result<CompiledDag, crate::dag::validator::DagValidationError> {
        crate::dag::validator::compile_dag(self)
    }
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
///
/// 代数的データ型 (タグ付き enum) として表現され、ノード種別ごとに必要なフィールドのみを保持する。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "node_type", rename_all = "snake_case")]
pub enum DagNode {
    /// 単一質問のモデル推論を実行するノード。
    Inference {
        /// 推論で実行する質問仕様。
        question: Question,
        /// ノード完了後の条件付き遷移規則リスト。先頭から順に評価される。
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        conditions: Vec<NodeCondition>,
        /// 分岐または推論の理由説明文。
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// 先行ノードの判定値・確信度のみに基づいて遷移先を決定する純粋な条件分岐ノード。
    Branch {
        /// ノード完了後の条件付き遷移規則リスト。先頭から順に評価される。
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        conditions: Vec<NodeCondition>,
        /// 分岐の理由説明文。
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// 独立した複数の質問を並行実行し合流するノード。
    Parallel {
        /// 並行実行対象となるノード ID 一覧。
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        parallel_nodes: Vec<String>,
        /// 並行実行完了後の合流先ノード ID。
        join_node: String,
        /// 並行実行の理由説明文。
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    /// 確信度不足や例外検知時に上位 LLM または人手へ委ねるエスカレーションノード。
    Escalate {
        /// エスカレーションの理由説明文。
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
}

impl DagNode {
    /// ノードの種別を返す。
    pub fn node_type(&self) -> DagNodeType {
        match self {
            Self::Inference { .. } => DagNodeType::Inference,
            Self::Branch { .. } => DagNodeType::Branch,
            Self::Parallel { .. } => DagNodeType::Parallel,
            Self::Escalate { .. } => DagNodeType::Escalate,
        }
    }

    /// 理由説明文を返す。
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Inference { reason, .. }
            | Self::Branch { reason, .. }
            | Self::Parallel { reason, .. }
            | Self::Escalate { reason, .. } => reason.as_deref(),
        }
    }

    /// 条件付き遷移規則リストへの参照を返す (Inference または Branch の場合)。
    pub fn conditions(&self) -> &[NodeCondition] {
        match self {
            Self::Inference { conditions, .. } | Self::Branch { conditions, .. } => conditions,
            Self::Parallel { .. } | Self::Escalate { .. } => &[],
        }
    }

    /// 推論ノードの質問定義への参照を返す (Inference の場合)。
    pub fn question(&self) -> Option<&Question> {
        match self {
            Self::Inference { question, .. } => Some(question),
            _ => None,
        }
    }

    /// 推論ノード (`Inference`) を構築するヘルパー。
    pub fn inference(question: Question, conditions: Vec<NodeCondition>) -> Self {
        Self::Inference {
            question,
            conditions,
            reason: None,
        }
    }

    /// 条件分岐ノード (`Branch`) を構築するヘルパー。
    pub fn branch(conditions: Vec<NodeCondition>) -> Self {
        Self::Branch {
            conditions,
            reason: None,
        }
    }

    /// 並行実行ノード (`Parallel`) を構築するヘルパー。
    pub fn parallel(parallel_nodes: Vec<String>, join_node: impl Into<String>) -> Self {
        Self::Parallel {
            parallel_nodes,
            join_node: join_node.into(),
            reason: None,
        }
    }

    /// エスカレーションノード (`Escalate`) を構築するヘルパー。
    pub fn escalate(reason: Option<impl Into<String>>) -> Self {
        Self::Escalate {
            reason: reason.map(Into::into),
        }
    }
}

/// 条件付き遷移規則。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

    /// コンテキスト値プロパティ比較用のスカラー値 (文字列、真偽値、有限数値) を文字列として抽出する。
    pub fn context_value_as_string(&self) -> Option<String> {
        match self.value.as_ref()? {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Bool(b) => Some(b.to_string()),
            serde_json::Value::Number(n) => {
                if n.as_f64().is_some_and(|f| f.is_finite()) {
                    Some(n.to_string())
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// ルート比較用の有効な識別子 ("auto_execute", "confirm_or_escalate", "fallback") を抽出する。
    pub fn route_value_as_str(&self) -> Option<&str> {
        let s = self.value.as_ref()?.as_str()?;
        match s {
            "auto_execute" | "confirm_or_escalate" | "fallback" => Some(s),
            _ => None,
        }
    }

    /// ルート比較用の有効な識別子を抽出し、型安全な `DecisionRoute` に変換する。
    pub fn route_value_as_decision_route(&self) -> Option<DecisionRoute> {
        let s = self.value.as_ref()?.as_str()?;
        match s {
            "auto_execute" => Some(DecisionRoute::AutoExecute),
            "confirm_or_escalate" => Some(DecisionRoute::ConfirmOrEscalate),
            "fallback" => Some(DecisionRoute::Fallback),
            _ => None,
        }
    }

    /// 自身を事前パース・型解決済みの `CompiledCondition` へ変換する。
    pub fn compile(&self) -> Option<CompiledCondition> {
        CompiledCondition::from_node_condition(self)
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
    /// 2 つの有限浮動小数点数を演算子のセマンティクスに従って許容誤差 (`1e-6`) 付きで比較する。
    ///
    /// 数値比較演算子 (`Eq`, `Neq`, `Gt`, `Gte`, `Lt`, `Lte`) および確信度エイリアス (`ConfidenceGte`, `ConfidenceLt`) を
    /// `to_comparison()` 経由で評価対象とし、集合演算子 (`In`) 等の数値比較不能な演算子に対しては `false` を返す。
    ///
    /// # 引数
    /// - `lhs`: 左辺値 (評価対象の実測値)。
    /// - `rhs`: 右辺値 (条件の閾値)。
    #[inline]
    pub fn eval_f64(&self, lhs: f64, rhs: f64) -> bool {
        self.to_comparison().is_some_and(|op| op.eval_f64(lhs, rhs))
    }

    /// 確信度評価用の演算子を正規化する。
    ///
    /// `ConfidenceGte` は `Gte` に、`ConfidenceLt` は `Lt` に統一する。
    /// `In` 演算子は確信度比較に対して無効であるため `None` を返す。
    #[inline]
    pub fn normalize_confidence(self) -> Option<Self> {
        match self {
            Self::ConfidenceGte | Self::Gte => Some(Self::Gte),
            Self::ConfidenceLt | Self::Lt => Some(Self::Lt),
            Self::In => None,
            other => Some(other),
        }
    }

    /// 等価専用演算子への変換を試みる。
    ///
    /// `Eq` または `Neq` の場合は `Some` を返し、大小比較や集合演算子等の無効な組み合わせは `None` を返す。
    #[inline]
    pub fn to_equality(self) -> Option<EqualityOp> {
        match self {
            Self::Eq => Some(EqualityOp::Eq),
            Self::Neq => Some(EqualityOp::Neq),
            _ => None,
        }
    }

    /// 数値・順序比較演算子への変換を試みる。
    ///
    /// 確信度エイリアス (`ConfidenceGte`, `ConfidenceLt`) は正規化され、
    /// 集合演算子 (`In`) は数値比較に不適合なため `None` を返す。
    #[inline]
    pub fn to_comparison(self) -> Option<ComparisonOp> {
        match self {
            Self::Eq => Some(ComparisonOp::Eq),
            Self::Neq => Some(ComparisonOp::Neq),
            Self::Gt => Some(ComparisonOp::Gt),
            Self::Gte | Self::ConfidenceGte => Some(ComparisonOp::Gte),
            Self::Lt | Self::ConfidenceLt => Some(ComparisonOp::Lt),
            Self::Lte => Some(ComparisonOp::Lte),
            Self::In => None,
        }
    }
}

/// 等価・不等価比較専用の演算子。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EqualityOp {
    /// 等価 (`==`)。
    Eq,
    /// 不等価 (`!=`)。
    Neq,
}

impl EqualityOp {
    /// 2 つの等価比較可能な値が条件を満たすかを判定する。
    #[inline]
    pub fn matches<T: PartialEq + ?Sized>(self, lhs: &T, rhs: &T) -> bool {
        match self {
            Self::Eq => lhs == rhs,
            Self::Neq => lhs != rhs,
        }
    }

    /// 2 つの有限浮動小数点数を許容誤差 (`1e-6`) 付きで等価比較する。
    #[inline]
    pub fn eval_f64(self, lhs: f64, rhs: f64) -> bool {
        ComparisonOp::from(self).eval_f64(lhs, rhs)
    }
}

impl From<EqualityOp> for ComparisonOp {
    #[inline]
    fn from(op: EqualityOp) -> Self {
        match op {
            EqualityOp::Eq => Self::Eq,
            EqualityOp::Neq => Self::Neq,
        }
    }
}

/// 数値・順序比較演算子。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonOp {
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
}

impl ComparisonOp {
    /// 浮動小数点数比較で用いる許容誤差イプシロン。
    pub const FLOAT_EPSILON: f64 = 1e-6;

    /// 2 つの有限浮動小数点数を演算子のセマンティクスに従って許容誤差 (`1e-6`) 付きで比較する。
    #[inline]
    pub fn eval_f64(self, lhs: f64, rhs: f64) -> bool {
        match self {
            Self::Eq => (lhs - rhs).abs() < Self::FLOAT_EPSILON,
            Self::Neq => (lhs - rhs).abs() >= Self::FLOAT_EPSILON,
            Self::Gt => lhs > rhs + Self::FLOAT_EPSILON,
            Self::Gte => lhs >= rhs - Self::FLOAT_EPSILON,
            Self::Lt => lhs < rhs - Self::FLOAT_EPSILON,
            Self::Lte => lhs <= rhs + Self::FLOAT_EPSILON,
        }
    }
}

/// コンテキストノード参照 (`node_id.prop`) で参照可能なプロパティ種別。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextProp {
    /// 先行ノードの回答値 (`value` / `answer`)。
    Value,
    /// 先行ノードの確信度 (`confidence`)。
    Confidence,
}

/// `NodeCondition.field` 文字列をパースした型安全な評価対象フィールド。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionField<'a> {
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
    pub fn parse(field: &'a str) -> Self {
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

/// コンパイル・型解決済みの回答値述語。
#[derive(Debug, Clone, PartialEq)]
pub enum CompiledAnswerPredicate {
    /// Noul または汎用真偽値判定 (実効確率 >= 0.5 の真偽判定)。
    Bool { op: EqualityOp, expected: bool },
    /// 数値閾値判定 (Score または Noul 連続確率)。
    Numeric { op: ComparisonOp, threshold: f64 },
    /// 文字列一致判定 (Choice) または文字列/真偽値/数値文字列の等価判定。
    String {
        op: EqualityOp,
        expected: String,
        /// もし数値としても解釈可能なら事前パースした有限 f64。
        as_f64: Option<f64>,
        /// もし "true" / "false" 文字列なら事前パースした真偽値。
        as_bool: Option<bool>,
    },
    /// 選択肢集合包含判定 (Choice `in`)。
    In { candidates: Vec<String> },
}

impl CompiledAnswerPredicate {
    /// 回答結果 (`Answer`) に対する適合判定を行う。
    pub fn matches(&self, ans: &Answer) -> bool {
        match self {
            Self::Bool { op, expected } => {
                if let Some(noul_val) = ans.as_noul() {
                    let is_true = noul_val >= 0.5;
                    op.matches(&is_true, expected)
                } else {
                    false
                }
            }
            Self::Numeric { op, threshold } => {
                if let Some(score_val) = ans.as_score() {
                    op.eval_f64(score_val, *threshold)
                } else if let Some(noul_val) = ans.as_noul() {
                    op.eval_f64(noul_val, *threshold)
                } else {
                    false
                }
            }
            Self::String {
                op,
                expected,
                as_f64,
                as_bool,
            } => {
                if let Some(noul_val) = ans.as_noul() {
                    if let Some(tb) = as_bool {
                        let is_true = noul_val >= 0.5;
                        return op.matches(&is_true, tb);
                    }
                    if let Some(num) = as_f64 {
                        return op.eval_f64(noul_val, *num);
                    }
                    return false;
                }
                if let Some(choice_val) = ans.as_choice() {
                    return op.matches(choice_val, expected);
                }
                if let Some(score_val) = ans.as_score() {
                    if let Some(num) = as_f64 {
                        return op.eval_f64(score_val, *num);
                    }
                    return false;
                }
                false
            }
            Self::In { candidates } => {
                if let Some(choice_val) = ans.as_choice() {
                    candidates.iter().any(|c| c == choice_val)
                } else {
                    false
                }
            }
        }
    }

    /// 生の `NodeCondition` から `CompiledAnswerPredicate` をパース・型解決して構築する。
    ///
    /// 構文不正またはサポートされない演算子・値の組み合わせの場合は `None` を返す。
    pub fn from_node_condition(cond: &NodeCondition) -> Option<Self> {
        if cond.op == NodeConditionOp::In {
            let arr = cond.value.as_ref()?.as_array()?;
            if arr.is_empty() {
                return None;
            }
            let mut candidates = Vec::with_capacity(arr.len());
            for v in arr {
                candidates.push(v.as_str()?.to_string());
            }
            Some(Self::In { candidates })
        } else if let Some(b) = cond.value.as_ref().and_then(|v| v.as_bool()) {
            let op = cond.op.to_equality()?;
            Some(Self::Bool { op, expected: b })
        } else if let Some(comp_op) = cond.op.to_comparison().filter(|o| {
            matches!(
                o,
                ComparisonOp::Gt | ComparisonOp::Gte | ComparisonOp::Lt | ComparisonOp::Lte
            )
        }) {
            // 大小比較の場合は数値として解釈 (数値リテラルまたは数値文字列)
            let threshold = cond.value_as_f64()?;
            Some(Self::Numeric {
                op: comp_op,
                threshold,
            })
        } else if let Some(n) = cond.value.as_ref().and_then(|v| v.as_f64()) {
            if !n.is_finite() {
                return None;
            }
            let op = cond.op.to_comparison()?;
            Some(Self::Numeric { op, threshold: n })
        } else if let Some(s) = cond.value.as_ref().and_then(|v| v.as_str()) {
            let op = cond.op.to_equality()?;
            let as_bool = match s {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            };
            let as_f64 = s.trim().parse::<f64>().ok().filter(|f| f.is_finite());
            Some(Self::String {
                op,
                expected: s.to_string(),
                as_f64,
                as_bool,
            })
        } else {
            None
        }
    }
}

/// コンパイル・型解決済みの条件判定述語。
#[derive(Debug, Clone, PartialEq)]
pub enum ConditionPredicate {
    /// 実効確信度 (`confidence`) に対する数値閾値比較。
    Confidence { op: ComparisonOp, threshold: f64 },
    /// ゲーティングルート (`route`) に対する等価判定。
    Route {
        op: EqualityOp,
        expected: DecisionRoute,
    },
    /// 先行ノードの回答値コンテキスト (`<node_id>.value` / `<node_id>.answer`) に対する判定。
    ContextValue {
        node_id: String,
        op: ComparisonOp,
        expected: String,
        /// 条件値が有限数値として解釈可能な場合の事前パース値。
        as_f64: Option<f64>,
        /// 条件値が真偽値として解釈可能な場合の事前パース値。
        as_bool: Option<bool>,
    },
    /// 先行ノードの確信度コンテキスト (`<node_id>.confidence`) に対する数値閾値比較。
    ContextConfidence {
        node_id: String,
        op: ComparisonOp,
        threshold: f64,
    },
    /// 現在のノードの判定結果 (`answer` / `value`) に対する型付き判定。
    Answer(CompiledAnswerPredicate),
}

impl ConditionPredicate {
    /// 生の `NodeCondition` から `ConditionPredicate` をパース・型解決して構築する。
    ///
    /// 構文不正またはサポートされない演算子・値の組み合わせの場合は `None` を返す。
    pub fn from_node_condition(cond: &NodeCondition) -> Option<Self> {
        match ConditionField::parse(&cond.field) {
            ConditionField::Confidence => {
                let threshold = cond.value_as_f64()?;
                let op = cond.op.to_comparison()?;
                Some(Self::Confidence { op, threshold })
            }
            ConditionField::Route => {
                let op = cond.op.to_equality()?;
                let expected = cond.route_value_as_decision_route()?;
                Some(Self::Route { op, expected })
            }
            ConditionField::ContextProperty { node_id, prop } => match prop {
                ContextProp::Confidence => {
                    let threshold = cond.value_as_f64()?;
                    let op = cond.op.to_comparison()?;
                    Some(Self::ContextConfidence {
                        node_id: node_id.to_string(),
                        op,
                        threshold,
                    })
                }
                ContextProp::Value => {
                    let op = cond.op.to_comparison()?;
                    let expected = cond.context_value_as_string()?;
                    let as_f64 = cond.value_as_f64();
                    let as_bool = cond.value.as_ref().and_then(|v| v.as_bool()).or_else(|| {
                        match expected.trim() {
                            "true" => Some(true),
                            "false" => Some(false),
                            _ => None,
                        }
                    });
                    if matches!(
                        op,
                        ComparisonOp::Gt | ComparisonOp::Gte | ComparisonOp::Lt | ComparisonOp::Lte
                    ) && as_f64.is_none()
                    {
                        return None;
                    }
                    Some(Self::ContextValue {
                        node_id: node_id.to_string(),
                        op,
                        expected,
                        as_f64,
                        as_bool,
                    })
                }
            },
            ConditionField::Answer => {
                let ans_pred = CompiledAnswerPredicate::from_node_condition(cond)?;
                Some(Self::Answer(ans_pred))
            }
            ConditionField::Unknown => None,
        }
    }
}

/// コンパイル・型解決済みのノード遷移条件。
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledCondition {
    /// 事前解決された述語。
    pub predicate: ConditionPredicate,
    /// 条件合致時の遷移先ノード ID。
    pub target_node: String,
}

impl CompiledCondition {
    /// 単一条件の適合判定を行う。
    #[inline]
    pub fn matches(
        &self,
        answer: Option<&Answer>,
        context: &IndexMap<String, crate::dag::interpolate::NodeContextValue>,
    ) -> bool {
        match &self.predicate {
            ConditionPredicate::Confidence { op, threshold } => {
                let conf = answer.map(Answer::effective_confidence).unwrap_or(0.0);
                op.eval_f64(conf, *threshold)
            }
            ConditionPredicate::Route { op, expected } => {
                let Some(gating) = answer.and_then(|a| a.gating.as_ref()) else {
                    return false;
                };
                op.matches(&gating.route, expected)
            }
            ConditionPredicate::ContextValue {
                node_id,
                op,
                expected,
                as_f64,
                as_bool,
            } => {
                let Some(ctx_val) = context.get(node_id) else {
                    return false;
                };

                // 1. 数値比較: 条件値が数値として解釈でき、かつコンテキスト値も数値として解釈できる場合
                let ctx_f64 = ctx_val.numeric_value.or_else(|| {
                    ctx_val
                        .value
                        .trim()
                        .parse::<f64>()
                        .ok()
                        .filter(|f| f.is_finite())
                });
                if let (Some(num_cond), Some(num_ctx)) = (as_f64, ctx_f64) {
                    return op.eval_f64(num_ctx, *num_cond);
                }

                // 2. 大小比較演算子の場合は、両辺が数値でない限り不一致
                if matches!(
                    op,
                    ComparisonOp::Gt | ComparisonOp::Gte | ComparisonOp::Lt | ComparisonOp::Lte
                ) {
                    return false;
                }

                // 3. 真偽値比較: 条件値が真偽値として解釈でき、かつコンテキスト値も真偽値の場合
                if let Some(expected_b) = as_bool {
                    let ctx_b = match ctx_val.value.as_str() {
                        "true" => Some(true),
                        "false" => Some(false),
                        _ => None,
                    };
                    if let Some(ctx_b) = ctx_b {
                        return match op {
                            ComparisonOp::Eq => ctx_b == *expected_b,
                            ComparisonOp::Neq => ctx_b != *expected_b,
                            _ => false,
                        };
                    }
                }

                // 4. 文字列等価比較
                match op {
                    ComparisonOp::Eq => &ctx_val.value == expected,
                    ComparisonOp::Neq => &ctx_val.value != expected,
                    _ => false,
                }
            }
            ConditionPredicate::ContextConfidence {
                node_id,
                op,
                threshold,
            } => context
                .get(node_id)
                .is_some_and(|ctx_val| op.eval_f64(ctx_val.confidence, *threshold)),
            ConditionPredicate::Answer(ans_pred) => answer.is_some_and(|ans| ans_pred.matches(ans)),
        }
    }

    /// 生の `NodeCondition` から `CompiledCondition` を構築する。
    ///
    /// 構文不正またはサポートされない演算子・値の組み合わせの場合は `None` を返す。
    pub fn from_node_condition(cond: &NodeCondition) -> Option<Self> {
        let predicate = ConditionPredicate::from_node_condition(cond)?;
        Some(Self {
            predicate,
            target_node: cond.target_node.clone(),
        })
    }
}

/// コンパイル・検証済みの実行用 DAG ノード。
#[derive(Debug, Clone, PartialEq)]
pub enum CompiledDagNode {
    /// 単一質問のモデル推論を実行するノード。
    Inference {
        question: Question,
        conditions: Vec<CompiledCondition>,
        reason: Option<String>,
    },
    /// 先行ノードの判定値・確信度のみに基づいて遷移先を決定する純粋な条件分岐ノード。
    Branch {
        conditions: Vec<CompiledCondition>,
        reason: Option<String>,
    },
    /// 独立した複数の質問を並行実行し合流するノード。
    Parallel {
        parallel_nodes: Vec<String>,
        join_node: String,
        reason: Option<String>,
    },
    /// 確信度不足や例外検知時に上位 LLM または人手へ委ねるエスカレーションノード。
    Escalate { reason: Option<String> },
}

impl CompiledDagNode {
    /// ノードの種別を返す。
    pub fn node_type(&self) -> DagNodeType {
        match self {
            Self::Inference { .. } => DagNodeType::Inference,
            Self::Branch { .. } => DagNodeType::Branch,
            Self::Parallel { .. } => DagNodeType::Parallel,
            Self::Escalate { .. } => DagNodeType::Escalate,
        }
    }

    /// 理由説明文を返す。
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Inference { reason, .. }
            | Self::Branch { reason, .. }
            | Self::Parallel { reason, .. }
            | Self::Escalate { reason, .. } => reason.as_deref(),
        }
    }

    /// 事前コンパイル済み条件付き遷移規則リストへの参照を返す。
    pub fn conditions(&self) -> &[CompiledCondition] {
        match self {
            Self::Inference { conditions, .. } | Self::Branch { conditions, .. } => conditions,
            Self::Parallel { .. } | Self::Escalate { .. } => &[],
        }
    }

    /// 推論ノードの質問定義への参照を返す (Inference の場合)。
    pub fn question(&self) -> Option<&Question> {
        match self {
            Self::Inference { question, .. } => Some(question),
            _ => None,
        }
    }
}

/// コンパイル・検証済みの実行用 DAG プラン。
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledDag {
    /// DAG の一意識別子。
    pub dag_id: String,
    /// 全体タイムアウト時間 (ミリ秒)。
    pub timeout_ms: u64,
    /// 実行開始ノードの識別子。
    pub entry_node: String,
    /// コンパイル済みノードマップ。
    pub nodes: IndexMap<String, CompiledDagNode>,
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
        assert_eq!(dag.nodes["step1"].node_type(), DagNodeType::Inference);
        assert_eq!(dag.nodes["escalate"].node_type(), DagNodeType::Escalate);
    }

    #[test]
    fn test_dag_node_deserialize_invalid_spec_fails_at_serde() {
        // inference なのに question が欠落している JSON
        let invalid_inference = r#"{
            "node_type": "inference",
            "conditions": []
        }"#;
        assert!(serde_json::from_str::<DagNode>(invalid_inference).is_err());

        // parallel なのに join_node が欠落している JSON
        let invalid_parallel = r#"{
            "node_type": "parallel",
            "parallel_nodes": ["task1"]
        }"#;
        assert!(serde_json::from_str::<DagNode>(invalid_parallel).is_err());
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
        assert!(NodeConditionOp::ConfidenceGte.eval_f64(1.0, 0.5));
        assert!(!NodeConditionOp::ConfidenceGte.eval_f64(0.4, 0.5));
        assert!(NodeConditionOp::ConfidenceLt.eval_f64(0.2, 0.5));
        assert!(!NodeConditionOp::ConfidenceLt.eval_f64(0.6, 0.5));
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

    #[test]
    fn test_compiled_condition_from_node_condition() {
        let cond_conf = NodeCondition {
            field: "confidence".to_string(),
            op: NodeConditionOp::ConfidenceGte,
            value: Some(serde_json::json!(0.85)),
            target_node: "target_a".to_string(),
        };
        let compiled = cond_conf.compile().expect("コンパイルに成功。");
        assert_eq!(compiled.target_node, "target_a");
        assert_eq!(
            compiled.predicate,
            ConditionPredicate::Confidence {
                op: ComparisonOp::Gte,
                threshold: 0.85,
            }
        );

        let cond_ctx_val = NodeCondition {
            field: "prev_node.value".to_string(),
            op: NodeConditionOp::Eq,
            value: Some(serde_json::json!("transfer")),
            target_node: "target_b".to_string(),
        };
        let compiled_ctx = cond_ctx_val.compile().expect("コンパイルに成功。");
        assert_eq!(
            compiled_ctx.predicate,
            ConditionPredicate::ContextValue {
                node_id: "prev_node".to_string(),
                op: ComparisonOp::Eq,
                expected: "transfer".to_string(),
                as_f64: None,
                as_bool: None,
            }
        );

        let cond_invalid = NodeCondition {
            field: "invalid_field".to_string(),
            op: NodeConditionOp::Eq,
            value: Some(serde_json::json!(1)),
            target_node: "target_c".to_string(),
        };
        assert!(cond_invalid.compile().is_none());

        // ルート条件のコンパイルテスト。
        let cond_route = NodeCondition {
            field: "route".to_string(),
            op: NodeConditionOp::Eq,
            value: Some(serde_json::json!("auto_execute")),
            target_node: "target_route".to_string(),
        };
        let compiled_route = cond_route.compile().expect("コンパイルに成功。");
        assert_eq!(
            compiled_route.predicate,
            ConditionPredicate::Route {
                op: EqualityOp::Eq,
                expected: DecisionRoute::AutoExecute,
            }
        );

        // コンテキスト確信度条件のコンパイルテスト。
        let cond_ctx_conf = NodeCondition {
            field: "upstream_node.confidence".to_string(),
            op: NodeConditionOp::ConfidenceLt,
            value: Some(serde_json::json!(0.5)),
            target_node: "target_conf".to_string(),
        };
        let compiled_ctx_conf = cond_ctx_conf.compile().expect("コンパイルに成功。");
        assert_eq!(
            compiled_ctx_conf.predicate,
            ConditionPredicate::ContextConfidence {
                node_id: "upstream_node".to_string(),
                op: ComparisonOp::Lt,
                threshold: 0.5,
            }
        );

        // 回答 'in' 条件のコンパイルテスト。
        let cond_in = NodeCondition {
            field: "answer".to_string(),
            op: NodeConditionOp::In,
            value: Some(serde_json::json!(["apple", "banana"])),
            target_node: "target_in".to_string(),
        };
        let compiled_in = cond_in.compile().expect("コンパイルに成功。");
        assert_eq!(
            compiled_in.predicate,
            ConditionPredicate::Answer(CompiledAnswerPredicate::In {
                candidates: vec!["apple".to_string(), "banana".to_string()],
            })
        );
    }

    #[test]
    fn test_equality_op() {
        assert!(EqualityOp::Eq.matches("hello", "hello"));
        assert!(!EqualityOp::Eq.matches("hello", "world"));
        assert!(EqualityOp::Neq.matches("hello", "world"));
        assert!(!EqualityOp::Neq.matches("hello", "hello"));

        assert!(EqualityOp::Eq.matches(&DecisionRoute::AutoExecute, &DecisionRoute::AutoExecute));
        assert!(!EqualityOp::Eq.matches(&DecisionRoute::AutoExecute, &DecisionRoute::Fallback));
        assert!(EqualityOp::Neq.matches(&DecisionRoute::AutoExecute, &DecisionRoute::Fallback));

        assert!(EqualityOp::Eq.eval_f64(1.0, 1.0 + 5e-7));
        assert!(!EqualityOp::Eq.eval_f64(1.0, 1.0 + 2e-6));
        assert!(EqualityOp::Neq.eval_f64(1.0, 1.0 + 2e-6));
        assert!(!EqualityOp::Neq.eval_f64(1.0, 1.0 + 5e-7));
    }

    #[test]
    fn test_comparison_op() {
        assert!(ComparisonOp::Eq.eval_f64(1.0, 1.0 + 5e-7));
        assert!(!ComparisonOp::Eq.eval_f64(1.0, 1.0 + 2e-6));
        assert!(ComparisonOp::Neq.eval_f64(1.0, 1.0 + 2e-6));
        assert!(ComparisonOp::Gt.eval_f64(1.0 + 2e-6, 1.0));
        assert!(!ComparisonOp::Gt.eval_f64(1.0, 1.0));
        assert!(ComparisonOp::Gte.eval_f64(1.0 - 5e-7, 1.0));
        assert!(ComparisonOp::Lt.eval_f64(1.0 - 2e-6, 1.0));
        assert!(ComparisonOp::Lte.eval_f64(1.0 + 5e-7, 1.0));
        assert_eq!(ComparisonOp::from(EqualityOp::Eq), ComparisonOp::Eq);
        assert_eq!(ComparisonOp::from(EqualityOp::Neq), ComparisonOp::Neq);
    }
}
