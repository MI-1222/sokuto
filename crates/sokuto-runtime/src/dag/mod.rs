//! # インプロセス DAG 実行器モジュール
//!
//! 非自己回帰型モデルの単一パスでは解決が難しい多段階の前提可変ルール、
//! 例外条項の優先度判定、および複合ワークフローをミリ秒オーダーの超低遅延で
//! 決定論的にオーケストレーションする基盤。

pub mod executor;
pub mod interpolate;
pub mod pool;
pub mod schema;
pub mod validator;

pub use executor::{
    DagError, InProcessDagExecutor, evaluate_compiled_conditions, evaluate_conditions,
};
pub use interpolate::{NodeContextValue, interpolate_prompt, interpolate_question};
pub use pool::{LowLatencyResourcePool, PoolError, PooledResource};
pub use schema::{
    ComparisonOp, CompiledAnswerPredicate, CompiledCondition, CompiledDag, CompiledDagNode,
    ConditionField, ConditionPredicate, ContextProp, DagDefinition, DagExecutionResult, DagNode,
    DagNodeType, EqualityOp, NodeCondition, NodeConditionOp, StepExecutionResult,
    parse_condition_f64,
};
pub use validator::{DagValidationError, compile_dag, validate_dag};
