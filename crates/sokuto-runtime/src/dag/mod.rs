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

pub use executor::{DagError, InProcessDagExecutor, evaluate_conditions};
pub use interpolate::{NodeContextValue, interpolate_prompt, interpolate_question};
pub use pool::{
    LowLatencyResourcePool, LowLatencySessionPool, PoolError, PooledResource, PooledSession,
};
pub use schema::{
    DagDefinition, DagExecutionResult, DagNode, DagNodeType, NodeCondition, NodeConditionOp,
    StepExecutionResult, parse_condition_f64,
};
pub use validator::{DagValidationError, validate_dag};
