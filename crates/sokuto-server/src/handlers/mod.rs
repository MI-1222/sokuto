//! # エンドポイントハンドラ集約モジュール
//!
//! 推論 API および運用監視 API のハンドラを再エクスポートする。

pub mod dag;
pub mod ops;
pub mod state_batch;
pub mod systemone;

pub use dag::{MAX_DAG_TIMEOUT_MS, dag_handler};
pub use ops::{HealthStatusResponse, health_handler, metrics_handler, ready_handler};
pub use state_batch::state_batch_handler;
pub use systemone::system_one_handler;
