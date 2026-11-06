//! # 共通サービス層モジュール
//!
//! 各 HTTP エンドポイントから利用されるビジネスロジックおよび推論実行パイプラインを集約する。

pub mod inference;

pub use inference::{InferenceOutput, InferenceService};
