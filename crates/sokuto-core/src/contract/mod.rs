//! # 成果物引き渡し契約(Contract Boundary)モジュール
//!
//! Python 学習側と Rust ランタイム推論側が合意する入出力テンソル仕様および、
//! キャリブレーション設定(`calibration.json`)のデータ構造を提供する。

pub mod calibration;
pub mod model_spec;

pub use calibration::{BucketSpec, CalibrationConfig, TemperatureMap};
pub use model_spec::*;
