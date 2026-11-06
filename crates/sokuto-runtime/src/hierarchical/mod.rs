//! # 粗密二段階階層ルーティング (Coarse-to-Fine Hierarchical Routing)
//!
//! 候補数が膨大な Choice 型質問 ($K > 20 \sim 30$) に対し、
//! 同一言語モデル重みをメモリ上で共有しながら、
//! 第 1 パス大分類 (Coarse: 4〜8 クラスタ) と第 2 パス細分類 (Fine: 8〜24 候補) を
//! 二段階で推論する高速・高精度ルーティング機構を提供する。
//!
//! ## 主な特徴
//! - **Soft-Beam ルーティング**: 第 1 パスの境界領域 ($M_{\text{coarse}} < \tau_{\text{beam}}$) で Top-1 と Top-2 クラスタを動的マージし、カスケードエラーを阻止。
//! - **階層 Early-Exit**: 確信度極小 ($S_{\text{coarse}} < \tau_{\text{escalate}}$) 時に第 2 パスを即座にスキップして System 2 へ委譲。
//! - **大分類エントロピー連動温度調整**: $T_{\text{fine}} = 1.0 + \gamma \tilde{H}(\mathbf{P}_C)$ により細分類の過信 Softmax を平滑化。
//! - **全候補空間 Gating 再計算**: 全 $K$ 候補空間に確率を復元し、適正な正規化エントロピーと Top-Margin を算出。

pub mod calibration;
pub mod mapping;
pub mod router;

pub use calibration::{
    DEFAULT_COARSE_POWER_ALPHA, DEFAULT_ENTROPY_TEMP_GAMMA, DirichletCalibrator,
    combine_and_reconstruct_probabilities, compute_fine_temperature, compute_normalized_entropy,
    softmax_f64, temperature_scaled_softmax,
};
pub use mapping::{HierarchicalMapping, HierarchicalMappingBuilder};
pub use router::{
    CoarseToFineRouter, DEFAULT_BEAM_MARGIN_THRESHOLD, DEFAULT_CATCHALL_THRESHOLD,
    DEFAULT_ESCALATE_CONFIDENCE_THRESHOLD, HierarchicalEngine, HierarchicalExecutionTrace,
    HierarchicalResult, HierarchicalRouterConfig,
};
