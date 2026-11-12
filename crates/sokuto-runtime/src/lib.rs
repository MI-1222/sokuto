//! # sokuto-runtime
//!
//! 高速トークナイズおよび ONNX Runtime による超低遅延推論パイプラインを提供するクレート。
//!
//! ## 概要
//! - Hugging Face `tokenizers` を用いたトークナイズ処理。
//! - ONNX Runtime (`ort`) を用いた単一フォワードパス決定ヘッド推論。
//! - 候補数が多い場合の粗密探索(Coarse-to-Fine)やバッチ推論。

pub use sokuto_core as core;

pub mod dag;
pub mod engine;
pub mod error;
pub mod hierarchical;
pub mod tokenizer;

pub use dag::{
    DagDefinition, DagError, DagExecutionResult, DagNode, DagNodeType, InProcessDagExecutor,
    LowLatencyResourcePool, LowLatencySessionPool, NodeCondition, NodeConditionOp, PooledResource,
    PooledSession, StepExecutionResult, validate_dag,
};

pub use engine::{
    AlignedBuffer64, BatchInferenceOptions, BatchScratchpad, CacheCompressionMode,
    CachedStateFeatures, CachedTensorPayload, CandidateEmbeddingCache, CoarseScorer,
    CoarseToFineConfig, DEFAULT_COARSE_THRESHOLD, DEFAULT_DUAL_MAX_BATCH_CHUNK_SIZE,
    DEFAULT_MAX_BATCH_CHUNK_SIZE, DEFAULT_NEGATIVE_KEYS, DEFAULT_TOP_M_CANDIDATES,
    DualSessionEngine, EmbeddingCoarseScorer, ExecutionProvider, FilteredCandidates,
    InferenceEngine, LexicalCoarseScorer, OptimizationLevel, SessionConfig, StateCacheManager,
    filter_top_candidates, is_aligned_64, is_negative_candidate, reconstruct_probabilities,
};
pub use error::{Result, RuntimeError};
pub use hierarchical::{
    CoarseToFineRouter, DEFAULT_BEAM_MARGIN_THRESHOLD, DEFAULT_CATCHALL_THRESHOLD,
    DEFAULT_COARSE_POWER_ALPHA, DEFAULT_ENTROPY_TEMP_GAMMA, DEFAULT_ESCALATE_CONFIDENCE_THRESHOLD,
    DirichletCalibrator, HierarchicalEngine, HierarchicalExecutionTrace, HierarchicalMapping,
    HierarchicalMappingBuilder, HierarchicalPredictor, HierarchicalResult,
    HierarchicalRouterConfig, combine_and_reconstruct_probabilities, compute_fine_temperature,
    compute_normalized_entropy, temperature_scaled_softmax,
};
pub use tokenizer::{BatchTokenizedQuestions, JevTokenizer, TokenizedQuestion};

/// ランタイムの初期化確認用関数。
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(version(), env!("CARGO_PKG_VERSION"));
    }
}
