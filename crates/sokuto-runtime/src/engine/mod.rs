//! # 推論エンジンモジュール
//!
//! ONNX Runtime を用いた低遅延フォワードパス、セッション管理、
//! および Execution Provider 解決機構を提供する。

pub mod batch;
pub mod cache;
pub mod coarse;
pub mod config;
pub mod escalation;
pub mod gating;
pub mod provider;
pub mod session;

pub use batch::{BatchInferenceOptions, BatchScratchpad, DEFAULT_MAX_BATCH_CHUNK_SIZE};
pub use cache::{
    AlignedBuffer64, CacheCompressionMode, CachedStateFeatures, CachedTensorPayload,
    DEFAULT_DUAL_MAX_BATCH_CHUNK_SIZE, DualSessionEngine, StateCacheManager, is_aligned_64,
};
pub use coarse::{
    CandidateEmbeddingCache, CoarseScorer, CoarseToFineConfig, DEFAULT_COARSE_THRESHOLD,
    DEFAULT_NEGATIVE_KEYS, DEFAULT_NEGATIVE_PHRASES, DEFAULT_TOP_M_CANDIDATES,
    EmbeddingCoarseScorer, FilteredCandidates, LexicalCoarseScorer, filter_top_candidates,
    is_negative_candidate, is_negative_candidate_with_phrases, reconstruct_and_gate_answer,
    reconstruct_choice_answer, reconstruct_probabilities,
};
pub use config::{ExecutionProvider, OptimizationLevel, SessionConfig};
pub use escalation::{
    EscalationPromptBuilder, EscalationTemplateConfig, build_rich_escalation_prompt,
};
pub use gating::{
    apply_gating_to_answer, apply_gating_to_answer_with_energy, apply_gating_to_answer_with_logits,
    apply_gating_to_answers, apply_gating_to_answers_with_logits, calculate_choice_energy,
    evaluate_and_gate_answer, evaluate_and_gate_answer_with_buf, resolve_gating_config,
};
pub use provider::register_execution_providers;
pub use session::{InferenceEngine, validate_op_indices};
