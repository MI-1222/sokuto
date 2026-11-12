//! # アプリケーション共有状態管理モジュール
//!
//! 推論エンジン、トークナイザー、較正設定、およびサーバー運用パラメータをスレッドセーフに保持する。

use std::sync::Arc;

use sokuto_core::contract::calibration::CalibrationConfig;
use sokuto_core::gating::GatingConfig;
use sokuto_runtime::dag::InProcessDagExecutor;
use sokuto_runtime::engine::{
    CacheCompressionMode, CoarseToFineConfig, DualSessionEngine, InferenceEngine, SessionConfig,
    StateCacheManager,
};
use sokuto_runtime::hierarchical::HierarchicalRouterConfig;
use sokuto_runtime::hierarchical::mapping::HierarchicalMapping;
use sokuto_runtime::tokenizer::JevTokenizer;

use crate::guardrails::{GuardrailConfig, GuardrailPipeline};

/// デフォルトのリクエストあたり最大質問数。
pub const DEFAULT_MAX_QUESTIONS_PER_REQUEST: usize = 128;

/// デフォルトのバッチチャンクサイズ。
pub const DEFAULT_CHUNK_SIZE: usize = 16;

/// サーバー全体で共有されるアプリケーション状態。
#[derive(Clone)]
pub struct AppState {
    /// ONNX Runtime 推論エンジン(内部でセッションプールと同期制御を保持)。
    pub engine: Arc<InferenceEngine>,
    /// Jev 高速トークナイザー。
    pub tokenizer: Arc<JevTokenizer>,
    /// 事後較正温度設定。
    pub calib_config: Arc<CalibrationConfig>,
    /// 大規模候補数向けの粗密 2 段階探索設定。
    pub coarse_config: CoarseToFineConfig,
    /// 1 リクエストで許容される最大質問数。
    pub max_questions_per_request: usize,
    /// マイクロバッチ分割時のチャンクサイズ。
    pub chunk_size: usize,
    /// 前処理ガードレールパイプライン。
    pub guardrail_pipeline: GuardrailPipeline,
    /// 確信度ゲーティング処理設定。
    pub gating_config: GatingConfig,
    /// インプロセス DAG 実行器(反事実・マイクロ決定グラフ向け)。
    pub dag_executor: Arc<InProcessDagExecutor>,
    /// 大分類・細分類オントロジー定義(存在する場合)。
    pub hierarchical_mapping: Option<Arc<HierarchicalMapping>>,
    /// 粗密二段階階層ルーティング設定。
    pub hierarchical_config: HierarchicalRouterConfig,
    /// 透過的粗密ルーティングモードの有効フラグ。
    pub auto_hierarchical: bool,
    /// 2 セッション分割型長文推論エンジン (存在する場合)。
    pub dual_engine: Option<Arc<DualSessionEngine>>,
    /// State 中間特徴量キャッシュマネージャー (存在する場合)。
    pub cache_manager: Option<Arc<StateCacheManager>>,
}

impl AppState {
    /// 新規 `AppState` を構築する。
    ///
    /// # 引数
    /// - `engine`: 推論エンジン。
    /// - `tokenizer`: 高速トークナイザー。
    /// - `calib_config`: 較正設定。
    pub fn new(
        engine: Arc<InferenceEngine>,
        tokenizer: Arc<JevTokenizer>,
        calib_config: Arc<CalibrationConfig>,
    ) -> Self {
        let mut guardrail_config = GuardrailConfig::default();
        guardrail_config.limits.max_questions = DEFAULT_MAX_QUESTIONS_PER_REQUEST;
        let guardrail_pipeline = GuardrailPipeline::new(guardrail_config);

        let auto_hierarchical = std::env::var("SOKUTO_AUTO_HIERARCHICAL")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let dag_executor = Arc::new(InProcessDagExecutor::new(
            Arc::clone(&engine),
            Arc::clone(&tokenizer),
            (*calib_config).clone(),
        ));

        Self {
            engine,
            tokenizer,
            calib_config,
            coarse_config: CoarseToFineConfig::default(),
            max_questions_per_request: DEFAULT_MAX_QUESTIONS_PER_REQUEST,
            chunk_size: DEFAULT_CHUNK_SIZE,
            guardrail_pipeline,
            gating_config: GatingConfig {
                enabled: false,
                ..Default::default()
            },
            dag_executor,
            hierarchical_mapping: None,
            hierarchical_config: HierarchicalRouterConfig::default(),
            auto_hierarchical,
            dual_engine: None,
            cache_manager: None,
        }
    }

    /// 設定値をカスタマイズして `AppState` を構築する。
    pub fn with_options(
        engine: Arc<InferenceEngine>,
        tokenizer: Arc<JevTokenizer>,
        calib_config: Arc<CalibrationConfig>,
        coarse_config: CoarseToFineConfig,
        max_questions_per_request: usize,
        chunk_size: usize,
    ) -> Self {
        let mut guardrail_config = GuardrailConfig::default();
        guardrail_config.limits.max_questions = max_questions_per_request;
        let guardrail_pipeline = GuardrailPipeline::new(guardrail_config);

        let auto_hierarchical = std::env::var("SOKUTO_AUTO_HIERARCHICAL")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let dag_executor = Arc::new(InProcessDagExecutor::new(
            Arc::clone(&engine),
            Arc::clone(&tokenizer),
            (*calib_config).clone(),
        ));

        Self {
            engine,
            tokenizer,
            calib_config,
            coarse_config,
            max_questions_per_request,
            chunk_size: chunk_size.max(1),
            guardrail_pipeline,
            gating_config: GatingConfig {
                enabled: false,
                ..Default::default()
            },
            dag_executor,
            hierarchical_mapping: None,
            hierarchical_config: HierarchicalRouterConfig::default(),
            auto_hierarchical,
            dual_engine: None,
            cache_manager: None,
        }
    }

    /// ガードレール設定も含めてフルカスタマイズした `AppState` を構築する。
    pub fn with_guardrails(
        engine: Arc<InferenceEngine>,
        tokenizer: Arc<JevTokenizer>,
        calib_config: Arc<CalibrationConfig>,
        coarse_config: CoarseToFineConfig,
        max_questions_per_request: usize,
        chunk_size: usize,
        guardrail_config: GuardrailConfig,
    ) -> Self {
        let auto_hierarchical = std::env::var("SOKUTO_AUTO_HIERARCHICAL")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let dag_executor = Arc::new(InProcessDagExecutor::new(
            Arc::clone(&engine),
            Arc::clone(&tokenizer),
            (*calib_config).clone(),
        ));

        Self {
            engine,
            tokenizer,
            calib_config,
            coarse_config,
            max_questions_per_request,
            chunk_size: chunk_size.max(1),
            guardrail_pipeline: GuardrailPipeline::new(guardrail_config),
            gating_config: GatingConfig {
                enabled: false,
                ..Default::default()
            },
            dag_executor,
            hierarchical_mapping: None,
            hierarchical_config: HierarchicalRouterConfig::default(),
            auto_hierarchical,
            dual_engine: None,
            cache_manager: None,
        }
    }

    /// デュアルセッション推論エンジンおよびキャッシュマネージャーを設定する。
    pub fn with_dual_engine(
        mut self,
        dual_engine: Arc<DualSessionEngine>,
        cache_manager: Arc<StateCacheManager>,
    ) -> Self {
        self.dual_engine = Some(dual_engine);
        self.cache_manager = Some(cache_manager);
        self
    }

    /// ゲーティング設定をカスタマイズして設定する。
    pub fn with_gating_config(mut self, gating_config: GatingConfig) -> Self {
        self.gating_config = gating_config;
        self
    }

    /// 階層オントロジーマッピングを設定する。
    pub fn with_hierarchical_mapping(mut self, mapping: Arc<HierarchicalMapping>) -> Self {
        self.hierarchical_mapping = Some(mapping);
        self
    }

    /// 粗密階層ルーティング設定をカスタマイズして設定する。
    pub fn with_hierarchical_config(mut self, config: HierarchicalRouterConfig) -> Self {
        self.hierarchical_config = config;
        self
    }

    /// 透過的粗密ルーティングモードの有効化を設定する。
    pub fn with_auto_hierarchical(mut self, enabled: bool) -> Self {
        self.auto_hierarchical = enabled;
        self
    }

    /// カスタム DAG 実行器を設定する。
    pub fn with_dag_executor(mut self, executor: Arc<InProcessDagExecutor>) -> Self {
        self.dag_executor = executor;
        self
    }

    /// 指定ディレクトリ内の `hierarchical_mapping.json` を読み込み、存在すれば登録する。
    pub fn load_hierarchical_mapping_from_dir<P: AsRef<std::path::Path>>(mut self, dir: P) -> Self {
        let mapping_path = dir.as_ref().join("hierarchical_mapping.json");
        if mapping_path.exists()
            && let Ok(content) = std::fs::read_to_string(&mapping_path)
        {
            if let Ok(mapping) = serde_json::from_str::<HierarchicalMapping>(&content) {
                if mapping.validate().is_ok() {
                    tracing::info!("粗密階層マッピングをロードしました: {}", mapping.name);
                    self.hierarchical_mapping = Some(Arc::new(mapping));
                } else {
                    tracing::warn!("階層マッピングの検証に失敗しました: {:?}", mapping_path);
                }
            } else {
                tracing::warn!(
                    "階層マッピング JSON のパースに失敗しました: {:?}",
                    mapping_path
                );
            }
        }

        self
    }

    /// モデルディレクトリ内のデュアルセッションモデル(`state_encoder.onnx`, `query_evaluator.onnx`)を検出し、
    /// 存在すれば環境変数の設定に基づいて `DualSessionEngine` と `StateCacheManager` を初期化して登録する。
    ///
    /// # 引数
    /// - `dir`: モデルファイルが配置されているディレクトリ。
    /// - `session_config`: ONNX Runtime セッション構築設定。
    pub fn init_dual_engine_from_dir<P: AsRef<std::path::Path>>(
        mut self,
        dir: P,
        session_config: SessionConfig,
    ) -> Self {
        let dir = dir.as_ref();
        let state_model_path = dir.join("state_encoder.onnx");
        let query_model_path = dir.join("query_evaluator.onnx");

        if state_model_path.exists() && query_model_path.exists() {
            tracing::info!(
                "2 セッション分割型モデルを検出しました: State={}, Query={}",
                state_model_path.display(),
                query_model_path.display()
            );

            let cache_max_bytes = std::env::var("SOKUTO_STATE_CACHE_MAX_BYTES")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(512 * 1024 * 1024);

            let cache_ttl_secs = std::env::var("SOKUTO_STATE_CACHE_TTL_SECS")
                .ok()
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(600);

            let compression_mode = match std::env::var("SOKUTO_STATE_CACHE_COMPRESSION")
                .unwrap_or_else(|_| "fp16".to_string())
                .to_lowercase()
                .as_str()
            {
                "none" | "fp32" => CacheCompressionMode::None,
                "int8" => CacheCompressionMode::Int8Symmetric,
                _ => CacheCompressionMode::Fp16,
            };

            let cache_manager = Arc::new(StateCacheManager::new(
                cache_max_bytes,
                cache_ttl_secs,
                compression_mode,
            ));

            match DualSessionEngine::new(
                &state_model_path,
                &query_model_path,
                session_config,
                Arc::clone(&cache_manager),
            ) {
                Ok(dual_engine) => {
                    tracing::info!(
                        "DualSessionEngine を初期化しました (キャッシュ上限: {}MB, 圧縮: {:?})。",
                        cache_max_bytes / (1024 * 1024),
                        compression_mode
                    );
                    self = self.with_dual_engine(Arc::new(dual_engine), cache_manager);
                }
                Err(err) => {
                    tracing::warn!(
                        "DualSessionEngine の構築に失敗したため、単一モデルモードで運用します: {err}"
                    );
                }
            }
        } else {
            tracing::warn!(
                "デュアルセッションモデル (state_encoder.onnx / query_evaluator.onnx) が '{}' に存在しないため、単一モデルフォールバックモードで起動します（State 中間特徴量キャッシュは無効化されます）。",
                dir.display()
            );
        }

        self
    }

    /// サーバーがリクエストを安全に処理可能(Readiness)であるかを検査する。
    pub fn is_ready(&self) -> bool {
        self.engine.pool_size() > 0 && self.tokenizer.max_sequence_length() > 0
    }
}
