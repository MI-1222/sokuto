//! # 推論セッション設定モジュール
//!
//! ONNX Runtime セッションの Execution Provider 優先度、
//! スレッド並列数、グラフ最適化レベル、メモリアリーナなどの構成パラメータを定義する。

use std::path::PathBuf;

use ort::session::builder::GraphOptimizationLevel;

/// システムの物理コア数とセッションプール数から最適な `intra_threads` を算出する。
///
/// ハイパースレッディング (SMT) による論理コアではなく、物理コア数 (`num_cpus::get_physical`) を基準とし、
/// スレッド間の演算器競合およびキャッシュスラッシングを防止する。
///
/// # 計算式
/// `max(1, min(max_cap, physical_cores / pool_size))`
///
/// # 引数
/// - `pool_size`: セッションプール数。
/// - `max_cap`: 単一セッションに割り当てるスレッド数の上限値。
pub fn auto_intra_threads(pool_size: usize, max_cap: usize) -> usize {
    let physical = num_cpus::get_physical().max(1);
    let pool = pool_size.max(1);
    let per_session = physical / pool;
    per_session.clamp(1, max_cap)
}

/// 単一モデルフォールバック時に長文とみなす State トークン数閾値。
pub const SINGLE_MODEL_LONG_STATE_THRESHOLD: usize = 512;

/// 単一モデルフォールバック時の長文マイクロバッチ上限チャンクサイズ。
pub const SINGLE_MODEL_LONG_STATE_MAX_CHUNK: usize = 4;

/// State トークン長に応じて単一モデル推論時の実効チャンクサイズを制限する。
///
/// 長文入力時(`state_tokens > SINGLE_MODEL_LONG_STATE_THRESHOLD`)において、
/// 巨大なパディング行列による L3 キャッシュスラッシングおよびメモリ消費の急増を抑えるため、
/// マイクロバッチ分割サイズを段階的に引き下げる。
#[inline]
pub fn effective_single_model_chunk_size(
    requested_chunk: Option<usize>,
    state_tokens: usize,
) -> Option<usize> {
    if state_tokens > SINGLE_MODEL_LONG_STATE_THRESHOLD {
        let cap = if state_tokens > 1536 {
            2
        } else {
            SINGLE_MODEL_LONG_STATE_MAX_CHUNK
        };
        Some(requested_chunk.map_or(cap, |c| c.max(1).min(cap)))
    } else {
        requested_chunk.map(|c| c.max(1))
    }
}

/// 利用可能な Execution Provider (実行バックエンド) の指定。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionProvider {
    /// 自動選択 (利用可能なアクセラレータを優先し、利用不可時は CPU へフォールバック)。
    #[default]
    Auto,
    /// Apple Silicon 向け CoreML バックエンド。
    CoreML,
    /// NVIDIA GPU 向け CUDA バックエンド。
    CUDA,
    /// NVIDIA GPU 向け TensorRT バックエンド。
    TensorRT,
    /// CPU バックエンド (高移植性・標準フォールバック)。
    CPU,
}

impl ExecutionProvider {
    /// バックエンドの表示名を取得する。
    pub fn name(&self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::CoreML => "CoreML",
            Self::CUDA => "CUDA",
            Self::TensorRT => "TensorRT",
            Self::CPU => "CPU",
        }
    }
}

impl std::fmt::Display for ExecutionProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for ExecutionProvider {
    type Err = String;

    /// 文字列から [`ExecutionProvider`] をパースする。
    ///
    /// 大小文字を区別せず、`auto`, `coreml`, `cuda`, `tensorrt`, `cpu` を解釈する。
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "coreml" => Ok(Self::CoreML),
            "cuda" => Ok(Self::CUDA),
            "tensorrt" => Ok(Self::TensorRT),
            "cpu" => Ok(Self::CPU),
            other => Err(format!(
                "未知の Execution Provider です: '{other}'。利用可能: auto, coreml, cuda, tensorrt, cpu。"
            )),
        }
    }
}

/// グラフ最適化レベル。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationLevel {
    /// 最適化を無効化。
    Disable,
    /// 基本的な最適化 (ノード結合、定数畳み込み等)。
    Basic,
    /// 拡張最適化 (レイヤー正規化融合、GELU融合等)。
    Extended,
    /// すべての最適化を適用 (デフォルト)。
    #[default]
    All,
}

impl OptimizationLevel {
    /// `ort` クレートの `GraphOptimizationLevel` に変換する。
    pub fn to_ort_level(self) -> GraphOptimizationLevel {
        match self {
            Self::Disable => GraphOptimizationLevel::Disable,
            Self::Basic => GraphOptimizationLevel::Level1,
            Self::Extended => GraphOptimizationLevel::Level2,
            Self::All => GraphOptimizationLevel::Level3,
        }
    }
}

/// ONNX Runtime セッション構築設定。
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// 優先的に試行する Execution Provider の順序付きリスト。
    pub preferred_providers: Vec<ExecutionProvider>,
    /// 単一オペレータ内の並列スレッド数 (intra_op)。None の場合は物理コア数から自動算出。
    pub intra_threads: Option<usize>,
    /// 複数オペレータ間の並列スレッド数 (inter_op)。None の場合は自動設定 (通常は 1)。
    pub inter_threads: Option<usize>,
    /// グラフ内の並列実行モード (false で Sequential、true で Parallel)。
    pub parallel_execution: bool,
    /// グラフ最適化レベル。
    pub optimization_level: OptimizationLevel,
    /// CPU メモリアリーナの有効化。
    pub enable_mem_arena: bool,
    /// 並行推論用セッションプールサイズ (デフォルト: 1)。
    pub pool_size: usize,
    /// 最適化済みモデルグラフのディスク保存・キャッシュパス (任意)。
    pub optimized_model_path: Option<PathBuf>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        #[cfg(target_os = "macos")]
        let default_providers = vec![ExecutionProvider::CoreML, ExecutionProvider::CPU];

        #[cfg(not(target_os = "macos"))]
        let default_providers = vec![
            ExecutionProvider::TensorRT,
            ExecutionProvider::CUDA,
            ExecutionProvider::CPU,
        ];

        Self {
            preferred_providers: default_providers,
            intra_threads: None,
            inter_threads: Some(1),
            parallel_execution: false,
            optimization_level: OptimizationLevel::All,
            enable_mem_arena: true,
            pool_size: 1,
            optimized_model_path: None,
        }
    }
}

impl SessionConfig {
    /// ホスト環境の物理コア数に基づき、スレッド数を自動計算した設定を生成する。
    pub fn auto() -> Self {
        let mut config = Self::default();
        config.intra_threads = Some(auto_intra_threads(config.pool_size, 8));
        config
    }

    /// CPU 推論に特化した設定を生成する。
    pub fn cpu_only() -> Self {
        Self {
            preferred_providers: vec![ExecutionProvider::CPU],
            intra_threads: None,
            inter_threads: Some(1),
            parallel_execution: false,
            optimization_level: OptimizationLevel::All,
            enable_mem_arena: true,
            pool_size: 1,
            optimized_model_path: None,
        }
    }

    /// Tier 1 (130M-INT8) モデル向けの推奨セッション設定を生成する。
    ///
    /// 軽量モデルの短時間フォワードパス (~12ms) においてスレッド同期オーバーヘッドを抑えるため、
    /// `intra_threads` の上限を 4 スレッドに制限する。
    pub fn for_tier1() -> Self {
        let mut config = Self::cpu_only();
        config.intra_threads = Some(auto_intra_threads(config.pool_size, 4));
        config
    }

    /// Tier 2 (310M-INT8) モデル向けの推奨セッション設定を生成する。
    ///
    /// 計算負荷の高い GEMM 演算を高速化するため、物理コア数に応じて最大 8 スレッドまで割り当てる。
    pub fn for_tier2() -> Self {
        let mut config = Self::cpu_only();
        config.intra_threads = Some(auto_intra_threads(config.pool_size, 8));
        config
    }

    /// 現在のセッションプール数に基づき、スレッド数を自動計算して適用する。
    pub fn with_auto_threads(mut self) -> Self {
        self.intra_threads = Some(auto_intra_threads(self.pool_size, 8));
        self
    }

    /// スレッド数を明示指定したビルダーメソッド。
    pub fn with_threads(mut self, intra: Option<usize>, inter: Option<usize>) -> Self {
        self.intra_threads = intra;
        self.inter_threads = inter;
        self
    }

    /// 並列実行フラグを指定したビルダーメソッド。
    pub fn with_parallel_execution(mut self, parallel: bool) -> Self {
        self.parallel_execution = parallel;
        self
    }

    /// 最適化レベルを指定したビルダーメソッド。
    pub fn with_optimization_level(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    /// メモリアリーナ有効化フラグを指定したビルダーメソッド。
    pub fn with_memory_arena(mut self, enable: bool) -> Self {
        self.enable_mem_arena = enable;
        self
    }

    /// セッションプールサイズを指定したビルダーメソッド。
    pub fn with_pool_size(mut self, size: usize) -> Self {
        self.pool_size = size.max(1);
        self
    }

    /// 最適化済みモデルの保存・キャッシュパスを指定したビルダーメソッド。
    pub fn with_optimized_model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.optimized_model_path = Some(path.into());
        self
    }

    /// モデルファイルおよびセッション設定からグラフキャッシュの識別ハッシュ (XXH3-64) を算出する。
    ///
    /// モデルパス、ファイルサイズ、最終更新日時、最適化レベル、優先プロバイダ列、
    /// 並列実行設定を組み合わせ、モデル差し替え時や設定変更時のキャッシュ陳腐化を防止する。
    pub(crate) fn compute_graph_cache_hash(&self, model_path: &std::path::Path) -> u64 {
        use xxhash_rust::xxh3::Xxh3;

        let mut hasher = Xxh3::new();

        // 1. カノニカルパス (解決不能な場合は元のパス)
        let canonical =
            std::fs::canonicalize(model_path).unwrap_or_else(|_| model_path.to_path_buf());
        hasher.update(canonical.to_string_lossy().as_bytes());

        // 2. モデルファイルのメタデータ (サイズ, 最終更新日時)
        if let Ok(metadata) = std::fs::metadata(&canonical) {
            hasher.update(&metadata.len().to_le_bytes());
            if let Some(duration) = metadata
                .modified()
                .ok()
                .and_then(|m| m.duration_since(std::time::UNIX_EPOCH).ok())
            {
                hasher.update(&duration.as_nanos().to_le_bytes());
            }
        }

        // 3. セッション設定パラメータ
        let opt_level_byte = match self.optimization_level {
            OptimizationLevel::Disable => 0u8,
            OptimizationLevel::Basic => 1u8,
            OptimizationLevel::Extended => 2u8,
            OptimizationLevel::All => 3u8,
        };
        hasher.update(&[opt_level_byte]);

        for provider in &self.preferred_providers {
            hasher.update(provider.name().as_bytes());
        }

        hasher.update(&[self.parallel_execution as u8]);

        hasher.digest()
    }

    /// 最適化済みモデルグラフのディスクキャッシュを安全に自動設定するビルダーメソッド。
    ///
    /// `optimized_model_path` が未指定の場合に、書き込み可能な一時ディレクトリ
    /// (`std::env::temp_dir()` 配下)にモデル識別ハッシュ付きのファイルパスを自動生成する。
    /// Docker 等の読み取り専用 (`:ro`) マウント環境でも安全に動作し、ディレクトリ生成に失敗した場合は
    /// キャッシュを有効化せず安全にスキップする。
    ///
    /// キャッシュファイル名のハッシュ値には、モデルファイルのパス、ファイルサイズ、
    /// 最終更新日時、最適化レベル、優先 Execution Provider、並列実行設定が含まれ、
    /// モデル差し替え時や設定変更時のキャッシュ陳腐化を確実に防止する。
    pub fn with_auto_graph_cache(mut self, model_path: &std::path::Path) -> Self {
        if self.optimized_model_path.is_some() {
            return self;
        }

        let file_stem = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        let path_hash = self.compute_graph_cache_hash(model_path);
        let cache_dir = std::env::temp_dir().join("sokuto_graph_cache");
        if std::fs::create_dir_all(&cache_dir).is_ok() {
            let cache_file = cache_dir.join(format!("{file_stem}_{path_hash:016x}.onnx"));
            self.optimized_model_path = Some(cache_file);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_intra_threads() {
        let physical = num_cpus::get_physical();
        assert!(physical >= 1);

        let t1 = auto_intra_threads(1, 4);
        assert!((1..=4).contains(&t1));

        let t2 = auto_intra_threads(1, 8);
        assert!((1..=8).contains(&t2));

        let t_pool = auto_intra_threads(4, 8);
        assert!(t_pool >= 1);
    }

    #[test]
    fn test_tier_presets() {
        let cfg1 = SessionConfig::for_tier1();
        assert_eq!(cfg1.preferred_providers, vec![ExecutionProvider::CPU]);
        assert_eq!(cfg1.inter_threads, Some(1));
        assert!(!cfg1.parallel_execution);
        assert!(cfg1.intra_threads.is_some());
        assert!(cfg1.intra_threads.unwrap() <= 4);

        let cfg2 = SessionConfig::for_tier2();
        assert_eq!(cfg2.preferred_providers, vec![ExecutionProvider::CPU]);
        assert_eq!(cfg2.inter_threads, Some(1));
        assert!(!cfg2.parallel_execution);
        assert!(cfg2.intra_threads.is_some());
        assert!(cfg2.intra_threads.unwrap() <= 8);
    }

    #[test]
    fn test_execution_provider_from_str() {
        use std::str::FromStr;

        assert_eq!(
            ExecutionProvider::from_str("auto").unwrap(),
            ExecutionProvider::Auto
        );
        assert_eq!(
            ExecutionProvider::from_str("CoreML").unwrap(),
            ExecutionProvider::CoreML
        );
        assert_eq!(
            ExecutionProvider::from_str("CUDA").unwrap(),
            ExecutionProvider::CUDA
        );
        assert_eq!(
            ExecutionProvider::from_str("tensorrt").unwrap(),
            ExecutionProvider::TensorRT
        );
        assert_eq!(
            ExecutionProvider::from_str("CPU").unwrap(),
            ExecutionProvider::CPU
        );

        assert!(ExecutionProvider::from_str("invalid_provider").is_err());
    }

    #[test]
    fn test_execution_provider_display() {
        assert_eq!(format!("{}", ExecutionProvider::Auto), "Auto");
        assert_eq!(format!("{}", ExecutionProvider::CoreML), "CoreML");
        assert_eq!(format!("{}", ExecutionProvider::CUDA), "CUDA");
        assert_eq!(format!("{}", ExecutionProvider::TensorRT), "TensorRT");
        assert_eq!(format!("{}", ExecutionProvider::CPU), "CPU");
    }

    #[test]
    fn test_effective_single_model_chunk_size() {
        // 短文時 (<= 512 トークン): 指定値がそのまま反映されること。
        assert_eq!(effective_single_model_chunk_size(None, 100), None);
        assert_eq!(effective_single_model_chunk_size(Some(8), 512), Some(8));
        assert_eq!(effective_single_model_chunk_size(Some(0), 200), Some(1));

        // 中長文時 (513..=1536 トークン): 最大 4 に制限されること。
        assert_eq!(effective_single_model_chunk_size(None, 513), Some(4));
        assert_eq!(effective_single_model_chunk_size(Some(16), 1000), Some(4));
        assert_eq!(effective_single_model_chunk_size(Some(2), 1000), Some(2));

        // 超長文時 (> 1536 トークン): 最大 2 に制限されること。
        assert_eq!(effective_single_model_chunk_size(None, 1537), Some(2));
        assert_eq!(effective_single_model_chunk_size(Some(8), 2000), Some(2));
        assert_eq!(effective_single_model_chunk_size(Some(1), 2000), Some(1));
    }

    #[test]
    fn test_with_auto_graph_cache() {
        let model_path = std::path::Path::new("/dummy/model.onnx");
        let cfg = SessionConfig::default().with_auto_graph_cache(model_path);

        assert!(cfg.optimized_model_path.is_some());
        let path = cfg.optimized_model_path.unwrap();
        assert!(path.to_string_lossy().contains("model_"));
        assert!(path.to_string_lossy().ends_with(".onnx"));

        // 明示的に指定済みの場合は上書きされないこと。
        let custom_path = std::path::PathBuf::from("/custom/opt.onnx");
        let cfg2 = SessionConfig::default()
            .with_optimized_model_path(&custom_path)
            .with_auto_graph_cache(model_path);
        assert_eq!(cfg2.optimized_model_path, Some(custom_path));
    }

    #[test]
    fn test_compute_graph_cache_hash_stale_prevention() {
        let temp_dir =
            std::env::temp_dir().join(format!("sokuto_hash_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&temp_dir);
        let model_file = temp_dir.join("test_model.onnx");

        std::fs::write(&model_file, b"model v1").expect("テスト用ファイル作成に失敗。");

        let cfg = SessionConfig::default();
        let hash1 = cfg.compute_graph_cache_hash(&model_file);

        // 同一ファイル・同一設定なら同じハッシュになること
        let hash1_retry = cfg.compute_graph_cache_hash(&model_file);
        assert_eq!(hash1, hash1_retry);

        // 最適化レベル変更時にハッシュが変わること
        let cfg_opt = SessionConfig::default().with_optimization_level(OptimizationLevel::Disable);
        let hash_opt = cfg_opt.compute_graph_cache_hash(&model_file);
        assert_ne!(hash1, hash_opt);

        // Execution Provider 変更時にハッシュが変わること
        let cfg_cpu = SessionConfig::cpu_only();
        let hash_cpu = cfg_cpu.compute_graph_cache_hash(&model_file);
        // default と cpu_only が異なる EP 構成なら異なるハッシュになること
        if cfg.preferred_providers != cfg_cpu.preferred_providers {
            assert_ne!(hash1, hash_cpu);
        }

        // ファイルサイズ変更時にハッシュが変わること (キャッシュ陳腐化防止)
        std::fs::write(&model_file, b"model v2 modified content")
            .expect("テスト用ファイル更新に失敗。");
        let hash2 = cfg.compute_graph_cache_hash(&model_file);
        assert_ne!(hash1, hash2);

        // クリーンアップ
        let _ = std::fs::remove_file(&model_file);
        let _ = std::fs::remove_dir(&temp_dir);
    }
}
