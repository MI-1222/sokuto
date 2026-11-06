//! # セッション設定およびプロバイダ解決モジュール
//!
//! CLI 引数に基づき ONNX Runtime の `SessionConfig` を構築するビルダーおよび
//! Execution Provider のパース処理を提供する。

use crate::error::CliError;
use sokuto_runtime::engine::{ExecutionProvider, SessionConfig};

/// CLI 引数用の Execution Provider (実行バックエンド) 指定。
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    clap::ValueEnum,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionProviderArg {
    /// 自動選択 (利用可能なアクセラレータを優先し、利用不可時は CPU へフォールバック)。
    #[default]
    Auto,
    /// Apple Silicon 向け CoreML バックエンド。
    Coreml,
    /// NVIDIA GPU 向け CUDA バックエンド。
    Cuda,
    /// NVIDIA GPU 向け TensorRT バックエンド。
    Tensorrt,
    /// CPU バックエンド。
    Cpu,
}

impl From<ExecutionProviderArg> for ExecutionProvider {
    fn from(arg: ExecutionProviderArg) -> Self {
        match arg {
            ExecutionProviderArg::Auto => Self::Auto,
            ExecutionProviderArg::Coreml => Self::CoreML,
            ExecutionProviderArg::Cuda => Self::CUDA,
            ExecutionProviderArg::Tensorrt => Self::TensorRT,
            ExecutionProviderArg::Cpu => Self::CPU,
        }
    }
}

impl From<ExecutionProvider> for ExecutionProviderArg {
    fn from(ep: ExecutionProvider) -> Self {
        match ep {
            ExecutionProvider::Auto => Self::Auto,
            ExecutionProvider::CoreML => Self::Coreml,
            ExecutionProvider::CUDA => Self::Cuda,
            ExecutionProvider::TensorRT => Self::Tensorrt,
            ExecutionProvider::CPU => Self::Cpu,
        }
    }
}

impl std::fmt::Display for ExecutionProviderArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Coreml => write!(f, "coreml"),
            Self::Cuda => write!(f, "cuda"),
            Self::Tensorrt => write!(f, "tensorrt"),
            Self::Cpu => write!(f, "cpu"),
        }
    }
}

impl std::str::FromStr for ExecutionProviderArg {
    type Err = CliError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let ep = parse_provider(s)?;
        Ok(Self::from(ep))
    }
}

/// 文字列から Execution Provider をパースする。
///
/// 大小文字を区別せず、`auto`, `coreml`, `cuda`, `tensorrt`, `cpu` を解釈する。
///
/// # エラー
///
/// サポートされていないプロバイダ文字列が指定された場合は [`CliError::UnknownProvider`] を返す。
pub fn parse_provider(name: &str) -> Result<ExecutionProvider, CliError> {
    name.parse().map_err(CliError::UnknownProvider)
}

/// CLI 引数から [`SessionConfig`] を段階的に構築するビルダー。
#[derive(Debug, Clone)]
pub struct SessionConfigBuilder {
    pool_size: usize,
    intra_threads: Option<usize>,
    inter_threads: Option<usize>,
    provider: Option<ExecutionProvider>,
    enable_mem_arena: Option<bool>,
}

impl Default for SessionConfigBuilder {
    fn default() -> Self {
        Self {
            pool_size: 1,
            intra_threads: None,
            inter_threads: None,
            provider: None,
            enable_mem_arena: None,
        }
    }
}

impl SessionConfigBuilder {
    /// 新しいビルダーインスタンスを生成する。
    pub fn new() -> Self {
        Self::default()
    }

    /// セッションプールサイズを設定する。
    ///
    /// 0 が指定された場合は最小値 1 に補正される。
    pub fn pool_size(mut self, size: usize) -> Self {
        self.pool_size = size.max(1);
        self
    }

    /// 単一オペレータ内の並列スレッド数 (intra-op) を設定する。
    pub fn intra_threads(mut self, threads: Option<usize>) -> Self {
        self.intra_threads = threads;
        self
    }

    /// 複数オペレータ間の並列スレッド数 (inter-op) を設定する。
    pub fn inter_threads(mut self, threads: Option<usize>) -> Self {
        self.inter_threads = threads;
        self
    }

    /// Execution Provider を設定する。
    pub fn provider(mut self, provider: impl Into<ExecutionProvider>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Option でラップされた Execution Provider を設定する。
    pub fn provider_opt(mut self, provider: Option<impl Into<ExecutionProvider>>) -> Self {
        self.provider = provider.map(Into::into);
        self
    }

    /// 文字列から Execution Provider をパースして設定する。
    ///
    /// # エラー
    ///
    /// 未知のプロバイダ文字列が渡された場合はエラーを返却する。
    pub fn provider_str(mut self, provider: &str) -> Result<Self, CliError> {
        self.provider = Some(parse_provider(provider)?);
        Ok(self)
    }

    /// CPU メモリアリーナの有効化フラグを設定する。
    pub fn enable_mem_arena(mut self, enable: bool) -> Self {
        self.enable_mem_arena = Some(enable);
        self
    }

    /// 設定内容を検証し、[`SessionConfig`] を構築する。
    ///
    /// `Auto` の場合はデフォルトの優先リストを、特定プロバイダの場合は
    /// `[指定プロバイダ, CPU]` というフォールバック順序を設定する。
    /// なお、指定プロバイダが `CPU` の場合は重複登録を防ぐため `[CPU]` のみを設定する。
    pub fn build(self) -> Result<SessionConfig, CliError> {
        let mut session_config = SessionConfig {
            pool_size: self.pool_size.max(1),
            intra_threads: self.intra_threads,
            inter_threads: self.inter_threads.or(Some(1)),
            ..Default::default()
        };

        if let Some(enable_mem_arena) = self.enable_mem_arena {
            session_config.enable_mem_arena = enable_mem_arena;
        }

        if let Some(provider) = self.provider {
            session_config.preferred_providers = match provider {
                ExecutionProvider::Auto => SessionConfig::default().preferred_providers,
                ExecutionProvider::CPU => vec![ExecutionProvider::CPU],
                other => vec![other, ExecutionProvider::CPU],
            };
        }

        Ok(session_config)
    }
}

/// CLI 引数から [`SessionConfig`] を直接構築するヘルパー関数。
pub fn build_session_config(
    pool_size: usize,
    intra_threads: Option<usize>,
    inter_threads: Option<usize>,
    provider: Option<impl Into<ExecutionProvider>>,
) -> Result<SessionConfig, CliError> {
    SessionConfigBuilder::new()
        .pool_size(pool_size)
        .intra_threads(intra_threads)
        .inter_threads(inter_threads)
        .provider_opt(provider)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_provider_valid() {
        assert_eq!(parse_provider("auto").unwrap(), ExecutionProvider::Auto);
        assert_eq!(parse_provider("CoreML").unwrap(), ExecutionProvider::CoreML);
        assert_eq!(parse_provider("CUDA").unwrap(), ExecutionProvider::CUDA);
        assert_eq!(
            parse_provider("TensorRT").unwrap(),
            ExecutionProvider::TensorRT
        );
        assert_eq!(parse_provider("cpu").unwrap(), ExecutionProvider::CPU);
    }

    #[test]
    fn test_parse_provider_invalid() {
        let result = parse_provider("unsupported_backend");
        assert!(result.is_err());
    }

    #[test]
    fn test_execution_provider_arg_conversions() {
        use std::str::FromStr;

        assert_eq!(
            ExecutionProvider::from(ExecutionProviderArg::Auto),
            ExecutionProvider::Auto
        );
        assert_eq!(
            ExecutionProvider::from(ExecutionProviderArg::Coreml),
            ExecutionProvider::CoreML
        );
        assert_eq!(
            ExecutionProvider::from(ExecutionProviderArg::Cuda),
            ExecutionProvider::CUDA
        );
        assert_eq!(
            ExecutionProvider::from(ExecutionProviderArg::Tensorrt),
            ExecutionProvider::TensorRT
        );
        assert_eq!(
            ExecutionProvider::from(ExecutionProviderArg::Cpu),
            ExecutionProvider::CPU
        );

        assert_eq!(
            ExecutionProviderArg::from(ExecutionProvider::Auto),
            ExecutionProviderArg::Auto
        );
        assert_eq!(
            ExecutionProviderArg::from(ExecutionProvider::CoreML),
            ExecutionProviderArg::Coreml
        );
        assert_eq!(
            ExecutionProviderArg::from(ExecutionProvider::CUDA),
            ExecutionProviderArg::Cuda
        );
        assert_eq!(
            ExecutionProviderArg::from(ExecutionProvider::TensorRT),
            ExecutionProviderArg::Tensorrt
        );
        assert_eq!(
            ExecutionProviderArg::from(ExecutionProvider::CPU),
            ExecutionProviderArg::Cpu
        );

        assert_eq!(
            ExecutionProviderArg::from_str("cuda").unwrap(),
            ExecutionProviderArg::Cuda
        );
        assert_eq!(format!("{}", ExecutionProviderArg::Cuda), "cuda");
    }

    #[test]
    fn test_builder_defaults() {
        let config = SessionConfigBuilder::new().build().unwrap();
        assert_eq!(config.pool_size, 1);
        assert_eq!(config.intra_threads, None);
        assert_eq!(config.inter_threads, Some(1));
    }

    #[test]
    fn test_builder_with_provider_auto() {
        let config = SessionConfigBuilder::new()
            .pool_size(4)
            .provider(ExecutionProviderArg::Auto)
            .build()
            .unwrap();
        assert_eq!(config.pool_size, 4);
        assert_eq!(
            config.preferred_providers,
            SessionConfig::default().preferred_providers
        );
    }

    #[test]
    fn test_builder_with_specific_provider() {
        let config = SessionConfigBuilder::new()
            .pool_size(0)
            .intra_threads(Some(2))
            .inter_threads(Some(4))
            .provider(ExecutionProvider::CPU)
            .enable_mem_arena(false)
            .build()
            .unwrap();
        assert_eq!(config.pool_size, 1);
        assert_eq!(config.intra_threads, Some(2));
        assert_eq!(config.inter_threads, Some(4));
        assert!(!config.enable_mem_arena);
        assert_eq!(config.preferred_providers, vec![ExecutionProvider::CPU]);
    }

    #[test]
    fn test_builder_with_non_cpu_provider() {
        let config = SessionConfigBuilder::new()
            .provider(ExecutionProviderArg::Cuda)
            .build()
            .unwrap();
        assert_eq!(
            config.preferred_providers,
            vec![ExecutionProvider::CUDA, ExecutionProvider::CPU]
        );
    }

    #[test]
    fn test_builder_with_provider_str() {
        let config = SessionConfigBuilder::new()
            .provider_str("coreml")
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(
            config.preferred_providers,
            vec![ExecutionProvider::CoreML, ExecutionProvider::CPU]
        );
    }

    #[test]
    fn test_helper_build_session_config() {
        let config = build_session_config(2, Some(4), None, Some(ExecutionProvider::CPU)).unwrap();
        assert_eq!(config.pool_size, 2);
        assert_eq!(config.intra_threads, Some(4));
        assert_eq!(config.inter_threads, Some(1));
    }
}
