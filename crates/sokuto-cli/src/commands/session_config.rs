//! # セッション設定およびプロバイダ解決モジュール
//!
//! CLI 引数に基づき ONNX Runtime の `SessionConfig` を構築するビルダーおよび
//! Execution Provider のパース処理を提供する。

use sokuto_runtime::engine::{ExecutionProvider, SessionConfig};

/// 文字列から Execution Provider をパースする。
///
/// 大小文字を区別せず、`auto`, `coreml`, `cuda`, `tensorrt`, `cpu` を解釈する。
///
/// # エラー
///
/// サポートされていないプロバイダ文字列が指定された場合はエラーメッセージを返す。
pub fn parse_provider(name: &str) -> Result<ExecutionProvider, String> {
    match name.to_lowercase().as_str() {
        "auto" => Ok(ExecutionProvider::Auto),
        "coreml" => Ok(ExecutionProvider::CoreML),
        "cuda" => Ok(ExecutionProvider::CUDA),
        "tensorrt" => Ok(ExecutionProvider::TensorRT),
        "cpu" => Ok(ExecutionProvider::CPU),
        other => Err(format!(
            "未知の Execution Provider です: '{other}'。利用可能: auto, coreml, cuda, tensorrt, cpu。"
        )),
    }
}

/// CLI 引数から [`SessionConfig`] を段階的に構築するビルダー。
#[derive(Debug, Clone)]
pub struct SessionConfigBuilder {
    pool_size: usize,
    intra_threads: Option<usize>,
    inter_threads: Option<usize>,
    provider: Option<String>,
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

    /// Execution Provider 名を設定する。
    pub fn provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Option でラップされた Execution Provider 名を設定する。
    pub fn provider_opt(mut self, provider: Option<impl Into<String>>) -> Self {
        self.provider = provider.map(Into::into);
        self
    }

    /// CPU メモリアリーナの有効化フラグを設定する。
    pub fn enable_mem_arena(mut self, enable: bool) -> Self {
        self.enable_mem_arena = Some(enable);
        self
    }

    /// 設定内容を検証し、[`SessionConfig`] を構築する。
    ///
    /// プロバイダ文字列が指定されている場合は [`parse_provider`] でパースし、
    /// `Auto` の場合はデフォルトの優先リストを、特定プロバイダの場合は
    /// `[指定プロバイダ, CPU]` というフォールバック順序を設定する。
    /// なお、指定プロバイダが `CPU` の場合は重複登録を防ぐため `[CPU]` のみを設定する。
    pub fn build(self) -> Result<SessionConfig, String> {
        let mut session_config = SessionConfig {
            pool_size: self.pool_size.max(1),
            intra_threads: self.intra_threads,
            inter_threads: self.inter_threads.or(Some(1)),
            ..Default::default()
        };

        if let Some(enable_mem_arena) = self.enable_mem_arena {
            session_config.enable_mem_arena = enable_mem_arena;
        }

        if let Some(ref provider_str) = self.provider {
            let provider = parse_provider(provider_str)?;
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
    provider: Option<&str>,
) -> Result<SessionConfig, String> {
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
            .provider("auto")
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
            .provider("cpu")
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
            .provider("cuda")
            .build()
            .unwrap();
        assert_eq!(
            config.preferred_providers,
            vec![ExecutionProvider::CUDA, ExecutionProvider::CPU]
        );
    }

    #[test]
    fn test_helper_build_session_config() {
        let config = build_session_config(2, Some(4), None, Some("cpu")).unwrap();
        assert_eq!(config.pool_size, 2);
        assert_eq!(config.intra_threads, Some(4));
        assert_eq!(config.inter_threads, Some(1));
    }
}
