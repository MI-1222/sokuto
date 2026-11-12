//! # Execution Provider 解決モジュール
//!
//! 利用可能なアクセラレータ(CoreML / CUDA / TensorRT)の登録を試行し、
//! 初期化失敗時やドライバ未検出時には CPU バックエンドへ安全にフォールバックする。

use ort::ep::ExecutionProvider as OrtExecutionProvider;
use ort::session::builder::SessionBuilder;

use crate::engine::config::ExecutionProvider;
use crate::error::{Result, RuntimeError};

/// 指定された優先順位リストに従って Execution Provider をセッションビルダーに登録する。
///
/// 実際に登録に成功した最優先の Execution Provider を返却する。
/// すべてのアクセラレータが失敗または未コンパイルの場合は、必ず CPU へフォールバックする。
pub fn register_execution_providers(
    builder: &mut SessionBuilder,
    preferred: &[ExecutionProvider],
    enable_mem_arena: bool,
) -> Result<ExecutionProvider> {
    let mut explicit_cpu = false;
    for provider in preferred {
        match provider {
            ExecutionProvider::Auto => {
                // Auto の場合はプラットフォーム標準の優先順位を再帰的に試行
                #[cfg(target_os = "macos")]
                let sub_preferred = [ExecutionProvider::CoreML, ExecutionProvider::CPU];
                #[cfg(not(target_os = "macos"))]
                let sub_preferred = [
                    ExecutionProvider::TensorRT,
                    ExecutionProvider::CUDA,
                    ExecutionProvider::CPU,
                ];
                if let Ok(active) =
                    register_execution_providers(builder, &sub_preferred, enable_mem_arena)
                {
                    return Ok(active);
                }
            }
            ExecutionProvider::CoreML => {
                #[cfg(feature = "coreml")]
                {
                    use ort::ep::coreml::CoreML;
                    let ep = CoreML::default();
                    match ep.is_available() {
                        Ok(true) => match ep.register(builder) {
                            Ok(()) => {
                                tracing::info!(
                                    "CoreML Execution Provider が正常に登録されました。"
                                );
                                return Ok(ExecutionProvider::CoreML);
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "CoreML の登録に失敗しました ({e})。CPU にフォールバックします。"
                                );
                            }
                        },
                        Ok(false) => {
                            tracing::info!(
                                "CoreML は利用可能ではありません。次の EP を試行します。"
                            );
                        }
                        Err(e) => {
                            tracing::warn!("CoreML 利用可否確認中にエラーが発生しました: {e}。");
                        }
                    }
                }
                #[cfg(not(feature = "coreml"))]
                {
                    tracing::debug!(
                        "CoreML feature は未コンパイルです。CPU にフォールバックします。"
                    );
                }
            }
            ExecutionProvider::CUDA => {
                #[cfg(feature = "cuda")]
                {
                    use ort::ep::cuda::CUDA;
                    let ep = CUDA::default();
                    match ep.is_available() {
                        Ok(true) => match ep.register(builder) {
                            Ok(()) => {
                                tracing::info!("CUDA Execution Provider が正常に登録されました。");
                                return Ok(ExecutionProvider::CUDA);
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "CUDA の登録に失敗しました ({e})。CPU にフォールバックします。"
                                );
                            }
                        },
                        Ok(false) => {
                            tracing::info!("CUDA は利用可能ではありません。次の EP を試行します。");
                        }
                        Err(e) => {
                            tracing::warn!("CUDA 利用可否確認中にエラーが発生しました: {e}。");
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    tracing::debug!(
                        "CUDA feature は未コンパイルです。CPU にフォールバックします。"
                    );
                }
            }
            ExecutionProvider::TensorRT => {
                #[cfg(feature = "tensorrt")]
                {
                    use ort::ep::tensorrt::TensorRT;
                    let ep = TensorRT::default();
                    match ep.is_available() {
                        Ok(true) => match ep.register(builder) {
                            Ok(()) => {
                                tracing::info!(
                                    "TensorRT Execution Provider が正常に登録されました。"
                                );
                                return Ok(ExecutionProvider::TensorRT);
                            }
                            Err(e) => {
                                tracing::warn!(
                                    "TensorRT の登録に失敗しました ({e})。CPU にフォールバックします。"
                                );
                            }
                        },
                        Ok(false) => {
                            tracing::info!(
                                "TensorRT は利用可能ではありません。次の EP を試行します。"
                            );
                        }
                        Err(e) => {
                            tracing::warn!("TensorRT 利用可否確認中にエラーが発生しました: {e}。");
                        }
                    }
                }
                #[cfg(not(feature = "tensorrt"))]
                {
                    tracing::debug!(
                        "TensorRT feature は未コンパイルです。CPU にフォールバックします。"
                    );
                }
            }
            ExecutionProvider::CPU => {
                explicit_cpu = true;
                break;
            }
        }
    }

    register_cpu_provider(builder, enable_mem_arena, !explicit_cpu)
}

/// CPU Execution Provider をセッションビルダーへ登録する共通ヘルパー。
///
/// ループ内で明示的な `ExecutionProvider::CPU` が選択された場合 (`is_fallback == false`) と、
/// 優先リストに `CPU` が含まれないまま全アクセラレータ登録に失敗・未コンパイルとなった
/// 最終フォールバック時 (`is_fallback == true`) の双方を単一経路で処理する。
fn register_cpu_provider(
    builder: &mut SessionBuilder,
    enable_mem_arena: bool,
    is_fallback: bool,
) -> Result<ExecutionProvider> {
    let ep = ort::ep::CPU::default().with_arena_allocator(enable_mem_arena);
    ep.register(builder).map_err(|e| {
        let context = if is_fallback {
            "CPUExecutionProvider への最終フォールバックに失敗しました"
        } else {
            "CPUExecutionProvider の登録に失敗しました"
        };
        RuntimeError::ExecutionProviderError(format!("{context}: {e}。"))
    })?;
    if is_fallback {
        tracing::info!(
            "CPU Execution Provider への最終フォールバックが完了しました (メモリアリーナ: {enable_mem_arena})。"
        );
    } else {
        tracing::info!(
            "CPU Execution Provider が正常に登録されました (メモリアリーナ: {enable_mem_arena})。"
        );
    }
    Ok(ExecutionProvider::CPU)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ort::session::Session;

    #[test]
    fn test_register_execution_providers_explicit_cpu_and_empty_fallback() {
        let mut cpu_builder = Session::builder().expect("SessionBuilder 作成に成功すること。");
        let ep_cpu =
            register_execution_providers(&mut cpu_builder, &[ExecutionProvider::CPU], true)
                .expect("明示的 CPU 指定で登録に成功すること。");
        assert_eq!(ep_cpu, ExecutionProvider::CPU);

        let mut empty_builder = Session::builder().expect("SessionBuilder 作成に成功すること。");
        let ep_empty = register_execution_providers(&mut empty_builder, &[], false)
            .expect("空リスト指定時も最終フォールバックで CPU 登録に成功すること。");
        assert_eq!(ep_empty, ExecutionProvider::CPU);
    }

    #[test]
    fn test_register_execution_providers_cpu_before_accelerator_respects_order() {
        let mut builder = Session::builder().expect("SessionBuilder 作成に成功すること。");
        let ep = register_execution_providers(
            &mut builder,
            &[ExecutionProvider::CPU, ExecutionProvider::CoreML],
            true,
        )
        .expect("CPU 優先指定で登録に成功すること。");
        assert_eq!(ep, ExecutionProvider::CPU);
    }
}
