//! # OpenAPI 仕様書エクスポートコマンド
//!
//! TypeSafe AI Jev 互換 HTTP API の OpenAPI 3.1 仕様書を
//! 静的 JSON 文字列として生成し、ファイルまたは標準出力へ出力する。

use std::fs;
use std::path::PathBuf;

use sokuto_server::openapi::export_openapi_json;

/// OpenAPI 3.1 仕様書エクスポートコマンドを実行する。
///
/// # 引数
/// - `output`: 出力先ファイルパス。None の場合は標準出力に出力する。
pub fn run_export_openapi(
    output: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let json_str = export_openapi_json();

    if let Some(path) = output {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, &json_str)?;
        eprintln!("OpenAPI 3.1 仕様書を正常に出力しました: {}", path.display());
    } else {
        println!("{json_str}");
    }

    Ok(())
}
