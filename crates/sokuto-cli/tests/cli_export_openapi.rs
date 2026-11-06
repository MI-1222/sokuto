//! # CLI OpenAPI エクスポート統合テスト
//!
//! `sokuto export-openapi` による静的スキーマ生成と
//! ファイル出力の完全性を検証する。

use serde_json::Value;
use sokuto_cli::commands::export_openapi::run_export_openapi;
use sokuto_server::openapi::export_openapi_json;
use std::fs;
use std::path::PathBuf;

#[test]
fn test_export_openapi_json_structure() {
    let json_str = export_openapi_json();
    assert!(!json_str.is_empty());

    let parsed: Value = serde_json::from_str(&json_str)
        .expect("エクスポートされた文字列が有効な JSON であること。");

    // OpenAPI 3.1 仕様書の必須フィールド
    assert!(parsed.get("openapi").is_some());
    assert!(parsed.get("info").is_some());
    assert!(parsed.get("paths").is_some());
    assert!(parsed.get("components").is_some());

    // エンドポイントおよび主要スキーマの存在確認
    assert!(parsed["paths"]["/v1/systemone"].is_object());
    assert!(parsed["paths"]["/health"].is_object());
    assert!(parsed["paths"]["/ready"].is_object());
    assert!(parsed["paths"]["/metrics"].is_object());
    assert!(parsed["components"]["schemas"]["SystemOneRequest"].is_object());
    assert!(parsed["components"]["schemas"]["SystemOneResponse"].is_object());
}

#[test]
fn test_export_openapi_file_output() {
    let json_str = export_openapi_json();

    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("sokuto_test_openapi.json");

    fs::write(&temp_file, &json_str).expect("一時ファイルへの書き込みに成功すること。");

    let read_content =
        fs::read_to_string(&temp_file).expect("書き込んだファイルが読み込めること。");
    assert_eq!(read_content, json_str);

    // テスト後クリーンアップ
    let _ = fs::remove_file(&temp_file);
}

#[test]
fn test_run_export_openapi_current_dir_relative_path() {
    let filename = format!("test_openapi_export_{}.json", std::process::id());
    let path = PathBuf::from(&filename);

    let result = run_export_openapi(Some(path.clone()));
    assert!(
        result.is_ok(),
        "カレントディレクトリ直下へのファイル出力に成功。"
    );

    assert!(path.exists(), "ファイルが正常に生成されていること。");
    let content = fs::read_to_string(&path).expect("出力ファイルが読み出せること。");
    assert!(!content.is_empty(), "ファイル内容が空でないこと。");

    let _ = fs::remove_file(&path);
}
