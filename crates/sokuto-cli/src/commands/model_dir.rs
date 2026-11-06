//! # モデルディレクトリ検証モジュール
//!
//! 推論サーバー起動や量子化、ベンチマーク実行に必要なモデルディレクトリの完全性を検証する。

use std::path::{Path, PathBuf};
use thiserror::Error;

/// モデルディレクトリ検証に関するエラー。
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ModelDirError {
    /// モデルディレクトリ自体が存在しない。
    #[error("モデルディレクトリが存在しません: {}。", .0.display())]
    NotFound(PathBuf),

    /// 必須モデルファイル (`model.onnx`) が見つからない。
    #[error("必須モデルファイルが見つかりません: {}。", .0.display())]
    MissingModelFile(PathBuf),

    /// 必須トークナイザー定義ファイル (`tokenizer.json`) が見つからない。
    #[error("必須トークナイザー定義ファイルが見つかりません: {}。", .0.display())]
    MissingTokenizerFile(PathBuf),
}

/// モデルディレクトリの完全性を検証する。
///
/// モデルディレクトリおよび必須ファイル (`model.onnx`, `tokenizer.json`) の
/// 存在を確認し、不足があれば対応するエラーを返す。
///
/// # 引数
/// - `dir`: 検証対象のモデルディレクトリパス。
///
/// # エラー
/// - ディレクトリまたは必須成果物が存在しない場合に [`ModelDirError`] を返す。
pub fn validate_model_dir(dir: &Path) -> Result<(), ModelDirError> {
    if !dir.exists() {
        return Err(ModelDirError::NotFound(dir.to_path_buf()));
    }

    let onnx_file = dir.join("model.onnx");
    if !onnx_file.exists() {
        return Err(ModelDirError::MissingModelFile(onnx_file));
    }

    let tok_file = dir.join("tokenizer.json");
    if !tok_file.exists() {
        return Err(ModelDirError::MissingTokenizerFile(tok_file));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_validate_model_dir_not_found() {
        let non_existent = PathBuf::from("target/non_existent_model_dir_for_test");
        let result = validate_model_dir(&non_existent);
        assert_eq!(result, Err(ModelDirError::NotFound(non_existent)));
    }

    #[test]
    fn test_validate_model_dir_missing_onnx() {
        let temp_dir = std::env::temp_dir().join("sokuto_test_missing_onnx");
        let _ = fs::create_dir_all(&temp_dir);

        let result = validate_model_dir(&temp_dir);
        let expected_onnx = temp_dir.join("model.onnx");
        assert_eq!(result, Err(ModelDirError::MissingModelFile(expected_onnx)));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_validate_model_dir_missing_tokenizer() {
        let temp_dir = std::env::temp_dir().join("sokuto_test_missing_tokenizer");
        let _ = fs::create_dir_all(&temp_dir);
        let onnx_file = temp_dir.join("model.onnx");
        let _ = fs::write(&onnx_file, b"dummy onnx");

        let result = validate_model_dir(&temp_dir);
        let expected_tok = temp_dir.join("tokenizer.json");
        assert_eq!(
            result,
            Err(ModelDirError::MissingTokenizerFile(expected_tok))
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_validate_model_dir_success() {
        let temp_dir = std::env::temp_dir().join("sokuto_test_valid_model_dir");
        let _ = fs::create_dir_all(&temp_dir);
        let onnx_file = temp_dir.join("model.onnx");
        let tok_file = temp_dir.join("tokenizer.json");
        let _ = fs::write(&onnx_file, b"dummy onnx");
        let _ = fs::write(&tok_file, b"dummy tokenizer");

        let result = validate_model_dir(&temp_dir);
        assert_eq!(result, Ok(()));

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
