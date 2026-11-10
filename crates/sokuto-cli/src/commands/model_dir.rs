//! # モデルディレクトリ検証および成果物ロードモジュール
//!
//! 推論サーバー起動や量子化、ベンチマーク実行に必要なモデルディレクトリの完全性を検証し、
//! 各種成果物パスの解決や事後較正温度設定のロード処理を集約・提供する。

use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

use sokuto_core::contract::calibration::CalibrationConfig;

use crate::error::CliError;

/// ONNX モデル成果物の標準ファイル名。
pub const MODEL_ONNX_FILENAME: &str = "model.onnx";

/// トークナイザー定義の標準ファイル名。
pub const TOKENIZER_JSON_FILENAME: &str = "tokenizer.json";

/// 事後較正温度設定の標準ファイル名。
pub const CALIBRATION_JSON_FILENAME: &str = "calibration.json";

/// モデルディレクトリ配下の主要成果物パス群。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelArtifactPaths {
    /// ONNX モデルファイルのパス (`model.onnx`)。
    pub model_onnx: PathBuf,
    /// トークナイザー定義ファイルのパス (`tokenizer.json`)。
    pub tokenizer_json: PathBuf,
    /// 較正設定ファイルのパス (`calibration.json`)。
    pub calibration_json: PathBuf,
}

impl ModelArtifactPaths {
    /// モデルディレクトリから各成果物のパスを解決する。
    ///
    /// # 引数
    /// - `model_dir`: モデルディレクトリのパス。
    pub fn resolve(model_dir: &Path) -> Self {
        Self {
            model_onnx: model_dir.join(MODEL_ONNX_FILENAME),
            tokenizer_json: model_dir.join(TOKENIZER_JSON_FILENAME),
            calibration_json: model_dir.join(CALIBRATION_JSON_FILENAME),
        }
    }
}

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
    if !dir.is_dir() {
        return Err(ModelDirError::NotFound(dir.to_path_buf()));
    }

    let onnx_file = dir.join(MODEL_ONNX_FILENAME);
    if !onnx_file.is_file() {
        return Err(ModelDirError::MissingModelFile(onnx_file));
    }

    let tok_file = dir.join(TOKENIZER_JSON_FILENAME);
    if !tok_file.is_file() {
        return Err(ModelDirError::MissingTokenizerFile(tok_file));
    }

    Ok(())
}

/// モデルディレクトリ配下の `calibration.json` をロードする。
///
/// ファイルが存在する場合はパースして返し、存在しない場合はデフォルト設定 ([`CalibrationConfig::default()`]) を返す。
///
/// # 引数
/// - `model_dir`: モデルディレクトリのパス。
///
/// # エラー
/// - ファイル読み込みエラーや JSON パースエラーが発生した場合に [`CliError`] を返す。
pub fn load_calibration_config(model_dir: &Path) -> Result<CalibrationConfig, CliError> {
    let calib_path = model_dir.join(CALIBRATION_JSON_FILENAME);
    if calib_path.is_file() {
        let json = fs::read_to_string(&calib_path)?;
        tracing::info!("較正温度設定をロードしました: {}", calib_path.display());
        let config = CalibrationConfig::from_json_str(&json)?;
        Ok(config)
    } else {
        tracing::info!("較正設定ファイルが存在しないため、デフォルト温度設定を使用します。");
        Ok(CalibrationConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sokuto_core::contract::calibration::BucketSpec;
    use std::fs;

    #[test]
    fn test_validate_model_dir_not_found() {
        let non_existent = PathBuf::from("target/non_existent_model_dir_for_test");
        let result = validate_model_dir(&non_existent);
        assert_eq!(result, Err(ModelDirError::NotFound(non_existent)));
    }

    #[test]
    fn test_validate_model_dir_not_a_directory() {
        let temp_file = std::env::temp_dir().join("sokuto_test_file_not_a_dir");
        let _ = fs::write(&temp_file, b"not a directory");

        let result = validate_model_dir(&temp_file);
        assert_eq!(result, Err(ModelDirError::NotFound(temp_file.clone())));

        let _ = fs::remove_file(&temp_file);
    }

    #[test]
    fn test_validate_model_dir_missing_onnx() {
        let temp_dir = std::env::temp_dir().join("sokuto_test_missing_onnx");
        let _ = fs::create_dir_all(&temp_dir);

        let result = validate_model_dir(&temp_dir);
        let expected_onnx = temp_dir.join(MODEL_ONNX_FILENAME);
        assert_eq!(result, Err(ModelDirError::MissingModelFile(expected_onnx)));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_validate_model_dir_missing_tokenizer() {
        let temp_dir = std::env::temp_dir().join("sokuto_test_missing_tokenizer");
        let _ = fs::create_dir_all(&temp_dir);
        let onnx_file = temp_dir.join(MODEL_ONNX_FILENAME);
        let _ = fs::write(&onnx_file, b"dummy onnx");

        let result = validate_model_dir(&temp_dir);
        let expected_tok = temp_dir.join(TOKENIZER_JSON_FILENAME);
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
        let onnx_file = temp_dir.join(MODEL_ONNX_FILENAME);
        let tok_file = temp_dir.join(TOKENIZER_JSON_FILENAME);
        let _ = fs::write(&onnx_file, b"dummy onnx");
        let _ = fs::write(&tok_file, b"dummy tokenizer");

        let result = validate_model_dir(&temp_dir);
        assert_eq!(result, Ok(()));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_model_artifact_paths_resolve() {
        let model_dir = Path::new("test_models/default");
        let paths = ModelArtifactPaths::resolve(model_dir);
        assert_eq!(paths.model_onnx, model_dir.join("model.onnx"));
        assert_eq!(paths.tokenizer_json, model_dir.join("tokenizer.json"));
        assert_eq!(paths.calibration_json, model_dir.join("calibration.json"));
    }

    #[test]
    fn test_load_calibration_config_fallback_default() {
        let temp_dir = std::env::temp_dir().join("sokuto_test_calib_not_found");
        let _ = fs::create_dir_all(&temp_dir);

        let config = load_calibration_config(&temp_dir).expect("デフォルト設定取得に成功。");
        assert_eq!(config, CalibrationConfig::default());

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_load_calibration_config_from_valid_file() {
        let temp_dir = std::env::temp_dir().join("sokuto_test_calib_valid");
        let _ = fs::create_dir_all(&temp_dir);
        let calib_file = temp_dir.join(CALIBRATION_JSON_FILENAME);

        let valid_json = r#"{
            "version": "1.0",
            "default_temperature": 1.0,
            "temperature_map": {
                "choice": {"3": 1.25},
                "score": {},
                "noul": 1.0
            }
        }"#;
        fs::write(&calib_file, valid_json).expect("テスト用較正ファイル作成に成功。");

        let config = load_calibration_config(&temp_dir).expect("較正設定ロードに成功。");
        assert_eq!(config.version, "1.0");
        assert_eq!(
            config.temperature_map.choice.get(&BucketSpec::Exact(3)),
            Some(&1.25)
        );

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_load_calibration_config_invalid_json() {
        let temp_dir = std::env::temp_dir().join("sokuto_test_calib_invalid");
        let _ = fs::create_dir_all(&temp_dir);
        let calib_file = temp_dir.join(CALIBRATION_JSON_FILENAME);

        fs::write(&calib_file, b"not valid json").expect("不正JSONファイル作成に成功。");

        let result = load_calibration_config(&temp_dir);
        assert!(result.is_err(), "不正JSONのパース失敗を期待。");

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
