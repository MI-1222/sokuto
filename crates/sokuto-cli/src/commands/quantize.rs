//! # ONNX INT8 PTQ 量子化コマンド
//!
//! FP32 ONNX モデルに対して動的 INT8 量子化 (Post-Training Quantization) を適用し、
//! デシジョンヘッド (OptionGatherLayer, out_proj) の FP32 精度を維持しつつ
//! モデルサイズ半減と低メモリ推論を実現する。

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use sokuto_runtime::engine::{InferenceEngine, SessionConfig};
use sokuto_runtime::tokenizer::JevTokenizer;

use crate::commands::model_dir::validate_model_dir;
use crate::error::CliError;

/// 量子化実行オプション構造体。
#[derive(Debug, Clone, clap::Args, PartialEq, Eq)]
pub struct QuantizeArgs {
    /// 元モデルファイル群が存在するディレクトリパス。
    #[arg(short, long, default_value = "models/default")]
    pub model_dir: PathBuf,

    /// 量子化後成果物バンドルの出力先ディレクトリパス。
    #[arg(short, long, default_value = "models/quantized")]
    pub output_dir: PathBuf,

    /// チャンネル単位で重みを量子化する (デフォルト: 有効)。
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub per_channel: bool,

    /// FP32 モデルとのパリティ検証を実行する (デフォルト: 有効)。
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub verify: bool,

    /// Python 量子化スクリプトのパス (省略時はリポジトリルートを自動探索)。
    #[arg(long, env = "SOKUTO_QUANTIZE_SCRIPT_PATH")]
    pub script_path: Option<PathBuf>,
}

/// 量子化結果メタデータ。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizeMetadata {
    /// タイムスタンプ。
    pub timestamp: String,
    /// 量子化結果詳細。
    pub quantize_result: QuantizeResultDetails,
    /// チャンネル単位量子化フラグ。
    pub per_channel: bool,
}

/// 量子化結果の詳細サマリー。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizeResultDetails {
    /// 入力モデルパス。
    pub input_model_path: String,
    /// 出力モデルパス。
    pub output_model_path: String,
    /// バンドル出力先ディレクトリ。
    pub bundle_dir: String,
    /// 元のファイルサイズ (バイト)。
    pub original_size_bytes: u64,
    /// 量子化後のファイルサイズ (バイト)。
    pub quantized_size_bytes: u64,
    /// 圧縮率。
    pub compression_ratio: f64,
    /// パリティ検証合否。
    pub parity_passed: bool,
    /// Top-1 決定一致率。
    pub top1_agreement_rate: f64,
    /// 最大絶対誤差。
    pub max_abs_error: f64,
    /// 平均絶対誤差。
    pub mean_abs_error: f64,
}

/// ONNX INT8 PTQ 量子化コマンドを実行する。
pub fn run_quantize(args: QuantizeArgs) -> Result<(), CliError> {
    // 1. 入力モデルディレクトリの存在と必須ファイルチェック
    validate_model_dir(&args.model_dir)?;

    eprintln!(
        "ONNX INT8 PTQ 量子化を開始します (入力: {}, 出力先: {})...",
        args.model_dir.display(),
        args.output_dir.display()
    );

    // 2. 出力先ディレクトリの確保
    fs::create_dir_all(&args.output_dir)?;

    // 3. train 側の Python 量子化スクリプトを実行 (uv run)
    let (quantize_script, project_dir) =
        resolve_quantize_script(args.script_path.as_deref(), &args.model_dir)?;

    let model_dir_abs = fs::canonicalize(&args.model_dir)?;
    let output_dir_abs = if args.output_dir.is_absolute() {
        args.output_dir.clone()
    } else {
        std::env::current_dir()?.join(&args.output_dir)
    };

    let mut cmd = Command::new("uv");
    cmd.arg("run")
        .arg("--project")
        .arg(&project_dir)
        .arg("python")
        .arg(&quantize_script)
        .arg("--model-dir")
        .arg(&model_dir_abs)
        .arg("--output-dir")
        .arg(&output_dir_abs);

    if args.per_channel {
        cmd.arg("--per-channel");
    } else {
        cmd.arg("--no-per-channel");
    }

    if args.verify {
        cmd.arg("--verify");
    } else {
        cmd.arg("--no-verify");
    }

    cmd.current_dir(&project_dir);

    eprintln!("Python 量子化パイプラインを起動中...");
    let output = match cmd.output() {
        Ok(out) => out,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Err(CliError::Quantize(
                "外部コマンド 'uv' の実行に失敗しました。uv がインストールされていないか、PATH に通っていません。Python 環境と uv を確認してください。".into(),
            ));
        }
        Err(err) => {
            return Err(CliError::Quantize(format!(
                "量子化サブプロセスの起動に失敗しました: {err}。"
            )));
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        eprintln!("量子化スクリプトの実行に失敗しました:\nSTDERR:\n{stderr}\nSTDOUT:\n{stdout}");
        return Err(CliError::Quantize(
            "量子化サブプロセスの実行に失敗しました。".into(),
        ));
    }

    // 4. 出力先成果物バンドルの完全性検証
    let quantized_onnx = output_dir_abs.join("model.onnx");
    let quantized_tok = output_dir_abs.join("tokenizer.json");

    if !quantized_onnx.exists() || !quantized_tok.exists() {
        return Err(CliError::Quantize(format!(
            "量子化成果物ファイルが不足しています: {} または {}。",
            quantized_onnx.display(),
            quantized_tok.display()
        )));
    }

    // 5. Rust ランタイムによるロード可能性の検証 (スモークテスト)
    eprintln!("Rust ランタイムによる量子化モデルの整合性チェックを実行中...");
    let test_engine = InferenceEngine::new(&quantized_onnx, SessionConfig::cpu_only())?;
    let _test_tok = JevTokenizer::from_file(&quantized_tok)?;
    drop(test_engine);

    // 6. メタデータの読み込みとサマリー表示
    let meta_file = output_dir_abs.join("quantize_metadata.json");
    if meta_file.exists() {
        let meta_str = fs::read_to_string(&meta_file)?;
        if let Ok(meta) = serde_json::from_str::<QuantizeMetadata>(&meta_str) {
            let res = &meta.quantize_result;
            println!();
            println!("==========================================================================");
            println!(" Sokuto ONNX INT8 PTQ 量子化サマリー");
            println!("==========================================================================");
            println!("成果物ディレクトリ : {}", args.output_dir.display());
            println!(
                "ファイルサイズ     : {:.1} MB -> {:.1} MB (削減率: {:.1}%)",
                res.original_size_bytes as f64 / 1_048_576.0,
                res.quantized_size_bytes as f64 / 1_048_576.0,
                res.compression_ratio * 100.0
            );
            println!(
                "Top-1 決定一致率   : {:.1}% ({})",
                res.top1_agreement_rate * 100.0,
                if res.parity_passed {
                    "合格"
                } else {
                    "警告"
                }
            );
            println!("最大絶対誤差 (Max) : {:.4}", res.max_abs_error);
            println!("平均絶対誤差 (Mean): {:.4}", res.mean_abs_error);
            println!("==========================================================================");
            println!();
        }
    } else {
        println!("量子化が正常に完了しました: {}", args.output_dir.display());
    }

    Ok(())
}

/// 量子化スクリプトおよび uv プロジェクトディレクトリを解決する。
pub fn resolve_quantize_script(
    script_path: Option<&Path>,
    model_dir: &Path,
) -> Result<(PathBuf, PathBuf), CliError> {
    if let Some(custom_script) = script_path {
        if !custom_script.exists() {
            return Err(CliError::Quantize(format!(
                "指定された量子化スクリプトが見つかりません: {}。",
                custom_script.display()
            )));
        }
        let script_abs = fs::canonicalize(custom_script)?;
        let project_dir = find_pyproject_root(&script_abs)
            .unwrap_or_else(|| script_abs.parent().unwrap_or(&script_abs).to_path_buf());
        Ok((script_abs, project_dir))
    } else {
        let project_root = find_project_root(model_dir)?;
        let train_dir = project_root.join("train");

        if !train_dir.exists() {
            return Err(CliError::Quantize(format!(
                "開発用リポジトリ内の train ディレクトリが見つかりません: {}。リポジトリ外から実行する場合は --script-path オプションで量子化スクリプトのパスを明示的に指定してください。",
                train_dir.display()
            )));
        }

        let quantize_script = train_dir.join("export").join("quantize.py");
        if !quantize_script.exists() {
            return Err(CliError::Quantize(format!(
                "開発用リポジトリ内の量子化スクリプトが見つかりません: {}。--script-path オプションで量子化スクリプトのパスを明示的に指定してください。",
                quantize_script.display()
            )));
        }

        Ok((quantize_script, train_dir))
    }
}

/// スクリプトパスの祖先から pyproject.toml を持つディレクトリを探索する。
fn find_pyproject_root(start_path: &Path) -> Option<PathBuf> {
    let mut current = if start_path.is_dir() {
        start_path.to_path_buf()
    } else {
        start_path.parent()?.to_path_buf()
    };

    loop {
        if current.join("pyproject.toml").exists() {
            return Some(current);
        }
        if let Some(parent) = current.parent() {
            current = parent.to_path_buf();
        } else {
            break;
        }
    }
    None
}

/// プロジェクトルートディレクトリを探索する。
fn find_project_root(start_dir: &Path) -> Result<PathBuf, CliError> {
    let find_root = |mut current: PathBuf| {
        loop {
            if current.join("Cargo.toml").exists() && current.join("crates").exists() {
                return Some(current);
            }
            if let Some(parent) = current.parent() {
                current = parent.to_path_buf();
            } else {
                return None;
            }
        }
    };

    let cwd = std::env::current_dir()?;
    let start_path = if start_dir.is_absolute() {
        start_dir.to_path_buf()
    } else {
        cwd.join(start_dir)
    };

    for candidate in [start_path, cwd] {
        if let Some(root) = find_root(candidate) {
            return Ok(root);
        }
    }

    Err(CliError::Quantize(
        "開発用リポジトリルート (Cargo.toml, crates/, train/ が存在する場所) を特定できませんでした。\
        'sokuto quantize' は開発用リポジトリ内の Python パイプライン (train/export/quantize.py) を利用するコマンドです。\
        リポジトリ外から実行する場合は '--script-path <PATH>' オプションで量子化スクリプトのパスを明示的に指定してください。"
            .into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_resolve_quantize_script_not_found() {
        let non_existent = Path::new("non_existent_dir/custom_quantize.py");
        let result = resolve_quantize_script(Some(non_existent), Path::new("models/default"));
        assert!(
            result.is_err(),
            "存在しないスクリプトパスを指定した場合はエラーを返すこと。"
        );
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("指定された量子化スクリプトが見つかりません"),
            "エラーメッセージにスクリプト未検出の旨が含まれていること。"
        );
    }

    #[test]
    fn test_find_pyproject_root() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let root = find_project_root(manifest_dir).expect("プロジェクトルートの特定に成功。");
        let train_pyproject = root.join("train").join("export").join("quantize.py");
        assert!(
            train_pyproject.exists(),
            "train/export/quantize.py が存在すること。"
        );
        let found = find_pyproject_root(&train_pyproject);
        assert!(
            found.is_some(),
            "train/export/quantize.py から pyproject.toml を検出できること。"
        );
        assert!(
            found.unwrap().ends_with("train"),
            "検出されたプロジェクトルートが train であること。"
        );
    }

    #[test]
    fn test_find_project_root() {
        let cwd = std::env::current_dir().expect("カレントディレクトリの取得に成功。");
        let root = find_project_root(&cwd);
        assert!(
            root.is_ok(),
            "リポジトリ内のカレントディレクトリからプロジェクトルートを特定できること。"
        );
        let root_path = root.unwrap();
        assert!(
            root_path.join("Cargo.toml").exists(),
            "特定されたルートに Cargo.toml が存在すること。"
        );
    }
}
