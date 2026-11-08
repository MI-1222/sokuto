//! # キャリブレーション設定スキーマモジュール
//!
//! Python 学習工程から出力される `calibration.json` のスキーマ定義および、
//! 質問タイプ・候補数に応じた最適温度パラメータ $\tau^*$ の解決ロジックを提供する。

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};
use crate::gating::GatingConfig;

pub use crate::gating::GatingThresholds;
use crate::schema::QuestionType;

/// 候補数バケットに基づく温度係数マップ。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemperatureMap {
    /// Choice 型向けの候補数バケット別温度テーブル(例: "2": 1.05, "3-5": 1.12, "6-10": 1.20, "11+": 1.35)。
    #[serde(default)]
    pub choice: IndexMap<String, f64>,

    /// Score 型向けの評価段階数バケット別温度テーブル(例: "2-5": 1.00, "6-10": 1.08)。
    #[serde(default)]
    pub score: IndexMap<String, f64>,

    /// Noul 型向けの二値判定温度係数(デフォルト: 1.0)。
    #[serde(default = "default_noul_temperature")]
    pub noul: f64,
}

fn default_noul_temperature() -> f64 {
    1.0
}

impl Default for TemperatureMap {
    fn default() -> Self {
        let mut choice = IndexMap::new();
        choice.insert("2".to_string(), 1.0);
        choice.insert("3-5".to_string(), 1.0);
        choice.insert("6-10".to_string(), 1.0);
        choice.insert("11+".to_string(), 1.0);

        let mut score = IndexMap::new();
        score.insert("2-5".to_string(), 1.0);
        score.insert("6-10".to_string(), 1.0);

        Self {
            choice,
            score,
            noul: 1.0,
        }
    }
}

/// 成果物引き渡し用 `calibration.json` のスキーマ構造体。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationConfig {
    /// キャリブレーションスキーマのバージョン(例: "1.0")。
    pub version: String,

    /// マップに該当がない場合に使用するデフォルト温度係数。
    #[serde(default = "default_temperature")]
    pub default_temperature: f64,

    /// 質問プリミティブ別の温度テーブルマップ。
    pub temperature_map: TemperatureMap,

    /// ゲーティング判定用閾値設定。
    #[serde(default)]
    pub gating_thresholds: GatingThresholds,
}

fn default_temperature() -> f64 {
    1.0
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            default_temperature: 1.0,
            temperature_map: TemperatureMap::default(),
            gating_thresholds: GatingThresholds::default(),
        }
    }
}

impl CalibrationConfig {
    /// 新規キャリブレーション設定を生成する。
    pub fn new(version: impl Into<String>, temperature_map: TemperatureMap) -> Self {
        Self {
            version: version.into(),
            default_temperature: 1.0,
            temperature_map,
            gating_thresholds: GatingThresholds::default(),
        }
    }

    /// ゲーティング閾値を設定して自身を返却する。
    pub fn with_gating_thresholds(mut self, gating_thresholds: GatingThresholds) -> Self {
        self.gating_thresholds = gating_thresholds;
        self
    }

    /// 内包されるゲーティング閾値から有効状態の `GatingConfig` を導出する。
    pub fn gating_config(&self) -> GatingConfig {
        self.gating_thresholds.to_gating_config(true)
    }

    /// JSON 文字列からデシリアライズする。
    pub fn from_json_str(json_str: &str) -> Result<Self> {
        serde_json::from_str(json_str).map_err(|e| CoreError::SerializationError {
            message: format!("calibration.json の解析に失敗しました: {e}。"),
        })
    }

    /// JSON 文字列へシリアライズする。
    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| CoreError::SerializationError {
            message: format!("calibration.json への変換に失敗しました: {e}。"),
        })
    }

    /// 質問種別と候補数(または評価段階数)に基づいて最適温度係数 $\tau^*$ を解決する。
    ///
    /// バケットパターン例:
    /// - `"2"`: 候補数がちょうど 2 の場合に一致。
    /// - `"3-5"`: 候補数が 3 以上 5 以下の場合に一致。
    /// - `"11+"`: 候補数が 11 以上の場合に一致。
    ///
    /// マッチする定義が存在しない場合は、`default_temperature` を返却する。
    pub fn get_temperature(&self, question_type: QuestionType, candidate_count: usize) -> f64 {
        match question_type {
            QuestionType::Choice => {
                self.resolve_bucket(&self.temperature_map.choice, candidate_count)
            }
            QuestionType::Score => {
                self.resolve_bucket(&self.temperature_map.score, candidate_count)
            }
            QuestionType::Noul => self.temperature_map.noul,
        }
    }

    /// バケットマップを走査し、候補数に一致する温度を探索する。
    fn resolve_bucket(&self, table: &IndexMap<String, f64>, count: usize) -> f64 {
        for (bucket_expr, &temperature) in table {
            if matches_bucket(bucket_expr, count) {
                return temperature;
            }
        }
        self.default_temperature
    }
}

/// バケット定義文字列(例: "2", "3-5", "11+")と候補数のマッチ判定を行う。
fn matches_bucket(expr: &str, count: usize) -> bool {
    let trimmed = expr.trim();
    if let Ok(exact) = trimmed.parse::<usize>() {
        return exact == count;
    }

    if let Some(lower_str) = trimmed.strip_suffix('+')
        && let Ok(lower) = lower_str.trim().parse::<usize>()
    {
        return count >= lower;
    }

    if let Some((start_str, end_str)) = trimmed.split_once('-')
        && let (Ok(start), Ok(end)) = (
            start_str.trim().parse::<usize>(),
            end_str.trim().parse::<usize>(),
        )
    {
        return count >= start && count <= end;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gating::{
        DEFAULT_HIGH_CONFIDENCE_THRESHOLD, DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        DEFAULT_TOP_MARGIN_THRESHOLD,
    };

    #[test]
    fn test_bucket_matching() {
        assert!(matches_bucket("2", 2));
        assert!(!matches_bucket("2", 3));

        assert!(matches_bucket("3-5", 3));
        assert!(matches_bucket("3-5", 4));
        assert!(matches_bucket("3-5", 5));
        assert!(!matches_bucket("3-5", 2));
        assert!(!matches_bucket("3-5", 6));

        assert!(matches_bucket("11+", 11));
        assert!(matches_bucket("11+", 50));
        assert!(!matches_bucket("11+", 10));
    }

    #[test]
    fn test_calibration_round_trip() {
        let mut choice = IndexMap::new();
        choice.insert("2".to_string(), 1.05);
        choice.insert("3-5".to_string(), 1.15);
        choice.insert("6-10".to_string(), 1.25);
        choice.insert("11+".to_string(), 1.40);

        let mut score = IndexMap::new();
        score.insert("2-5".to_string(), 1.02);
        score.insert("6-10".to_string(), 1.10);

        let config = CalibrationConfig {
            version: "1.0".to_string(),
            default_temperature: 1.0,
            temperature_map: TemperatureMap {
                choice,
                score,
                noul: 0.95,
            },
            gating_thresholds: GatingThresholds::default(),
        };

        let json = config.to_json_string().unwrap();
        let loaded = CalibrationConfig::from_json_str(&json).unwrap();
        assert_eq!(config, loaded);

        // 温度取得のテスト
        assert_eq!(loaded.get_temperature(QuestionType::Choice, 2), 1.05);
        assert_eq!(loaded.get_temperature(QuestionType::Choice, 4), 1.15);
        assert_eq!(loaded.get_temperature(QuestionType::Choice, 8), 1.25);
        assert_eq!(loaded.get_temperature(QuestionType::Choice, 20), 1.40);

        assert_eq!(loaded.get_temperature(QuestionType::Score, 3), 1.02);
        assert_eq!(loaded.get_temperature(QuestionType::Score, 7), 1.10);

        assert_eq!(loaded.get_temperature(QuestionType::Noul, 1), 0.95);

        // ゲーティング設定変換のテスト
        let gating_cfg = loaded.gating_config();
        assert!(gating_cfg.enabled);
        assert!((gating_cfg.high_threshold - DEFAULT_HIGH_CONFIDENCE_THRESHOLD).abs() < 1e-9);
        assert!((gating_cfg.low_threshold - DEFAULT_LOW_CONFIDENCE_THRESHOLD).abs() < 1e-9);
        assert!((gating_cfg.top_margin_threshold - DEFAULT_TOP_MARGIN_THRESHOLD).abs() < 1e-9);
    }
}
