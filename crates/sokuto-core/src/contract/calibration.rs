//! # キャリブレーション設定スキーマモジュール
//!
//! Python 学習工程から出力される `calibration.json` のスキーマ定義および、
//! 質問タイプ・候補数に応じた最適温度パラメータ $\tau^*$ の解決ロジックを提供する。

use std::fmt;
use std::str::FromStr;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};
use crate::gating::GatingConfig;

pub use crate::gating::GatingThresholds;
use crate::schema::QuestionType;

/// 温度バケットの仕様定義。
///
/// 文字列によるバケット表現(例: `"2"`, `"3-5"`, `"11+"`)をロード時に事前パースし、
/// 推論ホットパスにおいて整数の範囲比較のみで温度を解決できるようにする。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BucketSpec {
    /// 候補数が指定値と完全に一致する場合(例: `"2"` -> `Exact(2)`)。
    Exact(usize),
    /// 候補数が閉区間 `[start, end]` に含まれる場合(例: `"3-5"` -> `Range(3, 5)`)。
    Range(usize, usize),
    /// 候補数が下限値以上である場合(例: `"11+"` -> `AtLeast(11)`)。
    AtLeast(usize),
}

impl BucketSpec {
    /// 文字列スライスからバケット仕様をパースする。
    pub fn parse(expr: &str) -> Option<Self> {
        let trimmed = expr.trim();
        if trimmed.is_empty() || trimmed.starts_with('+') {
            return None;
        }

        if let Ok(exact) = trimmed.parse::<usize>() {
            return Some(Self::Exact(exact));
        }

        if let Some(lower_str) = trimmed.strip_suffix('+')
            && let Ok(lower) = lower_str.trim().parse::<usize>()
        {
            return Some(Self::AtLeast(lower));
        }

        if let Some((start_str, end_str)) = trimmed.split_once('-')
            && let (Ok(start), Ok(end)) = (
                start_str.trim().parse::<usize>(),
                end_str.trim().parse::<usize>(),
            )
        {
            return Some(Self::Range(start, end));
        }

        None
    }

    /// 候補数または段階数 `count` がバケット定義に合致するか判定する。
    #[inline]
    pub fn matches(&self, count: usize) -> bool {
        match *self {
            Self::Exact(exact) => count == exact,
            Self::Range(start, end) => count >= start && count <= end,
            Self::AtLeast(lower) => count >= lower,
        }
    }
}

impl FromStr for BucketSpec {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s).ok_or_else(|| CoreError::SerializationError {
            message: format!("不正な温度バケット定義形式です: '{s}'。'2', '3-5', '11+' などの形式を指定してください。"),
        })
    }
}

impl fmt::Display for BucketSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(n) => write!(f, "{n}"),
            Self::Range(start, end) => write!(f, "{start}-{end}"),
            Self::AtLeast(lower) => write!(f, "{lower}+"),
        }
    }
}

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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "RawCalibrationConfig")]
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

    /// 事前パースされた Choice バケット定義。
    #[serde(skip)]
    pub parsed_choice: Vec<(BucketSpec, f64)>,

    /// 事前パースされた Score バケット定義。
    #[serde(skip)]
    pub parsed_score: Vec<(BucketSpec, f64)>,
}

#[derive(Deserialize)]
struct RawCalibrationConfig {
    version: String,
    #[serde(default = "default_temperature")]
    default_temperature: f64,
    temperature_map: TemperatureMap,
    #[serde(default)]
    gating_thresholds: GatingThresholds,
}

impl From<RawCalibrationConfig> for CalibrationConfig {
    fn from(raw: RawCalibrationConfig) -> Self {
        let mut config = Self {
            version: raw.version,
            default_temperature: raw.default_temperature,
            temperature_map: raw.temperature_map,
            gating_thresholds: raw.gating_thresholds,
            parsed_choice: Vec::new(),
            parsed_score: Vec::new(),
        };
        config.init_parsed_buckets();
        config
    }
}

impl PartialEq for CalibrationConfig {
    fn eq(&self, other: &Self) -> bool {
        self.version == other.version
            && (self.default_temperature - other.default_temperature).abs() < f64::EPSILON
            && self.temperature_map == other.temperature_map
            && self.gating_thresholds == other.gating_thresholds
    }
}

fn default_temperature() -> f64 {
    1.0
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        let mut config = Self {
            version: "1.0".to_string(),
            default_temperature: 1.0,
            temperature_map: TemperatureMap::default(),
            gating_thresholds: GatingThresholds::default(),
            parsed_choice: Vec::new(),
            parsed_score: Vec::new(),
        };
        config.init_parsed_buckets();
        config
    }
}

impl CalibrationConfig {
    /// 新規キャリブレーション設定を生成する。
    pub fn new(version: impl Into<String>, temperature_map: TemperatureMap) -> Self {
        let mut config = Self {
            version: version.into(),
            default_temperature: 1.0,
            temperature_map,
            gating_thresholds: GatingThresholds::default(),
            parsed_choice: Vec::new(),
            parsed_score: Vec::new(),
        };
        config.init_parsed_buckets();
        config
    }

    /// バケット定義文字列を事前パースして内部キャッシュに格納する。
    pub fn init_parsed_buckets(&mut self) {
        self.parsed_choice = self
            .temperature_map
            .choice
            .iter()
            .filter_map(|(k, &v)| BucketSpec::parse(k).map(|spec| (spec, v)))
            .collect();

        self.parsed_score = self
            .temperature_map
            .score
            .iter()
            .filter_map(|(k, &v)| BucketSpec::parse(k).map(|spec| (spec, v)))
            .collect();
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
        let mut config: Self =
            serde_json::from_str(json_str).map_err(|e| CoreError::SerializationError {
                message: format!("calibration.json の解析に失敗しました: {e}。"),
            })?;
        config.init_parsed_buckets();
        Ok(config)
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
    /// 事前パース済みの `BucketSpec` を走査し、整数の範囲比較のみで決定する。
    /// マッチする定義が存在しない場合は、`default_temperature` を返却する。
    #[inline]
    pub fn get_temperature(&self, question_type: QuestionType, candidate_count: usize) -> f64 {
        match question_type {
            QuestionType::Choice => {
                if !self.parsed_choice.is_empty() {
                    self.resolve_parsed_bucket(&self.parsed_choice, candidate_count)
                } else {
                    self.resolve_bucket(&self.temperature_map.choice, candidate_count)
                }
            }
            QuestionType::Score => {
                if !self.parsed_score.is_empty() {
                    self.resolve_parsed_bucket(&self.parsed_score, candidate_count)
                } else {
                    self.resolve_bucket(&self.temperature_map.score, candidate_count)
                }
            }
            QuestionType::Noul => self.temperature_map.noul,
        }
    }

    /// 事前パース済みバケットスライスを走査し、候補数に一致する温度を探索する。
    #[inline]
    fn resolve_parsed_bucket(&self, table: &[(BucketSpec, f64)], count: usize) -> f64 {
        for &(spec, temperature) in table {
            if spec.matches(count) {
                return temperature;
            }
        }
        self.default_temperature
    }

    /// バケットマップを走査し、候補数に一致する温度を探索する (フォールバック用)。
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
pub fn matches_bucket(expr: &str, count: usize) -> bool {
    BucketSpec::parse(expr).is_some_and(|spec| spec.matches(count))
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
            parsed_choice: Vec::new(),
            parsed_score: Vec::new(),
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

        // 事前パースキャッシュが正しく生成されていることを検証
        assert_eq!(loaded.parsed_choice.len(), 4);
        assert_eq!(loaded.parsed_choice[0], (BucketSpec::Exact(2), 1.05));
        assert_eq!(loaded.parsed_choice[1], (BucketSpec::Range(3, 5), 1.15));
        assert_eq!(loaded.parsed_choice[2], (BucketSpec::Range(6, 10), 1.25));
        assert_eq!(loaded.parsed_choice[3], (BucketSpec::AtLeast(11), 1.40));

        assert_eq!(loaded.parsed_score.len(), 2);
        assert_eq!(loaded.parsed_score[0], (BucketSpec::Range(2, 5), 1.02));
        assert_eq!(loaded.parsed_score[1], (BucketSpec::Range(6, 10), 1.10));

        // ゲーティング設定変換のテスト
        let gating_cfg = loaded.gating_config();
        assert!(gating_cfg.enabled);
        assert!(
            (gating_cfg.thresholds.high_threshold - DEFAULT_HIGH_CONFIDENCE_THRESHOLD).abs() < 1e-9
        );
        assert!(
            (gating_cfg.thresholds.low_threshold - DEFAULT_LOW_CONFIDENCE_THRESHOLD).abs() < 1e-9
        );
        assert!(
            (gating_cfg.thresholds.top_margin_threshold - DEFAULT_TOP_MARGIN_THRESHOLD).abs()
                < 1e-9
        );
    }

    #[test]
    fn test_bucket_spec_parse_and_matches() {
        // Exact
        let exact = BucketSpec::parse(" 2 ").expect("Exact パース成功。");
        assert_eq!(exact, BucketSpec::Exact(2));
        assert!(exact.matches(2));
        assert!(!exact.matches(1));
        assert!(!exact.matches(3));

        // Range
        let range = BucketSpec::parse(" 3 - 5 ").expect("Range パース成功。");
        assert_eq!(range, BucketSpec::Range(3, 5));
        assert!(range.matches(3));
        assert!(range.matches(4));
        assert!(range.matches(5));
        assert!(!range.matches(2));
        assert!(!range.matches(6));

        // AtLeast
        let at_least = BucketSpec::parse(" 11+ ").expect("AtLeast パース成功。");
        assert_eq!(at_least, BucketSpec::AtLeast(11));
        assert!(at_least.matches(11));
        assert!(at_least.matches(100));
        assert!(!at_least.matches(10));

        // 不正形式
        assert!(BucketSpec::parse("abc").is_none());
        assert!(BucketSpec::parse("3-").is_none());
        assert!(BucketSpec::parse("+5").is_none());
        assert!(BucketSpec::parse("").is_none());
    }

    #[test]
    fn test_bucket_spec_from_str_and_display() {
        use std::str::FromStr;

        let spec = BucketSpec::from_str("3-5").expect("FromStr 成功。");
        assert_eq!(spec, BucketSpec::Range(3, 5));
        assert_eq!(format!("{spec}"), "3-5");

        let exact = BucketSpec::from_str("42").expect("FromStr 成功。");
        assert_eq!(exact, BucketSpec::Exact(42));
        assert_eq!(format!("{exact}"), "42");

        let at_least = BucketSpec::from_str("10+").expect("FromStr 成功。");
        assert_eq!(at_least, BucketSpec::AtLeast(10));
        assert_eq!(format!("{at_least}"), "10+");

        assert!(BucketSpec::from_str("invalid").is_err());
    }

    #[test]
    fn test_uninitialized_fallback() {
        // 手動構築で parsed_choice が空のインスタンスでもフォールバックで正常に温度解決できることを検証
        let mut choice = IndexMap::new();
        choice.insert("2".to_string(), 1.05);
        choice.insert("3-5".to_string(), 1.15);

        let config = CalibrationConfig {
            version: "1.0".to_string(),
            default_temperature: 1.0,
            temperature_map: TemperatureMap {
                choice,
                score: IndexMap::new(),
                noul: 1.0,
            },
            gating_thresholds: GatingThresholds::default(),
            parsed_choice: Vec::new(),
            parsed_score: Vec::new(),
        };

        // parsed_choice が空でも動的マッチで解決される
        assert_eq!(config.get_temperature(QuestionType::Choice, 2), 1.05);
        assert_eq!(config.get_temperature(QuestionType::Choice, 4), 1.15);
        assert_eq!(config.get_temperature(QuestionType::Choice, 99), 1.0);
    }
}
