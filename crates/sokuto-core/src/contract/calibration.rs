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
    ///
    /// 範囲指定(`Range(start, end)`)の場合は `start <= end` であることを検証し、範囲逆転している場合は `None` を返却する。
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
            && start <= end
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

/// バケット定義マップからバケット仕様と温度のペア一覧を事前パースする。
fn parse_bucket_map(map: &IndexMap<String, f64>) -> Vec<(BucketSpec, f64)> {
    map.iter()
        .filter_map(|(k, &v)| BucketSpec::parse(k).map(|spec| (spec, v)))
        .collect()
}

impl From<RawCalibrationConfig> for CalibrationConfig {
    fn from(raw: RawCalibrationConfig) -> Self {
        let parsed_choice = parse_bucket_map(&raw.temperature_map.choice);
        let parsed_score = parse_bucket_map(&raw.temperature_map.score);
        Self {
            version: raw.version,
            default_temperature: raw.default_temperature,
            parsed_choice,
            parsed_score,
            temperature_map: raw.temperature_map,
            gating_thresholds: raw.gating_thresholds,
        }
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
        let temperature_map = TemperatureMap::default();
        let parsed_choice = parse_bucket_map(&temperature_map.choice);
        let parsed_score = parse_bucket_map(&temperature_map.score);
        Self {
            version: "1.0".to_string(),
            default_temperature: 1.0,
            parsed_choice,
            parsed_score,
            temperature_map,
            gating_thresholds: GatingThresholds::default(),
        }
    }
}

impl CalibrationConfig {
    /// 新規キャリブレーション設定を生成する。
    ///
    /// 引数として渡された `temperature_map` から Choice / Score バケット仕様を即座に事前パースし、
    /// パース済みキャッシュを確定させた状態でインスタンスを構築する。
    pub fn new(version: impl Into<String>, temperature_map: TemperatureMap) -> Self {
        let parsed_choice = parse_bucket_map(&temperature_map.choice);
        let parsed_score = parse_bucket_map(&temperature_map.score);
        Self {
            version: version.into(),
            default_temperature: 1.0,
            parsed_choice,
            parsed_score,
            temperature_map,
            gating_thresholds: GatingThresholds::default(),
        }
    }

    /// バケット定義文字列を事前パースして内部キャッシュに格納する。
    pub fn init_parsed_buckets(&mut self) {
        self.parsed_choice = parse_bucket_map(&self.temperature_map.choice);
        self.parsed_score = parse_bucket_map(&self.temperature_map.score);
    }

    /// デフォルト温度係数を設定して自身を返却する。
    pub fn with_default_temperature(mut self, default_temperature: f64) -> Self {
        self.default_temperature = default_temperature;
        self
    }

    /// 温度テーブルマップを設定し、バケット定義を再パースして自身を返却する。
    pub fn with_temperature_map(mut self, temperature_map: TemperatureMap) -> Self {
        self.temperature_map = temperature_map;
        self.init_parsed_buckets();
        self
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
    ///
    /// デシリアライズ処理(`#[serde(from = "RawCalibrationConfig")]`)により、
    /// 構築時に必ず `BucketSpec` の事前パースが完了する不変条件が保証される。
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
    /// 構築時に確定した事前パース済みの `BucketSpec` を走査し、整数の範囲比較のみで決定する。
    /// 動的文字列パースによるフォールバック処理は行わない。
    /// マッチする定義が存在しない場合は、`default_temperature` を返却する。
    #[inline]
    pub fn get_temperature(&self, question_type: QuestionType, candidate_count: usize) -> f64 {
        match question_type {
            QuestionType::Choice => {
                self.resolve_parsed_bucket(&self.parsed_choice, candidate_count)
            }
            QuestionType::Score => self.resolve_parsed_bucket(&self.parsed_score, candidate_count),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gating::{
        DEFAULT_HIGH_CONFIDENCE_THRESHOLD, DEFAULT_LOW_CONFIDENCE_THRESHOLD,
        DEFAULT_TOP_MARGIN_THRESHOLD,
    };

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

        let config = CalibrationConfig::new(
            "1.0",
            TemperatureMap {
                choice,
                score,
                noul: 0.95,
            },
        );

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
    fn test_parsed_buckets_resolution() {
        let mut choice = IndexMap::new();
        choice.insert("2".to_string(), 1.05);
        choice.insert("3-5".to_string(), 1.15);

        // new 呼び出し時点で事前パースが完了している
        let config = CalibrationConfig::new(
            "1.0",
            TemperatureMap {
                choice,
                score: IndexMap::new(),
                noul: 1.0,
            },
        );

        // 事前パース結果に基づいて即座に温度解決される
        assert_eq!(config.get_temperature(QuestionType::Choice, 2), 1.05);
        assert_eq!(config.get_temperature(QuestionType::Choice, 4), 1.15);
        // マッチしない候補数は default_temperature が返却される
        assert_eq!(config.get_temperature(QuestionType::Choice, 99), 1.0);
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
        assert!(BucketSpec::parse("5-3").is_none());
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
        assert!(BucketSpec::from_str("5-3").is_err());
    }
}
