//! # 確信度ゲーティングおよび3系統ルーティングモジュール
//!
//! 事後較正(RLCD + 最適温度スケーリング $\tau^*$)を経た実効確信度に基づき、
//! 決定論的に 3 系統のアクションへ分岐する安全制御ロジックを提供する。
//!
//! ## 3系統ルーティングの判定基準
//! - **`AutoExecute` ($\ge 0.70$ & Margin $\ge 0.15$)**: 自動実行(高信頼・人手や大型LLMの介在なし)。
//! - **`ConfirmOrEscalate` ($0.35 \dots 0.70$ または Margin $< 0.15$)**: 確認/二次検証要求(System 2 LLMへの委託または人手確認)。
//! - **`Fallback` ($< 0.35$)**: 安全弁フォールバック(情報不足・一様迷い時のデフォルト動作・棄却)。

use std::fmt::{self, Write};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};
use crate::math::{normalized_entropy, normalized_entropy_iter, top_margin_iter};
use crate::schema::{Answer, AnswerPayload, Question, QuestionType};

/// デフォルトの高確信度閾値 (自動実行境界)。
pub const DEFAULT_HIGH_CONFIDENCE_THRESHOLD: f64 = 0.70;

/// デフォルトの中確信度下限閾値 (確認・二次検証要求境界)。
pub const DEFAULT_LOW_CONFIDENCE_THRESHOLD: f64 = 0.35;

/// デフォルトの上位 2 候補確率マージン閾値。
pub const DEFAULT_TOP_MARGIN_THRESHOLD: f64 = 0.15;

/// デフォルトの OOD 自由エネルギー棄却閾値 (正規化自由エネルギー基準)。
///
/// Choice 型の正規化自由エネルギー $E_{\text{norm}}(x; C)$ がこの値を超過した場合、
/// 未定義カテゴリまたは該当なし入力 (OOD) と判定して `Fallback` へ短絡ルーティングする。
pub const DEFAULT_ENERGY_THRESHOLD: f64 = -1.0;

/// 浮動小数点比較時の微小許容誤差。
const EPSILON: f64 = 1e-9;

/// ゲーティング判定に基づく 3 系統のアクション種別。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "openapi", schema(rename_all = "snake_case"))]
pub enum DecisionRoute {
    /// 自動実行: モデルの確信度が極めて高く、人手介在なしに後続処理を実行する。
    AutoExecute,
    /// 確認/二次検証要求: 候補間で迷いが生じており、System 2 (大型LLM) やユーザー確認を求める。
    ConfirmOrEscalate,
    /// 安全弁フォールバック: 信頼性が著しく低く、安全なデフォルト値の採用や人手転送を行う。
    Fallback,
}

impl DecisionRoute {
    /// ルーティング文字列スライスを取得する。
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::AutoExecute => "auto_execute",
            Self::ConfirmOrEscalate => "confirm_or_escalate",
            Self::Fallback => "fallback",
        }
    }
}

impl fmt::Display for DecisionRoute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// デフォルトの真偽値 (true)。
fn default_true() -> bool {
    true
}

fn default_high_threshold() -> f64 {
    DEFAULT_HIGH_CONFIDENCE_THRESHOLD
}

fn default_low_threshold() -> f64 {
    DEFAULT_LOW_CONFIDENCE_THRESHOLD
}

fn default_top_margin_threshold() -> f64 {
    DEFAULT_TOP_MARGIN_THRESHOLD
}

fn default_energy_threshold() -> f64 {
    DEFAULT_ENERGY_THRESHOLD
}

/// 確信度ゲーティングおよび OOD 判定で用いる各種閾値パラメータ。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct GatingThresholds {
    /// 自動実行 (AutoExecute) と判定するための確信度下限値 (デフォルト: 0.70)。
    #[serde(default = "default_high_threshold")]
    pub high_threshold: f64,

    /// 確認/エスカレーション (ConfirmOrEscalate) と判定するための確信度下限値 (デフォルト: 0.35)。
    #[serde(default = "default_low_threshold")]
    pub low_threshold: f64,

    /// 上位 2 候補の最小確率マージン閾値 (デフォルト: 0.15)。
    ///
    /// 確信度が `high_threshold` 以上であっても、1位と2位の確率差がこの値を下回る場合は
    /// 選択肢間で拮抗が生じていると判断し、強制的に `ConfirmOrEscalate` へ降格する。
    #[serde(default = "default_top_margin_threshold")]
    pub top_margin_threshold: f64,

    /// Energy-based OOD 安全弁を有効化するかどうか (デフォルト: true)。
    #[serde(default = "default_true")]
    pub ood_enabled: bool,

    /// OOD 判定用の正規化自由エネルギー閾値 (デフォルト: -1.0)。
    ///
    /// Choice 型において正規化自由エネルギー $E_{\text{norm}}$ がこの値を超過した場合、
    /// 未定義カテゴリ・該当なしとして強制的に `Fallback` へ降格する。
    #[serde(default = "default_energy_threshold")]
    pub energy_threshold: f64,

    /// OOD 自由エネルギー算出専用の温度パラメータ。
    ///
    /// Softmax 確率較正温度とは独立に、低発火・高コントラストな自由エネルギーを算出するために使用する。
    /// 未指定時はモデル成果物のキャリブレーション設定にフォールバックする。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy_temperature: Option<f64>,

    /// ハイブリッド OOD 判定用の緩和正規化自由エネルギー閾値 (任意)。
    ///
    /// 有効化する場合は [`Self::ood_min_confidence`] および [`Self::ood_min_margin`] と合わせて
    /// 3 項目すべてを指定し、かつ [`Self::energy_threshold`] 以下の有限実数を設定する必要がある。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loose_energy_threshold: Option<f64>,

    /// ハイブリッド OOD 判定用の最小確信度閾値 (任意)。
    ///
    /// [`Self::loose_energy_threshold`] および [`Self::ood_min_margin`] と同時に指定する。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ood_min_confidence: Option<f64>,

    /// ハイブリッド OOD 判定用の最小マージン閾値 (任意)。
    ///
    /// [`Self::loose_energy_threshold`] および [`Self::ood_min_confidence`] と同時に指定する。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ood_min_margin: Option<f64>,
}

impl Default for GatingThresholds {
    fn default() -> Self {
        Self {
            high_threshold: DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
            low_threshold: DEFAULT_LOW_CONFIDENCE_THRESHOLD,
            top_margin_threshold: DEFAULT_TOP_MARGIN_THRESHOLD,
            ood_enabled: true,
            energy_threshold: DEFAULT_ENERGY_THRESHOLD,
            energy_temperature: None,
            loose_energy_threshold: None,
            ood_min_confidence: None,
            ood_min_margin: None,
        }
    }
}

impl GatingThresholds {
    /// ゲーティング判定設定構造体へ変換する。
    ///
    /// # 引数
    /// - `enabled`: ゲーティング処理を有効化するかどうかの真偽値。
    ///
    /// # 戻り値
    /// - 閾値が反映された `GatingConfig`。
    pub fn to_gating_config(&self, enabled: bool) -> GatingConfig {
        GatingConfig {
            enabled,
            thresholds: self.clone(),
        }
    }

    /// 閾値設定値の数学的妥当性を検証する。
    ///
    /// - 確信度・マージン関連閾値が `[0.0, 1.0]` に収まり、`low_threshold <= high_threshold` であることを検証する。
    /// - `energy_threshold` が有限実数であり、`energy_temperature` 指定時は正の有限実数であることを検証する。
    /// - `loose_energy_threshold` 指定時は有限実数かつ `loose_energy_threshold <= energy_threshold` であることを検証する。
    /// - ハイブリッド OOD 閾値 (`loose_energy_threshold`, `ood_min_confidence`, `ood_min_margin`) は、
    ///   すべて指定またはすべて未指定のいずれかであることを検証する。
    pub fn validate(&self) -> Result<()> {
        if self.high_threshold < 0.0 || self.high_threshold > 1.0 {
            return Err(CoreError::InvalidGatingConfig {
                message: format!(
                    "high_threshold ({}) は 0.0〜1.0 の範囲である必要があります。",
                    self.high_threshold
                ),
            });
        }
        if self.low_threshold < 0.0 || self.low_threshold > 1.0 {
            return Err(CoreError::InvalidGatingConfig {
                message: format!(
                    "low_threshold ({}) は 0.0〜1.0 の範囲である必要があります。",
                    self.low_threshold
                ),
            });
        }
        if self.low_threshold > self.high_threshold {
            return Err(CoreError::InvalidGatingConfig {
                message: format!(
                    "low_threshold ({}) は high_threshold ({}) 以下である必要があります。",
                    self.low_threshold, self.high_threshold
                ),
            });
        }
        if self.top_margin_threshold < 0.0 || self.top_margin_threshold > 1.0 {
            return Err(CoreError::InvalidGatingConfig {
                message: format!(
                    "top_margin_threshold ({}) は 0.0〜1.0 の範囲である必要があります。",
                    self.top_margin_threshold
                ),
            });
        }
        if !self.energy_threshold.is_finite() {
            return Err(CoreError::InvalidGatingConfig {
                message: format!(
                    "energy_threshold ({}) は有限実数である必要があります。",
                    self.energy_threshold
                ),
            });
        }
        if let Some(t) = self.energy_temperature
            && (!t.is_finite() || t <= 0.0)
        {
            return Err(CoreError::InvalidGatingConfig {
                message: format!(
                    "energy_temperature ({}) は正の有限実数である必要があります。",
                    t
                ),
            });
        }
        if let Some(loose_th) = self.loose_energy_threshold {
            if !loose_th.is_finite() {
                return Err(CoreError::InvalidGatingConfig {
                    message: format!(
                        "loose_energy_threshold ({loose_th}) は有限実数である必要があります。"
                    ),
                });
            }
            if loose_th > self.energy_threshold {
                return Err(CoreError::InvalidGatingConfig {
                    message: format!(
                        "loose_energy_threshold ({loose_th}) は energy_threshold ({}) 以下である必要があります。",
                        self.energy_threshold
                    ),
                });
            }
        }
        if let Some(c) = self.ood_min_confidence
            && !(0.0..=1.0).contains(&c)
        {
            return Err(CoreError::InvalidGatingConfig {
                message: format!(
                    "ood_min_confidence ({}) は 0.0〜1.0 の範囲である必要があります。",
                    c
                ),
            });
        }
        if let Some(m) = self.ood_min_margin
            && !(0.0..=1.0).contains(&m)
        {
            return Err(CoreError::InvalidGatingConfig {
                message: format!(
                    "ood_min_margin ({}) は 0.0〜1.0 の範囲である必要があります。",
                    m
                ),
            });
        }
        let hybrid_set_count = usize::from(self.loose_energy_threshold.is_some())
            + usize::from(self.ood_min_confidence.is_some())
            + usize::from(self.ood_min_margin.is_some());
        if hybrid_set_count != 0 && hybrid_set_count != 3 {
            return Err(CoreError::InvalidGatingConfig {
                message: "ハイブリッド OOD 判定を有効化する場合、loose_energy_threshold、ood_min_confidence、ood_min_margin は 3 つすべて指定するか、すべて未指定にする必要があります。".to_string(),
            });
        }
        Ok(())
    }
}

/// ゲーティング判定の設定パラメータ。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct GatingConfig {
    /// ゲーティング処理を有効化するかどうか。
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// ゲーティング判定用閾値パラメータ群。
    #[serde(flatten)]
    pub thresholds: GatingThresholds,
}

impl Default for GatingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            thresholds: GatingThresholds::default(),
        }
    }
}

impl GatingConfig {
    /// 有効フラグと閾値設定から新規 `GatingConfig` を生成する。
    pub fn new(enabled: bool, thresholds: GatingThresholds) -> Self {
        Self {
            enabled,
            thresholds,
        }
    }

    /// 閾値設定への参照を取得する。
    #[inline]
    pub fn thresholds(&self) -> &GatingThresholds {
        &self.thresholds
    }

    /// 閾値設定への可変参照を取得する。
    #[inline]
    pub fn thresholds_mut(&mut self) -> &mut GatingThresholds {
        &mut self.thresholds
    }

    /// 閾値設定値の数学的妥当性を検証する。
    pub fn validate(&self) -> Result<()> {
        self.thresholds.validate()
    }
}

/// 候補とその予測確率値のペア。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct CandidateProbability {
    /// 候補識別子ラベル。
    pub candidate: String,
    /// 較正済み確率値 (0.0 <= p <= 1.0)。
    pub probability: f64,
}

/// 大型自己回帰 LLM (System 2) へのエスカレーション用コンテキスト。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct EscalationContext {
    /// 確率上位の候補一覧 (降順、最大3候補)。
    pub top_candidates: Vec<CandidateProbability>,

    /// 1位候補と2位候補の確率差。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub margin: Option<f64>,

    /// 不確実性および迷いが生じた理由の説明文。
    pub uncertainty_reason: String,

    /// 大型 LLM への入力プロンプト案。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_template: Option<String>,
}

/// 単一質問に対する確信度ゲーティング判定結果および監査メタデータ。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct GatingMetadata {
    /// 判定されたルーティングアクション種別。
    pub route: DecisionRoute,

    /// 判定に用いられた較正済み実効確信度スコア (0.0 <= C <= 1.0)。
    pub confidence: f64,

    /// 確率分布の不確実性・散らばり指標値。
    ///
    /// # 備考
    /// 質問タイプに応じて以下の正規化不確実性指標を格納する:
    /// - **Choice 型**: 逆算された正規化エントロピー指標 ($1 - C_{\text{entropy}}$)。
    /// - **Score 型**: 逆算された正規化分散指標 ($1 - C_{\text{var}}$)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entropy: Option<f64>,

    /// 上位 2 候補の確率マージン (Top-1 - Top-2)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub margin: Option<f64>,

    /// 算出された正規化自由エネルギー値。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy: Option<f64>,

    /// 未定義カテゴリ・該当なし (OOD) として検知されたかどうかの真偽値。
    #[serde(default)]
    pub is_ood: bool,

    /// 判定根拠の説明文 (監査証跡用)。
    pub reason: String,

    /// 中確信度または低確信度時の大型 LLM 連携コンテキスト。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub escalation: Option<EscalationContext>,
}

impl GatingMetadata {
    /// 確率分布の散らばり度合い(不確実性指標)を取得する。
    ///
    /// Choice 型のエントロピーおよび Score 型の分散指標を同一の不確実性スコアとして参照する。
    pub fn dispersion(&self) -> Option<f64> {
        self.entropy
    }
}

/// リクエスト全体の集約ルーティングサマリー。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct SystemRoutingSummary {
    /// リクエスト全体の集約ルーティング (最悪値ルール)。
    pub aggregate_route: DecisionRoute,

    /// 自動実行 (AutoExecute) と判定された質問数。
    pub auto_execute_count: usize,

    /// 確認・二次検証要求 (ConfirmOrEscalate) と判定された質問数。
    pub confirm_count: usize,

    /// 安全弁フォールバック (Fallback) と判定された質問数。
    pub fallback_count: usize,

    /// System 2 へのエスカレーションまたは人手介入が必要であるかどうか。
    pub escalation_needed: bool,
}

/// 単一の `Answer` に対して確信度ゲーティング評価を実施し、メタデータを生成する。
///
/// # 引数
/// - `answer`: 事後較正済みの判定結果。
/// - `question_id`: 質問識別子(プロンプト構築用、任意)。
/// - `question`: 質問定義(指示文やタイプ取得用、任意)。
/// - `config`: ゲーティング設定。
///
/// # 戻り値
/// - 導出された `GatingMetadata`。
pub fn evaluate_answer_gating(
    answer: &Answer,
    question_id: Option<&str>,
    question: Option<&Question>,
    config: &GatingConfig,
) -> GatingMetadata {
    evaluate_answer_gating_with_energy(answer, question_id, question, config, None)
}

/// 単一の `Answer` に対し、自由エネルギー情報を含めて確信度ゲーティング評価を実施し、メタデータを生成する。
///
/// # 概要
/// - 通常の確信度・マージン判定に先立ち、Energy-based OOD 安全弁 (Circuit Breaker) を評価する。
/// - Choice 型において正規化自由エネルギー $E_{\text{norm}}$ が `config.energy_threshold` を超過した場合、
///   高確信度判定をショートサーキットし、強制的に `DecisionRoute::Fallback` へ降格する。
/// - 内部でデフォルトの委託プロンプト案 (`prompt_template`) を自動生成する。
/// - プロンプト生成をスキップしたい場合は [`evaluate_answer_gating_opts`] を使用する。
///
/// # 引数
/// - `answer`: 事後較正済みの判定結果。
/// - `question_id`: 質問識別子(プロンプト構築用、任意)。
/// - `question`: 質問定義(指示文やタイプ取得用、任意)。
/// - `config`: ゲーティング設定。
/// - `energy`: 算出された正規化自由エネルギー値 (任意)。
///
/// # 戻り値
/// - 導出された `GatingMetadata`。
pub fn evaluate_answer_gating_with_energy(
    answer: &Answer,
    question_id: Option<&str>,
    question: Option<&Question>,
    config: &GatingConfig,
    energy: Option<f64>,
) -> GatingMetadata {
    evaluate_answer_gating_opts(answer, question_id, question, config, energy, true)
}

/// 確信度ゲーティングにおける判定理由の構造化種別。
///
/// ルーティング決定ロジックと監査証跡用メッセージ生成を分離するために使用する。
#[derive(Debug, Clone, Copy, PartialEq)]
enum RoutingReasonKind {
    /// ハード自由エネルギー閾値超過による OOD 検知。
    HardEnergyOod { energy: f64, threshold: f64 },
    /// 緩和自由エネルギー閾値と不確実性の複合条件による OOD 検知。
    HybridEnergyOod {
        energy: f64,
        loose_threshold: f64,
        conf: f64,
        min_conf: f64,
        margin: f64,
        min_margin: f64,
    },
    /// Choice 型の上位 2 候補拮抗によるマージンガード降格。
    MarginGuardDemoted {
        entropy_conf: f64,
        high_threshold: f64,
        margin: f64,
        top_margin_threshold: f64,
    },
    /// 高確信度による自動実行承認。
    HighConfidenceAuto { conf: f64, high_threshold: f64 },
    /// 中確信度による二次検証・確認要求。
    MediumConfidenceConfirm {
        conf: f64,
        low_threshold: f64,
        high_threshold: f64,
    },
    /// 低確信度による安全弁フォールバック。
    LowConfidenceFallback { conf: f64, low_threshold: f64 },
}

impl RoutingReasonKind {
    /// OOD 安全弁によるフォールバック判定であるかどうかを返す。
    fn is_ood(self) -> bool {
        matches!(
            self,
            Self::HardEnergyOod { .. } | Self::HybridEnergyOod { .. }
        )
    }

    /// 判定種別に対応する監査証跡用の説明文を生成する。
    fn format_reason(self) -> String {
        match self {
            Self::HardEnergyOod { energy, threshold } => format!(
                "正規化自由エネルギー({energy:.3} > {threshold:.3})が閾値を超過したため、未定義カテゴリまたは該当なし(OOD)として安全弁フォールバックを適用しました。"
            ),
            Self::HybridEnergyOod {
                energy,
                loose_threshold,
                conf,
                min_conf,
                margin,
                min_margin,
            } => format!(
                "正規化自由エネルギー({energy:.3} > {loose_threshold:.3})かつ不確実性(確信度={conf:.3} < {min_conf:.3} または マージン={margin:.3} < {min_margin:.3})の複合条件を満たしたため、該当なし(OOD)として安全弁フォールバックを適用しました。"
            ),
            Self::MarginGuardDemoted {
                entropy_conf,
                high_threshold,
                margin,
                top_margin_threshold,
            } => format!(
                "エントロピー確信度({entropy_conf:.3} >= {high_threshold:.3})は高水準ですが、上位2候補の確率差({margin:.3} < {top_margin_threshold:.3})が僅差のため確認要求に降格しました。"
            ),
            Self::HighConfidenceAuto {
                conf,
                high_threshold,
            } => format!(
                "実効確信度({conf:.3} >= {high_threshold:.3})が高水準であるため自動実行が承認されました。"
            ),
            Self::MediumConfidenceConfirm {
                conf,
                low_threshold,
                high_threshold,
            } => format!(
                "実効確信度({conf:.3})が確認要求境界 [{low_threshold:.3}, {high_threshold:.3}) 内のため二次検証が必要です。"
            ),
            Self::LowConfidenceFallback {
                conf,
                low_threshold,
            } => format!(
                "実効確信度({conf:.3} < {low_threshold:.3})が低水準のため安全弁フォールバックを適用しました。"
            ),
        }
    }
}

/// 判定結果の確率分布から Top-Margin と分布散らばり指標 (エントロピーまたは正規化分散) を算出する。
fn compute_margin_and_entropy(answer: &Answer) -> (Option<f64>, Option<f64>) {
    let margin = match &answer.payload {
        AnswerPayload::Choice { probabilities, .. } => top_margin_iter(probabilities.values()).ok(),
        AnswerPayload::Noul { probability } => {
            let p_true = *probability;
            let p_false = (1.0 - p_true).clamp(0.0, 1.0);
            Some((p_true - p_false).abs().clamp(0.0, 1.0))
        }
        _ => None,
    };

    // ゼロアロケーション走査による分布エントロピー・散らばり指標の直接算出。
    let entropy = match &answer.payload {
        AnswerPayload::Choice { probabilities, .. } => {
            normalized_entropy_iter(probabilities.values()).ok()
        }
        AnswerPayload::Score { confidence, .. } => {
            // Score 型の場合、confidence は正規化分散確信度 C_var であるため、散らばり指標として正規化分散 (1 - C_var) を記録する。
            Some((1.0 - confidence).clamp(0.0, 1.0))
        }
        AnswerPayload::Noul { probability } => {
            // Noul 型の場合、二値確率分布 [p, 1 - p] に対する正規化シャノンエントロピー指標を記録する。
            let p_true = probability.clamp(0.0, 1.0);
            let p_false = 1.0 - p_true;
            normalized_entropy(&[p_true, p_false]).ok()
        }
    };

    (margin, entropy)
}

/// Choice 型に対する Energy-based OOD 安全弁を評価し、該当時はその根拠を返却する。
fn evaluate_ood_trigger(
    is_choice: bool,
    energy: Option<f64>,
    conf: f64,
    margin: Option<f64>,
    thresholds: &GatingThresholds,
) -> Option<RoutingReasonKind> {
    if !thresholds.ood_enabled || !is_choice {
        return None;
    }
    let e = energy?;

    if e > thresholds.energy_threshold {
        return Some(RoutingReasonKind::HardEnergyOod {
            energy: e,
            threshold: thresholds.energy_threshold,
        });
    }

    if let (Some(loose_th), Some(min_conf), Some(min_margin)) = (
        thresholds.loose_energy_threshold,
        thresholds.ood_min_confidence,
        thresholds.ood_min_margin,
    ) {
        let m = margin.unwrap_or(0.0);
        if e > loose_th && (conf < min_conf || m < min_margin) {
            return Some(RoutingReasonKind::HybridEnergyOod {
                energy: e,
                loose_threshold: loose_th,
                conf,
                min_conf,
                margin: m,
                min_margin,
            });
        }
    }

    None
}

/// 確信度・エントロピー・マージン・自由エネルギーに基づき、最終的なルーティングと判定種別を決定する純粋関数。
fn resolve_routing_decision(
    is_choice: bool,
    conf: f64,
    entropy: Option<f64>,
    margin: Option<f64>,
    energy: Option<f64>,
    thresholds: &GatingThresholds,
) -> (DecisionRoute, RoutingReasonKind) {
    // 1. OOD 検知時は最優先で Fallback へ短絡降格する。
    if let Some(ood_kind) = evaluate_ood_trigger(is_choice, energy, conf, margin, thresholds) {
        return (DecisionRoute::Fallback, ood_kind);
    }

    // 2. Choice 型のマージンガード判定:
    //    純粋な分布平坦度指標であるエントロピー確信度 (1.0 - H_norm) を用いて、
    //    複合確信度 (S_conf <= Margin) によるデッドコード化を回避する。
    let entropy_confidence = if is_choice {
        entropy.map(|h| (1.0 - h).clamp(0.0, 1.0))
    } else {
        None
    };
    if is_choice
        && let Some(ent_conf) = entropy_confidence
        && ent_conf >= thresholds.high_threshold - EPSILON
        && let Some(m) = margin
        && m < thresholds.top_margin_threshold - EPSILON
    {
        return (
            DecisionRoute::ConfirmOrEscalate,
            RoutingReasonKind::MarginGuardDemoted {
                entropy_conf: ent_conf,
                high_threshold: thresholds.high_threshold,
                margin: m,
                top_margin_threshold: thresholds.top_margin_threshold,
            },
        );
    }

    // 3. 実効確信度に基づく 3 段階ルーティング判定。
    if conf >= thresholds.high_threshold - EPSILON {
        return (
            DecisionRoute::AutoExecute,
            RoutingReasonKind::HighConfidenceAuto {
                conf,
                high_threshold: thresholds.high_threshold,
            },
        );
    }

    if conf >= thresholds.low_threshold - EPSILON {
        return (
            DecisionRoute::ConfirmOrEscalate,
            RoutingReasonKind::MediumConfidenceConfirm {
                conf,
                low_threshold: thresholds.low_threshold,
                high_threshold: thresholds.high_threshold,
            },
        );
    }

    (
        DecisionRoute::Fallback,
        RoutingReasonKind::LowConfidenceFallback {
            conf,
            low_threshold: thresholds.low_threshold,
        },
    )
}

/// エスカレーション時 (`AutoExecute` 以外) に上位候補と任意のプロンプト案を構築する。
fn build_escalation_context(
    answer: &Answer,
    question_id: Option<&str>,
    question: Option<&Question>,
    margin: Option<f64>,
    reason: &str,
    generate_prompt: bool,
) -> EscalationContext {
    let top_candidates = match &answer.payload {
        AnswerPayload::Choice { probabilities, .. }
        | AnswerPayload::Score { probabilities, .. } => extract_top3_candidates(probabilities),
        AnswerPayload::Noul { probability } => {
            let p_true = *probability;
            let p_false = (1.0 - p_true).clamp(0.0, 1.0);
            let true_cand = CandidateProbability {
                candidate: "true".to_string(),
                probability: p_true,
            };
            let false_cand = CandidateProbability {
                candidate: "false".to_string(),
                probability: p_false,
            };
            if p_true >= p_false {
                vec![true_cand, false_cand]
            } else {
                vec![false_cand, true_cand]
            }
        }
    };

    let prompt_template = if generate_prompt {
        Some(build_escalation_prompt(
            question_id,
            question,
            &top_candidates,
            margin,
        ))
    } else {
        None
    };

    EscalationContext {
        top_candidates,
        margin,
        uncertainty_reason: reason.to_string(),
        prompt_template,
    }
}

/// 単一の `Answer` に対し、自由エネルギー情報およびプロンプト生成要否を指定して確信度ゲーティング評価を実施する。
///
/// # 概要
/// - `generate_prompt` に `false` を指定した場合、
///   コア層でのデフォルトエスカレーションプロンプト文字列のヒープ確保をスキップし、
///   `EscalationContext.prompt_template` を `None` とする。
/// - ランタイム層等で上位のリッチプロンプト(State 文脈や高度な推論理由等を含むプロンプト)に、
///   置換される前提の処理において、不要な二重アロケーションを抑止するために使用する。
///
/// # 引数
/// - `answer`: 事後較正済みの判定結果。
/// - `question_id`: 質問識別子(プロンプト構築用、任意)。
/// - `question`: 質問定義(指示文やタイプ取得用、任意)。
/// - `config`: ゲーティング設定。
/// - `energy`: 算出された正規化自由エネルギー値 (任意)。
/// - `generate_prompt`: デフォルトプロンプト案文字列を生成するかどうかのフラグ。
///
/// # 戻り値
/// - 導出された `GatingMetadata`。
pub fn evaluate_answer_gating_opts(
    answer: &Answer,
    question_id: Option<&str>,
    question: Option<&Question>,
    config: &GatingConfig,
    energy: Option<f64>,
    generate_prompt: bool,
) -> GatingMetadata {
    let conf = answer.effective_confidence().clamp(0.0, 1.0);

    // 質問種別の特定(question 定義からの取得、または answer の payload からの導出)。
    let question_type = question
        .map(|q| q.question_type)
        .unwrap_or_else(|| answer.question_type());
    let is_choice = question_type == QuestionType::Choice;

    let (margin, entropy) = compute_margin_and_entropy(answer);

    let (route, reason_kind) =
        resolve_routing_decision(is_choice, conf, entropy, margin, energy, &config.thresholds);
    let is_ood = reason_kind.is_ood();
    let reason = reason_kind.format_reason();

    // エスカレーションコンテキストの遅延生成 (ConfirmOrEscalate または Fallback 時のみアロケーションを実行)。
    let escalation = (route != DecisionRoute::AutoExecute).then(|| {
        build_escalation_context(
            answer,
            question_id,
            question,
            margin,
            &reason,
            generate_prompt,
        )
    });

    GatingMetadata {
        route,
        confidence: conf,
        entropy,
        margin,
        energy,
        is_ood,
        reason,
        escalation,
    }
}

/// 確率マップから確率降順で上位 3 件の候補を抽出する。
///
/// 全件の文字列クローンやヒープ確保・全ソートを行わず、
/// 要素数 3 の固定長スタック配列とインサーションシフトを用いて単一パスで上位候補を収集する。
///
/// # 引数
/// - `probabilities`: 候補ラベルと較正済み確率値の対応マップ。
///
/// # 戻り値
/// 確率降順の上位候補一覧 (最大 3 件)。
fn extract_top3_candidates(probabilities: &IndexMap<String, f64>) -> Vec<CandidateProbability> {
    let mut top: [Option<(&str, f64)>; 3] = [None, None, None];

    for (k, &p) in probabilities {
        // 3 枠が埋まっており、かつ第 3 位の確率以下の場合は走査を短絡スキップする。
        if let Some((_, p_min)) = top[2]
            && p.total_cmp(&p_min) != std::cmp::Ordering::Greater
        {
            continue;
        }

        // 挿入位置を特定し、後続スロットを右シフトして格納する。
        for i in 0..3 {
            let should_insert = match top[i] {
                Some((_, slot_p)) => p.total_cmp(&slot_p) == std::cmp::Ordering::Greater,
                None => true,
            };
            if should_insert {
                for j in (i + 1..3).rev() {
                    top[j] = top[j - 1];
                }
                top[i] = Some((k.as_str(), p));
                break;
            }
        }
    }

    top.into_iter()
        .flatten()
        .map(|(k, p)| CandidateProbability {
            candidate: k.to_string(),
            probability: p,
        })
        .collect()
}

/// System 2 (大型自己回帰LLM) に向けた委託用プロンプト案を生成する。
fn build_escalation_prompt(
    question_id: Option<&str>,
    question: Option<&Question>,
    candidates: &[CandidateProbability],
    margin: Option<f64>,
) -> String {
    let q_name = question_id.unwrap_or("target_question");
    let instructions = question
        .map(|q| q.instructions.as_str())
        .unwrap_or("指示文なし");

    if candidates.len() >= 2 {
        let c1 = &candidates[0];
        let c2 = &candidates[1];

        let mut prompt = String::with_capacity(256);
        let _ = write!(
            prompt,
            "質問「{q_name}」({instructions}) について、System 1 の判定では候補「{}」(確信度 {:.1}%) と「{}」(確信度 {:.1}%) の間で迷いが生じています ",
            c1.candidate,
            c1.probability * 100.0,
            c2.candidate,
            c2.probability * 100.0
        );
        if let Some(m) = margin {
            let _ = write!(prompt, "(確率差: {:.1}%)", m * 100.0);
        }
        prompt.push_str("。State コンテキストを精読し、思考連鎖 (Chain-of-Thought) に基づき最適な決定を行ってください。");
        return prompt;
    }

    let mut prompt = String::with_capacity(128);
    let _ = write!(
        prompt,
        "質問「{q_name}」({instructions}) について、System 1 の判定信頼性が低水準となっています。State コンテキストを精読し、論理的根拠に基づいて決定を下してください。"
    );
    prompt
}

/// リクエストに含まれる全回答のゲーティング情報から集約サマリーを算出する。
///
/// 最悪値ルール(Conservative Policy)を適用する:
/// - `answers` が空(0 件)の場合は、未処理・判定不能として安全側の `Fallback` (`escalation_needed = true`) とする。
/// - 1 つでも `Fallback` が存在する場合 $\rightarrow$ 全体も `Fallback`。
/// - そうでなく 1 つでも `ConfirmOrEscalate` が存在する場合 $\rightarrow$ 全体も `ConfirmOrEscalate`。
/// - 1 件以上存在し、かつ全ての質問が `AutoExecute` である場合 $\rightarrow$ 全体も `AutoExecute`。
pub fn evaluate_response_routing(answers: &IndexMap<String, Answer>) -> SystemRoutingSummary {
    if answers.is_empty() {
        return SystemRoutingSummary {
            aggregate_route: DecisionRoute::Fallback,
            auto_execute_count: 0,
            confirm_count: 0,
            fallback_count: 0,
            escalation_needed: true,
        };
    }

    let mut auto_execute_count = 0;
    let mut confirm_count = 0;
    let mut fallback_count = 0;

    for answer in answers.values() {
        match answer.route() {
            Some(DecisionRoute::AutoExecute) => auto_execute_count += 1,
            Some(DecisionRoute::ConfirmOrEscalate) => confirm_count += 1,
            Some(DecisionRoute::Fallback) => fallback_count += 1,
            None => {
                // ゲーティング未適用の場合は実効確信度から簡易分類
                let conf = answer.effective_confidence();
                if conf >= DEFAULT_HIGH_CONFIDENCE_THRESHOLD {
                    auto_execute_count += 1;
                } else if conf >= DEFAULT_LOW_CONFIDENCE_THRESHOLD {
                    confirm_count += 1;
                } else {
                    fallback_count += 1;
                }
            }
        }
    }

    let aggregate_route = if fallback_count > 0 {
        DecisionRoute::Fallback
    } else if confirm_count > 0 {
        DecisionRoute::ConfirmOrEscalate
    } else {
        DecisionRoute::AutoExecute
    };

    let escalation_needed = confirm_count > 0 || fallback_count > 0;

    SystemRoutingSummary {
        aggregate_route,
        auto_execute_count,
        confirm_count,
        fallback_count,
        escalation_needed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gating_config_default_and_validation() {
        let config = GatingConfig::default();
        assert!(config.enabled);
        assert!((config.thresholds.high_threshold - 0.70).abs() < EPSILON);
        assert!((config.thresholds.low_threshold - 0.35).abs() < EPSILON);
        assert!((config.thresholds.top_margin_threshold - 0.15).abs() < EPSILON);
        assert!(config.validate().is_ok());

        let invalid_config = GatingConfig {
            enabled: true,
            thresholds: GatingThresholds {
                high_threshold: 0.30,
                low_threshold: 0.50,
                top_margin_threshold: 0.15,
                ood_enabled: true,
                energy_threshold: -1.0,
                ..GatingThresholds::default()
            },
        };
        assert!(invalid_config.validate().is_err());

        let invalid_energy_config = GatingConfig {
            thresholds: GatingThresholds {
                energy_threshold: f64::NAN,
                ..GatingThresholds::default()
            },
            ..GatingConfig::default()
        };
        assert!(invalid_energy_config.validate().is_err());
    }

    #[test]
    fn test_evaluate_answer_gating_auto_execute() {
        let mut probs = IndexMap::new();
        probs.insert("opt_a".to_string(), 0.90);
        probs.insert("opt_b".to_string(), 0.10);
        let answer = Answer::choice("opt_a", probs, 0.92);

        let config = GatingConfig::default();
        let meta = evaluate_answer_gating(&answer, Some("q1"), None, &config);

        assert_eq!(meta.route, DecisionRoute::AutoExecute);
        assert!((meta.confidence - 0.92).abs() < EPSILON);
        assert!((meta.margin.unwrap() - 0.80).abs() < EPSILON);
        assert!(meta.entropy.is_some());
        assert!(meta.escalation.is_none());
    }

    #[test]
    fn test_evaluate_answer_gating_margin_guard_demote_to_confirm() {
        // エントロピー確信度 (1 - H_norm >= 0.70) は高水準だが、Top-1 と Top-2 が僅差 (0.50 vs 0.48 -> margin 0.02 < 0.15) のケース。
        // K=20 の分布において上位 2 候補へ確率が集中しつつ拮抗しているため、マージンガードにより ConfirmOrEscalate へ降格する。
        let mut probs = IndexMap::new();
        probs.insert("opt_a".to_string(), 0.50);
        probs.insert("opt_b".to_string(), 0.48);
        let rem_prob = 0.02 / 18.0;
        for i in 2..20 {
            probs.insert(format!("opt_{:02}", i), rem_prob);
        }
        let s_conf = crate::math::composite_confidence_iter(probs.values()).unwrap();
        let answer = Answer::choice("opt_a", probs, s_conf);

        let config = GatingConfig::default();
        let meta = evaluate_answer_gating(&answer, Some("q1"), None, &config);

        assert_eq!(meta.route, DecisionRoute::ConfirmOrEscalate);
        assert!(meta.reason.contains("上位2候補の確率差"));
        assert!(meta.escalation.is_some());
        let esc = meta.escalation.unwrap();
        assert_eq!(esc.top_candidates.len(), 3);
        assert_eq!(esc.top_candidates[0].candidate, "opt_a");
        assert_eq!(esc.top_candidates[1].candidate, "opt_b");
    }

    #[test]
    fn test_evaluate_answer_gating_margin_guard_with_derived_composite_confidence() {
        // 高基数 (K=80) において、分布は上位に集中 (エントロピー確信度 1 - H_norm >= 0.70) しているが、
        // Top-1 と Top-2 が僅差 (0.48 vs 0.44 -> margin 0.04 < 0.15) のリアルなケース。
        // 本番パイプラインでは複合確信度 S_conf = (1 - H_norm) * 0.04 ≈ 0.028 となり、
        // 旧仕様では low_threshold (0.35) 未満により Fallback へ誤判定されていた。
        let mut probs = IndexMap::new();
        probs.insert("opt_00".to_string(), 0.48);
        probs.insert("opt_01".to_string(), 0.44);
        let rem_prob = 0.08 / 78.0;
        for i in 2..80 {
            probs.insert(format!("opt_{:02}", i), rem_prob);
        }

        // 複合確信度 S_conf の算出
        let s_conf = crate::math::composite_confidence_iter(probs.values()).unwrap();
        assert!(
            s_conf < 0.05,
            "複合確信度はマージン 0.04 に制約されて 0.05 未満となる。"
        );

        let answer = Answer::choice("opt_00", probs, s_conf);
        let config = GatingConfig::default();
        let meta = evaluate_answer_gating(&answer, Some("q_high_k"), None, &config);

        // マージンガードが先行発動し、Fallback ではなく ConfirmOrEscalate へ正しく降格・ルーティングされること
        assert_eq!(meta.route, DecisionRoute::ConfirmOrEscalate);
        assert!(meta.reason.contains("エントロピー確信度"));
        assert!(meta.reason.contains("上位2候補の確率差"));
        assert!(meta.escalation.is_some());
        let esc = meta.escalation.unwrap();
        assert_eq!(esc.top_candidates[0].candidate, "opt_00");
        assert_eq!(esc.top_candidates[1].candidate, "opt_01");
    }

    #[test]
    fn test_evaluate_answer_gating_medium_confirm() {
        let mut probs = IndexMap::new();
        probs.insert("opt_a".to_string(), 0.60);
        probs.insert("opt_b".to_string(), 0.40);
        let answer = Answer::choice("opt_a", probs, 0.50);

        let config = GatingConfig::default();
        let meta = evaluate_answer_gating(&answer, Some("q1"), None, &config);

        assert_eq!(meta.route, DecisionRoute::ConfirmOrEscalate);
        assert!(meta.escalation.is_some());
    }

    #[test]
    fn test_evaluate_answer_gating_low_fallback() {
        let mut probs = IndexMap::new();
        probs.insert("opt_a".to_string(), 0.34);
        probs.insert("opt_b".to_string(), 0.33);
        probs.insert("opt_c".to_string(), 0.33);
        let answer = Answer::choice("opt_a", probs, 0.10);

        let config = GatingConfig::default();
        let meta = evaluate_answer_gating(&answer, Some("q1"), None, &config);

        assert_eq!(meta.route, DecisionRoute::Fallback);
        assert!(meta.escalation.is_some());
    }

    #[test]
    fn test_evaluate_answer_gating_noul_high_and_low() {
        let config = GatingConfig::default();

        // High: P=0.96 -> C=|2*0.96 - 1| = 0.92 >= 0.70
        let ans_high = Answer::noul(0.96);
        let meta_high = evaluate_answer_gating(&ans_high, Some("noul_q"), None, &config);
        assert_eq!(meta_high.route, DecisionRoute::AutoExecute);
        assert!(meta_high.entropy.is_some());
        let ent_high = meta_high.entropy.unwrap();
        assert!(ent_high > 0.20 && ent_high < 0.30);

        // Low: P=0.52 -> C=|2*0.52 - 1| = 0.04 < 0.35
        let ans_low = Answer::noul(0.52);
        let meta_low = evaluate_answer_gating(&ans_low, Some("noul_q"), None, &config);
        assert_eq!(meta_low.route, DecisionRoute::Fallback);
        assert!(meta_low.escalation.is_some());
        assert!(meta_low.entropy.is_some());
        let ent_low = meta_low.entropy.unwrap();
        assert!(ent_low > 0.99 && ent_low <= 1.0);

        // Extreme: P=1.0 -> H=0.0, P=0.5 -> H=1.0
        let ans_one = Answer::noul(1.0);
        let meta_one = evaluate_answer_gating(&ans_one, Some("noul_q"), None, &config);
        assert_eq!(meta_one.entropy, Some(0.0));

        let ans_half = Answer::noul(0.5);
        let meta_half = evaluate_answer_gating(&ans_half, Some("noul_q"), None, &config);
        assert!((meta_half.entropy.unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_evaluate_response_routing_aggregation() {
        let mut answers = IndexMap::new();

        let ans1 = Answer::choice("a", IndexMap::new(), 0.95).with_gating(GatingMetadata {
            route: DecisionRoute::AutoExecute,
            confidence: 0.95,
            entropy: None,
            margin: None,
            energy: None,
            is_ood: false,
            reason: String::new(),
            escalation: None,
        });

        let ans2 = Answer::choice("b", IndexMap::new(), 0.50).with_gating(GatingMetadata {
            route: DecisionRoute::ConfirmOrEscalate,
            confidence: 0.50,
            entropy: None,
            margin: None,
            energy: None,
            is_ood: false,
            reason: String::new(),
            escalation: None,
        });

        answers.insert("q1".to_string(), ans1);
        answers.insert("q2".to_string(), ans2);

        let summary = evaluate_response_routing(&answers);
        assert_eq!(summary.aggregate_route, DecisionRoute::ConfirmOrEscalate);
        assert_eq!(summary.auto_execute_count, 1);
        assert_eq!(summary.confirm_count, 1);
        assert_eq!(summary.fallback_count, 0);
        assert!(summary.escalation_needed);

        // Fallback が 1 つでも加わると全体が Fallback
        let ans3 = Answer::choice("c", IndexMap::new(), 0.20).with_gating(GatingMetadata {
            route: DecisionRoute::Fallback,
            confidence: 0.20,
            entropy: None,
            margin: None,
            energy: None,
            is_ood: false,
            reason: String::new(),
            escalation: None,
        });
        answers.insert("q3".to_string(), ans3);

        let summary_fallback = evaluate_response_routing(&answers);
        assert_eq!(summary_fallback.aggregate_route, DecisionRoute::Fallback);
        assert_eq!(summary_fallback.fallback_count, 1);
        assert!(summary_fallback.escalation_needed);

        // 空の回答マップの場合は安全弁として Fallback (escalation_needed = true) になること。
        let empty_answers: IndexMap<String, Answer> = IndexMap::new();
        let summary_empty = evaluate_response_routing(&empty_answers);
        assert_eq!(summary_empty.aggregate_route, DecisionRoute::Fallback);
        assert_eq!(summary_empty.auto_execute_count, 0);
        assert_eq!(summary_empty.confirm_count, 0);
        assert_eq!(summary_empty.fallback_count, 0);
        assert!(summary_empty.escalation_needed);
    }

    #[test]
    fn test_evaluate_answer_gating_ood_short_circuit() {
        // 見かけ上は高確信度 (conf=0.90, margin=0.80) だが、自由エネルギーが異常に高い (energy=0.5 > -1.0)
        let mut probs = IndexMap::new();
        probs.insert("tech_support".to_string(), 0.90);
        probs.insert("billing".to_string(), 0.10);
        let answer = Answer::choice("tech_support", probs, 0.90);

        let config = GatingConfig::default();
        // OOD 入力: energy = 0.5 (閾値 -1.0 を超過)
        let meta = evaluate_answer_gating_with_energy(
            &answer,
            Some("department"),
            None,
            &config,
            Some(0.5),
        );

        // 高確信度をショートサーキットして強制的に Fallback へ降格
        assert_eq!(meta.route, DecisionRoute::Fallback);
        assert!(meta.is_ood);
        assert_eq!(meta.energy, Some(0.5));
        assert!(meta.reason.contains("未定義カテゴリまたは該当なし(OOD)"));
        assert!(meta.escalation.is_some());
    }

    #[test]
    fn test_evaluate_answer_gating_ood_id_pass() {
        // 既知の適合入力 (ID): energy = -3.5 <= -1.0 で正常
        let mut probs = IndexMap::new();
        probs.insert("tech_support".to_string(), 0.90);
        probs.insert("billing".to_string(), 0.10);
        let answer = Answer::choice("tech_support", probs, 0.90);

        let config = GatingConfig::default();
        let meta = evaluate_answer_gating_with_energy(
            &answer,
            Some("department"),
            None,
            &config,
            Some(-3.5),
        );

        assert_eq!(meta.route, DecisionRoute::AutoExecute);
        assert!(!meta.is_ood);
        assert_eq!(meta.energy, Some(-3.5));
        assert!(meta.escalation.is_none());
    }

    #[test]
    fn test_evaluate_answer_gating_ood_disabled() {
        // ood_enabled が false の場合、エネルギーが高くても OOD 安全弁が発動しない
        let mut probs = IndexMap::new();
        probs.insert("tech_support".to_string(), 0.90);
        probs.insert("billing".to_string(), 0.10);
        let answer = Answer::choice("tech_support", probs, 0.90);

        let config = GatingConfig {
            thresholds: GatingThresholds {
                ood_enabled: false,
                ..Default::default()
            },
            ..Default::default()
        };

        let meta = evaluate_answer_gating_with_energy(
            &answer,
            Some("department"),
            None,
            &config,
            Some(0.5),
        );

        assert_eq!(meta.route, DecisionRoute::AutoExecute);
        assert!(!meta.is_ood);
    }

    #[test]
    fn test_evaluate_answer_gating_entropy_fallback_without_question() {
        let config = GatingConfig::default();

        // 1. Choice 型: question = None でも answer.choice からフォールバックしてエントロピーが算出される。
        let mut choice_probs = IndexMap::new();
        choice_probs.insert("opt_a".to_string(), 0.50);
        choice_probs.insert("opt_b".to_string(), 0.50);
        let ans_choice = Answer::choice("opt_a", choice_probs, 0.50);
        let meta_choice = evaluate_answer_gating(&ans_choice, Some("q_choice"), None, &config);
        assert!(meta_choice.entropy.is_some());
        // 2 候補均等分布の正規化エントロピーは 1.0。
        assert!((meta_choice.entropy.unwrap() - 1.0).abs() < EPSILON);

        // 2. Score 型: question = None でも answer.score からフォールバックして分散指標が算出される。
        let mut score_probs = IndexMap::new();
        score_probs.insert("1".to_string(), 0.2);
        score_probs.insert("2".to_string(), 0.8);
        let ans_score = Answer::score(1.8, score_probs, 0.70);
        let meta_score = evaluate_answer_gating(&ans_score, Some("q_score"), None, &config);
        assert!(meta_score.entropy.is_some());
        assert!((meta_score.entropy.unwrap() - 0.30).abs() < EPSILON);

        // 3. Noul 型: question = None でも answer.noul からフォールバックして二値エントロピーが算出される。
        let ans_noul = Answer::noul(0.50);
        let meta_noul = evaluate_answer_gating(&ans_noul, Some("q_noul"), None, &config);
        assert!(meta_noul.entropy.is_some());
        // P=0.50 の二値エントロピーは 1.0。
        assert!((meta_noul.entropy.unwrap() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_evaluate_answer_gating_score_adjacent_levels_not_demoted_by_margin() {
        // Score 型において、隣接する 2 段階に確率が集中している低分散・高確信度ケース。
        // 上位 2 段階の確率差 (Margin) は 0.00 となるが、Score 型は順序尺度であるため
        // Top-Margin による確認要求降格を行わず、分散確信度に基づいて AutoExecute となることを検証する。
        let mut score_probs = IndexMap::new();
        score_probs.insert("1".to_string(), 0.0);
        score_probs.insert("2".to_string(), 0.0);
        score_probs.insert("3".to_string(), 0.5);
        score_probs.insert("4".to_string(), 0.5);
        let ans_score = Answer::score(3.5, score_probs, 0.88);

        let config = GatingConfig::default();
        let meta = evaluate_answer_gating(&ans_score, Some("q_score"), None, &config);

        assert_eq!(meta.route, DecisionRoute::AutoExecute);
        assert_eq!(meta.margin, None);
        assert!(meta.escalation.is_none());
        assert!(meta.reason.contains("実効確信度"));
    }

    #[test]
    fn test_evaluate_answer_gating_opts_prompt_suppression() {
        // ConfirmOrEscalate 境界の判定結果を用意する。
        let mut choice_probs = IndexMap::new();
        choice_probs.insert("opt_a".to_string(), 0.60);
        choice_probs.insert("opt_b".to_string(), 0.40);
        let ans_choice = Answer::choice("opt_a", choice_probs, 0.60);

        let config = GatingConfig::default();

        // 1. generate_prompt = false の場合: prompt_template のアロケーションが抑止され None となる。
        let meta_suppressed =
            evaluate_answer_gating_opts(&ans_choice, Some("q_choice"), None, &config, None, false);
        assert_eq!(meta_suppressed.route, DecisionRoute::ConfirmOrEscalate);
        let esc_suppressed = meta_suppressed.escalation.as_ref().unwrap();
        assert!(esc_suppressed.prompt_template.is_none());
        assert_eq!(esc_suppressed.top_candidates.len(), 2);

        // 2. generate_prompt = true の場合: デフォルトの prompt_template が生成される。
        let meta_generated =
            evaluate_answer_gating_opts(&ans_choice, Some("q_choice"), None, &config, None, true);
        assert_eq!(meta_generated.route, DecisionRoute::ConfirmOrEscalate);
        let esc_generated = meta_generated.escalation.as_ref().unwrap();
        assert!(esc_generated.prompt_template.is_some());
    }

    #[test]
    fn test_extract_top3_candidates_comprehensive() {
        // 1. 空マップの場合
        let empty_probs = IndexMap::new();
        assert!(extract_top3_candidates(&empty_probs).is_empty());

        // 2. 1 件のみの場合
        let mut single_prob = IndexMap::new();
        single_prob.insert("single".to_string(), 0.95);
        let res_single = extract_top3_candidates(&single_prob);
        assert_eq!(res_single.len(), 1);
        assert_eq!(res_single[0].candidate, "single");
        assert!((res_single[0].probability - 0.95).abs() < EPSILON);

        // 3. 2 件の場合
        let mut two_probs = IndexMap::new();
        two_probs.insert("low".to_string(), 0.20);
        two_probs.insert("high".to_string(), 0.80);
        let res_two = extract_top3_candidates(&two_probs);
        assert_eq!(res_two.len(), 2);
        assert_eq!(res_two[0].candidate, "high");
        assert_eq!(res_two[1].candidate, "low");

        // 4. 4 件以上 (多数候補) の場合
        let mut multi_probs = IndexMap::new();
        multi_probs.insert("c1".to_string(), 0.10);
        multi_probs.insert("c2".to_string(), 0.40);
        multi_probs.insert("c3".to_string(), 0.05);
        multi_probs.insert("c4".to_string(), 0.25);
        multi_probs.insert("c5".to_string(), 0.20);
        let res_multi = extract_top3_candidates(&multi_probs);
        assert_eq!(res_multi.len(), 3);
        assert_eq!(res_multi[0].candidate, "c2");
        assert!((res_multi[0].probability - 0.40).abs() < EPSILON);
        assert_eq!(res_multi[1].candidate, "c4");
        assert!((res_multi[1].probability - 0.25).abs() < EPSILON);
        assert_eq!(res_multi[2].candidate, "c5");
        assert!((res_multi[2].probability - 0.20).abs() < EPSILON);

        // 5. 多数候補 (255 候補スケール) でトップ 3 が確実に抽出される検証
        let mut large_probs = IndexMap::new();
        for i in 0..255 {
            large_probs.insert(format!("opt_{i}"), (i as f64) / 1000.0);
        }
        let res_large = extract_top3_candidates(&large_probs);
        assert_eq!(res_large.len(), 3);
        assert_eq!(res_large[0].candidate, "opt_254");
        assert_eq!(res_large[1].candidate, "opt_253");
        assert_eq!(res_large[2].candidate, "opt_252");

        // 6. 同率タイ (Tie) を含むケースで先着順が維持され、正しく降順に整列する検証
        let mut tie_probs = IndexMap::new();
        tie_probs.insert("tie_a".to_string(), 0.40);
        tie_probs.insert("tie_b".to_string(), 0.40);
        tie_probs.insert("high".to_string(), 0.60);
        tie_probs.insert("tie_c".to_string(), 0.40);
        let res_tie = extract_top3_candidates(&tie_probs);
        assert_eq!(res_tie.len(), 3);
        assert_eq!(res_tie[0].candidate, "high");
        assert!((res_tie[0].probability - 0.60).abs() < EPSILON);
        assert_eq!(res_tie[1].candidate, "tie_a");
        assert!((res_tie[1].probability - 0.40).abs() < EPSILON);
        assert_eq!(res_tie[2].candidate, "tie_b");
        assert!((res_tie[2].probability - 0.40).abs() < EPSILON);

        // 7. 逆順 (昇順) 入力におけるスワップ・右シフト連鎖の整合性検証
        let mut asc_probs = IndexMap::new();
        asc_probs.insert("p1".to_string(), 0.10);
        asc_probs.insert("p2".to_string(), 0.20);
        asc_probs.insert("p3".to_string(), 0.30);
        asc_probs.insert("p4".to_string(), 0.40);
        let res_asc = extract_top3_candidates(&asc_probs);
        assert_eq!(res_asc.len(), 3);
        assert_eq!(res_asc[0].candidate, "p4");
        assert_eq!(res_asc[1].candidate, "p3");
        assert_eq!(res_asc[2].candidate, "p2");
    }

    #[test]
    fn test_evaluate_answer_gating_hybrid_ood_trigger() {
        let mut probs = IndexMap::new();
        probs.insert("opt_a".to_string(), 0.70);
        probs.insert("opt_b".to_string(), 0.30);
        // conf=0.60 (< 0.65), margin=0.40 (>= 0.20)
        let answer = Answer::choice("opt_a", probs, 0.60);

        let config = GatingConfig {
            thresholds: GatingThresholds {
                energy_threshold: -1.0,
                loose_energy_threshold: Some(-2.5),
                ood_min_confidence: Some(0.65),
                ood_min_margin: Some(0.20),
                ..Default::default()
            },
            ..Default::default()
        };

        // energy = -2.0 (> -2.5 かつ <= -1.0) で conf < 0.65 のためハイブリッド OOD が発動する。
        let meta_ood = evaluate_answer_gating_with_energy(
            &answer,
            Some("q_hybrid"),
            None,
            &config,
            Some(-2.0),
        );
        assert_eq!(meta_ood.route, DecisionRoute::Fallback);
        assert!(meta_ood.is_ood);
        assert!(meta_ood.reason.contains("複合条件"));

        // energy = -3.0 (<= -2.5) の場合は OOD にならず通常の確信度判定 (ConfirmOrEscalate) となる。
        let meta_id = evaluate_answer_gating_with_energy(
            &answer,
            Some("q_hybrid"),
            None,
            &config,
            Some(-3.0),
        );
        assert_eq!(meta_id.route, DecisionRoute::ConfirmOrEscalate);
        assert!(!meta_id.is_ood);
    }

    #[test]
    fn test_hybrid_ood_thresholds_validation() {
        // 1. 3 項目すべて妥当に指定された場合は検証成功 (境界値 loose == energy を含む)。
        let valid_hybrid = GatingThresholds {
            energy_threshold: -1.0,
            loose_energy_threshold: Some(-2.5),
            ood_min_confidence: Some(0.65),
            ood_min_margin: Some(0.20),
            ..Default::default()
        };
        assert!(valid_hybrid.validate().is_ok());

        let boundary_equal = GatingThresholds {
            energy_threshold: -1.0,
            loose_energy_threshold: Some(-1.0),
            ood_min_confidence: Some(0.65),
            ood_min_margin: Some(0.20),
            ..Default::default()
        };
        assert!(boundary_equal.validate().is_ok());

        // 2. loose_energy_threshold が非有限値 (NaN / +inf / -inf) の場合は拒否。
        for non_finite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let invalid = GatingThresholds {
                energy_threshold: -1.0,
                loose_energy_threshold: Some(non_finite),
                ood_min_confidence: Some(0.65),
                ood_min_margin: Some(0.20),
                ..Default::default()
            };
            assert!(invalid.validate().is_err());
        }

        // 3. loose_energy_threshold > energy_threshold (閾値逆転) の場合は拒否。
        let reversed_energy = GatingThresholds {
            energy_threshold: -1.0,
            loose_energy_threshold: Some(-0.5),
            ood_min_confidence: Some(0.65),
            ood_min_margin: Some(0.20),
            ..Default::default()
        };
        assert!(reversed_energy.validate().is_err());

        // 4. ハイブリッド OOD 項目の一部のみが設定された場合は拒否。
        let partial_cases = [
            GatingThresholds {
                loose_energy_threshold: Some(-2.5),
                ..Default::default()
            },
            GatingThresholds {
                ood_min_confidence: Some(0.65),
                ..Default::default()
            },
            GatingThresholds {
                ood_min_margin: Some(0.20),
                ..Default::default()
            },
            GatingThresholds {
                loose_energy_threshold: Some(-2.5),
                ood_min_confidence: Some(0.65),
                ..Default::default()
            },
            GatingThresholds {
                loose_energy_threshold: Some(-2.5),
                ood_min_margin: Some(0.20),
                ..Default::default()
            },
            GatingThresholds {
                ood_min_confidence: Some(0.65),
                ood_min_margin: Some(0.20),
                ..Default::default()
            },
        ];
        for partial in partial_cases {
            assert!(partial.validate().is_err());
        }
    }
}
