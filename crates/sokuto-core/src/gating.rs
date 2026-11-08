//! # 確信度ゲーティングおよび3系統ルーティングモジュール
//!
//! 事後較正(RLCD + 最適温度スケーリング $\tau^*$)を経た実効確信度に基づき、
//! 決定論的に 3 系統のアクションへ分岐する安全制御ロジックを提供する。
//!
//! ## 3系統ルーティングの判定基準
//! - **`AutoExecute` ($\ge 0.85$ & Margin $\ge 0.15$)**: 自動実行(高信頼・人手や大型LLMの介在なし)。
//! - **`ConfirmOrEscalate` ($0.50 \dots 0.85$ または Margin $< 0.15$)**: 確認/二次検証要求(System 2 LLMへの委託または人手確認)。
//! - **`Fallback` ($< 0.50$)**: 安全弁フォールバック(情報不足・一様迷い時のデフォルト動作・棄却)。

use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};
use crate::math::top_margin_iter;
use crate::schema::{Answer, Question, QuestionType};

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

/// ゲーティング判定の設定パラメータ。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct GatingConfig {
    /// ゲーティング処理を有効化するかどうか。
    #[serde(default = "default_true")]
    pub enabled: bool,

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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loose_energy_threshold: Option<f64>,

    /// ハイブリッド OOD 判定用の最小確信度閾値 (任意)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ood_min_confidence: Option<f64>,

    /// ハイブリッド OOD 判定用の最小マージン閾値 (任意)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ood_min_margin: Option<f64>,
}

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

impl Default for GatingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
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

impl GatingConfig {
    /// 閾値設定値の数学的妥当性を検証する。
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
        Ok(())
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
    let conf = answer.effective_confidence().unwrap_or(0.0).clamp(0.0, 1.0);

    // ゼロアロケーション走査による上位 2 候補の確率マージン (Top-Margin) 算出。
    let margin = if let Some(probs) = &answer.probabilities {
        top_margin_iter(probs.values()).ok()
    } else if let Some(p_true) = answer.noul {
        let p_false = (1.0 - p_true).clamp(0.0, 1.0);
        Some((p_true - p_false).abs().clamp(0.0, 1.0))
    } else {
        None
    };

    // 質問種別の特定(question 定義からの取得、または answer の格納フィールドからのフォールバック導出)。
    let question_type = question.map(|q| q.question_type).or_else(|| {
        if answer.choice.is_some() {
            Some(QuestionType::Choice)
        } else if answer.score.is_some() {
            Some(QuestionType::Score)
        } else if answer.noul.is_some() {
            Some(QuestionType::Noul)
        } else {
            None
        }
    });

    // 分布エントロピー・散らばり指標の直接算出。
    let entropy = match question_type {
        Some(QuestionType::Choice) => {
            if let Some(probs) = &answer.probabilities {
                let p_vals: Vec<f64> = probs.values().copied().collect();
                crate::math::normalized_entropy(&p_vals).ok()
            } else {
                None
            }
        }
        Some(QuestionType::Score) => {
            // Score 型の場合、confidence は正規化分散確信度 C_var であるため、散らばり指標として正規化分散 (1 - C_var) を記録する。
            answer.confidence.map(|c| (1.0 - c).clamp(0.0, 1.0))
        }
        Some(QuestionType::Noul) => {
            // Noul 型の場合、二値エントロピー指標 (底 2 のシャノンエントロピー) を記録する。
            answer.noul.map(|p_true| {
                let p_false = (1.0 - p_true).clamp(0.0, 1.0);
                if p_true > 0.0 && p_false > 0.0 {
                    let h =
                        -(p_true * p_true.ln() + p_false * p_false.ln()) / std::f64::consts::LN_2;
                    h.clamp(0.0, 1.0)
                } else {
                    0.0
                }
            })
        }
        None => None,
    };

    // OOD 判定 (Pre-filter / Circuit Breaker): Choice 型かつ ood_enabled。
    let is_choice = question_type == Some(QuestionType::Choice);

    let (is_ood, ood_reason) = if config.ood_enabled && is_choice {
        if let Some(e) = energy {
            let is_hard_energy_ood = e > config.energy_threshold;
            let is_hybrid_ood = if let (Some(loose_th), Some(min_conf), Some(min_margin)) = (
                config.loose_energy_threshold,
                config.ood_min_confidence,
                config.ood_min_margin,
            ) {
                let m = margin.unwrap_or(0.0);
                e > loose_th && (conf < min_conf || m < min_margin)
            } else {
                false
            };

            if is_hard_energy_ood {
                (
                    true,
                    Some(format!(
                        "正規化自由エネルギー({:.3} > {:.3})が閾値を超過したため、未定義カテゴリまたは該当なし(OOD)として安全弁フォールバックを適用しました。",
                        e, config.energy_threshold
                    )),
                )
            } else if is_hybrid_ood {
                let m = margin.unwrap_or(0.0);
                (
                    true,
                    Some(format!(
                        "正規化自由エネルギー({:.3} > {:.3})かつ不確実性(確信度={:.3} < {:.3} または マージン={:.3} < {:.3})の複合条件を満たしたため、該当なし(OOD)として安全弁フォールバックを適用しました。",
                        e,
                        config.loose_energy_threshold.unwrap(),
                        conf,
                        config.ood_min_confidence.unwrap(),
                        m,
                        config.ood_min_margin.unwrap()
                    )),
                )
            } else {
                (false, None)
            }
        } else {
            (false, None)
        }
    } else {
        (false, None)
    };

    // 閾値判定とマージンガード (OOD 検知時は最優先で Fallback へ短絡降格)。
    let (route, reason) = if is_ood {
        (
            DecisionRoute::Fallback,
            ood_reason.unwrap_or_else(|| {
                "OOD 安全弁回路が作動したため安全弁フォールバックを適用しました。".to_string()
            }),
        )
    } else if conf >= config.high_threshold - EPSILON {
        if let Some(m) = margin {
            if m < config.top_margin_threshold - EPSILON {
                (
                    DecisionRoute::ConfirmOrEscalate,
                    format!(
                        "実効確信度({:.3} >= {:.3})は高水準ですが、上位2候補の確率差({:.3} < {:.3})が僅差のため確認要求に降格しました。",
                        conf, config.high_threshold, m, config.top_margin_threshold
                    ),
                )
            } else {
                (
                    DecisionRoute::AutoExecute,
                    format!(
                        "実効確信度({:.3} >= {:.3})および上位候補マージン({:.3} >= {:.3})を満たすため自動実行が承認されました。",
                        conf, config.high_threshold, m, config.top_margin_threshold
                    ),
                )
            }
        } else {
            (
                DecisionRoute::AutoExecute,
                format!(
                    "実効確信度({:.3} >= {:.3})が高水準であるため自動実行が承認されました。",
                    conf, config.high_threshold
                ),
            )
        }
    } else if conf >= config.low_threshold - EPSILON {
        (
            DecisionRoute::ConfirmOrEscalate,
            format!(
                "実効確信度({:.3})が確認要求境界 [{:.3}, {:.3}) 内のため二次検証が必要です。",
                conf, config.low_threshold, config.high_threshold
            ),
        )
    } else {
        (
            DecisionRoute::Fallback,
            format!(
                "実効確信度({:.3} < {:.3})が低水準のため安全弁フォールバックを適用しました。",
                conf, config.low_threshold
            ),
        )
    };

    // エスカレーションコンテキストの遅延生成 (ConfirmOrEscalate または Fallback 時のみアロケーションを実行)。
    let escalation = if route != DecisionRoute::AutoExecute {
        let mut candidates: Vec<CandidateProbability> = Vec::new();
        if let Some(probs) = &answer.probabilities {
            candidates = probs
                .iter()
                .map(|(k, &p)| CandidateProbability {
                    candidate: k.clone(),
                    probability: p,
                })
                .collect();
            candidates.sort_by(|a, b| {
                b.probability
                    .partial_cmp(&a.probability)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else if let Some(p_true) = answer.noul {
            let p_false = (1.0 - p_true).clamp(0.0, 1.0);
            if p_true >= p_false {
                candidates.push(CandidateProbability {
                    candidate: "true".to_string(),
                    probability: p_true,
                });
                candidates.push(CandidateProbability {
                    candidate: "false".to_string(),
                    probability: p_false,
                });
            } else {
                candidates.push(CandidateProbability {
                    candidate: "false".to_string(),
                    probability: p_false,
                });
                candidates.push(CandidateProbability {
                    candidate: "true".to_string(),
                    probability: p_true,
                });
            }
        }

        let top_candidates = candidates.iter().take(3).cloned().collect();
        let prompt_template = build_escalation_prompt(question_id, question, &candidates, margin);

        Some(EscalationContext {
            top_candidates,
            margin,
            uncertainty_reason: reason.clone(),
            prompt_template: Some(prompt_template),
        })
    } else {
        None
    };

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
        let p1_str = format!("{:.1}", c1.probability * 100.0);
        let p2_str = format!("{:.1}", c2.probability * 100.0);
        let m_str = match margin {
            Some(m) => format!("(確率差: {:.1}%)", m * 100.0),
            None => String::new(),
        };

        let mut prompt = String::with_capacity(256);
        prompt.push_str("質問「");
        prompt.push_str(q_name);
        prompt.push_str("」(");
        prompt.push_str(instructions);
        prompt.push_str(") について、System 1 の判定では候補「");
        prompt.push_str(&c1.candidate);
        prompt.push_str("」(確信度 ");
        prompt.push_str(&p1_str);
        prompt.push_str("%) と「");
        prompt.push_str(&c2.candidate);
        prompt.push_str("」(確信度 ");
        prompt.push_str(&p2_str);
        prompt.push_str("%) の間で迷いが生じています ");
        prompt.push_str(&m_str);
        prompt.push_str("。State コンテキストを精読し、思考連鎖 (Chain-of-Thought) に基づき最適な決定を行ってください。");
        return prompt;
    }

    let mut prompt = String::with_capacity(128);
    prompt.push_str("質問「");
    prompt.push_str(q_name);
    prompt.push_str("」(");
    prompt.push_str(instructions);
    prompt.push_str(") について、System 1 の判定信頼性が低水準となっています。State コンテキストを精読し、論理的根拠に基づいて決定を下してください。");
    prompt
}

/// リクエストに含まれる全回答のゲーティング情報から集約サマリーを算出する。
///
/// 最悪値ルール(Conservative Policy)を適用する:
/// - 1 つでも `Fallback` が存在する場合 $\rightarrow$ 全体も `Fallback`
/// - そうでなく 1 つでも `ConfirmOrEscalate` が存在する場合 $\rightarrow$ 全体も `ConfirmOrEscalate`
/// - 全ての質問が `AutoExecute` である場合 $\rightarrow$ 全体も `AutoExecute`
pub fn evaluate_response_routing(answers: &IndexMap<String, Answer>) -> SystemRoutingSummary {
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
                let conf = answer.effective_confidence().unwrap_or(0.0);
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
        assert!((config.high_threshold - 0.70).abs() < EPSILON);
        assert!((config.low_threshold - 0.35).abs() < EPSILON);
        assert!((config.top_margin_threshold - 0.15).abs() < EPSILON);
        assert!(config.validate().is_ok());

        let invalid_config = GatingConfig {
            enabled: true,
            high_threshold: 0.30,
            low_threshold: 0.50,
            top_margin_threshold: 0.15,
            ood_enabled: true,
            energy_threshold: -1.0,
            ..GatingConfig::default()
        };
        assert!(invalid_config.validate().is_err());

        let invalid_energy_config = GatingConfig {
            energy_threshold: f64::NAN,
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
        // 確信度は 0.75 と高水準だが、Top-1 と Top-2 が僅差 (0.48 vs 0.44 -> margin 0.04 < 0.15) のケース
        let mut probs = IndexMap::new();
        probs.insert("opt_a".to_string(), 0.48);
        probs.insert("opt_b".to_string(), 0.44);
        probs.insert("opt_c".to_string(), 0.08);
        let answer = Answer::choice("opt_a", probs, 0.75);

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

        // Low: P=0.52 -> C=|2*0.52 - 1| = 0.04 < 0.35
        let ans_low = Answer::noul(0.52);
        let meta_low = evaluate_answer_gating(&ans_low, Some("noul_q"), None, &config);
        assert_eq!(meta_low.route, DecisionRoute::Fallback);
        assert!(meta_low.escalation.is_some());
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
            ood_enabled: false,
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
}
