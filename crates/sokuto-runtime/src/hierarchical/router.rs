//! # 粗密階層ルーターモジュール
//!
//! 第 1 パス大分類推論、Early-Exit 判定、Soft-Beam 動的クラスタマージ、
//! 第 2 パス細分類推論、および確率結合・全候補空間 Gating 再計算を実行する。

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use sokuto_core::contract::calibration::CalibrationConfig;
use sokuto_core::gating::{
    CandidateProbability, DecisionRoute, EscalationContext, GatingConfig, GatingMetadata,
};
use sokuto_core::math::{normalized_entropy, top_margin};
use sokuto_core::schema::{Answer, Criteria, Question, QuestionType};

use crate::engine::InferenceEngine;
use crate::engine::cache::{CachedStateFeatures, DualSessionEngine};
use crate::engine::coarse::{HierarchicalCoarseScorer, ModelDrivenCoarseScorer};
use crate::engine::escalation::build_rich_escalation_prompt;
use crate::error::{Result, RuntimeError};
use crate::hierarchical::calibration::{
    DEFAULT_COARSE_POWER_ALPHA, DEFAULT_ENTROPY_TEMP_GAMMA, DirichletCalibrator,
    combine_and_reconstruct_probabilities, compute_fine_temperature, compute_normalized_entropy,
    temperature_scaled_softmax,
};
use crate::hierarchical::mapping::HierarchicalMapping;
use crate::tokenizer::JevTokenizer;

/// デフォルトの Soft-Beam 発動差分マージン閾値 ($\tau_{\text{beam}}$)。
pub const DEFAULT_BEAM_MARGIN_THRESHOLD: f64 = 0.35;

/// デフォルトの階層 Early-Exit 複合確信度閾値 ($\tau_{\text{escalate}}$)。
pub const DEFAULT_ESCALATE_CONFIDENCE_THRESHOLD: f64 = 0.25;

/// デフォルトの受け皿候補バイアス閾値 ($\tau_{\text{catchall}}$)。
pub const DEFAULT_CATCHALL_THRESHOLD: f64 = 0.10;

/// 粗密階層ルーターのハイパーパラメータ設定。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalRouterConfig {
    /// Soft-Beam 発動閾値 ($M_{\text{coarse}} < \tau_{\text{beam}}$ で Top-1 と Top-2 をマージ)。
    pub beam_margin_threshold: f64,
    /// Early-Exit 閾値 ($S_{\text{coarse}} < \tau_{\text{escalate}}$ で第 2 パスをスキップ)。
    pub escalate_confidence_threshold: f64,
    /// 受け皿 (None / その他) 候補優先採択閾値 ($P \ge \tau_{\text{catchall}}$)。
    pub catchall_threshold: f64,
    /// 大分類エントロピー連動温度係数 ($\gamma$)。
    pub entropy_temp_gamma: f64,
    /// 大分類確率減衰べき乗指数 ($\alpha$)。
    pub coarse_power_alpha: f64,
    /// Soft-Beam 機構を有効にするか。
    pub enable_soft_beam: bool,
    /// Early-Exit 機構を有効にするか。
    pub enable_early_exit: bool,
    /// 受け皿候補優先バイアスを有効にするか。
    pub enable_catchall_bias: bool,
    /// 第 1 パス大分類確率に対する ODIR 正則化 Dirichlet 較正器 (任意)。
    pub coarse_calibrator: Option<DirichletCalibrator>,
}

impl Default for HierarchicalRouterConfig {
    fn default() -> Self {
        Self {
            beam_margin_threshold: DEFAULT_BEAM_MARGIN_THRESHOLD,
            escalate_confidence_threshold: DEFAULT_ESCALATE_CONFIDENCE_THRESHOLD,
            catchall_threshold: DEFAULT_CATCHALL_THRESHOLD,
            entropy_temp_gamma: DEFAULT_ENTROPY_TEMP_GAMMA,
            coarse_power_alpha: DEFAULT_COARSE_POWER_ALPHA,
            enable_soft_beam: true,
            enable_early_exit: true,
            enable_catchall_bias: true,
            coarse_calibrator: None,
        }
    }
}

impl HierarchicalRouterConfig {
    /// ディリクレ較正器を設定する。
    pub fn with_coarse_calibrator(mut self, calibrator: DirichletCalibrator) -> Self {
        self.coarse_calibrator = Some(calibrator);
        self
    }
    /// 設定パラメータの妥当性を検証する。
    pub fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.beam_margin_threshold) {
            return Err(RuntimeError::InvalidQuestion(
                "beam_margin_threshold は 0.0〜1.0 の範囲である必要があります。".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.escalate_confidence_threshold) {
            return Err(RuntimeError::InvalidQuestion(
                "escalate_confidence_threshold は 0.0〜1.0 の範囲である必要があります。"
                    .to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.catchall_threshold) {
            return Err(RuntimeError::InvalidQuestion(
                "catchall_threshold は 0.0〜1.0 の範囲である必要があります。".to_string(),
            ));
        }
        if self.entropy_temp_gamma < 0.0 {
            return Err(RuntimeError::InvalidQuestion(
                "entropy_temp_gamma は 0.0 以上である必要があります。".to_string(),
            ));
        }
        if !(0.0..=2.0).contains(&self.coarse_power_alpha) {
            return Err(RuntimeError::InvalidQuestion(
                "coarse_power_alpha は 0.0〜2.0 の範囲である必要があります。".to_string(),
            ));
        }
        Ok(())
    }
}

/// 粗密二段階推論の実行トレース詳細情報。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalExecutionTrace {
    /// 第 1 パス (大分類) の Top-1 クラスタキー。
    pub top1_coarse: String,
    /// 第 1 パス (大分類) の Top-1 確率。
    pub top1_coarse_prob: f64,
    /// 第 1 パス (大分類) の Top-2 クラスタキー。
    pub top2_coarse: Option<String>,
    /// 第 1 パス (大分類) の Top-2 確率。
    pub top2_coarse_prob: Option<f64>,
    /// 第 1 パスの差分マージン ($M_{\text{coarse}} = P(C_{(1)}) - P(C_{(2)})$)。
    pub coarse_margin: f64,
    /// 第 1 パスの正規化エントロピー ($\tilde{H}(\mathbf{P}_C)$)。
    pub coarse_normalized_entropy: f64,
    /// 第 1 パスの複合確信度スコア ($S_{\text{coarse}}$)。
    pub coarse_composite_confidence: f64,
    /// Early-Exit が発動したか。
    pub early_exit_triggered: bool,
    /// Early-Exit が発動した場合の理由。
    pub early_exit_reason: Option<String>,
    /// Soft-Beam が発動したか。
    pub soft_beam_triggered: bool,
    /// 第 2 パスへ投入された大分類クラスタ群。
    pub selected_coarse_clusters: Vec<String>,
    /// 第 2 パスで実際に評価された細分類候補数。
    pub evaluated_fine_count: usize,
    /// 第 2 パスに適用された動的温度 ($T_{\text{fine}}$)。
    pub fine_temperature: f64,
    /// 受け皿候補優先バイアスが発動したか。
    pub catchall_bias_triggered: bool,
}

/// 粗密階層ルーティングの最終実行結果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalResult {
    /// 既存 API 互換の型付き決定。
    pub answer: Answer,
    /// 階層推論の詳細実行トレース。
    pub trace: HierarchicalExecutionTrace,
}

impl HierarchicalResult {
    /// `Answer` 型へ消費変換する。
    pub fn into_answer(self) -> Answer {
        self.answer
    }

    /// `Answer` への参照を取得する。
    pub fn answer(&self) -> &Answer {
        &self.answer
    }

    /// 実行トレースへの参照を取得する。
    pub fn trace(&self) -> &HierarchicalExecutionTrace {
        &self.trace
    }
}

/// 粗密階層ルーティングで使用する推論エンジンバックエンド。
pub enum HierarchicalEngine<'a> {
    /// 単一 ONNX モデルによる通常推論エンジン。
    Single(&'a InferenceEngine),
    /// 中間特徴量キャッシュを活用するデュアルセッションエンジン。
    Dual {
        /// デュアルセッションエンジンへの参照。
        engine: &'a DualSessionEngine,
        /// キャッシュ済みの中間特徴量への参照。
        cached_state: &'a CachedStateFeatures,
    },
}

/// 粗密二段階階層ルーティングを実行するオーケストレータ。
pub struct CoarseToFineRouter<'a> {
    backend: HierarchicalEngine<'a>,
    tokenizer: &'a JevTokenizer,
    mapping: &'a HierarchicalMapping,
    calib_config: &'a CalibrationConfig,
    config: HierarchicalRouterConfig,
}

impl<'a> CoarseToFineRouter<'a> {
    /// 新しいルーターインスタンスを単一推論エンジンから生成する。
    pub fn new(
        engine: &'a InferenceEngine,
        tokenizer: &'a JevTokenizer,
        mapping: &'a HierarchicalMapping,
        calib_config: &'a CalibrationConfig,
        config: HierarchicalRouterConfig,
    ) -> Result<Self> {
        Self::with_backend(
            HierarchicalEngine::Single(engine),
            tokenizer,
            mapping,
            calib_config,
            config,
        )
    }

    /// キャッシュ済み中間特徴量を保持するデュアルセッションエンジンから新しいルーターインスタンスを生成する。
    pub fn new_dual(
        engine: &'a DualSessionEngine,
        cached_state: &'a CachedStateFeatures,
        tokenizer: &'a JevTokenizer,
        mapping: &'a HierarchicalMapping,
        calib_config: &'a CalibrationConfig,
        config: HierarchicalRouterConfig,
    ) -> Result<Self> {
        Self::with_backend(
            HierarchicalEngine::Dual {
                engine,
                cached_state,
            },
            tokenizer,
            mapping,
            calib_config,
            config,
        )
    }

    /// 任意の推論バックエンドからルーターインスタンスを生成する。
    pub fn with_backend(
        backend: HierarchicalEngine<'a>,
        tokenizer: &'a JevTokenizer,
        mapping: &'a HierarchicalMapping,
        calib_config: &'a CalibrationConfig,
        config: HierarchicalRouterConfig,
    ) -> Result<Self> {
        config.validate()?;
        mapping.validate()?;

        if let Some(ref calibrator) = config.coarse_calibrator
            && calibrator.num_classes != mapping.total_coarse_count()
        {
            return Err(RuntimeError::InvalidQuestion(format!(
                "DirichletCalibrator のクラス数 ({}) がマッピングの大分類数 ({}) と一致しません。",
                calibrator.num_classes,
                mapping.total_coarse_count()
            )));
        }

        Ok(Self {
            backend,
            tokenizer,
            mapping,
            calib_config,
            config,
        })
    }

    /// 階層ルーティング推論を実行する。
    ///
    /// # パイプライン
    /// 1. **第 1 パス (大分類推論 & Dirichlet 較正)**: `ModelDrivenCoarseScorer` で大分類確率を算出し、設定に応じて ODIR Dirichlet 較正を適用。
    /// 2. **Early-Exit 判定**: 複合確信度 $S_{\text{coarse}} < \tau_{\text{escalate}}$ の場合、即時脱出。
    /// 3. **Soft-Beam 判定**: Top-Margin $M_{\text{coarse}} < \tau_{\text{beam}}$ の場合、Top-1 と Top-2 配下の候補をマージ。
    /// 4. **第 2 パス (細分類推論 & 生ロジット取得)**: マージされた候補群に対して生ロジットを直接抽出。
    /// 5. **温度調整 & 確率結合**: エントロピー連動温度・ベース温度を生ロジットへ直接適用 (1 回の Softmax) し、大分類確率と減衰周辺化結合。
    /// 6. **受け皿候補バイアス**: $P(\text{その他}) \ge \tau_{\text{catchall}}$ 判定。
    /// 7. **全候補空間 Gating 再計算**: 全細分類候補空間基準で Gating メタデータを正規化。
    pub fn route(
        &self,
        state: &str,
        instructions: &str,
        gating_config: Option<&GatingConfig>,
    ) -> Result<HierarchicalResult> {
        // --- Step 1: 第 1 パス (大分類推論 & Dirichlet 較正) ---
        let coarse_probabilities = match self.backend {
            HierarchicalEngine::Single(single_engine) => {
                let coarse_scorer = ModelDrivenCoarseScorer::new(single_engine, self.tokenizer)
                    .with_calibration(self.calib_config);
                coarse_scorer.score_coarse(state, instructions, &self.mapping.coarse_categories)?
            }
            HierarchicalEngine::Dual {
                engine,
                cached_state,
            } => {
                let coarse_question = Question {
                    question_type: QuestionType::Choice,
                    instructions: instructions.to_string(),
                    criteria: Some(sokuto_core::schema::Criteria::Map(
                        self.mapping.coarse_categories.clone(),
                    )),
                };
                let raw_logits =
                    engine.forward_question_raw(self.tokenizer, cached_state, &coarse_question)?;
                let base_temp = self
                    .calib_config
                    .get_temperature(QuestionType::Choice, self.mapping.coarse_categories.len());
                let probs = temperature_scaled_softmax(&raw_logits, base_temp);
                let mut map = IndexMap::with_capacity(self.mapping.coarse_categories.len());
                for (i, key) in self.mapping.coarse_categories.keys().enumerate() {
                    map.insert(key.clone(), probs[i]);
                }
                map
            }
        };

        // ODIR 正則化 Dirichlet 較正の適用 (設定されている場合)
        let coarse_probabilities = if let Some(ref calibrator) = self.config.coarse_calibrator {
            let all_coarse = self.mapping.all_coarse_keys();
            let uncalibrated: Vec<f64> = all_coarse
                .iter()
                .map(|k| coarse_probabilities.get(k).copied().unwrap_or(0.0))
                .collect();
            let calibrated_vec = calibrator.calibrate(&uncalibrated)?;
            let mut calibrated_map = IndexMap::with_capacity(all_coarse.len());
            for (k, p) in all_coarse.iter().zip(calibrated_vec) {
                calibrated_map.insert(k.clone(), p);
            }
            calibrated_map
        } else {
            coarse_probabilities
        };

        // 決定論的降順ソート (確率同値時は元の大分類インデックス昇順)
        let all_coarse_keys = self.mapping.all_coarse_keys();
        let mut sorted_coarse: Vec<(&str, f64)> = all_coarse_keys
            .iter()
            .map(|k| {
                (
                    k.as_str(),
                    coarse_probabilities.get(k).copied().unwrap_or(0.0),
                )
            })
            .collect();

        sorted_coarse.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let idx_a = all_coarse_keys
                        .iter()
                        .position(|k| k == a.0)
                        .unwrap_or(usize::MAX);
                    let idx_b = all_coarse_keys
                        .iter()
                        .position(|k| k == b.0)
                        .unwrap_or(usize::MAX);
                    idx_a.cmp(&idx_b)
                })
        });

        if sorted_coarse.is_empty() {
            return Err(RuntimeError::InvalidQuestion(
                "第 1 パスの大分類確率分布が空です。".to_string(),
            ));
        }

        let top1_coarse = sorted_coarse[0].0.to_string();
        let top1_prob = sorted_coarse[0].1;
        let (top2_coarse, top2_prob): (Option<String>, Option<f64>) = if sorted_coarse.len() > 1 {
            (
                Some(sorted_coarse[1].0.to_string()),
                Some(sorted_coarse[1].1),
            )
        } else {
            (None, None)
        };

        let coarse_probs_slice: Vec<f64> = sorted_coarse.iter().map(|(_, p)| *p).collect();
        let (_, norm_entropy) = compute_normalized_entropy(&coarse_probs_slice);

        let coarse_margin: f64 = if let Some(p2) = top2_prob {
            (top1_prob - p2).max(0.0)
        } else {
            1.0
        };

        // 複合確信度 S_coarse = P(C_(1)) * (1 - H_norm)
        let composite_confidence: f64 = top1_prob * (1.0 - norm_entropy);

        // --- Step 2: Early-Exit 判定 ---
        if self.config.enable_early_exit
            && composite_confidence < self.config.escalate_confidence_threshold
        {
            return build_early_exit_result(
                self.mapping,
                state,
                instructions,
                &sorted_coarse,
                coarse_margin,
                norm_entropy,
                composite_confidence,
                self.config.escalate_confidence_threshold,
            );
        }

        // --- Step 3: Soft-Beam 判定と細分類候補抽出 ---
        let mut selected_clusters = vec![top1_coarse.clone()];
        let mut soft_beam_triggered = false;

        if self.config.enable_soft_beam
            && coarse_margin < self.config.beam_margin_threshold
            && let Some(ref c2) = top2_coarse
        {
            selected_clusters.push(c2.clone());
            soft_beam_triggered = true;
        }

        let mut fine_candidates = IndexMap::new();
        for cluster in &selected_clusters {
            if let Some(f_keys) = self.mapping.fine_keys_for_coarse(cluster) {
                for fk in f_keys {
                    if let Some(desc) = self.mapping.fine_criteria.get(fk) {
                        fine_candidates.insert(fk.clone(), desc.clone());
                    }
                }
            }
        }

        if fine_candidates.is_empty() {
            return Err(RuntimeError::InvalidQuestion(
                "選択された大分類クラスタ配下の細分類候補が空です。".to_string(),
            ));
        }

        // --- Step 4: 第 2 パス (細分類推論 & 生ロジット取得) ---
        let fine_question = Question {
            question_type: QuestionType::Choice,
            instructions: instructions.to_string(),
            criteria: Some(Criteria::Map(fine_candidates.clone())),
        };

        // Softmax 前の生ロジットを直接抽出 (二重 Softmax & 疑似ロジット対数逆算の歪みを根絶)
        let raw_logits = match self.backend {
            HierarchicalEngine::Single(single_engine) => {
                single_engine.forward_question_raw(self.tokenizer, state, &fine_question)?
            }
            HierarchicalEngine::Dual {
                engine,
                cached_state,
            } => engine.forward_question_raw(self.tokenizer, cached_state, &fine_question)?,
        };

        // --- Step 5: 温度調整 & 確率結合 ---
        let fine_temperature =
            compute_fine_temperature(norm_entropy, self.config.entropy_temp_gamma);

        // ベース較正温度と動的温度を乗算した実効温度
        let base_temp = self
            .calib_config
            .get_temperature(QuestionType::Choice, fine_candidates.len());
        let effective_temp = (base_temp * fine_temperature).max(0.01);

        // 生ロジットに対して実効温度で 1 回だけ Softmax を適用
        let temp_scaled_probs = temperature_scaled_softmax(&raw_logits, effective_temp);

        let mut evaluated_fine_probs = IndexMap::with_capacity(fine_candidates.len());
        for (i, (k, _)) in fine_candidates.iter().enumerate() {
            evaluated_fine_probs.insert(k.clone(), temp_scaled_probs[i]);
        }

        let full_probabilities = combine_and_reconstruct_probabilities(
            self.mapping.all_fine_keys(),
            &self.mapping.fine_to_coarse,
            &coarse_probabilities,
            &evaluated_fine_probs,
            self.config.coarse_power_alpha,
        )?;

        // --- Step 6: 決定論的勝者選定および受け皿候補バイアス ---
        let all_fine_keys = self.mapping.all_fine_keys();
        let mut sorted_full: Vec<(&str, f64)> = all_fine_keys
            .iter()
            .map(|k| {
                (
                    k.as_str(),
                    full_probabilities.get(k).copied().unwrap_or(0.0),
                )
            })
            .collect();

        sorted_full.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let idx_a = all_fine_keys
                        .iter()
                        .position(|k| k == a.0)
                        .unwrap_or(usize::MAX);
                    let idx_b = all_fine_keys
                        .iter()
                        .position(|k| k == b.0)
                        .unwrap_or(usize::MAX);
                    idx_a.cmp(&idx_b)
                })
        });

        let default_winner = sorted_full[0].0.to_string();
        let default_winner_prob = sorted_full[0].1;

        let mut final_winner = default_winner.clone();
        let mut catchall_bias_triggered = false;

        if self.config.enable_catchall_bias {
            // ネガティブ候補で閾値以上のものを探索
            for (key, &prob) in &full_probabilities {
                let desc = self
                    .mapping
                    .fine_criteria
                    .get(key)
                    .map(|s| s.as_str())
                    .unwrap_or("");
                if self.mapping.is_negative_candidate(key, desc)
                    && prob >= self.config.catchall_threshold
                {
                    final_winner = key.clone();
                    catchall_bias_triggered = true;
                    break;
                }
            }
        }

        // --- Step 7: 全候補空間基準での Gating 再計算 ---
        let all_probs_slice: Vec<f64> = all_fine_keys
            .iter()
            .map(|k| full_probabilities.get(k).copied().unwrap_or(0.0))
            .collect();

        let final_norm_entropy = normalized_entropy(&all_probs_slice).unwrap_or(0.0);
        let final_margin = top_margin(&all_probs_slice).unwrap_or(0.0);

        let final_confidence = if catchall_bias_triggered {
            full_probabilities
                .get(&final_winner)
                .copied()
                .unwrap_or(default_winner_prob)
        } else {
            default_winner_prob
        };

        // 生ロジットから正規化自由エネルギーを正確に算出
        let fine_energy = crate::engine::gating::calculate_choice_energy(
            &fine_question,
            &raw_logits,
            self.calib_config,
        );

        // Gating 判定
        let default_gating_config = GatingConfig::default();
        let gc = gating_config.unwrap_or(&default_gating_config);
        let final_route = if final_confidence >= gc.thresholds.high_threshold
            && final_margin >= gc.thresholds.top_margin_threshold
        {
            DecisionRoute::AutoExecute
        } else if final_confidence < gc.thresholds.low_threshold {
            DecisionRoute::Fallback
        } else {
            DecisionRoute::ConfirmOrEscalate
        };

        let gating = build_final_gating_metadata(
            self.mapping,
            state,
            instructions,
            &sorted_full,
            final_route,
            final_confidence,
            final_norm_entropy,
            final_margin,
            fine_energy,
        );

        let answer =
            Answer::choice(final_winner, full_probabilities, final_confidence).with_gating(gating);

        let trace = HierarchicalExecutionTrace {
            top1_coarse,
            top1_coarse_prob: top1_prob,
            top2_coarse,
            top2_coarse_prob: top2_prob,
            coarse_margin,
            coarse_normalized_entropy: norm_entropy,
            coarse_composite_confidence: composite_confidence,
            early_exit_triggered: false,
            early_exit_reason: None,
            soft_beam_triggered,
            selected_coarse_clusters: selected_clusters,
            evaluated_fine_count: fine_candidates.len(),
            fine_temperature,
            catchall_bias_triggered,
        };

        Ok(HierarchicalResult { answer, trace })
    }
}

/// `AutoExecute` 以外のルーティング時に、`GatingMetadata` へ `EscalationContext` を付与する。
///
/// 全細分類定義 (`mapping.fine_criteria`) を含む `Question` を構築し、
/// [`build_rich_escalation_prompt`] で System 2 への CoT 委託プロンプト案を生成して格納する。
/// `gating.route == DecisionRoute::AutoExecute` の場合はアロケーションを行わず何もしない。
fn attach_escalation_context(
    mapping: &HierarchicalMapping,
    state: &str,
    instructions: &str,
    gating: &mut GatingMetadata,
    top_candidates: Vec<CandidateProbability>,
) {
    if gating.route == DecisionRoute::AutoExecute {
        return;
    }

    let escalation_question = Question {
        question_type: QuestionType::Choice,
        instructions: instructions.to_string(),
        criteria: Some(Criteria::Map(mapping.fine_criteria.clone())),
    };

    let prompt_template = build_rich_escalation_prompt(
        Some(state),
        None,
        Some(&escalation_question),
        gating,
        &top_candidates,
    );

    gating.escalation = Some(EscalationContext {
        top_candidates,
        margin: gating.margin,
        uncertainty_reason: gating.reason.clone(),
        prompt_template: Some(prompt_template),
    });
}

/// 通常ルーティング終了時 (Step 7) の `GatingMetadata` を構築する。
///
/// `final_route != DecisionRoute::AutoExecute` の場合は、全候補空間へ復元された確率分布の上位 3 候補を抽出し、
/// `EscalationContext` および System 2 委託プロンプト案を付与する。
#[allow(clippy::too_many_arguments)]
fn build_final_gating_metadata(
    mapping: &HierarchicalMapping,
    state: &str,
    instructions: &str,
    sorted_full: &[(&str, f64)],
    final_route: DecisionRoute,
    final_confidence: f64,
    final_norm_entropy: f64,
    final_margin: f64,
    fine_energy: Option<f64>,
) -> GatingMetadata {
    let mut gating = GatingMetadata {
        route: final_route,
        confidence: final_confidence,
        entropy: Some(final_norm_entropy),
        margin: Some(final_margin),
        energy: fine_energy,
        is_ood: false,
        reason: format!("Hierarchical decision routing (route={})", final_route),
        escalation: None,
    };

    if final_route != DecisionRoute::AutoExecute {
        let top_candidates: Vec<CandidateProbability> = sorted_full
            .iter()
            .take(3)
            .map(|(k, p)| CandidateProbability {
                candidate: (*k).to_string(),
                probability: *p,
            })
            .collect();
        attach_escalation_context(mapping, state, instructions, &mut gating, top_candidates);
    }

    gating
}

/// 第 1 パスの複合確信度不足による Early-Exit 時のフォールバック結果を構築する。
///
/// 第 1 パスの Top-1 クラスタ配下にある先頭細分類を `Answer.choice` として選定し、
/// `DecisionRoute::ConfirmOrEscalate` に対応する `EscalationContext`
/// (上位候補リストおよび `build_rich_escalation_prompt` によるプロンプト案) を付与する。
///
/// # 引数
/// - `mapping`: 階層マッピング定義。
/// - `state`: 推論対象のコンテキストテキスト。
/// - `instructions`: 質問の判定指示文。
/// - `sorted_coarse`: 確率降順に並んだ第 1 パスの大分類スコア一覧。
/// - `coarse_margin`: 第 1 パスの Top-1 と Top-2 の確率差。
/// - `norm_entropy`: 第 1 パスの正規化エントロピー。
/// - `composite_confidence`: 第 1 パスの複合確信度スコア。
/// - `escalate_confidence_threshold`: Early-Exit 判定に用いた閾値。
///
/// # 戻り値
/// Early-Exit 用の `HierarchicalResult`。
#[allow(clippy::too_many_arguments)]
fn build_early_exit_result(
    mapping: &HierarchicalMapping,
    state: &str,
    instructions: &str,
    sorted_coarse: &[(&str, f64)],
    coarse_margin: f64,
    norm_entropy: f64,
    composite_confidence: f64,
    escalate_confidence_threshold: f64,
) -> Result<HierarchicalResult> {
    let (top1_coarse, top1_prob) = sorted_coarse.first().copied().ok_or_else(|| {
        RuntimeError::InvalidQuestion("第 1 パスの大分類確率分布が空です。".to_string())
    })?;
    let (top2_coarse, top2_prob): (Option<String>, Option<f64>) =
        if let Some((c2, p2)) = sorted_coarse.get(1) {
            (Some((*c2).to_string()), Some(*p2))
        } else {
            (None, None)
        };

    let all_fine_keys = mapping.all_fine_keys();
    if all_fine_keys.is_empty() {
        return Err(RuntimeError::InvalidQuestion(
            "階層マッピングの細分類候補が空です。".to_string(),
        ));
    }

    let selected_choice = mapping
        .fine_keys_for_coarse(top1_coarse)
        .and_then(|keys| keys.first())
        .cloned()
        .unwrap_or_else(|| all_fine_keys[0].clone());

    let uniform_prob = 1.0 / (all_fine_keys.len() as f64);
    let mut full_probs = IndexMap::with_capacity(all_fine_keys.len());
    for k in all_fine_keys {
        full_probs.insert(k.clone(), uniform_prob);
    }

    let mut top_candidates: Vec<CandidateProbability> = Vec::with_capacity(3);
    let mut seen = std::collections::HashSet::with_capacity(3);

    for (coarse_key, _) in sorted_coarse {
        if top_candidates.len() >= 3 {
            break;
        }
        if let Some(first_fine) = mapping
            .fine_keys_for_coarse(coarse_key)
            .and_then(|keys| keys.first())
            && seen.insert(first_fine.as_str())
        {
            top_candidates.push(CandidateProbability {
                candidate: first_fine.clone(),
                probability: uniform_prob,
            });
        }
    }

    if top_candidates.len() < 3
        && let Some(top1_fines) = mapping.fine_keys_for_coarse(top1_coarse)
    {
        for fk in top1_fines {
            if top_candidates.len() >= 3 {
                break;
            }
            if seen.insert(fk.as_str()) {
                top_candidates.push(CandidateProbability {
                    candidate: fk.clone(),
                    probability: uniform_prob,
                });
            }
        }
    }

    if top_candidates.len() < 3 {
        for fk in all_fine_keys {
            if top_candidates.len() >= 3 {
                break;
            }
            if seen.insert(fk.as_str()) {
                top_candidates.push(CandidateProbability {
                    candidate: fk.clone(),
                    probability: uniform_prob,
                });
            }
        }
    }

    let reason = "Hierarchical Early-Exit: composite confidence below threshold".to_string();
    let entropy_val = if all_fine_keys.len() > 1 { 1.0 } else { 0.0 };
    let mut fallback_gating = GatingMetadata {
        route: DecisionRoute::ConfirmOrEscalate,
        confidence: composite_confidence,
        entropy: Some(entropy_val),
        margin: Some(0.0),
        energy: None,
        is_ood: false,
        reason,
        escalation: None,
    };

    attach_escalation_context(
        mapping,
        state,
        instructions,
        &mut fallback_gating,
        top_candidates,
    );

    let fallback_answer = Answer::choice(selected_choice, full_probs, composite_confidence)
        .with_gating(fallback_gating);

    let trace = HierarchicalExecutionTrace {
        top1_coarse: top1_coarse.to_string(),
        top1_coarse_prob: top1_prob,
        top2_coarse,
        top2_coarse_prob: top2_prob,
        coarse_margin,
        coarse_normalized_entropy: norm_entropy,
        coarse_composite_confidence: composite_confidence,
        early_exit_triggered: true,
        early_exit_reason: Some(format!(
            "複合確信度 ({:.4}) が閾値 ({:.4}) 未満のため Early-Exit しました。",
            composite_confidence, escalate_confidence_threshold
        )),
        soft_beam_triggered: false,
        selected_coarse_clusters: vec![top1_coarse.to_string()],
        evaluated_fine_count: 0,
        fine_temperature: 1.0,
        catchall_bias_triggered: false,
    };

    Ok(HierarchicalResult {
        answer: fallback_answer,
        trace,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_exit_selects_top1_coarse_first_candidate_and_builds_escalation() {
        let mapping = HierarchicalMapping::banking77();
        let all_fine_keys = mapping.all_fine_keys();
        // オントロジー全体の先頭は card_ops 配下であることを前提検証する。
        assert_eq!(mapping.coarse_for_fine(&all_fine_keys[0]), Some("card_ops"));

        // 第 1 パスの Top-1 が card_ops 以外 (transfers) のケースを模擬する。
        let sorted_coarse = vec![
            ("transfers", 0.22),
            ("payments", 0.20),
            ("card_ops", 0.18),
            ("account_sec", 0.15),
            ("fx_fees", 0.10),
            ("top_up", 0.08),
            ("general", 0.07),
        ];

        let state = "海外への銀行送金がまだ着金していません。手数料も確認したいです。";
        let instructions = "お客様のお問い合わせ内容に最も適合するカテゴリを選択してください。";

        let res = build_early_exit_result(
            &mapping,
            state,
            instructions,
            &sorted_coarse,
            0.02,
            0.96,
            0.0088,
            0.25,
        )
        .expect("Early-Exit 結果の構築に成功すること。");

        let expected_top1_first = mapping
            .fine_keys_for_coarse("transfers")
            .and_then(|keys| keys.first())
            .expect("transfers 配下に細分類が存在すること。");
        let expected_top2_first = mapping
            .fine_keys_for_coarse("payments")
            .and_then(|keys| keys.first())
            .expect("payments 配下に細分類が存在すること。");
        let expected_top3_first = mapping
            .fine_keys_for_coarse("card_ops")
            .and_then(|keys| keys.first())
            .expect("card_ops 配下に細分類が存在すること。");

        // 1. choice がオントロジー先頭ではなく Top-1 クラスタ配下の先頭候補であること
        assert_ne!(res.answer().as_choice(), Some(all_fine_keys[0].as_str()));
        assert_eq!(res.answer().as_choice(), Some(expected_top1_first.as_str()));
        assert_eq!(
            mapping.coarse_for_fine(res.answer().as_choice().unwrap()),
            Some("transfers")
        );

        // 2. 全 77 候補の一様分布が保持されていること
        let probs = res
            .answer()
            .probabilities()
            .expect("確率分布が存在すること。");
        assert_eq!(probs.len(), 77);
        let sum: f64 = probs.values().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // 3. Gating 契約 (ConfirmOrEscalate 時は escalation: Some(...)) を満たすこと
        let gating = res
            .answer()
            .gating
            .as_ref()
            .expect("Gating が付与されていること。");
        assert_eq!(gating.route, DecisionRoute::ConfirmOrEscalate);
        let esc = gating
            .escalation
            .as_ref()
            .expect("ConfirmOrEscalate 時は EscalationContext が必須であること。");

        assert_eq!(esc.top_candidates.len(), 3);
        assert_eq!(esc.top_candidates[0].candidate, *expected_top1_first);
        assert_eq!(esc.top_candidates[1].candidate, *expected_top2_first);
        assert_eq!(esc.top_candidates[2].candidate, *expected_top3_first);
        assert_eq!(esc.margin, Some(0.0));
        assert_eq!(esc.uncertainty_reason, gating.reason);

        let prompt = esc
            .prompt_template
            .as_ref()
            .expect("prompt_template が生成されていること。");
        assert!(prompt.contains("<context>"));
        assert!(prompt.contains("</context>"));
        assert!(prompt.contains(state));
        assert!(prompt.contains(instructions));
        assert!(prompt.contains(expected_top1_first));

        // 4. トレースの整合性確認
        assert!(res.trace().early_exit_triggered);
        assert_eq!(res.trace().top1_coarse, "transfers");
        assert_eq!(res.trace().selected_coarse_clusters, vec!["transfers"]);
        assert_eq!(res.trace().evaluated_fine_count, 0);
    }

    #[test]
    fn test_early_exit_single_cluster_edge_case() {
        let mut builder = HierarchicalMapping::builder("SingleCluster");
        builder.add_category(
            "only_cluster",
            "唯一のクラスタ",
            vec![("fine_a", "候補 A の説明"), ("fine_b", "候補 B の説明")],
        );
        let mapping = builder.build().expect("マッピング構築に成功すること。");

        let sorted_coarse = vec![("only_cluster", 1.0)];
        let res = build_early_exit_result(
            &mapping,
            "判断対象のテキスト",
            "適切な候補を選択してください。",
            &sorted_coarse,
            1.0,
            0.0,
            0.1,
            0.25,
        )
        .expect("単一クラスタでも Early-Exit 構築に成功すること。");

        assert_eq!(res.answer().as_choice(), Some("fine_a"));
        let esc = res
            .answer()
            .gating
            .as_ref()
            .and_then(|g| g.escalation.as_ref())
            .expect("EscalationContext が付与されていること。");
        assert_eq!(esc.top_candidates.len(), 2);
        assert_eq!(esc.top_candidates[0].candidate, "fine_a");
        assert_eq!(esc.top_candidates[1].candidate, "fine_b");
        assert!(esc.prompt_template.is_some());
    }

    #[test]
    fn test_final_gating_metadata_escalation_contract() {
        let mapping = HierarchicalMapping::banking77();
        let state = "カードの請求金額に身に覚えのない引き落としがあります。";
        let instructions = "お客様のお問い合わせ内容に最も適合するカテゴリを選択してください。";
        let sorted_full = vec![
            ("card_payment_not_recognised", 0.42),
            ("direct_debit_payment_not_recognised", 0.31),
            ("compromised_card", 0.12),
            ("lost_or_stolen_card", 0.05),
        ];

        // 1. AutoExecute 時はゼロアロケーション契約により escalation が None であること
        let auto_meta = build_final_gating_metadata(
            &mapping,
            state,
            instructions,
            &sorted_full,
            DecisionRoute::AutoExecute,
            0.85,
            0.12,
            0.40,
            Some(-2.5),
        );
        assert_eq!(auto_meta.route, DecisionRoute::AutoExecute);
        assert!(auto_meta.escalation.is_none());

        // 2. ConfirmOrEscalate / Fallback 時は上位 3 候補と委託プロンプトが付与されること
        for route in [DecisionRoute::ConfirmOrEscalate, DecisionRoute::Fallback] {
            let meta = build_final_gating_metadata(
                &mapping,
                state,
                instructions,
                &sorted_full,
                route,
                0.42,
                0.55,
                0.11,
                Some(-1.2),
            );
            assert_eq!(meta.route, route);
            assert_eq!(meta.margin, Some(0.11));
            assert_eq!(meta.energy, Some(-1.2));

            let esc = meta
                .escalation
                .as_ref()
                .expect("非 AutoExecute 時は EscalationContext が必須であること。");
            assert_eq!(esc.top_candidates.len(), 3);
            assert_eq!(
                esc.top_candidates[0].candidate,
                "card_payment_not_recognised"
            );
            assert!((esc.top_candidates[0].probability - 0.42).abs() < 1e-9);
            assert_eq!(
                esc.top_candidates[1].candidate,
                "direct_debit_payment_not_recognised"
            );
            assert_eq!(esc.top_candidates[2].candidate, "compromised_card");
            assert_eq!(esc.margin, Some(0.11));
            assert_eq!(esc.uncertainty_reason, meta.reason);

            let prompt = esc
                .prompt_template
                .as_ref()
                .expect("prompt_template が生成されていること。");
            assert!(prompt.contains("<context>"));
            assert!(prompt.contains(state));
            assert!(prompt.contains(instructions));
            assert!(prompt.contains("card_payment_not_recognised"));
        }
    }
}
