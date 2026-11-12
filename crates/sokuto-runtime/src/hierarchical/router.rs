//! # 粗密階層ルーターモジュール
//!
//! 第 1 パス大分類推論、Early-Exit 判定、Soft-Beam 動的クラスタマージ、
//! 第 2 パス細分類推論、および確率結合・全候補空間 Gating 再計算を実行する。

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use sokuto_core::contract::calibration::CalibrationConfig;
use sokuto_core::gating::{
    CandidateProbability, DecisionRoute, EscalationContext, GatingConfig, GatingMetadata,
    evaluate_answer_gating_opts,
};
use sokuto_core::math::composite_confidence_iter;
use sokuto_core::schema::{Answer, Criteria, Question, QuestionType};

use crate::engine::InferenceEngine;
use crate::engine::cache::{CachedStateFeatures, DualSessionEngine};
use crate::engine::escalation::build_rich_escalation_prompt;
use crate::engine::gating::{calculate_choice_energy_with_config, resolve_gating_config};
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

/// 粗密階層ルーティングで使用する推論エンジンバックエンド (後方互換用列挙型)。
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

/// 粗密階層ルーティングにおける二段階推論バックエンドの抽象化トレイト。
///
/// [`CoarseToFineRouter`] へ静的ディスパッチ (ゼロコスト抽象化) により推論能力を供給する。
/// 第 1 パス (大分類確率推論) および第 2 パス (細分類生ロジット推論) を提供し、
/// 単一セッション・デュアルセッション・テスト用モック等を統一的に差し替え可能にする。
pub trait HierarchicalPredictor {
    /// 第 1 パス (大分類) の各カテゴリの確率分布を推論する。
    ///
    /// デフォルト実装では大分類用の `Choice` 質問を構築し、[`Self::forward_fine_raw`] で取得した
    /// 未正規化生ロジットに対して `calib` の温度スケーリング付き Softmax を適用して確率マップを生成する。
    ///
    /// # 引数
    /// - `state`: 推論対象のコンテキストテキスト。
    /// - `instructions`: 質問の判定指示文。
    /// - `categories`: 大分類の識別子と説明文の順序付きマップ。
    /// - `calib`: モデルの較正温度設定。
    ///
    /// # 戻り値
    /// 各大分類キーに対応する較正済み確率マップ。
    fn predict_coarse_probabilities(
        &self,
        state: &str,
        instructions: &str,
        categories: &IndexMap<String, String>,
        calib: &CalibrationConfig,
    ) -> Result<IndexMap<String, f64>> {
        let coarse_question = Question {
            question_type: QuestionType::Choice,
            instructions: instructions.to_string(),
            criteria: Some(Criteria::Map(categories.clone())),
        };
        let raw_logits = self.forward_fine_raw(state, &coarse_question)?;
        let base_temp = calib.get_temperature(QuestionType::Choice, categories.len());
        let probs = temperature_scaled_softmax(&raw_logits, base_temp)?;
        if probs.len() != categories.len() {
            return Err(RuntimeError::InvalidQuestion(format!(
                "第 1 パスの出力ロジット数 ({}) が大分類数 ({}) と一致しません。",
                probs.len(),
                categories.len()
            )));
        }
        let mut map = IndexMap::with_capacity(categories.len());
        for (i, key) in categories.keys().enumerate() {
            map.insert(key.clone(), probs[i]);
        }
        Ok(map)
    }

    /// 質問定義に対する未正規化生ロジット列 (`Vec<f64>`) を推論する。
    ///
    /// # 引数
    /// - `state`: 推論対象のコンテキストテキスト。
    /// - `question`: 評価対象の質問仕様。
    ///
    /// # 戻り値
    /// 各選択肢に対応する未正規化生ロジット配列。
    fn forward_fine_raw(&self, state: &str, question: &Question) -> Result<Vec<f64>>;
}

impl<P: HierarchicalPredictor + ?Sized> HierarchicalPredictor for &P {
    fn predict_coarse_probabilities(
        &self,
        state: &str,
        instructions: &str,
        categories: &IndexMap<String, String>,
        calib: &CalibrationConfig,
    ) -> Result<IndexMap<String, f64>> {
        (**self).predict_coarse_probabilities(state, instructions, categories, calib)
    }

    fn forward_fine_raw(&self, state: &str, question: &Question) -> Result<Vec<f64>> {
        (**self).forward_fine_raw(state, question)
    }
}

impl HierarchicalPredictor for (&InferenceEngine, &JevTokenizer) {
    fn forward_fine_raw(&self, state: &str, question: &Question) -> Result<Vec<f64>> {
        self.0.forward_question_raw(self.1, state, question)
    }
}

impl HierarchicalPredictor for (&DualSessionEngine, &CachedStateFeatures, &JevTokenizer) {
    fn forward_fine_raw(&self, _state: &str, question: &Question) -> Result<Vec<f64>> {
        self.0.forward_question_raw(self.2, self.1, question)
    }
}

impl HierarchicalPredictor for (HierarchicalEngine<'_>, &JevTokenizer) {
    fn forward_fine_raw(&self, state: &str, question: &Question) -> Result<Vec<f64>> {
        match &self.0 {
            HierarchicalEngine::Single(engine) => {
                engine.forward_question_raw(self.1, state, question)
            }
            HierarchicalEngine::Dual {
                engine,
                cached_state,
            } => engine.forward_question_raw(self.1, cached_state, question),
        }
    }
}

/// 粗密二段階階層ルーティングを実行するオーケストレータ。
pub struct CoarseToFineRouter<'a, P = (&'a InferenceEngine, &'a JevTokenizer)>
where
    P: HierarchicalPredictor,
{
    predictor: P,
    mapping: &'a HierarchicalMapping,
    calib_config: &'a CalibrationConfig,
    config: HierarchicalRouterConfig,
}

impl<'a> CoarseToFineRouter<'a, (&'a InferenceEngine, &'a JevTokenizer)> {
    /// 新しいルーターインスタンスを単一推論エンジンから生成する。
    pub fn new(
        engine: &'a InferenceEngine,
        tokenizer: &'a JevTokenizer,
        mapping: &'a HierarchicalMapping,
        calib_config: &'a CalibrationConfig,
        config: HierarchicalRouterConfig,
    ) -> Result<Self> {
        Self::with_predictor((engine, tokenizer), mapping, calib_config, config)
    }
}

impl<'a>
    CoarseToFineRouter<
        'a,
        (
            &'a DualSessionEngine,
            &'a CachedStateFeatures,
            &'a JevTokenizer,
        ),
    >
{
    /// キャッシュ済み中間特徴量を保持するデュアルセッションエンジンから新しいルーターインスタンスを生成する。
    pub fn new_dual(
        engine: &'a DualSessionEngine,
        cached_state: &'a CachedStateFeatures,
        tokenizer: &'a JevTokenizer,
        mapping: &'a HierarchicalMapping,
        calib_config: &'a CalibrationConfig,
        config: HierarchicalRouterConfig,
    ) -> Result<Self> {
        Self::with_predictor(
            (engine, cached_state, tokenizer),
            mapping,
            calib_config,
            config,
        )
    }
}

impl<'a> CoarseToFineRouter<'a, (HierarchicalEngine<'a>, &'a JevTokenizer)> {
    /// 任意の推論バックエンド (`HierarchicalEngine`) からルーターインスタンスを生成する。
    pub fn with_backend(
        backend: HierarchicalEngine<'a>,
        tokenizer: &'a JevTokenizer,
        mapping: &'a HierarchicalMapping,
        calib_config: &'a CalibrationConfig,
        config: HierarchicalRouterConfig,
    ) -> Result<Self> {
        Self::with_predictor((backend, tokenizer), mapping, calib_config, config)
    }
}

impl<'a, P: HierarchicalPredictor> CoarseToFineRouter<'a, P> {
    /// 任意の [`HierarchicalPredictor`] 実装から新しいルーターインスタンスを生成する。
    pub fn with_predictor(
        predictor: P,
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
            predictor,
            mapping,
            calib_config,
            config,
        })
    }

    /// 階層ルーティング推論を実行する。
    ///
    /// オントロジー内の全細分類候補を対象として推論を行う。
    pub fn route(
        &self,
        state: &str,
        instructions: &str,
        gating_config: Option<&GatingConfig>,
    ) -> Result<HierarchicalResult> {
        self.route_with_allowed_candidates(state, instructions, None, gating_config)
    }

    /// 許可された細分類候補（オントロジーのサブセット）を指定して階層ルーティング推論を実行する。
    ///
    /// # 引数
    /// - `state`: 文脈テキスト。
    /// - `instructions`: 質問の指示文。
    /// - `allowed_criteria`: 評価対象として制約する細分類候補マップ (任意)。`None` の場合はオントロジーの全細分類を対象とする。
    /// - `gating_config`: ゲーティング閾値設定 (任意)。
    ///
    /// # パイプライン
    /// 1. **第 1 パス (大分類推論 & Dirichlet 較正)**: [`HierarchicalPredictor::predict_coarse_probabilities`] で大分類確率を算出し、設定に応じて ODIR Dirichlet 較正を適用。
    /// 2. **大分類クラスタ制約・再正規化**: `allowed_criteria` が指定されている場合、配下に候補が存在する大分類のみに限定して確率を再正規化。
    /// 3. **Early-Exit 判定**: 複合確信度 $S_{\text{coarse}} < \tau_{\text{escalate}}$ の場合、即時脱出。
    /// 4. **Soft-Beam 判定**: Top-Margin $M_{\text{coarse}} < \tau_{\text{beam}}$ の場合、Top-1 と Top-2 配下の候補をマージ。
    /// 5. **第 2 パス (細分類推論 & 生ロジット取得)**: [`HierarchicalPredictor::forward_fine_raw`] でマージされた許可候補群に対して生ロジットを直接抽出。
    /// 6. **温度調整 & 確率結合**: エントロピー連動温度・ベース温度を生ロジットへ直接適用 (1 回の Softmax) し、大分類確率と減衰周辺化結合。
    /// 7. **受け皿候補バイアス**: $P(\text{その他}) \ge \tau_{\text{catchall}}$ 判定。
    /// 8. **全候補空間 Gating 再計算**: 許可細分類候補空間基準で Gating メタデータを正規化。
    pub fn route_with_allowed_candidates(
        &self,
        state: &str,
        instructions: &str,
        allowed_criteria: Option<&IndexMap<String, String>>,
        gating_config: Option<&GatingConfig>,
    ) -> Result<HierarchicalResult> {
        let allowed_fine_keys: Vec<String> = if let Some(allowed) = allowed_criteria {
            if allowed.is_empty() {
                return Err(RuntimeError::InvalidQuestion(
                    "指定された criteria が空です。".to_string(),
                ));
            }
            allowed.keys().cloned().collect()
        } else {
            self.mapping.all_fine_keys().to_vec()
        };

        // --- Step 1: 第 1 パス (大分類推論 & Dirichlet 較正) ---
        let coarse_probabilities = self.predictor.predict_coarse_probabilities(
            state,
            instructions,
            &self.mapping.coarse_categories,
            self.calib_config,
        )?;

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

        // 配下に許可候補が存在する大分類クラスタを特定
        let valid_coarse_keys: Vec<String> = if let Some(allowed) = allowed_criteria {
            self.mapping
                .all_coarse_keys()
                .iter()
                .filter(|c_key| {
                    self.mapping
                        .fine_keys_for_coarse(c_key)
                        .map(|f_keys| f_keys.iter().any(|fk| allowed.contains_key(fk)))
                        .unwrap_or(false)
                })
                .cloned()
                .collect()
        } else {
            self.mapping.all_coarse_keys().to_vec()
        };

        if valid_coarse_keys.is_empty() {
            return Err(RuntimeError::InvalidQuestion(
                "指定された criteria に合致する大分類クラスタが存在しません。".to_string(),
            ));
        }

        // 許可大分類クラスタに限定した確率分布の再正規化
        let mut filtered_coarse_probs = IndexMap::with_capacity(valid_coarse_keys.len());
        let total_coarse_prob: f64 = valid_coarse_keys
            .iter()
            .map(|k| coarse_probabilities.get(k).copied().unwrap_or(0.0))
            .sum();

        if total_coarse_prob > f64::EPSILON && total_coarse_prob.is_finite() {
            for k in &valid_coarse_keys {
                let p = coarse_probabilities.get(k).copied().unwrap_or(0.0);
                filtered_coarse_probs.insert(k.clone(), p / total_coarse_prob);
            }
        } else {
            let uniform = 1.0 / (valid_coarse_keys.len() as f64);
            for k in &valid_coarse_keys {
                filtered_coarse_probs.insert(k.clone(), uniform);
            }
        }

        // 決定論的降順ソート (確率同値時は元の大分類インデックス昇順)
        let sorted_coarse =
            sort_probabilities_descending(&valid_coarse_keys, &filtered_coarse_probs);

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
                &allowed_fine_keys,
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
                    if let Some(allowed) = allowed_criteria
                        && !allowed.contains_key(fk)
                    {
                        continue;
                    }
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
        let raw_logits = self.predictor.forward_fine_raw(state, &fine_question)?;
        if raw_logits.len() != fine_candidates.len() {
            return Err(RuntimeError::InvalidQuestion(format!(
                "第 2 パスの出力ロジット数 ({}) が細分類候補数 ({}) と一致しません。",
                raw_logits.len(),
                fine_candidates.len()
            )));
        }

        // --- Step 5: 温度調整 & 確率結合 ---
        let fine_temperature =
            compute_fine_temperature(norm_entropy, self.config.entropy_temp_gamma);

        // ベース較正温度と動的温度を乗算した実効温度
        let base_temp = self
            .calib_config
            .get_temperature(QuestionType::Choice, fine_candidates.len());
        let effective_temp = (base_temp * fine_temperature).max(0.01);

        // 生ロジットに対して実効温度で 1 回だけ Softmax を適用
        let temp_scaled_probs = temperature_scaled_softmax(&raw_logits, effective_temp)?;

        let mut evaluated_fine_probs = IndexMap::with_capacity(fine_candidates.len());
        for (i, (k, _)) in fine_candidates.iter().enumerate() {
            evaluated_fine_probs.insert(k.clone(), temp_scaled_probs[i]);
        }

        let full_probabilities = combine_and_reconstruct_probabilities(
            &allowed_fine_keys,
            &self.mapping.fine_to_coarse,
            &filtered_coarse_probs,
            &evaluated_fine_probs,
            self.config.coarse_power_alpha,
        )?;

        // --- Step 6: 決定論的勝者選定および受け皿候補バイアス ---
        let sorted_full = sort_probabilities_descending(&allowed_fine_keys, &full_probabilities);

        let (final_winner, _winner_prob, final_margin, catchall_bias_triggered) =
            resolve_winner_with_catchall_bias(
                self.mapping,
                &sorted_full,
                self.config.enable_catchall_bias,
                self.config.catchall_threshold,
            )?;

        // --- Step 7: 全候補空間基準での複合確信度算出・Gating 判定 (OOD 安全弁含む) ---
        let fine_energy = calculate_choice_energy_with_config(
            &fine_question,
            &raw_logits,
            self.calib_config,
            gating_config,
        );
        let resolved_gating_config = resolve_gating_config(gating_config, self.calib_config);
        let final_confidence = compute_final_answer_confidence(
            &full_probabilities,
            catchall_bias_triggered,
            final_margin,
        )?;

        let mut answer = Answer::choice(final_winner, full_probabilities, final_confidence);
        let gating = build_final_gating_metadata(
            self.mapping,
            state,
            instructions,
            &answer,
            &resolved_gating_config,
            fine_energy,
        );
        answer = answer.with_gating(gating);

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

/// 候補キー順と確率マップから、確率降順 (同値時は元の定義順昇順) に並べた `(key, prob)` 列を返す。
///
/// 事前に出現インデックスをタプル `(usize, &str, f64)` として保持し、
/// `f64::total_cmp` で全順序比較を行うことで $O(K \log K)$ で決定論的にソートする。
fn sort_probabilities_descending<'a>(
    keys: &'a [String],
    probabilities: &IndexMap<String, f64>,
) -> Vec<(&'a str, f64)> {
    let mut items: Vec<(usize, &'a str, f64)> = keys
        .iter()
        .enumerate()
        .map(|(idx, k)| {
            (
                idx,
                k.as_str(),
                probabilities.get(k).copied().unwrap_or(0.0),
            )
        })
        .collect();

    items.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| a.0.cmp(&b.0)));

    items.into_iter().map(|(_, k, p)| (k, p)).collect()
}

/// 受け皿候補バイアス (`catchall_bias`) を適用して最終勝者・確信度・マージンを決定する。
///
/// 確率降順にソート済みの `sorted_full` を先頭から走査するため、閾値以上のネガティブ候補が
/// 複数存在する場合でも走査順に依存せず最大確率の候補 (同率時は定義順先頭) を採択する。
/// また、2 位以下のネガティブ候補を昇格させた場合も、`final_winner` 以外の最大確率候補
/// (次点候補) との差 `(final_confidence - runner_up_prob).clamp(0.0, 1.0)` でマージンを再計算する。
///
/// # 引数
/// - `mapping`: 階層マッピング定義。
/// - `sorted_full`: 確率降順 (同率時は定義順昇順) にソート済みの全細分類確率一覧。
/// - `enable_catchall_bias`: 受け皿候補バイアスを有効にするか。
/// - `catchall_threshold`: 受け皿候補優先採択の確率閾値。
///
/// # 戻り値
/// `(final_winner, final_confidence, final_margin, catchall_bias_triggered)` のタプル。
fn resolve_winner_with_catchall_bias(
    mapping: &HierarchicalMapping,
    sorted_full: &[(&str, f64)],
    enable_catchall_bias: bool,
    catchall_threshold: f64,
) -> Result<(String, f64, f64, bool)> {
    let (default_winner, default_winner_prob) = sorted_full
        .first()
        .copied()
        .ok_or_else(|| RuntimeError::InvalidQuestion("全細分類の確率分布が空です。".to_string()))?;

    let mut final_winner = default_winner.to_string();
    let mut final_confidence = default_winner_prob;
    let mut catchall_bias_triggered = false;

    if enable_catchall_bias {
        for &(key, prob) in sorted_full {
            let desc = mapping
                .fine_criteria
                .get(key)
                .map(|s| s.as_str())
                .unwrap_or("");
            if mapping.is_negative_candidate(key, desc) && prob >= catchall_threshold {
                final_winner = key.to_string();
                final_confidence = prob;
                catchall_bias_triggered = true;
                break;
            }
        }
    }

    let final_margin = if sorted_full.len() <= 1 {
        1.0
    } else if let Some((_, runner_up_prob)) = sorted_full
        .iter()
        .find(|(k, _)| *k != final_winner.as_str())
    {
        (final_confidence - *runner_up_prob).clamp(0.0, 1.0)
    } else {
        0.0
    };

    Ok((
        final_winner,
        final_confidence,
        final_margin,
        catchall_bias_triggered,
    ))
}

/// 全細分類候補空間の確率分布から、`Answer::choice` に設定する複合確信度スコアを算出する。
///
/// # 数理仕様
/// - 通常時は [`composite_confidence_iter`] により全候補空間の正規化エントロピーと Top-Margin を統合した
///   $S_{\text{confidence}} = (1.0 - H_{\text{norm}}(p)) \times M(p)$ を算出する。
/// - `catchall_bias` によって 2 位以下のネガティブ候補が勝者に昇格された場合 (`final_margin == 0.0`) は、
///   分布全体の最大確率候補に対する確信度で昇格先候補を過信自動実行することを防ぐため `0.0` を返す。
fn compute_final_answer_confidence(
    full_probabilities: &IndexMap<String, f64>,
    catchall_bias_triggered: bool,
    final_margin: f64,
) -> Result<f64> {
    if catchall_bias_triggered && final_margin <= 0.0 {
        return Ok(0.0);
    }
    Ok(composite_confidence_iter(full_probabilities.values())?)
}

/// 全細分類定義を含む `Question` を構築し、System 2 への CoT 委託プロンプト案を生成する。
fn build_hierarchical_escalation_prompt(
    mapping: &HierarchicalMapping,
    state: &str,
    instructions: &str,
    gating: &GatingMetadata,
    top_candidates: &[CandidateProbability],
) -> String {
    let escalation_question = Question {
        question_type: QuestionType::Choice,
        instructions: instructions.to_string(),
        criteria: Some(Criteria::Map(mapping.fine_criteria.clone())),
    };

    build_rich_escalation_prompt(
        Some(state),
        None,
        Some(&escalation_question),
        gating,
        top_candidates,
    )
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

    let prompt_template =
        build_hierarchical_escalation_prompt(mapping, state, instructions, gating, &top_candidates);

    gating.escalation = Some(EscalationContext {
        top_candidates,
        margin: gating.margin,
        uncertainty_reason: gating.reason.clone(),
        prompt_template: Some(prompt_template),
    });
}

/// 通常ルーティング終了時 (Step 7) の `GatingMetadata` を構築する。
///
/// コア層の [`evaluate_answer_gating_opts`] へ自由エネルギー `fine_energy` とともに評価を委譲し、
/// OOD 安全弁・マージンガード・3 系統ルーティング判定・`is_ood` フラグ・監査理由を一貫して算出する。
/// `route != DecisionRoute::AutoExecute` の場合は、全候補定義を含む `Question` を構築して
/// [`build_rich_escalation_prompt`] による System 2 委託プロンプト案を `EscalationContext` へ格納する。
fn build_final_gating_metadata(
    mapping: &HierarchicalMapping,
    state: &str,
    instructions: &str,
    answer: &Answer,
    gating_config: &GatingConfig,
    fine_energy: Option<f64>,
) -> GatingMetadata {
    let mut gating =
        evaluate_answer_gating_opts(answer, None, None, gating_config, fine_energy, false);

    if gating.route != DecisionRoute::AutoExecute {
        let rich_prompt = gating.escalation.as_ref().map(|esc| {
            build_hierarchical_escalation_prompt(
                mapping,
                state,
                instructions,
                &gating,
                &esc.top_candidates,
            )
        });
        if let (Some(esc), Some(prompt)) = (&mut gating.escalation, rich_prompt) {
            esc.prompt_template = Some(prompt);
        }
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
/// - `allowed_fine_keys`: 評価対象として許可された細分類候補キー配列。
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
    allowed_fine_keys: &[String],
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

    if allowed_fine_keys.is_empty() {
        return Err(RuntimeError::InvalidQuestion(
            "階層マッピングの細分類候補が空です。".to_string(),
        ));
    }

    let selected_choice = mapping
        .fine_keys_for_coarse(top1_coarse)
        .and_then(|keys| keys.iter().find(|fk| allowed_fine_keys.contains(fk)))
        .cloned()
        .unwrap_or_else(|| allowed_fine_keys[0].clone());

    let uniform_prob = 1.0 / (allowed_fine_keys.len() as f64);
    let mut full_probs = IndexMap::with_capacity(allowed_fine_keys.len());
    for k in allowed_fine_keys {
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
            .and_then(|keys| keys.iter().find(|fk| allowed_fine_keys.contains(fk)))
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
            if allowed_fine_keys.contains(fk) && seen.insert(fk.as_str()) {
                top_candidates.push(CandidateProbability {
                    candidate: fk.clone(),
                    probability: uniform_prob,
                });
            }
        }
    }

    if top_candidates.len() < 3 {
        for fk in allowed_fine_keys {
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
    let entropy_val = if allowed_fine_keys.len() > 1 {
        1.0
    } else {
        0.0
    };
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
            mapping.all_fine_keys(),
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
            mapping.all_fine_keys(),
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
        let gating_config = GatingConfig::default();

        // 1. AutoExecute 時はゼロアロケーション契約により escalation が None であること
        let mut auto_probs = IndexMap::new();
        auto_probs.insert("card_payment_not_recognised".to_string(), 0.85);
        auto_probs.insert("direct_debit_payment_not_recognised".to_string(), 0.10);
        auto_probs.insert("compromised_card".to_string(), 0.03);
        auto_probs.insert("lost_or_stolen_card".to_string(), 0.02);
        let auto_answer = Answer::choice("card_payment_not_recognised", auto_probs, 0.85);

        let auto_meta = build_final_gating_metadata(
            &mapping,
            state,
            instructions,
            &auto_answer,
            &gating_config,
            Some(-2.5),
        );
        assert_eq!(auto_meta.route, DecisionRoute::AutoExecute);
        assert!(!auto_meta.is_ood);
        assert!(auto_meta.escalation.is_none());

        // 2. ConfirmOrEscalate / Fallback 時は上位 3 候補と委託プロンプトが付与されること
        let mut mid_probs = IndexMap::new();
        mid_probs.insert("card_payment_not_recognised".to_string(), 0.42);
        mid_probs.insert("direct_debit_payment_not_recognised".to_string(), 0.31);
        mid_probs.insert("compromised_card".to_string(), 0.15);
        mid_probs.insert("lost_or_stolen_card".to_string(), 0.12);

        let confirm_answer = Answer::choice("card_payment_not_recognised", mid_probs.clone(), 0.42);
        let fallback_answer = Answer::choice("card_payment_not_recognised", mid_probs, 0.25);

        for (answer, expected_route) in [
            (confirm_answer, DecisionRoute::ConfirmOrEscalate),
            (fallback_answer, DecisionRoute::Fallback),
        ] {
            let meta = build_final_gating_metadata(
                &mapping,
                state,
                instructions,
                &answer,
                &gating_config,
                Some(-1.2),
            );
            assert_eq!(meta.route, expected_route);
            assert!(!meta.is_ood);
            assert!((meta.margin.unwrap() - 0.11).abs() < 1e-9);
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
            assert!((esc.margin.unwrap() - 0.11).abs() < 1e-9);
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

    #[test]
    fn test_final_gating_metadata_ood_safety_valve() {
        use sokuto_core::gating::GatingThresholds;

        let mapping = HierarchicalMapping::banking77();
        let state = "今日の東京の天気と明日の傘の必要性を教えてください。";
        let instructions = "お客様のお問い合わせ内容に最も適合するカテゴリを選択してください。";

        // 見かけの softmax 確信度とマージンが高くても、自由エネルギーが閾値超過なら OOD 発動
        let mut probs = IndexMap::new();
        probs.insert("card_payment_not_recognised".to_string(), 0.90);
        probs.insert("direct_debit_payment_not_recognised".to_string(), 0.05);
        probs.insert("compromised_card".to_string(), 0.05);
        let high_conf_answer = Answer::choice("card_payment_not_recognised", probs.clone(), 0.90);

        let default_gc = GatingConfig::default(); // ood_enabled=true, energy_threshold=-1.0
        let ood_meta = build_final_gating_metadata(
            &mapping,
            state,
            instructions,
            &high_conf_answer,
            &default_gc,
            Some(0.25),
        );
        assert_eq!(ood_meta.route, DecisionRoute::Fallback);
        assert!(ood_meta.is_ood);
        assert_eq!(ood_meta.energy, Some(0.25));
        assert!(ood_meta.reason.contains("OOD"));
        let esc = ood_meta
            .escalation
            .as_ref()
            .expect("OOD Fallback 時は EscalationContext が必須であること。");
        let prompt = esc
            .prompt_template
            .as_ref()
            .expect("OOD 警告付きプロンプトが生成されていること。");
        assert!(prompt.contains("Out-of-Distribution"));

        // ood_enabled: false の場合は同じエネルギーでも AutoExecute になること
        let disabled_ood_gc = GatingConfig {
            enabled: true,
            thresholds: GatingThresholds {
                ood_enabled: false,
                ..Default::default()
            },
        };
        let non_ood_meta = build_final_gating_metadata(
            &mapping,
            state,
            instructions,
            &high_conf_answer,
            &disabled_ood_gc,
            Some(0.25),
        );
        assert_eq!(non_ood_meta.route, DecisionRoute::AutoExecute);
        assert!(!non_ood_meta.is_ood);
        assert!(non_ood_meta.escalation.is_none());

        // ハイブリッド OOD (loose_energy_threshold + 低マージン) による Fallback 発動
        let mut close_probs = IndexMap::new();
        close_probs.insert("card_payment_not_recognised".to_string(), 0.52);
        close_probs.insert("direct_debit_payment_not_recognised".to_string(), 0.45);
        close_probs.insert("compromised_card".to_string(), 0.03);
        let close_answer = Answer::choice("card_payment_not_recognised", close_probs, 0.52);

        let hybrid_gc = GatingConfig {
            enabled: true,
            thresholds: GatingThresholds {
                ood_enabled: true,
                energy_threshold: 0.0,
                loose_energy_threshold: Some(-1.5),
                ood_min_confidence: Some(0.40),
                ood_min_margin: Some(0.15),
                ..Default::default()
            },
        };
        let hybrid_meta = build_final_gating_metadata(
            &mapping,
            state,
            instructions,
            &close_answer,
            &hybrid_gc,
            Some(-1.0),
        );
        assert_eq!(hybrid_meta.route, DecisionRoute::Fallback);
        assert!(hybrid_meta.is_ood);
    }

    #[test]
    fn test_final_gating_metadata_margin_guard_demotion() {
        let mapping = HierarchicalMapping::banking77();
        let state = "カードの引き落としについて確認したいです。";
        let instructions = "お客様のお問い合わせ内容に最も適合するカテゴリを選択してください。";
        let gating_config = GatingConfig::default();

        // 多クラス空間 (例: 10クラス中上位2候補に集中し残りが 0) でエントロピー確信度 >= 0.70 だが
        // 上位2候補が僅差 (margin = 0.04 < 0.15) のとき、マージンガードにより ConfirmOrEscalate へ降格する。
        let mut probs = IndexMap::new();
        probs.insert("card_payment_not_recognised".to_string(), 0.52);
        probs.insert("direct_debit_payment_not_recognised".to_string(), 0.48);
        for key in mapping.all_fine_keys().iter().skip(2).take(18) {
            probs.insert(key.clone(), 0.0);
        }
        // confidence を 0.30 (< low_threshold=0.35) にしても、entropy_conf >= 0.70 かつ margin < 0.15 なら
        // コア層のマージンガードが先に発火して ConfirmOrEscalate に降格される。
        let answer = Answer::choice("card_payment_not_recognised", probs, 0.30);

        let meta = build_final_gating_metadata(
            &mapping,
            state,
            instructions,
            &answer,
            &gating_config,
            Some(-3.0),
        );
        assert_eq!(meta.route, DecisionRoute::ConfirmOrEscalate);
        assert!(!meta.is_ood);
        assert!(meta.escalation.is_some());
    }

    #[test]
    fn test_catchall_bias_selects_highest_prob_negative_and_recomputes_margin() {
        let mut builder = HierarchicalMapping::builder("CatchallTest");
        // 定義順では other を先に、none を後に配置して走査順依存バグを再現できるようにする。
        builder.add_category(
            "ops",
            "カード操作",
            vec![
                ("card_lost", "カードを紛失した。"),
                ("other", "その他の問い合わせ。"),
                ("card_damaged", "カードが破損した。"),
                ("none", "該当なし。"),
            ],
        );
        let mapping = builder.build().expect("マッピング構築に成功すること。");

        // 元の Top-1 は card_lost (0.50)、Top-2 は none (0.28) なので元のマージンは 0.22。
        let sorted_full = vec![
            ("card_lost", 0.50),
            ("none", 0.28),
            ("other", 0.14),
            ("card_damaged", 0.08),
        ];

        // 1. enable_catchall_bias = true: 定義順の other (0.14) ではなく最大確率の none (0.28) を選ぶ。
        let (winner, conf, margin, triggered) =
            resolve_winner_with_catchall_bias(&mapping, &sorted_full, true, 0.10)
                .expect("勝者解決に成功すること。");
        assert!(triggered);
        assert_eq!(winner, "none");
        assert!((conf - 0.28).abs() < 1e-9);
        // 2 位以下からの昇格時は次点候補 (card_lost=0.50) との差によりマージンが 0.0 へ再計算される。
        assert!((margin - 0.0).abs() < 1e-9);

        // 2. 元々 1 位がネガティブ候補の場合は 1 位と 2 位の差がマージンとなる。
        let sorted_top1_neg = vec![
            ("none", 0.75),
            ("card_lost", 0.15),
            ("other", 0.10),
            ("card_damaged", 0.00),
        ];
        let (winner2, conf2, margin2, triggered2) =
            resolve_winner_with_catchall_bias(&mapping, &sorted_top1_neg, true, 0.10)
                .expect("勝者解決に成功すること。");
        assert!(triggered2);
        assert_eq!(winner2, "none");
        assert!((conf2 - 0.75).abs() < 1e-9);
        assert!((margin2 - 0.60).abs() < 1e-9);

        // 3. enable_catchall_bias = false: デフォルト勝者 card_lost と元のマージン 0.22 を維持する。
        let (winner3, conf3, margin3, triggered3) =
            resolve_winner_with_catchall_bias(&mapping, &sorted_full, false, 0.10)
                .expect("勝者解決に成功すること。");
        assert!(!triggered3);
        assert_eq!(winner3, "card_lost");
        assert!((conf3 - 0.50).abs() < 1e-9);
        assert!((margin3 - 0.22).abs() < 1e-9);
    }

    #[test]
    fn test_sort_probabilities_descending_orders_and_breaks_ties() {
        let keys = vec![
            "k0".to_string(),
            "k1".to_string(),
            "k2".to_string(),
            "k3".to_string(),
            "k4_missing".to_string(),
        ];
        let mut probs = IndexMap::new();
        probs.insert("k0".to_string(), 0.25);
        probs.insert("k1".to_string(), 0.50);
        probs.insert("k2".to_string(), 0.25);
        probs.insert("k3".to_string(), 0.00);

        let sorted = sort_probabilities_descending(&keys, &probs);
        let ordered_keys: Vec<&str> = sorted.iter().map(|(k, _)| *k).collect();

        // 確率降順 (k1=0.50 が先頭) かつ同値 (0.25 の k0, k2 / 0.00 の k3, k4_missing) は元の定義順を保つ。
        assert_eq!(ordered_keys, vec!["k1", "k0", "k2", "k3", "k4_missing"]);
        assert!((sorted[0].1 - 0.50).abs() < 1e-12);
        assert!((sorted[4].1 - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_final_answer_confidence_uses_composite_confidence() {
        let mapping = HierarchicalMapping::banking77();
        let all_keys = mapping.all_fine_keys();

        // 1. 最大確率は 0.72 (>= high_threshold 0.70) かつ 2 位が 0.27 (margin 0.45 >= 0.15) でも、
        //    全 77 候補空間での複合確信度 (1 - H_norm) * M は 0.70 未満となるため、
        //    生確率の直接代入による誤った AutoExecute を防ぎ ConfirmOrEscalate となること。
        let mut probs = IndexMap::new();
        probs.insert(all_keys[0].clone(), 0.72);
        probs.insert(all_keys[1].clone(), 0.27);
        let tail_prob = 0.01 / ((all_keys.len() - 2) as f64);
        for key in all_keys.iter().skip(2) {
            probs.insert(key.clone(), tail_prob);
        }

        let probs_vec: Vec<f64> = probs.values().copied().collect();
        let expected_composite = sokuto_core::math::composite_confidence(&probs_vec).unwrap();
        assert!(expected_composite < 0.70);

        let actual = compute_final_answer_confidence(&probs, false, 0.72 - 0.27)
            .expect("複合確信度の算出に成功すること。");
        assert!((actual - expected_composite).abs() < 1e-12);

        let answer = Answer::choice(all_keys[0].clone(), probs, actual);
        let gating = build_final_gating_metadata(
            &mapping,
            "カードの件について相談です。",
            "カテゴリを選択してください。",
            &answer,
            &GatingConfig::default(),
            Some(-2.5),
        );
        assert_eq!(gating.route, DecisionRoute::ConfirmOrEscalate);

        // 2. 単一候補に極端に集中した分布では複合確信度が high_threshold を超え AutoExecute となること
        let mut peaked_probs = IndexMap::new();
        peaked_probs.insert(all_keys[0].clone(), 0.96);
        peaked_probs.insert(all_keys[1].clone(), 0.04);
        for key in all_keys.iter().skip(2) {
            peaked_probs.insert(key.clone(), 0.0);
        }
        let peaked_vec: Vec<f64> = peaked_probs.values().copied().collect();
        let expected_peaked = sokuto_core::math::composite_confidence(&peaked_vec).unwrap();
        assert!(expected_peaked >= 0.70);

        let actual_peaked = compute_final_answer_confidence(&peaked_probs, false, 0.92)
            .expect("複合確信度の算出に成功すること。");
        assert!((actual_peaked - expected_peaked).abs() < 1e-12);

        let peaked_answer = Answer::choice(all_keys[0].clone(), peaked_probs, actual_peaked);
        let peaked_gating = build_final_gating_metadata(
            &mapping,
            "カードを紛失しました。",
            "カテゴリを選択してください。",
            &peaked_answer,
            &GatingConfig::default(),
            Some(-2.5),
        );
        assert_eq!(peaked_gating.route, DecisionRoute::AutoExecute);

        // 3. catchall_bias により 2 位以下のネガティブ候補へ切り替わった場合 (final_margin == 0.0) は 0.0 となること
        let mut override_probs = IndexMap::new();
        override_probs.insert("card_lost".to_string(), 0.85);
        override_probs.insert("none".to_string(), 0.15);
        let override_conf = compute_final_answer_confidence(&override_probs, true, 0.0)
            .expect("昇格時確信度の算出に成功すること。");
        assert!((override_conf - 0.0).abs() < 1e-12);
    }

    /// `forward_fine_raw` のみを実装し、第 1 パスのデフォルト実装も検証するモック。
    struct MockRawPredictor {
        coarse_logits: Vec<f64>,
        fine_logits: Vec<f64>,
        coarse_called: std::cell::Cell<bool>,
        fine_call_count: std::cell::Cell<usize>,
    }

    impl HierarchicalPredictor for MockRawPredictor {
        fn forward_fine_raw(&self, _state: &str, _question: &Question) -> Result<Vec<f64>> {
            if !self.coarse_called.get() {
                self.coarse_called.set(true);
                Ok(self.coarse_logits.clone())
            } else {
                self.fine_call_count.set(self.fine_call_count.get() + 1);
                Ok(self.fine_logits.clone())
            }
        }
    }

    /// 第 1 パス確率を直接返すモック。
    struct MockDirectCoarsePredictor {
        coarse_probs: IndexMap<String, f64>,
        fine_logits: Vec<f64>,
        fine_call_count: std::cell::Cell<usize>,
    }

    impl HierarchicalPredictor for MockDirectCoarsePredictor {
        fn predict_coarse_probabilities(
            &self,
            _state: &str,
            _instructions: &str,
            _categories: &IndexMap<String, String>,
            _calib: &CalibrationConfig,
        ) -> Result<IndexMap<String, f64>> {
            Ok(self.coarse_probs.clone())
        }

        fn forward_fine_raw(&self, _state: &str, _question: &Question) -> Result<Vec<f64>> {
            self.fine_call_count.set(self.fine_call_count.get() + 1);
            Ok(self.fine_logits.clone())
        }
    }

    fn build_test_mapping() -> HierarchicalMapping {
        let mut builder = HierarchicalMapping::builder("MockDomain");
        builder.add_category(
            "c1",
            "クラスタ1",
            vec![("c1_f1", "細分類1-1"), ("c1_f2", "細分類1-2")],
        );
        builder.add_category(
            "c2",
            "クラスタ2",
            vec![("c2_f1", "細分類2-1"), ("c2_f2", "細分類2-2")],
        );
        builder.add_category(
            "c3",
            "クラスタ3",
            vec![("c3_f1", "細分類3-1"), ("c3_f2", "細分類3-2")],
        );
        builder.add_category(
            "c4",
            "クラスタ4",
            vec![("c4_f1", "細分類4-1"), ("c4_f2", "細分類4-2")],
        );
        builder
            .build()
            .expect("テスト用マッピング構築に成功すること。")
    }

    #[test]
    fn test_route_with_mock_predictor_normal_and_soft_beam() {
        let mapping = build_test_mapping();
        let calib = CalibrationConfig::default();

        // 1. Top-1 が圧倒的 (c1=6.0) な通常パス: Soft-Beam は発動せず c1 配下の 2 候補のみ評価
        let normal_predictor = MockRawPredictor {
            coarse_logits: vec![6.0, 0.0, 0.0, 0.0],
            fine_logits: vec![0.5, 4.0],
            coarse_called: std::cell::Cell::new(false),
            fine_call_count: std::cell::Cell::new(0),
        };
        let router = CoarseToFineRouter::with_predictor(
            &normal_predictor,
            &mapping,
            &calib,
            HierarchicalRouterConfig::default(),
        )
        .expect("ルーター構築に成功すること。");

        let res = router
            .route("入力コンテキスト", "分類してください。", None)
            .expect("通常推論に成功すること。");
        assert!(!res.trace().early_exit_triggered);
        assert!(!res.trace().soft_beam_triggered);
        assert_eq!(res.trace().selected_coarse_clusters, vec!["c1"]);
        assert_eq!(res.trace().evaluated_fine_count, 2);
        assert_eq!(res.answer().as_choice(), Some("c1_f2"));
        assert_eq!(res.answer().probabilities().unwrap().len(), 8);
        assert_eq!(normal_predictor.fine_call_count.get(), 1);

        // 2. Top-1 と Top-2 が僅差 (c1=5.0, c2=4.3, c3=0.0, c4=0.0) のケース:
        //    P(c1)~=0.662, P(c2)~=0.329 で margin < 0.35 かつ composite_confidence >= 0.25 となり Soft-Beam 発動
        let beam_predictor = MockRawPredictor {
            coarse_logits: vec![5.0, 4.3, 0.0, 0.0],
            fine_logits: vec![0.0, 0.0, 5.0, 0.0],
            coarse_called: std::cell::Cell::new(false),
            fine_call_count: std::cell::Cell::new(0),
        };
        let beam_router = CoarseToFineRouter::with_predictor(
            &beam_predictor,
            &mapping,
            &calib,
            HierarchicalRouterConfig::default(),
        )
        .expect("Soft-Beam ルーター構築に成功すること。");

        let beam_res = beam_router
            .route("入力コンテキスト", "分類してください。", None)
            .expect("Soft-Beam 推論に成功すること。");
        assert!(!beam_res.trace().early_exit_triggered);
        assert!(beam_res.trace().soft_beam_triggered);
        assert_eq!(beam_res.trace().selected_coarse_clusters, vec!["c1", "c2"]);
        assert_eq!(beam_res.trace().evaluated_fine_count, 4);
        assert_eq!(beam_res.answer().as_choice(), Some("c2_f1"));
        assert_eq!(beam_predictor.fine_call_count.get(), 1);
    }

    #[test]
    fn test_route_with_mock_predictor_early_exit_skips_fine_pass() {
        let mapping = build_test_mapping();
        let calib = CalibrationConfig::default();

        let mut coarse_probs = IndexMap::new();
        coarse_probs.insert("c1".to_string(), 0.25);
        coarse_probs.insert("c2".to_string(), 0.25);
        coarse_probs.insert("c3".to_string(), 0.25);
        coarse_probs.insert("c4".to_string(), 0.25);

        let predictor = MockDirectCoarsePredictor {
            coarse_probs,
            fine_logits: vec![1.0, 0.0],
            fine_call_count: std::cell::Cell::new(0),
        };
        let router = CoarseToFineRouter::with_predictor(
            &predictor,
            &mapping,
            &calib,
            HierarchicalRouterConfig::default(),
        )
        .expect("ルーター構築に成功すること。");

        let res = router
            .route("曖昧なコンテキスト", "分類してください。", None)
            .expect("Early-Exit 推論に成功すること。");
        assert!(res.trace().early_exit_triggered);
        assert_eq!(res.trace().evaluated_fine_count, 0);
        assert_eq!(predictor.fine_call_count.get(), 0);
    }

    #[test]
    fn test_route_with_mock_predictor_logit_length_mismatch_returns_error() {
        let mapping = build_test_mapping();
        let calib = CalibrationConfig::default();

        let predictor = MockRawPredictor {
            coarse_logits: vec![6.0, 0.0, 0.0, 0.0],
            // c1 配下は 2 候補だが 1 要素しか返さない不正モック
            fine_logits: vec![1.0],
            coarse_called: std::cell::Cell::new(false),
            fine_call_count: std::cell::Cell::new(0),
        };
        let router = CoarseToFineRouter::with_predictor(
            predictor,
            &mapping,
            &calib,
            HierarchicalRouterConfig::default(),
        )
        .expect("ルーター構築に成功すること。");

        let err = router
            .route("コンテキスト", "分類してください。", None)
            .expect_err("ロジット数不一致時はエラーを返すこと。");
        assert!(matches!(err, RuntimeError::InvalidQuestion(_)));
    }

    #[test]
    fn test_route_with_allowed_candidates_restricts_candidates() {
        let mapping = build_test_mapping();
        let calib = CalibrationConfig::default();

        // c3 の確率が一番高いが、c3 の候補は allowed_criteria に含まれていない状況を設定
        let mut coarse_probs = IndexMap::new();
        coarse_probs.insert("c1".to_string(), 0.3);
        coarse_probs.insert("c2".to_string(), 0.2);
        coarse_probs.insert("c3".to_string(), 0.5);
        coarse_probs.insert("c4".to_string(), 0.0);

        // c1_f1 と c2_f1 のみを許可候補として指定
        let mut allowed_criteria = IndexMap::new();
        allowed_criteria.insert("c1_f1".to_string(), "細分類1-1".to_string());
        allowed_criteria.insert("c2_f1".to_string(), "細分類2-1".to_string());

        let predictor = MockDirectCoarsePredictor {
            coarse_probs,
            // 許可候補 c1_f1 の生ロジット
            fine_logits: vec![2.0],
            fine_call_count: std::cell::Cell::new(0),
        };

        let config = HierarchicalRouterConfig {
            enable_soft_beam: false,
            enable_early_exit: false,
            ..Default::default()
        };

        let router = CoarseToFineRouter::with_predictor(&predictor, &mapping, &calib, config)
            .expect("ルーター構築に成功すること。");

        let res = router
            .route_with_allowed_candidates(
                "コンテキスト",
                "分類してください。",
                Some(&allowed_criteria),
                None,
            )
            .expect("制約付き階層ルーティングに成功すること。");

        // 1. 勝者が許可された候補に含まれること
        let choice = res
            .answer()
            .as_choice()
            .expect("Choice 判定が存在すること。");
        assert_eq!(choice, "c1_f1");

        // 2. 確率分布のキーが許可された候補のみで構成されていること
        let probs = res
            .answer()
            .probabilities()
            .expect("確率マップが存在すること。");
        assert_eq!(probs.len(), 2);
        assert!(probs.contains_key("c1_f1"));
        assert!(probs.contains_key("c2_f1"));
        assert!(!probs.contains_key("c1_f2"));
        assert!(!probs.contains_key("c3_f1"));

        // 3. 確率の総和が 1.0 であること
        let sum: f64 = probs.values().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // 4. 許可候補が存在しない大分類 (c3) ではなく、許可候補を持つ大分類 (c1) が Top-1 に選ばれていること
        assert_eq!(res.trace().top1_coarse, "c1");
    }
}
