//! # 粗密2段階探索 (Coarse-to-Fine) モジュール
//!
//! 候補数が膨大な Choice 型質問 ($K > 30$) に対し、推論遅延の悪化、
//! State 優先トランケーションによる文脈欠落、およびアテンション分散を抑制するため、
//! 1 次粗スクリーニング (Coarse) と 2 次精密決定 (Fine) を連携させる機構を提供する。

use indexmap::IndexMap;
use sokuto_core::contract::model_spec::MAX_NUM_OPTIONS;
use sokuto_core::schema::{AnswerPayload, Question, QuestionType};

use crate::engine::InferenceEngine;
use crate::error::{Result, RuntimeError};
use crate::tokenizer::JevTokenizer;

/// デフォルトの自動発動閾値。候補数がこれを超える場合に 2 段階探索を発動する。
pub const DEFAULT_COARSE_THRESHOLD: usize = 30;

/// デフォルトの Fine 段階抽出候補数 ($M$)。
pub const DEFAULT_TOP_M_CANDIDATES: usize = 20;

/// 「該当なし」とみなすデフォルトのネガティブ候補識別子セット。
pub const DEFAULT_NEGATIVE_KEYS: &[&str] = &[
    "none",
    "other",
    "others",
    "none_of_the_above",
    "out_of_scope",
    "unknown",
    "not_applicable",
    "na",
    "n/a",
    "該当なし",
    "その他",
    "不明",
    "対象外",
];

/// 粗密 2 段階探索の設定構造体。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoarseToFineConfig {
    /// 自動トリガー候補数閾値。候補数がこの値を超える場合に 2 段階探索を行う。
    pub threshold: usize,
    /// Fine 段階へ引き渡す最大候補数 ($M$)。通常 16〜24 の範囲を推奨する。
    pub top_m: usize,
    /// 「該当なし (None / Other)」候補の自動保護 (Pinning) を有効にするか。
    pub preserve_negative: bool,
    /// ネガティブ候補として判定するキー名のリスト。
    pub negative_keys: Vec<String>,
}

impl Default for CoarseToFineConfig {
    fn default() -> Self {
        Self {
            threshold: DEFAULT_COARSE_THRESHOLD,
            top_m: DEFAULT_TOP_M_CANDIDATES,
            preserve_negative: true,
            negative_keys: DEFAULT_NEGATIVE_KEYS
                .iter()
                .map(|&s| s.to_string())
                .collect(),
        }
    }
}

impl CoarseToFineConfig {
    /// 指定された候補数 $K$ において 2 段階探索を発動すべきかを判定する。
    ///
    /// # 引数
    /// - `candidate_count`: 質問の候補数。
    pub fn should_trigger(&self, candidate_count: usize) -> bool {
        candidate_count > self.threshold
    }

    /// 設定値の整合性を検証する。
    pub fn validate(&self) -> Result<()> {
        if self.top_m == 0 {
            return Err(RuntimeError::InvalidQuestion(
                "top_m は 1 以上である必要があります。".to_string(),
            ));
        }
        if self.top_m > MAX_NUM_OPTIONS {
            return Err(RuntimeError::InvalidQuestion(format!(
                "top_m ({}) はモデル上限 ({}) を超えることはできません。",
                self.top_m, MAX_NUM_OPTIONS
            )));
        }
        Ok(())
    }
}

/// 候補キーおよび説明文が「該当なし / その他」などのネガティブ候補であるかを判定する。
///
/// 閉域分類バイアスによる誤分類を防止するため、類似度に関わらず保護対象とする。
///
/// # 引数
/// - `key`: 候補識別子。
/// - `description`: 候補の説明文。
/// - `negative_keys`: 登録済みネガティブキー一覧。
pub fn is_negative_candidate(key: &str, description: &str, negative_keys: &[String]) -> bool {
    let lower_key = key.trim().to_lowercase();
    if negative_keys
        .iter()
        .any(|nk| nk.to_lowercase() == lower_key)
    {
        return true;
    }

    let lower_desc = description.trim().to_lowercase();
    let negative_desc_phrases = [
        "none of the above",
        "out of scope",
        "not applicable",
        "other inquiry",
        "general inquiry",
        "上記のいずれにも該当しない",
        "該当なし",
        "その他のお問い合わせ",
        "サポート外",
    ];

    negative_desc_phrases
        .iter()
        .any(|phrase| lower_desc.contains(phrase))
}

/// 事前エンコードされた候補埋め込みベクトルを保持・照会するキャッシュ。
///
/// 各候補の静的なテキスト説明文に対する正規化済みベクトルをキャッシュし、
/// 推論時の内積計算をミリ秒未満で完了させる。
#[derive(Debug, Clone, Default)]
pub struct CandidateEmbeddingCache {
    /// 候補識別子と L2 正規化済み埋め込みベクトルのマップ。
    embeddings: IndexMap<String, Vec<f32>>,
    /// ベクトルの次元数。未登録時は 0。
    dimension: usize,
}

impl CandidateEmbeddingCache {
    /// 空のキャッシュを作成する。
    pub fn new() -> Self {
        Self::default()
    }

    /// ベクトルを登録する。自動的に L2 正規化が行われる。
    ///
    /// # 引数
    /// - `key`: 候補識別子。
    /// - `mut vector`: 埋め込みベクトル。
    pub fn insert(&mut self, key: impl Into<String>, mut vector: Vec<f32>) -> Result<()> {
        let key_str = key.into();
        if vector.is_empty() {
            return Err(RuntimeError::InvalidTensorData(format!(
                "キー '{key_str}' の埋め込みベクトルが空です。"
            )));
        }

        if self.dimension == 0 {
            self.dimension = vector.len();
        } else if vector.len() != self.dimension {
            return Err(RuntimeError::InvalidTensorData(format!(
                "埋め込み次元数不一致: 期待値={}, 実際={} (キー='{key_str}')。",
                self.dimension,
                vector.len()
            )));
        }

        // L2 正規化
        let norm_sq: f32 = vector.iter().map(|v| v * v).sum();
        let norm = norm_sq.sqrt();
        if norm > 1e-12 {
            let inv_norm = 1.0 / norm;
            for v in &mut vector {
                *v *= inv_norm;
            }
        }

        self.embeddings.insert(key_str, vector);
        Ok(())
    }

    /// 登録済みベクトルを取得する。
    pub fn get(&self, key: &str) -> Option<&[f32]> {
        self.embeddings.get(key).map(|v| v.as_slice())
    }

    /// ベクトルの次元数を取得する。
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// 登録されている候補数を取得する。
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// キャッシュが空であるかを判定する。
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// 2 つの L2 正規化済みベクトル間のコサイン類似度 (内積) を計算する。
    pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
        if v1.len() != v2.len() || v1.is_empty() {
            return 0.0;
        }
        v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
    }
}

/// 粗選考段階におけるスコアリングインターフェース。
pub trait CoarseScorer {
    /// クエリ文字列 (State + Instructions) と候補群を受け取り、各候補の類似度スコア列を算出する。
    ///
    /// # 引数
    /// - `query`: 照会クエリテキスト。
    /// - `candidates`: 評価対象の候補識別子と説明文の順序付きマップ。
    ///
    /// # 戻り値
    /// `candidates` の出現順に対応する類似度スコア列 (`Vec<f32>`)。
    fn score_candidates(
        &self,
        query: &str,
        candidates: &IndexMap<String, String>,
    ) -> Result<Vec<f32>>;
}

/// 文脈対応型・階層大分類スコアリングインターフェース。
///
/// 単純な静的類似度ではなく、入力 State と文脈全体を反映して
/// 大分類クラスタごとの確率・確信度スコアを算出する。
pub trait HierarchicalCoarseScorer {
    /// State、指示文、および大分類候補群を受け取り、各大分類の確率・確信度マップを算出する。
    ///
    /// # 引数
    /// - `state`: 入力文脈テキスト。
    /// - `instructions`: 質問指示文。
    /// - `coarse_categories`: 大分類のキーおよび説明文マップ。
    ///
    /// # 戻り値
    /// 各大分類キーと算出された確率値のマッピング。
    fn score_coarse(
        &self,
        state: &str,
        instructions: &str,
        coarse_categories: &IndexMap<String, String>,
    ) -> Result<IndexMap<String, f64>>;
}

/// ModernBERT モデルの第 1 パスフォワード推論を直接駆動するモデル駆動型大分類スコアラー。
///
/// 外部の検索モデルや静的埋め込みインデックスを必要とせず、
/// 同一の言語モデル重みを活用して高精度な大分類確信度分布を出力する。
pub struct ModelDrivenCoarseScorer<'a> {
    engine: &'a InferenceEngine,
    tokenizer: &'a JevTokenizer,
    calib_config: Option<&'a sokuto_core::contract::calibration::CalibrationConfig>,
}

impl<'a> ModelDrivenCoarseScorer<'a> {
    /// 推論エンジンおよびトークナイザー参照からモデル駆動型スコアラーを生成する。
    pub fn new(engine: &'a InferenceEngine, tokenizer: &'a JevTokenizer) -> Self {
        Self {
            engine,
            tokenizer,
            calib_config: None,
        }
    }

    /// 較正温度設定を適用する。
    pub fn with_calibration(
        mut self,
        calib_config: &'a sokuto_core::contract::calibration::CalibrationConfig,
    ) -> Self {
        self.calib_config = Some(calib_config);
        self
    }
}

impl<'a> HierarchicalCoarseScorer for ModelDrivenCoarseScorer<'a> {
    fn score_coarse(
        &self,
        state: &str,
        instructions: &str,
        coarse_categories: &IndexMap<String, String>,
    ) -> Result<IndexMap<String, f64>> {
        let default_calib = sokuto_core::contract::calibration::CalibrationConfig::default();
        let calib = self.calib_config.unwrap_or(&default_calib);

        let question = Question {
            question_type: QuestionType::Choice,
            instructions: instructions.to_string(),
            criteria: Some(sokuto_core::schema::Criteria::Map(
                coarse_categories.clone(),
            )),
        };

        let answer =
            self.engine
                .predict_single_question(self.tokenizer, state, &question, calib)?;

        match answer.payload {
            AnswerPayload::Choice { probabilities, .. }
            | AnswerPayload::Score { probabilities, .. } => Ok(probabilities),
            _ => Err(RuntimeError::InvalidQuestion(
                "大分類推論結果に確率分布が含まれていません。".to_string(),
            )),
        }
    }
}

/// トークナイザーのサブワード語彙を活用した外部モデル不要の語彙スコアラー。
///
/// State + Instructions と各候補説明文の語彙重複度・頻度重み付けにより
/// 高速かつアロケーションフリーにスクリーニングを行う。
#[derive(Debug, Clone)]
pub struct LexicalCoarseScorer<'a> {
    /// Jev トークナイザーへの参照。
    tokenizer: &'a JevTokenizer,
}

impl<'a> LexicalCoarseScorer<'a> {
    /// トークナイザー参照から語彙スコアラーを生成する。
    pub fn new(tokenizer: &'a JevTokenizer) -> Self {
        Self { tokenizer }
    }
}

impl CoarseScorer for LexicalCoarseScorer<'_> {
    fn score_candidates(
        &self,
        query: &str,
        candidates: &IndexMap<String, String>,
    ) -> Result<Vec<f32>> {
        // クエリトークンの頻度マップを構築
        let query_encoding = self
            .tokenizer
            .inner()
            .encode(query, false)
            .map_err(|e| RuntimeError::TokenizerEncodeError(e.to_string()))?;
        let query_ids = query_encoding.get_ids();

        let mut query_counts = std::collections::HashMap::with_capacity(query_ids.len());
        for &id in query_ids {
            *query_counts.entry(id).or_insert(0usize) += 1;
        }

        let mut scores = Vec::with_capacity(candidates.len());

        for (key, desc) in candidates {
            // キー名と説明文を合わせてテキスト化
            let candidate_text = format!("{key} {desc}");
            let cand_encoding = self
                .tokenizer
                .inner()
                .encode(candidate_text, false)
                .map_err(|e| RuntimeError::TokenizerEncodeError(e.to_string()))?;
            let cand_ids = cand_encoding.get_ids();

            if cand_ids.is_empty() {
                scores.push(0.0);
                continue;
            }

            let mut cand_counts = std::collections::HashMap::with_capacity(cand_ids.len());
            for &id in cand_ids {
                *cand_counts.entry(id).or_insert(0usize) += 1;
            }

            // 単語重複度 (内積 / ノルム積 = コサイン類似度類似の重み付き重複度)
            let mut dot_product = 0.0f32;
            for (id, count) in &cand_counts {
                if let Some(q_count) = query_counts.get(id) {
                    dot_product += (*count as f32) * (*q_count as f32);
                }
            }

            let cand_len_sq: f32 = cand_counts.values().map(|c| (c * c) as f32).sum();
            let query_len_sq: f32 = query_counts.values().map(|c| (c * c) as f32).sum();

            let norm = (cand_len_sq * query_len_sq).sqrt();
            let score = if norm > 1e-12 {
                dot_product / norm
            } else {
                0.0
            };

            scores.push(score);
        }

        Ok(scores)
    }
}

/// 事前埋め込みキャッシュとクエリベクトルを用いたコサイン類似度スコアラー。
#[derive(Debug, Clone)]
pub struct EmbeddingCoarseScorer<'a> {
    /// 候補埋め込みキャッシュへの参照。
    cache: &'a CandidateEmbeddingCache,
    /// 事前エンコードされたクエリの正規化済みベクトル。
    query_vector: &'a [f32],
}

impl<'a> EmbeddingCoarseScorer<'a> {
    /// キャッシュとクエリベクトルからスコアラーを生成する。
    pub fn new(cache: &'a CandidateEmbeddingCache, query_vector: &'a [f32]) -> Self {
        Self {
            cache,
            query_vector,
        }
    }
}

impl CoarseScorer for EmbeddingCoarseScorer<'_> {
    fn score_candidates(
        &self,
        _query: &str,
        candidates: &IndexMap<String, String>,
    ) -> Result<Vec<f32>> {
        let mut scores = Vec::with_capacity(candidates.len());

        for key in candidates.keys() {
            if let Some(cand_vec) = self.cache.get(key) {
                let sim = CandidateEmbeddingCache::cosine_similarity(self.query_vector, cand_vec);
                scores.push(sim);
            } else {
                // キャッシュに存在しない候補はスコア 0.0
                scores.push(0.0);
            }
        }

        Ok(scores)
    }
}

/// スクリーニング結果を保持する構造体。
#[derive(Debug, Clone, PartialEq)]
pub struct FilteredCandidates {
    /// Fine 段階へ引き渡す縮小版候補マップ (元の出現順を維持)。
    pub selected: IndexMap<String, String>,
    /// Coarse 段階で除外された候補キーのリスト。
    pub excluded_keys: Vec<String>,
    /// 元の総候補数。
    pub total_candidates: usize,
}

/// 大規模候補群から Top-M 件を抽出し、ネガティブ候補を保護した部分マップを生成する。
///
/// # 計算量とタイブレーク
/// - `select_nth_unstable_by` による $O(K)$ クイックセレクトを採用。
/// - 類似度スコアが同等の場合は、元の Criteria 内の出現順序 (インデックス昇順) でタイブレークを行い、決定論的再現性を担保する。
/// - 最終的な候補マップは元の出現順序を維持して再構成される。
///
/// # 引数
/// - `query`: 照会クエリテキスト (State + Instructions)。
/// - `candidates`: 全候補の順序付きマップ。
/// - `config`: 粗密探索設定。
/// - `scorer`: 類似度スコアラー。
pub fn filter_top_candidates(
    query: &str,
    candidates: &IndexMap<String, String>,
    config: &CoarseToFineConfig,
    scorer: &impl CoarseScorer,
) -> Result<FilteredCandidates> {
    let total_candidates = candidates.len();

    // 候補数が閾値以下の場合はフィルタリングを行わずそのまま返却
    if total_candidates <= config.top_m {
        return Ok(FilteredCandidates {
            selected: candidates.clone(),
            excluded_keys: Vec::new(),
            total_candidates,
        });
    }

    // 1. ネガティブ候補の検出と保護枠の確保
    let mut pinned_indices = Vec::new();
    let mut candidate_entries: Vec<(usize, &String, &String)> =
        Vec::with_capacity(total_candidates);

    for (orig_idx, (key, desc)) in candidates.iter().enumerate() {
        if config.preserve_negative && is_negative_candidate(key, desc, &config.negative_keys) {
            pinned_indices.push(orig_idx);
        } else {
            candidate_entries.push((orig_idx, key, desc));
        }
    }

    // ネガティブ保護候補が既に top_m 以上存在する場合のガード
    let target_top_m = config.top_m.max(1);
    if pinned_indices.len() > target_top_m {
        pinned_indices.truncate(target_top_m);
    }
    let remaining_quota = target_top_m.saturating_sub(pinned_indices.len());

    // 2. 保護対象外の候補について類似度スコアを算出
    let mut selected_indices = pinned_indices;

    if remaining_quota > 0 && !candidate_entries.is_empty() {
        let mut unpinned_map = IndexMap::with_capacity(candidate_entries.len());

        for &(_, key, desc) in &candidate_entries {
            unpinned_map.insert(key.clone(), desc.clone());
        }

        let scores = scorer.score_candidates(query, &unpinned_map)?;

        // (score, orig_idx) のリストを作成
        let mut scored_items: Vec<(f32, usize)> = candidate_entries
            .iter()
            .zip(scores.iter())
            .map(|(&(orig_idx, _, _), &score)| {
                // NaN ガード: スコアが非有限値の場合は極小値を設定
                let s = if score.is_finite() {
                    score
                } else {
                    f32::NEG_INFINITY
                };
                (s, orig_idx)
            })
            .collect();

        // 3. select_nth_unstable_by による O(K) Top-M 抽出
        let select_k = remaining_quota.min(scored_items.len());
        if select_k < scored_items.len() {
            // スコア降順 (高い順)。同値ならインデックス昇順 (先勝ち)
            scored_items.select_nth_unstable_by(select_k - 1, |a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });
            scored_items.truncate(select_k);
        }

        for (_, orig_idx) in scored_items {
            selected_indices.push(orig_idx);
        }
    }

    // 4. 元の出現順序 (orig_idx 昇順) にソートしてマップを再構築
    selected_indices.sort_unstable();
    selected_indices.dedup();

    let mut selected_map = IndexMap::with_capacity(selected_indices.len());
    let mut excluded_keys = Vec::with_capacity(total_candidates - selected_indices.len());

    let mut selected_idx_set = std::collections::HashSet::with_capacity(selected_indices.len());
    for &idx in &selected_indices {
        selected_idx_set.insert(idx);
    }

    for (orig_idx, (key, desc)) in candidates.iter().enumerate() {
        if selected_idx_set.contains(&orig_idx) {
            selected_map.insert(key.clone(), desc.clone());
        } else {
            excluded_keys.push(key.clone());
        }
    }

    Ok(FilteredCandidates {
        selected: selected_map,
        excluded_keys,
        total_candidates,
    })
}

/// 縮小候補空間で計算された確率分布から、全候補空間の確率マップを再構成する。
///
/// Fine 段階で評価された候補には算出された確率を割り当て、
/// Coarse 段階で除外された候補には `0.0` を明示的に設定する。
///
/// # 引数
/// - `original_keys`: リクエストされた全候補キーの順序付きスライス。
/// - `sub_probabilities`: Fine 段階で算出された確率マップ。
pub fn reconstruct_probabilities(
    original_keys: &[String],
    sub_probabilities: &IndexMap<String, f64>,
) -> IndexMap<String, f64> {
    let mut full_probabilities = IndexMap::with_capacity(original_keys.len());

    for key in original_keys {
        let prob = sub_probabilities.get(key).copied().unwrap_or(0.0);
        full_probabilities.insert(key.clone(), prob);
    }

    full_probabilities
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 簡易モックスコアラー
    struct MockScorer {
        scores: Vec<f32>,
    }

    impl CoarseScorer for MockScorer {
        fn score_candidates(
            &self,
            _query: &str,
            candidates: &IndexMap<String, String>,
        ) -> Result<Vec<f32>> {
            assert_eq!(self.scores.len(), candidates.len());
            Ok(self.scores.clone())
        }
    }

    #[test]
    fn test_negative_candidate_detection() {
        let neg_keys = vec!["none".to_string(), "other".to_string()];
        assert!(is_negative_candidate("None", "該当なし", &neg_keys));
        assert!(is_negative_candidate("other", "その他", &neg_keys));
        assert!(is_negative_candidate(
            "custom",
            "上記のいずれにも該当しない",
            &neg_keys
        ));
        assert!(!is_negative_candidate("card_lost", "カード紛失", &neg_keys));
    }

    #[test]
    fn test_candidate_embedding_cache() {
        let mut cache = CandidateEmbeddingCache::new();
        assert!(cache.is_empty());

        cache.insert("opt1", vec![3.0, 4.0]).expect("挿入失敗。");
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.dimension(), 2);

        let vec = cache.get("opt1").unwrap();
        // L2 正規化確認: (3/5, 4/5) = (0.6, 0.8)
        assert!((vec[0] - 0.6).abs() < 1e-6);
        assert!((vec[1] - 0.8).abs() < 1e-6);

        // コサイン類似度
        let sim = CandidateEmbeddingCache::cosine_similarity(vec, &[0.6, 0.8]);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_filter_top_candidates_pinning_and_determinism() {
        let mut candidates = IndexMap::new();
        // 5つの候補を定義
        candidates.insert("card_issue".to_string(), "カード発行".to_string()); // idx 0
        candidates.insert("card_lost".to_string(), "カード紛失".to_string()); // idx 1
        candidates.insert("loan_apply".to_string(), "ローン申込".to_string()); // idx 2
        candidates.insert("loan_repay".to_string(), "ローン返済".to_string()); // idx 3
        candidates.insert(
            "other".to_string(),
            "上記のいずれにも該当しない".to_string(),
        ); // idx 4 (ネガティブ)

        let config = CoarseToFineConfig {
            threshold: 3,
            top_m: 3, // 3 件のみ選抜
            preserve_negative: true,
            negative_keys: vec!["other".to_string()],
        };

        // 非ネガティブ候補 (0, 1, 2, 3) に対するモックスコア
        // idx 0 -> 0.1, idx 1 -> 0.9, idx 2 -> 0.5, idx 3 -> 0.5 (タイ)
        let scorer = MockScorer {
            scores: vec![0.1, 0.9, 0.5, 0.5],
        };

        let result = filter_top_candidates("カード", &candidates, &config, &scorer).unwrap();

        assert_eq!(result.selected.len(), 3);
        assert_eq!(result.total_candidates, 5);

        // "other" はスコアに関わらず保護されていること
        assert!(result.selected.contains_key("other"));
        // 最高スコアの "card_lost" は選ばれていること
        assert!(result.selected.contains_key("card_lost"));
        // タイブレーク (idx 2 vs idx 3: 共に 0.5) で先頭優先の "loan_apply" が選ばれていること
        assert!(result.selected.contains_key("loan_apply"));
        assert!(!result.selected.contains_key("loan_repay"));
        assert!(!result.selected.contains_key("card_issue"));

        // 出現順が維持されていること (card_lost, loan_apply, other)
        let keys: Vec<&String> = result.selected.keys().collect();
        assert_eq!(keys, vec!["card_lost", "loan_apply", "other"]);
    }

    #[test]
    fn test_reconstruct_probabilities() {
        let original_keys = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let mut sub_probs = IndexMap::new();
        sub_probs.insert("b".to_string(), 0.7);
        sub_probs.insert("d".to_string(), 0.3);

        let full = reconstruct_probabilities(&original_keys, &sub_probs);
        assert_eq!(full.len(), 4);
        assert_eq!(full.get("a"), Some(&0.0));
        assert_eq!(full.get("b"), Some(&0.7));
        assert_eq!(full.get("c"), Some(&0.0));
        assert_eq!(full.get("d"), Some(&0.3));
    }

    #[test]
    fn test_filter_top_candidates_excessive_negative_keys_guard() {
        let mut candidates = IndexMap::new();
        // 10 件中 5 件がネガティブ候補
        for i in 0..5 {
            candidates.insert(format!("none_{i}"), format!("該当なし {i}"));
        }
        for i in 0..5 {
            candidates.insert(format!("regular_{i}"), format!("通常候補 {i}"));
        }

        // top_m = 3 の設定 (ネガティブ候補 5 件よりも小さい)
        let config = CoarseToFineConfig {
            threshold: 5,
            top_m: 3,
            preserve_negative: true,
            negative_keys: (0..5).map(|i| format!("none_{i}")).collect(),
        };

        let scorer = MockScorer {
            scores: vec![1.0; 5],
        };

        let result = filter_top_candidates("クエリ", &candidates, &config, &scorer).unwrap();

        // 厳密に top_m (3) 件以下に制限されていること
        assert_eq!(result.selected.len(), 3);
        assert_eq!(result.total_candidates, 10);
        // 先頭の 3 件のネガティブ候補のみが選ばれていること
        assert!(result.selected.contains_key("none_0"));
        assert!(result.selected.contains_key("none_1"));
        assert!(result.selected.contains_key("none_2"));
        assert!(!result.selected.contains_key("none_3"));
        assert!(!result.selected.contains_key("regular_0"));
    }
}
