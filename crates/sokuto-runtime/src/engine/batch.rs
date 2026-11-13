//! # 複数質問一括バッチ推論モジュール
//!
//! トークンレベルのプレフィックス共有と動的バッチ Collate により、
//! 単一フォワードパスでの並列推論および決定プリミティブ解決を実現する。

use indexmap::IndexMap;
use sokuto_core::contract::CalibrationConfig;
use sokuto_core::decision::evaluate_question_with_buf;
use sokuto_core::gating::GatingConfig;
use sokuto_core::schema::{Answer, Question, QuestionType};

use crate::engine::coarse::{
    CoarseScorer, CoarseToFineConfig, LexicalCoarseScorer, filter_top_candidates,
    reconstruct_and_gate_answer,
};
use crate::engine::config::effective_single_model_chunk_size;
use crate::engine::gating::{
    calculate_choice_energy_with_config, evaluate_and_gate_answer_with_buf,
};
use crate::engine::session::InferenceEngine;
use crate::error::{Result, RuntimeError};
use crate::hierarchical::{
    CoarseToFineRouter, HierarchicalMapping, HierarchicalResult, HierarchicalRouterConfig,
};
use crate::tokenizer::{BatchTokenizedQuestions, JevTokenizer};

/// サーバー層との連携において推奨されるデフォルトの最大バッチチャンクサイズ。
///
/// 質問数がこの値を超える場合、OOM (Out Of Memory) を防止するため
/// マイクロバッチに分割して実行することが推奨される。
pub const DEFAULT_MAX_BATCH_CHUNK_SIZE: usize = 16;

/// バッチ推論および決定数理解決用のスクラッチパッドバッファ。
///
/// スレッドローカルまたはリクエストコンテキスト内で再利用することで、
/// 推論ごとのメモリアロケーションを根絶する。
#[derive(Debug, Default, Clone)]
pub struct BatchScratchpad {
    /// 平坦化生ロジット書き込み用バッファ (`[batch_size * num_options]`)。
    pub flat_logits: Vec<f64>,
    /// ソフトマックス確率分布算出用作業バッファ (`[max_options_per_q]`)。
    pub probs_buf: Vec<f64>,
}

impl BatchScratchpad {
    /// 指定されたキャパシティでスクラッチパッドを初期化する。
    pub fn with_capacity(total_options: usize, max_options_per_q: usize) -> Self {
        Self {
            flat_logits: Vec::with_capacity(total_options),
            probs_buf: Vec::with_capacity(max_options_per_q),
        }
    }

    /// 要求されるサイズに合わせてバッファ領域を確保・リサイズする。
    pub fn prepare(&mut self, total_options: usize, max_options_per_q: usize) {
        if self.flat_logits.len() < total_options {
            self.flat_logits.resize(total_options, 0.0);
        }
        if self.probs_buf.len() < max_options_per_q {
            self.probs_buf.resize(max_options_per_q, 0.0);
        }
    }
}

/// バッチ推論パイプラインの動作を制御するオプション構造体 (Parameter Object)。
///
/// ゲーティング設定、スクラッチパッドバッファ、マイクロバッチチャンクサイズ、
/// および粗密 2 段階探索の設定を一元的に指定し、単一の一貫した推論エントリーポイントへ引き渡す。
#[derive(Default)]
pub struct BatchInferenceOptions<'a> {
    /// 明示的なゲーティング設定 (任意、未指定時は `CalibrationConfig` から導出)。
    pub gating_config: Option<&'a GatingConfig>,
    /// 再利用可能な作業用スクラッチパッドバッファ (任意、未指定時は内部で一時確保)。
    pub scratchpad: Option<&'a mut BatchScratchpad>,
    /// 1 回のフォワードパスで処理する最大質問数 (任意、未指定時は一括実行)。
    pub chunk_size: Option<usize>,
    /// 粗密 2 段階探索の設定 (任意、指定時は候補数閾値を超える `Choice` 質問を事前スクリーニング)。
    pub coarse_config: Option<&'a CoarseToFineConfig>,
    /// 粗密 2 段階探索で使用する類似度スコアラー (任意、未設定時は `LexicalCoarseScorer` を使用)。
    pub coarse_scorer: Option<&'a dyn CoarseScorer>,
}

impl<'a> BatchInferenceOptions<'a> {
    /// 既定値でオプションを生成する。
    pub fn new() -> Self {
        Self::default()
    }

    /// ゲーティング設定を指定する。
    pub fn with_gating(mut self, gating_config: Option<&'a GatingConfig>) -> Self {
        self.gating_config = gating_config;
        self
    }

    /// 再利用可能なスクラッチパッドバッファを指定する。
    pub fn with_scratchpad(mut self, scratchpad: &'a mut BatchScratchpad) -> Self {
        self.scratchpad = Some(scratchpad);
        self
    }

    /// マイクロバッチ分割の最大チャンクサイズを指定する。
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = Some(chunk_size);
        self
    }

    /// 粗密 2 段階探索の設定を指定する。
    pub fn with_coarse_to_fine(mut self, coarse_config: &'a CoarseToFineConfig) -> Self {
        self.coarse_config = Some(coarse_config);
        self
    }

    /// 粗密 2 段階探索で用いる類似度スコアラーを指定する。
    pub fn with_coarse_scorer(mut self, scorer: &'a dyn CoarseScorer) -> Self {
        self.coarse_scorer = Some(scorer);
        self
    }
}

impl InferenceEngine {
    /// 事前パッキング済みバッチトークン列を受け取り、単一フォワードパスで各質問のロジットを算出する。
    ///
    /// 出力されたテンソルから各質問の真の候補数 $K_i$ 個のロジットのみをスライス抽出することで、
    /// パディングスロット由来のダミーロジットを完全に除外して返却する。
    ///
    /// # 引数
    /// - `batch`: パディングおよび平坦化済みのバッチトークナイズ結果。
    ///
    /// # 戻り値
    /// 質問キーと有効ロジット列 (`Vec<f64>`) の順序付きマップ。
    ///
    /// # エラー
    /// ONNX Runtime 推論処理またはテンソル検証に失敗した場合にエラーを返す。
    pub fn forward_batch_tokenized(
        &self,
        batch: &BatchTokenizedQuestions,
    ) -> Result<IndexMap<String, Vec<f64>>> {
        let total_logits = batch.dims.batch_size * batch.dims.num_options;
        let mut flat_logits = vec![0.0f64; total_logits];

        self.forward_batch_tokenized_into(batch, &mut flat_logits)?;

        let mut results = IndexMap::with_capacity(batch.dims.batch_size);
        let num_options = batch.dims.num_options;

        for (i, q_key) in batch.question_keys.iter().enumerate() {
            let k_i = batch.candidate_counts[i];
            let start_idx = i * num_options;
            let valid_logits = flat_logits[start_idx..start_idx + k_i].to_vec();
            results.insert(q_key.clone(), valid_logits);
        }

        Ok(results)
    }

    /// 事前パッキング済みバッチトークン列を受け取り、推論結果を指定されたスライスに書き込む。
    ///
    /// 借用ビュー (`TensorRef`) によりゼロコピーで ONNX Runtime に渡される。
    ///
    /// # 引数
    /// - `batch`: パディングおよび平坦化済みのバッチトークナイズ結果。
    /// - `out_logits`: 平坦化生ロジットの書き込み先バッファ (`[batch_size * num_options]`)。
    pub fn forward_batch_tokenized_into(
        &self,
        batch: &BatchTokenizedQuestions,
        out_logits: &mut [f64],
    ) -> Result<()> {
        self.forward_raw_into(
            &batch.input_ids,
            &batch.attention_mask,
            &batch.op_indices,
            batch.dims,
            out_logits,
        )
    }

    /// 共通 State と複数質問群を受け取り、トークンプレフィックス共有バッチ推論を実行する。
    ///
    /// State のトークナイズを 1 度だけ実行し、各質問を単一のバッチ次元に集約して
    /// 1 回のフォワードパスで全質問の有効ロジットを返却する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問識別子と質問定義のマップ。
    ///
    /// # 戻り値
    /// 質問キーと有効ロジット列 (`Vec<f64>`) の順序付きマップ。
    pub fn forward_batch_questions(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
    ) -> Result<IndexMap<String, Vec<f64>>> {
        let batch = tokenizer.encode_batch_questions(state, questions)?;
        self.forward_batch_tokenized(&batch)
    }

    /// 指定された推論オプション (`BatchInferenceOptions`) に基づき、複数質問の一括評価を実行する統一コアパイプライン。
    ///
    /// 粗密 2 段階探索の前処理、マイクロバッチ分割実行、スクラッチパッドバッファ再利用、
    /// および確信度ゲーティング適用を単一のフローで統合制御する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問識別子と質問定義のマップ。
    /// - `calib_config`: 較正温度設定。
    /// - `options`: バッチ推論の各種実行オプション。
    pub fn evaluate_batch_questions_with_options(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
        calib_config: &CalibrationConfig,
        options: BatchInferenceOptions<'_>,
    ) -> Result<IndexMap<String, Answer>> {
        let BatchInferenceOptions {
            gating_config,
            scratchpad,
            chunk_size,
            coarse_config,
            coarse_scorer,
        } = options;

        if let Some(coarse_cfg) = coarse_config {
            coarse_cfg.validate()?;
        }

        if questions.is_empty() {
            return Ok(IndexMap::new());
        }

        let mut plan = if let Some(coarse_cfg) = coarse_config {
            let fallback_scorer;
            let scorer: &dyn CoarseScorer = match coarse_scorer {
                Some(s) => s,
                None => {
                    fallback_scorer = LexicalCoarseScorer::new(tokenizer);
                    &fallback_scorer
                }
            };
            let prepared = prepare_coarse_questions(state, questions, coarse_cfg, scorer)?;
            let energies = IndexMap::with_capacity(prepared.original_keys_map.len());
            CoarseBatchPlan::Enabled(Box::new(EnabledCoarseBatch {
                processed_questions: prepared.processed_questions,
                runtime_state: CoarseRuntimeState {
                    original_keys_map: prepared.original_keys_map,
                    energies,
                },
            }))
        } else {
            CoarseBatchPlan::Disabled { questions }
        };

        let (effective_questions, mut coarse_state) = plan.as_parts_mut();
        let mut local_scratchpad;
        let scratchpad = match scratchpad {
            Some(sp) => sp,
            None => {
                local_scratchpad = BatchScratchpad::default();
                &mut local_scratchpad
            }
        };

        let mut answers = IndexMap::with_capacity(effective_questions.len());
        let ctx = BatchEvalContext {
            state,
            calib_config,
            gating_config,
        };

        let state_ids = tokenizer.encode_state(state)?;
        let effective_chunk_size = effective_single_model_chunk_size(chunk_size, state_ids.len());

        if let Some(chunk_limit) = effective_chunk_size.filter(|&c| effective_questions.len() > c) {
            for sub_questions in iter_map_chunks(effective_questions, chunk_limit) {
                let batch = tokenizer.encode_batch_with_pretokenized_state(
                    &state_ids,
                    &sub_questions,
                    tokenizer.max_sequence_length(),
                )?;
                self.execute_batch_chunk(
                    &ctx,
                    &batch,
                    &sub_questions,
                    questions,
                    scratchpad,
                    coarse_state.as_deref_mut(),
                    &mut answers,
                )?;
            }
        } else {
            let batch = tokenizer.encode_batch_with_pretokenized_state(
                &state_ids,
                effective_questions,
                tokenizer.max_sequence_length(),
            )?;
            self.execute_batch_chunk(
                &ctx,
                &batch,
                effective_questions,
                questions,
                scratchpad,
                coarse_state,
                &mut answers,
            )?;
        }

        if let CoarseBatchPlan::Enabled(enabled) = &plan {
            reconstruct_and_gate_batch_answers(
                &ctx,
                questions,
                &enabled.runtime_state.original_keys_map,
                &enabled.runtime_state.energies,
                &mut answers,
            )?;
        }

        Ok(answers)
    }

    /// 1 チャンク分のパッキング済みバッチを推論し、決定解決または粗密中間結果を格納する。
    #[allow(clippy::too_many_arguments)]
    fn execute_batch_chunk(
        &self,
        ctx: &BatchEvalContext<'_>,
        batch: &BatchTokenizedQuestions,
        chunk_questions: &IndexMap<String, Question>,
        original_questions: &IndexMap<String, Question>,
        scratchpad: &mut BatchScratchpad,
        coarse_state: Option<&mut CoarseRuntimeState>,
        answers: &mut IndexMap<String, Answer>,
    ) -> Result<()> {
        let num_options = batch.dims.num_options;
        let total_logits = batch.dims.batch_size * num_options;
        scratchpad.prepare(total_logits, num_options);

        let flat_logits = &mut scratchpad.flat_logits[..total_logits];
        self.forward_batch_tokenized_into(batch, flat_logits)?;

        let probs_buf = &mut scratchpad.probs_buf[..num_options];
        if let Some(coarse) = coarse_state {
            resolve_coarse_batch_chunk(
                ctx,
                batch,
                chunk_questions,
                original_questions,
                &coarse.original_keys_map,
                flat_logits,
                probs_buf,
                &mut coarse.energies,
                answers,
            )?;
        } else {
            resolve_batch_answers(ctx, batch, chunk_questions, flat_logits, probs_buf, answers)?;
        }

        Ok(())
    }

    /// 共通 State、複数質問群、および較正設定を受け取り、単一フォワードパスで判定結果を一括導出する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問識別子と質問定義のマップ。
    /// - `calib_config`: 較正温度設定。
    ///
    /// # 戻り値
    /// 質問キーと決定結果 (`Answer`) の順序付きマップ。
    pub fn evaluate_batch_questions(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
        calib_config: &CalibrationConfig,
    ) -> Result<IndexMap<String, Answer>> {
        self.evaluate_batch_questions_with_options(
            tokenizer,
            state,
            questions,
            calib_config,
            BatchInferenceOptions::default(),
        )
    }

    /// ゲーティング設定を明示指定して、複数質問一括バッチ推論を実行する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問識別子と質問定義のマップ。
    /// - `calib_config`: 較正温度設定。
    /// - `gating_config`: 明示的なゲーティング設定 (任意、未指定時は較正設定から導出)。
    pub fn evaluate_batch_questions_with_gating(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
        calib_config: &CalibrationConfig,
        gating_config: Option<&GatingConfig>,
    ) -> Result<IndexMap<String, Answer>> {
        self.evaluate_batch_questions_with_options(
            tokenizer,
            state,
            questions,
            calib_config,
            BatchInferenceOptions::default().with_gating(gating_config),
        )
    }

    /// スクラッチパッドバッファを再利用し、共通 State と複数質問群の判定結果を一括導出する。
    ///
    /// アロケーションを最小化したい高頻度 API サーバー環境向けの高速推論インターフェース。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問識別子と質問定義のマップ。
    /// - `calib_config`: 較正温度設定。
    /// - `scratchpad`: 再利用可能な作業用スクラッチパッドバッファ。
    pub fn evaluate_batch_questions_with_scratchpad(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
        calib_config: &CalibrationConfig,
        scratchpad: &mut BatchScratchpad,
    ) -> Result<IndexMap<String, Answer>> {
        self.evaluate_batch_questions_with_options(
            tokenizer,
            state,
            questions,
            calib_config,
            BatchInferenceOptions::default().with_scratchpad(scratchpad),
        )
    }

    /// スクラッチパッドバッファとゲーティング設定を指定してバッチ推論を実行する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問識別子と質問定義のマップ。
    /// - `calib_config`: 較正温度設定。
    /// - `gating_config`: 明示的なゲーティング設定 (任意)。
    /// - `scratchpad`: 再利用可能な作業用スクラッチパッドバッファ。
    pub fn evaluate_batch_questions_with_scratchpad_and_gating(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
        calib_config: &CalibrationConfig,
        gating_config: Option<&GatingConfig>,
        scratchpad: &mut BatchScratchpad,
    ) -> Result<IndexMap<String, Answer>> {
        self.evaluate_batch_questions_with_options(
            tokenizer,
            state,
            questions,
            calib_config,
            BatchInferenceOptions::default()
                .with_gating(gating_config)
                .with_scratchpad(scratchpad),
        )
    }

    /// 共通 State と複数質問群を指定されたチャンクサイズで分割実行し、有効ロジットを返却する。
    ///
    /// 質問数が極端に多いリクエストにおいて、メモリ使用量 ($N \times L_{\max}$) の急増を抑制し
    /// OOM (Out Of Memory) を防止するマイクロバッチ推論インターフェース。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問識別子と質問定義のマップ。
    /// - `chunk_size`: 1 回のフォワードパスで処理する最大質問数。
    pub fn forward_batch_questions_chunked(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
        chunk_size: usize,
    ) -> Result<IndexMap<String, Vec<f64>>> {
        let state_ids = tokenizer.encode_state(state)?;
        let effective_chunk = effective_single_model_chunk_size(Some(chunk_size), state_ids.len())
            .unwrap_or(chunk_size)
            .max(1);

        if questions.len() <= effective_chunk {
            let batch = tokenizer.encode_batch_with_pretokenized_state(
                &state_ids,
                questions,
                tokenizer.max_sequence_length(),
            )?;
            return self.forward_batch_tokenized(&batch);
        }

        let mut results = IndexMap::with_capacity(questions.len());

        for sub_questions in iter_map_chunks(questions, effective_chunk) {
            let batch = tokenizer.encode_batch_with_pretokenized_state(
                &state_ids,
                &sub_questions,
                tokenizer.max_sequence_length(),
            )?;
            let sub_results = self.forward_batch_tokenized(&batch)?;
            results.extend(sub_results);
        }

        Ok(results)
    }

    /// 共通 State と複数質問群を指定されたチャンクサイズで分割実行し、判定結果を一括導出する。
    ///
    /// サーバー層 (sokuto-server) のガードレール等と連携し、大規模質問リクエストを
    /// 安全な固定バッチサイズで順次推論・解決する。較正設定の閾値に基づき、各質問に透過的にゲーティングを適用する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問識別子と質問定義のマップ。
    /// - `calib_config`: 較正温度設定。
    /// - `chunk_size`: 1 回のフォワードパスで処理する最大質問数。
    pub fn evaluate_batch_questions_chunked(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
        calib_config: &CalibrationConfig,
        chunk_size: usize,
    ) -> Result<IndexMap<String, Answer>> {
        self.evaluate_batch_questions_with_options(
            tokenizer,
            state,
            questions,
            calib_config,
            BatchInferenceOptions::default().with_chunk_size(chunk_size),
        )
    }

    /// チャンク分割およびゲーティング設定を指定してバッチ推論を実行する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問識別子と質問定義のマップ。
    /// - `calib_config`: 較正温度設定。
    /// - `chunk_size`: 1 回のフォワードパスで処理する最大質問数。
    /// - `gating_config`: 明示的なゲーティング設定 (任意)。
    pub fn evaluate_batch_questions_chunked_with_gating(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
        calib_config: &CalibrationConfig,
        chunk_size: usize,
        gating_config: Option<&GatingConfig>,
    ) -> Result<IndexMap<String, Answer>> {
        self.evaluate_batch_questions_with_options(
            tokenizer,
            state,
            questions,
            calib_config,
            BatchInferenceOptions::default()
                .with_chunk_size(chunk_size)
                .with_gating(gating_config),
        )
    }

    /// 任意の類似度スコアラーを指定して、粗密 2 段階探索を適用した複数質問一括バッチ推論を実行する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問マップ。
    /// - `calib_config`: 較正温度設定。
    /// - `coarse_config`: 粗密探索設定。
    /// - `scorer`: 類似度スコアラー実装。
    pub fn evaluate_batch_questions_coarse_to_fine_with_scorer<S: CoarseScorer>(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
        calib_config: &CalibrationConfig,
        coarse_config: &CoarseToFineConfig,
        scorer: &S,
    ) -> Result<IndexMap<String, Answer>> {
        self.evaluate_batch_questions_with_options(
            tokenizer,
            state,
            questions,
            calib_config,
            BatchInferenceOptions::default()
                .with_coarse_to_fine(coarse_config)
                .with_coarse_scorer(scorer),
        )
    }

    /// 既定の語彙スコアラー (`LexicalCoarseScorer`) を用いて、粗密 2 段階探索を適用した複数質問一括バッチ推論を実行する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問マップ。
    /// - `calib_config`: 較正温度設定。
    /// - `coarse_config`: 粗密探索設定。
    pub fn evaluate_batch_questions_coarse_to_fine(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
        calib_config: &CalibrationConfig,
        coarse_config: &CoarseToFineConfig,
    ) -> Result<IndexMap<String, Answer>> {
        self.evaluate_batch_questions_with_options(
            tokenizer,
            state,
            questions,
            calib_config,
            BatchInferenceOptions::default().with_coarse_to_fine(coarse_config),
        )
    }

    /// 任意の類似度スコアラーを指定して、粗密 2 段階探索とマイクロバッチチャンキングを統合した安全推論を実行する。
    ///
    /// 候補数が膨大な質問を事前に Top-$M$ 件へスクリーニングした上で、
    /// 共通 State を 1 度だけエンコードし、質問群を指定されたチャンクサイズごとに順次フォワードパスへ供給する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問マップ。
    /// - `calib_config`: 較正温度設定。
    /// - `coarse_config`: 粗密探索設定。
    /// - `chunk_size`: 1 回のフォワードパスで処理する最大質問数。
    /// - `scorer`: 類似度スコアラー実装。
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_batch_questions_coarse_to_fine_chunked_with_scorer<S: CoarseScorer>(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
        calib_config: &CalibrationConfig,
        coarse_config: &CoarseToFineConfig,
        chunk_size: usize,
        scorer: &S,
    ) -> Result<IndexMap<String, Answer>> {
        self.evaluate_batch_questions_coarse_to_fine_chunked_with_gating_and_scorer(
            tokenizer,
            state,
            questions,
            calib_config,
            coarse_config,
            chunk_size,
            None,
            scorer,
        )
    }

    /// 任意の類似度スコアラーおよび明示的なゲーティング設定を指定して、粗密 2 段階探索とマイクロバッチチャンキングを統合した安全推論を実行する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問マップ。
    /// - `calib_config`: 較正温度設定。
    /// - `coarse_config`: 粗密探索設定。
    /// - `chunk_size`: 1 回のフォワードパスで処理する最大質問数。
    /// - `gating_config`: 明示的なゲーティング設定 (任意)。
    /// - `scorer`: 類似度スコアラー実装。
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_batch_questions_coarse_to_fine_chunked_with_gating_and_scorer<
        S: CoarseScorer,
    >(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
        calib_config: &CalibrationConfig,
        coarse_config: &CoarseToFineConfig,
        chunk_size: usize,
        gating_config: Option<&GatingConfig>,
        scorer: &S,
    ) -> Result<IndexMap<String, Answer>> {
        self.evaluate_batch_questions_with_options(
            tokenizer,
            state,
            questions,
            calib_config,
            BatchInferenceOptions::default()
                .with_coarse_to_fine(coarse_config)
                .with_chunk_size(chunk_size)
                .with_gating(gating_config)
                .with_coarse_scorer(scorer),
        )
    }

    /// 既定の語彙スコアラー (`LexicalCoarseScorer`) を用いて、粗密 2 段階探索とマイクロバッチチャンキングを統合した推論を実行する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問マップ。
    /// - `calib_config`: 較正温度設定。
    /// - `coarse_config`: 粗密探索設定。
    /// - `chunk_size`: 1 回のフォワードパスで処理する最大質問数。
    pub fn evaluate_batch_questions_coarse_to_fine_chunked(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
        calib_config: &CalibrationConfig,
        coarse_config: &CoarseToFineConfig,
        chunk_size: usize,
    ) -> Result<IndexMap<String, Answer>> {
        self.evaluate_batch_questions_coarse_to_fine_chunked_with_gating(
            tokenizer,
            state,
            questions,
            calib_config,
            coarse_config,
            chunk_size,
            None,
        )
    }

    /// 既定の語彙スコアラーとゲーティング設定を指定して、粗密 2 段階探索とマイクロバッチチャンキングを統合した推論を実行する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 質問マップ。
    /// - `calib_config`: 較正温度設定。
    /// - `coarse_config`: 粗密探索設定。
    /// - `chunk_size`: 1 回のフォワードパスで処理する最大質問数。
    /// - `gating_config`: 明示的なゲーティング設定 (任意)。
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_batch_questions_coarse_to_fine_chunked_with_gating(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
        calib_config: &CalibrationConfig,
        coarse_config: &CoarseToFineConfig,
        chunk_size: usize,
        gating_config: Option<&GatingConfig>,
    ) -> Result<IndexMap<String, Answer>> {
        self.evaluate_batch_questions_with_options(
            tokenizer,
            state,
            questions,
            calib_config,
            BatchInferenceOptions::default()
                .with_coarse_to_fine(coarse_config)
                .with_chunk_size(chunk_size)
                .with_gating(gating_config),
        )
    }

    /// 粗密二段階階層ルーティング (Coarse-to-Fine Hierarchical Routing) を用いて複数質問を一括評価する。
    ///
    /// 共通の階層オントロジーマッピングとルーター設定を各質問に適用し、
    /// 大分類オントロジー、ODIR Dirichlet 較正、Soft-Beam、エントロピー連動温度較正による低遅延・高精度推論を実行する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 共通文脈テキスト。
    /// - `questions`: 評価対象の質問群マップ。
    /// - `mapping`: 大分類・細分類の階層オントロジー定義。
    /// - `calib_config`: 較正温度設定。
    /// - `config`: 階層ルーター設定。
    /// - `gating_config`: ゲーティング閾値設定 (任意)。
    ///
    /// # 戻り値
    /// 各質問キーと階層推論結果 (`HierarchicalResult`) の順序付きマップ。
    ///
    /// # エラー
    /// - いずれかの質問が [`QuestionType::Choice`] ではない場合。
    /// - 質問の `criteria` が Map 形式ではない、あるいは空の場合。
    /// - 質問の候補キーが階層オントロジーの細分類定義に含まれていない場合。
    /// - ルーター構築または推論実行に失敗した場合。
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_batch_questions_hierarchical(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        questions: &IndexMap<String, Question>,
        mapping: &HierarchicalMapping,
        calib_config: &CalibrationConfig,
        config: HierarchicalRouterConfig,
        gating_config: Option<&GatingConfig>,
    ) -> Result<IndexMap<String, HierarchicalResult>> {
        if questions.is_empty() {
            return Ok(IndexMap::new());
        }

        validate_hierarchical_questions(questions, mapping)?;

        let state_ids = tokenizer.encode_state(state)?;
        let router = CoarseToFineRouter::new_with_pretokenized_state(
            self,
            tokenizer,
            &state_ids,
            mapping,
            calib_config,
            config,
        )?;
        let mut results = IndexMap::with_capacity(questions.len());

        for (q_key, question) in questions {
            let allowed_criteria = question.criteria.as_ref().and_then(|c| c.as_map());
            let res = router.route_with_allowed_candidates_and_id(
                state,
                Some(q_key.as_str()),
                &question.instructions,
                allowed_criteria,
                gating_config,
            )?;
            results.insert(q_key.clone(), res);
        }

        Ok(results)
    }
}

/// 階層ルーティング推論に渡された質問群が階層オントロジーおよび Choice 型に適合しているかを検証する。
///
/// # 検証項目
/// - すべての質問が [`QuestionType::Choice`] であること。
/// - `criteria` が指定されている場合、Map 形式かつ非空であること。
/// - `criteria` に指定された候補キーがすべて [`HierarchicalMapping`] の細分類オントロジーに含まれていること。
fn validate_hierarchical_questions(
    questions: &IndexMap<String, Question>,
    mapping: &HierarchicalMapping,
) -> Result<()> {
    for (q_key, question) in questions {
        if question.question_type != QuestionType::Choice {
            return Err(RuntimeError::InvalidQuestion(format!(
                "階層ルーティング対象の質問 '{q_key}' は Choice 型である必要がありますが、{:?} が指定されました。",
                question.question_type
            )));
        }

        if let Some(ref criteria) = question.criteria {
            let map = criteria.as_map().ok_or_else(|| {
                RuntimeError::InvalidQuestion(format!(
                    "階層ルーティング対象の質問 '{q_key}' の criteria は Map 形式である必要があります。"
                ))
            })?;

            if map.is_empty() {
                return Err(RuntimeError::InvalidQuestion(format!(
                    "階層ルーティング対象の質問 '{q_key}' の criteria が空です。"
                )));
            }

            for cand_key in map.keys() {
                if !mapping.fine_to_coarse.contains_key(cand_key) {
                    return Err(RuntimeError::InvalidQuestion(format!(
                        "質問 '{q_key}' の候補キー '{cand_key}' は階層オントロジー '{}' の細分類に含まれていません。",
                        mapping.name
                    )));
                }
            }
        }
    }

    Ok(())
}

/// バッチ推論結果の決定解決およびゲーティング適用に必要な共通コンテキスト。
struct BatchEvalContext<'a> {
    state: &'a str,
    calib_config: &'a CalibrationConfig,
    gating_config: Option<&'a GatingConfig>,
}

/// 粗密探索の前処理結果として、縮小済み質問マップと元候補キー配列マップを保持する。
struct PreparedCoarseBatch {
    /// 閾値を超える `Choice` 質問が Top-$M$ 候補に縮小された質問マップ。
    processed_questions: IndexMap<String, Question>,
    /// 縮小対象となった質問の元候補キー配列マップ。
    original_keys_map: IndexMap<String, Vec<String>>,
}

/// 統一バッチ推論における粗密復元用の実行時状態。
struct CoarseRuntimeState {
    original_keys_map: IndexMap<String, Vec<String>>,
    energies: IndexMap<String, Option<f64>>,
}

/// バッチ推論における粗密 2 段階探索の有効時計画データ。
struct EnabledCoarseBatch {
    processed_questions: IndexMap<String, Question>,
    runtime_state: CoarseRuntimeState,
}

/// バッチ推論における粗密 2 段階探索の実行計画および状態管理 enum。
///
/// 粗密探索の有効・無効状態を一元管理し、縮小済み質問マップと復元用状態の食い違いをコンパイル時に排除する。
enum CoarseBatchPlan<'a> {
    /// 粗密探索が無効な状態(元の質問マップを直接参照)
    Disabled {
        questions: &'a IndexMap<String, Question>,
    },
    /// 粗密探索が有効な状態(縮小済み質問マップおよび復元用実行時状態を保持)
    Enabled(Box<EnabledCoarseBatch>),
}

impl<'a> CoarseBatchPlan<'a> {
    /// 評価対象となる質問マップおよび粗密探索実行時状態の可変参照を分解取得する。
    fn as_parts_mut(&mut self) -> (&IndexMap<String, Question>, Option<&mut CoarseRuntimeState>) {
        match self {
            Self::Disabled { questions } => (*questions, None),
            Self::Enabled(enabled) => (
                &enabled.processed_questions,
                Some(&mut enabled.runtime_state),
            ),
        }
    }
}

/// 粗密探索の前処理として、候補数が閾値を超える `Choice` 質問を Top-$M$ 候補へ縮小する。
///
/// 縮小対象となった質問については全候補の元キー配列を `original_keys_map` に記録し、
/// 閾値以下の `Choice` 質問および `Choice` 以外の質問は元の定義のまま保持する。
///
/// # 引数
/// - `state`: 共通文脈テキスト。
/// - `questions`: 評価対象の質問マップ。
/// - `coarse_config`: 粗密探索設定。
/// - `scorer`: 類似度スコアラー実装。
///
/// # 戻り値
/// 縮小済み質問マップと元候補キー配列マップを格納した `PreparedCoarseBatch`。
fn prepare_coarse_questions<S: CoarseScorer + ?Sized>(
    state: &str,
    questions: &IndexMap<String, Question>,
    coarse_config: &CoarseToFineConfig,
    scorer: &S,
) -> Result<PreparedCoarseBatch> {
    let mut processed_questions = IndexMap::with_capacity(questions.len());
    let mut original_keys_map: IndexMap<String, Vec<String>> = IndexMap::new();

    for (q_key, question) in questions {
        if let Some(map) = coarse_config.applicable_candidates(question) {
            let original_keys: Vec<String> = map.keys().cloned().collect();
            let query = format!("State: {}\nInstructions: {}", state, question.instructions);

            let filtered = filter_top_candidates(&query, map, coarse_config, scorer)?;
            let sub_question =
                Question::new_choice(question.instructions.clone(), filtered.selected);

            processed_questions.insert(q_key.clone(), sub_question);
            original_keys_map.insert(q_key.clone(), original_keys);
        } else {
            processed_questions.insert(q_key.clone(), question.clone());
        }
    }

    Ok(PreparedCoarseBatch {
        processed_questions,
        original_keys_map,
    })
}

/// 粗密バッチ推論の 1 チャンク分の生ロジットから決定解決を行い、スクリーニング対象外の質問は即座にゲーティングを適用する。
///
/// スクリーニング対象の質問は後続の全候補空間復元後にゲーティングを行うため、この段階では決定解決と自由エネルギーの事前算出のみを行う。
///
/// # 引数
/// - `ctx`: 状態テキスト・較正設定・ゲーティング設定を保持する評価コンテキスト。
/// - `batch`: トークナイズ済みバッチ情報。
/// - `chunk_questions`: 縮小済み質問定義マップ (当チャンク分)。
/// - `original_questions`: 元の全候補を含む質問定義マップ。
/// - `original_keys_map`: 粗密スクリーニング対象となった質問の元候補キー配列マップ。
/// - `flat_logits`: 平坦化されたバッチ生ロジット列。
/// - `probs_buf`: Softmax 確率分布計算用の一時バッファ。
/// - `coarse_energies`: 事前算出した自由エネルギー値の格納先マップ。
/// - `answers`: 解決済み `Answer` の格納先マップ。
#[allow(clippy::too_many_arguments)]
fn resolve_coarse_batch_chunk(
    ctx: &BatchEvalContext<'_>,
    batch: &BatchTokenizedQuestions,
    chunk_questions: &IndexMap<String, Question>,
    original_questions: &IndexMap<String, Question>,
    original_keys_map: &IndexMap<String, Vec<String>>,
    flat_logits: &[f64],
    probs_buf: &mut [f64],
    coarse_energies: &mut IndexMap<String, Option<f64>>,
    answers: &mut IndexMap<String, Answer>,
) -> Result<()> {
    let num_options = batch.dims.num_options;
    for (i, q_key) in batch.question_keys.iter().enumerate() {
        let k_i = batch.candidate_counts[i];
        let start_idx = i * num_options;
        let valid_logits = &flat_logits[start_idx..start_idx + k_i];
        let question = chunk_questions
            .get(q_key)
            .expect("キーの一致が保証されている。");

        let orig_q = original_questions.get(q_key).unwrap_or(question);
        let answer = if !original_keys_map.contains_key(q_key) {
            resolve_and_gate_answer(ctx, q_key, orig_q, valid_logits, &mut probs_buf[..k_i])?
        } else {
            let answer = evaluate_question_with_buf(
                question,
                valid_logits,
                ctx.calib_config,
                &mut probs_buf[..k_i],
                Some(q_key.as_str()),
            )?;
            let energy = calculate_choice_energy_with_config(
                orig_q,
                valid_logits,
                ctx.calib_config,
                ctx.gating_config,
            );
            coarse_energies.insert(q_key.clone(), energy);
            answer
        };
        answers.insert(q_key.clone(), answer);
    }
    Ok(())
}

/// 粗密探索で縮小推論された各質問の結果を全候補空間へ復元し、複合確信度の再計算とゲーティングを適用する。
///
/// # 引数
/// - `ctx`: 状態テキスト・較正設定・ゲーティング設定を保持する評価コンテキスト。
/// - `questions`: 元の全候補を含む質問定義マップ。
/// - `original_keys_map`: 粗密スクリーニング対象となった質問の元候補キー配列マップ。
/// - `energies`: 事前算出された自由エネルギーマップ。
/// - `answers`: 復元・ゲーティング反映先の回答マップ。
fn reconstruct_and_gate_batch_answers(
    ctx: &BatchEvalContext<'_>,
    questions: &IndexMap<String, Question>,
    original_keys_map: &IndexMap<String, Vec<String>>,
    energies: &IndexMap<String, Option<f64>>,
    answers: &mut IndexMap<String, Answer>,
) -> Result<()> {
    for (q_key, original_keys) in original_keys_map {
        if let Some(answer) = answers.get_mut(q_key)
            && answer.is_choice()
        {
            let energy = energies.get(q_key).copied().flatten();
            reconstruct_and_gate_answer(
                answer,
                original_keys,
                energy,
                Some(ctx.state),
                Some(q_key),
                questions.get(q_key),
                ctx.gating_config,
                ctx.calib_config,
            )?;
        }
    }
    Ok(())
}

/// 単一質問の有効ロジットから決定プリミティブを解決し、エネルギー算出とゲーティングを適用する。
///
/// # 引数
/// - `ctx`: 状態テキスト・較正設定・ゲーティング設定を保持する評価コンテキスト。
/// - `q_key`: 質問識別子キー。
/// - `question`: 決定解決およびエスカレーションプロンプト生成に用いる質問定義。
/// - `valid_logits`: 該当質問の有効候補数 $K_i$ に切り出された生ロジット列。
/// - `probs_buf`: Softmax 確率分布計算用の一時バッファ (`valid_logits` と同長)。
fn resolve_and_gate_answer(
    ctx: &BatchEvalContext<'_>,
    q_key: &str,
    question: &Question,
    valid_logits: &[f64],
    probs_buf: &mut [f64],
) -> Result<Answer> {
    evaluate_and_gate_answer_with_buf(
        question,
        valid_logits,
        probs_buf,
        Some(ctx.state),
        Some(q_key),
        ctx.gating_config,
        ctx.calib_config,
    )
}

/// `IndexMap` を指定されたチャンクサイズで分割し、挿入順を保った部分マップのイテレータを返す。
///
/// 一時ベクタへの `collect` を排除し、`IndexMap::get_index` による O(1) インデックス範囲走査で
/// 必要なキャパシティのみを確保した部分マップを逐次生成する。`chunk_size` が 0 の場合は 1 として扱う。
///
/// # 引数
/// - `map`: 分割元の順序付きマップ。
/// - `chunk_size`: 1 チャンクあたりの最大要素数。
fn iter_map_chunks<K, V>(
    map: &IndexMap<K, V>,
    chunk_size: usize,
) -> impl Iterator<Item = IndexMap<K, V>> + '_
where
    K: Clone + Eq + std::hash::Hash,
    V: Clone,
{
    let step = chunk_size.max(1);
    (0..map.len()).step_by(step).map(move |start| {
        let end = (start + step).min(map.len());
        let mut sub_map = IndexMap::with_capacity(end - start);
        for idx in start..end {
            if let Some((key, value)) = map.get_index(idx) {
                sub_map.insert(key.clone(), value.clone());
            }
        }
        sub_map
    })
}

/// トークナイズ済みバッチの全質問に対して決定解決とゲーティングを適用し、結果マップに格納する。
///
/// # 引数
/// - `ctx`: 状態テキスト・較正設定・ゲーティング設定を保持する評価コンテキスト。
/// - `batch`: バッチ次元・質問キー順・候補数を保持するトークナイズ済みバッチ。
/// - `questions`: 質問識別子から質問定義を引くマップ。
/// - `flat_logits`: 平坦化されたバッチ生ロジット列 (`batch_size * num_options`)。
/// - `probs_buf`: バッチ内の最大候補数以上の作業バッファ。
/// - `answers`: 解決済み `Answer` を格納する出力マップ。
fn resolve_batch_answers(
    ctx: &BatchEvalContext<'_>,
    batch: &BatchTokenizedQuestions,
    questions: &IndexMap<String, Question>,
    flat_logits: &[f64],
    probs_buf: &mut [f64],
    answers: &mut IndexMap<String, Answer>,
) -> Result<()> {
    let num_options = batch.dims.num_options;
    for (i, q_key) in batch.question_keys.iter().enumerate() {
        let k_i = batch.candidate_counts[i];
        let start_idx = i * num_options;
        let valid_logits = &flat_logits[start_idx..start_idx + k_i];
        let question = questions.get(q_key).expect("キーの一致が保証されている。");
        let answer =
            resolve_and_gate_answer(ctx, q_key, question, valid_logits, &mut probs_buf[..k_i])?;
        answers.insert(q_key.clone(), answer);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;
    use sokuto_core::contract::model_spec::{MAX_NUM_OPTIONS, ModelInputDimensions};
    use sokuto_core::gating::DecisionRoute;

    #[test]
    fn test_resolve_and_gate_answer_choice_and_noul() {
        let calib_config = CalibrationConfig::default();
        let gating_config = GatingConfig {
            enabled: true,
            ..Default::default()
        };
        let ctx = BatchEvalContext {
            state: "テスト用入力テキスト",
            calib_config: &calib_config,
            gating_config: Some(&gating_config),
        };

        // 僅差の Choice ロジットは ConfirmOrEscalate と EscalationContext を付与する
        let mut criteria = IndexMap::new();
        criteria.insert("a".to_string(), "候補A".to_string());
        criteria.insert("b".to_string(), "候補B".to_string());
        let q_choice = Question::new_choice("選択せよ", criteria);
        let mut probs_buf = [0.0f64; 2];
        let ans_choice =
            resolve_and_gate_answer(&ctx, "q_choice", &q_choice, &[0.2, 0.1], &mut probs_buf)
                .unwrap();

        assert_eq!(ans_choice.as_choice(), Some("a"));
        let gating = ans_choice.gating.as_ref().unwrap();
        assert_ne!(gating.route, DecisionRoute::AutoExecute);
        assert!(gating.escalation.is_some());
        assert!(gating.energy.is_some());

        // 強い正の Noul ロジットは AutoExecute となる
        let q_noul = Question::new_noul("有効か?");
        let mut noul_buf = [0.0f64; 2];
        let ans_noul =
            resolve_and_gate_answer(&ctx, "q_noul", &q_noul, &[6.0, 0.0], &mut noul_buf).unwrap();
        assert_eq!(ans_noul.as_noul().map(|p| p > 0.99), Some(true));
        let noul_gating = ans_noul.gating.as_ref().unwrap();
        assert_eq!(noul_gating.route, DecisionRoute::AutoExecute);
    }

    #[test]
    fn test_resolve_batch_answers_slices_padded_logits() {
        let calib_config = CalibrationConfig::default();
        let gating_config = GatingConfig {
            enabled: true,
            ..Default::default()
        };
        let ctx = BatchEvalContext {
            state: "バッチ文脈",
            calib_config: &calib_config,
            gating_config: Some(&gating_config),
        };

        let mut criteria = IndexMap::new();
        criteria.insert("x".to_string(), "選択肢X".to_string());
        criteria.insert("y".to_string(), "選択肢Y".to_string());
        criteria.insert("z".to_string(), "選択肢Z".to_string());

        let mut questions = IndexMap::new();
        questions.insert("q1".to_string(), Question::new_choice("どれ?", criteria));
        questions.insert("q2".to_string(), Question::new_noul("正しいか?"));

        let batch = BatchTokenizedQuestions {
            input_ids: vec![],
            attention_mask: vec![],
            op_indices: vec![],
            dims: ModelInputDimensions {
                batch_size: 2,
                sequence_length: 1,
                num_options: 3,
            },
            question_keys: vec!["q1".to_string(), "q2".to_string()],
            candidate_counts: vec![3, 2],
            option_keys_per_q: vec![
                vec!["x".to_string(), "y".to_string(), "z".to_string()],
                vec!["true".to_string(), "false".to_string()],
            ],
        };

        // q2 のパディングスロット (+100.0) は candidate_counts = 2 により無視されるべき
        let flat_logits = vec![-1.0, 5.0, 0.0, 0.0, 6.0, 100.0];
        let mut probs_buf = vec![0.0f64; batch.dims.num_options];
        let mut answers = IndexMap::new();

        resolve_batch_answers(
            &ctx,
            &batch,
            &questions,
            &flat_logits,
            &mut probs_buf,
            &mut answers,
        )
        .unwrap();

        assert_eq!(answers.len(), 2);
        assert_eq!(answers.get("q1").and_then(|a| a.as_choice()), Some("y"));
        assert!(answers.get("q1").and_then(|a| a.gating.as_ref()).is_some());
        assert_eq!(
            answers
                .get("q2")
                .and_then(|a| a.as_noul())
                .map(|p| p < 0.05),
            Some(true)
        );
    }

    #[test]
    fn test_prepare_coarse_questions_screens_only_large_choice() {
        struct MockBatchScorer;

        impl CoarseScorer for MockBatchScorer {
            fn score_candidates_iter(
                &self,
                _query: &str,
                candidates: &mut dyn ExactSizeIterator<Item = (&str, &str)>,
            ) -> Result<Vec<f32>> {
                Ok((0..candidates.len()).map(|idx| idx as f32).collect())
            }
        }

        let mut large_criteria = IndexMap::new();
        large_criteria.insert("c1".to_string(), "候補1".to_string());
        large_criteria.insert("c2".to_string(), "候補2".to_string());
        large_criteria.insert("c3".to_string(), "候補3".to_string());
        large_criteria.insert("c4".to_string(), "候補4".to_string());

        let mut small_criteria = IndexMap::new();
        small_criteria.insert("s1".to_string(), "小候補1".to_string());
        small_criteria.insert("s2".to_string(), "小候補2".to_string());

        let mut questions = IndexMap::new();
        questions.insert(
            "large".to_string(),
            Question::new_choice("大候補質問", large_criteria),
        );
        questions.insert(
            "small".to_string(),
            Question::new_choice("小候補質問", small_criteria),
        );
        questions.insert("noul".to_string(), Question::new_noul("真偽質問?"));

        let coarse_config = CoarseToFineConfig {
            threshold: 3,
            top_m: NonZeroUsize::new(2).expect("2 は非ゼロ。"),
            preserve_negative: false,
            negative_keys: vec![],
            ..Default::default()
        };

        let PreparedCoarseBatch {
            processed_questions: processed,
            original_keys_map,
        } = prepare_coarse_questions("状態テキスト", &questions, &coarse_config, &MockBatchScorer)
            .unwrap();

        assert_eq!(processed.len(), 3);
        assert_eq!(original_keys_map.len(), 1);
        assert_eq!(
            original_keys_map.get("large").map(|v| v.as_slice()),
            Some(
                &[
                    "c1".to_string(),
                    "c2".to_string(),
                    "c3".to_string(),
                    "c4".to_string()
                ][..]
            )
        );

        let screened_keys: Vec<&String> = processed
            .get("large")
            .and_then(|q| q.criteria.as_ref())
            .and_then(|c| c.as_map())
            .unwrap()
            .keys()
            .collect();
        assert_eq!(screened_keys, vec!["c3", "c4"]);

        let small_keys: Vec<&String> = processed
            .get("small")
            .and_then(|q| q.criteria.as_ref())
            .and_then(|c| c.as_map())
            .unwrap()
            .keys()
            .collect();
        assert_eq!(small_keys, vec!["s1", "s2"]);
        assert!(!original_keys_map.contains_key("small"));
        assert!(!original_keys_map.contains_key("noul"));
    }

    #[test]
    fn test_resolve_and_gate_answer_respects_gating_energy_temperature() {
        use sokuto_core::contract::calibration::GatingThresholds;
        use sokuto_core::math::normalized_free_energy;

        let calib_config = CalibrationConfig {
            gating_thresholds: GatingThresholds {
                energy_temperature: Some(1.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let gating_config = GatingConfig {
            enabled: true,
            thresholds: GatingThresholds {
                energy_temperature: Some(5.0),
                ..Default::default()
            },
        };
        let ctx = BatchEvalContext {
            state: "テスト用入力テキスト",
            calib_config: &calib_config,
            gating_config: Some(&gating_config),
        };

        let mut criteria = IndexMap::new();
        criteria.insert("a".to_string(), "候補A".to_string());
        criteria.insert("b".to_string(), "候補B".to_string());
        let q_choice = Question::new_choice("選択せよ", criteria);
        let logits = [3.0, -1.0];
        let mut probs_buf = [0.0f64; 2];

        let ans = resolve_and_gate_answer(&ctx, "q_temp", &q_choice, &logits, &mut probs_buf)
            .expect("ゲート解決に成功する。");

        let actual_energy = ans.gating.as_ref().and_then(|g| g.energy).unwrap();
        let expected_energy = normalized_free_energy(&logits, 5.0).unwrap();
        let fallback_energy = normalized_free_energy(&logits, 1.0).unwrap();

        assert!((actual_energy - expected_energy).abs() < 1e-12);
        assert!((actual_energy - fallback_energy).abs() > 1e-3);
    }

    #[test]
    fn test_resolve_coarse_batch_chunk_defers_gating_until_reconstruction() {
        let calib_config = CalibrationConfig::default();
        let gating_config = GatingConfig {
            enabled: true,
            ..Default::default()
        };
        let ctx = BatchEvalContext {
            state: "粗密バッチ文脈",
            calib_config: &calib_config,
            gating_config: Some(&gating_config),
        };

        let mut full_criteria = IndexMap::new();
        full_criteria.insert("c1".to_string(), "候補1".to_string());
        full_criteria.insert("c2".to_string(), "候補2".to_string());
        full_criteria.insert("c3".to_string(), "候補3".to_string());
        full_criteria.insert("c4".to_string(), "候補4".to_string());

        let mut sub_criteria = IndexMap::new();
        sub_criteria.insert("c3".to_string(), "候補3".to_string());
        sub_criteria.insert("c4".to_string(), "候補4".to_string());

        let mut original_questions = IndexMap::new();
        original_questions.insert(
            "q_coarse".to_string(),
            Question::new_choice("大候補", full_criteria),
        );
        original_questions.insert("q_noul".to_string(), Question::new_noul("真偽?"));

        let mut chunk_questions = IndexMap::new();
        chunk_questions.insert(
            "q_coarse".to_string(),
            Question::new_choice("大候補", sub_criteria),
        );
        chunk_questions.insert("q_noul".to_string(), Question::new_noul("真偽?"));

        let mut original_keys_map = IndexMap::new();
        original_keys_map.insert(
            "q_coarse".to_string(),
            vec![
                "c1".to_string(),
                "c2".to_string(),
                "c3".to_string(),
                "c4".to_string(),
            ],
        );

        let batch = BatchTokenizedQuestions {
            input_ids: vec![],
            attention_mask: vec![],
            op_indices: vec![],
            dims: ModelInputDimensions {
                batch_size: 2,
                sequence_length: 1,
                num_options: 2,
            },
            question_keys: vec!["q_coarse".to_string(), "q_noul".to_string()],
            candidate_counts: vec![2, 2],
            option_keys_per_q: vec![
                vec!["c3".to_string(), "c4".to_string()],
                vec!["true".to_string(), "false".to_string()],
            ],
        };

        let flat_logits = vec![4.0, -1.0, 6.0, 0.0];
        let mut probs_buf = vec![0.0f64; 2];
        let mut coarse_energies = IndexMap::new();
        let mut answers = IndexMap::new();

        resolve_coarse_batch_chunk(
            &ctx,
            &batch,
            &chunk_questions,
            &original_questions,
            &original_keys_map,
            &flat_logits,
            &mut probs_buf,
            &mut coarse_energies,
            &mut answers,
        )
        .unwrap();

        // スクリーニング対象質問はこの段階ではゲーティングされず、エネルギーのみ保持される
        assert!(answers.get("q_coarse").unwrap().gating.is_none());
        assert!(coarse_energies.get("q_coarse").and_then(|e| *e).is_some());
        // スクリーニング対象外質問はこの段階で 1 度だけゲーティングされる
        assert!(answers.get("q_noul").unwrap().gating.is_some());

        reconstruct_and_gate_batch_answers(
            &ctx,
            &original_questions,
            &original_keys_map,
            &coarse_energies,
            &mut answers,
        )
        .unwrap();

        let coarse_ans = answers.get("q_coarse").unwrap();
        assert_eq!(coarse_ans.probabilities().unwrap().len(), 4);
        assert_eq!(coarse_ans.probabilities().unwrap().get("c1"), Some(&0.0));
        assert!(coarse_ans.gating.is_some());
        assert_eq!(
            coarse_ans.gating.as_ref().unwrap().energy,
            coarse_energies.get("q_coarse").copied().flatten()
        );

        // 自由エネルギーが枝刈り後の M=2 ではなく全候補数 K=4 の母集団で正規化されていることを検証
        let coarse_energy = coarse_energies
            .get("q_coarse")
            .copied()
            .flatten()
            .expect("エネルギーが存在する。");
        let raw_energy = sokuto_core::math::free_energy(&[4.0, -1.0], 1.0).unwrap();
        let expected_energy_k4 = raw_energy + (4.0f64).ln();
        assert!((coarse_energy - expected_energy_k4).abs() < 1e-6);
    }

    #[test]
    fn test_iter_map_chunks_preserves_order_and_handles_edge_cases() {
        let mut map = IndexMap::new();
        for idx in 0..5 {
            map.insert(format!("q{idx}"), idx);
        }

        // 端数を含む通常分割で挿入順と値が維持されること。
        let chunks: Vec<Vec<(String, usize)>> = iter_map_chunks(&map, 2)
            .map(|sub| sub.into_iter().collect())
            .collect();
        assert_eq!(
            chunks,
            vec![
                vec![("q0".to_string(), 0), ("q1".to_string(), 1)],
                vec![("q2".to_string(), 2), ("q3".to_string(), 3)],
                vec![("q4".to_string(), 4)],
            ]
        );

        // チャンクサイズが 0 の場合は 1 として安全に分割されること。
        let single_chunks: Vec<usize> = iter_map_chunks(&map, 0).map(|sub| sub.len()).collect();
        assert_eq!(single_chunks, vec![1, 1, 1, 1, 1]);

        // チャンクサイズが要素数以上の場合は単一チャンクとなること。
        let full_chunks: Vec<usize> = iter_map_chunks(&map, 10).map(|sub| sub.len()).collect();
        assert_eq!(full_chunks, vec![5]);

        // 空マップの場合は空イテレータとなること。
        let empty: IndexMap<String, usize> = IndexMap::new();
        assert_eq!(iter_map_chunks(&empty, 2).count(), 0);
    }

    #[test]
    fn test_batch_inference_options_builder() {
        let default_opts = BatchInferenceOptions::default();
        assert!(default_opts.gating_config.is_none());
        assert!(default_opts.scratchpad.is_none());
        assert!(default_opts.chunk_size.is_none());
        assert!(default_opts.coarse_config.is_none());
        assert!(default_opts.coarse_scorer.is_none());

        let gating_cfg = GatingConfig::default();
        let coarse_cfg = CoarseToFineConfig::default();
        let mut scratchpad = BatchScratchpad::with_capacity(16, 4);
        struct DummyScorer;
        impl CoarseScorer for DummyScorer {
            fn score_candidates_iter(
                &self,
                _query: &str,
                candidates: &mut dyn ExactSizeIterator<Item = (&str, &str)>,
            ) -> Result<Vec<f32>> {
                Ok(vec![0.0; candidates.len()])
            }
        }
        let scorer = DummyScorer;

        let opts = BatchInferenceOptions::new()
            .with_gating(Some(&gating_cfg))
            .with_scratchpad(&mut scratchpad)
            .with_chunk_size(8)
            .with_coarse_to_fine(&coarse_cfg)
            .with_coarse_scorer(&scorer);

        assert!(opts.gating_config.is_some());
        assert!(opts.scratchpad.is_some());
        assert_eq!(opts.chunk_size, Some(8));
        assert!(opts.coarse_config.is_some());
        assert!(opts.coarse_scorer.is_some());
    }

    #[test]
    fn test_single_model_long_state_chunk_limiting() {
        use crate::engine::config::{
            SINGLE_MODEL_LONG_STATE_MAX_CHUNK, SINGLE_MODEL_LONG_STATE_THRESHOLD,
            effective_single_model_chunk_size,
        };

        // 短文時 (閾値以下) は要求サイズが維持されること。
        let short_len = SINGLE_MODEL_LONG_STATE_THRESHOLD;
        assert_eq!(effective_single_model_chunk_size(None, short_len), None);
        assert_eq!(
            effective_single_model_chunk_size(Some(16), short_len),
            Some(16)
        );

        // 長文時 (閾値超え) は最大 4 に制限されること。
        let long_len = SINGLE_MODEL_LONG_STATE_THRESHOLD + 1;
        assert_eq!(
            effective_single_model_chunk_size(None, long_len),
            Some(SINGLE_MODEL_LONG_STATE_MAX_CHUNK)
        );
        assert_eq!(
            effective_single_model_chunk_size(Some(16), long_len),
            Some(SINGLE_MODEL_LONG_STATE_MAX_CHUNK)
        );
        assert_eq!(
            effective_single_model_chunk_size(Some(2), long_len),
            Some(2)
        );

        // 極長文時 (> 1536) は最大 2 に制限されること。
        let very_long_len = 1600;
        assert_eq!(
            effective_single_model_chunk_size(None, very_long_len),
            Some(2)
        );
        assert_eq!(
            effective_single_model_chunk_size(Some(8), very_long_len),
            Some(2)
        );
        assert_eq!(
            effective_single_model_chunk_size(Some(1), very_long_len),
            Some(1)
        );
    }

    #[test]
    fn test_validate_hierarchical_questions() {
        let mapping = HierarchicalMapping::banking77();

        // 1. criteria が None の Choice 質問は正常に通過すること。
        let mut questions_none = IndexMap::new();
        questions_none.insert(
            "q_choice_none".to_string(),
            Question {
                question_type: QuestionType::Choice,
                instructions: "カテゴリを選択。".to_string(),
                criteria: None,
            },
        );
        assert!(validate_hierarchical_questions(&questions_none, &mapping).is_ok());

        // 2. criteria がオントロジー内の細分類キーを含む Choice 質問は正常に通過すること。
        let mut valid_criteria = IndexMap::new();
        valid_criteria.insert("lost_or_stolen_card".to_string(), "カード紛失".to_string());
        valid_criteria.insert("card_arrival".to_string(), "カード未着".to_string());
        let mut questions_valid = IndexMap::new();
        questions_valid.insert(
            "q_choice_valid".to_string(),
            Question::new_choice("カード関連の問い合わせ。", valid_criteria),
        );
        assert!(validate_hierarchical_questions(&questions_valid, &mapping).is_ok());

        // 3. Score 型の質問が含まれる場合は InvalidQuestion エラーとなること。
        let mut questions_score = IndexMap::new();
        questions_score.insert(
            "q_score".to_string(),
            Question::new_score("重要度評価。", vec!["低".to_string(), "高".to_string()]),
        );
        let err_score = validate_hierarchical_questions(&questions_score, &mapping)
            .expect_err("Score 型の質問は拒否されること。");
        assert!(matches!(err_score, RuntimeError::InvalidQuestion(_)));

        // 4. Noul 型の質問が含まれる場合は InvalidQuestion エラーとなること。
        let mut questions_noul = IndexMap::new();
        questions_noul.insert("q_noul".to_string(), Question::new_noul("本人確認済みか。"));
        let err_noul = validate_hierarchical_questions(&questions_noul, &mapping)
            .expect_err("Noul 型の質問は拒否されること。");
        assert!(matches!(err_noul, RuntimeError::InvalidQuestion(_)));

        // 5. criteria が List 形式の Choice 質問は InvalidQuestion エラーとなること。
        let mut questions_invalid_criteria_type = IndexMap::new();
        questions_invalid_criteria_type.insert(
            "q_list_criteria".to_string(),
            Question {
                question_type: QuestionType::Choice,
                instructions: "不正形式。".to_string(),
                criteria: Some(sokuto_core::schema::Criteria::List(vec!["a".to_string()])),
            },
        );
        let err_list_criteria =
            validate_hierarchical_questions(&questions_invalid_criteria_type, &mapping)
                .expect_err("List 形式の criteria は拒否されること。");
        assert!(matches!(
            err_list_criteria,
            RuntimeError::InvalidQuestion(_)
        ));

        // 6. criteria が空マップの Choice 質問は InvalidQuestion エラーとなること。
        let mut questions_empty_criteria = IndexMap::new();
        questions_empty_criteria.insert(
            "q_empty_criteria".to_string(),
            Question::new_choice("空マップ。", IndexMap::new()),
        );
        let err_empty_criteria =
            validate_hierarchical_questions(&questions_empty_criteria, &mapping)
                .expect_err("空の criteria マップは拒否されること。");
        assert!(matches!(
            err_empty_criteria,
            RuntimeError::InvalidQuestion(_)
        ));

        // 7. criteria にオントロジー外の細分類キーが含まれる場合は InvalidQuestion エラーとなること。
        let mut invalid_criteria = IndexMap::new();
        invalid_criteria.insert(
            "unknown_intent_category".to_string(),
            "未知のカテゴリ".to_string(),
        );
        let mut questions_invalid_key = IndexMap::new();
        questions_invalid_key.insert(
            "q_unknown_key".to_string(),
            Question::new_choice("オントロジー外。", invalid_criteria),
        );
        let err_unknown_key = validate_hierarchical_questions(&questions_invalid_key, &mapping)
            .expect_err("オントロジー外のキーは拒否されること。");
        assert!(matches!(err_unknown_key, RuntimeError::InvalidQuestion(_)));
    }

    #[test]
    fn test_evaluate_batch_questions_empty_returns_ok() {
        let engine = InferenceEngine::empty_for_test();
        let tokenizer = JevTokenizer::empty_for_test();
        let empty_questions = IndexMap::new();
        let calib = CalibrationConfig::default();
        let coarse_cfg = CoarseToFineConfig::default();

        // 1. evaluate_batch_questions
        let res = engine.evaluate_batch_questions(&tokenizer, "state", &empty_questions, &calib);
        assert!(res.is_ok());
        assert!(res.unwrap().is_empty());

        // 2. evaluate_batch_questions_with_options
        let res_opts = engine.evaluate_batch_questions_with_options(
            &tokenizer,
            "state",
            &empty_questions,
            &calib,
            BatchInferenceOptions::default().with_coarse_to_fine(&coarse_cfg),
        );
        assert!(res_opts.is_ok());
        assert!(res_opts.unwrap().is_empty());

        // 3. evaluate_batch_questions_chunked
        let res_chunked = engine.evaluate_batch_questions_chunked(
            &tokenizer,
            "state",
            &empty_questions,
            &calib,
            4,
        );
        assert!(res_chunked.is_ok());
        assert!(res_chunked.unwrap().is_empty());

        // 4. evaluate_batch_questions_coarse_to_fine
        let res_c2f = engine.evaluate_batch_questions_coarse_to_fine(
            &tokenizer,
            "state",
            &empty_questions,
            &calib,
            &coarse_cfg,
        );
        assert!(res_c2f.is_ok());
        assert!(res_c2f.unwrap().is_empty());

        // 5. evaluate_batch_questions_coarse_to_fine_chunked
        let res_c2f_chunked = engine.evaluate_batch_questions_coarse_to_fine_chunked(
            &tokenizer,
            "state",
            &empty_questions,
            &calib,
            &coarse_cfg,
            4,
        );
        assert!(res_c2f_chunked.is_ok());
        assert!(res_c2f_chunked.unwrap().is_empty());

        // 6. evaluate_batch_questions_hierarchical
        let mapping = HierarchicalMapping::banking77();
        let res_hierarchical = engine.evaluate_batch_questions_hierarchical(
            &tokenizer,
            "state",
            &empty_questions,
            &mapping,
            &calib,
            HierarchicalRouterConfig::default(),
            None,
        );
        assert!(res_hierarchical.is_ok());
        assert!(res_hierarchical.unwrap().is_empty());
    }

    #[test]
    fn test_evaluate_batch_questions_empty_with_invalid_coarse_config_fails() {
        let engine = InferenceEngine::empty_for_test();
        let tokenizer = JevTokenizer::empty_for_test();
        let empty_questions = IndexMap::new();
        let calib = CalibrationConfig::default();

        let invalid_coarse_cfg = CoarseToFineConfig {
            top_m: NonZeroUsize::new(MAX_NUM_OPTIONS + 1).expect("モデル上限超過は非ゼロ。"),
            ..Default::default()
        };

        let res = engine.evaluate_batch_questions_with_options(
            &tokenizer,
            "state",
            &empty_questions,
            &calib,
            BatchInferenceOptions::default().with_coarse_to_fine(&invalid_coarse_cfg),
        );
        assert!(res.is_err());
        assert!(matches!(res.unwrap_err(), RuntimeError::InvalidQuestion(_)));
    }
}
