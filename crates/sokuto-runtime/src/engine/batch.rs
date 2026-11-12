//! # 複数質問一括バッチ推論モジュール
//!
//! トークンレベルのプレフィックス共有と動的バッチ Collate により、
//! 単一フォワードパスでの並列推論および決定プリミティブ解決を実現する。

use indexmap::IndexMap;
use sokuto_core::contract::CalibrationConfig;
use sokuto_core::decision::evaluate_question_with_buf;
use sokuto_core::gating::GatingConfig;
use sokuto_core::schema::{Answer, Question};

use crate::engine::coarse::{
    CoarseScorer, CoarseToFineConfig, LexicalCoarseScorer, filter_top_candidates,
    reconstruct_and_gate_answer,
};
use crate::engine::gating::{
    apply_gating_to_answer_with_logits, calculate_choice_energy_with_config,
};
use crate::engine::session::InferenceEngine;
use crate::error::Result;
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

    /// 共通 State、複数質問群、および較正設定を受け取り、単一フォワードパスで判定結果を一括導出する。
    ///
    /// # 処理フロー
    /// 1. `tokenizer.encode_batch_questions` によるプレフィックス共有バッチ化。
    /// 2. `forward_batch_tokenized_into` による単一フォワードパス実行。
    /// 3. 各質問の真の候補数 $K_i$ によるスライス抽出とダミーロジット破棄。
    /// 4. `sokuto_core::decision::evaluate_question_with_buf` による決定論的プリミティブ解決。
    /// 5. 較正設定の閾値に基づき、透過的に確信度ゲーティングを適用して返却。
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
        self.evaluate_batch_questions_with_gating(tokenizer, state, questions, calib_config, None)
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
        let batch = tokenizer.encode_batch_questions(state, questions)?;
        let total_logits = batch.dims.batch_size * batch.dims.num_options;
        let mut flat_logits = vec![0.0f64; total_logits];

        self.forward_batch_tokenized_into(&batch, &mut flat_logits)?;

        let mut answers = IndexMap::with_capacity(batch.dims.batch_size);
        let mut probs_buf = vec![0.0f64; batch.dims.num_options];
        let ctx = BatchEvalContext {
            state,
            calib_config,
            gating_config,
        };
        resolve_batch_answers(
            &ctx,
            &batch,
            questions,
            &flat_logits,
            &mut probs_buf,
            &mut answers,
        )?;

        Ok(answers)
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
        self.evaluate_batch_questions_with_scratchpad_and_gating(
            tokenizer,
            state,
            questions,
            calib_config,
            None,
            scratchpad,
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
        let batch = tokenizer.encode_batch_questions(state, questions)?;
        let total_logits = batch.dims.batch_size * batch.dims.num_options;
        let num_options = batch.dims.num_options;

        scratchpad.prepare(total_logits, num_options);

        self.forward_batch_tokenized_into(&batch, &mut scratchpad.flat_logits[..total_logits])?;

        let mut answers = IndexMap::with_capacity(batch.dims.batch_size);
        let ctx = BatchEvalContext {
            state,
            calib_config,
            gating_config,
        };
        resolve_batch_answers(
            &ctx,
            &batch,
            questions,
            &scratchpad.flat_logits[..total_logits],
            &mut scratchpad.probs_buf,
            &mut answers,
        )?;

        Ok(answers)
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
        let chunk_size = chunk_size.max(1);
        if questions.len() <= chunk_size {
            return self.forward_batch_questions(tokenizer, state, questions);
        }

        let state_ids = tokenizer.encode_state(state)?;
        let mut results = IndexMap::with_capacity(questions.len());

        for chunk_slice in questions.iter().collect::<Vec<_>>().chunks(chunk_size) {
            let mut sub_questions = IndexMap::with_capacity(chunk_slice.len());
            for (k, v) in chunk_slice {
                sub_questions.insert((*k).clone(), (*v).clone());
            }

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
        self.evaluate_batch_questions_chunked_with_gating(
            tokenizer,
            state,
            questions,
            calib_config,
            chunk_size,
            None,
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
        let chunk_size = chunk_size.max(1);
        if questions.len() <= chunk_size {
            return self.evaluate_batch_questions_with_gating(
                tokenizer,
                state,
                questions,
                calib_config,
                gating_config,
            );
        }

        let state_ids = tokenizer.encode_state(state)?;
        let mut answers = IndexMap::with_capacity(questions.len());
        let ctx = BatchEvalContext {
            state,
            calib_config,
            gating_config,
        };

        for chunk_slice in questions.iter().collect::<Vec<_>>().chunks(chunk_size) {
            let mut sub_questions = IndexMap::with_capacity(chunk_slice.len());
            for (k, v) in chunk_slice {
                sub_questions.insert((*k).clone(), (*v).clone());
            }

            let batch = tokenizer.encode_batch_with_pretokenized_state(
                &state_ids,
                &sub_questions,
                tokenizer.max_sequence_length(),
            )?;

            let total_logits = batch.dims.batch_size * batch.dims.num_options;
            let mut flat_logits = vec![0.0f64; total_logits];
            self.forward_batch_tokenized_into(&batch, &mut flat_logits)?;

            let mut probs_buf = vec![0.0f64; batch.dims.num_options];
            resolve_batch_answers(
                &ctx,
                &batch,
                &sub_questions,
                &flat_logits,
                &mut probs_buf,
                &mut answers,
            )?;
        }

        Ok(answers)
    }

    /// 任意の類似度スコアラーを指定して、粗密 2 段階探索を適用した複数質問一括バッチ推論を実行する。
    ///
    /// # 処理フロー
    /// 1. 各質問を走査し、`Choice` かつ候補数 $K > \text{threshold}$ の質問を個別に Top-M 候補にスクリーニング。
    /// 2. 縮小質問群(および閾値以下の元質問群)をプレフィックス共有バッチ化。
    /// 3. バッチ全体の最大候補数 $K_{\max}$ が圧縮された状態で単一フォワードパスを実行。
    /// 4. 各質問の決定プリミティブを解決し、スクリーニングされた質問については除外候補を確率 0.0 として全候補確率マップを再構成。
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
        coarse_config.validate()?;

        if questions.is_empty() {
            return self.evaluate_batch_questions(tokenizer, state, questions, calib_config);
        }

        // 各質問の前処理と縮小
        let PreparedCoarseBatch {
            processed_questions,
            original_keys_map,
        } = prepare_coarse_questions(state, questions, coarse_config, scorer)?;

        // バッチ推論を実行 (最大候補数 K_max が縮小された状態で単一フォワードパス)
        let mut answers =
            self.evaluate_batch_questions(tokenizer, state, &processed_questions, calib_config)?;

        // スクリーニングされた質問の確率マップ・複合確信度を全候補空間に復元し、ゲーティングを再適用
        let ctx = BatchEvalContext {
            state,
            calib_config,
            gating_config: None,
        };
        reconstruct_and_gate_batch_answers(
            &ctx,
            questions,
            &original_keys_map,
            None,
            &mut answers,
        )?;

        Ok(answers)
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
        let scorer = LexicalCoarseScorer::new(tokenizer);
        self.evaluate_batch_questions_coarse_to_fine_with_scorer(
            tokenizer,
            state,
            questions,
            calib_config,
            coarse_config,
            &scorer,
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
        coarse_config.validate()?;

        if questions.is_empty() {
            return Ok(IndexMap::new());
        }

        let effective_chunk_size = chunk_size.max(1);

        // 1. 各質問の前処理と粗スクリーニング (Top-M 抽出)
        let PreparedCoarseBatch {
            processed_questions,
            original_keys_map,
        } = prepare_coarse_questions(state, questions, coarse_config, scorer)?;

        // 2. 共通 State を 1 度だけエンコードし全チャンクで再利用
        let state_ids = tokenizer.encode_state(state)?;
        let mut answers = IndexMap::with_capacity(questions.len());
        let mut coarse_energies: IndexMap<String, Option<f64>> =
            IndexMap::with_capacity(original_keys_map.len());
        let ctx = BatchEvalContext {
            state,
            calib_config,
            gating_config,
        };

        // 3. マイクロバッチ分割逐次推論
        for chunk_slice in processed_questions
            .iter()
            .collect::<Vec<_>>()
            .chunks(effective_chunk_size)
        {
            let mut sub_questions = IndexMap::with_capacity(chunk_slice.len());
            for (k, v) in chunk_slice {
                sub_questions.insert((*k).clone(), (*v).clone());
            }

            let batch = tokenizer.encode_batch_with_pretokenized_state(
                &state_ids,
                &sub_questions,
                tokenizer.max_sequence_length(),
            )?;

            let total_logits = batch.dims.batch_size * batch.dims.num_options;
            let mut flat_logits = vec![0.0f64; total_logits];
            self.forward_batch_tokenized_into(&batch, &mut flat_logits)?;

            let mut probs_buf = vec![0.0f64; batch.dims.num_options];
            let num_options = batch.dims.num_options;

            for (i, q_key) in batch.question_keys.iter().enumerate() {
                let k_i = batch.candidate_counts[i];
                let start_idx = i * num_options;
                let valid_logits = &flat_logits[start_idx..start_idx + k_i];
                let question = sub_questions
                    .get(q_key)
                    .expect("キーの一致が保証されている。");

                let answer = if !original_keys_map.contains_key(q_key) {
                    let orig_q = questions.get(q_key).unwrap_or(question);
                    resolve_and_gate_answer(
                        &ctx,
                        q_key,
                        orig_q,
                        valid_logits,
                        &mut probs_buf[..k_i],
                    )?
                } else {
                    let answer = evaluate_question_with_buf(
                        question,
                        valid_logits,
                        calib_config,
                        &mut probs_buf[..k_i],
                        Some(q_key.as_str()),
                    )?;
                    let energy = calculate_choice_energy_with_config(
                        question,
                        valid_logits,
                        calib_config,
                        gating_config,
                    );
                    coarse_energies.insert(q_key.clone(), energy);
                    answer
                };
                answers.insert(q_key.clone(), answer);
            }
        }

        // 4. スクリーニングされた質問の確率マップ・複合確信度を全候補空間に復元し、ゲーティングを適用
        reconstruct_and_gate_batch_answers(
            &ctx,
            questions,
            &original_keys_map,
            Some(&coarse_energies),
            &mut answers,
        )?;

        Ok(answers)
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
        let scorer = LexicalCoarseScorer::new(tokenizer);
        self.evaluate_batch_questions_coarse_to_fine_chunked_with_scorer(
            tokenizer,
            state,
            questions,
            calib_config,
            coarse_config,
            chunk_size,
            &scorer,
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
        let scorer = LexicalCoarseScorer::new(tokenizer);
        self.evaluate_batch_questions_coarse_to_fine_chunked_with_gating_and_scorer(
            tokenizer,
            state,
            questions,
            calib_config,
            coarse_config,
            chunk_size,
            gating_config,
            &scorer,
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
        let router = CoarseToFineRouter::new(self, tokenizer, mapping, calib_config, config)?;
        let mut results = IndexMap::with_capacity(questions.len());

        for (q_key, question) in questions {
            let res = router.route(state, &question.instructions, gating_config)?;
            results.insert(q_key.clone(), res);
        }

        Ok(results)
    }
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
fn prepare_coarse_questions<S: CoarseScorer>(
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

/// 粗密探索で縮小推論された各質問の結果を全候補空間へ復元し、複合確信度の再計算とゲーティングを適用する。
///
/// # 引数
/// - `ctx`: 状態テキスト・較正設定・ゲーティング設定を保持する評価コンテキスト。
/// - `questions`: 元の全候補を含む質問定義マップ。
/// - `original_keys_map`: 粗密スクリーニング対象となった質問の元候補キー配列マップ。
/// - `energies`: 事前算出された自由エネルギーマップ (`None` の場合は各 `Answer` 既存の `gating.energy` を引き継ぐ)。
/// - `answers`: 復元・ゲーティング反映先の回答マップ。
fn reconstruct_and_gate_batch_answers(
    ctx: &BatchEvalContext<'_>,
    questions: &IndexMap<String, Question>,
    original_keys_map: &IndexMap<String, Vec<String>>,
    energies: Option<&IndexMap<String, Option<f64>>>,
    answers: &mut IndexMap<String, Answer>,
) -> Result<()> {
    for (q_key, original_keys) in original_keys_map {
        if let Some(answer) = answers.get_mut(q_key)
            && answer.is_choice()
        {
            let energy = match energies {
                Some(map) => map.get(q_key).copied().flatten(),
                None => answer.gating.as_ref().and_then(|g| g.energy),
            };
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
    let mut answer = evaluate_question_with_buf(
        question,
        valid_logits,
        ctx.calib_config,
        probs_buf,
        Some(q_key),
    )?;
    apply_gating_to_answer_with_logits(
        &mut answer,
        Some(valid_logits),
        Some(ctx.state),
        Some(q_key),
        Some(question),
        ctx.gating_config,
        ctx.calib_config,
    );
    Ok(answer)
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
    use super::*;
    use sokuto_core::contract::model_spec::ModelInputDimensions;
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
            fn score_candidates(
                &self,
                _query: &str,
                candidates: &IndexMap<String, String>,
            ) -> Result<Vec<f32>> {
                Ok(candidates
                    .keys()
                    .enumerate()
                    .map(|(idx, _)| idx as f32)
                    .collect())
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
            top_m: 2,
            preserve_negative: false,
            negative_keys: vec![],
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
}
