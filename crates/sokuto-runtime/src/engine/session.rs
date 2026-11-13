//! # ONNX Runtime 推論セッション管理モジュール
//!
//! エクスポート済み ONNX モデルグラフのライフサイクル、入力テンソル整合性検証、
//! ゼロコピーテンソルバインド、およびセッションプールによる高並列フォワードパスを提供する。

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use indexmap::IndexMap;

use sokuto_core::contract::calibration::CalibrationConfig;
use sokuto_core::contract::model_spec::{
    ModelInputDimensions, OP_INDEX_PADDING, TENSOR_ATTENTION_MASK, TENSOR_INPUT_IDS, TENSOR_LOGITS,
    TENSOR_OP_INDICES,
};
use sokuto_core::decision::evaluate_question;
use sokuto_core::gating::GatingConfig;
use sokuto_core::schema::{Answer, Question};

use ort::session::Session;
use ort::session::builder::SessionBuilder;
use ort::value::TensorRef;

use crate::engine::coarse::{
    CoarseScorer, CoarseToFineConfig, LexicalCoarseScorer, filter_top_candidates,
    reconstruct_and_gate_answer,
};
use crate::engine::config::{ExecutionProvider, SessionConfig, auto_intra_threads};
use crate::engine::gating::{calculate_choice_energy_with_config, evaluate_and_gate_answer};
use crate::engine::provider::register_execution_providers;
use crate::error::{Result, RuntimeError};
use crate::hierarchical::{
    CoarseToFineRouter, HierarchicalMapping, HierarchicalResult, HierarchicalRouterConfig,
};
use crate::tokenizer::{JevTokenizer, TokenizedQuestion};

/// スレッドセーフな ONNX Runtime 推論エンジン。
///
/// 内部にセッションプールを保持し、複数スレッドからの並行推論要求に対して
/// アロケーションフリーかつ競合を最小化したフォワードパスを実行する。
/// `Arc<InferenceEngine>` として複数ワーカー間で安全に共有可能である。
pub struct InferenceEngine {
    /// セッションプール。並行度に応じた複数の ORT セッションを保持する。
    sessions: Vec<Mutex<Session>>,
    /// ラウンドロビン選択用インデックスカウンター。
    next_session_idx: AtomicUsize,
    /// 実際に有効化された Execution Provider。
    active_provider: ExecutionProvider,
    /// モデルが要求する入力テンソル名一覧。
    input_names: Vec<String>,
    /// モデルが出力するテンソル名一覧。
    output_names: Vec<String>,
}

impl InferenceEngine {
    /// ファイルパスからモデルを読み込み、推論エンジンを初期化する。
    ///
    /// # 引数
    /// - `model_path`: ONNX モデルファイルのパス。
    /// - `config`: セッション構築設定。
    pub fn new(model_path: impl AsRef<Path>, config: SessionConfig) -> Result<Self> {
        let model_path_ref = model_path.as_ref();
        if !model_path_ref.exists() {
            return Err(RuntimeError::ModelContractViolation(format!(
                "モデルファイルが存在しません: {}。",
                model_path_ref.display()
            )));
        }

        let pool_size = config.pool_size.max(1);

        tracing::info!(
            "推論セッションの初期化を開始します (パス: {}, プール数: {})...",
            model_path_ref.display(),
            pool_size
        );

        let (sessions, active_provider) =
            build_session_pool(&config, |mut b| b.commit_from_file(model_path_ref))?;

        Self::init_with_sessions(sessions, active_provider)
    }

    /// メモリ上のバイト列からモデルを読み込み、推論エンジンを初期化する。
    ///
    /// # 引数
    /// - `model_bytes`: ONNX モデルバイナリデータ。
    /// - `config`: セッション構築設定。
    pub fn from_bytes(model_bytes: &[u8], config: SessionConfig) -> Result<Self> {
        let pool_size = config.pool_size.max(1);

        tracing::info!(
            "バイト列からの推論セッション初期化を開始します (サイズ: {} bytes, プール数: {})...",
            model_bytes.len(),
            pool_size
        );

        let (sessions, active_provider) =
            build_session_pool(&config, |mut b| b.commit_from_memory(model_bytes))?;

        Self::init_with_sessions(sessions, active_provider)
    }

    /// 構築済みセッション群から契約を検証し、エンジンインスタンスを完成させる。
    fn init_with_sessions(
        sessions: Vec<Mutex<Session>>,
        active_provider: ExecutionProvider,
    ) -> Result<Self> {
        let first_session = sessions[0].lock().map_err(|e| {
            RuntimeError::ModelContractViolation(format!("セッションロック獲得失敗: {e}。"))
        })?;

        let input_names: Vec<String> = first_session
            .inputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let output_names: Vec<String> = first_session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        // 必須入力テンソル契約の検証
        if !input_names.iter().any(|name| name == TENSOR_INPUT_IDS) {
            return Err(RuntimeError::ModelContractViolation(format!(
                "モデルに必須入力テンソル '{TENSOR_INPUT_IDS}' が存在しません: 検出={input_names:?}。"
            )));
        }
        if !input_names.iter().any(|name| name == TENSOR_ATTENTION_MASK) {
            return Err(RuntimeError::ModelContractViolation(format!(
                "モデルに必須入力テンソル '{TENSOR_ATTENTION_MASK}' が存在しません: 検出={input_names:?}。"
            )));
        }
        if !input_names.iter().any(|name| name == TENSOR_OP_INDICES) {
            return Err(RuntimeError::ModelContractViolation(format!(
                "モデルに必須入力テンソル '{TENSOR_OP_INDICES}' が存在しません: 検出={input_names:?}。"
            )));
        }

        // 必須出力テンソル契約の検証
        if !output_names.iter().any(|name| name == TENSOR_LOGITS) {
            return Err(RuntimeError::ModelContractViolation(format!(
                "モデルに必須出力テンソル '{TENSOR_LOGITS}' が存在しません: 検出={output_names:?}。"
            )));
        }

        drop(first_session);

        tracing::info!(
            "モデル契約検証が完了しました: 入力={input_names:?}, 出力={output_names:?}, アクティブEP={}",
            active_provider.name()
        );

        Ok(Self {
            sessions,
            next_session_idx: AtomicUsize::new(0),
            active_provider,
            input_names,
            output_names,
        })
    }

    /// 有効化された Execution Provider を取得する。
    pub fn active_provider(&self) -> ExecutionProvider {
        self.active_provider
    }

    /// モデルの入力テンソル名一覧を取得する。
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// モデルの出力テンソル名一覧を取得する。
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// セッションプールサイズを取得する。
    pub fn pool_size(&self) -> usize {
        self.sessions.len()
    }

    /// 推論を呼び出さないユニットテスト用の空エンジンを生成する。
    #[cfg(test)]
    pub(crate) fn empty_for_test() -> Self {
        Self {
            sessions: Vec::new(),
            next_session_idx: AtomicUsize::new(0),
            active_provider: ExecutionProvider::CPU,
            input_names: Vec::new(),
            output_names: Vec::new(),
        }
    }

    /// 平坦化された生テンソルスライスを受け取り、推論を実行してロジットを返却する。
    ///
    /// # 引数
    /// - `input_ids`: トークン ID 列 (`[batch_size * seq_len]`)。
    /// - `attention_mask`: アテンションマスク列 (`[batch_size * seq_len]`)。
    /// - `op_indices`: オプションマーカー位置インデックス (`[batch_size * num_options]`)。
    /// - `dims`: 入力テンソルの次元情報 (`batch_size`, `sequence_length`, `num_options`)。
    ///
    /// # 戻り値
    /// 各バッチサンプルのロジット配列 (`Vec<Vec<f64>>`)。
    pub fn forward_raw(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        op_indices: &[i64],
        dims: ModelInputDimensions,
    ) -> Result<Vec<Vec<f64>>> {
        let total_logits = dims.batch_size * dims.num_options;
        let mut flat_logits = vec![0.0f64; total_logits];

        self.forward_raw_into(
            input_ids,
            attention_mask,
            op_indices,
            dims,
            &mut flat_logits,
        )?;

        let mut batched_logits = Vec::with_capacity(dims.batch_size);
        for chunk in flat_logits.chunks_exact(dims.num_options) {
            batched_logits.push(chunk.to_vec());
        }

        Ok(batched_logits)
    }

    /// 平坦化された生テンソルスライスを受け取り、推論結果を指定されたスライスに書き込む。
    ///
    /// 入力テンソルは借用ビュー (`TensorRef`) によりゼロコピーで ORT に渡され、
    /// 出力も事前確保済みスライスに直接展開されるため、ヒープ確保が発生しない。
    ///
    /// # 引数
    /// - `input_ids`: トークン ID 列 (`[batch_size * seq_len]`)。
    /// - `attention_mask`: アテンションマスク列 (`[batch_size * seq_len]`)。
    /// - `op_indices`: オプションマーカー位置インデックス (`[batch_size * num_options]`)。
    /// - `dims`: 入力テンソルの次元情報 (`batch_size`, `sequence_length`, `num_options`)。
    /// - `out_logits`: 結果ロジットの書き込み先バッファ (`[batch_size * num_options]`)。
    pub fn forward_raw_into(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        op_indices: &[i64],
        dims: ModelInputDimensions,
        out_logits: &mut [f64],
    ) -> Result<()> {
        let batch_size = dims.batch_size;
        let seq_len = dims.sequence_length;
        let num_options = dims.num_options;

        // 1. 次元整合性検証
        if input_ids.len() != batch_size * seq_len {
            return Err(RuntimeError::InvalidTensorData(format!(
                "input_ids 長さ ({}) が期待値 ({} * {} = {}) と一致しません。",
                input_ids.len(),
                batch_size,
                seq_len,
                batch_size * seq_len
            )));
        }
        if attention_mask.len() != batch_size * seq_len {
            return Err(RuntimeError::InvalidTensorData(format!(
                "attention_mask 長さ ({}) が期待値 ({} * {} = {}) と一致しません。",
                attention_mask.len(),
                batch_size,
                seq_len,
                batch_size * seq_len
            )));
        }
        if op_indices.len() != batch_size * num_options {
            return Err(RuntimeError::InvalidTensorData(format!(
                "op_indices 長さ ({}) が期待値 ({} * {} = {}) と一致しません。",
                op_indices.len(),
                batch_size,
                num_options,
                batch_size * num_options
            )));
        }
        if out_logits.len() != batch_size * num_options {
            return Err(RuntimeError::InvalidTensorData(format!(
                "出力バッファ長 ({}) が期待値 ({} * {} = {}) と一致しません。",
                out_logits.len(),
                batch_size,
                num_options,
                batch_size * num_options
            )));
        }

        // 2. op_indices の範囲外インデックスチェック (Gather 層クラッシュ・セグフォ防止、パディング -1 は許容)
        validate_op_indices(op_indices, seq_len)?;

        // 3. 入力 ORT TensorRef をゼロコピー借用ビューで構築 (.to_vec() 排除)
        let input_ids_tensor = TensorRef::from_array_view(([batch_size, seq_len], input_ids))
            .map_err(RuntimeError::Ort)?;
        let attention_mask_tensor =
            TensorRef::from_array_view(([batch_size, seq_len], attention_mask))
                .map_err(RuntimeError::Ort)?;
        let op_indices_tensor = TensorRef::from_array_view(([batch_size, num_options], op_indices))
            .map_err(RuntimeError::Ort)?;

        #[cfg(test)]
        if self.sessions.is_empty() {
            // テスト用空エンジンの場合、ゼロロジットでダミー応答
            out_logits.fill(0.0);
            return Ok(());
        }

        // 4. セッションプールから空きセッションを獲得 (非ブロッキング探索 -> ラウンドロビン待機)
        let mut session = acquire_pooled_session(&self.sessions, &self.next_session_idx, "")?;

        // 5. 推論実行
        let outputs = session
            .run(ort::inputs![
                TENSOR_INPUT_IDS => input_ids_tensor,
                TENSOR_ATTENTION_MASK => attention_mask_tensor,
                TENSOR_OP_INDICES => op_indices_tensor,
            ])
            .map_err(RuntimeError::Ort)?;

        // 6. 出力抽出と検証
        let logits_output = outputs.get(TENSOR_LOGITS).ok_or_else(|| {
            RuntimeError::ModelContractViolation(format!(
                "出力テンソル '{TENSOR_LOGITS}' が存在しません。"
            ))
        })?;

        let (shape, logits_f32) = logits_output
            .try_extract_tensor::<f32>()
            .map_err(RuntimeError::Ort)?;

        // 出力形状の契約検証 (スタック配列によるゼロアロケーション化)
        if shape.len() != 2 {
            return Err(RuntimeError::ModelContractViolation(format!(
                "出力テンソル '{TENSOR_LOGITS}' は 2 次元テンソル [batch, num_options] である必要があります: {shape:?}。"
            )));
        }
        let shape_stack = [shape[0] as usize, shape[1] as usize];
        dims.validate_output_shape(&shape_stack)?;

        // 7. 出力値の有限性 (NaN/Inf) 検証および f32 -> f64 キャスト
        for (i, &val) in logits_f32.iter().enumerate() {
            if !val.is_finite() {
                return Err(RuntimeError::InvalidTensorData(format!(
                    "logits[{i}] に非有限値 (NaN または Inf) が検出されました: {val}。"
                )));
            }
            out_logits[i] = val as f64;
        }

        Ok(())
    }

    /// 単一のトークナイズ済み質問を受け取り、ロジット配列を算出する。
    ///
    /// # 引数
    /// - `tokenized`: 高速トークナイザーから得られた `TokenizedQuestion`。
    ///
    /// # 戻り値
    /// 各選択肢に対応する決定ロジット (`Vec<f64>`)。
    pub fn forward_question(&self, tokenized: &TokenizedQuestion) -> Result<Vec<f64>> {
        let dims =
            ModelInputDimensions::new(1, tokenized.input_ids.len(), tokenized.op_indices.len())?;

        let logits = self.forward_raw(
            &tokenized.input_ids,
            &tokenized.attention_mask,
            &tokenized.op_indices,
            dims,
        )?;

        logits
            .into_iter()
            .next()
            .ok_or_else(|| RuntimeError::InvalidTensorData("推論出力が空です。".to_string()))
    }

    /// 任意の類似度スコアラーを指定して、粗密 2 段階探索 (Coarse-to-Fine) で単一質問を評価する。
    ///
    /// # 処理フロー
    /// 1. `question.question_type == QuestionType::Choice` かつ `criteria.len() > coarse_config.threshold` の場合にのみ 2 段階探索を発動。
    /// 2. ネガティブ候補保護 (Pinning) およびスコアラーによる Top-M スクリーニングを実施。
    /// 3. 絞り込み後の候補マップを用いて部分質問 `sub_question` を構築。
    /// 4. `tokenizer.encode_question` および `self.forward_question` を実行。
    /// 5. 較正設定 `calib_config` (Fine 候補数 $M$ に対応するバケット) を適用して `sub_answer` を導出。
    /// 6. `reconstruct_probabilities` により除外候補を確率 0.0 として全候補確率マップを再構成して返却。
    ///
    /// 発動条件を満たさない質問 (候補数 $K \le \text{threshold}$、または Score/Noul 型) は、
    /// 直接通常推論 (`encode_question` -> `forward_question` -> `evaluate_question`) へバイパスする。
    ///
    /// # 引数
    /// 任意の類似度スコアラーを指定して、粗密 2 段階探索 (Coarse-to-Fine) で単一質問を評価する。
    ///
    /// # 処理フロー
    /// 1. `question.question_type == QuestionType::Choice` かつ `criteria.len() > coarse_config.threshold` の場合にのみ 2 段階探索を発動。
    /// 2. ネガティブ候補保護 (Pinning) およびスコアラーによる Top-M スクリーニングを実施。
    /// 3. 絞り込み後の候補マップを用いて部分質問 `sub_question` を構築。
    /// 4. `tokenizer.encode_question` および `self.forward_question` を実行。
    /// 5. 較正設定 `calib_config` (Fine 候補数 $M$ に対応するバケット) を適用して `sub_answer` を導出。
    /// 6. `reconstruct_probabilities` により除外候補を確率 0.0 として全候補確率マップを再構成。
    /// 7. 較正設定の閾値またはリクエスト指定設定に基づき、透過的に確信度ゲーティングを適用して返却。
    ///
    /// 発動条件を満たさない質問 (候補数 $K \le \text{threshold}$、または Score/Noul 型) は、
    /// 直接通常推論 (`encode_question` -> `forward_question` -> `evaluate_question`) へバイパスする。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 文脈テキスト。
    /// - `question`: 質問定義。
    /// - `calib_config`: 較正温度設定。
    /// - `coarse_config`: 粗密探索設定。
    /// - `scorer`: 類似度スコアラー実装。
    pub fn evaluate_question_coarse_to_fine_with_scorer<S: CoarseScorer>(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        question: &Question,
        calib_config: &CalibrationConfig,
        coarse_config: &CoarseToFineConfig,
        scorer: &S,
    ) -> Result<Answer> {
        self.evaluate_question_coarse_to_fine_with_gating_and_scorer(
            tokenizer,
            state,
            question,
            calib_config,
            coarse_config,
            None,
            scorer,
        )
    }

    /// 任意の類似度スコアラーとゲーティング設定を指定して、粗密 2 段階探索で単一質問を評価する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 文脈テキスト。
    /// - `question`: 質問定義。
    /// - `calib_config`: 較正温度設定。
    /// - `coarse_config`: 粗密探索設定。
    /// - `gating_config`: 明示的なゲーティング設定 (任意、未指定時は較正設定から導出)。
    /// - `scorer`: 類似度スコアラー実装。
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_question_coarse_to_fine_with_gating_and_scorer<S: CoarseScorer>(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        question: &Question,
        calib_config: &CalibrationConfig,
        coarse_config: &CoarseToFineConfig,
        gating_config: Option<&GatingConfig>,
        scorer: &S,
    ) -> Result<Answer> {
        coarse_config.validate()?;

        // 発動条件の検査: Choice 型 かつ Criteria が Map 形式 かつ 候補数 > threshold
        let Some(map) = coarse_config.applicable_candidates(question) else {
            // バイパス: 通常の単一推論を実行
            let tokenized = tokenizer.encode_question(state, question)?;
            let logits = self.forward_question(&tokenized)?;
            return evaluate_and_gate_answer(
                question,
                &logits,
                Some(state),
                None,
                gating_config,
                calib_config,
            );
        };

        let original_keys: Vec<String> = map.keys().cloned().collect();
        let query = format!("State: {}\nInstructions: {}", state, question.instructions);

        // 1. Coarse 粗スクリーニング
        let filtered = filter_top_candidates(&query, map, coarse_config, scorer)?;

        // 2. 縮小質問の構築
        let sub_question = Question::new_choice(question.instructions.clone(), filtered.selected);

        // 3. Fine 精密推論
        let sub_tokenized = tokenizer.encode_question(state, &sub_question)?;
        let sub_logits = self.forward_question(&sub_tokenized)?;

        // 4. 縮小候補空間での決定数理解決 (M 候補の温度バケットが適用される)
        let mut answer = evaluate_question(&sub_question, &sub_logits, calib_config)
            .map_err(RuntimeError::Core)?;

        // 5. 全候補空間に対する確率マップの再構成・複合確信度の再計算およびゲーティング判定の適用
        let energy =
            calculate_choice_energy_with_config(question, &sub_logits, calib_config, gating_config);
        reconstruct_and_gate_answer(
            &mut answer,
            &original_keys,
            energy,
            Some(state),
            None,
            Some(question),
            gating_config,
            calib_config,
        )?;

        Ok(answer)
    }

    /// 既定の語彙スコアラー (`LexicalCoarseScorer`) を用いて、粗密 2 段階探索で単一質問を評価する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 文脈テキスト。
    /// - `question`: 質問定義。
    /// - `calib_config`: 較正温度設定。
    /// - `coarse_config`: 粗密探索設定。
    pub fn evaluate_question_coarse_to_fine(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        question: &Question,
        calib_config: &CalibrationConfig,
        coarse_config: &CoarseToFineConfig,
    ) -> Result<Answer> {
        let scorer = LexicalCoarseScorer::new(tokenizer);
        self.evaluate_question_coarse_to_fine_with_scorer(
            tokenizer,
            state,
            question,
            calib_config,
            coarse_config,
            &scorer,
        )
    }

    /// 既定の語彙スコアラーとゲーティング設定を指定して、粗密 2 段階探索で単一質問を評価する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 文脈テキスト。
    /// - `question`: 質問定義。
    /// - `calib_config`: 較正温度設定。
    /// - `coarse_config`: 粗密探索設定。
    /// - `gating_config`: 明示的なゲーティング設定 (任意)。
    pub fn evaluate_question_coarse_to_fine_with_gating(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        question: &Question,
        calib_config: &CalibrationConfig,
        coarse_config: &CoarseToFineConfig,
        gating_config: Option<&GatingConfig>,
    ) -> Result<Answer> {
        let scorer = LexicalCoarseScorer::new(tokenizer);
        self.evaluate_question_coarse_to_fine_with_gating_and_scorer(
            tokenizer,
            state,
            question,
            calib_config,
            coarse_config,
            gating_config,
            &scorer,
        )
    }

    /// 粗密二段階階層ルーティング (Coarse-to-Fine Hierarchical Routing) により単一質問を評価する。
    ///
    /// 大分類オントロジーと Soft-Beam、大分類エントロピー連動温度調整を活用し、
    /// 高基数分類 ($K > 20 \sim 30$) においても高精度かつ低遅延に決定を出力する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 文脈テキスト。
    /// - `instructions`: 質問指示文。
    /// - `mapping`: 大分類と細分類のオントロジー階層マッピング。
    /// - `calib_config`: 較正温度設定。
    /// - `config`: 階層ルーター設定 (Soft-Beam, Early-Exit 閾値など)。
    /// - `gating_config`: ゲーティング閾値設定 (任意)。
    ///
    /// # 戻り値
    /// 階層推論結果 (`HierarchicalResult`)。
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_question_hierarchical(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        instructions: &str,
        mapping: &HierarchicalMapping,
        calib_config: &CalibrationConfig,
        config: HierarchicalRouterConfig,
        gating_config: Option<&GatingConfig>,
    ) -> Result<HierarchicalResult> {
        let router = CoarseToFineRouter::new(self, tokenizer, mapping, calib_config, config)?;
        router.route(state, instructions, gating_config)
    }

    /// 許可された細分類候補制約を指定して単一質問に対する階層推論を実行する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 文脈テキスト。
    /// - `instructions`: 質問指示文。
    /// - `allowed_criteria`: 評価対象として制約する細分類候補マップ (任意)。
    /// - `mapping`: 大分類と細分類のオントロジー階層マッピング。
    /// - `calib_config`: 較正温度設定。
    /// - `config`: 階層ルーター設定。
    /// - `gating_config`: ゲーティング閾値設定 (任意)。
    ///
    /// # 戻り値
    /// 階層推論結果 (`HierarchicalResult`)。
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_question_hierarchical_with_allowed_candidates(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        instructions: &str,
        allowed_criteria: Option<&IndexMap<String, String>>,
        mapping: &HierarchicalMapping,
        calib_config: &CalibrationConfig,
        config: HierarchicalRouterConfig,
        gating_config: Option<&GatingConfig>,
    ) -> Result<HierarchicalResult> {
        let router = CoarseToFineRouter::new(self, tokenizer, mapping, calib_config, config)?;
        router.route_with_allowed_candidates(state, instructions, allowed_criteria, gating_config)
    }

    /// 単一の質問をトークナイズして生ロジット配列 (Softmax 正規化前) を算出する。
    ///
    /// 粗密推論における第 2 パス動的温度適用や、独自の後処理を行うために使用する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 文脈テキスト。
    /// - `question`: 質問仕様。
    ///
    /// # 戻り値
    /// 各選択肢の未正規化生ロジット列 (`Vec<f64>`)。
    pub fn forward_question_raw(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        question: &Question,
    ) -> Result<Vec<f64>> {
        let tokenized = tokenizer.encode_question(state, question)?;
        self.forward_question(&tokenized)
    }

    /// 単一の質問をトークナイズから決定数理評価まで一括して実行し、決定結果を返却する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `state`: 文脈テキスト。
    /// - `question`: 質問仕様。
    /// - `calib_config`: 較正温度設定。
    ///
    /// # 戻り値
    /// 確率較正済みの `Answer`。
    pub fn predict_single_question(
        &self,
        tokenizer: &JevTokenizer,
        state: &str,
        question: &Question,
        calib_config: &CalibrationConfig,
    ) -> Result<Answer> {
        let tokenized = tokenizer.encode_question(state, question)?;
        let logits = self.forward_question(&tokenized)?;
        evaluate_and_gate_answer(question, &logits, Some(state), None, None, calib_config)
    }
}

/// 指定された設定とプールサイズに基づいて `SessionBuilder` を構成する内部共通ヘルパー関数。
///
/// グラフ最適化レベル、メモリアリーナ、並列実行モード、物理コア数に応じた
/// `intra_threads` 自動設定、および最適化済みモデルのキャッシュ設定を適用する。
pub(crate) fn create_session_builder(
    config: &SessionConfig,
    pool_size: usize,
) -> Result<(SessionBuilder, ExecutionProvider)> {
    let mut builder = Session::builder().map_err(RuntimeError::Ort)?;

    builder = builder
        .with_optimization_level(config.optimization_level.to_ort_level())
        .map_err(|e| RuntimeError::OrtConfig(e.to_string()))?;

    builder = builder
        .with_memory_pattern(config.enable_mem_arena)
        .map_err(|e| RuntimeError::OrtConfig(e.to_string()))?;

    builder = builder
        .with_parallel_execution(config.parallel_execution)
        .map_err(|e| RuntimeError::OrtConfig(e.to_string()))?;

    // スレッド数の決定: 未指定時は物理コア数とセッションプール数から自動算出
    let intra = config
        .intra_threads
        .unwrap_or_else(|| auto_intra_threads(pool_size, 8));
    builder = builder
        .with_intra_threads(intra)
        .map_err(|e| RuntimeError::OrtConfig(e.to_string()))?;

    if let Some(inter) = config.inter_threads {
        builder = builder
            .with_inter_threads(inter)
            .map_err(|e| RuntimeError::OrtConfig(e.to_string()))?;
    }

    if let Some(ref opt_path) = config.optimized_model_path {
        if let Some(parent) = opt_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        builder = builder
            .with_optimized_model_path(opt_path)
            .map_err(|e| RuntimeError::OrtConfig(e.to_string()))?;
    }

    let provider = register_execution_providers(
        &mut builder,
        &config.preferred_providers,
        config.enable_mem_arena,
    )?;

    Ok((builder, provider))
}

/// 指定された設定から `Session` プールを構築する内部共通ヘルパー関数。
///
/// `config.pool_size` (最小値 1) 分の [`SessionBuilder`] を生成し、
/// `commit_fn` でモデルをコミットして `Mutex<Session>` のプールと有効な [`ExecutionProvider`] を返す。
/// スロット 0 の Execution Provider を基準値とし、2 番目以降 (`i >= 1`) で異なる
/// Execution Provider にフォールバックした場合は警告ログを出力する。
///
/// # 引数
/// - `config`: セッション構築設定。
/// - `commit_fn`: 構成済み [`SessionBuilder`] から [`Session`] を確定するクロージャ。
pub(crate) fn build_session_pool<F>(
    config: &SessionConfig,
    mut commit_fn: F,
) -> Result<(Vec<Mutex<Session>>, ExecutionProvider)>
where
    F: FnMut(SessionBuilder) -> ort::Result<Session>,
{
    build_session_pool_with(
        config.pool_size,
        |pool_size| create_session_builder(config, pool_size),
        |builder| commit_fn(builder).map_err(RuntimeError::Ort),
    )
}

/// ビルダー生成処理とセッション確定処理を注入してセッションプールを構築する内部補助関数。
///
/// [`build_session_pool`] の実体であり、ユニットテスト時にモックを使って
/// プールサイズ正規化や Execution Provider 不一致検知を検証できるように分離している。
fn build_session_pool_with<B, S, BF, CF>(
    pool_size: usize,
    mut build_fn: BF,
    mut commit_fn: CF,
) -> Result<(Vec<Mutex<S>>, ExecutionProvider)>
where
    BF: FnMut(usize) -> Result<(B, ExecutionProvider)>,
    CF: FnMut(B) -> Result<S>,
{
    let effective_pool_size = pool_size.max(1);
    let mut sessions = Vec::with_capacity(effective_pool_size);
    let mut active_provider = ExecutionProvider::CPU;

    for i in 0..effective_pool_size {
        let (builder, provider) = build_fn(effective_pool_size)?;
        if i == 0 {
            active_provider = provider;
        } else if provider != active_provider {
            tracing::warn!(
                "セッションプール内の Execution Provider が不一致です (スロット 0: {}, スロット {i}: {})。",
                active_provider.name(),
                provider.name()
            );
        }

        let session = commit_fn(builder)?;
        sessions.push(Mutex::new(session));
    }

    Ok((sessions, active_provider))
}

/// 同期セッションプールから利用可能なスロットを借り受ける内部共通ヘルパー関数。
///
/// `counter` によるラウンドロビン開始位置から非ブロッキング (`try_lock`) で空きスロットを優先探索し、
/// 全スロットが使用中の場合は開始位置のスロットをブロック待機 (`lock`) で獲得する。
///
/// 本実装は Tokio ランタイム非依存の同期関数(ブロッキング推論スレッド内)において、
/// ロック競合を最小化しながらゼロコストでスロットを貸し出す責務を持つ。
/// 非同期タスク側での許可証制御(`LowLatencyResourcePool`)とは異なり、
/// 各推論エンジンインスタンス(`InferenceEngine` / `DualSessionEngine`)の内部状態として自己完結する。
///
/// # 引数
/// - `sessions`: プール内のミューテックス保護セッション配列。
/// - `counter`: ラウンドロビン開始インデックスを供給するアトミックカウンター。
/// - `name`: エラーメッセージに付与するセッション種別名 (空文字列の場合は接頭辞なし)。
pub(crate) fn acquire_pooled_session<'a, T>(
    sessions: &'a [Mutex<T>],
    counter: &AtomicUsize,
    name: &str,
) -> Result<MutexGuard<'a, T>> {
    let build_err = |msg: String| {
        if name.is_empty() {
            RuntimeError::ModelContractViolation(msg)
        } else {
            RuntimeError::ModelContractViolation(format!("{name} {msg}"))
        }
    };

    let pool_len = sessions.len();
    if pool_len == 0 {
        return Err(build_err("セッションプールが空です。".to_string()));
    }

    let start_idx = counter.fetch_add(1, Ordering::Relaxed) % pool_len;
    for offset in 0..pool_len {
        let idx = (start_idx + offset) % pool_len;
        if let Ok(guard) = sessions[idx].try_lock() {
            return Ok(guard);
        }
    }

    sessions[start_idx]
        .lock()
        .map_err(|e| build_err(format!("セッションロック獲得失敗: {e}。")))
}

/// 候補インデックス列 (`op_indices`) の範囲外アクセスを検証する共通関数。
///
/// パディング値 (`OP_INDEX_PADDING = -1`) は許容し、それ以外のインデックスが系列長 `0..seq_len` の範囲内にあることを確認する。
pub fn validate_op_indices(op_indices: &[i64], seq_len: usize) -> Result<()> {
    for (i, &idx) in op_indices.iter().enumerate() {
        if idx != OP_INDEX_PADDING && (idx < 0 || idx >= seq_len as i64) {
            return Err(RuntimeError::InvalidTensorData(format!(
                "op_indices[{i}] = {idx} が有効範囲 (-1 または 0..{seq_len}) の範囲外です。"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_op_indices_semantics() {
        let seq_len = 32;

        // 有効なインデックス列 (-1 パディングおよび 0..32 の範囲内)
        let valid_indices = vec![0, 5, 15, 31, OP_INDEX_PADDING, OP_INDEX_PADDING];
        assert!(validate_op_indices(&valid_indices, seq_len).is_ok());

        // -2 以下の負値は不正
        let invalid_neg = vec![0, -2, 10];
        assert!(validate_op_indices(&invalid_neg, seq_len).is_err());

        // 系列長以上のインデックスは不正
        let invalid_oob = vec![0, 5, 32];
        assert!(validate_op_indices(&invalid_oob, seq_len).is_err());
    }

    #[test]
    fn test_acquire_pooled_session_round_robin_and_busy_skip() {
        let pool = vec![
            Mutex::new(10usize),
            Mutex::new(20usize),
            Mutex::new(30usize),
        ];
        let counter = AtomicUsize::new(0);

        // 0 番目がロック中の場合でも次の空きスロット 1 番目を非ブロッキング取得できること
        let held_zero = pool[0].lock().unwrap();
        let guard1 = acquire_pooled_session(&pool, &counter, "Query").unwrap();
        assert_eq!(*guard1, 20);
        drop(guard1);

        // カウンタ進行に従いスロット 1 -> 2 とラウンドロビン取得でき、
        // 先頭 0 番目は依然ロック中なのでスキップしてスロット 2 -> 1 と選ばれること
        let guard2 = acquire_pooled_session(&pool, &counter, "Query").unwrap();
        assert_eq!(*guard2, 20);
        let guard3 = acquire_pooled_session(&pool, &counter, "Query").unwrap();
        assert_eq!(*guard3, 30);

        drop(guard3);
        drop(guard2);
        drop(held_zero);

        // 全解放後に空プールエラーハンドリングも確認
        let empty: Vec<Mutex<usize>> = Vec::new();
        let err = acquire_pooled_session(&empty, &counter, "State").unwrap_err();
        assert!(err.to_string().contains("State セッションプールが空です。"));
    }

    #[test]
    fn test_build_session_pool_with_clamps_pool_size_and_tracks_provider() {
        // pool_size = 0 指定時は最小値 1 へ正規化され、先頭スロットの EP が採用されること。
        let mut observed_pool_sizes = Vec::new();
        let (sessions, provider) = build_session_pool_with(
            0,
            |effective_size| {
                observed_pool_sizes.push(effective_size);
                Ok(("builder", ExecutionProvider::CoreML))
            },
            |b| Ok(format!("{b}-session")),
        )
        .unwrap();

        assert_eq!(observed_pool_sizes, vec![1]);
        assert_eq!(sessions.len(), 1);
        assert_eq!(provider, ExecutionProvider::CoreML);
        assert_eq!(*sessions[0].lock().unwrap(), "builder-session");
    }

    #[test]
    fn test_build_session_pool_with_keeps_first_provider_on_fallback_and_propagates_errors() {
        // スロット 0 の EP が基準値として維持され、後続スロットのフォールバック時も全スロットが構築されること。
        let mut call_idx = 0usize;
        let (sessions, provider) = build_session_pool_with(
            3,
            |_| {
                let ep = if call_idx == 0 {
                    ExecutionProvider::CUDA
                } else {
                    ExecutionProvider::CPU
                };
                call_idx += 1;
                Ok((call_idx, ep))
            },
            |idx| Ok(idx * 10),
        )
        .unwrap();

        assert_eq!(sessions.len(), 3);
        assert_eq!(provider, ExecutionProvider::CUDA);
        assert_eq!(*sessions[0].lock().unwrap(), 10);
        assert_eq!(*sessions[1].lock().unwrap(), 20);
        assert_eq!(*sessions[2].lock().unwrap(), 30);

        // commit_fn が失敗した場合は即座にエラーが伝播すること。
        let mut commit_count = 0usize;
        let err = build_session_pool_with(
            3,
            |_| Ok(((), ExecutionProvider::CPU)),
            |()| {
                commit_count += 1;
                if commit_count == 2 {
                    Err(RuntimeError::ModelContractViolation(
                        "commit failed".to_string(),
                    ))
                } else {
                    Ok(())
                }
            },
        )
        .unwrap_err();
        assert_eq!(commit_count, 2);
        assert!(err.to_string().contains("commit failed"));
    }
}
