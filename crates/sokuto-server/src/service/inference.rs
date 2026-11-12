//! # 共通推論サービスモジュール
//!
//! 単一リクエストおよび一括バッチリクエスト双方から利用される
//! コア推論ディスパッチ、粗密 2 段階ルーティング、キャッシュ制御、
//! ガードレール適用、およびメトリクス収集を集約する。

use std::sync::Arc;
use std::time::Instant;

use axum::http::{HeaderMap, HeaderValue};
use indexmap::IndexMap;
use serde_json::Value;
use sokuto_core::gating::{SystemRoutingSummary, evaluate_response_routing};
use sokuto_core::schema::{Answer, QuestionType, SystemOneRequest, Usage};
use sokuto_runtime::engine::StateCacheManager;
use sokuto_runtime::hierarchical::CoarseToFineRouter;

use crate::error::ServerError;
use crate::metrics::{
    record_confidence, record_gating_route, record_hierarchical_early_exit,
    record_hierarchical_soft_beam, record_inference_metrics, record_question_type,
};
use crate::state::AppState;

/// コア推論パイプラインの実行結果サマリー。
#[derive(Debug, Clone)]
pub struct InferenceOutput {
    /// 各質問に対応する判定結果マップ。
    pub answers: IndexMap<String, Answer>,
    /// トークン消費統計。
    pub usage: Usage,
    /// 確信度ゲーティングサマリー (有効時のみ)。
    pub routing_summary: Option<SystemRoutingSummary>,
    /// State キャッシュステータス文字列 ("HIT", "MISS", "BYPASS")。
    pub cache_status: &'static str,
    /// キャッシュがヒットしたかどうかのフラグ。
    pub cache_hit: bool,
    /// 対象 State のキャッシュキー (任意)。
    pub state_key: Option<u64>,
}

impl InferenceOutput {
    /// 共通レスポンスヘッダー群 (キャッシュ状態および State ハッシュ) を生成する。
    pub fn to_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if let Some(key) = self.state_key
            && let Ok(hv) = HeaderValue::from_str(&format!("{:016x}", key))
        {
            headers.insert("X-State-Hash", hv);
        }
        let cache_status_val = HeaderValue::from_static(self.cache_status);
        headers.insert("X-Cache-Status", cache_status_val.clone());
        headers.insert("X-State-Cache", cache_status_val.clone());
        headers.insert("X-Cache", cache_status_val);
        headers
    }
}

/// 共通推論ディスパッチおよび実行制御サービス。
#[derive(Clone)]
pub struct InferenceService {
    state: Arc<AppState>,
}

impl InferenceService {
    /// 新規推論サービスを生成する。
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }

    /// 推論リクエストを実行し、結果およびキャッシュ・ゲーティングメタデータを返却する。
    ///
    /// # 処理フロー
    /// 1. ガードレールパイプラインによる前処理 (無毒化、特殊トークン除去等)。
    /// 2. 質問プリミティブ種別ごとのメトリクス集計。
    /// 3. State テキスト展開とキャッシュキー (ハッシュ) の検証および解決。
    /// 4. `tokio::task::spawn_blocking` による非同期ワーカースレッド分離。
    /// 5. デュアルエンジン中間特徴量キャッシュまたは通常粗密エンジンの透過的ディスパッチ。
    /// 6. 後処理ガードレール (二峰性分布補正、キー復元等) の適用。
    /// 7. 確信度ゲーティング判定と集約サマリーの算出。
    /// 8. 推論遅延、確信度分布、ゲーティングルーティングメトリクスの一括記録。
    pub async fn execute(
        &self,
        mut req: SystemOneRequest,
        client_hash_str: Option<String>,
    ) -> Result<InferenceOutput, ServerError> {
        let original_questions = req.questions.clone();

        // 1. ガードレール適用前に空 State オフロード判定を実施
        let is_raw_state_empty = match &req.state {
            Value::Null => true,
            Value::String(s) => s.trim().is_empty(),
            _ => false,
        };

        if is_raw_state_empty {
            // 空 State 時はメタデータ追記をバイパスし、質問側のガードレール検証のみ実行
            self.state
                .guardrail_pipeline
                .process_questions_only(&mut req, None)?;
        } else {
            self.state.guardrail_pipeline.process(&mut req, None)?;
        }

        let question_count = req.questions.len();

        // 2. 質問タイプ別メトリクスの記録
        for question in req.questions.values() {
            match question.question_type {
                QuestionType::Choice => record_question_type("choice"),
                QuestionType::Score => record_question_type("score"),
                QuestionType::Noul => record_question_type("noul"),
            }
        }

        // 3. State 本文の展開とキャッシュ検証
        let state_text = match &req.state {
            Value::String(s) => s.clone(),
            Value::Null => String::new(),
            other => other.to_string(),
        };

        let is_state_empty = state_text.trim().is_empty();

        let (state_key, _state_cache_hit_initial) = if is_state_empty {
            if let Some(ref h_str) = client_hash_str {
                let key = u64::from_str_radix(h_str.trim().trim_start_matches("0x"), 16).map_err(
                    |e| ServerError::BadRequest(format!("無効な State ハッシュです: {e}。")),
                )?;
                let has_cache = self
                    .state
                    .cache_manager
                    .as_ref()
                    .map(|cm| cm.contains_key(key))
                    .unwrap_or(false);
                if !has_cache {
                    return Err(ServerError::StateCacheMiss(
                        "指定された State ハッシュのキャッシュが見つかりません。State 本文を含めて再送してください。".to_string(),
                    ));
                }
                (Some(key), true)
            } else {
                return Err(ServerError::BadRequest(
                    "State 本文が空で、有効な State ハッシュも指定されていません。".to_string(),
                ));
            }
        } else {
            let key = StateCacheManager::compute_key(&state_text);
            let has_cache = self
                .state
                .cache_manager
                .as_ref()
                .map(|cm| cm.contains_key(key))
                .unwrap_or(false);
            (Some(key), has_cache)
        };

        let engine = Arc::clone(&self.state.engine);
        let dual_engine = self.state.dual_engine.clone();
        let cache_manager = self.state.cache_manager.clone();
        let tokenizer = Arc::clone(&self.state.tokenizer);
        let calib_config = Arc::clone(&self.state.calib_config);
        let coarse_config = self.state.coarse_config.clone();
        let guardrail_pipeline = self.state.guardrail_pipeline.clone();
        let chunk_size = self.state.chunk_size;
        let questions = req.questions;
        let gating_config = match req.gating.as_ref() {
            Some(patch) => self.state.gating_config.apply_patch(patch)?,
            None => self.state.gating_config.clone(),
        };
        let auto_hierarchical = req
            .auto_hierarchical
            .unwrap_or(self.state.auto_hierarchical);
        let hierarchical_mapping = self.state.hierarchical_mapping.clone();
        let hierarchical_config = self.state.hierarchical_config.clone();

        let inference_start = Instant::now();

        // 4. ブロッキング推論タスクの隔離実行
        let (answers, prompt_tokens, routing_summary, route_records, cache_hit, cache_status_str) =
            tokio::task::spawn_blocking(move || {
                let state_tokens = if !is_state_empty {
                    tokenizer
                        .encode_state(&state_text)
                        .map(|toks| toks.len())
                        .unwrap_or(0)
                } else {
                    0
                };
                let questions_tokens: usize = questions
                    .values()
                    .map(|q| {
                        sokuto_runtime::tokenizer::prompt::format_prompt(q)
                            .and_then(|f| {
                                tokenizer
                                    .inner()
                                    .encode(f.suffix.as_str(), false)
                                    .map_err(|e| {
                                        sokuto_runtime::error::RuntimeError::TokenizerEncodeError(
                                            e.to_string(),
                                        )
                                    })
                            })
                            .map(|toks| toks.len())
                            .unwrap_or(0)
                    })
                    .sum();
                let total_prompt_tokens = state_tokens + questions_tokens;

                // 透過的粗密ルーティング対象質問の有無を判定 (マッピング適合性および候補数閾値ガード)
                let has_hierarchical_target = if let Some(ref mapping) = hierarchical_mapping {
                    questions.values().any(|q| {
                        q.question_type == QuestionType::Choice
                            && q.criteria
                                .as_ref()
                                .and_then(|c| c.as_map())
                                .map(|m| {
                                    let k = m.len();
                                    (auto_hierarchical && k > 8 || k > 16)
                                        && m.keys()
                                            .any(|key| mapping.fine_to_coarse.contains_key(key))
                                })
                                .unwrap_or(false)
                    })
                } else {
                    false
                };

                let (mut answers, cache_hit, status_str) = if let (Some(de), Some(cm), Some(key)) =
                    (&dual_engine, &cache_manager, state_key)
                {
                    let (cached_state, hit, status) = if let Some(cached) = cm.get(key) {
                        (cached, true, "HIT")
                    } else {
                        let (s_ids, s_mask) = tokenizer.encode_state_tensor(&state_text)?;
                        let cached = de.encode_and_cache_state(
                            &state_text,
                            &s_ids,
                            &s_mask,
                            [1, s_ids.len()],
                        )?;
                        (cached, false, "MISS")
                    };

                    if has_hierarchical_target {
                        let mapping = hierarchical_mapping
                            .as_ref()
                            .expect("マッピングが存在すること。");
                        let router = CoarseToFineRouter::new_dual(
                            de,
                            &cached_state,
                            &tokenizer,
                            mapping.as_ref(),
                            &calib_config,
                            hierarchical_config,
                        )?;

                        let mut combined_answers =
                            indexmap::IndexMap::with_capacity(questions.len());
                        let mut fallback_questions = indexmap::IndexMap::new();

                        let effective_state = if is_state_empty {
                            cached_state.state_text.as_deref().unwrap_or(&state_text)
                        } else {
                            &state_text
                        };

                        for (qid, question) in &questions {
                            let is_hierarchical = question.question_type == QuestionType::Choice
                                && question
                                    .criteria
                                    .as_ref()
                                    .and_then(|c| c.as_map())
                                    .map(|m| {
                                        let k = m.len();
                                        (auto_hierarchical && k > 8 || k > 16)
                                            && m.keys()
                                                .any(|key| mapping.fine_to_coarse.contains_key(key))
                                    })
                                    .unwrap_or(false);

                            if is_hierarchical {
                                let allowed_criteria =
                                    question.criteria.as_ref().and_then(|c| c.as_map());
                                let res = router.route_with_allowed_candidates(
                                    effective_state,
                                    &question.instructions,
                                    allowed_criteria,
                                    Some(&gating_config),
                                )?;
                                if res.trace().soft_beam_triggered {
                                    record_hierarchical_soft_beam();
                                }
                                if res.trace().early_exit_triggered {
                                    record_hierarchical_early_exit();
                                }
                                combined_answers.insert(qid.clone(), res.into_answer());
                            } else {
                                fallback_questions.insert(qid.clone(), question.clone());
                            }
                        }

                        if !fallback_questions.is_empty() {
                            let fallback_answers = de.evaluate_batch_questions_with_cache(
                                &tokenizer,
                                &cached_state,
                                if is_state_empty {
                                    None
                                } else {
                                    Some(&state_text)
                                },
                                &fallback_questions,
                                &calib_config,
                                Some(&gating_config),
                                chunk_size,
                            )?;
                            for (qid, answer) in fallback_answers {
                                combined_answers.insert(qid, answer);
                            }
                        }

                        (combined_answers, hit, status)
                    } else {
                        let ans = de.evaluate_batch_questions_with_cache(
                            &tokenizer,
                            &cached_state,
                            if is_state_empty {
                                None
                            } else {
                                Some(&state_text)
                            },
                            &questions,
                            &calib_config,
                            Some(&gating_config),
                            chunk_size,
                        )?;

                        (ans, hit, status)
                    }
                } else if has_hierarchical_target {
                    let mapping = hierarchical_mapping
                        .as_ref()
                        .expect("マッピングが存在すること。");
                    let router = CoarseToFineRouter::new(
                        &engine,
                        &tokenizer,
                        mapping.as_ref(),
                        &calib_config,
                        hierarchical_config,
                    )?;

                    let mut combined_answers = indexmap::IndexMap::with_capacity(questions.len());
                    let mut fallback_questions = indexmap::IndexMap::new();

                    for (qid, question) in &questions {
                        let is_hierarchical = question.question_type == QuestionType::Choice
                            && question
                                .criteria
                                .as_ref()
                                .and_then(|c| c.as_map())
                                .map(|m| {
                                    let k = m.len();
                                    (auto_hierarchical && k > 8 || k > 16)
                                        && m.keys()
                                            .any(|key| mapping.fine_to_coarse.contains_key(key))
                                })
                                .unwrap_or(false);

                        if is_hierarchical {
                            let allowed_criteria =
                                question.criteria.as_ref().and_then(|c| c.as_map());
                            let res = router.route_with_allowed_candidates(
                                &state_text,
                                &question.instructions,
                                allowed_criteria,
                                Some(&gating_config),
                            )?;
                            if res.trace().soft_beam_triggered {
                                record_hierarchical_soft_beam();
                            }
                            if res.trace().early_exit_triggered {
                                record_hierarchical_early_exit();
                            }
                            combined_answers.insert(qid.clone(), res.into_answer());
                        } else {
                            fallback_questions.insert(qid.clone(), question.clone());
                        }
                    }

                    if !fallback_questions.is_empty() {
                        let fallback_answers = engine
                            .evaluate_batch_questions_coarse_to_fine_chunked_with_gating(
                                &tokenizer,
                                &state_text,
                                &fallback_questions,
                                &calib_config,
                                &coarse_config,
                                chunk_size,
                                Some(&gating_config),
                            )?;
                        for (qid, answer) in fallback_answers {
                            combined_answers.insert(qid, answer);
                        }
                    }

                    (combined_answers, false, "BYPASS")
                } else {
                    let ans = engine.evaluate_batch_questions_coarse_to_fine_chunked_with_gating(
                        &tokenizer,
                        &state_text,
                        &questions,
                        &calib_config,
                        &coarse_config,
                        chunk_size,
                        Some(&gating_config),
                    )?;
                    (ans, false, "BYPASS")
                };

                // 6. 後処理ガードレール
                guardrail_pipeline.post_process(&mut answers, &original_questions);

                // 7. 確信度ゲーティングサマリーおよびルーティングレコードの収集
                let (routing_summary, route_records) = if gating_config.enabled {
                    let mut records = Vec::with_capacity(answers.len());
                    for (qid, answer) in answers.iter() {
                        let q_def = questions.get(qid);
                        let q_type_str = q_def
                            .map(|q| match q.question_type {
                                QuestionType::Choice => "choice",
                                QuestionType::Score => "score",
                                QuestionType::Noul => "noul",
                            })
                            .unwrap_or("unknown");

                        if let Some(ref meta) = answer.gating {
                            records.push((
                                meta.route.as_str(),
                                q_type_str,
                                qid.clone(),
                                meta.route,
                                meta.confidence,
                                meta.reason.clone(),
                            ));
                        }
                    }
                    let summary = evaluate_response_routing(&answers);
                    (Some(summary), records)
                } else {
                    (None, Vec::new())
                };

                Ok::<_, ServerError>((
                    answers,
                    total_prompt_tokens,
                    routing_summary,
                    route_records,
                    cache_hit,
                    status_str,
                ))
            })
            .await
            .map_err(|join_err| {
                ServerError::Internal(format!(
                    "ブロッキング推論タスクが異常終了しました: {join_err}。"
                ))
            })??;

        let inference_duration = inference_start.elapsed().as_secs_f64();

        // 8. メトリクスの一括記録
        record_inference_metrics(inference_duration, question_count);

        for answer in answers.values() {
            record_confidence(answer.effective_confidence());
        }

        for (route_str, q_type_str, qid, route, confidence, reason) in route_records {
            record_gating_route(route_str, q_type_str);
            tracing::info!(
                question_id = %qid,
                route = %route,
                confidence = confidence,
                reason = %reason,
                "確信度ゲーティング判定を適用しました。"
            );
        }

        let usage = Usage::new(prompt_tokens);

        Ok(InferenceOutput {
            answers,
            usage,
            routing_summary,
            cache_status: cache_status_str,
            cache_hit,
            state_key,
        })
    }
}
