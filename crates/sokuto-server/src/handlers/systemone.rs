//! # Jev 互換推論エンドポイントモジュール
//!
//! `POST /v1/systemone` に対するリクエスト検証、非同期ワーカースレッド隔離、
//! 粗密 2 段階探索連携、および型付き判定結果のシリアライズを提供する。

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue};
use axum::response::IntoResponse;
use serde_json::Value;
use sokuto_core::gating::evaluate_response_routing;
use sokuto_core::schema::{QuestionType, SystemOneRequest, SystemOneResponse, Usage};
use sokuto_runtime::engine::StateCacheManager;
use sokuto_runtime::hierarchical::CoarseToFineRouter;

use crate::error::{ErrorResponse, ServerError};
use crate::metrics::{
    record_confidence, record_gating_route, record_hierarchical_early_exit,
    record_hierarchical_soft_beam, record_http_duration, record_http_request,
    record_inference_metrics, record_question_type,
};
use crate::state::AppState;

/// TypeSafe AI Jev 互換の非自己回帰型判断推論エンドポイント。
///
/// 共通 State と複数質問群を受け取り、単一フォワードパスおよび
/// 決定プリミティブ数理に基づく型付き確率決定を一括返却する。
#[utoipa::path(
    post,
    path = "/v1/systemone",
    tag = "Inference",
    request_body = SystemOneRequest,
    responses(
        (status = 200, description = "推論および決定解決成功", body = SystemOneResponse),
        (status = 400, description = "スキーマまたはパラメータ検証違反", body = ErrorResponse),
        (status = 413, description = "質問数またはペイロード上限超過", body = ErrorResponse),
        (status = 500, description = "推論エンジンまたはサーバー内部障害", body = ErrorResponse)
    )
)]
pub async fn system_one_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(mut req): axum::Json<SystemOneRequest>,
) -> Result<impl IntoResponse, ServerError> {
    let start_time = Instant::now();

    req.validate()?;

    let client_hash_str = headers
        .get("x-state-hash")
        .or_else(|| headers.get("X-State-Hash"))
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .or_else(|| req.state_hash.clone());

    let original_questions = req.questions.clone();

    state.guardrail_pipeline.process(&mut req, None)?;

    let question_count = req.questions.len();

    for question in req.questions.values() {
        match question.question_type {
            QuestionType::Choice => record_question_type("choice"),
            QuestionType::Score => record_question_type("score"),
            QuestionType::Noul => record_question_type("noul"),
        }
    }

    let state_text = match &req.state {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    };

    let is_state_empty = state_text.trim().is_empty();

    let (state_key, _state_cache_hit_initial) = if is_state_empty {
        if let Some(ref h_str) = client_hash_str {
            let key = u64::from_str_radix(h_str.trim_start_matches("0x"), 16).map_err(|e| {
                ServerError::BadRequest(format!("無効な State ハッシュです: {e}。"))
            })?;
            let has_cache = state
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
        let has_cache = state
            .cache_manager
            .as_ref()
            .map(|cm| cm.contains_key(key))
            .unwrap_or(false);
        (Some(key), has_cache)
    };

    let engine = Arc::clone(&state.engine);
    let dual_engine = state.dual_engine.clone();
    let cache_manager = state.cache_manager.clone();
    let tokenizer = Arc::clone(&state.tokenizer);
    let calib_config = Arc::clone(&state.calib_config);
    let coarse_config = state.coarse_config.clone();
    let guardrail_pipeline = state.guardrail_pipeline.clone();
    let chunk_size = state.chunk_size;
    let questions = req.questions;
    let gating_config = req.gating.unwrap_or_else(|| state.gating_config.clone());
    let auto_hierarchical = req.auto_hierarchical.unwrap_or(state.auto_hierarchical);
    let hierarchical_mapping = state.hierarchical_mapping.clone();
    let hierarchical_config = state.hierarchical_config.clone();

    let inference_start = Instant::now();
    let (answers, prompt_tokens, routing_summary, route_records, cache_status_str) =
        tokio::task::spawn_blocking(move || {
            // 正規化後ペイロードの正味トークン数を正確に算出
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
                                    && m.keys().any(|key| mapping.fine_to_coarse.contains_key(key))
                            })
                            .unwrap_or(false)
                })
            } else {
                false
            };

            let (mut answers, cache_status_str) = if let (Some(de), Some(cm), Some(key)) =
                (&dual_engine, &cache_manager, state_key)
            {
                // デュアルセッションエンジンによる中間特徴量キャッシュ推論
                let (cached_state, status_str) = if let Some(cached) = cm.get(key) {
                    (cached, "HIT")
                } else {
                    let (s_ids, s_mask) = tokenizer.encode_state_tensor(&state_text)?;
                    let cached =
                        de.encode_and_cache_state(&state_text, &s_ids, &s_mask, [1, s_ids.len()])?;
                    (cached, "MISS")
                };

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

                (ans, status_str)
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
                                    && m.keys().any(|key| mapping.fine_to_coarse.contains_key(key))
                            })
                            .unwrap_or(false);

                    if is_hierarchical {
                        let res = router.route(
                            &state_text,
                            &question.instructions,
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

                (combined_answers, "BYPASS")
            } else {
                // 粗密 2 段階探索とマイクロバッチチャンキングを統合した通常推論を実行
                let ans = engine.evaluate_batch_questions_coarse_to_fine_chunked_with_gating(
                    &tokenizer,
                    &state_text,
                    &questions,
                    &calib_config,
                    &coarse_config,
                    chunk_size,
                    Some(&gating_config),
                )?;
                (ans, "BYPASS")
            };

            // 後処理ガードレール (二峰性分布検出による安全弁降格、および元ラベルキー復元) の適用
            guardrail_pipeline.post_process(&mut answers, &original_questions);

            // 確信度ゲーティング (3系統ルーティング) メトリクス収集および集約サマリー算出
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
                cache_status_str,
            ))
        })
        .await
        .map_err(|join_err| {
            ServerError::Internal(format!(
                "ブロッキング推論タスクが異常終了しました: {join_err}。"
            ))
        })??;

    let inference_duration = inference_start.elapsed().as_secs_f64();
    record_inference_metrics(inference_duration, question_count);

    // 確信度メトリクスの記録
    for answer in answers.values() {
        if let Some(conf) = answer.effective_confidence() {
            record_confidence(conf);
        }
    }

    // 5. ゲーティング判定ログおよびメトリクスの記録
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

    // 6. トークン消費量のメタデータ付与
    let usage = Usage::new(prompt_tokens);

    let mut response = SystemOneResponse::new(answers, usage);
    if let Some(summary) = routing_summary {
        response = response.with_routing(summary);
    }

    let total_duration = start_time.elapsed().as_secs_f64();
    record_http_duration("/v1/systemone", total_duration);
    record_http_request("/v1/systemone", 200);

    // 7. レスポンスヘッダーの設定 (透過的キャッシュプロトコル)
    let mut resp_headers = HeaderMap::new();
    if let Some(key) = state_key
        && let Ok(hv) = HeaderValue::from_str(&format!("{:016x}", key))
    {
        resp_headers.insert("X-State-Hash", hv);
    }
    let cache_status_val = HeaderValue::from_static(match cache_status_str {
        "HIT" => "HIT",
        "MISS" => "MISS",
        _ => "BYPASS",
    });
    resp_headers.insert("X-Cache-Status", cache_status_val.clone());
    resp_headers.insert("X-State-Cache", cache_status_val.clone());
    resp_headers.insert("X-Cache", cache_status_val);

    Ok((resp_headers, axum::Json(response)))
}
