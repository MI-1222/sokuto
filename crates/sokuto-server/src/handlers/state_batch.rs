//! # 一括バッチ質問エンドポイントモジュール
//!
//! 単一の長文 `State` に対して複数の異種質問 (`Choice`/`Score`/`Noul`) を一括投入し、
//! `State` エンコードを 1 度きりに抑えてミリ秒級で並列判定結果を返却する。

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue};
use axum::response::IntoResponse;
use serde_json::Value;
use sokuto_core::gating::evaluate_response_routing;
use sokuto_core::schema::{
    QuestionType, StateBatchRequest, StateBatchResponse, SystemOneRequest, Usage,
};
use sokuto_runtime::engine::StateCacheManager;

use crate::error::{ErrorResponse, ServerError};
use crate::metrics::{
    record_confidence, record_gating_route, record_http_duration, record_http_request,
    record_inference_metrics, record_question_type,
};
use crate::state::AppState;

/// 単一 `State` に対する一括バッチ質問推論エンドポイント。
///
/// 共通 `State` を 1 回のみエンコード (またはキャッシュ取得) し、
/// 全質問をプレフィックス共有バッチ化して一括評価する。
#[utoipa::path(
    post,
    path = "/v1/systemone/state-batch",
    tag = "Inference",
    request_body = StateBatchRequest,
    responses(
        (status = 200, description = "バッチ推論および決定解決成功", body = StateBatchResponse),
        (status = 400, description = "スキーマまたはパラメータ検証違反", body = ErrorResponse),
        (status = 413, description = "質問数またはペイロード上限超過", body = ErrorResponse),
        (status = 500, description = "推論エンジンまたはサーバー内部障害", body = ErrorResponse)
    )
)]
pub async fn state_batch_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    axum::Json(mut req): axum::Json<StateBatchRequest>,
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

    let mut dummy_s1_req = SystemOneRequest {
        model: req.model.clone(),
        state: req.state.clone(),
        questions: req.questions.clone(),
        gating: req.gating.clone(),
        auto_hierarchical: req.auto_hierarchical,
        state_hash: req.state_hash.clone(),
    };
    state.guardrail_pipeline.process(&mut dummy_s1_req, None)?;
    req.questions = dummy_s1_req.questions;

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
            let key =
                u64::from_str_radix(h_str.trim().trim_start_matches("0x"), 16).map_err(|e| {
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

    let inference_start = Instant::now();
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

            let (mut answers, cache_hit, status_str) = if let (Some(de), Some(cm), Some(key)) =
                (&dual_engine, &cache_manager, state_key)
            {
                let (cached_state, hit, status) = if let Some(cached) = cm.get(key) {
                    (cached, true, "HIT")
                } else {
                    let (s_ids, s_mask) = tokenizer.encode_state_tensor(&state_text)?;
                    let cached =
                        de.encode_and_cache_state(&state_text, &s_ids, &s_mask, [1, s_ids.len()])?;
                    (cached, false, "MISS")
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

                (ans, hit, status)
            } else {
                // 単一モデル運用環境でのフォールバック
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

            guardrail_pipeline.post_process(&mut answers, &original_questions);

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
                "バッチ推論ブロッキングタスクが異常終了しました: {join_err}。"
            ))
        })??;

    let inference_duration = inference_start.elapsed().as_secs_f64();
    record_inference_metrics(inference_duration, question_count);

    for answer in answers.values() {
        if let Some(conf) = answer.effective_confidence() {
            record_confidence(conf);
        }
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
    let hex_key_opt = state_key.map(|k| format!("{:016x}", k));

    let mut response = StateBatchResponse::new(answers, usage, cache_hit);
    if let Some(summary) = routing_summary {
        response = response.with_routing(summary);
    }
    if let Some(ref hex_key) = hex_key_opt {
        response = response.with_state_hash(hex_key);
    }

    let total_duration = start_time.elapsed().as_secs_f64();
    record_http_duration("/v1/systemone/state-batch", total_duration);
    record_http_request("/v1/systemone/state-batch", 200);

    let mut resp_headers = HeaderMap::new();
    if let Some(ref hex_key) = hex_key_opt
        && let Ok(hv) = HeaderValue::from_str(hex_key)
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
