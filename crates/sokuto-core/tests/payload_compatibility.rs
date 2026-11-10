//! # Jev 公式互換ペイロードのシリアライズ/デシリアライズ双方向結合テスト
//!
//! TypeSafe AI の `POST /v1/systemone` リクエスト・レスポンススキーマとの
//! 完全な互換性を検証する。

use indexmap::IndexMap;
use serde_json::json;
use sokuto_core::{Answer, CoreError, QuestionType, SystemOneRequest, SystemOneResponse, Usage};

/// 典型的な Choice, Score, Noul が混在するリクエスト JSON のデシリアライズテスト。
#[test]
fn test_deserialize_mixed_request() {
    let payload = json!({
        "model": "sokuto-large",
        "state": "ユーザーから注文 #12345 の配送遅延について苦情が届いた。",
        "questions": {
            "intent": {
                "type": "choice",
                "instructions": "問い合わせの主な意図を分類せよ。",
                "criteria": {
                    "shipping_inquiry": "配送状況の確認",
                    "complaint": "対応への苦情・クレーム",
                    "cancellation": "注文のキャンセル要望"
                }
            },
            "urgency": {
                "type": "score",
                "instructions": "対応の緊急度を 3 段階で評価せよ。",
                "criteria": [
                    "通常対応(24時間以内)",
                    "優先対応(本日中)",
                    "至急対応(即時)"
                ]
            },
            "is_angry": {
                "type": "noul",
                "instructions": "顧客は強い怒りを示しているか。"
            }
        }
    });

    let json_str = serde_json::to_string(&payload).unwrap();
    let request: SystemOneRequest = serde_json::from_str(&json_str).unwrap();

    assert_eq!(request.model.as_deref(), Some("sokuto-large"));
    assert_eq!(request.questions.len(), 3);

    // バリデーションが正常に通ることを検証する。
    assert!(request.validate().is_ok());

    // 各質問のプロパティを検証する。
    let q_intent = request.questions.get("intent").unwrap();
    assert_eq!(q_intent.question_type, QuestionType::Choice);
    assert_eq!(q_intent.criteria.as_ref().unwrap().len(), 3);

    let q_urgency = request.questions.get("urgency").unwrap();
    assert_eq!(q_urgency.question_type, QuestionType::Score);
    assert_eq!(q_urgency.criteria.as_ref().unwrap().len(), 3);

    let q_angry = request.questions.get("is_angry").unwrap();
    assert_eq!(q_angry.question_type, QuestionType::Noul);
    assert!(q_angry.criteria.is_none() || q_angry.criteria.as_ref().unwrap().is_none());
}

/// 構造化オブジェクトや配列形式の State を受け入れるテスト。
#[test]
fn test_structured_state_deserialization() {
    let payload = json!({
        "state": {
            "user_id": 9999,
            "cart_total": 45000,
            "items": ["keyboard", "monitor"]
        },
        "questions": {
            "high_value": {
                "type": "noul",
                "instructions": "高額決済に該当するか。"
            }
        }
    });

    let request: SystemOneRequest = serde_json::from_value(payload).unwrap();
    assert!(request.validate().is_ok());
    assert!(request.state.is_object());
}

/// Jev 互換レスポンスのシリアライズおよびデシリアライズ双方向(Round-trip)テスト。
#[test]
fn test_response_round_trip() {
    let mut answers = IndexMap::new();

    let mut choice_probs = IndexMap::new();
    choice_probs.insert("refund".to_string(), 0.88);
    choice_probs.insert("other".to_string(), 0.12);
    answers.insert(
        "intent".to_string(),
        Answer::choice("refund", choice_probs, 0.91),
    );

    let mut score_probs = IndexMap::new();
    score_probs.insert("level_0".to_string(), 0.05);
    score_probs.insert("level_1".to_string(), 0.25);
    score_probs.insert("level_2".to_string(), 0.70);
    answers.insert(
        "severity".to_string(),
        Answer::score(1.65, score_probs, 0.84),
    );

    answers.insert("is_verified".to_string(), Answer::noul(0.96));

    let usage = Usage::new(142);
    let original_response = SystemOneResponse::new(answers, usage);

    // シリアライズして JSON 文字列化する。
    let serialized = serde_json::to_string_pretty(&original_response).unwrap();

    // 出力 JSON に completion_tokens: 0 が含まれていることを検証する。
    assert!(serialized.contains(r#""completion_tokens": 0"#));

    // デシリアライズして元の構造体と一致することを検証する。
    let deserialized: SystemOneResponse = serde_json::from_str(&serialized).unwrap();
    assert_eq!(original_response, deserialized);
}

/// バリデーション異常系: 空の questions。
#[test]
fn test_validation_empty_questions() {
    let payload = json!({
        "state": "テキスト",
        "questions": {}
    });

    let request: SystemOneRequest = serde_json::from_value(payload).unwrap();
    assert_eq!(request.validate().unwrap_err(), CoreError::EmptyQuestions);
}

/// バリデーション異常系: Choice で criteria が欠落している場合。
#[test]
fn test_validation_choice_missing_criteria() {
    let payload = json!({
        "state": "テキスト",
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "分類せよ。"
            }
        }
    });

    let request: SystemOneRequest = serde_json::from_value(payload).unwrap();
    assert_eq!(
        request.validate().unwrap_err(),
        CoreError::MissingCriteria {
            question_id: Some("category".to_string()),
            question_type: "choice".to_string(),
        }
    );
}

/// バリデーション異常系: Choice で criteria が配列(List)形式で渡された場合。
#[test]
fn test_validation_choice_invalid_criteria_type() {
    let payload = json!({
        "state": "テキスト",
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "分類せよ。",
                "criteria": ["opt1", "opt2"]
            }
        }
    });

    let request: SystemOneRequest = serde_json::from_value(payload).unwrap();
    assert_eq!(
        request.validate().unwrap_err(),
        CoreError::InvalidCriteriaType {
            question_id: Some("category".to_string()),
            question_type: "choice".to_string(),
            expected: "Map (オブジェクト)",
            actual: "List (配列)",
        }
    );
}

/// バリデーション異常系: Score の段階数が 2 未満または 10 超過の場合。
#[test]
fn test_validation_score_invalid_count() {
    // 段階数が 1 の場合
    let payload_single = json!({
        "state": "テキスト",
        "questions": {
            "rating": {
                "type": "score",
                "instructions": "評価せよ。",
                "criteria": ["レベル1のみ"]
            }
        }
    });

    let req_single: SystemOneRequest = serde_json::from_value(payload_single).unwrap();
    assert_eq!(
        req_single.validate().unwrap_err(),
        CoreError::InvalidScoreLevelCount {
            question_id: Some("rating".to_string()),
            count: 1,
        }
    );

    // 段階数が 11 の場合
    let payload_eleven = json!({
        "state": "テキスト",
        "questions": {
            "rating": {
                "type": "score",
                "instructions": "評価せよ。",
                "criteria": (0..11).map(|i| format!("レベル{i}")).collect::<Vec<_>>()
            }
        }
    });

    let req_eleven: SystemOneRequest = serde_json::from_value(payload_eleven).unwrap();
    assert_eq!(
        req_eleven.validate().unwrap_err(),
        CoreError::InvalidScoreLevelCount {
            question_id: Some("rating".to_string()),
            count: 11,
        }
    );
}

/// バリデーション異常系: ゲーティング閾値が逆転している場合 (low_threshold > high_threshold)。
#[test]
fn test_validation_gating_config_reversed_thresholds() {
    let payload = json!({
        "state": "テキスト",
        "questions": {
            "is_flagged": {
                "type": "noul",
                "instructions": "フラグ対象か判定せよ。"
            }
        },
        "gating": {
            "enabled": true,
            "high_threshold": 0.4,
            "low_threshold": 0.8
        }
    });

    let request: SystemOneRequest = serde_json::from_value(payload).unwrap();
    let err = request.validate().unwrap_err();
    assert_eq!(
        err,
        CoreError::InvalidGatingConfig {
            message: "low_threshold (0.8) は high_threshold (0.4) 以下である必要があります。"
                .to_string()
        }
    );
}

/// 正常系: 妥当なゲーティング設定を含むリクエストの検証。
#[test]
fn test_validation_gating_config_valid() {
    let payload = json!({
        "state": "テキスト",
        "questions": {
            "is_flagged": {
                "type": "noul",
                "instructions": "フラグ対象か判定せよ。"
            }
        },
        "gating": {
            "enabled": true,
            "high_threshold": 0.85,
            "low_threshold": 0.45
        }
    });

    let request: SystemOneRequest = serde_json::from_value(payload).unwrap();
    assert!(request.validate().is_ok());
    let gating = request.gating.unwrap();
    assert!(gating.enabled);
    assert_eq!(gating.thresholds.high_threshold, 0.85);
    assert_eq!(gating.thresholds.low_threshold, 0.45);
}

/// StateBatchRequest と SystemOneRequest の共通 DecisionRequest トレイトおよび型エイリアス互換性テスト。
#[test]
fn test_decision_request_trait_and_conversion() {
    use sokuto_core::schema::{DecisionRequest, StateBatchRequest};

    let payload = json!({
        "state": "長文テキスト",
        "questions": {
            "flag": {
                "type": "noul",
                "instructions": "フラグ判定"
            }
        }
    });

    let batch_req: StateBatchRequest = serde_json::from_value(payload).unwrap();
    // 固有 validate と DecisionRequest トレイト経由の検証。
    assert!(batch_req.validate().is_ok());
    assert!(DecisionRequest::validate(&batch_req).is_ok());
    assert_eq!(batch_req.questions().len(), 1);

    // 型エイリアスとしての完全互換性 (SystemOneRequest としての扱い) を検証。
    let s1_req: SystemOneRequest = batch_req;
    assert_eq!(s1_req.questions.len(), 1);
    assert!(s1_req.validate().is_ok());
}

/// `Answer` 構造体の直和型 (AnswerPayload) による型安全性と相互変換の検証テスト。
#[test]
fn test_answer_payload_type_safety_and_roundtrip() {
    use sokuto_core::schema::AnswerPayload;

    // 1. Choice 型の生成とアクセサ
    let mut choice_probs = IndexMap::new();
    choice_probs.insert("opt_a".to_string(), 0.9);
    choice_probs.insert("opt_b".to_string(), 0.1);
    let ans_choice = Answer::choice("opt_a", choice_probs.clone(), 0.85);

    assert!(ans_choice.is_choice());
    assert!(!ans_choice.is_score());
    assert!(!ans_choice.is_noul());
    assert_eq!(ans_choice.as_choice(), Some("opt_a"));
    assert_eq!(ans_choice.as_score(), None);
    assert_eq!(ans_choice.as_noul(), None);
    assert_eq!(ans_choice.confidence(), Some(0.85));
    assert_eq!(ans_choice.effective_confidence(), Some(0.85));
    assert_eq!(ans_choice.question_type(), QuestionType::Choice);

    // 非推奨アクセサの動作検証
    #[allow(deprecated)]
    {
        assert_eq!(ans_choice.payload.selected(), Some("opt_a"));
        assert_eq!(ans_choice.payload.choice(), Some("opt_a"));
    }

    // シリアライズ検証: choice, probabilities, confidence のみが出力され、score, noul は出力されないこと。
    let choice_val = serde_json::to_value(&ans_choice).unwrap();
    assert_eq!(choice_val["choice"], "opt_a");
    assert_eq!(choice_val["confidence"], 0.85);
    assert!(choice_val.get("score").is_none());
    assert!(choice_val.get("noul").is_none());

    // デシリアライズ検証: ラウンドトリップで同一の AnswerPayload::Choice に復元されること。
    let restored_choice: Answer = serde_json::from_value(choice_val).unwrap();
    assert_eq!(restored_choice, ans_choice);
    match &restored_choice.payload {
        AnswerPayload::Choice {
            selected,
            probabilities,
            confidence,
        } => {
            assert_eq!(selected, "opt_a");
            assert_eq!(probabilities, &choice_probs);
            assert_eq!(*confidence, 0.85);
        }
        _ => panic!("Choice 型に復元される必要があります。"),
    }

    // 2. Score 型の生成とアクセサ
    let mut score_probs = IndexMap::new();
    score_probs.insert("1".to_string(), 0.2);
    score_probs.insert("2".to_string(), 0.8);
    let ans_score = Answer::score(1.8, score_probs.clone(), 0.75);

    assert!(ans_score.is_score());
    assert_eq!(ans_score.as_score(), Some(1.8));
    assert_eq!(ans_score.as_choice(), None);
    assert_eq!(ans_score.as_noul(), None);
    assert_eq!(ans_score.confidence(), Some(0.75));
    assert_eq!(ans_score.effective_confidence(), Some(0.75));
    assert_eq!(ans_score.question_type(), QuestionType::Score);

    // 非推奨アクセサの動作検証
    #[allow(deprecated)]
    {
        assert_eq!(ans_score.payload.score(), Some(1.8));
        assert_eq!(ans_score.payload.score_value(), Some(1.8));
        assert_eq!(ans_score.payload.selected(), None);
    }

    let score_val = serde_json::to_value(&ans_score).unwrap();
    assert_eq!(score_val["score"], 1.8);
    assert!(score_val.get("choice").is_none());
    assert!(score_val.get("noul").is_none());

    let restored_score: Answer = serde_json::from_value(score_val).unwrap();
    assert_eq!(restored_score, ans_score);

    // 3. Noul 型の生成とアクセサ
    let ans_noul = Answer::noul(0.92);
    assert!(ans_noul.is_noul());
    assert_eq!(ans_noul.as_noul(), Some(0.92));
    assert_eq!(ans_noul.as_choice(), None);
    assert_eq!(ans_noul.as_score(), None);
    assert_eq!(ans_noul.confidence(), None);

    // 非推奨アクセサの動作検証
    #[allow(deprecated)]
    {
        assert_eq!(ans_noul.payload.noul(), Some(0.92));
        assert_eq!(ans_noul.payload.noul_probability(), Some(0.92));
        assert_eq!(ans_noul.payload.selected(), None);
    }
    // Noul の実効確信度: |2 * 0.92 - 1.0| = 0.84
    assert!((ans_noul.effective_confidence().unwrap() - 0.84).abs() < 1e-6);
    assert_eq!(ans_noul.question_type(), QuestionType::Noul);

    let noul_val = serde_json::to_value(&ans_noul).unwrap();
    assert_eq!(noul_val["noul"], 0.92);
    assert!(noul_val.get("choice").is_none());
    assert!(noul_val.get("score").is_none());
    assert!(noul_val.get("probabilities").is_none());
    assert!(noul_val.get("confidence").is_none());

    let restored_noul: Answer = serde_json::from_value(noul_val).unwrap();
    assert_eq!(restored_noul, ans_noul);
}

/// Jev プロトコル上あり得ない不正な矛盾状態がデシリアライズ時に厳密に拒絶されることを検証するテスト。
#[test]
fn test_answer_rejects_invalid_contradictory_states() {
    // 1. Choice なのに score が含まれている矛盾状態の拒絶
    let conflict_choice_score = json!({
        "choice": "opt_a",
        "score": 2.5,
        "probabilities": {"opt_a": 1.0},
        "confidence": 0.95
    });
    let result: Result<Answer, _> = serde_json::from_value(conflict_choice_score);
    assert!(
        result.is_err(),
        "choice と score の両方が含まれる矛盾 JSON は拒絶される必要があります。"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("mutually exclusive"));

    // 2. Score なのに noul が含まれている矛盾状態の拒絶
    let conflict_score_noul = json!({
        "score": 3.0,
        "noul": 0.8
    });
    let result: Result<Answer, _> = serde_json::from_value(conflict_score_noul);
    assert!(
        result.is_err(),
        "score と noul の両方が含まれる矛盾 JSON は拒絶される必要があります。"
    );

    // 3. choice, score, noul のいずれも含まれていない空状態の拒絶
    let empty_answer = json!({
        "confidence": 0.5
    });
    let result: Result<Answer, _> = serde_json::from_value(empty_answer);
    assert!(
        result.is_err(),
        "判定フィールドを何も持たない空の Answer は拒絶される必要があります。"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("must contain exactly one of"));
}
