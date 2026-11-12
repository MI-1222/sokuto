//! # インプロセス DAG 実行器結合テストスイート
//!
//! 宣言的 DAG 定義のパース、静的検証 (循環検出・未定義ノード検知)、
//! ゼロアロケーション動的プロンプト注入、条件付き分岐制御、
//! タイムアウト制御、および反事実ルール (マイクロ決定 DAG) の動作を検証する。

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use indexmap::IndexMap;
use sokuto_core::contract::calibration::CalibrationConfig;
use sokuto_core::schema::{Answer, Question};
use sokuto_runtime::dag::{
    DagDefinition, DagNode, InProcessDagExecutor, LowLatencyResourcePool, NodeCondition,
    NodeConditionOp, evaluate_conditions, interpolate_prompt, validate_dag,
};
use sokuto_runtime::engine::{InferenceEngine, SessionConfig};
use sokuto_runtime::tokenizer::JevTokenizer;

/// ワークスペースのルートディレクトリを取得する。
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ワークスペースルートの解決に失敗しました。")
        .to_path_buf()
}

/// 配布モデルディレクトリを取得する。
fn default_model_dir() -> PathBuf {
    workspace_root().join("models").join("default")
}

#[test]
fn test_dag_validation_clean_dag() {
    let mut nodes = IndexMap::new();

    // Node 1: セール品判定 (Noul)
    nodes.insert(
        "is_sale_item".to_string(),
        DagNode::inference(
            Question::new_noul("対象商品はセール品ですか？".to_string()),
            vec![
                NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!(true)),
                    target_node: "escalate_reject".to_string(),
                },
                NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!(false)),
                    target_node: "check_within_30days".to_string(),
                },
            ],
        ),
    );

    // Node 2: 30日以内判定 (Noul)
    nodes.insert(
        "check_within_30days".to_string(),
        DagNode::inference(
            Question::new_noul("商品到着後30日以内ですか？".to_string()),
            vec![
                NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!(false)),
                    target_node: "escalate_reject".to_string(),
                },
                NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!(true)),
                    target_node: "check_package_status".to_string(),
                },
            ],
        ),
    );

    // Node 3: 開封状態判定 (Choice)
    let mut package_criteria = IndexMap::new();
    package_criteria.insert("unopened".to_string(), "未開封".to_string());
    package_criteria.insert("opened".to_string(), "開封済み".to_string());
    nodes.insert(
        "check_package_status".to_string(),
        DagNode::inference(
            Question::new_choice(
                "商品の状態を選択してください。".to_string(),
                package_criteria,
            ),
            vec![
                NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!("unopened")),
                    target_node: "approve_return".to_string(),
                },
                NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!("opened")),
                    target_node: "check_initial_defect".to_string(),
                },
            ],
        ),
    );

    // Node 4: 初期不良判定 (Noul)
    nodes.insert(
        "check_initial_defect".to_string(),
        DagNode::inference(
            Question::new_noul("初期不良の申告がありますか？".to_string()),
            vec![
                NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!(true)),
                    target_node: "approve_return".to_string(),
                },
                NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!(false)),
                    target_node: "escalate_reject".to_string(),
                },
            ],
        ),
    );

    // 終端ノード
    nodes.insert(
        "approve_return".to_string(),
        DagNode::Branch {
            conditions: Vec::new(),
            reason: Some("返品条件を満たしているため承認。".to_string()),
        },
    );

    nodes.insert(
        "escalate_reject".to_string(),
        DagNode::escalate(Some("規約により返品不可のため却下または上位判断。")),
    );

    let dag = DagDefinition {
        dag_id: "return_policy_counterfactual_dag".to_string(),
        timeout_ms: 100,
        entry_node: "is_sale_item".to_string(),
        nodes,
    };

    let val_res = validate_dag(&dag);
    assert!(val_res.is_ok(), "正常な DAG は検証成功すること。");
}

#[test]
fn test_dag_validation_cycle_rejection() {
    let mut nodes = IndexMap::new();

    nodes.insert(
        "step_a".to_string(),
        DagNode::inference(
            Question::new_noul("質問A".to_string()),
            vec![NodeCondition {
                field: "answer".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!(true)),
                target_node: "step_b".to_string(),
            }],
        ),
    );

    nodes.insert(
        "step_b".to_string(),
        DagNode::inference(
            Question::new_noul("質問B".to_string()),
            vec![NodeCondition {
                field: "answer".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!(true)),
                target_node: "step_c".to_string(),
            }],
        ),
    );

    nodes.insert(
        "step_c".to_string(),
        DagNode::inference(
            Question::new_noul("質問C".to_string()),
            vec![NodeCondition {
                field: "answer".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!(true)),
                target_node: "step_a".to_string(), // 循環
            }],
        ),
    );

    let dag = DagDefinition {
        dag_id: "cycle_test_dag".to_string(),
        timeout_ms: 50,
        entry_node: "step_a".to_string(),
        nodes,
    };

    let val_res = validate_dag(&dag);
    assert!(val_res.is_err(), "循環 DAG は拒絶されること。");
}

#[test]
fn test_interpolate_prompt_zero_allocation_performance() {
    let mut context = IndexMap::new();
    context.insert(
        "user_check".to_string(),
        sokuto_runtime::dag::NodeContextValue::new("premium", 0.985),
    );
    context.insert(
        "fraud_check".to_string(),
        sokuto_runtime::dag::NodeContextValue::new("false", 0.999),
    );

    let template = "ユーザーランク={user_check.value} (確信度: {user_check.confidence})、不正フラグ={fraud_check.answer}";
    let mut output = String::new();

    // 10,000回の実行で所要時間を計測 (0.8μs/回以下であることを確認)
    let start = std::time::Instant::now();
    for _ in 0..10_000 {
        interpolate_prompt(template, &context, &mut output);
    }
    let elapsed = start.elapsed();
    let per_op_micros = elapsed.as_secs_f64() * 1_000_000.0 / 10_000.0;

    assert_eq!(
        output,
        "ユーザーランク=premium (確信度: 0.985)、不正フラグ=false"
    );
    println!("interpolate_prompt 平均実行時間: {:.3} μs", per_op_micros);
    // スケジューラ目標 0.8μs 未満 (テスト環境のマージン込みで 10μs 以下をアサート)
    assert!(per_op_micros < 10.0, "補間処理が高速であること。");
}

#[tokio::test]
async fn test_pool_borrow_and_drop_lifecycle() {
    let pool = LowLatencyResourcePool::new(vec!["sess1".to_string(), "sess2".to_string()]);
    assert_eq!(pool.capacity(), 2);
    assert_eq!(pool.available(), 2);

    let guard1 = pool
        .acquire(Duration::from_millis(10))
        .await
        .expect("第1セッション借用成功。");
    assert_eq!(pool.available(), 1);

    let guard2 = pool
        .acquire(Duration::from_millis(10))
        .await
        .expect("第2セッション借用成功。");
    assert_eq!(pool.available(), 0);

    // プール枯渇時のタイムアウト検証
    let timeout_res = pool.acquire(Duration::from_millis(5)).await;
    assert!(timeout_res.is_err(), "枯渇時はタイムアウトすること。");

    // drop による自動返却
    drop(guard1);
    assert_eq!(pool.available(), 1);

    drop(guard2);
    assert_eq!(pool.available(), 2);
}

#[test]
fn test_evaluate_conditions_short_circuit() {
    let answer = Answer::choice("card_payment", IndexMap::new(), 0.55);
    let context = IndexMap::new();

    // 確信度不足ガードレールを先頭に配置
    let conditions = vec![
        NodeCondition {
            field: "confidence".to_string(),
            op: NodeConditionOp::ConfidenceLt,
            value: Some(serde_json::json!(0.60)),
            target_node: "escalate_low_confidence".to_string(),
        },
        NodeCondition {
            field: "answer".to_string(),
            op: NodeConditionOp::Eq,
            value: Some(serde_json::json!("card_payment")),
            target_node: "process_card".to_string(),
        },
    ];

    let target = evaluate_conditions(&conditions, Some(&answer), &context);
    // 確信度が 0.55 < 0.60 のため、後の条件に合致しても先頭のエスカレーションが選ばれること
    assert_eq!(target, Some("escalate_low_confidence".to_string()));
}

#[tokio::test]
async fn test_end_to_end_dag_with_onnx_model() {
    let model_dir = default_model_dir();
    let model_path = model_dir.join("model.onnx");
    let tokenizer_path = model_dir.join("tokenizer.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        eprintln!("スキップ: model.onnx または tokenizer.json が存在しません。");
        return;
    }

    let tokenizer =
        Arc::new(JevTokenizer::from_file(&tokenizer_path).expect("トークナイザー初期化失敗。"));
    let engine = Arc::new(
        InferenceEngine::new(&model_path, SessionConfig::cpu_only())
            .expect("推論エンジン初期化失敗。"),
    );

    let calib_config = CalibrationConfig::default();
    let executor = InProcessDagExecutor::new(engine, tokenizer, calib_config);

    // 2 ステップ DAG:
    // Step 1: 問い合わせ種別判定 (Choice: card / transfer / other)
    // - card -> Escalate (カード専門窓口)
    // - transfer -> Step 2 (送金トラブル詳細)
    let mut step1_criteria = IndexMap::new();
    step1_criteria.insert(
        "card".to_string(),
        "カード利用や紛失に関する照会".to_string(),
    );
    step1_criteria.insert(
        "transfer".to_string(),
        "振込や口座間送金に関する照会".to_string(),
    );
    step1_criteria.insert("other".to_string(), "その他一般的な問い合わせ".to_string());

    let mut nodes = IndexMap::new();
    nodes.insert(
        "triage".to_string(),
        DagNode::inference(
            Question::new_choice(
                "問い合わせ内容の大分類を選択してください。".to_string(),
                step1_criteria,
            ),
            vec![
                NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!("card")),
                    target_node: "escalate_card".to_string(),
                },
                NodeCondition {
                    field: "answer".to_string(),
                    op: NodeConditionOp::Eq,
                    value: Some(serde_json::json!("transfer")),
                    target_node: "transfer_detail".to_string(),
                },
            ],
        ),
    );

    let mut step2_criteria = IndexMap::new();
    step2_criteria.insert("delay".to_string(), "送金が遅延している".to_string());
    step2_criteria.insert("cancel".to_string(), "送金を取り消したい".to_string());
    nodes.insert(
        "transfer_detail".to_string(),
        DagNode::inference(
            Question::new_choice(
                "送金トラブルの具体的内容を選択してください。".to_string(),
                step2_criteria,
            ),
            vec![],
        ),
    );

    nodes.insert(
        "escalate_card".to_string(),
        DagNode::escalate(Some("カード関連の専任担当者へエスカレーション。")),
    );

    let dag = DagDefinition {
        dag_id: "banking_triage_dag".to_string(),
        timeout_ms: 200,
        entry_node: "triage".to_string(),
        nodes,
    };

    let state = "クレジットカードを落としてしまったので至急利用を停止したいです。";
    let result = executor
        .execute(state, &dag)
        .await
        .expect("DAG 実行成功すること。");

    println!("実行結果: {:?}", result.execution_path);
    println!("合計所要時間: {:.2} ms", result.total_duration_ms);

    assert_eq!(result.execution_path[0], "triage");
    // "card" が選択されて escalate_card に遷移するか、正常に終了すること
    assert!(result.steps.contains_key("triage"));
}

#[tokio::test]
async fn test_dag_timeout_enforcement() {
    let model_dir = default_model_dir();
    let model_path = model_dir.join("model.onnx");
    let tokenizer_path = model_dir.join("tokenizer.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        eprintln!("スキップ: model.onnx または tokenizer.json が存在しません。");
        return;
    }

    let tokenizer =
        Arc::new(JevTokenizer::from_file(&tokenizer_path).expect("トークナイザー初期化失敗。"));
    let engine = Arc::new(
        InferenceEngine::new(&model_path, SessionConfig::cpu_only())
            .expect("推論エンジン初期化失敗。"),
    );

    let calib_config = CalibrationConfig::default();
    let executor = InProcessDagExecutor::new(engine, tokenizer, calib_config);

    let mut nodes = IndexMap::new();
    nodes.insert(
        "q1".to_string(),
        DagNode::inference(Question::new_noul("質問内容".to_string()), vec![]),
    );

    // 0ms (即時タイムアウト) を指定
    let dag = DagDefinition {
        dag_id: "timeout_dag".to_string(),
        timeout_ms: 0,
        entry_node: "q1".to_string(),
        nodes,
    };

    let result = executor.execute("入力文", &dag).await;
    assert!(
        matches!(result, Err(sokuto_runtime::dag::DagError::Timeout(_))),
        "0ms タイムアウトで即時 Timeout エラーが返却されること。"
    );
}

#[test]
fn test_parallel_nodes_pure_task_validation() {
    let mut nodes = IndexMap::new();

    // conditions を持った並行タスクノード (不正仕様)
    nodes.insert(
        "p_task_with_conditions".to_string(),
        DagNode::inference(
            Question::new_noul("並行質問1".to_string()),
            vec![NodeCondition {
                field: "answer".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!(true)),
                target_node: "join".to_string(),
            }],
        ),
    );

    nodes.insert(
        "p_task_clean".to_string(),
        DagNode::inference(Question::new_noul("並行質問2".to_string()), vec![]),
    );

    nodes.insert(
        "parallel_parent".to_string(),
        DagNode::parallel(
            vec![
                "p_task_with_conditions".to_string(),
                "p_task_clean".to_string(),
            ],
            "join",
        ),
    );

    nodes.insert(
        "join".to_string(),
        DagNode::Branch {
            conditions: vec![],
            reason: Some("合流完了".to_string()),
        },
    );

    let dag = DagDefinition {
        dag_id: "parallel_validation_dag".to_string(),
        timeout_ms: 100,
        entry_node: "parallel_parent".to_string(),
        nodes,
    };

    let res = validate_dag(&dag);
    assert!(
        res.is_err(),
        "parallel_nodes に conditions を持つノードが含まれる場合はバリデーションエラーとなること。"
    );
}

#[tokio::test]
async fn test_escalation_diagnostics_aggregation() {
    let model_dir = default_model_dir();
    let model_path = model_dir.join("model.onnx");
    let tokenizer_path = model_dir.join("tokenizer.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        eprintln!("スキップ: model.onnx または tokenizer.json が存在しません。");
        return;
    }

    let tokenizer =
        Arc::new(JevTokenizer::from_file(&tokenizer_path).expect("トークナイザー初期化失敗。"));
    let engine = Arc::new(
        InferenceEngine::new(&model_path, SessionConfig::cpu_only())
            .expect("推論エンジン初期化失敗。"),
    );

    let calib_config = CalibrationConfig::default();
    let executor = InProcessDagExecutor::new(engine, tokenizer, calib_config);

    let mut nodes = IndexMap::new();
    nodes.insert(
        "fraud_check".to_string(),
        DagNode::inference(
            Question::new_noul("不正送金の疑いがありますか？".to_string()),
            vec![
                // 無条件に escalate へ誘導
                NodeCondition {
                    field: "confidence".to_string(),
                    op: NodeConditionOp::ConfidenceGte,
                    value: Some(serde_json::json!(0.0)),
                    target_node: "escalate_security".to_string(),
                },
            ],
        ),
    );

    nodes.insert(
        "escalate_security".to_string(),
        DagNode::escalate(Some("セキュリティ疑いによる即時エスカレーション。")),
    );

    let dag = DagDefinition {
        dag_id: "escalation_diagnostic_dag".to_string(),
        timeout_ms: 200,
        entry_node: "fraud_check".to_string(),
        nodes,
    };

    let result = executor
        .execute("至急確認してください", &dag)
        .await
        .expect("実行成功すること。");

    assert!(result.escalated);
    assert_eq!(
        result.escalation_reason.as_deref(),
        Some("セキュリティ疑いによる即時エスカレーション。")
    );
    assert_eq!(result.escalation_node_id.as_deref(), Some("fraud_check"));
    assert!(
        result.escalation_prompt.is_some(),
        "System 2 向け Guided CoT プロンプト案が集約されていること。"
    );
}

#[tokio::test]
async fn test_escalation_via_branch_diagnostics_aggregation() {
    let model_dir = default_model_dir();
    let model_path = model_dir.join("model.onnx");
    let tokenizer_path = model_dir.join("tokenizer.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        eprintln!("スキップ: model.onnx または tokenizer.json が存在しません。");
        return;
    }

    let tokenizer =
        Arc::new(JevTokenizer::from_file(&tokenizer_path).expect("トークナイザー初期化失敗。"));
    let engine = Arc::new(
        InferenceEngine::new(&model_path, SessionConfig::cpu_only())
            .expect("推論エンジン初期化失敗。"),
    );

    let calib_config = CalibrationConfig::default();
    let executor = InProcessDagExecutor::new(engine, tokenizer, calib_config);

    let mut nodes = IndexMap::new();
    nodes.insert(
        "fraud_check".to_string(),
        DagNode::inference(
            Question::new_noul("不正送金の疑いがありますか？".to_string()),
            vec![NodeCondition {
                field: "confidence".to_string(),
                op: NodeConditionOp::ConfidenceGte,
                value: Some(serde_json::json!(0.0)),
                target_node: "branch_security".to_string(),
            }],
        ),
    );

    nodes.insert(
        "branch_security".to_string(),
        DagNode::branch(vec![NodeCondition {
            field: "confidence".to_string(),
            op: NodeConditionOp::ConfidenceGte,
            value: Some(serde_json::json!(0.0)),
            target_node: "escalate_security".to_string(),
        }]),
    );

    nodes.insert(
        "escalate_security".to_string(),
        DagNode::escalate(Some("セキュリティ疑いによる即時エスカレーション。")),
    );

    let dag = DagDefinition {
        dag_id: "escalation_via_branch_dag".to_string(),
        timeout_ms: 200,
        entry_node: "fraud_check".to_string(),
        nodes,
    };

    let result = executor
        .execute("至急確認してください", &dag)
        .await
        .expect("実行に成功。");

    assert!(result.escalated);
    assert_eq!(
        result.execution_path,
        vec!["fraud_check", "branch_security", "escalate_security"]
    );
    assert_eq!(
        result.escalation_reason.as_deref(),
        Some("セキュリティ疑いによる即時エスカレーション。")
    );
    assert_eq!(
        result.escalation_node_id.as_deref(),
        Some("fraud_check"),
        "直前ノードが Branch の場合でも、最後に実行された推論ノードが特定されること。"
    );
    assert!(
        result.escalation_prompt.is_some(),
        "System 2 向け Guided CoT プロンプト案が集約されていること。"
    );
}

#[test]
fn test_validate_dag_condition_rejections() {
    use sokuto_runtime::dag::DagValidationError;

    // 1. 未知フィールド (non_existent) の静的拒否
    let mut nodes_unknown = IndexMap::new();
    nodes_unknown.insert(
        "q1".to_string(),
        DagNode::inference(
            Question::new_noul("質問".to_string()),
            vec![NodeCondition {
                field: "non_existent".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!(true)),
                target_node: "term".to_string(),
            }],
        ),
    );
    nodes_unknown.insert(
        "term".to_string(),
        DagNode::escalate(Option::<String>::None),
    );
    let dag_unknown = DagDefinition {
        dag_id: "unknown_field_test".to_string(),
        timeout_ms: 100,
        entry_node: "q1".to_string(),
        nodes: nodes_unknown,
    };
    let err_unknown = validate_dag(&dag_unknown).expect_err("未知フィールドは拒否されること。");
    assert!(matches!(
        err_unknown,
        DagValidationError::InvalidCondition {
            ref node_id,
            ref field,
            ..
        } if node_id == "q1" && field == "non_existent"
    ));

    // 2. 未定義ノード参照 (ghost.value) の静的拒否
    let mut nodes_ghost = IndexMap::new();
    nodes_ghost.insert(
        "q1".to_string(),
        DagNode::inference(
            Question::new_noul("質問".to_string()),
            vec![NodeCondition {
                field: "ghost.value".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!("true")),
                target_node: "term".to_string(),
            }],
        ),
    );
    nodes_ghost.insert(
        "term".to_string(),
        DagNode::escalate(Option::<String>::None),
    );
    let dag_ghost = DagDefinition {
        dag_id: "ghost_ref_test".to_string(),
        timeout_ms: 100,
        entry_node: "q1".to_string(),
        nodes: nodes_ghost,
    };
    let err_ghost = validate_dag(&dag_ghost).expect_err("未定義ノード参照は拒否されること。");
    assert!(matches!(
        err_ghost,
        DagValidationError::InvalidCondition {
            ref node_id,
            ref field,
            ..
        } if node_id == "q1" && field == "ghost.value"
    ));

    // 3. 演算子・値型不整合 (Score に対する In 演算子) の静的拒否
    let mut nodes_type = IndexMap::new();
    nodes_type.insert(
        "q1".to_string(),
        DagNode::inference(
            Question::new_score(
                "スコア評価".to_string(),
                vec!["1".to_string(), "2".to_string(), "3".to_string()],
            ),
            vec![NodeCondition {
                field: "answer".to_string(),
                op: NodeConditionOp::In,
                value: Some(serde_json::json!([1, 2])),
                target_node: "term".to_string(),
            }],
        ),
    );
    nodes_type.insert(
        "term".to_string(),
        DagNode::escalate(Option::<String>::None),
    );
    let dag_type = DagDefinition {
        dag_id: "type_mismatch_test".to_string(),
        timeout_ms: 100,
        entry_node: "q1".to_string(),
        nodes: nodes_type,
    };
    let err_type = validate_dag(&dag_type).expect_err("型不整合条件は拒否されること。");
    assert!(matches!(
        err_type,
        DagValidationError::InvalidCondition {
            ref node_id,
            ref field,
            ..
        } if node_id == "q1" && field == "answer"
    ));
}

#[test]
fn test_validate_dag_interpolation_rejections() {
    use sokuto_runtime::dag::DagValidationError;

    // 1. 未定義ノード補間 ({ghost.value}) の静的拒否
    let mut nodes_ghost = IndexMap::new();
    nodes_ghost.insert(
        "q1".to_string(),
        DagNode::inference(
            Question::new_noul("前段結果 {ghost.value} に基づき判定せよ。".to_string()),
            vec![NodeCondition {
                field: "answer".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!(true)),
                target_node: "term".to_string(),
            }],
        ),
    );
    nodes_ghost.insert(
        "term".to_string(),
        DagNode::escalate(Option::<String>::None),
    );
    let dag_ghost = DagDefinition {
        dag_id: "interpolate_ghost_test".to_string(),
        timeout_ms: 100,
        entry_node: "q1".to_string(),
        nodes: nodes_ghost,
    };
    let err_ghost = validate_dag(&dag_ghost).expect_err("未定義ノード補間は拒否されること。");
    assert!(matches!(
        err_ghost,
        DagValidationError::InvalidInterpolation {
            ref node_id,
            ref placeholder,
            ..
        } if node_id == "q1" && placeholder == "ghost.value"
    ));

    // 2. 自己参照 ({q1.value}) の静的拒否
    let mut nodes_self = IndexMap::new();
    nodes_self.insert(
        "q1".to_string(),
        DagNode::inference(
            Question::new_noul("自己値 {q1.value} に基づき判定せよ。".to_string()),
            vec![NodeCondition {
                field: "answer".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!(true)),
                target_node: "term".to_string(),
            }],
        ),
    );
    nodes_self.insert(
        "term".to_string(),
        DagNode::escalate(Option::<String>::None),
    );
    let dag_self = DagDefinition {
        dag_id: "interpolate_self_test".to_string(),
        timeout_ms: 100,
        entry_node: "q1".to_string(),
        nodes: nodes_self,
    };
    let err_self = validate_dag(&dag_self).expect_err("自己参照補間は拒否されること。");
    assert!(matches!(
        err_self,
        DagValidationError::InvalidInterpolation {
            ref node_id,
            ref placeholder,
            ..
        } if node_id == "q1" && placeholder == "q1.value"
    ));

    // 3. 上流逆転参照 (後続ノード {q2.value} を参照) の静的拒否
    let mut nodes_forward = IndexMap::new();
    nodes_forward.insert(
        "q1".to_string(),
        DagNode::inference(
            Question::new_noul("後段結果 {q2.value} を参照。".to_string()),
            vec![NodeCondition {
                field: "answer".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!(true)),
                target_node: "q2".to_string(),
            }],
        ),
    );
    nodes_forward.insert(
        "q2".to_string(),
        DagNode::inference(
            Question::new_noul("後段質問。".to_string()),
            vec![NodeCondition {
                field: "answer".to_string(),
                op: NodeConditionOp::Eq,
                value: Some(serde_json::json!(true)),
                target_node: "term".to_string(),
            }],
        ),
    );
    nodes_forward.insert(
        "term".to_string(),
        DagNode::escalate(Option::<String>::None),
    );
    let dag_forward = DagDefinition {
        dag_id: "interpolate_forward_test".to_string(),
        timeout_ms: 100,
        entry_node: "q1".to_string(),
        nodes: nodes_forward,
    };
    let err_forward =
        validate_dag(&dag_forward).expect_err("後続ノード参照 (逆転補間) は拒否されること。");
    assert!(matches!(
        err_forward,
        DagValidationError::InvalidInterpolation {
            ref node_id,
            ref placeholder,
            ..
        } if node_id == "q1" && placeholder == "q2.value"
    ));
}
