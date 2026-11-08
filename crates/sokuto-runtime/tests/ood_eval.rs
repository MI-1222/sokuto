//! # OOD (Out-of-Distribution: 未定義カテゴリ・異常入力) 安全弁評価テストスイート
//!
//! ヘルムホルツ自由エネルギーおよび候補数正規化自由エネルギーに基づく
//! OOD 安全弁回路が、未定義カテゴリやクロスドメイン異常入力に対して
//! 目標検知率 90% 以上 (Exit Criteria 4) を達成しているかを実測・検証する。

use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use sokuto_core::contract::calibration::CalibrationConfig;
use sokuto_core::gating::{GatingConfig, GatingThresholds};
use sokuto_core::schema::{Answer, Question};
use sokuto_runtime::engine::coarse::CoarseToFineConfig;
use sokuto_runtime::engine::{InferenceEngine, SessionConfig};
use sokuto_runtime::tokenizer::JevTokenizer;

/// ワークスペースのルートディレクトリを取得する。
fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("ワークスペースルートの解決に失敗した。")
        .to_path_buf()
}

/// 指定モデルディレクトリから推論エンジンとトークナイザーを初期化する。
fn init_engine(model_dir: &Path) -> Option<(InferenceEngine, JevTokenizer, CalibrationConfig)> {
    let onnx_path = model_dir.join("model.onnx");
    let tok_path = model_dir.join("tokenizer.json");
    let calib_path = model_dir.join("calibration.json");

    if !onnx_path.exists() || !tok_path.exists() {
        eprintln!(
            "スキップ: モデルまたはトークナイザーが存在しない ({:?})。",
            model_dir
        );
        return None;
    }

    let session_config = SessionConfig::cpu_only();
    let engine = InferenceEngine::new(&onnx_path, session_config).ok()?;
    let tokenizer = JevTokenizer::from_file(&tok_path).ok()?;
    let calib_config = if calib_path.exists() {
        std::fs::read_to_string(&calib_path)
            .ok()
            .and_then(|s| CalibrationConfig::from_json_str(&s).ok())
            .unwrap_or_default()
    } else {
        CalibrationConfig::default()
    };

    Some((engine, tokenizer, calib_config))
}

/// 単一の評価サンプル定義。
#[derive(Clone)]
struct Sample {
    /// 入力コンテキストテキスト (State)。
    state: &'static str,
    /// 評価指示文 (Instructions)。
    instructions: &'static str,
    /// 候補選択肢マップ (識別子 -> 説明文)。
    criteria: Vec<(&'static str, &'static str)>,
    /// サンプルの説明・種別。
    description: &'static str,
}

/// 評価データセットを構築する。
fn build_evaluation_datasets() -> (Vec<Sample>, Vec<Sample>) {
    // -------------------------------------------------------------------------
    // 1. In-Distribution (ID: 分布内通常サンプル群)
    // -------------------------------------------------------------------------
    let id_samples = vec![
        Sample {
            state: "昨日購入したキーボードのBluetooth接続が頻繁に切断されます。初期不良でしょうか？交換可能ですか？",
            instructions: "問い合わせの担当部門を判定してください。",
            criteria: vec![
                ("tech_support", "技術サポート・初期不良対応・修理窓口"),
                ("billing", "請求書・クレジットカード決済・領収書発行"),
                ("sales", "製品導入相談・見積もり・新規購入"),
                ("general", "総合受付・その他の一般的な問い合わせ"),
            ],
            description: "ID: サポート窓口問い合わせ",
        },
        Sample {
            state: "クレジットカードの請求明細に見覚えのない引き落としがありました。詳細を確認したいです。",
            instructions: "問い合わせの担当部門を判定してください。",
            criteria: vec![
                ("tech_support", "技術サポート・初期不良対応・修理窓口"),
                ("billing", "請求書・クレジットカード決済・領収書発行"),
                ("sales", "製品導入相談・見積もり・新規購入"),
                ("general", "総合受付・その他の一般的な問い合わせ"),
            ],
            description: "ID: 請求・決済問い合わせ",
        },
        Sample {
            state: "法人向けプランの契約を検討しています。100アカウントの一括見積もりをお願いできますか？",
            instructions: "問い合わせの担当部門を判定してください。",
            criteria: vec![
                ("tech_support", "技術サポート・初期不良対応・修理窓口"),
                ("billing", "請求書・クレジットカード決済・領収書発行"),
                ("sales", "製品導入相談・見積もり・新規購入"),
                ("general", "総合受付・その他の一般的な問い合わせ"),
            ],
            description: "ID: 営業・購入相談",
        },
        Sample {
            state: "迅速かつ丁寧なご対応をいただき誠にありがとうございました。大変助かりました。",
            instructions: "お客様の声の感情極性を判定してください。",
            criteria: vec![
                ("positive", "肯定的・感謝・満足・改善"),
                ("neutral", "中立・事実の連絡・変化なし"),
                ("negative", "否定的・不満・クレーム・落胆"),
            ],
            description: "ID: ポジティブ感謝",
        },
        Sample {
            state: "サービス品質に非常に満足しており、友人や同僚にもぜひ薦めたいと思います。",
            instructions: "お客様の声の感情極性を判定してください。",
            criteria: vec![
                ("positive", "肯定的・感謝・満足・改善"),
                ("neutral", "中立・事実の連絡・変化なし"),
                ("negative", "否定的・不満・クレーム・落胆"),
            ],
            description: "ID: ポジティブ推薦",
        },
        Sample {
            state: "配送が予定より3日も遅れ、事前の連絡も一切ありませんでした。非常に失望しました。",
            instructions: "お客様の声の感情極性を判定してください。",
            criteria: vec![
                ("positive", "肯定的・感謝・満足・改善"),
                ("neutral", "中立・事実の連絡・変化なし"),
                ("negative", "否定的・不満・クレーム・落胆"),
            ],
            description: "ID: ネガティブ遅延不満",
        },
        Sample {
            state: "操作方法が極めて難解でマニュアルも不親切です。二度と利用したくありません。",
            instructions: "お客様の声の感情極性を判定してください。",
            criteria: vec![
                ("positive", "肯定的・感謝・満足・改善"),
                ("neutral", "中立・事実の連絡・変化なし"),
                ("negative", "否定的・不満・クレーム・落胆"),
            ],
            description: "ID: ネガティブ製品酷評",
        },
        Sample {
            state: "管理画面にログインしようとすると『認証エラー 401』が表示され、全社員が業務を行えません。",
            instructions: "障害・インシデントの重大度を判定してください。",
            criteria: vec![
                (
                    "critical",
                    "システム全損・主要サービス停止・全社業務継続不可",
                ),
                ("high", "主要機能の障害・一部ユーザーの業務停止"),
                ("medium", "軽微な機能障害・代替手段あり"),
                ("low", "表示崩れや要望・運用影響なし"),
            ],
            description: "ID: クリティカル障害",
        },
        Sample {
            state: "フッターの著作権表示の年号が2025年のままになっています。修正をお願いします。",
            instructions: "障害・インシデントの重大度を判定してください。",
            criteria: vec![
                (
                    "critical",
                    "システム全損・主要サービス停止・全社業務継続不可",
                ),
                ("high", "主要機能の障害・一部ユーザーの業務停止"),
                ("medium", "軽微な機能障害・代替手段あり"),
                ("low", "表示崩れや要望・運用影響なし"),
            ],
            description: "ID: 軽微な表示崩れ",
        },
        Sample {
            state: "本日出荷された荷物の追跡番号を教えてください。",
            instructions: "要求のカテゴリを判定してください。",
            criteria: vec![
                ("tracking", "荷物の配送状況・追跡番号の確認"),
                ("cancel", "注文のキャンセル依頼"),
                ("address_change", "お届け先住所の変更依頼"),
                ("return", "返品・返金手続きの依頼"),
            ],
            description: "ID: 配送追跡要求",
        },
        Sample {
            state: "間違えて同じ商品を2重に注文してしまいました。1件キャンセルしてください。",
            instructions: "要求のカテゴリを判定してください。",
            criteria: vec![
                ("tracking", "荷物の配送状況・追跡番号の確認"),
                ("cancel", "注文のキャンセル依頼"),
                ("address_change", "お届け先住所の変更依頼"),
                ("return", "返品・返金手続きの依頼"),
            ],
            description: "ID: 注文キャンセル要求",
        },
        Sample {
            state: "来週引っ越すことになったため、配送先を新しい住所に変更したいです。",
            instructions: "要求のカテゴリを判定してください。",
            criteria: vec![
                ("tracking", "荷物の配送状況・追跡番号の確認"),
                ("cancel", "注文のキャンセル依頼"),
                ("address_change", "お届け先住所の変更依頼"),
                ("return", "返品・返金手続きの依頼"),
            ],
            description: "ID: 住所変更要求",
        },
        Sample {
            state: "届いた服のサイズが合いませんでした。サイズ交換または返品は可能でしょうか？",
            instructions: "要求のカテゴリを判定してください。",
            criteria: vec![
                ("tracking", "荷物の配送状況・追跡番号の確認"),
                ("cancel", "注文のキャンセル依頼"),
                ("address_change", "お届け先住所の変更依頼"),
                ("return", "返品・返金手続きの依頼"),
            ],
            description: "ID: 返品・交換要求",
        },
    ];

    // -------------------------------------------------------------------------
    // 2. Out-of-Distribution (OOD: 未定義カテゴリ・異常入力・無関係サンプル群)
    // -------------------------------------------------------------------------
    let ood_samples = vec![
        // (A) 未定義カテゴリ (正解選択肢が候補群に存在せず、該当なしを選ぶべきケース)
        Sample {
            state: "ノートパソコンの液晶画面が割れてしまい、バックライトが点灯しません。修理費用はいくらですか？",
            instructions: "問い合わせの担当部門を判定してください。",
            criteria: vec![
                ("billing", "請求書・クレジットカード決済・領収書発行"),
                ("hr", "人事部・採用・労務管理"),
                ("legal", "法務部・利用規約・知的財産・コンプライアンス"),
            ],
            description: "OOD-A1: 技術修理なのに選択肢が請求・人事・法務のみ",
        },
        Sample {
            state: "社内の有給休暇申請フローと年末調整の書類提出期限を教えてください。",
            instructions: "問い合わせの担当部門を判定してください。",
            criteria: vec![
                ("tech_support", "技術サポート・初期不良対応・修理窓口"),
                ("logistics", "物流管理・倉庫入出荷・在庫引当"),
                ("sales", "製品導入相談・見積もり・新規購入"),
            ],
            description: "OOD-A2: 社内人事の質問なのに選択肢が外部向け窓口のみ",
        },
        Sample {
            state: "レストランの個室を大人4名で金曜日の19時から予約したいのですが空いていますか？",
            instructions: "要求のカテゴリを判定してください。",
            criteria: vec![
                ("tracking", "荷物の配送状況・追跡番号の確認"),
                ("cancel", "注文のキャンセル依頼"),
                ("address_change", "お届け先住所の変更依頼"),
                ("return", "返品・返金手続きの依頼"),
            ],
            description: "OOD-A3: レストラン予約なのに選択肢がEC通販手続きのみ",
        },
        Sample {
            state: "最新のセキュリティ脆弱性 CVE-2026-9999 に対するパッチ適用手順を教えてください。",
            instructions: "お客様の声の感情極性を判定してください。",
            criteria: vec![
                ("positive", "肯定的・感謝・満足・改善"),
                ("negative", "否定的・不満・クレーム・落胆"),
            ],
            description: "OOD-A4: セキュリティ技術手順なのに選択肢が肯定的/否定的のみ",
        },
        // (B) クロスドメイン・完全無関係テキスト (料理、科学、文学、スポーツなど)
        Sample {
            state: "玉ねぎをみじん切りにして弱火で飴色になるまでじっくり炒め、牛すじ肉とスパイスを加えて煮込みます。",
            instructions: "問い合わせの担当部門を判定してください。",
            criteria: vec![
                ("tech_support", "技術サポート・初期不良対応・修理窓口"),
                ("billing", "請求書・クレジットカード決済・領収書発行"),
                ("sales", "製品導入相談・見積もり・新規購入"),
            ],
            description: "OOD-B1: カレーの調理レシピテキスト",
        },
        Sample {
            state: "アンドロメダ銀河 (M31) は地球から約250万光年の距離に位置する渦巻銀河であり、局所銀河群で最大です。",
            instructions: "障害・インシデントの重大度を判定してください。",
            criteria: vec![
                ("critical", "システム全損・主要サービス停止・業務継続不可"),
                ("high", "主要機能の障害・一部ユーザーの業務停止"),
                ("low", "表示崩れや要望・運用影響なし"),
            ],
            description: "OOD-B2: 天文学の解説テキスト",
        },
        Sample {
            state: "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。",
            instructions: "要求のカテゴリを判定してください。",
            criteria: vec![
                ("tracking", "荷物の配送状況・追跡番号の確認"),
                ("cancel", "注文のキャンセル依頼"),
                ("return", "返品・返金手続きの依頼"),
            ],
            description: "OOD-B3: 夏目漱石の文学作品冒頭テキスト",
        },
        Sample {
            state: "9回裏ツーアウト満塁、カウント3ボール2ストライクから放たれた打球はレフトスタンドへ飛び込むサヨナラ満塁ホームランとなりました。",
            instructions: "問い合わせの担当部門を判定してください。",
            criteria: vec![
                ("tech_support", "技術サポート・初期不良対応・修理窓口"),
                ("billing", "請求書・クレジットカード決済・領収書発行"),
                ("sales", "製品導入相談・見積もり・新規購入"),
            ],
            description: "OOD-B4: 野球のスポーツ実況テキスト",
        },
        Sample {
            state: "光速は真空中において約秒速30万キロメートルであり、アインシュタインの特殊相対性理論において不変の物理定数とされます。",
            instructions: "問い合わせの緊急度を判定してください。",
            criteria: vec![
                ("urgent", "緊急・即時対応が必要"),
                ("normal", "通常・当日中の対応"),
                ("low", "低優先・数日以内の対応"),
            ],
            description: "OOD-B5: 物理学の理論解説テキスト",
        },
        // (C) 異常ノイズ・非対応言語・無意味な文字列 (過信防止対象)
        Sample {
            state: "ភាសាខ្មែរ គឺជាភាសាផ្លូវការនៃប្រទេសកម្ពុជា ដែលនិយាយដោយប្រជាជនខ្មែរ។",
            instructions: "問い合わせの担当部門を判定してください。",
            criteria: vec![
                ("tech_support", "技術サポート・初期不良対応・修理窓口"),
                ("billing", "請求書・クレジットカード決済・領収書発行"),
                ("sales", "製品導入相談・見積もり・新規購入"),
            ],
            description: "OOD-C1: クメール語 (未対応言語)",
        },
        Sample {
            state: "ภาษาไทยเป็นภาษาทางการและภาษาประจำชาติของประเทศไทย มีอักษรไทยเป็นเอกลักษณ์",
            instructions: "お客様の声の感情極性を判定してください。",
            criteria: vec![
                ("positive", "肯定的・感謝・満足・改善"),
                ("neutral", "中立・事実の連絡・変化なし"),
                ("negative", "否定的・不満・クレーム・落胆"),
            ],
            description: "OOD-C2: タイ語 (未対応言語)",
        },
        Sample {
            state: "aslkd!@#$0982341kjhasdf &*^%#@! kjhwef0982341lkjsdf ??? /// ===",
            instructions: "問い合わせの担当部門を判定してください。",
            criteria: vec![
                ("tech_support", "技術サポート・初期不良対応・修理窓口"),
                ("billing", "請求書・クレジットカード決済・領収書発行"),
                ("sales", "製品導入相談・見積もり・新規購入"),
            ],
            description: "OOD-C3: ランダム記号英数ノイズ",
        },
        Sample {
            state: "あああああああああああああああああああああああああああああああああああああああああああああ",
            instructions: "問い合わせの担当部門を判定してください。",
            criteria: vec![
                ("tech_support", "技術サポート・初期不良対応・修理窓口"),
                ("billing", "請求書・クレジットカード決済・領収書発行"),
                ("sales", "製品導入相談・見積もり・新規購入"),
            ],
            description: "OOD-C4: 極端な同一文字反復",
        },
    ];

    (id_samples, ood_samples)
}

/// モデルの各サンプルに対する正規化自由エネルギーを収集・分析する。
fn analyze_model_energy_distribution(
    engine: &InferenceEngine,
    tokenizer: &JevTokenizer,
    calib_config: &CalibrationConfig,
    tier_name: &str,
) -> (Vec<f64>, Vec<f64>) {
    let (id_samples, ood_samples) = build_evaluation_datasets();
    let coarse_config = CoarseToFineConfig::default();

    // ゲーティング閾値は十分に高く設定し、すべてのサンプルで素の energy 値を取得する
    let gating_config = GatingConfig {
        enabled: true,
        thresholds: GatingThresholds {
            high_threshold: 0.70,
            low_threshold: 0.35,
            top_margin_threshold: 0.15,
            ood_enabled: true,
            energy_temperature: None,
            energy_threshold: 100.0,
            ..GatingThresholds::default()
        },
    };

    println!("\n================================================================================");
    println!(">>> 【エネルギー分布解析】: {}", tier_name);
    println!("================================================================================");

    // ID サンプルのエネルギー測定
    println!(
        "--- [In-Distribution (ID) サンプル群 ({} 件)] ---",
        id_samples.len()
    );
    let mut id_energies = Vec::new();
    for sample in &id_samples {
        let mut criteria_map = IndexMap::new();
        for (k, v) in &sample.criteria {
            criteria_map.insert(k.to_string(), v.to_string());
        }
        let question = Question::new_choice(sample.instructions, criteria_map);

        let answer: Answer = engine
            .evaluate_question_coarse_to_fine_with_gating(
                tokenizer,
                sample.state,
                &question,
                calib_config,
                &coarse_config,
                Some(&gating_config),
            )
            .expect("推論に成功した。");

        let mut flat_logits = vec![0.0f64; sample.criteria.len()];
        let state_ids = tokenizer.encode_state(sample.state).unwrap();
        let mut sub_questions = IndexMap::new();
        sub_questions.insert("q".to_string(), question.clone());
        let batch = tokenizer
            .encode_batch_with_pretokenized_state(&state_ids, &sub_questions, 512)
            .unwrap();
        engine
            .forward_batch_tokenized_into(&batch, &mut flat_logits)
            .unwrap();

        let gating = answer.gating.expect("gating メタデータが存在する。");
        let energy = gating.energy.expect("energy 値が存在する。");
        let _choice = answer.choice.unwrap_or_default();
        id_energies.push(energy);
        println!(
            "  ID  | E = {:>7.3} | Conf = {:.3} | Logits = {:<25?} | {}",
            energy, gating.confidence, flat_logits, sample.description
        );
    }

    // OOD サンプルのエネルギー測定
    println!(
        "\n--- [Out-of-Distribution (OOD) サンプル群 ({} 件)] ---",
        ood_samples.len()
    );
    let mut ood_energies = Vec::new();
    for sample in &ood_samples {
        let mut criteria_map = IndexMap::new();
        for (k, v) in &sample.criteria {
            criteria_map.insert(k.to_string(), v.to_string());
        }
        let question = Question::new_choice(sample.instructions, criteria_map);

        let answer: Answer = engine
            .evaluate_question_coarse_to_fine_with_gating(
                tokenizer,
                sample.state,
                &question,
                calib_config,
                &coarse_config,
                Some(&gating_config),
            )
            .expect("推論に成功した。");

        let mut flat_logits = vec![0.0f64; sample.criteria.len()];
        let state_ids = tokenizer.encode_state(sample.state).unwrap();
        let mut sub_questions = IndexMap::new();
        sub_questions.insert("q".to_string(), question.clone());
        let batch = tokenizer
            .encode_batch_with_pretokenized_state(&state_ids, &sub_questions, 512)
            .unwrap();
        engine
            .forward_batch_tokenized_into(&batch, &mut flat_logits)
            .unwrap();

        let gating = answer.gating.expect("gating メタデータが存在する。");
        let energy = gating.energy.expect("energy 値が存在する。");
        let _choice = answer.choice.unwrap_or_default();
        ood_energies.push(energy);
        println!(
            "  OOD | E = {:>7.3} | Conf = {:.3} | Logits = {:<25?} | {}",
            energy, gating.confidence, flat_logits, sample.description
        );
    }

    // 統計量
    id_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ood_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let id_min = id_energies.first().copied().unwrap();
    let id_max = id_energies.last().copied().unwrap();
    let id_mean = id_energies.iter().sum::<f64>() / id_energies.len() as f64;
    let id_p90 = id_energies[(id_energies.len() as f64 * 0.90) as usize];

    let ood_min = ood_energies.first().copied().unwrap();
    let ood_max = ood_energies.last().copied().unwrap();
    let ood_mean = ood_energies.iter().sum::<f64>() / ood_energies.len() as f64;
    let ood_p10 = ood_energies[(ood_energies.len() as f64 * 0.10) as usize];

    println!("\n--- [統計比較サマリー: {}] ---", tier_name);
    println!(
        "  ID  (分布内): Min={:.3}, Mean={:.3}, P90={:.3}, Max={:.3}",
        id_min, id_mean, id_p90, id_max
    );
    println!(
        "  OOD (分布外): Min={:.3}, P10={:.3}, Mean={:.3}, Max={:.3}",
        ood_min, ood_p10, ood_mean, ood_max
    );
    println!(
        "  エネルギー分離マージン (ΔE_mean): {:.3}",
        ood_mean - id_mean
    );

    // AUROC (Area Under ROC) の計算
    // OOD の方がエネルギーが高いと仮定し、すべての (OOD, ID) ペアで E_ood > E_id の比率を算出
    let mut pairs_correct = 0.0;
    let total_pairs = (id_energies.len() * ood_energies.len()) as f64;
    for &e_ood in &ood_energies {
        for &e_id in &id_energies {
            if e_ood > e_id {
                pairs_correct += 1.0;
            } else if (e_ood - e_id).abs() < 1e-6 {
                pairs_correct += 0.5;
            }
        }
    }
    let auroc = pairs_correct / total_pairs;
    println!("  AUROC (判別能): {:.4} ({:.2}%)", auroc, auroc * 100.0);

    // 閾値スイープテーブルの表示
    println!("\n  [閾値スイープ解析 (tau_energy)]");
    println!("  | 閾値 (tau) | OOD 検知率 (TPR) | ID 誤棄却率 (FPR) | 判定評価 |");
    println!("  | :---: | :---: | :---: | :--- |");
    let test_thresholds = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    for &th in &test_thresholds {
        let tpr = (ood_energies.iter().filter(|&&e| e > th).count() as f64
            / ood_energies.len() as f64)
            * 100.0;
        let fpr = (id_energies.iter().filter(|&&e| e > th).count() as f64
            / id_energies.len() as f64)
            * 100.0;
        let status = if tpr >= 90.0 && fpr <= 15.0 {
            "★ 最適バランス達成"
        } else if tpr >= 90.0 {
            "OOD 90% 達成 (FPR 高め)"
        } else {
            "検知率不足"
        };
        println!(
            "  |  {:>5.1}     |      {:>5.1}%     |      {:>5.1}%     | {} |",
            th, tpr, fpr, status
        );
    }

    (id_energies, ood_energies)
}

#[test]
fn test_ood_safety_valve_benchmark() {
    let root = workspace_root();

    // 1. Tier 1 (130M-INT8) の評価 (目標達成の検証)
    let tier1_dir = root.join("models").join("quantized");
    if let Some((engine1, tok1, calib1)) = init_engine(&tier1_dir) {
        let (id1, ood1) =
            analyze_model_energy_distribution(&engine1, &tok1, &calib1, "Tier 1 (130M-INT8)");

        let mut pairs_correct = 0.0;
        for &e_ood in &ood1 {
            for &e_id in &id1 {
                if e_ood > e_id {
                    pairs_correct += 1.0;
                } else if (e_ood - e_id).abs() < 1e-6 {
                    pairs_correct += 0.5;
                }
            }
        }
        let auroc1 = pairs_correct / (id1.len() * ood1.len()) as f64;
        assert!(
            auroc1 >= 0.90,
            "Tier 1 AUROC が 0.90 以上であること。実際: {:.4}。",
            auroc1
        );

        // 閾値 tau = 2.5 における OOD 検知率 90% 以上 (Exit Criteria 4 達成検証)
        let tpr1 = (ood1.iter().filter(|&&e| e > 2.5).count() as f64 / ood1.len() as f64) * 100.0;
        assert!(
            tpr1 >= 90.0,
            "Tier 1 OOD 検知率が 90.0% 以上であること。実際: {:.1}%。",
            tpr1
        );
    }

    // 2. Tier 2 (310M-INT8) の評価 (最適化されたエネルギー温度とハイブリッドOOD安全弁の検証)
    let tier2_dir = root.join("models").join("modernbert-310m-int8");
    if let Some((engine2, tok2, calib2)) = init_engine(&tier2_dir) {
        let (id2, ood2) =
            analyze_model_energy_distribution(&engine2, &tok2, &calib2, "Tier 2 (310M-INT8)");

        let mut pairs_correct = 0.0;
        for &e_ood in &ood2 {
            for &e_id in &id2 {
                if e_ood > e_id {
                    pairs_correct += 1.0;
                } else if (e_ood - e_id).abs() < 1e-6 {
                    pairs_correct += 0.5;
                }
            }
        }
        let auroc2 = pairs_correct / (id2.len() * ood2.len()) as f64;
        println!("Tier 2 (310M-INT8) AUROC: {:.4} (目標 0.80 以上)。", auroc2);
        assert!(
            auroc2 >= 0.80,
            "Tier 2 AUROC が 0.80 以上であること。実際: {:.4}。",
            auroc2
        );

        // calibration.json の設定に基づく GatingConfig での実推論 OOD 検知率検証
        let (id_samples, ood_samples) = build_evaluation_datasets();
        let coarse_config = CoarseToFineConfig::default();
        let gating_config2 = calib2.gating_config();

        let mut ood_detected = 0;
        for sample in &ood_samples {
            let mut criteria_map = IndexMap::new();
            for (k, v) in &sample.criteria {
                criteria_map.insert(k.to_string(), v.to_string());
            }
            let question = Question::new_choice(sample.instructions, criteria_map);
            let answer = engine2
                .evaluate_question_coarse_to_fine_with_gating(
                    &tok2,
                    sample.state,
                    &question,
                    &calib2,
                    &coarse_config,
                    Some(&gating_config2),
                )
                .expect("推論に成功した。");

            let gating = answer.gating.expect("gating メタデータが存在する。");
            if gating.is_ood {
                ood_detected += 1;
            }
        }

        let tpr2 = (ood_detected as f64 / ood_samples.len() as f64) * 100.0;
        println!(
            "Tier 2 (310M-INT8) ハイブリッド OOD 検知率: {:.1}% ({}/{}) (目標 90.0% 以上)。",
            tpr2,
            ood_detected,
            ood_samples.len()
        );

        // ID サンプルの誤棄却率 (FPR) も計測
        let mut id_rejected = 0;
        for sample in &id_samples {
            let mut criteria_map = IndexMap::new();
            for (k, v) in &sample.criteria {
                criteria_map.insert(k.to_string(), v.to_string());
            }
            let question = Question::new_choice(sample.instructions, criteria_map);
            let answer = engine2
                .evaluate_question_coarse_to_fine_with_gating(
                    &tok2,
                    sample.state,
                    &question,
                    &calib2,
                    &coarse_config,
                    Some(&gating_config2),
                )
                .expect("推論に成功した。");

            let gating = answer.gating.expect("gating メタデータが存在する。");
            if gating.is_ood {
                id_rejected += 1;
            }
        }
        let fpr2 = (id_rejected as f64 / id_samples.len() as f64) * 100.0;
        println!(
            "Tier 2 (310M-INT8) ID 誤棄却率 (FPR): {:.1}% ({}/{})。",
            fpr2,
            id_rejected,
            id_samples.len()
        );

        assert!(
            tpr2 >= 90.0,
            "Tier 2 OOD 検知率が 90.0% 以上であること。実際: {:.1}%。",
            tpr2
        );
    }
}
