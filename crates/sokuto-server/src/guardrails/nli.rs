//! # 語彙正規化・エンティティ投影型 Post-Execution ハルシネーション検証モジュール
//!
//! System 2 (上位 LLM) が生成した主張文 (Claim) と前提文 (State) の整合性を検証する。
//!
//! ## 二段構えの検証パイプライン
//! 1. **エンティティ・数値整合性検査 (`Entity Mismatch Flag`)**:
//!    `arithmetic.rs` と連動し、数値・金額・パーセンテージ・日付表現を正規化抽出。
//!    State と Claim の間で数値や単位が食い違っている場合、NLI の埋め込み空間の曖昧さを
//!    待たずに即座に `Entity Mismatch Flag` を立てて遮断する。
//! 2. **同義語カバレッジ連動の Soft Margin Calibration (動的マージン補正)**:
//!    正当な言い換えによる過剰遮断 (偽陽性) を防止するため、同義語カバレッジに応じて
//!    要求含意確率の閾値を動的に緩和する:
//!    $$\tau_{\text{effective}} = \tau_0 - \beta \cdot \text{SynonymCoverage}(\text{State}, \text{Claim})$$

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

/// 数値・通貨・パーセント抽出用正規表現 (全角数字・ピリオド・カンマ対応)。
static NUMERIC_ENTITY_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?P<val>[0-9０-９]+(?:[,\.，．][0-9０-９]+)*)\s*(?P<unit>万|億|兆|円|ドル|%|％|個|本|件|人|倍)?")
        .expect("数値エンティティ抽出正規表現のコンパイルに成功すること。")
});

/// 年月日日付エンティティ抽出用正規表現 (2026年10月4日, 2026-10-04, 2026/10/04 等)。
static DATE_ENTITY_REGEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?P<year>[0-9０-９]{4})\s*[年\-/]\s*(?P<month>[0-9０-９]{1,2})\s*[月\-/]\s*(?P<day>[0-9０-９]{1,2})\s*日?")
        .expect("日付エンティティ抽出正規表現のコンパイルに成功すること。")
});

/// 代表的な日本語同義語・言い換えペアリスト。
static SYNONYM_PAIRS: LazyLock<Vec<(&'static str, &'static str)>> = LazyLock::new(|| {
    vec![
        ("倍増", "100%増"),
        ("倍増", "2倍"),
        ("二倍", "2倍"),
        ("減少", "低下"),
        ("増加", "上昇"),
        ("完了", "終了"),
        ("完了", "済"),
        ("開始", "スタート"),
        ("売上", "売上高"),
        ("利益", "純利益"),
        ("不具合", "バグ"),
        ("欠陥", "不良"),
        ("承認", "許可"),
        ("拒否", "却下"),
    ]
});

/// ハルシネーション検証設定構造体。
#[derive(Debug, Clone, PartialEq)]
pub struct HallucinationVerifierConfig {
    /// ハルシネーション検証を有効にするか。
    pub enabled: bool,
    /// 基準要求含意確率閾値 $\tau_0$ (デフォルト: 0.60)。
    /// 含意確率 $P(\text{Entailment}) < \tau_{\text{effective}}$ でハルシネーションと判定。
    pub base_threshold: f64,
    /// 同義語カバレッジによる緩和係数 $\beta$ (デフォルト: 0.20)。
    pub synonym_margin_beta: f64,
    /// 数値・エンティティ不一致時の即時遮断フラグを有効にするか。
    pub enable_entity_mismatch_check: bool,
}

impl Default for HallucinationVerifierConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            base_threshold: 0.60,
            synonym_margin_beta: 0.20,
            enable_entity_mismatch_check: true,
        }
    }
}

/// 抽出された正規化数値エンティティ。
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedEntity {
    /// 表層文字列。
    pub raw: String,
    /// 正規化数値。
    pub normalized_value: f64,
    /// 単位 (存在する場合)。
    pub unit: Option<String>,
}

/// 出力ハルシネーション検証結果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HallucinationVerificationResult {
    /// 前提文に対する含意確率 $P(\text{Entailment} \mid S, c)$。
    pub entailment_prob: f64,
    /// 中立確率 $P(\text{Neutral} \mid S, c)$。
    pub neutral_prob: f64,
    /// 矛盾確率 $P(\text{Contradiction} \mid S, c)$。
    pub contradiction_prob: f64,
    /// ハルシネーションスコア $H(c \mid S) = 1.0 - P(\text{Entailment})$。
    pub hallucination_score: f64,
    /// 動的補正後の実効閾値 $\tau_{\text{effective}}$。
    pub effective_threshold: f64,
    /// 同義語カバレッジ比率。
    pub synonym_coverage: f64,
    /// 数値・固有表現不一致即死フラグが発火したか。
    pub entity_mismatch: bool,
    /// ハルシネーション (不整合) と判定されたか。
    pub is_hallucination: bool,
    /// 遮断または警告理由。
    pub reason: Option<String>,
}

/// テキストから数値および日付エンティティを正規化抽出する。
pub fn extract_normalized_entities(text: &str) -> Vec<NormalizedEntity> {
    let mut entities = Vec::new();

    // 1. 年月日日付エンティティの抽出 (YYYYMMDD 整数値として正規化)
    for caps in DATE_ENTITY_REGEX.captures_iter(text) {
        let (Some(y_m), Some(m_m), Some(d_m)) =
            (caps.name("year"), caps.name("month"), caps.name("day"))
        else {
            continue;
        };

        let parse_digits = |s: &str| -> Option<u32> {
            let clean = s
                .chars()
                .map(|c| match c {
                    '０'..='９' => (c as u32 - '０' as u32 + '0' as u32) as u8 as char,
                    other => other,
                })
                .collect::<String>();
            clean.parse::<u32>().ok()
        };

        if let (Some(y), Some(m), Some(d)) = (
            parse_digits(y_m.as_str()),
            parse_digits(m_m.as_str()),
            parse_digits(d_m.as_str()),
        ) {
            let date_val = (y * 10_000 + m * 100 + d) as f64;
            entities.push(NormalizedEntity {
                raw: caps
                    .get(0)
                    .map(|m| m.as_str().to_string())
                    .unwrap_or_default(),
                normalized_value: date_val,
                unit: Some("date".to_string()),
            });
        }
    }

    // 2. 数値・通貨・パーセントエンティティの抽出
    for caps in NUMERIC_ENTITY_REGEX.captures_iter(text) {
        let Some(val_match) = caps.name("val") else {
            continue;
        };

        // 全角数字、全角カンマ、全角ピリオドを半角へ正規化
        let clean_val_str = val_match
            .as_str()
            .replace([',', '，'], "")
            .chars()
            .map(|c| match c {
                '０'..='９' => (c as u32 - '０' as u32 + '0' as u32) as u8 as char,
                '．' => '.',
                other => other,
            })
            .collect::<String>();

        let Ok(mut val) = clean_val_str.parse::<f64>() else {
            continue;
        };

        let unit = caps.name("unit").map(|m| m.as_str().to_string());

        // 単位に応じた桁数正規化 (万 -> 1e4, 億 -> 1e8, 兆 -> 1e12)
        if let Some(ref u) = unit {
            match u.as_str() {
                "万" => val *= 10_000.0,
                "億" => val *= 100_000_000.0,
                "兆" => val *= 1_000_000_000_000.0,
                _ => {}
            }
        }

        entities.push(NormalizedEntity {
            raw: caps
                .get(0)
                .map(|m| m.as_str().to_string())
                .unwrap_or_default(),
            normalized_value: val,
            unit,
        });
    }

    entities
}

/// State と Claim の数値エンティティを照合し、不整合 (未記載の数値や食い違い) があるか判定する。
///
/// Claim 側に現れる数値が State 側に存在しない場合、または値が異なる場合に不整合と判定する。
pub fn check_entity_mismatch(state: &str, claim: &str) -> bool {
    let state_entities = extract_normalized_entities(state);
    let claim_entities = extract_normalized_entities(claim);

    if claim_entities.is_empty() {
        return false;
    }

    // State 側に存在する正規化値の集合
    let state_values: Vec<f64> = state_entities.iter().map(|e| e.normalized_value).collect();

    for claim_ent in &claim_entities {
        // 浮動小数点誤差を許容 (相対誤差 1e-4 以内)
        let found = state_values
            .iter()
            .any(|&sv| (sv - claim_ent.normalized_value).abs() / (sv.abs().max(1.0)) < 1e-4);

        if !found {
            // 同義語表現 (例: state に「倍増」があり、claim_ent が「2倍」「100%増」の場合) を救済
            let is_covered_by_synonym = SYNONYM_PAIRS.iter().any(|(s1, s2)| {
                (state.contains(s1)
                    && claim.contains(s2)
                    && (claim_ent.raw.contains(s2) || s2.contains(&claim_ent.raw)))
                    || (state.contains(s2)
                        && claim.contains(s1)
                        && (claim_ent.raw.contains(s1) || s1.contains(&claim_ent.raw)))
            });

            if !is_covered_by_synonym {
                // Claim に State にない数値が含まれている
                return true;
            }
        }
    }

    false
}

/// State と Claim 間の同義語カバレッジ比率を算出する。
pub fn compute_synonym_coverage(state: &str, claim: &str) -> f64 {
    let mut state_words = HashSet::new();
    for (s1, s2) in SYNONYM_PAIRS.iter() {
        if state.contains(s1) {
            state_words.insert(*s1);
            state_words.insert(*s2);
        }
        if state.contains(s2) {
            state_words.insert(*s1);
            state_words.insert(*s2);
        }
    }

    if state_words.is_empty() {
        return 0.0;
    }

    let mut matches = 0;
    let mut total_claim_terms = 0;

    for (s1, s2) in SYNONYM_PAIRS.iter() {
        let has_s1 = claim.contains(s1);
        let has_s2 = claim.contains(s2);
        if has_s1 || has_s2 {
            total_claim_terms += 1;
            if (has_s1 && state_words.contains(s1)) || (has_s2 && state_words.contains(s2)) {
                matches += 1;
            }
        }
    }

    if total_claim_terms == 0 {
        0.0
    } else {
        (matches as f64) / (total_claim_terms as f64)
    }
}

/// NLI ロジット (含意, 中立, 矛盾) と前提・主張テキストからハルシネーション検証を実行する。
///
/// # 引数
/// - `state`: 前提文 (Context / State)。
/// - `claim`: 生成主張文 (Generated Claim)。
/// - `nli_logits`: [含意, 中立, 矛盾] の 3 クラス未正規化ロジット。
/// - `config`: 検証設定。
pub fn verify_claim_nli(
    state: &str,
    claim: &str,
    nli_logits: [f64; 3],
    config: &HallucinationVerifierConfig,
) -> HallucinationVerificationResult {
    // Softmax 計算
    let max_logit = nli_logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_e = (nli_logits[0] - max_logit).exp();
    let exp_n = (nli_logits[1] - max_logit).exp();
    let exp_c = (nli_logits[2] - max_logit).exp();
    let sum_exp = exp_e + exp_n + exp_c;

    let p_entailment = exp_e / sum_exp;
    let p_neutral = exp_n / sum_exp;
    let p_contradiction = exp_c / sum_exp;

    let hallucination_score = 1.0 - p_entailment;

    // 1. エンティティ照合
    let entity_mismatch = if config.enable_entity_mismatch_check {
        check_entity_mismatch(state, claim)
    } else {
        false
    };

    // 2. 同義語カバレッジと動的マージン補正
    let synonym_cov = compute_synonym_coverage(state, claim);
    // 同義語が多いほど要求含意確率閾値 tau を緩和 (引き下げ) する
    let effective_threshold =
        (config.base_threshold - config.synonym_margin_beta * synonym_cov).max(0.10);

    // 含意確率が要求閾値を下回るか、エンティティ不一致の場合にハルシネーションと判定
    let is_hallucination = entity_mismatch || (p_entailment < effective_threshold);

    let reason = if entity_mismatch {
        Some("数値・エンティティの不整合を検知しました (Entity Mismatch)。".to_string())
    } else if is_hallucination {
        Some(format!(
            "含意確率 ({:.3}) が要求閾値 ({:.3}) を下回りました (ハルシネーションスコア: {:.3})。",
            p_entailment, effective_threshold, hallucination_score
        ))
    } else {
        None
    };

    HallucinationVerificationResult {
        entailment_prob: p_entailment,
        neutral_prob: p_neutral,
        contradiction_prob: p_contradiction,
        hallucination_score,
        effective_threshold,
        synonym_coverage: synonym_cov,
        entity_mismatch,
        is_hallucination,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_normalized_entities_scales() {
        let text = "契約金額は 1,000万円で、売上は 5億円、前年比 150% です。";
        let entities = extract_normalized_entities(text);
        assert_eq!(entities.len(), 3);
        assert_eq!(entities[0].normalized_value, 10_000_000.0);
        assert_eq!(entities[1].normalized_value, 500_000_000.0);
        assert_eq!(entities[2].normalized_value, 150.0);
    }

    #[test]
    fn test_entity_mismatch_flag() {
        let state = "契約金は 1,000万円です。";
        let claim_ok = "金額は 10,000,000円と合意されました。";
        assert!(!check_entity_mismatch(state, claim_ok));

        let claim_bad = "契約金は 1,000億円です。";
        assert!(check_entity_mismatch(state, claim_bad));
    }

    #[test]
    fn test_synonym_coverage_and_soft_margin() {
        let state = "今期の売上は前年比倍増しました。";
        let claim = "売上高は 2倍になりました。";
        let cov = compute_synonym_coverage(state, claim);
        assert!(cov > 0.0);

        let config = HallucinationVerifierConfig::default();
        // 微妙なロジットでも同義語緩和で通過することを検証
        let res = verify_claim_nli(state, claim, [1.5, 0.5, -1.0], &config);
        assert!(res.effective_threshold < config.base_threshold);
        assert!(!res.is_hallucination);
    }

    #[test]
    fn test_extract_normalized_decimal_and_dates() {
        let text = "締結日は 2026年10月4日、投資額は 1.5億円 (全角: １．５億円) です。";
        let entities = extract_normalized_entities(text);

        // 日付 1 件 + 数値 2 件 = 3 件
        assert!(
            entities
                .iter()
                .any(|e| (e.normalized_value - 20261004.0).abs() < 1e-4)
        );
        assert!(
            entities
                .iter()
                .any(|e| (e.normalized_value - 150_000_000.0).abs() < 1e-4)
        );

        // 日付不整合の検知 (2026年10月4日 vs 2026年10月5日)
        let state = "締結日は 2026年10月4日です。";
        let claim_bad_date = "締結日は 2026年10月5日です。";
        assert!(check_entity_mismatch(state, claim_bad_date));

        let claim_good_date = "締結日は 2026-10-04 です。";
        assert!(!check_entity_mismatch(state, claim_good_date));
    }
}
