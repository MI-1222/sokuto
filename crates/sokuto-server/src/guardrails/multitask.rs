//! # マルチタスク動的マーカー並列スキャンモジュール (1 パス推論)
//!
//! インジェクション (INJ)、有害性 (TOX)、個人情報漏洩 (PII) の 3 項目を個別推論せず、
//! 系列先頭に `[INJ]`, `[TOX]`, `[PII]` の動的マーカーを配置して単一のフォワードパスで同時評価する。
//! これにより推論パス数を 3 回から 1 回に集約し、66% の計算量を削減する。
//! また、ヘルムホルツ自由エネルギー $F(x) = -T \ln \sum \exp(z_k / T)$ による OOD 異常検知を行う。

use serde::{Deserialize, Serialize};

use crate::guardrails::fast_fail::{GuardrailAction, GuardrailViolation};

/// 特殊マーカー定義。
pub const MARKER_INJECTION: &str = "[INJ]";
pub const MARKER_TOXICITY: &str = "[TOX]";
pub const MARKER_PII: &str = "[PII]";

/// マルチタスクスキャナー設定。
#[derive(Debug, Clone, PartialEq)]
pub struct MultiTaskScannerConfig {
    /// マルチタスクスキャンを有効にするか。
    pub enabled: bool,
    /// インジェクション判定閾値 (デフォルト: 0.5)。
    pub threshold_injection: f64,
    /// 有害性判定閾値 (デフォルト: 0.5)。
    pub threshold_toxicity: f64,
    /// PII 漏洩判定閾値 (デフォルト: 0.5)。
    pub threshold_pii: f64,
    /// ヘルムホルツ自由エネルギー閾値 (この値を超えると OOD と判定、デフォルト: -2.5)。
    pub threshold_energy: f64,
    /// 自由エネルギー計算時の温度パラメータ (デフォルト: 1.0)。
    pub temperature: f64,
    /// 違反時のアクション。
    pub action: GuardrailAction,
}

impl Default for MultiTaskScannerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold_injection: 0.5,
            threshold_toxicity: 0.5,
            threshold_pii: 0.5,
            threshold_energy: -2.5,
            temperature: 1.0,
            action: GuardrailAction::Block,
        }
    }
}

/// 単一パス並列スキャン評価結果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiTaskScanResult {
    /// プロンプトインジェクション確率 $P(\text{inj} = 1 \mid X)$。
    pub injection_prob: f64,
    /// 有害性確率 $P(\text{tox} = 1 \mid X)$。
    pub toxicity_prob: f64,
    /// 個人情報漏洩確率 $P(\text{pii} = 1 \mid X)$。
    pub pii_prob: f64,
    /// ヘルムホルツ自由エネルギー $F(x)$。
    pub free_energy: f64,
    /// 脅威が検知されたか。
    pub is_threat_detected: bool,
    /// 分布外 (OOD) または判断不確実であるか。
    pub is_ood: bool,
    /// 発生したガードレール違反一覧。
    pub violations: Vec<GuardrailViolation>,
}

/// マルチタスク推論用の合成プロンプト系列を構築する。
///
/// 先頭に `[INJ] [TOX] [PII]` マーカーを配置した文字列を返却する。
pub fn build_multitask_prompt(text: &str) -> String {
    format!(
        "{} {} {} {}",
        MARKER_INJECTION, MARKER_TOXICITY, MARKER_PII, text
    )
}

/// ロジット列からシグモイド確率を算出する。
#[inline]
pub fn sigmoid(logit: f64) -> f64 {
    1.0 / (1.0 + (-logit).exp())
}

/// 二値分類マルチタスクにおけるヘルムホルツ自由エネルギーを計算する。
///
/// 各タスク $k$ の二値決定状態 $\{0, 1\}$ (未正規化ロジット $[-z_k/2, z_k/2]$)
/// に対する全自由エネルギーを算出する。
/// 確信度が高い ($|z_k|$ が大きい) ほど自由エネルギーは低くなり、
/// 判断が曖昧または不確実な異常入力 ($z_k \approx 0$) では自由エネルギーが上昇する。
///
/// $$F(x; T) = -T \sum_{k=1}^K \ln \left( \exp\left(\frac{z_k}{2T}\right) + \exp\left(-\frac{z_k}{2T}\right) \right)$$
pub fn compute_free_energy(logits: &[f64], temperature: f64) -> f64 {
    if logits.is_empty() {
        return 0.0;
    }
    let t = if temperature <= 0.0 { 1.0 } else { temperature };

    let mut total_energy = 0.0;
    for &z in logits {
        let abs_z = z.abs();
        // LogSumExp トリックによる安定計算:
        // ln(exp(a) + exp(-a)) = a + ln(1 + exp(-2a)) (where a = abs_z / (2t))
        let a = abs_z / (2.0 * t);
        let log_sum_exp = a + (-2.0 * a).exp().ln_1p();
        total_energy += -t * log_sum_exp;
    }

    total_energy
}

/// ロジットから各タスクの確率と自由エネルギーを算出し、総合判定を下す。
///
/// # 引数
/// - `inj_logit`: `[INJ]` マーカー位置のロジット。
/// - `tox_logit`: `[TOX]` マーカー位置のロジット。
/// - `pii_logit`: `[PII]` マーカー位置のロジット。
/// - `config`: マルチタスク設定。
pub fn evaluate_multitask_logits(
    inj_logit: f64,
    tox_logit: f64,
    pii_logit: f64,
    config: &MultiTaskScannerConfig,
) -> MultiTaskScanResult {
    let inj_prob = sigmoid(inj_logit);
    let tox_prob = sigmoid(tox_logit);
    let pii_prob = sigmoid(pii_logit);

    let logits = [inj_logit, tox_logit, pii_logit];
    let free_energy = compute_free_energy(&logits, config.temperature);

    let mut violations = Vec::new();
    let mut threat_detected = false;

    if inj_prob >= config.threshold_injection {
        threat_detected = true;
        violations.push(GuardrailViolation {
            rule: "multitask_prompt_injection".to_string(),
            confidence: inj_prob,
            snippet: None,
            action_taken: config.action,
        });
    }

    if tox_prob >= config.threshold_toxicity {
        threat_detected = true;
        violations.push(GuardrailViolation {
            rule: "multitask_toxicity".to_string(),
            confidence: tox_prob,
            snippet: None,
            action_taken: config.action,
        });
    }

    if pii_prob >= config.threshold_pii {
        threat_detected = true;
        violations.push(GuardrailViolation {
            rule: "multitask_pii_leakage".to_string(),
            confidence: pii_prob,
            snippet: None,
            action_taken: config.action,
        });
    }

    let is_ood = free_energy > config.threshold_energy;
    if is_ood {
        violations.push(GuardrailViolation {
            rule: "multitask_free_energy_ood".to_string(),
            confidence: sigmoid(free_energy),
            snippet: Some(format!("free_energy={:.3}", free_energy)),
            action_taken: config.action,
        });
    }

    MultiTaskScanResult {
        injection_prob: inj_prob,
        toxicity_prob: tox_prob,
        pii_prob,
        free_energy,
        is_threat_detected: threat_detected,
        is_ood,
        violations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_multitask_prompt() {
        let prompt = build_multitask_prompt("ユーザー入力テスト");
        assert_eq!(prompt, "[INJ] [TOX] [PII] ユーザー入力テスト");
    }

    #[test]
    fn test_evaluate_multitask_safe() {
        let config = MultiTaskScannerConfig::default();
        // 低いロジット (すべて安全)
        let res = evaluate_multitask_logits(-5.0, -4.0, -6.0, &config);
        assert!(!res.is_threat_detected);
        assert!(res.injection_prob < 0.1);
        assert!(res.toxicity_prob < 0.1);
        assert!(res.pii_prob < 0.1);
    }

    #[test]
    fn test_evaluate_multitask_injection_violation() {
        let config = MultiTaskScannerConfig::default();
        // インジェクションロジットが高い
        let res = evaluate_multitask_logits(3.0, -4.0, -5.0, &config);
        assert!(res.is_threat_detected);
        assert!(res.injection_prob > 0.9);
        assert!(
            res.violations
                .iter()
                .any(|v| v.rule == "multitask_prompt_injection")
        );
    }

    #[test]
    fn test_free_energy_ood() {
        let config = MultiTaskScannerConfig {
            threshold_energy: -3.0,
            ..Default::default()
        };
        // 高確信度ロジット (安全側または脅威側): 低エネルギー (safe)
        let res_confident = evaluate_multitask_logits(-5.0, -4.0, -5.0, &config);
        assert!(!res_confident.is_ood);
        assert!(res_confident.free_energy < -3.0);

        // 曖昧・判断不能ロジット (0.0 近傍): 高エネルギー (OOD)
        let res_ambiguous = evaluate_multitask_logits(0.0, 0.0, 0.0, &config);
        assert!(res_ambiguous.is_ood);
        assert!(
            res_ambiguous
                .violations
                .iter()
                .any(|v| v.rule == "multitask_free_energy_ood")
        );
    }
}
