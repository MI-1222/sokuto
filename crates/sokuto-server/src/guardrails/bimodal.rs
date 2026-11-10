//! # 二峰性 (バイモーダル) 分布検出モジュール
//!
//! 順序尺度 (Score 型) の事後確率分布を検査し、「低評価と高評価に真っ向から分裂」
//! している意見二極化インシデントを数理的 (双峰性係数 BC および最大分散比率) に検知する。
//! 検知時は Gating 安全弁を作動させ、誤った自動実行 (AutoExecute) を防止して
//! `ConfirmOrEscalate` へ強制降格する。

use std::borrow::Borrow;

use indexmap::IndexMap;
use sokuto_core::error::Result;
use sokuto_core::gating::DecisionRoute;
use sokuto_core::math::{
    bimodality_coefficient, expected_score, score_variance, score_variance_ratio,
};
use sokuto_core::schema::{Answer, Question, QuestionType};

/// 二峰性分布検出設定構造体。
#[derive(Debug, Clone, PartialEq)]
pub struct BimodalConfig {
    /// 二峰性検出ガードレールの有効化フラグ。
    pub enabled: bool,
    /// 双峰性係数 (BC) の判定閾値 (デフォルト: 0.555、一様分布基準)。
    pub bc_threshold: f64,
    /// 理論最大分散に対する比率閾値 (デフォルト: 0.35)。
    pub var_ratio_threshold: f64,
    /// 二峰性判定を適用する最小段階数 (デフォルト: 3)。
    pub min_categories: usize,
}

impl Default for BimodalConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bc_threshold: 0.555,
            var_ratio_threshold: 0.35,
            min_categories: 3,
        }
    }
}

/// 二峰性検出レポート構造体。
#[derive(Debug, Clone, PartialEq)]
pub struct BimodalReport {
    /// 二峰性意見対立が検出されたか。
    pub is_bimodal: bool,
    /// 算出された双峰性係数 (BC)。
    pub bc: f64,
    /// 理論最大分散に対する事後分散比率。
    pub var_ratio: f64,
    /// 事後分散。
    pub variance: f64,
    /// 加重平均期待値スコア。
    pub expected_score: f64,
}

/// 二峰性検出エンジン。
#[derive(Debug, Clone)]
pub struct BimodalDetector {
    config: BimodalConfig,
}

impl BimodalDetector {
    /// 新規 `BimodalDetector` を生成する。
    pub fn new(config: BimodalConfig) -> Self {
        Self { config }
    }

    /// 設定への参照を取得する。
    pub fn config(&self) -> &BimodalConfig {
        &self.config
    }

    /// 確率分布の局所極大値 (ピーク) の個数を $O(M)$ で走査・カウントする。
    ///
    /// 端点 ($k=0, M-1$) および内部の極大値を検出し、単峰性 (単調増加・単調減少・中央山型) と
    /// 意見二極化 (2 箇所以上のピーク) を幾何学的に識別する。
    pub fn count_local_peaks(probabilities: &[f64]) -> usize {
        let m = probabilities.len();
        if m < 2 {
            return m;
        }

        let mut peaks = 0;

        // 左端 (k = 0)
        if probabilities[0] > probabilities[1] + 1e-4 {
            peaks += 1;
        }

        // 内部 (0 < k < m - 1)
        for k in 1..m - 1 {
            if probabilities[k] > probabilities[k - 1] + 1e-4
                && probabilities[k] > probabilities[k + 1] + 1e-4
            {
                peaks += 1;
            }
        }

        // 右端 (k = m - 1)
        if probabilities[m - 1] > probabilities[m - 2] + 1e-4 {
            peaks += 1;
        }

        peaks
    }

    /// 確率分布スライスを検査し、二峰性レポートを生成する。
    ///
    /// # 判定基準 (ハイブリッド判定)
    /// 1. 段階数が `min_categories` 以上であること。
    /// 2. 確率分布の局所ピーク数が 2 箇所以上存在すること (単調推移や単一山型を排除)。
    /// 3. 双峰性係数 $BC > \text{bc\_threshold}$ (一様分布水準 0.555 を超過)。
    /// 4. 分散比率 $\text{VarRatio} > \text{var\_ratio\_threshold}$ (理論最大分散の 35% 以上)。
    pub fn detect(&self, probabilities: &[f64]) -> Result<BimodalReport> {
        let m = probabilities.len();
        if m < 2 {
            return Ok(BimodalReport {
                is_bimodal: false,
                bc: 0.0,
                var_ratio: 0.0,
                variance: 0.0,
                expected_score: 0.0,
            });
        }

        let mean = expected_score(probabilities)?;
        let var = score_variance(probabilities)?;
        let var_ratio = score_variance_ratio(probabilities)?;
        let bc = bimodality_coefficient(probabilities)?;
        let peaks = Self::count_local_peaks(probabilities);

        let is_bimodal = self.config.enabled
            && m >= self.config.min_categories
            && peaks >= 2
            && bc > self.config.bc_threshold
            && var_ratio > self.config.var_ratio_threshold;

        Ok(BimodalReport {
            is_bimodal,
            bc,
            var_ratio,
            variance: var,
            expected_score: mean,
        })
    }

    /// 任意の確率値イテレータを検査し、二峰性レポートを生成する。
    ///
    /// # 備考
    /// - スタックバッファを活用し、Score 型の通常スコープ ($M \le 10$) において
    ///   ヒープアロケーションなしで二峰性を検査する。
    pub fn detect_iter<I>(&self, iter: I) -> Result<BimodalReport>
    where
        I: IntoIterator,
        I::Item: Borrow<f64>,
    {
        const STACK_CAPACITY: usize = 16;
        let mut stack_buf = [0.0; STACK_CAPACITY];
        let mut heap_buf: Option<Vec<f64>> = None;
        let mut m = 0;

        for item in iter {
            let p = *item.borrow();
            if m < STACK_CAPACITY {
                stack_buf[m] = p;
            } else {
                let vec = heap_buf.get_or_insert_with(|| {
                    let mut v = Vec::with_capacity(32);
                    v.extend_from_slice(&stack_buf);
                    v
                });
                vec.push(p);
            }
            m += 1;
        }

        let slice = if let Some(ref v) = heap_buf {
            v.as_slice()
        } else {
            &stack_buf[..m]
        };

        self.detect(slice)
    }
}

impl Default for BimodalDetector {
    fn default() -> Self {
        Self::new(BimodalConfig::default())
    }
}

/// 推論結果の各 Answer を走査し、Score 型の二峰性意見対立を検知した場合は Gating を降格する。
///
/// # 挙動仕様
/// - Score 型の質問であり、かつ確率分布が存在する場合に二峰性判定を実施する。
/// - 二峰性対立が検知された場合：
///   - Gating ルートが `AutoExecute` であれば、`ConfirmOrEscalate` へ強制降格する。
///   - 確信度 (`confidence`) を抑制し、安全マージンを確保する。
///   - 監査理由 (`reason`) に二峰性検出ログを付加する。
pub fn verify_and_adjust_bimodal(
    answers: &mut IndexMap<String, Answer>,
    questions: &IndexMap<String, Question>,
    detector: &BimodalDetector,
) {
    if !detector.config.enabled {
        return;
    }

    for (qid, answer) in answers.iter_mut() {
        let Some(question) = questions.get(qid) else {
            continue;
        };

        if question.question_type != QuestionType::Score {
            continue;
        }

        let Some(probs_map) = answer.probabilities() else {
            continue;
        };

        let Ok(report) = detector.detect_iter(probs_map.values()) else {
            continue;
        };

        if report.is_bimodal {
            tracing::warn!(
                question_id = %qid,
                bc = report.bc,
                var_ratio = report.var_ratio,
                expected_score = report.expected_score,
                "Score 型判定において意見二極化 (バイモーダル分布) を検出しました。"
            );

            // Gating メタデータへの介入 (安全弁降格と確信度抑制)
            if let Some(ref mut gating) = answer.gating {
                gating.confidence = gating.confidence.min(0.20);

                if gating.route == DecisionRoute::AutoExecute {
                    gating.route = DecisionRoute::ConfirmOrEscalate;
                }

                let bimodal_reason = format!(
                    "[Bimodal Contradiction] 意見二極化(バイモーダル分布)を検出したため二次検証を要求 (BC: {:.3} > {:.3}, VarRatio: {:.3} > {:.3})。",
                    report.bc,
                    detector.config.bc_threshold,
                    report.var_ratio,
                    detector.config.var_ratio_threshold
                );

                if gating.reason.is_empty() {
                    gating.reason = bimodal_reason;
                } else {
                    gating.reason = format!("{} | {}", bimodal_reason, gating.reason);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sokuto_core::gating::GatingMetadata;

    #[test]
    fn test_bimodal_detector_extremes() {
        let detector = BimodalDetector::default();

        // 1. 完全両端対立: [0.5, 0.0, 0.0, 0.5] -> 二峰性検知
        let p_bimodal = vec![0.5, 0.0, 0.0, 0.5];
        let report = detector.detect(&p_bimodal).unwrap();
        assert!(report.is_bimodal);
        assert!((report.bc - 1.0).abs() < 1e-6);
        assert!((report.var_ratio - 1.0).abs() < 1e-6);

        // 2. ワンホット (完全単峰): [0.0, 1.0, 0.0, 0.0] -> 検知なし
        let p_onehot = vec![0.0, 1.0, 0.0, 0.0];
        let report_onehot = detector.detect(&p_onehot).unwrap();
        assert!(!report_onehot.is_bimodal);
        assert_eq!(report_onehot.bc, 0.0);

        // 3. 隣接割れ (低分散): [0.0, 0.5, 0.5, 0.0] -> 分散比率が低いため検知なし
        let p_adjacent = vec![0.0, 0.5, 0.5, 0.0];
        let report_adjacent = detector.detect(&p_adjacent).unwrap();
        assert!(!report_adjacent.is_bimodal);
        assert!(report_adjacent.var_ratio < 0.20);

        // 4. 一様分布: [0.25, 0.25, 0.25, 0.25] -> 分散比率は中程度だが BC は 0.61 付近
        // VarRatio = 1.25 / 2.25 ≈ 0.555 -> 4段階の一様分布
        let p_uniform = vec![0.25, 0.25, 0.25, 0.25];
        let _report_uniform = detector.detect(&p_uniform).unwrap();
    }

    #[test]
    fn test_verify_and_adjust_bimodal_demotion() {
        let detector = BimodalDetector::default();

        let mut questions = IndexMap::new();
        questions.insert(
            "q_score".to_string(),
            Question::new_score(
                "障害深刻度を評価せよ。".to_string(),
                vec![
                    "軽微".to_string(),
                    "中度".to_string(),
                    "重大".to_string(),
                    "致命的".to_string(),
                ],
            ),
        );

        let mut probs = IndexMap::new();
        probs.insert("軽微".to_string(), 0.50);
        probs.insert("中度".to_string(), 0.0);
        probs.insert("重大".to_string(), 0.0);
        probs.insert("致命的".to_string(), 0.50);

        let mut answer = Answer::score(1.5, probs, 0.95);
        answer.gating = Some(GatingMetadata {
            route: DecisionRoute::AutoExecute,
            confidence: 0.95,
            entropy: None,
            margin: None,
            energy: None,
            is_ood: false,
            reason: "高確信度判定".to_string(),
            escalation: None,
        });

        let mut answers = IndexMap::new();
        answers.insert("q_score".to_string(), answer);

        // ガードレール後処理適用
        verify_and_adjust_bimodal(&mut answers, &questions, &detector);

        let adjusted = answers.get("q_score").unwrap();
        let gating = adjusted.gating.as_ref().unwrap();

        // AutoExecute から ConfirmOrEscalate へ強制降格されていること
        assert_eq!(gating.route, DecisionRoute::ConfirmOrEscalate);
        assert_eq!(gating.confidence, 0.20);
        assert!(gating.reason.contains("[Bimodal Contradiction]"));
        assert_eq!(adjusted.confidence(), Some(0.95));
    }
}
