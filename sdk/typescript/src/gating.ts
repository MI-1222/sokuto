/**
 * 3軸 Pareto 最適ゲーティングエンジンモジュール。
 *
 * 単一 Softmax 確率への過信を排し、Top-Margin、正規化エントロピー、
 * およびヘルムホルツ自由エネルギーの 3 条件を同時評価して
 * 高確信度即時確定と System 2 エスカレーションを判定する。
 */

import type { DecisionRoute, GatingConfig, GatingDecision } from "./types.js";

/**
 * 確率分布からシャノンエントロピーおよび正規化エントロピーを算出する。
 *
 * @param probabilities 各候補の確率値列または確率マップ。
 * @returns [未正規化エントロピー, 0.0〜1.0 にクランプされた正規化エントロピー] のタプル。
 */
export function calculateEntropy(
  probabilities: number[] | Record<string, number>
): [number, number] {
  const probs = Array.isArray(probabilities)
    ? probabilities
    : Object.values(probabilities);

  const k = probs.length;
  if (k <= 1) {
    return [0.0, 0.0];
  }

  let entropy = 0.0;
  for (const p of probs) {
    if (p > 0.0) {
      const clampedP = Math.min(p, 1.0);
      entropy -= clampedP * Math.log(clampedP);
    }
  }

  const maxEntropy = Math.log(k);
  if (maxEntropy <= 0.0) {
    return [0.0, 0.0];
  }

  const normalized = entropy / maxEntropy;
  return [entropy, Math.max(0.0, Math.min(1.0, normalized))];
}

/**
 * 確率分布の上位2候補の差分 (Top-Margin) を算出する。
 *
 * @param probabilities 各候補の確率値列または確率マップ。
 * @returns 0.0〜1.0 にクランプされた Top-Margin。
 */
export function calculateTopMargin(
  probabilities: number[] | Record<string, number>
): number {
  const probs = Array.isArray(probabilities)
    ? probabilities
    : Object.values(probabilities);

  const k = probs.length;
  if (k === 0) {
    return 0.0;
  }
  if (k === 1) {
    return 1.0;
  }

  let max1 = -Infinity;
  let max2 = -Infinity;

  for (const p of probs) {
    const val = Math.max(0.0, Math.min(1.0, p));
    if (val > max1) {
      max2 = max1;
      max1 = val;
    } else if (val > max2) {
      max2 = val;
    }
  }

  const margin = max1 - max2;
  return Math.max(0.0, Math.min(1.0, margin));
}

/**
 * 未正規化生ロジット列からヘルムホルツ自由エネルギーを算出する。
 *
 * 数理仕様:
 * F(x; T) = -T * ln(sum(exp(z_k / T)))
 * オーバーフロー防止のため最大ロジット減算を行う。
 *
 * @param logits 未正規化ロジット列。
 * @param temperature 較正温度パラメータ (デフォルト: 1.0)。
 * @returns ヘルムホルツ自由エネルギー実数値。
 */
export function computeFreeEnergy(
  logits: number[],
  temperature = 1.0
): number {
  if (logits.length === 0) {
    return 0.0;
  }
  if (logits.length === 1) {
    return -logits[0];
  }

  const tau = Math.max(1e-4, temperature);
  const maxLogit = Math.max(...logits);

  let sumExp = 0.0;
  for (const z of logits) {
    sumExp += Math.exp((z - maxLogit) / tau);
  }

  if (sumExp <= 0.0) {
    return -maxLogit;
  }

  return -maxLogit - tau * Math.log(sumExp);
}

/**
 * Score 型確率分布から期待値、分散、分散指標、双峰性係数を算出する。
 *
 * @param probabilities 各段階レベルの確率分布列。
 * @returns [期待値スコア, 分散, 正規化分散指標, 双峰性係数] のタプル。
 */
export function calculateScoreMetrics(
  probabilities: number[]
): [number, number, number, number] {
  const m = probabilities.length;
  if (m < 2) {
    return [0.0, 0.0, 0.0, 0.0];
  }

  let mean = 0.0;
  for (let k = 0; k < m; k++) {
    mean += k * probabilities[k];
  }

  let variance = 0.0;
  let mu3 = 0.0;
  let mu4 = 0.0;

  for (let k = 0; k < m; k++) {
    const diff = k - mean;
    const diff2 = diff * diff;
    variance += diff2 * probabilities[k];
    mu3 += diff2 * diff * probabilities[k];
    mu4 += diff2 * diff2 * probabilities[k];
  }

  const maxVariance = ((m - 1.0) ** 2) / 4.0;
  const cVar = maxVariance > 0.0 ? 1.0 - variance / maxVariance : 1.0;
  const dispersion = 1.0 - Math.max(0.0, Math.min(1.0, cVar));

  if (variance < 1e-6) {
    return [mean, variance, dispersion, 0.0];
  }

  const sigma = Math.sqrt(variance);
  const sigma3 = variance * sigma;
  const sigma4 = variance * variance;

  if (sigma4 <= 0.0) {
    return [mean, variance, dispersion, 0.0];
  }

  const gamma = mu3 / sigma3;
  const kurtosis = mu4 / sigma4;
  if (kurtosis <= 0.0) {
    return [mean, variance, dispersion, 0.0];
  }

  const bimodality = (gamma * gamma + 1.0) / kurtosis;
  return [mean, variance, dispersion, Math.max(0.0, Math.min(1.0, bimodality))];
}

/**
 * 3軸 Pareto 最適ゲーティング評価を実行し、意思決定ルートを判定する。
 *
 * @param params 評価パラメータ。
 * @returns GatingDecision 判定結果。
 */
export function evaluateGating(params: {
  probabilities?: number[] | Record<string, number> | null;
  freeEnergy?: number | null;
  questionType?: string;
  config?: GatingConfig;
  labels?: string[];
  noulProbability?: number | null;
  serverGating?: {
    energy?: number;
    free_energy?: number;
    confidence?: number;
    route?: string;
    margin?: number;
    entropy?: number;
    is_ood?: boolean;
    reason?: string;
  } | null;
}): GatingDecision {
  const marginThreshold = params.config?.marginThreshold ?? 0.15;
  const entropyThreshold = params.config?.entropyThreshold ?? 0.65;
  const energyThreshold = params.config?.energyThreshold ?? -1.0;
  const scoreVarThreshold = params.config?.scoreVarThreshold ?? 0.50;
  const bimodalityThreshold = params.config?.bimodalityThreshold ?? 0.555;

  const qtype = (params.questionType ?? "choice").toLowerCase();

  let resolvedFreeEnergy = params.freeEnergy;
  if (resolvedFreeEnergy === undefined || resolvedFreeEnergy === null) {
    const serverEnergy = params.serverGating?.energy ?? params.serverGating?.free_energy;
    if (serverEnergy !== undefined && serverEnergy !== null) {
      resolvedFreeEnergy = serverEnergy;
    }
  }

  // 1. Noul 型 (真偽値)
  if (qtype === "noul") {
    let pTrue = params.noulProbability;
    if (pTrue === undefined || pTrue === null) {
      if (params.probabilities) {
        if (Array.isArray(params.probabilities) && params.probabilities.length > 0) {
          pTrue = params.probabilities[0];
        } else if (!Array.isArray(params.probabilities)) {
          pTrue = params.probabilities["true"] ?? params.probabilities["True"] ?? 0.5;
        }
      }
    }
    if (pTrue === undefined || pTrue === null) {
      pTrue = 0.5;
    }

    pTrue = Math.max(0.0, Math.min(1.0, pTrue));
    const pFalse = 1.0 - pTrue;

    const margin = Math.abs(2.0 * pTrue - 1.0);
    let hNorm = 0.0;
    if (pTrue > 0.0 && pTrue < 1.0) {
      const hRaw = -(pTrue * Math.log(pTrue) + pFalse * Math.log(pFalse));
      hNorm = Math.max(0.0, Math.min(1.0, hRaw / Math.log(2.0)));
    }

    const confidence = Math.max(pTrue, pFalse);
    const topCandidates: Array<[string, number]> = [
      [pTrue >= pFalse ? "true" : "false", Math.max(pTrue, pFalse)],
      [pTrue >= pFalse ? "false" : "true", Math.min(pTrue, pFalse)],
    ];

    if (margin >= marginThreshold && hNorm <= entropyThreshold) {
      return {
        route: "auto_execute",
        escalate: false,
        confidence,
        topMargin: margin,
        normalizedEntropy: hNorm,
        freeEnergy: resolvedFreeEnergy ?? undefined,
        isOod: false,
        reason: "Noul 条件充足 (高確信度真偽判定)。",
        topCandidates,
      };
    } else {
      return {
        route: "confirm_or_escalate",
        escalate: true,
        confidence,
        topMargin: margin,
        normalizedEntropy: hNorm,
        freeEnergy: resolvedFreeEnergy ?? undefined,
        isOod: false,
        reason: `Noul 境界事例検知 (Margin=${margin.toFixed(3)} < ${marginThreshold} または Entropy=${hNorm.toFixed(3)} > ${entropyThreshold})。`,
        topCandidates,
      };
    }
  }

  // 2. Score 型 (順序尺度)
  if (qtype === "score") {
    let probsList: number[] = [];
    if (params.probabilities) {
      probsList = Array.isArray(params.probabilities)
        ? params.probabilities
        : Object.values(params.probabilities);
    }

    if (probsList.length < 2) {
      return {
        route: "confirm_or_escalate",
        escalate: true,
        confidence: 0.5,
        isOod: false,
        reason: "Score 確率分布データ不足。",
        topCandidates: [],
      };
    }

    const [meanVal, , dispersion, bc] = calculateScoreMetrics(probsList);
    const confidence = 1.0 - dispersion;

    if (bc > bimodalityThreshold) {
      return {
        route: "confirm_or_escalate",
        escalate: true,
        confidence,
        normalizedEntropy: dispersion,
        freeEnergy: resolvedFreeEnergy ?? undefined,
        isOod: false,
        reason: `Score 双峰性検知 (BC=${bc.toFixed(3)} > ${bimodalityThreshold}: 評価対立)。`,
        topCandidates: [[`score_${Math.round(meanVal)}`, confidence]],
      };
    }

    if (dispersion > scoreVarThreshold) {
      return {
        route: "confirm_or_escalate",
        escalate: true,
        confidence,
        normalizedEntropy: dispersion,
        freeEnergy: resolvedFreeEnergy ?? undefined,
        isOod: false,
        reason: `Score 分散過大検知 (Dispersion=${dispersion.toFixed(3)} > ${scoreVarThreshold})。`,
        topCandidates: [[`score_${Math.round(meanVal)}`, confidence]],
      };
    }

    return {
      route: "auto_execute",
      escalate: false,
      confidence,
      normalizedEntropy: dispersion,
      freeEnergy: resolvedFreeEnergy ?? undefined,
      isOod: false,
      reason: "Score 条件充足 (安定単峰性スコア)。",
      topCandidates: [[`score_${Math.round(meanVal)}`, confidence]],
    };
  }

  // 3. Choice 型 (多クラス選択)
  const probMap: Record<string, number> = {};
  if (params.probabilities) {
    if (Array.isArray(params.probabilities)) {
      if (params.labels && params.labels.length === params.probabilities.length) {
        for (let i = 0; i < params.probabilities.length; i++) {
          probMap[params.labels[i]] = params.probabilities[i];
        }
      } else {
        for (let i = 0; i < params.probabilities.length; i++) {
          probMap[`c_${i}`] = params.probabilities[i];
        }
      }
    } else {
      Object.assign(probMap, params.probabilities);
    }
  }

  const probsVals = Object.values(probMap);
  if (probsVals.length === 0) {
    return {
      route: "fallback",
      escalate: true,
      confidence: 0.0,
      isOod: true,
      reason: "確率分布が空です。",
      topCandidates: [],
    };
  }

  const [, hNorm] = calculateEntropy(probsVals);
  const margin = calculateTopMargin(probsVals);

  const sortedCandidates = Object.entries(probMap).sort((a, b) => b[1] - a[1]);
  const topCandidates: Array<[string, number]> = sortedCandidates.slice(0, 3);
  const topConf = topCandidates.length > 0 ? topCandidates[0][1] : 0.0;

  // 自由エネルギーによる OOD 評価
  if (resolvedFreeEnergy !== undefined && resolvedFreeEnergy !== null && resolvedFreeEnergy > energyThreshold) {
    return {
      route: "fallback",
      escalate: true,
      confidence: topConf,
      topMargin: margin,
      normalizedEntropy: hNorm,
      freeEnergy: resolvedFreeEnergy,
      isOod: true,
      reason: `自由エネルギー超過による OOD 検知 (F=${resolvedFreeEnergy.toFixed(3)} > ${energyThreshold})。`,
      topCandidates,
    };
  }

  // 3 条件充足
  if (margin >= marginThreshold && hNorm <= entropyThreshold) {
    return {
      route: "auto_execute",
      escalate: false,
      confidence: topConf,
      topMargin: margin,
      normalizedEntropy: hNorm,
      freeEnergy: resolvedFreeEnergy ?? undefined,
      isOod: false,
      reason: "3軸 Pareto 条件充足 (高確信度即時確定)。",
      topCandidates,
    };
  }

  const failureReasons: string[] = [];
  if (margin < marginThreshold) {
    failureReasons.push(`Top-Margin 拮抗 (Margin=${margin.toFixed(3)} < ${marginThreshold})`);
  }
  if (hNorm > entropyThreshold) {
    failureReasons.push(`エントロピー拡散 (Entropy=${hNorm.toFixed(3)} > ${entropyThreshold})`);
  }

  return {
    route: "confirm_or_escalate",
    escalate: true,
    confidence: topConf,
    topMargin: margin,
    normalizedEntropy: hNorm,
    freeEnergy: resolvedFreeEnergy ?? undefined,
    isOod: false,
    reason: `不確実性検知による System 2 エスカレーション: ${failureReasons.join(" / ")}。`,
    topCandidates,
  };
}
