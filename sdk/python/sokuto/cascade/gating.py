"""3軸 Pareto 最適ゲーティングエンジンモジュール。

単一の Softmax 最大確率依存を排し、Top-Margin (僅差拮抗の排除)、
正規化シャノンエントロピー (全体的不確実性の検知)、
およびヘルムホルツ自由エネルギー (未知ドメイン・OOD の遮断) の
3 指標を同時評価して高信頼判定と System 2 エスカレーションを統括する。
"""

from __future__ import annotations

import math
from typing import Any

from .types import DecisionRoute, GatingConfig, GatingDecision


def calculate_entropy(
    probabilities: list[float] | dict[str, float],
) -> tuple[float, float]:
    """確率分布からシャノンエントロピーおよび正規化エントロピーを算出する。

    数理仕様:
        H(p) = - \\sum_{k=1}^K p_k \\ln p_k
        H_{norm}(p) = H(p) / \\ln K (K >= 2 の場合)
        K = 1 の場合は特異点として H = 0.0, H_{norm} = 0.0 を返却する。

    Args:
        probabilities (list[float] | dict[str, float]): 各候補の確率値列または辞書。

    Returns:
        tuple[float, float]: (未正規化シャノンエントロピー, [0.0, 1.0] にクランプされた正規化エントロピー)。
    """
    probs = list(probabilities.values()) if isinstance(probabilities, dict) else list(probabilities)
    k = len(probs)
    if k <= 1:
        return 0.0, 0.0

    entropy = 0.0
    for p in probs:
        if p > 0.0:
            clamped_p = min(p, 1.0)
            entropy -= clamped_p * math.log(clamped_p)

    max_entropy = math.log(float(k))
    if max_entropy <= 0.0:
        return 0.0, 0.0

    normalized = entropy / max_entropy
    return entropy, max(0.0, min(1.0, normalized))


def calculate_top_margin(probabilities: list[float] | dict[str, float]) -> float:
    """確率分布の上位2候補の差分 (Top-Margin) を算出する。

    数理仕様:
        M(p) = p_{(1)} - p_{(2)} (K >= 2 の場合)
        K = 1 の場合は 1.0 を返却する。

    Args:
        probabilities (list[float] | dict[str, float]): 各候補の確率値列または辞書。

    Returns:
        float: [0.0, 1.0] にクランプされた Top-Margin。
    """
    probs = list(probabilities.values()) if isinstance(probabilities, dict) else list(probabilities)
    k = len(probs)
    if k == 0:
        return 0.0
    if k == 1:
        return 1.0

    # 上位2候補の走査
    max1 = -float("inf")
    max2 = -float("inf")
    for p in probs:
        val = max(0.0, min(1.0, p))
        if val > max1:
            max2 = max1
            max1 = val
        elif val > max2:
            max2 = val

    margin = max1 - max2
    return max(0.0, min(1.0, margin))


def compute_free_energy(logits: list[float], temperature: float = 1.0) -> float:
    """未正規化生ロジット列からヘルムホルツ自由エネルギーを算出する。

    数理仕様:
        F(x; T) = -T \\ln \\sum_{k=1}^K \\exp(z_k / T)
        オーバーフロー防止のため、最大ロジット減算による安定化を適用する。

    Args:
        logits (list[float]): 未正規化ロジット列。
        temperature (float): 較正温度パラメータ (デフォルト: 1.0)。

    Returns:
        float: ヘルムホルツ自由エネルギー実数値。
    """
    if not logits:
        return 0.0
    if len(logits) == 1:
        return -logits[0]

    tau = max(1e-4, temperature)
    max_logit = max(logits)

    sum_exp = sum(math.exp((z - max_logit) / tau) for z in logits)
    if sum_exp <= 0.0:
        return -max_logit

    return -max_logit - (tau * math.log(sum_exp))


def calculate_score_metrics(
    probabilities: list[float],
) -> tuple[float, float, float, float]:
    """Score 型確率分布から期待値、分散、正規化分散指標、双峰性係数を算出する。

    Args:
        probabilities (list[float]): 各段階レベルの確率分布列 (長さ M >= 2)。

    Returns:
        tuple[float, float, float, float]:
            - 期待値スコア (mu)
            - 分散 (sigma^2)
            - 正規化分散指標 (1 - C_var)
            - 双峰性係数 (BC)
    """
    m = len(probabilities)
    if m < 2:
        return 0.0, 0.0, 0.0, 0.0

    mean = sum(k * p for k, p in enumerate(probabilities))
    variance = 0.0
    mu3 = 0.0
    mu4 = 0.0

    for k, p in enumerate(probabilities):
        diff = float(k) - mean
        diff2 = diff * diff
        variance += diff2 * p
        mu3 += diff2 * diff * p
        mu4 += diff2 * diff2 * p

    max_variance = ((float(m) - 1.0) ** 2) / 4.0
    c_var = 1.0 - (variance / max_variance) if max_variance > 0.0 else 1.0
    dispersion = 1.0 - max(0.0, min(1.0, c_var))

    if variance < 1e-6:
        return mean, variance, dispersion, 0.0

    sigma = math.sqrt(variance)
    sigma3 = variance * sigma
    sigma4 = variance * variance

    if sigma4 <= 0.0:
        return mean, variance, dispersion, 0.0

    gamma = mu3 / sigma3
    kurtosis = mu4 / sigma4
    if kurtosis <= 0.0:
        return mean, variance, dispersion, 0.0

    bimodality = (gamma * gamma + 1.0) / kurtosis
    return mean, variance, dispersion, max(0.0, min(1.0, bimodality))


def evaluate_gating(
    probabilities: list[float] | dict[str, float] | None = None,
    free_energy: float | None = None,
    question_type: str = "choice",
    config: GatingConfig | None = None,
    labels: list[str] | None = None,
    noul_probability: float | None = None,
    server_gating: dict[str, Any] | None = None,
) -> GatingDecision:
    """3軸 Pareto 最適ゲーティング評価を実行し、エスカレーション要否を判定する。

    Choice, Noul, Score の各決定プリミティブに応じた不確実性を計測し、
    Top-Margin、正規化エントロピー、自由エネルギーの全制約を満たす場合のみ
    System 1 (sokuto) を確定採用 (AutoExecute) とする。

    Args:
        probabilities (list[float] | dict[str, float] | None): 確率分布。
        free_energy (float | None): ヘルムホルツ自由エネルギー値。
        question_type (str): 質問種別 ('choice', 'noul', 'score')。
        config (GatingConfig | None): 閾値設定。未指定時はデフォルト。
        labels (list[str] | None): 候補ラベル名列。
        noul_probability (float | None): Noul 型の P(true) 確率。
        server_gating (dict[str, Any] | None): sokuto-server から返却された既存ゲーティングメタデータ。

    Returns:
        GatingDecision: ルーティング判定結果および診断詳細。
    """
    cfg = config or GatingConfig()
    qtype = question_type.lower()

    # サーバー側で算出された自由エネルギーの反映
    if free_energy is None and server_gating is not None:
        energy_val = server_gating.get("energy")
        if energy_val is None:
            energy_val = server_gating.get("free_energy")
        if energy_val is not None:
            free_energy = float(energy_val)

    # 1. Noul 型 (真偽値) の評価
    if qtype == "noul":
        p_true = noul_probability
        if p_true is None and probabilities is not None:
            if isinstance(probabilities, dict):
                p_true = probabilities.get("true", probabilities.get("True", 0.5))
            elif len(probabilities) >= 1:
                p_true = probabilities[0]
        if p_true is None:
            p_true = 0.5

        p_true = max(0.0, min(1.0, float(p_true)))
        p_false = 1.0 - p_true

        margin = abs(2.0 * p_true - 1.0)
        h_norm = 0.0
        if 0.0 < p_true < 1.0:
            h_raw = -(p_true * math.log(p_true) + p_false * math.log(p_false))
            h_norm = max(0.0, min(1.0, h_raw / math.log(2.0)))

        confidence = max(p_true, p_false)
        top_candidates = [
            ("true" if p_true >= p_false else "false", max(p_true, p_false)),
            ("false" if p_true >= p_false else "true", min(p_true, p_false)),
        ]

        # 判定
        if margin >= cfg.margin_threshold and h_norm <= cfg.entropy_threshold:
            return GatingDecision(
                route=DecisionRoute.AUTO_EXECUTE,
                escalate=False,
                confidence=confidence,
                top_margin=margin,
                normalized_entropy=h_norm,
                free_energy=free_energy,
                is_ood=False,
                reason="Noul 条件充足 (高確信度真偽判定)。",
                top_candidates=top_candidates,
            )
        else:
            return GatingDecision(
                route=DecisionRoute.CONFIRM_OR_ESCALATE,
                escalate=True,
                confidence=confidence,
                top_margin=margin,
                normalized_entropy=h_norm,
                free_energy=free_energy,
                is_ood=False,
                reason=f"Noul 境界事例検知 (Margin={margin:.3f} < {cfg.margin_threshold} または Entropy={h_norm:.3f} > {cfg.entropy_threshold})。",
                top_candidates=top_candidates,
            )

    # 2. Score 型 (順序尺度) の評価
    if qtype == "score":
        probs_list: list[float] = []
        if isinstance(probabilities, dict):
            probs_list = [float(v) for v in probabilities.values()]
        elif isinstance(probabilities, list):
            probs_list = [float(v) for v in probabilities]

        if len(probs_list) < 2:
            return GatingDecision(
                route=DecisionRoute.CONFIRM_OR_ESCALATE,
                escalate=True,
                confidence=0.5,
                reason="Score 確率分布データ不足。",
            )

        mean_val, _, dispersion, bc = calculate_score_metrics(probs_list)
        confidence = 1.0 - dispersion

        # 双峰性・意見分裂の検知
        if bc > cfg.bimodality_threshold:
            return GatingDecision(
                route=DecisionRoute.CONFIRM_OR_ESCALATE,
                escalate=True,
                confidence=confidence,
                top_margin=None,
                normalized_entropy=dispersion,
                free_energy=free_energy,
                is_ood=False,
                reason=f"Score 双峰性検知 (BC={bc:.3f} > {cfg.bimodality_threshold}: 評価対立)。",
                top_candidates=[(f"score_{round(mean_val)}", confidence)],
            )

        if dispersion > cfg.score_var_threshold:
            return GatingDecision(
                route=DecisionRoute.CONFIRM_OR_ESCALATE,
                escalate=True,
                confidence=confidence,
                top_margin=None,
                normalized_entropy=dispersion,
                free_energy=free_energy,
                is_ood=False,
                reason=f"Score 分散過大検知 (Dispersion={dispersion:.3f} > {cfg.score_var_threshold})。",
                top_candidates=[(f"score_{round(mean_val)}", confidence)],
            )

        return GatingDecision(
            route=DecisionRoute.AUTO_EXECUTE,
            escalate=False,
            confidence=confidence,
            top_margin=None,
            normalized_entropy=dispersion,
            free_energy=free_energy,
            is_ood=False,
            reason="Score 条件充足 (安定単峰性スコア)。",
            top_candidates=[(f"score_{round(mean_val)}", confidence)],
        )

    # 3. Choice 型 (多クラス選択) の 3軸 Pareto 評価
    prob_map: dict[str, float] = {}
    if isinstance(probabilities, dict):
        prob_map = {str(k): float(v) for k, v in probabilities.items()}
    elif isinstance(probabilities, list):
        if labels and len(labels) == len(probabilities):
            prob_map = {lbl: float(p) for lbl, p in zip(labels, probabilities, strict=False)}
        else:
            prob_map = {f"c_{i}": float(p) for i, p in enumerate(probabilities)}

    probs_vals = list(prob_map.values())
    if not probs_vals:
        return GatingDecision(
            route=DecisionRoute.FALLBACK,
            escalate=True,
            confidence=0.0,
            is_ood=True,
            reason="確率分布が空です。",
        )

    _, h_norm = calculate_entropy(probs_vals)
    margin = calculate_top_margin(probs_vals)

    # 上位候補の抽出
    sorted_candidates = sorted(prob_map.items(), key=lambda item: item[1], reverse=True)
    top_candidates = sorted_candidates[:3]
    top_conf = sorted_candidates[0][1] if sorted_candidates else 0.0

    # OOD 自由エネルギー検査
    if free_energy is not None and free_energy > cfg.energy_threshold:
        return GatingDecision(
            route=DecisionRoute.FALLBACK,
            escalate=True,
            confidence=top_conf,
            top_margin=margin,
            normalized_entropy=h_norm,
            free_energy=free_energy,
            is_ood=True,
            reason=f"自由エネルギー超過による OOD 検知 (F={free_energy:.3f} > {cfg.energy_threshold})。",
            top_candidates=top_candidates,
        )

    # 3 条件充足の検査
    if margin >= cfg.margin_threshold and h_norm <= cfg.entropy_threshold:
        return GatingDecision(
            route=DecisionRoute.AUTO_EXECUTE,
            escalate=False,
            confidence=top_conf,
            top_margin=margin,
            normalized_entropy=h_norm,
            free_energy=free_energy,
            is_ood=False,
            reason="3軸 Pareto 条件充足 (高確信度即時確定)。",
            top_candidates=top_candidates,
        )

    # 僅差拮抗またはエントロピー拡散
    failure_reasons: list[str] = []
    if margin < cfg.margin_threshold:
        failure_reasons.append(f"Top-Margin 拮抗 (Margin={margin:.3f} < {cfg.margin_threshold})")
    if h_norm > cfg.entropy_threshold:
        failure_reasons.append(f"エントロピー拡散 (Entropy={h_norm:.3f} > {cfg.entropy_threshold})")

    reason_str = " / ".join(failure_reasons)
    return GatingDecision(
        route=DecisionRoute.CONFIRM_OR_ESCALATE,
        escalate=True,
        confidence=top_conf,
        top_margin=margin,
        normalized_entropy=h_norm,
        free_energy=free_energy,
        is_ood=False,
        reason=f"不確実性検知による System 2 エスカレーション: {reason_str}。",
        top_candidates=top_candidates,
    )
