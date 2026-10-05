"""アンカリング思考汚染防止ニュートラル対比型 Triage プロンプト合成モジュール。

不確実な境界事例や未知ドメイン (OOD) をフロンティア自己回帰 LLM (System 2) へ
委託する際、単一候補への迎合 (Sycophancy) や誤判定追従を防ぐため、
上位 2 候補を対等な対立構図として中立的に提示する Adversarial Balanced Guided CoT プロンプトを生成する。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .types import GatingDecision, TriagePrompt


def synthesize_triage_prompt(
    state: Any,
    instruction: str,
    gating: GatingDecision,
    criteria: dict[str, str] | list[str] | None = None,
    allow_none_of_the_above: bool = True,
    context_keys: list[str] | None = None,
    context_reducer: Callable[[Any], str] | None = None,
) -> TriagePrompt:
    """アンカリング思考汚染を防ぐニュートラル対比型エスカレーションプロンプトを合成する。

    設計規則:
        1. 単一の推奨候補を絶対に提示しない (事前確信度や 1 位ラベルの単独漏洩を遮断)。
        2. 上位 2 候補 (Top-1 と Top-2) を「候補 A」「候補 B」として対等に並列提示する。
        3. 機械学習モデルの事前確率に惑わされず、提供された事実のみに基づく中立的検証を強制する。
        4. OOD (自由エネルギー超過) 時は「該当なし (None of the above)」の可能性を動的追加する。
        5. 思考連鎖 (<thinking>...</thinking>) の後に確定決定を出力する構造化スキーマを定義する。

    Args:
        state (Any): 判断の根拠となる入力文脈 (文字列または辞書)。
        instruction (str): 判断タスクの指示文。
        gating (GatingDecision): 3軸ゲーティング評価結果。
        criteria (dict[str, str] | list[str] | None): 選択肢の説明・基準定義。
        allow_none_of_the_above (bool): 該当なし選択肢の検討を許可するか。
        context_keys (list[str] | None): state が辞書の場合に抽出するキーのリスト。
        context_reducer (Callable[[Any], str] | None): 長文文脈の縮約・トリミングコールバック。

    Returns:
        TriagePrompt: システムプロンプト、ユーザープロンプト、および構造化出力スキーマ。
    """
    # 1. システムプロンプトの構成 (中立性・批判的思考の強制)
    system_prompt = (
        "あなたは厳密かつ中立な論理検証エージェントです。\n"
        "機械学習モデルの事前判定において判断が拮抗した境界事例を中立・客観的に再検証します。\n"
        "【重要規則】\n"
        "- 機械学習モデルの予測や事前確率に一切迎合せず、与えられた事実のみに基づいて判断してください。\n"
        "- 提示された対立候補を批判的に比較検証し、十分な根拠がある場合のみ結論を下してください。\n"
        "- まず <thinking> タグ内で事実の照合と思考連鎖 (Chain-of-Thought) を展開し、その後に最終決定を出力してください。"
    )

    # 2. 文脈 (State) の文字列整形
    state_str: str
    if context_reducer is not None:
        state_str = context_reducer(state)
    elif isinstance(state, dict):
        if context_keys:
            filtered = {k: state[k] for k in context_keys if k in state}
            state_str = json.dumps(filtered, ensure_ascii=False, indent=2)
        else:
            state_str = json.dumps(state, ensure_ascii=False, indent=2)
    else:
        state_str = str(state)

    # 3. 候補の対照提示 (Top-1 vs Top-2)
    top_candidates = gating.top_candidates
    candidate_sections: list[str] = []

    criteria_map: dict[str, str] = {}
    if isinstance(criteria, dict):
        criteria_map = {str(k): str(v) for k, v in criteria.items()}
    elif isinstance(criteria, list):
        criteria_map = {c: c for c in criteria}

    if len(top_candidates) >= 2:
        cand_a_name, _ = top_candidates[0]
        cand_b_name, _ = top_candidates[1]
        desc_a = criteria_map.get(cand_a_name, "")
        desc_b = criteria_map.get(cand_b_name, "")

        cand_a_text = f"- 候補 A: `{cand_a_name}`" + (f" ({desc_a})" if desc_a else "")
        cand_b_text = f"- 候補 B: `{cand_b_name}`" + (f" ({desc_b})" if desc_b else "")
        candidate_sections.extend([cand_a_text, cand_b_text])
    elif len(top_candidates) == 1:
        cand_name, _ = top_candidates[0]
        desc = criteria_map.get(cand_name, "")
        candidate_sections.append(f"- 候補: `{cand_name}`" + (f" ({desc})" if desc else ""))
    else:
        for c in criteria_map:
            candidate_sections.append(f"- 選択肢: `{c}` ({criteria_map[c]})")

    candidates_text = "\n".join(candidate_sections)

    # 4. OOD または該当なしの注記
    ood_notice = ""
    if gating.is_ood or (allow_none_of_the_above and gating.route.value == "fallback"):
        ood_notice = (
            "\n【未知ドメイン警告】\n"
            "本事例はどの定義済み候補にも適合しない可能性 (Out-of-Distribution) が検知されています。\n"
            "提供された文脈がいずれの候補の成立要件も満たさない場合は、無理に選択せず 'none_of_the_above' (該当なし) を選択してください。\n"
        )

    # 5. ユーザープロンプトの組み立て
    user_prompt = (
        f"以下の判断タスクにおいて、判定が拮抗している候補を中立・批判的に検証してください。\n\n"
        f"### 指示\n{instruction}\n\n"
        f"### 検証対象候補 (拮抗)\n{candidates_text}\n"
        f"{ood_notice}\n"
        f"### 根拠情報 (State)\n```text\n{state_str}\n```\n\n"
        "### 要求事項\n"
        "1. <thinking> タグ内で、文脈の客観的事実と各候補の要件を 1 つずつ照合してください。\n"
        "2. 照合完了後、指定された JSON フォーマットで最終決定を出力してください。"
    )

    # 6. Structured Outputs スキーマ
    allowed_decisions = [c[0] for c in top_candidates]
    if allow_none_of_the_above:
        allowed_decisions.append("none_of_the_above")

    structured_schema = {
        "type": "object",
        "properties": {
            "thinking": {
                "type": "string",
                "description": "客観的事実に基づく中立・批判的なステップ思考過程。",
            },
            "decision": {
                "type": "string",
                "enum": allowed_decisions if allowed_decisions else None,
                "description": "確定された最終決定ラベル。",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "自己評価確信度。",
            },
        },
        "required": ["thinking", "decision"],
        "additionalProperties": False,
    }

    return TriagePrompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        structured_schema=structured_schema,
    )
