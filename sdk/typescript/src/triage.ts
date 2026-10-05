/**
 * アンカリング思考汚染防止ニュートラル対比型 Triage プロンプト合成モジュール。
 *
 * 不確実な境界事例や未知ドメイン (OOD) を System 2 (フロンティア LLM) へ
 * 委託する際、単一候補への迎合 (Sycophancy) や誤判定追従を防ぐため、
 * 上位 2 候補を対等な対立構図として中立的に提示する Adversarial Balanced Guided CoT プロンプトを生成する。
 */

import type { GatingDecision, TriagePrompt } from "./types.js";

/**
 * アンカリング思考汚染を防ぐニュートラル対比型エスカレーションプロンプトを合成する。
 *
 * @param params 合成パラメータ。
 * @returns TriagePrompt 構造体。
 */
export function synthesizeTriagePrompt(params: {
  state: unknown;
  instruction: string;
  gating: GatingDecision;
  criteria?: Record<string, string> | string[] | null;
  allowNoneOfTheAbove?: boolean;
  contextReducer?: (state: unknown) => string;
}): TriagePrompt {
  const allowNone = params.allowNoneOfTheAbove ?? true;

  // 1. システムプロンプト
  const systemPrompt = [
    "あなたは厳密かつ中立な論理検証エージェントです。",
    "機械学習モデルの事前判定において判断が拮抗した境界事例を中立・客観的に再検証します。",
    "【重要規則】",
    "- 機械学習モデルの予測や事前確率に一切迎合せず、与えられた事実のみに基づいて判断してください。",
    "- 提示された対立候補を批判的に比較検証し、十分な根拠がある場合のみ結論を下してください。",
    "- まず <thinking> タグ内で事実の照合と思考連鎖 (Chain-of-Thought) を展開し、その後に最終決定を出力してください。",
  ].join("\n");

  // 2. 文脈の文字列化 (contextReducer が指定されていれば優先適用)
  let stateStr: string;
  if (params.contextReducer) {
    stateStr = params.contextReducer(params.state);
  } else if (typeof params.state === "string") {
    stateStr = params.state;
  } else {
    stateStr = JSON.stringify(params.state, null, 2);
  }

  // 3. 候補の対立構造化 (Top-1 vs Top-2)
  const topCandidates = params.gating.topCandidates;
  const candidateSections: string[] = [];

  const criteriaMap: Record<string, string> = {};
  if (params.criteria) {
    if (Array.isArray(params.criteria)) {
      for (const c of params.criteria) {
        criteriaMap[c] = c;
      }
    } else {
      Object.assign(criteriaMap, params.criteria);
    }
  }

  if (topCandidates.length >= 2) {
    const [candAName] = topCandidates[0];
    const [candBName] = topCandidates[1];
    const descA = criteriaMap[candAName] ? ` (${criteriaMap[candAName]})` : "";
    const descB = criteriaMap[candBName] ? ` (${criteriaMap[candBName]})` : "";

    candidateSections.push(`- 候補 A: \`${candAName}\`${descA}`);
    candidateSections.push(`- 候補 B: \`${candBName}\`${descB}`);
  } else if (topCandidates.length === 1) {
    const [candName] = topCandidates[0];
    const desc = criteriaMap[candName] ? ` (${criteriaMap[candName]})` : "";
    candidateSections.push(`- 候補: \`${candName}\`${desc}`);
  } else {
    for (const [key, desc] of Object.entries(criteriaMap)) {
      candidateSections.push(`- 選択肢: \`${key}\` (${desc})`);
    }
  }

  const candidatesText = candidateSections.join("\n");

  // 4. OOD 警告
  let oodNotice = "";
  if (params.gating.isOod || (allowNone && params.gating.route === "fallback")) {
    oodNotice = [
      "",
      "【未知ドメイン警告】",
      "本事例はどの定義済み候補にも適合しない可能性 (Out-of-Distribution) が検知されています。",
      "提供された文脈がいずれの候補の成立要件も満たさない場合は、無理に選択せず 'none_of_the_above' (該当なし) を選択してください。",
      "",
    ].join("\n");
  }

  // 5. ユーザープロンプト
  const userPrompt = [
    "以下の判断タスクにおいて、判定が拮抗している候補を中立・批判的に検証してください。",
    "",
    `### 指示\n${params.instruction}`,
    "",
    `### 検証対象候補 (拮抗)\n${candidatesText}`,
    oodNotice,
    `### 根拠情報 (State)\n\`\`\`text\n${stateStr}\n\`\`\``,
    "",
    "### 要求事項",
    "1. <thinking> タグ内で、文脈の客観的事実と各候補の要件を 1 つずつ照合してください。",
    "2. 照合完了後、指定された JSON フォーマットで最終決定を出力してください。",
  ].join("\n");

  // 6. Structured Outputs スキーマ
  const allowedDecisions = topCandidates.map(([name]) => name);
  if (allowNone) {
    allowedDecisions.push("none_of_the_above");
  }

  const structuredSchema: Record<string, unknown> = {
    type: "object",
    properties: {
      thinking: {
        type: "string",
        description: "客観的事実に基づく中立・批判的なステップ思考過程。",
      },
      decision: {
        type: "string",
        enum: allowedDecisions.length > 0 ? allowedDecisions : undefined,
        description: "確定された最終決定ラベル。",
      },
      confidence: {
        type: "number",
        minimum: 0.0,
        maximum: 1.0,
        description: "自己評価確信度。",
      },
    },
    required: ["thinking", "decision"],
    additionalProperties: false,
  };

  return {
    systemPrompt,
    userPrompt,
    structuredSchema,
  };
}
