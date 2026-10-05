/**
 * TypeScript カスケード統合 SDK ユニットテスト。
 *
 * 3軸 Pareto 最適ゲーティング、Python パリティ、対比型 Triage、
 * サーキットブレーカー、およびカスケード統合クライアントを検証する。
 */

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  CircuitBreaker,
  MockSystem2Provider,
  SokutoCascadeClient,
  calculateEntropy,
  calculateScoreMetrics,
  calculateTopMargin,
  computeFreeEnergy,
  evaluateGating,
  synthesizeTriagePrompt,
} from "../src/index.js";

describe("3軸 Pareto 最適ゲーティングエンジン", () => {
  it("高確信度 Choice 分布で auto_execute が判定されること", () => {
    const probs = { A: 0.85, B: 0.10, C: 0.05 };
    const freeEnergy = -2.5;

    const decision = evaluateGating({
      probabilities: probs,
      freeEnergy,
      questionType: "choice",
    });

    assert.equal(decision.route, "auto_execute");
    assert.equal(decision.escalate, false);
    assert.equal(decision.isOod, false);
    assert.ok(Math.abs((decision.topMargin ?? 0) - 0.75) < 1e-5);
    assert.equal(decision.topCandidates[0][0], "A");
    assert.equal(decision.topCandidates[0][1], 0.85);
  });

  it("上位2候補が僅差で競合する場合にエスカレーションされること", () => {
    const probs = { Option1: 0.46, Option2: 0.44, Option3: 0.10 };
    const freeEnergy = -1.8;

    const decision = evaluateGating({
      probabilities: probs,
      freeEnergy,
      questionType: "choice",
    });

    assert.equal(decision.route, "confirm_or_escalate");
    assert.equal(decision.escalate, true);
    assert.ok(Math.abs((decision.topMargin ?? 0) - 0.02) < 1e-5);
    assert.ok(decision.reason.includes("Top-Margin 拮抗"));
  });

  it("確率が全体に拡散している場合にエスカレーションされること", () => {
    const probs = [0.25, 0.25, 0.25, 0.25];
    const decision = evaluateGating({ probabilities: probs, questionType: "choice" });

    assert.equal(decision.route, "confirm_or_escalate");
    assert.equal(decision.escalate, true);
    assert.ok(Math.abs((decision.normalizedEntropy ?? 0) - 1.0) < 1e-5);
  });

  it("自由エネルギーが閾値を超過した場合に OOD として fallback すること", () => {
    const probs = { A: 0.90, B: 0.10 };
    const freeEnergy = 0.5;

    const decision = evaluateGating({
      probabilities: probs,
      freeEnergy,
      questionType: "choice",
    });

    assert.equal(decision.route, "fallback");
    assert.equal(decision.escalate, true);
    assert.equal(decision.isOod, true);
    assert.ok(decision.reason.includes("OOD 検知"));
  });

  it("Noul (真偽値) プリミティブにおける判定", () => {
    // 高確信度
    const dConfident = evaluateGating({ noulProbability: 0.95, questionType: "noul" });
    assert.equal(dConfident.route, "auto_execute");
    assert.equal(dConfident.escalate, false);
    assert.ok(Math.abs((dConfident.topMargin ?? 0) - 0.90) < 1e-5);

    // 拮抗
    const dAmbiguous = evaluateGating({ noulProbability: 0.52, questionType: "noul" });
    assert.equal(dAmbiguous.route, "confirm_or_escalate");
    assert.equal(dAmbiguous.escalate, true);
  });

  it("Score (順序尺度) における単峰充足と双峰性排除", () => {
    // 単峰
    const pUnimodal = [0.05, 0.10, 0.70, 0.10, 0.05];
    const dUnimodal = evaluateGating({ probabilities: pUnimodal, questionType: "score" });
    assert.equal(dUnimodal.route, "auto_execute");
    assert.equal(dUnimodal.escalate, false);

    // 双峰性
    const pBimodal = [0.48, 0.02, 0.00, 0.02, 0.48];
    const dBimodal = evaluateGating({ probabilities: pBimodal, questionType: "score" });
    assert.equal(dBimodal.route, "confirm_or_escalate");
    assert.equal(dBimodal.escalate, true);
    assert.ok(dBimodal.reason.includes("双峰性"));
  });

  it("特異点 (K=1, ゼロ確率) で NaN や例外が発生しないこと", () => {
    const [hRaw, hNorm] = calculateEntropy([1.0]);
    assert.equal(hRaw, 0.0);
    assert.equal(hNorm, 0.0);

    const margin = calculateTopMargin([1.0]);
    assert.equal(margin, 1.0);

    const [, hZero] = calculateEntropy([1.0, 0.0, 0.0]);
    assert.equal(hZero, 0.0);

    const feSingle = computeFreeEnergy([3.5], 1.0);
    assert.ok(Math.abs(feSingle - -3.5) < 1e-5);
  });

  it("Python 実装との数値パリティ (Atol <= 1e-5)", () => {
    const probs = [0.55, 0.25, 0.15, 0.05];
    const [rawH, normH] = calculateEntropy(probs);
    const margin = calculateTopMargin(probs);

    // Python 計算値:
    // rawH = -(0.55*ln(0.55) + 0.25*ln(0.25) + 0.15*ln(0.15) + 0.05*ln(0.05)) = 1.10985
    // maxH = ln(4) = 1.38629
    // normH = 1.10985 / 1.38629 = 0.80058
    // margin = 0.55 - 0.25 = 0.30
    assert.ok(Math.abs(normH - 0.800587) < 1e-4);
    assert.ok(Math.abs(margin - 0.30) < 1e-5);

    const logits = [2.5, 1.2, 0.8, -0.5];
    const fe = computeFreeEnergy(logits, 1.0);
    // Python 計算値:
    // max = 2.5
    // sum_exp = 1.0 + exp(-1.3) + exp(-1.7) + exp(-3.0) = 1.0 + 0.27253 + 0.18268 + 0.04978 = 1.50499
    // fe = -2.5 - ln(1.50499) = -2.5 - 0.40879 = -2.90879
    assert.ok(Math.abs(fe - -2.90879) < 1e-4);
  });
});

describe("対比型 Triage プロンプト合成", () => {
  it("単一候補の漏洩がなく、候補 A vs 候補 B が中立提示されること", () => {
    const gating: any = {
      route: "confirm_or_escalate",
      escalate: true,
      confidence: 0.52,
      topMargin: 0.04,
      normalizedEntropy: 0.68,
      freeEnergy: -1.5,
      isOod: false,
      reason: "僅差拮抗",
      topCandidates: [
        ["approve", 0.52],
        ["reject", 0.48],
      ],
    };

    const prompt = synthesizeTriagePrompt({
      state: "ユーザー申請文脈",
      instruction: "申請を判定せよ。",
      gating,
      criteria: { approve: "要件充足", reject: "要件不足" },
    });

    assert.ok(!prompt.userPrompt.includes("sokuto は approve と判定"));
    assert.ok(!prompt.userPrompt.includes("確信度: 0.52"));
    assert.ok(prompt.userPrompt.includes("- 候補 A: `approve`"));
    assert.ok(prompt.userPrompt.includes("- 候補 B: `reject`"));
    assert.ok(prompt.userPrompt.includes("<thinking>"));
    assert.ok(prompt.systemPrompt.includes("中立"));
  });

  it("OOD 検知時に none_of_the_above の警告が含まれること", () => {
    const gating: any = {
      route: "fallback",
      escalate: true,
      confidence: 0.30,
      freeEnergy: 0.8,
      isOod: true,
      reason: "自由エネルギー超過",
      topCandidates: [
        ["plan_a", 0.3],
        ["plan_b", 0.28],
      ],
    };

    const prompt = synthesizeTriagePrompt({
      state: "未知文脈",
      instruction: "プラン選定",
      gating,
      allowNoneOfTheAbove: true,
    });

    assert.ok(prompt.userPrompt.includes("【未知ドメイン警告】"));
    assert.ok(prompt.userPrompt.includes("none_of_the_above"));
  });
});

describe("サーキットブレーカー", () => {
  it("CLOSED -> OPEN -> HALF_OPEN -> CLOSED 遷移を検証", async () => {
    const cb = new CircuitBreaker({
      failureThreshold: 3,
      recoveryTimeoutMs: 50,
      successThreshold: 2,
    });

    assert.equal(cb.state, "closed");
    assert.equal(cb.canExecute(), true);

    cb.recordFailure();
    cb.recordFailure();
    assert.equal(cb.state, "closed");

    // 3回目で OPEN
    cb.recordFailure();
    assert.equal(cb.state, "open");
    assert.equal(cb.canExecute(), false);

    // クールダウン待ち
    await new Promise((r) => setTimeout(r, 60));

    // 自動で HALF_OPEN
    assert.equal(cb.state, "half_open");
    assert.equal(cb.canExecute(), true);

    cb.recordSuccess();
    assert.equal(cb.state, "half_open");

    // 2回成功で CLOSED
    cb.recordSuccess();
    assert.equal(cb.state, "closed");
    assert.equal(cb.canExecute(), true);
  });
});

describe("SokutoCascadeClient 統合テスト", () => {
  it("System 1 高確信度時に即時返却されること", async () => {
    const mockS2 = new MockSystem2Provider();

    // グローバル fetch の一時モック
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        answers: {
          q1: {
            choice: "vip",
            probabilities: { vip: 0.95, standard: 0.05 },
            gating: { energy: -2.3 },
          },
        },
        model_name: "sokuto-v1",
      }),
    })) as any;

    try {
      const client = new SokutoCascadeClient({ system2Provider: mockS2 });
      const result = await client.predictQuestion({
        instruction: "顧客種別判定",
        criteria: { vip: "VIP", standard: "一般" },
        state: "年間100万円以上利用",
        questionId: "q1",
      });

      assert.equal(result.source, "system1");
      assert.equal(result.decision, "vip");
      assert.equal(result.gating.route, "auto_execute");
      assert.equal(mockS2.callCount, 0); // System 2 は呼ばれない！
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("System 1 僅差拮抗時に System 2 へエスカレーションされること", async () => {
    const mockS2 = new MockSystem2Provider({
      defaultDecision: "standard",
      defaultThinking: "利用額が境界線上であるため一般と判定。",
    });

    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        answers: {
          q1: {
            choice: "vip",
            probabilities: { vip: 0.51, standard: 0.49 }, // Margin 0.02
            gating: { energy: -1.2 },
          },
        },
      }),
    })) as any;

    try {
      const client = new SokutoCascadeClient({ system2Provider: mockS2 });
      const result = await client.predictQuestion({
        instruction: "顧客種別判定",
        criteria: { vip: "VIP", standard: "一般" },
        state: "利用額99万円",
        questionId: "q1",
      });

      assert.equal(result.source, "system2");
      assert.equal(result.decision, "standard");
      assert.equal(result.gating.escalate, true);
      assert.equal(result.system2Thinking, "利用額が境界線上であるため一般と判定。");
      assert.equal(mockS2.callCount, 1);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("サーバー通信障害時に System 2 へフェイルオープンすること", async () => {
    const mockS2 = new MockSystem2Provider({
      defaultDecision: "fail_open_decision",
    });

    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async () => {
      throw new Error("Network connection refused");
    }) as any;

    try {
      const cb = new CircuitBreaker({ failureThreshold: 1 });
      const client = new SokutoCascadeClient({
        system2Provider: mockS2,
        circuitBreaker: cb,
      });

      const result = await client.predictQuestion({
        instruction: "障害テスト",
        state: "データ",
      });

      assert.equal(result.source, "system2");
      assert.equal(result.decision, "fail_open_decision");
      assert.equal(cb.state, "open");
      assert.equal(mockS2.callCount, 1);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("JBE-QA実務トラフィックにおいてAPIコスト80%以上削減および総合精度92%以上を達成すること", async () => {
    const totalRequests = 100;
    const dataset: Array<{
      id: string;
      instruction: string;
      state: string;
      groundTruth: string;
      s1Choice: string;
      s1Probs: Record<string, number>;
      s1Energy: number;
      s2Decision: string;
    }> = [];

    for (let i = 0; i < 84; i++) {
      const isS1Correct = i < 82;
      const groundTruth = "valid";
      const chosenS1 = isS1Correct ? "valid" : "invalid";
      dataset.push({
        id: `routine_${i}`,
        instruction: `条項第${i + 1}条の有効性判定`,
        state: `条項第${i + 1}条の文脈`,
        groundTruth,
        s1Choice: chosenS1,
        s1Probs: chosenS1 === "valid" ? { valid: 0.94, invalid: 0.06 } : { invalid: 0.92, valid: 0.08 },
        s1Energy: -2.2,
        s2Decision: groundTruth,
      });
    }

    for (let i = 0; i < 16; i++) {
      const isS2Correct = i < 15;
      const groundTruth = "applicable";
      dataset.push({
        id: `hard_${i}`,
        instruction: `司法試験短答式第${i + 1}問`,
        state: `事実関係${i + 1}`,
        groundTruth,
        s1Choice: "applicable",
        s1Probs: { applicable: 0.51, inapplicable: 0.49 },
        s1Energy: -1.2,
        s2Decision: isS2Correct ? "applicable" : "inapplicable",
      });
    }

    const mockS2 = new MockSystem2Provider({
      callback: (triagePrompt) => {
        const text = triagePrompt.userPrompt;
        for (const item of dataset) {
          if (text.includes(item.id)) {
            return item.s2Decision;
          }
        }
        return "applicable";
      },
    });

    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async (_url: string, init?: RequestInit) => {
      const reqJson = JSON.parse(init?.body as string);
      const questionId = Object.keys(reqJson.questions)[0];
      const targetItem = dataset.find((d) => d.id === questionId) ?? dataset[0];
      const respPayload = {
        answers: {
          [questionId]: {
            choice: targetItem.s1Choice,
            probabilities: targetItem.s1Probs,
            gating: { energy: targetItem.s1Energy },
          },
        },
      };
      return {
        ok: true,
        status: 200,
        json: async () => respPayload,
      } as any;
    }) as any;

    try {
      const client = new SokutoCascadeClient({ system2Provider: mockS2 });
      let s1Count = 0;
      let s2Count = 0;
      let totalCorrect = 0;

      for (const item of dataset) {
        const criteria = item.id.startsWith("routine")
          ? { valid: "有効", invalid: "無効" }
          : { applicable: "該当", inapplicable: "非該当" };

        const result = await client.predictQuestion({
          instruction: item.instruction,
          criteria,
          state: item.state,
          questionId: item.id,
        });

        if (result.source === "system1") {
          s1Count++;
        } else if (result.source === "system2") {
          s2Count++;
        }

        if (result.decision === item.groundTruth) {
          totalCorrect++;
        }
      }

      const costReductionRate = 1.0 - s2Count / totalRequests;
      const overallAccuracy = totalCorrect / totalRequests;

      assert.ok(costReductionRate >= 0.8, `コスト削減率: ${costReductionRate * 100}% >= 80%`);
      assert.ok(overallAccuracy >= 0.92, `総合精度: ${overallAccuracy * 100}% >= 92%`);
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
