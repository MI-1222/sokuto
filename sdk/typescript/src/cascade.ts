/**
 * System 1 / System 2 カスケード統合統括クライアントモジュール。
 *
 * 超低遅延・高確信度のローカル非自己回帰判断エンジン (sokuto / System 1) と、
 * 推論連鎖に優れるフロンティア自己回帰 LLM (Claude, GPT 等 / System 2) を協調させ、
 * 定常トラフィックの 80% 以上をミリ秒で完結させつつ境界事例を高精度に救済する。
 */

import { CircuitBreaker } from "./circuit-breaker.js";
import { evaluateGating } from "./gating.js";
import { MockSystem2Provider, type System2Provider } from "./providers.js";
import { synthesizeTriagePrompt } from "./triage.js";
import type {
  CascadeResult,
  CascadeSource,
  GatingConfig,
  GatingDecision,
  SokutoRequest,
  SokutoResponse,
  StateBatchRequest,
  StateBatchResponse,
} from "./types.js";
import { SokutoClient } from "./client.js";

/**
 * System 1 / System 2 カスケード統合クライアント。
 */
export class SokutoCascadeClient {
  public readonly baseUrl: string;
  public readonly system2Provider: System2Provider;
  public readonly gatingConfig: GatingConfig;
  public readonly timeoutMs: number;
  public readonly circuitBreaker: CircuitBreaker;
  public readonly sokutoClient: SokutoClient;

  /**
   * カスケードクライアントを初期化する。
   *
   * @param options 設定オプション。
   */
  constructor(options?: {
    baseUrl?: string;
    system2Provider?: System2Provider;
    gatingConfig?: GatingConfig;
    timeoutMs?: number;
    circuitBreaker?: CircuitBreaker;
    sokutoClient?: SokutoClient;
  }) {
    this.baseUrl = (options?.baseUrl ?? "http://localhost:8080").replace(/\/+$/, "");
    this.system2Provider = options?.system2Provider ?? new MockSystem2Provider();
    this.gatingConfig = options?.gatingConfig ?? {};
    this.timeoutMs = options?.timeoutMs ?? 50;
    this.circuitBreaker = options?.circuitBreaker ?? new CircuitBreaker();
    this.sokutoClient =
      options?.sokutoClient ??
      new SokutoClient({ baseUrl: this.baseUrl, timeoutMs: this.timeoutMs });
  }

  /**
   * 単一の質問定義に対してカスケード判定を実行する。
   *
   * @param params 質問パラメータ。
   * @returns CascadeResult<T> 確定判定結果。
   */
  async predictQuestion<T = unknown>(params: {
    instruction: string;
    criteria?: Record<string, string> | string[] | null;
    state: unknown;
    questionType?: "choice" | "score" | "noul" | string;
    questionId?: string;
    contextReducer?: (state: unknown) => string;
  }): Promise<CascadeResult<T>> {
    const startTime = performance.now();
    const qId = params.questionId ?? "q1";
    const qType = params.questionType ?? "choice";

    // サーキットブレーカー検査
    if (!this.circuitBreaker.canExecute()) {
      return this.escalateToSystem2<T>({
        instruction: params.instruction,
        criteria: params.criteria,
        state: params.state,
        gating: {
          route: "fallback",
          escalate: true,
          confidence: 0.0,
          isOod: false,
          reason: "サーキットブレーカー遮断中 (Fail-Open 直行)。",
          topCandidates: [],
        },
        startTime,
        reason: "circuit_breaker_open",
        contextReducer: params.contextReducer,
      });
    }

    // System 1 への呼び出し
    let s1Result: SokutoResponse | null = null;
    try {
      const payload: SokutoRequest = {
        state: params.state,
        questions: {
          [qId]: {
            instructions: params.instruction,
            criteria: params.criteria,
            type: qType,
          },
        },
      };

      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);

      const resp = await fetch(`${this.baseUrl}/v1/systemone`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      clearTimeout(timer);

      if (resp.ok) {
        s1Result = (await resp.json()) as SokutoResponse;
        this.circuitBreaker.recordSuccess();
      } else {
        this.circuitBreaker.recordFailure();
      }
    } catch {
      this.circuitBreaker.recordFailure();
    }

    // 通信断または障害時のフェイルオープン
    if (!s1Result) {
      return this.escalateToSystem2<T>({
        instruction: params.instruction,
        criteria: params.criteria,
        state: params.state,
        gating: {
          route: "fallback",
          escalate: true,
          confidence: 0.0,
          isOod: false,
          reason: "System 1 通信タイムアウトまたはサーバーエラー (Fail-Open)。",
          topCandidates: [],
        },
        startTime,
        reason: "system1_timeout_or_error",
        contextReducer: params.contextReducer,
      });
    }

    // ゲーティング評価
    const answerData = s1Result.answers?.[qId] ?? {};
    const probs = answerData.probabilities;
    const noulP = answerData.noul;
    const serverG = answerData.gating;

    const gating = evaluateGating({
      probabilities: probs,
      questionType: qType,
      config: this.gatingConfig,
      noulProbability: noulP,
      serverGating: serverG,
    });

    // 充足時は System 1 で即時返却
    if (!gating.escalate && gating.route === "auto_execute") {
      const elapsedMs = performance.now() - startTime;
      let decision: unknown =
        answerData.choice ??
        answerData.score ??
        answerData.noul ??
        (gating.topCandidates.length > 0 ? gating.topCandidates[0][0] : null);

      return {
        decision: decision as T,
        source: "system1",
        gating,
        latencyMs: elapsedMs,
        modelName: s1Result.model_name,
        rawS1Response: s1Result as unknown as Record<string, unknown>,
      };
    }

    // 不確実性検知による System 2 エスカレーション
    return this.escalateToSystem2<T>({
      instruction: params.instruction,
      criteria: params.criteria,
      state: params.state,
      gating,
      startTime,
      reason: gating.reason,
      rawS1Response: s1Result as unknown as Record<string, unknown>,
      contextReducer: params.contextReducer,
    });
  }

  /**
   * 複合推論リクエストに対するカスケード判定を実行する。
   *
   * @param request 推論リクエスト。
   * @param customSchema System 2 向けの構造化スキーマ。
   * @returns CascadeResult<T> 確定判定結果。
   */
  async execute<T = unknown>(
    request: SokutoRequest,
    customSchema?: Record<string, unknown>
  ): Promise<CascadeResult<T>> {
    const startTime = performance.now();

    // サーキットブレーカー検査
    if (!this.circuitBreaker.canExecute()) {
      return this.escalateRequestToSystem2<T>({
        request,
        gating: {
          route: "fallback",
          escalate: true,
          confidence: 0.0,
          isOod: false,
          reason: "サーキットブレーカー遮断中 (Fail-Open 直行)。",
          topCandidates: [],
        },
        startTime,
        reason: "circuit_breaker_open",
        customSchema,
      });
    }

    let s1Result: SokutoResponse | null = null;
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);

      const resp = await fetch(`${this.baseUrl}/v1/systemone`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal: controller.signal,
      });
      clearTimeout(timer);

      if (resp.ok) {
        s1Result = (await resp.json()) as SokutoResponse;
        this.circuitBreaker.recordSuccess();
      } else {
        this.circuitBreaker.recordFailure();
      }
    } catch {
      this.circuitBreaker.recordFailure();
    }

    if (!s1Result) {
      return this.escalateRequestToSystem2<T>({
        request,
        gating: {
          route: "fallback",
          escalate: true,
          confidence: 0.0,
          isOod: false,
          reason: "System 1 通信タイムアウトまたはサーバー障害 (Fail-Open)。",
          topCandidates: [],
        },
        startTime,
        reason: "system1_timeout_or_error",
        customSchema,
      });
    }

    // 全質問に対する集約ゲーティング判定
    let overallEscalate = false;
    let worstGating: GatingDecision | null = null;
    const decisions: Record<string, unknown> = {};

    for (const [qId, qSpec] of Object.entries(request.questions)) {
      const ans = s1Result.answers?.[qId] ?? {};
      const qType = qSpec.type ?? "choice";
      const probs = ans.probabilities;
      const noulVal = ans.noul;
      const serverG = ans.gating;

      const g = evaluateGating({
        probabilities: probs,
        questionType: qType,
        config: this.gatingConfig,
        noulProbability: noulVal,
        serverGating: serverG,
      });

      if (g.escalate) {
        overallEscalate = true;
        worstGating = g;
        break;
      } else if (!worstGating || g.confidence < worstGating.confidence) {
        worstGating = g;
      }

      decisions[qId] =
        ans.choice ??
        ans.score ??
        ans.noul ??
        (g.topCandidates.length > 0 ? g.topCandidates[0][0] : null);
    }

    const finalGating: GatingDecision = worstGating ?? {
      route: "auto_execute",
      escalate: false,
      confidence: 1.0,
      isOod: false,
      reason: "全フィールド充足。",
      topCandidates: [],
    };

    if (!overallEscalate) {
      const elapsedMs = performance.now() - startTime;
      return {
        decision: decisions as T,
        source: "system1",
        gating: finalGating,
        latencyMs: elapsedMs,
        modelName: s1Result.model_name,
        rawS1Response: s1Result as unknown as Record<string, unknown>,
      };
    }

    return this.escalateRequestToSystem2<T>({
      request,
      gating: finalGating,
      startTime,
      reason: finalGating.reason,
      customSchema,
      rawS1Response: s1Result as unknown as Record<string, unknown>,
    });
  }

  private async escalateToSystem2<T>(params: {
    instruction: string;
    criteria?: Record<string, string> | string[] | null;
    state: unknown;
    gating: GatingDecision;
    startTime: number;
    reason: string;
    rawS1Response?: Record<string, unknown>;
    contextReducer?: (state: unknown) => string;
  }): Promise<CascadeResult<T>> {
    const prompt = synthesizeTriagePrompt({
      state: params.state,
      instruction: params.instruction,
      gating: params.gating,
      criteria: params.criteria,
      contextReducer: params.contextReducer,
    });

    const s2Resp = await this.system2Provider.call(prompt);
    const elapsedMs = performance.now() - params.startTime;

    let decisionVal = s2Resp.decision;
    if (
      typeof decisionVal === "object" &&
      decisionVal !== null &&
      "decision" in (decisionVal as Record<string, unknown>)
    ) {
      decisionVal = (decisionVal as Record<string, unknown>).decision;
    }

    return {
      decision: decisionVal as T,
      source: "system2",
      gating: params.gating,
      latencyMs: elapsedMs,
      escalationReason: params.reason,
      system2Thinking: s2Resp.thinking,
      rawS1Response: params.rawS1Response,
      rawS2Response: s2Resp as unknown as Record<string, unknown>,
    };
  }

  private async escalateRequestToSystem2<T>(params: {
    request: SokutoRequest;
    gating: GatingDecision;
    startTime: number;
    reason: string;
    customSchema?: Record<string, unknown>;
    rawS1Response?: Record<string, unknown>;
  }): Promise<CascadeResult<T>> {
    const instructions = Object.entries(params.request.questions)
      .map(([id, q]) => `[${id}] ${q.instructions}`)
      .join("\n");

    const prompt = synthesizeTriagePrompt({
      state: params.request.state,
      instruction: instructions,
      gating: params.gating,
    });

    if (params.customSchema) {
      prompt.structuredSchema = params.customSchema;
    }

    const s2Resp = await this.system2Provider.call(prompt);
    const elapsedMs = performance.now() - params.startTime;

    return {
      decision: s2Resp.decision as T,
      source: "system2",
      gating: params.gating,
      latencyMs: elapsedMs,
      escalationReason: params.reason,
      system2Thinking: s2Resp.thinking,
      rawS1Response: params.rawS1Response,
      rawS2Response: s2Resp as unknown as Record<string, unknown>,
    };
  }

  /**
   * 単一 State に対する複数質問を一括評価し、部分カスケード (Partial Cascade) を実行する。
   *
   * 高確信度な質問は System 1 で即時確定し、僅差・OOD の質問のみ System 2 へ選択的エスカレーションする。
   *
   * @param request StateBatchRequest パラメータ。
   * @param options 最適化オプション (State-Offloading等)。
   * @returns 各質問 ID に対応する CascadeResult の辞書。
   */
  async executeBatch(
    request: StateBatchRequest,
    options?: { enableStateOffloading?: boolean },
  ): Promise<Record<string, CascadeResult<unknown>>> {
    const startTime = performance.now();
    const results: Record<string, CascadeResult<unknown>> = {};

    // 1. サーキットブレーカー検査 (遮断時は全質問を System 2 直行)
    if (!this.circuitBreaker.canExecute()) {
      for (const [qid, q] of Object.entries(request.questions)) {
        results[qid] = await this.escalateToSystem2<unknown>({
          instruction: q.instructions,
          criteria: q.criteria,
          state: request.state,
          gating: {
            route: "fallback",
            escalate: true,
            confidence: 0.0,
            isOod: false,
            reason: "サーキットブレーカー遮断中 (Fail-Open 直行)。",
            topCandidates: [],
          },
          startTime,
          reason: "circuit_breaker_open",
        });
      }
      return results;
    }

    // 2. System 1 へのバッチ問い合わせ
    let batchResp: StateBatchResponse;
    try {
      batchResp = await this.sokutoClient.evaluateBatch(request, options);
      this.circuitBreaker.recordSuccess();
    } catch (err) {
      this.circuitBreaker.recordFailure();
      // 障害時は全質問を System 2 へフェイルオープン
      for (const [qid, q] of Object.entries(request.questions)) {
        results[qid] = await this.escalateToSystem2<unknown>({
          instruction: q.instructions,
          criteria: q.criteria,
          state: request.state,
          gating: {
            route: "fallback",
            escalate: true,
            confidence: 0.0,
            isOod: false,
            reason: `System 1 呼び出し失敗: ${err instanceof Error ? err.message : String(err)}`,
            topCandidates: [],
          },
          startTime,
          reason: "s1_fetch_error",
        });
      }
      return results;
    }

    // 3. 各質問の判定と部分エスカレーション (Partial Cascade)
    for (const [qid, q] of Object.entries(request.questions)) {
      const answer = batchResp.answers[qid];
      if (!answer) {
        continue;
      }

      const gating = evaluateGating({
        probabilities: answer.probabilities,
        questionType: q.type,
        config: this.gatingConfig,
        noulProbability: answer.noul,
        serverGating: answer.gating,
      });

      if (gating.escalate) {
        results[qid] = await this.escalateToSystem2<unknown>({
          instruction: q.instructions,
          criteria: q.criteria,
          state: request.state,
          gating,
          startTime,
          reason: gating.reason,
          rawS1Response: { answer, batch_usage: batchResp.usage },
        });
      } else {
        const decisionVal = answer.choice ?? answer.score ?? answer.noul;
        results[qid] = {
          decision: decisionVal,
          source: "system1",
          gating,
          latencyMs: performance.now() - startTime,
          rawS1Response: { answer, batch_usage: batchResp.usage },
        };
      }
    }

    return results;
  }
}
