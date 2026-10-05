/**
 * System 2 (フロンティア自己回帰 LLM) プロバイダー抽象化モジュール。
 *
 * 特定ベンダーの SDK への直接密結合を排し、
 * Strategy パターンによる差し替え可能なエスカレーション実行インターフェースを提供する。
 */

import type { TriagePrompt } from "./types.js";

/**
 * System 2 からの実行結果コンテナ。
 */
export interface System2Response<T = unknown> {
  /** 確定決定ラベルまたは値。 */
  decision: T;
  /** 思考連鎖 (CoT) のテキスト。 */
  thinking?: string;
  /** LLM からの生出力文字列。 */
  rawOutput?: string;
  /** 自己申告確信度。 */
  confidence?: number;
  /** トークン消費統計。 */
  usage?: {
    promptTokens?: number;
    completionTokens?: number;
    totalTokens?: number;
  };
}

/**
 * System 2 実行プロバイダーの共通インターフェース。
 */
export interface System2Provider {
  /**
   * エスカレーションプロンプトを受け取り、判定結果を返却する。
   *
   * @param prompt 対比型エスカレーションプロンプト。
   * @returns System2Response 判定結果。
   */
  call(prompt: TriagePrompt): Promise<System2Response>;
}

/**
 * テストおよびオフライン検証用モックプロバイダー。
 */
export class MockSystem2Provider implements System2Provider {
  public defaultDecision: unknown;
  public defaultThinking: string;
  public callCount = 0;
  public lastPrompt?: TriagePrompt;
  public callback?: (prompt: TriagePrompt) => Promise<unknown> | unknown;

  constructor(options?: {
    defaultDecision?: unknown;
    defaultThinking?: string;
    callback?: (prompt: TriagePrompt) => Promise<unknown> | unknown;
  }) {
    this.defaultDecision = options?.defaultDecision ?? "candidate_a";
    this.defaultThinking =
      options?.defaultThinking ?? "モック思考連鎖による検証。";
    this.callback = options?.callback;
  }

  async call(prompt: TriagePrompt): Promise<System2Response> {
    this.callCount++;
    this.lastPrompt = prompt;

    if (this.callback) {
      const res = await this.callback(prompt);
      if (
        typeof res === "object" &&
        res !== null &&
        "decision" in (res as Record<string, unknown>)
      ) {
        return res as System2Response;
      }
      return {
        decision: res,
        thinking: this.defaultThinking,
        rawOutput: String(res),
        confidence: 0.95,
      };
    }

    return {
      decision: this.defaultDecision,
      thinking: this.defaultThinking,
      rawOutput: JSON.stringify({
        thinking: this.defaultThinking,
        decision: this.defaultDecision,
      }),
      confidence: 0.95,
      usage: { promptTokens: 120, completionTokens: 45, totalTokens: 165 },
    };
  }
}

/**
 * 任意の関数または非同期関数をラップするプロバイダー。
 */
export class CallableSystem2Provider implements System2Provider {
  constructor(
    private readonly fn: (prompt: TriagePrompt) => Promise<unknown> | unknown
  ) {}

  async call(prompt: TriagePrompt): Promise<System2Response> {
    const res = await this.fn(prompt);
    if (
      typeof res === "object" &&
      res !== null &&
      "decision" in (res as Record<string, unknown>)
    ) {
      return res as System2Response;
    }

    let decision = res;
    let thinking: string | undefined;
    if (typeof res === "object" && res !== null) {
      const obj = res as Record<string, unknown>;
      if ("decision" in obj) decision = obj.decision;
      if ("thinking" in obj && typeof obj.thinking === "string") thinking = obj.thinking;
    }

    return {
      decision,
      thinking,
      rawOutput: String(res),
      confidence: 0.90,
    };
  }
}

/**
 * OpenAI 互換 (chat/completions) エンドポイントと通信する汎用 HTTP プロバイダー。
 */
export class GenericHttpSystem2Provider implements System2Provider {
  constructor(
    public readonly endpointUrl: string,
    public readonly apiKey = "",
    public readonly model = "gpt-4o",
    public readonly timeoutMs = 30000
  ) {}

  async call(prompt: TriagePrompt): Promise<System2Response> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (this.apiKey) {
      headers["Authorization"] = `Bearer ${this.apiKey}`;
    }

    const payload = {
      model: this.model,
      messages: [
        { role: "system", content: prompt.systemPrompt },
        { role: "user", content: prompt.userPrompt },
      ],
      response_format: { type: "json_object" },
      temperature: 0.0,
    };

    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const resp = await fetch(this.endpointUrl, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (!resp.ok) {
        throw new Error(`HTTP error ${resp.status}: ${await resp.text()}`);
      }

      const data = (await resp.json()) as {
        choices: Array<{ message: { content: string } }>;
        usage?: {
          prompt_tokens?: number;
          completion_tokens?: number;
          total_tokens?: number;
        };
      };

      const rawContent = data.choices[0]?.message?.content ?? "";
      let thinking: string | undefined;
      let decision: unknown = rawContent.trim();
      let confidence: number | undefined;

      try {
        const parsed = JSON.parse(rawContent) as Record<string, unknown>;
        if ("thinking" in parsed && typeof parsed.thinking === "string") {
          thinking = parsed.thinking;
        }
        if ("decision" in parsed) {
          decision = parsed.decision;
        }
        if ("confidence" in parsed && typeof parsed.confidence === "number") {
          confidence = parsed.confidence;
        }
      } catch {
        const match = /<thinking>(.*?)<\/thinking>/s.exec(rawContent);
        if (match) {
          thinking = match[1].trim();
        }
      }

      return {
        decision,
        thinking,
        rawOutput: rawContent,
        confidence,
        usage: {
          promptTokens: data.usage?.prompt_tokens,
          completionTokens: data.usage?.completion_tokens,
          totalTokens: data.usage?.total_tokens,
        },
      };
    } finally {
      clearTimeout(id);
    }
  }
}
