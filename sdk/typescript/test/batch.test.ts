import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { SokutoClient } from "../src/client.js";
import { SokutoCascadeClient } from "../src/cascade.js";
import type { StateBatchRequest, StateBatchResponse } from "../src/types.js";
import { MockSystem2Provider } from "../src/providers.js";

describe("SokutoClient バッチ推論 & 透過キャッシュ", () => {
  it("evaluateBatch で state-batch エンドポイントへ送信しレスポンスを受信できること", async () => {
    const originalFetch = globalThis.fetch;
    try {
      globalThis.fetch = async (input, init) => {
        const url = String(input);
        assert.ok(url.endsWith("/v1/systemone/state-batch"));
        const body = JSON.parse(String(init?.body)) as StateBatchRequest;
        assert.equal(body.state, "テスト用State文書");
        assert.ok(body.questions.q1);

        const respData: StateBatchResponse = {
          answers: {
            q1: { choice: "承認", confidence: 0.95 },
            q2: { choice: "否認", confidence: 0.88 },
          },
          usage: { prompt_tokens: 30, completion_tokens: 0, total_tokens: 30 },
          cache_hit: true,
          state_hash: "0123456789abcdef",
        };

        return new Response(JSON.stringify(respData), {
          status: 200,
          headers: {
            "Content-Type": "application/json",
            "X-State-Hash": "0123456789abcdef",
            "X-Cache-Status": "HIT",
          },
        });
      };

      const client = new SokutoClient({ baseUrl: "http://localhost:8080" });
      const resp = await client.evaluateBatch({
        state: "テスト用State文書",
        questions: {
          q1: { instructions: "申請を承認しますか？", type: "choice" },
          q2: { instructions: "申請を否認しますか？", type: "choice" },
        },
      });

      assert.equal(resp.cache_hit, true);
      assert.equal(resp.cache_status, "HIT");
      assert.equal(resp.state_hash, "0123456789abcdef");
      assert.equal(client.getLastStateHash(), "0123456789abcdef");
      assert.equal(resp.answers.q1.choice, "承認");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("enableStateOffloading 有効時にハッシュ送信および自動再送が機能すること", async () => {
    const originalFetch = globalThis.fetch;
    let callCount = 0;
    try {
      globalThis.fetch = async (input, init) => {
        callCount++;
        const headers = init?.headers as Record<string, string>;
        const body = JSON.parse(String(init?.body)) as StateBatchRequest;

        // 初回: state が空でキャッシュミス (400)
        if (callCount === 1) {
          assert.equal(body.state, "");
          assert.equal(headers["X-State-Hash"], "prev_hash_123");
          return new Response(
            JSON.stringify({ error: { code: "state_cache_miss", message: "Miss" } }),
            { status: 400 },
          );
        }

        // 2回目: 自動再送で本文が含まれる
        assert.equal(body.state, "巨大なState本文");
        assert.equal(headers["X-State-Hash"], undefined);
        return new Response(
          JSON.stringify({
            answers: { q1: { choice: "OK", confidence: 0.99 } },
            usage: { prompt_tokens: 50, completion_tokens: 0, total_tokens: 50 },
            cache_hit: false,
          }),
          {
            status: 200,
            headers: {
              "Content-Type": "application/json",
              "X-State-Hash": "new_hash_456",
              "X-Cache-Status": "MISS",
            },
          },
        );
      };

      const client = new SokutoClient();
      (client as unknown as { lastStateHash: string }).lastStateHash = "prev_hash_123";

      const res = await client.evaluateBatch(
        {
          state: "巨大なState本文",
          questions: { q1: { instructions: "確認", type: "choice" } },
        },
        { enableStateOffloading: true },
      );

      assert.equal(callCount, 2);
      assert.equal(res.answers.q1.choice, "OK");
      assert.equal(client.getLastStateHash(), "new_hash_456");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});

describe("SokutoCascadeClient executeBatch", () => {
  it("複数質問を一括評価し、部分エスカレーション (Partial Cascade) が動作すること", async () => {
    const originalFetch = globalThis.fetch;
    try {
      globalThis.fetch = async () => {
        const respData: StateBatchResponse = {
          answers: {
            q_high: {
              choice: "承認",
              confidence: 0.98,
              probabilities: { 承認: 0.98, 却下: 0.02 },
            },
            q_low: {
              choice: "保留",
              confidence: 0.45,
              probabilities: { 保留: 0.45, 却下: 0.44, 承認: 0.11 },
            },
          },
          usage: { prompt_tokens: 40, completion_tokens: 0, total_tokens: 40 },
          cache_hit: true,
        };

        return new Response(JSON.stringify(respData), {
          status: 200,
          headers: {
            "Content-Type": "application/json",
            "X-State-Hash": "hash123",
            "X-Cache-Status": "HIT",
          },
        });
      };

      const cascadeClient = new SokutoCascadeClient({
        system2Provider: new MockSystem2Provider({
          defaultDecision: "System 2 救済判定",
          defaultThinking: "詳細文脈精査による救済",
        }),
      });

      const batchResults = await cascadeClient.executeBatch({
        state: "契約書文脈",
        questions: {
          q_high: { instructions: "通常承認要件", type: "choice" },
          q_low: { instructions: "境界事例要件", type: "choice" },
        },
      });

      // q_high は System 1 で即時確定
      assert.equal(batchResults.q_high.source, "system1");
      assert.equal(batchResults.q_high.decision, "承認");

      // q_low は System 2 へ部分エスカレーション
      assert.equal(batchResults.q_low.source, "system2");
      assert.equal(batchResults.q_low.decision, "System 2 救済判定");
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  it("enableStateOffloading 有効時にエスカレーションされた際、手元の State で Triage プロンプトが再構成されること", async () => {
    const originalFetch = globalThis.fetch;
    try {
      let sentBody = "";
      globalThis.fetch = async (_url, init) => {
        sentBody = String(init?.body ?? "");
        const respData: StateBatchResponse = {
          answers: {
            q_ambiguous: {
              confidence: 0.51,
              probabilities: { 承認: 0.51, 却下: 0.49 },
              gating: {
                route: "confirm_or_escalate",
                confidence: 0.51,
                reason: "僅差拮抗",
                escalation: {
                  uncertainty_reason: "僅差拮抗",
                  prompt_template: "<context>\n(コンテキストが指定されていません。)\n</context>",
                },
              },
            },
          },
          usage: { prompt_tokens: 30, completion_tokens: 0, total_tokens: 30 },
          cache_hit: true,
        };

        return new Response(JSON.stringify(respData), {
          status: 200,
          headers: {
            "Content-Type": "application/json",
            "X-State-Hash": "cached_state_hash",
            "X-Cache-Status": "HIT",
          },
        });
      };

      const originalStateText = "重要顧客の特別契約書データ（氏名: 佐藤花子）";
      const cascadeClient = new SokutoCascadeClient({
        system2Provider: new MockSystem2Provider({
          defaultDecision: "承認",
        }),
      });
      // 前回キャッシュハッシュを設定
      cascadeClient.sokutoClient.setLastStateHash("cached_state_hash");

      const results = await cascadeClient.executeBatch(
        {
          state: originalStateText,
          questions: {
            q_ambiguous: { instructions: "特例条項を精査せよ", type: "choice" },
          },
        },
        { enableStateOffloading: true },
      );

      // サーバーへ送信された body では state が空文字 "" になっていること
      const parsedBody = JSON.parse(sentBody);
      assert.equal(parsedBody.state, "");

      // エスカレーション結果の検証
      assert.equal(results.q_ambiguous.source, "system2");
      assert.ok(results.q_ambiguous.triagePrompt);
      // 手元の originalStateText が triagePrompt に再構成されて埋め込まれていること
      assert.ok(results.q_ambiguous.triagePrompt?.userPrompt.includes(originalStateText));
      assert.ok(results.q_ambiguous.triagePrompt?.userPrompt.includes("佐藤花子"));
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});
