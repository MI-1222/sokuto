/**
 * sokuto 基本 HTTP 通信クライアントモジュール。
 *
 * sokuto-server (Axum) の /v1/systemone エンドポイントと通信し、
 * 非自己回帰の単一フォワードパス推論を実行する。
 */

import type {
  SokutoRequest,
  SokutoResponse,
  StateBatchRequest,
  StateBatchResponse,
} from "./types.js";

/**
 * sokuto-server 向け HTTP 推論クライアント。
 */
export class SokutoClient {
  public readonly baseUrl: string;
  public readonly timeoutMs: number;
  private lastStateHash: string | null = null;

  /**
   * クライアントを初期化する。
   *
   * @param options 設定オプション。
   */
  constructor(options?: { baseUrl?: string; timeoutMs?: number }) {
    this.baseUrl = (options?.baseUrl ?? "http://localhost:8080").replace(/\/+$/, "");
    this.timeoutMs = options?.timeoutMs ?? 50;
  }

  /** 直近にサーバーから受信した State キャッシュキーを取得する。 */
  getLastStateHash(): string | null {
    return this.lastStateHash;
  }

  /** 記憶している State キャッシュキーをクリアする。 */
  clearStateHash(): void {
    this.lastStateHash = null;
  }

  /**
   * /v1/systemone エンドポイントへ推論リクエストを送信する。
   *
   * @param request リクエストデータ。
   * @param options 最適化オプション (State-Offloading等)。
   * @returns SokutoResponse レスポンスデータ。
   */
  async predict(
    request: SokutoRequest,
    options?: { enableStateOffloading?: boolean },
  ): Promise<SokutoResponse> {
    const url = `${this.baseUrl}/v1/systemone`;
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), this.timeoutMs);

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    const effectiveHash = request.state_hash ?? this.lastStateHash;
    if (effectiveHash) {
      headers["X-State-Hash"] = effectiveHash;
    }

    let payload = { ...request };
    if (options?.enableStateOffloading && effectiveHash && request.state) {
      payload = { ...payload, state: "" };
    }

    try {
      let resp = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      // State-Offloading でミスした場合は本文を含めて自動フォールバック再送
      if (
        !resp.ok &&
        resp.status === 400 &&
        options?.enableStateOffloading &&
        payload.state === "" &&
        request.state
      ) {
        resp = await fetch(url, {
          method: "POST",
          headers,
          body: JSON.stringify(request),
          signal: controller.signal,
        });
      }

      if (!resp.ok) {
        throw new Error(`sokuto-server HTTP ${resp.status}: ${await resp.text()}`);
      }

      const receivedHash = resp.headers.get("x-state-hash") ?? resp.headers.get("X-State-Hash");
      if (receivedHash) {
        this.lastStateHash = receivedHash;
      }
      const cacheStatus =
        resp.headers.get("x-cache-status") ??
        resp.headers.get("X-Cache-Status") ??
        resp.headers.get("x-cache") ??
        "BYPASS";

      const data = (await resp.json()) as SokutoResponse;
      data.cache_status = cacheStatus;
      if (receivedHash) {
        data.state_hash = receivedHash;
      }
      return data;
    } finally {
      clearTimeout(id);
    }
  }

  /**
   * /v1/systemone/state-batch エンドポイントへ一括バッチ質問リクエストを送信する。
   *
   * @param request バッチリクエストデータ。
   * @param options 最適化オプション。
   * @returns StateBatchResponse レスポンスデータ。
   */
  async evaluateBatch(
    request: StateBatchRequest,
    options?: { enableStateOffloading?: boolean },
  ): Promise<StateBatchResponse> {
    const url = `${this.baseUrl}/v1/systemone/state-batch`;
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), this.timeoutMs);

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };

    const effectiveHash = request.state_hash ?? this.lastStateHash;
    if (effectiveHash) {
      headers["X-State-Hash"] = effectiveHash;
    }

    let payload = { ...request };
    if (options?.enableStateOffloading && effectiveHash && request.state) {
      payload = { ...payload, state: "" };
    }

    try {
      let resp = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      if (
        !resp.ok &&
        resp.status === 400 &&
        options?.enableStateOffloading &&
        payload.state === "" &&
        request.state
      ) {
        resp = await fetch(url, {
          method: "POST",
          headers,
          body: JSON.stringify(request),
          signal: controller.signal,
        });
      }

      if (!resp.ok) {
        throw new Error(`sokuto-server HTTP ${resp.status}: ${await resp.text()}`);
      }

      const receivedHash = resp.headers.get("x-state-hash") ?? resp.headers.get("X-State-Hash");
      if (receivedHash) {
        this.lastStateHash = receivedHash;
      }
      const cacheStatus =
        resp.headers.get("x-cache-status") ??
        resp.headers.get("X-Cache-Status") ??
        resp.headers.get("x-cache") ??
        "BYPASS";

      const data = (await resp.json()) as StateBatchResponse;
      data.cache_status = cacheStatus;
      if (receivedHash) {
        data.state_hash = receivedHash;
      }
      return data;
    } finally {
      clearTimeout(id);
    }
  }

  /** サーバーの稼働状態 (/ready) を確認する。 */
  async isReady(): Promise<boolean> {
    const url = `${this.baseUrl}/ready`;
    try {
      const resp = await fetch(url);
      return resp.ok;
    } catch {
      return false;
    }
  }
}
