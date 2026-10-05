/**
 * sokuto 基本 HTTP 通信クライアントモジュール。
 *
 * sokuto-server (Axum) の /v1/systemone エンドポイントと通信し、
 * 非自己回帰の単一フォワードパス推論を実行する。
 */

import type { SokutoRequest, SokutoResponse } from "./types.js";

/**
 * sokuto-server 向け HTTP 推論クライアント。
 */
export class SokutoClient {
  public readonly baseUrl: string;
  public readonly timeoutMs: number;

  /**
   * クライアントを初期化する。
   *
   * @param options 設定オプション。
   */
  constructor(options?: { baseUrl?: string; timeoutMs?: number }) {
    this.baseUrl = (options?.baseUrl ?? "http://localhost:8080").replace(/\/+$/, "");
    this.timeoutMs = options?.timeoutMs ?? 50;
  }

  /**
   * /v1/systemone エンドポイントへ推論リクエストを送信する。
   *
   * @param request リクエストデータ。
   * @returns SokutoResponse レスポンスデータ。
   */
  async predict(request: SokutoRequest): Promise<SokutoResponse> {
    const url = `${this.baseUrl}/v1/systemone`;
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const resp = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      if (!resp.ok) {
        throw new Error(`sokuto-server HTTP ${resp.status}: ${await resp.text()}`);
      }

      return (await resp.json()) as SokutoResponse;
    } finally {
      clearTimeout(id);
    }
  }

  /**
   * サーバーの稼働状態 (/ready) を確認する。
   */
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
