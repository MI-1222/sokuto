/**
 * サーキットブレーカーおよび高可用性フェイルオープン機構モジュール。
 *
 * sokuto サーバーの障害、過負荷、タイムアウトを検知した際に
 * System 1 へのアクセスを遮断して System 2 へ即時フェイルオープンし、
 * ヘルスチェックエンドポイント (/ready) による自動復帰を管理する。
 */

export type CircuitState = "closed" | "open" | "half_open";

/**
 * System 1 通信を保護するインメモリサーキットブレーカー。
 */
export class CircuitBreaker {
  public readonly failureThreshold: number;
  public readonly recoveryTimeoutMs: number;
  public readonly successThreshold: number;

  private _state: CircuitState = "closed";
  private _consecutiveFailures = 0;
  private _consecutiveSuccesses = 0;
  private _lastStateChange = Date.now();

  /**
   * サーキットブレーカーを初期化する。
   *
   * @param options 設定オプション。
   */
  constructor(options?: {
    failureThreshold?: number;
    recoveryTimeoutMs?: number;
    successThreshold?: number;
  }) {
    this.failureThreshold = options?.failureThreshold ?? 5;
    this.recoveryTimeoutMs = options?.recoveryTimeoutMs ?? 10000;
    this.successThreshold = options?.successThreshold ?? 2;
  }

  /**
   * 現在のサーキット状態を取得する。クールダウン経過時は自動で half_open に遷移する。
   */
  get state(): CircuitState {
    const now = Date.now();
    if (
      this._state === "open" &&
      now - this._lastStateChange >= this.recoveryTimeoutMs
    ) {
      this._state = "half_open";
      this._lastStateChange = now;
      this._consecutiveSuccesses = 0;
    }
    return this._state;
  }

  /**
   * System 1 を呼び出してよいかどうかを判定する。
   */
  canExecute(): boolean {
    return this.state !== "open";
  }

  /**
   * System 1 呼び出しの成功を記録し、必要に応じて closed へ復帰する。
   */
  recordSuccess(): void {
    if (this._state === "half_open") {
      this._consecutiveSuccesses++;
      if (this._consecutiveSuccesses >= this.successThreshold) {
        this._state = "closed";
        this._consecutiveFailures = 0;
        this._consecutiveSuccesses = 0;
        this._lastStateChange = Date.now();
      }
    } else if (this._state === "closed") {
      this._consecutiveFailures = 0;
    }
  }

  /**
   * System 1 呼び出しの失敗を記録し、閾値超過時に open へ遮断する。
   */
  recordFailure(): void {
    this._consecutiveFailures++;
    if (this._state === "half_open") {
      this._state = "open";
      this._lastStateChange = Date.now();
      this._consecutiveSuccesses = 0;
    } else if (
      this._state === "closed" &&
      this._consecutiveFailures >= this.failureThreshold
    ) {
      this._state = "open";
      this._lastStateChange = Date.now();
      this._consecutiveSuccesses = 0;
    }
  }

  /**
   * サーキットブレーカーを closed 状態へ強制リセットする。
   */
  reset(): void {
    this._state = "closed";
    this._consecutiveFailures = 0;
    this._consecutiveSuccesses = 0;
    this._lastStateChange = Date.now();
  }

  /**
   * サーバーの /ready エンドポイントへヘルスチェックを行い、回復を確認する。
   *
   * @param baseUrl sokuto サーバーの URL。
   * @param timeoutMs タイムアウト (ミリ秒)。
   * @returns 正常稼働していれば true。
   */
  async checkHealth(baseUrl: string, timeoutMs = 1000): Promise<boolean> {
    try {
      const url = `${baseUrl.replace(/\/+$/, "")}/ready`;
      const controller = new AbortController();
      const id = setTimeout(() => controller.abort(), timeoutMs);

      const resp = await fetch(url, { signal: controller.signal });
      clearTimeout(id);

      if (resp.ok) {
        this.recordSuccess();
        return true;
      }
    } catch {
      // 通信失敗
    }

    this.recordFailure();
    return false;
  }
}
