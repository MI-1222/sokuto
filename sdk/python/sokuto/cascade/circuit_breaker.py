"""サーキットブレーカーおよび高可用性フェイルオープン機構モジュール。

sokuto サーバーの障害、過負荷、タイムアウトを検知した際に
System 1 への無駄なアクセスを遮断し、System 2 へ即時フェイルオープンするとともに、
ヘルスチェックエンドポイント (/ready) による自動復帰を管理する。
"""

from __future__ import annotations

import threading
import time
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx


class CircuitState(StrEnum):
    """サーキットブレーカーの動作状態。"""

    CLOSED = "closed"  # 正常稼働: System 1 へリクエストを通す
    OPEN = "open"  # 障害遮断: System 1 をバイパスして System 2 へフェイルオープン
    HALF_OPEN = "half_open"  # 復帰試験: 試験的にリクエストを送信して回復を確認


class CircuitBreaker:
    """System 1 通信を保護するインメモリサーキットブレーカー。

    Attributes:
        failure_threshold (int): OPEN へ遷移する連続失敗回数閾値。
        recovery_timeout (float): OPEN から HALF_OPEN へ移行するまでのクールダウン秒数。
        success_threshold (int): HALF_OPEN から CLOSED へ復帰するために必要な連続成功回数。
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 10.0,
        success_threshold: int = 2,
    ) -> None:
        """サーキットブレーカーを初期化する。

        Args:
            failure_threshold (int): 連続失敗許容回数。
            recovery_timeout (float): クールダウン秒数。
            success_threshold (int): 復帰に必要な連続成功回数。
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_state_change = time.monotonic()
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """現在のサーキット状態を取得する。クールダウン経過時は自動で HALF_OPEN に遷移する。"""
        with self._lock:
            now = time.monotonic()
            if (
                self._state == CircuitState.OPEN
                and (now - self._last_state_change) >= self.recovery_timeout
            ):
                self._state = CircuitState.HALF_OPEN
                self._last_state_change = now
                self._consecutive_successes = 0
            return self._state

    def can_execute(self) -> bool:
        """System 1 を呼び出してよいかどうかを判定する。

        Returns:
            bool: CLOSED または HALF_OPEN のとき True、OPEN のとき False。
        """
        return self.state != CircuitState.OPEN

    def record_success(self) -> None:
        """System 1 呼び出しの成功を記録し、必要に応じて CLOSED へ復帰する。"""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._consecutive_failures = 0
                    self._consecutive_successes = 0
                    self._last_state_change = time.monotonic()
            elif self._state == CircuitState.CLOSED:
                self._consecutive_failures = 0

    def record_failure(self) -> None:
        """System 1 呼び出しの失敗を記録し、閾値超過時に OPEN へ遮断する。"""
        with self._lock:
            self._consecutive_failures += 1
            if self._state == CircuitState.HALF_OPEN:
                # 復帰試験中の失敗は即座に OPEN へ再突入
                self._state = CircuitState.OPEN
                self._last_state_change = time.monotonic()
                self._consecutive_successes = 0
            elif (
                self._state == CircuitState.CLOSED
                and self._consecutive_failures >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._last_state_change = time.monotonic()
                self._consecutive_successes = 0

    def reset(self) -> None:
        """サーキットブレーカーを CLOSED 状態へ強制リセットする。"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._last_state_change = time.monotonic()

    async def check_health_async(
        self,
        http_client: httpx.AsyncClient,
        ready_endpoint: str = "/ready",
        timeout: float = 1.0,
    ) -> bool:
        """sokuto-server の /ready エンドポイントへヘルスチェックを行い、成功時に復帰する。

        Args:
            http_client (httpx.AsyncClient): 非同期 HTTP クライアント。
            ready_endpoint (str): ヘルスチェックパス。
            timeout (float): ヘルスチェックのタイムアウト秒数。

        Returns:
            bool: 正常稼働していれば True。
        """
        try:
            resp = await http_client.get(ready_endpoint, timeout=timeout)
            if resp.status_code == 200:
                self.record_success()
                return True
        except Exception:
            pass

        self.record_failure()
        return False
