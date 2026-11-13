//! # ロックフリー非同期リソース借用プールモジュール
//!
//! Tokio 非同期タスクとブロッキング推論スレッド間の排他借用を
//! `crossbeam_queue::ArrayQueue` と `tokio::sync::Semaphore` でロックフリーに制御する。
//!
//! 同期推論エンジン(`InferenceEngine` / `DualSessionEngine`)内部の同期ラウンドロビン方式
//! (`Vec<Mutex<Session>>`)とは異なり、非同期スケジューラ上で待機・タイムアウト・
//! スレッド間転送を行うタスク向けに特化した汎用リソース管理プリミティブを提供する。

use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::time::Duration;

use crossbeam_queue::ArrayQueue;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// プール借用時のエラー型。
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PoolError {
    /// セッション取得のハードタイムアウト。
    #[error("セッション借用がタイムアウトしました ({0:?})。プールが枯渇しています。")]
    Timeout(Duration),

    /// セマフォがクローズされた。
    #[error("セッションプールのセマフォがクローズされました。")]
    Closed,

    /// プール内部キューの不整合。
    #[error("セマフォ許可証の取得後にキューが空でした。")]
    QueueEmpty,
}

/// ロックフリーなリソース借用プール。
///
/// Tokio の非同期スケジューラ上で許可証を獲得後、RAII ガード構造体として
/// リソースを取り出し、`spawn_blocking` などのスレッド境界を越えて安全に転送可能にする。
#[derive(Debug, Clone)]
pub struct LowLatencyResourcePool<T: Send + 'static> {
    resources: Arc<ArrayQueue<T>>,
    permits: Arc<Semaphore>,
    capacity: usize,
}

impl<T: Send + 'static> LowLatencyResourcePool<T> {
    /// リソースリストからプールを新規作成する。
    ///
    /// `items` が空の場合は `capacity = 0` の無効化プールとして構築し、セマフォをクローズして
    /// `acquire` を即座に拒否する。
    ///
    /// # 引数
    /// - `items`: プールに初期格納するリソース一覧。
    pub fn new(items: Vec<T>) -> Self {
        let capacity = items.len();
        // ArrayQueue::new(0) はパニックするため、キュー自体の確保容量のみ最小 1 に補正する。
        let queue = Arc::new(ArrayQueue::new(capacity.max(1)));
        for item in items {
            let _ = queue.push(item);
        }
        let permits = Arc::new(Semaphore::new(capacity));
        if capacity == 0 {
            permits.close();
        }
        Self {
            resources: queue,
            permits,
            capacity,
        }
    }

    /// プールの最大キャパシティを取得する。
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// 現在キュー内で待機中の利用可能リソース数を取得する。
    pub fn available(&self) -> usize {
        self.permits.available_permits()
    }

    /// ハードタイムアウト付きでリソースを非同期借用する。
    ///
    /// Tokio 非同期コンテキストでセマフォ許可証を取得後、キューから `pop` する。
    /// 返却された `PooledResource` は `'static` であり、`spawn_blocking` への転送が可能である。
    /// キャパシティが 0 の空プールに対してはタイムアウト待機を行わず即座に `PoolError::Closed` を返す。
    ///
    /// # 引数
    /// - `timeout`: 取得待機タイムアウト時間。
    pub async fn acquire(&self, timeout: Duration) -> Result<PooledResource<T>, PoolError> {
        if self.capacity == 0 {
            return Err(PoolError::Closed);
        }

        let permit_result =
            tokio::time::timeout(timeout, self.permits.clone().acquire_owned()).await;

        let permit = match permit_result {
            Ok(Ok(p)) => p,
            Ok(Err(_)) => return Err(PoolError::Closed),
            Err(_) => return Err(PoolError::Timeout(timeout)),
        };

        let resource = self.resources.pop().ok_or(PoolError::QueueEmpty)?;

        Ok(PooledResource {
            resources: self.resources.clone(),
            resource: Some(resource),
            _permit: permit,
        })
    }
}

/// 借用中のリソースを保持する RAII ガード。
///
/// スコープを抜けた際に自動で元のプールキューへリソースを返却し、
/// セマフォ許可証を解放する。
pub struct PooledResource<T: Send + 'static> {
    resources: Arc<ArrayQueue<T>>,
    resource: Option<T>,
    _permit: OwnedSemaphorePermit,
}

impl<T: Send + 'static> Deref for PooledResource<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.resource
            .as_ref()
            .expect("リソースは生存期間中常に存在します。")
    }
}

impl<T: Send + 'static> DerefMut for PooledResource<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.resource
            .as_mut()
            .expect("リソースは生存期間中常に存在します。")
    }
}

impl<T: Send + 'static> Drop for PooledResource<T> {
    fn drop(&mut self) {
        if let Some(item) = self.resource.take() {
            let _ = self.resources.push(item);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_resource_pool_acquire_and_release() {
        let pool = LowLatencyResourcePool::new(vec![100, 200]);
        assert_eq!(pool.capacity(), 2);
        assert_eq!(pool.available(), 2);

        {
            let guard1 = pool
                .acquire(Duration::from_millis(50))
                .await
                .expect("借用成功すること。");
            assert_eq!(*guard1, 100);
            assert_eq!(pool.available(), 1);

            let guard2 = pool
                .acquire(Duration::from_millis(50))
                .await
                .expect("借用成功すること。");
            assert_eq!(*guard2, 200);
            assert_eq!(pool.available(), 0);

            // タイムアウトテスト (空きがない場合)
            let timeout_res = pool.acquire(Duration::from_millis(10)).await;
            assert!(matches!(timeout_res, Err(PoolError::Timeout(_))));
        }

        // drop 後に返却されていること
        assert_eq!(pool.available(), 2);
        let guard_reacquired = pool
            .acquire(Duration::from_millis(50))
            .await
            .expect("再借用成功すること。");
        assert!(*guard_reacquired == 100 || *guard_reacquired == 200);
    }

    #[tokio::test]
    async fn test_transfer_to_spawn_blocking() {
        let pool = LowLatencyResourcePool::new(vec!["session_a".to_string()]);
        let guard = pool
            .acquire(Duration::from_millis(50))
            .await
            .expect("借用成功すること。");

        let handle = tokio::task::spawn_blocking(move || {
            let val = guard.clone();
            format!("{val}_processed")
        });

        let res = handle.await.expect("スレッド完了すること。");
        assert_eq!(res, "session_a_processed");
        assert_eq!(pool.available(), 1);
    }

    #[tokio::test]
    async fn test_empty_resource_pool_rejects_acquire_immediately() {
        let pool: LowLatencyResourcePool<i32> = LowLatencyResourcePool::new(vec![]);
        assert_eq!(pool.capacity(), 0);
        assert_eq!(pool.available(), 0);

        let start = std::time::Instant::now();
        let res = pool.acquire(Duration::from_secs(5)).await;
        assert!(matches!(res, Err(PoolError::Closed)));
        assert!(
            start.elapsed() < Duration::from_millis(100),
            "空プールではタイムアウト待機せず即座に拒否されること。"
        );
    }
}
