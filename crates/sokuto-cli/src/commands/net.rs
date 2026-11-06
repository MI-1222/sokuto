//! # ネットワーク共通ユーティリティモジュール
//!
//! ホスト名の正規化およびソケットアドレスの非同期名前解決処理を提供する。

use std::net::{IpAddr, SocketAddr};
use std::time::Instant;
use thiserror::Error;

/// アドレス解決処理で発生するエラー型。
#[derive(Debug, Error, PartialEq, Eq)]
pub enum NetError {
    /// ホスト名解決の失敗（DNS 解決エラー等）。
    #[error("ホスト名 '{host}:{port}' の名前解決に失敗しました: {message}。")]
    LookupFailed {
        /// 対象ホスト名。
        host: String,
        /// 対象ポート番号。
        port: u16,
        /// エラーメッセージ。
        message: String,
    },

    /// ホスト名に対応するアドレスが存在しない。
    #[error("ホスト名 '{host}:{port}' に対応するソケットアドレスが見つかりませんでした。")]
    NoAddressesFound {
        /// 対象ホスト名。
        host: String,
        /// 対象ポート番号。
        port: u16,
    },

    /// 名前解決処理のタイムアウト。
    #[error("ホスト名 '{host}:{port}' の名前解決がタイムアウトしました。")]
    Timeout {
        /// 対象ホスト名。
        host: String,
        /// 対象ポート番号。
        port: u16,
    },
}

/// ホスト名から角括弧（IPv6 リテラル用の `[` と `]`）を除去して正規化する。
///
/// 前後両方に角括弧が存在する場合にのみ除去し、角括弧のないホスト名や IP アドレスはそのまま返す。
///
/// # 引数
/// - `host`: 対象のホスト名または IP 文字列。
#[must_use]
pub fn normalize_host(host: &str) -> &str {
    host.strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host)
}

/// ホスト名およびポート番号からソケットアドレス一覧を非同期解決する。
///
/// IP アドレス文字列（IPv4 または `[::1]` などの IPv6 リテラル）の場合は
/// DNS 問い合わせを行わずに即座にパースし、
/// ホスト名の場合は Tokio の非同期名前解決 (`tokio::net::lookup_host`) を用いて解決する。
///
/// `deadline` が指定されている場合は、指定時刻を超過した場合にタイムアウトエラーを返す。
///
/// # 引数
/// - `host`: ホスト名または IP アドレス文字列。
/// - `port`: ポート番号。
/// - `deadline`: タイムアウト期限（任意）。
///
/// # エラー
/// 名前解決に失敗した場合、対応アドレスが存在しない場合、またはタイムアウトした場合に [`NetError`] を返す。
pub async fn resolve_host_port_with_deadline(
    host: &str,
    port: u16,
    deadline: Option<Instant>,
) -> Result<Vec<SocketAddr>, NetError> {
    let normalized = normalize_host(host);

    if let Ok(ip) = normalized.parse::<IpAddr>() {
        return Ok(vec![SocketAddr::new(ip, port)]);
    }

    let lookup_future = tokio::net::lookup_host((normalized, port));

    let iter = match deadline {
        Some(dl) => {
            let now = Instant::now();
            let remaining = dl.checked_duration_since(now).unwrap_or_default();
            if remaining.is_zero() {
                return Err(NetError::Timeout {
                    host: host.to_string(),
                    port,
                });
            }

            match tokio::time::timeout(remaining, lookup_future).await {
                Ok(Ok(iter)) => iter,
                Ok(Err(err)) => {
                    return Err(NetError::LookupFailed {
                        host: host.to_string(),
                        port,
                        message: err.to_string(),
                    });
                }
                Err(_) => {
                    return Err(NetError::Timeout {
                        host: host.to_string(),
                        port,
                    });
                }
            }
        }
        None => lookup_future.await.map_err(|err| NetError::LookupFailed {
            host: host.to_string(),
            port,
            message: err.to_string(),
        })?,
    };

    let addrs: Vec<SocketAddr> = iter.collect();
    if addrs.is_empty() {
        return Err(NetError::NoAddressesFound {
            host: host.to_string(),
            port,
        });
    }

    Ok(addrs)
}

/// ホスト名およびポート番号からソケットアドレス一覧を非同期解決する。
///
/// タイムアウト制限を設けずに名前解決を実行する。
///
/// # 引数
/// - `host`: ホスト名または IP アドレス文字列。
/// - `port`: ポート番号。
///
/// # エラー
/// 名前解決に失敗した場合、または対応アドレスが存在しない場合に [`NetError`] を返す。
pub async fn resolve_host_port(host: &str, port: u16) -> Result<Vec<SocketAddr>, NetError> {
    resolve_host_port_with_deadline(host, port, None).await
}

/// ホスト名およびポート番号から単一のソケットアドレスを非同期解決する。
///
/// 最初に見つかったソケットアドレスを返す。
///
/// # 引数
/// - `host`: ホスト名または IP アドレス文字列。
/// - `port`: ポート番号。
///
/// # エラー
/// 名前解決に失敗した場合、または対応アドレスが存在しない場合に [`NetError`] を返す。
pub async fn resolve_single_socket_addr(host: &str, port: u16) -> Result<SocketAddr, NetError> {
    let mut addrs = resolve_host_port(host, port).await?;
    Ok(addrs.swap_remove(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_normalize_host() {
        assert_eq!(normalize_host("127.0.0.1"), "127.0.0.1");
        assert_eq!(normalize_host("localhost"), "localhost");
        assert_eq!(normalize_host("::1"), "::1");
        assert_eq!(normalize_host("[::1]"), "::1");
        assert_eq!(normalize_host("[localhost]"), "localhost");
        assert_eq!(normalize_host("[incomplete"), "[incomplete");
        assert_eq!(normalize_host("incomplete]"), "incomplete]");
    }

    #[tokio::test]
    async fn test_resolve_host_port_ipv4() {
        let addrs = resolve_host_port("127.0.0.1", 8080)
            .await
            .expect("IPv4 アドレスの解決に失敗。");
        assert_eq!(addrs, vec!["127.0.0.1:8080".parse::<SocketAddr>().unwrap()]);
    }

    #[tokio::test]
    async fn test_resolve_host_port_ipv6() {
        let addrs = resolve_host_port("::1", 8080)
            .await
            .expect("IPv6 アドレス (角括弧なし) の解決に失敗。");
        assert_eq!(addrs, vec!["[::1]:8080".parse::<SocketAddr>().unwrap()]);
    }

    #[tokio::test]
    async fn test_resolve_host_port_ipv6_bracketed() {
        let addrs = resolve_host_port("[::1]", 8080)
            .await
            .expect("角括弧付き IPv6 アドレスの解決に失敗。");
        assert_eq!(addrs, vec!["[::1]:8080".parse::<SocketAddr>().unwrap()]);
    }

    #[tokio::test]
    async fn test_resolve_single_socket_addr_localhost() {
        let addr = resolve_single_socket_addr("localhost", 8080)
            .await
            .expect("localhost の解決に失敗。");
        assert_eq!(addr.port(), 8080);
        assert!(addr.ip().is_loopback());
    }

    #[tokio::test]
    async fn test_resolve_host_port_timeout() {
        let expired_deadline = Instant::now() - Duration::from_millis(10);
        let err = resolve_host_port_with_deadline("localhost", 8080, Some(expired_deadline)).await;
        assert!(matches!(err, Err(NetError::Timeout { .. })));
    }
}
