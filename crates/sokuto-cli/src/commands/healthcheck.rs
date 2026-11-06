//! # ヘルスチェック実行コマンド
//!
//! 最小コンテナ環境など curl が存在しない環境において、
//! 外部依存なしで HTTP エンドポイントの死活監視・準備状態確認 (Liveness / Readiness) を行う。

use std::time::{Duration, Instant};

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::error::CliError;

/// ヘルスチェック実行引数構造体。
#[derive(Debug, Clone, clap::Args, PartialEq, Eq)]
pub struct HealthcheckArgs {
    /// 監視対象 URL (例: `http://127.0.0.1:3000/ready`)。
    #[arg(
        short,
        long,
        env = "SOKUTO_HEALTHCHECK_URL",
        default_value = "http://127.0.0.1:3000/ready"
    )]
    pub url: String,

    /// 接続および応答タイムアウト秒数。
    #[arg(short, long, default_value_t = 3, value_parser = clap::value_parser!(u64).range(1..))]
    pub timeout_secs: u64,
}

/// URL をホスト、ポート、パスにパースする。
fn parse_http_url(url: &str) -> Result<(String, u16, String), CliError> {
    let stripped = url.strip_prefix("http://").ok_or_else(|| {
        CliError::HealthcheckFailed(format!(
            "無効な URL スキームです。http:// で始まる必要があります: '{url}'。"
        ))
    })?;

    let (host_port, path) = match stripped.find('/') {
        Some(idx) => (&stripped[..idx], &stripped[idx..]),
        None => (stripped, "/"),
    };

    let (host, port) = if let Some(rest) = host_port.strip_prefix('[') {
        let close_idx = rest.find(']').ok_or_else(|| {
            CliError::HealthcheckFailed(format!(
                "無効な IPv6 アドレス表記です。']' がありません: '{host_port}'。"
            ))
        })?;
        let ip_str = &rest[..close_idx];
        if ip_str.is_empty() {
            return Err(CliError::HealthcheckFailed(format!(
                "IPv6 アドレスが空です: '{host_port}'。"
            )));
        }
        ip_str.parse::<std::net::Ipv6Addr>().map_err(|e| {
            CliError::HealthcheckFailed(format!("無効な IPv6 アドレスです: '{ip_str}' ({e})。"))
        })?;

        let after = &rest[close_idx + 1..];
        let port = if after.is_empty() {
            80
        } else if let Some(port_str) = after.strip_prefix(':') {
            port_str.parse::<u16>().map_err(|e| {
                CliError::HealthcheckFailed(format!(
                    "ポート番号のパースに失敗しました: '{port_str}' ({e})。"
                ))
            })?
        } else {
            return Err(CliError::HealthcheckFailed(format!(
                "無効な IPv6 ホスト・ポート形式です: '{host_port}'。"
            )));
        };
        (format!("[{ip_str}]"), port)
    } else if let Some(idx) = host_port.rfind(':') {
        let host = &host_port[..idx];
        if host.is_empty() {
            return Err(CliError::HealthcheckFailed(format!(
                "ホスト名が空です: '{host_port}'。"
            )));
        }
        let port_str = &host_port[idx + 1..];
        let port = port_str.parse::<u16>().map_err(|e| {
            CliError::HealthcheckFailed(format!(
                "ポート番号のパースに失敗しました: '{port_str}' ({e})。"
            ))
        })?;
        (host.to_string(), port)
    } else {
        if host_port.is_empty() {
            return Err(CliError::HealthcheckFailed(format!(
                "ホスト名が空です: '{url}'。"
            )));
        }
        (host_port.to_string(), 80)
    };

    let path = if path.is_empty() {
        "/".to_string()
    } else {
        path.to_string()
    };

    Ok((host, port, path))
}

/// ホスト名およびポート番号からソケットアドレスを解決する。
///
/// IP アドレス文字列（IPv4 または `[::1]` などの IPv6 リテラル）の場合は DNS 問い合わせを行わずに即座にパースし、
/// ホスト名の場合は Tokio の非同期名前解決とタイムアウトを用いて指定期限 (`deadline`) 内での解決を保証する。
async fn resolve_socket_addrs(
    host: &str,
    port: u16,
    deadline: Instant,
) -> Result<Vec<std::net::SocketAddr>, CliError> {
    let ip_str = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);

    if let Ok(ip) = ip_str.parse::<std::net::IpAddr>() {
        return Ok(vec![std::net::SocketAddr::new(ip, port)]);
    }

    let now = Instant::now();
    let remaining = deadline
        .checked_duration_since(now)
        .ok_or_else(|| CliError::HealthcheckFailed("名前解決がタイムアウトしました。".into()))?;
    if remaining.is_zero() {
        return Err(CliError::HealthcheckFailed(
            "名前解決がタイムアウトしました。".into(),
        ));
    }

    let host_port = format!("{host}:{port}");
    match tokio::time::timeout(remaining, tokio::net::lookup_host(&host_port)).await {
        Ok(Ok(iter)) => Ok(iter.collect()),
        Ok(Err(e)) => Err(CliError::HealthcheckFailed(format!(
            "ホスト名 '{host}' の名前解決に失敗しました: {e}。"
        ))),
        Err(_) => Err(CliError::HealthcheckFailed(format!(
            "ホスト名 '{host}' の名前解決がタイムアウトしました。"
        ))),
    }
}

/// HTTP GET リクエストを送信し、ステータスコード 200 OK であるかを判定する。
///
/// # 引数
/// - `args`: ヘルスチェック対象 URL およびタイムアウト設定。
pub async fn run_healthcheck(args: HealthcheckArgs) -> Result<(), CliError> {
    if args.timeout_secs == 0 {
        return Err(CliError::HealthcheckFailed(
            "タイムアウト秒数には 1 以上の値を指定してください。".into(),
        ));
    }

    let (host, port, path) = parse_http_url(&args.url)?;
    let timeout = Duration::from_secs(args.timeout_secs);
    let deadline = Instant::now() + timeout;

    let socket_addrs = resolve_socket_addrs(&host, port, deadline).await?;
    if socket_addrs.is_empty() {
        return Err(CliError::HealthcheckFailed(format!(
            "ホスト名のアドレス解決に失敗しました: {host}:{port}。"
        )));
    }

    let mut last_err = None;
    let mut stream = None;

    for addr in socket_addrs {
        let now = Instant::now();
        let Some(remaining) = deadline.checked_duration_since(now) else {
            return Err(CliError::HealthcheckFailed(
                "接続試行がタイムアウトしました。".into(),
            ));
        };
        if remaining.is_zero() {
            return Err(CliError::HealthcheckFailed(
                "接続試行がタイムアウトしました。".into(),
            ));
        }

        match tokio::time::timeout(remaining, TcpStream::connect(&addr)).await {
            Ok(Ok(s)) => {
                stream = Some(s);
                break;
            }
            Ok(Err(e)) => {
                last_err = Some(e);
            }
            Err(_) => {
                last_err = Some(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "接続試行がタイムアウトしました。",
                ));
            }
        }
    }

    let mut stream = match stream {
        Some(s) => s,
        None => {
            let err_msg = last_err
                .map(|e| e.to_string())
                .unwrap_or_else(|| "接続先アドレスがありません。".to_string());
            return Err(CliError::HealthcheckFailed(format!(
                "{host}:{port} への接続に失敗しました: {err_msg}。"
            )));
        }
    };

    let now = Instant::now();
    let remaining = deadline.checked_duration_since(now).ok_or_else(|| {
        CliError::HealthcheckFailed("リクエスト送信がタイムアウトしました。".into())
    })?;
    if remaining.is_zero() {
        return Err(CliError::HealthcheckFailed(
            "リクエスト送信がタイムアウトしました。".into(),
        ));
    }

    let user_agent = concat!("sokuto-healthcheck/", env!("CARGO_PKG_VERSION"));
    let request = format!(
        "GET {path} HTTP/1.1\r\n\
         Host: {host}:{port}\r\n\
         User-Agent: {user_agent}\r\n\
         Connection: close\r\n\
         \r\n"
    );

    tokio::time::timeout(remaining, stream.write_all(request.as_bytes()))
        .await
        .map_err(|_| CliError::HealthcheckFailed("リクエスト送信がタイムアウトしました。".into()))?
        .map_err(CliError::Io)?;

    let (reader, _) = stream.into_split();
    let mut reader = tokio::io::BufReader::new(reader);
    let mut status_line = String::new();
    const MAX_STATUS_LINE_LEN: usize = 8192;

    loop {
        status_line.clear();
        let now = Instant::now();
        let Some(remaining) = deadline.checked_duration_since(now) else {
            return Err(CliError::HealthcheckFailed(
                "応答読み取りがタイムアウトしました。".into(),
            ));
        };
        if remaining.is_zero() {
            return Err(CliError::HealthcheckFailed(
                "応答読み取りがタイムアウトしました。".into(),
            ));
        }

        let mut handle = (&mut reader).take(MAX_STATUS_LINE_LEN as u64);
        let read_res = tokio::time::timeout(remaining, handle.read_line(&mut status_line)).await;
        match read_res {
            Ok(Ok(0)) => {
                return Err(CliError::HealthcheckFailed(
                    "空の HTTP レスポンスを受信しました。".into(),
                ));
            }
            Ok(Ok(_)) => {
                if !status_line.ends_with('\n') {
                    return Err(CliError::HealthcheckFailed(format!(
                        "HTTP ステータス行が長すぎるか、不完全です (上限 {MAX_STATUS_LINE_LEN} バイト)。"
                    )));
                }
                let trimmed = status_line.trim_matches(|c| c == '\r' || c == '\n');
                if trimmed.is_empty() {
                    // RFC 7230 §3.5: 先行する空行 (CRLF) を読み飛ばす。
                    continue;
                }
                break;
            }
            Ok(Err(e)) => return Err(CliError::Io(e)),
            Err(_) => {
                return Err(CliError::HealthcheckFailed(
                    "応答読み取りがタイムアウトしました。".into(),
                ));
            }
        }
    }

    let trimmed = status_line.trim_matches(|c| c == '\r' || c == '\n');
    let status_code = parse_http_status_line(trimmed)?;

    if (200..=299).contains(&status_code) {
        tracing::debug!("ヘルスチェック成功: {trimmed}。");
        Ok(())
    } else {
        Err(CliError::HealthcheckFailed(format!(
            "ヘルスチェック失敗 (ステータスコード {status_code}): '{trimmed}'。"
        )))
    }
}

/// HTTP ステータス行を検証し、ステータスコードを抽出する。
fn parse_http_status_line(status_line: &str) -> Result<u16, CliError> {
    let trimmed = status_line.trim_matches(|c| c == '\r' || c == '\n');
    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    if parts.len() < 2 || !parts[0].starts_with("HTTP/") {
        return Err(CliError::HealthcheckFailed(format!(
            "不正な HTTP ステータス行です: '{trimmed}'。"
        )));
    }

    let status_code = parts[1].parse::<u16>().map_err(|e| {
        CliError::HealthcheckFailed(format!(
            "ステータスコードのパースに失敗しました: '{}' ({e})。",
            parts[1]
        ))
    })?;

    Ok(status_code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_http_url() {
        let (host, port, path) = parse_http_url("http://127.0.0.1:3000/ready").unwrap();
        assert_eq!(host, "127.0.0.1");
        assert_eq!(port, 3000);
        assert_eq!(path, "/ready");

        let (host, port, path) = parse_http_url("http://localhost/health").unwrap();
        assert_eq!(host, "localhost");
        assert_eq!(port, 80);
        assert_eq!(path, "/health");

        // IPv6 リテラル (ポート指定あり)
        let (host, port, path) = parse_http_url("http://[::1]:3000/ready").unwrap();
        assert_eq!(host, "[::1]");
        assert_eq!(port, 3000);
        assert_eq!(path, "/ready");

        // IPv6 リテラル (ポート省略)
        let (host, port, path) = parse_http_url("http://[::1]/ready").unwrap();
        assert_eq!(host, "[::1]");
        assert_eq!(port, 80);
        assert_eq!(path, "/ready");

        // IPv6 リテラル (パス省略)
        let (host, port, path) = parse_http_url("http://[::1]").unwrap();
        assert_eq!(host, "[::1]");
        assert_eq!(port, 80);
        assert_eq!(path, "/");

        // 不正なスキーム
        let err = parse_http_url("https://localhost:3000/health");
        assert!(err.is_err());

        // 不正な IPv6 表記 (閉じ括弧なし)
        let err = parse_http_url("http://[::1:3000/ready");
        assert!(err.is_err());

        // 不正な IPv6 アドレス
        let err = parse_http_url("http://[not-an-ip]:3000/ready");
        assert!(err.is_err());

        // ホスト名が空
        let err = parse_http_url("http://:3000/ready");
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn test_run_healthcheck_ipv6() {
        use std::io::Write;
        use std::net::TcpListener;
        use std::thread;

        let Ok(listener) = TcpListener::bind("[::1]:0") else {
            // IPv6 ループバックが利用できない環境ではスキップ。
            return;
        };
        let port = listener.local_addr().unwrap().port();

        let server_handle = thread::spawn(move || {
            if let Ok((mut socket, _)) = listener.accept() {
                let response = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK";
                let _ = socket.write_all(response.as_bytes());
            }
        });

        let args = HealthcheckArgs {
            url: format!("http://[::1]:{port}/health"),
            timeout_secs: 5,
        };

        let result = run_healthcheck(args).await;
        assert!(result.is_ok());

        let _ = server_handle.join();
    }

    #[tokio::test]
    async fn test_run_healthcheck_immediate_return_without_close() {
        use std::io::Write;
        use std::net::TcpListener;
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        let server_handle = thread::spawn(move || {
            if let Ok((mut socket, _)) = listener.accept() {
                // HTTP 200 OK を送信するが、ソケットを close せず待機する。
                let response = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK";
                let _ = socket.write_all(response.as_bytes());
                // クライアントが切断または早期脱出するまで少し待つ。
                thread::sleep(Duration::from_millis(500));
            }
        });

        let args = HealthcheckArgs {
            url: format!("http://127.0.0.1:{port}/health"),
            timeout_secs: 5,
        };

        let start = std::time::Instant::now();
        let result = run_healthcheck(args).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        // 5秒のタイムアウトを待たずに即座 (500ms 未満) に脱出できたことを検証。
        assert!(elapsed < Duration::from_millis(400));

        let _ = server_handle.join();
    }

    #[tokio::test]
    async fn test_run_healthcheck_fragmented_status_line() {
        use std::io::Write;
        use std::net::TcpListener;
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        let server_handle = thread::spawn(move || {
            if let Ok((mut socket, _)) = listener.accept() {
                // パケット分割を模倣: "HTTP/1.1 200 OK\r" の \r 直後で分割して送信。
                let chunk1 = b"HTTP/1.1 200 OK\r";
                socket.write_all(chunk1).unwrap();
                socket.flush().unwrap();

                thread::sleep(Duration::from_millis(30));

                let chunk2 = b"\nContent-Length: 2\r\n\r\nOK";
                socket.write_all(chunk2).unwrap();
                socket.flush().unwrap();
            }
        });

        let args = HealthcheckArgs {
            url: format!("http://127.0.0.1:{port}/health"),
            timeout_secs: 5,
        };

        let result = run_healthcheck(args).await;
        assert!(result.is_ok());

        let _ = server_handle.join();
    }

    #[tokio::test]
    async fn test_run_healthcheck_leading_empty_lines() {
        use std::io::Write;
        use std::net::TcpListener;
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        let server_handle = thread::spawn(move || {
            if let Ok((mut socket, _)) = listener.accept() {
                // 先行する CRLF 空行を含むレスポンスを送信。
                let response = "\r\n\r\nHTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK";
                socket.write_all(response.as_bytes()).unwrap();
                socket.flush().unwrap();
            }
        });

        let args = HealthcheckArgs {
            url: format!("http://127.0.0.1:{port}/health"),
            timeout_secs: 5,
        };

        let result = run_healthcheck(args).await;
        assert!(result.is_ok());

        let _ = server_handle.join();
    }

    #[tokio::test]
    async fn test_run_healthcheck_timeout_secs_zero() {
        let args = HealthcheckArgs {
            url: "http://127.0.0.1:3000/ready".to_string(),
            timeout_secs: 0,
        };
        let result = run_healthcheck(args).await;
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "ヘルスチェック失敗: タイムアウト秒数には 1 以上の値を指定してください。"
        );
    }

    #[test]
    fn test_healthcheck_args_clap_validation() {
        use clap::Parser;

        #[derive(Debug, Parser)]
        struct TestCli {
            #[command(flatten)]
            args: HealthcheckArgs,
        }

        // 0 はバリデーションエラーとなる
        let res = TestCli::try_parse_from(["test", "--timeout-secs", "0"]);
        assert!(res.is_err());

        // 1 以上は正常にパースできる
        let parsed = TestCli::try_parse_from(["test", "--timeout-secs", "1"]).unwrap();
        assert_eq!(parsed.args.timeout_secs, 1);
    }

    #[tokio::test]
    async fn test_resolve_socket_addrs() {
        let deadline = Instant::now() + Duration::from_secs(5);

        // IPv4 アドレス
        let addrs = resolve_socket_addrs("127.0.0.1", 8080, deadline)
            .await
            .unwrap();
        assert_eq!(addrs.len(), 1);
        assert_eq!(addrs[0], "127.0.0.1:8080".parse().unwrap());

        // IPv6 アドレス (ブラケット付き)
        let addrs = resolve_socket_addrs("[::1]", 8080, deadline).await.unwrap();
        assert_eq!(addrs.len(), 1);
        assert_eq!(addrs[0], "[::1]:8080".parse().unwrap());

        // 期限切れ
        let expired_deadline = Instant::now() - Duration::from_millis(10);
        let err = resolve_socket_addrs("localhost", 8080, expired_deadline).await;
        assert!(err.is_err());
    }

    #[test]
    fn test_parse_http_status_line() {
        // 正常な HTTP ステータス行
        assert_eq!(parse_http_status_line("HTTP/1.1 200 OK").unwrap(), 200);
        assert_eq!(
            parse_http_status_line("HTTP/1.1 204 No Content").unwrap(),
            204
        );
        assert_eq!(parse_http_status_line("HTTP/2 200").unwrap(), 200);
        assert_eq!(
            parse_http_status_line("HTTP/1.0 503 Service Unavailable\r\n").unwrap(),
            503
        );

        // 非 HTTP サービスの応答 (HTTP/ から始まらない)
        let err = parse_http_status_line("SSH-2.0-OpenSSH_8.9").unwrap_err();
        assert!(err.to_string().contains("不正な HTTP ステータス行です"));

        let err = parse_http_status_line("220 Service ready").unwrap_err();
        assert!(err.to_string().contains("不正な HTTP ステータス行です"));

        // トークン不足
        let err = parse_http_status_line("HTTP/1.1").unwrap_err();
        assert!(err.to_string().contains("不正な HTTP ステータス行です"));

        // ステータスコードが数値でない
        let err = parse_http_status_line("HTTP/1.1 BAD OK").unwrap_err();
        assert!(
            err.to_string()
                .contains("ステータスコードのパースに失敗しました")
        );
    }

    #[tokio::test]
    async fn test_run_healthcheck_204_no_content() {
        use std::io::Write;
        use std::net::TcpListener;
        use std::thread;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();

        let server_handle = thread::spawn(move || {
            if let Ok((mut socket, _)) = listener.accept() {
                let response = "HTTP/1.1 204 No Content\r\n\r\n";
                socket.write_all(response.as_bytes()).unwrap();
                socket.flush().unwrap();
            }
        });

        let args = HealthcheckArgs {
            url: format!("http://127.0.0.1:{port}/ready"),
            timeout_secs: 5,
        };

        let result = run_healthcheck(args).await;
        assert!(result.is_ok());

        let _ = server_handle.join();
    }
}
