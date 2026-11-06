//! # ヘルスチェック実行コマンド
//!
//! 最小コンテナ環境など curl が存在しない環境において、
//! 外部依存なしで HTTP エンドポイントの死活監視・準備状態確認 (Liveness / Readiness) を行う。

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

/// ヘルスチェック実行引数構造体。
#[derive(Debug, Clone)]
pub struct HealthcheckArgs {
    /// 監視対象 URL (例: `http://127.0.0.1:3000/ready`)。
    pub url: String,
    /// 接続および応答タイムアウト秒数。
    pub timeout_secs: u64,
}

/// URL をホスト、ポート、パスにパースする。
fn parse_http_url(url: &str) -> Result<(String, u16, String), String> {
    let stripped = url.strip_prefix("http://").ok_or_else(|| {
        format!("無効な URL スキームです。http:// で始まる必要があります: '{url}'。")
    })?;

    let (host_port, path) = match stripped.find('/') {
        Some(idx) => (&stripped[..idx], &stripped[idx..]),
        None => (stripped, "/"),
    };

    let (host, port) = if let Some(rest) = host_port.strip_prefix('[') {
        let close_idx = rest.find(']').ok_or_else(|| {
            format!("無効な IPv6 アドレス表記です。']' がありません: '{host_port}'。")
        })?;
        let ip_str = &rest[..close_idx];
        if ip_str.is_empty() {
            return Err(format!("IPv6 アドレスが空です: '{host_port}'。"));
        }
        ip_str
            .parse::<std::net::Ipv6Addr>()
            .map_err(|e| format!("無効な IPv6 アドレスです: '{ip_str}' ({e})。"))?;

        let after = &rest[close_idx + 1..];
        let port = if after.is_empty() {
            80
        } else if let Some(port_str) = after.strip_prefix(':') {
            port_str
                .parse::<u16>()
                .map_err(|e| format!("ポート番号のパースに失敗しました: '{port_str}' ({e})。"))?
        } else {
            return Err(format!(
                "無効な IPv6 ホスト・ポート形式です: '{host_port}'。"
            ));
        };
        (format!("[{ip_str}]"), port)
    } else if let Some(idx) = host_port.rfind(':') {
        let host = &host_port[..idx];
        if host.is_empty() {
            return Err(format!("ホスト名が空です: '{host_port}'。"));
        }
        let port_str = &host_port[idx + 1..];
        let port = port_str
            .parse::<u16>()
            .map_err(|e| format!("ポート番号のパースに失敗しました: '{port_str}' ({e})。"))?;
        (host.to_string(), port)
    } else {
        if host_port.is_empty() {
            return Err(format!("ホスト名が空です: '{url}'。"));
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

/// HTTP GET リクエストを送信し、ステータスコード 200 OK であるかを判定する。
///
/// # 引数
/// - `args`: ヘルスチェック対象 URL およびタイムアウト設定。
pub fn run_healthcheck(
    args: HealthcheckArgs,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let (host, port, path) = parse_http_url(&args.url)?;
    let timeout = Duration::from_secs(args.timeout_secs);
    let deadline = Instant::now() + timeout;

    let socket_addrs: Vec<_> = format!("{host}:{port}").to_socket_addrs()?.collect();
    if socket_addrs.is_empty() {
        return Err(format!("ホスト名のアドレス解決に失敗しました: {host}:{port}。").into());
    }

    let mut last_err = None;
    let mut stream = None;

    for addr in socket_addrs {
        let now = Instant::now();
        let Some(remaining) = deadline.checked_duration_since(now) else {
            return Err("接続試行がタイムアウトしました。".into());
        };
        if remaining.is_zero() {
            return Err("接続試行がタイムアウトしました。".into());
        }

        match TcpStream::connect_timeout(&addr, remaining) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(e) => {
                last_err = Some(e);
            }
        }
    }

    let mut stream = match stream {
        Some(s) => s,
        None => {
            let err_msg = last_err
                .map(|e| e.to_string())
                .unwrap_or_else(|| "接続先アドレスがありません。".to_string());
            return Err(format!("{host}:{port} への接続に失敗しました: {err_msg}。").into());
        }
    };

    let now = Instant::now();
    let remaining = deadline
        .checked_duration_since(now)
        .unwrap_or(Duration::from_millis(1));
    let io_timeout = remaining.max(Duration::from_millis(1));

    stream.set_read_timeout(Some(io_timeout))?;
    stream.set_write_timeout(Some(io_timeout))?;

    let user_agent = concat!("sokuto-healthcheck/", env!("CARGO_PKG_VERSION"));
    let request = format!(
        "GET {path} HTTP/1.1\r\n\
         Host: {host}:{port}\r\n\
         User-Agent: {user_agent}\r\n\
         Connection: close\r\n\
         \r\n"
    );

    stream.write_all(request.as_bytes())?;

    let mut response_bytes = Vec::new();
    let mut buffer = [0u8; 1024];
    loop {
        match stream.read(&mut buffer) {
            Ok(0) => break,
            Ok(n) => {
                response_bytes.extend_from_slice(&buffer[..n]);
                if response_bytes.contains(&b'\n') {
                    break;
                }
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                return Err("応答読み取りがタイムアウトしました。".into());
            }
            Err(e) => return Err(e.into()),
        }
    }

    let response_text = String::from_utf8_lossy(&response_bytes);
    let mut lines = response_text.lines();
    let status_line = lines.next().ok_or("空の HTTP レスポンスを受信しました。")?;

    // 例: "HTTP/1.1 200 OK"
    let parts: Vec<&str> = status_line.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(format!("不正な HTTP ステータス行です: '{status_line}'。").into());
    }

    let status_code = parts[1].parse::<u16>().map_err(|e| {
        format!(
            "ステータスコードのパースに失敗しました: '{}' ({e})。",
            parts[1]
        )
    })?;

    if status_code == 200 {
        tracing::debug!("ヘルスチェック成功: {status_line}。");
        Ok(())
    } else {
        Err(
            format!("ヘルスチェック失敗 (ステータスコード {status_code}): '{status_line}'。")
                .into(),
        )
    }
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

    #[test]
    fn test_run_healthcheck_ipv6() {
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

        let result = run_healthcheck(args);
        assert!(result.is_ok());

        let _ = server_handle.join();
    }

    #[test]
    fn test_run_healthcheck_immediate_return_without_close() {
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
        let result = run_healthcheck(args);
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        // 5秒のタイムアウトを待たずに即座 (500ms 未満) に脱出できたことを検証。
        assert!(elapsed < Duration::from_millis(400));

        let _ = server_handle.join();
    }
}
