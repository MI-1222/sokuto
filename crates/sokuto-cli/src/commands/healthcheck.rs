//! # ヘルスチェック実行コマンド
//!
//! 最小コンテナ環境など curl が存在しない環境において、
//! 外部依存なしで HTTP エンドポイントの死活監視・準備状態確認 (Liveness / Readiness) を行う。

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::commands::net::resolve_host_port_with_deadline;
use crate::error::CliError;

/// 複数アドレスへの接続試行におけるフォールバック開始遅延時間 (RFC 8305 Happy Eyeballs 準拠)。
const CONNECTION_ATTEMPT_DELAY: Duration = Duration::from_millis(250);

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

    let (host_port, path) = match stripped.find(&['/', '?'][..]) {
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
    } else if path.starts_with('?') {
        format!("/{path}")
    } else {
        path.to_string()
    };

    Ok((host, port, path))
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

    let socket_addrs = resolve_host_port_with_deadline(&host, port, Some(deadline))
        .await
        .map_err(|err| CliError::HealthcheckFailed(err.to_string()))?;
    if socket_addrs.is_empty() {
        return Err(CliError::HealthcheckFailed(format!(
            "ホスト名のアドレス解決に失敗しました: {host}:{port}。"
        )));
    }

    let mut stream = connect_to_any(&socket_addrs, deadline, &host, port).await?;

    let remaining = check_remaining(deadline, "リクエスト送信がタイムアウトしました。")?;

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

    let mut reader = tokio::io::BufReader::new(stream);
    let mut status_line = String::new();
    const MAX_STATUS_LINE_LEN: usize = 8192;

    loop {
        status_line.clear();
        let remaining = check_remaining(deadline, "応答読み取りがタイムアウトしました。")?;

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

/// デッドラインまでの残余時間を算出し、期限切れの場合はエラーを返す。
fn check_remaining(deadline: Instant, timeout_msg: &'static str) -> Result<Duration, CliError> {
    let remaining = deadline
        .checked_duration_since(Instant::now())
        .unwrap_or_default();
    if remaining.is_zero() {
        Err(CliError::HealthcheckFailed(timeout_msg.into()))
    } else {
        Ok(remaining)
    }
}

/// 次の候補アドレスへの非同期接続タスクを生成し、後続候補が存在するかどうかを返す。
///
/// 後続の候補アドレスが存在する場合は、
/// 指定されたタイマーを接続試行遅延 (`CONNECTION_ATTEMPT_DELAY`) でリセットする。
fn spawn_next_candidate<I>(
    addrs: &mut std::iter::Peekable<I>,
    join_set: &mut tokio::task::JoinSet<(SocketAddr, Result<TcpStream, std::io::Error>)>,
    mut sleep: std::pin::Pin<&mut tokio::time::Sleep>,
) -> bool
where
    I: Iterator<Item = SocketAddr>,
{
    if let Some(next_addr) = addrs.next() {
        join_set.spawn(async move { (next_addr, TcpStream::connect(next_addr).await) });
        let has_more = addrs.peek().is_some();
        if has_more {
            sleep
                .as_mut()
                .reset(tokio::time::Instant::now() + CONNECTION_ATTEMPT_DELAY);
        }
        has_more
    } else {
        false
    }
}

/// 解決されたソケットアドレス一覧へ接続を試行し、最初に確立された TCP ストリームを返す。
///
/// 複数アドレスが存在する場合、RFC 8305 (Happy Eyeballs v2) に基づき、
/// 先頭アドレスへの接続が一定時間内に完了しない場合に後続アドレスへの並行接続を開始する。
/// これにより、IPv6 がルーティング不能な環境等において、タイムアウトまで待機することなく
/// 正常なアドレス (IPv4 等) への迅速なフォールバックを実現する。
async fn connect_to_any(
    socket_addrs: &[SocketAddr],
    deadline: Instant,
    host: &str,
    port: u16,
) -> Result<TcpStream, CliError> {
    if socket_addrs.is_empty() {
        return Err(CliError::HealthcheckFailed(format!(
            "{host}:{port} への接続に失敗しました: 接続先アドレスがありません。"
        )));
    }

    // 単一アドレスの場合はフォールバック不要のため直接接続する。
    if socket_addrs.len() == 1 {
        let addr = socket_addrs[0];
        let remaining = check_remaining(deadline, "接続試行がタイムアウトしました。")?;
        return match tokio::time::timeout(remaining, TcpStream::connect(&addr)).await {
            Ok(Ok(stream)) => Ok(stream),
            Ok(Err(err)) => Err(CliError::HealthcheckFailed(format!(
                "{host}:{port} への接続に失敗しました: {err}。"
            ))),
            Err(_) => Err(CliError::HealthcheckFailed(
                "接続試行がタイムアウトしました。".into(),
            )),
        };
    }

    // 複数アドレスに対する Happy Eyeballs 接続制御。
    let mut addrs = socket_addrs.iter().copied().peekable();
    let mut join_set = tokio::task::JoinSet::new();
    let mut last_err = None;

    let first_addr = addrs.next().unwrap();
    join_set.spawn(async move { (first_addr, TcpStream::connect(first_addr).await) });

    let sleep = tokio::time::sleep(CONNECTION_ATTEMPT_DELAY);
    tokio::pin!(sleep);

    let deadline_sleep = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline));
    tokio::pin!(deadline_sleep);

    let mut has_more = addrs.peek().is_some();

    loop {
        tokio::select! {
            _ = &mut deadline_sleep => {
                return Err(CliError::HealthcheckFailed(
                    "接続試行がタイムアウトしました。".into(),
                ));
            }
            () = &mut sleep, if has_more => {
                has_more = spawn_next_candidate(&mut addrs, &mut join_set, sleep.as_mut());
            }
            res = join_set.join_next(), if !join_set.is_empty() => {
                match res {
                    Some(Ok((_addr, Ok(stream)))) => {
                        return Ok(stream);
                    }
                    Some(Ok((addr, Err(err)))) => {
                        last_err = Some(format!("{addr}: {err}"));
                        if has_more {
                            has_more =
                                spawn_next_candidate(&mut addrs, &mut join_set, sleep.as_mut());
                        }
                    }
                    Some(Err(join_err)) => {
                        last_err = Some(join_err.to_string());
                    }
                    None => {}
                }
            }
        }

        if join_set.is_empty() && !has_more {
            let err_msg = last_err.unwrap_or_else(|| "接続先アドレスがありません。".to_string());
            return Err(CliError::HealthcheckFailed(format!(
                "{host}:{port} への接続に失敗しました: {err_msg}。"
            )));
        }
    }
}

/// HTTP ステータス行を検証し、ステータスコードを抽出する。
fn parse_http_status_line(status_line: &str) -> Result<u16, CliError> {
    let trimmed = status_line.trim_matches(|c| c == '\r' || c == '\n');
    let mut parts = trimmed.split_whitespace();

    match (parts.next(), parts.next()) {
        (Some(proto), Some(code_str)) if proto.starts_with("HTTP/") => {
            code_str.parse::<u16>().map_err(|e| {
                CliError::HealthcheckFailed(format!(
                    "ステータスコードのパースに失敗しました: '{code_str}' ({e})。"
                ))
            })
        }
        _ => Err(CliError::HealthcheckFailed(format!(
            "不正な HTTP ステータス行です: '{trimmed}'。"
        ))),
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

        // クエリパラメータ付き URL (パス省略およびパス指定あり)
        let (host, port, path) = parse_http_url("http://localhost:3000?ready=1").unwrap();
        assert_eq!(host, "localhost");
        assert_eq!(port, 3000);
        assert_eq!(path, "/?ready=1");

        let (host, port, path) = parse_http_url("http://localhost?ready=1").unwrap();
        assert_eq!(host, "localhost");
        assert_eq!(port, 80);
        assert_eq!(path, "/?ready=1");

        let (host, port, path) = parse_http_url("http://[::1]:3000?ready=1").unwrap();
        assert_eq!(host, "[::1]");
        assert_eq!(port, 3000);
        assert_eq!(path, "/?ready=1");

        let (host, port, path) = parse_http_url("http://localhost:3000/?ready=1").unwrap();
        assert_eq!(host, "localhost");
        assert_eq!(port, 3000);
        assert_eq!(path, "/?ready=1");

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

    #[test]
    fn test_check_remaining() {
        let future = Instant::now() + Duration::from_secs(10);
        let res = check_remaining(future, "タイムアウト。");
        assert!(res.is_ok());
        assert!(res.unwrap() > Duration::from_secs(8));

        let past = Instant::now() - Duration::from_secs(1);
        let res = check_remaining(past, "タイムアウト。");
        assert!(res.is_err());
        let err = res.unwrap_err();
        assert!(err.to_string().contains("タイムアウト。"));
    }

    #[tokio::test]
    async fn test_connect_to_any_single_addr() {
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let deadline = Instant::now() + Duration::from_secs(2);
        let res = connect_to_any(&[addr], deadline, "127.0.0.1", addr.port()).await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_connect_to_any_fallback_on_immediate_error() {
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let ok_addr = listener.local_addr().unwrap();

        // 接続不可なポートを取得 (バインド直後にクローズ)。
        let unused_listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let closed_addr = unused_listener.local_addr().unwrap();
        drop(unused_listener);

        let deadline = Instant::now() + Duration::from_secs(2);
        // 先頭が接続不可アドレスで、2つ目が正常アドレスのケース。
        let res = connect_to_any(
            &[closed_addr, ok_addr],
            deadline,
            "127.0.0.1",
            ok_addr.port(),
        )
        .await;
        assert!(res.is_ok());
    }

    #[tokio::test]
    async fn test_connect_to_any_all_failed() {
        use std::net::TcpListener;

        let l1 = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr1 = l1.local_addr().unwrap();
        drop(l1);

        let l2 = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr2 = l2.local_addr().unwrap();
        drop(l2);

        let deadline = Instant::now() + Duration::from_secs(1);
        let res = connect_to_any(&[addr1, addr2], deadline, "127.0.0.1", addr1.port()).await;
        assert!(res.is_err());
        let err_msg = res.unwrap_err().to_string();
        assert!(err_msg.contains("への接続に失敗しました:"));
    }

    #[tokio::test]
    async fn test_connect_to_any_past_deadline() {
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let past_deadline = Instant::now() - Duration::from_millis(10);
        let res = connect_to_any(&[addr], past_deadline, "127.0.0.1", addr.port()).await;
        assert!(res.is_err());
        assert!(
            res.unwrap_err()
                .to_string()
                .contains("接続試行がタイムアウトしました。")
        );
    }

    #[tokio::test]
    async fn test_connect_to_any_empty_addrs() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let res = connect_to_any(&[], deadline, "localhost", 80).await;
        assert!(res.is_err());
        assert!(
            res.unwrap_err()
                .to_string()
                .contains("接続先アドレスがありません。")
        );
    }
}
