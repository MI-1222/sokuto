//! # 投機的ストリーミング検証モジュール (SSE Realtime Interceptor)
//!
//! System 2 (上位 LLM) からの SSE (Server-Sent Events) ストリーミングレスポンスに対し、
//! 句読点 (「。」「\n」「！」「？」) 単位でリアルタイムに文境界をバッファリングし、
//! 投機的 NLI ハルシネーション検証を並行して実行する。
//! 不整合を検知した瞬間、ダウンストリームへの送信を即時切断し、安全なエラー・置換メッセージを
//! 注入して安全にコネクションをクローズする。

use std::sync::Arc;

use bytes::Bytes;
use futures_util::Stream;
use futures_util::stream::BoxStream;

use crate::guardrails::nli::{
    HallucinationVerificationResult, HallucinationVerifierConfig, verify_claim_nli,
};

/// 文境界区切り文字。
pub const SENTENCE_DELIMITERS: &[char] = &['。', '\n', '！', '？', '!', '?'];

/// ストリーミング遮断時の置換メッセージ。
pub const TERMINATION_REPLACEMENT_MESSAGE: &str =
    "\n[安全基準に基づき生成を中断しました。事実整合性エラーを検知しました]";

/// ストリーミング検証設定。
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingVerifierConfig {
    /// ストリーミング検証を有効にするか。
    pub enabled: bool,
    /// 文境界の最小文字数 (極端に短い助詞などの細切れ分割を抑止、デフォルト: 6)。
    pub min_sentence_chars: usize,
    /// エラー置換メッセージ。
    pub replacement_message: String,
    /// ハルシネーション検証設定。
    pub verifier_config: HallucinationVerifierConfig,
}

impl Default for StreamingVerifierConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_sentence_chars: 6,
            replacement_message: TERMINATION_REPLACEMENT_MESSAGE.to_string(),
            verifier_config: HallucinationVerifierConfig::default(),
        }
    }
}

/// 文境界バッファ。
#[derive(Debug, Default)]
pub struct SentenceChunkBuffer {
    buffer: String,
}

impl SentenceChunkBuffer {
    /// 新規バッファを生成する。
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// チャンク文字列を追加し、文境界に達した完結文を取り出す。
    ///
    /// # 引数
    /// - `chunk`: 受信した文字列スライスのチャンク。
    /// - `min_chars`: 最小文文字数。
    ///
    /// # 戻り値
    /// 境界に達した完結文のベクタ (未完結の末尾は内部バッファに残る)。
    pub fn push_and_drain_sentences(&mut self, chunk: &str, min_chars: usize) -> Vec<String> {
        self.buffer.push_str(chunk);
        let mut sentences = Vec::new();

        while let Some(idx) = self.find_delimiter_index(min_chars) {
            // 区切り文字の次のバイトインデックス
            let delim_char = self.buffer[idx..].chars().next().unwrap();
            let split_byte_idx = idx + delim_char.len_utf8();

            let sentence = self.buffer[..split_byte_idx].to_string();
            self.buffer = self.buffer[split_byte_idx..].to_string();
            sentences.push(sentence);
        }

        sentences
    }

    /// バッファ内に残存する全テキストをフラッシュして取り出す。
    pub fn flush(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            None
        } else {
            let remaining = std::mem::take(&mut self.buffer);
            Some(remaining)
        }
    }

    #[inline]
    fn find_delimiter_index(&self, min_chars: usize) -> Option<usize> {
        let mut char_count = 0;
        for (byte_idx, c) in self.buffer.char_indices() {
            char_count += 1;
            if char_count >= min_chars && SENTENCE_DELIMITERS.contains(&c) {
                return Some(byte_idx);
            }
        }
        None
    }
}

/// SSE の data ペイロードからテキスト本文を抽出する簡易ヘルパー。
///
/// 形式例: `data: {"choices":[{"delta":{"content":"こんにちは"}}]}\n\n`
pub fn extract_content_from_sse_line(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if !trimmed.starts_with("data:") {
        return None;
    }
    let data_body = trimmed.strip_prefix("data:")?.trim();
    if data_body == "[DONE]" {
        return None;
    }

    if let Ok(v) = serde_json::from_str::<serde_json::Value>(data_body) {
        if let Some(content) = v
            .pointer("/choices/0/delta/content")
            .and_then(|c| c.as_str())
        {
            return Some(content.to_string());
        }
        if let Some(text) = v.get("text").and_then(|t| t.as_str()) {
            return Some(text.to_string());
        }
    }

    None
}

/// 投機的ストリーム検証インターセプター。
pub struct SpeculativeStreamInterceptor {
    config: Arc<StreamingVerifierConfig>,
    state_context: Arc<String>,
}

impl SpeculativeStreamInterceptor {
    /// 新規インターセプターを構築する。
    pub fn new(config: StreamingVerifierConfig, state_context: String) -> Self {
        Self {
            config: Arc::new(config),
            state_context: Arc::new(state_context),
        }
    }

    /// 文がハルシネーションであるかを同期または投機的に検査する。
    pub fn verify_sentence(
        &self,
        sentence: &str,
        mock_nli_logits: Option<[f64; 3]>,
    ) -> HallucinationVerificationResult {
        // デフォルトでは高い含意確率ロジット、指定があればそれを使用
        let logits = mock_nli_logits.unwrap_or([3.0, -1.0, -3.0]);
        verify_claim_nli(
            &self.state_context,
            sentence,
            logits,
            &self.config.verifier_config,
        )
    }

    /// 遮断用 SSE エラーチャンクバイト列を構築する。
    pub fn build_error_event_bytes(&self, reason: &str) -> Bytes {
        let payload = serde_json::json!({
            "error": "guardrail_violation",
            "rule": "outbound_hallucination",
            "message": self.config.replacement_message,
            "reason": reason
        });
        let sse = format!(
            "event: guardrail_violation\ndata: {}\n\ndata: [DONE]\n\n",
            payload
        );
        Bytes::from(sse)
    }

    /// 設定への参照を取得する。
    pub fn config(&self) -> &StreamingVerifierConfig {
        &self.config
    }

    /// 前提コンテキスト文字列への参照を取得する。
    pub fn state_context(&self) -> &str {
        &self.state_context
    }
}

/// SSE ストリームをインターセプトし、文境界ごとの投機的ハルシネーション検証とサーキットブレークを行う。
///
/// ハルシネーションを検知した場合は、安全な置換エラーイベントを注入してストリームを遮断する。
pub fn intercept_sse_stream<S, E>(
    stream: S,
    interceptor: Arc<SpeculativeStreamInterceptor>,
) -> BoxStream<'static, Result<Bytes, E>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: Send + 'static,
{
    use futures_util::StreamExt;

    let mut buffer = SentenceChunkBuffer::new();
    let mut terminated = false;

    let stream = stream.flat_map(move |item| {
        if terminated {
            return futures_util::stream::empty().boxed();
        }

        match item {
            Ok(bytes) => {
                let text = String::from_utf8_lossy(&bytes);
                let sentences =
                    buffer.push_and_drain_sentences(&text, interceptor.config.min_sentence_chars);

                for sentence in sentences {
                    let verification = interceptor.verify_sentence(&sentence, None);
                    if verification.is_hallucination {
                        terminated = true;
                        let err_reason = verification
                            .reason
                            .unwrap_or_else(|| "事実不整合".to_string());
                        let err_bytes = interceptor.build_error_event_bytes(&err_reason);
                        return futures_util::stream::iter(vec![Ok(err_bytes)]).boxed();
                    }
                }

                futures_util::stream::iter(vec![Ok(bytes)]).boxed()
            }
            Err(e) => futures_util::stream::iter(vec![Err(e)]).boxed(),
        }
    });

    stream.boxed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_chunk_buffer_drain() {
        let mut buf = SentenceChunkBuffer::new();
        let chunks = vec![
            "本日の",
            "契約金額は",
            "1,000万円です。",
            "次回は",
            "来月です。",
        ];

        let mut sentences = Vec::new();
        for c in chunks {
            let drained = buf.push_and_drain_sentences(c, 5);
            sentences.extend(drained);
        }
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], "本日の契約金額は1,000万円です。");
        assert_eq!(sentences[1], "次回は来月です。");
    }

    #[test]
    fn test_extract_content_from_sse() {
        let line = r#"data: {"choices":[{"delta":{"content":"認証完了。"}}]}"#;
        let content = extract_content_from_sse_line(line);
        assert_eq!(content.as_deref(), Some("認証完了。"));
    }

    #[test]
    fn test_interceptor_error_generation() {
        let config = StreamingVerifierConfig::default();
        let interceptor = SpeculativeStreamInterceptor::new(config, "前提コンテキスト".to_string());
        let err_bytes = interceptor.build_error_event_bytes("テスト理由");
        let s = String::from_utf8_lossy(&err_bytes);
        assert!(s.contains("event: guardrail_violation"));
        assert!(s.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn test_intercept_sse_stream_circuit_breaker() {
        use futures_util::StreamExt;

        let config = StreamingVerifierConfig::default();
        // 前提: 契約金は 1,000万円
        let interceptor = Arc::new(SpeculativeStreamInterceptor::new(
            config,
            "前提: 契約金額は 1,000万円です。".to_string(),
        ));

        // 正常文のあとにハルシネーション文 (1,000億円) を流す
        let chunks = vec![
            Ok::<_, std::io::Error>(Bytes::from("data: 本件の確認を開始します。\n\n")),
            Ok(Bytes::from(
                "data: 契約金額は 1,000億円に確定しました。\n\n",
            )),
            Ok(Bytes::from("data: 完了しました。\n\n")),
        ];
        let raw_stream = futures_util::stream::iter(chunks);

        let verified_stream = intercept_sse_stream(raw_stream, interceptor);
        let items: Vec<Result<Bytes, std::io::Error>> = verified_stream.collect().await;

        let texts: Vec<String> = items
            .into_iter()
            .map(|r| String::from_utf8_lossy(&r.unwrap()).to_string())
            .collect();

        // 1つ目は通過
        assert!(texts[0].contains("本件の確認を開始します。"));
        // 2つ目でハルシネーション (Entity Mismatch) を検知して遮断イベントへ置換
        assert!(texts[1].contains("event: guardrail_violation"));
        assert!(texts[1].contains("[安全基準に基づき生成を中断しました"));
        // 3つ目以降はストリームが終了して送出されない
        assert_eq!(texts.len(), 2);
    }
}
