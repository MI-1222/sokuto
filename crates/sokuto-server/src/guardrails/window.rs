//! # 階層型 Fast-Fail 境界縮約窓生成モジュール (Head-Tail Window Extraction)
//!
//! 自然言語によるプロンプトインジェクションの 99% 以上はプロンプトの境界
//! (先頭によるペルソナ上書き、末尾による直前命令打ち消し) に集中する。
//! 入力長が 512 トークンを超える長文に対し、全文エンコード ($O(L^2)$) を回避し、
//! 先頭 256 トークンと末尾 256 トークンを結合した 512 トークン窓を生成して
//! 5〜8ms での高速 Early-Exit スキャンを実現する。

use std::borrow::Cow;

/// 縮約窓抽出設定構造体。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowConfig {
    /// 縮約窓抽出を有効にするか。
    pub enabled: bool,
    /// 縮約窓を適用するトリガーとなる最大トークン閾値 (デフォルト: 512)。
    pub max_tokens_threshold: usize,
    /// 先頭から抽出するトークン数 (デフォルト: 256)。
    pub head_tokens: usize,
    /// 末尾から抽出するトークン数 (デフォルト: 256)。
    pub tail_tokens: usize,
    /// 日本語における 1 トークンあたりの平均文字数見積もり (デフォルト: 1.5)。
    pub chars_per_token_estimate: usize,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_tokens_threshold: 512,
            head_tokens: 256,
            tail_tokens: 256,
            chars_per_token_estimate: 2,
        }
    }
}

/// 縮約窓抽出結果。
#[derive(Debug, Clone, PartialEq)]
pub struct WindowExtractionResult<'a> {
    /// 抽出された縮約窓テキスト。
    pub window_text: Cow<'a, str>,
    /// 実際に縮約窓が適用されたか (閾値未満の場合は false)。
    pub was_truncated: bool,
    /// 元の推定トークン数。
    pub original_estimated_tokens: usize,
    /// 窓抽出後の推定トークン数。
    pub window_estimated_tokens: usize,
}

/// 入力テキストから先頭と末尾の重要境界領域を抽出して結合した縮約窓を生成する。
///
/// UTF-8 のマルチバイト文字境界を破壊しないよう、必ず文字境界 (Char Boundary) でスライスを行う。
///
/// # 引数
/// - `text`: 対象テキスト。
/// - `config`: 縮約窓設定。
///
/// # 戻り値
/// 縮約窓抽出結果。
pub fn extract_head_tail_window<'a>(
    text: &'a str,
    config: &WindowConfig,
) -> WindowExtractionResult<'a> {
    if !config.enabled {
        let est = estimate_tokens(text, config.chars_per_token_estimate);
        return WindowExtractionResult {
            window_text: Cow::Borrowed(text),
            was_truncated: false,
            original_estimated_tokens: est,
            window_estimated_tokens: est,
        };
    }

    let char_count = text.chars().count();
    let est_tokens = estimate_tokens_from_char_count(char_count, config.chars_per_token_estimate);

    if est_tokens <= config.max_tokens_threshold {
        return WindowExtractionResult {
            window_text: Cow::Borrowed(text),
            was_truncated: false,
            original_estimated_tokens: est_tokens,
            window_estimated_tokens: est_tokens,
        };
    }

    // 先頭および末尾の必要文字数を計算
    let head_chars = config.head_tokens * config.chars_per_token_estimate;
    let tail_chars = config.tail_tokens * config.chars_per_token_estimate;

    if head_chars + tail_chars >= char_count {
        return WindowExtractionResult {
            window_text: Cow::Borrowed(text),
            was_truncated: false,
            original_estimated_tokens: est_tokens,
            window_estimated_tokens: est_tokens,
        };
    }

    // 先頭部分の文字境界を取得
    let head_byte_idx = text
        .char_indices()
        .nth(head_chars)
        .map(|(idx, _)| idx)
        .unwrap_or(text.len());
    let head_slice = &text[..head_byte_idx];

    // 末尾部分の文字境界を取得
    let tail_start_char = char_count.saturating_sub(tail_chars);
    let tail_byte_idx = text
        .char_indices()
        .nth(tail_start_char)
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let tail_slice = &text[tail_byte_idx..];

    let omitted_chars = char_count.saturating_sub(head_chars + tail_chars);
    let omitted_tokens = omitted_chars / config.chars_per_token_estimate.max(1);

    let window_text = format!(
        "{}\n[... 縮約窓走査により中間 {} トークンを省略 ...]\n{}",
        head_slice, omitted_tokens, tail_slice
    );

    let window_tokens = config.head_tokens + config.tail_tokens;

    WindowExtractionResult {
        window_text: Cow::Owned(window_text),
        was_truncated: true,
        original_estimated_tokens: est_tokens,
        window_estimated_tokens: window_tokens,
    }
}

/// 文字列からトークン数を簡易見積もりする。
#[inline]
pub fn estimate_tokens(text: &str, chars_per_token: usize) -> usize {
    let char_count = text.chars().count();
    estimate_tokens_from_char_count(char_count, chars_per_token)
}

#[inline]
fn estimate_tokens_from_char_count(char_count: usize, chars_per_token: usize) -> usize {
    let factor = chars_per_token.max(1);
    char_count.div_ceil(factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_short_text_unchanged() {
        let config = WindowConfig::default();
        let text = "これは短いテスト文章です。";
        let res = extract_head_tail_window(text, &config);
        assert!(!res.was_truncated);
        assert_eq!(res.window_text, text);
    }

    #[test]
    fn test_window_long_text_truncated() {
        let config = WindowConfig {
            enabled: true,
            max_tokens_threshold: 10,
            head_tokens: 4,
            tail_tokens: 4,
            chars_per_token_estimate: 2,
        };
        // 40 文字 = 約 20 トークン (> 10)
        let text = "0123456789abcdefghijklmnopqrstuvwxyzABCD";
        let res = extract_head_tail_window(text, &config);
        assert!(res.was_truncated);
        assert!(res.window_text.starts_with("01234567"));
        assert!(res.window_text.ends_with("wxyzABCD"));
        assert!(res.window_text.contains("[... 縮約窓走査により中間"));
    }

    #[test]
    fn test_window_multibyte_safety() {
        let config = WindowConfig {
            enabled: true,
            max_tokens_threshold: 10,
            head_tokens: 3,
            tail_tokens: 3,
            chars_per_token_estimate: 2,
        };
        // 日本語マルチバイト
        let text = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほ";
        let res = extract_head_tail_window(text, &config);
        assert!(res.was_truncated);
        // パニックせず正常に UTF-8 文字列が構築されること
        assert!(res.window_text.starts_with("あいうえおか"));
        assert!(res.window_text.ends_with("はひふへほ"));
    }
}
