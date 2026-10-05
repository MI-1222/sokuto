//! # 構造的分離アテンションマスクモジュール (Segment-Isolated Attention)
//!
//! 因果マスキングを持たない双方向エンコーダ (ModernBERT 等) において、
//! ユーザー入力末尾に含まれる悪意ある再帰的指示 (「指示を無視せよ」等) が
//! 文頭のシステム指示 (Instructions / Criteria) の注意重みを後方から汚染することを
//! 物理的に防止する半双方向アテンションマスクを提供する。
//!
//! ## アテンション結合仕様
//! - $S \to S$ (システム指示同士): 許可 ($0.0$)
//! - $S \to U$ (システム指示がユーザー入力を精査): 許可 ($0.0$)
//! - $U \to U$ (ユーザー入力同士): 許可 ($0.0$)
//! - $U \to S$ (ユーザー入力がシステム指示を変調): **物理遮断 ($-10000.0$)**

/// マスク値の定数定義。
pub const ATTN_MASK_ALLOW: f32 = 0.0;
/// 物理遮断を表すアテンションバイアス値 ($-\infty$ 相当)。
pub const ATTN_MASK_BLOCK: f32 = -10000.0;

/// セグメント種別。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentType {
    /// システム指示 (Instructions, Criteria, System Marker)。
    System,
    /// 外部ユーザー入力 (User State, Untrusted Prompt)。
    User,
    /// パディング領域。
    Padding,
}

/// 構造的分離アテンションマスク設定。
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionIsolationConfig {
    /// 構造的分離アテンションマスクを有効にするか。
    pub enabled: bool,
    /// 遮断時に加算するバイアス値 (デフォルト: -10000.0)。
    pub block_bias: f32,
}

impl Default for AttentionIsolationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            block_bias: ATTN_MASK_BLOCK,
        }
    }
}

/// 系列長 $L$ のセグメント割り当てに基づいて 2 次元のアテンションマスク $(L, L)$ を生成する。
///
/// 行 $i$ (Query) から列 $j$ (Key) へのアテンション重みを決定する。
///
/// # 引数
/// - `segments`: 各トークン位置のセグメント種別スライス。
/// - `config`: アテンション分離設定。
///
/// # 戻り値
/// サイズ $(L, L)$ のフラット化されたベクトルまたは 2 次元ベクトル。
pub fn build_segment_isolated_attention_mask(
    segments: &[SegmentType],
    config: &AttentionIsolationConfig,
) -> Vec<Vec<f32>> {
    let len = segments.len();
    let mut mask = vec![vec![ATTN_MASK_ALLOW; len]; len];

    if !config.enabled {
        // パディングのみ遮断
        for row in mask.iter_mut() {
            for (j, seg) in segments.iter().enumerate() {
                if *seg == SegmentType::Padding {
                    row[j] = config.block_bias;
                }
            }
        }
        return mask;
    }

    for (i, &query_seg) in segments.iter().enumerate() {
        for (j, &key_seg) in segments.iter().enumerate() {
            if key_seg == SegmentType::Padding {
                // パディングへの注意は常に遮断
                mask[i][j] = config.block_bias;
            } else if query_seg == SegmentType::User && key_seg == SegmentType::System {
                // ユーザー入力からシステム指示へのアテンションを物理遮断 (U -> S)
                mask[i][j] = config.block_bias;
            } else {
                // それ以外はすべて相互作用を許可 (S -> S, S -> U, U -> U)
                mask[i][j] = ATTN_MASK_ALLOW;
            }
        }
    }

    mask
}

/// システム指示長とユーザー入力長からフラット化された 1 次元アテンションマスク配列 $(L \times L)$ を生成する。
///
/// テンソル入力 (ONNX Runtime 等) への直接変換に最適化されている。
///
/// # 引数
/// - `sys_len`: システム指示トークン数。
/// - `user_len`: ユーザー入力トークン数。
/// - `pad_len`: パディングトークン数。
/// - `config`: アテンション分離設定。
pub fn build_flat_attention_mask(
    sys_len: usize,
    user_len: usize,
    pad_len: usize,
    config: &AttentionIsolationConfig,
) -> Vec<f32> {
    let total_len = sys_len + user_len + pad_len;
    let mut flat = vec![ATTN_MASK_ALLOW; total_len * total_len];

    let block_val = if config.enabled {
        config.block_bias
    } else {
        ATTN_MASK_ALLOW
    };

    for i in 0..total_len {
        let is_query_user = i >= sys_len && i < sys_len + user_len;

        for j in 0..total_len {
            let idx = i * total_len + j;
            let is_key_system = j < sys_len;
            let is_key_pad = j >= sys_len + user_len;

            if is_key_pad {
                flat[idx] = config.block_bias;
            } else if config.enabled && is_query_user && is_key_system {
                // User -> System 遮断
                flat[idx] = block_val;
            } else {
                flat[idx] = ATTN_MASK_ALLOW;
            }
        }
    }

    flat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_isolated_attention_mask_behavior() {
        let config = AttentionIsolationConfig::default();
        let segments = vec![
            SegmentType::System,  // 0: Instruction
            SegmentType::System,  // 1: Criteria
            SegmentType::User,    // 2: User Input
            SegmentType::User,    // 3: User Input
            SegmentType::Padding, // 4: Pad
        ];

        let mask = build_segment_isolated_attention_mask(&segments, &config);

        // S -> S (0 -> 1): 許可
        assert_eq!(mask[0][1], ATTN_MASK_ALLOW);
        // S -> U (0 -> 2): 許可 (システムがユーザー入力を精査)
        assert_eq!(mask[0][2], ATTN_MASK_ALLOW);
        // U -> U (2 -> 3): 許可
        assert_eq!(mask[2][3], ATTN_MASK_ALLOW);

        // U -> S (2 -> 0, 3 -> 1): 物理遮断 (ユーザーがシステム指示を変調不可)
        assert_eq!(mask[2][0], ATTN_MASK_BLOCK);
        assert_eq!(mask[2][1], ATTN_MASK_BLOCK);
        assert_eq!(mask[3][0], ATTN_MASK_BLOCK);
        assert_eq!(mask[3][1], ATTN_MASK_BLOCK);

        // 任意 -> Padding: 物理遮断
        assert_eq!(mask[0][4], ATTN_MASK_BLOCK);
        assert_eq!(mask[2][4], ATTN_MASK_BLOCK);
    }

    #[test]
    fn test_flat_attention_mask_parity() {
        let config = AttentionIsolationConfig::default();
        let sys_len = 2;
        let user_len = 2;
        let pad_len = 1;
        let total = sys_len + user_len + pad_len;

        let flat = build_flat_attention_mask(sys_len, user_len, pad_len, &config);
        assert_eq!(flat.len(), total * total);

        // User (i=2) -> System (j=0) は遮断
        assert_eq!(flat[2 * total], ATTN_MASK_BLOCK);
        // System (i=0) -> User (j=2) は許可
        assert_eq!(flat[2], ATTN_MASK_ALLOW);
        // Key is Padding (j=4) は遮断
        assert_eq!(flat[4], ATTN_MASK_BLOCK);
    }
}
