//! # 中間特徴量キャッシュおよびゼロコピー連携モジュール
//!
//! 長文 `State` の中間特徴量キャッシュ、
//! `xxHash3` 高速キー生成、`moka` 並行キャッシュストレージ、
//! `ort` の `IoBinding` によるゼロコピーバインド、
//! L3キャッシュ局所性最適化（`FP16`/`INT8` 圧縮）、
//! 64バイトメモリアライメント、および `Tokio` ワーカースレッド競合隔離を提供する。

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::borrow::{Borrow, BorrowMut};
use std::fmt;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use half::f16;
use moka::sync::Cache;
use ort::memory::{Allocator, MemoryInfo};
use ort::session::Session;
use ort::value::{Tensor, Value};
use sokuto_core::contract::calibration::CalibrationConfig;
use sokuto_core::contract::model_spec::{
    ModelInputDimensions, TENSOR_ATTENTION_MASK, TENSOR_INPUT_IDS, TENSOR_LOGITS,
    TENSOR_OP_INDICES, TENSOR_QUERY_IDS, TENSOR_QUERY_MASK, TENSOR_STATE_HIDDEN_STATES,
    TENSOR_STATE_MASK,
};
use sokuto_core::gating::GatingConfig;
use sokuto_core::schema::{Answer, Question};
use unicode_normalization::{IsNormalized, UnicodeNormalization, is_nfc_quick};
use xxhash_rust::xxh3::xxh3_64;

use crate::engine::config::{ExecutionProvider, SessionConfig};
use crate::engine::gating::{evaluate_and_gate_answer, evaluate_and_gate_answer_with_buf};
use crate::engine::session::{acquire_pooled_session, build_session_pool};
use crate::error::{Result, RuntimeError};
use crate::tokenizer::{JevTokenizer, TokenizedQuestion};

/// 64バイト境界メモリアライメント判定関数。
///
/// `AVX-512` や `AVX2` のベクトル化命令において、未アライメントアクセスによる
/// セグメンテーション違反や性能劣化を防ぐためにアライメント状態を検証する。
pub fn is_aligned_64<T>(ptr: *const T) -> bool {
    (ptr as usize).is_multiple_of(64)
}

/// デュアルセッション推論における推奨デフォルト最大チャンクサイズ。
///
/// 長文 `State` 特徴量のリピート展開による一時的なヒープメモリ確保スパイク(OOM Killer リスク) を抑制するため、
/// 4 サンプル程度を推奨上限とする。
pub const DEFAULT_DUAL_MAX_BATCH_CHUNK_SIZE: usize = 4;

/// 64バイト境界にアライメントされた連続ヒープメモリバッファ。
///
/// `SIMD` カーネルの高速演算要件（64バイトアライメント）を満たす安全な連続バッファを提供する。
pub struct AlignedBuffer64<T: Copy> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    layout: Layout,
}

impl<T: Copy> AlignedBuffer64<T> {
    /// 指定された要素数で64バイトアライメントされたゼロ初期化バッファを生成する。
    ///
    /// # 引数
    /// - `len`: 要素数。
    pub fn with_capacity_zeroed(len: usize) -> Self {
        assert!(
            std::mem::size_of::<T>() > 0,
            "ZST はサポートされていません。"
        );

        let align = std::mem::align_of::<T>().max(64);
        let size = len
            .checked_mul(std::mem::size_of::<T>())
            .expect("メモリ確保サイズがオーバーフローしました。");
        let layout = Layout::from_size_align(size, align)
            .expect("64バイトアライメントレイアウトの作成に失敗しました。");

        if len == 0 {
            let empty_align =
                NonZeroUsize::new(align).expect("アライメント値は非ゼロである必要があります。");
            return Self {
                ptr: NonNull::without_provenance(empty_align),
                len: 0,
                capacity: 0,
                layout,
            };
        }

        // SAFETY: 有効なレイアウトと非ゼロサイズを検証済みであり、直後にゼロ初期化メモリとして確保する。
        let raw_ptr = unsafe { alloc_zeroed(layout) as *mut T };
        let ptr = NonNull::new(raw_ptr).expect("アライメントメモリの割り当てに失敗しました。");

        Self {
            ptr,
            len,
            capacity: len,
            layout,
        }
    }

    /// スライスから64バイトアライメントされたバッファを複製生成する。
    ///
    /// # 引数
    /// - `slice`: 複製元スライス。
    pub fn from_slice(slice: &[T]) -> Self {
        let mut buf = Self::with_capacity_zeroed(slice.len());
        buf.copy_from_slice(slice);
        buf
    }

    /// バッファの要素数を取得する。
    pub fn len(&self) -> usize {
        self.len
    }

    /// バッファが空であるかを判定する。
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 内部生ポインタが64バイト境界に整列しているかを検証する。
    pub fn is_aligned_64(&self) -> bool {
        is_aligned_64(self.ptr.as_ptr())
    }
}

impl<T: Copy> Deref for AlignedBuffer64<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        // SAFETY: `len == 0` 時も `ptr` は非ヌルかつ `align_of::<T>()` 以上 (64バイト境界) に整列しているため、
        // スライス生成は常に安全である。
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> DerefMut for AlignedBuffer64<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: 排他的借用下で、`len == 0` 時も `ptr` は非ヌルかつ `align_of::<T>()` 以上に整列している。
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy> AsRef<[T]> for AlignedBuffer64<T> {
    fn as_ref(&self) -> &[T] {
        self.deref()
    }
}

impl<T: Copy> AsMut<[T]> for AlignedBuffer64<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.deref_mut()
    }
}

impl<T: Copy> Borrow<[T]> for AlignedBuffer64<T> {
    fn borrow(&self) -> &[T] {
        self.deref()
    }
}

impl<T: Copy> BorrowMut<[T]> for AlignedBuffer64<T> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.deref_mut()
    }
}

impl<T: Copy> Drop for AlignedBuffer64<T> {
    fn drop(&mut self) {
        if self.capacity > 0 && self.layout.size() > 0 {
            // SAFETY: 構築時に確保した同一のポインタとレイアウトで解放する。
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
            }
        }
    }
}

// SAFETY: 保持する型 `T` が `Send` であれば、スレッド間で安全に所有権を移動可能である。
unsafe impl<T: Copy + Send> Send for AlignedBuffer64<T> {}
// SAFETY: 保持する型 `T` が `Sync` であれば、スレッド間で安全に参照を共有可能である。
unsafe impl<T: Copy + Sync> Sync for AlignedBuffer64<T> {}

impl<T: Copy + fmt::Debug> fmt::Debug for AlignedBuffer64<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.deref().iter()).finish()
    }
}

/// 中間特徴量キャッシュのメモリ圧縮モード。
///
/// 8kトークンの隠れ状態（`FP32` で25.16MB）によるL3キャッシュ破壊を防ぎ、
/// メモリバス帯域の飽和を抑制するための圧縮方式を指定する。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CacheCompressionMode {
    /// 圧縮なし（`FP32` の `Value` を直接キャッシュ保持し、ゼロコピーで直接バインド）。
    #[default]
    None,
    /// `FP16` (`half::f16`) 半精度浮動小数点圧縮（50%の容量削減、誤差 <= 2e-5）。
    Fp16,
    /// トークン単位対称 `INT8` 量子化（75%の容量削減、誤差 <= 8e-4）。
    Int8Symmetric,
}

/// `f32` スライスを `half::f16` へ圧縮する。
///
/// # 引数
/// - `src`: 入力 `f32` スライス。
pub fn compress_f32_to_fp16(src: &[f32]) -> Vec<f16> {
    src.iter().map(|&x| f16::from_f32(x)).collect()
}

/// `half::f16` スライスを `f32` スライスへ展開する。
///
/// # 引数
/// - `src`: 入力 `half::f16` スライス。
/// - `dst`: 出力先 `f32` スライス。
pub fn decompress_fp16_to_f32(src: &[f16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len(), "展開元と先の要素数が一致しません。");
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = s.to_f32();
    }
}

/// `f32` スライスをトークン単位対称 `INT8` へ圧縮する。
///
/// 各トークンベクトル（長さ `hidden_size`）ごとに最大絶対値 $M = \max |x|$ を算出し、
/// スケール $s = M / 127.0$ を用いて対称量子化を行う。
///
/// # 引数
/// - `src`: 入力 `f32` スライス。
/// - `hidden_size`: 隠れ層次元数。
pub fn compress_f32_to_int8_symmetric(src: &[f32], hidden_size: usize) -> (Vec<i8>, Vec<f32>) {
    assert!(hidden_size > 0, "hidden_sizeは1以上である必要があります。");
    assert_eq!(
        src.len() % hidden_size,
        0,
        "データ長はhidden_sizeの倍数である必要があります。"
    );

    let num_tokens = src.len() / hidden_size;
    let mut quantized = Vec::with_capacity(src.len());
    let mut scales = Vec::with_capacity(num_tokens);

    for token_slice in src.chunks_exact(hidden_size) {
        let mut max_abs = 0.0f32;
        for &val in token_slice {
            let abs_val = val.abs();
            if abs_val > max_abs {
                max_abs = abs_val;
            }
        }

        let scale = if max_abs > 1e-12 {
            max_abs / 127.0
        } else {
            1.0
        };
        scales.push(scale);

        let inv_scale = 1.0 / scale;
        for &val in token_slice {
            let q = (val * inv_scale).round().clamp(-128.0, 127.0) as i8;
            quantized.push(q);
        }
    }

    (quantized, scales)
}

/// トークン単位対称 `INT8` スライスを `f32` スライスへ展開する。
///
/// # 引数
/// - `src`: 量子化データスライス。
/// - `scales`: 各トークンのスケールスライス。
/// - `hidden_size`: 隠れ層次元数。
/// - `dst`: 出力先 `f32` スライス。
pub fn decompress_int8_symmetric_to_f32(
    src: &[i8],
    scales: &[f32],
    hidden_size: usize,
    dst: &mut [f32],
) {
    assert!(hidden_size > 0, "hidden_sizeは1以上である必要があります。");
    assert_eq!(src.len(), dst.len(), "展開元と先の要素数が一致しません。");
    assert_eq!(
        src.len() % hidden_size,
        0,
        "データ長はhidden_sizeの倍数である必要があります。"
    );
    assert_eq!(
        src.len() / hidden_size,
        scales.len(),
        "トークン数とスケール数が一致しません。"
    );

    for (q_chunk, (dst_chunk, &scale)) in src
        .chunks_exact(hidden_size)
        .zip(dst.chunks_exact_mut(hidden_size).zip(scales.iter()))
    {
        for (&q, d) in q_chunk.iter().zip(dst_chunk.iter_mut()) {
            *d = (q as f32) * scale;
        }
    }
}

/// キャッシュ内に保持されるテンソル実体ペイロード。
pub enum CachedTensorPayload {
    /// 非圧縮の `Value`。直接 `IoBinding::bind_input` へゼロコピーで受け渡し可能。
    Raw(Arc<Value>),
    /// `FP16` 圧縮済みテンソルデータ。
    Fp16(Arc<Vec<f16>>),
    /// トークン単位対称 `INT8` 量子化済みテンソルデータ。
    Int8Symmetric {
        /// 量子化された `i8` 配列。
        quantized: Arc<Vec<i8>>,
        /// 各トークンのスケール値配列。
        scales: Arc<Vec<f32>>,
    },
}

impl CachedTensorPayload {
    /// ペイロードの保持形式に対応する [`CacheCompressionMode`] を返す。
    pub fn compression_mode(&self) -> CacheCompressionMode {
        match self {
            Self::Raw(_) => CacheCompressionMode::None,
            Self::Fp16(_) => CacheCompressionMode::Fp16,
            Self::Int8Symmetric { .. } => CacheCompressionMode::Int8Symmetric,
        }
    }
}

impl fmt::Debug for CachedTensorPayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Raw(_) => write!(f, "CachedTensorPayload::Raw(Value)"),
            Self::Fp16(data) => write!(f, "CachedTensorPayload::Fp16(len={})", data.len()),
            Self::Int8Symmetric { quantized, scales } => write!(
                f,
                "CachedTensorPayload::Int8Symmetric(q_len={}, scales_len={})",
                quantized.len(),
                scales.len()
            ),
        }
    }
}

/// `State` テキストの中間特徴量キャッシュキーを表す Newtype ラッパー。
///
/// `xxHash3` (64-bit) によるハッシュ値を保持し、16進文字列フォーマット(`{:016x}`)やパース処理をカプセル化する。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct StateHash(pub u64);

impl StateHash {
    /// 生の `u64` ハッシュ値からキャッシュキーを構築する。
    #[inline]
    pub const fn from_raw(val: u64) -> Self {
        Self(val)
    }

    /// 保持している生の `u64` ハッシュ値を取得する。
    #[inline]
    pub const fn as_u64(&self) -> u64 {
        self.0
    }

    /// 内部の `u64` 値を取り出す。
    #[inline]
    pub const fn into_inner(self) -> u64 {
        self.0
    }
}

impl fmt::Display for StateHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

impl fmt::LowerHex for StateHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

impl fmt::UpperHex for StateHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::UpperHex::fmt(&self.0, f)
    }
}

impl std::str::FromStr for StateHash {
    type Err = std::num::ParseIntError;

    /// 16進数文字列(任意で `0x` または `0X` プレフィックス付き)から [`StateHash`] をパースする。
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let trimmed = s.trim();
        let hex_str = trimmed
            .strip_prefix("0x")
            .or_else(|| trimmed.strip_prefix("0X"))
            .unwrap_or(trimmed);
        let val = u64::from_str_radix(hex_str, 16)?;
        Ok(Self(val))
    }
}

impl From<u64> for StateHash {
    #[inline]
    fn from(val: u64) -> Self {
        Self(val)
    }
}

impl From<StateHash> for u64 {
    #[inline]
    fn from(hash: StateHash) -> Self {
        hash.0
    }
}

/// キャッシュに格納される `State` 中間特徴量エントリ。
#[derive(Debug)]
pub struct CachedStateFeatures {
    /// `xxHash3` (64-bit) によるキー。
    pub hash_key: StateHash,
    /// テンソル形状 (`[batch_size, seq_len, hidden_size]`)。
    pub shape: Vec<usize>,
    /// テンソル実体ペイロード。
    pub payload: CachedTensorPayload,
    /// テンソルが占有する実メモリバイト数（`moka` の `weigher` に使用）。
    pub byte_size: usize,
    /// `State` の有効トークンアテンションマスク列 (`[batch_size, seq_len]`)。
    pub state_mask: Vec<i64>,
    /// キャッシュされた State 原文テキスト（または先頭抜粋）。
    /// State-Offloading 時に System 2 へのエスカレーションプロンプト生成で活用される。
    pub state_text: Option<String>,
}

/// `State` 有効トークンマスクの長さを検証し、空の場合は全有効 (`1`) で初期化する。
fn validate_and_normalize_state_mask(
    shape: &[usize],
    mut state_mask: Vec<i64>,
) -> Result<Vec<i64>> {
    let total_tokens = if shape.len() >= 2 {
        shape[0] * shape[1]
    } else {
        0
    };
    if state_mask.is_empty() {
        if total_tokens > 0 {
            state_mask.resize(total_tokens, 1i64);
        }
        Ok(state_mask)
    } else if shape.len() < 2 || state_mask.len() != total_tokens {
        Err(RuntimeError::InvalidTensorData(format!(
            "state_mask 長さ ({}) が期待値 ({total_tokens}) と一致しません。",
            state_mask.len()
        )))
    } else {
        Ok(state_mask)
    }
}

impl CachedStateFeatures {
    /// 非圧縮の生 `Value` からキャッシュエントリを生成する。
    ///
    /// # 引数
    /// - `hash_key`: 64-bitハッシュキー。
    /// - `shape`: テンソル形状。
    /// - `ort_value`: ONNX Runtimeの出力 `Value`。
    /// - `byte_size`: 占有バイト数。
    /// - `state_mask`: `State` 有効トークンマスク列。空の場合は全トークン有効 (`1`) として補完される。
    ///
    /// # エラー
    /// - `shape` が 3 次元でない場合、または `shape[0] != 1` の場合は [`RuntimeError::ModelContractViolation`] を返す。
    /// - `state_mask` が非空かつ要素数が `shape[0] * shape[1]` と一致しない場合は [`RuntimeError::InvalidTensorData`] を返す。
    pub fn new_raw(
        hash_key: impl Into<StateHash>,
        shape: Vec<usize>,
        ort_value: Value,
        byte_size: usize,
        state_mask: Vec<i64>,
    ) -> Result<Self> {
        Self::new_raw_with_text(hash_key, shape, ort_value, byte_size, state_mask, None)
    }

    /// 非圧縮の生 `Value` と State 原文テキストからキャッシュエントリを生成する。
    ///
    /// # エラー
    /// - `shape` が 3 次元でない場合、または `shape[0] != 1` の場合は [`RuntimeError::ModelContractViolation`] を返す。
    /// - `state_mask` が非空かつ要素数が `shape[0] * shape[1]` と一致しない場合は [`RuntimeError::InvalidTensorData`] を返す。
    pub fn new_raw_with_text(
        hash_key: impl Into<StateHash>,
        shape: Vec<usize>,
        ort_value: Value,
        byte_size: usize,
        state_mask: Vec<i64>,
        state_text: Option<String>,
    ) -> Result<Self> {
        if shape.len() != 3 {
            return Err(RuntimeError::ModelContractViolation(format!(
                "State特徴量は3次元テンソルである必要があります: {shape:?}。"
            )));
        }

        if shape[0] != 1 {
            return Err(RuntimeError::ModelContractViolation(format!(
                "バッチ展開元のキャッシュエントリは batch_size == 1 である必要があります: shape={shape:?}。"
            )));
        }

        let state_mask = validate_and_normalize_state_mask(&shape, state_mask)?;
        let hash_key = hash_key.into();

        Ok(Self {
            hash_key,
            shape,
            payload: CachedTensorPayload::Raw(Arc::new(ort_value)),
            byte_size,
            state_mask,
            state_text,
        })
    }

    /// 明示指定された文脈テキストとキャッシュ保持テキストから、ゲーティング用の有効文脈を解決する。
    fn resolve_state_text<'a>(&'a self, state_text: Option<&'a str>) -> Option<&'a str> {
        match state_text {
            Some(s) if !s.trim().is_empty() => Some(s),
            _ => self.state_text.as_deref(),
        }
    }

    /// `f32` スライスを指定された圧縮モードで変換してキャッシュエントリを生成する。
    ///
    /// # 引数
    /// - `hash_key`: 64-bitハッシュキー。
    /// - `shape`: テンソル形状 (`[batch_size, seq_len, hidden_size]`)。
    /// - `raw_slice`: 展開済み `f32` スライス。
    /// - `mode`: 圧縮モード。
    /// - `state_mask`: `State` 有効トークンマスク列。空の場合は全トークン有効 (`1`) として補完される。
    ///
    /// # エラー
    /// - `shape` が 3 次元でない場合、または `shape[0] != 1` の場合は [`RuntimeError::ModelContractViolation`] を返す。
    /// - `raw_slice` の要素数が `shape` の総要素数と一致しない場合、または要素数計算がオーバーフローした場合は [`RuntimeError::InvalidTensorData`] を返す。
    /// - `state_mask` が非空かつ要素数が `shape[0] * shape[1]` と一致しない場合は [`RuntimeError::InvalidTensorData`] を返す。
    pub fn new_compressed(
        hash_key: impl Into<StateHash>,
        shape: Vec<usize>,
        raw_slice: &[f32],
        mode: CacheCompressionMode,
        state_mask: Vec<i64>,
    ) -> Result<Self> {
        Self::new_compressed_with_text(hash_key, shape, raw_slice, mode, state_mask, None)
    }

    /// `f32` スライスと State 原文テキストを指定された圧縮モードで変換してキャッシュエントリを生成する。
    ///
    /// # エラー
    /// - `shape` が 3 次元でない場合、または `shape[0] != 1` の場合は [`RuntimeError::ModelContractViolation`] を返す。
    /// - `raw_slice` の要素数が `shape` の総要素数と一致しない場合、または要素数計算がオーバーフローした場合は [`RuntimeError::InvalidTensorData`] を返す。
    /// - `state_mask` が非空かつ要素数が `shape[0] * shape[1]` と一致しない場合は [`RuntimeError::InvalidTensorData`] を返す。
    pub fn new_compressed_with_text(
        hash_key: impl Into<StateHash>,
        shape: Vec<usize>,
        raw_slice: &[f32],
        mode: CacheCompressionMode,
        state_mask: Vec<i64>,
        state_text: Option<String>,
    ) -> Result<Self> {
        if shape.len() != 3 {
            return Err(RuntimeError::ModelContractViolation(format!(
                "State特徴量は3次元テンソルである必要があります: {shape:?}。"
            )));
        }

        if shape[0] != 1 {
            return Err(RuntimeError::ModelContractViolation(format!(
                "バッチ展開元のキャッシュエントリは batch_size == 1 である必要があります: shape={shape:?}。"
            )));
        }

        let total_elements = shape[0]
            .checked_mul(shape[1])
            .and_then(|v| v.checked_mul(shape[2]))
            .ok_or_else(|| {
                RuntimeError::InvalidTensorData(
                    "テンソル要素数の計算がオーバーフローしました。".to_string(),
                )
            })?;

        if raw_slice.len() != total_elements {
            return Err(RuntimeError::InvalidTensorData(format!(
                "raw_slice の要素数 ({}) が shape ({shape:?}) の総要素数 ({total_elements}) と一致しません。",
                raw_slice.len()
            )));
        }

        let hidden_size = shape[2];
        if mode == CacheCompressionMode::Int8Symmetric && hidden_size == 0 {
            return Err(RuntimeError::InvalidTensorData(
                "Int8Symmetric モードでは hidden_size は 1 以上である必要があります。".to_string(),
            ));
        }

        let state_mask = validate_and_normalize_state_mask(&shape, state_mask)?;
        let hash_key = hash_key.into();

        match mode {
            CacheCompressionMode::None => {
                let allocator = Allocator::default();
                let mut tensor =
                    Tensor::<f32>::new(&allocator, shape.clone()).map_err(RuntimeError::Ort)?;
                let (_, dst) = tensor
                    .try_extract_tensor_mut::<f32>()
                    .map_err(RuntimeError::Ort)?;
                dst.copy_from_slice(raw_slice);
                let byte_size = std::mem::size_of_val(raw_slice);
                Ok(Self {
                    hash_key,
                    shape,
                    payload: CachedTensorPayload::Raw(Arc::new(tensor.into_dyn())),
                    byte_size,
                    state_mask,
                    state_text,
                })
            }
            CacheCompressionMode::Fp16 => {
                let compressed = compress_f32_to_fp16(raw_slice);
                let byte_size = std::mem::size_of_val(compressed.as_slice());
                Ok(Self {
                    hash_key,
                    shape,
                    payload: CachedTensorPayload::Fp16(Arc::new(compressed)),
                    byte_size,
                    state_mask,
                    state_text,
                })
            }
            CacheCompressionMode::Int8Symmetric => {
                let (quantized, scales) = compress_f32_to_int8_symmetric(raw_slice, hidden_size);
                let byte_size = quantized.len() * std::mem::size_of::<i8>()
                    + scales.len() * std::mem::size_of::<f32>();
                Ok(Self {
                    hash_key,
                    shape,
                    payload: CachedTensorPayload::Int8Symmetric {
                        quantized: Arc::new(quantized),
                        scales: Arc::new(scales),
                    },
                    byte_size,
                    state_mask,
                    state_text,
                })
            }
        }
    }

    /// テンソルペイロードの圧縮モードを取得する。
    pub fn compression_mode(&self) -> CacheCompressionMode {
        self.payload.compression_mode()
    }

    /// State 原文テキストを設定するビルダーメソッド。
    pub fn with_state_text(mut self, text: Option<String>) -> Self {
        self.state_text = text;
        self
    }

    /// 推論入力としてバインド可能な `Arc<Value>` を取得する。
    ///
    /// 非圧縮時は保持している `Arc<Value>` をそのままゼロコピーで返却し、
    /// 圧縮時はオンザフライで ONNX Runtime のアロケータメモリへ直接デコードを展開してテンソルを生成する。
    pub fn as_ort_value(&self) -> Result<Arc<Value>> {
        match &self.payload {
            CachedTensorPayload::Raw(val) => Ok(Arc::clone(val)),
            CachedTensorPayload::Fp16(compressed) => {
                let allocator = Allocator::default();
                let mut tensor = Tensor::<f32>::new(&allocator, self.shape.clone())
                    .map_err(RuntimeError::Ort)?;
                let (_, dst) = tensor
                    .try_extract_tensor_mut::<f32>()
                    .map_err(RuntimeError::Ort)?;
                decompress_fp16_to_f32(compressed, dst);
                Ok(Arc::new(tensor.into_dyn()))
            }
            CachedTensorPayload::Int8Symmetric { quantized, scales } => {
                let hidden_size = self.hidden_size();
                let allocator = Allocator::default();
                let mut tensor = Tensor::<f32>::new(&allocator, self.shape.clone())
                    .map_err(RuntimeError::Ort)?;
                let (_, dst) = tensor
                    .try_extract_tensor_mut::<f32>()
                    .map_err(RuntimeError::Ort)?;
                decompress_int8_symmetric_to_f32(quantized, scales, hidden_size, dst);
                Ok(Arc::new(tensor.into_dyn()))
            }
        }
    }

    /// 指定されたバッチサイズに合わせてバインド可能な `Arc<Value>` を取得する。
    ///
    /// 単一サンプル (`[1, seq_len, hidden_size]`) のキャッシュエントリを前提とし、
    /// `target_batch == 1` の場合はゼロコピー返却(圧縮時は 1 サンプル分を展開)を行う。
    /// `target_batch > 1` の場合は、単一サンプルの特徴量データを `target_batch` 回リピートした
    /// テンソル (`[target_batch, seq_len, hidden_size]`) を ONNX Runtime アロケータメモリへ直接構築して返却する。
    /// 中間ベクタのヒープ確保と再アロケーションを回避し、64バイトアライメントを維持する。
    ///
    /// # エラー
    /// - `target_batch == 0`、サイズ計算オーバーフロー、次元長 0、またはペイロード長不整合の場合は [`RuntimeError::InvalidTensorData`] を返す。
    /// - キャッシュテンソルが 3 次元でない場合や `self.batch_size() != 1` の場合は [`RuntimeError::ModelContractViolation`] を返す。
    pub fn as_ort_value_for_batch(&self, target_batch: usize) -> Result<Arc<Value>> {
        if target_batch == 0 {
            return Err(RuntimeError::InvalidTensorData(
                "target_batch は 1 以上である必要があります。".to_string(),
            ));
        }

        if self.shape.len() != 3 {
            return Err(RuntimeError::ModelContractViolation(format!(
                "State特徴量は3次元テンソルである必要があります: {:?}。",
                self.shape
            )));
        }

        let batch_size = self.batch_size();
        if batch_size != 1 {
            return Err(RuntimeError::ModelContractViolation(format!(
                "バッチ展開元のキャッシュエントリは batch_size == 1 である必要があります: batch_size={batch_size}。"
            )));
        }

        let seq_len = self.seq_len();
        let hidden_size = self.hidden_size();
        let single_len = seq_len.checked_mul(hidden_size).ok_or_else(|| {
            RuntimeError::InvalidTensorData(
                "テンソル要素数の計算がオーバーフローしました。".to_string(),
            )
        })?;
        if single_len == 0 {
            return Err(RuntimeError::InvalidTensorData(format!(
                "State特徴量の系列長および隠れ層次元数は 1 以上である必要があります: {:?}。",
                self.shape
            )));
        }

        if target_batch == 1 {
            return self.as_ort_value();
        }

        target_batch.checked_mul(single_len).ok_or_else(|| {
            RuntimeError::InvalidTensorData(
                "バッチ展開後のテンソル要素数がオーバーフローしました。".to_string(),
            )
        })?;

        let target_shape = vec![target_batch, seq_len, hidden_size];

        let allocator = Allocator::default();
        let mut tensor = Tensor::<f32>::new(&allocator, target_shape).map_err(RuntimeError::Ort)?;
        let (_, dst) = tensor
            .try_extract_tensor_mut::<f32>()
            .map_err(RuntimeError::Ort)?;

        match &self.payload {
            CachedTensorPayload::Raw(val) => {
                let (_, single_slice) =
                    val.try_extract_tensor::<f32>().map_err(RuntimeError::Ort)?;
                if single_slice.len() != single_len {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "Raw キャッシュデータの要素数 ({}) が期待値 ({single_len}) と一致しません。",
                        single_slice.len()
                    )));
                }
                for chunk in dst.chunks_exact_mut(single_len) {
                    chunk.copy_from_slice(single_slice);
                }
            }
            CachedTensorPayload::Fp16(compressed) => {
                if compressed.len() != single_len {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "FP16 キャッシュデータの要素数 ({}) が期待値 ({single_len}) と一致しません。",
                        compressed.len()
                    )));
                }
                let (first_chunk, rest) = dst.split_at_mut(single_len);
                decompress_fp16_to_f32(compressed, first_chunk);
                for chunk in rest.chunks_exact_mut(single_len) {
                    chunk.copy_from_slice(first_chunk);
                }
            }
            CachedTensorPayload::Int8Symmetric { quantized, scales } => {
                if quantized.len() != single_len || scales.len() != seq_len {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "INT8 キャッシュデータの要素数 (quantized={}, scales={}) が期待値 ({single_len}, {seq_len}) と一致しません。",
                        quantized.len(),
                        scales.len()
                    )));
                }
                let (first_chunk, rest) = dst.split_at_mut(single_len);
                decompress_int8_symmetric_to_f32(quantized, scales, hidden_size, first_chunk);
                for chunk in rest.chunks_exact_mut(single_len) {
                    chunk.copy_from_slice(first_chunk);
                }
            }
        }

        Ok(Arc::new(tensor.into_dyn()))
    }

    /// バッチサイズを取得する。
    pub fn batch_size(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }

    /// 系列長（トークン数）を取得する。
    pub fn seq_len(&self) -> usize {
        if self.shape.len() >= 2 {
            self.shape[1]
        } else {
            0
        }
    }

    /// 隠れ層次元数を取得する。
    pub fn hidden_size(&self) -> usize {
        if self.shape.len() >= 3 {
            self.shape[2]
        } else {
            0
        }
    }

    /// 指定されたバッチサイズに合わせて複製された `state_mask` を取得する。
    ///
    /// キャッシュされた単一 `State` のマスク列 (`[state_seq_len]`) を `target_batch` 回リピートした
    /// テンソル用フラットベクタ (`[target_batch * state_seq_len]`) を生成する。
    pub fn state_mask_for_batch(&self, target_batch: usize) -> Vec<i64> {
        let seq_len = self.seq_len();
        let mut repeated = Vec::with_capacity(target_batch * seq_len);
        let mask_slice = if self.state_mask.len() == seq_len {
            &self.state_mask[..]
        } else if self.state_mask.len() >= seq_len {
            &self.state_mask[..seq_len]
        } else {
            &[]
        };

        if mask_slice.len() == seq_len {
            for _ in 0..target_batch {
                repeated.extend_from_slice(mask_slice);
            }
        } else {
            let fallback = vec![1i64; seq_len];
            for _ in 0..target_batch {
                repeated.extend_from_slice(&fallback);
            }
        }
        repeated
    }
}

/// 並行 `State` 特徴量キャッシュマネージャー。
///
/// `xxHash3` による高速キー計算と、`moka` の TinyLFU/LRU 並行キャッシュにより、
/// 複数スレッドからの同時参照をロックフリーに処理し、バイト重み付き退避を行う。
pub struct StateCacheManager {
    cache: Cache<StateHash, Arc<CachedStateFeatures>>,
    compression_mode: CacheCompressionMode,
}

impl StateCacheManager {
    /// 新規キャッシュマネージャーを初期化する。
    ///
    /// # 引数
    /// - `max_capacity_bytes`: 最大常駐メモリ容量（バイト数）。
    /// - `ttl_secs`: エントリの有効期限（秒数）。
    /// - `compression_mode`: キャッシュ圧縮モード。
    pub fn new(
        max_capacity_bytes: u64,
        ttl_secs: u64,
        compression_mode: CacheCompressionMode,
    ) -> Self {
        let cache = Cache::builder()
            .weigher(|_key, value: &Arc<CachedStateFeatures>| {
                value.byte_size.min(u32::MAX as usize) as u32
            })
            .max_capacity(max_capacity_bytes)
            .time_to_live(Duration::from_secs(ttl_secs))
            .build();

        Self {
            cache,
            compression_mode,
        }
    }

    /// `State` テキストから `xxHash3` (64-bit) ハッシュキーを高速算出する。
    ///
    /// Unicode NFC 正規化を行い、NFD/NFC 間の正準等価性を保証する。
    /// 既に NFC 形式である通常の文字列に対してはゼロアロケーションで高速通過する。
    /// `AVX2`、`AVX-512`、`ARM NEON` 命令により、8kトークンのテキストでも1μs未満で完了する。
    ///
    /// # 引数
    /// - `state_text`: `State` 原文テキスト。
    pub fn compute_key(state_text: &str) -> StateHash {
        let raw = match is_nfc_quick(state_text.chars()) {
            IsNormalized::Yes => xxh3_64(state_text.as_bytes()),
            IsNormalized::No | IsNormalized::Maybe => {
                let normalized: String = state_text.nfc().collect();
                xxh3_64(normalized.as_bytes())
            }
        };
        StateHash(raw)
    }

    /// 指定されたキーの特徴量を取得する。
    ///
    /// # 引数
    /// - `key`: `xxHash3` キー。
    pub fn get(&self, key: impl Into<StateHash>) -> Option<Arc<CachedStateFeatures>> {
        let key = key.into();
        self.cache.get(&key)
    }

    /// 特徴量をキャッシュに登録する。
    ///
    /// # 引数
    /// - `key`: `xxHash3` キー。
    /// - `features`: キャッシュエントリ。
    pub fn insert(&self, key: impl Into<StateHash>, features: Arc<CachedStateFeatures>) {
        let key = key.into();
        self.cache.insert(key, features);
    }

    /// 指定されたキーが存在するかを判定する。
    pub fn contains_key(&self, key: impl Into<StateHash>) -> bool {
        let key = key.into();
        self.cache.contains_key(&key)
    }

    /// 指定されたキーのエントリを無効化（削除）する。
    pub fn invalidate(&self, key: impl Into<StateHash>) {
        let key = key.into();
        self.cache.invalidate(&key);
    }

    /// 全キャッシュエントリを破棄する。
    pub fn clear(&self) {
        self.cache.invalidate_all();
    }

    /// 現在保持しているキャッシュエントリ数を取得する。
    pub fn entry_count(&self) -> u64 {
        self.cache.entry_count()
    }

    /// キャッシュ内の総メモリ占有バイト数を取得する。
    pub fn weighted_size(&self) -> u64 {
        self.cache.weighted_size()
    }

    /// 設定されている圧縮モードを取得する。
    pub fn compression_mode(&self) -> CacheCompressionMode {
        self.compression_mode
    }

    /// 保留中のキャッシュメンテナンスタスク（退避、カウント更新等）を直ちに同期実行する。
    pub fn run_pending_tasks(&self) {
        self.cache.run_pending_tasks();
    }
}

/// 2セッション分割型推論エンジン (`DualSessionEngine`)。
///
/// `state_session` で `State` をエンコードし、得られた隠れ状態をキャッシュした上で、
/// `query_session` の入力スロットへ `IoBinding` でゼロコピーバインドして超低遅延推論を実行する。
///
/// 内部に同期セッションプール（`query_sessions`、`state_sessions`）を保持し、
/// 複数スレッドからの並行評価要求に対して非ブロッキングラウンドロビン走査でロック競合を回避し、並列推論を実行する。
pub struct DualSessionEngine {
    state_sessions: Vec<Mutex<Session>>,
    query_sessions: Vec<Mutex<Session>>,
    next_state_idx: AtomicUsize,
    next_query_idx: AtomicUsize,
    cache_manager: Arc<StateCacheManager>,
    active_provider: ExecutionProvider,
}

impl DualSessionEngine {
    /// ファイルパスからデュアルモデルを読み込み、エンジンを初期化する。
    ///
    /// # 引数
    /// - `state_model_path`: `state_encoder.onnx` ファイルパス。
    /// - `query_model_path`: `query_evaluator.onnx` ファイルパス。
    /// - `config`: セッション構築設定。
    /// - `cache_manager`: キャッシュマネージャー。
    pub fn new(
        state_model_path: impl AsRef<Path>,
        query_model_path: impl AsRef<Path>,
        config: SessionConfig,
        cache_manager: Arc<StateCacheManager>,
    ) -> Result<Self> {
        let state_path = state_model_path.as_ref();
        let query_path = query_model_path.as_ref();
        let (state_sessions, state_provider) =
            build_session_pool(&config, |mut b| b.commit_from_file(state_path))?;
        let (query_sessions, query_provider) =
            build_session_pool(&config, |mut b| b.commit_from_file(query_path))?;

        Ok(Self::init_with_pools(
            state_sessions,
            query_sessions,
            state_provider,
            query_provider,
            cache_manager,
        ))
    }

    /// メモリ上のバイト列からデュアルモデルを読み込み、エンジンを初期化する。
    ///
    /// # 引数
    /// - `state_model_bytes`: `state_encoder.onnx` バイナリ。
    /// - `query_model_bytes`: `query_evaluator.onnx` バイナリ。
    /// - `config`: セッション構築設定。
    /// - `cache_manager`: キャッシュマネージャー。
    pub fn from_bytes(
        state_model_bytes: &[u8],
        query_model_bytes: &[u8],
        config: SessionConfig,
        cache_manager: Arc<StateCacheManager>,
    ) -> Result<Self> {
        let (state_sessions, state_provider) =
            build_session_pool(&config, |mut b| b.commit_from_memory(state_model_bytes))?;
        let (query_sessions, query_provider) =
            build_session_pool(&config, |mut b| b.commit_from_memory(query_model_bytes))?;

        Ok(Self::init_with_pools(
            state_sessions,
            query_sessions,
            state_provider,
            query_provider,
            cache_manager,
        ))
    }

    /// 構築済みの State/Query セッションプールを統合し、エンジンインスタンスを完成させる。
    fn init_with_pools(
        state_sessions: Vec<Mutex<Session>>,
        query_sessions: Vec<Mutex<Session>>,
        state_provider: ExecutionProvider,
        query_provider: ExecutionProvider,
        cache_manager: Arc<StateCacheManager>,
    ) -> Self {
        if state_provider != query_provider {
            tracing::warn!(
                "State セッションと Query セッションで Execution Provider が不一致です (State: {}, Query: {})。",
                state_provider.name(),
                query_provider.name()
            );
        }

        Self {
            state_sessions,
            query_sessions,
            next_state_idx: AtomicUsize::new(0),
            next_query_idx: AtomicUsize::new(0),
            cache_manager,
            active_provider: state_provider,
        }
    }

    /// 構築済みのセッションからエンジンインスタンスを生成する。
    pub fn from_sessions(
        state_session: Session,
        query_session: Session,
        active_provider: ExecutionProvider,
        cache_manager: Arc<StateCacheManager>,
    ) -> Self {
        Self {
            state_sessions: vec![Mutex::new(state_session)],
            query_sessions: vec![Mutex::new(query_session)],
            next_state_idx: AtomicUsize::new(0),
            next_query_idx: AtomicUsize::new(0),
            cache_manager,
            active_provider,
        }
    }

    /// セッションプールから利用可能な `Query` セッションを取得する。
    ///
    /// ロックされていない空きセッションを優先探索し、全セッションが使用中の場合はブロック待機する。
    fn acquire_query_session(&self) -> Result<std::sync::MutexGuard<'_, Session>> {
        acquire_pooled_session(&self.query_sessions, &self.next_query_idx, "Query")
    }

    /// セッションプールから利用可能な `State` セッションを取得する。
    ///
    /// ロックされていない空きセッションを優先探索し、全セッションが使用中の場合はブロック待機する。
    fn acquire_state_session(&self) -> Result<std::sync::MutexGuard<'_, Session>> {
        acquire_pooled_session(&self.state_sessions, &self.next_state_idx, "State")
    }

    /// 保持しているキャッシュマネージャーへの参照を取得する。
    pub fn cache_manager(&self) -> &Arc<StateCacheManager> {
        &self.cache_manager
    }

    /// 稼働中の Execution Provider を取得する。
    pub fn active_provider(&self) -> ExecutionProvider {
        self.active_provider
    }

    /// `State` をエンコードしてキャッシュへ格納し、キャッシュエントリを返す。
    ///
    /// 既にキャッシュに存在する場合は推論を省略し、即座にキャッシュエントリを返す。
    ///
    /// # 引数
    /// - `state_text`: キャッシュキー生成用の `State` 原文テキスト。
    /// - `input_ids`: トークンID列 (`[batch_size, seq_len]`)。
    /// - `attention_mask`: アテンションマスク列 (`[batch_size, seq_len]`)。
    /// - `dims`: `[batch_size, seq_len]` の次元配列。
    ///
    /// # エラー
    /// - `dims[0] != 1` またはモデル出力のバッチサイズが 1 でない場合は [`RuntimeError::ModelContractViolation`] を返す。
    /// - `input_ids` や `attention_mask` の長さが `dims` と一致しない場合は [`RuntimeError::InvalidTensorData`] を返す。
    pub fn encode_and_cache_state(
        &self,
        state_text: &str,
        input_ids: &[i64],
        attention_mask: &[i64],
        dims: [usize; 2],
    ) -> Result<Arc<CachedStateFeatures>> {
        let key = StateCacheManager::compute_key(state_text);
        if let Some(cached) = self.cache_manager.get(key) {
            return Ok(cached);
        }

        let [batch_size, seq_len] = dims;
        if batch_size != 1 {
            return Err(RuntimeError::ModelContractViolation(format!(
                "Stateエンコードのバッチサイズは 1 である必要があります: batch_size={batch_size}。"
            )));
        }

        if input_ids.len() != batch_size * seq_len {
            return Err(RuntimeError::InvalidTensorData(format!(
                "input_ids 長さ ({}) が期待値 ({batch_size} * {seq_len} = {}) と一致しません。",
                input_ids.len(),
                batch_size * seq_len
            )));
        }
        if attention_mask.len() != batch_size * seq_len {
            return Err(RuntimeError::InvalidTensorData(format!(
                "attention_mask 長さ ({}) が期待値 ({batch_size} * {seq_len} = {}) と一致しません。",
                attention_mask.len(),
                batch_size * seq_len
            )));
        }

        let mut session = self.acquire_state_session()?;

        let mut binding = session.create_binding().map_err(RuntimeError::Ort)?;
        let mem_info = MemoryInfo::default();

        let ids_val = Value::from_array(([batch_size, seq_len], input_ids.to_vec()))
            .map_err(RuntimeError::Ort)?;
        let mask_val = Value::from_array(([batch_size, seq_len], attention_mask.to_vec()))
            .map_err(RuntimeError::Ort)?;

        binding
            .bind_input(TENSOR_INPUT_IDS, &ids_val)
            .map_err(RuntimeError::Ort)?;
        binding
            .bind_input(TENSOR_ATTENTION_MASK, &mask_val)
            .map_err(RuntimeError::Ort)?;
        binding
            .bind_output_to_device(TENSOR_STATE_HIDDEN_STATES, &mem_info)
            .map_err(RuntimeError::Ort)?;

        let mut outputs = session.run_binding(&binding).map_err(RuntimeError::Ort)?;
        let state_val = outputs.remove(TENSOR_STATE_HIDDEN_STATES).ok_or_else(|| {
            RuntimeError::ModelContractViolation(format!(
                "出力テンソル '{TENSOR_STATE_HIDDEN_STATES}' が存在しません。"
            ))
        })?;

        let (shape, flat_data) = state_val
            .try_extract_tensor::<f32>()
            .map_err(RuntimeError::Ort)?;
        let shape_vec: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        if shape_vec.len() != 3 {
            return Err(RuntimeError::ModelContractViolation(format!(
                "State特徴量は3次元テンソルである必要があります: {shape_vec:?}。"
            )));
        }
        if shape_vec[0] != 1 {
            return Err(RuntimeError::ModelContractViolation(format!(
                "Stateモデルの出力テンソルは batch_size == 1 である必要があります: shape={shape_vec:?}。"
            )));
        }

        let compression = self.cache_manager.compression_mode();
        let state_text_owned = Some(state_text.to_string());
        let cached_entry = if compression == CacheCompressionMode::None {
            let total_elements: usize = shape_vec.iter().product();
            let byte_size = total_elements * std::mem::size_of::<f32>();
            Arc::new(CachedStateFeatures::new_raw_with_text(
                key,
                shape_vec,
                state_val,
                byte_size,
                attention_mask.to_vec(),
                state_text_owned,
            )?)
        } else {
            Arc::new(CachedStateFeatures::new_compressed_with_text(
                key,
                shape_vec,
                flat_data,
                compression,
                attention_mask.to_vec(),
                state_text_owned,
            )?)
        };

        self.cache_manager.insert(key, Arc::clone(&cached_entry));
        Ok(cached_entry)
    }

    /// キャッシュされた `State` 特徴量をゼロコピーバインドして質問を推論する。
    ///
    /// # 引数
    /// - `cached_state`: キャッシュされた `State` 特徴量。
    /// - `state_mask`: `State` マスク列 (`[batch_size, state_seq_len]`)。
    /// - `query_ids`: 質問トークンID列 (`[batch_size, query_seq_len]`)。
    /// - `query_mask`: 質問アテンションマスク列 (`[batch_size, query_seq_len]`)。
    /// - `op_indices`: 候補アンカーインデックス (`[batch_size, num_options]`)。
    /// - `query_dims`: 質問側の入力次元情報。
    ///
    /// # 戻り値
    /// 各バッチサンプルのロジット配列 (`Vec<Vec<f64>>`)。
    pub fn evaluate_query_with_cache(
        &self,
        cached_state: &CachedStateFeatures,
        state_mask: &[i64],
        query_ids: &[i64],
        query_mask: &[i64],
        op_indices: &[i64],
        query_dims: ModelInputDimensions,
    ) -> Result<Vec<Vec<f64>>> {
        let state_val = cached_state.as_ort_value_for_batch(query_dims.batch_size)?;
        self.evaluate_query_with_state_value(
            state_val.as_ref(),
            cached_state.seq_len(),
            state_mask,
            query_ids,
            query_mask,
            op_indices,
            query_dims,
        )
    }

    /// 構築済みの `State` テンソル `Value` を直接バインドして質問を推論する。
    ///
    /// 複数チャンクのバッチ推論において、同一チャンク長の `state_val` を再利用して
    /// リピートテンソルの再生成アロケーションをスキップするために使用する。
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_query_with_state_value(
        &self,
        state_val: &Value,
        state_seq_len: usize,
        state_mask: &[i64],
        query_ids: &[i64],
        query_mask: &[i64],
        op_indices: &[i64],
        query_dims: ModelInputDimensions,
    ) -> Result<Vec<Vec<f64>>> {
        let batch_size = query_dims.batch_size;
        let query_seq_len = query_dims.sequence_length;
        let num_options = query_dims.num_options;

        // 1. 動的軸の事前検証 (Fast-Fail)。
        if state_mask.len() != batch_size * state_seq_len {
            return Err(RuntimeError::CacheDimensionMismatch {
                expected: batch_size * state_seq_len,
                actual: state_mask.len(),
            });
        }
        if query_ids.len() != batch_size * query_seq_len {
            return Err(RuntimeError::InvalidTensorData(format!(
                "query_ids 長さ ({}) が期待値 ({batch_size} * {query_seq_len} = {}) と一致しません。",
                query_ids.len(),
                batch_size * query_seq_len
            )));
        }
        if query_mask.len() != batch_size * query_seq_len {
            return Err(RuntimeError::InvalidTensorData(format!(
                "query_mask 長さ ({}) が期待値 ({batch_size} * {query_seq_len} = {}) と一致しません。",
                query_mask.len(),
                batch_size * query_seq_len
            )));
        }
        if op_indices.len() != batch_size * num_options {
            return Err(RuntimeError::InvalidTensorData(format!(
                "op_indices 長さ ({}) が期待値 ({batch_size} * {num_options} = {}) と一致しません。",
                op_indices.len(),
                batch_size * num_options
            )));
        }

        // 2. `op_indices` の範囲外検証（クエリ系列長基準、パディング -1 は許容）。
        crate::engine::session::validate_op_indices(op_indices, query_seq_len)?;

        let mut session = self.acquire_query_session()?;

        let mut binding = session.create_binding().map_err(RuntimeError::Ort)?;
        let mem_info = MemoryInfo::default();

        // 3. キャッシュ特徴量のゼロコピー / バッチ適合バインド。
        binding
            .bind_input(TENSOR_STATE_HIDDEN_STATES, state_val)
            .map_err(RuntimeError::Ort)?;

        // 4. `Query` 側入力テンソルのバインド。
        let s_mask_val = Value::from_array(([batch_size, state_seq_len], state_mask.to_vec()))
            .map_err(RuntimeError::Ort)?;
        let q_ids_val = Value::from_array(([batch_size, query_seq_len], query_ids.to_vec()))
            .map_err(RuntimeError::Ort)?;
        let q_mask_val = Value::from_array(([batch_size, query_seq_len], query_mask.to_vec()))
            .map_err(RuntimeError::Ort)?;
        let op_idx_val = Value::from_array(([batch_size, num_options], op_indices.to_vec()))
            .map_err(RuntimeError::Ort)?;

        binding
            .bind_input(TENSOR_STATE_MASK, &s_mask_val)
            .map_err(RuntimeError::Ort)?;
        binding
            .bind_input(TENSOR_QUERY_IDS, &q_ids_val)
            .map_err(RuntimeError::Ort)?;
        binding
            .bind_input(TENSOR_QUERY_MASK, &q_mask_val)
            .map_err(RuntimeError::Ort)?;
        binding
            .bind_input(TENSOR_OP_INDICES, &op_idx_val)
            .map_err(RuntimeError::Ort)?;

        binding
            .bind_output_to_device(TENSOR_LOGITS, &mem_info)
            .map_err(RuntimeError::Ort)?;

        // 5. 推論実行。
        let outputs = session.run_binding(&binding).map_err(RuntimeError::Ort)?;
        let logits_out = outputs.get(TENSOR_LOGITS).ok_or_else(|| {
            RuntimeError::ModelContractViolation(format!(
                "出力テンソル '{TENSOR_LOGITS}' が存在しません。"
            ))
        })?;

        let (shape, logits_f32) = logits_out
            .try_extract_tensor::<f32>()
            .map_err(RuntimeError::Ort)?;

        if shape.len() != 2 {
            return Err(RuntimeError::ModelContractViolation(format!(
                "出力テンソル '{TENSOR_LOGITS}' は 2 次元テンソル [batch, num_options] である必要があります: {shape:?}。"
            )));
        }
        let shape_stack = [shape[0] as usize, shape[1] as usize];
        query_dims.validate_output_shape(&shape_stack)?;

        // 6. 出力値の有限性 (`NaN`/`Inf`) 検証および `f32` から `f64` へのキャスト。
        let mut batched_logits = Vec::with_capacity(batch_size);
        for chunk in logits_f32.chunks_exact(num_options) {
            let mut sample_logits = Vec::with_capacity(num_options);
            for &val in chunk {
                if !val.is_finite() {
                    return Err(RuntimeError::InvalidTensorData(format!(
                        "logits に非有限値 (NaN または Inf) が検出されました: {val}。"
                    )));
                }
                sample_logits.push(val as f64);
            }
            batched_logits.push(sample_logits);
        }

        Ok(batched_logits)
    }

    /// `Tokio` ワーカースレッドの枯渇を防ぐため、`tokio::task::spawn_blocking` で隔離して推論を実行する。
    ///
    /// # 引数
    /// - `engine`: デュアルセッションエンジン。
    /// - `cached_state`: キャッシュされた `State` 特徴量。
    /// - `state_mask`: `State` マスク列。
    /// - `query_ids`: 質問トークンID列。
    /// - `query_mask`: 質問アテンションマスク列。
    /// - `op_indices`: 候補アンカーインデックス。
    /// - `query_dims`: 質問側の入力次元情報。
    pub fn evaluate_query_async(
        engine: Arc<Self>,
        cached_state: Arc<CachedStateFeatures>,
        state_mask: Vec<i64>,
        query_ids: Vec<i64>,
        query_mask: Vec<i64>,
        op_indices: Vec<i64>,
        query_dims: ModelInputDimensions,
    ) -> tokio::task::JoinHandle<Result<Vec<Vec<f64>>>> {
        tokio::task::spawn_blocking(move || {
            engine.evaluate_query_with_cache(
                &cached_state,
                &state_mask,
                &query_ids,
                &query_mask,
                &op_indices,
                query_dims,
            )
        })
    }

    /// キャッシュされた `State` 特徴量をゼロコピー参照し、複数の質問を一括またはマイクロバッチ評価する。
    ///
    /// 内部で各質問の `Query` 部分を事前トークナイズし、`chunk_size` 単位のマイクロバッチとして
    /// `evaluate_query_with_cache` を実行することで、L3 キャッシュ保護と並列推論を両立する。
    ///
    /// # 引数
    /// - `tokenizer`: 高速トークナイザー。
    /// - `cached_state`: キャッシュされた `State` 特徴量。
    /// - `state_text`: `State` テキスト (ゲーティングや OOD 判定用、任意)。
    /// - `questions`: 評価対象質問群のマップ。
    /// - `calib_config`: 事後較正設定。
    /// - `gating_config`: 確信度ゲーティング設定 (任意)。
    /// - `chunk_size`: マイクロバッチ分割サイズ (OOM 防止)。
    ///
    /// # 戻り値
    /// 各質問 ID に対応する決定結果 (`IndexMap<String, Answer>`)。
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_batch_questions_with_cache(
        &self,
        tokenizer: &JevTokenizer,
        cached_state: &CachedStateFeatures,
        state_text: Option<&str>,
        questions: &indexmap::IndexMap<String, Question>,
        calib_config: &CalibrationConfig,
        gating_config: Option<&GatingConfig>,
        chunk_size: usize,
    ) -> Result<indexmap::IndexMap<String, Answer>> {
        if questions.is_empty() {
            return Ok(indexmap::IndexMap::new());
        }

        let state_seq_len = cached_state.seq_len();
        let single_sample_bytes =
            state_seq_len * cached_state.hidden_size() * std::mem::size_of::<f32>();

        let max_safe_chunk = if single_sample_bytes > 12 * 1024 * 1024 {
            2
        } else if single_sample_bytes > 4 * 1024 * 1024 {
            DEFAULT_DUAL_MAX_BATCH_CHUNK_SIZE
        } else {
            DEFAULT_DUAL_MAX_BATCH_CHUNK_SIZE * 2
        };

        let effective_chunk_size = chunk_size.max(1).min(max_safe_chunk);
        let mut answers = indexmap::IndexMap::with_capacity(questions.len());

        let mut tokenized_items = Vec::with_capacity(questions.len());
        for (qid, q) in questions {
            let t = tokenizer.encode_query(q)?;
            tokenized_items.push((qid, q, t));
        }

        let effective_state_text = cached_state.resolve_state_text(state_text);
        let mut scratch_probs = Vec::new();
        let mut reusable_state_val: Option<(usize, Arc<Value>)> = None;

        for chunk in tokenized_items.chunks(effective_chunk_size) {
            let chunk_len = chunk.len();
            let packed = tokenizer.pack_tokenized_questions(chunk.iter().map(|(_, _, t)| t))?;
            let batch_state_mask = cached_state.state_mask_for_batch(chunk_len);

            let state_val = match &reusable_state_val {
                Some((len, val)) if *len == chunk_len => Arc::clone(val),
                _ => {
                    let val = cached_state.as_ort_value_for_batch(chunk_len)?;
                    reusable_state_val = Some((chunk_len, Arc::clone(&val)));
                    val
                }
            };

            let logits_per_sample = self.evaluate_query_with_state_value(
                state_val.as_ref(),
                state_seq_len,
                &batch_state_mask,
                &packed.input_ids,
                &packed.attention_mask,
                &packed.op_indices,
                packed.dims,
            )?;

            for (i, (qid, q, t)) in chunk.iter().enumerate() {
                let k_i = t.op_indices.len();
                let sample_logits = &logits_per_sample[i][..k_i];

                if scratch_probs.len() < k_i {
                    scratch_probs.resize(k_i, 0.0);
                }

                let answer = evaluate_and_gate_answer_with_buf(
                    q,
                    sample_logits,
                    &mut scratch_probs[..k_i],
                    effective_state_text,
                    Some(qid.as_str()),
                    gating_config,
                    calib_config,
                )?;

                answers.insert((*qid).clone(), answer);
            }
        }

        Ok(answers)
    }

    /// 単一の質問クエリに対する各選択肢の生ロジット列 (`Vec<f64>`) を抽出する。
    ///
    /// # 引数
    /// - `cached_state`: エンコード済みの中間特徴量キャッシュ。
    /// - `tokenized`: トークナイズ済みの質問クエリ。
    ///
    /// # 戻り値
    /// 該当質問の各選択肢 (`[OP]`) に対応する生ロジット列。
    pub fn forward_query_raw(
        &self,
        cached_state: &CachedStateFeatures,
        tokenized: &TokenizedQuestion,
    ) -> Result<Vec<f64>> {
        let batch_state_mask = cached_state.state_mask_for_batch(1);
        let query_dims = ModelInputDimensions {
            batch_size: 1,
            sequence_length: tokenized.input_ids.len(),
            num_options: tokenized.op_indices.len(),
        };
        let mut logits = self.evaluate_query_with_cache(
            cached_state,
            &batch_state_mask,
            &tokenized.input_ids,
            &tokenized.attention_mask,
            &tokenized.op_indices,
            query_dims,
        )?;
        let sample_logits = logits.pop().unwrap_or_default();
        let k = tokenized.op_indices.len();
        Ok(sample_logits.into_iter().take(k).collect())
    }

    /// 単一の質問定義とキャッシュ済み `State` から各選択肢の未正規化生ロジット列 (`Vec<f64>`) を直接抽出する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `cached_state`: エンコード済みの中間特徴量キャッシュ。
    /// - `question`: 質問仕様。
    ///
    /// # 戻り値
    /// 各選択肢の未正規化生ロジット列。
    pub fn forward_question_raw(
        &self,
        tokenizer: &JevTokenizer,
        cached_state: &CachedStateFeatures,
        question: &Question,
    ) -> Result<Vec<f64>> {
        let tokenized = tokenizer.encode_query(question)?;
        self.forward_query_raw(cached_state, &tokenized)
    }

    /// 単一の質問をキャッシュ済み `State` に対して推論し、決定結果 (`Answer`) を返却する。
    ///
    /// # 引数
    /// - `tokenizer`: Jev 高速トークナイザー。
    /// - `cached_state`: エンコード済みの中間特徴量キャッシュ。
    /// - `state_text`: 任意で渡す文脈テキスト (ゲーティングメタデータ用)。
    /// - `question`: 質問仕様。
    /// - `calib_config`: 較正温度設定。
    ///
    /// # 戻り値
    /// 確率較正済みの `Answer`。
    pub fn predict_single_question(
        &self,
        tokenizer: &JevTokenizer,
        cached_state: &CachedStateFeatures,
        state_text: Option<&str>,
        question: &Question,
        calib_config: &CalibrationConfig,
    ) -> Result<Answer> {
        let tokenized = tokenizer.encode_query(question)?;
        let logits = self.forward_query_raw(cached_state, &tokenized)?;
        let effective_state_text = cached_state.resolve_state_text(state_text);
        evaluate_and_gate_answer(
            question,
            &logits,
            effective_state_text,
            None,
            None,
            calib_config,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_hash_formatting_and_parsing() {
        let raw_val: u64 = 0x0123_4567_89ab_cdef;
        let hash = StateHash::from_raw(raw_val);

        assert_eq!(hash.as_u64(), raw_val);
        assert_eq!(hash.into_inner(), raw_val);
        assert_eq!(StateHash::from(raw_val), hash);
        assert_eq!(u64::from(hash), raw_val);

        // Display (16桁ゼロ埋め小文字16進)
        assert_eq!(hash.to_string(), "0123456789abcdef");
        assert_eq!(format!("{hash}"), "0123456789abcdef");

        // LowerHex / UpperHex
        assert_eq!(format!("{hash:x}"), "123456789abcdef");
        assert_eq!(format!("{hash:016x}"), "0123456789abcdef");
        assert_eq!(format!("{hash:X}"), "123456789ABCDEF");
        assert_eq!(format!("{hash:016X}"), "0123456789ABCDEF");

        // FromStr パース
        let parsed: StateHash = "0123456789abcdef".parse().expect("パースに成功。");
        assert_eq!(parsed, hash);

        let parsed_with_prefix: StateHash = "0x0123456789abcdef"
            .parse()
            .expect("0xプレフィックス付きパースに成功。");
        assert_eq!(parsed_with_prefix, hash);

        let parsed_upper: StateHash = "0X0123456789ABCDEF"
            .parse()
            .expect("0Xプレフィックス付き大文字パースに成功。");
        assert_eq!(parsed_upper, hash);

        let parsed_trimmed: StateHash = "   0x0123456789abcdef  "
            .parse()
            .expect("トリム付きパースに成功。");
        assert_eq!(parsed_trimmed, hash);

        assert!("invalid_hex_string".parse::<StateHash>().is_err());
    }

    #[test]
    fn test_xxh3_key_computation_deterministic() {
        let text1 = "本契約において、甲は乙に対し対価を支払うものとする。";
        let text2 = "本契約において、甲は乙に対し対価を支払うものとする。";
        let text3 = "別のテキストです。";

        let key1 = StateCacheManager::compute_key(text1);
        let key2 = StateCacheManager::compute_key(text2);
        let key3 = StateCacheManager::compute_key(text3);

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
        assert_ne!(key1, StateHash(0));
    }

    #[test]
    fn test_compute_key_unicode_nfc_normalization() {
        // NFC 形式（結合済み文字）と NFD 形式（基底文字 + 結合用濁点）の等価性検証。
        let nfc_text = "がっこう";
        let nfd_text = "か\u{3099}っこう";

        assert_ne!(nfc_text.as_bytes(), nfd_text.as_bytes());

        let key_nfc = StateCacheManager::compute_key(nfc_text);
        let key_nfd = StateCacheManager::compute_key(nfd_text);

        assert_eq!(key_nfc, key_nfd);
        assert_ne!(key_nfc, StateHash(0));
    }

    #[test]
    fn test_aligned_buffer64_alignment() {
        let buf = AlignedBuffer64::<f32>::with_capacity_zeroed(128);
        assert_eq!(buf.len(), 128);
        assert!(buf.is_aligned_64());
        assert!(is_aligned_64(buf.as_ptr()));

        for &val in buf.iter() {
            assert_eq!(val, 0.0);
        }

        let slice = vec![1.5f32, 2.5, 3.5, 4.5];
        let buf_from_slice = AlignedBuffer64::from_slice(&slice);
        assert_eq!(buf_from_slice.len(), 4);
        assert!(buf_from_slice.is_aligned_64());
        assert_eq!(&buf_from_slice[..], &slice[..]);
    }

    #[test]
    fn test_aligned_buffer64_empty() {
        let mut buf = AlignedBuffer64::<f32>::with_capacity_zeroed(0);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        assert!(buf.is_aligned_64());
        assert!(is_aligned_64(buf.as_ptr()));
        assert!(is_aligned_64(buf.as_mut_ptr()));
        assert_eq!(&buf[..], &[] as &[f32]);

        let empty_slice: [f32; 0] = [];
        let buf_from_empty = AlignedBuffer64::from_slice(&empty_slice);
        assert_eq!(buf_from_empty.len(), 0);
        assert!(buf_from_empty.is_empty());
        assert!(buf_from_empty.is_aligned_64());
        assert!(is_aligned_64(buf_from_empty.as_ptr()));

        // 64バイト超のアライメント要件を持つ型でも、型のアライメントと 64B 境界の双方を満たすこと
        #[repr(align(128))]
        #[derive(Clone, Copy)]
        struct OverAligned([u8; 128]);
        let over_buf = AlignedBuffer64::<OverAligned>::with_capacity_zeroed(0);
        assert!(over_buf.is_aligned_64());
        assert_eq!((over_buf.as_ptr() as usize) % 128, 0);
        let over_buf_nonempty = AlignedBuffer64::<OverAligned>::with_capacity_zeroed(1);
        assert!(over_buf_nonempty.is_aligned_64());
        assert_eq!((over_buf_nonempty.as_ptr() as usize) % 128, 0);
        assert_eq!(over_buf_nonempty[0].0[0], 0);
    }

    #[test]
    #[should_panic(expected = "ZST はサポートされていません。")]
    fn test_aligned_buffer64_zst_panic() {
        let _ = AlignedBuffer64::<()>::with_capacity_zeroed(1);
    }

    #[test]
    fn test_aligned_buffer64_slice_traits() {
        fn sum_as_ref(buf: impl AsRef<[f32]>) -> f32 {
            buf.as_ref().iter().sum()
        }
        fn scale_as_mut(mut buf: impl AsMut<[f32]>, factor: f32) {
            for v in buf.as_mut() {
                *v *= factor;
            }
        }
        fn sum_borrow(buf: &impl Borrow<[f32]>) -> f32 {
            buf.borrow().iter().sum()
        }
        fn add_borrow_mut(buf: &mut impl BorrowMut<[f32]>, delta: f32) {
            for v in buf.borrow_mut() {
                *v += delta;
            }
        }

        let mut buf = AlignedBuffer64::from_slice(&[1.0f32, 2.0, 3.0]);
        assert!((sum_as_ref(&buf) - 6.0).abs() < 1e-6);
        assert!((sum_borrow(&buf) - 6.0).abs() < 1e-6);

        scale_as_mut(&mut buf, 2.0);
        assert_eq!(buf.as_ref(), &[2.0, 4.0, 6.0]);

        add_borrow_mut(&mut buf, 1.0);
        assert_eq!(
            <AlignedBuffer64<f32> as Borrow<[f32]>>::borrow(&buf),
            &[3.0, 5.0, 7.0]
        );
    }

    #[test]
    fn test_fp16_compression_roundtrip_precision() {
        let original: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.005 - 2.5).collect();
        let compressed = compress_f32_to_fp16(&original);
        assert_eq!(compressed.len(), original.len());

        let mut restored = vec![0.0f32; original.len()];
        decompress_fp16_to_f32(&compressed, &mut restored);

        let mut max_err = 0.0f32;
        for (&orig, &rest) in original.iter().zip(restored.iter()) {
            let err = (orig - rest).abs();
            if err > max_err {
                max_err = err;
            }
        }

        // `FP16` の最大絶対誤差は 2e-3 未満（正規化範囲内で高精度）。
        assert!(max_err < 2e-3, "FP16 誤差が大きすぎます: {max_err}");
    }

    #[test]
    fn test_int8_symmetric_compression_roundtrip() {
        let hidden_size = 64;
        let num_tokens = 4;
        let mut original = Vec::with_capacity(num_tokens * hidden_size);

        for t in 0..num_tokens {
            let scale = (t + 1) as f32 * 0.8;
            for i in 0..hidden_size {
                original.push(((i as f32) / (hidden_size as f32) - 0.5) * scale);
            }
        }

        let (quantized, scales) = compress_f32_to_int8_symmetric(&original, hidden_size);
        assert_eq!(quantized.len(), original.len());
        assert_eq!(scales.len(), num_tokens);

        let mut restored = vec![0.0f32; original.len()];
        decompress_int8_symmetric_to_f32(&quantized, &scales, hidden_size, &mut restored);

        let mut max_err = 0.0f32;
        for (&orig, &rest) in original.iter().zip(restored.iter()) {
            let err = (orig - rest).abs();
            if err > max_err {
                max_err = err;
            }
        }

        // `INT8` 対称量子化の量子化ステップ幅は `scale / 127`。最大誤差が 0.05 未満であることを確認。
        assert!(max_err < 0.05, "INT8 誤差が大きすぎます: {max_err}");
    }

    #[test]
    fn test_state_cache_manager_crud_and_weigher() {
        let manager = StateCacheManager::new(
            1024 * 1024, // 1MB
            60,
            CacheCompressionMode::Fp16,
        );

        let text = "特許権侵害訴訟における証拠保全手続きの概要。";
        let key = StateCacheManager::compute_key(text);
        assert!(!manager.contains_key(key));
        assert!(manager.get(key).is_none());

        let shape = vec![1, 8, 16];
        let raw_data = vec![0.1f32; 8 * 16];
        let state_mask = vec![1i64; 8];
        let features = Arc::new(
            CachedStateFeatures::new_compressed(
                key,
                shape,
                &raw_data,
                CacheCompressionMode::Fp16,
                state_mask,
            )
            .unwrap(),
        );

        manager.insert(key, Arc::clone(&features));
        assert!(manager.contains_key(key));
        manager.run_pending_tasks();

        let retrieved = manager.get(key).expect("キャッシュヒットするはずです。");
        assert_eq!(retrieved.hash_key, key);
        assert_eq!(retrieved.seq_len(), 8);
        assert_eq!(retrieved.hidden_size(), 16);
        assert_eq!(retrieved.state_mask, vec![1i64; 8]);
        assert!(manager.weighted_size() > 0);

        manager.invalidate(key);
        manager.run_pending_tasks();
        assert!(!manager.contains_key(key));
    }

    #[test]
    fn test_as_ort_value_for_batch_repeat() {
        let text = "バッチリピートテスト用のStateテキスト。";
        let key = StateCacheManager::compute_key(text);
        let seq_len = 4;
        let hidden_size = 8;
        let shape = vec![1, seq_len, hidden_size];
        let raw_data: Vec<f32> = (0..seq_len * hidden_size).map(|i| i as f32).collect();
        // パディングを含むマスク列 (先頭3トークン有効、4トークン目パディング)
        let custom_state_mask = vec![1i64, 1, 1, 0];

        let features = CachedStateFeatures::new_compressed(
            key,
            shape,
            &raw_data,
            CacheCompressionMode::Fp16,
            custom_state_mask.clone(),
        )
        .expect("キャッシュエントリの作成に成功すること。");

        // 単一バッチ時は [1, seq_len, hidden_size]
        let val_b1 = features
            .as_ort_value_for_batch(1)
            .expect("単一バッチ取得成功。");
        let (s1, d1) = val_b1
            .try_extract_tensor::<f32>()
            .expect("テンソル抽出成功。");
        assert_eq!(s1.as_ref(), &[1, seq_len as i64, hidden_size as i64]);
        assert_eq!(d1.len(), seq_len * hidden_size);

        // 複数バッチ (例: batch=3) 時は [3, seq_len, hidden_size] へリピート展開
        let val_b3 = features
            .as_ort_value_for_batch(3)
            .expect("3バッチ取得成功。");
        let (s3, d3) = val_b3
            .try_extract_tensor::<f32>()
            .expect("テンソル抽出成功。");
        assert_eq!(s3.as_ref(), &[3, seq_len as i64, hidden_size as i64]);
        assert_eq!(d3.len(), 3 * seq_len * hidden_size);

        // 1番目と2番目のスライスが元のデータと一致しているか検証
        for b in 0..3 {
            let offset = b * seq_len * hidden_size;
            let slice = &d3[offset..offset + seq_len * hidden_size];
            for (v_orig, v_rep) in d1.iter().zip(slice.iter()) {
                assert!((v_orig - v_rep).abs() < 1e-3);
            }
        }

        // state_mask_for_batch のリピート展開とパディング維持の検証
        let mask_b3 = features.state_mask_for_batch(3);
        assert_eq!(mask_b3.len(), 3 * seq_len);
        assert_eq!(mask_b3, vec![1i64, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0]);

        // 非圧縮 (Raw) キャッシュ時の中間アロケーション回避とリピート展開の検証
        let raw_shape = vec![1, seq_len, hidden_size];
        let raw_val = Value::from_array((raw_shape.clone(), raw_data.clone()))
            .expect("Value の作成に成功すること。");
        let raw_features = CachedStateFeatures::new_raw(
            key,
            raw_shape,
            raw_val.into(),
            raw_data.len() * std::mem::size_of::<f32>(),
            custom_state_mask,
        )
        .expect("Raw キャッシュエントリの作成に成功すること。");

        let raw_b2 = raw_features
            .as_ort_value_for_batch(2)
            .expect("Raw バッチ取得成功。");
        let (s_raw, d_raw) = raw_b2
            .try_extract_tensor::<f32>()
            .expect("Raw テンソル抽出成功。");
        assert_eq!(s_raw.as_ref(), &[2, seq_len as i64, hidden_size as i64]);
        assert_eq!(d_raw.len(), 2 * seq_len * hidden_size);
        assert_eq!(&d_raw[..seq_len * hidden_size], &raw_data[..]);
        assert_eq!(&d_raw[seq_len * hidden_size..], &raw_data[..]);

        // バッチサイズ 0 の不正指定時にエラーとなるゼロガードを検証。
        assert!(features.as_ort_value_for_batch(0).is_err());
    }

    #[test]
    fn test_as_ort_value_for_batch_rejects_multi_batch_and_invalid_shapes_without_panic() {
        let key = StateCacheManager::compute_key("multi-batch-guard-test");
        let seq_len = 4;
        let hidden_size = 8;
        let batch_size = 2;
        let shape = vec![batch_size, seq_len, hidden_size];
        let raw_data: Vec<f32> = (0..batch_size * seq_len * hidden_size)
            .map(|i| i as f32)
            .collect();
        let mask = vec![1i64; batch_size * seq_len];

        // 1. コンストラクタ時点で batch_size != 1 の不正エントリが ModelContractViolation で拒絶されることの検証。
        for mode in [
            CacheCompressionMode::None,
            CacheCompressionMode::Fp16,
            CacheCompressionMode::Int8Symmetric,
        ] {
            let err = CachedStateFeatures::new_compressed(
                key,
                shape.clone(),
                &raw_data,
                mode,
                mask.clone(),
            )
            .expect_err("batch_size != 1 のエントリ構築は拒絶される必要があります。");
            assert!(
                matches!(err, RuntimeError::ModelContractViolation(_)),
                "ModelContractViolation が返る必要があります: mode={mode:?}, err={err:?}"
            );
        }

        // batch_size == 0 も ModelContractViolation を返すこと。
        let zero_batch_val =
            Value::from_array((vec![0usize, seq_len, hidden_size], Vec::<f32>::new()))
                .expect("空バッチ Value の作成に成功すること。");
        let zero_batch_err = CachedStateFeatures::new_raw(
            key,
            vec![0, seq_len, hidden_size],
            zero_batch_val.into(),
            0,
            vec![],
        )
        .expect_err("batch_size == 0 のキャッシュエントリ作成は拒絶される必要があります。");
        assert!(matches!(
            zero_batch_err,
            RuntimeError::ModelContractViolation(_)
        ));

        // 3次元以外の形状 (空形状・2次元) も ModelContractViolation を返すこと。
        let empty_shape_val = Value::from_array((
            vec![1usize, seq_len, hidden_size],
            vec![0.0f32; seq_len * hidden_size],
        ))
        .expect("Value の作成に成功すること。");
        let empty_shape_err =
            CachedStateFeatures::new_raw(key, vec![], empty_shape_val.into(), 0, vec![])
                .expect_err("空形状キャッシュエントリの作成は拒絶される必要があります。");
        assert!(matches!(
            empty_shape_err,
            RuntimeError::ModelContractViolation(_)
        ));

        // 2. 万一の不正内部状態に対する as_ort_value_for_batch の多層防御ガードの検証。
        let dummy_val = Value::from_array((
            vec![2usize, seq_len, hidden_size],
            vec![0.0f32; 2 * seq_len * hidden_size],
        ))
        .expect("Value の作成に成功すること。");
        let corrupt_features = CachedStateFeatures {
            hash_key: key,
            shape: vec![2, seq_len, hidden_size],
            payload: CachedTensorPayload::Raw(Arc::new(dummy_val.into())),
            byte_size: 2 * seq_len * hidden_size * std::mem::size_of::<f32>(),
            state_mask: vec![1i64; 2 * seq_len],
            state_text: None,
        };
        assert!(matches!(
            corrupt_features.as_ort_value_for_batch(1),
            Err(RuntimeError::ModelContractViolation(_))
        ));

        // 次元長 0 (seq_len == 0 / hidden_size == 0) は chunks_exact_mut(0) でパニックせず InvalidTensorData を返すこと。
        let zero_seq_val = Value::from_array((vec![1usize, 0, hidden_size], Vec::<f32>::new()))
            .expect("Value の作成に成功すること。");
        let zero_seq_features = CachedStateFeatures::new_raw(
            key,
            vec![1, 0, hidden_size],
            zero_seq_val.into(),
            0,
            vec![],
        )
        .expect("ゼロ系列長キャッシュエントリの作成に成功すること。");
        assert!(matches!(
            zero_seq_features.as_ort_value_for_batch(2),
            Err(RuntimeError::InvalidTensorData(_))
        ));

        // shape と実ペイロード長が不一致の場合も copy_from_slice でパニックせず InvalidTensorData を返すこと。
        let mismatched_val =
            Value::from_array((vec![1usize, 2, hidden_size], vec![1.0f32; 2 * hidden_size]))
                .expect("Value の作成に成功すること。");
        let mismatched_features = CachedStateFeatures::new_raw(
            key,
            vec![1, seq_len, hidden_size],
            mismatched_val.into(),
            2 * hidden_size * std::mem::size_of::<f32>(),
            vec![1; seq_len],
        )
        .expect("データ長不一致キャッシュエントリの作成に成功すること。");
        assert!(matches!(
            mismatched_features.as_ort_value_for_batch(2),
            Err(RuntimeError::InvalidTensorData(_))
        ));
    }

    #[test]
    fn test_cached_state_features_validates_state_mask_length() {
        let key = StateCacheManager::compute_key("state-mask-len-validation-test");
        let seq_len = 4;
        let hidden_size = 8;
        let shape = vec![1, seq_len, hidden_size];
        let raw_data: Vec<f32> = vec![0.5f32; seq_len * hidden_size];

        // 1. 要素数不足の state_mask 指定時に InvalidTensorData が返ること。
        let short_mask = vec![1i64; seq_len - 1];
        let val1 = Value::from_array((shape.clone(), raw_data.clone()))
            .expect("Value の作成に成功すること。");
        let raw_short_err = CachedStateFeatures::new_raw(
            key,
            shape.clone(),
            val1.into(),
            raw_data.len() * std::mem::size_of::<f32>(),
            short_mask.clone(),
        );
        assert!(matches!(
            raw_short_err,
            Err(RuntimeError::InvalidTensorData(_))
        ));

        let comp_short_err = CachedStateFeatures::new_compressed(
            key,
            shape.clone(),
            &raw_data,
            CacheCompressionMode::Fp16,
            short_mask,
        );
        assert!(matches!(
            comp_short_err,
            Err(RuntimeError::InvalidTensorData(_))
        ));

        // 2. 要素数超過の state_mask 指定時に InvalidTensorData が返ること。
        let long_mask = vec![1i64; seq_len + 1];
        let val2 = Value::from_array((shape.clone(), raw_data.clone()))
            .expect("Value の作成に成功すること。");
        let raw_long_err = CachedStateFeatures::new_raw(
            key,
            shape.clone(),
            val2.into(),
            raw_data.len() * std::mem::size_of::<f32>(),
            long_mask.clone(),
        );
        assert!(matches!(
            raw_long_err,
            Err(RuntimeError::InvalidTensorData(_))
        ));

        let comp_long_err = CachedStateFeatures::new_compressed(
            key,
            shape.clone(),
            &raw_data,
            CacheCompressionMode::Int8Symmetric,
            long_mask,
        );
        assert!(matches!(
            comp_long_err,
            Err(RuntimeError::InvalidTensorData(_))
        ));

        // 3. 空の state_mask 指定時は全トークン有効 (1i64) で自動補完されること。
        let val3 = Value::from_array((shape.clone(), raw_data.clone()))
            .expect("Value の作成に成功すること。");
        let raw_empty_mask = CachedStateFeatures::new_raw(
            key,
            shape.clone(),
            val3.into(),
            raw_data.len() * std::mem::size_of::<f32>(),
            vec![],
        )
        .expect("空マスク時は自動補完されて成功すること。");
        assert_eq!(raw_empty_mask.state_mask, vec![1i64; seq_len]);

        let comp_empty_mask = CachedStateFeatures::new_compressed(
            key,
            shape.clone(),
            &raw_data,
            CacheCompressionMode::None,
            vec![],
        )
        .expect("空マスク時は自動補完されて成功すること。");
        assert_eq!(comp_empty_mask.state_mask, vec![1i64; seq_len]);

        // 4. 正しい要素数のマスク(パディング 0 含む)がそのまま保持されること。
        let valid_mask = vec![1i64, 1, 0, 0];
        let val4 = Value::from_array((shape.clone(), raw_data.clone()))
            .expect("Value の作成に成功すること。");
        let raw_valid = CachedStateFeatures::new_raw(
            key,
            shape.clone(),
            val4.into(),
            raw_data.len() * std::mem::size_of::<f32>(),
            valid_mask.clone(),
        )
        .expect("正常マスク時は成功すること。");
        assert_eq!(raw_valid.state_mask, valid_mask);
        assert_eq!(
            raw_valid.state_mask_for_batch(2),
            vec![1, 1, 0, 0, 1, 1, 0, 0]
        );
    }

    #[test]
    fn test_cached_state_features_validates_raw_slice_length() {
        let key = StateCacheManager::compute_key("raw-slice-len-validation-test");
        let seq_len = 4;
        let hidden_size = 8;
        let shape = vec![1, seq_len, hidden_size];
        let total_elements = seq_len * hidden_size;

        // 1. スライス長不足時に各モードでパニックせず InvalidTensorData が返ること。
        let short_data: Vec<f32> = vec![0.5f32; total_elements - 1];
        for mode in [
            CacheCompressionMode::None,
            CacheCompressionMode::Fp16,
            CacheCompressionMode::Int8Symmetric,
        ] {
            let res =
                CachedStateFeatures::new_compressed(key, shape.clone(), &short_data, mode, vec![]);
            assert!(
                matches!(res, Err(RuntimeError::InvalidTensorData(_))),
                "{mode:?} モードでスライス長不足時に InvalidTensorData が返る必要があります。"
            );
        }

        // 2. スライス長超過時に各モードでパニックせず InvalidTensorData が返ること。
        let long_data: Vec<f32> = vec![0.5f32; total_elements + 1];
        for mode in [
            CacheCompressionMode::None,
            CacheCompressionMode::Fp16,
            CacheCompressionMode::Int8Symmetric,
        ] {
            let res =
                CachedStateFeatures::new_compressed(key, shape.clone(), &long_data, mode, vec![]);
            assert!(
                matches!(res, Err(RuntimeError::InvalidTensorData(_))),
                "{mode:?} モードでスライス長超過時に InvalidTensorData が返る必要があります。"
            );
        }

        // 3. shape が 3 次元でない場合は ModelContractViolation が返ること。
        let valid_data: Vec<f32> = vec![0.5f32; total_elements];
        let invalid_shapes = vec![
            vec![seq_len, hidden_size],
            vec![1, 1, seq_len, hidden_size],
            vec![],
        ];
        for invalid_shape in invalid_shapes {
            let res = CachedStateFeatures::new_compressed(
                key,
                invalid_shape,
                &valid_data,
                CacheCompressionMode::None,
                vec![],
            );
            assert!(
                matches!(res, Err(RuntimeError::ModelContractViolation(_))),
                "3次元以外の形状では ModelContractViolation が返る必要があります。"
            );
        }

        // 4. Int8Symmetric で hidden_size == 0 の場合は InvalidTensorData が返ること。
        let zero_hidden_shape = vec![1, 4, 0];
        let zero_data: Vec<f32> = vec![];
        let res = CachedStateFeatures::new_compressed(
            key,
            zero_hidden_shape,
            &zero_data,
            CacheCompressionMode::Int8Symmetric,
            vec![],
        );
        assert!(matches!(res, Err(RuntimeError::InvalidTensorData(_))));
    }
}
