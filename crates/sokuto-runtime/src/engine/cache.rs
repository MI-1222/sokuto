//! # 中間特徴量キャッシュおよびゼロコピー連携モジュール
//!
//! 長文 `State` の中間特徴量キャッシュ、
//! `xxHash3` 高速キー生成、`moka` 並行キャッシュストレージ、
//! `ort` の `IoBinding` によるゼロコピーバインド、
//! L3キャッシュ局所性最適化（`FP16`/`INT8` 圧縮）、
//! 64バイトメモリアライメント、および `Tokio` ワーカースレッド競合隔離を提供する。

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
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
use sokuto_core::decision::evaluate_question_with_buf;
use sokuto_core::gating::GatingConfig;
use sokuto_core::schema::{Answer, Question};
use xxhash_rust::xxh3::xxh3_64;

use crate::engine::config::{ExecutionProvider, SessionConfig};
use crate::engine::gating::{apply_gating_to_answer_with_energy, calculate_choice_energy};
use crate::engine::session::create_session_builder;
use crate::error::{Result, RuntimeError};
use crate::tokenizer::JevTokenizer;

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
        if len == 0 {
            let layout = Layout::from_size_align(0, 64).unwrap_or(Layout::new::<u8>());
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                capacity: 0,
                layout,
            };
        }

        let size = len
            .checked_mul(std::mem::size_of::<T>())
            .expect("メモリ確保サイズがオーバーフローしました。");
        let layout = Layout::from_size_align(size, 64)
            .expect("64バイトアライメントレイアウトの作成に失敗しました。");

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
        let buf = Self::with_capacity_zeroed(slice.len());
        if !slice.is_empty() {
            // SAFETY: `buf` の長さは `slice.len()` と完全に一致しており、重なり合わない有効なポインタである。
            unsafe {
                std::ptr::copy_nonoverlapping(slice.as_ptr(), buf.ptr.as_ptr(), slice.len());
            }
        }
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
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: 有効な割り当て領域と要素数を保持している。
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }
}

impl<T: Copy> DerefMut for AlignedBuffer64<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.len == 0 {
            &mut []
        } else {
            // SAFETY: 排他的借用下で有効な割り当て領域と要素数を保持している。
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
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
    assert_eq!(src.len(), dst.len(), "展開元と先の要素数が一致しません。");
    assert_eq!(
        src.len() / hidden_size,
        scales.len(),
        "トークン数とスケール数が一致しません。"
    );

    for (token_idx, (q_chunk, &scale)) in
        src.chunks_exact(hidden_size).zip(scales.iter()).enumerate()
    {
        let offset = token_idx * hidden_size;
        for (i, &q) in q_chunk.iter().enumerate() {
            dst[offset + i] = (q as f32) * scale;
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

/// キャッシュに格納される `State` 中間特徴量エントリ。
#[derive(Debug)]
pub struct CachedStateFeatures {
    /// `xxHash3` (64-bit) によるキー。
    pub hash_key: u64,
    /// テンソル形状 (`[batch_size, seq_len, hidden_size]`)。
    pub shape: Vec<usize>,
    /// テンソル実体ペイロード。
    pub payload: CachedTensorPayload,
    /// テンソルが占有する実メモリバイト数（`moka` の `weigher` に使用）。
    pub byte_size: usize,
    /// 採用された圧縮モード。
    pub compression_mode: CacheCompressionMode,
    /// `State` の有効トークンアテンションマスク列 (`[batch_size, seq_len]`)。
    pub state_mask: Vec<i64>,
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
    pub fn new_raw(
        hash_key: u64,
        shape: Vec<usize>,
        ort_value: Value,
        byte_size: usize,
        state_mask: Vec<i64>,
    ) -> Self {
        let total_tokens = if shape.len() >= 2 {
            shape[0] * shape[1]
        } else {
            0
        };
        let mut state_mask = state_mask;
        if state_mask.is_empty() && total_tokens > 0 {
            state_mask.resize(total_tokens, 1i64);
        }

        Self {
            hash_key,
            shape,
            payload: CachedTensorPayload::Raw(Arc::new(ort_value)),
            byte_size,
            compression_mode: CacheCompressionMode::None,
            state_mask,
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
    pub fn new_compressed(
        hash_key: u64,
        shape: Vec<usize>,
        raw_slice: &[f32],
        mode: CacheCompressionMode,
        state_mask: Vec<i64>,
    ) -> Result<Self> {
        let hidden_size = if shape.len() >= 3 {
            shape[2]
        } else {
            return Err(RuntimeError::ModelContractViolation(format!(
                "State特徴量は3次元テンソルである必要があります: {shape:?}。"
            )));
        };

        let total_tokens = if shape.len() >= 2 {
            shape[0] * shape[1]
        } else {
            0
        };
        let mut state_mask = state_mask;
        if state_mask.is_empty() && total_tokens > 0 {
            state_mask.resize(total_tokens, 1i64);
        }

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
                    compression_mode: mode,
                    state_mask,
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
                    compression_mode: mode,
                    state_mask,
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
                    compression_mode: mode,
                    state_mask,
                })
            }
        }
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
    /// `target_batch == 1` かつ自身のバッチサイズが 1 の場合はゼロコピー返却する。
    /// `target_batch > 1` の場合は、単一サンプルの特徴量データを `target_batch` 回リピートした
    /// テンソル (`[target_batch, seq_len, hidden_size]`) を ONNX Runtime アロケータメモリへ直接構築して返却する。
    /// 中間ベクタのヒープ確保と再アロケーションを回避し、64バイトアライメントを維持する。
    pub fn as_ort_value_for_batch(&self, target_batch: usize) -> Result<Arc<Value>> {
        if target_batch == 0 {
            return Err(RuntimeError::InvalidTensorData(
                "target_batch は 1 以上である必要があります。".to_string(),
            ));
        }

        if target_batch == 1 && self.batch_size() == 1 {
            return self.as_ort_value();
        }

        let seq_len = self.seq_len();
        let hidden_size = self.hidden_size();
        let single_len = seq_len * hidden_size;
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
                for chunk in dst.chunks_exact_mut(single_len) {
                    chunk.copy_from_slice(single_slice);
                }
            }
            CachedTensorPayload::Fp16(compressed) => {
                let (first_chunk, rest) = dst.split_at_mut(single_len);
                decompress_fp16_to_f32(compressed, first_chunk);
                for chunk in rest.chunks_exact_mut(single_len) {
                    chunk.copy_from_slice(first_chunk);
                }
            }
            CachedTensorPayload::Int8Symmetric { quantized, scales } => {
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
    cache: Cache<u64, Arc<CachedStateFeatures>>,
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
    /// `AVX2`、`AVX-512`、`ARM NEON` 命令により、8kトークンのテキストでも1μs未満で完了する。
    ///
    /// # 引数
    /// - `state_text`: `State` 原文テキスト。
    pub fn compute_key(state_text: &str) -> u64 {
        xxh3_64(state_text.as_bytes())
    }

    /// 指定されたキーの特徴量を取得する。
    ///
    /// # 引数
    /// - `key`: `xxHash3` キー。
    pub fn get(&self, key: u64) -> Option<Arc<CachedStateFeatures>> {
        self.cache.get(&key)
    }

    /// 特徴量をキャッシュに登録する。
    ///
    /// # 引数
    /// - `key`: `xxHash3` キー。
    /// - `features`: キャッシュエントリ。
    pub fn insert(&self, key: u64, features: Arc<CachedStateFeatures>) {
        self.cache.insert(key, features);
    }

    /// 指定されたキーが存在するかを判定する。
    pub fn contains_key(&self, key: u64) -> bool {
        self.cache.contains_key(&key)
    }

    /// 指定されたキーのエントリを無効化（削除）する。
    pub fn invalidate(&self, key: u64) {
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
/// 内部にセッションプール（`query_sessions`、`state_sessions`）を保持し、
/// 複数スレッドからの並行評価要求に対してロック競合を回避し、並列推論を実行する。
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
        let pool_size = config.pool_size.max(1);
        let mut state_sessions = Vec::with_capacity(pool_size);
        let mut query_sessions = Vec::with_capacity(pool_size);
        let mut active_provider = ExecutionProvider::CPU;

        for i in 0..pool_size {
            let (mut state_builder, provider) = create_session_builder(&config, pool_size)?;
            if i == 0 {
                active_provider = provider;
            }
            let s_session = state_builder
                .commit_from_file(state_model_path.as_ref())
                .map_err(RuntimeError::Ort)?;
            state_sessions.push(Mutex::new(s_session));

            let (mut query_builder, _) = create_session_builder(&config, pool_size)?;
            let q_session = query_builder
                .commit_from_file(query_model_path.as_ref())
                .map_err(RuntimeError::Ort)?;
            query_sessions.push(Mutex::new(q_session));
        }

        Ok(Self {
            state_sessions,
            query_sessions,
            next_state_idx: AtomicUsize::new(0),
            next_query_idx: AtomicUsize::new(0),
            cache_manager,
            active_provider,
        })
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
        let pool_size = config.pool_size.max(1);
        let mut state_sessions = Vec::with_capacity(pool_size);
        let mut query_sessions = Vec::with_capacity(pool_size);
        let mut active_provider = ExecutionProvider::CPU;

        for i in 0..pool_size {
            let (mut state_builder, provider) = create_session_builder(&config, pool_size)?;
            if i == 0 {
                active_provider = provider;
            }
            let s_session = state_builder
                .commit_from_memory(state_model_bytes)
                .map_err(RuntimeError::Ort)?;
            state_sessions.push(Mutex::new(s_session));

            let (mut query_builder, _) = create_session_builder(&config, pool_size)?;
            let q_session = query_builder
                .commit_from_memory(query_model_bytes)
                .map_err(RuntimeError::Ort)?;
            query_sessions.push(Mutex::new(q_session));
        }

        Ok(Self {
            state_sessions,
            query_sessions,
            next_state_idx: AtomicUsize::new(0),
            next_query_idx: AtomicUsize::new(0),
            cache_manager,
            active_provider,
        })
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
        let pool_len = self.query_sessions.len();
        let start_idx = self.next_query_idx.fetch_add(1, Ordering::Relaxed) % pool_len;
        for offset in 0..pool_len {
            let idx = (start_idx + offset) % pool_len;
            if let Ok(guard) = self.query_sessions[idx].try_lock() {
                return Ok(guard);
            }
        }
        self.query_sessions[start_idx].lock().map_err(|e| {
            RuntimeError::ModelContractViolation(format!("Query セッションロック獲得失敗: {e}。"))
        })
    }

    /// セッションプールから利用可能な `State` セッションを取得する。
    ///
    /// ロックされていない空きセッションを優先探索し、全セッションが使用中の場合はブロック待機する。
    fn acquire_state_session(&self) -> Result<std::sync::MutexGuard<'_, Session>> {
        let pool_len = self.state_sessions.len();
        let start_idx = self.next_state_idx.fetch_add(1, Ordering::Relaxed) % pool_len;
        for offset in 0..pool_len {
            let idx = (start_idx + offset) % pool_len;
            if let Ok(guard) = self.state_sessions[idx].try_lock() {
                return Ok(guard);
            }
        }
        self.state_sessions[start_idx].lock().map_err(|e| {
            RuntimeError::ModelContractViolation(format!("State セッションロック獲得失敗: {e}。"))
        })
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

        let compression = self.cache_manager.compression_mode();
        let cached_entry = if compression == CacheCompressionMode::None {
            let total_elements: usize = shape_vec.iter().product();
            let byte_size = total_elements * std::mem::size_of::<f32>();
            Arc::new(CachedStateFeatures::new_raw(
                key,
                shape_vec,
                state_val,
                byte_size,
                attention_mask.to_vec(),
            ))
        } else {
            Arc::new(CachedStateFeatures::new_compressed(
                key,
                shape_vec,
                flat_data,
                compression,
                attention_mask.to_vec(),
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

        // 2. `op_indices` の範囲外検証（クエリ系列長基準、パディング-1は許容）。
        for (i, &idx) in op_indices.iter().enumerate() {
            if idx != -1 && (idx < 0 || idx >= query_seq_len as i64) {
                return Err(RuntimeError::InvalidTensorData(format!(
                    "op_indices[{i}] = {idx} が有効範囲 (-1 または 0..{query_seq_len}) の範囲外です。"
                )));
            }
        }

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
        let pad_id = tokenizer.pad_token_id().unwrap_or(0) as i64;

        let mut tokenized_items = Vec::with_capacity(questions.len());
        for (qid, q) in questions {
            let t = tokenizer.encode_query(q)?;
            tokenized_items.push((qid, q, t));
        }

        let mut scratch_probs = Vec::new();
        let mut reusable_state_val: Option<(usize, Arc<Value>)> = None;

        for chunk in tokenized_items.chunks(effective_chunk_size) {
            let chunk_len = chunk.len();
            let max_q_len = chunk
                .iter()
                .map(|(_, _, t)| t.input_ids.len())
                .max()
                .unwrap_or(1);
            let max_opts = chunk
                .iter()
                .map(|(_, _, t)| t.op_indices.len())
                .max()
                .unwrap_or(1);

            let mut batch_query_ids = vec![pad_id; chunk_len * max_q_len];
            let mut batch_query_mask = vec![0i64; chunk_len * max_q_len];
            let mut batch_op_indices = vec![-1i64; chunk_len * max_opts];
            let batch_state_mask = cached_state.state_mask_for_batch(chunk_len);

            for (i, (_, _, t)) in chunk.iter().enumerate() {
                let q_len = t.input_ids.len();
                let q_start = i * max_q_len;
                batch_query_ids[q_start..q_start + q_len].copy_from_slice(&t.input_ids);
                batch_query_mask[q_start..q_start + q_len].copy_from_slice(&t.attention_mask);

                let opt_len = t.op_indices.len();
                let opt_start = i * max_opts;
                batch_op_indices[opt_start..opt_start + opt_len].copy_from_slice(&t.op_indices);
            }

            let query_dims = ModelInputDimensions {
                batch_size: chunk_len,
                sequence_length: max_q_len,
                num_options: max_opts,
            };

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
                &batch_query_ids,
                &batch_query_mask,
                &batch_op_indices,
                query_dims,
            )?;

            for (i, (qid, q, t)) in chunk.iter().enumerate() {
                let k_i = t.op_indices.len();
                let sample_logits = &logits_per_sample[i][..k_i];

                if scratch_probs.len() < k_i {
                    scratch_probs.resize(k_i, 0.0);
                }

                let mut answer = evaluate_question_with_buf(
                    q,
                    sample_logits,
                    calib_config,
                    &mut scratch_probs[..k_i],
                )?;

                let energy = calculate_choice_energy(q, sample_logits, calib_config);
                apply_gating_to_answer_with_energy(
                    &mut answer,
                    energy,
                    state_text,
                    Some(qid.as_str()),
                    Some(q),
                    gating_config,
                    calib_config,
                );

                answers.insert((*qid).clone(), answer);
            }
        }

        Ok(answers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_ne!(key1, 0);
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
        );

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
}
