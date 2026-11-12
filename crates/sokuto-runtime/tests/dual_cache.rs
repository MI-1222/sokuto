//! # 中間特徴量キャッシュおよびデュアルセッション統合テスト
//!
//! `xxHash3` 高速キー計算、`moka` 並行キャッシュ管理、
//! L3キャッシュ局所性最適化（`FP16`/`INT8` 量子化圧縮）、
//! 64バイトメモリアライメント、および動的軸の Fast-Fail 検証を実施する。

use std::sync::Arc;

use sokuto_runtime::engine::{
    AlignedBuffer64, CacheCompressionMode, CachedStateFeatures, DualSessionEngine, SessionConfig,
    StateCacheManager, is_aligned_64,
};

#[test]
fn test_cached_state_features_dimensions() {
    let key = 0xABCD_1234;
    let shape = vec![1, 32, 128];
    let data = vec![0.0f32; 32 * 128];
    let mask = vec![1i64; 32];
    let entry = CachedStateFeatures::new_compressed(
        key,
        shape,
        &data,
        CacheCompressionMode::None,
        mask.clone(),
    )
    .expect("キャッシュエントリ作成に失敗しました。");

    assert_eq!(entry.batch_size(), 1);
    assert_eq!(entry.seq_len(), 32);
    assert_eq!(entry.hidden_size(), 128);
    assert_eq!(entry.compression_mode(), CacheCompressionMode::None);
    assert_eq!(entry.state_mask, mask);

    // batch_size != 1 の場合は ModelContractViolation で拒絶されることの検証。
    let invalid_shape = vec![2, 32, 128];
    let invalid_data = vec![0.0f32; 2 * 32 * 128];
    let invalid_mask = vec![1i64; 2 * 32];
    let err = CachedStateFeatures::new_compressed(
        key,
        invalid_shape,
        &invalid_data,
        CacheCompressionMode::None,
        invalid_mask,
    )
    .expect_err("batch_size != 1 のエントリ構築は拒絶される必要があります。");
    assert!(matches!(
        err,
        sokuto_runtime::error::RuntimeError::ModelContractViolation(_)
    ));
}

#[test]
fn test_8_4_1_xxhash3_and_moka_cache_concurrency() {
    let manager = Arc::new(StateCacheManager::new(
        10 * 1024 * 1024, // 10MB
        300,
        CacheCompressionMode::None,
    ));

    let state_text = "第1条（定義）本契約において「秘密情報」とは、甲または乙が開示した技術上、営業上の情報をいう。";
    let key = StateCacheManager::compute_key(state_text);

    // 1. 同一テキストに対する決定論的ハッシュ計算。
    let key_repeated = StateCacheManager::compute_key(state_text);
    assert_eq!(
        key, key_repeated,
        "xxHash3キーは完全に一致する必要があります。"
    );

    // 2. 特徴量エントリの作成と登録。
    let shape = vec![1, 16, 64];
    let raw_data = vec![0.5f32; 16 * 64];
    let mask = vec![1i64; 16];
    let features = Arc::new(
        CachedStateFeatures::new_compressed(
            key,
            shape.clone(),
            &raw_data,
            CacheCompressionMode::None,
            mask.clone(),
        )
        .expect("キャッシュエントリの作成に失敗しました。"),
    );

    manager.insert(key, Arc::clone(&features));
    manager.run_pending_tasks();

    assert!(manager.contains_key(key));
    assert_eq!(manager.entry_count(), 1);
    assert!(manager.weighted_size() > 0);

    // 3. 複数スレッドからの並行参照安全性の検証。
    let mut handles = Vec::new();
    for _ in 0..8 {
        let mgr = Arc::clone(&manager);
        let handle = std::thread::spawn(move || {
            let item = mgr.get(key).expect("キャッシュに存在する必要があります。");
            assert_eq!(item.hash_key, key);
            assert_eq!(item.seq_len(), 16);
            assert_eq!(item.hidden_size(), 64);
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().expect("スレッド実行に失敗しました。");
    }

    // 4. 無効化の検証。
    manager.invalidate(key);
    manager.run_pending_tasks();
    assert!(!manager.contains_key(key));
    assert_eq!(manager.entry_count(), 0);
}

#[test]
fn test_8_4_3_fp16_and_int8_compression_fidelity() {
    let hidden_size = 768; // ModernBERT-ja 隠れ層次元
    let seq_len = 128;
    let total_elements = seq_len * hidden_size;

    // 正規分布を模した合成ベクトル（平均0、分散1程度）。
    let mut synthetic_data = Vec::with_capacity(total_elements);
    for i in 0..total_elements {
        let val = ((i as f32) * 0.017).sin() * 2.0;
        synthetic_data.push(val);
    }

    let key = 0x1234_5678_9ABC_DEF0;
    let shape = vec![1, seq_len, hidden_size];
    let mask = vec![1i64; seq_len];

    // 1. `FP16` 圧縮保持と展開精度の検証 (目標: Atol <= 2e-3)。
    let fp16_entry = CachedStateFeatures::new_compressed(
        key,
        shape.clone(),
        &synthetic_data,
        CacheCompressionMode::Fp16,
        mask.clone(),
    )
    .expect("FP16 エントリの作成に失敗しました。");

    assert_eq!(fp16_entry.compression_mode(), CacheCompressionMode::Fp16);
    // `FP16` は `f32` の半分のメモリサイズ (要素数 * 2バイト)。
    assert_eq!(fp16_entry.byte_size, total_elements * 2);

    let fp16_val = fp16_entry.as_ort_value().expect("展開に失敗しました。");
    let (_, fp16_restored) = fp16_val
        .try_extract_tensor::<f32>()
        .expect("テンソル抽出に失敗しました。");

    // ort デフォルト CPU アロケータによる 64 バイト整列メモリ確保の検証。
    assert!(
        is_aligned_64(fp16_restored.as_ptr()),
        "FP16 展開後のテンソルが 64 バイト整列していません。"
    );

    let mut max_fp16_err = 0.0f32;
    for (&orig, &rest) in synthetic_data.iter().zip(fp16_restored.iter()) {
        let err = (orig - rest).abs();
        if err > max_fp16_err {
            max_fp16_err = err;
        }
    }
    assert!(
        max_fp16_err <= 2e-3,
        "FP16 最大絶対誤差超過: {max_fp16_err} > 2e-3。"
    );

    // 2. トークン単位対称 `INT8` 量子化保持と展開精度の検証 (目標: Atol <= 0.05)。
    let int8_entry = CachedStateFeatures::new_compressed(
        key,
        shape.clone(),
        &synthetic_data,
        CacheCompressionMode::Int8Symmetric,
        mask,
    )
    .expect("INT8 エントリの作成に失敗しました。");

    assert_eq!(
        int8_entry.compression_mode(),
        CacheCompressionMode::Int8Symmetric
    );
    // `INT8` は約75%削減 (要素数 * 1バイト + トークン数 * 4バイト)。
    assert!(int8_entry.byte_size < total_elements * 2);

    let int8_val = int8_entry.as_ort_value().expect("展開に失敗しました。");
    let (_, int8_restored) = int8_val
        .try_extract_tensor::<f32>()
        .expect("テンソル抽出に失敗しました。");

    // ort デフォルト CPU アロケータによる 64 バイト整列メモリ確保の検証。
    assert!(
        is_aligned_64(int8_restored.as_ptr()),
        "INT8 展開後のテンソルが 64 バイト整列していません。"
    );

    // バッチ展開時 (as_ort_value_for_batch) の直接展開と 64 バイト整列メモリ確保の検証。
    let b2_val = int8_entry
        .as_ort_value_for_batch(2)
        .expect("バッチ展開に失敗しました。");
    let (_, b2_restored) = b2_val
        .try_extract_tensor::<f32>()
        .expect("バッチテンソル抽出に失敗しました。");
    assert!(
        is_aligned_64(b2_restored.as_ptr()),
        "バッチ展開後のテンソルが 64 バイト整列していません。"
    );
    assert_eq!(b2_restored.len(), 2 * total_elements);

    let mut max_int8_err = 0.0f32;
    for (&orig, &rest) in synthetic_data.iter().zip(int8_restored.iter()) {
        let err = (orig - rest).abs();
        if err > max_int8_err {
            max_int8_err = err;
        }
    }
    assert!(
        max_int8_err <= 0.05,
        "INT8 最大絶対誤差超過: {max_int8_err} > 0.05。"
    );
}

#[test]
fn test_8_4_4_64byte_memory_alignment_guarantee() {
    // 1. 各種サイズでの64バイトメモリアライメント検証。
    for &size in &[1usize, 16, 64, 128, 512, 1024, 8192] {
        let buf = AlignedBuffer64::<f32>::with_capacity_zeroed(size);
        assert_eq!(buf.len(), size);
        assert!(
            buf.is_aligned_64(),
            "バッファが 64B アライメントされていません (サイズ: {size})。"
        );
        assert!(
            is_aligned_64(buf.as_ptr()),
            "生ポインタが 64B 整列していません。"
        );

        // ゼロ初期化の検証。
        assert!(buf.iter().all(|&v| v == 0.0f32));
    }

    // 2. スライスからのアライメント複製。
    let sample = vec![1.0f32, 2.0, 3.0, 4.0];
    let aligned = AlignedBuffer64::from_slice(&sample);
    assert_eq!(aligned.len(), 4);
    assert!(aligned.is_aligned_64());
    assert_eq!(&aligned[..], &sample[..]);
}

#[test]
fn test_dual_engine_non_existent_file_handling() {
    let dummy_state = "/non/existent/state_encoder.onnx";
    let dummy_query = "/non/existent/query_evaluator.onnx";
    let mgr = Arc::new(StateCacheManager::new(
        1024 * 1024,
        60,
        CacheCompressionMode::None,
    ));

    let res = DualSessionEngine::new(dummy_state, dummy_query, SessionConfig::cpu_only(), mgr);
    assert!(
        res.is_err(),
        "存在しないモデルパスはエラーとなる必要があります。"
    );
}

#[tokio::test]
async fn test_8_4_4_tokio_blocking_isolation() {
    // ワーカースレッドのブロック隔離が `Tokio` 非同期タスク内で正常に実行できるかを検証。
    let handle = tokio::task::spawn_blocking(|| {
        let buf = AlignedBuffer64::<f32>::with_capacity_zeroed(64);
        assert!(buf.is_aligned_64());
        42
    });

    let res = handle
        .await
        .expect("ブロッキングタスクの待機に失敗しました。");
    assert_eq!(res, 42);
}

#[test]
fn test_op_indices_minus_one_padding_semantics() {
    let query_seq_len = 64;
    // -1パディングおよび有効インデックス (0..64) は正常。
    let valid_indices = vec![10i64, 25, 40, -1];
    for &idx in &valid_indices {
        assert!(
            idx == -1 || (idx >= 0 && idx < query_seq_len as i64),
            "-1または有効なインデックスである必要があります。"
        );
    }

    // 範囲外インデックス (-2 や 64 以上) の検出。
    let invalid_indices = vec![-2i64, 64, 100];
    for &idx in &invalid_indices {
        let is_valid = idx == -1 || (idx >= 0 && idx < query_seq_len as i64);
        assert!(!is_valid, "無効なインデックスである必要があります: {idx}");
    }
}

#[test]
fn test_state_mask_padding_preservation_and_batch_replication() {
    let key = 0x5555_AAAA_1234_5678;
    let seq_len = 6;
    let hidden_size = 16;
    let shape = vec![1, seq_len, hidden_size];
    let data = vec![0.1f32; seq_len * hidden_size];

    // 先頭4トークンが有効で末尾2トークンがパディング（0）のマスク。
    let padded_mask = vec![1i64, 1, 1, 1, 0, 0];

    let entry = CachedStateFeatures::new_compressed(
        key,
        shape,
        &data,
        CacheCompressionMode::None,
        padded_mask.clone(),
    )
    .expect("キャッシュエントリ作成に成功すること。");

    assert_eq!(entry.state_mask, padded_mask);

    // バッチサイズ 1 への複製検証。
    let mask_b1 = entry.state_mask_for_batch(1);
    assert_eq!(mask_b1, vec![1i64, 1, 1, 1, 0, 0]);

    // バッチサイズ 3 への複製検証。各サンプルでパディング位置が正確に維持されることを確認。
    let mask_b3 = entry.state_mask_for_batch(3);
    assert_eq!(
        mask_b3,
        vec![
            1i64, 1, 1, 1, 0, 0, // サンプル 0
            1, 1, 1, 1, 0, 0, // サンプル 1
            1, 1, 1, 1, 0, 0, // サンプル 2
        ]
    );

    // 空マスク指定時の全有効フォールバック検証。
    let empty_mask_entry = CachedStateFeatures::new_compressed(
        key,
        vec![1, seq_len, hidden_size],
        &data,
        CacheCompressionMode::None,
        vec![],
    )
    .expect("空マスク指定時でも自動補完されること。");

    assert_eq!(empty_mask_entry.state_mask, vec![1i64; seq_len]);
    assert_eq!(
        empty_mask_entry.state_mask_for_batch(2),
        vec![1i64; 2 * seq_len]
    );

    // 長さ不一致マスク指定時はエラーとなることの検証。
    let mismatched_result = CachedStateFeatures::new_compressed(
        key,
        vec![1, seq_len, hidden_size],
        &data,
        CacheCompressionMode::None,
        vec![1i64; seq_len - 1],
    );
    assert!(mismatched_result.is_err());
}

#[test]
fn test_state_cache_manager_nfc_nfd_normalization_cache_hit() {
    let manager = StateCacheManager::new(1024 * 1024, 60, CacheCompressionMode::None);

    let nfc_text = "がっこう";
    let nfd_text = "か\u{3099}っこう";

    let key_nfc = StateCacheManager::compute_key(nfc_text);
    let key_nfd = StateCacheManager::compute_key(nfd_text);

    assert_eq!(key_nfc, key_nfd);

    let shape = vec![1, 4, 16];
    let raw_data = vec![0.1f32; 4 * 16];
    let mask = vec![1i64; 4];
    let features = Arc::new(
        CachedStateFeatures::new_compressed(
            key_nfc,
            shape,
            &raw_data,
            CacheCompressionMode::None,
            mask,
        )
        .expect("キャッシュエントリ作成に成功。"),
    );

    manager.insert(key_nfc, features);

    // NFD テキストから計算したキーでキャッシュヒットすることを検証。
    assert!(manager.contains_key(key_nfd));
    let hit = manager.get(key_nfd);
    assert!(hit.is_some());
    assert_eq!(hit.unwrap().hash_key, key_nfc);
}
