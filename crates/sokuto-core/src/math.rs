//! # 決定プリミティブ数理モジュール
//!
//! 確率較正、ソフトマックス、エントロピー確信度、およびスコア期待値の計算ユーティリティを提供する。

use std::borrow::Borrow;

use crate::error::{CoreError, Result};

/// 確率分布の総和に対する有効性判定用の微小許容閾値。
const WEIGHT_SUM_EPSILON: f64 = 1e-12;

/// ロジット列から温度付きソフトマックス確率分布を計算し、指定されたバッファへ書き込む。
///
/// # 概要
/// - 推論ランタイム等の高スループット処理向けに、ヒープ割り当てを行わずに確率分布を算出する。
/// - 数値安定性を担保するため、最大値減算(Max-subtraction)および温度除算を適用する。
/// - 算出された確率値は浮動小数点演算の累積丸め誤差を考慮し、`[0.0, 1.0]` の閉区間にクランプされる。
///
/// ```mermaid
/// flowchart TD
///     Start["入力: logits, out, temperature"] --> Val{"バリデーション<br>(空配列 / 長さ不一致 / 非有限値 / tau <= 0)"}
///     Val -- 不正 --> Err["CoreError::MathError 返却"]
///     Val -- 正常 --> KCheck{"候補数 K == 1 ?"}
///     KCheck -- Yes --> Single["out[0] = 1.0 (ショートサーキット)"]
///     KCheck -- No --> MaxFind["Pass 1: 最大値 z_max 探索<br>(同時 NaN/Inf 検出)"]
///     MaxFind --> ExpSum["Pass 2: 指数計算 & 累積和<br>exp((z_i - z_max) / tau_eff)"]
///     ExpSum --> Norm["Pass 3: 分母除算正規化 & clamp(0.0, 1.0)"]
///     Single --> Done["Ok(()) 返却"]
///     Norm --> Done
/// ```
///
/// # 数理仕様
/// $$p_i = \frac{\exp\left(\frac{z_i - z_{\max}}{\tau_{\text{eff}}}\right)}{\sum_{j=1}^K \exp\left(\frac{z_j - z_{\max}}{\tau_{\text{eff}}}\right)}, \quad z_{\max} = \max_{1 \le j \le K} z_j$$
/// ここで、有効温度 $\tau_{\text{eff}} = \max(\tau, 10^{-4})$ である。
///
/// # 引数
/// - `logits`: 未正規化の生ロジットスライス(`f64`)。
/// - `out`: 算出された確率値を書き込む出力先バッファスライス(`f64`)。`logits` と同一の要素数が必要。
/// - `temperature`: 較正温度パラメータ $\tau$。正の有限実数。
///
/// # 戻り値
/// - 成功時は `Ok(())`、入力異常時は `CoreError::MathError` を返却する。
#[aquamarine::aquamarine]
pub fn softmax_into(logits: &[f64], out: &mut [f64], temperature: f64) -> Result<()> {
    if logits.is_empty() {
        return Err(CoreError::MathError {
            message: "ロジット配列が空です。".to_string(),
        });
    }

    if logits.len() != out.len() {
        return Err(CoreError::MathError {
            message: format!(
                "出力バッファ長({out_len})がロジット長({logits_len})と一致しません。",
                out_len = out.len(),
                logits_len = logits.len()
            ),
        });
    }

    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(CoreError::MathError {
            message: format!("温度パラメータは正の有限実数である必要があります: {temperature}。"),
        });
    }

    // 単一候補(K=1)のエッジケースは指数計算をスキップして即座に 1.0 を設定する。
    if logits.len() == 1 {
        if !logits[0].is_finite() {
            return Err(CoreError::MathError {
                message: "ロジットに非有限値(NaNまたは無限大)が含まれています。".to_string(),
            });
        }
        out[0] = 1.0;
        return Ok(());
    }

    // Python 側の較正評価器(evaluator.py)と平仄を合わせ、極小温度によるオーバーフローを防ぐ安全下限クランプ。
    let effective_tau = temperature.max(1e-4);

    // Pass 1: 最大値探索および非有限値(NaN, Inf)の厳密な検出。
    // IEEE 754 の f64::max による NaN すり抜けを防ぐため、明示的な走査を行う。
    let mut max_logit = f64::NEG_INFINITY;
    for &logit in logits {
        if !logit.is_finite() {
            return Err(CoreError::MathError {
                message: "ロジットに非有限値(NaNまたは無限大)が含まれています。".to_string(),
            });
        }
        if logit > max_logit {
            max_logit = logit;
        }
    }

    // Pass 2: 最大値減算後の温度スケーリング、指数計算、および総和累積。
    // (logit - max_logit) <= 0.0 であるため、指数関数の引数は常に (-inf, 0.0] となりオーバーフローしない。
    // かつ最大値要素自身は exp(0.0) = 1.0 となるため、sum_exp >= 1.0 が数学的に保証される。
    let mut sum_exp = 0.0;
    for (i, &logit) in logits.iter().enumerate() {
        let scaled = (logit - max_logit) / effective_tau;
        let exp_val = scaled.exp();
        out[i] = exp_val;
        sum_exp += exp_val;
    }

    if sum_exp <= 0.0 || !sum_exp.is_finite() {
        return Err(CoreError::MathError {
            message: "ソフトマックスの分母が不正な値になりました。".to_string(),
        });
    }

    // Pass 3: 分母除算による正規化と丸め誤差対策のクランプ。
    for val in out.iter_mut() {
        *val = (*val / sum_exp).clamp(0.0, 1.0);
    }

    Ok(())
}

/// ロジット列から温度付きソフトマックス確率分布を計算し、新しいベクタとして返却する。
///
/// # 概要
/// - 内部でゼロアロケーション版関数 [`softmax_into`] を呼び出す。
/// - 返却される確率ベクタの総和はほぼ 1.0 となり、各要素は `[0.0, 1.0]` に収まる。
///
/// # 引数
/// - `logits`: 未正規化の生ロジットスライス(`f64`)。
/// - `temperature`: 較正温度パラメータ $\tau$。正の有限実数。
///
/// # 戻り値
/// - 成功時は計算された確率分布ベクタ(`Vec<f64>`)、異常時は `CoreError::MathError`。
pub fn softmax(logits: &[f64], temperature: f64) -> Result<Vec<f64>> {
    let mut probs: Vec<f64> = vec![0.0; logits.len()];
    softmax_into(logits, &mut probs, temperature)?;
    Ok(probs)
}

/// 任意の確率値イテレータから候補数 $K$ に非依存な正規化シャノンエントロピー $H_{\text{norm}}(p) \in [0.0, 1.0]$ を単一パス走査で算出する。
///
/// # 数理仕様
/// $$H_{\text{norm}}(p) = \begin{cases} 0.0 & (K = 1) \\ \frac{-\sum_{k=1}^K p_k \ln p_k}{\ln K} & (K \ge 2) \end{cases}$$
///
/// - $O(K)$ 単一パス走査により、ヒープメモリ割り当て(ゼロアロケーション)で計算する。
/// - $p_k \le 0.0$ の要素は情報量 $0 \ln 0 = 0$ として対数計算をスキップする。
/// - $K=0$ の場合は空配列エラーとして `CoreError::MathError` を返却する。
/// - $K=1$ の特異点ではゼロ除算を回避し、決定的な $0.0$(不確実性なし)を即座に返却する。
/// - 非有限値(NaNまたは無限大)や負の確率値が含まれる場合は `CoreError::MathError` を返却する。
/// - 戻り値は浮動小数点誤差を考慮して `[0.0, 1.0]` の閉区間にクランプされる。
///
/// # 引数
/// - `iter`: 各候補の確率値(`f64` または `&f64`)を提供するイテレータ。
///
/// # 戻り値
/// - 成功時は `[0.0, 1.0]` にクランプされた正規化エントロピー、不正時または空時は `CoreError::MathError`。
pub fn normalized_entropy_iter<I>(iter: I) -> Result<f64>
where
    I: IntoIterator,
    I::Item: Borrow<f64>,
{
    let mut count: usize = 0;
    let mut entropy = 0.0;

    for item in iter {
        let p = *item.borrow();
        if !p.is_finite() {
            return Err(CoreError::MathError {
                message: "確率値に非有限値(NaNまたは無限大)が含まれています。".to_string(),
            });
        }
        if p < 0.0 {
            return Err(CoreError::MathError {
                message: format!("負の確率値が含まれています: {p}。"),
            });
        }

        count += 1;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }

    match count {
        0 => Err(CoreError::MathError {
            message: "確率分布配列が空です。".to_string(),
        }),
        1 => Ok(0.0),
        _ => {
            let max_entropy = (count as f64).ln();
            if max_entropy <= 0.0 {
                return Ok(0.0);
            }

            let h_norm = entropy / max_entropy;
            Ok(h_norm.clamp(0.0, 1.0))
        }
    }
}

/// 確率分布から候補数 $K$ に非依存な正規化シャノンエントロピー $H_{\text{norm}}(p) \in [0.0, 1.0]$ を算出する。
///
/// # 数理仕様
/// $$H_{\text{norm}}(p) = \begin{cases} 0.0 & (K = 1) \\ \frac{-\sum_{k=1}^K p_k \ln p_k}{\ln K} & (K \ge 2) \end{cases}$$
///
/// - [`normalized_entropy_iter`] を呼び出すゼロコストラッパーとして動作する。
/// - $p_k \le 0.0$ の要素は情報量 $0 \ln 0 = 0$ として対数計算をスキップする。
/// - $K=1$ の特異点ではゼロ除算を回避し、決定的な $0.0$(不確実性なし)を即座に返却する。
/// - 戻り値は浮動小数点誤差を考慮して `[0.0, 1.0]` の閉区間にクランプされる。
///
/// # 引数
/// - `probabilities`: 各候補の確率分布スライス(`f64`)。
///
/// # 戻り値
/// - 成功時は `[0.0, 1.0]` にクランプされた正規化エントロピー、不正時は `CoreError::MathError`。
pub fn normalized_entropy(probabilities: &[f64]) -> Result<f64> {
    normalized_entropy_iter(probabilities)
}

/// 任意の確率値イテレータから上位2候補の確率差(Top-Margin) $M(p) \in [0.0, 1.0]$ を単一パス走査で算出する。
///
/// # 数理仕様
/// $$M(p) = \begin{cases} 1.0 & (K = 1) \\ p_{(1)} - p_{(2)} & (K \ge 2) \end{cases}$$
///
/// - $O(K)$ 単一パス走査により、ヒープメモリ割り当て(ゼロアロケーション)で計算する。
/// - $K=0$ の場合は空配列エラーとして `CoreError::MathError` を返却する。
/// - $K=1$ の特異点では $1.0$(他候補なし・完全なマージン)を即座に返却する。
/// - 同率1位(タイ)の場合は $M(p) = 0.0$ となる。
/// - 非有限値(NaNまたは無限大)や負の確率値が含まれる場合は `CoreError::MathError` を返却する。
/// - 戻り値は `[0.0, 1.0]` の閉区間にクランプされる。
///
/// # 引数
/// - `iter`: 確率値(`f64` または `&f64`)を提供するイテレータ。
///
/// # 戻り値
/// - 成功時は `[0.0, 1.0]` にクランプされた Top-Margin、不正時または空時は `CoreError::MathError`。
pub fn top_margin_iter<I>(iter: I) -> Result<f64>
where
    I: IntoIterator,
    I::Item: Borrow<f64>,
{
    let mut count: usize = 0;
    let mut max1 = f64::NEG_INFINITY;
    let mut max2 = f64::NEG_INFINITY;

    for item in iter {
        let p = *item.borrow();
        if !p.is_finite() {
            return Err(CoreError::MathError {
                message: "確率値に非有限値(NaNまたは無限大)が含まれています。".to_string(),
            });
        }
        if p < 0.0 {
            return Err(CoreError::MathError {
                message: format!("負の確率値が含まれています: {p}。"),
            });
        }

        count += 1;
        if p > max1 {
            max2 = max1;
            max1 = p;
        } else if p > max2 {
            max2 = p;
        }
    }

    match count {
        0 => Err(CoreError::MathError {
            message: "確率分布配列が空です。".to_string(),
        }),
        1 => Ok(1.0),
        _ => {
            let margin = max1 - max2;
            Ok(margin.clamp(0.0, 1.0))
        }
    }
}

/// 確率分布から上位2候補の確率差(Top-Margin) $M(p) \in [0.0, 1.0]$ を算出する。
///
/// # 数理仕様
/// $$M(p) = \begin{cases} 1.0 & (K = 1) \\ p_{(1)} - p_{(2)} & (K \ge 2) \end{cases}$$
///
/// - [`top_margin_iter`] を呼び出すゼロコストラッパーとして動作する。
/// - $O(K)$ 単一パス走査により、ヒープメモリ割り当て(アロケーション)ゼロで計算する。
/// - $K=1$ の特異点では $1.0$(他候補なし・完全なマージン)を即座に返却する。
/// - 同率1位(タイ)の場合は $M(p) = 0.0$ となる。
/// - 戻り値は `[0.0, 1.0]` の閉区間にクランプされる。
///
/// # 引数
/// - `probabilities`: 各候補の確率分布スライス(`f64`)。
///
/// # 戻り値
/// - 成功時は `[0.0, 1.0]` にクランプされた Top-Margin、不正時は `CoreError::MathError`。
#[inline]
pub fn top_margin(probabilities: &[f64]) -> Result<f64> {
    top_margin_iter(probabilities)
}

/// 任意の確率値イテレータから、正規化エントロピーと Top-Margin を統合した複合確信度スコア $S_{\text{confidence}} \in [0.0, 1.0]$ を単一パス走査で算出する。
///
/// # 数理仕様
/// $$S_{\text{confidence}} = \begin{cases} 1.0 & (K = 1) \\ (1.0 - H_{\text{norm}}(p)) \times M(p) & (K \ge 2) \end{cases}$$
///
/// - $O(K)$ 単一パス走査により、要素数 $K$、エントロピー累積値 $-\sum p \ln p$、上位2候補 $p_{(1)}, p_{(2)}$ を同時に集計する。
/// - 同一スライスの2回走査を回避し、ゼロアロケーションかつキャッシュ局所性を最大化する。
/// - $K=0$ の場合は空配列エラーとして `CoreError::MathError` を返却する。
/// - $K=1$ の特異点では $1.0$(不確実性なし・完全確信)を即座に返却する。
/// - 同率タイブレーク時はマージンが $0.0$ となるため、複合確信度も厳密に $0.0$ に収束する。
/// - 非有限値(NaNまたは無限大)や負の確率値が含まれる場合は `CoreError::MathError` を返却する。
/// - 戻り値は浮動小数点誤差を考慮して `[0.0, 1.0]` の閉区間にクランプされる。
///
/// # 引数
/// - `iter`: 確率値(`f64` または `&f64`)を提供するイテレータ。
///
/// # 戻り値
/// - 成功時は `[0.0, 1.0]` にクランプされた複合確信度スコア、不正時または空時は `CoreError::MathError`。
pub fn composite_confidence_iter<I>(iter: I) -> Result<f64>
where
    I: IntoIterator,
    I::Item: Borrow<f64>,
{
    let mut count: usize = 0;
    let mut entropy = 0.0;
    let mut max1 = f64::NEG_INFINITY;
    let mut max2 = f64::NEG_INFINITY;

    for item in iter {
        let p = *item.borrow();
        if !p.is_finite() {
            return Err(CoreError::MathError {
                message: "確率値に非有限値(NaNまたは無限大)が含まれています。".to_string(),
            });
        }
        if p < 0.0 {
            return Err(CoreError::MathError {
                message: format!("負の確率値が含まれています: {p}。"),
            });
        }

        count += 1;
        if p > 0.0 {
            entropy -= p * p.ln();
        }

        if p > max1 {
            max2 = max1;
            max1 = p;
        } else if p > max2 {
            max2 = p;
        }
    }

    match count {
        0 => Err(CoreError::MathError {
            message: "確率分布配列が空です。".to_string(),
        }),
        1 => Ok(1.0),
        _ => {
            let max_entropy = (count as f64).ln();
            let h_norm = if max_entropy <= 0.0 {
                0.0
            } else {
                (entropy / max_entropy).clamp(0.0, 1.0)
            };
            let margin = (max1 - max2).clamp(0.0, 1.0);
            let score = (1.0 - h_norm) * margin;
            Ok(score.clamp(0.0, 1.0))
        }
    }
}

/// 確率分布から、正規化エントロピーと Top-Margin を統合した複合確信度スコア $S_{\text{confidence}} \in [0.0, 1.0]$ を算出する。
///
/// # 数理仕様
/// $$S_{\text{confidence}} = (1.0 - H_{\text{norm}}(p)) \times M(p)$$
///
/// - [`composite_confidence_iter`] を呼び出すゼロコストラッパーとして動作する。
/// - $O(K)$ 単一パス走査により、要素数 $K$、エントロピー累積値、上位2候補のマージンを同時に集計する。
/// - 分布全体が鋭利であり(平坦度 $H_{\text{norm}}$ が小)、かつ第1位と第2位の差分(マージン $M(p)$)が明瞭な場合にのみ
///   高い確信度を出力する。
/// - $K=1$ の特異点では $1.0$ を返却する。
/// - 同率タイブレーク時はマージンが $0.0$ となるため、複合確信度も厳密に $0.0$ に収束する。
/// - 戻り値は `[0.0, 1.0]` の閉区間にクランプされる。
///
/// # 引数
/// - `probabilities`: 各候補の確率分布スライス(`f64`)。
///
/// # 戻り値
/// - 成功時は `[0.0, 1.0]` にクランプされた複合確信度スコア、不正時は `CoreError::MathError`。
#[inline]
pub fn composite_confidence(probabilities: &[f64]) -> Result<f64> {
    composite_confidence_iter(probabilities)
}

/// シャノンエントロピーに基づき、0.0〜1.0 の正規化確信度(Confidence)を算出する。
///
/// 候補数 $K=1$ の場合は完全に確信しているため 1.0 を返す。
/// 一様分布のとき 0.0、単一の候補に確率が集中しているとき 1.0 となる。
///
/// # 備考
/// 本関数は後方互換性のために維持されており、$1.0 - H_{\text{norm}}(p)$ を返却する。
pub fn normalized_entropy_confidence(probabilities: &[f64]) -> Result<f64> {
    let h_norm = normalized_entropy(probabilities)?;
    Ok((1.0 - h_norm).clamp(0.0, 1.0))
}

/// スライス内から最大値を持つ要素のインデックスを取得する。
///
/// # 概要
/// - 決定論的タイブレーク規則として、同一の最大値が複数存在する場合は「最小のインデックス(先勝ち)」を確定的に採択する。
/// - 配列内に `NaN` が含まれている場合は `CoreError::MathError` を返却する。
///
/// # 引数
/// - `values`: 探索対象の実数値スライス(`f64`)。
///
/// # 戻り値
/// - 成功時は最大値のインデックス(`usize`)、配列が空または非有限値を含む場合は `CoreError::MathError`。
pub fn argmax(values: &[f64]) -> Result<usize> {
    if values.is_empty() {
        return Err(CoreError::MathError {
            message: "探索対象のスライスが空です。".to_string(),
        });
    }

    let mut best_index = 0;
    let mut max_val = f64::NEG_INFINITY;

    for (i, &val) in values.iter().enumerate() {
        if !val.is_finite() {
            return Err(CoreError::MathError {
                message: "スライス内に非有限値(NaNまたは無限大)が含まれています。".to_string(),
            });
        }
        // 厳密な不等号(val > max_val)を用いることで、同率値の場合は先頭インデックスを維持する。
        if val > max_val {
            max_val = val;
            best_index = i;
        }
    }

    Ok(best_index)
}

/// 任意の確率値イテレータから加重平均スコア実数値 $\text{Score} \in [0.0, M-1.0]$ を単一パス走査で算出する。
///
/// # 概要
/// - 各段階レベル $k \in \{0, \dots, M-1\}$ に対する加重平均期待値 $\hat{s} = E[S]$ を計算する。
/// - [`score_moments_iter`] を呼び出し、単一パス走査で算出された期待値を抽出して返却する。
///
/// # 引数
/// - `iter`: 各段階レベルの確率値(`f64` または `&f64`)を提供するイテレータ。
///
/// # 戻り値
/// - 成功時は加重平均スコア実数値、不正時は `CoreError::MathError`。
pub fn expected_score_iter<I>(iter: I) -> Result<f64>
where
    I: IntoIterator,
    I::Item: Borrow<f64>,
{
    score_moments_iter(iter).map(|(_, mean, _)| mean)
}

/// Score 型の確率分布から加重平均スコア実数値を算出する。
///
/// 各段階レベル $k \in \{0, \dots, M-1\}$ に対して $\text{Score} = \sum k \cdot p_k$ を計算する。
///
/// # 備考
/// - [`expected_score_iter`] を呼び出すゼロコストラッパーとして動作する。
#[inline]
pub fn expected_score(probabilities: &[f64]) -> Result<f64> {
    expected_score_iter(probabilities)
}

/// 任意の確率値イテレータから、段階数 $M$、加重平均期待値 $E[S]$、事後分散 $\text{Var}[S]$ を単一パス走査で算出する。
///
/// # 数理仕様
/// Welford の重み付きオンラインアルゴリズムに基づき、単一パス走査で桁落ちなく正確に集計する。
///
/// - $O(M)$ 単一パス走査により、二重走査や中間 `Vec` 確保(ヒープアロケーション)を完全に排除する。
/// - $M < 2$ の場合は段階数不足エラーとして `CoreError::MathError` を返却する。
/// - 非有限値(NaNまたは無限大)や負の確率値が含まれる場合は `CoreError::MathError` を返却する。
/// - 全要素がゼロまたは重みの総和が微小閾値以下($\le 10^{-12}$)の不正な分布の場合は `CoreError::MathError` を返却する。
/// - 確率総和の丸め誤差や非正規化重みに対応するため、加重二乗和 $M_2$ を重みの総和で正規化して加重分散を算出する。
///
/// # 引数
/// - `iter`: 各段階レベルの確率値(`f64` または `&f64`)を提供するイテレータ。
///
/// # 戻り値
/// - 成功時は `(count, expected_score, variance)` のタプル、不正時は `CoreError::MathError`。
pub fn score_moments_iter<I>(iter: I) -> Result<(usize, f64, f64)>
where
    I: IntoIterator,
    I::Item: Borrow<f64>,
{
    let mut count: usize = 0;
    let mut weight_sum = 0.0;
    let mut mean = 0.0;
    let mut m2 = 0.0;

    for (k, item) in iter.into_iter().enumerate() {
        let p = *item.borrow();
        if !p.is_finite() {
            return Err(CoreError::MathError {
                message: "確率値に非有限値(NaNまたは無限大)が含まれています。".to_string(),
            });
        }
        if p < 0.0 {
            return Err(CoreError::MathError {
                message: format!("負の確率値が含まれています: {p}。"),
            });
        }

        count += 1;
        let x = k as f64;
        let temp = weight_sum + p;
        let delta = x - mean;
        let r = if temp > 0.0 { delta * p / temp } else { 0.0 };
        mean += r;
        m2 += weight_sum * delta * r;
        weight_sum = temp;
    }

    if count < 2 {
        return Err(CoreError::MathError {
            message: "Score の確率分布は 2 段階以上必要です。".to_string(),
        });
    }

    if !weight_sum.is_finite() || weight_sum <= WEIGHT_SUM_EPSILON {
        return Err(CoreError::MathError {
            message: format!("確率値の総和が正数ではありません: {weight_sum}。"),
        });
    }

    let var = (m2 / weight_sum).max(0.0);

    Ok((count, mean, var))
}

/// Score 型の確率分布から、段階数 $M$、加重平均期待値 $E[S]$、事後分散 $\text{Var}[S]$ を単一パス走査で算出する。
///
/// # 備考
/// - [`score_moments_iter`] を呼び出すゼロコストラッパーとして動作する。
#[inline]
pub fn score_moments(probabilities: &[f64]) -> Result<(usize, f64, f64)> {
    score_moments_iter(probabilities)
}

/// 任意の確率値イテレータからスコアの事後分散 $\text{Var}[S]$ を単一パス走査で算出する。
///
/// # 備考
/// - [`score_moments_iter`] を呼び出し、単一パス走査で事後分散を取得する。
pub fn score_variance_iter<I>(iter: I) -> Result<f64>
where
    I: IntoIterator,
    I::Item: Borrow<f64>,
{
    let (_, _, var) = score_moments_iter(iter)?;
    Ok(var)
}

/// Score 型の確率分布からスコアの事後分散 $\text{Var}[S]$ を算出する。
///
/// 段階レベル $k \in \{0, \dots, M-1\}$ に対し、加重平均期待値 $\hat{s} = E[S]$ を用いて
/// $\text{Var}[S] = \sum_{k=0}^{M-1} (k - \hat{s})^2 p_k$ を計算する。
///
/// # 備考
/// - [`score_variance_iter`] を呼び出すゼロコストラッパーとして動作する。
#[inline]
pub fn score_variance(probabilities: &[f64]) -> Result<f64> {
    score_variance_iter(probabilities)
}

/// 段階数 $M$ と事後分散 $\text{Var}[S]$ から、最大可能分散で正規化した確信度 $C_{\text{var}} \in [0.0, 1.0]$ を算出する。
///
/// # 概要
/// - $M$ 段階評価における理論上の最大分散 $V_{\max} = \frac{(M-1)^2}{4}$ (両極端に 0.5 ずつ配分された二峰性分布)を用いて事後分散を正規化する。
/// - $M < 2$ の場合は分散による不確実性を定義できないため、確信度 1.0 を返却する。
///
/// # 数理仕様
/// $$C_{\text{var}} = 1.0 - \frac{\text{Var}[S]}{V_{\max}} = 1.0 - \frac{4 \cdot \text{Var}[S]}{(M-1)^2}$$
///
/// # 引数
/// - `m`: 評価段階数 $M$。
/// - `var`: 算出済みの事後分散 $\text{Var}[S]$。
///
/// # 戻り値
/// - `[0.0, 1.0]` の閉区間にクランプされた正規化分散確信度。
#[inline]
pub fn normalized_variance_confidence_from_variance(m: usize, var: f64) -> f64 {
    if m < 2 {
        return 1.0;
    }
    let max_var = ((m - 1) as f64).powi(2) / 4.0;
    if max_var <= 0.0 {
        return 1.0;
    }
    let confidence = 1.0 - (var / max_var);
    confidence.clamp(0.0, 1.0)
}

/// 任意の確率値イテレータから、最大可能分散で正規化した確信度 $C_{\text{var}} \in [0.0, 1.0]$ を単一パス走査で算出する。
///
/// # 備考
/// - [`score_moments_iter`] を呼び出し、単一パス走査で段階数と事後分散を取得して正規化確信度を計算する。
pub fn normalized_variance_confidence_iter<I>(iter: I) -> Result<f64>
where
    I: IntoIterator,
    I::Item: Borrow<f64>,
{
    let (m, _, var) = score_moments_iter(iter)?;
    Ok(normalized_variance_confidence_from_variance(m, var))
}

/// Score 型の確率分布から、最大可能分散で正規化した確信度 $C_{\text{var}} \in [0.0, 1.0]$ を算出する。
///
/// # 概要
/// - 順序尺度(Score)においてカテゴリカルエントロピーを用いると、両極端への分裂(高分散・バイモーダル)と
///   隣接段階への分裂(低分散)を区別できず、誤った過信を招く問題がある。
/// - 本関数では事後分散 $\text{Var}\[S\]$ を計算し、$M$ 段階における理論上の最大分散
///   $V_{\max} = \frac{(M-1)^2}{4}$(両極端に 0.5 ずつ配分された二峰性分布)で正規化する。
///
/// # 数理仕様
/// $$C_{\text{var}} = 1.0 - \frac{\text{Var}\[S\]}{V_{\max}} = 1.0 - \frac{4 \cdot \text{Var}\[S\]}{(M-1)^2}$$
///
/// # 引数
/// - `probabilities`: 各段階レベルの確率分布スライス($M \ge 2$)。
///
/// # 戻り値
/// - 成功時は `[0.0, 1.0]` の閉区間にクランプされた正規化確信度、不正時は `CoreError::MathError`。
#[inline]
pub fn normalized_variance_confidence(probabilities: &[f64]) -> Result<f64> {
    normalized_variance_confidence_iter(probabilities)
}

/// 任意の確率値イテレータから、理論最大分散に対する事後分散の比率 $\text{VarRatio} \in [0.0, 1.0]$ を単一パス走査で算出する。
///
/// # 備考
/// - [`score_moments_iter`] を呼び出し、単一パス走査で段階数と事後分散を取得して分散比率を計算する。
pub fn score_variance_ratio_iter<I>(iter: I) -> Result<f64>
where
    I: IntoIterator,
    I::Item: Borrow<f64>,
{
    let (m, _, var) = score_moments_iter(iter)?;
    let max_var = ((m - 1) as f64).powi(2) / 4.0;

    if max_var <= 0.0 {
        return Ok(0.0);
    }

    let ratio = var / max_var;
    Ok(ratio.clamp(0.0, 1.0))
}

/// Score 型の確率分布から、理論最大分散に対する事後分散の比率 $\text{VarRatio} \in [0.0, 1.0]$ を算出する。
///
/// # 概要
/// - $M$ 段階における理論最大分散 $V_{\max} = \frac{(M-1)^2}{4}$ に対する比率を計算する。
/// - 両端($0$ と $M-1$)に $0.5$ ずつ配分された完全二峰性分布では $1.0$ となる。
///
/// # 引数
/// - `probabilities`: 各段階レベルの確率分布スライス ($M \ge 2$)。
///
/// # 戻り値
/// - 成功時は `[0.0, 1.0]` にクランプされた分散比率、不正時は `CoreError::MathError`。
#[inline]
pub fn score_variance_ratio(probabilities: &[f64]) -> Result<f64> {
    score_variance_ratio_iter(probabilities)
}

/// 任意の確率値イテレータから双峰性係数 (Bimodality Coefficient: $BC$) を算出する。
///
/// # 概要
/// - 離散確率分布における母集団中心モーメント(2次、3次、4次)を直接算出し、
///   標本補正項による $M \le 3$ でのゼロ除算クラッシュを完全に排除した閉じた数理計算を行う。
/// - Welford / Terriberry の加重オンライン高次モーメント更新式を採用し、平均、分散 ($M_2$)、
///   3次中心モーメント ($M_3$)、4次中心モーメント ($M_4$) を完全単一パス・ゼロアロケーションでストリーミング算出する。
/// - 中間バッファを一切必要としないため、メモリ計算量は $O(1)$ であり、要素数に関わらずヒープ割り当ては完全ゼロとなる。
/// - 単峰性分布では $BC < 0.555$ となり、完全一様分布では $BC \approx 0.555$、
///   両端対立(二極化)分布では $BC \to 1.0$ へ漸近する。
/// - 極小分散 ($\sigma^2 < 10^{-6}$) の場合は、完全単峰・高確信度と判定してゼロ除算を回避し、
///   即座に $0.0$ を返却する。
///
/// # 数理仕様
/// $$BC = \frac{\gamma^2 + 1}{\kappa}$$
/// ここで、歪度 $\gamma = \frac{\mu_3}{\sigma^3}$、尖度 $\kappa = \frac{\mu_4}{\sigma^4}$、
/// $\mu_r = \frac{1}{W} \sum_{k=0}^{M-1} (k - \mu)^r p_k$ ($W = \sum p_k$) である。
/// 確率総和の丸め誤差や非正規化重みに対応するため、各中心モーメントを $W$ で正規化して歪度・尖度を算出する。
///
/// # 引数
/// - `iter`: 各段階レベルの確率値(`f64` または `&f64`)を提供するイテレータ ($M \ge 2$)。
///
/// # 戻り値
/// - 成功時は `[0.0, 1.0]` の閉区間にクランプされた双峰性係数、不正時は `CoreError::MathError`。
pub fn bimodality_coefficient_iter<I>(iter: I) -> Result<f64>
where
    I: IntoIterator,
    I::Item: Borrow<f64>,
{
    let mut count: usize = 0;
    let mut weight_sum = 0.0;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;

    for (k, item) in iter.into_iter().enumerate() {
        let p = *item.borrow();
        if !p.is_finite() {
            return Err(CoreError::MathError {
                message: "確率値に非有限値(NaNまたは無限大)が含まれています。".to_string(),
            });
        }
        if p < 0.0 {
            return Err(CoreError::MathError {
                message: format!("負の確率値が含まれています: {p}。"),
            });
        }

        count += 1;
        let x = k as f64;
        let temp_weight = weight_sum + p;
        let delta = x - mean;
        let r = if temp_weight > 0.0 {
            delta * p / temp_weight
        } else {
            0.0
        };
        let delta2 = delta - r;

        let delta_m2 = weight_sum * delta * r;
        let r2 = r * r;
        let term_m4 = weight_sum * (r2 * r2) + p * (delta2 * delta2 * delta2 * delta2);

        // 高次モーメントの更新 (M4 -> M3 -> M2 -> mean の順序で依存関係を解決)
        m4 += -4.0 * r * m3 + 6.0 * r2 * m2 + term_m4;
        m3 += -3.0 * r * m2 + delta_m2 * (delta - 2.0 * r);
        m2 += delta_m2;
        mean += r;
        weight_sum = temp_weight;
    }

    if count < 2 {
        return Err(CoreError::MathError {
            message: "双峰性係数の計算には 2 段階以上の確率分布が必要です。".to_string(),
        });
    }

    let (variance, mu3, mu4) = if weight_sum > 0.0 {
        (m2 / weight_sum, m3 / weight_sum, m4 / weight_sum)
    } else {
        (0.0, 0.0, 0.0)
    };

    // 極小分散 (ワンホット等) のショートサーキット判定
    if variance < 1e-6 {
        return Ok(0.0);
    }

    let sigma = variance.sqrt();
    let sigma3 = variance * sigma;
    let sigma4 = variance * variance;

    if sigma4 <= 0.0 {
        return Ok(0.0);
    }

    let gamma = mu3 / sigma3;
    let kurtosis = mu4 / sigma4;

    if kurtosis <= 0.0 {
        return Ok(0.0);
    }

    let bc = (gamma * gamma + 1.0) / kurtosis;
    Ok(bc.clamp(0.0, 1.0))
}

/// Score 型の確率分布から双峰性係数 (Bimodality Coefficient: $BC$) を算出する。
///
/// # 概要
/// - 離散確率分布における母集団中心モーメント(2次、3次、4次)を直接算出し、
///   標本補正項による $M \le 3$ でのゼロ除算クラッシュを完全に排除した閉じた数理計算を行う。
/// - 単峰性分布では $BC < 0.555$ となり、完全一様分布では $BC \approx 0.555$、
///   両端対立(二極化)分布では $BC \to 1.0$ へ漸近する。
/// - 極小分散 ($\sigma^2 < 10^{-6}$) の場合は、完全単峰・高確信度と判定してゼロ除算を回避し、
///   即座に $0.0$ を返却する。
///
/// # 数理仕様
/// $$BC = \frac{\gamma^2 + 1}{\kappa}$$
/// ここで、歪度 $\gamma = \frac{\mu_3}{\sigma^3}$、尖度 $\kappa = \frac{\mu_4}{\sigma^4}$、
/// $\mu_r = \frac{1}{W} \sum_{k=0}^{M-1} (k - \mu)^r p_k$ ($W = \sum p_k$) である。
/// 確率総和の丸め誤差や非正規化重みに対応するため、各中心モーメントを $W$ で正規化して歪度・尖度を算出する。
///
/// # 引数
/// - `probabilities`: 各段階レベルの確率分布スライス ($M \ge 2$)。
///
/// # 戻り値
/// - 成功時は `[0.0, 1.0]` の閉区間にクランプされた双峰性係数、不正時は `CoreError::MathError`。
///
/// # 備考
/// - [`bimodality_coefficient_iter`] を呼び出すゼロコストラッパーとして動作する。
#[inline]
pub fn bimodality_coefficient(probabilities: &[f64]) -> Result<f64> {
    bimodality_coefficient_iter(probabilities)
}

/// Noul 型の言明真実確率値 $P(\text{true}) \in [0.0, 1.0]$ を算出する。
///
/// # 概要
/// - ロジット差分 $x = (z_{\text{true}} - z_{\text{false}}) / \tau_{\text{eff}}$ に対し、
///   符号分岐型シグモイド $\sigma(x)$ を用いることで指数オーバーフローおよびアンダーフローを完全に排除する。
/// - $x \ge 0$ のときは $\frac{1}{1 + \exp(-x)}$、$x < 0$ のときは $\frac{\exp(x)}{1 + \exp(x)}$ を計算する。
///
/// # 引数
/// - `logit_true`: 真(True)候補の生ロジット。
/// - `logit_false`: 偽(False)候補の生ロジット。
/// - `temperature`: 較正温度パラメータ $\tau$。正の有限実数。
///
/// # 戻り値
/// - 成功時は `[0.0, 1.0]` にクランプされた真実確率、不正時は `CoreError::MathError`。
pub fn noul_probability(logit_true: f64, logit_false: f64, temperature: f64) -> Result<f64> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(CoreError::MathError {
            message: format!("温度パラメータは正の実数である必要があります: {temperature}。"),
        });
    }

    if !logit_true.is_finite() || !logit_false.is_finite() {
        return Err(CoreError::MathError {
            message: "ロジットに非有限値(NaNまたは無限大)が含まれています。".to_string(),
        });
    }

    let effective_tau = temperature.max(1e-4);
    let diff = (logit_true - logit_false) / effective_tau;

    let prob = if diff >= 0.0 {
        1.0 / (1.0 + (-diff).exp())
    } else {
        let exp_diff = diff.exp();
        exp_diff / (1.0 + exp_diff)
    };

    Ok(prob.clamp(0.0, 1.0))
}

/// Noul 型の真実確率から、決定の尖り度に基づく正規化確信度 $C_{\text{noul}} \in [0.0, 1.0]$ を算出する。
///
/// # 数理仕様
/// $$C_{\text{noul}} = |2 \cdot P(\text{true}) - 1.0|$$
/// - $P(\text{true}) = 0.5$(完全な迷い)のとき 0.0。
/// - $P(\text{true}) = 1.0$ または $0.0$(完全な確信)のとき 1.0。
pub fn noul_confidence(probability: f64) -> Result<f64> {
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(CoreError::MathError {
            message: format!("確率は [0.0, 1.0] の有限実数である必要があります: {probability}。"),
        });
    }

    let conf = (2.0 * probability - 1.0).abs();
    Ok(conf.clamp(0.0, 1.0))
}

/// 任意のロジット値イテレータから自由エネルギー値および要素数を単一パス走査で算出する内部ヘルパー。
fn free_energy_with_count_iter<I>(iter: I, temperature: f64) -> Result<(f64, usize)>
where
    I: IntoIterator,
    I::Item: Borrow<f64>,
{
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err(CoreError::MathError {
            message: format!("温度パラメータは正の有限実数である必要があります: {temperature}。"),
        });
    }

    let effective_tau = temperature.max(1e-4);
    let mut count = 0;
    let mut max_logit = f64::NEG_INFINITY;
    let mut sum_exp = 0.0;

    for item in iter {
        let z = *item.borrow();
        if !z.is_finite() {
            return Err(CoreError::MathError {
                message: "ロジットに非有限値(NaNまたは無限大)が含まれています。".to_string(),
            });
        }

        count += 1;
        if count == 1 {
            max_logit = z;
            sum_exp = 1.0;
        } else if z > max_logit {
            let scale = ((max_logit - z) / effective_tau).exp();
            sum_exp = sum_exp * scale + 1.0;
            max_logit = z;
        } else {
            let scaled = ((z - max_logit) / effective_tau).exp();
            sum_exp += scaled;
        }
    }

    if count == 0 {
        return Err(CoreError::MathError {
            message: "ロジット配列が空です。".to_string(),
        });
    }

    if count == 1 {
        return Ok((-max_logit, 1));
    }

    let energy = -max_logit - effective_tau * sum_exp.ln();
    Ok((energy, count))
}

/// 任意の未正規化生ロジットのイテレータからヘルムホルツ自由エネルギー (Free Energy) を単一パス走査で算出する。
///
/// # 概要
/// - オンライン LogSumExp アルゴリズムにより、単一パス走査かつゼロアロケーションで計算する。
/// - 指数オーバーフローを防ぐため、オンライン最大値追跡によるスケーリングを適用する。
///
/// # 数理仕様
/// $$E(x; C) = -T \cdot \ln \sum_{i=1}^K \exp\left( \frac{z_i}{T} \right)$$
/// 有効温度 $T_{\text{eff}} = \max(T, 10^{-4})$ を適用する。
///
/// # 引数
/// - `iter`: 未正規化の生ロジット(`f64` または `&f64`)を提供するイテレータ。
/// - `temperature`: 較正温度パラメータ $T$。正の有限実数。
///
/// # 戻り値
/// - 成功時は自由エネルギー値 $E(x; C)$、入力不正時は `CoreError::MathError`。
pub fn free_energy_iter<I>(iter: I, temperature: f64) -> Result<f64>
where
    I: IntoIterator,
    I::Item: Borrow<f64>,
{
    let (energy, _) = free_energy_with_count_iter(iter, temperature)?;
    Ok(energy)
}

/// ロジット列からヘルムホルツ自由エネルギー (Free Energy) を算出する。
///
/// # 概要
/// - Softmax 正規化前の全ロジットの絶対的な発火強度を測定し、未定義カテゴリや未知の入力 (Out-of-Distribution: OOD) を検知するための指標である。
/// - 数値安定性を担保するため、最大値減算 (Max-trick) を適用して指数関数の浮動小数点オーバーフローを物理的に排除する。
/// - ヒープ割り当てを一切行わないゼロアロケーション走査で計算する。
///
/// # 数理仕様
/// $$E(x; C) = -T \cdot \ln \sum_{i=1}^K \exp\left( \frac{z_i}{T} \right) = -z_{\max} - T_{\text{eff}} \cdot \ln \sum_{i=1}^K \exp\left( \frac{z_i - z_{\max}}{T_{\text{eff}}} \right)$$
/// ここで、有効温度 $T_{\text{eff}} = \max(T, 10^{-4})$、最大ロジット $z_{\max} = \max_{1 \le i \le K} z_i$ である。
///
/// # 引数
/// - `logits`: 未正規化の生ロジットスライス (`f64`)。
/// - `temperature`: 較正温度パラメータ $T$。正の有限実数。
///
/// # 戻り値
/// - 成功時は自由エネルギー値 $E(x; C)$、入力不正時は `CoreError::MathError` を返却する。
///
/// # 備考
/// - [`free_energy_iter`] を呼び出すゼロコストラッパーとして動作する。
#[inline]
pub fn free_energy(logits: &[f64], temperature: f64) -> Result<f64> {
    free_energy_iter(logits, temperature)
}

/// 任意の未正規化生ロジットのイテレータから、候補数 $K$ に対する基底エネルギー偏向を補正した正規化自由エネルギーを単一パス走査で算出する。
///
/// # 概要
/// - 単一パス走査で自由エネルギーと候補数 $K$ を同時に集計し、ゼロアロケーションで補正エネルギーを算出する。
///
/// # 数理仕様
/// $$E_{\text{norm}}(x; C) = E(x; C) + T \ln K = -T \cdot \ln\left( \frac{1}{K} \sum_{i=1}^K \exp\left(\frac{z_i}{T}\right) \right)$$
///
/// # 引数
/// - `iter`: 未正規化の生ロジット(`f64` または `&f64`)を提供するイテレータ。
/// - `temperature`: 較正温度パラメータ $T$。正の有限実数。
///
/// # 戻り値
/// - 成功時は正規化自由エネルギー値 $E_{\text{norm}}(x; C)$、入力不正時は `CoreError::MathError`。
pub fn normalized_free_energy_iter<I>(iter: I, temperature: f64) -> Result<f64>
where
    I: IntoIterator,
    I::Item: Borrow<f64>,
{
    let (raw_energy, k) = free_energy_with_count_iter(iter, temperature)?;
    if k <= 1 {
        return Ok(raw_energy);
    }

    let effective_tau = temperature.max(1e-4);
    let shift = effective_tau * (k as f64).ln();
    Ok(raw_energy + shift)
}

/// 候補数 $K$ に対する基底エネルギー偏向を補正した、正規化ヘルムホルツ自由エネルギー (Normalized Free Energy) を算出する。
///
/// # 概要
/// - 単純な自由エネルギー $E(x; C) = -T \ln \sum \exp(z_i/T)$ は、候補数 $K$ が増加するだけで理論的に自然低下 (約 $-T \ln K$) する。
/// - sokuto では候補数 $K \in [1, 255]$ が動的に変動するため、候補数で補正した正規化自由エネルギーを採用することで、
///   任意の候補数に対して単一の閾値 $\tau_{\text{energy}}$ で安定した境界判定を可能にする。
/// - すべてのロジットが同一の定数 $c$ (完全な一様分布) であるとき、本関数の出力は厳密に $-c$ (ロジットが0のときは0.0) となる。
///
/// # 数理仕様
/// $$E_{\text{norm}}(x; C) = E(x; C) + T \ln K = -T \cdot \ln\left( \frac{1}{K} \sum_{i=1}^K \exp\left(\frac{z_i}{T}\right) \right) = -z_{\max} - T_{\text{eff}} \cdot \ln\left( \frac{1}{K} \sum_{i=1}^K \exp\left(\frac{z_i - z_{\max}}{T_{\text{eff}}} \right) \right)$$
///
/// # 引数
/// - `logits`: 未正規化の生ロジットスライス (`f64`)。
/// - `temperature`: 較正温度パラメータ $T$。正の有限実数。
///
/// # 戻り値
/// - 成功時は正規化自由エネルギー値 $E_{\text{norm}}(x; C)$、入力不正時は `CoreError::MathError` を返却する。
///
/// # 備考
/// - [`normalized_free_energy_iter`] を呼び出すゼロコストラッパーとして動作する。
#[inline]
pub fn normalized_free_energy(logits: &[f64], temperature: f64) -> Result<f64> {
    normalized_free_energy_iter(logits, temperature)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let logits = vec![2.0, 1.0, 0.1];
        let probs = softmax(&logits, 1.0).unwrap();
        assert_eq!(probs.len(), 3);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
    }

    #[test]
    fn test_softmax_into_basic() {
        let logits = vec![2.0, 1.0, 0.1];
        let mut probs = vec![0.0; 3];
        softmax_into(&logits, &mut probs, 1.0).unwrap();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(probs[0] > probs[1]);
        assert!(probs[1] > probs[2]);
    }

    #[test]
    fn test_softmax_extreme_logits() {
        // 巨大なロジット差でもオーバーフロー・アンダーフローでクラッシュしないことを検証する。
        let logits = vec![10000.0, 0.0, -10000.0];
        let probs = softmax(&logits, 1.0).unwrap();
        assert_eq!(probs.len(), 3);
        assert!((probs[0] - 1.0).abs() < 1e-6);
        assert_eq!(probs[1], 0.0);
        assert_eq!(probs[2], 0.0);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_identical_logits() {
        // 全要素が同一ロジットの場合、厳密に一様分布となることを検証する。
        let logits = vec![5.0, 5.0, 5.0, 5.0];
        let probs = softmax(&logits, 1.0).unwrap();
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-6);
        }
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_single_candidate() {
        // K=1 のエッジケースにおいて、即座に 1.0 が返却されることを検証する。
        let logits = vec![std::f64::consts::PI];
        let probs = softmax(&logits, 0.5).unwrap();
        assert_eq!(probs.len(), 1);
        assert_eq!(probs[0], 1.0);

        let mut out = vec![0.0];
        softmax_into(&logits, &mut out, 2.0).unwrap();
        assert_eq!(out[0], 1.0);
    }

    #[test]
    fn test_softmax_temperature_limits() {
        let logits = vec![2.0, 1.0, 0.0];

        // 高温 (tau -> inf): 一様分布へ平滑化される。
        let probs_high = softmax(&logits, 1000.0).unwrap();
        for &p in &probs_high {
            assert!((p - 1.0 / 3.0).abs() < 1e-2);
        }

        // 低温 (tau -> 0): 最大値へ確率が 1.0 集中する (One-hot)。
        let probs_low = softmax(&logits, 1e-4).unwrap();
        assert!((probs_low[0] - 1.0).abs() < 1e-6);
        assert!(probs_low[1] < 1e-6);
        assert!(probs_low[2] < 1e-6);

        // 安全下限クランプ (tau = 1e-9): ゼロ除算・オーバーフローなく計算できる。
        let probs_micro = softmax(&logits, 1e-9).unwrap();
        assert!((probs_micro[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_errors() {
        // 空ロジット
        let empty_logits: Vec<f64> = vec![];
        assert!(softmax(&empty_logits, 1.0).is_err());

        // 出力バッファ長不一致
        let logits = vec![1.0, 2.0];
        let mut short_out = vec![0.0];
        assert!(softmax_into(&logits, &mut short_out, 1.0).is_err());

        // 不正な温度 (tau <= 0, NaN, Inf)
        assert!(softmax(&logits, 0.0).is_err());
        assert!(softmax(&logits, -1.0).is_err());
        assert!(softmax(&logits, f64::NAN).is_err());
        assert!(softmax(&logits, f64::INFINITY).is_err());

        // ロジットに NaN または Inf
        let nan_logits = vec![1.0, f64::NAN, 3.0];
        assert!(softmax(&nan_logits, 1.0).is_err());
        let inf_logits = vec![1.0, f64::INFINITY, 3.0];
        assert!(softmax(&inf_logits, 1.0).is_err());
        let neg_inf_logits = vec![f64::NEG_INFINITY, 1.0];
        assert!(softmax(&neg_inf_logits, 1.0).is_err());
    }

    #[test]
    fn test_normalized_entropy_confidence() {
        // 完全な一様分布 -> 確信度は 0.0
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let conf_uniform = normalized_entropy_confidence(&uniform).unwrap();
        assert!(conf_uniform < 1e-6);

        // 完全にピークした分布 -> 確信度は 1.0
        let peaked = vec![1.0, 0.0, 0.0, 0.0];
        let conf_peaked = normalized_entropy_confidence(&peaked).unwrap();
        assert!((conf_peaked - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_expected_score() {
        let probs = vec![0.1, 0.2, 0.7]; // k=0, 1, 2
        let score = expected_score(&probs).unwrap();
        let expected = 0.0 * 0.1 + 1.0 * 0.2 + 2.0 * 0.7;
        assert!((score - expected).abs() < 1e-6);

        // 異常系: 段階数不足
        assert!(expected_score(&[]).is_err());
        assert!(expected_score(&[1.0]).is_err());

        // 異常系: 負の確率値
        assert!(expected_score(&[-0.1, 1.1]).is_err());
        assert!(expected_score(&[0.5, -0.5]).is_err());
    }

    #[test]
    fn test_expected_score_iter() {
        let probs = vec![0.1, 0.2, 0.7];
        let expected = 0.0 * 0.1 + 1.0 * 0.2 + 2.0 * 0.7;

        // イテレータ (参照、クローン、所有権) での一致検証
        assert!((expected_score_iter(probs.iter()).unwrap() - expected).abs() < 1e-12);
        assert!((expected_score_iter(probs.clone()).unwrap() - expected).abs() < 1e-12);
        assert!((expected_score_iter(probs.into_iter()).unwrap() - expected).abs() < 1e-12);

        // 異常系: 段階数不足
        let empty: [f64; 0] = [];
        assert!(expected_score_iter(empty).is_err());
        assert!(expected_score_iter([1.0]).is_err());

        // 異常系: 非有限値・負値
        assert!(expected_score_iter([0.5, f64::NAN]).is_err());
        assert!(expected_score_iter([f64::INFINITY, 0.5]).is_err());
        assert!(expected_score_iter([-0.1, 1.1]).is_err());
    }

    #[test]
    fn test_score_non_finite_inputs() {
        // 先頭、中間、末尾に非有限値(NaN, +Inf, -Inf)が含まれる場合の捕捉を検証する。
        let test_cases = vec![
            vec![f64::NAN, 0.5, 0.5],
            vec![0.5, f64::NAN, 0.5],
            vec![0.5, 0.5, f64::NAN],
            vec![f64::INFINITY, 0.5, 0.5],
            vec![0.5, f64::INFINITY, 0.5],
            vec![0.5, 0.5, f64::INFINITY],
            vec![f64::NEG_INFINITY, 0.5, 0.5],
            vec![0.5, f64::NEG_INFINITY, 0.5],
            vec![0.5, 0.5, f64::NEG_INFINITY],
        ];

        for (i, case) in test_cases.iter().enumerate() {
            assert!(
                expected_score(case).is_err(),
                "expected_score ケース {i} で非有限値がすり抜けました: {case:?}。"
            );
            assert!(
                score_variance(case).is_err(),
                "score_variance ケース {i} で非有限値がすり抜けました: {case:?}。"
            );
            assert!(
                bimodality_coefficient(case).is_err(),
                "bimodality_coefficient ケース {i} で非有限値がすり抜けました: {case:?}。"
            );
            assert!(
                normalized_variance_confidence(case).is_err(),
                "normalized_variance_confidence ケース {i} で非有限値がすり抜けました: {case:?}。"
            );
            assert!(
                score_variance_ratio(case).is_err(),
                "score_variance_ratio ケース {i} で非有限値がすり抜けました: {case:?}。"
            );
        }
    }

    #[test]
    fn test_noul_probability() {
        let prob_equal = noul_probability(1.0, 1.0, 1.0).unwrap();
        assert!((prob_equal - 0.5).abs() < 1e-6);

        let prob_high = noul_probability(10.0, 0.0, 1.0).unwrap();
        assert!(prob_high > 0.99);

        // 不正な温度
        assert!(noul_probability(1.0, 0.0, 0.0).is_err());
        assert!(noul_probability(1.0, 0.0, -1.0).is_err());
        assert!(noul_probability(1.0, 0.0, f64::NAN).is_err());
    }

    #[test]
    fn test_argmax() {
        // 基本的な最大値探索
        let values = vec![0.1, 0.8, 0.4];
        assert_eq!(argmax(&values).unwrap(), 1);

        // 決定論的タイブレーク(同率最大値の場合は先頭インデックスを採択)
        let tied = vec![0.5, 0.9, 0.9, 0.2];
        assert_eq!(argmax(&tied).unwrap(), 1);

        // 先頭が最大値の場合
        let first_max = vec![10.0, 5.0, 2.0];
        assert_eq!(argmax(&first_max).unwrap(), 0);

        // 末尾が最大値の場合
        let last_max = vec![1.0, 2.0, 3.0];
        assert_eq!(argmax(&last_max).unwrap(), 2);

        // 異常系: 空スライス
        let empty: Vec<f64> = vec![];
        assert!(argmax(&empty).is_err());

        // 異常系: NaN または Inf
        let nan_vals = vec![1.0, f64::NAN, 2.0];
        assert!(argmax(&nan_vals).is_err());
        let inf_vals = vec![1.0, f64::INFINITY];
        assert!(argmax(&inf_vals).is_err());
    }

    #[test]
    fn test_score_variance_and_confidence() {
        // 1. ワンホット分布(単一レベルに集中): 分散 0.0、確信度 1.0
        let one_hot = vec![0.0, 0.0, 1.0, 0.0];
        let var_one_hot = score_variance(&one_hot).unwrap();
        let conf_one_hot = normalized_variance_confidence(&one_hot).unwrap();
        assert!(var_one_hot < 1e-6);
        assert!((conf_one_hot - 1.0).abs() < 1e-6);

        // 2. 両極端に分裂した二峰性分布 [0.5, 0.0, 0.0, 0.5] (M=4)
        // 最大可能分散 V_max = (4-1)^2 / 4 = 2.25
        // Mean = 0 * 0.5 + 3 * 0.5 = 1.5
        // Var = (0 - 1.5)^2 * 0.5 + (3 - 1.5)^2 * 0.5 = 2.25
        // 確信度 C_var = 1.0 - 2.25 / 2.25 = 0.0
        let bimodal = vec![0.5, 0.0, 0.0, 0.5];
        let var_bimodal = score_variance(&bimodal).unwrap();
        let conf_bimodal = normalized_variance_confidence(&bimodal).unwrap();
        assert!((var_bimodal - 2.25).abs() < 1e-6);
        assert!(conf_bimodal < 1e-6);

        // 3. 隣接段階に分裂した低分散分布 [0.0, 0.5, 0.5, 0.0] (M=4)
        // Mean = 1 * 0.5 + 2 * 0.5 = 1.5
        // Var = (1 - 1.5)^2 * 0.5 + (2 - 1.5)^2 * 0.5 = 0.25
        // 確信度 C_var = 1.0 - 0.25 / 2.25 = 1.0 - 1/9 ≈ 0.888889
        let adjacent = vec![0.0, 0.5, 0.5, 0.0];
        let var_adjacent = score_variance(&adjacent).unwrap();
        let conf_adjacent = normalized_variance_confidence(&adjacent).unwrap();
        assert!((var_adjacent - 0.25).abs() < 1e-6);
        assert!((conf_adjacent - (8.0 / 9.0)).abs() < 1e-6);

        // エントロピー確信度との対比検証:
        // bimodal と adjacent はカテゴリカルエントロピーが完全に同一だが、
        // 分散確信度では bimodal(0.0) と adjacent(0.889) で明確に区別される。
        let ent_bimodal = normalized_entropy_confidence(&bimodal).unwrap();
        let ent_adjacent = normalized_entropy_confidence(&adjacent).unwrap();
        assert!((ent_bimodal - ent_adjacent).abs() < 1e-6);
        assert!(conf_adjacent > conf_bimodal);

        // 異常系: 段階数が 2 未満
        let single_level = vec![1.0];
        assert!(score_variance(&single_level).is_err());
        assert!(normalized_variance_confidence(&single_level).is_err());
    }

    #[test]
    fn test_score_moments_and_iter_parity() {
        let test_distributions = vec![
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.5, 0.0, 0.0, 0.5],
            vec![0.0, 0.5, 0.5, 0.0],
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.7, 0.2, 0.1],
        ];

        for dist in test_distributions {
            // スライス版
            let slice_mean = expected_score(&dist).unwrap();
            let slice_var = score_variance(&dist).unwrap();
            let slice_conf = normalized_variance_confidence(&dist).unwrap();
            let slice_ratio = score_variance_ratio(&dist).unwrap();

            // moments 単一パス集計版
            let (m, moments_mean, moments_var) = score_moments_iter(dist.iter()).unwrap();
            let (m_slice, s_mean, s_var) = score_moments(&dist).unwrap();
            assert_eq!(m, dist.len());
            assert_eq!(m_slice, m);
            assert!((moments_mean - slice_mean).abs() < 1e-12);
            assert!((moments_var - slice_var).abs() < 1e-12);
            assert!((s_mean - slice_mean).abs() < 1e-12);
            assert!((s_var - slice_var).abs() < 1e-12);

            let conf_from_var = normalized_variance_confidence_from_variance(m, moments_var);
            assert!((conf_from_var - slice_conf).abs() < 1e-12);

            // 各種 iter 版の一致検証
            let iter_mean = expected_score_iter(dist.iter()).unwrap();
            let iter_var = score_variance_iter(dist.iter()).unwrap();
            let iter_conf = normalized_variance_confidence_iter(dist.iter()).unwrap();
            let iter_ratio = score_variance_ratio_iter(dist.iter()).unwrap();

            assert!((iter_mean - slice_mean).abs() < 1e-12);
            assert!((iter_var - slice_var).abs() < 1e-12);
            assert!((iter_conf - slice_conf).abs() < 1e-12);
            assert!((iter_ratio - slice_ratio).abs() < 1e-12);
        }

        // 非正規化重み配列に対する加重分散の正規化検証
        // 重みの総和が 1.0 でない場合でも、分散が真の加重分散として正規化されることを検証する。
        let unnormalized_scaled_2x = vec![1.0, 0.0, 0.0, 1.0]; // 総和 2.0 (両端 0 と 3 に均等配分)
        let unnormalized_scaled_half = vec![0.25, 0.0, 0.0, 0.25]; // 総和 0.5 (両端 0 と 3 に均等配分)
        let normalized = vec![0.5, 0.0, 0.0, 0.5]; // 総和 1.0
        let (_, mean_norm, var_norm) = score_moments_iter(&normalized).unwrap();
        let (_, mean_2x, var_2x) = score_moments_iter(&unnormalized_scaled_2x).unwrap();
        let (_, mean_half, var_half) = score_moments_iter(&unnormalized_scaled_half).unwrap();

        assert!((mean_2x - mean_norm).abs() < 1e-12);
        assert!((mean_half - mean_norm).abs() < 1e-12);
        assert!((var_2x - var_norm).abs() < 1e-12);
        assert!((var_half - var_norm).abs() < 1e-12);
        assert!((var_norm - 2.25).abs() < 1e-12);

        // 異常系: 段階数不足
        let empty: [f64; 0] = [];
        assert!(score_moments_iter(empty).is_err());
        assert!(score_variance_iter([1.0]).is_err());
        assert!(normalized_variance_confidence_iter([1.0]).is_err());
        assert!(score_variance_ratio_iter([1.0]).is_err());

        // 異常系: 非有限値・負値
        assert!(score_moments_iter([0.5, f64::NAN]).is_err());
        assert!(score_variance_iter([f64::INFINITY, 0.5]).is_err());
        assert!(normalized_variance_confidence_iter([-0.1, 1.1]).is_err());
        assert!(score_variance_ratio_iter([0.5, -0.5]).is_err());

        // 異常系: 全要素ゼロおよび微小総和(破綻分布)の拒絶
        assert!(score_moments_iter([0.0, 0.0, 0.0]).is_err());
        assert!(score_moments(&[0.0, 0.0]).is_err());
        assert!(expected_score(&[0.0, 0.0]).is_err());
        assert!(score_variance(&[0.0, 0.0]).is_err());
        assert!(normalized_variance_confidence(&[0.0, 0.0, 0.0, 0.0]).is_err());
        assert!(score_variance_ratio(&[0.0, 0.0]).is_err());
        assert!(score_moments_iter([1e-15, 1e-15]).is_err());
    }

    #[test]
    fn test_noul_stability_and_confidence() {
        // 極端なロジット差でもオーバーフロー・アンダーフローせず安定計算できることの検証
        let prob_huge_pos = noul_probability(10000.0, 0.0, 1.0).unwrap();
        assert!((prob_huge_pos - 1.0).abs() < 1e-6);

        let prob_huge_neg = noul_probability(-10000.0, 0.0, 1.0).unwrap();
        assert!(prob_huge_neg < 1e-6);

        // 確信度計算の検証
        // P = 0.5 -> 確信度 0.0
        assert!((noul_confidence(0.5).unwrap() - 0.0).abs() < 1e-6);
        // P = 1.0 -> 確信度 1.0
        assert!((noul_confidence(1.0).unwrap() - 1.0).abs() < 1e-6);
        // P = 0.0 -> 確信度 1.0
        assert!((noul_confidence(0.0).unwrap() - 1.0).abs() < 1e-6);
        // P = 0.8 -> 確信度 0.6
        assert!((noul_confidence(0.8).unwrap() - 0.6).abs() < 1e-6);

        // 異常系: 範囲外または非有限値
        assert!(noul_confidence(-0.1).is_err());
        assert!(noul_confidence(1.1).is_err());
        assert!(noul_confidence(f64::NAN).is_err());
    }

    #[test]
    fn test_softmax_non_finite_positions() {
        // 先頭、中間、末尾の各位置に非有限値(NaN, +Inf, -Inf)が注入された場合の捕捉を検証する。
        let test_cases = vec![
            vec![f64::NAN, 1.0, 2.0],
            vec![1.0, f64::NAN, 2.0],
            vec![1.0, 2.0, f64::NAN],
            vec![f64::INFINITY, 1.0, 2.0],
            vec![1.0, f64::INFINITY, 2.0],
            vec![1.0, 2.0, f64::INFINITY],
            vec![f64::NEG_INFINITY, 1.0, 2.0],
            vec![1.0, f64::NEG_INFINITY, 2.0],
            vec![1.0, 2.0, f64::NEG_INFINITY],
        ];

        for (i, case) in test_cases.iter().enumerate() {
            let res = softmax(case, 1.0);
            assert!(
                res.is_err(),
                "ケース {i} で非有限値がすり抜けました: {case:?}。"
            );
        }

        // K=1 における非有限値の遮断を検証する。
        assert!(softmax(&[f64::NAN], 1.0).is_err());
        assert!(softmax(&[f64::INFINITY], 1.0).is_err());
        assert!(softmax(&[f64::NEG_INFINITY], 1.0).is_err());

        // 全要素が -Inf の場合、指数計算で分母が 0 になるため安全にエラー捕捉されることを検証する。
        let all_neg_inf = vec![f64::NEG_INFINITY, f64::NEG_INFINITY];
        assert!(softmax(&all_neg_inf, 1.0).is_err());
    }

    #[test]
    fn test_softmax_mask_value_convergence() {
        // Python 側のパディングマスク値 -1e4 が混入した場合の挙動を検証する。
        // 有効ロジットに対してマスクされた位置の確率が厳密に 0.0 となり、
        // かつ有効要素間の確率比率が保たれることを確認する。
        let logits = vec![3.0, 1.0, -10000.0];
        let probs = softmax(&logits, 1.0).unwrap();
        assert_eq!(probs.len(), 3);

        // マスク位置は 0.0 にアンダーフローして安全に収束する。
        assert_eq!(probs[2], 0.0);

        // 有効要素の和が 1.0 になる。
        let valid_sum = probs[0] + probs[1];
        assert!((valid_sum - 1.0).abs() < 1e-6);

        // 3.0 と 1.0 の二値ソフトマックス理論値と一致する。
        let expected_ratio = (2.0_f64).exp(); // exp(3 - 1)
        let actual_ratio = probs[0] / probs[1];
        assert!((actual_ratio - expected_ratio).abs() < 1e-4);
    }

    #[test]
    fn test_argmax_non_finite_positions() {
        // argmax において先頭、中間、末尾に非有限値が含まれる場合にエラーとなることを検証する。
        let test_cases = vec![
            vec![f64::NAN, 0.5, 0.2],
            vec![0.5, f64::NAN, 0.2],
            vec![0.5, 0.2, f64::NAN],
            vec![f64::INFINITY, 0.5, 0.2],
            vec![0.5, f64::INFINITY, 0.2],
            vec![0.5, 0.2, f64::INFINITY],
            vec![f64::NEG_INFINITY, 0.5, 0.2],
            vec![0.5, f64::NEG_INFINITY, 0.2],
            vec![0.5, 0.2, f64::NEG_INFINITY],
        ];

        for (i, case) in test_cases.iter().enumerate() {
            assert!(
                argmax(case).is_err(),
                "argmax ケース {i} で非有限値がすり抜けました: {case:?}。"
            );
        }
    }

    #[test]
    fn test_noul_non_finite_inputs() {
        // noul_probability の真偽ロジットに非有限値が渡された場合の拒絶を検証する。
        assert!(noul_probability(f64::NAN, 0.0, 1.0).is_err());
        assert!(noul_probability(0.0, f64::NAN, 1.0).is_err());
        assert!(noul_probability(f64::INFINITY, 0.0, 1.0).is_err());
        assert!(noul_probability(0.0, f64::INFINITY, 1.0).is_err());
        assert!(noul_probability(f64::NEG_INFINITY, 0.0, 1.0).is_err());
        assert!(noul_probability(0.0, f64::NEG_INFINITY, 1.0).is_err());
    }

    #[test]
    fn test_normalized_entropy() {
        // 1. K=1 の特異点: 不確実性ゼロのため 0.0 を返却する。
        assert_eq!(normalized_entropy(&[1.0]).unwrap(), 0.0);

        // 2. 完全な一様分布: 理論値 H(p) = ln(K) により正規化エントロピーは厳密に 1.0 となる。
        let uniform2 = vec![0.5, 0.5];
        assert!((normalized_entropy(&uniform2).unwrap() - 1.0).abs() < 1e-6);

        let uniform4 = vec![0.25, 0.25, 0.25, 0.25];
        assert!((normalized_entropy(&uniform4).unwrap() - 1.0).abs() < 1e-6);

        // 3. ワンホット(完全確信)分布: H(p) = 0.0 により正規化エントロピーは 0.0 となる。
        let peaked = vec![1.0, 0.0, 0.0, 0.0];
        assert!(normalized_entropy(&peaked).unwrap() < 1e-6);

        // 4. 一般分布 (K=3)
        // p = [0.7, 0.2, 0.1]
        // H(p) = -(0.7 * ln 0.7 + 0.2 * ln 0.2 + 0.1 * ln 0.1) ≈ 0.80181855
        // H_norm = 0.80181855 / ln(3) ≈ 0.80181855 / 1.09861229 ≈ 0.7298466
        let p3 = vec![0.7, 0.2, 0.1];
        let h_norm3 = normalized_entropy(&p3).unwrap();
        assert!((h_norm3 - 0.7298466).abs() < 1e-5);

        // 5. 異常系
        assert!(normalized_entropy(&[]).is_err());
        assert!(normalized_entropy(&[-0.1, 1.1]).is_err());
        assert!(normalized_entropy(&[0.5, f64::NAN]).is_err());
        assert!(normalized_entropy(&[f64::INFINITY, 0.5]).is_err());
        assert!(normalized_entropy(&[f64::NAN]).is_err());
    }

    #[test]
    fn test_normalized_entropy_iter() {
        // 1. K=1 の特異点: 不確実性ゼロのため 0.0 を返却する。
        assert_eq!(normalized_entropy_iter([1.0]).unwrap(), 0.0);
        assert_eq!(normalized_entropy_iter(&[1.0][..]).unwrap(), 0.0);

        // 2. イテレータ (値および参照) の透過的サポート。
        let uniform2 = vec![0.5, 0.5];
        assert!((normalized_entropy_iter(uniform2.iter()).unwrap() - 1.0).abs() < 1e-6);
        assert!((normalized_entropy_iter(uniform2.clone()).unwrap() - 1.0).abs() < 1e-6);
        assert!((normalized_entropy_iter(uniform2.into_iter()).unwrap() - 1.0).abs() < 1e-6);

        // 3. 空イテレータの異常系。
        let empty: [f64; 0] = [];
        assert!(normalized_entropy_iter(empty).is_err());
        assert!(normalized_entropy_iter(std::iter::empty::<f64>()).is_err());

        // 4. 非有限値・負値の異常系。
        assert!(normalized_entropy_iter([0.5, f64::NAN]).is_err());
        assert!(normalized_entropy_iter([-0.1, 0.8]).is_err());
    }

    #[test]
    fn test_top_margin() {
        // 1. K=1 の特異点: 競合候補が存在しないため 1.0 を返却する。
        assert_eq!(top_margin(&[1.0]).unwrap(), 1.0);

        // 2. 明確な差がある場合
        let p_clear = vec![0.7, 0.2, 0.1];
        assert!((top_margin(&p_clear).unwrap() - 0.5).abs() < 1e-6);

        // 3. 同率1位(タイ)の場合: マージンは 0.0 となる。
        let p_tied = vec![0.4, 0.4, 0.2];
        assert!(top_margin(&p_tied).unwrap() < 1e-6);

        let p_all_equal = vec![0.25, 0.25, 0.25, 0.25];
        assert!(top_margin(&p_all_equal).unwrap() < 1e-6);

        // 4. ワンホット極限: マージンは 1.0 となる。
        let p_onehot = vec![1.0, 0.0, 0.0];
        assert!((top_margin(&p_onehot).unwrap() - 1.0).abs() < 1e-6);

        // 5. 最大値が末尾にある場合
        let p_last = vec![0.1, 0.2, 0.7];
        assert!((top_margin(&p_last).unwrap() - 0.5).abs() < 1e-6);

        // 6. 異常系
        assert!(top_margin(&[]).is_err());
        assert!(top_margin(&[-0.5, 1.5]).is_err());
        assert!(top_margin(&[0.5, f64::NAN]).is_err());
        assert!(top_margin(&[f64::INFINITY, 0.0]).is_err());
        assert!(top_margin(&[f64::NAN]).is_err());
    }

    #[test]
    fn test_top_margin_iter() {
        // 1. K=1 の特異点: 競合候補が存在しないため 1.0 を返却する。
        assert_eq!(top_margin_iter([1.0]).unwrap(), 1.0);
        assert_eq!(top_margin_iter(&[1.0][..]).unwrap(), 1.0);

        // 2. イテレータ (値および参照) の透過的サポート。
        let p_clear = vec![0.7, 0.2, 0.1];
        assert!((top_margin_iter(p_clear.iter()).unwrap() - 0.5).abs() < 1e-6);
        assert!((top_margin_iter(p_clear.clone()).unwrap() - 0.5).abs() < 1e-6);
        assert!((top_margin_iter(p_clear.into_iter()).unwrap() - 0.5).abs() < 1e-6);

        // 3. 空イテレータの異常系。
        let empty: [f64; 0] = [];
        assert!(top_margin_iter(empty).is_err());
        assert!(top_margin_iter(std::iter::empty::<f64>()).is_err());

        // 4. 非有限値・負値の異常系。
        assert!(top_margin_iter([0.5, f64::NAN]).is_err());
        assert!(top_margin_iter([-0.1, 0.8]).is_err());
    }

    #[test]
    fn test_composite_confidence() {
        // 1. K=1 の特異点: 完全に決定しているため 1.0 を返却する。
        assert_eq!(composite_confidence(&[1.0]).unwrap(), 1.0);

        // 2. ワンホット極限: H_norm = 0.0, Margin = 1.0 -> S_conf = 1.0
        let p_onehot = vec![1.0, 0.0, 0.0];
        assert!((composite_confidence(&p_onehot).unwrap() - 1.0).abs() < 1e-6);

        // 3. 完全一様分布極限: H_norm = 1.0, Margin = 0.0 -> S_conf = 0.0
        let p_uniform = vec![0.25, 0.25, 0.25, 0.25];
        assert!(composite_confidence(&p_uniform).unwrap() < 1e-6);

        // 4. Margin Collapse (上位2候補激突):
        // p = [0.49, 0.49, 0.02]
        // 分布全体としてはそこそこ尖っているが、Margin = 0.0 のため S_conf は 0.0 となる。
        let p_collapse = vec![0.49, 0.49, 0.02];
        assert!(composite_confidence(&p_collapse).unwrap() < 1e-6);

        // 5. 高確信判定ケース:
        // p = [0.85, 0.10, 0.05]
        // H(p) = -(0.85*ln 0.85 + 0.1*ln 0.1 + 0.05*ln 0.05) ≈ 0.518
        // H_norm = 0.518 / ln(3) ≈ 0.518 / 1.0986 ≈ 0.4715
        // Margin = 0.85 - 0.10 = 0.75
        // S_conf = (1.0 - 0.4715) * 0.75 ≈ 0.396
        let p_high = vec![0.85, 0.10, 0.05];
        let s_conf = composite_confidence(&p_high).unwrap();
        assert!(s_conf > 0.35 && s_conf < 0.45);

        // 6. 圧倒的確信ケース:
        // p = [0.98, 0.01, 0.01]
        // Margin = 0.97
        // 1 - H_norm ≈ 0.89
        // S_conf ≈ 0.86
        let p_extreme = vec![0.98, 0.01, 0.01];
        let s_extreme = composite_confidence(&p_extreme).unwrap();
        assert!(s_extreme > 0.80);

        // 7. 異常系
        assert!(composite_confidence(&[]).is_err());
        assert!(composite_confidence(&[-0.1, 1.1]).is_err());
        assert!(composite_confidence(&[0.5, f64::NAN]).is_err());
    }

    #[test]
    fn test_composite_confidence_iter() {
        // 1. K=1 の特異点: 競合候補が存在しないため 1.0 を返却する。
        assert_eq!(composite_confidence_iter([1.0]).unwrap(), 1.0);
        assert_eq!(composite_confidence_iter(&[1.0][..]).unwrap(), 1.0);

        // 2. イテレータ (値および参照) の透過的サポート。
        let p_high = vec![0.85, 0.10, 0.05];
        let expected = composite_confidence(&p_high).unwrap();
        assert!((composite_confidence_iter(p_high.iter()).unwrap() - expected).abs() < 1e-12);
        assert!((composite_confidence_iter(p_high.clone()).unwrap() - expected).abs() < 1e-12);
        assert!((composite_confidence_iter(p_high.into_iter()).unwrap() - expected).abs() < 1e-12);

        // 3. 空イテレータの異常系。
        let empty: [f64; 0] = [];
        assert!(composite_confidence_iter(empty).is_err());
        assert!(composite_confidence_iter(std::iter::empty::<f64>()).is_err());

        // 4. 非有限値・負値の異常系。
        assert!(composite_confidence_iter([0.5, f64::NAN]).is_err());
        assert!(composite_confidence_iter([f64::INFINITY, 0.5]).is_err());
        assert!(composite_confidence_iter([-0.1, 0.8]).is_err());

        // 5. スライス版とイテレータ版の完全一致検証。
        let distributions = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.49, 0.49, 0.02],
            vec![0.7, 0.2, 0.1],
            vec![0.98, 0.01, 0.01],
            vec![0.1, 0.2, 0.3, 0.4],
        ];
        for dist in distributions {
            let slice_res = composite_confidence(&dist).unwrap();
            let iter_res = composite_confidence_iter(dist.iter()).unwrap();
            assert!(
                (slice_res - iter_res).abs() < 1e-12,
                "スライス版とイテレータ版で計算結果が不一致です: {dist:?}。"
            );
        }
    }

    #[test]
    fn test_bimodality_coefficient() {
        // 1. ワンホット極限: 分散ゼロのため 0.0 を返却する。
        let p_onehot = vec![1.0, 0.0, 0.0, 0.0];
        assert_eq!(bimodality_coefficient(&p_onehot).unwrap(), 0.0);

        // 2. 完全両端対立 (二峰性): [0.5, 0.0, 0.0, 0.5]
        // 平均 1.5, mu_2 = 2.25, mu_3 = 0.0, mu_4 = 5.0625, kurtosis = 1.0 -> BC = 1.0
        let p_bimodal = vec![0.5, 0.0, 0.0, 0.5];
        let bc_bimodal = bimodality_coefficient(&p_bimodal).unwrap();
        assert!((bc_bimodal - 1.0).abs() < 1e-6);

        // 3. 5段階の両端対立: [0.5, 0.0, 0.0, 0.0, 0.5]
        let p_bimodal_5 = vec![0.5, 0.0, 0.0, 0.0, 0.5];
        let bc_bimodal_5 = bimodality_coefficient(&p_bimodal_5).unwrap();
        assert!((bc_bimodal_5 - 1.0).abs() < 1e-6);

        // 4. 一様分布: 離散一様分布では M に応じて 0.555〜0.65 付近となる (連続極限で 5/9 ≈ 0.555)。
        let p_uniform_4 = vec![0.25, 0.25, 0.25, 0.25];
        let bc_uniform_4 = bimodality_coefficient(&p_uniform_4).unwrap();
        assert!(bc_uniform_4 > 0.55 && bc_uniform_4 < 0.65);

        let p_uniform_10 = vec![0.1; 10];
        let bc_uniform_10 = bimodality_coefficient(&p_uniform_10).unwrap();
        assert!(bc_uniform_10 > 0.55 && bc_uniform_10 < 0.60);

        // 5. 単峰山型分布: [0.05, 0.2, 0.5, 0.2, 0.05] -> BC は 0.555 未満
        let p_unimodal = vec![0.05, 0.2, 0.5, 0.2, 0.05];
        let bc_unimodal = bimodality_coefficient(&p_unimodal).unwrap();
        assert!(bc_unimodal < 0.555);

        // 6. 隣接割れ (低分散): [0.0, 0.5, 0.5, 0.0]
        // 平均 1.5, 分散 0.25, kurtosis = 1.0 -> BC = 1.0 だが分散比率は極めて小さい
        let p_adjacent = vec![0.0, 0.5, 0.5, 0.0];
        let var_ratio = score_variance_ratio(&p_adjacent).unwrap();
        assert!(var_ratio < 0.20);

        // 7. 非正規化重み配列に対する双峰性係数のスケール不変性検証
        // 確率の総和が 1.0 でない場合でも、高次中心モーメントの正規化により BC が厳密に一致することを検証する。
        let p_asymmetric = vec![0.1, 0.2, 0.4, 0.3];
        let p_scaled_2x = vec![0.2, 0.4, 0.8, 0.6];
        let p_scaled_half = vec![0.05, 0.1, 0.2, 0.15];
        let bc_norm = bimodality_coefficient(&p_asymmetric).unwrap();
        let bc_2x = bimodality_coefficient(&p_scaled_2x).unwrap();
        let bc_half = bimodality_coefficient(&p_scaled_half).unwrap();
        assert!((bc_2x - bc_norm).abs() < 1e-12);
        assert!((bc_half - bc_norm).abs() < 1e-12);

        // 8. 異常系
        assert!(bimodality_coefficient(&[]).is_err());
        assert!(bimodality_coefficient(&[1.0]).is_err());
        assert!(bimodality_coefficient(&[-0.1, 1.1]).is_err());
    }

    #[test]
    fn test_score_variance_ratio() {
        // 1. 完全両端対立: 理論最大分散と一致するため 1.0
        let p_bimodal = vec![0.5, 0.0, 0.0, 0.5];
        let ratio = score_variance_ratio(&p_bimodal).unwrap();
        assert!((ratio - 1.0).abs() < 1e-6);

        // 2. ワンホット: 分散 0.0
        let p_onehot = vec![0.0, 1.0, 0.0, 0.0];
        let ratio_onehot = score_variance_ratio(&p_onehot).unwrap();
        assert!(ratio_onehot < 1e-6);

        // 3. 異常系
        assert!(score_variance_ratio(&[]).is_err());
        assert!(score_variance_ratio(&[1.0]).is_err());
    }

    #[test]
    fn test_free_energy_basic() {
        let logits = vec![2.0, 1.0, 0.0];
        let t = 1.0;
        let fe = free_energy(&logits, t).unwrap();
        // 理論値: -ln(exp(2) + exp(1) + exp(0)) = -ln(7.389056 + 2.718282 + 1.0) = -ln(11.107338) ≈ -2.4076
        let expected = -((2.0f64).exp() + (1.0f64).exp() + (0.0f64).exp()).ln();
        assert!((fe - expected).abs() < 1e-9);

        // K=1 の場合は -z_0
        let fe_single = free_energy(&[3.5], 1.0).unwrap();
        assert!((fe_single - (-3.5)).abs() < 1e-9);
    }

    #[test]
    fn test_normalized_free_energy_uniform() {
        // 全ロジットが同一 c の場合、E_norm は厳密に -c に一致する。
        for &k in &[2, 3, 5, 10, 50] {
            let logits_zero = vec![0.0; k];
            let nfe_zero = normalized_free_energy(&logits_zero, 1.0).unwrap();
            assert!(
                nfe_zero.abs() < 1e-9,
                "k={k} でロジットが 0 のとき E_norm は 0.0 であるべき"
            );

            let c = 2.5;
            let logits_const = vec![c; k];
            let nfe_const = normalized_free_energy(&logits_const, 1.2).unwrap();
            assert!(
                (nfe_const - (-c)).abs() < 1e-9,
                "k={k} でロジットが c のとき E_norm は -c であるべき"
            );
        }
    }

    #[test]
    fn test_free_energy_extreme_logits() {
        // 大きなロジットでも Max-trick によりオーバーフローせずに計算可能。
        let logits = vec![1000.0, 999.0, 998.0];
        let fe = free_energy(&logits, 1.0).unwrap();
        assert!(fe.is_finite());
        // E = -1000 - ln(1 + exp(-1) + exp(-2)) ≈ -1000 - 0.4076 = -1000.4076
        assert!(fe < -999.0);

        let nfe = normalized_free_energy(&logits, 1.0).unwrap();
        assert!(nfe.is_finite());
        // nfe = fe + ln(3) ≈ -1000.4076 + 1.0986 = -999.309
        assert!(nfe < -998.0);
    }

    #[test]
    fn test_free_energy_ood_vs_id() {
        let t = 1.0;
        // In-Distribution: 1つの候補が強く発火 (低エネルギー = 負の大きな値)
        let id_logits = vec![5.0, 0.2, -0.5];
        let id_energy = normalized_free_energy(&id_logits, t).unwrap();
        assert!(id_energy < -3.5);

        // Out-of-Distribution: 全ての候補が低く沈んでいる (高エネルギー = 正または0近傍の大きな値)
        let ood_logits = vec![-4.0, -4.5, -5.0];
        let ood_energy = normalized_free_energy(&ood_logits, t).unwrap();
        assert!(ood_energy > 3.0);

        // OOD のエネルギーは ID より遥かに高い
        assert!(ood_energy > id_energy + 6.0);
    }

    #[test]
    fn test_free_energy_errors_and_edge_cases() {
        // 空配列
        assert!(free_energy(&[], 1.0).is_err());
        assert!(normalized_free_energy(&[], 1.0).is_err());

        // 温度が 0 以下または非有限
        assert!(free_energy(&[1.0, 2.0], 0.0).is_err());
        assert!(free_energy(&[1.0, 2.0], -1.0).is_err());
        assert!(free_energy(&[1.0, 2.0], f64::NAN).is_err());

        // ロジットに非有限値
        assert!(free_energy(&[1.0, f64::NAN], 1.0).is_err());
        assert!(free_energy(&[1.0, f64::INFINITY], 1.0).is_err());
    }

    #[test]
    fn test_bimodality_coefficient_iter_parity() {
        use indexmap::IndexMap;

        // 1. IndexMap からのゼロアロケーション走査
        let mut map = IndexMap::new();
        map.insert("cat_0".to_string(), 0.5);
        map.insert("cat_1".to_string(), 0.0);
        map.insert("cat_2".to_string(), 0.0);
        map.insert("cat_3".to_string(), 0.5);

        let slice = [0.5, 0.0, 0.0, 0.5];
        let bc_slice = bimodality_coefficient(&slice).unwrap();
        let bc_iter = bimodality_coefficient_iter(map.values()).unwrap();
        assert_eq!(bc_slice, bc_iter);

        // 2. 大規模要素数 (50要素) における単一パス計算とスライス計算の一致検証
        let large_p: Vec<f64> = (0..50).map(|_| 1.0 / 50.0).collect();
        let bc_large_slice = bimodality_coefficient(&large_p).unwrap();
        let bc_large_iter = bimodality_coefficient_iter(large_p.iter()).unwrap();
        assert!((bc_large_slice - bc_large_iter).abs() < 1e-9);
    }

    #[test]
    fn test_free_energy_iter_parity() {
        use indexmap::IndexMap;

        let mut map = IndexMap::new();
        map.insert("k0".to_string(), 2.0);
        map.insert("k1".to_string(), 1.0);
        map.insert("k2".to_string(), 0.0);

        let slice = [2.0, 1.0, 0.0];
        let t = 1.0;

        let fe_slice = free_energy(&slice, t).unwrap();
        let fe_iter = free_energy_iter(map.values(), t).unwrap();
        assert!((fe_slice - fe_iter).abs() < 1e-9);

        let nfe_slice = normalized_free_energy(&slice, t).unwrap();
        let nfe_iter = normalized_free_energy_iter(map.values(), t).unwrap();
        assert!((nfe_slice - nfe_iter).abs() < 1e-9);
    }
}
