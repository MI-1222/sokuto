"""長系列拡張および Late Chunking 事後プーリングモジュール。

ModernBERT の 1G+2L ハイブリッド注意構造において局所窓層 (128幅) を固定保護しながら、
大域注意層のみに YaRN (周波数帯域分割補間) を適用してコンテキスト長を 8,192〜16,384+ へ安全拡張する。
また、全文エンコード後の隠れ状態から条項・段落スパン単位で集約する Late Chunking プーリング層、
上位 k チャンクを選択して集約する Top-k Cross-Attention、
および契約書・規程の全条項を一括真偽判定する並列 Noul スキャナーを提供する。
"""

import logging
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

DEFAULT_MASK_VALUE = -1e4


class YaRNScaledRotaryEmbedding(nn.Module):
    """ModernBERT の Global Layer に適用する YaRN RoPE スケーリングモジュール。

    周波数帯域分割補間 (YaRN) により、短波長 (高周波・局所順序) を変更せず維持し、
    長波長 (低周波・大域関係) のみをスケーリング倍率 s に応じて補間する。
    さらに系列拡張に伴うアテンションエントロピー爆発を防ぐ温度補正係数 mscale を適用する。

    数理仕様:
    $$s = L_{\\text{target}} / L_{\\text{original}}$$
    $$r(i) = L_{\\text{original}} / \\lambda_i$$
    $$\\gamma(r(i)) = \\mathrm{clamp}\\left(\\frac{r(i) - \\alpha}{\\beta - \\alpha}, 0, 1\\right)$$
    $$\\theta_i' = (1 - \\gamma(r(i))) \\frac{\\theta_i}{s} + \\gamma(r(i)) \\theta_i$$
    $$m = 0.1 \\ln(s) + 1.0 \\quad (s > 1.0)$$
    """

    inv_freq: Tensor

    def __init__(
        self,
        dim: int,
        original_max_position: int = 8192,
        target_max_position: int = 16384,
        base: float = 160000.0,
        alpha: float = 1.0,
        beta: float = 32.0,
    ) -> None:
        """YaRN RoPE スケーリングモジュールを初期化する。

        Args:
            dim (int): ヘッドあたりの RoPE 埋め込み次元数 (head_dim)。
            original_max_position (int): 事前学習時の最大系列長。
            target_max_position (int): 拡張後の目標最大系列長。
            base (float): RoPE の基底周波数 theta (大域層は 160,000.0)。
            alpha (float): 低周波補間の下限しきい値。
            beta (float): 高周波保護の上限しきい値。
        """
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim ({dim}) は偶数である必要があります。")
        if target_max_position < original_max_position:
            raise ValueError(
                f"target_max_position ({target_max_position}) は original_max_position "
                f"({original_max_position}) 以上である必要があります。"
            )

        self.dim = dim
        self.original_max_position = original_max_position
        self.target_max_position = target_max_position
        self.scale = float(target_max_position) / float(original_max_position)
        self.base = base
        self.alpha = alpha
        self.beta = beta

        inv_freq, mscale = self.compute_yarn_parameters(
            dim=dim,
            original_max_position=original_max_position,
            target_max_position=target_max_position,
            base=base,
            alpha=alpha,
            beta=beta,
        )
        self.register_buffer("inv_freq", inv_freq)
        self.mscale = mscale

    @staticmethod
    def compute_yarn_parameters(
        dim: int,
        original_max_position: int = 8192,
        target_max_position: int = 16384,
        base: float = 160000.0,
        alpha: float = 1.0,
        beta: float = 32.0,
        device: torch.device | None = None,
    ) -> tuple[Tensor, float]:
        """YaRN スケーリング後の周波数テーブルおよび温度補正係数を算出する。

        Args:
            dim (int): ヘッド次元数。
            original_max_position (int): 事前学習時系列長。
            target_max_position (int): 目標系列長。
            base (float): RoPE 基底値。
            alpha (float): 補間開始しきい値。
            beta (float): 補間完了しきい値。
            device (torch.device | None): 出力デバイス。

        Returns:
            tuple[Tensor, float]: (スケーリング後 inv_freq, 温度補正係数 mscale)。
        """
        scale = float(target_max_position) / float(original_max_position)
        pos = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (pos / dim))

        if scale <= 1.0:
            return inv_freq, 1.0

        wavelength = 2.0 * math.pi / inv_freq
        ratio = original_max_position / wavelength

        # 高周波 (ratio > beta): gamma=1 (未補間), 低周波 (ratio < alpha): gamma=0 (1/scale 補間)
        gamma = torch.clamp((ratio - alpha) / (beta - alpha), min=0.0, max=1.0)
        yarn_inv_freq = (1.0 - gamma) * (inv_freq / scale) + gamma * inv_freq
        mscale = 0.1 * math.log(scale) + 1.0

        return yarn_inv_freq, mscale

    def forward(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """指定系列長に対する Cosine および Sine 埋め込みを算出する。

        Args:
            seq_len (int): 計算対象のトークン系列長。
            device (torch.device): テンソル配置デバイス。

        Returns:
            tuple[Tensor, Tensor]: (cos テンソル [seq_len, dim], sin テンソル [seq_len, dim])。
        """
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.mscale
        sin = emb.sin() * self.mscale
        return cos, sin


def apply_yarn_to_modernbert(
    model: PreTrainedModel,
    target_max_position: int = 16384,
    alpha: float = 1.0,
    beta: float = 32.0,
) -> None:
    """ModernBERT モデルの大域注意層 (Global Layers) のみに YaRN RoPE スケーリングを安全適用する。

    ModernBERT の局所スライディングウィンドウ注意層 (128幅, local_rope_theta = 10,000) は、
    相対距離が常に 128 トークン以内であるため元の周波数を完全に固定保護する。
    大域注意層 (global_rope_theta = 160,000) のみ周波数帯域分割補間と温度補正 (mscale) を適用する。

    注意:
        rotary_emb の inv_freq バッファをインプレースで書き換えるため、
        モデルを Accelerator や DDP/FSDP、または特定デバイスへ転送 (.to(device)) する前に
        本関数を実行してください (初期化順序: バックボーン生成 -> YaRN 適用 -> Accelerator 準備)。

    Args:
        model (PreTrainedModel): 対象の ModernBERT バックボーンモデル。
        target_max_position (int): 拡張後の目標最大位置長 (デフォルト: 16,384)。
        alpha (float): YaRN 補間下限パラメータ。
        beta (float): YaRN 補間上限パラメータ。
    """

    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("モデルに config 属性が存在しません。")

    original_max_position = getattr(config, "max_position_embeddings", 8192)
    if target_max_position <= original_max_position:
        logger.info(
            "target_max_position (%d) <= original_max_position (%d) のため、YaRN スケーリングをスキップします。",
            target_max_position,
            original_max_position,
        )
        return

    # ModernBERT の rotary_emb 属性を探索
    rotary_emb = None
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        rotary_emb = model.model.rotary_emb
    elif hasattr(model, "rotary_emb"):
        rotary_emb = model.rotary_emb

    if rotary_emb is None:
        logger.warning(
            "モデル内に rotary_emb が検出されませんでした。YaRN のインプレース適用をスキップします。"
        )
        return

    full_inv_freq = getattr(rotary_emb, "full_attention_inv_freq", None)
    if isinstance(full_inv_freq, Tensor):
        dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        base = config.rope_parameters["full_attention"]["rope_theta"]
        curr_device = full_inv_freq.device

        yarn_inv_freq, mscale = YaRNScaledRotaryEmbedding.compute_yarn_parameters(
            dim=dim,
            original_max_position=original_max_position,
            target_max_position=target_max_position,
            base=base,
            alpha=alpha,
            beta=beta,
            device=curr_device,
        )

        with torch.no_grad():
            full_inv_freq.copy_(yarn_inv_freq)
            if hasattr(rotary_emb, "full_attention_attention_scaling"):
                object.__setattr__(
                    rotary_emb, "full_attention_attention_scaling", mscale
                )

        logger.info(
            "ModernBERT の Global Attention 層に YaRN を適用しました (original: %d -> target: %d, mscale: %.4f)。局所層は固定保護されています。",
            original_max_position,
            target_max_position,
            mscale,
        )
    else:
        logger.warning(
            "rotary_emb 内に full_attention_inv_freq が見つかりませんでした。"
        )

    # config の最大系列長を更新
    config.max_position_embeddings = target_max_position


class LateChunkingPooling(nn.Module):
    """全文隠れ状態から条項・段落スパンごとに事後集約するプーリング層。

    長文 State 全体を双方向エンコーダで一括処理した後の隠れ状態テンソル H から、
    各条項・段落の境界スパン [t_start, t_end] に対応するトークン表現を集約する。
    Mean Pooling (算術平均) および学習可能な Attention-Weighted Pooling に対応する。

    数理仕様:
    - Mean Pooling:
      $$v_{\\text{mean}}^{(k)} = \\frac{1}{t_{\\text{end}}^{(k)} - t_{\\text{start}}^{(k)} + 1} \\sum_{t=t_{\\text{start}}^{(k)}}^{t_{\\text{end}}^{(k)}} h_t$$
    - Attention-Weighted Pooling:
      $$\\alpha_t = \\frac{\\exp(w_{\\text{pool}}^T h_t / \\sqrt{d})}{\\sum_{j=t_{\\text{start}}^{(k)}}^{t_{\\text{end}}^{(k)}} \\exp(w_{\\text{pool}}^T h_j / \\sqrt{d})}$$
      $$v_{\\text{attn}}^{(k)} = \\sum_{t=t_{\\text{start}}^{(k)}}^{t_{\\text{end}}^{(k)}} \\alpha_t h_t$$
    """

    def __init__(
        self,
        hidden_size: int,
        pooling_strategy: Literal["mean", "attention"] = "mean",
    ) -> None:
        """LateChunkingPooling 層を初期化する。

        Args:
            hidden_size (int): 隠れ層次元数。
            pooling_strategy (Literal["mean", "attention"]): プーリング方式 ('mean' または 'attention')。
        """
        super().__init__()
        if pooling_strategy not in ("mean", "attention"):
            raise ValueError(
                f"未知の pooling_strategy: {pooling_strategy}。'mean' または 'attention' を指定してください。"
            )

        self.hidden_size = hidden_size
        self.pooling_strategy = pooling_strategy
        if pooling_strategy == "attention":
            self.query_proj = nn.Linear(hidden_size, 1)

    def forward(
        self,
        hidden_states: Tensor,
        chunk_spans: list[list[tuple[int, int]]] | None = None,
        chunk_starts: Tensor | None = None,
        chunk_ends: Tensor | None = None,
        chunk_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """全文隠れ状態を指定スパン境界に従ってチャンク表現へ集約する。

        Python のリスト形式 `chunk_spans`、またはテンソル形式 (`chunk_starts`, `chunk_ends`, `chunk_mask`)
        のいずれかを入力として受け付ける。

        Args:
            hidden_states (Tensor): 全文隠れ状態テンソル `[batch_size, seq_len, hidden_size]`。
            chunk_spans (list[list[tuple[int, int]]] | None): バッチ各サンプルの `[(start, end), ...]` スパンリスト。
            chunk_starts (Tensor | None): 開始トークンインデックス `[batch_size, max_chunks]`。
            chunk_ends (Tensor | None): 終了トークンインデックス `[batch_size, max_chunks]`。
            chunk_mask (Tensor | None): 有効チャンクマスク `[batch_size, max_chunks]` (True: 有効)。

        Returns:
            tuple[Tensor, Tensor]:
                - pooled_chunks: 集約チャンクテンソル `[batch_size, max_chunks, hidden_size]`。
                - out_chunk_mask: 有効チャンクマスク `[batch_size, max_chunks]` (bool)。
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        if chunk_spans is not None:
            max_chunks = (
                max(len(spans) for spans in chunk_spans) if batch_size > 0 else 0
            )
            if max_chunks == 0:
                empty_chunks = torch.zeros(
                    (batch_size, 0, self.hidden_size),
                    dtype=dtype,
                    device=device,
                )
                empty_mask = torch.zeros(
                    (batch_size, 0), dtype=torch.bool, device=device
                )
                return empty_chunks, empty_mask

            pooled_chunks = torch.zeros(
                (batch_size, max_chunks, self.hidden_size),
                dtype=dtype,
                device=device,
            )
            out_mask = torch.zeros(
                (batch_size, max_chunks), dtype=torch.bool, device=device
            )

            for b in range(batch_size):
                spans = chunk_spans[b]
                for c_idx, (start, end) in enumerate(spans):
                    s = max(0, min(start, seq_len - 1))
                    e = max(s, min(end, seq_len - 1))
                    span_h = hidden_states[b, s : e + 1, :]
                    if span_h.size(0) == 0:
                        continue

                    if self.pooling_strategy == "mean":
                        pooled_chunks[b, c_idx] = span_h.mean(dim=0)
                    elif self.pooling_strategy == "attention":
                        attn_logits = self.query_proj(span_h).squeeze(-1) / math.sqrt(
                            self.hidden_size
                        )
                        attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1)
                        pooled_chunks[b, c_idx] = (span_h * attn_weights).sum(dim=0)

                    out_mask[b, c_idx] = True

            return pooled_chunks, out_mask

        if chunk_starts is not None and chunk_ends is not None:
            max_chunks = chunk_starts.size(1)
            if chunk_mask is None:
                chunk_mask = torch.ones(
                    (batch_size, max_chunks), dtype=torch.bool, device=device
                )

            # バッチ行列積 (GEMM) による完全ベクトル化プーリング。
            # 位置インデックス [1, 1, L_seq] とスパン境界 [B, N_chunks, 1] のブロードキャスト比較。
            pos_indices = torch.arange(seq_len, device=device).view(1, 1, seq_len)
            starts = chunk_starts.clamp(min=0, max=seq_len - 1).unsqueeze(
                -1
            )  # [B, N, 1]
            ends = chunk_ends.clamp(min=0, max=seq_len - 1).unsqueeze(-1)  # [B, N, 1]
            ends = torch.maximum(starts, ends)

            span_mask = (pos_indices >= starts) & (pos_indices <= ends)  # [B, N, L]
            span_mask = span_mask & chunk_mask.unsqueeze(-1)  # [B, N, L]

            if self.pooling_strategy == "mean":
                span_counts = span_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                weights = span_mask.to(dtype) / span_counts  # [B, N, L]
                pooled_chunks = torch.bmm(weights, hidden_states)  # [B, N, D]
            elif self.pooling_strategy == "attention":
                attn_logits = self.query_proj(hidden_states).squeeze(-1) / math.sqrt(
                    self.hidden_size
                )  # [B, L]
                attn_logits = attn_logits.unsqueeze(1).expand(
                    -1, max_chunks, -1
                )  # [B, N, L]
                attn_logits = attn_logits.masked_fill(~span_mask, DEFAULT_MASK_VALUE)
                attn_weights = F.softmax(attn_logits, dim=-1)  # [B, N, L]
                # 無効チャンクのゼロ埋め。
                attn_weights = attn_weights * chunk_mask.unsqueeze(-1).to(dtype)
                pooled_chunks = torch.bmm(attn_weights, hidden_states)  # [B, N, D]

            return pooled_chunks, chunk_mask

        raise ValueError(
            "chunk_spans または (chunk_starts, chunk_ends) のいずれかを指定する必要があります。"
        )


class TopKChunkCrossAttention(nn.Module):
    """集約されたチャンク表現群から上位 k チャンクを選択して Cross-Attention を適用する層。

    長文ドキュメント全体の全トークンではなく、プーリング後の N_chunks 個のベクトルから
    質問 (Query) 表現と関連度の高い上位 k 個 (例: 5〜8 個) を動的に選択し、
    スパースに交差アテンションを適用することでアテンション希釈 (Attention Dilution) を物理的に防ぐ。
    """

    def __init__(
        self,
        hidden_size: int,
        top_k: int = 5,
        num_heads: int = 4,
        dropout: float = 0.0,
        eps: float = 1e-5,
    ) -> None:
        """Top-k チャンク交差アテンション層を初期化する。

        Args:
            hidden_size (int): 隠れ層次元数。
            top_k (int): 選択する上位チャンク数。
            num_heads (int): アテンションヘッド数。
            dropout (float): ドロップアウト率。
            eps (float): LayerNorm イプシロン。
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm = nn.LayerNorm(hidden_size, eps=eps)

    def select_topk_chunks(
        self,
        query_states: Tensor,
        chunk_states: Tensor,
        chunk_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """質問表現との類似度に基づき上位 k チャンクを抽出する。

        Args:
            query_states (Tensor): 質問表現 `[batch_size, query_len, hidden_size]`。
            chunk_states (Tensor): チャンク表現 `[batch_size, num_chunks, hidden_size]`。
            chunk_mask (Tensor | None): 有効チャンクマスク `[batch_size, num_chunks]`。

        Returns:
            tuple[Tensor, Tensor]: (選択されたチャンクテンソル `[batch_size, actual_k, hidden_size]`, 選択インデックス)。
        """
        batch_size, _, _ = query_states.shape
        num_chunks = chunk_states.size(1)
        actual_k = min(self.top_k, num_chunks)

        if actual_k == 0:
            empty = torch.zeros(
                (batch_size, 0, self.hidden_size),
                dtype=chunk_states.dtype,
                device=chunk_states.device,
            )
            empty_idx = torch.zeros(
                (batch_size, 0), dtype=torch.long, device=chunk_states.device
            )
            return empty, empty_idx

        # 質問全体の代表ベクトル (Mean Pooling) を用いてチャンクとの類似度をスコアリング
        query_summary = query_states.mean(dim=1)  # [B, D]
        # コサイン類似度計算
        q_norm = F.normalize(query_summary, p=2, dim=-1)  # [B, D]
        c_norm = F.normalize(chunk_states, p=2, dim=-1)  # [B, N, D]
        sim_scores = torch.bmm(c_norm, q_norm.unsqueeze(-1)).squeeze(-1)  # [B, N]

        if chunk_mask is not None:
            sim_scores = sim_scores.masked_fill(~chunk_mask, DEFAULT_MASK_VALUE)

        # 上位 k チャンクのインデックスを取得
        _, topk_indices = torch.topk(sim_scores, k=actual_k, dim=-1)  # [B, k]

        # チャンクテンソルから gather
        gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        selected_chunks = torch.gather(chunk_states, dim=1, index=gather_indices)

        return selected_chunks, topk_indices

    def forward(
        self,
        query_states: Tensor,
        chunk_states: Tensor,
        chunk_mask: Tensor | None = None,
    ) -> Tensor:
        """Top-k 選択されたチャンクを Key/Value として交差アテンションを適用する。

        Args:
            query_states (Tensor): 質問表現 `[batch_size, query_len, hidden_size]`。
            chunk_states (Tensor): チャンク表現 `[batch_size, num_chunks, hidden_size]`。
            chunk_mask (Tensor | None): 有効チャンクマスク `[batch_size, num_chunks]`。

        Returns:
            Tensor: 文脈集約後の質問表現 `[batch_size, query_len, hidden_size]`。
        """
        selected_chunks, _ = self.select_topk_chunks(
            query_states=query_states,
            chunk_states=chunk_states,
            chunk_mask=chunk_mask,
        )

        batch_size, q_len, _ = query_states.shape
        k_len = selected_chunks.size(1)

        if k_len == 0:
            return query_states

        q = (
            self.q_proj(query_states)
            .view(batch_size, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(selected_chunks)
            .view(batch_size, k_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(selected_chunks)
            .view(batch_size, k_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, q_len, self.hidden_size)
        )
        out = self.out_proj(context)

        return self.norm(query_states + self.dropout(out))


class ParallelClauseNoulScanner(nn.Module):
    """集約された全条項チャンク表現を一括判定する並列 Noul スキャナーヘッド。

    契約書や規程文書内の全条項 (例: 30〜60 条項) のチャンクベクトル V_chunks に対し、
    共有の単一線形層 (または Noul 分類ヘッド) を並列適用し、
    1 回のフォワードパスで全条項の真偽確率ベクトル p in [B, N_chunks] を出力する。
    """

    def __init__(self, hidden_size: int, dropout: float = 0.0) -> None:
        """並列条項スキャナーを初期化する。

        Args:
            hidden_size (int): チャンク表現の隠れ層次元数。
            dropout (float): ドロップアウト率。
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(
        self,
        chunk_states: Tensor,
        chunk_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """全チャンク表現を一括スキャンし、各条項の真偽ロジットと確率を算出する。

        Args:
            chunk_states (Tensor): チャンク表現テンソル `[batch_size, num_chunks, hidden_size]`。
            chunk_mask (Tensor | None): 有効チャンクマスク `[batch_size, num_chunks]`。

        Returns:
            tuple[Tensor, Tensor]:
                - logits: 各条項のロジット `[batch_size, num_chunks]`。
                - probs: 各条項の真偽確率 `[batch_size, num_chunks]` (Sigmoid 適用後)。
        """
        h = self.dropout(chunk_states)
        logits = self.classifier(h).squeeze(-1)  # [B, N_chunks]

        if chunk_mask is not None:
            logits = logits.masked_fill(~chunk_mask, DEFAULT_MASK_VALUE)
            probs = torch.sigmoid(logits).masked_fill(~chunk_mask, 0.0)
        else:
            probs = torch.sigmoid(logits)

        return logits, probs
