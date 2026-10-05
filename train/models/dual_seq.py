"""Dual-Sequence 2セッション分割型アーキテクチャモジュール。

長文 State(契約書、規程、カルテ等)を 1 回だけ ModernBERT で処理して最終層隠れ状態を抽出し、
質問側(Instructions + Criteria + [OP])のエンコード結果と 2 層 Multi-Head Cross-Attention および
Context-Aware Gating で融合した後に、Pre-Gather 置換同変自己注意(SAB) を経て決定ロジットを算出する。
"""

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import PreTrainedModel

from contract import (
    TENSOR_ATTENTION_MASK,
    TENSOR_INPUT_IDS,
    TENSOR_LOGITS,
    TENSOR_OP_INDICES,
    TENSOR_QUERY_IDS,
    TENSOR_QUERY_MASK,
    TENSOR_STATE_HIDDEN_STATES,
    TENSOR_STATE_MASK,
)
from models.decision_head import (
    DEFAULT_MASK_VALUE,
    ChoiceHead,
    OptionGatherLayer,
    SetAttentionBlock,
)

logger = logging.getLogger(__name__)


class CrossAttentionBlock(nn.Module):
    """単一の多頭交差アテンションブロック (Multi-Head Cross-Attention Block)。

    Query 側の隠れ状態系列を Query ($Q$)、State 側の隠れ状態系列を Key ($K$) および Value ($V$)
    として受け取り、Scaled Dot-Product Cross-Attention を計算する。
    残差接続、2層 FeedForward ネットワーク、および LayerNorm (`eps=1e-5`) を含む。

    数理仕様:
    $$A = \\text{Softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M_{\\text{state}}\\right) V$$
    $$H_1 = \\text{LayerNorm}(Q + \\text{Dropout}(W_o A))$$
    $$H_2 = \\text{LayerNorm}(H_1 + \\text{FFN}(H_1))$$
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        eps: float = 1e-5,
    ) -> None:
        """交差アテンションブロックを初期化する。

        Args:
            hidden_size (int): 隠れ層次元数。
            num_heads (int): マルチヘッドアテンションのヘッド数。
            ffn_dim (int | None): FFN 中間層次元数 (未指定時は `hidden_size * 2`)。
            dropout (float): ドロップアウト率。
            eps (float): LayerNorm のイプシロン値 (ModernBERT と厳密一致の 1e-5)。
        """
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) は num_heads ({num_heads}) で割り切れる必要があります。"
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        ffn_inner_dim = ffn_dim if ffn_dim is not None else hidden_size * 2

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.norm1 = nn.LayerNorm(hidden_size, eps=eps)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_inner_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(ffn_inner_dim, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=eps)

    def forward(
        self,
        query_states: Tensor,
        key_value_states: Tensor,
        key_mask: Tensor | None = None,
    ) -> Tensor:
        """Query 表現と State 表現間で多頭交差注意を計算する。

        内部ロジック:
        1. Query を線形変換し [B, num_heads, L_q, head_dim] に展開する。
        2. Key および Value を線形変換し [B, num_heads, L_s, head_dim] に展開する。
        3. Scaled Dot-Product 注意スコア [B, num_heads, L_q, L_s] を計算する。
        4. `key_mask` が提供されている場合、State 無効位置を -1e4 でマスクする。
        5. Softmax と Dropout を経て Value と内積し、出力線形射影を行う。
        6. 残差接続と LayerNorm、および FFN と LayerNorm を適用する。

        Args:
            query_states (Tensor): Query 側隠れ状態 `[batch_size, query_seq_len, hidden_size]`。
            key_value_states (Tensor): State 側隠れ状態 `[batch_size, state_seq_len, hidden_size]`。
            key_mask (Tensor | None): State 系列マスク `[batch_size, state_seq_len]` (有効: 1, 無効: 0)。

        Returns:
            Tensor: 交差アテンション適用後の隠れ状態 `[batch_size, query_seq_len, hidden_size]`。
        """
        b_sz, q_len, _ = query_states.shape
        s_len = key_value_states.shape[1]

        q = (
            self.q_proj(query_states)
            .view(b_sz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key_value_states)
            .view(b_sz, s_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(key_value_states)
            .view(b_sz, s_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scale = float(self.head_dim) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if key_mask is not None:
            # key_mask: 1 is valid, 0 is invalid
            # shape: [B, 1, 1, L_s]
            invalid_mask = (key_mask == 0).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(invalid_mask, DEFAULT_MASK_VALUE)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v)
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(b_sz, q_len, self.hidden_size)
        )
        attn_out = self.out_proj(attn_out)

        h1 = self.norm1(query_states + self.dropout(attn_out))
        h2 = self.norm2(h1 + self.ffn(h1))
        return h2


class CrossAttentionStack(nn.Module):
    """2層直列の交差アテンションスタック (2-Layer Cross-Attention Stack)。

    第 1 層で State 内の関連スパンを大域走査し、
    第 2 層で特定されたスパンと質問トークン間の論理的整合性を検証する。
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        eps: float = 1e-5,
    ) -> None:
        """交差アテンションスタックを初期化する。

        Args:
            hidden_size (int): 隠れ層次元数。
            num_layers (int): 交差注意ブロックの積層数 (デフォルト: 2)。
            num_heads (int): マルチヘッドアテンションのヘッド数。
            ffn_dim (int | None): FFN 中間層次元数。
            dropout (float): ドロップアウト率。
            eps (float): LayerNorm のイプシロン値。
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        query_states: Tensor,
        key_value_states: Tensor,
        key_mask: Tensor | None = None,
    ) -> Tensor:
        """2層直列で交差注意を適用する。

        Args:
            query_states (Tensor): Query 側隠れ状態 `[batch_size, query_seq_len, hidden_size]`。
            key_value_states (Tensor): State 側隠れ状態 `[batch_size, state_seq_len, hidden_size]`。
            key_mask (Tensor | None): State 系列マスク `[batch_size, state_seq_len]`。

        Returns:
            Tensor: 全層適用後の集約隠れ状態 `[batch_size, query_seq_len, hidden_size]`。
        """
        hidden = query_states
        for layer in self.layers:
            hidden = layer(hidden, key_value_states, key_mask=key_mask)
        return hidden


class ContextAwareGating(nn.Module):
    """文脈適応型残差ゲート機構 (Context-Aware Gating)。

    Question 自身の自己表現 $H_{\\text{query}}$ と、State から集約された交差表現 $H_{\\text{cross}}$
    を入力適応型の Sigmoid ゲート $G$ で動的合成する。
    State 参照が不要な質問文自体の構文構造破壊を抑止する。

    数理仕様:
    $$G = \\sigma(W_g [H_{\\text{query}} \\,\\Vert{}\\; H_{\\text{cross}}] + b_g)$$
    $$H_{\\text{fused}} = G \\odot H_{\\text{cross}} + (1 - G) \\odot H_{\\text{query}}$$
    """

    def __init__(self, hidden_size: int) -> None:
        """文脈適応型ゲート層を初期化する。

        Args:
            hidden_size (int): 隠れ層次元数。
        """
        super().__init__()
        self.gate_proj = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, query_hidden: Tensor, cross_hidden: Tensor) -> Tensor:
        """Question 表現と交差集約表現を動的ゲートで融合する。

        Args:
            query_hidden (Tensor): Question 側の自己表現 `[batch_size, seq_len, hidden_size]`。
            cross_hidden (Tensor): State から集約された交差表現 `[batch_size, seq_len, hidden_size]`。

        Returns:
            Tensor: 動的融合された隠れ状態 `[batch_size, seq_len, hidden_size]`。
        """
        concat = torch.cat([query_hidden, cross_hidden], dim=-1)
        gate = torch.sigmoid(self.gate_proj(concat))
        return gate * cross_hidden + (1.0 - gate) * query_hidden


class AnchorPoolingLayer(nn.Module):
    """State 大域文脈プーリング注入層 (Anchor Pooling)。

    State 全体の平均隠れベクトルまたは CLS トークン表現を抽出し、
    Question 側の初期系列表現に射影バイアスとして加算することで
    大域文脈の把握を促進する。
    """

    def __init__(self, hidden_size: int) -> None:
        """Anchor Pooling 層を初期化する。

        Args:
            hidden_size (int): 隠れ層次元数。
        """
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(
        self,
        query_hidden: Tensor,
        state_hidden: Tensor,
        state_mask: Tensor | None = None,
    ) -> Tensor:
        """State の大域平均表現を Question 側の全トークンへ加算注入する。

        Args:
            query_hidden (Tensor): Question 側隠れ状態 `[batch_size, query_seq_len, hidden_size]`。
            state_hidden (Tensor): State 側隠れ状態 `[batch_size, state_seq_len, hidden_size]`。
            state_mask (Tensor | None): State 系列マスク `[batch_size, state_seq_len]`。

        Returns:
            Tensor: 大域文脈が注入された隠れ状態 `[batch_size, query_seq_len, hidden_size]`。
        """
        if state_mask is not None:
            mask_exp = state_mask.unsqueeze(-1).float()
            state_mean = (state_hidden * mask_exp).sum(dim=1) / mask_exp.sum(
                dim=1
            ).clamp(min=1e-8)
        else:
            state_mean = state_hidden.mean(dim=1)

        anchor_bias = self.proj(state_mean).unsqueeze(1)
        return self.norm(query_hidden + anchor_bias)


class SokutoStateEncoder(nn.Module):
    """長文 State を 1 回だけエンコードし最終層隠れ状態を出力する独立エンコーダ。

    ONNX エクスポート時および推論ランタイムにおいて `state_encoder.onnx` として動作する。
    """

    def __init__(self, backbone: PreTrainedModel) -> None:
        """State エンコーダを初期化する。

        Args:
            backbone (PreTrainedModel): ModernBERT などの双方向エンコーダバックボーン。
        """
        super().__init__()
        self.backbone = backbone

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """長文 State から最終層隠れ状態 H_state [batch_size, state_seq_len, hidden_size] を算出する。

        Args:
            input_ids (Tensor): トークンID列 `[batch_size, state_seq_len]`。
            attention_mask (Tensor): アテンションマスク `[batch_size, state_seq_len]`。

        Returns:
            Tensor: 最終層隠れ状態テンソル `[batch_size, state_seq_len, hidden_size]`。
        """
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden: Tensor = outputs.last_hidden_state
        return last_hidden

    def get_input_names(self) -> list[str]:
        """ONNX エクスポート用の入力テンソル名一覧を取得する。

        Returns:
            list[str]: 入力テンソル名リスト。
        """
        return [TENSOR_INPUT_IDS, TENSOR_ATTENTION_MASK]

    def get_output_names(self) -> list[str]:
        """ONNX エクスポート用の出力テンソル名一覧を取得する。

        Returns:
            list[str]: 出力テンソル名リスト。
        """
        return [TENSOR_STATE_HIDDEN_STATES]

    def get_dynamic_axes(self) -> dict[str, dict[int, str]]:
        """ONNX エクスポート用の動的軸定義辞書を取得する。

        Returns:
            dict[str, dict[int, str]]: 動的軸辞書。
        """
        return {
            TENSOR_INPUT_IDS: {0: "batch_size", 1: "state_seq_len"},
            TENSOR_ATTENTION_MASK: {0: "batch_size", 1: "state_seq_len"},
            TENSOR_STATE_HIDDEN_STATES: {0: "batch_size", 1: "state_seq_len"},
        }


class SokutoQueryEvaluator(nn.Module):
    """キャッシュされた State 隠れ状態を参照して質問を高速評価する決定エンジン。

    トポロジー A (Pre-Gather Cross-Attention) を採用：
    1. Question 系列の ModernBERT エンコード
    2. 2層 Multi-Head Cross-Attention による State 集約
    3. Context-Aware Gating による動的文脈融合
    4. OptionGatherLayer による `[OP]` マーカー位置抽出
    5. SetAttentionBlock (SAB) による置換同変自己注意
    6. Choice 射影ヘッドによるロジット出力
    """

    def __init__(
        self,
        backbone: PreTrainedModel,
        hidden_size: int = 768,
        num_cross_heads: int = 4,
        num_cross_layers: int = 2,
        num_sab_heads: int = 4,
        mlp_hidden_size: int | None = None,
        dropout: float = 0.0,
        use_anchor_pooling: bool = False,
    ) -> None:
        """Query 評価エンジンを初期化する。

        Args:
            backbone (PreTrainedModel): ModernBERT などの双方向エンコーダバックボーン。
            hidden_size (int): 隠れ層次元数。
            num_cross_heads (int): Cross-Attention のヘッド数。
            num_cross_layers (int): Cross-Attention の積層数。
            num_sab_heads (int): SAB のヘッド数。
            mlp_hidden_size (int | None): Choice ヘッド MLP 中間層次元数。
            dropout (float): ドロップアウト率。
            use_anchor_pooling (bool): Anchor Pooling 層を適用するかどうか。
        """
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.use_anchor_pooling = use_anchor_pooling

        self.anchor_layer = (
            AnchorPoolingLayer(hidden_size=hidden_size) if use_anchor_pooling else None
        )
        self.cross_attention = CrossAttentionStack(
            hidden_size=hidden_size,
            num_layers=num_cross_layers,
            num_heads=num_cross_heads,
            dropout=dropout,
            eps=1e-5,
        )
        self.gating = ContextAwareGating(hidden_size=hidden_size)
        self.gather_layer = OptionGatherLayer()
        self.sab = SetAttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_sab_heads,
            dropout=dropout,
        )
        self.choice_head = ChoiceHead(
            hidden_size=hidden_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=dropout,
            mask_value=DEFAULT_MASK_VALUE,
        )

    def forward(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        op_indices: Tensor,
        state_hidden_states: Tensor,
        state_mask: Tensor,
    ) -> Tensor:
        """キャッシュされた State 表現と照合して質問の選択肢ロジットを算出する。

        Args:
            query_ids (Tensor): 質問トークン列 `[batch_size, query_seq_len]`。
            query_mask (Tensor): 質問アテンションマスク `[batch_size, query_seq_len]`。
            op_indices (Tensor): 候補マーカー位置インデックス `[batch_size, num_options]`。
            state_hidden_states (Tensor): State 側最終層隠れ状態 `[batch_size, state_seq_len, hidden_size]`。
            state_mask (Tensor): State 側マスク `[batch_size, state_seq_len]`。

        Returns:
            Tensor: 候補ロジットテンソル `[batch_size, num_options]`。
        """
        outputs = self.backbone(input_ids=query_ids, attention_mask=query_mask)
        q_hidden: Tensor = outputs.last_hidden_state

        if self.anchor_layer is not None:
            q_hidden = self.anchor_layer(
                q_hidden, state_hidden_states, state_mask=state_mask
            )

        cross_hidden = self.cross_attention(
            q_hidden,
            state_hidden_states,
            key_mask=state_mask,
        )
        fused_hidden = self.gating(q_hidden, cross_hidden)

        op_mask = op_indices != -1
        raw_op_vectors = self.gather_layer(fused_hidden, op_indices)
        sab_op_vectors = self.sab(raw_op_vectors, op_mask=op_mask)
        logits = self.choice_head(sab_op_vectors, op_mask=op_mask)
        return logits

    def get_input_names(self) -> list[str]:
        """ONNX エクスポート用の入力テンソル名一覧を取得する。

        Returns:
            list[str]: 入力テンソル名リスト。
        """
        return [
            TENSOR_QUERY_IDS,
            TENSOR_QUERY_MASK,
            TENSOR_OP_INDICES,
            TENSOR_STATE_HIDDEN_STATES,
            TENSOR_STATE_MASK,
        ]

    def get_output_names(self) -> list[str]:
        """ONNX エクスポート用の出力テンソル名一覧を取得する。

        Returns:
            list[str]: 出力テンソル名リスト。
        """
        return [TENSOR_LOGITS]

    def get_dynamic_axes(self) -> dict[str, dict[int, str]]:
        """ONNX エクスポート用の動的軸定義辞書を取得する。

        Returns:
            dict[str, dict[int, str]]: 動的軸辞書。
        """
        return {
            TENSOR_QUERY_IDS: {0: "batch_size", 1: "query_seq_len"},
            TENSOR_QUERY_MASK: {0: "batch_size", 1: "query_seq_len"},
            TENSOR_OP_INDICES: {0: "batch_size", 1: "num_options"},
            TENSOR_STATE_HIDDEN_STATES: {0: "batch_size", 1: "state_seq_len"},
            TENSOR_STATE_MASK: {0: "batch_size", 1: "state_seq_len"},
            TENSOR_LOGITS: {0: "batch_size", 1: "num_options"},
        }


class DualSeqDecisionModel(nn.Module):
    """StateEncoder と QueryEvaluator を連動実行する統合 Dual-Sequence モデル。

    学習時および E2E テスト・パリティ検証時に使用する。
    """

    def __init__(
        self,
        state_encoder: SokutoStateEncoder,
        query_evaluator: SokutoQueryEvaluator,
    ) -> None:
        """統合モデルを初期化する。

        Args:
            state_encoder (SokutoStateEncoder): State エンコーダ。
            query_evaluator (SokutoQueryEvaluator): Query 評価エンジン。
        """
        super().__init__()
        self.state_encoder = state_encoder
        self.query_evaluator = query_evaluator

    def forward(
        self,
        state_ids: Tensor,
        state_mask: Tensor,
        query_ids: Tensor,
        query_mask: Tensor,
        op_indices: Tensor,
    ) -> Tensor:
        """State をエンコードし、その隠れ状態を用いて質問の決定ロジットを算出する。

        Args:
            state_ids (Tensor): State トークン列 `[batch_size, state_seq_len]`。
            state_mask (Tensor): State アテンションマスク `[batch_size, state_seq_len]`。
            query_ids (Tensor): 質問トークン列 `[batch_size, query_seq_len]`。
            query_mask (Tensor): 質問アテンションマスク `[batch_size, query_seq_len]`。
            op_indices (Tensor): 候補マーカー位置インデックス `[batch_size, num_options]`。

        Returns:
            Tensor: 候補ロジットテンソル `[batch_size, num_options]`。
        """
        state_hidden = self.state_encoder(
            input_ids=state_ids, attention_mask=state_mask
        )
        logits = self.query_evaluator(
            query_ids=query_ids,
            query_mask=query_mask,
            op_indices=op_indices,
            state_hidden_states=state_hidden,
            state_mask=state_mask,
        )
        return logits
