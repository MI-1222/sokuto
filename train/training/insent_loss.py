"""InSeNT (In-Sequence Negative Training) 対照損失モジュール。

Late Chunking による「文脈平均化による段落間コントラスト希釈」を防ぐため、
同一文書内の別チャンクを最も判別が困難な Hard Negative とする In-Sequence 対照損失 L_seq と、
バッチ内他文書のチャンクを負例とする In-Batch 対照損失 L_batch を線形結合して最適化する。

数理仕様:
$$s(q, k) = \\frac{q \\cdot k}{\\|q\\|_2 \\|k\\|_2 \\tau}$$
$$\\mathcal{L}_{\\text{seq}} = -\\log \\frac{\\exp(s(q, k^+))}{\\exp(s(q, k^+)) + \\sum_{k_i \\in N_{\\text{seq}} \\setminus \\{k^+\\}} \\exp(s(q, k_i))}$$
$$\\mathcal{L}_{\\text{batch}} = -\\log \\frac{\\exp(s(q, k^+))}{\\exp(s(q, k^+)) + \\sum_{k_j \\in N_{\\text{batch}}} \\exp(s(q, k_j))}$$
$$\\mathcal{L}_{\\text{insent}} = \\lambda_{\\text{seq}} \\mathcal{L}_{\\text{seq}} + (1 - \\lambda_{\\text{seq}}) \\mathcal{L}_{\\text{batch}} \\quad (\\lambda_{\\text{seq}} = 0.2)$$
"""

import logging

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)

DEFAULT_TEMPERATURE = 0.05
DEFAULT_LAMBDA_SEQ = 0.2


class InSeNTLoss(nn.Module):
    """In-Sequence & In-Batch 二重対照損失層 (InSeNT Loss)。

    Attributes:
        temperature (float): コサイン類似度 Softmax の温度パラメータ tau。
        lambda_seq (float): In-Sequence 損失 L_seq の加重比率 (デフォルト: 0.2)。
    """

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        lambda_seq: float = DEFAULT_LAMBDA_SEQ,
    ) -> None:
        """InSeNT 損失層を初期化する。

        Args:
            temperature (float): 温度パラメータ。
            lambda_seq (float): In-Sequence 損失の加重比率 (0.0 <= lambda_seq <= 1.0)。
        """
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(
                f"temperature ({temperature}) は正の値である必要があります。"
            )
        if not (0.0 <= lambda_seq <= 1.0):
            raise ValueError(
                f"lambda_seq ({lambda_seq}) は [0.0, 1.0] の範囲である必要があります。"
            )

        self.temperature = temperature
        self.lambda_seq = lambda_seq

    def forward(
        self,
        query_embeddings: Tensor,
        chunk_embeddings: Tensor,
        target_chunk_indices: Tensor,
        chunk_mask: Tensor | None = None,
        return_dict: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, float]]:
        """InSeNT 二重対照損失を計算する。

        Args:
            query_embeddings (Tensor): 質問側の表現ベクトル `[batch_size, hidden_size]`。
            chunk_embeddings (Tensor): 集約チャンク表現 `[batch_size, num_chunks, hidden_size]`。
            target_chunk_indices (Tensor): 各サンプルの正解条項インデックス `[batch_size]`。
            chunk_mask (Tensor | None): 有効チャンクマスク `[batch_size, num_chunks]`。
            return_dict (bool): メトリクス内訳辞書を返却するかどうか。

        Returns:
            Tensor | tuple[Tensor, dict[str, float]]:
                - 通常時: スカラーの InSeNT 損失テンソル。
                - return_dict=True 時: (insent_loss, metrics_dict) のタプル。
        """
        batch_size, num_chunks, hidden_size = chunk_embeddings.shape
        device = query_embeddings.device

        if batch_size == 0 or num_chunks == 0:
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if return_dict:
                return zero_loss, {
                    "loss_insent": 0.0,
                    "loss_seq": 0.0,
                    "loss_batch": 0.0,
                }
            return zero_loss

        # 3次元質問系列 [B, L, D] が渡された場合は平均プーリングで要約
        if query_embeddings.dim() == 3:
            q_vec = query_embeddings.mean(dim=1)
        else:
            q_vec = query_embeddings

        # L2 正規化 (コサイン類似度準備)
        q_norm = F.normalize(q_vec, p=2, dim=-1)  # [B, D]
        c_norm = F.normalize(chunk_embeddings, p=2, dim=-1)  # [B, N, D]

        if chunk_mask is None:
            chunk_mask = torch.ones(
                (batch_size, num_chunks), dtype=torch.bool, device=device
            )

        seq_losses: list[Tensor] = []
        batch_losses: list[Tensor] = []

        # 全バッチの正解チャンクベクトルを抽出: [B, D]
        # target_chunk_indices を安全クランプ
        clamped_targets = torch.clamp(target_chunk_indices, min=0, max=num_chunks - 1)
        target_indices_expanded = clamped_targets.view(batch_size, 1, 1).expand(
            -1, -1, hidden_size
        )
        target_chunk_vecs = torch.gather(
            c_norm, dim=1, index=target_indices_expanded
        ).squeeze(1)  # [B, D]

        # 1. In-Sequence 対照損失 (同一ドキュメント内の別条項をハードネガティブとする)
        for b in range(batch_size):
            valid_mask_b = chunk_mask[b]  # [N]
            target_idx = int(clamped_targets[b].item())

            # 正解チャンクがマスクされている場合はスキップ
            if not valid_mask_b[target_idx]:
                continue

            # 同一文書内の全チャンクとのコサイン類似度
            sim_b = torch.matmul(c_norm[b], q_norm[b]) / self.temperature  # [N]
            # 無効チャンクをマスク
            sim_b = sim_b.masked_fill(~valid_mask_b, -1e4)

            # 有効チャンク数が 2 以上のときのみ対照損失を計算 (他チャンクが存在する)
            num_valid = int(valid_mask_b.sum().item())
            if num_valid >= 2:
                # Cross-Entropy 損失 (正解ラベルは target_idx)
                target_label = torch.tensor(target_idx, device=device)
                loss_seq_b = F.cross_entropy(
                    sim_b.unsqueeze(0), target_label.unsqueeze(0)
                )
                seq_losses.append(loss_seq_b)

        # 2. In-Batch 対照損失 (他ドキュメントのチャンクを一般ネガティブとする)
        if batch_size >= 2:
            for b in range(batch_size):
                target_idx = int(clamped_targets[b].item())
                if not chunk_mask[b, target_idx]:
                    continue

                pos_sim = (
                    torch.dot(q_norm[b], target_chunk_vecs[b]) / self.temperature
                )  # スカラー

                # 他文書 (b' != b) の有効チャンクを集約
                neg_sims: list[Tensor] = []
                for other_b in range(batch_size):
                    if other_b == b:
                        continue
                    other_mask = chunk_mask[other_b]  # [N]
                    if other_mask.any():
                        other_chunks = c_norm[other_b][other_mask]  # [M, D]
                        sim_others = (
                            torch.matmul(other_chunks, q_norm[b]) / self.temperature
                        )  # [M]
                        neg_sims.append(sim_others)

                if neg_sims:
                    all_neg_sims = torch.cat(neg_sims, dim=0)  # [Total_Neg]
                    # logits: [pos_sim, neg_sim_1, neg_sim_2, ...] (正解インデックスは 0)
                    all_logits = torch.cat([pos_sim.unsqueeze(0), all_neg_sims], dim=0)
                    zero_target = torch.tensor(0, device=device)
                    loss_batch_b = F.cross_entropy(
                        all_logits.unsqueeze(0), zero_target.unsqueeze(0)
                    )
                    batch_losses.append(loss_batch_b)

        # 各損失の平均算出 (ゼロ除算ガード)
        mean_seq_loss = (
            torch.stack(seq_losses).mean()
            if seq_losses
            else torch.tensor(0.0, device=device)
        )
        mean_batch_loss = (
            torch.stack(batch_losses).mean()
            if batch_losses
            else torch.tensor(0.0, device=device)
        )

        # InSeNT 損失の加重合算
        if seq_losses and batch_losses:
            total_insent = (
                self.lambda_seq * mean_seq_loss
                + (1.0 - self.lambda_seq) * mean_batch_loss
            )
        elif seq_losses:
            total_insent = mean_seq_loss
        elif batch_losses:
            total_insent = mean_batch_loss
        else:
            total_insent = torch.tensor(0.0, device=device, requires_grad=True)

        if not return_dict:
            return total_insent

        metrics = {
            "loss_insent": float(total_insent.detach().item()),
            "loss_seq": float(mean_seq_loss.detach().item()) if seq_losses else 0.0,
            "loss_batch": float(mean_batch_loss.detach().item())
            if batch_losses
            else 0.0,
        }
        return total_insent, metrics
