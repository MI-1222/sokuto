"""長系列拡張および Late Chunking / InSeNT マルチタスク学習トレーナー。

全文エンコード後の隠れ状態からの Late Chunking 集約、
In-Sequence (同一文書内隣接条項) & In-Batch 対照損失 (InSeNT) と
意思決定タスク損失 (Choice / Score / Noul) のマルチタスク結合最適化を実行する。
"""

import logging

from torch import Tensor

from models.decision_head import JevDecisionModel
from models.long_context import (
    LateChunkingPooling,
    apply_yarn_to_modernbert,
)
from training.insent_loss import InSeNTLoss
from training.long_context_config import LongContextConfig
from training.loss import JevMultiTaskLoss

logger = logging.getLogger(__name__)


class LongContextTrainer:
    """長系列・Late Chunking・InSeNT パイプラインの学習トレーナー。

    Attributes:
        config (LongContextConfig): 長系列学習ハイパーパラメータ設定。
        model (JevDecisionModel): Jev 決定モデル。
        pooler (LateChunkingPooling): Late Chunking 事後プーリング層。
        insent_loss_fn (InSeNTLoss): InSeNT 対照損失層。
        task_loss_fn (JevMultiTaskLoss): 意思決定タスク損失層。
    """

    def __init__(
        self,
        config: LongContextConfig,
        model: JevDecisionModel,
        pooler: LateChunkingPooling | None = None,
    ) -> None:
        """LongContextTrainer を初期化する。

        Args:
            config (LongContextConfig): 長系列学習設定。
            model (JevDecisionModel): バックボーンおよびデシジョンヘッドを含むモデル。
            pooler (LateChunkingPooling | None): チャンクプーリング層。未指定時は config に従い自動生成。
        """
        self.config = config
        self.model = model

        # YaRN RoPE スケーリングの適用
        if config.use_yarn and hasattr(model, "backbone"):
            apply_yarn_to_modernbert(
                model.backbone,
                target_max_position=config.target_max_position,
                alpha=config.yarn_alpha,
                beta=config.yarn_beta,
            )

        # 勾配チェックポインティングの適用 (VRAM 節約)
        if (
            config.gradient_checkpointing
            and hasattr(model, "backbone")
            and hasattr(model.backbone, "gradient_checkpointing_enable")
        ):
            model.backbone.gradient_checkpointing_enable()
            logger.info("バックボーンの勾配チェックポインティングを有効化しました。")

        # プーラーの初期化
        hidden_size = model.backbone.config.hidden_size
        self.pooler = (
            pooler
            if pooler is not None
            else LateChunkingPooling(
                hidden_size=hidden_size,
                pooling_strategy=config.pooling_strategy,
            )
        )

        # 損失関数の初期化
        self.insent_loss_fn = InSeNTLoss(
            temperature=config.insent_temperature,
            lambda_seq=config.lambda_seq,
        )
        self.task_loss_fn = JevMultiTaskLoss()

    def compute_step_loss(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        op_indices: Tensor,
        op_mask: Tensor,
        labels: Tensor,
        chunk_spans: list[list[tuple[int, int]]] | None = None,
        chunk_starts: Tensor | None = None,
        chunk_ends: Tensor | None = None,
        target_chunk_indices: Tensor | None = None,
        question_types: list[str] | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """1 ステップの順伝播とマルチタスク損失 (タスク損失 + InSeNT 損失) を計算する。

        内部ロジック:
        1. バックボーンの隠れ状態を出力し、[OP] マーカー位置から決定ロジットを算出。
        2. 全文隠れ状態から LateChunkingPooling によりチャンク表現 V_chunks を抽出。
        3. タスク損失 L_task (Choice/Score/Noul) を計算。
        4. target_chunk_indices が提供されている場合、InSeNT 対照損失 L_insent を計算。
        5. L_total = L_task + lambda_insent * L_insent を合算。

        Args:
            input_ids (Tensor): 入力トークン ID `[B, L]`.
            attention_mask (Tensor): アテンションマスク `[B, L]`.
            op_indices (Tensor): [OP] マーカー位置 `[B, K]`.
            op_mask (Tensor): 有効候補マスク `[B, K]`.
            labels (Tensor): 正解インデックス `[B]`.
            chunk_spans (list[list[tuple[int, int]]] | None): 各バッチのチャンクスパン境界。
            chunk_starts (Tensor | None): チャンク開始テンソル `[B, N]`.
            chunk_ends (Tensor | None): チャンク終了テンソル `[B, N]`.
            target_chunk_indices (Tensor | None): 正解チャンクインデックス `[B]`.
            question_types (list[str] | None): 質問種別リスト。

        Returns:
            tuple[Tensor, dict[str, float]]: (総損失テンソル, メトリクス内訳辞書)。
        """
        # バックボーンで全文エンコード (単一フォワードパス)
        backbone_outputs = self.model.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = backbone_outputs.last_hidden_state  # [B, L, D]

        # 1. 決定ロジットの算出 (抽出された隠れ状態からデシジョンヘッドへ伝播)
        option_vectors = self.model.gather_layer(hidden_states, op_indices)
        sab_option_vectors = self.model.sab(option_vectors, op_mask)
        logits = self.model.choice_head(sab_option_vectors, op_mask)

        # 2. タスク損失の計算

        task_loss, loss_dict = self.task_loss_fn(
            logits=logits,
            labels=labels,
            op_mask=op_mask,
            question_types=question_types,
            return_dict=True,
        )

        total_loss = task_loss
        insent_loss_val = 0.0

        # 3. Late Chunking および InSeNT 対照損失の計算
        if self.config.use_insent and target_chunk_indices is not None:
            # チャンク集約
            if chunk_spans is not None:
                chunk_embeddings, chunk_mask = self.pooler(
                    hidden_states=hidden_states,
                    chunk_spans=chunk_spans,
                )
            elif chunk_starts is not None and chunk_ends is not None:
                chunk_embeddings, chunk_mask = self.pooler(
                    hidden_states=hidden_states,
                    chunk_starts=chunk_starts,
                    chunk_ends=chunk_ends,
                )
            else:
                chunk_embeddings, chunk_mask = None, None

            if chunk_embeddings is not None and chunk_mask is not None:
                # 質問側表現として [CLS] トークン (index 0) またはシーケンス平均を使用
                query_embeddings = hidden_states[:, 0, :]  # [B, D]
                insent_loss, insent_metrics = self.insent_loss_fn(
                    query_embeddings=query_embeddings,
                    chunk_embeddings=chunk_embeddings,
                    target_chunk_indices=target_chunk_indices,
                    chunk_mask=chunk_mask,
                    return_dict=True,
                )
                total_loss = total_loss + self.config.lambda_insent * insent_loss
                insent_loss_val = float(insent_loss.detach().item())
                loss_dict.update(insent_metrics)

        loss_dict["total_loss"] = float(total_loss.detach().item())
        loss_dict["loss_insent"] = insent_loss_val

        return total_loss, loss_dict
