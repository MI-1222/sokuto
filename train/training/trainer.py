"""SFT 学習ループおよびモデル検証・保存モジュール。

Hugging Face Accelerate を用いて、可変長・可変候補数の指示データに対する
クロスエントロピー教師ありファインチューニングを実行し、
差分学習率、エポック連動シャッフル、勾配クリッピング、および再現性メタデータの記録を行う。
"""

import json
import logging
import os
import platform
import random
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# マルチワーカー実行時の Hugging Face Tokenizers デッドロックを防止
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizerFast,
    get_cosine_schedule_with_warmup,
)

from data.builders import UnifiedDatasetBuilder
from data.dataset import JevDataset, pad_jev_collate_fn
from data.formatter import format_prompt, tokenize_sample
from data.schema import QuestionType, UnifiedSample
from models.backbone import prepare_backbone_and_tokenizer, save_tokenizer_for_runtime
from models.decision_head import JevDecisionModel
from models.long_context import apply_yarn_to_modernbert
from training.config import SFTConfig
from training.loss import JevMultiTaskLoss
from training.metrics import MetricsTracker

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """全乱数生成器のシードを一括初期化し、厳密な再現性を確保する。

    Args:
        seed (int): 乱数シード。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("乱数シードを %d に固定しました。", seed)


def collect_run_metadata(config: SFTConfig) -> dict[str, Any]:
    """実行環境の情報(Git コミット、ライブラリバージョン、ハードウェア情報)を収集する。

    Args:
        config (SFTConfig): SFT 設定オブジェクト。

    Returns:
        dict[str, Any]: 環境メタデータ辞書。
    """
    git_commit = "unknown"
    git_dirty = False
    try:
        commit_res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if commit_res.returncode == 0:
            git_commit = commit_res.stdout.strip()

        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        if status_res.returncode == 0:
            git_dirty = len(status_res.stdout.strip()) > 0
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("Git 情報の取得に失敗しました: %s。", e)

    import accelerate
    import transformers

    device_name = "cpu"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_name = "Apple Silicon MPS"

    return {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": device_name,
        "packages": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
        },
        "config": config.to_dict(),
    }


class SFTDataset(JevDataset):
    """評価・集計用のメタデータ(質問タイプ、合成ネガティブフラグ)を保持する Dataset。"""

    def __getitem__(self, index: int) -> dict[str, Any]:
        """指定インデックスのサンプルをトークナイズし、評価用メタデータを付与して返却する。

        Args:
            index (int): サンプルインデックス。

        Returns:
            dict[str, Any]: トークンテンソルおよびメタデータを含む辞書。
        """
        item: dict[str, Any] = super().__getitem__(index)
        sample = self.samples[index]
        item["question_type"] = sample.question_type.value
        item["is_negative"] = bool(sample.metadata.get("is_synthetic_negative", False))
        return item


def sft_collate_fn(
    batch: list[dict[str, Any]],
    pad_token_id: int,
) -> dict[str, Any]:
    """可変候補数・系列長のパディングを行い、評価用メタデータを保持してバッチ化する。

    Args:
        batch (list[dict[str, Any]]): 単一サンプルの辞書リスト。
        pad_token_id (int): パディングトークン ID。

    Returns:
        dict[str, Any]: パディング済みテンソルおよびメタデータリスト。
    """
    tensor_batch: list[dict[str, Tensor]] = []
    for item in batch:
        tensor_batch.append(
            {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
                "op_indices": item["op_indices"],
                "label": item["label"],
            }
        )

    collated: dict[str, Any] = dict(
        pad_jev_collate_fn(tensor_batch, pad_token_id=pad_token_id)
    )
    collated["question_types"] = [item.get("question_type", "") for item in batch]
    collated["is_negatives"] = torch.tensor(
        [item.get("is_negative", False) for item in batch], dtype=torch.bool
    )
    return collated


def get_optimizer_grouped_parameters(
    model: JevDecisionModel,
    lr_backbone: float,
    lr_embed: float,
    lr_head: float,
    weight_decay: float,
    head_weight_decay: float = 0.001,
) -> list[dict[str, Any]]:
    """バックボーン、埋め込み層、およびデシジョンヘッドの学習率を分離し、重み減衰をグループ化する。

    内部ロジック:
    1. バイアス項および正規化層(LayerNorm / RMSNorm / norm)の重みは weight_decay=0.0 とする。
    2. 埋め込み層 (embeddings / tok_embeddings) には lr_embed を適用する。
    3. デシジョンヘッド (decision_head / gather_layer) には lr_head を適用する。
    4. それ以外のバックボーン Transformer レイヤーには lr_backbone を適用する。

    Args:
        model (JevDecisionModel): Jev 決定モデル。
        lr_backbone (float): バックボーン Transformer レイヤーの学習率。
        lr_embed (float): [OP] を含む入力埋め込み層の学習率。
        lr_head (float): デシジョンヘッドの学習率。
        weight_decay (float): バックボーンおよび埋め込み層の重み減衰率。
        head_weight_decay (float): デシジョンヘッドの重み減衰率(デフォルト: 0.001)。

    Returns:
        list[dict[str, Any]]: 最適化パラメータグループリスト。
    """
    no_decay = [
        "bias",
        "LayerNorm.weight",
        "layer_norm.weight",
        "norm.weight",
        "norm.bias",
    ]

    head_decay = []
    head_no_decay = []
    embed_decay = []
    embed_no_decay = []
    backbone_decay = []
    backbone_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = any(nd in name for nd in no_decay)

        if name.startswith(
            (
                "decision_head",
                "choice_head",
                "score_head",
                "noul_head",
                "sab",
                "gather_layer",
            )
        ):
            if is_no_decay:
                head_no_decay.append(param)
            else:
                head_decay.append(param)
        elif "embeddings" in name or "tok_embeddings" in name:
            if is_no_decay:
                embed_no_decay.append(param)
            else:
                embed_decay.append(param)
        else:
            if is_no_decay:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)

    groups: list[dict[str, Any]] = []
    if backbone_decay:
        groups.append(
            {"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay}
        )
    if backbone_no_decay:
        groups.append(
            {"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0}
        )
    if embed_decay:
        groups.append(
            {"params": embed_decay, "lr": lr_embed, "weight_decay": weight_decay}
        )
    if embed_no_decay:
        groups.append({"params": embed_no_decay, "lr": lr_embed, "weight_decay": 0.0})
    if head_decay:
        groups.append(
            {"params": head_decay, "lr": lr_head, "weight_decay": head_weight_decay}
        )
    if head_no_decay:
        groups.append({"params": head_no_decay, "lr": lr_head, "weight_decay": 0.0})

    # パラメータグループの分類漏れを厳格にアサーション検証
    total_grouped_params = sum(len(g["params"]) for g in groups)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert total_grouped_params == len(trainable_params), (
        f"オプティマイザのパラメータグループ分類漏れが発生しました: "
        f"grouped={total_grouped_params}, total_trainable={len(trainable_params)}。"
    )

    return groups


class SFTTrainer:
    """Jev モデルの SFT (教師あり指示学習) を統括する Trainer クラス。

    Attributes:
        config (SFTConfig): SFT 設定。
        accelerator (Accelerator): Hugging Face Accelerate インスタンス。
        model (JevDecisionModel): Jev 決定モデル。
        tokenizer (PreTrainedTokenizerFast): トークナイザー。
        op_token_id (int): [OP] トークン ID。
        run_dir (Path): 今回の実行成果物の保存先ディレクトリ。
    """

    def __init__(
        self,
        config: SFTConfig,
        model: JevDecisionModel | None = None,
        tokenizer: PreTrainedTokenizerFast | None = None,
        op_token_id: int | None = None,
        accelerator: Accelerator | None = None,
    ) -> None:
        """Trainer を初期化する。

        Args:
            config (SFTConfig): 学習ハイパーパラメータ設定。
            model (JevDecisionModel | None): 学習対象モデル。未指定時はバックボーンから自動生成。
            tokenizer (PreTrainedTokenizerFast | None): トークナイザー。
            op_token_id (int | None): [OP] トークン ID。
            accelerator (Accelerator | None): Accelerate インスタンス。
        """
        self.config = config
        self.accelerator = accelerator or Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )

        set_seed(config.seed)

        # 実行成果物ディレクトリの作成 (タイムスタンプ付き)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        model_tag = Path(config.model_name_or_path).name.replace("/", "_")
        self.run_dir = (
            Path(config.output_dir) / f"sft_{timestamp}_{model_tag}"
        ).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if model is None or tokenizer is None or op_token_id is None:
            backbone, tok, op_id = prepare_backbone_and_tokenizer(
                model_name_or_path=config.model_name_or_path
            )
            self.tokenizer = tok
            self.op_token_id = op_id
            self.model = JevDecisionModel(
                backbone=backbone,
                mlp_hidden_size=config.mlp_hidden_size,
            )
        else:
            self.model = model
            self.tokenizer = tokenizer
            self.op_token_id = op_token_id

        # 長系列 YaRN RoPE スケーリングの適用 (局所窓層を保護し大域層のみ拡張)
        if getattr(config, "use_yarn", False) and hasattr(self.model, "backbone"):
            apply_yarn_to_modernbert(
                self.model.backbone,
                target_max_position=getattr(config, "target_max_position", 16384),
                alpha=getattr(config, "yarn_alpha", 1.0),
                beta=getattr(config, "yarn_beta", 32.0),
            )

        # 勾配チェックポインティングの適用 (長系列時の VRAM 節約)
        if (
            getattr(config, "gradient_checkpointing", False)
            and hasattr(self.model, "backbone")
            and hasattr(self.model.backbone, "gradient_checkpointing_enable")
        ):
            self.model.backbone.gradient_checkpointing_enable()
            logger.info("バックボーンの勾配チェックポインティングを有効化しました。")

        self.loss_fn: nn.Module = JevMultiTaskLoss(
            label_smoothing=config.label_smoothing,
            focal_gamma=config.focal_gamma,
            score_loss_power=config.score_loss_power,
            noul_pos_weight=config.noul_pos_weight,
            contrastive_weight=config.contrastive_weight,
            contrastive_temperature=config.contrastive_temperature,
            choice_weight=config.choice_weight,
            score_weight=config.score_weight,
            noul_weight=config.noul_weight,
            insent_weight=getattr(config, "insent_weight", 0.0),
            insent_temperature=getattr(config, "insent_temperature", 0.05),
            insent_lambda_seq=getattr(config, "insent_lambda_seq", 0.2),
        )
        self.metrics_tracker = MetricsTracker()

        self.best_metric = -1.0
        self.metrics_history: list[dict[str, Any]] = []
        self.val_samples: list[UnifiedSample] = []

    def prepare_data(
        self,
    ) -> tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]], SFTDataset]:
        """指示データセットを構築し、DataLoader を生成する。

        Returns:
            tuple[DataLoader[dict[str, Any]], DataLoader[dict[str, Any]], SFTDataset]:
                - 訓練用 DataLoader。
                - 検証用 DataLoader。
                - 訓練用 SFTDataset (エポック連動シャッフル呼び出し用)。
        """
        builder = UnifiedDatasetBuilder(
            seed=self.config.seed,
            negative_ratio=self.config.negative_ratio,
        )

        train_samples = list(
            builder.stream_samples(
                dataset_names=self.config.dataset_names,
                split="train",
                max_samples_per_dataset=self.config.max_samples_per_dataset,
            )
        )
        # 訓練スプリットへの合成ネガティブ注入
        train_samples = builder.negative_injector.inject(train_samples, split="train")

        val_samples = list(
            builder.stream_samples(
                dataset_names=self.config.dataset_names,
                split="validation",
                max_samples_per_dataset=self.config.max_samples_per_dataset,
            )
        )
        self.val_samples = val_samples

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

        train_dataset = SFTDataset(
            samples=train_samples,
            tokenizer=self.tokenizer,
            op_token_id=self.op_token_id,
            max_length=self.config.max_sequence_length,
            is_train=True,
            base_seed=self.config.seed,
        )

        val_dataset = SFTDataset(
            samples=val_samples,
            tokenizer=self.tokenizer,
            op_token_id=self.op_token_id,
            max_length=self.config.max_sequence_length,
            is_train=False,
            base_seed=self.config.seed,
        )

        train_loader: DataLoader[dict[str, Any]] = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda b: sft_collate_fn(b, pad_token_id=pad_id),
        )

        val_loader: DataLoader[dict[str, Any]] = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=lambda b: sft_collate_fn(b, pad_token_id=pad_id),
        )

        logger.info(
            "データセット準備完了: 訓練サンプル=%d, 検証サンプル=%d。",
            len(train_samples),
            len(val_samples),
        )
        return train_loader, val_loader, train_dataset

    def train_epoch(
        self,
        epoch: int,
        model: nn.Module,
        train_loader: DataLoader[dict[str, Any]],
        train_dataset: SFTDataset,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Any,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """単一エポックの訓練を実行する。

        内部ロジック:
        1. train_dataset.set_epoch(epoch) を呼び出し、候補シャッフルシードを更新する。
        2. 各バッチでフォワード計算、op_mask によるマスク付き損失計算を実施する。
        3. 勾配蓄積と clip_grad_norm_ を経てオプティマイザを更新する。

        Args:
            epoch (int): 現在のエポック番号。
            model (nn.Module): 学習モデル。
            train_loader (DataLoader[dict[str, Any]]): 訓練 DataLoader。
            train_dataset (SFTDataset): 訓練データセット。
            optimizer (torch.optim.Optimizer): オプティマイザ。
            lr_scheduler (Any): 学習率スケジューラ。
            progress_callback (Callable[[dict[str, Any]], None] | None): 進捗通知コールバック。

        Returns:
            dict[str, Any]: 訓練エポックのメトリクス集計結果。
        """
        # エポック連動動的シャッフルを更新 (位置バイアス排除)
        train_dataset.set_epoch(epoch)

        model.train()
        self.metrics_tracker.reset()

        total_batches = len(train_loader)
        device = next(model.parameters()).device
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            op_indices = batch["op_indices"].to(device)
            op_mask = batch["op_mask"].to(device)
            labels = batch["labels"].to(device)

            with self.accelerator.accumulate(model):
                q_types = batch.get("question_types")
                if self.config.contrastive_weight > 0.0:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        op_indices=op_indices,
                        op_mask=op_mask,
                        question_type=q_types,
                        return_features=True,
                    )
                    logits, state_repr, option_vectors = outputs  # type: ignore[misc]
                else:
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        op_indices=op_indices,
                        op_mask=op_mask,
                        question_type=q_types,
                    )
                    state_repr = None
                    option_vectors = None

                if isinstance(self.loss_fn, JevMultiTaskLoss):
                    loss = self.loss_fn(
                        logits=logits,
                        labels=labels,
                        op_mask=op_mask,
                        question_types=batch.get("question_types"),
                        state_repr=state_repr,
                        option_repr=option_vectors,
                    )
                else:
                    loss = self.loss_fn(
                        logits=logits,
                        labels=labels,
                        op_mask=op_mask,
                    )

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        model.parameters(), max_norm=self.config.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            loss_val = float(loss.item())
            self.metrics_tracker.update(
                loss=loss_val,
                logits=logits.detach(),
                labels=labels.detach(),
                op_mask=op_mask.detach(),
                question_types=batch["question_types"],
                is_negatives=batch["is_negatives"],
            )

            # MPS アロケータのキャッシュ累積によるメモリ枯渇を防止
            if step % 20 == 0 and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            if progress_callback is not None and (
                step % 10 == 0 or step == total_batches - 1
            ):
                current_metrics = self.metrics_tracker.compute()
                progress_callback(
                    {
                        "stage": "train",
                        "epoch": epoch,
                        "step": step,
                        "total_steps": total_batches,
                        "loss": current_metrics["loss"],
                        "accuracy": current_metrics["accuracy"],
                    }
                )

        return self.metrics_tracker.compute()

    def evaluate(
        self,
        model: nn.Module,
        val_loader: DataLoader[dict[str, Any]],
    ) -> dict[str, Any]:
        """検証データセットに対する評価を実行し、多面的メトリクスを算出する。

        Args:
            model (nn.Module): 評価対象モデル。
            val_loader (DataLoader[dict[str, Any]]): 検証 DataLoader。

        Returns:
            dict[str, Any]: 検証メトリクス集計結果。
        """
        model.eval()
        self.metrics_tracker.reset()

        device = next(model.parameters()).device
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                op_indices = batch["op_indices"].to(device)
                op_mask = batch["op_mask"].to(device)
                labels = batch["labels"].to(device)

                q_types = batch.get("question_types")
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    op_indices=op_indices,
                    op_mask=op_mask,
                    question_type=q_types,
                )
                if isinstance(self.loss_fn, JevMultiTaskLoss):
                    loss = self.loss_fn(
                        logits=logits,
                        labels=labels,
                        op_mask=op_mask,
                        question_types=batch.get("question_types"),
                    )
                else:
                    loss = self.loss_fn(
                        logits=logits,
                        labels=labels,
                        op_mask=op_mask,
                    )

                loss_val = float(loss.item())
                self.metrics_tracker.update(
                    loss=loss_val,
                    logits=logits,
                    labels=labels,
                    op_mask=op_mask,
                    question_types=batch["question_types"],
                    is_negatives=batch["is_negatives"],
                )

        metrics = self.metrics_tracker.compute()

        # 位置バイアス検証の実行 (Choice 型の順序不変性評価)
        if self.config.evaluate_position_bias and self.val_samples:
            bias_metrics = self.evaluate_position_bias(model=model)
            metrics.update(bias_metrics)

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return metrics

    @torch.no_grad()
    def evaluate_position_bias(
        self,
        model: nn.Module,
        val_samples: list[UnifiedSample] | None = None,
        max_samples: int = 200,
        threshold: float = 0.03,
    ) -> dict[str, Any]:
        """Choice 型候補の選択肢順序シャッフルに対する予測確率の不変性 (位置バイアス) を評価する。

        内部ロジック:
        1. Choice 型の検証サンプルを抽出する。
        2. 元の選択肢順序でモデルに入力し、各候補の Softmax 確率 p_orig を算出する。
        3. 選択肢を逆順に並び替えた入力を作成し、Softmax 確率 p_permuted を算出する。
        4. 各選択肢について、元の順序と置換後の順序の間で予測確率の絶対差分 |p_orig[k] - p_permuted[k]| を計測する。
        5. 最大乖離幅 (max_delta)、平均乖離幅 (mean_delta)、および閾値 (デフォルト: 0.03) 以内に収まった割合 (pass_rate) を集計する。

        Args:
            model (nn.Module): 評価対象モデル。
            val_samples (list[UnifiedSample] | None): 検証サンプルリスト。未指定時は self.val_samples。
            max_samples (int): 評価に使用する最大サンプル数。
            threshold (float): 位置バイアス許容変動閾値。

        Returns:
            dict[str, Any]: 位置バイアス評価指標辞書。
        """
        samples = val_samples if val_samples is not None else self.val_samples
        choice_samples = [
            s
            for s in samples
            if s.question_type == QuestionType.CHOICE and len(s.criteria) >= 2
        ][:max_samples]

        if not choice_samples:
            return {
                "position_bias_max_delta": 0.0,
                "position_bias_mean_delta": 0.0,
                "position_bias_pass_rate": 1.0,
                "position_bias_evaluated_samples": 0,
            }

        device = next(model.parameters()).device
        model.eval()

        deltas: list[float] = []
        max_sample_deltas: list[float] = []

        for sample_idx, sample in enumerate(choice_samples):
            num_options = len(sample.criteria)
            # 1. 元の順序での推論
            _, orig_keys = format_prompt(sample, shuffle_options=False)
            orig_tokenized = tokenize_sample(
                sample=sample,
                tokenizer=self.tokenizer,
                op_token_id=self.op_token_id,
                max_length=self.config.max_sequence_length,
                shuffle_options=False,
            )
            input_ids_orig = orig_tokenized["input_ids"].unsqueeze(0).to(device)
            attention_mask_orig = (
                orig_tokenized["attention_mask"].unsqueeze(0).to(device)
            )
            op_indices_orig = orig_tokenized["op_indices"].unsqueeze(0).to(device)

            logits_orig = model(input_ids_orig, attention_mask_orig, op_indices_orig)
            probs_orig = F.softmax(logits_orig[0, :num_options], dim=-1)

            # 2. 選択肢をシャッフルした順序での推論
            perm_seed = self.config.seed + sample_idx + 1000
            _, perm_keys = format_prompt(sample, shuffle_options=True, seed=perm_seed)
            perm_tokenized = tokenize_sample(
                sample=sample,
                tokenizer=self.tokenizer,
                op_token_id=self.op_token_id,
                max_length=self.config.max_sequence_length,
                shuffle_options=True,
                seed=perm_seed,
            )
            input_ids_perm = perm_tokenized["input_ids"].unsqueeze(0).to(device)
            attention_mask_perm = (
                perm_tokenized["attention_mask"].unsqueeze(0).to(device)
            )
            op_indices_perm = perm_tokenized["op_indices"].unsqueeze(0).to(device)

            logits_perm = model(input_ids_perm, attention_mask_perm, op_indices_perm)
            probs_perm = F.softmax(logits_perm[0, :num_options], dim=-1)

            # 3. 同一候補キーごとに確率差分を計測
            sample_diffs: list[float] = []
            for key in sample.criteria:
                orig_pos = orig_keys.index(key)
                perm_pos = perm_keys.index(key)
                p_orig = float(probs_orig[orig_pos].item())
                p_perm = float(probs_perm[perm_pos].item())
                diff = abs(p_orig - p_perm)
                deltas.append(diff)
                sample_diffs.append(diff)

            max_sample_deltas.append(max(sample_diffs) if sample_diffs else 0.0)

        mean_delta = float(np.mean(deltas)) if deltas else 0.0
        max_delta = float(np.max(max_sample_deltas)) if max_sample_deltas else 0.0
        passed_samples = sum(1 for d in max_sample_deltas if d <= threshold)
        pass_rate = passed_samples / len(choice_samples) if choice_samples else 1.0

        return {
            "position_bias_max_delta": round(max_delta, 4),
            "position_bias_mean_delta": round(mean_delta, 4),
            "position_bias_pass_rate": round(pass_rate, 4),
            "position_bias_evaluated_samples": len(choice_samples),
        }

    def save_checkpoint(
        self,
        model: nn.Module,
        val_metrics: dict[str, Any],
        is_best: bool = False,
    ) -> Path:
        """チェックポイントおよびトークナイザー、実行設定を保存する。

        Args:
            model (nn.Module): 保存対象モデル。
            val_metrics (dict[str, Any]): 検証メトリクス。
            is_best (bool): 最良モデルフラグ。

        Returns:
            Path: 保存先ディレクトリパス。
        """
        checkpoint_dir = self.run_dir / (
            "best_checkpoint" if is_best else "latest_checkpoint"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        unwrapped = self.accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), checkpoint_dir / "model.pt")

        # Rust 推論ランタイム用の tokenizer.json 一式を保存
        save_tokenizer_for_runtime(self.tokenizer, checkpoint_dir / "tokenizer")

        # バックボーン設定の保存 (Hugging Face AutoConfig 互換性の担保)
        if (
            hasattr(unwrapped, "backbone")
            and hasattr(unwrapped.backbone, "config")
            and hasattr(unwrapped.backbone.config, "save_pretrained")
        ):
            unwrapped.backbone.config.save_pretrained(checkpoint_dir)

        # 再現性のための設定ファイルもチェックポイント内に同期保存 (完全性保証)
        self.config.save_yaml(checkpoint_dir / "config.yaml")
        self.config.save_json(checkpoint_dir / "config.json")

        metrics_file = checkpoint_dir / "metrics.json"
        metrics_file.write_text(
            json.dumps(val_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "チェックポイントを保存しました (is_best=%s): %s。",
            is_best,
            checkpoint_dir,
        )
        return checkpoint_dir

    def train(
        self,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """SFT 学習ループ全体を実行し、成果物一式を出力する。

        Args:
            progress_callback (Callable[[dict[str, Any]], None] | None): 進捗通知コールバック。

        Returns:
            dict[str, Any]: 最終結果サマリー辞書。
        """
        logger.info("=== SFT 学習開始 (出力先: %s) ===", self.run_dir)

        # 1. 再現性メタデータおよび設定の保存
        self.config.save_yaml(self.run_dir / "config.yaml")
        self.config.save_json(self.run_dir / "config.json")

        metadata = collect_run_metadata(self.config)
        (self.run_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # 2. データ準備
        train_loader, val_loader, train_dataset = self.prepare_data()

        # 3. 最適化オプティマイザ・スケジューラ構築 (3系統 Differential LR)
        optimizer_params = get_optimizer_grouped_parameters(
            model=self.model,
            lr_backbone=self.config.learning_rate_backbone,
            lr_embed=self.config.learning_rate_embed,
            lr_head=self.config.learning_rate_head,
            weight_decay=self.config.weight_decay,
        )
        optimizer = AdamW(optimizer_params)

        num_update_steps_per_epoch = max(
            1, len(train_loader) // self.config.gradient_accumulation_steps
        )
        max_train_steps = self.config.num_epochs * num_update_steps_per_epoch
        num_warmup_steps = int(max_train_steps * self.config.warmup_ratio)

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )
        # 勾配チェックポインティングの適用 (VRAM/メモリ削減)
        if getattr(self.config, "gradient_checkpointing", False):
            raw_backbone = getattr(self.model, "backbone", None)
            if raw_backbone is not None and hasattr(
                raw_backbone, "gradient_checkpointing_enable"
            ):
                raw_backbone.gradient_checkpointing_enable()
                logger.info(
                    "勾配チェックポインティング (Gradient Checkpointing) を有効化しました。"
                )

        # 4. Accelerate によるラップ
        (
            prepared_model,
            prepared_optimizer,
            prepared_train_loader,
            prepared_val_loader,
            prepared_scheduler,
        ) = self.accelerator.prepare(
            self.model, optimizer, train_loader, val_loader, lr_scheduler
        )

        # 5. 初期ゼロショット検証 (ランダム水準の記録)
        initial_val_metrics = self.evaluate(prepared_model, prepared_val_loader)
        eval_key = self.config.eval_metric
        initial_eval_metric = float(
            initial_val_metrics.get(eval_key, initial_val_metrics.get("accuracy", 0.0))
        )
        logger.info(
            "初期ゼロショット検証: Loss=%.4f, Acc=%.4f (%s=%.4f)。",
            initial_val_metrics["loss"],
            initial_val_metrics["accuracy"],
            eval_key,
            initial_eval_metric,
        )
        self.metrics_history.append(
            {"epoch": 0, "stage": "zero_shot", "metrics": initial_val_metrics}
        )

        # 6. 学習ループ
        best_metric_val = -1.0
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, self.config.num_epochs + 1):
            logger.info("--- エポック %d/%d 開始 ---", epoch, self.config.num_epochs)

            train_metrics = self.train_epoch(
                epoch=epoch,
                model=prepared_model,
                train_loader=prepared_train_loader,
                train_dataset=train_dataset,
                optimizer=prepared_optimizer,
                lr_scheduler=prepared_scheduler,
                progress_callback=progress_callback,
            )

            val_metrics = self.evaluate(prepared_model, prepared_val_loader)
            current_metric = float(
                val_metrics.get(eval_key, val_metrics.get("accuracy", 0.0))
            )
            logger.info(
                "エポック %d 完了: Train Loss=%.4f, Val Loss=%.4f, Val Acc=%.4f (%s=%.4f)。",
                epoch,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                eval_key,
                current_metric,
            )

            self.metrics_history.append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                }
            )

            is_best = current_metric > best_metric_val
            if is_best:
                best_metric_val = current_metric
                best_epoch = epoch
                patience_counter = 0
                self.save_checkpoint(prepared_model, val_metrics, is_best=True)
            else:
                patience_counter += 1

            self.save_checkpoint(prepared_model, val_metrics, is_best=False)

            # メトリクス履歴の順次更新
            (self.run_dir / "metrics_history.json").write_text(
                json.dumps(self.metrics_history, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "epoch_end",
                        "epoch": epoch,
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                        "is_best": is_best,
                    }
                )

            # 早期終了判定
            if (
                self.config.early_stopping_patience is not None
                and patience_counter >= self.config.early_stopping_patience
            ):
                logger.info(
                    "早期終了 (Early Stopping): %d エポック連続で %s の改善が見られなかったため学習を打ち切ります。",
                    patience_counter,
                    eval_key,
                )
                break

        # 7. 最終サマリーの書き出し
        best_val_data = (
            self.metrics_history[best_epoch].get("val", {})
            if best_epoch < len(self.metrics_history)
            else {}
        )
        summary_text = (
            f"# SFT Training Summary\n\n"
            f"- **Run Directory**: `{self.run_dir}`\n"
            f"- **Model**: `{self.config.model_name_or_path}`\n"
            f"- **Epochs Trained**: {min(epoch, self.config.num_epochs)}\n"
            f"- **Best Epoch**: {best_epoch}\n"
            f"- **Best {eval_key}**: {best_metric_val:.4f}\n"
            f"- **Initial Zero-Shot Accuracy**: {initial_val_metrics['accuracy']:.4f}\n\n"
            f"## Final Validation Metrics (Best Checkpoint)\n\n"
            f"```json\n{json.dumps(best_val_data, indent=2, ensure_ascii=False)}\n```\n"
        )
        (self.run_dir / "final_summary.md").write_text(summary_text, encoding="utf-8")
        logger.info(
            "SFT 学習完了。最良 %s: %.4f (Epoch %d)。",
            eval_key,
            best_metric_val,
            best_epoch,
        )

        best_accuracy = float(best_val_data.get("accuracy", best_metric_val))
        return {
            "run_dir": str(self.run_dir),
            "best_epoch": best_epoch,
            "best_accuracy": best_accuracy,
            "best_metric": best_metric_val,
            "eval_metric": eval_key,
            "initial_accuracy": initial_val_metrics["accuracy"],
            "metrics_history": self.metrics_history,
        }
