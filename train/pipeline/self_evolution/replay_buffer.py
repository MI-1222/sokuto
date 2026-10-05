"""80:20 黄金比リプレイバッファモジュール。

本番認定済み教師データ (Gold Standard) 80% と新規難例データ (Silver Data) 20%
を厳格な固定比率でブレンドしてバッチを編成する。
生成データのみの反復学習による決定境界の歪みとモデル崩壊 (Model Collapse)、
および破滅的忘却 (Catastrophic Forgetting) を数学的に防止する。
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

from torch.utils.data import Dataset, Sampler

from data.schema import UnifiedSample
from pipeline.self_evolution.config import ReplayConfig


@dataclass
class DistillationSample:
    """ソフトラベル確率分布を保持する蒸留学習用データサンプル。

    Attributes:
        unified_sample (UnifiedSample): 基本的な中間学習サンプル。
        soft_labels (dict[str, float] | None): 教師モデルのソフト確率分布 (Choice/Score/Noul)。
        is_silver (bool): 新規難例データ (Silver) か本番認定済み過去データ (Gold) かのフラグ。
    """

    unified_sample: UnifiedSample
    soft_labels: dict[str, float] | None = None
    is_silver: bool = False

    @property
    def sample_id(self) -> str:
        """サンプル固有識別子。"""
        return self.unified_sample.sample_id


class GoldenRatioReplayBuffer(Dataset[DistillationSample]):
    """Gold 80% : Silver 20% の黄金比リプレイバッファ。

    Reservoir Sampling により、有限容量内で均一なデータ分布を維持しながら
    新規難例データを安全に蓄積・提供する。
    """

    def __init__(self, config: ReplayConfig | None = None) -> None:
        """リプレイバッファを初期化する。

        Args:
            config (ReplayConfig | None): リプレイバッファ設定。
        """
        self.config = config or ReplayConfig()
        self._gold_pool: list[DistillationSample] = []
        self._silver_pool: list[DistillationSample] = []
        self._total_silver_seen: int = 0

    def add_gold_samples(self, samples: list[UnifiedSample]) -> None:
        """本番認定済み Gold Standard サンプルを追加する。

        Args:
            samples (list[UnifiedSample]): 追加する Gold サンプル列。
        """
        for s in samples:
            dist_sample = DistillationSample(
                unified_sample=s,
                soft_labels=None,
                is_silver=False,
            )
            self._gold_pool.append(dist_sample)

    def add_silver_samples(
        self,
        samples: list[UnifiedSample],
        soft_labels_list: list[dict[str, float]] | None = None,
    ) -> None:
        """検証通過済みの新規難例 Silver サンプルをリザーバサンプリングで追加する。

        Args:
            samples (list[UnifiedSample]): 追加する Silver サンプル列。
            soft_labels_list (list[dict[str, float]] | None): 対応するソフト確率分布列。
        """
        capacity = int(self.config.buffer_capacity * self.config.silver_ratio)
        capacity = max(capacity, 1)

        for i, s in enumerate(samples):
            soft = (
                soft_labels_list[i]
                if soft_labels_list and i < len(soft_labels_list)
                else None
            )
            dist_sample = DistillationSample(
                unified_sample=s,
                soft_labels=soft,
                is_silver=True,
            )

            self._total_silver_seen += 1
            if len(self._silver_pool) < capacity:
                self._silver_pool.append(dist_sample)
            elif self.config.reservoir_sampling:
                # リザーバサンプリングによる確率的置換
                replace_idx = random.randint(0, self._total_silver_seen - 1)
                if replace_idx < capacity:
                    self._silver_pool[replace_idx] = dist_sample

    def __len__(self) -> int:
        """バッファ内の総サンプル数を返す。"""
        return len(self._gold_pool) + len(self._silver_pool)

    def __getitem__(self, index: int) -> DistillationSample:
        """インデックスによるサンプル取得。"""
        if index < len(self._gold_pool):
            return self._gold_pool[index]
        return self._silver_pool[index - len(self._gold_pool)]

    @property
    def num_gold(self) -> int:
        """保持している Gold サンプル数。"""
        return len(self._gold_pool)

    @property
    def num_silver(self) -> int:
        """保持している Silver サンプル数。"""
        return len(self._silver_pool)


class GoldenRatioBatchSampler(Sampler[list[int]]):
    """バッチごとに厳格に Gold 80% / Silver 20% の比率を保つバッチサンプラー。"""

    def __init__(
        self,
        buffer: GoldenRatioReplayBuffer,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        """バッチサンプラーを初期化する。

        Args:
            buffer (GoldenRatioReplayBuffer): サンプリング対象のリプレイバッファ。
            batch_size (int): 1 バッチあたりの総サンプル数。
            shuffle (bool): シャッフルを行うか。
            seed (int): 乱数シード。
        """
        super().__init__()
        self.buffer = buffer
        self.batch_size = max(batch_size, 2)
        self.shuffle = shuffle
        self.rng = random.Random(seed)

        # 比率に応じたバッチ内サンプル数の決定
        self.silver_per_batch = max(
            round(self.batch_size * buffer.config.silver_ratio), 1
        )
        self.gold_per_batch = self.batch_size - self.silver_per_batch

    def __iter__(self) -> Iterator[list[int]]:
        """80:20 の比率でインデックスリストを生成・反復する。"""
        num_gold = self.buffer.num_gold
        num_silver = self.buffer.num_silver

        if num_gold == 0 and num_silver == 0:
            return

        gold_indices = list(range(num_gold))
        silver_indices = list(range(num_gold, num_gold + num_silver))

        if self.shuffle:
            self.rng.shuffle(gold_indices)
            self.rng.shuffle(silver_indices)

        # サンプル数が不足する場合は循環サンプリング
        g_idx = 0
        s_idx = 0

        # イテレーション回数は Gold プール基準
        total_batches = max(len(gold_indices) // max(self.gold_per_batch, 1), 1)

        for _ in range(total_batches):
            batch: list[int] = []

            # Gold サンプルの抽出
            if gold_indices:
                for _ in range(self.gold_per_batch):
                    batch.append(gold_indices[g_idx % len(gold_indices)])
                    g_idx += 1

            # Silver サンプルの抽出
            if silver_indices:
                for _ in range(self.silver_per_batch):
                    batch.append(silver_indices[s_idx % len(silver_indices)])
                    s_idx += 1

            if self.shuffle:
                self.rng.shuffle(batch)

            yield batch

    def __len__(self) -> int:
        """1 エポックあたりのバッチ数を返す。"""
        if self.buffer.num_gold == 0:
            return 1 if self.buffer.num_silver > 0 else 0
        return max(self.buffer.num_gold // max(self.gold_per_batch, 1), 1)
