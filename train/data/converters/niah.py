"""NIAH (Needle In A Haystack) 長文ベンチマークコンバータモジュール。

`NIAHBenchmarkGenerator` をラップし、`BaseDatasetConverter` インターフェースを通じて
Hugging Face Dataset や `UnifiedDatasetBuilder` から直接ストリーミング供給可能にする。
"""

from collections.abc import Iterator

from data.benchmarks.niah_long import (
    DEFAULT_NEEDLE_BANK,
    NeedleRule,
    NIAHBenchmarkGenerator,
)
from data.converters.base import BaseDatasetConverter
from data.schema import UnifiedSample


class NIAHBenchmarkConverter(BaseDatasetConverter):
    """NIAH 長文ベンチマークコンバータ。

    Attributes:
        generator (NIAHBenchmarkGenerator): ベンチマーク生成器。
        target_char_length (int): 背景テキスト目標文字数。
    """

    def __init__(
        self,
        needle_bank: list[NeedleRule] | None = None,
        target_char_length: int = 12000,
        seed: int = 42,
    ) -> None:
        """NIAH コンバータを初期化する。

        Args:
            needle_bank (list[NeedleRule] | None): 特約ルール一覧。
            target_char_length (int): 背景テキスト目標文字数。
            seed (int): 乱数シード。
        """
        super().__init__(seed=seed)
        self.generator = NIAHBenchmarkGenerator(
            needle_bank=needle_bank or DEFAULT_NEEDLE_BANK,
            target_char_length=target_char_length,
            seed=seed,
        )

    def convert_split(self, split: str = "test") -> Iterator[UnifiedSample]:
        """NIAH ベンチマークサンプルをストリーミング生成する。

        NIAH は主に評価用('test' / 'validation')として利用されるが、
        'train' が指定された場合も一貫したサンプル列を供給する。

        Args:
            split (str): スプリット名。

        Yields:
            Iterator[UnifiedSample]: 生成された統一サンプル列。
        """
        samples = self.generator.generate_benchmark_samples(include_ood=True)
        for s in samples:
            # スプリット情報の付与
            metadata = dict(s.metadata)
            metadata["split"] = split
            yield UnifiedSample(
                dataset_name=s.dataset_name,
                sample_id=s.sample_id,
                question_type=s.question_type,
                state=s.state,
                instructions=s.instructions,
                criteria=s.criteria,
                target=s.target,
                metadata=metadata,
            )
