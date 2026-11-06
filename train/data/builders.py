"""データセット一括構築・統合ビルダーモジュール。

複数の公開コーパス (Banking77, CLINC150, MNLI, AG News) のコンバータを統合し、
マルチタスク学習用の中間データセットを一括生成・結合するパイプラインを提供する。
"""

import logging
from collections.abc import Iterator

from datasets import Dataset

from data.converters.ag_news import AGNewsConverter
from data.converters.banking77 import Banking77Converter
from data.converters.base import BaseDatasetConverter
from data.converters.clinc150 import Clinc150Converter
from data.converters.compliance_ja import ComplianceJAConverter
from data.converters.jbe_qa import JBEQAConverter
from data.converters.jglue_jcommonsenseqa import JGlueJCommonsenseQAConverter
from data.converters.jglue_jnli import JGlueJNLIConverter
from data.converters.jglue_jsts import JGlueJSTSConverter
from data.converters.jglue_marc_ja import JGlueMarcJaConverter
from data.converters.legal_rikai import LegalRikaiConverter
from data.converters.mnli import MNLIConverter
from data.converters.niah import NIAHBenchmarkConverter
from data.converters.sst5 import SST5Converter
from data.converters.synthetic import SyntheticDatasetConverter
from data.converters.synthetic_long import LongContextSyntheticConverter
from data.negative_sampler import SyntheticNegativeInjector
from data.schema import UnifiedSample

logger = logging.getLogger(__name__)


class UnifiedDatasetBuilder:
    """複数 NLP コーパスの統合データセットビルダー。

    Attributes:
        converters (dict[str, BaseDatasetConverter]): 登録済みコンバータマップ。
        negative_injector (SyntheticNegativeInjector): 合成ネガティブ混入器。
    """

    def __init__(self, seed: int = 42, negative_ratio: float = 0.15) -> None:
        """ビルダーを初期化し、標準コンバータを登録する。

        Args:
            seed (int): 乱数シード。
            negative_ratio (float): train スプリット結合時に混入する合成ネガティブ比率。
        """
        self.seed = seed
        self.negative_ratio = negative_ratio
        self.negative_injector = SyntheticNegativeInjector(
            negative_ratio=negative_ratio, seed=seed
        )
        self.converters: dict[str, BaseDatasetConverter] = {
            "banking77": Banking77Converter(seed=seed),
            "clinc150": Clinc150Converter(seed=seed),
            "mnli_choice": MNLIConverter(mode="choice", seed=seed),
            "mnli_noul": MNLIConverter(mode="noul", seed=seed),
            "ag_news": AGNewsConverter(seed=seed),
            "sst5": SST5Converter(seed=seed),
            "jglue_marc_ja": JGlueMarcJaConverter(mode="noul", seed=seed),
            "jglue_marc_ja_choice": JGlueMarcJaConverter(mode="choice", seed=seed),
            "jglue_jnli": JGlueJNLIConverter(mode="choice", seed=seed),
            "jglue_jnli_noul": JGlueJNLIConverter(mode="noul", seed=seed),
            "jglue_jsts": JGlueJSTSConverter(seed=seed),
            "jglue_jcommonsenseqa": JGlueJCommonsenseQAConverter(seed=seed),
            "synthetic_all": SyntheticDatasetConverter(mode="all", seed=seed),
            "synthetic_choice": SyntheticDatasetConverter(mode="choice", seed=seed),
            "synthetic_score": SyntheticDatasetConverter(mode="score", seed=seed),
            "synthetic_noul": SyntheticDatasetConverter(mode="noul", seed=seed),
            # 長文実務データセット & ベンチマーク
            "jbe_qa": JBEQAConverter(mode="both", seed=seed),
            "jbe_qa_noul": JBEQAConverter(mode="noul", seed=seed),
            "jbe_qa_choice": JBEQAConverter(mode="choice", seed=seed),
            "legal_rikai": LegalRikaiConverter(mode="both", seed=seed),
            "legal_rikai_choice": LegalRikaiConverter(mode="choice", seed=seed),
            "legal_rikai_noul": LegalRikaiConverter(mode="noul", seed=seed),
            "compliance_ja": ComplianceJAConverter(mode="all", seed=seed),
            "compliance_ja_score": ComplianceJAConverter(mode="score", seed=seed),
            "compliance_ja_choice": ComplianceJAConverter(mode="choice", seed=seed),
            "compliance_ja_noul": ComplianceJAConverter(mode="noul", seed=seed),
            "niah_long": NIAHBenchmarkConverter(seed=seed),
            "synthetic_long_context": LongContextSyntheticConverter(seed=seed),
        }

    def register_converter(self, name: str, converter: BaseDatasetConverter) -> None:
        """新規コンバータを登録する。

        Args:
            name (str): コンバータ識別名。
            converter (BaseDatasetConverter): コンバーラインスタンス。
        """
        self.converters[name] = converter

    def stream_samples(
        self,
        dataset_names: list[str] | None = None,
        split: str = "train",
        max_samples_per_dataset: int | None = None,
    ) -> Iterator[UnifiedSample]:
        """指定されたデータセット群から UnifiedSample を順次ストリーミング抽出する。

        Args:
            dataset_names (list[str] | None): 対象データセット名リスト。None の場合は全登録データセット。
            split (str): 抽出対象スプリット名。
            max_samples_per_dataset (int | None): データセットごとの最大取得サンプル数。

        Yields:
            Iterator[UnifiedSample]: 統一サンプル列。
        """
        targets = dataset_names or list(self.converters.keys())

        for name in targets:
            if name not in self.converters:
                logger.warning(
                    "登録されていないデータセットです: %s。スキップします。",
                    name,
                )
                continue

            converter = self.converters[name]
            logger.info("データセット '%s' (split: %s) の変換を開始...", name, split)
            count = 0
            try:
                for sample in converter.convert_split(split):
                    yield sample
                    count += 1
                    if (
                        max_samples_per_dataset is not None
                        and count >= max_samples_per_dataset
                    ):
                        break
            except Exception as e:
                logger.error(
                    "データセット '%s' のロード中にエラーが発生しました: %s",
                    name,
                    e,
                )
                raise

    def build_combined_dataset(
        self,
        dataset_names: list[str] | None = None,
        split: str = "train",
        max_samples_per_dataset: int | None = None,
        inject_negatives: bool = True,
    ) -> Dataset:
        """指定データセット群を走査・変換し、結合された単一の Hugging Face Dataset を構築する。

        train スプリット時は inject_negatives が真であれば合成ネガティブサンプルが混入される。

        Args:
            dataset_names (list[str] | None): 対象データセット名リスト。
            split (str): スプリット名。
            max_samples_per_dataset (int | None): データセットごとの最大取得件数。
            inject_negatives (bool): 合成ネガティブ混入器を適用するかどうか (デフォルト: True)。

        Returns:
            Dataset: 全サンプルの辞書列から構成された統合 Dataset。
        """
        raw_samples = list(
            self.stream_samples(
                dataset_names=dataset_names,
                split=split,
                max_samples_per_dataset=max_samples_per_dataset,
            )
        )

        if inject_negatives:
            final_samples = self.negative_injector.inject(raw_samples, split=split)
        else:
            final_samples = raw_samples

        return Dataset.from_list([sample.to_dict() for sample in final_samples])
