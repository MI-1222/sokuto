"""長文意思決定合成データコンバータモジュール。

長文実務意思決定パイプライン (`LongContextSyntheticPipeline`) により生成された
JSONL ファイルを読み込むか、ビルトイン契約書・規程から動的に合成して
`UnifiedSample` 列としてストリーミング供給する。
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

from data.converters.base import BaseDatasetConverter
from data.converters.compliance_ja import BUILTIN_WORK_RULES_TEXT
from data.converters.legal_rikai import (
    BUILTIN_LEGAL_CONTRACT_NDA,
    BUILTIN_LEGAL_TERMS_SAAS,
)
from data.schema import UnifiedSample
from data.synthetic.long_context.pipeline import (
    LongContextSyntheticPipeline,
)

logger = logging.getLogger(__name__)


class LongContextSyntheticConverter(BaseDatasetConverter):
    """長文意思決定合成データセットコンバータ。

    Attributes:
        file_path (Path | None): 読み込み対象 JSONL パス。
        pipeline (LongContextSyntheticPipeline): 動的合成用パイプライン。
    """

    def __init__(
        self,
        file_path: str
        | Path
        | None = "train/data/synthetic/output/synthetic_long_context.jsonl",
        seed: int = 42,
    ) -> None:
        """コンバータを初期化する。

        Args:
            file_path (str | Path | None): JSONL ファイルパス。
            seed (int): 乱数シード。
        """
        super().__init__(seed=seed)
        self.file_path = Path(file_path) if file_path is not None else None
        self.pipeline = LongContextSyntheticPipeline(
            target_ood_ratio=0.135,
            include_contrast_sets=True,
            seed=seed,
        )

    def _resolve_file_path(self) -> Path | None:
        """有効な JSONL ファイルパスを探索する。

        Returns:
            Path | None: 発見されたファイルパス。
        """
        if self.file_path is None:
            return None

        candidates = [
            self.file_path,
            Path("train") / self.file_path,
        ]
        path_str = str(self.file_path)
        if path_str.startswith("train/"):
            candidates.append(Path(path_str[len("train/") :]))

        for cand in candidates:
            if cand.is_file():
                return cand
        return None

    def convert_split(self, split: str = "train") -> Iterator[UnifiedSample]:
        """指定スプリットを走査し、UnifiedSample を順次生成する。

        JSONL ファイルが存在すればそれを読み込み、存在しない場合は
        ビルトイン契約書・規程から動的合成を行う。

        Args:
            split (str): スプリット名 ('train', 'validation', 'test')。

        Yields:
            Iterator[UnifiedSample]: 統一サンプル列。
        """
        resolved = self._resolve_file_path()
        if resolved is not None:
            with open(resolved, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        sample = UnifiedSample.from_dict(data)
                        sample_hash = hash(sample.sample_id) % 100
                        if split in ("validation", "test"):
                            if sample_hash >= 15:
                                continue
                        else:
                            if sample_hash < 15:
                                continue
                        yield sample
                    except (
                        json.JSONDecodeError,
                        OSError,
                        ValueError,
                        KeyError,
                    ) as e:
                        logger.warning(
                            "合成データ行のパースに失敗しました: %s。スキップします。",
                            e,
                        )
            return

        # ファイルが存在しない場合はビルトイン契約書から動的合成
        corpus = [
            {"doc_id": "nda_alpha_beta", "text": BUILTIN_LEGAL_CONTRACT_NDA},
            {
                "doc_id": "saas_terms_cloud",
                "text": BUILTIN_LEGAL_TERMS_SAAS,
            },
            {
                "doc_id": "work_rules_sokuto",
                "text": BUILTIN_WORK_RULES_TEXT,
            },
        ]

        dynamic_samples = self.pipeline.process_corpus(corpus)
        for s in dynamic_samples:
            sample_hash = hash(s.sample_id) % 100
            if split in ("validation", "test"):
                if sample_hash >= 15:
                    continue
            else:
                if sample_hash < 15:
                    continue
            yield s
