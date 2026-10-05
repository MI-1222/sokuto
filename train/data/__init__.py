"""Jev データセットパイプラインパッケージ。

NLP コーパスのスキーマ統一、指示文多様化、プロンプトフォーマット、
およびトークナイズ機能を提供する。
"""

from data.benchmarks.niah_long import (
    DEFAULT_NEEDLE_BANK,
    NeedleRule,
    NIAHBenchmarkGenerator,
    NIAHEvaluationResult,
)
from data.builders import UnifiedDatasetBuilder
from data.converters import (
    AGNewsConverter,
    Banking77Converter,
    BaseDatasetConverter,
    Clinc150Converter,
    ComplianceJAConverter,
    JBEQAConverter,
    LegalRikaiConverter,
    LongContextSyntheticConverter,
    MNLIConverter,
    NIAHBenchmarkConverter,
)
from data.dataset import (
    JevDataset,
    JevDynamicDataset,
    jev_collate_fn,
    pad_jev_collate_fn,
)
from data.formatter import (
    char_spans_to_token_spans,
    format_prompt,
    tokenize_sample,
)
from data.hierarchical import HierarchicalMapping
from data.hierarchical_dataset import (
    HierarchicalDatasetGenerator,
    HierarchicalJevDataset,
    HierarchicalSamplePair,
    hierarchical_collate_fn,
)
from data.negative_sampler import NEGATIVE_OPTION_POOL, SyntheticNegativeInjector
from data.prompt_pool import sample_instruction
from data.schema import QuestionType, UnifiedSample
from data.synthetic.long_context import (
    ClauseDAG,
    ClauseDAGExtractor,
    ContrastSetGenerator,
    LongContextOODInjector,
    LongContextSyntheticPipeline,
    MultiHopSynthesizer,
)

__all__ = [
    "DEFAULT_NEEDLE_BANK",
    "NEGATIVE_OPTION_POOL",
    "AGNewsConverter",
    "Banking77Converter",
    "BaseDatasetConverter",
    "ClauseDAG",
    "ClauseDAGExtractor",
    "Clinc150Converter",
    "ComplianceJAConverter",
    "ContrastSetGenerator",
    "HierarchicalDatasetGenerator",
    "HierarchicalJevDataset",
    "HierarchicalMapping",
    "HierarchicalSamplePair",
    "JBEQAConverter",
    "JevDataset",
    "JevDynamicDataset",
    "LegalRikaiConverter",
    "LongContextOODInjector",
    "LongContextSyntheticConverter",
    "LongContextSyntheticPipeline",
    "MNLIConverter",
    "MultiHopSynthesizer",
    "NIAHBenchmarkConverter",
    "NIAHBenchmarkGenerator",
    "NIAHEvaluationResult",
    "NeedleRule",
    "QuestionType",
    "SyntheticNegativeInjector",
    "UnifiedDatasetBuilder",
    "UnifiedSample",
    "char_spans_to_token_spans",
    "format_prompt",
    "hierarchical_collate_fn",
    "jev_collate_fn",
    "pad_jev_collate_fn",
    "sample_instruction",
    "tokenize_sample",
]
