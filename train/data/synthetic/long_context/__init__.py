"""長文実務意思決定合成データパッケージ。

条項依存関係 DAG 抽出、Multi-hop 意思決定シナリオ合成、
最小反事実ペア (Contrast Sets) 生成、および最適比率 (12.5%〜15.0%) OOD サンプル注入
を含む統合生成パイプラインを提供する。
"""

from data.synthetic.long_context.contrast_generator import (
    ContrastPair,
    ContrastSetGenerator,
)
from data.synthetic.long_context.dag_extractor import (
    ClauseDAG,
    ClauseDAGExtractor,
    ClauseEdge,
    ClauseNode,
    ClauseRelationType,
)
from data.synthetic.long_context.multihop_synthesizer import (
    MultiHopScenario,
    MultiHopSynthesizer,
)
from data.synthetic.long_context.ood_injector import (
    LongContextOODInjector,
)
from data.synthetic.long_context.pipeline import (
    LongContextSyntheticPipeline,
)

__all__ = [
    "ClauseDAG",
    "ClauseDAGExtractor",
    "ClauseEdge",
    "ClauseNode",
    "ClauseRelationType",
    "ContrastPair",
    "ContrastSetGenerator",
    "LongContextOODInjector",
    "LongContextSyntheticPipeline",
    "MultiHopScenario",
    "MultiHopSynthesizer",
]
