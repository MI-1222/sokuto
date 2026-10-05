"""sokuto System 1 / System 2 カスケード統合 SDK パッケージ。

定常トラフィックの 80% 以上をローカルのミリ秒非自己回帰判断エンジン (sokuto / System 1) で即座に完結させ、
僅差の拮抗や未知ドメイン (OOD) の境界事例のみをフロンティア自己回帰 LLM (System 2) へ
中立的対比型 Triage プロンプトによりエスカレーションする。
"""

from .circuit_breaker import CircuitBreaker, CircuitState
from .client import SokutoCascadeClient, SokutoCascadeClientSync
from .gating import (
    calculate_entropy,
    calculate_score_metrics,
    calculate_top_margin,
    compute_free_energy,
    evaluate_gating,
)
from .providers import (
    CallableSystem2Provider,
    GenericHttpSystem2Provider,
    MockSystem2Provider,
    System2Provider,
    System2Response,
)
from .triage import synthesize_triage_prompt
from .types import (
    CascadeResult,
    CascadeSource,
    DecisionRoute,
    GatingConfig,
    GatingDecision,
    TriagePrompt,
)

__all__ = [
    "CallableSystem2Provider",
    "CascadeResult",
    "CascadeSource",
    "CircuitBreaker",
    "CircuitState",
    "DecisionRoute",
    "GatingConfig",
    "GatingDecision",
    "GenericHttpSystem2Provider",
    "MockSystem2Provider",
    "SokutoCascadeClient",
    "SokutoCascadeClientSync",
    "System2Provider",
    "System2Response",
    "TriagePrompt",
    "calculate_entropy",
    "calculate_score_metrics",
    "calculate_top_margin",
    "compute_free_energy",
    "evaluate_gating",
    "synthesize_triage_prompt",
]
