"""
ShiphaEvolve - LLM-Based Code Evolution Framework

Combines OpenAlpha_Evolve's IDE/LSP integration with ShinkaEvolve's
advanced evolutionary algorithms for production-ready code optimization.

Key Features:
- IDE/LSP integration with EVOLVE-BLOCK markers
- Multi-armed bandit LLM selection
- Tiered evaluation (Docker sandbox → subprocess → direct)
- Crossover operator for genetic recombination
- Novelty enforcement via embeddings
- Meta-recommendations for guided evolution
"""

__version__ = "0.1.0"
__author__ = "Jakob Faber"

# Core evaluation components
from shipha.core import (
    TrustLevel,
    EvaluatorConfig,
    TestCase,
    TestGroup,
    EvaluationResult,
    ParallelEvaluator,
    evaluate_programs,
)

# LLM selection with multi-armed bandits
from shipha.llm.bandit import AsymmetricUCB, BanditBase, FixedSampler

__all__ = [
    # Core
    "TrustLevel",
    "EvaluatorConfig",
    "TestCase",
    "TestGroup",
    "EvaluationResult",
    "ParallelEvaluator",
    "evaluate_programs",
    # LLM
    "AsymmetricUCB",
    "BanditBase",
    "FixedSampler",
]
