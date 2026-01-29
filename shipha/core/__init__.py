"""ShiphaEvolve core evolution engine."""

# NOTE: EvolutionRunner and EvolutionConfig will be implemented in runner.py
# For now, we export the evaluator which is complete
from shipha.core.evaluator import (
    TrustLevel,
    EvaluatorConfig,
    TestCase,
    TestGroup,
    EvaluationResult,
    ParallelEvaluator,
    evaluate_programs,
)

__all__ = [
    "TrustLevel",
    "EvaluatorConfig",
    "TestCase",
    "TestGroup",
    "EvaluationResult",
    "ParallelEvaluator",
    "evaluate_programs",
]

# Placeholder for future imports
# from shipha.core.runner import EvolutionRunner, EvolutionConfig
