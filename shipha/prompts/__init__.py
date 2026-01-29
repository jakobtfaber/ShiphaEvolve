"""ShiphaEvolve prompt templates for evolution."""

from shipha.prompts.base import BASE_SYSTEM_MSG
from shipha.prompts.diff import DIFF_SYS_FORMAT, DIFF_ITER_MSG
from shipha.prompts.crossover import CROSS_SYS_FORMAT, CROSS_ITER_MSG
from shipha.prompts.meta import MetaPrompts

__all__ = [
    "BASE_SYSTEM_MSG",
    "DIFF_SYS_FORMAT",
    "DIFF_ITER_MSG",
    "CROSS_SYS_FORMAT",
    "CROSS_ITER_MSG",
    "MetaPrompts",
]
