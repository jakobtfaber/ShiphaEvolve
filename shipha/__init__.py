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

from shipha.core import EvolutionRunner, EvolutionConfig
from shipha.database import ProgramDatabase, DatabaseConfig, Program

__all__ = [
    "EvolutionRunner",
    "EvolutionConfig",
    "ProgramDatabase",
    "DatabaseConfig",
    "Program",
]
