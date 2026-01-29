"""
Prompt sampler for diverse prompt generation.

Orchestrates selection of prompt strategies (diff, full, crossover, meta)
and parent program sampling to maximize exploration of the solution space.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shipha.database import Program, ProgramDatabase
    from shipha.prompts.base import PromptBuilder

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Strategy
# =============================================================================


class PromptStrategy(str, Enum):
    """Available prompting strategies."""

    DIFF = "diff"  # SEARCH/REPLACE incremental edits
    FULL = "full"  # Complete code rewrite
    CROSSOVER = "crossover"  # Multi-parent combination
    META = "meta"  # Meta-analysis guided
    RANDOM = "random"  # Random mutation


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SamplerConfig:
    """Configuration for prompt sampling.

    Attributes:
        strategy_weights: Weights for each strategy (must sum to 1).
        num_inspirations: Number of inspiration programs for crossover.
        include_feedback: Include evaluation feedback in prompts.
        include_lineage: Include parent history in prompts.
        max_context_tokens: Maximum prompt context size.
        temperature_range: Temperature range for sampling.
        elite_bias: Bias towards elite programs in parent selection.
    """

    strategy_weights: dict[PromptStrategy, float] = field(
        default_factory=lambda: {
            PromptStrategy.DIFF: 0.5,
            PromptStrategy.FULL: 0.2,
            PromptStrategy.CROSSOVER: 0.2,
            PromptStrategy.META: 0.1,
        }
    )
    num_inspirations: int = 3
    include_feedback: bool = True
    include_lineage: bool = True
    max_context_tokens: int = 8000
    temperature_range: tuple[float, float] = (0.6, 1.0)
    elite_bias: float = 2.0  # Bias factor for elite selection


# =============================================================================
# Prompt Sample
# =============================================================================


@dataclass
class PromptSample:
    """A sampled prompt ready for LLM query.

    Attributes:
        system_msg: System message for the LLM.
        user_msg: User message with problem/code context.
        strategy: The strategy used to generate this prompt.
        parent: The parent program (if any).
        inspirations: Inspiration programs (for crossover).
        temperature: Recommended sampling temperature.
        metadata: Additional metadata.
    """

    system_msg: str
    user_msg: str
    strategy: PromptStrategy
    parent: Program | None = None
    inspirations: list[Program] = field(default_factory=list)
    temperature: float = 0.7
    metadata: dict = field(default_factory=dict)


# =============================================================================
# Prompt Sampler
# =============================================================================


class PromptSampler:
    """Samples diverse prompts for program evolution.

    Balances exploration (random strategies, diverse parents) with
    exploitation (focusing on successful strategies and elite parents).
    """

    def __init__(
        self,
        config: SamplerConfig | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the sampler.

        Args:
            config: Sampler configuration.
            seed: Random seed for reproducibility.
        """
        self.config = config or SamplerConfig()
        self._rng = random.Random(seed)

        # Strategy statistics
        self._strategy_counts: dict[PromptStrategy, int] = {
            s: 0 for s in PromptStrategy
        }
        self._strategy_successes: dict[PromptStrategy, int] = {
            s: 0 for s in PromptStrategy
        }

        # Normalize weights
        self._normalized_weights = self._normalize_weights(
            self.config.strategy_weights
        )

    def _normalize_weights(
        self, weights: dict[PromptStrategy, float]
    ) -> dict[PromptStrategy, float]:
        """Normalize strategy weights to sum to 1."""
        total = sum(weights.values())
        if total == 0:
            # Uniform if all zero
            return {s: 1 / len(PromptStrategy) for s in PromptStrategy}
        return {s: w / total for s, w in weights.items()}

    # -------------------------------------------------------------------------
    # Strategy Selection
    # -------------------------------------------------------------------------

    def select_strategy(self) -> PromptStrategy:
        """Select a prompting strategy based on weights.

        Returns:
            Selected strategy.
        """
        strategies = list(self._normalized_weights.keys())
        weights = [self._normalized_weights[s] for s in strategies]
        return self._rng.choices(strategies, weights=weights, k=1)[0]

    def select_temperature(self) -> float:
        """Sample a temperature value.

        Returns:
            Temperature value.
        """
        low, high = self.config.temperature_range
        return self._rng.uniform(low, high)

    # -------------------------------------------------------------------------
    # Parent Selection
    # -------------------------------------------------------------------------

    async def select_parent(
        self,
        database: ProgramDatabase,
        exclude_ids: set[str] | None = None,
    ) -> Program | None:
        """Select a parent program for mutation.

        Uses elite-biased sampling to favor higher-scoring programs
        while maintaining diversity.

        Args:
            database: Program database.
            exclude_ids: Program IDs to exclude.

        Returns:
            Selected parent program, or None if no candidates.
        """
        # Get correct programs from archive
        archive = await database.get_archive()
        if not archive:
            # Fall back to any programs
            archive = list((await database.get_all())[:100])

        if not archive:
            return None

        # Filter excluded
        if exclude_ids:
            archive = [p for p in archive if p.id not in exclude_ids]

        if not archive:
            return None

        # Elite-biased selection: weight by score^bias
        weights = []
        for prog in archive:
            # Ensure positive weight
            score = max(prog.combined_score, 0.01)
            weight = score ** self.config.elite_bias
            weights.append(weight)

        selected = self._rng.choices(archive, weights=weights, k=1)[0]
        return selected

    async def select_inspirations(
        self,
        database: ProgramDatabase,
        current_id: str | None = None,
        num: int | None = None,
    ) -> list[Program]:
        """Select inspiration programs for crossover.

        Args:
            database: Program database.
            current_id: ID of current program to exclude.
            num: Number of inspirations (defaults to config).

        Returns:
            List of inspiration programs.
        """
        num = num or self.config.num_inspirations

        archive = await database.get_archive()
        if not archive:
            return []

        # Exclude current
        candidates = [p for p in archive if p.id != current_id]

        if len(candidates) <= num:
            return candidates

        # Diverse selection: mix of top scores and random
        candidates = sorted(
            candidates, key=lambda p: p.combined_score, reverse=True
        )

        # Take some top performers
        top_k = min(num // 2, len(candidates))
        inspirations = candidates[:top_k]

        # Add random samples
        remaining = [p for p in candidates[top_k:]]
        random_k = min(num - len(inspirations), len(remaining))
        if random_k > 0:
            inspirations.extend(self._rng.sample(remaining, random_k))

        return inspirations

    # -------------------------------------------------------------------------
    # Prompt Generation
    # -------------------------------------------------------------------------

    async def sample(
        self,
        database: ProgramDatabase,
        problem_description: str,
        initial_code: str = "",
        language: str = "python",
    ) -> PromptSample:
        """Sample a complete prompt for LLM query.

        Args:
            database: Program database.
            problem_description: Problem description.
            initial_code: Initial/seed code.
            language: Programming language.

        Returns:
            PromptSample ready for LLM query.
        """
        from shipha.prompts.base import PromptBuilder
        from shipha.prompts.crossover import build_crossover_prompt
        from shipha.prompts.diff import build_diff_prompt

        strategy = self.select_strategy()
        temperature = self.select_temperature()
        parent = await self.select_parent(database)

        self._strategy_counts[strategy] += 1

        # Build prompt based on strategy
        if strategy == PromptStrategy.CROSSOVER and parent:
            inspirations = await self.select_inspirations(database, parent.id)
            system_msg, user_msg = build_crossover_prompt(
                current=parent,
                inspirations=inspirations,
                language=language,
            )
            return PromptSample(
                system_msg=system_msg,
                user_msg=user_msg,
                strategy=strategy,
                parent=parent,
                inspirations=inspirations,
                temperature=temperature,
            )

        elif strategy == PromptStrategy.DIFF and parent:
            system_msg, user_msg, _ = build_diff_prompt(
                code=parent.code,
                problem=problem_description,
                feedback=parent.text_feedback if self.config.include_feedback else None,
                language=language,
            )
            return PromptSample(
                system_msg=system_msg,
                user_msg=user_msg,
                strategy=strategy,
                parent=parent,
                temperature=temperature,
            )

        else:
            # Fall back to full rewrite
            builder = PromptBuilder(
                problem_description=problem_description,
                language=language,
            )

            if parent:
                system_msg, user_msg = builder.build_improvement_prompt(
                    code=parent.code,
                    feedback=parent.text_feedback if self.config.include_feedback else None,
                )
            else:
                system_msg, user_msg = builder.build_initial_prompt(
                    examples=initial_code,
                )

            return PromptSample(
                system_msg=system_msg,
                user_msg=user_msg,
                strategy=PromptStrategy.FULL,
                parent=parent,
                temperature=temperature,
            )

    async def sample_batch(
        self,
        database: ProgramDatabase,
        problem_description: str,
        batch_size: int,
        initial_code: str = "",
        language: str = "python",
    ) -> list[PromptSample]:
        """Sample multiple prompts for batch LLM queries.

        Args:
            database: Program database.
            problem_description: Problem description.
            batch_size: Number of prompts to sample.
            initial_code: Initial/seed code.
            language: Programming language.

        Returns:
            List of PromptSamples.
        """
        samples = []
        used_parent_ids: set[str] = set()

        for _ in range(batch_size):
            sample = await self.sample(
                database=database,
                problem_description=problem_description,
                initial_code=initial_code,
                language=language,
            )

            # Track parent usage for diversity
            if sample.parent:
                used_parent_ids.add(sample.parent.id)

            samples.append(sample)

        return samples

    # -------------------------------------------------------------------------
    # Feedback
    # -------------------------------------------------------------------------

    def record_success(self, strategy: PromptStrategy) -> None:
        """Record a successful program from a strategy.

        Args:
            strategy: The strategy that produced a success.
        """
        self._strategy_successes[strategy] += 1

    def update_weights(self, learning_rate: float = 0.1) -> None:
        """Update strategy weights based on success rates.

        Increases weights for successful strategies using
        multiplicative weight update.

        Args:
            learning_rate: Weight update rate.
        """
        for strategy in PromptStrategy:
            if strategy in self._normalized_weights:
                count = self._strategy_counts.get(strategy, 0)
                successes = self._strategy_successes.get(strategy, 0)

                if count > 0:
                    success_rate = successes / count
                    # Multiplicative update
                    self._normalized_weights[strategy] *= (
                        1 + learning_rate * (success_rate - 0.5)
                    )

        # Re-normalize
        self._normalized_weights = self._normalize_weights(
            self._normalized_weights
        )

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def stats(self) -> dict:
        """Get sampler statistics.

        Returns:
            Statistics dictionary.
        """
        stats: dict = {
            "strategy_counts": dict(self._strategy_counts),
            "strategy_successes": dict(self._strategy_successes),
            "current_weights": dict(self._normalized_weights),
        }

        # Compute success rates
        success_rates = {}
        for strategy in PromptStrategy:
            count = self._strategy_counts.get(strategy, 0)
            if count > 0:
                success_rates[strategy.value] = (
                    self._strategy_successes.get(strategy, 0) / count
                )
        stats["success_rates"] = success_rates

        return stats

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._strategy_counts = {s: 0 for s in PromptStrategy}
        self._strategy_successes = {s: 0 for s in PromptStrategy}
