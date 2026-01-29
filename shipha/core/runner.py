"""
Evolution runner - main orchestrator for program evolution.

Coordinates the full evolution loop:
1. Sample prompts (parent selection, strategy selection)
2. Generate candidate programs via LLM
3. Apply edits to parent code
4. Evaluate candidates
5. Update database and statistics
6. Repeat until convergence or budget exhausted
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from shipha.core.config import EvolutionConfig, ShiphaConfig
from shipha.core.sampler import PromptSample, PromptSampler, SamplerConfig
from shipha.database import Program, ProgramDatabase
from shipha.edit import apply_diff_edit, apply_full_rewrite
from shipha.launch.scheduler import JobScheduler, LocalJobConfig
from shipha.llm import LLMClient

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class Evaluator(Protocol):
    """Protocol for program evaluators."""

    async def evaluate(self, program: Program) -> Program:
        """Evaluate a program and return with updated metrics."""
        ...


class NoveltyFilter(Protocol):
    """Protocol for novelty filtering."""

    async def is_novel(self, program: Program) -> bool:
        """Check if program is sufficiently novel."""
        ...

    async def add_to_archive(self, program: Program) -> None:
        """Add program to novelty archive."""
        ...

    async def filter_novel(self, programs: list[Program]) -> list[Program]:
        """Filter programs to keep only novel ones."""
        ...


# =============================================================================
# Callbacks
# =============================================================================


@dataclass
class EvolutionCallbacks:
    """Callbacks for evolution events.

    All callbacks are optional and receive relevant context.
    """

    on_iteration_start: Callable[[int], None] | None = None
    on_iteration_end: Callable[[int, dict[str, Any]], None] | None = None
    on_new_best: Callable[[Program], None] | None = None
    on_candidate_generated: Callable[[Program], None] | None = None
    on_candidate_evaluated: Callable[[Program], None] | None = None
    on_checkpoint: Callable[[int, Path], None] | None = None


# =============================================================================
# Evolution State
# =============================================================================


@dataclass
class EvolutionState:
    """Current state of evolution.

    Tracks progress, statistics, and best results.
    """

    iteration: int = 0
    total_candidates: int = 0
    total_evaluations: int = 0
    total_llm_cost: float = 0.0

    best_score: float = 0.0
    best_program_id: str | None = None

    stagnation_count: int = 0
    start_time: float = field(default_factory=time.time)

    # Per-iteration stats
    candidates_this_iter: int = 0
    correct_this_iter: int = 0
    novel_this_iter: int = 0
    improved_this_iter: int = 0

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def candidates_per_second(self) -> float:
        """Average candidates generated per second."""
        if self.elapsed_seconds == 0:
            return 0.0
        return self.total_candidates / self.elapsed_seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "iteration": self.iteration,
            "total_candidates": self.total_candidates,
            "total_evaluations": self.total_evaluations,
            "total_llm_cost": self.total_llm_cost,
            "best_score": self.best_score,
            "best_program_id": self.best_program_id,
            "stagnation_count": self.stagnation_count,
            "elapsed_seconds": self.elapsed_seconds,
            "candidates_per_second": self.candidates_per_second,
        }


# =============================================================================
# Evolution Runner
# =============================================================================


class EvolutionRunner:
    """Main evolution loop orchestrator.

    Coordinates all components to evolve programs towards
    higher scores.
    """

    # Type annotations for instance variables
    config: EvolutionConfig
    _full_config: ShiphaConfig | None
    llm: LLMClient | None
    database: ProgramDatabase | None
    evaluator: Evaluator | None
    novelty: NoveltyFilter | None
    sampler: PromptSampler
    scheduler: JobScheduler
    callbacks: EvolutionCallbacks
    state: EvolutionState
    problem_description: str
    initial_code: str
    language: str

    def __init__(
        self,
        config: ShiphaConfig | EvolutionConfig | None = None,
        llm_client: LLMClient | None = None,
        database: ProgramDatabase | None = None,
        evaluator: Evaluator | None = None,
        novelty_filter: NoveltyFilter | None = None,
        sampler: PromptSampler | None = None,
        scheduler: JobScheduler | None = None,
        callbacks: EvolutionCallbacks | None = None,
    ) -> None:
        """Initialize the evolution runner.

        Args:
            config: Evolution configuration.
            llm_client: LLM client for code generation.
            database: Program database.
            evaluator: Program evaluator.
            novelty_filter: Novelty filter (optional).
            sampler: Prompt sampler.
            scheduler: Job scheduler.
            callbacks: Event callbacks.
        """
        # Handle config types
        if isinstance(config, ShiphaConfig):
            self.config = config.evolution
            self._full_config = config
        elif isinstance(config, EvolutionConfig):
            self.config = config
            self._full_config = None
        else:
            self.config = EvolutionConfig()
            self._full_config = None

        # Components
        self.llm = llm_client
        self.database = database
        self.evaluator = evaluator
        self.novelty = novelty_filter
        self.sampler = sampler or PromptSampler(SamplerConfig())
        self.scheduler = scheduler or JobScheduler(LocalJobConfig())
        self.callbacks = callbacks or EvolutionCallbacks()

        # State
        self.state = EvolutionState()

        # Problem specification (set via configure)
        self.problem_description = ""
        self.initial_code: str = ""
        self.language: str = "python"

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def configure(
        self,
        problem_description: str,
        initial_code: str = "",
        language: str = "python",
    ) -> EvolutionRunner:
        """Configure the problem to solve.

        Args:
            problem_description: Problem statement.
            initial_code: Starting code template.
            language: Programming language.

        Returns:
            Self for chaining.
        """
        self.problem_description = problem_description
        self.initial_code = initial_code
        self.language = language
        return self

    def set_components(
        self,
        llm_client: LLMClient | None = None,
        database: ProgramDatabase | None = None,
        evaluator: Evaluator | None = None,
        novelty_filter: NoveltyFilter | None = None,
    ) -> EvolutionRunner:
        """Set or update components.

        Args:
            llm_client: LLM client.
            database: Program database.
            evaluator: Program evaluator.
            novelty_filter: Novelty filter.

        Returns:
            Self for chaining.
        """
        if llm_client:
            self.llm = llm_client
        if database:
            self.database = database
        if evaluator:
            self.evaluator = evaluator
        if novelty_filter:
            self.novelty = novelty_filter
        return self

    # -------------------------------------------------------------------------
    # Main Evolution Loop
    # -------------------------------------------------------------------------

    async def run(self) -> Program:
        """Run the full evolution loop.

        Returns:
            The best program found.

        Raises:
            ValueError: If required components are not configured.
        """
        self._validate_setup()

        logger.info(
            f"Starting evolution: {self.config.num_iterations} iterations, "
            f"batch_size={self.config.batch_size}"
        )

        try:
            for iteration in range(1, self.config.num_iterations + 1):
                self.state.iteration = iteration

                # Callbacks
                if self.callbacks.on_iteration_start:
                    self.callbacks.on_iteration_start(iteration)

                # Run one iteration
                iter_stats = await self._run_iteration()

                # Callbacks
                if self.callbacks.on_iteration_end:
                    self.callbacks.on_iteration_end(iteration, iter_stats)

                # Check early stopping
                if self._should_stop():
                    logger.info(
                        f"Early stopping at iteration {iteration}: "
                        f"score={self.state.best_score:.4f}"
                    )
                    break

                # Checkpoint
                if iteration % self.config.checkpoint_interval == 0:
                    await self._checkpoint(iteration)

                # Meta-analysis
                if iteration % self.config.meta_analysis_interval == 0:
                    await self._run_meta_analysis()

        except KeyboardInterrupt:
            logger.info("Evolution interrupted by user")

        # Get best program
        best = await self.database.get_best()  # type: ignore
        if best:
            logger.info(
                f"Evolution complete: best_score={best.combined_score:.4f}, "
                f"iterations={self.state.iteration}"
            )

        return best or Program(code="", combined_score=0.0)

    async def _run_iteration(self) -> dict[str, Any]:
        """Run a single evolution iteration.

        Returns:
            Iteration statistics.
        """
        # Reset per-iteration counters
        self.state.candidates_this_iter = 0
        self.state.correct_this_iter = 0
        self.state.novel_this_iter = 0
        self.state.improved_this_iter = 0

        # Sample prompts
        prompts = await self.sampler.sample_batch(
            database=self.database,  # type: ignore
            problem_description=self.problem_description,
            batch_size=self.config.batch_size,
            initial_code=self.initial_code,
            language=self.language,
        )

        # Generate candidates
        candidates = await self._generate_candidates(prompts)

        # Evaluate candidates
        evaluated = await self._evaluate_candidates(candidates)

        # Filter for novelty
        if self.novelty:
            novel = await self.novelty.filter_novel(evaluated)
            self.state.novel_this_iter = len(novel)
        else:
            novel = evaluated

        # Add to database
        for prog in novel:
            await self.database.add(prog)  # type: ignore

            # Track improvements
            if prog.correct:
                self.state.correct_this_iter += 1
                if prog.combined_score > self.state.best_score:
                    self.state.best_score = prog.combined_score
                    self.state.best_program_id = prog.id
                    self.state.stagnation_count = 0
                    self.state.improved_this_iter += 1

                    if self.callbacks.on_new_best:
                        self.callbacks.on_new_best(prog)

                # Add to novelty archive
                if self.novelty:
                    await self.novelty.add_to_archive(prog)

        # Update sampler weights based on results
        for prompt in prompts:
            if prompt.parent is not None:
                parent_id = prompt.parent.id
                has_correct_child = any(
                    p.correct
                    for p in novel
                    if p.parent and p.parent.id == parent_id
                )
                if has_correct_child:
                    self.sampler.record_success(prompt.strategy)

        if self.state.improved_this_iter == 0:
            self.state.stagnation_count += 1

        return {
            "candidates": self.state.candidates_this_iter,
            "correct": self.state.correct_this_iter,
            "novel": self.state.novel_this_iter,
            "improved": self.state.improved_this_iter,
            "best_score": self.state.best_score,
        }

    # -------------------------------------------------------------------------
    # Candidate Generation
    # -------------------------------------------------------------------------

    async def _generate_candidates(
        self,
        prompts: list[PromptSample],
    ) -> list[Program]:
        """Generate candidate programs from prompts.

        Args:
            prompts: List of sampled prompts.

        Returns:
            List of generated programs.
        """
        async def generate_one(prompt: PromptSample) -> Program | None:
            try:
                result = await self.llm.query(  # type: ignore
                    msg=prompt.user_msg,
                    system_msg=prompt.system_msg,
                    temperature=prompt.temperature,
                )

                if result is None:
                    return None

                # Apply edit
                code = self._apply_edit(prompt, result.content)

                if code is None:
                    return None

                # Create program
                prog = Program(
                    code=code,
                    language=self.language,
                    parent_id=prompt.parent.id if prompt.parent else None,
                    generation=self.state.iteration,
                )

                self.state.candidates_this_iter += 1
                self.state.total_candidates += 1
                self.state.total_llm_cost += result.cost

                if self.callbacks.on_candidate_generated:
                    self.callbacks.on_candidate_generated(prog)

                return prog

            except Exception as e:
                logger.warning(f"Failed to generate candidate: {e}")
                return None

        # Run in parallel
        tasks = [generate_one(p) for p in prompts]
        results = await asyncio.gather(*tasks)

        return [r for r in results if r is not None]

    def _apply_edit(self, prompt: PromptSample, response: str) -> str | None:
        """Apply LLM response to parent code.

        Args:
            prompt: The prompt used.
            response: LLM response.

        Returns:
            Modified code, or None if failed.
        """
        from shipha.core.sampler import PromptStrategy

        parent_code = prompt.parent.code if prompt.parent else self.initial_code

        if prompt.strategy == PromptStrategy.DIFF:
            outcome = apply_diff_edit(parent_code, response)
            if outcome.result.value in ("success", "partial"):
                return outcome.new_code
            # Fall back to full rewrite
            rewrite = apply_full_rewrite(parent_code, response, self.language)
            return rewrite.new_code if rewrite.success else None

        else:
            # Full rewrite
            rewrite = apply_full_rewrite(parent_code, response, self.language)
            return rewrite.new_code if rewrite.success else None

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    async def _evaluate_candidates(
        self,
        candidates: list[Program],
    ) -> list[Program]:
        """Evaluate candidate programs.

        Args:
            candidates: Programs to evaluate.

        Returns:
            Evaluated programs with updated metrics.
        """
        if not self.evaluator:
            # No evaluator - return as-is
            return candidates

        async def eval_one(prog: Program) -> Program:
            try:
                evaluated = await self.evaluator.evaluate(prog)  # type: ignore
                self.state.total_evaluations += 1

                if self.callbacks.on_candidate_evaluated:
                    self.callbacks.on_candidate_evaluated(evaluated)

                return evaluated

            except Exception as e:
                logger.warning(f"Evaluation failed for {prog.id}: {e}")
                return prog

        tasks = [eval_one(p) for p in candidates]
        return list(await asyncio.gather(*tasks))

    # -------------------------------------------------------------------------
    # Meta-Analysis
    # -------------------------------------------------------------------------

    async def _run_meta_analysis(self) -> None:
        """Run meta-analysis to extract improvement insights."""
        from shipha.prompts.meta import run_meta_analysis

        logger.info("Running meta-analysis...")

        try:
            # Get top programs for analysis
            archive = await self.database.get_archive()  # type: ignore
            if len(archive) < 3:
                return

            # Take diverse sample
            sample = archive[:min(5, len(archive))]
            best = await self.database.get_best()  # type: ignore

            # Run meta-analysis
            async def query_wrapper(msg: str, system_msg: str) -> str:
                result = await self.llm.query(  # type: ignore
                    msg=msg,
                    system_msg=system_msg,
                )
                return result.content if result else ""

            _, insights, improvements = await run_meta_analysis(
                programs=sample,
                llm_query_fn=query_wrapper,
                best_program=best,
            )

            logger.info(f"Meta-analysis insights: {insights[:200]}...")

            # Update sampler weights based on meta-analysis
            self.sampler.update_weights(learning_rate=0.1)

        except Exception as e:
            logger.warning(f"Meta-analysis failed: {e}")

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    async def _checkpoint(self, iteration: int) -> None:
        """Save checkpoint.

        Args:
            iteration: Current iteration.
        """
        if not self._full_config:
            return

        checkpoint_dir = Path(self._full_config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_{iteration:04d}.json"

        # Save state
        import json

        state_dict = {
            "iteration": iteration,
            "state": self.state.to_dict(),
            "sampler_stats": self.sampler.stats(),
            "llm_cost": self.state.total_llm_cost,
        }

        with open(checkpoint_path, "w") as f:
            json.dump(state_dict, f, indent=2)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

        if self.callbacks.on_checkpoint:
            self.callbacks.on_checkpoint(iteration, checkpoint_path)

    # -------------------------------------------------------------------------
    # Stopping Conditions
    # -------------------------------------------------------------------------

    def _should_stop(self) -> bool:
        """Check if evolution should stop early.

        Returns:
            True if should stop.
        """
        # Score threshold reached
        if self.state.best_score >= self.config.early_stop_score:
            return True

        # Stagnation
        if self.state.stagnation_count >= self.config.early_stop_patience:
            return True

        return False

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate_setup(self) -> None:
        """Validate that required components are configured."""
        errors = []

        if not self.llm:
            errors.append("LLM client is required")
        if not self.database:
            errors.append("Database is required")
        if not self.problem_description:
            errors.append("Problem description is required")

        if errors:
            raise ValueError(f"Setup incomplete: {', '.join(errors)}")

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Get runner statistics.

        Returns:
            Statistics dictionary.
        """
        stats = self.state.to_dict()
        stats["sampler"] = self.sampler.stats()
        stats["scheduler"] = self.scheduler.stats()

        if self.llm:
            stats["llm"] = {
                "total_cost": self.llm.total_cost,
                "query_count": self.llm.query_count,
            }

        return stats
