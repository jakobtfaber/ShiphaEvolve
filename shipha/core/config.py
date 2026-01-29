"""
Hydra-compatible structured configuration for ShiphaEvolve.

Defines dataclass configs for all major components with sensible defaults.
Supports Hydra instantiation via `_target_` fields and composition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# Enums for Type-Safe Configuration
# =============================================================================


class EvaluatorTrust(str, Enum):
    """Trust levels for program evaluation."""

    SANDBOX = "sandbox"  # Full isolation, resource limits
    SUBPROCESS = "subprocess"  # Subprocess with timeout
    DIRECT = "direct"  # Direct execution (trusted code only)


class BanditType(str, Enum):
    """Multi-armed bandit algorithms for model selection."""

    FIXED = "fixed"  # Uniform random selection
    UCB = "ucb"  # Asymmetric UCB with exploration bonus
    THOMPSON = "thompson"  # Thompson sampling (future)


class PromptStrategy(str, Enum):
    """Prompt strategy for code generation."""

    DIFF = "diff"  # SEARCH/REPLACE diff prompts
    FULL = "full"  # Full code rewrite
    CROSSOVER = "crossover"  # Multi-parent crossover
    META = "meta"  # Meta-analysis guided


# =============================================================================
# LLM Configuration
# =============================================================================


@dataclass
class LLMConfig:
    """Configuration for LLM client.

    Attributes:
        model_names: List of model identifiers (litellm format).
        temperatures: Temperature(s) for sampling.
        max_tokens: Maximum output tokens per query.
        max_retries: Retry count for transient failures.
        timeout: Request timeout in seconds.
        verbose: Enable detailed logging.
    """

    _target_: str = "shipha.llm.LLMClient"
    model_names: list[str] = field(
        default_factory=lambda: ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet-20241022"]
    )
    temperatures: list[float] = field(default_factory=lambda: [0.7, 0.9])
    max_tokens: int = 4096
    max_retries: int = 3
    timeout: float = 120.0
    verbose: bool = True


@dataclass
class BanditConfig:
    """Configuration for model selection bandit.

    Attributes:
        bandit_type: Algorithm type (fixed, ucb, thompson).
        exploration_coef: UCB exploration coefficient.
        epsilon: Minimum exploration probability.
        auto_decay: Decay factor for adaptive learning.
        asymmetric_scaling: Only count positive improvements.
    """

    _target_: str = "shipha.llm.bandit.AsymmetricUCB"
    bandit_type: BanditType = BanditType.UCB
    exploration_coef: float = 1.0
    epsilon: float = 0.2
    auto_decay: float = 0.95
    asymmetric_scaling: bool = True
    shift_by_baseline: bool = True
    shift_by_parent: bool = True


@dataclass
class EmbeddingConfig:
    """Configuration for embedding client.

    Attributes:
        primary_model: Primary embedding model.
        fallback_model: Fallback if primary fails.
        batch_size: Batch size for embedding requests.
        cache_embeddings: Enable embedding cache.
    """

    _target_: str = "shipha.llm.embedding.EmbeddingClient"
    primary_model: str = "text-embedding-3-large"
    fallback_model: str = "text-embedding-3-small"
    batch_size: int = 100
    cache_embeddings: bool = True
    dimensions: int = 3072  # text-embedding-3-large default


# =============================================================================
# Database Configuration
# =============================================================================


@dataclass
class DatabaseConfig:
    """Configuration for program database.

    Attributes:
        db_path: SQLite database file path.
        archive_size: Maximum elite archive size.
        num_islands: Number of island populations.
        enable_wal: Enable SQLite WAL mode.
    """

    _target_: str = "shipha.database.ProgramDatabase"
    db_path: str = "evolution_db.sqlite"
    archive_size: int = 100
    num_islands: int = 4
    enable_wal: bool = True


# =============================================================================
# Evaluator Configuration
# =============================================================================


@dataclass
class EvaluatorConfig:
    """Configuration for program evaluator.

    Attributes:
        trust_level: Execution trust level.
        timeout_seconds: Per-program timeout.
        memory_limit_mb: Memory limit in MB.
        num_workers: Parallel evaluation workers.
        tiered_evaluation: Enable tier-based trust progression.
    """

    _target_: str = "shipha.core.evaluator.TieredEvaluator"
    trust_level: EvaluatorTrust = EvaluatorTrust.SANDBOX
    timeout_seconds: float = 30.0
    memory_limit_mb: int = 512
    num_workers: int = 4
    tiered_evaluation: bool = True


# =============================================================================
# Prompt Configuration
# =============================================================================


@dataclass
class PromptConfig:
    """Configuration for prompt generation.

    Attributes:
        strategy: Primary prompting strategy.
        include_feedback: Include evaluation feedback.
        include_lineage: Include parent program history.
        max_context_tokens: Maximum context size.
        num_inspirations: Number of crossover inspirations.
    """

    strategy: PromptStrategy = PromptStrategy.DIFF
    include_feedback: bool = True
    include_lineage: bool = True
    max_context_tokens: int = 8000
    num_inspirations: int = 3


# =============================================================================
# Novelty Configuration
# =============================================================================


@dataclass
class NoveltyConfig:
    """Configuration for novelty judge.

    Attributes:
        enabled: Enable novelty filtering.
        similarity_threshold: Maximum similarity to archive (0-1).
        embedding_model: Model for code embeddings.
        cache_size: LRU cache size for embeddings.
    """

    _target_: str = "shipha.core.novelty.NoveltyJudge"
    enabled: bool = True
    similarity_threshold: float = 0.85
    use_semantic_similarity: bool = True
    use_syntactic_similarity: bool = True
    syntactic_weight: float = 0.3


# =============================================================================
# Scheduler Configuration
# =============================================================================


@dataclass
class SchedulerConfig:
    """Configuration for job scheduler.

    Attributes:
        max_concurrent_jobs: Maximum parallel LLM queries.
        max_concurrent_evals: Maximum parallel evaluations.
        queue_size: Job queue buffer size.
        priority_boost_for_correct: Boost priority for correct parents.
    """

    _target_: str = "shipha.launch.scheduler.JobScheduler"
    max_concurrent_jobs: int = 10
    max_concurrent_evals: int = 4
    queue_size: int = 100
    priority_boost_for_correct: float = 1.5


# =============================================================================
# Evolution Configuration
# =============================================================================


@dataclass
class EvolutionConfig:
    """Configuration for evolution runner.

    Attributes:
        num_iterations: Total evolution iterations.
        batch_size: Programs per iteration.
        elite_ratio: Fraction of population preserved.
        mutation_rate: Probability of mutation vs crossover.
        meta_analysis_interval: Iterations between meta-analysis.
        checkpoint_interval: Iterations between checkpoints.
    """

    num_iterations: int = 100
    batch_size: int = 8
    elite_ratio: float = 0.1
    mutation_rate: float = 0.7
    meta_analysis_interval: int = 10
    checkpoint_interval: int = 5
    early_stop_score: float = 1.0
    early_stop_patience: int = 20


# =============================================================================
# Root Configuration
# =============================================================================


@dataclass
class ShiphaConfig:
    """Root configuration for ShiphaEvolve.

    Composes all sub-configurations into a single structured config.
    Use with Hydra: `@hydra.main(config_path=".", config_name="config")`
    """

    # Core components
    llm: LLMConfig = field(default_factory=LLMConfig)
    bandit: BanditConfig = field(default_factory=BanditConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)

    # Strategy components
    prompt: PromptConfig = field(default_factory=PromptConfig)
    novelty: NoveltyConfig = field(default_factory=NoveltyConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)

    # Task specification
    task_file: str = ""
    problem_description: str = ""
    initial_code: str = ""
    language: str = "python"

    # Runtime options
    seed: int = 42
    output_dir: str = "outputs"
    experiment_name: str = "shipha_run"
    log_level: str = "INFO"


# =============================================================================
# Factory Functions
# =============================================================================


def create_default_config() -> ShiphaConfig:
    """Create a ShiphaConfig with all defaults."""
    return ShiphaConfig()


def create_fast_config() -> ShiphaConfig:
    """Create a fast config for testing/debugging."""
    return ShiphaConfig(
        llm=LLMConfig(
            model_names=["gpt-4o-mini"],
            max_tokens=1024,
        ),
        evolution=EvolutionConfig(
            num_iterations=10,
            batch_size=4,
        ),
        evaluator=EvaluatorConfig(
            timeout_seconds=10.0,
            num_workers=2,
        ),
    )


def create_production_config() -> ShiphaConfig:
    """Create a production config for serious runs."""
    return ShiphaConfig(
        llm=LLMConfig(
            model_names=["gpt-4o", "claude-3-5-sonnet-20241022", "gpt-4o-mini"],
            temperatures=[0.7, 0.8, 0.9],
            max_tokens=8192,
        ),
        bandit=BanditConfig(
            exploration_coef=1.5,
            epsilon=0.1,
        ),
        evolution=EvolutionConfig(
            num_iterations=500,
            batch_size=16,
            meta_analysis_interval=25,
        ),
        evaluator=EvaluatorConfig(
            timeout_seconds=60.0,
            memory_limit_mb=1024,
            num_workers=8,
        ),
        novelty=NoveltyConfig(
            similarity_threshold=0.9,
        ),
    )
