"""
Pytest fixtures for ShiphaEvolve tests.

Provides shared fixtures for testing LLM clients, database operations,
and evaluation pipelines.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shipha.database import Program, ProgramDatabase
from shipha.llm import LLMClient, QueryResult
from shipha.llm.bandit import AsymmetricUCB, FixedSampler


# =============================================================================
# Environment Configuration
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests requiring external APIs"
    )


# =============================================================================
# Database Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary database file path."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)
    # Also cleanup WAL and SHM files
    for suffix in ["-wal", "-shm"]:
        wal_path = db_path + suffix
        if os.path.exists(wal_path):
            os.unlink(wal_path)


@pytest.fixture
async def memory_db() -> AsyncGenerator[ProgramDatabase, None]:
    """Create an in-memory database for testing."""
    db = ProgramDatabase(":memory:")
    await db.connect()
    yield db
    await db.close()


@pytest.fixture
async def temp_db(temp_db_path: str) -> AsyncGenerator[ProgramDatabase, None]:
    """Create a temporary file-based database for testing."""
    db = ProgramDatabase(temp_db_path)
    await db.connect()
    yield db
    await db.close()


@pytest.fixture
def sample_program() -> Program:
    """Create a sample program for testing."""
    return Program(
        id="test_prog_001",
        code="def solve(x): return x * 2",
        language="python",
        combined_score=0.75,
        public_metrics={"accuracy": 0.9, "speed": 0.6},
        private_metrics={"memory_mb": 128},
        correct=True,
        generation=1,
        text_feedback="Good performance on accuracy",
    )


@pytest.fixture
def sample_programs() -> list[Program]:
    """Create multiple sample programs for testing."""
    return [
        Program(
            id=f"test_prog_{i:03d}",
            code=f"def solve(x): return x * {i}",
            combined_score=0.5 + i * 0.1,
            correct=i % 2 == 0,
            generation=i,
        )
        for i in range(5)
    ]


# =============================================================================
# LLM Fixtures
# =============================================================================


@pytest.fixture
def mock_litellm_response() -> MagicMock:
    """Create a mock litellm response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "def improved(x): return x * 3"
    response.usage = MagicMock()
    response.usage.prompt_tokens = 100
    response.usage.completion_tokens = 50
    return response


@pytest.fixture
def mock_llm_client(mock_litellm_response: MagicMock) -> Generator[LLMClient, None, None]:
    """Create an LLM client with mocked litellm."""
    with patch("shipha.llm.client.litellm") as mock_litellm:
        mock_litellm.acompletion = AsyncMock(return_value=mock_litellm_response)
        client = LLMClient(model_names=["gpt-4o-mini"], verbose=False)
        yield client


@pytest.fixture
def fixed_sampler() -> FixedSampler:
    """Create a fixed sampler for deterministic testing."""
    return FixedSampler(
        n_arms=3,
        arm_names=["gpt-4o", "claude-3-5-sonnet-20241022", "gpt-4o-mini"],
    )


@pytest.fixture
def ucb_bandit() -> AsymmetricUCB:
    """Create a UCB bandit for testing."""
    return AsymmetricUCB(
        n_arms=3,
        arm_names=["gpt-4o", "claude-3-5-sonnet-20241022", "gpt-4o-mini"],
        exploration_coef=1.0,
        epsilon=0.2,
    )


# =============================================================================
# Query Result Fixtures
# =============================================================================


@pytest.fixture
def sample_query_result() -> QueryResult:
    """Create a sample query result."""
    return QueryResult(
        content="def improved(x): return x * 3",
        model_name="gpt-4o",
        input_tokens=100,
        output_tokens=50,
        cost=0.0025,
        input_cost=0.00025,
        output_cost=0.002,
        latency_ms=1500.0,
        msg="Improve this function",
        system_msg="You are an expert",
        arm_index=0,
        model_posteriors={"gpt-4o": 0.5, "claude": 0.3, "gpt-4o-mini": 0.2},
    )


# =============================================================================
# Integration Test Helpers
# =============================================================================


def has_openai_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def has_anthropic_key() -> bool:
    """Check if Anthropic API key is available."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


skip_without_openai = pytest.mark.skipif(
    not has_openai_key(),
    reason="OPENAI_API_KEY not set",
)

skip_without_anthropic = pytest.mark.skipif(
    not has_anthropic_key(),
    reason="ANTHROPIC_API_KEY not set",
)

