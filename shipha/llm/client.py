"""
Async LLM Client for ShiphaEvolve

Provides unified async interface to multiple LLM providers via litellm,
with multi-armed bandit integration for intelligent model selection.

Usage:
    from shipha.llm import LLMClient

    client = LLMClient(["gpt-4o", "claude-3-5-sonnet-20241022"])
    result = await client.query("Optimize this function", system_msg="You are a code expert")

Features:
    - Async-first with asyncio.gather() for concurrent queries
    - Multi-armed bandit model selection (AsymmetricUCB)
    - Unified interface via litellm (OpenAI, Anthropic, DeepSeek, Gemini)
    - Cost tracking and token usage metrics
    - Automatic retries with exponential backoff
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

import litellm
import numpy as np
from pydantic import BaseModel

from shipha.llm.bandit import AsymmetricUCB, BanditBase, FixedSampler
from shipha.llm.models import (
    QueryResult,
    REASONING_MODELS,
    calculate_cost,
)

# Configure litellm
litellm.drop_params = True  # Drop unsupported params instead of erroring
litellm.set_verbose = False

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY_BASE = 1.0  # seconds


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    model_names: list[str] = field(default_factory=lambda: ["gpt-4o"])
    """List of model names to use for selection."""

    temperatures: list[float] = field(default_factory=lambda: [0.7])
    """Temperature(s) to sample from."""

    max_tokens: int = 4096
    """Maximum output tokens."""

    reasoning_effort: str = "auto"
    """Reasoning effort for compatible models (auto, low, medium, high)."""

    timeout: float = 120.0
    """Request timeout in seconds."""

    verbose: bool = True
    """Whether to log query details."""


class LLMClient:
    """
    Async LLM client with multi-armed bandit model selection.

    Uses litellm as unified backend to support OpenAI, Anthropic, DeepSeek,
    Gemini, and OpenRouter models through a single interface.

    Args:
        model_names: Model(s) to use. If multiple, bandit selects among them.
        model_selection: Bandit algorithm for model selection. Defaults to FixedSampler.
        temperatures: Temperature(s) to sample from.
        max_tokens: Maximum output tokens.
        reasoning_effort: For reasoning models (o1, deepseek-reasoner).
        verbose: Whether to log query details.

    Example:
        >>> client = LLMClient(["gpt-4o", "claude-3-5-sonnet-20241022"])
        >>> result = await client.query(
        ...     msg="Explain quicksort",
        ...     system_msg="You are a CS professor"
        ... )
        >>> print(result.content)
    """

    def __init__(
        self,
        model_names: str | list[str] = "gpt-4o",
        model_selection: BanditBase | None = None,
        temperatures: float | list[float] = 0.7,
        max_tokens: int = 4096,
        reasoning_effort: str = "auto",
        output_model: type[BaseModel] | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize the LLM client."""
        # Normalize to lists
        if isinstance(model_names, str):
            model_names = [model_names]
        if isinstance(temperatures, float):
            temperatures = [temperatures]

        self.model_names = model_names
        self.temperatures = temperatures
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.output_model = output_model
        self.verbose = verbose

        # Initialize bandit for model selection
        if model_selection is None:
            model_selection = FixedSampler(
                n_arms=len(model_names),
                arm_names=model_names,
            )
        self.bandit = model_selection

        # Track cumulative costs
        self._total_cost = 0.0
        self._query_count = 0

    @property
    def total_cost(self) -> float:
        """Total cost across all queries."""
        return self._total_cost

    @property
    def query_count(self) -> int:
        """Total number of queries made."""
        return self._query_count

    def _sample_kwargs(self) -> dict[str, Any]:
        """Sample query parameters using bandit."""
        # Get posterior from bandit
        posterior = self.bandit.posterior()
        model_posteriors = dict(zip(self.model_names, posterior))

        # Sample model based on posterior probabilities
        arm_idx = int(np.random.choice(len(self.model_names), p=posterior))
        model_name = self.model_names[arm_idx]

        # Sample temperature
        temperature = random.choice(self.temperatures)

        # Reasoning models require temperature=1.0
        if model_name in REASONING_MODELS:
            temperature = 1.0

        return {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            "arm_index": arm_idx,
            "model_posteriors": model_posteriors,
        }

    async def query(
        self,
        msg: str,
        system_msg: str = "",
        msg_history: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> QueryResult | None:
        """
        Execute a single async query to the LLM.

        Args:
            msg: User message to send.
            system_msg: System prompt.
            msg_history: Previous messages in conversation.
            **kwargs: Additional kwargs passed to litellm.

        Returns:
            QueryResult with response content and metadata, or None on failure.
        """
        if msg_history is None:
            msg_history = []

        # Sample parameters
        sampled = self._sample_kwargs()
        model = sampled["model"]
        temperature = sampled["temperature"]
        arm_idx = sampled["arm_index"]
        model_posteriors = sampled["model_posteriors"]

        # Build messages
        messages: list[dict[str, str]] = []
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        messages.extend(msg_history)
        messages.append({"role": "user", "content": msg})

        if self.verbose:
            logger.info(f"[QUERY] model={model} temp={temperature:.2f}")

        # Execute with retries
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.perf_counter()

                # Call litellm async
                response = await litellm.acompletion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    **kwargs,
                )

                latency_ms = (time.perf_counter() - start_time) * 1000

                # Extract content
                content = response.choices[0].message.content or ""

                # Get token usage
                usage = response.usage
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0

                # Calculate cost
                input_cost, output_cost, total_cost = calculate_cost(
                    model, input_tokens, output_tokens
                )
                self._total_cost += total_cost
                self._query_count += 1

                if self.verbose:
                    logger.info(
                        f"[QUERY] {input_tokens}→{output_tokens} tokens, "
                        f"${total_cost:.4f}, {latency_ms:.0f}ms"
                    )

                # Record submission for bandit (actual reward comes later)
                self.bandit.update_submitted(arm_idx)

                return QueryResult(
                    content=content,
                    model_name=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=total_cost,
                    input_cost=input_cost,
                    output_cost=output_cost,
                    latency_ms=latency_ms,
                    msg=msg,
                    system_msg=system_msg,
                    msg_history=msg_history,
                    kwargs={"temperature": temperature, "max_tokens": self.max_tokens},
                    model_posteriors=model_posteriors,
                    arm_index=arm_idx,
                )

            except Exception as e:
                last_error = e
                wait_time = RETRY_DELAY_BASE * (2**attempt)
                logger.warning(
                    f"[QUERY] Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}. "
                    f"Retrying in {wait_time:.1f}s..."
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(wait_time)

        logger.error(f"[QUERY] All {MAX_RETRIES} attempts failed: {last_error}")
        return None

    async def batch_query(
        self,
        num_samples: int,
        msg: str | list[str],
        system_msg: str | list[str] = "",
        msg_history: list[dict[str, str]] | list[list[dict[str, str]]] | None = None,
    ) -> list[QueryResult]:
        """
        Execute multiple queries concurrently.

        Args:
            num_samples: Number of queries to execute.
            msg: Message(s) to send. If string, repeated num_samples times.
            system_msg: System prompt(s). If string, repeated.
            msg_history: Message history/histories.

        Returns:
            List of successful QueryResults (may be fewer than num_samples on errors).
        """
        # Normalize inputs to lists
        if isinstance(msg, str):
            msgs = [msg] * num_samples
        else:
            msgs = msg

        if isinstance(system_msg, str):
            system_msgs = [system_msg] * num_samples
        else:
            system_msgs = system_msg

        if msg_history is None:
            histories: list[list[dict[str, str]]] = [[]] * num_samples
        elif len(msg_history) > 0 and isinstance(msg_history[0], dict):
            # Single history, repeat
            histories = [msg_history] * num_samples  # type: ignore
        else:
            histories = msg_history  # type: ignore

        # Log sampling distribution
        if self.verbose:
            posterior = self.bandit.posterior()
            lines = [f"[BATCH] Sampling {num_samples} queries:"]
            for name, prob in zip(self.model_names, posterior):
                lines.append(f"  {name:<35} {prob:>6.2%}")
            logger.info("\n".join(lines))

        # Create concurrent tasks
        tasks = [
            self.query(msgs[i], system_msgs[i], histories[i])
            for i in range(num_samples)
        ]

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        successful: list[QueryResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"[BATCH] Query {i} failed: {result}")
            elif result is not None:
                successful.append(result)

        # Log summary
        if self.verbose:
            total_cost = sum(r.cost for r in successful)
            logger.info(
                f"[BATCH] Completed {len(successful)}/{num_samples} queries, "
                f"total cost: ${total_cost:.4f}"
            )

        return successful

    def update_bandit_reward(
        self,
        arm_index: int,
        reward: float,
        parent_score: float = 0.0,
        baseline_score: float = 0.0,
    ) -> None:
        """
        Update bandit with evaluation reward.

        Call this after evaluating the LLM's output to improve future selections.

        Args:
            arm_index: Which model arm to update (from QueryResult.arm_index).
            reward: Evaluation score (e.g., 0.0-1.0 for correctness).
            parent_score: Parent program's score for relative improvement.
            baseline_score: Baseline score to shift rewards.
        """
        # Use the bandit's update method
        self.bandit.update(
            arm=arm_index,
            reward=reward - parent_score if parent_score else reward,
            baseline=baseline_score,
        )

    def get_bandit_stats(self) -> str:
        """Get bandit statistics as formatted string.

        Prints the summary using rich console and returns empty string.
        For a proper return value, capture stdout or override.
        """
        import io
        from contextlib import redirect_stdout
        from rich.console import Console

        # Capture print_summary output
        f = io.StringIO()
        console = Console(file=f, force_terminal=True, width=120)
        self.bandit.print_summary()
        return f"Bandit: {type(self.bandit).__name__} with {self.bandit.n_arms} arms"


class SyncLLMClient:
    """
    Synchronous wrapper around LLMClient for non-async contexts.

    Usage:
        client = SyncLLMClient(["gpt-4o"])
        result = client.query("Hello", system_msg="You are helpful")
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize with same args as LLMClient."""
        self._async_client = LLMClient(*args, **kwargs)

    def query(
        self,
        msg: str,
        system_msg: str = "",
        msg_history: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> QueryResult | None:
        """Execute a synchronous query."""
        return asyncio.run(
            self._async_client.query(msg, system_msg, msg_history, **kwargs)
        )

    def batch_query(
        self,
        num_samples: int,
        msg: str | list[str],
        system_msg: str | list[str] = "",
        msg_history: list[dict[str, str]] | list[list[dict[str, str]]] | None = None,
    ) -> list[QueryResult]:
        """Execute multiple synchronous queries."""
        return asyncio.run(
            self._async_client.batch_query(num_samples, msg, system_msg, msg_history)
        )

    @property
    def total_cost(self) -> float:
        """Total cost across all queries."""
        return self._async_client.total_cost

    @property
    def query_count(self) -> int:
        """Total number of queries made."""
        return self._async_client.query_count
