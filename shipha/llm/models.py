"""
LLM Response Models

Data structures for LLM query results and pricing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryResult:
    """
    Result from an LLM query.

    Captures the response content, token usage, costs, and metadata
    for tracking and bandit feedback.
    """

    content: str
    """The generated text content from the LLM."""

    model_name: str
    """The model that generated this response."""

    input_tokens: int = 0
    """Number of input tokens used."""

    output_tokens: int = 0
    """Number of output tokens generated."""

    cost: float = 0.0
    """Total cost of this query (input + output)."""

    input_cost: float = 0.0
    """Cost of input tokens."""

    output_cost: float = 0.0
    """Cost of output tokens."""

    latency_ms: float = 0.0
    """Query latency in milliseconds."""

    # Request metadata
    msg: str = ""
    """The user message sent."""

    system_msg: str = ""
    """The system message used."""

    msg_history: list[dict[str, str]] = field(default_factory=list)
    """Message history before this query."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    """Additional kwargs passed to the LLM."""

    # Extended thinking (for reasoning models)
    thought: str = ""
    """Chain-of-thought or reasoning trace if available."""

    # Bandit selection metadata
    model_posteriors: dict[str, float] = field(default_factory=dict)
    """Posterior probabilities from bandit at selection time."""

    arm_index: int = -1
    """Which arm (model index) was selected by the bandit."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "content": self.content,
            "model_name": self.model_name,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "latency_ms": self.latency_ms,
            "msg": self.msg,
            "system_msg": self.system_msg,
            "msg_history": self.msg_history,
            "kwargs": self.kwargs,
            "thought": self.thought,
            "model_posteriors": self.model_posteriors,
            "arm_index": self.arm_index,
        }


# -----------------------------------------------------------------
# Model Pricing (per 1M tokens)
# Updated: January 2026
# -----------------------------------------------------------------

OPENAI_MODELS: dict[str, dict[str, float]] = {
    # GPT-4o family
    "gpt-4o": {"input_price": 2.50, "output_price": 10.00},
    "gpt-4o-2024-11-20": {"input_price": 2.50, "output_price": 10.00},
    "gpt-4o-2024-08-06": {"input_price": 2.50, "output_price": 10.00},
    "gpt-4o-2024-05-13": {"input_price": 5.00, "output_price": 15.00},
    # GPT-4o-mini
    "gpt-4o-mini": {"input_price": 0.15, "output_price": 0.60},
    "gpt-4o-mini-2024-07-18": {"input_price": 0.15, "output_price": 0.60},
    # o1 reasoning models
    "o1": {"input_price": 15.00, "output_price": 60.00},
    "o1-2024-12-17": {"input_price": 15.00, "output_price": 60.00},
    "o1-preview": {"input_price": 15.00, "output_price": 60.00},
    "o1-mini": {"input_price": 3.00, "output_price": 12.00},
    # o3-mini
    "o3-mini": {"input_price": 1.10, "output_price": 4.40},
    "o3-mini-2025-01-31": {"input_price": 1.10, "output_price": 4.40},
    # GPT-4.5 preview
    "gpt-4.5-preview": {"input_price": 75.00, "output_price": 150.00},
}

ANTHROPIC_MODELS: dict[str, dict[str, float]] = {
    # Claude 3.5 Sonnet
    "claude-3-5-sonnet-20241022": {"input_price": 3.00, "output_price": 15.00},
    "claude-3-5-sonnet-20240620": {"input_price": 3.00, "output_price": 15.00},
    # Claude 3.5 Haiku
    "claude-3-5-haiku-20241022": {"input_price": 0.80, "output_price": 4.00},
    # Claude 3 Opus
    "claude-3-opus-20240229": {"input_price": 15.00, "output_price": 75.00},
    # Claude 3.6 Sonnet (Jan 2026)
    "claude-3-6-sonnet-20260115": {"input_price": 3.00, "output_price": 15.00},
    # Claude 4 Opus (Jan 2026)
    "claude-opus-4-20260120": {"input_price": 15.00, "output_price": 75.00},
    "claude-sonnet-4-20250514": {"input_price": 3.00, "output_price": 15.00},
}

DEEPSEEK_MODELS: dict[str, dict[str, float]] = {
    "deepseek-chat": {"input_price": 0.14, "output_price": 0.28},
    "deepseek-reasoner": {"input_price": 0.55, "output_price": 2.19},
}

GEMINI_MODELS: dict[str, dict[str, float]] = {
    "gemini-2.0-flash": {"input_price": 0.10, "output_price": 0.40},
    "gemini-2.0-flash-lite": {"input_price": 0.075, "output_price": 0.30},
    "gemini-1.5-pro": {"input_price": 1.25, "output_price": 5.00},
    "gemini-1.5-flash": {"input_price": 0.075, "output_price": 0.30},
    "gemini-2.5-pro-preview-05-06": {"input_price": 1.25, "output_price": 10.00},
    "gemini-2.5-flash-preview-04-17": {"input_price": 0.15, "output_price": 0.60},
}

# Reasoning models that require temperature=1.0
REASONING_MODELS: list[str] = [
    "o1",
    "o1-2024-12-17",
    "o1-preview",
    "o1-mini",
    "o3-mini",
    "o3-mini-2025-01-31",
    "deepseek-reasoner",
]


def get_model_pricing(model_name: str) -> dict[str, float]:
    """
    Get pricing for a model.

    Args:
        model_name: Model identifier

    Returns:
        Dict with input_price and output_price per 1M tokens
    """
    all_models = {
        **OPENAI_MODELS,
        **ANTHROPIC_MODELS,
        **DEEPSEEK_MODELS,
        **GEMINI_MODELS,
    }

    if model_name in all_models:
        return all_models[model_name]

    # Default fallback pricing (conservative estimate)
    return {"input_price": 5.00, "output_price": 15.00}


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
) -> tuple[float, float, float]:
    """
    Calculate query cost.

    Args:
        model_name: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Tuple of (input_cost, output_cost, total_cost)
    """
    pricing = get_model_pricing(model_name)
    input_cost = (input_tokens / 1_000_000) * pricing["input_price"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_price"]
    return input_cost, output_cost, input_cost + output_cost
