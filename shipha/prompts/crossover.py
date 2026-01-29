"""
Crossover prompts for multi-parent program synthesis.

Ported from ShinkaEvolve: prompts for combining multiple code solutions
into a novel hybrid implementation.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shipha.database import Program

# =============================================================================
# System Prompt for Crossover
# =============================================================================

CROSS_SYS_FORMAT = """You are given multiple code scripts implementing the same algorithm.
You are tasked with generating a new code snippet that combines these code scripts, ideally in a way that leverages the strengths of each.
The synthesized code should be correct, clear, and efficient.

Guidelines:
- Analyze each input script to understand its approach and trade-offs.
- Identify complementary techniques from different scripts.
- Combine the best ideas into a coherent, unified solution.
- Do not simply copy one script; produce a genuine hybrid.
- Preserve correctness and handle edge cases from all inputs.
- Include brief comments explaining the rationale for key design choices.

Output only the final synthesized code, with no additional explanation."""


# =============================================================================
# Iteration Message Template
# =============================================================================

CROSS_ITER_MSG = """# Current program
Here is the current program we are trying to improve:

```{language}
{current_code}
```

# Inspiration programs
Below are alternative implementations that may contain useful ideas:

{inspirations}

Please combine the best aspects of these programs into an improved solution."""


# =============================================================================
# Helper Functions
# =============================================================================


def format_inspiration(program: Program, index: int) -> str:
    """Format a single inspiration program for display.

    Args:
        program: The inspiration program.
        index: The 1-based index of this inspiration.

    Returns:
        Formatted string showing the inspiration code.
    """
    lang = program.language or "python"
    return f"""## Inspiration {index}
Score: {program.combined_score:.4f}

```{lang}
{program.code}
```"""


def get_cross_component(
    current: Program,
    archive_inspirations: list[Program],
    top_k_inspirations: int = 3,
    language: str = "python",
) -> tuple[str, str]:
    """Build crossover prompt components.

    Samples random inspirations from the archive and formats
    the system/user messages for crossover prompting.

    Args:
        current: The current program to improve.
        archive_inspirations: Pool of candidate inspiration programs.
        top_k_inspirations: Maximum number of inspirations to include.
        language: Programming language for code blocks.

    Returns:
        Tuple of (system_message, user_message) for the LLM.
    """
    # Sample diverse inspirations (exclude the current program)
    candidates = [p for p in archive_inspirations if p.id != current.id]

    if len(candidates) > top_k_inspirations:
        # Prefer higher-scoring programs with some randomness
        candidates = sorted(candidates, key=lambda p: p.combined_score, reverse=True)
        # Take top half, then sample
        top_half = candidates[: max(len(candidates) // 2, top_k_inspirations)]
        sampled = random.sample(top_half, min(top_k_inspirations, len(top_half)))
    else:
        sampled = candidates

    # Format inspiration blocks
    inspiration_blocks = [
        format_inspiration(prog, i + 1) for i, prog in enumerate(sampled)
    ]
    inspirations_text = "\n\n".join(inspiration_blocks) if inspiration_blocks else "(No inspirations available)"

    # Build user message
    user_msg = CROSS_ITER_MSG.format(
        language=language,
        current_code=current.code,
        inspirations=inspirations_text,
    )

    return CROSS_SYS_FORMAT, user_msg


def build_crossover_prompt(
    current: Program,
    inspirations: list[Program],
    language: str = "python",
) -> tuple[str, str]:
    """Convenience wrapper for get_cross_component.

    Args:
        current: The current program to improve.
        inspirations: List of inspiration programs.
        language: Programming language for code blocks.

    Returns:
        Tuple of (system_message, user_message).
    """
    return get_cross_component(
        current=current,
        archive_inspirations=inspirations,
        top_k_inspirations=min(3, len(inspirations)),
        language=language,
    )
