"""
Meta-analysis prompts for multi-step program improvement.

Ported from ShinkaEvolve: 3-step meta-prompting strategy that:
1. Analyzes individual programs to extract insights
2. Synthesizes global patterns across programs
3. Generates actionable improvement suggestions
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shipha.database import Program

# =============================================================================
# Step 1: Individual Program Analysis
# =============================================================================

META_STEP1_SYSTEM_MSG = """You are an expert programming assistant analyzing code solutions.
Your task is to examine a single program and extract key insights about its approach,
strengths, weaknesses, and notable implementation details.

Focus on:
- Algorithm design and data structure choices
- Time and space complexity
- Edge case handling
- Code clarity and maintainability
- Potential bottlenecks or inefficiencies
- Unique or clever techniques used

Be concise but thorough. Your analysis will be used to synthesize improvements."""

META_STEP1_USER_MSG = """# Program to Analyze

```{language}
{code}
```

Performance Metrics:
- Combined Score: {score:.4f}
- Correct: {correct}
{metrics}

Please provide a structured analysis of this program, highlighting its key characteristics
and potential areas for improvement."""


# =============================================================================
# Step 2: Global Pattern Synthesis
# =============================================================================

META_STEP2_SYSTEM_MSG = """You are an expert programming assistant analyzing patterns across multiple program solutions.
Given summaries of individual programs, your task is to identify:

1. Common successful patterns across high-scoring solutions
2. Recurring failure modes in lower-scoring solutions
3. Trade-offs observed (e.g., speed vs. memory, clarity vs. brevity)
4. Unexplored combinations of techniques that might yield improvements
5. Broad strategic insights for algorithm design

Synthesize a cohesive understanding that transcends any single solution."""

META_STEP2_USER_MSG = """# Individual Program Summaries

{individual_summaries}

Based on these program analyses, please identify global patterns, common strengths,
recurring issues, and strategic opportunities for improvement."""


# =============================================================================
# Step 3: Actionable Improvement Generation
# =============================================================================

META_STEP3_SYSTEM_MSG = """You are an expert programming assistant generating actionable improvement suggestions.
Given global insights about a family of programs, your task is to propose specific,
concrete changes that would likely improve the next iteration.

Your suggestions should be:
- Specific enough to implement directly
- Grounded in the observed patterns and insights
- Prioritized by expected impact
- Feasible given the problem constraints

Output 3-5 ranked improvement suggestions with brief rationale for each."""

META_STEP3_USER_MSG = """# Global Insights

{global_insights}

# Current Best Program

```{language}
{best_code}
```
Score: {best_score:.4f}

Based on the global insights and the current best program, please generate
3-5 specific, actionable improvement suggestions ranked by expected impact."""


# =============================================================================
# Helper Functions
# =============================================================================


def format_metrics(program: Program) -> str:
    """Format program metrics for display.

    Args:
        program: Program with metrics to format.

    Returns:
        Formatted string of public metrics.
    """
    if not program.public_metrics:
        return "- No additional metrics available"

    lines = []
    for key, value in program.public_metrics.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def build_step1_prompt(program: Program) -> tuple[str, str]:
    """Build Step 1 prompt for individual program analysis.

    Args:
        program: The program to analyze.

    Returns:
        Tuple of (system_message, user_message).
    """
    user_msg = META_STEP1_USER_MSG.format(
        language=program.language or "python",
        code=program.code,
        score=program.combined_score,
        correct="Yes" if program.correct else "No",
        metrics=format_metrics(program),
    )
    return META_STEP1_SYSTEM_MSG, user_msg


def build_step2_prompt(summaries: list[str]) -> tuple[str, str]:
    """Build Step 2 prompt for global pattern synthesis.

    Args:
        summaries: List of individual program analysis summaries.

    Returns:
        Tuple of (system_message, user_message).
    """
    numbered_summaries = []
    for i, summary in enumerate(summaries, 1):
        numbered_summaries.append(f"## Program {i} Analysis\n{summary}")

    combined = "\n\n".join(numbered_summaries)
    user_msg = META_STEP2_USER_MSG.format(individual_summaries=combined)
    return META_STEP2_SYSTEM_MSG, user_msg


def build_step3_prompt(
    global_insights: str,
    best_program: Program,
) -> tuple[str, str]:
    """Build Step 3 prompt for actionable improvements.

    Args:
        global_insights: Synthesized insights from Step 2.
        best_program: The current best program.

    Returns:
        Tuple of (system_message, user_message).
    """
    user_msg = META_STEP3_USER_MSG.format(
        global_insights=global_insights,
        language=best_program.language or "python",
        best_code=best_program.code,
        best_score=best_program.combined_score,
    )
    return META_STEP3_SYSTEM_MSG, user_msg


async def run_meta_analysis(
    programs: list[Program],
    llm_query_fn,
    best_program: Program | None = None,
) -> tuple[list[str], str, str]:
    """Run the full 3-step meta-analysis pipeline.

    Args:
        programs: List of programs to analyze.
        llm_query_fn: Async function to query the LLM (msg, system_msg) -> response.
        best_program: The current best program (defaults to highest scoring).

    Returns:
        Tuple of (individual_summaries, global_insights, improvement_suggestions).
    """
    import asyncio

    # Step 1: Analyze individual programs in parallel
    step1_tasks = []
    for prog in programs:
        sys_msg, user_msg = build_step1_prompt(prog)
        step1_tasks.append(llm_query_fn(msg=user_msg, system_msg=sys_msg))

    individual_summaries = await asyncio.gather(*step1_tasks)

    # Step 2: Synthesize global patterns
    sys_msg, user_msg = build_step2_prompt(individual_summaries)
    global_insights = await llm_query_fn(msg=user_msg, system_msg=sys_msg)

    # Step 3: Generate actionable improvements
    if best_program is None:
        best_program = max(programs, key=lambda p: p.combined_score)

    sys_msg, user_msg = build_step3_prompt(global_insights, best_program)
    improvements = await llm_query_fn(msg=user_msg, system_msg=sys_msg)

    return list(individual_summaries), global_insights, improvements

