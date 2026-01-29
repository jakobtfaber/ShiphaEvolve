"""
Base prompt templates for ShiphaEvolve.

Provides common system messages and formatting utilities for LLM prompts.
"""

from __future__ import annotations

from typing import Any

from shipha.database import Program


# =============================================================================
# System Messages
# =============================================================================

BASE_SYSTEM_MSG = """You are an expert software engineer tasked with improving the performance of a given program. Your job is to analyze the current program and suggest improvements based on the collected feedback from previous attempts."""

EXPERT_SYSTEM_MSG = """You are a world-class algorithm designer and optimization expert. You have deep knowledge of:
- Algorithm design and complexity analysis
- Performance optimization techniques
- Numerical methods and scientific computing
- Machine learning and data science patterns

Your task is to improve the given code to maximize its performance metrics."""


# =============================================================================
# Formatting Utilities
# =============================================================================


def format_metrics(combined_score: float, public_metrics: dict[str, Any]) -> str:
    """
    Format performance metrics for display in prompts.

    Args:
        combined_score: Overall fitness score.
        public_metrics: Dict of metric name to value.

    Returns:
        Formatted metrics string.
    """
    lines = [f"Combined score to maximize: {combined_score:.4f}"]

    for key, value in public_metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)


def format_text_feedback(text_feedback: str | list[str] | None) -> str:
    """
    Format text feedback for inclusion in prompts.

    Args:
        text_feedback: Raw feedback string or list of strings.

    Returns:
        Formatted feedback section (empty string if no feedback).
    """
    if not text_feedback:
        return ""

    if isinstance(text_feedback, list):
        text_feedback = "\n".join(str(item) for item in text_feedback)

    feedback_text = text_feedback.strip()
    if not feedback_text:
        return ""

    return f"""
## Text Feedback

{feedback_text}
"""


def format_program(
    program: Program,
    language: str = "python",
    include_feedback: bool = True,
) -> str:
    """
    Format a program for display in prompts.

    Args:
        program: The Program to format.
        language: Programming language for code block.
        include_feedback: Whether to include text feedback.

    Returns:
        Formatted program string.
    """
    sections = []

    # Code block
    sections.append(f"```{language}\n{program.code}\n```")

    # Metrics
    sections.append("\n### Performance Metrics")
    sections.append(format_metrics(program.combined_score, program.public_metrics))

    # Correctness
    if program.correct:
        sections.append("\n✓ The program is correct and passes all tests.")
    else:
        sections.append("\n✗ The program is incorrect or has failing tests.")

    # Feedback
    if include_feedback and program.text_feedback:
        sections.append(format_text_feedback(program.text_feedback))

    return "\n".join(sections)


def format_inspiration_programs(
    programs: list[Program],
    language: str = "python",
    include_feedback: bool = True,
) -> str:
    """
    Format a list of inspiration programs for prompts.

    Args:
        programs: List of programs to use as inspiration.
        language: Programming language for code blocks.
        include_feedback: Whether to include text feedback.

    Returns:
        Formatted inspiration section.
    """
    if not programs:
        return ""

    sections = ["## Prior Programs (for inspiration)\n"]

    for i, prog in enumerate(programs, 1):
        sections.append(f"### Program {i}")
        sections.append(format_program(prog, language, include_feedback))
        sections.append("")  # Blank line between programs

    return "\n".join(sections)


def construct_evolution_prompt(
    parent: Program,
    inspirations: list[Program] | None = None,
    language: str = "python",
) -> str:
    """
    Construct the full evolution prompt.

    Args:
        parent: The parent program to evolve from.
        inspirations: Optional list of inspiration programs.
        language: Programming language for code blocks.

    Returns:
        Complete prompt for the LLM.
    """
    sections = []

    # Inspiration programs
    if inspirations:
        sections.append(format_inspiration_programs(inspirations, language))

    # Current program
    sections.append("## Current Program (to improve)")
    sections.append(format_program(parent, language))

    # Instructions
    sections.append("""
## Task

Analyze the current program and suggest improvements to maximize the `combined_score`.
Consider the performance patterns from prior programs when available.
""")

    return "\n".join(sections)


# =============================================================================
# Prompt Builders
# =============================================================================


class PromptBuilder:
    """
    Builder for constructing LLM prompts.

    Provides fluent interface for building prompts with optional sections.

    Usage:
        prompt = (
            PromptBuilder()
            .with_system_msg(EXPERT_SYSTEM_MSG)
            .with_parent(parent_program)
            .with_inspirations(inspiration_list)
            .build()
        )
    """

    def __init__(self, language: str = "python") -> None:
        self.language = language
        self.system_msg = BASE_SYSTEM_MSG
        self.parent: Program | None = None
        self.inspirations: list[Program] = []
        self.custom_sections: list[str] = []
        self.task_instructions: str = ""

    def with_system_msg(self, msg: str) -> PromptBuilder:
        """Set the system message."""
        self.system_msg = msg
        return self

    def with_parent(self, program: Program) -> PromptBuilder:
        """Set the parent program to evolve."""
        self.parent = program
        return self

    def with_inspirations(self, programs: list[Program]) -> PromptBuilder:
        """Set inspiration programs."""
        self.inspirations = programs
        return self

    def with_section(self, section: str) -> PromptBuilder:
        """Add a custom section."""
        self.custom_sections.append(section)
        return self

    def with_task(self, instructions: str) -> PromptBuilder:
        """Set task instructions."""
        self.task_instructions = instructions
        return self

    def build(self) -> tuple[str, str]:
        """
        Build the prompt.

        Returns:
            Tuple of (system_msg, user_msg).
        """
        sections = []

        # Custom sections first
        for section in self.custom_sections:
            sections.append(section)

        # Inspiration programs
        if self.inspirations:
            sections.append(
                format_inspiration_programs(self.inspirations, self.language)
            )

        # Parent program
        if self.parent:
            sections.append("## Current Program (to improve)")
            sections.append(format_program(self.parent, self.language))

        # Task instructions
        if self.task_instructions:
            sections.append(f"\n## Task\n\n{self.task_instructions}")

        user_msg = "\n".join(sections)
        return self.system_msg, user_msg
