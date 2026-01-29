"""ShiphaEvolve prompt templates for evolution."""

from shipha.prompts.base import (
    BASE_SYSTEM_MSG,
    EXPERT_SYSTEM_MSG,
    PromptBuilder,
    construct_evolution_prompt,
    format_metrics,
    format_program,
    format_text_feedback,
    format_inspiration_programs,
)
from shipha.prompts.diff import (
    DIFF_SYSTEM_MSG,
    DIFF_EVOLUTION_MSG,
    EditBlock,
    apply_edit,
    apply_edits,
    compute_diff,
    construct_diff_prompt,
    parse_diff_response,
    validate_edit_block,
)

__all__ = [
    # Base prompts
    "BASE_SYSTEM_MSG",
    "EXPERT_SYSTEM_MSG",
    "PromptBuilder",
    "construct_evolution_prompt",
    "format_metrics",
    "format_program",
    "format_text_feedback",
    "format_inspiration_programs",
    # Diff prompts
    "DIFF_SYSTEM_MSG",
    "DIFF_EVOLUTION_MSG",
    "EditBlock",
    "apply_edit",
    "apply_edits",
    "compute_diff",
    "construct_diff_prompt",
    "parse_diff_response",
    "validate_edit_block",
]
