"""
Edit module for applying code transformations.

Provides utilities for applying diff-based and full-rewrite edits
to source code programs.
"""

from shipha.edit.apply_diff import apply_diff_edit, parse_diff_blocks
from shipha.edit.apply_full import apply_full_rewrite, validate_syntax

__all__ = [
    "apply_diff_edit",
    "parse_diff_blocks",
    "apply_full_rewrite",
    "validate_syntax",
]
