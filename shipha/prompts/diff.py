"""
Diff-based edit prompts for ShiphaEvolve.

Provides prompts for SEARCH/REPLACE style code modifications,
which are more reliable than full code rewrites.

The diff format uses:
    <<<<<<< SEARCH
    original code to match
    =======
    replacement code
    >>>>>>> REPLACE

This format allows for targeted edits while preserving code structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from shipha.database import Program
from shipha.prompts.base import format_metrics, format_text_feedback


# =============================================================================
# Diff Format Templates
# =============================================================================

DIFF_SYSTEM_MSG = """You are an expert software engineer improving code performance.

You MUST respond using the exact SEARCH/REPLACE diff format shown below:

<NAME>
short_edit_name_lowercase_underscores
</NAME>

<DESCRIPTION>
Explain your reasoning and what this edit improves.
</DESCRIPTION>

<DIFF>
<<<<<<< SEARCH
# Original code to find and replace (must match exactly including indentation)
=======
# New replacement code
>>>>>>> REPLACE

</DIFF>

Rules:
* Only modify code between EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers.
* Do NOT include the markers themselves in SEARCH/REPLACE blocks.
* SEARCH section must be copied VERBATIM from the current file, including indentation.
* You can propose multiple edits - each SEARCH/REPLACE block is applied sequentially.
* Ensure the file still runs after your changes.
* Focus on targeted improvements, not full rewrites."""


DIFF_EVOLUTION_MSG = """# Current Program

Here is the current program we are trying to improve:

```{language}
{code}
```

## Performance Metrics

{metrics}{feedback_section}

# Instructions

Suggest improvements to maximize the `combined_score`.
Use your expert knowledge to identify optimization opportunities.

Describe each change with a SEARCH/REPLACE block in the format shown in the system message.

IMPORTANT: Focus on targeted improvements. Do not rewrite the entire program."""


# =============================================================================
# Diff Prompt Builder
# =============================================================================


def construct_diff_prompt(
    program: Program,
    language: str = "python",
) -> tuple[str, str]:
    """
    Construct a diff-style evolution prompt.

    Args:
        program: The program to improve.
        language: Programming language for code blocks.

    Returns:
        Tuple of (system_msg, user_msg).
    """
    feedback_section = ""
    if program.text_feedback:
        feedback_section = format_text_feedback(program.text_feedback)

    user_msg = DIFF_EVOLUTION_MSG.format(
        language=language,
        code=program.code,
        metrics=format_metrics(program.combined_score, program.public_metrics),
        feedback_section=feedback_section,
    )

    return DIFF_SYSTEM_MSG, user_msg


# =============================================================================
# Diff Parsing
# =============================================================================


@dataclass
class EditBlock:
    """A parsed SEARCH/REPLACE edit block."""

    name: str
    """Short name for this edit."""

    description: str
    """Description of what this edit does."""

    search: str
    """The original code to search for."""

    replace: str
    """The replacement code."""


def parse_diff_response(response: str) -> list[EditBlock]:
    """
    Parse an LLM response containing SEARCH/REPLACE blocks.

    Args:
        response: The raw LLM response text.

    Returns:
        List of EditBlock objects representing the proposed edits.
    """
    edits: list[EditBlock] = []

    # Extract name
    name_match = re.search(r"<NAME>\s*(.*?)\s*</NAME>", response, re.DOTALL)
    name = name_match.group(1).strip() if name_match else "unnamed_edit"

    # Extract description
    desc_match = re.search(r"<DESCRIPTION>\s*(.*?)\s*</DESCRIPTION>", response, re.DOTALL)
    description = desc_match.group(1).strip() if desc_match else ""

    # Extract all DIFF blocks
    diff_pattern = r"<DIFF>\s*(.*?)\s*</DIFF>"
    diff_matches = re.findall(diff_pattern, response, re.DOTALL)

    for i, diff_content in enumerate(diff_matches):
        # Parse SEARCH/REPLACE within each diff block
        sr_pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
        sr_matches = re.findall(sr_pattern, diff_content, re.DOTALL)

        for j, (search, replace) in enumerate(sr_matches):
            edit_name = f"{name}_{i}_{j}" if len(sr_matches) > 1 else name
            edits.append(
                EditBlock(
                    name=edit_name,
                    description=description,
                    search=search,
                    replace=replace,
                )
            )

    return edits


def apply_edit(code: str, edit: EditBlock) -> tuple[str, bool]:
    """
    Apply a single edit to code.

    Args:
        code: The original source code.
        edit: The EditBlock to apply.

    Returns:
        Tuple of (modified_code, success).
    """
    search_text = edit.search.strip("\n")
    replace_text = edit.replace.strip("\n")

    # Try exact match first
    if search_text in code:
        modified = code.replace(search_text, replace_text, 1)
        return modified, True

    # Try with normalized whitespace
    search_normalized = " ".join(search_text.split())
    code_lines = code.split("\n")

    for i, line in enumerate(code_lines):
        line_normalized = " ".join(line.split())
        if search_normalized.startswith(line_normalized):
            # Found potential match start
            # Try to match the full search block
            search_lines = search_text.split("\n")
            match_lines = code_lines[i : i + len(search_lines)]

            if len(match_lines) == len(search_lines):
                # Check if lines match (ignoring trailing whitespace)
                matches = all(
                    a.rstrip() == b.rstrip()
                    for a, b in zip(match_lines, search_lines)
                )
                if matches:
                    # Apply replacement
                    new_lines = (
                        code_lines[:i]
                        + replace_text.split("\n")
                        + code_lines[i + len(search_lines) :]
                    )
                    return "\n".join(new_lines), True

    return code, False


def apply_edits(code: str, edits: list[EditBlock]) -> tuple[str, list[str]]:
    """
    Apply multiple edits to code sequentially.

    Args:
        code: The original source code.
        edits: List of EditBlocks to apply.

    Returns:
        Tuple of (modified_code, list_of_applied_edit_names).
    """
    current_code = code
    applied: list[str] = []

    for edit in edits:
        new_code, success = apply_edit(current_code, edit)
        if success:
            current_code = new_code
            applied.append(edit.name)

    return current_code, applied


# =============================================================================
# Diff Utilities
# =============================================================================


def compute_diff(old_code: str, new_code: str) -> str:
    """
    Compute a unified diff between old and new code.

    Args:
        old_code: Original code.
        new_code: Modified code.

    Returns:
        Unified diff string.
    """
    import difflib

    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile="original",
        tofile="modified",
        lineterm="",
    )

    return "".join(diff)


def validate_edit_block(edit: EditBlock, code: str) -> dict[str, Any]:
    """
    Validate an edit block against source code.

    Args:
        edit: The EditBlock to validate.
        code: The source code.

    Returns:
        Dict with validation results.
    """
    search_in_code = edit.search.strip() in code

    # Check for EVOLVE-BLOCK markers
    has_markers = "EVOLVE-BLOCK-START" in code and "EVOLVE-BLOCK-END" in code

    # If markers exist, check if edit is within bounds
    edit_in_bounds = True
    if has_markers:
        start_idx = code.find("EVOLVE-BLOCK-START")
        end_idx = code.find("EVOLVE-BLOCK-END")
        search_idx = code.find(edit.search.strip())

        if search_idx != -1:
            edit_in_bounds = start_idx < search_idx < end_idx

    return {
        "name": edit.name,
        "search_found": search_in_code,
        "has_markers": has_markers,
        "edit_in_bounds": edit_in_bounds,
        "valid": search_in_code and (not has_markers or edit_in_bounds),
    }
