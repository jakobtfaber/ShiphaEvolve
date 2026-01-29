"""
Diff-based code editing with SEARCH/REPLACE blocks.

Applies structured edits using the diff format from prompt responses.
Handles partial matches, fuzzy matching, and conflict resolution.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple


# =============================================================================
# Data Structures
# =============================================================================


class EditResult(str, Enum):
    """Result of applying an edit."""

    SUCCESS = "success"
    PARTIAL = "partial"  # Some blocks applied
    NOT_FOUND = "not_found"  # Search block not found
    CONFLICT = "conflict"  # Multiple matches
    SYNTAX_ERROR = "syntax_error"  # Result has syntax errors
    UNCHANGED = "unchanged"  # No changes made


@dataclass
class DiffBlock:
    """A single SEARCH/REPLACE edit block.

    Attributes:
        search: The exact text to find.
        replace: The replacement text.
        line_hint: Optional line number hint.
        confidence: Match confidence (0-1).
    """

    search: str
    replace: str
    line_hint: int | None = None
    confidence: float = 1.0

    @property
    def is_deletion(self) -> bool:
        """Check if this block deletes code."""
        return not self.replace.strip()

    @property
    def is_insertion(self) -> bool:
        """Check if this block inserts new code."""
        return not self.search.strip()


@dataclass
class EditOutcome:
    """Result of applying edits to code.

    Attributes:
        result: The outcome status.
        new_code: The modified code (or original if failed).
        applied_blocks: Number of successfully applied blocks.
        failed_blocks: Blocks that failed to apply.
        error_message: Description of any errors.
    """

    result: EditResult
    new_code: str
    applied_blocks: int = 0
    failed_blocks: list[DiffBlock] = field(default_factory=list)
    error_message: str = ""


class MatchLocation(NamedTuple):
    """Location of a match in source code."""

    start: int  # Start index
    end: int  # End index
    line: int  # Line number (1-indexed)
    confidence: float  # Match confidence


# =============================================================================
# Diff Block Parsing
# =============================================================================

# Regex patterns for SEARCH/REPLACE blocks
SEARCH_PATTERN = re.compile(
    r"<<<<<<+\s*SEARCH\s*\n(.*?)\n======+\s*\n(.*?)\n>>>>>>+\s*REPLACE",
    re.DOTALL,
)

ALT_SEARCH_PATTERN = re.compile(
    r"```\s*search\s*\n(.*?)\n```\s*\n```\s*replace\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)

SIMPLE_PATTERN = re.compile(
    r"SEARCH:\s*\n```[^\n]*\n(.*?)\n```\s*\nREPLACE:\s*\n```[^\n]*\n(.*?)\n```",
    re.DOTALL,
)


def parse_diff_blocks(response: str) -> list[DiffBlock]:
    """Parse SEARCH/REPLACE blocks from LLM response.

    Supports multiple formats:
    - <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE
    - ```search ... ``` ```replace ... ```
    - SEARCH: ```...``` REPLACE: ```...```

    Args:
        response: Raw LLM response text.

    Returns:
        List of parsed DiffBlock objects.
    """
    blocks: list[DiffBlock] = []

    # Try primary format
    for match in SEARCH_PATTERN.finditer(response):
        blocks.append(
            DiffBlock(
                search=match.group(1).strip(),
                replace=match.group(2).strip(),
            )
        )

    # Try alternative formats if no primary matches
    if not blocks:
        for match in ALT_SEARCH_PATTERN.finditer(response):
            blocks.append(
                DiffBlock(
                    search=match.group(1).strip(),
                    replace=match.group(2).strip(),
                )
            )

    if not blocks:
        for match in SIMPLE_PATTERN.finditer(response):
            blocks.append(
                DiffBlock(
                    search=match.group(1).strip(),
                    replace=match.group(2).strip(),
                )
            )

    return blocks


# =============================================================================
# Match Finding
# =============================================================================


def find_exact_match(code: str, search: str) -> MatchLocation | None:
    """Find exact match of search text in code.

    Args:
        code: Source code to search.
        search: Text to find.

    Returns:
        MatchLocation if found, None otherwise.
    """
    idx = code.find(search)
    if idx == -1:
        return None

    # Calculate line number
    line = code[:idx].count("\n") + 1

    return MatchLocation(
        start=idx,
        end=idx + len(search),
        line=line,
        confidence=1.0,
    )


def find_fuzzy_match(
    code: str,
    search: str,
    threshold: float = 0.8,
) -> MatchLocation | None:
    """Find fuzzy match of search text in code.

    Uses difflib SequenceMatcher to find approximate matches.
    Useful when LLM slightly modifies the search text.

    Args:
        code: Source code to search.
        search: Text to find.
        threshold: Minimum similarity ratio (0-1).

    Returns:
        MatchLocation if found with sufficient confidence, None otherwise.
    """
    search_lines = search.strip().split("\n")
    code_lines = code.split("\n")

    if not search_lines:
        return None

    # Sliding window search
    window_size = len(search_lines)
    best_match = None
    best_ratio = threshold

    for i in range(len(code_lines) - window_size + 1):
        window = "\n".join(code_lines[i : i + window_size])
        matcher = difflib.SequenceMatcher(None, search, window)
        ratio = matcher.ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            # Calculate character positions
            start = sum(len(line) + 1 for line in code_lines[:i])
            end = start + len(window)
            best_match = MatchLocation(
                start=start,
                end=end,
                line=i + 1,
                confidence=ratio,
            )

    return best_match


def find_match(
    code: str,
    search: str,
    fuzzy: bool = True,
    threshold: float = 0.8,
) -> MatchLocation | None:
    """Find match of search text in code.

    Tries exact match first, then falls back to fuzzy matching.

    Args:
        code: Source code to search.
        search: Text to find.
        fuzzy: Enable fuzzy matching fallback.
        threshold: Fuzzy match threshold.

    Returns:
        MatchLocation if found, None otherwise.
    """
    # Try exact match first
    exact = find_exact_match(code, search)
    if exact:
        return exact

    # Try with normalized whitespace
    normalized_search = normalize_whitespace(search)
    normalized_code = normalize_whitespace(code)
    norm_exact = find_exact_match(normalized_code, normalized_search)
    if norm_exact:
        # Map back to original positions (approximate)
        return MatchLocation(
            start=norm_exact.start,
            end=norm_exact.end,
            line=norm_exact.line,
            confidence=0.95,
        )

    # Fall back to fuzzy matching
    if fuzzy:
        return find_fuzzy_match(code, search, threshold)

    return None


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text for comparison.

    Args:
        text: Input text.

    Returns:
        Text with normalized whitespace.
    """
    lines = text.split("\n")
    # Strip trailing whitespace, normalize indentation to spaces
    normalized = []
    for line in lines:
        line = line.rstrip()
        line = line.replace("\t", "    ")
        normalized.append(line)
    return "\n".join(normalized)


# =============================================================================
# Edit Application
# =============================================================================


def apply_single_diff(
    code: str,
    block: DiffBlock,
    fuzzy: bool = True,
) -> tuple[str, EditResult]:
    """Apply a single diff block to code.

    Args:
        code: Source code to modify.
        block: The diff block to apply.
        fuzzy: Enable fuzzy matching.

    Returns:
        Tuple of (modified_code, result_status).
    """
    # Handle insertions (empty search)
    if block.is_insertion:
        # Append to end if no context
        return code + "\n" + block.replace, EditResult.SUCCESS

    # Find the search text
    match = find_match(code, block.search, fuzzy=fuzzy)

    if match is None:
        return code, EditResult.NOT_FOUND

    # Check for multiple matches (conflict)
    second_match = find_exact_match(code[match.end :], block.search)
    if second_match:
        # Multiple matches - try to use line hint
        if block.line_hint:
            # Use the match closest to the hint
            pass  # TODO: Implement line hint disambiguation
        else:
            return code, EditResult.CONFLICT

    # Apply the replacement
    new_code = code[: match.start] + block.replace + code[match.end :]

    return new_code, EditResult.SUCCESS


def apply_diff_edit(
    code: str,
    blocks: list[DiffBlock] | str,
    fuzzy: bool = True,
    validate: bool = True,
) -> EditOutcome:
    """Apply multiple diff blocks to code.

    Args:
        code: Source code to modify.
        blocks: List of DiffBlock objects or raw response to parse.
        fuzzy: Enable fuzzy matching.
        validate: Validate syntax after edits.

    Returns:
        EditOutcome with result status and modified code.
    """
    # Parse if string
    if isinstance(blocks, str):
        blocks = parse_diff_blocks(blocks)

    if not blocks:
        return EditOutcome(
            result=EditResult.UNCHANGED,
            new_code=code,
            error_message="No edit blocks found in response",
        )

    current_code = code
    applied = 0
    failed: list[DiffBlock] = []

    for block in blocks:
        new_code, result = apply_single_diff(current_code, block, fuzzy=fuzzy)

        if result == EditResult.SUCCESS:
            current_code = new_code
            applied += 1
        else:
            failed.append(block)

    # Determine overall result
    if applied == 0:
        result = EditResult.NOT_FOUND
    elif failed:
        result = EditResult.PARTIAL
    else:
        result = EditResult.SUCCESS

    # Validate syntax if requested
    if validate and result in (EditResult.SUCCESS, EditResult.PARTIAL):
        from shipha.edit.apply_full import validate_syntax

        if not validate_syntax(current_code):
            return EditOutcome(
                result=EditResult.SYNTAX_ERROR,
                new_code=current_code,
                applied_blocks=applied,
                failed_blocks=failed,
                error_message="Result has syntax errors",
            )

    return EditOutcome(
        result=result,
        new_code=current_code,
        applied_blocks=applied,
        failed_blocks=failed,
    )


# =============================================================================
# Utility Functions
# =============================================================================


def create_diff_preview(
    original: str,
    modified: str,
    context_lines: int = 3,
) -> str:
    """Create a unified diff preview.

    Args:
        original: Original code.
        modified: Modified code.
        context_lines: Context lines around changes.

    Returns:
        Unified diff string.
    """
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile="original",
        tofile="modified",
        n=context_lines,
    )

    return "".join(diff)


def count_changes(original: str, modified: str) -> dict[str, int]:
    """Count lines added, removed, and modified.

    Args:
        original: Original code.
        modified: Modified code.

    Returns:
        Dict with 'added', 'removed', 'modified' counts.
    """
    original_lines = set(original.splitlines())
    modified_lines = set(modified.splitlines())

    added = len(modified_lines - original_lines)
    removed = len(original_lines - modified_lines)

    return {
        "added": added,
        "removed": removed,
        "total_delta": added - removed,
    }
