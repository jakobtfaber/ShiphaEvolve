"""
Full code rewrite application.

Handles cases where the LLM generates complete new code rather than
incremental diffs. Includes syntax validation and safety checks.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import NamedTuple


# =============================================================================
# Data Structures
# =============================================================================


class CodeBlock(NamedTuple):
    """Extracted code block from LLM response."""

    code: str
    language: str
    start_line: int
    end_line: int


@dataclass
class RewriteOutcome:
    """Result of applying a full rewrite.

    Attributes:
        success: Whether the rewrite was successful.
        new_code: The new code (or original if failed).
        syntax_valid: Whether the new code has valid syntax.
        error_message: Description of any errors.
        language: Detected programming language.
    """

    success: bool
    new_code: str
    syntax_valid: bool = True
    error_message: str = ""
    language: str = "python"


# =============================================================================
# Code Extraction
# =============================================================================

# Pattern to match fenced code blocks
CODE_BLOCK_PATTERN = re.compile(
    r"```(\w*)\n(.*?)\n```",
    re.DOTALL,
)

# Pattern for indented code blocks (4 spaces or tab)
INDENTED_PATTERN = re.compile(
    r"(?:^|\n)((?:[ ]{4}|\t)[^\n]+(?:\n(?:[ ]{4}|\t)[^\n]+)*)",
    re.MULTILINE,
)


def extract_code_blocks(response: str) -> list[CodeBlock]:
    """Extract all code blocks from LLM response.

    Args:
        response: Raw LLM response text.

    Returns:
        List of CodeBlock objects.
    """
    blocks: list[CodeBlock] = []
    lines = response.split("\n")

    for match in CODE_BLOCK_PATTERN.finditer(response):
        language = match.group(1) or "python"
        code = match.group(2)

        # Calculate line numbers
        start_char = match.start()
        start_line = response[:start_char].count("\n") + 1
        end_line = start_line + code.count("\n")

        blocks.append(
            CodeBlock(
                code=code,
                language=language.lower(),
                start_line=start_line,
                end_line=end_line,
            )
        )

    return blocks


def extract_primary_code(
    response: str,
    expected_language: str = "python",
) -> str | None:
    """Extract the primary code block from response.

    Prefers:
    1. Code blocks matching expected language
    2. Longest code block
    3. First code block

    Args:
        response: Raw LLM response text.
        expected_language: Expected programming language.

    Returns:
        Extracted code string, or None if no code found.
    """
    blocks = extract_code_blocks(response)

    if not blocks:
        # Try to find indented code
        indented = INDENTED_PATTERN.findall(response)
        if indented:
            # Return the longest indented block
            return max(indented, key=len)
        return None

    # Prefer matching language
    matching = [b for b in blocks if b.language == expected_language]
    if matching:
        # Return longest matching
        return max(matching, key=lambda b: len(b.code)).code

    # Return longest block
    return max(blocks, key=lambda b: len(b.code)).code


# =============================================================================
# Syntax Validation
# =============================================================================


def validate_syntax(
    code: str,
    language: str = "python",
) -> bool:
    """Check if code has valid syntax.

    Args:
        code: Source code to validate.
        language: Programming language.

    Returns:
        True if syntax is valid, False otherwise.
    """
    if language == "python":
        return validate_python_syntax(code)
    # Add other language validators as needed
    return True  # Unknown languages pass by default


def validate_python_syntax(code: str) -> bool:
    """Validate Python syntax.

    Args:
        code: Python source code.

    Returns:
        True if syntax is valid.
    """
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def get_syntax_error(code: str, language: str = "python") -> str | None:
    """Get syntax error message if code is invalid.

    Args:
        code: Source code to validate.
        language: Programming language.

    Returns:
        Error message string, or None if valid.
    """
    if language != "python":
        return None

    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"Line {e.lineno}: {e.msg}"


# =============================================================================
# Full Rewrite Application
# =============================================================================


def apply_full_rewrite(
    original: str,
    response: str,
    language: str = "python",
    validate: bool = True,
    preserve_signature: bool = False,
) -> RewriteOutcome:
    """Extract and apply a full code rewrite from LLM response.

    Args:
        original: Original source code.
        response: LLM response containing new code.
        language: Expected programming language.
        validate: Validate syntax before accepting.
        preserve_signature: Ensure function signatures match.

    Returns:
        RewriteOutcome with result status.
    """
    # Extract the new code
    new_code = extract_primary_code(response, language)

    if new_code is None:
        return RewriteOutcome(
            success=False,
            new_code=original,
            error_message="No code block found in response",
        )

    # Clean up the code
    new_code = clean_code(new_code)

    # Validate syntax if requested
    if validate:
        error = get_syntax_error(new_code, language)
        if error:
            return RewriteOutcome(
                success=False,
                new_code=original,
                syntax_valid=False,
                error_message=f"Syntax error: {error}",
                language=language,
            )

    # Check signature preservation if requested
    if preserve_signature and language == "python":
        if not signatures_match(original, new_code):
            return RewriteOutcome(
                success=False,
                new_code=original,
                syntax_valid=True,
                error_message="Function signatures do not match original",
                language=language,
            )

    return RewriteOutcome(
        success=True,
        new_code=new_code,
        syntax_valid=True,
        language=language,
    )


def clean_code(code: str) -> str:
    """Clean up extracted code.

    Removes common artifacts from LLM responses.

    Args:
        code: Raw extracted code.

    Returns:
        Cleaned code string.
    """
    # Remove leading/trailing whitespace
    code = code.strip()

    # Remove common prefixes
    prefixes_to_remove = [
        "Here is the improved code:",
        "Here's the updated code:",
        "Here is the code:",
        "```python",
        "```",
    ]
    for prefix in prefixes_to_remove:
        if code.startswith(prefix):
            code = code[len(prefix) :].lstrip()

    # Remove common suffixes
    suffixes_to_remove = ["```"]
    for suffix in suffixes_to_remove:
        if code.endswith(suffix):
            code = code[: -len(suffix)].rstrip()

    # Ensure trailing newline
    if not code.endswith("\n"):
        code += "\n"

    return code


# =============================================================================
# Signature Matching
# =============================================================================


def extract_function_signatures(code: str) -> dict[str, str]:
    """Extract function signatures from Python code.

    Args:
        code: Python source code.

    Returns:
        Dict mapping function names to their signature strings.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}

    signatures: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Build signature string
            args = []
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                args.append(arg_str)

            sig = f"{node.name}({', '.join(args)})"
            if node.returns:
                sig += f" -> {ast.unparse(node.returns)}"

            signatures[node.name] = sig

    return signatures


def signatures_match(original: str, modified: str) -> bool:
    """Check if function signatures match between original and modified code.

    Args:
        original: Original source code.
        modified: Modified source code.

    Returns:
        True if all original signatures are preserved.
    """
    orig_sigs = extract_function_signatures(original)
    mod_sigs = extract_function_signatures(modified)

    # Check that all original functions exist with same signatures
    for name, sig in orig_sigs.items():
        if name not in mod_sigs:
            return False
        if mod_sigs[name] != sig:
            return False

    return True


# =============================================================================
# Hybrid Edit Strategy
# =============================================================================


def apply_edit(
    original: str,
    response: str,
    language: str = "python",
    prefer_diff: bool = True,
) -> RewriteOutcome:
    """Apply edits using the most appropriate strategy.

    Tries diff-based editing first, falls back to full rewrite.

    Args:
        original: Original source code.
        response: LLM response.
        language: Programming language.
        prefer_diff: Try diff-based editing first.

    Returns:
        RewriteOutcome with result status.
    """
    from shipha.edit.apply_diff import apply_diff_edit, parse_diff_blocks, EditResult

    if prefer_diff:
        # Try diff-based editing
        blocks = parse_diff_blocks(response)
        if blocks:
            outcome = apply_diff_edit(original, blocks, validate=True)
            if outcome.result == EditResult.SUCCESS:
                return RewriteOutcome(
                    success=True,
                    new_code=outcome.new_code,
                    syntax_valid=True,
                    language=language,
                )

    # Fall back to full rewrite
    return apply_full_rewrite(original, response, language)
