"""File path sanitization and injection prevention.

Requirements: 20.4, 20.5
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

# Patterns commonly used in injection attacks within workflow/config strings
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r";\s*\w"),  # shell command chaining
    re.compile(r"\|\s*\w"),  # pipe to command
    re.compile(r"`[^`]+`"),  # backtick execution
    re.compile(r"\$\("),  # command substitution
    re.compile(r"\$\{"),  # variable expansion
    re.compile(r"__import__\s*\("),  # Python import injection
    re.compile(r"eval\s*\("),  # eval injection
    re.compile(r"exec\s*\("),  # exec injection
    re.compile(r"os\s*\.\s*system\s*\("),  # os.system call
    re.compile(r"subprocess\s*\."),  # subprocess usage
]

# Characters that should never appear in identifiers
_UNSAFE_IDENTIFIER_RE = re.compile(r"[^a-zA-Z0-9_\-]")


class SanitizationError(Exception):
    """Raised when a value fails sanitization checks."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def sanitize_file_path(
    path: str,
    allowed_base_dirs: Sequence[str],
) -> str:
    """Sanitize a file path and ensure it resides under an allowed directory.

    Resolves the path (following symlinks), then verifies the resolved
    path starts with one of *allowed_base_dirs*.  This prevents path
    traversal attacks (Req 20.5).

    Args:
        path: The raw file path to sanitize.
        allowed_base_dirs: Directories the resolved path must fall under.

    Returns:
        The resolved, validated absolute path string.

    Raises:
        SanitizationError: If the path is empty, contains null bytes,
            or resolves outside every allowed base directory.
    """
    if not path or not path.strip():
        raise SanitizationError("File path cannot be empty")

    if "\x00" in path:
        raise SanitizationError("File path contains null bytes")

    if not allowed_base_dirs:
        raise SanitizationError("At least one allowed base directory is required")

    resolved = Path(path).resolve()

    for base in allowed_base_dirs:
        base_resolved = Path(base).resolve()
        try:
            resolved.relative_to(base_resolved)
            return str(resolved)
        except ValueError:
            continue

    raise SanitizationError(
        f"Path '{path}' resolves to '{resolved}' which is outside allowed directories"
    )


def sanitize_string_input(
    value: str,
    max_length: int = 1024,
    allow_patterns: list[re.Pattern[str]] | None = None,
) -> str:
    """Sanitize a generic string input.

    Strips leading/trailing whitespace, enforces a maximum length, and
    checks for known injection patterns (Req 20.4).

    Args:
        value: The raw string to sanitize.
        max_length: Maximum allowed length after stripping.
        allow_patterns: Optional regex patterns that, if matched, exempt
            the value from injection-pattern checks.  Useful for values
            known to contain shell-like syntax legitimately.

    Returns:
        The sanitized (stripped, truncated) string.

    Raises:
        SanitizationError: If the value contains a detected injection pattern.
    """
    if not isinstance(value, str):
        raise SanitizationError(f"Expected string, got {type(value).__name__}")

    cleaned = value.strip()

    if len(cleaned) > max_length:
        raise SanitizationError(f"String length {len(cleaned)} exceeds maximum {max_length}")

    if "\x00" in cleaned:
        raise SanitizationError("String contains null bytes")

    # Check injection patterns unless explicitly allowed
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(cleaned):
            if allow_patterns and any(ap.search(cleaned) for ap in allow_patterns):
                continue
            raise SanitizationError("Potential injection pattern detected in input")

    return cleaned


def is_safe_identifier(value: str) -> bool:
    """Check whether *value* is a safe identifier.

    A safe identifier contains only alphanumeric characters, hyphens,
    and underscores.  It must be non-empty and at most 256 characters.

    Args:
        value: The candidate identifier.

    Returns:
        True if safe, False otherwise.
    """
    if not value or len(value) > 256:
        return False
    return _UNSAFE_IDENTIFIER_RE.search(value) is None


def detect_injection(value: str) -> str | None:
    """Return the first injection pattern description found in *value*, or None."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(value):
            return pattern.pattern
    return None
