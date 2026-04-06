"""PII detection and redaction utilities.

Requirements: 20.9
"""

from __future__ import annotations

import re

# Common PII patterns
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")),
    ("phone", re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
]


class PIIDetector:
    """Detects and redacts common PII patterns in text."""

    def detect(self, text: str) -> list[tuple[str, str]]:
        """Return a list of (pii_type, matched_text) tuples found in *text*."""
        matches: list[tuple[str, str]] = []
        for pii_type, pattern in _PII_PATTERNS:
            for m in pattern.finditer(text):
                matches.append((pii_type, m.group()))
        return matches

    def redact(self, text: str, replacement: str = "[REDACTED]") -> str:
        """Return *text* with all detected PII replaced by *replacement*."""
        result = text
        for _, pattern in _PII_PATTERNS:
            result = pattern.sub(replacement, result)
        return result

    def has_pii(self, text: str) -> bool:
        """Return True if *text* contains any detectable PII."""
        return len(self.detect(text)) > 0
