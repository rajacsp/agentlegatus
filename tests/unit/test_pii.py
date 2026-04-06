"""Unit tests for PII detection and redaction.

Requirements: 20.9
"""

import pytest

from agentlegatus.security.pii import PIIDetector


def _make_email(user: str = "user", domain: str = "example.com") -> str:
    """Build an email string at runtime to avoid environment transformations."""
    return f"{user}@{domain}"


class TestPIIDetector:

    def setup_method(self):
        self.detector = PIIDetector()

    # -- detect --

    def test_detect_email(self):
        email = _make_email()
        matches = self.detector.detect(f"Contact {email} for info")
        assert any(pii_type == "email" for pii_type, _ in matches)

    def test_detect_phone(self):
        matches = self.detector.detect("Call 555-123-4567 now")
        assert any(pii_type == "phone" for pii_type, _ in matches)

    def test_detect_ssn(self):
        matches = self.detector.detect("SSN: 123-45-6789")
        assert any(pii_type == "ssn" for pii_type, _ in matches)

    def test_detect_credit_card(self):
        matches = self.detector.detect("Card: 4111111111111111")
        assert any(pii_type == "credit_card" for pii_type, _ in matches)

    def test_detect_no_pii(self):
        matches = self.detector.detect("Just a normal sentence.")
        assert matches == []

    def test_detect_multiple(self):
        email = _make_email("alice", "test.org")
        text = f"Email {email}, phone 555-123-4567"
        matches = self.detector.detect(text)
        types = {t for t, _ in matches}
        assert "email" in types
        assert "phone" in types

    # -- redact --

    def test_redact_email(self):
        email = _make_email()
        result = self.detector.redact(f"Send to {email}")
        assert "[REDACTED]" in result
        assert email not in result

    def test_redact_custom_replacement(self):
        result = self.detector.redact("SSN: 123-45-6789", replacement="***")
        assert "***" in result
        assert "123-45-6789" not in result

    def test_redact_no_pii_unchanged(self):
        text = "Nothing sensitive here"
        assert self.detector.redact(text) == text

    # -- has_pii --

    def test_has_pii_true(self):
        assert self.detector.has_pii(_make_email()) is True

    def test_has_pii_false(self):
        assert self.detector.has_pii("no pii here") is False
