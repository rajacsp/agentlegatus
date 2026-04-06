"""Unit tests for the security module (validation & sanitization).

Requirements: 20.4, 20.5
"""

import os
import pytest

from agentlegatus.security.sanitization import (
    SanitizationError,
    detect_injection,
    is_safe_identifier,
    sanitize_file_path,
    sanitize_string_input,
)
from agentlegatus.security.validation import (
    ValidationError,
    validate_agent_config,
    validate_workflow_definition,
)


# ---------------------------------------------------------------------------
# sanitize_file_path
# ---------------------------------------------------------------------------

class TestSanitizeFilePath:
    """Req 20.5 — prevent path traversal attacks."""

    def test_valid_path_under_allowed_dir(self, tmp_path):
        sub = tmp_path / "data"
        sub.mkdir()
        target = sub / "file.txt"
        target.touch()
        result = sanitize_file_path(str(target), [str(tmp_path)])
        assert result == str(target.resolve())

    def test_traversal_rejected(self, tmp_path):
        evil = str(tmp_path / ".." / ".." / "etc" / "passwd")
        with pytest.raises(SanitizationError, match="outside allowed"):
            sanitize_file_path(evil, [str(tmp_path)])

    def test_empty_path_rejected(self):
        with pytest.raises(SanitizationError, match="empty"):
            sanitize_file_path("", ["/tmp"])

    def test_null_byte_rejected(self):
        with pytest.raises(SanitizationError, match="null"):
            sanitize_file_path("/tmp/foo\x00bar", ["/tmp"])

    def test_no_allowed_dirs_rejected(self):
        with pytest.raises(SanitizationError, match="At least one"):
            sanitize_file_path("/tmp/file", [])

    def test_multiple_allowed_dirs(self, tmp_path):
        d1 = tmp_path / "a"
        d2 = tmp_path / "b"
        d1.mkdir()
        d2.mkdir()
        target = d2 / "ok.txt"
        target.touch()
        result = sanitize_file_path(str(target), [str(d1), str(d2)])
        assert result == str(target.resolve())

    def test_symlink_resolved(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        target = real / "data.txt"
        target.touch()
        link = tmp_path / "link"
        link.symlink_to(real)
        result = sanitize_file_path(str(link / "data.txt"), [str(real)])
        assert result == str(target.resolve())


# ---------------------------------------------------------------------------
# sanitize_string_input
# ---------------------------------------------------------------------------

class TestSanitizeStringInput:
    """Req 20.4 — sanitize inputs to prevent injection."""

    def test_clean_string_passes(self):
        assert sanitize_string_input("hello world") == "hello world"

    def test_strips_whitespace(self):
        assert sanitize_string_input("  hi  ") == "hi"

    def test_exceeds_max_length(self):
        with pytest.raises(SanitizationError, match="exceeds maximum"):
            sanitize_string_input("a" * 2000, max_length=1024)

    def test_null_byte_rejected(self):
        with pytest.raises(SanitizationError, match="null"):
            sanitize_string_input("foo\x00bar")

    def test_shell_injection_rejected(self):
        with pytest.raises(SanitizationError, match="injection"):
            sanitize_string_input("hello; rm -rf /")

    def test_backtick_injection_rejected(self):
        with pytest.raises(SanitizationError, match="injection"):
            sanitize_string_input("value `whoami`")

    def test_command_substitution_rejected(self):
        with pytest.raises(SanitizationError, match="injection"):
            sanitize_string_input("$(cat /etc/passwd)")

    def test_eval_injection_rejected(self):
        with pytest.raises(SanitizationError, match="injection"):
            sanitize_string_input("eval('malicious')")

    def test_non_string_rejected(self):
        with pytest.raises(SanitizationError, match="Expected string"):
            sanitize_string_input(123)  # type: ignore[arg-type]

    def test_allow_patterns_bypass(self):
        import re
        allow = [re.compile(r"\$\(")]
        result = sanitize_string_input("$(ok)", allow_patterns=[allow[0]])
        assert result == "$(ok)"


# ---------------------------------------------------------------------------
# is_safe_identifier
# ---------------------------------------------------------------------------

class TestIsSafeIdentifier:

    @pytest.mark.parametrize("val", ["hello", "my-agent", "step_1", "A123"])
    def test_safe_values(self, val):
        assert is_safe_identifier(val) is True

    @pytest.mark.parametrize("val", ["", "a b", "foo;bar", "x$(y)", "a" * 257])
    def test_unsafe_values(self, val):
        assert is_safe_identifier(val) is False


# ---------------------------------------------------------------------------
# detect_injection
# ---------------------------------------------------------------------------

class TestDetectInjection:

    def test_clean_string(self):
        assert detect_injection("normal text") is None

    def test_detects_shell_chain(self):
        assert detect_injection("; rm -rf") is not None

    def test_detects_python_import(self):
        assert detect_injection("__import__('os')") is not None


# ---------------------------------------------------------------------------
# validate_workflow_definition
# ---------------------------------------------------------------------------

class TestValidateWorkflowDefinition:
    """Req 20.4 — validate workflow definitions."""

    def test_valid_definition(self):
        data = {
            "workflow_id": "wf-1",
            "name": "Test Workflow",
            "description": "A test",
            "provider": "mock",
            "steps": [
                {"step_id": "s1", "step_type": "agent", "config": {"key": "val"}}
            ],
        }
        valid, errors = validate_workflow_definition(data)
        assert valid is True
        assert errors == []

    def test_injection_in_step_config(self):
        data = {
            "workflow_id": "wf-1",
            "name": "Test",
            "provider": "mock",
            "steps": [
                {
                    "step_id": "s1",
                    "step_type": "agent",
                    "config": {"cmd": "eval('bad')"},
                }
            ],
        }
        valid, errors = validate_workflow_definition(data)
        assert valid is False
        assert any("injection" in e for e in errors)

    def test_unsafe_workflow_id(self):
        data = {
            "workflow_id": "wf; drop table",
            "name": "Test",
            "provider": "mock",
            "steps": [],
        }
        valid, errors = validate_workflow_definition(data)
        assert valid is False
        assert any("unsafe" in e.lower() for e in errors)

    def test_non_dict_rejected(self):
        valid, errors = validate_workflow_definition("not a dict")  # type: ignore[arg-type]
        assert valid is False

    def test_injection_in_metadata(self):
        data = {
            "workflow_id": "wf-1",
            "name": "Test",
            "provider": "mock",
            "steps": [],
            "metadata": {"note": "__import__('os')"},
        }
        valid, errors = validate_workflow_definition(data)
        assert valid is False
        assert any("injection" in e for e in errors)

    def test_steps_not_list(self):
        data = {
            "workflow_id": "wf-1",
            "name": "Test",
            "provider": "mock",
            "steps": "not-a-list",
        }
        valid, errors = validate_workflow_definition(data)
        assert valid is False
        assert any("list" in e for e in errors)


# ---------------------------------------------------------------------------
# validate_agent_config
# ---------------------------------------------------------------------------

class TestValidateAgentConfig:
    """Req 20.4 — validate agent configuration."""

    def test_valid_config(self):
        config = {
            "agent_id": "agent-1",
            "name": "My Agent",
            "model": "gpt-4",
            "tools": ["search", "calculator"],
        }
        valid, errors = validate_agent_config(config)
        assert valid is True
        assert errors == []

    def test_unsafe_agent_id(self):
        config = {"agent_id": "agent; rm", "name": "Bad"}
        valid, errors = validate_agent_config(config)
        assert valid is False
        assert any("unsafe" in e.lower() for e in errors)

    def test_injection_in_system_prompt(self):
        config = {
            "agent_id": "a1",
            "name": "Agent",
            "system_prompt": "eval('code')",
        }
        valid, errors = validate_agent_config(config)
        assert valid is False
        assert any("injection" in e for e in errors)

    def test_unsafe_tool_name(self):
        config = {
            "agent_id": "a1",
            "name": "Agent",
            "tools": ["good-tool", "bad tool!"],
        }
        valid, errors = validate_agent_config(config)
        assert valid is False

    def test_non_dict_rejected(self):
        valid, errors = validate_agent_config([1, 2])  # type: ignore[arg-type]
        assert valid is False

    def test_tools_not_list(self):
        config = {"agent_id": "a1", "name": "Agent", "tools": "not-a-list"}
        valid, errors = validate_agent_config(config)
        assert valid is False
        assert any("list" in e for e in errors)
