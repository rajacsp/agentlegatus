"""Unit tests for the centralized exception hierarchy and error recovery."""

import asyncio
from typing import Any, Dict, Optional

import pytest

from agentlegatus.exceptions import (
    AgentLegatusError,
    CapabilityNotSupportedError,
    MemoryOperationError,
    ProviderNotFoundError,
    ProviderSwitchError,
    StateBackendUnavailableError,
    WorkflowTimeoutError,
    WorkflowValidationError,
)


class TestExceptionHierarchy:
    """All custom exceptions inherit from AgentLegatusError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            ProviderNotFoundError,
            ProviderSwitchError,
            CapabilityNotSupportedError,
            StateBackendUnavailableError,
            MemoryOperationError,
            WorkflowValidationError,
            WorkflowTimeoutError,
        ],
    )
    def test_inherits_from_base(self, exc_class):
        assert issubclass(exc_class, AgentLegatusError)
        assert issubclass(exc_class, Exception)

    def test_catch_all_with_base(self):
        """Catching AgentLegatusError catches any custom exception."""
        with pytest.raises(AgentLegatusError):
            raise ProviderNotFoundError("x", [])

        with pytest.raises(AgentLegatusError):
            raise WorkflowTimeoutError("wf1", 30.0)


class TestProviderNotFoundError:
    def test_attributes(self):
        err = ProviderNotFoundError("missing", ["a", "b"])
        assert err.provider_name == "missing"
        assert err.available_providers == ["a", "b"]
        assert "missing" in str(err)
        assert "a, b" in str(err)

    def test_empty_providers(self):
        err = ProviderNotFoundError("x", [])
        assert "none" in str(err)


class TestProviderSwitchError:
    def test_attributes(self):
        orig = RuntimeError("boom")
        err = ProviderSwitchError("old", "new", "state import failed", orig)
        assert err.old_provider == "old"
        assert err.new_provider == "new"
        assert err.reason == "state import failed"
        assert err.original_error is orig
        assert "old" in str(err)
        assert "new" in str(err)

    def test_without_original_error(self):
        err = ProviderSwitchError("a", "b", "reason")
        assert err.original_error is None


class TestCapabilityNotSupportedError:
    def test_with_enum_capability(self):
        from agentlegatus.providers.base import ProviderCapability

        err = CapabilityNotSupportedError(
            "TestProvider",
            ProviderCapability.STREAMING,
            [ProviderCapability.TOOL_CALLING],
        )
        assert err.provider_name == "TestProvider"
        assert "streaming" in str(err)
        assert "tool_calling" in str(err)

    def test_with_string_capability(self):
        err = CapabilityNotSupportedError("P", "custom_cap")
        assert "custom_cap" in str(err)


class TestStateBackendUnavailableError:
    def test_attributes(self):
        orig = ConnectionError("refused")
        err = StateBackendUnavailableError("Redis", "connection refused", orig)
        assert err.backend_type == "Redis"
        assert err.reason == "connection refused"
        assert err.original_error is orig
        assert "Redis" in str(err)


class TestMemoryOperationError:
    def test_attributes(self):
        err = MemoryOperationError("store", "backend timeout")
        assert err.operation == "store"
        assert err.reason == "backend timeout"
        assert "store" in str(err)


class TestWorkflowValidationError:
    def test_attributes(self):
        errors = ["empty workflow_id", "cycle detected"]
        err = WorkflowValidationError("wf1", errors)
        assert err.workflow_id == "wf1"
        assert err.validation_errors == errors
        assert "empty workflow_id" in str(err)
        assert "cycle detected" in str(err)


class TestWorkflowTimeoutError:
    def test_workflow_level(self):
        err = WorkflowTimeoutError("wf1", 30.0)
        assert err.workflow_id == "wf1"
        assert err.timeout == 30.0
        assert err.step_id is None
        assert "wf1" in str(err)
        assert "30" in str(err)

    def test_step_level(self):
        err = WorkflowTimeoutError("wf1", 10.0, step_id="step_a")
        assert err.step_id == "step_a"
        assert "step_a" in str(err)


class TestBackwardCompatibility:
    """Existing import paths still work."""

    def test_provider_not_found_from_registry(self):
        from agentlegatus.providers.registry import ProviderNotFoundError as PNF
        assert PNF is ProviderNotFoundError

    def test_capability_not_supported_from_base(self):
        from agentlegatus.providers.base import CapabilityNotSupportedError as CNS
        assert CNS is CapabilityNotSupportedError

    def test_top_level_exports(self):
        import agentlegatus
        assert hasattr(agentlegatus, "ProviderNotFoundError")
        assert hasattr(agentlegatus, "WorkflowTimeoutError")
        assert hasattr(agentlegatus, "AgentLegatusError")
