"""Centralized exception hierarchy for AgentLegatus.

All custom exceptions inherit from ``AgentLegatusError`` so callers can
catch the entire family with a single ``except`` clause when needed.

Existing exceptions that were defined in individual modules
(``ProviderNotFoundError``, ``CapabilityNotSupportedError``, etc.) are
re-exported from here for convenience, but the canonical definitions
remain in their original modules to avoid breaking existing imports.

Requirements: 15.1-15.10
"""

from typing import Any

# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------


class AgentLegatusError(Exception):
    """Base exception for all AgentLegatus errors."""

    pass


# ---------------------------------------------------------------------------
# Provider errors
# ---------------------------------------------------------------------------


class ProviderNotFoundError(AgentLegatusError):
    """Raised when a requested provider is not registered.

    Requirement 15.1: Include a list of available providers.
    """

    def __init__(self, provider_name: str, available_providers: list[str]):
        self.provider_name = provider_name
        self.available_providers = available_providers
        message = (
            f"Provider '{provider_name}' not found. "
            f"Available providers: {', '.join(available_providers) if available_providers else 'none'}"
        )
        super().__init__(message)


class ProviderSwitchError(AgentLegatusError):
    """Raised when switching between providers fails.

    Requirement 15.8: Rollback to previous provider and preserve state.
    """

    def __init__(
        self,
        old_provider: str,
        new_provider: str,
        reason: str,
        original_error: Exception | None = None,
    ):
        self.old_provider = old_provider
        self.new_provider = new_provider
        self.reason = reason
        self.original_error = original_error
        message = (
            f"Failed to switch provider from '{old_provider}' to " f"'{new_provider}': {reason}"
        )
        super().__init__(message)


class CapabilityNotSupportedError(AgentLegatusError):
    """Raised when a provider does not support a required capability.

    Requirement 15.7 / 24.4.
    """

    def __init__(
        self,
        provider_name: str,
        capability: Any,
        supported_capabilities: list[Any] | None = None,
    ):
        self.provider_name = provider_name
        self.capability = capability
        self.supported_capabilities = supported_capabilities or []
        supported_names = [
            c.value if hasattr(c, "value") else str(c) for c in self.supported_capabilities
        ]
        message = (
            f"Provider '{provider_name}' does not support capability "
            f"'{capability.value if hasattr(capability, 'value') else capability}'. "
            f"Supported: {', '.join(supported_names) if supported_names else 'none'}"
        )
        super().__init__(message)


# ---------------------------------------------------------------------------
# State / backend errors
# ---------------------------------------------------------------------------


class StateBackendUnavailableError(AgentLegatusError):
    """Raised when a state backend is unreachable.

    Requirement 15.4: Attempt reconnection; 15.5: preserve in-memory state.
    """

    def __init__(self, backend_type: str, reason: str, original_error: Exception | None = None):
        self.backend_type = backend_type
        self.reason = reason
        self.original_error = original_error
        message = f"State backend '{backend_type}' unavailable: {reason}"
        super().__init__(message)


class MemoryOperationError(AgentLegatusError):
    """Raised when a memory backend operation fails.

    Requirement 15.7 (tool/memory invocation failures).
    """

    def __init__(self, operation: str, reason: str, original_error: Exception | None = None):
        self.operation = operation
        self.reason = reason
        self.original_error = original_error
        message = f"Memory operation '{operation}' failed: {reason}"
        super().__init__(message)


# ---------------------------------------------------------------------------
# Workflow errors
# ---------------------------------------------------------------------------


class WorkflowValidationError(AgentLegatusError):
    """Raised when a workflow definition fails validation.

    Requirement 15.6: Include specific validation failures.
    """

    def __init__(self, workflow_id: str, validation_errors: list[str]):
        self.workflow_id = workflow_id
        self.validation_errors = validation_errors
        errors_str = "; ".join(validation_errors)
        message = f"Workflow '{workflow_id}' validation failed: {errors_str}"
        super().__init__(message)


class WorkflowTimeoutError(AgentLegatusError):
    """Raised when a workflow or step exceeds its configured timeout.

    Requirement 15.9 / 28.3.
    """

    def __init__(
        self,
        workflow_id: str,
        timeout: float,
        step_id: str | None = None,
    ):
        self.workflow_id = workflow_id
        self.timeout = timeout
        self.step_id = step_id
        if step_id:
            message = (
                f"Step '{step_id}' in workflow '{workflow_id}' " f"exceeded timeout of {timeout}s"
            )
        else:
            message = f"Workflow '{workflow_id}' exceeded timeout of {timeout}s"
        super().__init__(message)
