"""
AgentLegatus - Vendor-agnostic agent framework abstraction layer.

Terraform for AI Agents: Switch between different agent frameworks with a single line of code.
"""

__version__ = "0.1.0"

from agentlegatus.core.workflow import WorkflowDefinition, WorkflowStatus, WorkflowStep
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

__all__ = [
    "AgentLegatusError",
    "CapabilityNotSupportedError",
    "MemoryOperationError",
    "ProviderNotFoundError",
    "ProviderSwitchError",
    "StateBackendUnavailableError",
    "WorkflowDefinition",
    "WorkflowStep",
    "WorkflowStatus",
    "WorkflowTimeoutError",
    "WorkflowValidationError",
]
