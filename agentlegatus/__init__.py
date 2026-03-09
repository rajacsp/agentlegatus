"""
AgentLegatus - Vendor-agnostic agent framework abstraction layer.

Terraform for AI Agents: Switch between different agent frameworks with a single line of code.
"""

__version__ = "0.1.0"

from agentlegatus.core.workflow import WorkflowDefinition, WorkflowStep, WorkflowStatus

__all__ = [
    "WorkflowDefinition",
    "WorkflowStep",
    "WorkflowStatus",
]
