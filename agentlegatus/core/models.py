"""Core data models for execution context and configuration."""

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class AgentCapability(Enum):
    """Capabilities that an agent can have."""

    TOOL_USE = "tool_use"
    MEMORY = "memory"
    PLANNING = "planning"
    REFLECTION = "reflection"


@dataclass
class ExecutionContext:
    """Runtime context for workflow execution."""

    workflow_id: str
    execution_id: str
    current_step: str
    state: dict[str, Any]
    metadata: dict[str, Any]
    start_time: datetime
    parent_context: Optional["ExecutionContext"] = None
    trace_id: str | None = None

    def create_child_context(self, step_id: str) -> "ExecutionContext":
        """
        Create a child context for nested execution.

        Args:
            step_id: ID of the step for the child context

        Returns:
            New ExecutionContext with this context as parent
        """
        return ExecutionContext(
            workflow_id=self.workflow_id,
            execution_id=f"{self.execution_id}_{step_id}",
            current_step=step_id,
            state=self.state.copy(),
            metadata=self.metadata.copy(),
            start_time=datetime.now(),
            parent_context=self,
            trace_id=self.trace_id,
        )

    def get_elapsed_time(self) -> float:
        """
        Get elapsed execution time in seconds.

        Returns:
            Elapsed time in seconds
        """
        return (datetime.now() - self.start_time).total_seconds()


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    agent_id: str
    name: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str | None = None
    capabilities: list[AgentCapability] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    memory_config: dict[str, Any] | None = None
    provider_specific: dict[str, Any] = field(default_factory=dict)

    def validate(self, available_tools: list[str]) -> tuple[bool, list[str]]:
        """
        Validate agent configuration.

        Args:
            available_tools: List of available tool names

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not self.agent_id:
            errors.append("agent_id cannot be empty")
        if not self.name:
            errors.append("name cannot be empty")
        if not self.model:
            errors.append("model cannot be empty")
        if not (0.0 <= self.temperature <= 2.0):
            errors.append(f"temperature must be between 0.0 and 2.0, got {self.temperature}")
        if self.max_tokens <= 0:
            errors.append(f"max_tokens must be positive, got {self.max_tokens}")

        # Validate tools reference available tools
        for tool in self.tools:
            if tool not in available_tools:
                errors.append(f"tool '{tool}' is not registered")

        return len(errors) == 0, errors


@dataclass
class ProviderConfig:
    """Configuration for a provider."""

    provider_name: str
    api_key: str | None = None
    api_base: str | None = None
    timeout: float = 30.0
    max_retries: int = 3
    rate_limit: int | None = None
    custom_settings: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate provider configuration.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not self.provider_name:
            errors.append("provider_name cannot be empty")
        if self.timeout <= 0:
            errors.append(f"timeout must be positive, got {self.timeout}")
        if self.max_retries < 0:
            errors.append(f"max_retries must be non-negative, got {self.max_retries}")
        if self.rate_limit is not None and self.rate_limit <= 0:
            errors.append(f"rate_limit must be positive, got {self.rate_limit}")

        return len(errors) == 0, errors

    @classmethod
    def from_env(cls, provider_name: str) -> "ProviderConfig":
        """
        Load configuration from environment variables.

        Environment variables:
        - {PROVIDER}_API_KEY: API key for the provider
        - {PROVIDER}_API_BASE: Base URL for API
        - {PROVIDER}_TIMEOUT: Request timeout
        - {PROVIDER}_MAX_RETRIES: Maximum retry attempts
        - {PROVIDER}_RATE_LIMIT: Rate limit per minute

        Args:
            provider_name: Name of the provider

        Returns:
            ProviderConfig loaded from environment
        """
        prefix = provider_name.upper().replace("-", "_")

        return cls(
            provider_name=provider_name,
            api_key=os.getenv(f"{prefix}_API_KEY"),
            api_base=os.getenv(f"{prefix}_API_BASE"),
            timeout=float(os.getenv(f"{prefix}_TIMEOUT", "30.0")),
            max_retries=int(os.getenv(f"{prefix}_MAX_RETRIES", "3")),
            rate_limit=(
                int(os.getenv(f"{prefix}_RATE_LIMIT"))
                if os.getenv(f"{prefix}_RATE_LIMIT")
                else None
            ),
        )
