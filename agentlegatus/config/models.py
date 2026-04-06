"""Pydantic configuration models with validation and sensible defaults.

Requirements: 19.1-19.9, 20.1, 20.2
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# Sensitive field names that must never be logged or exposed
SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"api[_-]?key", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"credential", re.IGNORECASE),
]


class ConfigError(Exception):
    """Raised when configuration validation fails.

    Attributes:
        errors: List of specific validation failure messages.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Configuration errors: {'; '.join(errors)}")


def _is_sensitive_key(key: str) -> bool:
    """Return True if *key* matches a known sensitive pattern."""
    return any(p.search(key) for p in SENSITIVE_PATTERNS)


def redact_sensitive(data: dict[str, Any], replacement: str = "***") -> dict[str, Any]:
    """Return a shallow copy of *data* with sensitive values replaced."""
    out: dict[str, Any] = {}
    for k, v in data.items():
        if _is_sensitive_key(k):
            out[k] = replacement
        elif isinstance(v, dict):
            out[k] = redact_sensitive(v, replacement)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ProviderEntry(BaseModel):
    """Configuration for a single provider."""

    name: str
    api_key: str | None = Field(default=None, exclude=True)
    api_base: str | None = None
    timeout: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    rate_limit: int | None = Field(default=None, gt=0)
    custom_settings: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("provider name cannot be empty")
        return v


class StateConfig(BaseModel):
    """State backend configuration."""

    backend: str = "memory"
    connection_url: str | None = None
    ttl: int | None = Field(default=None, gt=0)
    pool_size: int = Field(default=5, gt=0)


class MemoryConfig(BaseModel):
    """Memory backend configuration."""

    backend: str = "memory"
    connection_url: str | None = None
    ttl: int = Field(default=3600, gt=0)
    embedding_model: str | None = None


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    json_format: bool = True

    @field_validator("level")
    @classmethod
    def _valid_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"level must be one of {allowed}")
        return v.upper()


class ObservabilityConfig(BaseModel):
    """Observability / metrics configuration."""

    enable_tracing: bool = False
    enable_prometheus: bool = False
    otlp_endpoint: str | None = None


class SecretsConfig(BaseModel):
    """Secrets management integration.

    Supports a pluggable *backend* (e.g. ``env``, ``aws_ssm``, ``vault``).
    ``prefix`` is prepended when resolving secret names.
    """

    backend: str = "env"
    prefix: str = ""
    config: dict[str, Any] = Field(default_factory=dict)


class WorkflowDefConfig(BaseModel):
    """Inline workflow defaults that can be overridden per-workflow."""

    timeout: float | None = Field(default=None, gt=0)
    execution_strategy: str = "sequential"
    max_parallel: int = Field(default=5, gt=0)

    @field_validator("execution_strategy")
    @classmethod
    def _valid_strategy(cls, v: str) -> str:
        allowed = {"sequential", "parallel", "conditional"}
        if v.lower() not in allowed:
            raise ValueError(f"execution_strategy must be one of {allowed}")
        return v.lower()


# ---------------------------------------------------------------------------
# Root configuration model
# ---------------------------------------------------------------------------


class AgentLegConfig(BaseModel):
    """Root configuration for AgentLegatus.

    Validates the full configuration tree and provides sensible defaults
    for every optional value (Req 19.5, 19.9).
    """

    default_provider: str = "mock"
    providers: list[ProviderEntry] = Field(default_factory=list)
    state: StateConfig = Field(default_factory=StateConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    secrets: SecretsConfig = Field(default_factory=SecretsConfig)
    workflow_defaults: WorkflowDefConfig = Field(default_factory=WorkflowDefConfig)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_default_provider(self) -> AgentLegConfig:
        """Ensure default_provider references a declared provider (if any)."""
        if self.providers:
            names = {p.name for p in self.providers}
            if self.default_provider not in names:
                raise ValueError(
                    f"default_provider '{self.default_provider}' "
                    f"not found in declared providers: {sorted(names)}"
                )
        return self

    def get_provider(self, name: str | None = None) -> ProviderEntry:
        """Return the ProviderEntry for *name* (or the default)."""
        target = name or self.default_provider
        for p in self.providers:
            if p.name == target:
                return p
        raise ConfigError([f"provider '{target}' not found in configuration"])

    def safe_dict(self) -> dict[str, Any]:
        """Return a dict representation with sensitive values redacted."""
        raw = self.model_dump()
        return redact_sensitive(raw)
