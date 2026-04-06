"""Configuration loader with YAML/JSON support, env overrides, and secrets.

Requirements: 19.1-19.9, 20.1, 20.2
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from agentlegatus.config.models import (
    AgentLegConfig,
    ConfigError,
    SecretsConfig,
)
from agentlegatus.utils.logging import get_logger

logger = get_logger(__name__)

# Environment variable prefix for AgentLegatus settings
ENV_PREFIX = "AGENTLEGATUS_"

# Mapping of flat env-var suffixes → nested config paths
_ENV_MAPPING: dict[str, list[str]] = {
    "DEFAULT_PROVIDER": ["default_provider"],
    "STATE_BACKEND": ["state", "backend"],
    "STATE_CONNECTION_URL": ["state", "connection_url"],
    "MEMORY_BACKEND": ["memory", "backend"],
    "MEMORY_CONNECTION_URL": ["memory", "connection_url"],
    "LOG_LEVEL": ["logging", "level"],
    "LOG_JSON": ["logging", "json_format"],
    "OBSERVABILITY_TRACING": ["observability", "enable_tracing"],
    "OBSERVABILITY_PROMETHEUS": ["observability", "enable_prometheus"],
    "OBSERVABILITY_OTLP_ENDPOINT": ["observability", "otlp_endpoint"],
    "SECRETS_BACKEND": ["secrets", "backend"],
    "SECRETS_PREFIX": ["secrets", "prefix"],
    "WORKFLOW_TIMEOUT": ["workflow_defaults", "timeout"],
    "WORKFLOW_STRATEGY": ["workflow_defaults", "execution_strategy"],
    "WORKFLOW_MAX_PARALLEL": ["workflow_defaults", "max_parallel"],
}


def _coerce_value(value: str) -> Any:
    """Best-effort coercion of string env values to Python types."""
    low = value.lower()
    if low in ("true", "1", "yes"):
        return True
    if low in ("false", "0", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _set_nested(data: dict[str, Any], path: list[str], value: Any) -> None:
    """Set a value in a nested dict following *path*."""
    for key in path[:-1]:
        data = data.setdefault(key, {})
    data[path[-1]] = value


# ---------------------------------------------------------------------------
# Secrets resolution
# ---------------------------------------------------------------------------


class SecretsResolver:
    """Resolve secret references in configuration values.

    Currently supports:
    - ``env`` backend: reads from environment variables.

    The resolver never logs or exposes resolved values (Req 20.1, 20.2).
    """

    def __init__(self, secrets_cfg: SecretsConfig) -> None:
        self._backend = secrets_cfg.backend
        self._prefix = secrets_cfg.prefix
        self._config = secrets_cfg.config

    def resolve(self, ref: str) -> str | None:
        """Resolve a single secret reference.

        Args:
            ref: The secret name (without prefix).

        Returns:
            The resolved value, or None if not found.
        """
        full_key = f"{self._prefix}{ref}" if self._prefix else ref

        if self._backend == "env":
            return os.environ.get(full_key)

        # Extensible: add aws_ssm, vault, etc. here
        return None

    def resolve_provider_secrets(self, data: dict[str, Any]) -> dict[str, Any]:
        """Walk *data* and resolve any values that look like ``secret:<ref>``.

        Returns a new dict with secrets resolved in-place.
        """
        out: dict[str, Any] = {}
        for k, v in data.items():
            if isinstance(v, str) and v.startswith("secret:"):
                ref = v[len("secret:") :]
                resolved = self.resolve(ref)
                out[k] = resolved  # None if not found
            elif isinstance(v, dict):
                out[k] = self.resolve_provider_secrets(v)
            else:
                out[k] = v
        return out


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------


class ConfigLoader:
    """Load, merge, and validate AgentLegatus configuration.

    Precedence (highest wins):
        1. Explicit overrides passed to ``load()``
        2. Environment variables (``AGENTLEGATUS_*``)
        3. Configuration file (YAML or JSON)
        4. Built-in defaults (from Pydantic model)

    Usage::

        config = ConfigLoader.load("config.yaml")
        config = ConfigLoader.load()  # env + defaults only
        config = ConfigLoader.load(overrides={"default_provider": "langgraph"})
    """

    @staticmethod
    def load(
        path: str | Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> AgentLegConfig:
        """Load configuration from file + env + overrides.

        Args:
            path: Optional path to a YAML or JSON config file.
            overrides: Optional dict merged on top of everything else.

        Returns:
            Validated ``AgentLegConfig``.

        Raises:
            ConfigError: When the file cannot be read or validation fails.
        """
        data: dict[str, Any] = {}

        # 1. File
        if path is not None:
            data = ConfigLoader._load_file(Path(path))

        # 2. Env overrides
        env_data = ConfigLoader._collect_env_overrides()
        ConfigLoader._deep_merge(data, env_data)

        # 3. Per-provider env API keys  (e.g. LANGGRAPH_API_KEY)
        ConfigLoader._apply_provider_env_keys(data)

        # 4. Explicit overrides
        if overrides:
            ConfigLoader._deep_merge(data, overrides)

        # 5. Resolve secrets
        data = ConfigLoader._resolve_secrets(data)

        # 6. Validate via Pydantic
        try:
            config = AgentLegConfig.model_validate(data)
        except ValidationError as exc:
            errors = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
            raise ConfigError(errors) from exc

        # Log (redacted) config at debug level
        try:
            logger.debug("configuration_loaded", config=config.safe_dict())
        except Exception:
            pass

        return config

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_file(path: Path) -> dict[str, Any]:
        """Read a YAML or JSON file and return its contents as a dict."""
        if not path.exists():
            raise ConfigError([f"configuration file not found: {path}"])

        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError([f"cannot read configuration file: {exc}"]) from exc

        suffix = path.suffix.lower()
        try:
            if suffix in (".yaml", ".yml"):
                data = yaml.safe_load(raw) or {}
            elif suffix == ".json":
                data = json.loads(raw)
            else:
                raise ConfigError(
                    [f"unsupported config file format '{suffix}'; use .yaml, .yml, or .json"]
                )
        except (yaml.YAMLError, json.JSONDecodeError) as exc:
            raise ConfigError([f"failed to parse configuration file: {exc}"]) from exc

        if not isinstance(data, dict):
            raise ConfigError(["configuration file must contain a mapping at the top level"])

        return data

    # ------------------------------------------------------------------
    # Environment variable collection
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_env_overrides() -> dict[str, Any]:
        """Collect AGENTLEGATUS_* env vars and map them to config paths."""
        data: dict[str, Any] = {}
        for suffix, path in _ENV_MAPPING.items():
            env_key = f"{ENV_PREFIX}{suffix}"
            value = os.environ.get(env_key)
            if value is not None:
                _set_nested(data, path, _coerce_value(value))
        return data

    @staticmethod
    def _apply_provider_env_keys(data: dict[str, Any]) -> None:
        """Inject per-provider API keys from env (e.g. LANGGRAPH_API_KEY)."""
        providers = data.get("providers")
        if not isinstance(providers, list):
            return
        for entry in providers:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name", "")
            env_key = f"{name.upper().replace('-', '_')}_API_KEY"
            env_val = os.environ.get(env_key)
            if env_val and not entry.get("api_key"):
                entry["api_key"] = env_val

    # ------------------------------------------------------------------
    # Secrets resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_secrets(data: dict[str, Any]) -> dict[str, Any]:
        """Resolve ``secret:`` references using the configured secrets backend."""
        secrets_raw = data.get("secrets", {})
        try:
            secrets_cfg = SecretsConfig.model_validate(secrets_raw)
        except Exception:
            secrets_cfg = SecretsConfig()

        resolver = SecretsResolver(secrets_cfg)

        # Walk provider entries
        providers = data.get("providers")
        if isinstance(providers, list):
            data["providers"] = [
                resolver.resolve_provider_secrets(p) if isinstance(p, dict) else p
                for p in providers
            ]

        return data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
        """Recursively merge *override* into *base* (mutates *base*)."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                ConfigLoader._deep_merge(base[key], value)
            else:
                base[key] = value
