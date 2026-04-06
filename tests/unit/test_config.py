"""Unit tests for configuration management.

Covers: ConfigLoader, AgentLegConfig validation, env overrides,
secrets resolution, and sensible defaults.
"""

import json
import os
from pathlib import Path

import pytest
import yaml

from agentlegatus.config.loader import ConfigLoader, SecretsResolver, _coerce_value
from agentlegatus.config.models import (
    AgentLegConfig,
    ConfigError,
    LoggingConfig,
    ProviderEntry,
    SecretsConfig,
    WorkflowDefConfig,
    _is_sensitive_key,
    redact_sensitive,
)


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------

class TestAgentLegConfig:
    def test_defaults(self):
        cfg = AgentLegConfig()
        assert cfg.default_provider == "mock"
        assert cfg.state.backend == "memory"
        assert cfg.memory.backend == "memory"
        assert cfg.memory.ttl == 3600
        assert cfg.logging.level == "INFO"
        assert cfg.logging.json_format is True
        assert cfg.observability.enable_tracing is False
        assert cfg.secrets.backend == "env"
        assert cfg.workflow_defaults.execution_strategy == "sequential"
        assert cfg.workflow_defaults.max_parallel == 5

    def test_valid_with_providers(self):
        cfg = AgentLegConfig(
            default_provider="langgraph",
            providers=[ProviderEntry(name="langgraph")],
        )
        assert cfg.default_provider == "langgraph"

    def test_default_provider_not_in_list(self):
        with pytest.raises(Exception):
            AgentLegConfig(
                default_provider="missing",
                providers=[ProviderEntry(name="langgraph")],
            )

    def test_get_provider(self):
        cfg = AgentLegConfig(
            default_provider="mock",
            providers=[
                ProviderEntry(name="mock"),
                ProviderEntry(name="langgraph", timeout=60.0),
            ],
        )
        p = cfg.get_provider("langgraph")
        assert p.timeout == 60.0

    def test_get_provider_missing(self):
        cfg = AgentLegConfig(
            default_provider="mock",
            providers=[ProviderEntry(name="mock")],
        )
        with pytest.raises(ConfigError):
            cfg.get_provider("nope")

    def test_safe_dict_redacts_secrets(self):
        cfg = AgentLegConfig(
            default_provider="mock",
            providers=[ProviderEntry(name="mock", api_key="supersecret")],
        )
        safe = cfg.safe_dict()
        # api_key is excluded by Pydantic (exclude=True), so it won't appear
        # but custom_settings with sensitive keys should be redacted
        assert "supersecret" not in json.dumps(safe)


class TestProviderEntry:
    def test_valid(self):
        p = ProviderEntry(name="test", timeout=10.0, max_retries=5)
        assert p.name == "test"

    def test_empty_name_rejected(self):
        with pytest.raises(Exception):
            ProviderEntry(name="  ")

    def test_negative_timeout_rejected(self):
        with pytest.raises(Exception):
            ProviderEntry(name="x", timeout=-1)


class TestLoggingConfig:
    def test_valid_levels(self):
        for lvl in ("DEBUG", "info", "Warning", "ERROR", "critical"):
            cfg = LoggingConfig(level=lvl)
            assert cfg.level == lvl.upper()

    def test_invalid_level(self):
        with pytest.raises(Exception):
            LoggingConfig(level="TRACE")


class TestWorkflowDefConfig:
    def test_valid_strategies(self):
        for s in ("sequential", "parallel", "conditional"):
            cfg = WorkflowDefConfig(execution_strategy=s)
            assert cfg.execution_strategy == s

    def test_invalid_strategy(self):
        with pytest.raises(Exception):
            WorkflowDefConfig(execution_strategy="random")


# ---------------------------------------------------------------------------
# Sensitive key detection & redaction
# ---------------------------------------------------------------------------

class TestSensitiveDetection:
    @pytest.mark.parametrize("key", ["api_key", "API_KEY", "apiKey", "secret", "password", "token", "credential"])
    def test_sensitive_keys(self, key):
        assert _is_sensitive_key(key) is True

    @pytest.mark.parametrize("key", ["name", "backend", "timeout", "level"])
    def test_non_sensitive_keys(self, key):
        assert _is_sensitive_key(key) is False

    def test_redact_sensitive(self):
        data = {"api_key": "abc123", "name": "test", "nested": {"password": "pw", "host": "localhost"}}
        redacted = redact_sensitive(data)
        assert redacted["api_key"] == "***"
        assert redacted["name"] == "test"
        assert redacted["nested"]["password"] == "***"
        assert redacted["nested"]["host"] == "localhost"


# ---------------------------------------------------------------------------
# Coercion helper
# ---------------------------------------------------------------------------

class TestCoerceValue:
    def test_bool_true(self):
        for v in ("true", "True", "1", "yes"):
            assert _coerce_value(v) is True

    def test_bool_false(self):
        for v in ("false", "False", "0", "no"):
            assert _coerce_value(v) is False

    def test_int(self):
        assert _coerce_value("42") == 42

    def test_float(self):
        assert _coerce_value("3.14") == 3.14

    def test_string(self):
        assert _coerce_value("hello") == "hello"


# ---------------------------------------------------------------------------
# ConfigLoader — file loading
# ---------------------------------------------------------------------------

class TestConfigLoaderFile:
    def test_load_yaml(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({
            "default_provider": "mock",
            "providers": [{"name": "mock"}],
            "logging": {"level": "DEBUG"},
        }))
        cfg = ConfigLoader.load(cfg_file)
        assert cfg.default_provider == "mock"
        assert cfg.logging.level == "DEBUG"

    def test_load_json(self, tmp_path):
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps({
            "default_provider": "mock",
            "providers": [{"name": "mock"}],
        }))
        cfg = ConfigLoader.load(cfg_file)
        assert cfg.default_provider == "mock"

    def test_missing_file(self):
        with pytest.raises(ConfigError, match="not found"):
            ConfigLoader.load("/nonexistent/config.yaml")

    def test_unsupported_format(self, tmp_path):
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("")
        with pytest.raises(ConfigError, match="unsupported"):
            ConfigLoader.load(cfg_file)

    def test_invalid_yaml(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("{{invalid yaml")
        with pytest.raises(ConfigError, match="parse"):
            ConfigLoader.load(cfg_file)

    def test_non_dict_yaml(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("- item1\n- item2\n")
        with pytest.raises(ConfigError, match="mapping"):
            ConfigLoader.load(cfg_file)

    def test_defaults_only(self):
        cfg = ConfigLoader.load()
        assert cfg.default_provider == "mock"
        assert cfg.state.backend == "memory"


# ---------------------------------------------------------------------------
# ConfigLoader — env overrides
# ---------------------------------------------------------------------------

class TestConfigLoaderEnv:
    def test_env_overrides_log_level(self, monkeypatch):
        monkeypatch.setenv("AGENTLEGATUS_LOG_LEVEL", "ERROR")
        cfg = ConfigLoader.load()
        assert cfg.logging.level == "ERROR"

    def test_env_overrides_state_backend(self, monkeypatch):
        monkeypatch.setenv("AGENTLEGATUS_STATE_BACKEND", "redis")
        cfg = ConfigLoader.load()
        assert cfg.state.backend == "redis"

    def test_env_overrides_bool(self, monkeypatch):
        monkeypatch.setenv("AGENTLEGATUS_LOG_JSON", "false")
        cfg = ConfigLoader.load()
        assert cfg.logging.json_format is False

    def test_env_overrides_file(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({
            "default_provider": "mock",
            "providers": [{"name": "mock"}],
            "logging": {"level": "DEBUG"},
        }))
        monkeypatch.setenv("AGENTLEGATUS_LOG_LEVEL", "WARNING")
        cfg = ConfigLoader.load(cfg_file)
        # Env should override file
        assert cfg.logging.level == "WARNING"

    def test_provider_api_key_from_env(self, tmp_path, monkeypatch):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({
            "default_provider": "langgraph",
            "providers": [{"name": "langgraph"}],
        }))
        monkeypatch.setenv("LANGGRAPH_API_KEY", "env-key-123")
        cfg = ConfigLoader.load(cfg_file)
        p = cfg.get_provider("langgraph")
        assert p.api_key == "env-key-123"

    def test_explicit_overrides_win(self, monkeypatch):
        monkeypatch.setenv("AGENTLEGATUS_LOG_LEVEL", "ERROR")
        cfg = ConfigLoader.load(overrides={"logging": {"level": "CRITICAL"}})
        assert cfg.logging.level == "CRITICAL"


# ---------------------------------------------------------------------------
# ConfigLoader — secrets resolution
# ---------------------------------------------------------------------------

class TestSecretsResolver:
    def test_env_backend_resolve(self, monkeypatch):
        monkeypatch.setenv("MY_SECRET", "resolved_value")
        resolver = SecretsResolver(SecretsConfig(backend="env"))
        assert resolver.resolve("MY_SECRET") == "resolved_value"

    def test_env_backend_with_prefix(self, monkeypatch):
        monkeypatch.setenv("APP_MY_SECRET", "prefixed_value")
        resolver = SecretsResolver(SecretsConfig(backend="env", prefix="APP_"))
        assert resolver.resolve("MY_SECRET") == "prefixed_value"

    def test_resolve_missing(self):
        resolver = SecretsResolver(SecretsConfig(backend="env"))
        assert resolver.resolve("NONEXISTENT_KEY_XYZ") is None

    def test_resolve_provider_secrets(self, monkeypatch):
        monkeypatch.setenv("MY_API_KEY", "secret123")
        resolver = SecretsResolver(SecretsConfig(backend="env"))
        data = {"name": "test", "api_key": "secret:MY_API_KEY", "timeout": 30}
        resolved = resolver.resolve_provider_secrets(data)
        assert resolved["api_key"] == "secret123"
        assert resolved["name"] == "test"
        assert resolved["timeout"] == 30

    def test_secret_ref_in_config_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROD_KEY", "real-key")
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({
            "default_provider": "mock",
            "providers": [{"name": "mock", "api_key": "secret:PROD_KEY"}],
        }))
        cfg = ConfigLoader.load(cfg_file)
        p = cfg.get_provider("mock")
        assert p.api_key == "real-key"


# ---------------------------------------------------------------------------
# ConfigLoader — validation errors
# ---------------------------------------------------------------------------

class TestConfigLoaderValidation:
    def test_invalid_provider_timeout(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({
            "default_provider": "mock",
            "providers": [{"name": "mock", "timeout": -5}],
        }))
        with pytest.raises(ConfigError):
            ConfigLoader.load(cfg_file)

    def test_invalid_log_level(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump({
            "logging": {"level": "TRACE"},
        }))
        with pytest.raises(ConfigError):
            ConfigLoader.load(cfg_file)
