"""Unit tests for data models.

Covers:
- WorkflowDefinition validation (valid/invalid workflow_id, DAG, cycles, deps, timeout)
- AgentConfig validation (temperature range, max_tokens, tools)
- ExecutionContext methods (create_child_context, get_elapsed_time)
- ProviderConfig validation and from_env
- Metrics data models (MetricsData, ExecutionMetrics, StepMetrics, BenchmarkMetrics)
"""

import os
from datetime import datetime, timedelta

import pytest

from agentlegatus.core.models import AgentCapability, AgentConfig, ExecutionContext, ProviderConfig
from agentlegatus.core.workflow import (
    ExecutionStrategy,
    RetryPolicy,
    WorkflowDefinition,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStep,
)
from agentlegatus.observability.metrics import (
    BenchmarkMetrics,
    ExecutionMetrics,
    MetricsData,
    StepMetrics,
    calculate_token_cost,
)


# ---------------------------------------------------------------------------
# WorkflowDefinition validation
# ---------------------------------------------------------------------------


class TestWorkflowDefinition:
    def _make_step(self, step_id: str, depends_on: list[str] | None = None) -> WorkflowStep:
        return WorkflowStep(
            step_id=step_id,
            step_type="agent",
            config={"model": "gpt-4"},
            depends_on=depends_on or [],
        )

    def _make_workflow(self, **overrides) -> WorkflowDefinition:
        defaults = dict(
            workflow_id="wf-1",
            name="Test Workflow",
            description="desc",
            version="1.0",
            provider="mock",
            steps=[self._make_step("step1")],
        )
        defaults.update(overrides)
        return WorkflowDefinition(**defaults)

    # --- workflow_id ---

    def test_valid_workflow(self):
        wf = self._make_workflow()
        valid, errors = wf.validate()
        assert valid is True
        assert errors == []

    def test_empty_workflow_id(self):
        wf = self._make_workflow(workflow_id="")
        valid, errors = wf.validate()
        assert valid is False
        assert any("workflow_id" in e for e in errors)

    def test_empty_name(self):
        wf = self._make_workflow(name="")
        valid, errors = wf.validate()
        assert valid is False
        assert any("name" in e for e in errors)

    def test_empty_provider(self):
        wf = self._make_workflow(provider="")
        valid, errors = wf.validate()
        assert valid is False
        assert any("provider" in e for e in errors)

    def test_no_steps(self):
        wf = self._make_workflow(steps=[])
        valid, errors = wf.validate()
        assert valid is False
        assert any("at least one step" in e for e in errors)

    # --- timeout ---

    def test_positive_timeout(self):
        wf = self._make_workflow(timeout=60.0)
        valid, errors = wf.validate()
        assert valid is True

    def test_zero_timeout(self):
        wf = self._make_workflow(timeout=0)
        valid, errors = wf.validate()
        assert valid is False
        assert any("timeout" in e for e in errors)

    def test_negative_timeout(self):
        wf = self._make_workflow(timeout=-5.0)
        valid, errors = wf.validate()
        assert valid is False
        assert any("timeout" in e for e in errors)

    def test_none_timeout_is_valid(self):
        wf = self._make_workflow(timeout=None)
        valid, errors = wf.validate()
        assert valid is True

    # --- DAG / dependency validation ---

    def test_valid_dag(self):
        steps = [
            self._make_step("a"),
            self._make_step("b", depends_on=["a"]),
            self._make_step("c", depends_on=["a", "b"]),
        ]
        wf = self._make_workflow(steps=steps)
        valid, errors = wf.validate()
        assert valid is True

    def test_invalid_dependency_reference(self):
        steps = [
            self._make_step("a"),
            self._make_step("b", depends_on=["nonexistent"]),
        ]
        wf = self._make_workflow(steps=steps)
        valid, errors = wf.validate()
        assert valid is False
        assert any("nonexistent" in e for e in errors)

    def test_duplicate_step_ids(self):
        steps = [
            self._make_step("a"),
            self._make_step("a"),
        ]
        wf = self._make_workflow(steps=steps)
        valid, errors = wf.validate()
        assert valid is False
        assert any("duplicate" in e for e in errors)

    def test_cycle_detection_simple(self):
        steps = [
            self._make_step("a", depends_on=["b"]),
            self._make_step("b", depends_on=["a"]),
        ]
        wf = self._make_workflow(steps=steps)
        valid, errors = wf.validate()
        assert valid is False
        assert any("cycle" in e for e in errors)

    def test_cycle_detection_three_nodes(self):
        steps = [
            self._make_step("a", depends_on=["c"]),
            self._make_step("b", depends_on=["a"]),
            self._make_step("c", depends_on=["b"]),
        ]
        wf = self._make_workflow(steps=steps)
        valid, errors = wf.validate()
        assert valid is False
        assert any("cycle" in e for e in errors)

    def test_step_timeout_validation(self):
        step = WorkflowStep(
            step_id="s1",
            step_type="agent",
            config={},
            timeout=-1.0,
        )
        wf = self._make_workflow(steps=[step])
        valid, errors = wf.validate()
        assert valid is False
        assert any("timeout" in e for e in errors)

    def test_step_retry_policy_validation(self):
        step = WorkflowStep(
            step_id="s1",
            step_type="agent",
            config={},
            retry_policy=RetryPolicy(max_attempts=0),
        )
        wf = self._make_workflow(steps=[step])
        valid, errors = wf.validate()
        assert valid is False
        assert any("max_attempts" in e for e in errors)


# ---------------------------------------------------------------------------
# RetryPolicy validation
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_valid_defaults(self):
        rp = RetryPolicy()
        valid, errors = rp.validate()
        assert valid is True

    def test_max_attempts_zero(self):
        rp = RetryPolicy(max_attempts=0)
        valid, errors = rp.validate()
        assert valid is False
        assert any("max_attempts" in e for e in errors)

    def test_backoff_multiplier_below_one(self):
        rp = RetryPolicy(backoff_multiplier=0.5)
        valid, errors = rp.validate()
        assert valid is False
        assert any("backoff_multiplier" in e for e in errors)

    def test_negative_initial_delay(self):
        rp = RetryPolicy(initial_delay=-1.0)
        valid, errors = rp.validate()
        assert valid is False

    def test_initial_delay_exceeds_max_delay(self):
        rp = RetryPolicy(initial_delay=100.0, max_delay=10.0)
        valid, errors = rp.validate()
        assert valid is False
        assert any("initial_delay cannot exceed max_delay" in e for e in errors)


# ---------------------------------------------------------------------------
# WorkflowStep validation
# ---------------------------------------------------------------------------


class TestWorkflowStep:
    def test_valid_step(self):
        step = WorkflowStep(step_id="s1", step_type="agent", config={})
        valid, errors = step.validate(["s1"])
        assert valid is True

    def test_empty_step_id(self):
        step = WorkflowStep(step_id="", step_type="agent", config={})
        valid, errors = step.validate([])
        assert valid is False
        assert any("step_id" in e for e in errors)

    def test_empty_step_type(self):
        step = WorkflowStep(step_id="s1", step_type="", config={})
        valid, errors = step.validate(["s1"])
        assert valid is False
        assert any("step_type" in e for e in errors)


# ---------------------------------------------------------------------------
# WorkflowResult
# ---------------------------------------------------------------------------


class TestWorkflowResult:
    def test_completed_result(self):
        result = WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            output={"answer": 42},
            metrics={"duration": 1.5},
            execution_time=1.5,
        )
        assert result.status == WorkflowStatus.COMPLETED
        assert result.output == {"answer": 42}
        assert result.error is None

    def test_failed_result_with_error(self):
        err = RuntimeError("boom")
        result = WorkflowResult(
            status=WorkflowStatus.FAILED,
            output=None,
            metrics={},
            execution_time=0.1,
            error=err,
        )
        assert result.status == WorkflowStatus.FAILED
        assert result.error is err


# ---------------------------------------------------------------------------
# AgentConfig validation
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def _make_config(self, **overrides) -> AgentConfig:
        defaults = dict(
            agent_id="agent-1",
            name="Test Agent",
            model="gpt-4",
        )
        defaults.update(overrides)
        return AgentConfig(**defaults)

    def test_valid_config(self):
        cfg = self._make_config()
        valid, errors = cfg.validate(available_tools=[])
        assert valid is True

    def test_empty_agent_id(self):
        cfg = self._make_config(agent_id="")
        valid, errors = cfg.validate(available_tools=[])
        assert valid is False
        assert any("agent_id" in e for e in errors)

    def test_empty_name(self):
        cfg = self._make_config(name="")
        valid, errors = cfg.validate(available_tools=[])
        assert valid is False
        assert any("name" in e for e in errors)

    def test_empty_model(self):
        cfg = self._make_config(model="")
        valid, errors = cfg.validate(available_tools=[])
        assert valid is False
        assert any("model" in e for e in errors)

    # --- temperature range 0.0 - 2.0 ---

    def test_temperature_zero(self):
        cfg = self._make_config(temperature=0.0)
        valid, _ = cfg.validate(available_tools=[])
        assert valid is True

    def test_temperature_two(self):
        cfg = self._make_config(temperature=2.0)
        valid, _ = cfg.validate(available_tools=[])
        assert valid is True

    def test_temperature_below_zero(self):
        cfg = self._make_config(temperature=-0.1)
        valid, errors = cfg.validate(available_tools=[])
        assert valid is False
        assert any("temperature" in e for e in errors)

    def test_temperature_above_two(self):
        cfg = self._make_config(temperature=2.1)
        valid, errors = cfg.validate(available_tools=[])
        assert valid is False
        assert any("temperature" in e for e in errors)

    # --- max_tokens positive ---

    def test_max_tokens_positive(self):
        cfg = self._make_config(max_tokens=100)
        valid, _ = cfg.validate(available_tools=[])
        assert valid is True

    def test_max_tokens_zero(self):
        cfg = self._make_config(max_tokens=0)
        valid, errors = cfg.validate(available_tools=[])
        assert valid is False
        assert any("max_tokens" in e for e in errors)

    def test_max_tokens_negative(self):
        cfg = self._make_config(max_tokens=-10)
        valid, errors = cfg.validate(available_tools=[])
        assert valid is False
        assert any("max_tokens" in e for e in errors)

    # --- tool references ---

    def test_valid_tool_references(self):
        cfg = self._make_config(tools=["search", "calculator"])
        valid, _ = cfg.validate(available_tools=["search", "calculator", "browser"])
        assert valid is True

    def test_invalid_tool_reference(self):
        cfg = self._make_config(tools=["search", "missing_tool"])
        valid, errors = cfg.validate(available_tools=["search"])
        assert valid is False
        assert any("missing_tool" in e for e in errors)

    # --- capabilities ---

    def test_capabilities_default_empty(self):
        cfg = self._make_config()
        assert cfg.capabilities == []

    def test_capabilities_set(self):
        cfg = self._make_config(
            capabilities=[AgentCapability.TOOL_USE, AgentCapability.MEMORY]
        )
        assert AgentCapability.TOOL_USE in cfg.capabilities
        assert AgentCapability.MEMORY in cfg.capabilities


# ---------------------------------------------------------------------------
# ProviderConfig validation and from_env
# ---------------------------------------------------------------------------


class TestProviderConfig:
    def test_valid_defaults(self):
        cfg = ProviderConfig(provider_name="mock")
        valid, errors = cfg.validate()
        assert valid is True

    def test_empty_provider_name(self):
        cfg = ProviderConfig(provider_name="")
        valid, errors = cfg.validate()
        assert valid is False
        assert any("provider_name" in e for e in errors)

    def test_negative_timeout(self):
        cfg = ProviderConfig(provider_name="mock", timeout=-1.0)
        valid, errors = cfg.validate()
        assert valid is False
        assert any("timeout" in e for e in errors)

    def test_negative_max_retries(self):
        cfg = ProviderConfig(provider_name="mock", max_retries=-1)
        valid, errors = cfg.validate()
        assert valid is False
        assert any("max_retries" in e for e in errors)

    def test_zero_rate_limit(self):
        cfg = ProviderConfig(provider_name="mock", rate_limit=0)
        valid, errors = cfg.validate()
        assert valid is False
        assert any("rate_limit" in e for e in errors)

    def test_none_rate_limit_is_valid(self):
        cfg = ProviderConfig(provider_name="mock", rate_limit=None)
        valid, _ = cfg.validate()
        assert valid is True

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("MOCK_API_KEY", "test-key-123")
        monkeypatch.setenv("MOCK_API_BASE", "https://api.example.com")
        monkeypatch.setenv("MOCK_TIMEOUT", "45.0")
        monkeypatch.setenv("MOCK_MAX_RETRIES", "5")
        monkeypatch.setenv("MOCK_RATE_LIMIT", "100")

        cfg = ProviderConfig.from_env("mock")
        assert cfg.provider_name == "mock"
        assert cfg.api_key == "test-key-123"
        assert cfg.api_base == "https://api.example.com"
        assert cfg.timeout == 45.0
        assert cfg.max_retries == 5
        assert cfg.rate_limit == 100

    def test_from_env_defaults(self, monkeypatch):
        # Ensure no env vars are set for this provider
        for key in ["TESTPROV_API_KEY", "TESTPROV_API_BASE", "TESTPROV_TIMEOUT",
                     "TESTPROV_MAX_RETRIES", "TESTPROV_RATE_LIMIT"]:
            monkeypatch.delenv(key, raising=False)

        cfg = ProviderConfig.from_env("testprov")
        assert cfg.provider_name == "testprov"
        assert cfg.api_key is None
        assert cfg.api_base is None
        assert cfg.timeout == 30.0
        assert cfg.max_retries == 3
        assert cfg.rate_limit is None

    def test_from_env_hyphenated_name(self, monkeypatch):
        monkeypatch.setenv("MY_PROVIDER_API_KEY", "key-456")
        cfg = ProviderConfig.from_env("my-provider")
        assert cfg.api_key == "key-456"


# ---------------------------------------------------------------------------
# ExecutionContext
# ---------------------------------------------------------------------------


class TestExecutionContext:
    def _make_context(self, **overrides) -> ExecutionContext:
        defaults = dict(
            workflow_id="wf-1",
            execution_id="exec-1",
            current_step="step-1",
            state={"key": "value"},
            metadata={"env": "test"},
            start_time=datetime.now(),
            trace_id="trace-abc",
        )
        defaults.update(overrides)
        return ExecutionContext(**defaults)

    def test_create_child_context(self):
        parent = self._make_context()
        child = parent.create_child_context("step-2")

        assert child.workflow_id == parent.workflow_id
        assert child.execution_id == "exec-1_step-2"
        assert child.current_step == "step-2"
        assert child.parent_context is parent
        assert child.trace_id == parent.trace_id

    def test_child_context_state_is_copy(self):
        parent = self._make_context()
        child = parent.create_child_context("step-2")

        # Mutating child state should not affect parent
        child.state["new_key"] = "new_value"
        assert "new_key" not in parent.state

    def test_child_context_metadata_is_copy(self):
        parent = self._make_context()
        child = parent.create_child_context("step-2")

        child.metadata["extra"] = True
        assert "extra" not in parent.metadata

    def test_get_elapsed_time(self):
        ctx = self._make_context(start_time=datetime.now() - timedelta(seconds=2))
        elapsed = ctx.get_elapsed_time()
        assert elapsed >= 2.0

    def test_get_elapsed_time_fresh(self):
        ctx = self._make_context(start_time=datetime.now())
        elapsed = ctx.get_elapsed_time()
        assert elapsed >= 0.0
        assert elapsed < 1.0


# ---------------------------------------------------------------------------
# MetricsData
# ---------------------------------------------------------------------------


class TestMetricsData:
    def _make_metric(self, **overrides) -> MetricsData:
        defaults = dict(
            execution_id="exec-1",
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
            metric_type="workflow_duration_seconds",
            value=5.5,
            unit="seconds",
            labels={"workflow_id": "wf-1", "provider": "mock"},
        )
        defaults.update(overrides)
        return MetricsData(**defaults)

    def test_to_prometheus_format_with_labels(self):
        m = self._make_metric()
        prom = m.to_prometheus_format()
        assert "workflow_duration_seconds" in prom
        assert 'workflow_id="wf-1"' in prom
        assert 'provider="mock"' in prom
        assert "5.5" in prom

    def test_to_prometheus_format_no_labels(self):
        m = self._make_metric(labels={})
        prom = m.to_prometheus_format()
        assert prom == "workflow_duration_seconds 5.5"

    def test_to_opentelemetry_format(self):
        m = self._make_metric()
        otel = m.to_opentelemetry_format()
        assert otel["name"] == "workflow_duration_seconds"
        assert otel["value"] == 5.5
        assert otel["unit"] == "seconds"
        assert "timestamp" in otel
        assert otel["attributes"]["execution_id"] == "exec-1"
        assert otel["attributes"]["workflow_id"] == "wf-1"

    def test_to_opentelemetry_format_timestamp_is_iso(self):
        m = self._make_metric()
        otel = m.to_opentelemetry_format()
        # Should be parseable ISO format
        datetime.fromisoformat(otel["timestamp"])


# ---------------------------------------------------------------------------
# StepMetrics
# ---------------------------------------------------------------------------


class TestStepMetrics:
    def test_finalize_sets_end_time_and_duration(self):
        start = datetime.now() - timedelta(seconds=3)
        sm = StepMetrics(step_id="s1", start_time=start)
        sm.finalize()
        assert sm.end_time is not None
        assert sm.duration is not None
        assert sm.duration >= 3.0

    def test_finalize_with_provider_calculates_cost(self):
        sm = StepMetrics(
            step_id="s1",
            start_time=datetime.now(),
            input_tokens=1_000_000,
            output_tokens=500_000,
        )
        sm.finalize(provider="openai")
        assert sm.provider == "openai"
        assert sm.cost > 0.0

    def test_finalize_preserves_existing_end_time(self):
        end = datetime(2024, 1, 15, 12, 0, 5)
        sm = StepMetrics(
            step_id="s1",
            start_time=datetime(2024, 1, 15, 12, 0, 0),
            end_time=end,
        )
        sm.finalize()
        assert sm.end_time == end
        assert sm.duration == 5.0

    def test_finalize_does_not_overwrite_existing_cost(self):
        sm = StepMetrics(
            step_id="s1",
            start_time=datetime.now(),
            input_tokens=1000,
            output_tokens=500,
            cost=99.99,
        )
        sm.finalize(provider="openai")
        # cost was already set, should not be overwritten
        assert sm.cost == 99.99


# ---------------------------------------------------------------------------
# ExecutionMetrics
# ---------------------------------------------------------------------------


class TestExecutionMetrics:
    def _make_exec_metrics(self) -> ExecutionMetrics:
        step1 = StepMetrics(
            step_id="s1",
            start_time=datetime.now(),
            input_tokens=1000,
            output_tokens=500,
            cost=0.05,
        )
        step2 = StepMetrics(
            step_id="s2",
            start_time=datetime.now(),
            input_tokens=2000,
            output_tokens=1000,
            cost=0.10,
        )
        return ExecutionMetrics(
            execution_id="exec-1",
            workflow_id="wf-1",
            provider="mock",
            start_time=datetime.now(),
            step_metrics=[step1, step2],
        )

    def test_calculate_cost(self):
        em = self._make_exec_metrics()
        total = em.calculate_cost()
        assert total == pytest.approx(0.15)
        assert em.total_cost == pytest.approx(0.15)

    def test_aggregate_tokens(self):
        em = self._make_exec_metrics()
        tokens = em.aggregate_tokens()
        assert tokens["input"] == 3000
        assert tokens["output"] == 1500
        assert tokens["total"] == 4500
        assert em.token_usage == tokens

    def test_get_summary(self):
        em = self._make_exec_metrics()
        em.end_time = em.start_time + timedelta(seconds=10)
        summary = em.get_summary()

        assert summary["execution_id"] == "exec-1"
        assert summary["workflow_id"] == "wf-1"
        assert summary["provider"] == "mock"
        assert summary["status"] == "running"
        assert summary["duration"] == pytest.approx(10.0)
        assert summary["step_count"] == 2

    def test_get_summary_without_end_time(self):
        em = self._make_exec_metrics()
        summary = em.get_summary()
        assert summary["duration"] is None

    def test_empty_step_metrics(self):
        em = ExecutionMetrics(
            execution_id="exec-1",
            workflow_id="wf-1",
            provider="mock",
            start_time=datetime.now(),
        )
        assert em.calculate_cost() == 0.0
        tokens = em.aggregate_tokens()
        assert tokens["total"] == 0


# ---------------------------------------------------------------------------
# BenchmarkMetrics
# ---------------------------------------------------------------------------


class TestBenchmarkMetrics:
    def test_creation(self):
        bm = BenchmarkMetrics(
            provider_name="openai",
            execution_time=5.2,
            total_cost=0.25,
            token_usage={"input": 5000, "output": 2000, "total": 7000},
            success_rate=0.95,
            error_count=1,
            latency_p50=1.0,
            latency_p95=3.5,
            latency_p99=5.0,
        )
        assert bm.provider_name == "openai"
        assert bm.execution_time == 5.2
        assert bm.total_cost == 0.25
        assert bm.success_rate == 0.95
        assert bm.error_count == 1
        assert bm.latency_p50 == 1.0
        assert bm.latency_p95 == 3.5
        assert bm.latency_p99 == 5.0
        assert bm.custom_metrics == {}

    def test_custom_metrics(self):
        bm = BenchmarkMetrics(
            provider_name="anthropic",
            execution_time=3.0,
            total_cost=0.10,
            token_usage={"input": 1000, "output": 500, "total": 1500},
            success_rate=1.0,
            error_count=0,
            latency_p50=0.5,
            latency_p95=1.5,
            latency_p99=2.5,
            custom_metrics={"quality_score": 0.92},
        )
        assert bm.custom_metrics["quality_score"] == 0.92


# ---------------------------------------------------------------------------
# calculate_token_cost
# ---------------------------------------------------------------------------


class TestCalculateTokenCost:
    def test_known_provider(self):
        cost = calculate_token_cost("openai", 1_000_000, 1_000_000)
        # openai: input 2.50/1M + output 10.00/1M = 12.50
        assert cost == pytest.approx(12.50)

    def test_unknown_provider_uses_default(self):
        cost = calculate_token_cost("unknown_provider", 1_000_000, 1_000_000)
        # default: input 2.00/1M + output 8.00/1M = 10.00
        assert cost == pytest.approx(10.00)

    def test_zero_tokens(self):
        cost = calculate_token_cost("openai", 0, 0)
        assert cost == 0.0

    def test_anthropic_pricing(self):
        cost = calculate_token_cost("anthropic", 1_000_000, 1_000_000)
        # anthropic: input 3.00/1M + output 15.00/1M = 18.00
        assert cost == pytest.approx(18.00)


# ---------------------------------------------------------------------------
# WorkflowStatus enum
# ---------------------------------------------------------------------------


class TestWorkflowStatus:
    def test_all_statuses_exist(self):
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"
        assert WorkflowStatus.CANCELLED.value == "cancelled"


# ---------------------------------------------------------------------------
# ExecutionStrategy enum
# ---------------------------------------------------------------------------


class TestExecutionStrategy:
    def test_all_strategies_exist(self):
        assert ExecutionStrategy.SEQUENTIAL.value == "sequential"
        assert ExecutionStrategy.PARALLEL.value == "parallel"
        assert ExecutionStrategy.CONDITIONAL.value == "conditional"
