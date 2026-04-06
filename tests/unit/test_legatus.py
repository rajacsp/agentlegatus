"""Unit tests for the Legatus orchestrator."""

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentlegatus.core.event_bus import Event, EventBus, EventType
from agentlegatus.core.executor import WorkflowExecutor
from agentlegatus.core.state import InMemoryStateBackend, StateManager, StateScope
from agentlegatus.core.workflow import (
    ExecutionStrategy,
    WorkflowDefinition,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStep,
)
from agentlegatus.hierarchy.centurion import Centurion
from agentlegatus.hierarchy.legatus import Legatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def state_manager(event_bus: EventBus) -> StateManager:
    backend = InMemoryStateBackend()
    return StateManager(backend=backend, event_bus=event_bus)


@pytest.fixture
def legatus(event_bus: EventBus) -> Legatus:
    return Legatus(config={}, event_bus=event_bus)


def _make_workflow(
    workflow_id: str = "wf-1",
    timeout: Optional[float] = None,
    strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
) -> WorkflowDefinition:
    """Helper to create a minimal WorkflowDefinition."""
    return WorkflowDefinition(
        workflow_id=workflow_id,
        name="Test Workflow",
        description="A test workflow",
        version="1.0",
        provider="mock",
        steps=[
            WorkflowStep(step_id="s1", step_type="agent", config={}),
        ],
        timeout=timeout,
        execution_strategy=strategy,
    )


def _make_executor(state_manager: StateManager, event_bus: EventBus) -> WorkflowExecutor:
    """Create a WorkflowExecutor with a mock provider and working execute_step."""
    provider = MagicMock()
    provider.execute_agent = AsyncMock(return_value={"result": "ok"})
    provider.export_state = MagicMock(return_value={})
    tool_registry = MagicMock()
    executor = WorkflowExecutor(
        provider=provider,
        state_manager=state_manager,
        tool_registry=tool_registry,
        event_bus=event_bus,
    )
    # Patch execute_step since task 11.1 is not yet implemented
    executor.execute_step = AsyncMock(return_value={"result": "ok"})
    return executor


# ---------------------------------------------------------------------------
# 13.1 — Legatus class basics
# ---------------------------------------------------------------------------

class TestLegatusInit:
    """Tests for Legatus __init__, add_centurion, get_status."""

    @pytest.mark.asyncio
    async def test_add_centurion(self, legatus: Legatus, event_bus: EventBus):
        centurion = Centurion("c1", ExecutionStrategy.SEQUENTIAL, event_bus)
        await legatus.add_centurion(centurion)
        assert "c1" in legatus._centurions

    @pytest.mark.asyncio
    async def test_add_multiple_centurions(self, legatus: Legatus, event_bus: EventBus):
        c1 = Centurion("c1", ExecutionStrategy.SEQUENTIAL, event_bus)
        c2 = Centurion("c2", ExecutionStrategy.PARALLEL, event_bus)
        await legatus.add_centurion(c1)
        await legatus.add_centurion(c2)
        assert len(legatus._centurions) == 2

    def test_get_status_unknown_workflow(self, legatus: Legatus):
        with pytest.raises(KeyError, match="not found"):
            legatus.get_status("nonexistent")

    @pytest.mark.asyncio
    async def test_get_status_after_execution(
        self, legatus: Legatus, state_manager: StateManager, event_bus: EventBus
    ):
        executor = _make_executor(state_manager, event_bus)
        wf = _make_workflow()
        await legatus.execute_workflow(wf, executor=executor, state_manager=state_manager)
        assert legatus.get_status("wf-1") == WorkflowStatus.COMPLETED


# ---------------------------------------------------------------------------
# 13.2 — Workflow execution
# ---------------------------------------------------------------------------

class TestWorkflowExecution:
    """Tests for execute_workflow."""

    @pytest.mark.asyncio
    async def test_successful_execution_returns_completed(
        self, legatus: Legatus, state_manager: StateManager, event_bus: EventBus
    ):
        executor = _make_executor(state_manager, event_bus)
        wf = _make_workflow()
        result = await legatus.execute_workflow(wf, executor=executor, state_manager=state_manager)

        assert result.status == WorkflowStatus.COMPLETED
        assert result.execution_time > 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_execution_emits_started_and_completed_events(
        self, legatus: Legatus, state_manager: StateManager, event_bus: EventBus
    ):
        captured: List[Event] = []

        async def capture(event: Event) -> None:
            captured.append(event)

        event_bus.subscribe(EventType.WORKFLOW_STARTED, capture)
        event_bus.subscribe(EventType.WORKFLOW_COMPLETED, capture)

        executor = _make_executor(state_manager, event_bus)
        wf = _make_workflow()
        await legatus.execute_workflow(wf, executor=executor, state_manager=state_manager)

        # Allow fire-and-forget handlers to complete
        await asyncio.sleep(0.05)

        types = [e.event_type for e in captured]
        assert EventType.WORKFLOW_STARTED in types
        assert EventType.WORKFLOW_COMPLETED in types

    @pytest.mark.asyncio
    async def test_execution_includes_trace_id(
        self, legatus: Legatus, state_manager: StateManager, event_bus: EventBus
    ):
        captured: List[Event] = []

        async def capture(event: Event) -> None:
            captured.append(event)

        event_bus.subscribe(EventType.WORKFLOW_STARTED, capture)

        executor = _make_executor(state_manager, event_bus)
        wf = _make_workflow()
        await legatus.execute_workflow(wf, executor=executor, state_manager=state_manager)
        await asyncio.sleep(0.05)

        started = [e for e in captured if e.event_type == EventType.WORKFLOW_STARTED]
        assert len(started) == 1
        assert started[0].trace_id is not None

    @pytest.mark.asyncio
    async def test_execution_metrics_in_result(
        self, legatus: Legatus, state_manager: StateManager, event_bus: EventBus
    ):
        executor = _make_executor(state_manager, event_bus)
        wf = _make_workflow()
        result = await legatus.execute_workflow(wf, executor=executor, state_manager=state_manager)

        assert "workflow_id" in result.metrics
        assert "duration" in result.metrics
        assert "trace_id" in result.metrics
        assert result.metrics["workflow_id"] == "wf-1"

    @pytest.mark.asyncio
    async def test_execution_initialises_state(
        self, legatus: Legatus, state_manager: StateManager, event_bus: EventBus
    ):
        executor = _make_executor(state_manager, event_bus)
        wf = _make_workflow()
        initial = {"foo": "bar"}
        await legatus.execute_workflow(
            wf, initial_state=initial, executor=executor, state_manager=state_manager
        )

        val = await state_manager.get("foo", scope=StateScope.WORKFLOW)
        assert val == "bar"

    @pytest.mark.asyncio
    async def test_failed_execution_returns_failed(
        self, legatus: Legatus, state_manager: StateManager, event_bus: EventBus
    ):
        executor = _make_executor(state_manager, event_bus)
        # Make execute_step raise an error
        executor.execute_step = AsyncMock(side_effect=RuntimeError("boom"))

        wf = _make_workflow()
        result = await legatus.execute_workflow(wf, executor=executor, state_manager=state_manager)

        assert result.status == WorkflowStatus.FAILED
        assert result.error is not None
        assert "boom" in str(result.error)

    @pytest.mark.asyncio
    async def test_failed_execution_emits_failed_event(
        self, legatus: Legatus, state_manager: StateManager, event_bus: EventBus
    ):
        captured: List[Event] = []

        async def capture(event: Event) -> None:
            captured.append(event)

        event_bus.subscribe(EventType.WORKFLOW_FAILED, capture)

        executor = _make_executor(state_manager, event_bus)
        executor.execute_step = AsyncMock(side_effect=RuntimeError("boom"))

        wf = _make_workflow()
        await legatus.execute_workflow(wf, executor=executor, state_manager=state_manager)
        await asyncio.sleep(0.05)

        assert len(captured) == 1
        assert captured[0].data["error"] == "boom"

    @pytest.mark.asyncio
    async def test_uses_registered_centurion(
        self, legatus: Legatus, state_manager: StateManager, event_bus: EventBus
    ):
        """When a centurion is registered, Legatus should use it."""
        centurion = Centurion("custom", ExecutionStrategy.SEQUENTIAL, event_bus)
        await legatus.add_centurion(centurion)

        executor = _make_executor(state_manager, event_bus)
        wf = _make_workflow()
        result = await legatus.execute_workflow(wf, executor=executor, state_manager=state_manager)
        assert result.status == WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_creates_default_centurion_when_none_registered(
        self, legatus: Legatus, state_manager: StateManager, event_bus: EventBus
    ):
        """When no centurion is registered, a default one is created."""
        executor = _make_executor(state_manager, event_bus)
        wf = _make_workflow()
        result = await legatus.execute_workflow(wf, executor=executor, state_manager=state_manager)
        assert result.status == WorkflowStatus.COMPLETED


# ---------------------------------------------------------------------------
# 13.3 — Workflow cancellation
# ---------------------------------------------------------------------------

class TestWorkflowCancellation:
    """Tests for cancel_workflow."""

    @pytest.mark.asyncio
    async def test_cancel_non_running_returns_false(self, legatus: Legatus):
        result = await legatus.cancel_workflow("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_running_workflow(
        self, legatus: Legatus, state_manager: StateManager, event_bus: EventBus
    ):
        """Start a slow workflow, cancel it, and verify status."""
        executor = _make_executor(state_manager, event_bus)

        async def slow_step(*args, **kwargs):
            await asyncio.sleep(10)
            return {"result": "done"}

        executor.execute_step = slow_step

        wf = _make_workflow()

        # Run workflow in background
        task = asyncio.create_task(
            legatus.execute_workflow(wf, executor=executor, state_manager=state_manager)
        )
        # Give it a moment to start
        await asyncio.sleep(0.05)

        cancelled = await legatus.cancel_workflow("wf-1", executor=executor)
        assert cancelled is True
        assert legatus.get_status("wf-1") == WorkflowStatus.CANCELLED

        # Let the task finish
        result = await task
        assert result.status == WorkflowStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_emits_cancelled_event(
        self, legatus: Legatus, state_manager: StateManager, event_bus: EventBus
    ):
        captured: List[Event] = []

        async def capture(event: Event) -> None:
            captured.append(event)

        event_bus.subscribe(EventType.WORKFLOW_CANCELLED, capture)

        # Manually set a workflow as running so we can cancel it
        legatus._workflow_statuses["wf-1"] = WorkflowStatus.RUNNING

        await legatus.cancel_workflow("wf-1")
        await asyncio.sleep(0.05)

        assert len(captured) == 1
        assert captured[0].data["workflow_id"] == "wf-1"
        assert captured[0].data["reason"] == "user_requested"


# ---------------------------------------------------------------------------
# 13.4 — Timeout enforcement
# ---------------------------------------------------------------------------

class TestWorkflowTimeout:
    """Tests for workflow timeout enforcement."""

    @pytest.mark.asyncio
    async def test_timeout_cancels_workflow(
        self, legatus: Legatus, state_manager: StateManager, event_bus: EventBus
    ):
        executor = _make_executor(state_manager, event_bus)

        async def slow_step(*args, **kwargs):
            await asyncio.sleep(10)
            return {"result": "done"}

        executor.execute_step = slow_step

        wf = _make_workflow(timeout=0.1)
        result = await legatus.execute_workflow(wf, executor=executor, state_manager=state_manager)

        assert result.status == WorkflowStatus.CANCELLED
        assert legatus.get_status("wf-1") == WorkflowStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_timeout_emits_cancelled_event_with_reason(
        self, legatus: Legatus, state_manager: StateManager, event_bus: EventBus
    ):
        captured: List[Event] = []

        async def capture(event: Event) -> None:
            captured.append(event)

        event_bus.subscribe(EventType.WORKFLOW_CANCELLED, capture)

        executor = _make_executor(state_manager, event_bus)

        async def slow_step(*args, **kwargs):
            await asyncio.sleep(10)
            return {}

        executor.execute_step = slow_step

        wf = _make_workflow(timeout=0.1)
        await legatus.execute_workflow(wf, executor=executor, state_manager=state_manager)
        await asyncio.sleep(0.05)

        cancelled_events = [
            e for e in captured if e.data.get("reason") == "timeout"
        ]
        assert len(cancelled_events) == 1

    @pytest.mark.asyncio
    async def test_no_timeout_when_within_limit(
        self, legatus: Legatus, state_manager: StateManager, event_bus: EventBus
    ):
        executor = _make_executor(state_manager, event_bus)
        wf = _make_workflow(timeout=10.0)
        result = await legatus.execute_workflow(wf, executor=executor, state_manager=state_manager)
        assert result.status == WorkflowStatus.COMPLETED


# ---------------------------------------------------------------------------
# Rate Limiting Integration
# ---------------------------------------------------------------------------

class TestLegatusRateLimiting:

    @pytest.mark.asyncio
    async def test_configure_rate_limit(self, legatus: Legatus):
        from agentlegatus.security.rate_limiter import RateLimitConfig

        legatus.configure_rate_limit("wf-1", RateLimitConfig(max_requests=5, window_seconds=60))
        # No error means success; internal state is opaque but we can verify
        # via subsequent execute calls.

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_excess_executions(
        self, legatus: Legatus, state_manager: StateManager, event_bus: EventBus
    ):
        from agentlegatus.security.rate_limiter import RateLimitConfig, RateLimitExceededError

        legatus.configure_rate_limit("wf-1", RateLimitConfig(max_requests=2, window_seconds=60))
        executor = _make_executor(state_manager, event_bus)
        wf = _make_workflow(workflow_id="wf-1")

        # First two should succeed
        r1 = await legatus.execute_workflow(wf, executor=executor, state_manager=state_manager)
        assert r1.status == WorkflowStatus.COMPLETED
        r2 = await legatus.execute_workflow(wf, executor=executor, state_manager=state_manager)
        assert r2.status == WorkflowStatus.COMPLETED

        # Third should be rejected
        with pytest.raises(RateLimitExceededError, match="wf-1"):
            await legatus.execute_workflow(wf, executor=executor, state_manager=state_manager)

    @pytest.mark.asyncio
    async def test_unconfigured_workflow_not_rate_limited(
        self, legatus: Legatus, state_manager: StateManager, event_bus: EventBus
    ):
        executor = _make_executor(state_manager, event_bus)
        wf = _make_workflow(workflow_id="wf-no-limit")

        # Should succeed many times without any rate limit configured
        for _ in range(10):
            result = await legatus.execute_workflow(wf, executor=executor, state_manager=state_manager)
            assert result.status == WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_rate_limits_are_per_workflow(
        self, legatus: Legatus, state_manager: StateManager, event_bus: EventBus
    ):
        from agentlegatus.security.rate_limiter import RateLimitConfig, RateLimitExceededError

        legatus.configure_rate_limit("wf-a", RateLimitConfig(max_requests=1, window_seconds=60))
        legatus.configure_rate_limit("wf-b", RateLimitConfig(max_requests=1, window_seconds=60))
        executor = _make_executor(state_manager, event_bus)

        wf_a = _make_workflow(workflow_id="wf-a")
        wf_b = _make_workflow(workflow_id="wf-b")

        # Both should succeed once
        r1 = await legatus.execute_workflow(wf_a, executor=executor, state_manager=state_manager)
        assert r1.status == WorkflowStatus.COMPLETED
        r2 = await legatus.execute_workflow(wf_b, executor=executor, state_manager=state_manager)
        assert r2.status == WorkflowStatus.COMPLETED

        # Both should now be blocked
        with pytest.raises(RateLimitExceededError, match="wf-a"):
            await legatus.execute_workflow(wf_a, executor=executor, state_manager=state_manager)
        with pytest.raises(RateLimitExceededError, match="wf-b"):
            await legatus.execute_workflow(wf_b, executor=executor, state_manager=state_manager)
