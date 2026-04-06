"""Integration test: end-to-end workflow execution.

Tests the complete workflow lifecycle from Legatus down through
Centurion → WorkflowExecutor → MockProvider, verifying state
persistence and event emission along the way.

Note: execute_step is not yet implemented (task 11.1), so we patch it
to delegate to the real MockProvider, keeping the rest of the stack
fully integrated.
"""

import asyncio
from datetime import datetime

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
from agentlegatus.providers.mock import MockProvider
from agentlegatus.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _EventCollector:
    """Subscribe to event types and collect emitted events."""

    def __init__(self, event_bus: EventBus, *event_types: EventType):
        self.events: list[Event] = []
        for et in event_types:
            event_bus.subscribe(et, self._handler)

    async def _handler(self, event: Event):
        self.events.append(event)

    def of_type(self, et: EventType) -> list[Event]:
        return [e for e in self.events if e.event_type == et]


def _make_workflow_def(
    workflow_id: str = "integ-wf-1",
    strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
    steps: list[WorkflowStep] | None = None,
    initial_state: dict | None = None,
    timeout: float | None = None,
) -> WorkflowDefinition:
    if steps is None:
        steps = [
            WorkflowStep(step_id="s1", step_type="agent", config={"task": "greet"}),
            WorkflowStep(
                step_id="s2",
                step_type="agent",
                config={"task": "farewell"},
                depends_on=["s1"],
            ),
        ]
    return WorkflowDefinition(
        workflow_id=workflow_id,
        name="Integration Test Workflow",
        description="End-to-end integration test",
        version="1.0.0",
        provider="mock",
        steps=steps,
        initial_state=initial_state or {},
        execution_strategy=strategy,
        timeout=timeout,
    )


def _patch_execute_step(executor: WorkflowExecutor, provider: MockProvider):
    """Patch execute_step to delegate to the real MockProvider.

    Task 11.1 (execute_step) is a stub, so we wire it to the provider
    manually. Everything else in the stack remains real.
    """

    async def _execute_step(step, context):
        agent = await provider.create_agent(
            step.config.get("agent", {"agent_id": step.step_id})
        )
        return await provider.execute_agent(agent, step.config, state=context)

    executor.execute_step = _execute_step


async def _build_stack(
    workflow_def: WorkflowDefinition | None = None,
) -> tuple[Legatus, WorkflowExecutor, StateManager, EventBus, MockProvider]:
    """Wire up the full Legatus → Centurion → Executor stack."""
    workflow_def = workflow_def or _make_workflow_def()

    event_bus = EventBus()
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend=backend, event_bus=event_bus)
    provider = MockProvider(config={})
    tool_registry = ToolRegistry()

    executor = WorkflowExecutor(
        provider=provider,
        state_manager=state_manager,
        tool_registry=tool_registry,
        event_bus=event_bus,
    )
    _patch_execute_step(executor, provider)

    legatus = Legatus(config={"provider": "mock"}, event_bus=event_bus)
    centurion = Centurion(
        name="main",
        strategy=workflow_def.execution_strategy,
        event_bus=event_bus,
    )
    await legatus.add_centurion(centurion)

    return legatus, executor, state_manager, event_bus, provider


# ===================================================================
# Tests
# ===================================================================


class TestEndToEndWorkflowExecution:
    """Full lifecycle: define → execute → verify result, state, events."""

    @pytest.mark.asyncio
    async def test_sequential_workflow_completes(self):
        """A two-step sequential workflow should complete with COMPLETED status."""
        wf = _make_workflow_def()
        legatus, executor, sm, eb, provider = await _build_stack(wf)

        result = await legatus.execute_workflow(
            wf, initial_state={}, executor=executor, state_manager=sm,
        )

        assert result.status == WorkflowStatus.COMPLETED
        assert result.output is not None
        assert result.execution_time > 0
        assert provider.execution_count == 2

    @pytest.mark.asyncio
    async def test_workflow_events_lifecycle(self):
        """WorkflowStarted and WorkflowCompleted events must be emitted."""
        wf = _make_workflow_def()
        legatus, executor, sm, eb, provider = await _build_stack(wf)
        collector = _EventCollector(
            eb,
            EventType.WORKFLOW_STARTED,
            EventType.WORKFLOW_COMPLETED,
            EventType.STEP_STARTED,
            EventType.STEP_COMPLETED,
        )

        result = await legatus.execute_workflow(
            wf, initial_state={}, executor=executor, state_manager=sm,
        )

        assert result.status == WorkflowStatus.COMPLETED

        started = collector.of_type(EventType.WORKFLOW_STARTED)
        completed = collector.of_type(EventType.WORKFLOW_COMPLETED)
        assert len(started) == 1
        assert len(completed) == 1
        assert started[0].data["workflow_id"] == "integ-wf-1"

        # Two steps → two StepStarted + two StepCompleted
        assert len(collector.of_type(EventType.STEP_STARTED)) == 2
        assert len(collector.of_type(EventType.STEP_COMPLETED)) == 2

    @pytest.mark.asyncio
    async def test_state_persisted_across_steps(self):
        """State set during execution should be readable after completion."""
        wf = _make_workflow_def(initial_state={"counter": 0})
        legatus, executor, sm, eb, provider = await _build_stack(wf)

        result = await legatus.execute_workflow(
            wf, initial_state={"counter": 0}, executor=executor, state_manager=sm,
        )

        assert result.status == WorkflowStatus.COMPLETED

        # Centurion stores step results in WORKFLOW scope as step_<id>_result
        s1_result = await sm.get("step_s1_result", scope=StateScope.WORKFLOW)
        s2_result = await sm.get("step_s2_result", scope=StateScope.WORKFLOW)
        assert s1_result is not None
        assert s2_result is not None

        # Step statuses should be marked completed
        s1_status = await sm.get("step_s1_status", scope=StateScope.WORKFLOW)
        assert s1_status == "completed"

    @pytest.mark.asyncio
    async def test_workflow_result_includes_metrics(self):
        """WorkflowResult.metrics should contain duration and trace info."""
        wf = _make_workflow_def()
        legatus, executor, sm, eb, _ = await _build_stack(wf)

        result = await legatus.execute_workflow(
            wf, initial_state={}, executor=executor, state_manager=sm,
        )

        assert isinstance(result.metrics, dict)
        assert result.execution_time > 0
        # Legatus builds metrics with workflow_id and trace_id
        assert "workflow_id" in result.metrics
        assert "trace_id" in result.metrics

    @pytest.mark.asyncio
    async def test_single_step_workflow(self):
        """A workflow with a single step should complete normally."""
        wf = _make_workflow_def(
            steps=[WorkflowStep(step_id="only", step_type="agent", config={"task": "solo"})],
        )
        legatus, executor, sm, eb, provider = await _build_stack(wf)

        result = await legatus.execute_workflow(
            wf, initial_state={}, executor=executor, state_manager=sm,
        )

        assert result.status == WorkflowStatus.COMPLETED
        assert provider.execution_count == 1

    @pytest.mark.asyncio
    async def test_provider_call_log_records_all_executions(self):
        """MockProvider.call_log should record every create_agent + execute_agent."""
        wf = _make_workflow_def()
        legatus, executor, sm, eb, provider = await _build_stack(wf)

        await legatus.execute_workflow(
            wf, initial_state={}, executor=executor, state_manager=sm,
        )

        create_calls = [c for c in provider.call_log if c["action"] == "create_agent"]
        exec_calls = [c for c in provider.call_log if c["action"] == "execute_agent"]
        assert len(create_calls) == 2
        assert len(exec_calls) == 2
