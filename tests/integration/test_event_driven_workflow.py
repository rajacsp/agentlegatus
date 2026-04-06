"""Integration test: event-driven workflow.

Registers event handlers before workflow execution, then verifies
event ordering, correlation IDs, and that all expected events fire.
"""

import pytest

from agentlegatus.core.event_bus import Event, EventBus, EventType
from agentlegatus.core.executor import WorkflowExecutor
from agentlegatus.core.state import InMemoryStateBackend, StateManager, StateScope
from agentlegatus.core.workflow import (
    ExecutionStrategy,
    WorkflowDefinition,
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
    """Collects all events for specified types."""

    def __init__(self, event_bus: EventBus, *event_types: EventType):
        self.events: list[Event] = []
        for et in event_types:
            event_bus.subscribe(et, self._handler)

    async def _handler(self, event: Event):
        self.events.append(event)

    def of_type(self, et: EventType) -> list[Event]:
        return [e for e in self.events if e.event_type == et]

    @property
    def types_in_order(self) -> list[EventType]:
        return [e.event_type for e in self.events]


def _patch_execute_step(executor: WorkflowExecutor, provider: MockProvider):
    async def _execute_step(step, context):
        agent = await provider.create_agent(
            step.config.get("agent", {"agent_id": step.step_id})
        )
        return await provider.execute_agent(agent, step.config, state=context)

    executor.execute_step = _execute_step


def _make_workflow(
    num_steps: int = 3,
    strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
) -> WorkflowDefinition:
    steps = []
    for i in range(1, num_steps + 1):
        deps = [f"s{i - 1}"] if i > 1 else []
        steps.append(
            WorkflowStep(
                step_id=f"s{i}",
                step_type="agent",
                config={"task": f"task_{i}"},
                depends_on=deps,
            )
        )
    return WorkflowDefinition(
        workflow_id="event-wf",
        name="Event Integration Workflow",
        description="Tests event emission",
        version="1.0.0",
        provider="mock",
        steps=steps,
        execution_strategy=strategy,
    )


async def _run_workflow(
    wf: WorkflowDefinition,
    event_bus: EventBus,
):
    """Build the full stack and execute the workflow."""
    backend = InMemoryStateBackend()
    sm = StateManager(backend=backend, event_bus=event_bus)
    provider = MockProvider(config={})
    tr = ToolRegistry()
    executor = WorkflowExecutor(
        provider=provider, state_manager=sm, tool_registry=tr, event_bus=event_bus,
    )
    _patch_execute_step(executor, provider)

    legatus = Legatus(config={"provider": "mock"}, event_bus=event_bus)
    centurion = Centurion(name="main", strategy=wf.execution_strategy, event_bus=event_bus)
    await legatus.add_centurion(centurion)

    return await legatus.execute_workflow(wf, initial_state={}, executor=executor, state_manager=sm)


# ===================================================================
# Tests
# ===================================================================


class TestEventDrivenWorkflow:

    @pytest.mark.asyncio
    async def test_all_lifecycle_events_emitted(self):
        """WorkflowStarted, StepStarted/Completed per step, WorkflowCompleted."""
        eb = EventBus()
        collector = _EventCollector(
            eb,
            EventType.WORKFLOW_STARTED,
            EventType.WORKFLOW_COMPLETED,
            EventType.STEP_STARTED,
            EventType.STEP_COMPLETED,
        )

        wf = _make_workflow(num_steps=3)
        result = await _run_workflow(wf, eb)

        assert result.status == WorkflowStatus.COMPLETED
        assert len(collector.of_type(EventType.WORKFLOW_STARTED)) == 1
        assert len(collector.of_type(EventType.WORKFLOW_COMPLETED)) == 1
        assert len(collector.of_type(EventType.STEP_STARTED)) == 3
        assert len(collector.of_type(EventType.STEP_COMPLETED)) == 3

    @pytest.mark.asyncio
    async def test_event_ordering(self):
        """Events should follow: WF_STARTED → (STEP_STARTED → STEP_COMPLETED)* → WF_COMPLETED."""
        eb = EventBus()
        collector = _EventCollector(
            eb,
            EventType.WORKFLOW_STARTED,
            EventType.WORKFLOW_COMPLETED,
            EventType.STEP_STARTED,
            EventType.STEP_COMPLETED,
        )

        wf = _make_workflow(num_steps=2)
        await _run_workflow(wf, eb)

        types = collector.types_in_order
        assert types[0] == EventType.WORKFLOW_STARTED
        assert types[-1] == EventType.WORKFLOW_COMPLETED

        # Between start and end, steps should alternate started/completed
        step_events = types[1:-1]
        for i in range(0, len(step_events), 2):
            assert step_events[i] == EventType.STEP_STARTED
            assert step_events[i + 1] == EventType.STEP_COMPLETED

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self):
        """WorkflowStarted and WorkflowCompleted should share the same trace_id."""
        eb = EventBus()
        collector = _EventCollector(
            eb,
            EventType.WORKFLOW_STARTED,
            EventType.WORKFLOW_COMPLETED,
        )

        wf = _make_workflow(num_steps=1)
        await _run_workflow(wf, eb)

        started = collector.of_type(EventType.WORKFLOW_STARTED)[0]
        completed = collector.of_type(EventType.WORKFLOW_COMPLETED)[0]
        assert started.trace_id is not None
        assert started.trace_id == completed.trace_id

    @pytest.mark.asyncio
    async def test_step_events_contain_step_id(self):
        """Each StepStarted/Completed event should carry the step_id in data."""
        eb = EventBus()
        collector = _EventCollector(eb, EventType.STEP_STARTED, EventType.STEP_COMPLETED)

        wf = _make_workflow(num_steps=2)
        await _run_workflow(wf, eb)

        started_ids = {e.data["step_id"] for e in collector.of_type(EventType.STEP_STARTED)}
        completed_ids = {e.data["step_id"] for e in collector.of_type(EventType.STEP_COMPLETED)}
        assert started_ids == {"s1", "s2"}
        assert completed_ids == {"s1", "s2"}

    @pytest.mark.asyncio
    async def test_event_history_accessible(self):
        """EventBus.get_event_history should return all emitted events."""
        eb = EventBus()
        wf = _make_workflow(num_steps=2)
        await _run_workflow(wf, eb)

        history = eb.get_event_history(limit=100)
        event_types = {e.event_type for e in history}
        assert EventType.WORKFLOW_STARTED in event_types
        assert EventType.WORKFLOW_COMPLETED in event_types

    @pytest.mark.asyncio
    async def test_custom_handler_receives_events(self):
        """A custom async handler should be invoked for subscribed events."""
        eb = EventBus()
        received = []

        async def my_handler(event: Event):
            received.append(event.data.get("workflow_id") or event.data.get("step_id"))

        eb.subscribe(EventType.WORKFLOW_STARTED, my_handler)
        eb.subscribe(EventType.STEP_STARTED, my_handler)

        wf = _make_workflow(num_steps=2)
        await _run_workflow(wf, eb)

        # Should have received workflow_id from WORKFLOW_STARTED + step_ids from STEP_STARTED
        assert "event-wf" in received
        assert "s1" in received
        assert "s2" in received
