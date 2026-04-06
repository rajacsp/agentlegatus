"""Property-based tests for Legatus.

Property 1: Workflow Execution Completeness
Property 2: Workflow Event Lifecycle
Validates: Requirements 1.1-1.6
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

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
from agentlegatus.hierarchy.legatus import Legatus


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def workflow_steps(draw, min_steps=1, max_steps=8):
    """Generate a list of WorkflowSteps forming a valid DAG."""
    n = draw(st.integers(min_value=min_steps, max_value=max_steps))
    steps: List[WorkflowStep] = []
    for i in range(n):
        possible_deps = list(range(i))
        deps = draw(
            st.lists(
                st.sampled_from(possible_deps) if possible_deps else st.nothing(),
                max_size=min(len(possible_deps), 3),
                unique=True,
            )
        )
        steps.append(
            WorkflowStep(
                step_id=f"step_{i}",
                step_type="agent",
                config={"task": f"task_{i}"},
                depends_on=[f"step_{d}" for d in deps],
            )
        )
    return steps


@st.composite
def workflow_definition(draw, strategy=None):
    """Generate a valid WorkflowDefinition with random steps."""
    steps = draw(workflow_steps())
    strat = strategy or draw(st.sampled_from([ExecutionStrategy.SEQUENTIAL]))
    wf_id = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
        min_size=1,
        max_size=20,
    ))
    name = draw(st.text(min_size=1, max_size=30))
    return WorkflowDefinition(
        workflow_id=wf_id,
        name=name,
        description="test workflow",
        version="1.0",
        provider="mock",
        steps=steps,
        execution_strategy=strat,
    )


@st.composite
def initial_state_strategy(draw):
    """Generate a random initial state dictionary."""
    return draw(
        st.dictionaries(
            st.text(
                alphabet=st.characters(whitelist_categories=("L",)),
                min_size=1,
                max_size=10,
            ),
            st.one_of(st.integers(), st.text(min_size=1, max_size=20)),
            min_size=0,
            max_size=5,
        )
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_executor(state_manager: StateManager, event_bus: EventBus) -> WorkflowExecutor:
    """Create a mock WorkflowExecutor that records step executions."""
    executor = AsyncMock(spec=WorkflowExecutor)

    async def mock_execute_step(step, context):
        return {"step_id": step.step_id, "status": "done"}

    executor.execute_step = AsyncMock(side_effect=mock_execute_step)
    return executor


# ---------------------------------------------------------------------------
# Property 1: Workflow Execution Completeness
#
# For any valid WorkflowDefinition, executing it through Legatus MUST
# return a WorkflowResult whose status is one of COMPLETED, FAILED, or
# CANCELLED — never PENDING or RUNNING.  Additionally, the result MUST
# include execution metrics with non-negative execution_time.
#
# Validates: Requirements 1.1, 1.3, 1.4, 1.5
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@given(wf=workflow_definition(), init_state=initial_state_strategy())
@settings(max_examples=10, deadline=5000)
async def test_property_1_workflow_execution_completeness(
    wf: WorkflowDefinition,
    init_state: Dict[str, Any],
):
    """
    Property 1: Workflow Execution Completeness

    For any valid WorkflowDefinition and initial state, Legatus.execute_workflow
    MUST return a WorkflowResult with a terminal status (COMPLETED, FAILED, or
    CANCELLED), non-negative execution_time, and a non-null metrics dict.

    Validates: Requirements 1.1, 1.3, 1.4, 1.5, 1.8
    """
    event_bus = EventBus()
    legatus = Legatus(config={}, event_bus=event_bus)

    backend = InMemoryStateBackend()
    state_manager = StateManager(backend=backend, event_bus=event_bus)
    executor = _make_executor(state_manager, event_bus)

    result = await legatus.execute_workflow(
        workflow_def=wf,
        initial_state=init_state,
        executor=executor,
        state_manager=state_manager,
    )

    # Status must be terminal
    assert result.status in (
        WorkflowStatus.COMPLETED,
        WorkflowStatus.FAILED,
        WorkflowStatus.CANCELLED,
    ), f"Workflow ended with non-terminal status: {result.status}"

    # execution_time must be non-negative
    assert result.execution_time >= 0, (
        f"execution_time must be non-negative, got {result.execution_time}"
    )

    # metrics must be present
    assert isinstance(result.metrics, dict), "metrics must be a dict"

    # A completed workflow must have non-null output
    if result.status == WorkflowStatus.COMPLETED:
        assert result.output is not None, (
            "Completed workflow must have non-null output"
        )

    # A failed workflow must have non-null error
    if result.status == WorkflowStatus.FAILED:
        assert result.error is not None, (
            "Failed workflow must have non-null error"
        )


@pytest.mark.asyncio
@given(wf=workflow_definition())
@settings(max_examples=10, deadline=5000)
async def test_property_1_failed_execution_returns_failed(
    wf: WorkflowDefinition,
):
    """
    Property 1 (failure path): When the underlying executor raises an
    exception, Legatus MUST return a WorkflowResult with status FAILED
    and the error field set.

    Validates: Requirements 1.4
    """
    event_bus = EventBus()
    legatus = Legatus(config={}, event_bus=event_bus)

    backend = InMemoryStateBackend()
    state_manager = StateManager(backend=backend, event_bus=event_bus)
    executor = AsyncMock(spec=WorkflowExecutor)
    executor.execute_step = AsyncMock(side_effect=RuntimeError("boom"))

    result = await legatus.execute_workflow(
        workflow_def=wf,
        executor=executor,
        state_manager=state_manager,
    )

    assert result.status == WorkflowStatus.FAILED, (
        f"Expected FAILED status, got {result.status}"
    )
    assert result.error is not None, "Failed result must carry the error"
    assert result.execution_time >= 0


# ---------------------------------------------------------------------------
# Property 2: Workflow Event Lifecycle
#
# For any workflow execution, the EventBus MUST contain a WorkflowStarted
# event.  If the workflow completes successfully, a WorkflowCompleted event
# MUST follow.  If it fails, a WorkflowFailed event MUST follow.  The
# started event MUST appear before the terminal event in history.
#
# Validates: Requirements 1.2, 1.3, 1.6
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@given(wf=workflow_definition(), init_state=initial_state_strategy())
@settings(max_examples=10, deadline=5000)
async def test_property_2_workflow_event_lifecycle_success(
    wf: WorkflowDefinition,
    init_state: Dict[str, Any],
):
    """
    Property 2 (success path): Workflow Event Lifecycle

    For any successful workflow execution, the event history MUST contain
    exactly one WORKFLOW_STARTED followed by exactly one WORKFLOW_COMPLETED,
    and the started event MUST precede the completed event.

    Validates: Requirements 1.2, 1.3, 1.6
    """
    event_bus = EventBus()
    legatus = Legatus(config={}, event_bus=event_bus)

    backend = InMemoryStateBackend()
    state_manager = StateManager(backend=backend, event_bus=event_bus)
    executor = _make_executor(state_manager, event_bus)

    result = await legatus.execute_workflow(
        workflow_def=wf,
        initial_state=init_state,
        executor=executor,
        state_manager=state_manager,
    )

    # Only check lifecycle for completed workflows
    assume(result.status == WorkflowStatus.COMPLETED)

    history = event_bus.get_event_history(limit=1000)

    started_events = [
        e for e in history if e.event_type == EventType.WORKFLOW_STARTED
    ]
    completed_events = [
        e for e in history if e.event_type == EventType.WORKFLOW_COMPLETED
    ]

    assert len(started_events) >= 1, "Must have at least one WORKFLOW_STARTED event"
    assert len(completed_events) >= 1, "Must have at least one WORKFLOW_COMPLETED event"

    # Find the events for this workflow
    wf_started = [
        e for e in started_events
        if e.data.get("workflow_id") == wf.workflow_id
    ]
    wf_completed = [
        e for e in completed_events
        if e.data.get("workflow_id") == wf.workflow_id
    ]

    assert len(wf_started) == 1, (
        f"Expected exactly 1 WORKFLOW_STARTED for '{wf.workflow_id}', "
        f"got {len(wf_started)}"
    )
    assert len(wf_completed) == 1, (
        f"Expected exactly 1 WORKFLOW_COMPLETED for '{wf.workflow_id}', "
        f"got {len(wf_completed)}"
    )

    # Started must appear before completed in history
    started_idx = history.index(wf_started[0])
    completed_idx = history.index(wf_completed[0])
    assert started_idx < completed_idx, (
        "WORKFLOW_STARTED must appear before WORKFLOW_COMPLETED in history"
    )

    # Both events must share the same trace_id
    assert wf_started[0].trace_id is not None, "Started event must have trace_id"
    assert wf_started[0].trace_id == wf_completed[0].trace_id, (
        "Started and completed events must share the same trace_id"
    )


@pytest.mark.asyncio
@given(wf=workflow_definition())
@settings(max_examples=10, deadline=5000)
async def test_property_2_workflow_event_lifecycle_failure(
    wf: WorkflowDefinition,
):
    """
    Property 2 (failure path): Workflow Event Lifecycle

    For any failed workflow execution, the event history MUST contain
    exactly one WORKFLOW_STARTED followed by exactly one WORKFLOW_FAILED,
    and the started event MUST precede the failed event.

    Validates: Requirements 1.2, 1.4, 1.6
    """
    event_bus = EventBus()
    legatus = Legatus(config={}, event_bus=event_bus)

    backend = InMemoryStateBackend()
    state_manager = StateManager(backend=backend, event_bus=event_bus)
    executor = AsyncMock(spec=WorkflowExecutor)
    executor.execute_step = AsyncMock(side_effect=RuntimeError("intentional"))

    result = await legatus.execute_workflow(
        workflow_def=wf,
        executor=executor,
        state_manager=state_manager,
    )

    assert result.status == WorkflowStatus.FAILED

    history = event_bus.get_event_history(limit=1000)

    wf_started = [
        e for e in history
        if e.event_type == EventType.WORKFLOW_STARTED
        and e.data.get("workflow_id") == wf.workflow_id
    ]
    wf_failed = [
        e for e in history
        if e.event_type == EventType.WORKFLOW_FAILED
        and e.data.get("workflow_id") == wf.workflow_id
    ]

    assert len(wf_started) == 1, "Must have exactly 1 WORKFLOW_STARTED"
    assert len(wf_failed) == 1, "Must have exactly 1 WORKFLOW_FAILED"

    started_idx = history.index(wf_started[0])
    failed_idx = history.index(wf_failed[0])
    assert started_idx < failed_idx, (
        "WORKFLOW_STARTED must appear before WORKFLOW_FAILED in history"
    )

    # Both events must share the same trace_id
    assert wf_started[0].trace_id == wf_failed[0].trace_id, (
        "Started and failed events must share the same trace_id"
    )

    # Failed event must contain error details
    assert "error" in wf_failed[0].data, "WORKFLOW_FAILED must contain error details"
    assert "error_type" in wf_failed[0].data, "WORKFLOW_FAILED must contain error_type"


@pytest.mark.asyncio
@given(wf=workflow_definition())
@settings(max_examples=10, deadline=5000)
async def test_property_2_no_completed_event_on_failure(
    wf: WorkflowDefinition,
):
    """
    Property 2 (exclusivity): When a workflow fails, there MUST NOT be
    a WORKFLOW_COMPLETED event for that workflow.

    Validates: Requirements 1.6
    """
    event_bus = EventBus()
    legatus = Legatus(config={}, event_bus=event_bus)

    backend = InMemoryStateBackend()
    state_manager = StateManager(backend=backend, event_bus=event_bus)
    executor = AsyncMock(spec=WorkflowExecutor)
    executor.execute_step = AsyncMock(side_effect=ValueError("fail"))

    result = await legatus.execute_workflow(
        workflow_def=wf,
        executor=executor,
        state_manager=state_manager,
    )

    assert result.status == WorkflowStatus.FAILED

    history = event_bus.get_event_history(limit=1000)
    wf_completed = [
        e for e in history
        if e.event_type == EventType.WORKFLOW_COMPLETED
        and e.data.get("workflow_id") == wf.workflow_id
    ]

    assert len(wf_completed) == 0, (
        "A failed workflow must NOT emit a WORKFLOW_COMPLETED event"
    )


@pytest.mark.asyncio
@given(wf=workflow_definition())
@settings(max_examples=10, deadline=5000)
async def test_property_2_no_failed_event_on_success(
    wf: WorkflowDefinition,
):
    """
    Property 2 (exclusivity): When a workflow succeeds, there MUST NOT be
    a WORKFLOW_FAILED event for that workflow.

    Validates: Requirements 1.6
    """
    event_bus = EventBus()
    legatus = Legatus(config={}, event_bus=event_bus)

    backend = InMemoryStateBackend()
    state_manager = StateManager(backend=backend, event_bus=event_bus)
    executor = _make_executor(state_manager, event_bus)

    result = await legatus.execute_workflow(
        workflow_def=wf,
        executor=executor,
        state_manager=state_manager,
    )

    assume(result.status == WorkflowStatus.COMPLETED)

    history = event_bus.get_event_history(limit=1000)
    wf_failed = [
        e for e in history
        if e.event_type == EventType.WORKFLOW_FAILED
        and e.data.get("workflow_id") == wf.workflow_id
    ]

    assert len(wf_failed) == 0, (
        "A successful workflow must NOT emit a WORKFLOW_FAILED event"
    )


@pytest.mark.asyncio
@given(wf=workflow_definition())
@settings(max_examples=10, deadline=5000)
async def test_property_1_status_reflects_result(
    wf: WorkflowDefinition,
):
    """
    Property 1 (status consistency): After execution, Legatus.get_status()
    MUST return the same status as the WorkflowResult.

    Validates: Requirements 1.1, 1.3, 1.4
    """
    event_bus = EventBus()
    legatus = Legatus(config={}, event_bus=event_bus)

    backend = InMemoryStateBackend()
    state_manager = StateManager(backend=backend, event_bus=event_bus)
    executor = _make_executor(state_manager, event_bus)

    result = await legatus.execute_workflow(
        workflow_def=wf,
        executor=executor,
        state_manager=state_manager,
    )

    stored_status = legatus.get_status(wf.workflow_id)
    assert stored_status == result.status, (
        f"get_status() returned {stored_status} but result had {result.status}"
    )
