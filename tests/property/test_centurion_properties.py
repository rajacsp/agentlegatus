"""Property-based tests for Centurion.

Property 9: Dependency Execution Order
Validates: Requirement 2.5 — WHEN a step has dependencies, THE Centurion
SHALL execute the step only after all dependencies complete successfully.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agentlegatus.core.event_bus import EventBus, EventType
from agentlegatus.core.executor import WorkflowExecutor
from agentlegatus.core.state import InMemoryStateBackend, StateManager, StateScope
from agentlegatus.core.workflow import (
    ExecutionStrategy,
    WorkflowDefinition,
    WorkflowStep,
)
from agentlegatus.hierarchy.centurion import Centurion


# ---------------------------------------------------------------------------
# Hypothesis strategies for generating random DAGs
# ---------------------------------------------------------------------------

@st.composite
def dag_steps(draw, min_steps=2, max_steps=15):
    """Generate a list of WorkflowSteps that form a valid DAG.

    Nodes are numbered 0..n-1.  Each node may only depend on nodes with
    a *smaller* index, which guarantees acyclicity by construction.
    """
    n = draw(st.integers(min_value=min_steps, max_value=max_steps))
    steps: List[WorkflowStep] = []

    for i in range(n):
        # Each step can depend on any subset of earlier steps
        possible_deps = list(range(i))
        deps = draw(
            st.lists(
                st.sampled_from(possible_deps) if possible_deps else st.nothing(),
                max_size=min(len(possible_deps), 5),
                unique=True,
            )
        )
        steps.append(
            WorkflowStep(
                step_id=f"step_{i}",
                step_type="agent",
                config={},
                depends_on=[f"step_{d}" for d in deps],
            )
        )

    return steps


# ---------------------------------------------------------------------------
# Property 9: Dependency Execution Order
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@given(steps=dag_steps())
@settings(max_examples=10, deadline=5000)
async def test_property_9_dependency_execution_order(steps: List[WorkflowStep]):
    """
    Property 9: Dependency Execution Order

    For any valid DAG of workflow steps, the execution plan produced by
    build_execution_plan MUST place every step *after* all of its
    dependencies.

    Formally: for every step S in the plan, if S depends on D, then
    index(D) < index(S).

    Validates: Requirement 2.5
    """
    centurion = Centurion(
        name="test",
        strategy=ExecutionStrategy.SEQUENTIAL,
        event_bus=EventBus(),
    )

    plan = centurion.build_execution_plan(steps)

    # Build a position map: step_id -> index in the plan
    position = {s.step_id: idx for idx, s in enumerate(plan)}

    for step in plan:
        for dep in step.depends_on:
            assert position[dep] < position[step.step_id], (
                f"Step '{step.step_id}' appears at position {position[step.step_id]} "
                f"but its dependency '{dep}' appears at position {position[dep]}. "
                f"Dependencies must come first."
            )


@pytest.mark.asyncio
@given(steps=dag_steps())
@settings(max_examples=10, deadline=5000)
async def test_property_9_plan_preserves_all_steps(steps: List[WorkflowStep]):
    """
    The execution plan must contain exactly the same steps as the input
    (no steps lost or duplicated).
    """
    centurion = Centurion(
        name="test",
        strategy=ExecutionStrategy.SEQUENTIAL,
        event_bus=EventBus(),
    )

    plan = centurion.build_execution_plan(steps)

    input_ids = sorted(s.step_id for s in steps)
    plan_ids = sorted(s.step_id for s in plan)

    assert input_ids == plan_ids, (
        f"Plan step IDs {plan_ids} do not match input step IDs {input_ids}"
    )


@pytest.mark.asyncio
@given(steps=dag_steps())
@settings(max_examples=10, deadline=5000)
async def test_property_9_sequential_execution_respects_order(
    steps: List[WorkflowStep],
):
    """
    When executing sequentially, steps must actually run in an order
    that respects dependencies.  We record the execution order via a
    mock executor and verify the same dependency invariant holds.

    Validates: Requirement 2.5
    """
    event_bus = EventBus()
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend=backend, event_bus=event_bus)

    centurion = Centurion(
        name="test",
        strategy=ExecutionStrategy.SEQUENTIAL,
        event_bus=event_bus,
    )

    # Track execution order
    execution_order: List[str] = []

    # Create a mock executor whose execute_step records the step id
    executor = AsyncMock(spec=WorkflowExecutor)

    async def mock_execute_step(step, context):
        execution_order.append(step.step_id)
        return f"result_{step.step_id}"

    executor.execute_step = AsyncMock(side_effect=mock_execute_step)

    plan = centurion.build_execution_plan(steps)
    await centurion.execute_sequential(plan, executor, state_manager)

    # Verify every step executed after its dependencies
    position = {sid: idx for idx, sid in enumerate(execution_order)}

    for step in steps:
        assert step.step_id in position, (
            f"Step '{step.step_id}' was never executed"
        )
        for dep in step.depends_on:
            assert position[dep] < position[step.step_id], (
                f"Step '{step.step_id}' executed at position {position[step.step_id]} "
                f"but dependency '{dep}' executed at position {position[dep]}"
            )


@pytest.mark.asyncio
async def test_centurion_rejects_cyclic_dependencies():
    """build_execution_plan must raise ValueError for cyclic graphs."""
    centurion = Centurion(
        name="test",
        strategy=ExecutionStrategy.SEQUENTIAL,
        event_bus=EventBus(),
    )

    cyclic_steps = [
        WorkflowStep(step_id="a", step_type="agent", config={}, depends_on=["b"]),
        WorkflowStep(step_id="b", step_type="agent", config={}, depends_on=["a"]),
    ]

    with pytest.raises(ValueError, match="cycle"):
        centurion.build_execution_plan(cyclic_steps)


@pytest.mark.asyncio
async def test_centurion_rejects_missing_dependency():
    """build_execution_plan must raise ValueError for missing dependency refs."""
    centurion = Centurion(
        name="test",
        strategy=ExecutionStrategy.SEQUENTIAL,
        event_bus=EventBus(),
    )

    steps = [
        WorkflowStep(
            step_id="a", step_type="agent", config={}, depends_on=["nonexistent"]
        ),
    ]

    with pytest.raises(ValueError, match="non-existent"):
        centurion.build_execution_plan(steps)
