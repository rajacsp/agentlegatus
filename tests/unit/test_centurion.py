"""Unit tests for Centurion workflow controller."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentlegatus.core.event_bus import EventBus, EventType
from agentlegatus.core.state import InMemoryStateBackend, StateManager, StateScope
from agentlegatus.core.workflow import (
    ExecutionStrategy,
    RetryPolicy,
    WorkflowDefinition,
    WorkflowStep,
)
from agentlegatus.hierarchy.centurion import Centurion
from agentlegatus.hierarchy.cohort import Cohort, CohortStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event_bus() -> EventBus:
    return EventBus()


def _make_state_manager(event_bus: EventBus | None = None) -> StateManager:
    return StateManager(
        backend=InMemoryStateBackend(),
        event_bus=event_bus,
    )


def _make_executor() -> MagicMock:
    """Return a mock WorkflowExecutor whose execute_step is async."""
    executor = MagicMock()
    executor.execute_step = AsyncMock(return_value={"output": "ok"})
    return executor


def _step(step_id: str, depends_on: list[str] | None = None, **config) -> WorkflowStep:
    return WorkflowStep(
        step_id=step_id,
        step_type="agent",
        config=config,
        depends_on=depends_on or [],
    )


def _workflow(
    steps: list[WorkflowStep],
    strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
    **metadata,
) -> WorkflowDefinition:
    return WorkflowDefinition(
        workflow_id="wf-test",
        name="Test Workflow",
        description="test",
        version="1.0",
        provider="mock",
        steps=steps,
        execution_strategy=strategy,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# 12.1 — Centurion creation & add_cohort
# ---------------------------------------------------------------------------

class TestCenturionCreation:
    def test_init(self):
        eb = _make_event_bus()
        c = Centurion("c1", ExecutionStrategy.SEQUENTIAL, eb)
        assert c.name == "c1"
        assert c.strategy == ExecutionStrategy.SEQUENTIAL
        assert c.event_bus is eb
        assert c._cohorts == {}

    def test_init_parallel(self):
        c = Centurion("c2", ExecutionStrategy.PARALLEL, _make_event_bus())
        assert c.strategy == ExecutionStrategy.PARALLEL

    def test_init_conditional(self):
        c = Centurion("c3", ExecutionStrategy.CONDITIONAL, _make_event_bus())
        assert c.strategy == ExecutionStrategy.CONDITIONAL

    @pytest.mark.asyncio
    async def test_add_cohort(self):
        c = Centurion("c1", ExecutionStrategy.SEQUENTIAL, _make_event_bus())
        cohort = Cohort("cohort-a", CohortStrategy.ROUND_ROBIN)
        await c.add_cohort(cohort)
        assert "cohort-a" in c._cohorts
        assert c._cohorts["cohort-a"] is cohort



# ---------------------------------------------------------------------------
# 12.2 — Execution plan building
# ---------------------------------------------------------------------------

class TestBuildExecutionPlan:
    def test_linear_chain(self):
        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, _make_event_bus())
        steps = [
            _step("a"),
            _step("b", ["a"]),
            _step("c", ["b"]),
        ]
        plan = c.build_execution_plan(steps)
        ids = [s.step_id for s in plan]
        assert ids.index("a") < ids.index("b") < ids.index("c")

    def test_diamond_dependency(self):
        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, _make_event_bus())
        steps = [
            _step("a"),
            _step("b", ["a"]),
            _step("c", ["a"]),
            _step("d", ["b", "c"]),
        ]
        plan = c.build_execution_plan(steps)
        ids = [s.step_id for s in plan]
        assert ids.index("a") < ids.index("b")
        assert ids.index("a") < ids.index("c")
        assert ids.index("b") < ids.index("d")
        assert ids.index("c") < ids.index("d")

    def test_no_dependencies(self):
        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, _make_event_bus())
        steps = [_step("a"), _step("b"), _step("c")]
        plan = c.build_execution_plan(steps)
        assert len(plan) == 3

    def test_cycle_raises(self):
        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, _make_event_bus())
        steps = [
            _step("a", ["c"]),
            _step("b", ["a"]),
            _step("c", ["b"]),
        ]
        with pytest.raises(ValueError, match="cycle"):
            c.build_execution_plan(steps)

    def test_missing_dependency_raises(self):
        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, _make_event_bus())
        steps = [_step("a", ["nonexistent"])]
        with pytest.raises(ValueError, match="non-existent"):
            c.build_execution_plan(steps)


# ---------------------------------------------------------------------------
# 12.3 — Sequential execution
# ---------------------------------------------------------------------------

class TestSequentialExecution:
    @pytest.mark.asyncio
    async def test_executes_all_steps_in_order(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        call_order: list[str] = []

        async def side_effect(step, ctx):
            call_order.append(step.step_id)
            return {"step": step.step_id}

        executor.execute_step.side_effect = side_effect

        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, eb)
        plan = [_step("a"), _step("b", ["a"]), _step("c", ["b"])]
        result = await c.execute_sequential(plan, executor, sm)

        assert call_order == ["a", "b", "c"]
        assert result[f"step_a_status"] == "completed"
        assert result[f"step_c_status"] == "completed"

    @pytest.mark.asyncio
    async def test_emits_step_events(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        events_received: list[EventType] = []

        async def handler(event):
            events_received.append(event.event_type)

        eb.subscribe(EventType.STEP_STARTED, handler)
        eb.subscribe(EventType.STEP_COMPLETED, handler)

        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, eb)
        await c.execute_sequential([_step("a")], executor, sm)

        # Give fire-and-forget handlers a moment
        await asyncio.sleep(0.05)

        assert EventType.STEP_STARTED in events_received
        assert EventType.STEP_COMPLETED in events_received

    @pytest.mark.asyncio
    async def test_step_failure_emits_failed_event(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()
        executor.execute_step.side_effect = RuntimeError("boom")

        events_received: list[EventType] = []

        async def handler(event):
            events_received.append(event.event_type)

        eb.subscribe(EventType.STEP_FAILED, handler)

        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, eb)
        with pytest.raises(RuntimeError, match="boom"):
            await c.execute_sequential([_step("a")], executor, sm)

        await asyncio.sleep(0.05)
        assert EventType.STEP_FAILED in events_received

    @pytest.mark.asyncio
    async def test_updates_state_after_each_step(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()
        executor.execute_step = AsyncMock(return_value="result_val")

        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, eb)
        await c.execute_sequential([_step("x")], executor, sm)

        val = await sm.get("step_x_result", scope=StateScope.WORKFLOW)
        assert val == "result_val"


# ---------------------------------------------------------------------------
# 12.4 — Parallel execution
# ---------------------------------------------------------------------------

class TestParallelExecution:
    @pytest.mark.asyncio
    async def test_independent_steps_run_concurrently(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        started: list[str] = []

        async def side_effect(step, ctx):
            started.append(step.step_id)
            await asyncio.sleep(0.01)
            return {"step": step.step_id}

        executor.execute_step.side_effect = side_effect

        c = Centurion("c", ExecutionStrategy.PARALLEL, eb)
        # a and b are independent; c depends on both
        plan = [_step("a"), _step("b"), _step("c", ["a", "b"])]
        result = await c.execute_parallel(plan, executor, sm)

        assert result["step_a_status"] == "completed"
        assert result["step_b_status"] == "completed"
        assert result["step_c_status"] == "completed"

    @pytest.mark.asyncio
    async def test_respects_concurrency_limit(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        concurrent_count = 0
        max_concurrent = 0

        async def side_effect(step, ctx):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.02)
            concurrent_count -= 1
            return {"step": step.step_id}

        executor.execute_step.side_effect = side_effect

        c = Centurion("c", ExecutionStrategy.PARALLEL, eb)
        plan = [_step("a"), _step("b"), _step("c"), _step("d")]
        await c.execute_parallel(plan, executor, sm, max_concurrency=2)

        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_failure_propagates(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        async def side_effect(step, ctx):
            if step.step_id == "b":
                raise RuntimeError("b failed")
            return "ok"

        executor.execute_step.side_effect = side_effect

        c = Centurion("c", ExecutionStrategy.PARALLEL, eb)
        plan = [_step("a"), _step("b")]
        with pytest.raises(RuntimeError, match="b failed"):
            await c.execute_parallel(plan, executor, sm)



# ---------------------------------------------------------------------------
# 12.5 — Conditional execution
# ---------------------------------------------------------------------------

class TestConditionalExecution:
    @pytest.mark.asyncio
    async def test_condition_true_executes_step(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        step = _step("a", condition=lambda state: True)
        c = Centurion("c", ExecutionStrategy.CONDITIONAL, eb)
        await c.execute_conditional([step], executor, sm)

        status = await sm.get("step_a_status", scope=StateScope.WORKFLOW)
        assert status == "completed"

    @pytest.mark.asyncio
    async def test_condition_false_skips_step(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        step = _step("a", condition=lambda state: False)
        c = Centurion("c", ExecutionStrategy.CONDITIONAL, eb)
        await c.execute_conditional([step], executor, sm)

        status = await sm.get("step_a_status", scope=StateScope.WORKFLOW)
        assert status == "skipped"
        executor.execute_step.assert_not_called()

    @pytest.mark.asyncio
    async def test_condition_uses_workflow_state(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        await sm.set("flag", True, scope=StateScope.WORKFLOW)

        step = _step("a", condition=lambda state: state.get("flag", False))
        c = Centurion("c", ExecutionStrategy.CONDITIONAL, eb)
        await c.execute_conditional([step], executor, sm)

        status = await sm.get("step_a_status", scope=StateScope.WORKFLOW)
        assert status == "completed"

    @pytest.mark.asyncio
    async def test_condition_exception_propagates(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        def bad_condition(state):
            raise ValueError("bad condition")

        step = _step("a", condition=bad_condition)
        c = Centurion("c", ExecutionStrategy.CONDITIONAL, eb)
        with pytest.raises(ValueError, match="bad condition"):
            await c.execute_conditional([step], executor, sm)

    @pytest.mark.asyncio
    async def test_no_condition_executes_normally(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        step = _step("a")  # no condition in config
        c = Centurion("c", ExecutionStrategy.CONDITIONAL, eb)
        await c.execute_conditional([step], executor, sm)

        status = await sm.get("step_a_status", scope=StateScope.WORKFLOW)
        assert status == "completed"

    @pytest.mark.asyncio
    async def test_evaluate_condition_sync(self):
        c = Centurion("c", ExecutionStrategy.CONDITIONAL, _make_event_bus())
        assert await c.evaluate_condition(lambda s: True, {}) is True
        assert await c.evaluate_condition(lambda s: False, {}) is False

    @pytest.mark.asyncio
    async def test_evaluate_condition_async(self):
        async def async_cond(state):
            return state.get("go", False)

        c = Centurion("c", ExecutionStrategy.CONDITIONAL, _make_event_bus())
        assert await c.evaluate_condition(async_cond, {"go": True}) is True
        assert await c.evaluate_condition(async_cond, {}) is False


# ---------------------------------------------------------------------------
# 12.6 — Orchestrate
# ---------------------------------------------------------------------------

class TestOrchestrate:
    @pytest.mark.asyncio
    async def test_sequential_orchestrate(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        wf = _workflow(
            [_step("a"), _step("b", ["a"])],
            strategy=ExecutionStrategy.SEQUENTIAL,
        )
        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, eb)
        result = await c.orchestrate(wf, sm, executor)

        assert result["step_a_status"] == "completed"
        assert result["step_b_status"] == "completed"

    @pytest.mark.asyncio
    async def test_parallel_orchestrate(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        wf = _workflow(
            [_step("a"), _step("b")],
            strategy=ExecutionStrategy.PARALLEL,
        )
        c = Centurion("c", ExecutionStrategy.PARALLEL, eb)
        result = await c.orchestrate(wf, sm, executor)

        assert result["step_a_status"] == "completed"
        assert result["step_b_status"] == "completed"

    @pytest.mark.asyncio
    async def test_conditional_orchestrate(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        wf = _workflow(
            [
                _step("a"),
                _step("b", ["a"], condition=lambda s: False),
            ],
            strategy=ExecutionStrategy.CONDITIONAL,
        )
        c = Centurion("c", ExecutionStrategy.CONDITIONAL, eb)
        result = await c.orchestrate(wf, sm, executor)

        assert result["step_a_status"] == "completed"
        assert result["step_b_status"] == "skipped"

    @pytest.mark.asyncio
    async def test_orchestrate_stores_initial_state(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        wf = _workflow([_step("a")], strategy=ExecutionStrategy.SEQUENTIAL)
        wf.initial_state = {"input_key": "input_val"}

        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, eb)
        result = await c.orchestrate(wf, sm, executor)

        assert result["input_key"] == "input_val"

    @pytest.mark.asyncio
    async def test_orchestrate_no_executor_raises(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        wf = _workflow([_step("a")])

        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, eb)
        with pytest.raises(ValueError, match="No WorkflowExecutor"):
            await c.orchestrate(wf, sm)

    @pytest.mark.asyncio
    async def test_orchestrate_failure_propagates(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()
        executor.execute_step.side_effect = RuntimeError("step failed")

        wf = _workflow([_step("a")])
        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, eb)
        with pytest.raises(RuntimeError, match="step failed"):
            await c.orchestrate(wf, sm, executor)
