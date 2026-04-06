"""Unit tests for hierarchy components: Agent, Cohort, Centurion, Legatus."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentlegatus.core.event_bus import Event, EventBus, EventType
from agentlegatus.core.models import AgentCapability
from agentlegatus.core.state import InMemoryStateBackend, StateManager, StateScope
from agentlegatus.core.workflow import (
    ExecutionStrategy,
    WorkflowDefinition,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStep,
)
from agentlegatus.hierarchy.agent import Agent
from agentlegatus.hierarchy.centurion import Centurion
from agentlegatus.hierarchy.cohort import Cohort, CohortFullError, CohortStrategy
from agentlegatus.hierarchy.legatus import Legatus
from agentlegatus.memory.base import InMemoryMemoryBackend, MemoryType
from agentlegatus.memory.manager import MemoryManager
from agentlegatus.providers.mock import MockProvider
from agentlegatus.tools.registry import ToolRegistry
from agentlegatus.tools.tool import Tool, ToolParameter


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_event_bus() -> EventBus:
    return EventBus()


def _make_state_manager(event_bus: EventBus | None = None) -> StateManager:
    return StateManager(backend=InMemoryStateBackend(), event_bus=event_bus)


def _make_mock_provider() -> MockProvider:
    return MockProvider(config={"model": "mock-model"})


def _make_tool(name: str = "test_tool") -> Tool:
    async def handler(data):
        return {"result": f"executed_{name}"}

    return Tool(
        name=name,
        description=f"A test tool named {name}",
        parameters=[
            ToolParameter(name="input", type="string", description="input value"),
        ],
        handler=handler,
    )


def _make_agent(
    agent_id: str = "agent-1",
    name: str = "TestAgent",
    capabilities: List[AgentCapability] | None = None,
    provider: MockProvider | None = None,
    tool_registry: ToolRegistry | None = None,
    memory_manager: MemoryManager | None = None,
) -> Agent:
    if capabilities is None:
        capabilities = [AgentCapability.TOOL_USE, AgentCapability.MEMORY]
    return Agent(
        agent_id=agent_id,
        name=name,
        capabilities=capabilities,
        provider=provider or _make_mock_provider(),
        tool_registry=tool_registry,
        memory_manager=memory_manager,
    )


def _step(step_id: str, depends_on: list[str] | None = None, **config) -> WorkflowStep:
    return WorkflowStep(
        step_id=step_id,
        step_type="agent",
        config=config,
        depends_on=depends_on or [],
    )


def _workflow(
    steps: list[WorkflowStep] | None = None,
    strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
    workflow_id: str = "wf-test",
    timeout: float | None = None,
    **metadata,
) -> WorkflowDefinition:
    return WorkflowDefinition(
        workflow_id=workflow_id,
        name="Test Workflow",
        description="test",
        version="1.0",
        provider="mock",
        steps=steps or [_step("s1")],
        execution_strategy=strategy,
        timeout=timeout,
        metadata=metadata,
    )


def _make_executor() -> MagicMock:
    executor = MagicMock()
    executor.execute_step = AsyncMock(return_value={"output": "ok"})
    executor.checkpoint_state = AsyncMock()
    return executor


# =========================================================================
# AGENT TESTS
# =========================================================================

class TestAgentCreation:
    """Test Agent creation with capabilities."""

    def test_agent_init_basic(self):
        provider = _make_mock_provider()
        agent = Agent(
            agent_id="a1",
            name="Alpha",
            capabilities=[AgentCapability.TOOL_USE],
            provider=provider,
        )
        assert agent.agent_id == "a1"
        assert agent.name == "Alpha"
        assert AgentCapability.TOOL_USE in agent.capabilities
        assert agent.provider is provider

    def test_agent_init_multiple_capabilities(self):
        agent = _make_agent(
            capabilities=[
                AgentCapability.TOOL_USE,
                AgentCapability.MEMORY,
                AgentCapability.PLANNING,
                AgentCapability.REFLECTION,
            ]
        )
        assert len(agent.capabilities) == 4

    def test_agent_init_no_capabilities(self):
        agent = _make_agent(capabilities=[])
        assert agent.capabilities == []

    def test_agent_initial_status_is_idle(self):
        agent = _make_agent()
        status = agent.get_status()
        assert status["status"] == "idle"
        assert status["task_count"] == 0
        assert status["error_count"] == 0


class TestAgentRun:
    """Test Agent.run() delegates to provider."""

    @pytest.mark.asyncio
    async def test_run_delegates_to_provider(self):
        provider = _make_mock_provider()
        agent = _make_agent(provider=provider)

        result = await agent.run(input_data="hello")

        assert result["output"].startswith("mock:agent-1:result:")
        assert result["input"] == "hello"
        assert provider.execution_count == 1

    @pytest.mark.asyncio
    async def test_run_passes_state(self):
        provider = _make_mock_provider()
        agent = _make_agent(provider=provider)

        result = await agent.run(input_data="test", state={"key": "val"})

        assert result["state"]["key"] == "val"

    @pytest.mark.asyncio
    async def test_run_updates_status_to_idle_after_success(self):
        agent = _make_agent()
        await agent.run(input_data="task1")

        status = agent.get_status()
        assert status["status"] == "idle"
        assert status["task_count"] == 1
        assert status["current_task"] is None

    @pytest.mark.asyncio
    async def test_run_increments_task_count(self):
        agent = _make_agent()
        await agent.run(input_data="task1")
        await agent.run(input_data="task2")

        assert agent.get_status()["task_count"] == 2

    @pytest.mark.asyncio
    async def test_run_error_sets_error_status(self):
        provider = MagicMock()
        provider.create_agent = AsyncMock(return_value={"agent_id": "a1"})
        provider.execute_agent = AsyncMock(side_effect=RuntimeError("provider error"))

        agent = _make_agent(provider=provider)

        with pytest.raises(RuntimeError, match="provider error"):
            await agent.run(input_data="fail")

        status = agent.get_status()
        assert status["status"] == "error"
        assert status["error_count"] == 1


class TestAgentInvokeTool:
    """Test Agent.invoke_tool() uses ToolRegistry."""

    @pytest.mark.asyncio
    async def test_invoke_tool_success(self):
        registry = ToolRegistry()
        tool = _make_tool("greet")
        registry.register_tool(tool)

        agent = _make_agent(
            capabilities=[AgentCapability.TOOL_USE],
            tool_registry=registry,
        )

        result = await agent.invoke_tool("greet", {"input": "world"})
        assert result["result"] == "executed_greet"

    @pytest.mark.asyncio
    async def test_invoke_tool_without_capability_raises(self):
        agent = _make_agent(capabilities=[AgentCapability.MEMORY])

        with pytest.raises(ValueError, match="TOOL_USE"):
            await agent.invoke_tool("any_tool", {})

    @pytest.mark.asyncio
    async def test_invoke_tool_without_registry_raises(self):
        agent = _make_agent(
            capabilities=[AgentCapability.TOOL_USE],
            tool_registry=None,
        )

        with pytest.raises(RuntimeError, match="no tool registry"):
            await agent.invoke_tool("any_tool", {})

    @pytest.mark.asyncio
    async def test_invoke_tool_not_found_raises(self):
        registry = ToolRegistry()
        agent = _make_agent(
            capabilities=[AgentCapability.TOOL_USE],
            tool_registry=registry,
        )

        with pytest.raises(KeyError, match="not found"):
            await agent.invoke_tool("nonexistent", {})


class TestAgentMemory:
    """Test Agent.store_memory() / retrieve_memory() uses MemoryManager."""

    @pytest.mark.asyncio
    async def test_store_short_term_memory(self):
        backend = InMemoryMemoryBackend()
        mm = MemoryManager(backend)
        agent = _make_agent(
            capabilities=[AgentCapability.MEMORY],
            memory_manager=mm,
        )

        await agent.store_memory("key1", "value1", memory_type="short_term")

        # Verify stored via manager
        recent = await mm.get_recent(MemoryType.SHORT_TERM, limit=10)
        assert "value1" in recent

    @pytest.mark.asyncio
    async def test_store_long_term_memory(self):
        backend = InMemoryMemoryBackend()
        mm = MemoryManager(backend)
        agent = _make_agent(
            capabilities=[AgentCapability.MEMORY],
            memory_manager=mm,
        )

        await agent.store_memory("key2", "value2", memory_type="long_term")

        recent = await mm.get_recent(MemoryType.LONG_TERM, limit=10)
        assert "value2" in recent

    @pytest.mark.asyncio
    async def test_retrieve_short_term_memory(self):
        backend = InMemoryMemoryBackend()
        mm = MemoryManager(backend)
        agent = _make_agent(
            capabilities=[AgentCapability.MEMORY],
            memory_manager=mm,
        )

        await mm.store_short_term("item1", "data1")
        results = await agent.retrieve_memory("item1", memory_type="short_term", limit=5)
        assert "data1" in results

    @pytest.mark.asyncio
    async def test_retrieve_long_term_memory(self):
        backend = InMemoryMemoryBackend()
        mm = MemoryManager(backend)
        agent = _make_agent(
            capabilities=[AgentCapability.MEMORY],
            memory_manager=mm,
        )

        await mm.store_long_term("doc1", "long_data")
        results = await agent.retrieve_memory("doc1", memory_type="long_term", limit=5)
        assert "long_data" in results

    @pytest.mark.asyncio
    async def test_store_memory_without_capability_raises(self):
        agent = _make_agent(capabilities=[AgentCapability.TOOL_USE])

        with pytest.raises(ValueError, match="MEMORY"):
            await agent.store_memory("k", "v")

    @pytest.mark.asyncio
    async def test_retrieve_memory_without_capability_raises(self):
        agent = _make_agent(capabilities=[AgentCapability.TOOL_USE])

        with pytest.raises(ValueError, match="MEMORY"):
            await agent.retrieve_memory("q")

    @pytest.mark.asyncio
    async def test_store_memory_without_manager_raises(self):
        agent = _make_agent(
            capabilities=[AgentCapability.MEMORY],
            memory_manager=None,
        )

        with pytest.raises(RuntimeError, match="no memory manager"):
            await agent.store_memory("k", "v")

    @pytest.mark.asyncio
    async def test_retrieve_memory_without_manager_raises(self):
        agent = _make_agent(
            capabilities=[AgentCapability.MEMORY],
            memory_manager=None,
        )

        with pytest.raises(RuntimeError, match="no memory manager"):
            await agent.retrieve_memory("q")


class TestAgentGetStatus:
    """Test Agent.get_status() returns current state."""

    def test_status_contains_required_fields(self):
        agent = _make_agent()
        status = agent.get_status()

        assert "agent_id" in status
        assert "name" in status
        assert "status" in status
        assert "capabilities" in status
        assert "task_count" in status
        assert "error_count" in status
        assert "total_execution_time" in status
        assert "average_execution_time" in status

    @pytest.mark.asyncio
    async def test_status_reflects_execution_metrics(self):
        agent = _make_agent()
        await agent.run(input_data="task")

        status = agent.get_status()
        assert status["task_count"] == 1
        assert status["total_execution_time"] > 0
        assert status["average_execution_time"] > 0


# =========================================================================
# COHORT TESTS
# =========================================================================

class TestCohortCreation:
    """Test Cohort creation with strategy and max_agents."""

    def test_cohort_init(self):
        cohort = Cohort("c1", CohortStrategy.ROUND_ROBIN, max_agents=5)
        assert cohort.name == "c1"
        assert cohort.strategy == CohortStrategy.ROUND_ROBIN
        assert cohort.max_agents == 5

    def test_cohort_default_max_agents(self):
        cohort = Cohort("c2", CohortStrategy.LOAD_BALANCED)
        assert cohort.max_agents == 10

    def test_cohort_all_strategies(self):
        for strategy in CohortStrategy:
            cohort = Cohort(f"c_{strategy.value}", strategy)
            assert cohort.strategy == strategy


class TestCohortAddRemoveAgent:
    """Test add_agent() / remove_agent() with capacity enforcement."""

    @pytest.mark.asyncio
    async def test_add_agent(self):
        cohort = Cohort("c1", CohortStrategy.ROUND_ROBIN)
        agent = _make_agent(agent_id="a1")
        await cohort.add_agent(agent)

        assert len(cohort._agents) == 1
        assert "a1" in cohort._agents

    @pytest.mark.asyncio
    async def test_add_multiple_agents(self):
        cohort = Cohort("c1", CohortStrategy.ROUND_ROBIN, max_agents=3)
        for i in range(3):
            await cohort.add_agent(_make_agent(agent_id=f"a{i}"))
        assert len(cohort._agents) == 3

    @pytest.mark.asyncio
    async def test_add_agent_exceeds_capacity_raises(self):
        cohort = Cohort("c1", CohortStrategy.ROUND_ROBIN, max_agents=1)
        await cohort.add_agent(_make_agent(agent_id="a1"))

        with pytest.raises(CohortFullError, match="max capacity"):
            await cohort.add_agent(_make_agent(agent_id="a2"))

    @pytest.mark.asyncio
    async def test_remove_agent_success(self):
        cohort = Cohort("c1", CohortStrategy.ROUND_ROBIN)
        await cohort.add_agent(_make_agent(agent_id="a1"))

        removed = await cohort.remove_agent("a1")
        assert removed is True
        assert len(cohort._agents) == 0

    @pytest.mark.asyncio
    async def test_remove_agent_not_found(self):
        cohort = Cohort("c1", CohortStrategy.ROUND_ROBIN)
        removed = await cohort.remove_agent("nonexistent")
        assert removed is False

    @pytest.mark.asyncio
    async def test_remove_leader_reassigns(self):
        cohort = Cohort("c1", CohortStrategy.LEADER_FOLLOWER)
        await cohort.add_agent(_make_agent(agent_id="leader"))
        await cohort.add_agent(_make_agent(agent_id="follower"))

        assert cohort._leader_id == "leader"
        await cohort.remove_agent("leader")
        assert cohort._leader_id == "follower"


class TestCohortRoundRobin:
    """Test ROUND_ROBIN strategy distributes tasks."""

    @pytest.mark.asyncio
    async def test_round_robin_distributes_evenly(self):
        cohort = Cohort("c1", CohortStrategy.ROUND_ROBIN)
        provider = _make_mock_provider()
        sm = _make_state_manager()

        a1 = _make_agent(agent_id="a1", provider=provider)
        a2 = _make_agent(agent_id="a2", provider=provider)
        await cohort.add_agent(a1)
        await cohort.add_agent(a2)

        # Execute 4 tasks — should alternate between a1 and a2
        results = []
        for i in range(4):
            r = await cohort.execute_task({"input": f"task{i}"}, sm)
            results.append(r["agent_id"])

        assert results[0] == "a1"
        assert results[1] == "a2"
        assert results[2] == "a1"
        assert results[3] == "a2"


class TestCohortLoadBalanced:
    """Test LOAD_BALANCED strategy assigns to least busy."""

    @pytest.mark.asyncio
    async def test_load_balanced_picks_least_busy(self):
        cohort = Cohort("c1", CohortStrategy.LOAD_BALANCED)
        provider = _make_mock_provider()
        sm = _make_state_manager()

        a1 = _make_agent(agent_id="a1", provider=provider)
        a2 = _make_agent(agent_id="a2", provider=provider)
        await cohort.add_agent(a1)
        await cohort.add_agent(a2)

        # Both idle, first task goes to agent with lowest task_count
        result = await cohort.execute_task({"input": "task1"}, sm)
        assert result["agent_id"] in ("a1", "a2")


class TestCohortBroadcast:
    """Test BROADCAST strategy sends to all agents."""

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all(self):
        cohort = Cohort("c1", CohortStrategy.BROADCAST)
        provider = _make_mock_provider()
        sm = _make_state_manager()

        a1 = _make_agent(agent_id="a1", provider=provider)
        a2 = _make_agent(agent_id="a2", provider=provider)
        a3 = _make_agent(agent_id="a3", provider=provider)
        await cohort.add_agent(a1)
        await cohort.add_agent(a2)
        await cohort.add_agent(a3)

        results = await cohort.execute_task({"input": "broadcast_task"}, sm)

        # Should return a list with one result per agent
        assert isinstance(results, list)
        assert len(results) == 3


class TestCohortLeaderFollower:
    """Test LEADER_FOLLOWER strategy routes through leader."""

    @pytest.mark.asyncio
    async def test_leader_follower_routes_to_leader(self):
        cohort = Cohort("c1", CohortStrategy.LEADER_FOLLOWER)
        provider = _make_mock_provider()
        sm = _make_state_manager()

        leader = _make_agent(agent_id="leader", provider=provider)
        follower = _make_agent(agent_id="follower", provider=provider)
        await cohort.add_agent(leader)
        await cohort.add_agent(follower)

        result = await cohort.execute_task({"input": "task"}, sm)
        assert result["agent_id"] == "leader"

    @pytest.mark.asyncio
    async def test_leader_follower_no_leader_raises(self):
        cohort = Cohort("c1", CohortStrategy.LEADER_FOLLOWER)
        cohort._leader_id = None
        provider = _make_mock_provider()
        sm = _make_state_manager()

        # Add agent but force leader to None
        await cohort.add_agent(_make_agent(agent_id="a1", provider=provider))
        cohort._leader_id = None

        with pytest.raises(RuntimeError, match="no leader"):
            await cohort.execute_task({"input": "task"}, sm)


class TestCohortGetAvailableAgents:
    """Test get_available_agents() returns non-busy agents."""

    @pytest.mark.asyncio
    async def test_all_idle_agents_available(self):
        cohort = Cohort("c1", CohortStrategy.ROUND_ROBIN)
        a1 = _make_agent(agent_id="a1")
        a2 = _make_agent(agent_id="a2")
        await cohort.add_agent(a1)
        await cohort.add_agent(a2)

        available = cohort.get_available_agents()
        assert len(available) == 2

    def test_empty_cohort_returns_empty(self):
        cohort = Cohort("c1", CohortStrategy.ROUND_ROBIN)
        assert cohort.get_available_agents() == []

    @pytest.mark.asyncio
    async def test_no_agents_raises_on_execute(self):
        cohort = Cohort("c1", CohortStrategy.ROUND_ROBIN)
        sm = _make_state_manager()

        with pytest.raises(ValueError, match="no agents"):
            await cohort.execute_task({"input": "task"}, sm)


# =========================================================================
# CENTURION TESTS
# =========================================================================

class TestCenturionBuildPlan:
    """Test build_execution_plan() using topological sort."""

    def test_linear_chain(self):
        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, _make_event_bus())
        steps = [_step("a"), _step("b", ["a"]), _step("c", ["b"])]
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
        assert ids.index("a") < ids.index("d")
        assert ids.index("b") < ids.index("d")
        assert ids.index("c") < ids.index("d")

    def test_cycle_raises(self):
        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, _make_event_bus())
        steps = [_step("a", ["c"]), _step("b", ["a"]), _step("c", ["b"])]
        with pytest.raises(ValueError, match="cycle"):
            c.build_execution_plan(steps)

    def test_missing_dependency_raises(self):
        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, _make_event_bus())
        with pytest.raises(ValueError, match="non-existent"):
            c.build_execution_plan([_step("a", ["missing"])])


class TestCenturionSequential:
    """Test execute_sequential() runs steps in order."""

    @pytest.mark.asyncio
    async def test_executes_in_order(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()
        call_order: list[str] = []

        async def side_effect(step, ctx):
            call_order.append(step.step_id)
            return {"step": step.step_id}

        executor.execute_step.side_effect = side_effect

        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, eb)
        plan = [_step("a"), _step("b", ["a"])]
        await c.execute_sequential(plan, executor, sm)

        assert call_order == ["a", "b"]

    @pytest.mark.asyncio
    async def test_stores_step_results(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()
        executor.execute_step = AsyncMock(return_value="result_val")

        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, eb)
        await c.execute_sequential([_step("x")], executor, sm)

        val = await sm.get("step_x_result", scope=StateScope.WORKFLOW)
        assert val == "result_val"


class TestCenturionParallel:
    """Test execute_parallel() runs independent steps concurrently."""

    @pytest.mark.asyncio
    async def test_independent_steps_complete(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        c = Centurion("c", ExecutionStrategy.PARALLEL, eb)
        plan = [_step("a"), _step("b"), _step("c", ["a", "b"])]
        result = await c.execute_parallel(plan, executor, sm)

        assert result["step_a_status"] == "completed"
        assert result["step_b_status"] == "completed"
        assert result["step_c_status"] == "completed"

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
        with pytest.raises(RuntimeError, match="b failed"):
            await c.execute_parallel([_step("a"), _step("b")], executor, sm)


class TestCenturionConditional:
    """Test execute_conditional() evaluates conditions."""

    @pytest.mark.asyncio
    async def test_condition_true_executes(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        step = _step("a", condition=lambda state: True)
        c = Centurion("c", ExecutionStrategy.CONDITIONAL, eb)
        await c.execute_conditional([step], executor, sm)

        status = await sm.get("step_a_status", scope=StateScope.WORKFLOW)
        assert status == "completed"

    @pytest.mark.asyncio
    async def test_condition_false_skips(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        step = _step("a", condition=lambda state: False)
        c = Centurion("c", ExecutionStrategy.CONDITIONAL, eb)
        await c.execute_conditional([step], executor, sm)

        status = await sm.get("step_a_status", scope=StateScope.WORKFLOW)
        assert status == "skipped"
        executor.execute_step.assert_not_called()


class TestCenturionOrchestrate:
    """Test orchestrate() coordinates workflow execution."""

    @pytest.mark.asyncio
    async def test_orchestrate_sequential(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        wf = _workflow([_step("a"), _step("b", ["a"])], strategy=ExecutionStrategy.SEQUENTIAL)
        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, eb)
        result = await c.orchestrate(wf, sm, executor)

        assert result["step_a_status"] == "completed"
        assert result["step_b_status"] == "completed"

    @pytest.mark.asyncio
    async def test_orchestrate_no_executor_raises(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        wf = _workflow()

        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, eb)
        with pytest.raises(ValueError, match="No WorkflowExecutor"):
            await c.orchestrate(wf, sm)

    @pytest.mark.asyncio
    async def test_orchestrate_stores_initial_state(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        wf = _workflow([_step("a")])
        wf.initial_state = {"init_key": "init_val"}

        c = Centurion("c", ExecutionStrategy.SEQUENTIAL, eb)
        result = await c.orchestrate(wf, sm, executor)

        assert result["init_key"] == "init_val"


# =========================================================================
# LEGATUS TESTS
# =========================================================================

class TestLegatusWorkflowExecution:
    """Test execute_workflow() returns WorkflowResult."""

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        legatus = Legatus(config={}, event_bus=eb)
        wf = _workflow()
        result = await legatus.execute_workflow(wf, executor=executor, state_manager=sm)

        assert result.status == WorkflowStatus.COMPLETED
        assert result.execution_time > 0
        assert result.error is None
        assert "workflow_id" in result.metrics

    @pytest.mark.asyncio
    async def test_failed_execution(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()
        executor.execute_step = AsyncMock(side_effect=RuntimeError("boom"))

        legatus = Legatus(config={}, event_bus=eb)
        wf = _workflow()
        result = await legatus.execute_workflow(wf, executor=executor, state_manager=sm)

        assert result.status == WorkflowStatus.FAILED
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_execution_with_initial_state(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        legatus = Legatus(config={}, event_bus=eb)
        wf = _workflow()
        await legatus.execute_workflow(
            wf, initial_state={"foo": "bar"}, executor=executor, state_manager=sm
        )

        val = await sm.get("foo", scope=StateScope.WORKFLOW)
        assert val == "bar"


class TestLegatusWorkflowEvents:
    """Test workflow events emitted (WorkflowStarted, WorkflowCompleted, WorkflowFailed)."""

    @pytest.mark.asyncio
    async def test_emits_started_and_completed(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()
        captured: list[EventType] = []

        async def capture(event):
            captured.append(event.event_type)

        eb.subscribe(EventType.WORKFLOW_STARTED, capture)
        eb.subscribe(EventType.WORKFLOW_COMPLETED, capture)

        legatus = Legatus(config={}, event_bus=eb)
        await legatus.execute_workflow(_workflow(), executor=executor, state_manager=sm)
        await asyncio.sleep(0.05)

        assert EventType.WORKFLOW_STARTED in captured
        assert EventType.WORKFLOW_COMPLETED in captured

    @pytest.mark.asyncio
    async def test_emits_failed_event_on_error(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()
        executor.execute_step = AsyncMock(side_effect=RuntimeError("fail"))
        captured: list[Event] = []

        async def capture(event):
            captured.append(event)

        eb.subscribe(EventType.WORKFLOW_FAILED, capture)

        legatus = Legatus(config={}, event_bus=eb)
        await legatus.execute_workflow(_workflow(), executor=executor, state_manager=sm)
        await asyncio.sleep(0.05)

        assert len(captured) == 1
        assert captured[0].data["error"] == "fail"

    @pytest.mark.asyncio
    async def test_started_event_has_trace_id(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()
        captured: list[Event] = []

        async def capture(event):
            captured.append(event)

        eb.subscribe(EventType.WORKFLOW_STARTED, capture)

        legatus = Legatus(config={}, event_bus=eb)
        await legatus.execute_workflow(_workflow(), executor=executor, state_manager=sm)
        await asyncio.sleep(0.05)

        assert len(captured) == 1
        assert captured[0].trace_id is not None


class TestLegatusCancelWorkflow:
    """Test cancel_workflow() stops execution."""

    @pytest.mark.asyncio
    async def test_cancel_non_running_returns_false(self):
        eb = _make_event_bus()
        legatus = Legatus(config={}, event_bus=eb)
        result = await legatus.cancel_workflow("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_running_workflow(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        async def slow_step(*args, **kwargs):
            await asyncio.sleep(10)
            return {}

        executor.execute_step = slow_step

        legatus = Legatus(config={}, event_bus=eb)
        wf = _workflow()

        task = asyncio.create_task(
            legatus.execute_workflow(wf, executor=executor, state_manager=sm)
        )
        await asyncio.sleep(0.05)

        cancelled = await legatus.cancel_workflow("wf-test", executor=executor)
        assert cancelled is True

        result = await task
        assert result.status == WorkflowStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_emits_cancelled_event(self):
        eb = _make_event_bus()
        captured: list[Event] = []

        async def capture(event):
            captured.append(event)

        eb.subscribe(EventType.WORKFLOW_CANCELLED, capture)

        legatus = Legatus(config={}, event_bus=eb)
        legatus._workflow_statuses["wf-1"] = WorkflowStatus.RUNNING

        await legatus.cancel_workflow("wf-1")
        await asyncio.sleep(0.05)

        assert len(captured) == 1
        assert captured[0].data["reason"] == "user_requested"


class TestLegatusGetStatus:
    """Test get_status() returns workflow status."""

    @pytest.mark.asyncio
    async def test_get_status_after_execution(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        legatus = Legatus(config={}, event_bus=eb)
        await legatus.execute_workflow(_workflow(), executor=executor, state_manager=sm)

        assert legatus.get_status("wf-test") == WorkflowStatus.COMPLETED

    def test_get_status_unknown_raises(self):
        eb = _make_event_bus()
        legatus = Legatus(config={}, event_bus=eb)

        with pytest.raises(KeyError, match="not found"):
            legatus.get_status("nonexistent")

    @pytest.mark.asyncio
    async def test_timeout_sets_cancelled_status(self):
        eb = _make_event_bus()
        sm = _make_state_manager(eb)
        executor = _make_executor()

        async def slow_step(*args, **kwargs):
            await asyncio.sleep(10)
            return {}

        executor.execute_step = slow_step

        legatus = Legatus(config={}, event_bus=eb)
        wf = _workflow(timeout=0.1)
        result = await legatus.execute_workflow(wf, executor=executor, state_manager=sm)

        assert result.status == WorkflowStatus.CANCELLED
        assert legatus.get_status("wf-test") == WorkflowStatus.CANCELLED
