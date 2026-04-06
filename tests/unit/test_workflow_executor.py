"""Comprehensive unit tests for WorkflowExecutor.

Tests cover:
- Step execution (via execute_graph and execute_step)
- Checkpoint/restore
- Provider switching (using real MockProvider instances)
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import pytest

from agentlegatus.core.event_bus import Event, EventBus, EventType
from agentlegatus.core.executor import WorkflowExecutor
from agentlegatus.core.graph import PEGEdge, PEGNode, PortableExecutionGraph
from agentlegatus.core.state import InMemoryStateBackend, StateManager, StateScope
from agentlegatus.core.workflow import RetryPolicy, WorkflowStep
from agentlegatus.exceptions import ProviderSwitchError
from agentlegatus.providers.mock import MockProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(nodes, edges=None):
    """Build a PortableExecutionGraph from simple dicts."""
    graph = PortableExecutionGraph()
    for n in nodes:
        graph.add_node(
            PEGNode(
                node_id=n["id"],
                node_type=n.get("type", "agent"),
                config=n.get("config", {}),
                inputs=n.get("inputs", []),
                outputs=n.get("outputs", []),
            )
        )
    for e in edges or []:
        graph.add_edge(PEGEdge(source=e[0], target=e[1]))
    return graph


def _make_executor(provider=None, state_manager=None, event_bus=None):
    """Create a WorkflowExecutor with sensible defaults."""
    provider = provider or MockProvider(config={})
    state_manager = state_manager or StateManager(backend=InMemoryStateBackend())
    event_bus = event_bus or EventBus()
    return WorkflowExecutor(
        provider=provider,
        state_manager=state_manager,
        tool_registry=Mock(),
        event_bus=event_bus,
    )


class _EventCollector:
    """Subscribes to event types and collects emitted events."""

    def __init__(self, event_bus: EventBus, *event_types: EventType):
        self.events: list[Event] = []
        for et in event_types:
            event_bus.subscribe(et, self._handler)

    async def _handler(self, event: Event):
        self.events.append(event)

    def of_type(self, et: EventType) -> list[Event]:
        return [e for e in self.events if e.event_type == et]


# ===================================================================
# Step Execution Tests
# ===================================================================

class TestStepExecution:
    """Tests for step execution via execute_graph (the working path)."""

    async def test_execute_step_raises_not_implemented(self):
        """execute_step() is a stub and raises NotImplementedError."""
        executor = _make_executor()
        step = WorkflowStep(step_id="s1", step_type="agent", config={})
        with pytest.raises(NotImplementedError):
            await executor.execute_step(step, {})

    async def test_single_step_executes_provider(self):
        """A single-node graph delegates execution to the provider."""
        provider = MockProvider(config={})
        executor = _make_executor(provider=provider)

        graph = _make_graph([{"id": "step1", "config": {"agent": {"agent_id": "a1"}}}])
        results = await executor.execute_graph(graph, {})

        assert "step1" in results
        assert provider.execution_count == 1

    async def test_execute_step_emits_started_and_completed_events(self):
        """StepStarted and StepCompleted events are emitted for each node."""
        event_bus = EventBus()
        collector = _EventCollector(
            event_bus, EventType.STEP_STARTED, EventType.STEP_COMPLETED
        )
        executor = _make_executor(event_bus=event_bus)

        graph = _make_graph([{"id": "n1", "config": {"agent": {"agent_id": "a1"}}}])
        await executor.execute_graph(graph, {})

        started = collector.of_type(EventType.STEP_STARTED)
        completed = collector.of_type(EventType.STEP_COMPLETED)
        assert len(started) == 1
        assert started[0].data["node_id"] == "n1"
        assert len(completed) == 1
        assert completed[0].data["node_id"] == "n1"

    async def test_execute_step_emits_failed_event_on_error(self):
        """StepFailed event is emitted when a node raises an exception."""
        provider = Mock()
        provider.execute_agent = AsyncMock(side_effect=RuntimeError("kaboom"))
        event_bus = EventBus()
        collector = _EventCollector(event_bus, EventType.STEP_FAILED)
        executor = _make_executor(provider=provider, event_bus=event_bus)

        graph = _make_graph([{"id": "bad"}])
        with pytest.raises(RuntimeError, match="kaboom"):
            await executor.execute_graph(graph, {})

        failed = collector.of_type(EventType.STEP_FAILED)
        assert len(failed) == 1
        assert failed[0].data["node_id"] == "bad"
        assert failed[0].data["error"] == "kaboom"
        assert failed[0].data["error_type"] == "RuntimeError"

    async def test_execute_step_collects_result_in_state(self):
        """Step results are stored in the STEP scope of the state manager."""
        sm = StateManager(backend=InMemoryStateBackend())
        provider = MockProvider(config={})
        executor = _make_executor(provider=provider, state_manager=sm)

        graph = _make_graph([{"id": "s1", "config": {"agent": {"agent_id": "a1"}}}])
        await executor.execute_graph(graph, {})

        stored = await sm.get("result_s1", scope=StateScope.STEP)
        assert stored is not None

    async def test_multi_step_sequential_execution_order(self):
        """Steps execute in topological order when chained."""
        provider = Mock()
        call_order = []

        async def track(agent, input_data, state):
            call_order.append(input_data["node_id"])
            return f"res_{input_data['node_id']}"

        provider.execute_agent = AsyncMock(side_effect=track)
        executor = _make_executor(provider=provider)

        graph = _make_graph(
            [{"id": "a"}, {"id": "b", "inputs": ["a"]}, {"id": "c", "inputs": ["b"]}],
            [("a", "b"), ("b", "c")],
        )
        await executor.execute_graph(graph, {})
        assert call_order == ["a", "b", "c"]

    async def test_step_timeout_enforcement(self):
        """A node with a timeout that is exceeded raises TimeoutError."""
        provider = Mock()

        async def slow(agent, input_data, state):
            await asyncio.sleep(10)
            return "done"

        provider.execute_agent = AsyncMock(side_effect=slow)
        executor = _make_executor(provider=provider)

        graph = _make_graph([{"id": "slow", "config": {"timeout": 0.05}}])
        with pytest.raises(asyncio.TimeoutError):
            await executor.execute_graph(graph, {})

    async def test_events_emitted_per_step_in_multi_step(self):
        """Each step in a multi-step graph emits its own started/completed pair."""
        event_bus = EventBus()
        collector = _EventCollector(
            event_bus, EventType.STEP_STARTED, EventType.STEP_COMPLETED
        )
        executor = _make_executor(event_bus=event_bus)

        graph = _make_graph(
            [
                {"id": "x", "config": {"agent": {"agent_id": "ax"}}},
                {"id": "y", "config": {"agent": {"agent_id": "ay"}}},
            ],
            [("x", "y")],
        )
        await executor.execute_graph(graph, {})

        assert len(collector.of_type(EventType.STEP_STARTED)) == 2
        assert len(collector.of_type(EventType.STEP_COMPLETED)) == 2


# ===================================================================
# Checkpoint / Restore Tests
# ===================================================================

class TestCheckpointRestore:
    """Tests for checkpoint_state() and restore_from_checkpoint()."""

    async def test_checkpoint_saves_current_state(self):
        """checkpoint_state() persists a snapshot that can be listed."""
        executor = _make_executor()
        await executor.state_manager.set("k", "v", StateScope.WORKFLOW)
        await executor.checkpoint_state("cp1")

        snapshots = await executor.state_manager.list_snapshots()
        assert "cp1" in snapshots

    async def test_restore_brings_back_state(self):
        """restore_from_checkpoint() restores workflow state to the snapshot."""
        sm = StateManager(backend=InMemoryStateBackend())
        executor = _make_executor(state_manager=sm)
        executor._current_workflow_id = "wf1"
        executor._completed_steps = {"s1"}

        await sm.set("data", "original", StateScope.WORKFLOW)
        await executor.checkpoint_state("cp")

        # mutate after checkpoint
        await sm.set("data", "changed", StateScope.WORKFLOW)
        executor._completed_steps = {"s1", "s2", "s3"}

        ctx = await executor.restore_from_checkpoint("cp")

        assert await sm.get("data", StateScope.WORKFLOW) == "original"
        assert executor._completed_steps == {"s1"}
        assert ctx["workflow_id"] == "wf1"

    async def test_restore_nonexistent_checkpoint_raises(self):
        """Restoring a checkpoint that doesn't exist raises ValueError."""
        executor = _make_executor()
        with pytest.raises(ValueError, match="not found"):
            await executor.restore_from_checkpoint("does_not_exist")

    async def test_checkpoint_metadata_includes_completed_steps(self):
        """Checkpoint metadata stores the set of completed steps."""
        executor = _make_executor()
        executor._completed_steps = {"a", "b", "c"}
        executor._current_workflow_id = "wf_test"

        await executor.checkpoint_state("cp_meta")

        meta = await executor.state_manager.get(
            "checkpoint_metadata_cp_meta", StateScope.WORKFLOW
        )
        assert set(meta["completed_steps"]) == {"a", "b", "c"}
        assert meta["workflow_id"] == "wf_test"

    async def test_multiple_checkpoints_are_independent(self):
        """Two checkpoints capture different states and restore independently."""
        sm = StateManager(backend=InMemoryStateBackend())
        executor = _make_executor(state_manager=sm)
        executor._current_workflow_id = "wf"

        # checkpoint 1
        executor._completed_steps = {"s1"}
        await sm.set("val", 1, StateScope.WORKFLOW)
        await executor.checkpoint_state("cp1")

        # checkpoint 2
        executor._completed_steps = {"s1", "s2"}
        await sm.set("val", 2, StateScope.WORKFLOW)
        await executor.checkpoint_state("cp2")

        # restore cp1
        await executor.restore_from_checkpoint("cp1")
        assert await sm.get("val", StateScope.WORKFLOW) == 1
        assert executor._completed_steps == {"s1"}

        # restore cp2
        await executor.restore_from_checkpoint("cp2")
        assert await sm.get("val", StateScope.WORKFLOW) == 2
        assert executor._completed_steps == {"s1", "s2"}

    async def test_checkpoint_with_empty_state(self):
        """Checkpointing with no completed steps and no extra state works."""
        executor = _make_executor()
        executor._completed_steps = set()
        executor._current_workflow_id = None

        await executor.checkpoint_state("cp_empty")
        ctx = await executor.restore_from_checkpoint("cp_empty")

        assert ctx["completed_steps"] == []
        assert ctx["workflow_id"] is None


# ===================================================================
# Provider Switching Tests
# ===================================================================

class TestProviderSwitching:
    """Tests for switch_provider() using real MockProvider instances."""

    async def test_switch_exports_state_from_current_provider(self):
        """switch_provider() calls export_state() on the old provider."""
        old = MockProvider(config={})
        old._state = {"key": "value"}
        executor = _make_executor(provider=old)

        new = MockProvider(config={})
        await executor.switch_provider(new)

        # The new provider should have received the exported state
        assert new._state == old._state

    async def test_switch_imports_state_to_new_provider(self):
        """The new provider receives the exported state via import_state()."""
        old = MockProvider(config={})
        await old.create_agent({"agent_id": "agent1", "name": "Agent 1"})
        old._state["counter"] = 42

        executor = _make_executor(provider=old)
        new = MockProvider(config={})
        await executor.switch_provider(new)

        assert new._state.get("counter") == 42
        assert "agent1" in new._agents

    async def test_switch_converts_workflow_via_portable_graph(self):
        """switch_provider() converts the current workflow through PEG."""
        old = MockProvider(config={})
        sm = StateManager(backend=InMemoryStateBackend())
        executor = _make_executor(provider=old, state_manager=sm)

        # Store a workflow that MockProvider can convert
        workflow = {
            "nodes": [
                {"node_id": "n1", "node_type": "agent", "config": {"model": "gpt-4"}},
                {"node_id": "n2", "node_type": "agent", "config": {}},
            ],
            "edges": [{"source": "n1", "target": "n2"}],
            "metadata": {"version": "1.0"},
        }
        await sm.set("current_workflow", workflow, StateScope.WORKFLOW)

        new = MockProvider(config={})
        await executor.switch_provider(new)

        # Verify the new provider's workflow was stored
        stored = await sm.get("current_workflow", StateScope.WORKFLOW)
        assert stored is not None
        assert stored["provider"] == "mock"
        assert len(stored["nodes"]) == 2

    async def test_switch_emits_provider_switched_event(self):
        """switch_provider() emits a PROVIDER_SWITCHED event."""
        event_bus = EventBus()
        collector = _EventCollector(event_bus, EventType.PROVIDER_SWITCHED)
        old = MockProvider(config={})
        executor = _make_executor(provider=old, event_bus=event_bus)
        executor._current_workflow_id = "wf_switch"

        new = MockProvider(config={})
        await executor.switch_provider(new)

        events = collector.of_type(EventType.PROVIDER_SWITCHED)
        assert len(events) == 1
        ev = events[0]
        assert ev.data["old_provider"] == "MockProvider"
        assert ev.data["new_provider"] == "MockProvider"
        assert ev.data["workflow_id"] == "wf_switch"

    async def test_switch_validates_portable_graph(self):
        """switch_provider() rejects an invalid portable graph (with cycle)."""
        old = Mock()
        # Build a cyclic graph
        cyclic = PortableExecutionGraph()
        cyclic.nodes["a"] = PEGNode(node_id="a", node_type="agent", config={})
        cyclic.nodes["b"] = PEGNode(node_id="b", node_type="agent", config={})
        cyclic.edges = [
            PEGEdge(source="a", target="b"),
            PEGEdge(source="b", target="a"),
        ]
        old.export_state = Mock(return_value={})
        old.to_portable_graph = Mock(return_value=cyclic)
        old.import_state = Mock()

        sm = StateManager(backend=InMemoryStateBackend())
        await sm.set("current_workflow", {"nodes": []}, StateScope.WORKFLOW)
        executor = _make_executor(provider=old, state_manager=sm)

        new = MockProvider(config={})
        with pytest.raises(ProviderSwitchError, match="validation failed"):
            await executor.switch_provider(new)

        # Old provider should be restored
        assert executor.provider is old

    async def test_switch_rollback_on_import_failure(self):
        """If import_state fails, the executor rolls back to the old provider."""
        old = MockProvider(config={})
        executor = _make_executor(provider=old)

        new = Mock()
        new.import_state = Mock(side_effect=RuntimeError("import boom"))
        new.from_portable_graph = Mock()

        with pytest.raises(ProviderSwitchError):
            await executor.switch_provider(new)

        assert executor.provider is old

    async def test_switch_updates_executor_provider_ref(self):
        """After a successful switch, executor.provider points to the new provider."""
        old = MockProvider(config={})
        executor = _make_executor(provider=old)

        new = MockProvider(config={})
        await executor.switch_provider(new)

        assert executor.provider is new

    async def test_switch_with_no_stored_workflow_uses_empty_graph(self):
        """When no workflow is in state, switch uses an empty portable graph."""
        old = MockProvider(config={})
        executor = _make_executor(provider=old)

        new = MockProvider(config={})
        # Should not raise — empty graph is valid
        await executor.switch_provider(new)
        assert executor.provider is new

    async def test_switch_event_contains_graph_metadata(self):
        """The ProviderSwitched event includes graph_nodes and graph_edges counts."""
        event_bus = EventBus()
        collector = _EventCollector(event_bus, EventType.PROVIDER_SWITCHED)
        old = MockProvider(config={})
        sm = StateManager(backend=InMemoryStateBackend())
        executor = _make_executor(provider=old, state_manager=sm, event_bus=event_bus)

        workflow = {
            "nodes": [
                {"node_id": "n1", "node_type": "agent", "config": {}},
                {"node_id": "n2", "node_type": "agent", "config": {}},
            ],
            "edges": [{"source": "n1", "target": "n2"}],
            "metadata": {},
        }
        await sm.set("current_workflow", workflow, StateScope.WORKFLOW)

        new = MockProvider(config={})
        await executor.switch_provider(new)

        ev = collector.events[0]
        assert ev.data["graph_nodes"] == 2
        assert ev.data["graph_edges"] == 1
        assert isinstance(ev.data["state_keys"], list)
