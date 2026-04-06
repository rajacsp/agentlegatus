"""Integration test: provider switching.

Verifies that the WorkflowExecutor can switch from one MockProvider to
another while preserving state and emitting the correct events.
"""

import pytest

from agentlegatus.core.event_bus import Event, EventBus, EventType
from agentlegatus.core.executor import WorkflowExecutor
from agentlegatus.core.graph import PEGEdge, PEGNode, PortableExecutionGraph
from agentlegatus.core.state import InMemoryStateBackend, StateManager, StateScope
from agentlegatus.exceptions import ProviderSwitchError
from agentlegatus.providers.mock import MockProvider
from agentlegatus.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _EventCollector:
    def __init__(self, event_bus: EventBus, *event_types: EventType):
        self.events: list[Event] = []
        for et in event_types:
            event_bus.subscribe(et, self._handler)

    async def _handler(self, event: Event):
        self.events.append(event)

    def of_type(self, et: EventType) -> list[Event]:
        return [e for e in self.events if e.event_type == et]


def _make_graph() -> PortableExecutionGraph:
    """Build a simple two-node graph."""
    graph = PortableExecutionGraph()
    graph.add_node(PEGNode(node_id="a", node_type="agent", config={"agent": {"agent_id": "a1"}}, inputs=[], outputs=[]))
    graph.add_node(PEGNode(node_id="b", node_type="agent", config={"agent": {"agent_id": "a2"}}, inputs=[], outputs=[]))
    graph.add_edge(PEGEdge(source="a", target="b"))
    return graph


def _make_executor(
    provider: MockProvider | None = None,
) -> tuple[WorkflowExecutor, StateManager, EventBus, MockProvider]:
    provider = provider or MockProvider(config={})
    event_bus = EventBus()
    sm = StateManager(backend=InMemoryStateBackend(), event_bus=event_bus)
    tr = ToolRegistry()
    executor = WorkflowExecutor(provider=provider, state_manager=sm, tool_registry=tr, event_bus=event_bus)
    return executor, sm, event_bus, provider


# ===================================================================
# Tests
# ===================================================================


class TestProviderSwitching:
    """Execute with one provider, switch, verify state migration."""

    @pytest.mark.asyncio
    async def test_switch_preserves_exported_state(self):
        """State exported from provider A should be importable by provider B."""
        provider_a = MockProvider(config={"label": "A"})
        executor, sm, eb, _ = _make_executor(provider_a)

        # Execute a graph so provider_a accumulates state
        graph = _make_graph()
        await executor.execute_graph(graph, {"seed": 42})

        assert provider_a.execution_count == 2

        # Store the workflow representation so switch_provider can find it
        workflow_repr = provider_a.from_portable_graph(graph)
        await sm.set("current_workflow", workflow_repr, scope=StateScope.WORKFLOW)

        # Switch to provider B
        provider_b = MockProvider(config={"label": "B"})
        await executor.switch_provider(provider_b)

        # Provider B should have imported the state from A
        exported = provider_b.export_state()
        assert exported["execution_count"] == provider_a.execution_count

    @pytest.mark.asyncio
    async def test_switch_emits_provider_switched_event(self):
        """A ProviderSwitched event must be emitted on successful switch."""
        provider_a = MockProvider(config={})
        executor, sm, eb, _ = _make_executor(provider_a)
        collector = _EventCollector(eb, EventType.PROVIDER_SWITCHED)

        provider_b = MockProvider(config={})
        await executor.switch_provider(provider_b)

        switched = collector.of_type(EventType.PROVIDER_SWITCHED)
        assert len(switched) == 1
        assert switched[0].data["old_provider"] == "MockProvider"
        assert switched[0].data["new_provider"] == "MockProvider"

    @pytest.mark.asyncio
    async def test_switch_updates_executor_provider(self):
        """After switching, the executor should use the new provider."""
        provider_a = MockProvider(config={})
        executor, sm, eb, _ = _make_executor(provider_a)

        provider_b = MockProvider(config={})
        await executor.switch_provider(provider_b)

        assert executor.provider is provider_b

    @pytest.mark.asyncio
    async def test_execution_continues_after_switch(self):
        """Workflow execution should work with the new provider after switch."""
        provider_a = MockProvider(config={})
        executor, sm, eb, _ = _make_executor(provider_a)

        # Execute once with provider A
        graph = _make_graph()
        results_a = await executor.execute_graph(graph, {})
        assert provider_a.execution_count == 2

        # Store workflow for switch
        workflow_repr = provider_a.from_portable_graph(graph)
        await sm.set("current_workflow", workflow_repr, scope=StateScope.WORKFLOW)

        # Switch to provider B
        provider_b = MockProvider(config={})
        await executor.switch_provider(provider_b)

        # Execute again — should use provider B
        results_b = await executor.execute_graph(graph, {})
        # provider_b imported execution_count=2 from A, then ran 2 more
        assert provider_b.execution_count == 4

    @pytest.mark.asyncio
    async def test_switch_rollback_on_failure(self):
        """If switch fails, the executor should keep the old provider."""
        provider_a = MockProvider(config={})
        executor, sm, eb, _ = _make_executor(provider_a)

        # Create a provider that will fail on import_state
        class FailingProvider(MockProvider):
            def import_state(self, state):
                raise RuntimeError("import boom")

        bad_provider = FailingProvider(config={})

        with pytest.raises(ProviderSwitchError):
            await executor.switch_provider(bad_provider)

        # Executor should still point to provider_a
        assert executor.provider is provider_a
