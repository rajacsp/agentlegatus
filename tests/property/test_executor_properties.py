"""Property-based tests for WorkflowExecutor."""

import asyncio
from typing import Any, Dict, List, Optional

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agentlegatus.core.event_bus import EventBus, EventType
from agentlegatus.core.executor import WorkflowExecutor
from agentlegatus.core.graph import PEGEdge, PEGNode, PortableExecutionGraph
from agentlegatus.core.state import InMemoryStateBackend, StateManager
from agentlegatus.exceptions import ProviderSwitchError
from agentlegatus.providers.base import BaseProvider, ProviderCapability
from agentlegatus.tools.registry import ToolRegistry


# Mock Provider for Testing
class MockTestProvider(BaseProvider):
    """Mock test provider with state and graph conversion support."""
    
    def __init__(self, config: Dict[str, Any], provider_id: str = "test"):
        """Initialize test provider with ID for tracking."""
        super().__init__(config)
        self.provider_id = provider_id
        self._state: Dict[str, Any] = {}
        self._workflow: Optional[Any] = None
    
    def _get_capabilities(self) -> List[ProviderCapability]:
        return [
            ProviderCapability.TOOL_CALLING,
            ProviderCapability.STATE_PERSISTENCE,
        ]
    
    async def create_agent(self, agent_config: Dict[str, Any]) -> Any:
        return {"agent_id": f"{self.provider_id}_agent", "config": agent_config}
    
    async def execute_agent(
        self,
        agent: Any,
        input_data: Any,
        state: Optional[Dict[str, Any]] = None
    ) -> Any:
        return {"result": f"{self.provider_id}_result", "input": input_data}
    
    async def invoke_tool(
        self,
        tool: Any,
        tool_input: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        return {"tool": "mock_tool", "output": f"{self.provider_id}_output"}
    
    def export_state(self) -> Dict[str, Any]:
        """Export state in provider-agnostic format."""
        return {
            "provider_id": self.provider_id,
            "state": self._state.copy(),
            "workflow": self._workflow,
        }
    
    def import_state(self, state: Dict[str, Any]) -> None:
        """Import state from provider-agnostic format."""
        self._state = state.get("state", {}).copy()
        self._workflow = state.get("workflow")
    
    def to_portable_graph(self, workflow: Any) -> PortableExecutionGraph:
        """Convert provider-specific workflow to portable graph."""
        graph = PortableExecutionGraph()
        
        # If workflow is None or empty, return empty graph
        if workflow is None:
            return graph
        
        # If workflow is already a PortableExecutionGraph, return it
        if isinstance(workflow, PortableExecutionGraph):
            return workflow
        
        # If workflow is a dict with nodes and edges, reconstruct graph
        if isinstance(workflow, dict):
            if "nodes" in workflow:
                for node_data in workflow["nodes"]:
                    node = PEGNode(**node_data)
                    graph.add_node(node)
            
            if "edges" in workflow:
                for edge_data in workflow["edges"]:
                    edge = PEGEdge(**edge_data)
                    graph.add_edge(edge)
            
            if "metadata" in workflow:
                graph.metadata = workflow["metadata"]
        
        return graph
    
    def from_portable_graph(self, graph: PortableExecutionGraph) -> Any:
        """Convert portable graph to provider-specific workflow."""
        # Convert graph to a simple dict representation
        workflow = {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "config": node.config,
                    "inputs": node.inputs,
                    "outputs": node.outputs,
                }
                for node in graph.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "condition": edge.condition,
                }
                for edge in graph.edges
            ],
            "metadata": graph.metadata,
            "provider_id": self.provider_id,
        }
        
        self._workflow = workflow
        return workflow


# Helper strategies
@st.composite
def state_dict_strategy(draw):
    """Generate state dictionaries."""
    return draw(st.dictionaries(
        st.text(min_size=1, max_size=20, alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters="_"
        )),
        st.one_of(
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(max_size=50),
            st.booleans(),
            st.lists(st.integers(), max_size=5),
        ),
        min_size=1,
        max_size=10
    ))


@st.composite
def simple_graph_strategy(draw):
    """Generate a simple valid graph."""
    num_nodes = draw(st.integers(min_value=1, max_value=5))
    
    graph = PortableExecutionGraph()
    
    # Create nodes
    for i in range(num_nodes):
        node = PEGNode(
            node_id=f"node_{i}",
            node_type=draw(st.sampled_from(["agent", "tool", "condition"])),
            config=draw(st.dictionaries(
                st.text(min_size=1, max_size=10),
                st.integers(),
                max_size=3
            )),
            inputs=[],
            outputs=[],
        )
        graph.add_node(node)
    
    # Create edges (no cycles)
    for i in range(num_nodes - 1):
        if draw(st.booleans()):
            edge = PEGEdge(
                source=f"node_{i}",
                target=f"node_{i+1}",
                condition=None,
            )
            graph.add_edge(edge)
    
    # Add metadata
    graph.metadata = {"test": "metadata", "version": "1.0"}
    
    return graph


# Property 5: Provider State Round-Trip
@pytest.mark.asyncio
@given(state_dict_strategy())
@settings(max_examples=10, deadline=2000)
async def test_property_5_provider_state_round_trip(state_data: Dict[str, Any]):
    """
    Property 5: Provider State Round-Trip
    
    For any provider state S, exporting S and then importing it into 
    another provider instance results in equivalent state.
    
    Validates: Requirements 3.5, 3.6
    """
    # Create first provider with state
    provider1 = MockTestProvider({"api_key": "test1"}, provider_id="provider1")
    provider1._state = state_data.copy()
    
    # Export state
    exported_state = provider1.export_state()
    
    # Verify exported state contains the data
    assert "state" in exported_state
    assert exported_state["state"] == state_data
    
    # Create second provider and import state
    provider2 = MockTestProvider({"api_key": "test2"}, provider_id="provider2")
    provider2.import_state(exported_state)
    
    # Verify state was imported correctly
    assert provider2._state == state_data, (
        f"Imported state should match original state. "
        f"Expected {state_data}, got {provider2._state}"
    )


# Property 6: Portable Graph Round-Trip
@pytest.mark.asyncio
@given(simple_graph_strategy())
@settings(max_examples=10, deadline=2000)
async def test_property_6_portable_graph_round_trip(graph: PortableExecutionGraph):
    """
    Property 6: Portable Graph Round-Trip
    
    For any PortableExecutionGraph G, converting G to provider-specific 
    format and back to portable graph results in an equivalent graph.
    
    Validates: Requirements 3.7, 3.8
    """
    # Create provider
    provider1 = MockTestProvider({"api_key": "test"}, provider_id="provider1")
    
    # Convert to provider-specific format
    provider_workflow = provider1.from_portable_graph(graph)
    
    # Verify workflow was created
    assert provider_workflow is not None
    assert isinstance(provider_workflow, dict)
    
    # Convert back to portable graph
    restored_graph = provider1.to_portable_graph(provider_workflow)
    
    # Verify graphs are equivalent
    assert len(restored_graph.nodes) == len(graph.nodes), (
        f"Number of nodes should match. Expected {len(graph.nodes)}, "
        f"got {len(restored_graph.nodes)}"
    )
    
    # Verify all nodes match
    for node_id, original_node in graph.nodes.items():
        assert node_id in restored_graph.nodes, (
            f"Node {node_id} should exist in restored graph"
        )
        restored_node = restored_graph.nodes[node_id]
        
        assert restored_node.node_id == original_node.node_id
        assert restored_node.node_type == original_node.node_type
        assert restored_node.config == original_node.config
        assert restored_node.inputs == original_node.inputs
        assert restored_node.outputs == original_node.outputs
    
    # Verify edges match
    assert len(restored_graph.edges) == len(graph.edges), (
        f"Number of edges should match. Expected {len(graph.edges)}, "
        f"got {len(restored_graph.edges)}"
    )
    
    for i, original_edge in enumerate(graph.edges):
        restored_edge = restored_graph.edges[i]
        assert restored_edge.source == original_edge.source
        assert restored_edge.target == original_edge.target
        assert restored_edge.condition == original_edge.condition
    
    # Verify metadata matches
    assert restored_graph.metadata == graph.metadata


# Combined test: Provider switching with state and graph preservation
@pytest.mark.asyncio
@given(
    state_dict_strategy(),
    simple_graph_strategy(),
)
@settings(max_examples=10, deadline=3000)
async def test_provider_switching_preserves_state_and_graph(
    state_data: Dict[str, Any],
    graph: PortableExecutionGraph,
):
    """
    Test that provider switching preserves both state and workflow graph.
    
    This test validates the complete provider switching flow:
    1. Export state from old provider
    2. Convert workflow to portable graph
    3. Validate portable graph
    4. Import state into new provider
    5. Convert portable graph to new provider format
    
    Validates: Requirements 5.1-5.8
    """
    # Setup
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend)
    event_bus = EventBus()
    tool_registry = ToolRegistry()
    
    # Create first provider with state and workflow
    provider1 = MockTestProvider({"api_key": "test1"}, provider_id="provider1")
    provider1._state = state_data.copy()
    
    # Convert graph to provider1's format and store it
    workflow1 = provider1.from_portable_graph(graph)
    await state_manager.set("current_workflow", workflow1)
    
    # Create executor with provider1
    executor = WorkflowExecutor(provider1, state_manager, tool_registry, event_bus)
    
    # Create second provider
    provider2 = MockTestProvider({"api_key": "test2"}, provider_id="provider2")
    
    # Switch to provider2
    await executor.switch_provider(provider2)
    
    # Verify provider was switched
    assert executor.provider is provider2
    
    # Verify state was transferred
    assert provider2._state == state_data, (
        "State should be preserved during provider switch"
    )
    
    # Verify workflow was transferred
    current_workflow = await state_manager.get("current_workflow")
    assert current_workflow is not None
    
    # Convert back to portable graph to verify
    restored_graph = provider2.to_portable_graph(current_workflow)
    
    # Verify graph structure is preserved
    assert len(restored_graph.nodes) == len(graph.nodes)
    assert len(restored_graph.edges) == len(graph.edges)
    
    for node_id in graph.nodes:
        assert node_id in restored_graph.nodes


# Test: Provider switching with empty workflow
@pytest.mark.asyncio
@given(state_dict_strategy())
@settings(max_examples=10, deadline=2000)
async def test_provider_switching_with_empty_workflow(state_data: Dict[str, Any]):
    """
    Test that provider switching works when no workflow is stored.
    
    Validates: Requirements 5.1-5.8
    """
    # Setup
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend)
    event_bus = EventBus()
    tool_registry = ToolRegistry()
    
    # Create first provider with state but no workflow
    provider1 = MockTestProvider({"api_key": "test1"}, provider_id="provider1")
    provider1._state = state_data.copy()
    
    # Create executor with provider1 (no workflow stored)
    executor = WorkflowExecutor(provider1, state_manager, tool_registry, event_bus)
    
    # Create second provider
    provider2 = MockTestProvider({"api_key": "test2"}, provider_id="provider2")
    
    # Switch to provider2 (should handle empty workflow gracefully)
    await executor.switch_provider(provider2)
    
    # Verify provider was switched
    assert executor.provider is provider2
    
    # Verify state was transferred
    assert provider2._state == state_data


# Test: Provider switching emits event
@pytest.mark.asyncio
@given(state_dict_strategy())
@settings(max_examples=10, deadline=2000)
async def test_provider_switching_emits_event(state_data: Dict[str, Any]):
    """
    Test that provider switching emits ProviderSwitched event.
    
    Validates: Requirement 5.8
    """
    # Setup
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend)
    event_bus = EventBus()
    tool_registry = ToolRegistry()
    
    # Track events
    events_received = []
    
    async def event_handler(event):
        events_received.append(event)
    
    event_bus.subscribe(EventType.PROVIDER_SWITCHED, event_handler)
    
    # Create providers
    provider1 = MockTestProvider({"api_key": "test1"}, provider_id="provider1")
    provider1._state = state_data.copy()
    
    provider2 = MockTestProvider({"api_key": "test2"}, provider_id="provider2")
    
    # Create executor and switch
    executor = WorkflowExecutor(provider1, state_manager, tool_registry, event_bus)
    await executor.switch_provider(provider2)
    
    # Verify event was emitted
    assert len(events_received) == 1
    event = events_received[0]
    
    assert event.event_type == EventType.PROVIDER_SWITCHED
    assert event.data["old_provider"] == "MockTestProvider"
    assert event.data["new_provider"] == "MockTestProvider"


# Test: Provider switching with invalid graph fails
@pytest.mark.asyncio
async def test_provider_switching_with_invalid_graph_fails():
    """
    Test that provider switching fails when portable graph validation fails.
    
    Validates: Requirement 5.3
    """
    # Setup
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend)
    event_bus = EventBus()
    tool_registry = ToolRegistry()
    
    # Create a provider that returns an invalid graph
    class InvalidGraphProvider(MockTestProvider):
        def to_portable_graph(self, workflow: Any) -> PortableExecutionGraph:
            # Create a graph with a cycle
            graph = PortableExecutionGraph()
            
            node1 = PEGNode(
                node_id="node1",
                node_type="agent",
                config={},
                inputs=[],
                outputs=[],
            )
            node2 = PEGNode(
                node_id="node2",
                node_type="agent",
                config={},
                inputs=[],
                outputs=[],
            )
            
            graph.add_node(node1)
            graph.add_node(node2)
            
            # Create cycle
            graph.add_edge(PEGEdge(source="node1", target="node2"))
            graph.add_edge(PEGEdge(source="node2", target="node1"))
            
            return graph
    
    # Create providers
    provider1 = InvalidGraphProvider({"api_key": "test1"}, provider_id="provider1")
    await state_manager.set("current_workflow", {"test": "workflow"})
    
    provider2 = MockTestProvider({"api_key": "test2"}, provider_id="provider2")
    
    # Create executor
    executor = WorkflowExecutor(provider1, state_manager, tool_registry, event_bus)
    
    # Switch should fail due to invalid graph
    with pytest.raises(ProviderSwitchError, match="validation failed"):
        await executor.switch_provider(provider2)
    
    # Verify provider was NOT switched (rollback)
    assert executor.provider is provider1


# Test: Provider switching failure rolls back
@pytest.mark.asyncio
@given(state_dict_strategy())
@settings(max_examples=10, deadline=2000)
async def test_provider_switching_failure_rolls_back(state_data: Dict[str, Any]):
    """
    Test that provider switching rolls back on failure.
    
    Validates: Requirement 15.8 (error recovery)
    """
    # Setup
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend)
    event_bus = EventBus()
    tool_registry = ToolRegistry()
    
    # Create a provider that fails on import_state
    class FailingProvider(MockTestProvider):
        def import_state(self, state: Dict[str, Any]) -> None:
            raise RuntimeError("Import failed")
    
    # Create providers
    provider1 = MockTestProvider({"api_key": "test1"}, provider_id="provider1")
    provider1._state = state_data.copy()
    
    provider2 = FailingProvider({"api_key": "test2"}, provider_id="provider2")
    
    # Create executor
    executor = WorkflowExecutor(provider1, state_manager, tool_registry, event_bus)
    
    # Switch should fail
    with pytest.raises(ProviderSwitchError, match="Import failed"):
        await executor.switch_provider(provider2)
    
    # Verify provider was NOT switched (rollback)
    assert executor.provider is provider1
    
    # Verify original state is intact
    assert provider1._state == state_data


# Test: State export includes all necessary data
@pytest.mark.asyncio
@given(state_dict_strategy())
@settings(max_examples=10, deadline=2000)
async def test_state_export_includes_all_data(state_data: Dict[str, Any]):
    """
    Test that exported state includes all necessary data for import.
    
    Validates: Requirement 3.5
    """
    provider = MockTestProvider({"api_key": "test"}, provider_id="test_provider")
    provider._state = state_data.copy()
    provider._workflow = {"test": "workflow"}
    
    # Export state
    exported = provider.export_state()
    
    # Verify all necessary fields are present
    assert "provider_id" in exported
    assert "state" in exported
    assert "workflow" in exported
    
    # Verify data is correct
    assert exported["provider_id"] == "test_provider"
    assert exported["state"] == state_data
    assert exported["workflow"] == {"test": "workflow"}


# Test: Empty graph conversion
@pytest.mark.asyncio
async def test_empty_graph_conversion():
    """
    Test that empty graphs can be converted between formats.
    
    Validates: Requirements 3.7, 3.8
    """
    provider = MockTestProvider({"api_key": "test"}, provider_id="test")
    
    # Create empty graph
    empty_graph = PortableExecutionGraph()
    
    # Convert to provider format
    workflow = provider.from_portable_graph(empty_graph)
    
    # Convert back
    restored_graph = provider.to_portable_graph(workflow)
    
    # Verify empty graph is preserved
    assert len(restored_graph.nodes) == 0
    assert len(restored_graph.edges) == 0


# Test: Graph metadata preservation
@pytest.mark.asyncio
@given(
    simple_graph_strategy(),
    st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.text(max_size=50),
        min_size=1,
        max_size=5
    )
)
@settings(max_examples=10, deadline=2000)
async def test_graph_metadata_preservation(
    graph: PortableExecutionGraph,
    metadata: Dict[str, str]
):
    """
    Test that graph metadata is preserved during conversion.
    
    Validates: Requirements 3.7, 3.8
    """
    # Set metadata
    graph.metadata = metadata
    
    provider = MockTestProvider({"api_key": "test"}, provider_id="test")
    
    # Convert to provider format and back
    workflow = provider.from_portable_graph(graph)
    restored_graph = provider.to_portable_graph(workflow)
    
    # Verify metadata is preserved
    assert restored_graph.metadata == metadata
