"""Property-based tests for PortableExecutionGraph."""

import json
from typing import Any, Dict, List, Set, Tuple

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agentlegatus.core.graph import (
    PEGEdge,
    PEGNode,
    PortableExecutionGraph,
)


# Helper strategies
@st.composite
def node_id_strategy(draw):
    """Generate valid node IDs."""
    return draw(st.text(
        min_size=1,
        max_size=30,
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters="_-"
        )
    ))


@st.composite
def node_type_strategy(draw):
    """Generate valid node types."""
    return draw(st.sampled_from(["agent", "tool", "condition", "loop"]))


@st.composite
def node_config_strategy(draw):
    """Generate node configuration dictionaries."""
    return draw(st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.one_of(
            st.integers(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(max_size=50),
            st.booleans(),
        ),
        max_size=5
    ))


@st.composite
def peg_node_strategy(draw):
    """Generate a valid PEGNode."""
    node_id = draw(node_id_strategy())
    node_type = draw(node_type_strategy())
    config = draw(node_config_strategy())
    inputs = draw(st.lists(st.text(max_size=20), max_size=3))
    outputs = draw(st.lists(st.text(max_size=20), max_size=3))
    
    return PEGNode(
        node_id=node_id,
        node_type=node_type,
        config=config,
        inputs=inputs,
        outputs=outputs,
    )


@st.composite
def simple_graph_strategy(draw, min_nodes=1, max_nodes=10):
    """Generate a simple graph without cycles."""
    num_nodes = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    
    # Generate unique node IDs
    node_ids = []
    for i in range(num_nodes):
        node_id = f"node_{i}"
        node_ids.append(node_id)
    
    # Create nodes
    nodes = []
    for node_id in node_ids:
        node_type = draw(node_type_strategy())
        config = draw(node_config_strategy())
        node = PEGNode(
            node_id=node_id,
            node_type=node_type,
            config=config,
            inputs=[],
            outputs=[],
        )
        nodes.append(node)
    
    # Create edges (ensuring no cycles by only connecting to later nodes)
    edges = []
    for i, source_id in enumerate(node_ids[:-1]):
        # Randomly connect to some later nodes
        num_edges = draw(st.integers(min_value=0, max_value=min(2, len(node_ids) - i - 1)))
        if num_edges > 0:
            targets = draw(st.lists(
                st.sampled_from(node_ids[i+1:]),
                min_size=num_edges,
                max_size=num_edges,
                unique=True
            ))
            for target_id in targets:
                condition = draw(st.one_of(st.none(), st.text(max_size=20)))
                edges.append(PEGEdge(source=source_id, target=target_id, condition=condition))
    
    return nodes, edges


@st.composite
def cyclic_graph_strategy(draw, min_nodes=2, max_nodes=6):
    """Generate a graph with at least one cycle."""
    num_nodes = draw(st.integers(min_value=min_nodes, max_value=max_nodes))
    
    # Generate unique node IDs
    node_ids = [f"node_{i}" for i in range(num_nodes)]
    
    # Create nodes
    nodes = []
    for node_id in node_ids:
        node_type = draw(node_type_strategy())
        config = draw(node_config_strategy())
        node = PEGNode(
            node_id=node_id,
            node_type=node_type,
            config=config,
            inputs=[],
            outputs=[],
        )
        nodes.append(node)
    
    # Create a cycle: node_0 -> node_1 -> ... -> node_n -> node_0
    edges = []
    for i in range(num_nodes):
        source = node_ids[i]
        target = node_ids[(i + 1) % num_nodes]
        edges.append(PEGEdge(source=source, target=target))
    
    # Optionally add more edges
    num_extra_edges = draw(st.integers(min_value=0, max_value=3))
    for _ in range(num_extra_edges):
        source = draw(st.sampled_from(node_ids))
        target = draw(st.sampled_from(node_ids))
        if source != target:
            edges.append(PEGEdge(source=source, target=target))
    
    return nodes, edges


# Property 16: Graph Serialization Round-Trip
@given(simple_graph_strategy(min_nodes=1, max_nodes=15))
@settings(max_examples=100, deadline=2000)
def test_property_16_graph_serialization_round_trip(graph_data: Tuple[List[PEGNode], List[PEGEdge]]):
    """
    Property 16: Graph Serialization Round-Trip
    
    For any valid PortableExecutionGraph, serializing to JSON then 
    deserializing produces an equivalent graph with the same nodes and edges.
    
    Validates: Requirements 12.9, 12.10
    """
    nodes, edges = graph_data
    
    # Create graph
    graph = PortableExecutionGraph()
    
    # Add nodes
    for node in nodes:
        graph.add_node(node)
    
    # Add edges
    for edge in edges:
        graph.add_edge(edge)
    
    # Add some metadata
    graph.metadata = {"test": "metadata", "version": "1.0"}
    
    # Serialize to JSON
    json_str = graph.to_json()
    
    # Deserialize from JSON
    restored_graph = PortableExecutionGraph.from_json(json_str)
    
    # Verify nodes match
    assert len(restored_graph.nodes) == len(graph.nodes), (
        f"Number of nodes should match. Expected {len(graph.nodes)}, got {len(restored_graph.nodes)}"
    )
    
    for node_id, original_node in graph.nodes.items():
        assert node_id in restored_graph.nodes, f"Node {node_id} should exist in restored graph"
        restored_node = restored_graph.nodes[node_id]
        
        assert restored_node.node_id == original_node.node_id
        assert restored_node.node_type == original_node.node_type
        assert restored_node.config == original_node.config
        assert restored_node.inputs == original_node.inputs
        assert restored_node.outputs == original_node.outputs
    
    # Verify edges match
    assert len(restored_graph.edges) == len(graph.edges), (
        f"Number of edges should match. Expected {len(graph.edges)}, got {len(restored_graph.edges)}"
    )
    
    for i, original_edge in enumerate(graph.edges):
        restored_edge = restored_graph.edges[i]
        assert restored_edge.source == original_edge.source
        assert restored_edge.target == original_edge.target
        assert restored_edge.condition == original_edge.condition
    
    # Verify metadata matches
    assert restored_graph.metadata == graph.metadata


# Property 17: Graph Node Removal Completeness
@given(
    simple_graph_strategy(min_nodes=3, max_nodes=10),
    st.integers(min_value=0, max_value=9)
)
@settings(max_examples=100, deadline=2000)
def test_property_17_graph_node_removal_completeness(
    graph_data: Tuple[List[PEGNode], List[PEGEdge]],
    node_index: int
):
    """
    Property 17: Graph Node Removal Completeness
    
    For any graph and node, removing the node also removes all edges 
    connected to that node.
    
    Validates: Requirements 12.3
    """
    nodes, edges = graph_data
    
    # Ensure we have at least one node
    assume(len(nodes) > 0)
    
    # Select a valid node index
    node_index = node_index % len(nodes)
    node_to_remove = nodes[node_index]
    
    # Create graph
    graph = PortableExecutionGraph()
    
    # Add all nodes
    for node in nodes:
        graph.add_node(node)
    
    # Add all edges
    for edge in edges:
        graph.add_edge(edge)
    
    # Count edges connected to the node before removal
    edges_before = len(graph.edges)
    connected_edges_count = sum(
        1 for edge in graph.edges
        if edge.source == node_to_remove.node_id or edge.target == node_to_remove.node_id
    )
    
    # Remove the node
    result = graph.remove_node(node_to_remove.node_id)
    
    # Verify removal was successful
    assert result is True, "remove_node should return True for existing node"
    
    # Verify node is removed
    assert node_to_remove.node_id not in graph.nodes, (
        f"Node {node_to_remove.node_id} should be removed from graph"
    )
    
    # Verify all connected edges are removed
    for edge in graph.edges:
        assert edge.source != node_to_remove.node_id, (
            f"Edge with source {node_to_remove.node_id} should be removed"
        )
        assert edge.target != node_to_remove.node_id, (
            f"Edge with target {node_to_remove.node_id} should be removed"
        )
    
    # Verify edge count is correct
    expected_edges_after = edges_before - connected_edges_count
    assert len(graph.edges) == expected_edges_after, (
        f"Expected {expected_edges_after} edges after removal, got {len(graph.edges)}"
    )
    
    # Verify removing non-existent node returns False
    result_second = graph.remove_node(node_to_remove.node_id)
    assert result_second is False, "remove_node should return False for non-existent node"


# Property 25: Graph Cycle Detection
@given(cyclic_graph_strategy(min_nodes=2, max_nodes=8))
@settings(max_examples=100, deadline=2000)
def test_property_25_graph_cycle_detection(graph_data: Tuple[List[PEGNode], List[PEGEdge]]):
    """
    Property 25: Graph Cycle Detection
    
    For any PortableExecutionGraph with a cycle, validation returns 
    False with cycle detection errors.
    
    Validates: Requirements 12.7, 29.1
    """
    nodes, edges = graph_data
    
    # Create graph with cycle
    graph = PortableExecutionGraph()
    
    # Add nodes
    for node in nodes:
        graph.add_node(node)
    
    # Add edges (includes cycle)
    for edge in edges:
        graph.add_edge(edge)
    
    # Validate graph
    is_valid, errors = graph.validate()
    
    # Verify validation fails
    assert is_valid is False, (
        "Graph with cycle should fail validation"
    )
    
    # Verify error message mentions cycles
    assert len(errors) > 0, "Validation should return error messages"
    assert any("cycle" in error.lower() for error in errors), (
        f"Error messages should mention cycles. Got: {errors}"
    )


# Property 26: Graph Reference Validation
@given(
    simple_graph_strategy(min_nodes=2, max_nodes=8),
    node_id_strategy(),
    node_id_strategy(),
)
@settings(max_examples=100, deadline=2000)
def test_property_26_graph_reference_validation(
    graph_data: Tuple[List[PEGNode], List[PEGEdge]],
    invalid_source: str,
    invalid_target: str,
):
    """
    Property 26: Graph Reference Validation
    
    For any PortableExecutionGraph with invalid node references, 
    validation returns False with reference validation errors.
    
    Validates: Requirements 12.8, 29.2
    """
    nodes, edges = graph_data
    
    # Ensure invalid IDs don't accidentally match existing nodes
    existing_node_ids = {node.node_id for node in nodes}
    assume(invalid_source not in existing_node_ids)
    assume(invalid_target not in existing_node_ids)
    assume(invalid_source != invalid_target)
    
    # Create graph
    graph = PortableExecutionGraph()
    
    # Add nodes
    for node in nodes:
        graph.add_node(node)
    
    # Add valid edges
    for edge in edges:
        graph.add_edge(edge)
    
    # Manually add invalid edge (bypassing add_edge validation)
    invalid_edge = PEGEdge(source=invalid_source, target=invalid_target)
    graph.edges.append(invalid_edge)
    
    # Validate graph
    is_valid, errors = graph.validate()
    
    # Verify validation fails
    assert is_valid is False, (
        "Graph with invalid node references should fail validation"
    )
    
    # Verify error messages mention the invalid references
    assert len(errors) > 0, "Validation should return error messages"
    
    error_text = " ".join(errors).lower()
    assert "non-existent" in error_text or "invalid" in error_text or "reference" in error_text, (
        f"Error messages should mention invalid references. Got: {errors}"
    )
    
    # Verify the specific invalid node IDs are mentioned
    assert any(invalid_source in error for error in errors) or any(invalid_target in error for error in errors), (
        f"Error messages should mention the invalid node IDs. Got: {errors}"
    )


# Additional test: Valid acyclic graph passes validation
@given(simple_graph_strategy(min_nodes=1, max_nodes=10))
@settings(max_examples=50, deadline=2000)
def test_valid_acyclic_graph_passes_validation(graph_data: Tuple[List[PEGNode], List[PEGEdge]]):
    """
    Test that a valid acyclic graph passes validation.
    
    Validates: Requirements 29.4, 29.5
    """
    nodes, edges = graph_data
    
    # Create graph
    graph = PortableExecutionGraph()
    
    # Add nodes
    for node in nodes:
        graph.add_node(node)
    
    # Add edges
    for edge in edges:
        graph.add_edge(edge)
    
    # Validate graph
    is_valid, errors = graph.validate()
    
    # Verify validation succeeds
    assert is_valid is True, f"Valid acyclic graph should pass validation. Errors: {errors}"
    assert len(errors) == 0, f"Valid graph should have no errors. Got: {errors}"


# Additional test: Serialization preserves empty graph
def test_serialization_preserves_empty_graph():
    """
    Test that serialization works correctly for empty graphs.
    
    Validates: Requirements 12.9, 12.10
    """
    # Create empty graph
    graph = PortableExecutionGraph()
    graph.metadata = {"empty": True}
    
    # Serialize and deserialize
    json_str = graph.to_json()
    restored_graph = PortableExecutionGraph.from_json(json_str)
    
    # Verify structure
    assert len(restored_graph.nodes) == 0
    assert len(restored_graph.edges) == 0
    assert restored_graph.metadata == {"empty": True}


# Additional test: Dictionary serialization round-trip
@given(simple_graph_strategy(min_nodes=1, max_nodes=10))
@settings(max_examples=50, deadline=2000)
def test_dictionary_serialization_round_trip(graph_data: Tuple[List[PEGNode], List[PEGEdge]]):
    """
    Test that to_dict/from_dict round-trip works correctly.
    
    Validates: Requirements 12.9, 12.10
    """
    nodes, edges = graph_data
    
    # Create graph
    graph = PortableExecutionGraph()
    
    for node in nodes:
        graph.add_node(node)
    
    for edge in edges:
        graph.add_edge(edge)
    
    graph.metadata = {"test": "data"}
    
    # Serialize to dict
    graph_dict = graph.to_dict()
    
    # Deserialize from dict
    restored_graph = PortableExecutionGraph.from_dict(graph_dict)
    
    # Verify nodes match
    assert len(restored_graph.nodes) == len(graph.nodes)
    for node_id in graph.nodes:
        assert node_id in restored_graph.nodes
    
    # Verify edges match
    assert len(restored_graph.edges) == len(graph.edges)
    
    # Verify metadata matches
    assert restored_graph.metadata == graph.metadata


# Additional test: Get successors and predecessors
@given(simple_graph_strategy(min_nodes=3, max_nodes=8))
@settings(max_examples=50, deadline=2000)
def test_get_successors_and_predecessors(graph_data: Tuple[List[PEGNode], List[PEGEdge]]):
    """
    Test that get_successors and get_predecessors return correct nodes.
    
    Validates: Requirements 12.5, 12.6
    """
    nodes, edges = graph_data
    
    # Create graph
    graph = PortableExecutionGraph()
    
    for node in nodes:
        graph.add_node(node)
    
    for edge in edges:
        graph.add_edge(edge)
    
    # Verify successors and predecessors for each node
    for node in nodes:
        node_id = node.node_id
        
        # Get successors
        successors = graph.get_successors(node_id)
        expected_successors = [e.target for e in edges if e.source == node_id]
        assert set(successors) == set(expected_successors), (
            f"Successors for {node_id} should match. Expected {expected_successors}, got {successors}"
        )
        
        # Get predecessors
        predecessors = graph.get_predecessors(node_id)
        expected_predecessors = [e.source for e in edges if e.target == node_id]
        assert set(predecessors) == set(expected_predecessors), (
            f"Predecessors for {node_id} should match. Expected {expected_predecessors}, got {predecessors}"
        )


# Additional test: Cannot add duplicate node IDs
@given(peg_node_strategy())
@settings(max_examples=50, deadline=2000)
def test_cannot_add_duplicate_node_ids(node: PEGNode):
    """
    Test that adding a node with duplicate ID raises ValueError.
    
    Validates: Requirements 29.3
    """
    graph = PortableExecutionGraph()
    
    # Add node first time
    graph.add_node(node)
    
    # Try to add same node again
    with pytest.raises(ValueError, match="already exists"):
        graph.add_node(node)


# Additional test: Cannot add edge with non-existent nodes
@given(
    peg_node_strategy(),
    node_id_strategy(),
    node_id_strategy(),
)
@settings(max_examples=50, deadline=2000)
def test_cannot_add_edge_with_nonexistent_nodes(
    node: PEGNode,
    invalid_source: str,
    invalid_target: str,
):
    """
    Test that adding an edge with non-existent nodes raises ValueError.
    
    Validates: Requirements 12.2
    """
    # Ensure invalid IDs don't match the existing node
    assume(invalid_source != node.node_id)
    assume(invalid_target != node.node_id)
    
    graph = PortableExecutionGraph()
    graph.add_node(node)
    
    # Try to add edge with non-existent source
    with pytest.raises(ValueError, match="does not exist"):
        graph.add_edge(PEGEdge(source=invalid_source, target=node.node_id))
    
    # Try to add edge with non-existent target
    with pytest.raises(ValueError, match="does not exist"):
        graph.add_edge(PEGEdge(source=node.node_id, target=invalid_target))


# Additional test: Get node returns correct node or None
@given(simple_graph_strategy(min_nodes=1, max_nodes=5), node_id_strategy())
@settings(max_examples=50, deadline=2000)
def test_get_node_returns_correct_node_or_none(
    graph_data: Tuple[List[PEGNode], List[PEGEdge]],
    nonexistent_id: str,
):
    """
    Test that get_node returns the correct node or None.
    
    Validates: Requirements 12.4
    """
    nodes, edges = graph_data
    
    # Ensure nonexistent_id doesn't match any existing node
    existing_ids = {node.node_id for node in nodes}
    assume(nonexistent_id not in existing_ids)
    
    # Create graph
    graph = PortableExecutionGraph()
    
    for node in nodes:
        graph.add_node(node)
    
    # Test getting existing nodes
    for node in nodes:
        retrieved = graph.get_node(node.node_id)
        assert retrieved is not None
        assert retrieved.node_id == node.node_id
        assert retrieved.node_type == node.node_type
    
    # Test getting non-existent node
    retrieved = graph.get_node(nonexistent_id)
    assert retrieved is None
