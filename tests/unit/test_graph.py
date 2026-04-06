"""Unit tests for Portable Execution Graph (PEG)."""

import json
import pytest
from agentlegatus.core.graph import PEGNode, PEGEdge, PortableExecutionGraph


class TestPEGNode:
    """Tests for PEGNode dataclass."""

    def test_node_creation(self):
        """Test creating a PEGNode."""
        node = PEGNode(
            node_id="node1",
            node_type="agent",
            config={"model": "gpt-4"},
            inputs=["input1"],
            outputs=["output1"],
        )
        assert node.node_id == "node1"
        assert node.node_type == "agent"
        assert node.config == {"model": "gpt-4"}
        assert node.inputs == ["input1"]
        assert node.outputs == ["output1"]

    def test_node_serialization(self):
        """Test node serialization to dict."""
        node = PEGNode(
            node_id="node1",
            node_type="tool",
            config={"name": "calculator"},
            inputs=["a", "b"],
            outputs=["result"],
        )
        data = node.to_dict()
        assert data["node_id"] == "node1"
        assert data["node_type"] == "tool"
        assert data["config"] == {"name": "calculator"}
        assert data["inputs"] == ["a", "b"]
        assert data["outputs"] == ["result"]

    def test_node_deserialization(self):
        """Test node deserialization from dict."""
        data = {
            "node_id": "node2",
            "node_type": "condition",
            "config": {"expression": "x > 0"},
            "inputs": ["x"],
            "outputs": ["result"],
        }
        node = PEGNode.from_dict(data)
        assert node.node_id == "node2"
        assert node.node_type == "condition"
        assert node.config == {"expression": "x > 0"}
        assert node.inputs == ["x"]
        assert node.outputs == ["result"]


class TestPEGEdge:
    """Tests for PEGEdge dataclass."""

    def test_edge_creation(self):
        """Test creating a PEGEdge."""
        edge = PEGEdge(source="node1", target="node2")
        assert edge.source == "node1"
        assert edge.target == "node2"
        assert edge.condition is None

    def test_edge_with_condition(self):
        """Test creating a PEGEdge with condition."""
        edge = PEGEdge(source="node1", target="node2", condition="x > 0")
        assert edge.source == "node1"
        assert edge.target == "node2"
        assert edge.condition == "x > 0"

    def test_edge_serialization(self):
        """Test edge serialization to dict."""
        edge = PEGEdge(source="node1", target="node2", condition="success")
        data = edge.to_dict()
        assert data["source"] == "node1"
        assert data["target"] == "node2"
        assert data["condition"] == "success"

    def test_edge_deserialization(self):
        """Test edge deserialization from dict."""
        data = {"source": "node1", "target": "node2", "condition": "failure"}
        edge = PEGEdge.from_dict(data)
        assert edge.source == "node1"
        assert edge.target == "node2"
        assert edge.condition == "failure"


class TestPortableExecutionGraph:
    """Tests for PortableExecutionGraph."""

    def test_graph_initialization(self):
        """Test creating an empty graph."""
        graph = PortableExecutionGraph()
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert len(graph.metadata) == 0

    def test_add_node(self):
        """Test adding nodes to graph."""
        graph = PortableExecutionGraph()
        node1 = PEGNode("node1", "agent", {"model": "gpt-4"})
        node2 = PEGNode("node2", "tool", {"name": "calculator"})
        
        graph.add_node(node1)
        graph.add_node(node2)
        
        assert len(graph.nodes) == 2
        assert "node1" in graph.nodes
        assert "node2" in graph.nodes

    def test_add_duplicate_node(self):
        """Test that adding duplicate node raises error."""
        graph = PortableExecutionGraph()
        node = PEGNode("node1", "agent", {})
        
        graph.add_node(node)
        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(node)

    def test_add_edge(self):
        """Test adding edges to graph."""
        graph = PortableExecutionGraph()
        node1 = PEGNode("node1", "agent", {})
        node2 = PEGNode("node2", "agent", {})
        
        graph.add_node(node1)
        graph.add_node(node2)
        
        edge = PEGEdge("node1", "node2")
        graph.add_edge(edge)
        
        assert len(graph.edges) == 1
        assert graph.edges[0].source == "node1"
        assert graph.edges[0].target == "node2"

    def test_add_edge_invalid_source(self):
        """Test that adding edge with invalid source raises error."""
        graph = PortableExecutionGraph()
        node = PEGNode("node1", "agent", {})
        graph.add_node(node)
        
        edge = PEGEdge("nonexistent", "node1")
        with pytest.raises(ValueError, match="does not exist"):
            graph.add_edge(edge)

    def test_add_edge_invalid_target(self):
        """Test that adding edge with invalid target raises error."""
        graph = PortableExecutionGraph()
        node = PEGNode("node1", "agent", {})
        graph.add_node(node)
        
        edge = PEGEdge("node1", "nonexistent")
        with pytest.raises(ValueError, match="does not exist"):
            graph.add_edge(edge)

    def test_remove_node(self):
        """Test removing a node and its edges."""
        graph = PortableExecutionGraph()
        node1 = PEGNode("node1", "agent", {})
        node2 = PEGNode("node2", "agent", {})
        node3 = PEGNode("node3", "agent", {})
        
        graph.add_node(node1)
        graph.add_node(node2)
        graph.add_node(node3)
        
        graph.add_edge(PEGEdge("node1", "node2"))
        graph.add_edge(PEGEdge("node2", "node3"))
        
        result = graph.remove_node("node2")
        
        assert result is True
        assert "node2" not in graph.nodes
        assert len(graph.edges) == 0  # Both edges should be removed

    def test_remove_nonexistent_node(self):
        """Test removing a node that doesn't exist."""
        graph = PortableExecutionGraph()
        result = graph.remove_node("nonexistent")
        assert result is False

    def test_get_node(self):
        """Test getting a node by ID."""
        graph = PortableExecutionGraph()
        node = PEGNode("node1", "agent", {"model": "gpt-4"})
        graph.add_node(node)
        
        retrieved = graph.get_node("node1")
        assert retrieved is not None
        assert retrieved.node_id == "node1"
        assert retrieved.config == {"model": "gpt-4"}

    def test_get_nonexistent_node(self):
        """Test getting a node that doesn't exist."""
        graph = PortableExecutionGraph()
        result = graph.get_node("nonexistent")
        assert result is None

    def test_get_successors(self):
        """Test getting successor nodes."""
        graph = PortableExecutionGraph()
        for i in range(4):
            graph.add_node(PEGNode(f"node{i}", "agent", {}))
        
        graph.add_edge(PEGEdge("node0", "node1"))
        graph.add_edge(PEGEdge("node0", "node2"))
        graph.add_edge(PEGEdge("node1", "node3"))
        
        successors = graph.get_successors("node0")
        assert len(successors) == 2
        assert "node1" in successors
        assert "node2" in successors

    def test_get_predecessors(self):
        """Test getting predecessor nodes."""
        graph = PortableExecutionGraph()
        for i in range(4):
            graph.add_node(PEGNode(f"node{i}", "agent", {}))
        
        graph.add_edge(PEGEdge("node0", "node2"))
        graph.add_edge(PEGEdge("node1", "node2"))
        graph.add_edge(PEGEdge("node2", "node3"))
        
        predecessors = graph.get_predecessors("node2")
        assert len(predecessors) == 2
        assert "node0" in predecessors
        assert "node1" in predecessors

    def test_validate_valid_dag(self):
        """Test validation of a valid DAG."""
        graph = PortableExecutionGraph()
        for i in range(3):
            graph.add_node(PEGNode(f"node{i}", "agent", {}))
        
        graph.add_edge(PEGEdge("node0", "node1"))
        graph.add_edge(PEGEdge("node1", "node2"))
        
        is_valid, errors = graph.validate()
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_cycle_detection(self):
        """Test that validation detects cycles."""
        graph = PortableExecutionGraph()
        for i in range(3):
            graph.add_node(PEGNode(f"node{i}", "agent", {}))
        
        graph.add_edge(PEGEdge("node0", "node1"))
        graph.add_edge(PEGEdge("node1", "node2"))
        graph.add_edge(PEGEdge("node2", "node0"))  # Creates cycle
        
        is_valid, errors = graph.validate()
        assert is_valid is False
        assert any("cycle" in error.lower() for error in errors)

    def test_validate_self_loop(self):
        """Test that validation detects self-loops."""
        graph = PortableExecutionGraph()
        graph.add_node(PEGNode("node0", "agent", {}))
        graph.add_edge(PEGEdge("node0", "node0"))  # Self-loop
        
        is_valid, errors = graph.validate()
        assert is_valid is False
        assert any("cycle" in error.lower() for error in errors)

    def test_graph_serialization(self):
        """Test graph serialization to dict."""
        graph = PortableExecutionGraph()
        graph.add_node(PEGNode("node1", "agent", {"model": "gpt-4"}))
        graph.add_node(PEGNode("node2", "tool", {"name": "calc"}))
        graph.add_edge(PEGEdge("node1", "node2"))
        graph.metadata = {"version": "1.0"}
        
        data = graph.to_dict()
        
        assert "nodes" in data
        assert "edges" in data
        assert "metadata" in data
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        assert data["metadata"]["version"] == "1.0"

    def test_graph_deserialization(self):
        """Test graph deserialization from dict."""
        data = {
            "nodes": {
                "node1": {
                    "node_id": "node1",
                    "node_type": "agent",
                    "config": {"model": "gpt-4"},
                    "inputs": [],
                    "outputs": [],
                },
                "node2": {
                    "node_id": "node2",
                    "node_type": "tool",
                    "config": {"name": "calc"},
                    "inputs": [],
                    "outputs": [],
                },
            },
            "edges": [{"source": "node1", "target": "node2"}],
            "metadata": {"version": "1.0"},
        }
        
        graph = PortableExecutionGraph.from_dict(data)
        
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.metadata["version"] == "1.0"
        assert graph.get_node("node1").node_type == "agent"
        assert graph.get_node("node2").node_type == "tool"

    def test_json_serialization(self):
        """Test graph serialization to JSON."""
        graph = PortableExecutionGraph()
        graph.add_node(PEGNode("node1", "agent", {"model": "gpt-4"}))
        graph.add_node(PEGNode("node2", "tool", {"name": "calc"}))
        graph.add_edge(PEGEdge("node1", "node2"))
        
        json_str = graph.to_json()
        
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert "nodes" in data
        assert "edges" in data

    def test_json_deserialization(self):
        """Test graph deserialization from JSON."""
        json_str = """
        {
            "nodes": {
                "node1": {
                    "node_id": "node1",
                    "node_type": "agent",
                    "config": {"model": "gpt-4"},
                    "inputs": [],
                    "outputs": []
                }
            },
            "edges": [],
            "metadata": {}
        }
        """
        
        graph = PortableExecutionGraph.from_json(json_str)
        
        assert len(graph.nodes) == 1
        assert graph.get_node("node1").node_type == "agent"

    def test_round_trip_serialization(self):
        """Test that serialization and deserialization preserve graph."""
        original = PortableExecutionGraph()
        original.add_node(PEGNode("node1", "agent", {"model": "gpt-4"}))
        original.add_node(PEGNode("node2", "tool", {"name": "calc"}))
        original.add_edge(PEGEdge("node1", "node2", "success"))
        original.metadata = {"version": "1.0", "author": "test"}
        
        # Round trip through JSON
        json_str = original.to_json()
        restored = PortableExecutionGraph.from_json(json_str)
        
        assert len(restored.nodes) == len(original.nodes)
        assert len(restored.edges) == len(original.edges)
        assert restored.metadata == original.metadata
        assert restored.get_node("node1").config == original.get_node("node1").config
        assert restored.edges[0].condition == original.edges[0].condition

    # --- Additional tests for comprehensive coverage ---

    def test_add_node_stores_by_node_id(self):
        """Test that add_node stores the node keyed by its node_id."""
        graph = PortableExecutionGraph()
        node = PEGNode("mynode", "agent", {"key": "val"})
        graph.add_node(node)
        assert graph.nodes["mynode"] is node

    def test_add_edge_stores_edge_object(self):
        """Test that add_edge appends the edge to the edges list."""
        graph = PortableExecutionGraph()
        graph.add_node(PEGNode("a", "agent", {}))
        graph.add_node(PEGNode("b", "agent", {}))
        edge = PEGEdge("a", "b", condition="ok")
        graph.add_edge(edge)
        assert graph.edges[0] is edge

    def test_remove_node_preserves_unrelated_edges(self):
        """Test that removing a node only removes edges connected to it."""
        graph = PortableExecutionGraph()
        for nid in ["a", "b", "c", "d"]:
            graph.add_node(PEGNode(nid, "agent", {}))
        graph.add_edge(PEGEdge("a", "b"))
        graph.add_edge(PEGEdge("c", "d"))
        graph.add_edge(PEGEdge("a", "c"))

        graph.remove_node("a")

        assert len(graph.edges) == 1
        assert graph.edges[0].source == "c"
        assert graph.edges[0].target == "d"

    def test_get_successors_empty(self):
        """Test get_successors returns empty list for node with no outgoing edges."""
        graph = PortableExecutionGraph()
        graph.add_node(PEGNode("solo", "agent", {}))
        assert graph.get_successors("solo") == []

    def test_get_predecessors_empty(self):
        """Test get_predecessors returns empty list for node with no incoming edges."""
        graph = PortableExecutionGraph()
        graph.add_node(PEGNode("solo", "agent", {}))
        assert graph.get_predecessors("solo") == []

    def test_get_successors_for_nonexistent_node(self):
        """Test get_successors returns empty list for a node not in the graph."""
        graph = PortableExecutionGraph()
        assert graph.get_successors("ghost") == []

    def test_get_predecessors_for_nonexistent_node(self):
        """Test get_predecessors returns empty list for a node not in the graph."""
        graph = PortableExecutionGraph()
        assert graph.get_predecessors("ghost") == []

    def test_validate_empty_graph(self):
        """Test that an empty graph is valid."""
        graph = PortableExecutionGraph()
        is_valid, errors = graph.validate()
        assert is_valid is True
        assert errors == []

    def test_validate_single_node(self):
        """Test that a single-node graph with no edges is valid."""
        graph = PortableExecutionGraph()
        graph.add_node(PEGNode("only", "agent", {}))
        is_valid, errors = graph.validate()
        assert is_valid is True
        assert errors == []

    def test_validate_disconnected_nodes(self):
        """Test that a graph with disconnected nodes (no edges) is valid."""
        graph = PortableExecutionGraph()
        for i in range(5):
            graph.add_node(PEGNode(f"n{i}", "agent", {}))
        is_valid, errors = graph.validate()
        assert is_valid is True
        assert errors == []

    def test_validate_detects_invalid_edge_references(self):
        """Test that validate detects edges referencing non-existent nodes."""
        graph = PortableExecutionGraph()
        graph.add_node(PEGNode("a", "agent", {}))
        # Manually inject a bad edge (bypassing add_edge validation)
        graph.edges.append(PEGEdge("a", "missing_target"))
        graph.edges.append(PEGEdge("missing_source", "a"))

        is_valid, errors = graph.validate()
        assert is_valid is False
        assert any("missing_target" in e for e in errors)
        assert any("missing_source" in e for e in errors)

    def test_validate_complex_dag(self):
        """Test validation of a complex but valid DAG (diamond shape)."""
        graph = PortableExecutionGraph()
        for nid in ["start", "left", "right", "end"]:
            graph.add_node(PEGNode(nid, "agent", {}))
        graph.add_edge(PEGEdge("start", "left"))
        graph.add_edge(PEGEdge("start", "right"))
        graph.add_edge(PEGEdge("left", "end"))
        graph.add_edge(PEGEdge("right", "end"))

        is_valid, errors = graph.validate()
        assert is_valid is True
        assert errors == []

    def test_edge_serialization_without_condition(self):
        """Test that edge serialization omits condition when it's None."""
        edge = PEGEdge(source="a", target="b")
        data = edge.to_dict()
        assert "condition" not in data
        assert data == {"source": "a", "target": "b"}

    def test_edge_deserialization_without_condition(self):
        """Test edge deserialization when condition is absent."""
        data = {"source": "x", "target": "y"}
        edge = PEGEdge.from_dict(data)
        assert edge.condition is None

    def test_node_deserialization_missing_optional_fields(self):
        """Test node deserialization when inputs/outputs are missing."""
        data = {
            "node_id": "n1",
            "node_type": "agent",
            "config": {},
        }
        node = PEGNode.from_dict(data)
        assert node.inputs == []
        assert node.outputs == []

    def test_node_default_inputs_outputs(self):
        """Test that PEGNode defaults inputs and outputs to empty lists."""
        node = PEGNode(node_id="n", node_type="agent", config={})
        assert node.inputs == []
        assert node.outputs == []

    def test_dict_round_trip_preserves_all_node_fields(self):
        """Test to_dict/from_dict round-trip preserves all node fields."""
        graph = PortableExecutionGraph()
        node = PEGNode(
            "n1", "condition",
            config={"expr": "x > 0", "nested": {"a": [1, 2, 3]}},
            inputs=["in1", "in2"],
            outputs=["out1"],
        )
        graph.add_node(node)
        graph.metadata = {"key": "value"}

        restored = PortableExecutionGraph.from_dict(graph.to_dict())
        rn = restored.get_node("n1")
        assert rn.node_id == "n1"
        assert rn.node_type == "condition"
        assert rn.config == {"expr": "x > 0", "nested": {"a": [1, 2, 3]}}
        assert rn.inputs == ["in1", "in2"]
        assert rn.outputs == ["out1"]
        assert restored.metadata == {"key": "value"}

    def test_from_dict_empty_data(self):
        """Test from_dict with empty/minimal data."""
        graph = PortableExecutionGraph.from_dict({})
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0
        assert graph.metadata == {}

    def test_json_round_trip_with_edge_conditions(self):
        """Test JSON round-trip preserves edge conditions."""
        graph = PortableExecutionGraph()
        graph.add_node(PEGNode("a", "agent", {}))
        graph.add_node(PEGNode("b", "agent", {}))
        graph.add_node(PEGNode("c", "agent", {}))
        graph.add_edge(PEGEdge("a", "b", condition="success"))
        graph.add_edge(PEGEdge("a", "c"))  # No condition

        restored = PortableExecutionGraph.from_json(graph.to_json())
        conditions = [e.condition for e in restored.edges]
        assert "success" in conditions
        assert None in conditions

    def test_json_round_trip_metadata(self):
        """Test JSON round-trip preserves metadata."""
        graph = PortableExecutionGraph()
        graph.metadata = {"version": "2.0", "tags": ["test", "ci"], "count": 42}

        restored = PortableExecutionGraph.from_json(graph.to_json())
        assert restored.metadata == {"version": "2.0", "tags": ["test", "ci"], "count": 42}

    def test_multiple_edges_between_same_nodes(self):
        """Test that multiple edges between the same pair of nodes are allowed."""
        graph = PortableExecutionGraph()
        graph.add_node(PEGNode("a", "agent", {}))
        graph.add_node(PEGNode("b", "agent", {}))
        graph.add_edge(PEGEdge("a", "b", condition="success"))
        graph.add_edge(PEGEdge("a", "b", condition="failure"))

        assert len(graph.edges) == 2
        successors = graph.get_successors("a")
        assert successors == ["b", "b"]

    def test_remove_node_returns_true_for_existing(self):
        """Test remove_node returns True when node exists."""
        graph = PortableExecutionGraph()
        graph.add_node(PEGNode("x", "agent", {}))
        assert graph.remove_node("x") is True
        assert graph.get_node("x") is None

    def test_remove_node_then_readd(self):
        """Test that a removed node can be re-added."""
        graph = PortableExecutionGraph()
        graph.add_node(PEGNode("x", "agent", {"v": 1}))
        graph.remove_node("x")
        graph.add_node(PEGNode("x", "tool", {"v": 2}))
        assert graph.get_node("x").node_type == "tool"
        assert graph.get_node("x").config == {"v": 2}
