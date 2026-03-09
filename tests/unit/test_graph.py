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
