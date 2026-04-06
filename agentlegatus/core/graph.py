"""Portable Execution Graph (PEG) for framework-agnostic workflow definitions."""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PEGNode:
    """Node in a Portable Execution Graph."""

    node_id: str
    node_type: str  # "agent", "tool", "condition", "loop"
    config: dict[str, Any]
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize node to dictionary.

        Returns:
            Dictionary representation of the node
        """
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "config": self.config,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PEGNode":
        """
        Deserialize node from dictionary.

        Args:
            data: Dictionary representation of the node

        Returns:
            PEGNode instance
        """
        return cls(
            node_id=data["node_id"],
            node_type=data["node_type"],
            config=data["config"],
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
        )


@dataclass
class PEGEdge:
    """Edge connecting nodes in a Portable Execution Graph."""

    source: str
    target: str
    condition: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize edge to dictionary.

        Returns:
            Dictionary representation of the edge
        """
        result = {
            "source": self.source,
            "target": self.target,
        }
        if self.condition is not None:
            result["condition"] = self.condition
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PEGEdge":
        """
        Deserialize edge from dictionary.

        Args:
            data: Dictionary representation of the edge

        Returns:
            PEGEdge instance
        """
        return cls(
            source=data["source"],
            target=data["target"],
            condition=data.get("condition"),
        )


class PortableExecutionGraph:
    """Framework-agnostic workflow representation."""

    def __init__(self):
        """Initialize empty PEG."""
        self.nodes: dict[str, PEGNode] = {}
        self.edges: list[PEGEdge] = []
        self.metadata: dict[str, Any] = {}

    def add_node(self, node: PEGNode) -> None:
        """
        Add a node to the graph.

        Args:
            node: PEGNode to add

        Raises:
            ValueError: If node with same ID already exists
        """
        if node.node_id in self.nodes:
            raise ValueError(f"Node with ID '{node.node_id}' already exists")
        self.nodes[node.node_id] = node

    def add_edge(self, edge: PEGEdge) -> None:
        """
        Add an edge to the graph.

        Args:
            edge: PEGEdge to add

        Raises:
            ValueError: If source or target node doesn't exist
        """
        if edge.source not in self.nodes:
            raise ValueError(f"Source node '{edge.source}' does not exist")
        if edge.target not in self.nodes:
            raise ValueError(f"Target node '{edge.target}' does not exist")
        self.edges.append(edge)

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node and all its connected edges.

        Args:
            node_id: ID of the node to remove

        Returns:
            True if node was removed, False if node didn't exist
        """
        if node_id not in self.nodes:
            return False

        # Remove the node
        del self.nodes[node_id]

        # Remove all edges connected to this node
        self.edges = [
            edge for edge in self.edges if edge.source != node_id and edge.target != node_id
        ]

        return True

    def get_node(self, node_id: str) -> PEGNode | None:
        """
        Get a node by ID.

        Args:
            node_id: ID of the node to retrieve

        Returns:
            PEGNode if found, None otherwise
        """
        return self.nodes.get(node_id)

    def get_successors(self, node_id: str) -> list[str]:
        """
        Get successor node IDs.

        Args:
            node_id: ID of the node

        Returns:
            List of successor node IDs
        """
        return [edge.target for edge in self.edges if edge.source == node_id]

    def get_predecessors(self, node_id: str) -> list[str]:
        """
        Get predecessor node IDs.

        Args:
            node_id: ID of the node

        Returns:
            List of predecessor node IDs
        """
        return [edge.source for edge in self.edges if edge.target == node_id]

    def _has_cycle_dfs(self, node_id: str, visited: set[str], rec_stack: set[str]) -> bool:
        """
        Detect cycles using depth-first search.

        Args:
            node_id: Current node being visited
            visited: Set of all visited nodes
            rec_stack: Set of nodes in current recursion stack

        Returns:
            True if cycle detected, False otherwise
        """
        visited.add(node_id)
        rec_stack.add(node_id)

        # Check all successors
        for successor in self.get_successors(node_id):
            if successor not in visited:
                if self._has_cycle_dfs(successor, visited, rec_stack):
                    return True
            elif successor in rec_stack:
                # Back edge found - cycle detected
                return True

        rec_stack.remove(node_id)
        return False

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate graph structure.

        Checks:
        - No cycles (DAG requirement)
        - All edge references point to existing nodes
        - All node IDs are unique (enforced by dict)

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check for cycles using DFS
        visited: set[str] = set()
        for node_id in self.nodes:
            if node_id not in visited:
                if self._has_cycle_dfs(node_id, visited, set()):
                    errors.append("Graph contains cycles")
                    break

        # Check for invalid node references in edges
        for edge in self.edges:
            if edge.source not in self.nodes:
                errors.append(f"Edge references non-existent source node: {edge.source}")
            if edge.target not in self.nodes:
                errors.append(f"Edge references non-existent target node: {edge.target}")

        return len(errors) == 0, errors

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize graph to dictionary.

        Returns:
            Dictionary representation of the graph
        """
        return {
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PortableExecutionGraph":
        """
        Deserialize graph from dictionary.

        Args:
            data: Dictionary representation of the graph

        Returns:
            PortableExecutionGraph instance
        """
        graph = cls()

        # Load nodes
        for node_id, node_data in data.get("nodes", {}).items():
            graph.nodes[node_id] = PEGNode.from_dict(node_data)

        # Load edges
        for edge_data in data.get("edges", []):
            graph.edges.append(PEGEdge.from_dict(edge_data))

        # Load metadata
        graph.metadata = data.get("metadata", {})

        return graph

    def to_json(self) -> str:
        """
        Serialize graph to JSON string.

        Returns:
            JSON string representation of the graph
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "PortableExecutionGraph":
        """
        Deserialize graph from JSON string.

        Args:
            json_str: JSON string representation of the graph

        Returns:
            PortableExecutionGraph instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
