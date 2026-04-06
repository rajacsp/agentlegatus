"""Mock provider for deterministic testing."""

from typing import Any

from agentlegatus.core.graph import (
    PEGEdge,
    PEGNode,
    PortableExecutionGraph,
)
from agentlegatus.providers.base import BaseProvider, ProviderCapability
from agentlegatus.tools.tool import Tool


class MockProvider(BaseProvider):
    """Deterministic mock provider for testing.

    Supports all capabilities and produces predictable outputs so that
    unit and integration tests can assert on exact values without
    relying on external services.
    """

    def __init__(self, config: dict[str, Any]):
        self._state: dict[str, Any] = {}
        self._agents: dict[str, dict[str, Any]] = {}
        self._call_log: list[dict[str, Any]] = []
        self._execution_count: int = 0
        super().__init__(config)

    def _get_capabilities(self) -> list[ProviderCapability]:
        return list(ProviderCapability)

    @property
    def call_log(self) -> list[dict[str, Any]]:
        """Return the log of all calls made to this provider."""
        return list(self._call_log)

    @property
    def execution_count(self) -> int:
        """Return the total number of agent executions."""
        return self._execution_count

    async def create_agent(self, agent_config: dict[str, Any]) -> Any:
        agent_id = agent_config.get("agent_id", "mock-agent")
        agent = {
            "agent_id": agent_id,
            "name": agent_config.get("name", agent_id),
            "model": agent_config.get("model", "mock-model"),
            "provider": "mock",
            "config": agent_config,
        }
        self._agents[agent_id] = agent
        self._call_log.append({"action": "create_agent", "agent_id": agent_id})
        return agent

    async def execute_agent(
        self,
        agent: Any,
        input_data: Any,
        state: dict[str, Any] | None = None,
    ) -> Any:
        """Execute agent with deterministic output.

        The output is always ``mock:<agent_id>:result:<n>`` where *n* is
        the 1-based execution counter, making assertions straightforward.
        """
        self._execution_count += 1
        agent_id = agent.get("agent_id", "unknown")

        merged_state = {**self._state}
        if state:
            merged_state.update(state)

        result = {
            "output": f"mock:{agent_id}:result:{self._execution_count}",
            "agent_id": agent_id,
            "input": input_data,
            "state": merged_state,
            "execution_number": self._execution_count,
        }

        self._state.update(merged_state)
        self._call_log.append(
            {
                "action": "execute_agent",
                "agent_id": agent_id,
                "execution_number": self._execution_count,
            }
        )
        return result

    async def invoke_tool(
        self,
        tool: Tool,
        tool_input: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:
        self._call_log.append(
            {
                "action": "invoke_tool",
                "tool_name": tool.name,
                "tool_input": tool_input,
            }
        )
        return await tool.invoke(tool_input, context)

    def export_state(self) -> dict[str, Any]:
        return {
            "provider": "mock",
            "state": dict(self._state),
            "agents": {aid: dict(a) for aid, a in self._agents.items()},
            "execution_count": self._execution_count,
        }

    def import_state(self, state: dict[str, Any]) -> None:
        self._state = dict(state.get("state", {}))
        self._agents = {aid: dict(a) for aid, a in state.get("agents", {}).items()}
        self._execution_count = state.get("execution_count", 0)

    def to_portable_graph(self, workflow: Any) -> PortableExecutionGraph:
        graph = PortableExecutionGraph()
        graph.metadata = {
            "source_provider": "mock",
            "original_metadata": workflow.get("metadata", {}),
        }

        for node_data in workflow.get("nodes", []):
            node = PEGNode(
                node_id=node_data["node_id"],
                node_type=node_data.get("node_type", "agent"),
                config=node_data.get("config", {}),
                inputs=node_data.get("inputs", []),
                outputs=node_data.get("outputs", []),
            )
            graph.add_node(node)

        for edge_data in workflow.get("edges", []):
            edge = PEGEdge(
                source=edge_data["source"],
                target=edge_data["target"],
                condition=edge_data.get("condition"),
            )
            graph.add_edge(edge)

        return graph

    def from_portable_graph(self, graph: PortableExecutionGraph) -> Any:
        return {
            "provider": "mock",
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
        }

    def reset(self) -> None:
        """Reset all internal state for clean test isolation."""
        self._state.clear()
        self._agents.clear()
        self._call_log.clear()
        self._execution_count = 0
