"""LangGraph provider implementation for AgentLegatus."""

from typing import Any

from agentlegatus.core.graph import (
    PEGEdge,
    PEGNode,
    PortableExecutionGraph,
)
from agentlegatus.providers.base import BaseProvider, ProviderCapability
from agentlegatus.tools.tool import Tool


class LangGraphProvider(BaseProvider):
    """Provider adapter for the LangGraph framework.

    This is a minimal implementation that wraps LangGraph's graph-based
    agent execution model behind the BaseProvider interface. It supports
    state persistence, tool calling, and parallel execution capabilities.
    """

    def __init__(self, config: dict[str, Any]):
        self._state: dict[str, Any] = {}
        self._agents: dict[str, dict[str, Any]] = {}
        self._workflows: dict[str, Any] = {}
        super().__init__(config)

    def _get_capabilities(self) -> list[ProviderCapability]:
        return [
            ProviderCapability.STATE_PERSISTENCE,
            ProviderCapability.TOOL_CALLING,
            ProviderCapability.PARALLEL_EXECUTION,
        ]

    async def create_agent(self, agent_config: dict[str, Any]) -> Any:
        """Create a LangGraph agent node configuration.

        Args:
            agent_config: Agent configuration dict with keys like
                agent_id, name, model, temperature, etc.

        Returns:
            Dict representing the agent node in LangGraph terms.
        """
        agent_id = agent_config.get("agent_id", "default")
        agent = {
            "agent_id": agent_id,
            "name": agent_config.get("name", agent_id),
            "model": agent_config.get("model", "gpt-4"),
            "temperature": agent_config.get("temperature", 0.7),
            "max_tokens": agent_config.get("max_tokens", 4096),
            "system_prompt": agent_config.get("system_prompt"),
            "tools": agent_config.get("tools", []),
            "provider": "langgraph",
        }
        self._agents[agent_id] = agent
        return agent

    async def execute_agent(
        self,
        agent: Any,
        input_data: Any,
        state: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a LangGraph agent with input data.

        In a real implementation this would invoke the LangGraph runtime.
        Here we simulate execution by returning a structured result dict.

        Args:
            agent: Agent dict returned by create_agent()
            input_data: Input payload for the agent
            state: Optional state dict carried across invocations

        Returns:
            Dict with output, agent_id, model, and updated state.
        """
        agent_id = agent.get("agent_id", "unknown")
        merged_state = {**self._state}
        if state:
            merged_state.update(state)

        result = {
            "output": f"langgraph:{agent_id}:processed",
            "agent_id": agent_id,
            "model": agent.get("model"),
            "input": input_data,
            "state": merged_state,
        }

        self._state.update(merged_state)
        return result

    async def invoke_tool(
        self,
        tool: Tool,
        tool_input: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:
        """Invoke a tool through LangGraph's tool system.

        Delegates to the tool's own invoke method, which handles
        validation and execution.

        Args:
            tool: Tool instance to invoke
            tool_input: Input parameters for the tool
            context: Execution context

        Returns:
            Tool execution result
        """
        return await tool.invoke(tool_input, context)

    def export_state(self) -> dict[str, Any]:
        """Export internal state in a provider-agnostic format.

        Returns:
            Dict with provider name, state snapshot, and agent configs.
        """
        return {
            "provider": "langgraph",
            "state": dict(self._state),
            "agents": {aid: dict(a) for aid, a in self._agents.items()},
            "workflows": {
                wid: dict(w) if isinstance(w, dict) else w for wid, w in self._workflows.items()
            },
        }

    def import_state(self, state: dict[str, Any]) -> None:
        """Import state from a provider-agnostic format.

        Args:
            state: State dict previously exported via export_state()
        """
        self._state = dict(state.get("state", {}))
        self._agents = {aid: dict(a) for aid, a in state.get("agents", {}).items()}
        self._workflows = dict(state.get("workflows", {}))

    def to_portable_graph(self, workflow: Any) -> PortableExecutionGraph:
        """Convert a LangGraph workflow dict to a PortableExecutionGraph.

        Expects *workflow* to be a dict with ``nodes`` (list of node dicts)
        and ``edges`` (list of edge dicts).

        Args:
            workflow: Dict describing the LangGraph workflow

        Returns:
            PortableExecutionGraph preserving the workflow semantics
        """
        graph = PortableExecutionGraph()
        graph.metadata = {
            "source_provider": "langgraph",
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
        """Convert a PortableExecutionGraph to a LangGraph workflow dict.

        Args:
            graph: PortableExecutionGraph to convert

        Returns:
            Dict representing the workflow in LangGraph terms
        """
        workflow = {
            "provider": "langgraph",
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
        self._workflows[graph.metadata.get("workflow_id", "default")] = workflow
        return workflow
