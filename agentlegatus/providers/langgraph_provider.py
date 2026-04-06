"""LangGraph provider implementation for AgentLegatus.

Uses the actual LangGraph public API:
- ``langgraph.graph.StateGraph`` for declarative graph construction
- ``langgraph.graph.START`` / ``END`` sentinel nodes
- ``compiled_graph.ainvoke()`` for async execution
- ``langgraph.prebuilt.create_react_agent`` for ReAct-style agents
- ``langgraph.checkpoint.memory.MemorySaver`` for in-memory checkpointing
"""

from __future__ import annotations

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

    Wraps LangGraph's graph-based agent execution model behind the
    BaseProvider interface.  Supports state persistence via
    checkpointers, tool calling through LangChain tool integration,
    parallel node execution, and human-in-the-loop interrupts.
    """

    def __init__(self, config: dict[str, Any]):
        self._state: dict[str, Any] = {}
        self._agents: dict[str, dict[str, Any]] = {}
        self._workflows: dict[str, Any] = {}
        self._compiled_graphs: dict[str, Any] = {}
        super().__init__(config)

    def _get_capabilities(self) -> list[ProviderCapability]:
        return [
            ProviderCapability.STATE_PERSISTENCE,
            ProviderCapability.TOOL_CALLING,
            ProviderCapability.PARALLEL_EXECUTION,
            ProviderCapability.HUMAN_IN_LOOP,
        ]

    # ------------------------------------------------------------------
    # Lazy imports
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_langgraph() -> None:
        """Verify that langgraph is importable.

        Raises:
            ImportError: If the ``langgraph`` package is not installed.
        """
        try:
            import langgraph  # noqa: F401
        except ImportError:
            raise ImportError(
                "langgraph is required for LangGraphProvider. "
                "Install it with: pip install langgraph"
            )

    # ------------------------------------------------------------------
    # Agent creation
    # ------------------------------------------------------------------

    async def create_agent(self, agent_config: dict[str, Any]) -> Any:
        """Create a LangGraph agent.

        If ``agent_config`` contains a ``"tools"`` list of LangChain
        tool objects, a prebuilt ReAct agent is created via
        ``langgraph.prebuilt.create_react_agent``.  Otherwise a simple
        single-node ``StateGraph`` is compiled.

        The compiled graph is stored internally and returned inside a
        wrapper dict.

        Args:
            agent_config: Configuration dict.  Recognised keys:

                - ``agent_id`` (str)
                - ``name`` (str)
                - ``model`` – a ``langchain_core`` chat model instance,
                  or a string model name (used to build a
                  ``ChatOpenAI``).
                - ``system_prompt`` (str)
                - ``tools`` (list) – LangChain tool objects
                - ``checkpointer`` – optional ``BaseCheckpointSaver``

        Returns:
            Wrapper dict containing the compiled graph under
            ``"_compiled_graph"``.
        """
        self._ensure_langgraph()

        agent_id = agent_config.get("agent_id", "default")
        model = self._resolve_model(agent_config)
        tools = agent_config.get("tools", [])
        checkpointer = agent_config.get("checkpointer")

        if tools:
            compiled = self._build_react_agent(
                model, tools, agent_config, checkpointer,
            )
        else:
            compiled = self._build_simple_graph(
                model, agent_config, checkpointer,
            )

        wrapper: dict[str, Any] = {
            "agent_id": agent_id,
            "name": agent_config.get("name", agent_id),
            "provider": "langgraph",
            "config": agent_config,
            "_compiled_graph": compiled,
        }
        self._agents[agent_id] = wrapper
        self._compiled_graphs[agent_id] = compiled
        return wrapper

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    async def execute_agent(
        self,
        agent: Any,
        input_data: Any,
        state: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a compiled LangGraph agent via ``ainvoke``.

        Args:
            agent: Wrapper dict returned by ``create_agent()``.
            input_data: Input payload.  If a string, it is wrapped
                into a ``HumanMessage``.
            state: Optional extra state merged before invocation.

        Returns:
            Dict with ``output``, ``agent_id``, ``input``, and
            ``state``.
        """
        self._ensure_langgraph()
        from langchain_core.messages import HumanMessage

        agent_id = agent.get("agent_id", "unknown")
        compiled = agent["_compiled_graph"]

        merged_state = {**self._state, **(state or {})}

        # Build the invocation input
        if isinstance(input_data, str):
            invoke_input = {"messages": [HumanMessage(content=input_data)]}
        elif isinstance(input_data, dict):
            invoke_input = input_data
        else:
            invoke_input = {"messages": [HumanMessage(content=str(input_data))]}

        # Merge any extra state keys into the invocation input
        for k, v in merged_state.items():
            if k not in invoke_input:
                invoke_input[k] = v

        result = await compiled.ainvoke(invoke_input)

        # Extract the last message content as the output
        messages = result.get("messages", [])
        output = messages[-1].content if messages else str(result)

        self._state.update(merged_state)

        return {
            "output": output,
            "agent_id": agent_id,
            "input": input_data,
            "state": merged_state,
            "raw_result": result,
        }

    async def invoke_tool(
        self,
        tool: Tool,
        tool_input: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:
        """Invoke a tool. Delegates to the tool's own invoke method."""
        return await tool.invoke(tool_input, context)

    # ------------------------------------------------------------------
    # Graph builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_react_agent(
        model: Any,
        tools: list[Any],
        agent_config: dict[str, Any],
        checkpointer: Any | None,
    ) -> Any:
        """Build a prebuilt ReAct agent graph.

        Uses ``langgraph.prebuilt.create_react_agent`` which returns a
        compiled ``CompiledStateGraph``.
        """
        from langgraph.prebuilt import create_react_agent

        kwargs: dict[str, Any] = {
            "model": model,
            "tools": tools,
        }
        prompt = agent_config.get("system_prompt")
        if prompt:
            kwargs["prompt"] = prompt
        if checkpointer is not None:
            kwargs["checkpointer"] = checkpointer

        return create_react_agent(**kwargs)

    @staticmethod
    def _build_simple_graph(
        model: Any,
        agent_config: dict[str, Any],
        checkpointer: Any | None,
    ) -> Any:
        """Build a minimal single-node StateGraph.

        The graph has one node (``"agent"``) that calls the model and
        returns the response as a message list update.
        """
        import operator
        from typing import Annotated

        from langchain_core.messages import BaseMessage
        from langgraph.graph import END, START, StateGraph
        from typing_extensions import TypedDict

        class AgentState(TypedDict):
            messages: Annotated[list[BaseMessage], operator.add]

        async def agent_node(state: AgentState) -> dict[str, Any]:
            messages = state["messages"]
            response = await model.ainvoke(messages)
            return {"messages": [response]}

        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.add_edge(START, "agent")
        graph.add_edge("agent", END)

        compile_kwargs: dict[str, Any] = {}
        if checkpointer is not None:
            compile_kwargs["checkpointer"] = checkpointer

        return graph.compile(**compile_kwargs)

    @staticmethod
    def _resolve_model(agent_config: dict[str, Any]) -> Any:
        """Return a LangChain chat model from config.

        If ``agent_config["model"]`` is already a chat model object it
        is returned as-is.  Otherwise a ``ChatOpenAI`` is constructed
        from the string model name.
        """
        model = agent_config.get("model")
        if model is not None and not isinstance(model, (str, dict)):
            return model  # already a ChatModelBase instance

        model_name = model if isinstance(model, str) else "gpt-4"
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            temperature=agent_config.get("temperature", 0.7),
            max_tokens=agent_config.get("max_tokens", 4096),
        )

    # ------------------------------------------------------------------
    # Multi-node graph construction
    # ------------------------------------------------------------------

    def build_graph(
        self,
        nodes: dict[str, Any],
        edges: list[tuple[str, str]],
        conditional_edges: dict[str, Any] | None = None,
        checkpointer: Any | None = None,
    ) -> Any:
        """Build and compile a custom multi-node StateGraph.

        This is a convenience method for constructing arbitrary
        LangGraph workflows beyond the single-agent pattern.

        Args:
            nodes: Mapping of node name → async callable.
            edges: List of ``(source, target)`` tuples.  Use
                ``"__start__"`` and ``"__end__"`` for the sentinel
                nodes.
            conditional_edges: Optional mapping of
                ``source_node → (condition_fn, path_map)`` for
                conditional routing.
            checkpointer: Optional checkpointer for persistence.

        Returns:
            A compiled ``CompiledStateGraph``.
        """
        self._ensure_langgraph()

        import operator
        from typing import Annotated

        from langchain_core.messages import BaseMessage
        from langgraph.graph import END, START, StateGraph
        from typing_extensions import TypedDict

        class GraphState(TypedDict):
            messages: Annotated[list[BaseMessage], operator.add]

        graph = StateGraph(GraphState)

        for name, fn in nodes.items():
            graph.add_node(name, fn)

        for source, target in edges:
            src = START if source == "__start__" else source
            tgt = END if target == "__end__" else target
            graph.add_edge(src, tgt)

        if conditional_edges:
            for source, (condition_fn, path_map) in conditional_edges.items():
                graph.add_conditional_edges(source, condition_fn, path_map)

        compile_kwargs: dict[str, Any] = {}
        if checkpointer is not None:
            compile_kwargs["checkpointer"] = checkpointer

        compiled = graph.compile(**compile_kwargs)
        graph_id = f"custom_{len(self._compiled_graphs)}"
        self._compiled_graphs[graph_id] = compiled
        return compiled

    # ------------------------------------------------------------------
    # State import / export
    # ------------------------------------------------------------------

    def export_state(self) -> dict[str, Any]:
        """Export internal state in a provider-agnostic format."""
        # Strip non-serialisable compiled graphs from agent wrappers
        agents_export: dict[str, Any] = {}
        for aid, wrapper in self._agents.items():
            agents_export[aid] = {
                k: v for k, v in wrapper.items()
                if k != "_compiled_graph"
            }

        return {
            "provider": "langgraph",
            "state": dict(self._state),
            "agents": agents_export,
            "workflows": {
                wid: dict(w) if isinstance(w, dict) else w
                for wid, w in self._workflows.items()
            },
        }

    def import_state(self, state: dict[str, Any]) -> None:
        """Import state from a provider-agnostic format."""
        self._state = dict(state.get("state", {}))
        self._agents = {
            aid: dict(a) for aid, a in state.get("agents", {}).items()
        }
        self._workflows = dict(state.get("workflows", {}))

    # ------------------------------------------------------------------
    # Portable graph conversion
    # ------------------------------------------------------------------

    def to_portable_graph(self, workflow: Any) -> PortableExecutionGraph:
        """Convert a LangGraph workflow dict to a PortableExecutionGraph."""
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
        """Convert a PortableExecutionGraph to a LangGraph workflow dict."""
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
        self._workflows[
            graph.metadata.get("workflow_id", "default")
        ] = workflow
        return workflow
