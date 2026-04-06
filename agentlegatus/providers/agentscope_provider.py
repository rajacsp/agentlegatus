"""AgentScope provider implementation for AgentLegatus.

Supports single-agent and multi-agent patterns including sequential
pipelines, fanout (parallel) pipelines, and group discussion via
AgentScope's MsgHub.

Uses the actual AgentScope public API:
- ``agentscope.init()`` for global initialisation
- ``agentscope.agent.ReActAgent`` as the primary agent class
- ``agentscope.pipeline.sequential_pipeline`` / ``fanout_pipeline``
- ``agentscope.pipeline.MsgHub`` for shared-message group discussion
- ``agentscope.message.Msg`` for message passing
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any

from agentlegatus.core.graph import (
    PEGEdge,
    PEGNode,
    PortableExecutionGraph,
)
from agentlegatus.providers.base import BaseProvider, ProviderCapability
from agentlegatus.tools.tool import Tool


class MultiAgentStrategy(Enum):
    """Execution strategies for multi-agent groups."""

    SEQUENTIAL = "sequential"
    FANOUT = "fanout"
    DISCUSSION = "discussion"


class AgentScopeProvider(BaseProvider):
    """Provider adapter for the AgentScope framework.

    Wraps AgentScope behind the BaseProvider interface.  Supports
    single-agent execution as well as multi-agent patterns:

    - **Sequential pipeline**: agents process a message one after
      another; each receives the previous agent's output.
    - **Fanout pipeline**: every agent receives a deep-copy of the
      same input concurrently; results are collected into a list.
    - **Discussion (MsgHub)**: agents participate in a group chat
      for a configurable number of rounds with shared message state.
    """

    def __init__(self, config: dict[str, Any]):
        self._state: dict[str, Any] = {}
        self._agents: dict[str, dict[str, Any]] = {}
        self._groups: dict[str, dict[str, Any]] = {}
        self._pipelines: dict[str, Any] = {}
        self._as_initialised = False
        super().__init__(config)

    def _get_capabilities(self) -> list[ProviderCapability]:
        return [
            ProviderCapability.TOOL_CALLING,
            ProviderCapability.PARALLEL_EXECUTION,
            ProviderCapability.STREAMING,
        ]

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_init(self) -> None:
        """Lazily import and initialise AgentScope.

        Calls ``agentscope.init()`` exactly once with the project /
        name values from the provider config.

        Raises:
            ImportError: If the ``agentscope`` package is not installed.
        """
        if self._as_initialised:
            return
        try:
            import agentscope
        except ImportError:
            raise ImportError(
                "agentscope is required for AgentScopeProvider. "
                "Install it with: pip install agentscope"
            )

        agentscope.init(
            project=self.config.get("project", "agentlegatus"),
            name=self.config.get("name", "default"),
        )
        self._as_initialised = True

    # ------------------------------------------------------------------
    # Agent factory
    # ------------------------------------------------------------------

    def _build_as_agent(self, agent_config: dict[str, Any]) -> Any:
        """Instantiate the appropriate AgentScope agent.

        Supported ``agent_type`` values:

        - ``react`` (default) → ``ReActAgent``
        - ``user`` → ``UserAgent``

        Args:
            agent_config: Configuration dict.

        Returns:
            An AgentScope agent instance.
        """
        from agentscope.agent import ReActAgent, UserAgent
        from agentscope.memory import InMemoryMemory
        

        agent_type = agent_config.get("agent_type", "react")
        name = agent_config.get("name", agent_config.get("agent_id", "agent"))

        if agent_type == "user":
            return UserAgent(name=name)

        # Default: ReActAgent
        model = self._resolve_model(agent_config)
        formatter = self._resolve_formatter(agent_config)
        sys_prompt = agent_config.get(
            "system_prompt", "You are a helpful assistant.",
        )
        toolkit = self._resolve_toolkit(agent_config)
        memory = agent_config.get("memory") or InMemoryMemory()

        return ReActAgent(
            name=name,
            sys_prompt=sys_prompt,
            model=model,
            formatter=formatter,
            toolkit=toolkit,
            memory=memory,
            max_iters=agent_config.get("max_iters", 10),
        )

    @staticmethod
    def _resolve_model(agent_config: dict[str, Any]) -> Any:
        """Build a ChatModelBase from config or return a pre-built one.

        If ``agent_config["model"]`` is already a model object it is
        returned as-is.  Otherwise a model is constructed from the
        ``model_type`` and ``model_name`` keys.
        """
        model = agent_config.get("model")
        if model is not None and not isinstance(model, (str, dict)):
            return model  # already a ChatModelBase instance

        # Build from config dict / string
        model_cfg: dict[str, Any] = (
            model if isinstance(model, dict) else {"model_name": model or "gpt-4"}
        )
        model_type = model_cfg.pop("model_type", "openai")

        if model_type == "dashscope":
            from agentscope.model import DashScopeChatModel
            return DashScopeChatModel(**model_cfg)

        # Default: OpenAI-compatible
        from agentscope.model import OpenAIChatModel
        return OpenAIChatModel(**model_cfg)

    @staticmethod
    def _resolve_formatter(agent_config: dict[str, Any]) -> Any:
        """Return a FormatterBase from config or build a default one."""
        formatter = agent_config.get("formatter")
        if formatter is not None:
            return formatter

        from agentscope.formatter import OpenAIFormatter
        return OpenAIFormatter()

    @staticmethod
    def _resolve_toolkit(agent_config: dict[str, Any]) -> Any:
        """Return a Toolkit from config or build an empty one."""
        toolkit = agent_config.get("toolkit")
        if toolkit is not None:
            return toolkit

        from agentscope.tool import Toolkit
        return Toolkit()

    # ------------------------------------------------------------------
    # Single-agent interface (BaseProvider)
    # ------------------------------------------------------------------

    async def create_agent(self, agent_config: dict[str, Any]) -> Any:
        """Create an AgentScope agent instance.

        Args:
            agent_config: Configuration dict with keys like agent_id,
                name, model, system_prompt, agent_type, formatter,
                toolkit, memory, max_iters, etc.

        Returns:
            Dict wrapping the underlying AgentScope agent.
        """
        self._ensure_init()

        agent_id = agent_config.get("agent_id", "default")
        as_agent = self._build_as_agent(agent_config)

        wrapper: dict[str, Any] = {
            "agent_id": agent_id,
            "name": agent_config.get("name", agent_id),
            "agent_type": agent_config.get("agent_type", "react"),
            "provider": "agentscope",
            "config": agent_config,
            "_as_agent": as_agent,
        }
        self._agents[agent_id] = wrapper
        return wrapper

    async def execute_agent(
        self,
        agent: Any,
        input_data: Any,
        state: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a single AgentScope agent.

        The agent's ``__call__`` is awaited with an ``agentscope.message.Msg``.

        Args:
            agent: Wrapper dict returned by ``create_agent()``.
            input_data: String, dict, or Msg-compatible value.
            state: Optional shared state dict.

        Returns:
            Dict with ``output``, ``agent_id``, ``input``, and ``state``.
        """
        self._ensure_init()

        agent_id = agent.get("agent_id", "unknown")
        as_agent = agent["_as_agent"]

        merged_state = {**self._state, **(state or {})}
        msg = self._to_msg(input_data)

        response = await as_agent(msg)

        result = {
            "output": response.content if hasattr(response, "content") else str(response),
            "agent_id": agent_id,
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
        """Invoke a tool. Delegates to the tool's own invoke method."""
        return await tool.invoke(tool_input, context)

    # ------------------------------------------------------------------
    # Multi-agent: creation
    # ------------------------------------------------------------------

    async def create_agents(
        self, agent_configs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Batch-create multiple agents.

        Args:
            agent_configs: List of agent configuration dicts.

        Returns:
            List of agent wrapper dicts.
        """
        return [await self.create_agent(cfg) for cfg in agent_configs]

    async def create_group(
        self,
        group_id: str,
        agent_ids: list[str],
        strategy: MultiAgentStrategy = MultiAgentStrategy.SEQUENTIAL,
        max_rounds: int = 3,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Define a multi-agent group from previously created agents.

        Args:
            group_id: Unique identifier for the group.
            agent_ids: Ordered list of agent IDs belonging to this group.
            strategy: Execution strategy for the group.
            max_rounds: Discussion rounds (``DISCUSSION`` strategy only).
            metadata: Optional extra metadata.

        Returns:
            Group descriptor dict.

        Raises:
            KeyError: If any agent_id has not been created yet.
        """
        for aid in agent_ids:
            if aid not in self._agents:
                raise KeyError(
                    f"Agent '{aid}' not found. "
                    "Create it with create_agent() first."
                )

        group: dict[str, Any] = {
            "group_id": group_id,
            "agent_ids": list(agent_ids),
            "strategy": strategy,
            "max_rounds": max_rounds,
            "metadata": metadata or {},
        }
        self._groups[group_id] = group
        return group

    # ------------------------------------------------------------------
    # Multi-agent: execution
    # ------------------------------------------------------------------

    async def execute_group(
        self,
        group_id: str,
        input_data: Any,
        state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a multi-agent group using its configured strategy.

        Args:
            group_id: ID of a group created via ``create_group()``.
            input_data: Initial input for the pipeline / discussion.
            state: Optional shared state dict.

        Returns:
            Dict with per-agent results, strategy, group_id, and state.

        Raises:
            KeyError: If group_id is unknown.
            ValueError: If the strategy is unrecognised.
        """
        if group_id not in self._groups:
            raise KeyError(f"Group '{group_id}' not found.")

        group = self._groups[group_id]
        strategy: MultiAgentStrategy = group["strategy"]

        if strategy == MultiAgentStrategy.SEQUENTIAL:
            return await self._run_sequential(group, input_data, state)
        if strategy == MultiAgentStrategy.FANOUT:
            return await self._run_fanout(group, input_data, state)
        if strategy == MultiAgentStrategy.DISCUSSION:
            return await self._run_discussion(group, input_data, state)

        raise ValueError(f"Unknown multi-agent strategy: {strategy}")

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    async def _run_sequential(
        self,
        group: dict[str, Any],
        input_data: Any,
        state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Run agents via ``agentscope.pipeline.sequential_pipeline``.

        Each agent receives the output of the previous one.
        """
        self._ensure_init()
        from agentscope.pipeline import sequential_pipeline

        merged_state = {**self._state, **(state or {})}
        as_agents = [
            self._agents[aid]["_as_agent"] for aid in group["agent_ids"]
        ]
        msg = self._to_msg(input_data)

        final_msg = await sequential_pipeline(as_agents, msg)

        final_output = (
            final_msg.content
            if hasattr(final_msg, "content")
            else str(final_msg)
        )
        self._state.update(merged_state)

        return {
            "group_id": group["group_id"],
            "strategy": MultiAgentStrategy.SEQUENTIAL.value,
            "final_output": final_output,
            "state": merged_state,
        }

    async def _run_fanout(
        self,
        group: dict[str, Any],
        input_data: Any,
        state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Run agents via ``agentscope.pipeline.fanout_pipeline``.

        Every agent receives a deep-copy of the same input concurrently.
        """
        self._ensure_init()
        from agentscope.pipeline import fanout_pipeline

        merged_state = {**self._state, **(state or {})}
        as_agents = [
            self._agents[aid]["_as_agent"] for aid in group["agent_ids"]
        ]
        msg = self._to_msg(input_data)

        responses = await fanout_pipeline(as_agents, msg)

        outputs: list[dict[str, Any]] = []
        for aid, resp in zip(group["agent_ids"], responses):
            output_text = (
                resp.content if hasattr(resp, "content") else str(resp)
            )
            outputs.append({
                "agent_id": aid,
                "name": self._agents[aid]["name"],
                "output": output_text,
            })

        self._state.update(merged_state)

        return {
            "group_id": group["group_id"],
            "strategy": MultiAgentStrategy.FANOUT.value,
            "outputs": outputs,
            "state": merged_state,
        }

    async def _run_discussion(
        self,
        group: dict[str, Any],
        input_data: Any,
        state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Run a group discussion via ``agentscope.pipeline.MsgHub``.

        Agents take turns speaking for ``max_rounds`` rounds.  Every
        message is automatically broadcast to all participants through
        the hub's subscriber mechanism.
        """
        self._ensure_init()
        from agentscope.pipeline import MsgHub, sequential_pipeline

        merged_state = {**self._state, **(state or {})}
        agent_ids = group["agent_ids"]
        as_agents = [self._agents[aid]["_as_agent"] for aid in agent_ids]
        max_rounds: int = group.get("max_rounds", 3)

        announcement = self._to_msg(input_data)
        history: list[dict[str, Any]] = []

        async with MsgHub(
            participants=as_agents,
            announcement=announcement,
        ):
            for round_num in range(max_rounds):
                for aid, as_agent in zip(agent_ids, as_agents):
                    response = await as_agent()
                    output_text = (
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )
                    history.append({
                        "round": round_num + 1,
                        "agent_id": aid,
                        "name": self._agents[aid]["name"],
                        "output": output_text,
                    })

        self._state.update(merged_state)

        return {
            "group_id": group["group_id"],
            "strategy": MultiAgentStrategy.DISCUSSION.value,
            "max_rounds": max_rounds,
            "history": history,
            "final_output": history[-1]["output"] if history else None,
            "state": merged_state,
        }

    # ------------------------------------------------------------------
    # State import / export
    # ------------------------------------------------------------------

    def export_state(self) -> dict[str, Any]:
        """Export internal state in a provider-agnostic format."""
        agents_export: dict[str, Any] = {}
        for aid, wrapper in self._agents.items():
            agents_export[aid] = {
                k: v for k, v in wrapper.items() if k != "_as_agent"
            }

        groups_export: dict[str, Any] = {}
        for gid, g in self._groups.items():
            groups_export[gid] = {
                k: (v.value if isinstance(v, MultiAgentStrategy) else v)
                for k, v in g.items()
            }

        return {
            "provider": "agentscope",
            "state": dict(self._state),
            "agents": agents_export,
            "groups": groups_export,
            "pipelines": {
                pid: dict(p) if isinstance(p, dict) else p
                for pid, p in self._pipelines.items()
            },
        }

    def import_state(self, state: dict[str, Any]) -> None:
        """Import state from a provider-agnostic format."""
        self._state = dict(state.get("state", {}))
        self._agents = {
            aid: dict(a) for aid, a in state.get("agents", {}).items()
        }
        self._pipelines = dict(state.get("pipelines", {}))

        self._groups = {}
        for gid, g in state.get("groups", {}).items():
            restored = dict(g)
            strategy_val = restored.get("strategy")
            if isinstance(strategy_val, str):
                restored["strategy"] = MultiAgentStrategy(strategy_val)
            self._groups[gid] = restored

    # ------------------------------------------------------------------
    # Portable graph conversion
    # ------------------------------------------------------------------

    def to_portable_graph(self, workflow: Any) -> PortableExecutionGraph:
        """Convert an AgentScope pipeline dict to a PortableExecutionGraph."""
        graph = PortableExecutionGraph()
        graph.metadata = {
            "source_provider": "agentscope",
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
        """Convert a PortableExecutionGraph to an AgentScope pipeline dict."""
        pipeline = {
            "provider": "agentscope",
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
        self._pipelines[
            graph.metadata.get("pipeline_id", "default")
        ] = pipeline
        return pipeline

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_msg(input_data: Any) -> Any:
        """Convert arbitrary input to an ``agentscope.message.Msg``."""
        from agentscope.message import Msg

        if isinstance(input_data, str):
            return Msg(name="user", content=input_data, role="user")
        if isinstance(input_data, dict):
            return Msg(
                name=input_data.get("name", "user"),
                content=input_data.get("content", ""),
                role=input_data.get("role", "user"),
            )
        return Msg(name="user", content=str(input_data), role="user")
