"""Base provider abstraction for agent frameworks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

from agentlegatus.tools.tool import Tool

if TYPE_CHECKING:
    from agentlegatus.core.graph import PortableExecutionGraph


class ProviderCapability(Enum):
    """Capabilities that providers can support."""

    STREAMING = "streaming"
    PARALLEL_EXECUTION = "parallel_execution"
    STATE_PERSISTENCE = "state_persistence"
    TOOL_CALLING = "tool_calling"
    HUMAN_IN_LOOP = "human_in_loop"


class BaseProvider(ABC):
    """Abstract base class for framework providers.

    All provider implementations must inherit from this class and implement
    the abstract methods to provide a unified interface across different
    agent frameworks.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize provider with configuration.

        Args:
            config: Provider-specific configuration dictionary
        """
        self.config = config
        self.capabilities = self._get_capabilities()

    @abstractmethod
    def _get_capabilities(self) -> list[ProviderCapability]:
        """Return list of capabilities supported by this provider.

        Returns:
            List of ProviderCapability enums indicating what features
            this provider supports
        """
        pass

    @abstractmethod
    async def create_agent(self, agent_config: dict[str, Any]) -> Any:
        """Create an agent instance using the underlying framework.

        Args:
            agent_config: Configuration for the agent including model,
                         temperature, max_tokens, etc.

        Returns:
            Agent instance compatible with the underlying framework
        """
        pass

    @abstractmethod
    async def execute_agent(
        self, agent: Any, input_data: Any, state: dict[str, Any] | None = None
    ) -> Any:
        """Execute an agent with input and state.

        Args:
            agent: Agent instance created by create_agent()
            input_data: Input data for the agent
            state: Optional state dictionary

        Returns:
            Execution result from the agent
        """
        pass

    @abstractmethod
    async def invoke_tool(
        self, tool: Tool, tool_input: dict[str, Any], context: dict[str, Any]
    ) -> Any:
        """Invoke a tool through the provider's tool system.

        Args:
            tool: Tool instance to invoke
            tool_input: Input parameters for the tool
            context: Execution context

        Returns:
            Tool execution result
        """
        pass

    @abstractmethod
    def export_state(self) -> dict[str, Any]:
        """Export current state in provider-agnostic format.

        Returns:
            Dictionary containing provider state in a format that can
            be imported by other providers
        """
        pass

    @abstractmethod
    def import_state(self, state: dict[str, Any]) -> None:
        """Import state from provider-agnostic format.

        Args:
            state: State dictionary exported from another provider
        """
        pass

    @abstractmethod
    def to_portable_graph(self, workflow: Any) -> PortableExecutionGraph:
        """Convert provider-specific workflow to portable graph.

        Args:
            workflow: Provider-specific workflow representation

        Returns:
            PortableExecutionGraph that preserves workflow semantics
        """
        pass

    @abstractmethod
    def from_portable_graph(self, graph: PortableExecutionGraph) -> Any:
        """Convert portable graph to provider-specific workflow.

        Args:
            graph: PortableExecutionGraph to convert

        Returns:
            Provider-specific workflow equivalent to the portable graph
        """
        pass

    def supports_capability(self, capability: ProviderCapability) -> bool:
        """Check if provider supports a specific capability.

        Args:
            capability: ProviderCapability to check

        Returns:
            True if the capability is supported, False otherwise
        """
        return capability in self.capabilities

    def require_capability(self, capability: ProviderCapability) -> None:
        """Assert that this provider supports the given capability.

        Args:
            capability: ProviderCapability that is required

        Raises:
            CapabilityNotSupportedError: If the capability is not supported
        """
        if not self.supports_capability(capability):
            raise CapabilityNotSupportedError(
                provider_name=self.__class__.__name__,
                capability=capability,
                supported_capabilities=self.capabilities,
            )


# Import from centralized exceptions module; re-export for backward compat
from agentlegatus.exceptions import CapabilityNotSupportedError  # noqa: E402, F401
