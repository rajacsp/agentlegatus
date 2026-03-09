"""Agent implementation for task execution."""

from typing import Any, Dict, List, Optional
from datetime import datetime

from agentlegatus.core.models import AgentCapability
from agentlegatus.providers.base import BaseProvider
from agentlegatus.tools.registry import ToolRegistry
from agentlegatus.memory.manager import MemoryManager


class Agent:
    """Individual agent worker that executes specific tasks.
    
    An agent is the lowest level in the Roman military hierarchy,
    responsible for executing individual tasks using the underlying
    framework capabilities provided by the BaseProvider.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        capabilities: List[AgentCapability],
        provider: BaseProvider,
        tool_registry: Optional[ToolRegistry] = None,
        memory_manager: Optional[MemoryManager] = None
    ):
        """Initialize agent with capabilities and provider.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            capabilities: List of capabilities this agent supports
            provider: Provider instance for framework-specific operations
            tool_registry: Optional tool registry for tool invocation
            memory_manager: Optional memory manager for memory operations
        """
        self.agent_id = agent_id
        self.name = name
        self.capabilities = capabilities
        self.provider = provider
        self.tool_registry = tool_registry
        self.memory_manager = memory_manager
        
        # Status tracking
        self._status = "idle"
        self._current_task: Optional[str] = None
        self._task_count = 0
        self._error_count = 0
        self._start_time: Optional[datetime] = None
        self._total_execution_time = 0.0
        
        # Provider-specific agent instance (created lazily)
        self._agent_instance: Optional[Any] = None
    
    async def run(
        self,
        input_data: Any,
        state: Optional[Dict[str, Any]] = None,
        tools: Optional[List[str]] = None
    ) -> Any:
        """Execute agent task.
        
        This method delegates to the provider's execute_agent method,
        which handles the framework-specific execution logic.
        
        Args:
            input_data: Input data for the agent task
            state: Optional state dictionary for stateful execution
            tools: Optional list of tool names to make available
            
        Returns:
            Execution result from the agent
            
        Raises:
            RuntimeError: If agent instance creation fails
        """
        self._status = "running"
        self._current_task = str(input_data)[:100]  # Store truncated task description
        self._task_count += 1
        start_time = datetime.now()
        
        try:
            # Create agent instance if not already created
            if self._agent_instance is None:
                agent_config = {
                    "agent_id": self.agent_id,
                    "name": self.name,
                    "capabilities": [cap.value for cap in self.capabilities],
                }
                
                # Add tools if provided and TOOL_USE capability is present
                if tools and AgentCapability.TOOL_USE in self.capabilities:
                    agent_config["tools"] = tools
                
                self._agent_instance = await self.provider.create_agent(agent_config)
            
            # Execute the agent task
            result = await self.provider.execute_agent(
                self._agent_instance,
                input_data,
                state
            )
            
            # Update metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self._total_execution_time += execution_time
            self._status = "idle"
            self._current_task = None
            
            return result
            
        except Exception as e:
            self._error_count += 1
            self._status = "error"
            execution_time = (datetime.now() - start_time).total_seconds()
            self._total_execution_time += execution_time
            raise
    
    async def invoke_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Invoke a tool through abstraction layer.
        
        Args:
            tool_name: Name of the tool to invoke
            tool_input: Input parameters for the tool
            context: Optional execution context
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If TOOL_USE capability is not supported
            RuntimeError: If tool registry is not configured
            KeyError: If tool is not found in registry
        """
        if AgentCapability.TOOL_USE not in self.capabilities:
            raise ValueError(
                f"Agent {self.agent_id} does not have TOOL_USE capability"
            )
        
        if self.tool_registry is None:
            raise RuntimeError(
                f"Agent {self.agent_id} has TOOL_USE capability but no tool registry configured"
            )
        
        # Get tool from registry
        tool = self.tool_registry.get_tool(tool_name)
        if tool is None:
            raise KeyError(f"Tool '{tool_name}' not found in registry")
        
        # Invoke tool through provider
        context = context or {}
        context["agent_id"] = self.agent_id
        context["agent_name"] = self.name
        
        return await self.provider.invoke_tool(tool, tool_input, context)
    
    async def store_memory(
        self,
        key: str,
        value: Any,
        memory_type: str = "short_term",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store data in memory backend.
        
        Args:
            key: Memory key
            value: Value to store
            memory_type: Type of memory ("short_term" or "long_term")
            metadata: Optional metadata for the memory
            
        Raises:
            ValueError: If MEMORY capability is not supported
            RuntimeError: If memory manager is not configured
        """
        if AgentCapability.MEMORY not in self.capabilities:
            raise ValueError(
                f"Agent {self.agent_id} does not have MEMORY capability"
            )
        
        if self.memory_manager is None:
            raise RuntimeError(
                f"Agent {self.agent_id} has MEMORY capability but no memory manager configured"
            )
        
        # Store memory based on type
        if memory_type == "short_term":
            await self.memory_manager.store_short_term(key, value)
        elif memory_type == "long_term":
            # Extract embedding if provided in metadata
            embedding = metadata.get("embedding") if metadata else None
            await self.memory_manager.store_long_term(key, value, embedding)
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")
    
    async def retrieve_memory(
        self,
        query: str,
        memory_type: str = "short_term",
        limit: int = 10
    ) -> List[Any]:
        """Retrieve data from memory backend.
        
        Args:
            query: Query string for memory retrieval
            memory_type: Type of memory to query
            limit: Maximum number of results to return
            
        Returns:
            List of memory items matching the query
            
        Raises:
            ValueError: If MEMORY capability is not supported
            RuntimeError: If memory manager is not configured
        """
        if AgentCapability.MEMORY not in self.capabilities:
            raise ValueError(
                f"Agent {self.agent_id} does not have MEMORY capability"
            )
        
        if self.memory_manager is None:
            raise RuntimeError(
                f"Agent {self.agent_id} has MEMORY capability but no memory manager configured"
            )
        
        # Retrieve memory based on type
        if memory_type == "short_term":
            from agentlegatus.memory.base import MemoryType
            return await self.memory_manager.get_recent(
                MemoryType.SHORT_TERM,
                limit
            )
        elif memory_type == "long_term":
            # Use semantic search for long-term memory
            return await self.memory_manager.semantic_search(
                query,
                limit=limit
            )
        else:
            raise ValueError(f"Unknown memory type: {memory_type}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status and metrics.
        
        Returns:
            Dictionary containing agent status including:
            - agent_id: Agent identifier
            - name: Agent name
            - status: Current status (idle, running, error)
            - capabilities: List of agent capabilities
            - current_task: Current task description (if running)
            - task_count: Total number of tasks executed
            - error_count: Total number of errors encountered
            - total_execution_time: Total time spent executing tasks
            - average_execution_time: Average time per task
        """
        avg_time = (
            self._total_execution_time / self._task_count
            if self._task_count > 0
            else 0.0
        )
        
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self._status,
            "capabilities": [cap.value for cap in self.capabilities],
            "current_task": self._current_task,
            "task_count": self._task_count,
            "error_count": self._error_count,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": avg_time,
        }
