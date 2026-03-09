"""Cohort implementation for agent group coordination."""

from typing import Any, Dict, List, Optional
from enum import Enum
import asyncio
from datetime import datetime

from agentlegatus.hierarchy.agent import Agent
from agentlegatus.core.state import StateManager


class CohortStrategy(Enum):
    """Coordination strategies for cohort task execution."""
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    BROADCAST = "broadcast"
    LEADER_FOLLOWER = "leader_follower"


class CohortFullError(Exception):
    """Raised when attempting to add an agent to a full cohort."""
    pass


class Cohort:
    """Group of agents working on related tasks.
    
    A cohort manages a group of agents and coordinates their execution
    using different strategies (round-robin, load-balanced, broadcast,
    or leader-follower).
    """
    
    def __init__(
        self,
        name: str,
        strategy: CohortStrategy,
        max_agents: int = 10
    ):
        """Initialize Cohort with coordination strategy.
        
        Args:
            name: Human-readable name for the cohort
            strategy: Coordination strategy for task execution
            max_agents: Maximum number of agents allowed in the cohort
        """
        self.name = name
        self.strategy = strategy
        self.max_agents = max_agents
        
        # Agent management
        self._agents: Dict[str, Agent] = {}
        self._agent_order: List[str] = []  # For round-robin
        self._current_index = 0  # For round-robin
        self._leader_id: Optional[str] = None  # For leader-follower
        
        # Metrics
        self._task_count = 0
        self._created_at = datetime.now()
    
    async def add_agent(self, agent: Agent) -> None:
        """Add an agent to the cohort.
        
        Args:
            agent: Agent instance to add
            
        Raises:
            CohortFullError: If cohort is at max capacity
        """
        if len(self._agents) >= self.max_agents:
            raise CohortFullError(
                f"Cohort '{self.name}' is at max capacity ({self.max_agents} agents)"
            )
        
        self._agents[agent.agent_id] = agent
        self._agent_order.append(agent.agent_id)
        
        # Set first agent as leader for LEADER_FOLLOWER strategy
        if self.strategy == CohortStrategy.LEADER_FOLLOWER and self._leader_id is None:
            self._leader_id = agent.agent_id
    
    async def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the cohort.
        
        Args:
            agent_id: ID of the agent to remove
            
        Returns:
            True if agent was removed, False if agent was not found
        """
        if agent_id not in self._agents:
            return False
        
        del self._agents[agent_id]
        self._agent_order.remove(agent_id)
        
        # Reset round-robin index if needed
        if self._current_index >= len(self._agent_order):
            self._current_index = 0
        
        # Reassign leader if removed
        if self.strategy == CohortStrategy.LEADER_FOLLOWER and agent_id == self._leader_id:
            self._leader_id = self._agent_order[0] if self._agent_order else None
        
        return True
    
    async def execute_task(
        self,
        task: Dict[str, Any],
        state: StateManager
    ) -> Any:
        """Execute task using cohort strategy.
        
        Args:
            task: Task data to execute
            state: State manager for accessing workflow state
            
        Returns:
            Task execution result (strategy-dependent)
            
        Raises:
            ValueError: If no agents are available
            RuntimeError: If strategy execution fails
        """
        if not self._agents:
            raise ValueError(f"Cohort '{self.name}' has no agents")
        
        self._task_count += 1
        
        # Get current workflow state
        workflow_state = await state.get_all()
        
        # Route to appropriate strategy
        if self.strategy == CohortStrategy.ROUND_ROBIN:
            return await self._execute_round_robin(task, workflow_state)
        elif self.strategy == CohortStrategy.LOAD_BALANCED:
            return await self._execute_load_balanced(task, workflow_state)
        elif self.strategy == CohortStrategy.BROADCAST:
            return await self._execute_broadcast(task, workflow_state)
        elif self.strategy == CohortStrategy.LEADER_FOLLOWER:
            return await self._execute_leader_follower(task, workflow_state)
        else:
            raise RuntimeError(f"Unknown strategy: {self.strategy}")
    
    async def _execute_round_robin(
        self,
        task: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Any:
        """Execute task using round-robin strategy.
        
        Distributes tasks evenly across agents in a circular fashion.
        
        Args:
            task: Task data to execute
            state: Current workflow state
            
        Returns:
            Result from the selected agent
        """
        # Select next agent in round-robin order
        agent_id = self._agent_order[self._current_index]
        agent = self._agents[agent_id]
        
        # Advance to next agent for next task
        self._current_index = (self._current_index + 1) % len(self._agent_order)
        
        # Execute task with selected agent
        return await agent.run(
            input_data=task.get("input"),
            state=state,
            tools=task.get("tools")
        )
    
    async def _execute_load_balanced(
        self,
        task: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Any:
        """Execute task using load-balanced strategy.
        
        Assigns task to the least busy agent based on current status.
        
        Args:
            task: Task data to execute
            state: Current workflow state
            
        Returns:
            Result from the least busy agent
        """
        # Find least busy agent
        least_busy_agent = None
        min_task_count = float('inf')
        
        for agent in self._agents.values():
            status = agent.get_status()
            
            # Prefer idle agents
            if status["status"] == "idle":
                task_count = status["task_count"]
                if task_count < min_task_count:
                    min_task_count = task_count
                    least_busy_agent = agent
        
        # If no idle agents, use first available
        if least_busy_agent is None:
            least_busy_agent = list(self._agents.values())[0]
        
        # Execute task with selected agent
        return await least_busy_agent.run(
            input_data=task.get("input"),
            state=state,
            tools=task.get("tools")
        )
    
    async def _execute_broadcast(
        self,
        task: Dict[str, Any],
        state: Dict[str, Any]
    ) -> List[Any]:
        """Execute task using broadcast strategy.
        
        Sends task to all agents and collects all results.
        
        Args:
            task: Task data to execute
            state: Current workflow state
            
        Returns:
            List of results from all agents
        """
        # Execute task on all agents concurrently
        tasks = [
            agent.run(
                input_data=task.get("input"),
                state=state,
                tools=task.get("tools")
            )
            for agent in self._agents.values()
        ]
        
        # Wait for all agents to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def _execute_leader_follower(
        self,
        task: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Any:
        """Execute task using leader-follower strategy.
        
        Routes all tasks through the leader agent. The leader can
        delegate to followers if needed.
        
        Args:
            task: Task data to execute
            state: Current workflow state
            
        Returns:
            Result from the leader agent
            
        Raises:
            RuntimeError: If no leader is assigned
        """
        if self._leader_id is None:
            raise RuntimeError(f"Cohort '{self.name}' has no leader assigned")
        
        leader = self._agents[self._leader_id]
        
        # Execute task with leader
        return await leader.run(
            input_data=task.get("input"),
            state=state,
            tools=task.get("tools")
        )
    
    async def broadcast_message(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all agents in cohort.
        
        This is a fire-and-forget operation that sends a message
        to all agents without waiting for responses.
        
        Args:
            message: Message data to broadcast
        """
        # Create tasks for all agents
        tasks = []
        for agent in self._agents.values():
            # Store message in agent's memory if it has MEMORY capability
            from agentlegatus.core.models import AgentCapability
            if AgentCapability.MEMORY in agent.capabilities:
                tasks.append(
                    agent.store_memory(
                        key=f"broadcast_{datetime.now().isoformat()}",
                        value=message,
                        memory_type="short_term"
                    )
                )
        
        # Execute all broadcasts concurrently (fire-and-forget)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_available_agents(self) -> List[Agent]:
        """Get list of available agents.
        
        Returns only agents that are not currently executing tasks.
        
        Returns:
            List of available Agent instances
        """
        available = []
        for agent in self._agents.values():
            status = agent.get_status()
            if status["status"] == "idle":
                available.append(agent)
        
        return available
    
    def get_status(self) -> Dict[str, Any]:
        """Get cohort status and metrics.
        
        Returns:
            Dictionary containing cohort status including:
            - name: Cohort name
            - strategy: Coordination strategy
            - agent_count: Number of agents in cohort
            - max_agents: Maximum capacity
            - available_agents: Number of idle agents
            - task_count: Total tasks executed
            - leader_id: Leader agent ID (for LEADER_FOLLOWER strategy)
        """
        available_count = len(self.get_available_agents())
        
        return {
            "name": self.name,
            "strategy": self.strategy.value,
            "agent_count": len(self._agents),
            "max_agents": self.max_agents,
            "available_agents": available_count,
            "task_count": self._task_count,
            "leader_id": self._leader_id,
            "created_at": self._created_at.isoformat(),
        }
