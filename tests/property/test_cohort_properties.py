"""Property-based tests for Cohort."""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agentlegatus.hierarchy.cohort import Cohort, CohortStrategy, CohortFullError
from agentlegatus.hierarchy.agent import Agent
from agentlegatus.core.models import AgentCapability
from agentlegatus.providers.base import BaseProvider, ProviderCapability
from agentlegatus.tools.tool import Tool
from typing import Any, Dict, List, Optional


# Mock provider for testing
class MockProvider(BaseProvider):
    """Mock provider for testing agents."""
    
    def _get_capabilities(self) -> List[ProviderCapability]:
        return [ProviderCapability.TOOL_CALLING]
    
    async def create_agent(self, agent_config: Dict[str, Any]) -> Any:
        return {"agent_id": agent_config.get("agent_id"), "config": agent_config}
    
    async def execute_agent(
        self,
        agent: Any,
        input_data: Any,
        state: Optional[Dict[str, Any]] = None
    ) -> Any:
        return {"result": "mock_result", "input": input_data}
    
    async def invoke_tool(
        self,
        tool: Tool,
        tool_input: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        return {"tool": tool.name, "output": "mock_output"}
    
    def export_state(self) -> Dict[str, Any]:
        return {"state": "exported"}
    
    def import_state(self, state: Dict[str, Any]) -> None:
        pass
    
    def to_portable_graph(self, workflow: Any) -> Any:
        return {"graph": "portable"}
    
    def from_portable_graph(self, graph: Any) -> Any:
        return {"workflow": "provider_specific"}


# Helper function to create test agents
def create_test_agent(agent_id: str, name: str) -> Agent:
    """Create a test agent with mock provider."""
    provider = MockProvider({})
    return Agent(
        agent_id=agent_id,
        name=name,
        capabilities=[AgentCapability.TOOL_USE],
        provider=provider
    )


# Property 24: Cohort Capacity Enforcement
@pytest.mark.asyncio
@given(
    max_agents=st.integers(min_value=1, max_value=20),
    num_agents_to_add=st.integers(min_value=1, max_value=30),
    strategy=st.sampled_from(list(CohortStrategy)),
)
@settings(max_examples=10, deadline=2000)
async def test_property_24_cohort_capacity_enforcement(
    max_agents: int,
    num_agents_to_add: int,
    strategy: CohortStrategy,
):
    """
    Property 24: Cohort Capacity Enforcement
    
    For any cohort with max_agents capacity, adding agents succeeds 
    up to max_agents, and subsequent additions raise CohortFullError.
    
    Validates: Requirements 17.1, 17.2
    
    Requirement 17.1: WHEN adding an agent to a cohort, THE Cohort SHALL 
    register the agent if the cohort is not at max capacity
    
    Requirement 17.2: WHEN adding an agent to a full cohort, THE Cohort 
    SHALL raise a CohortFullError
    """
    # Create cohort with specified capacity
    cohort = Cohort(
        name=f"test_cohort_{max_agents}",
        strategy=strategy,
        max_agents=max_agents
    )
    
    # Track successfully added agents
    added_agents = []
    errors_raised = []
    
    # Attempt to add agents
    for i in range(num_agents_to_add):
        agent = create_test_agent(
            agent_id=f"agent_{i}",
            name=f"Agent {i}"
        )
        
        try:
            await cohort.add_agent(agent)
            added_agents.append(agent)
        except CohortFullError as e:
            errors_raised.append((i, e))
    
    # Get cohort status
    status = cohort.get_status()
    
    # Property assertions
    
    # 1. Number of successfully added agents should not exceed max_agents
    assert len(added_agents) <= max_agents, (
        f"Cohort added {len(added_agents)} agents but max_agents is {max_agents}"
    )
    
    # 2. If we tried to add more than max_agents, errors should have been raised
    if num_agents_to_add > max_agents:
        expected_errors = num_agents_to_add - max_agents
        assert len(errors_raised) == expected_errors, (
            f"Expected {expected_errors} CohortFullError exceptions, "
            f"but got {len(errors_raised)}"
        )
        
        # 3. All errors should occur after max_agents have been added
        for error_index, error in errors_raised:
            assert error_index >= max_agents, (
                f"CohortFullError raised at index {error_index}, "
                f"but should only occur at index >= {max_agents}"
            )
    
    # 4. If we tried to add <= max_agents, no errors should have been raised
    if num_agents_to_add <= max_agents:
        assert len(errors_raised) == 0, (
            f"No errors should be raised when adding {num_agents_to_add} agents "
            f"to cohort with capacity {max_agents}, but got {len(errors_raised)} errors"
        )
        
        # All agents should have been added successfully
        assert len(added_agents) == num_agents_to_add, (
            f"Expected {num_agents_to_add} agents to be added, "
            f"but only {len(added_agents)} were added"
        )
    
    # 5. Cohort status should reflect the correct number of agents
    assert status["agent_count"] == len(added_agents), (
        f"Cohort status reports {status['agent_count']} agents, "
        f"but {len(added_agents)} were actually added"
    )
    
    # 6. Cohort status should report the correct max_agents
    assert status["max_agents"] == max_agents, (
        f"Cohort status reports max_agents={status['max_agents']}, "
        f"but cohort was created with max_agents={max_agents}"
    )
    
    # 7. Number of agents in cohort should equal min(num_agents_to_add, max_agents)
    expected_count = min(num_agents_to_add, max_agents)
    assert status["agent_count"] == expected_count, (
        f"Expected {expected_count} agents in cohort, "
        f"but got {status['agent_count']}"
    )


# Additional property test: Capacity enforcement after removals
@pytest.mark.asyncio
@given(
    max_agents=st.integers(min_value=2, max_value=10),
    num_to_remove=st.integers(min_value=1, max_value=5),
)
@settings(max_examples=10, deadline=2000)
async def test_cohort_capacity_after_removal(
    max_agents: int,
    num_to_remove: int,
):
    """
    Test that capacity enforcement works correctly after removing agents.
    
    After removing agents from a full cohort, we should be able to add
    new agents up to the capacity again.
    """
    # Ensure we don't try to remove more agents than exist
    assume(num_to_remove < max_agents)
    
    # Create cohort and fill it to capacity
    cohort = Cohort(
        name="test_cohort",
        strategy=CohortStrategy.ROUND_ROBIN,
        max_agents=max_agents
    )
    
    # Add agents to fill capacity
    initial_agents = []
    for i in range(max_agents):
        agent = create_test_agent(f"agent_{i}", f"Agent {i}")
        await cohort.add_agent(agent)
        initial_agents.append(agent)
    
    # Verify cohort is full
    status = cohort.get_status()
    assert status["agent_count"] == max_agents
    
    # Verify adding another agent raises CohortFullError
    overflow_agent = create_test_agent("overflow", "Overflow Agent")
    with pytest.raises(CohortFullError):
        await cohort.add_agent(overflow_agent)
    
    # Remove some agents
    for i in range(num_to_remove):
        removed = await cohort.remove_agent(initial_agents[i].agent_id)
        assert removed is True
    
    # Verify agent count decreased
    status = cohort.get_status()
    expected_count = max_agents - num_to_remove
    assert status["agent_count"] == expected_count
    
    # Now we should be able to add num_to_remove agents again
    for i in range(num_to_remove):
        new_agent = create_test_agent(f"new_agent_{i}", f"New Agent {i}")
        await cohort.add_agent(new_agent)  # Should not raise
    
    # Verify cohort is full again
    status = cohort.get_status()
    assert status["agent_count"] == max_agents
    
    # Verify adding another agent still raises CohortFullError
    final_overflow = create_test_agent("final_overflow", "Final Overflow")
    with pytest.raises(CohortFullError):
        await cohort.add_agent(final_overflow)


# Additional property test: Capacity is independent of strategy
@pytest.mark.asyncio
@given(
    max_agents=st.integers(min_value=1, max_value=15),
    strategy=st.sampled_from(list(CohortStrategy)),
)
@settings(max_examples=10, deadline=2000)
async def test_cohort_capacity_independent_of_strategy(
    max_agents: int,
    strategy: CohortStrategy,
):
    """
    Test that capacity enforcement is independent of cohort strategy.
    
    All strategies should enforce the same capacity limits.
    """
    cohort = Cohort(
        name=f"test_cohort_{strategy.value}",
        strategy=strategy,
        max_agents=max_agents
    )
    
    # Add agents up to capacity
    for i in range(max_agents):
        agent = create_test_agent(f"agent_{i}", f"Agent {i}")
        await cohort.add_agent(agent)
    
    # Verify cohort is at capacity
    status = cohort.get_status()
    assert status["agent_count"] == max_agents
    assert status["strategy"] == strategy.value
    
    # Verify adding one more agent raises CohortFullError regardless of strategy
    overflow_agent = create_test_agent("overflow", "Overflow Agent")
    with pytest.raises(CohortFullError) as exc_info:
        await cohort.add_agent(overflow_agent)
    
    # Verify error message contains cohort name and capacity
    error_message = str(exc_info.value)
    assert f"test_cohort_{strategy.value}" in error_message
    assert str(max_agents) in error_message


# Additional property test: Empty cohort has zero agents
@pytest.mark.asyncio
@given(
    max_agents=st.integers(min_value=1, max_value=20),
    strategy=st.sampled_from(list(CohortStrategy)),
)
@settings(max_examples=10, deadline=1000)
async def test_cohort_empty_on_creation(
    max_agents: int,
    strategy: CohortStrategy,
):
    """
    Test that newly created cohorts start with zero agents.
    """
    cohort = Cohort(
        name="empty_cohort",
        strategy=strategy,
        max_agents=max_agents
    )
    
    status = cohort.get_status()
    
    # Verify cohort starts empty
    assert status["agent_count"] == 0
    assert status["max_agents"] == max_agents
    
    # Verify we can add agents up to capacity
    for i in range(max_agents):
        agent = create_test_agent(f"agent_{i}", f"Agent {i}")
        await cohort.add_agent(agent)  # Should not raise
    
    # Verify cohort is now full
    status = cohort.get_status()
    assert status["agent_count"] == max_agents
