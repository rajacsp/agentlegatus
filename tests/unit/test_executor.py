"""Unit tests for WorkflowExecutor."""

import pytest
from unittest.mock import AsyncMock, Mock

from agentlegatus.core.event_bus import EventBus
from agentlegatus.core.executor import WorkflowExecutor
from agentlegatus.core.state import InMemoryStateBackend, StateManager, StateScope


class TestWorkflowExecutorCheckpoint:
    """Tests for WorkflowExecutor checkpoint and recovery."""

    @pytest.fixture
    def state_manager(self):
        """Create a state manager for testing."""
        backend = InMemoryStateBackend()
        return StateManager(backend=backend)

    @pytest.fixture
    def event_bus(self):
        """Create an event bus for testing."""
        return EventBus()

    @pytest.fixture
    def executor(self, state_manager, event_bus):
        """Create a workflow executor for testing."""
        mock_provider = Mock()
        mock_tool_registry = Mock()
        
        executor = WorkflowExecutor(
            provider=mock_provider,
            state_manager=state_manager,
            tool_registry=mock_tool_registry,
            event_bus=event_bus,
        )
        
        # Set up some initial state
        executor._current_workflow_id = "test_workflow_123"
        executor._completed_steps = {"step1", "step2", "step3"}
        
        return executor

    @pytest.mark.asyncio
    async def test_checkpoint_state_creates_snapshot(self, executor, state_manager):
        """Test that checkpoint_state creates a snapshot with all state scopes."""
        # Arrange
        checkpoint_id = "checkpoint_001"
        
        # Add some state to different scopes
        await state_manager.set("key1", "value1", StateScope.WORKFLOW)
        await state_manager.set("key2", "value2", StateScope.STEP)
        await state_manager.set("key3", "value3", StateScope.AGENT)
        await state_manager.set("key4", "value4", StateScope.GLOBAL)
        
        # Act
        await executor.checkpoint_state(checkpoint_id)
        
        # Assert - verify snapshot was created
        snapshots = await state_manager.list_snapshots()
        assert checkpoint_id in snapshots
        
        # Verify checkpoint metadata was stored
        metadata = await state_manager.get(
            f"checkpoint_metadata_{checkpoint_id}",
            StateScope.WORKFLOW
        )
        assert metadata is not None
        assert metadata["workflow_id"] == "test_workflow_123"
        assert set(metadata["completed_steps"]) == {"step1", "step2", "step3"}
        assert metadata["checkpoint_id"] == checkpoint_id

    @pytest.mark.asyncio
    async def test_checkpoint_state_saves_completed_steps(self, executor):
        """Test that checkpoint saves completed steps tracking."""
        # Arrange
        checkpoint_id = "checkpoint_002"
        executor._completed_steps = {"step_a", "step_b", "step_c", "step_d"}
        
        # Act
        await executor.checkpoint_state(checkpoint_id)
        
        # Assert
        metadata = await executor.state_manager.get(
            f"checkpoint_metadata_{checkpoint_id}",
            StateScope.WORKFLOW
        )
        assert len(metadata["completed_steps"]) == 4
        assert set(metadata["completed_steps"]) == {"step_a", "step_b", "step_c", "step_d"}

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint_restores_state(self, executor, state_manager):
        """Test that restore_from_checkpoint restores all state scopes."""
        # Arrange
        checkpoint_id = "checkpoint_003"
        
        # Set up initial state
        await state_manager.set("workflow_key", "workflow_value", StateScope.WORKFLOW)
        await state_manager.set("step_key", "step_value", StateScope.STEP)
        await state_manager.set("agent_key", "agent_value", StateScope.AGENT)
        await state_manager.set("global_key", "global_value", StateScope.GLOBAL)
        
        # Create checkpoint
        await executor.checkpoint_state(checkpoint_id)
        
        # Modify state after checkpoint
        await state_manager.set("workflow_key", "modified_value", StateScope.WORKFLOW)
        await state_manager.set("new_key", "new_value", StateScope.WORKFLOW)
        
        # Act - restore from checkpoint
        context = await executor.restore_from_checkpoint(checkpoint_id)
        
        # Assert - verify state was restored
        workflow_key_value = await state_manager.get("workflow_key", StateScope.WORKFLOW)
        assert workflow_key_value == "workflow_value"
        
        step_key_value = await state_manager.get("step_key", StateScope.STEP)
        assert step_key_value == "step_value"
        
        agent_key_value = await state_manager.get("agent_key", StateScope.AGENT)
        assert agent_key_value == "agent_value"
        
        global_key_value = await state_manager.get("global_key", StateScope.GLOBAL)
        assert global_key_value == "global_value"

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint_restores_completed_steps(self, executor):
        """Test that restore restores completed steps tracking."""
        # Arrange
        checkpoint_id = "checkpoint_004"
        original_steps = {"step1", "step2", "step3"}
        executor._completed_steps = original_steps.copy()
        
        # Create checkpoint
        await executor.checkpoint_state(checkpoint_id)
        
        # Modify completed steps after checkpoint
        executor._completed_steps = {"step1", "step2", "step3", "step4", "step5"}
        
        # Act - restore from checkpoint
        context = await executor.restore_from_checkpoint(checkpoint_id)
        
        # Assert
        assert executor._completed_steps == original_steps
        assert set(context["completed_steps"]) == original_steps

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint_restores_workflow_id(self, executor):
        """Test that restore restores workflow ID."""
        # Arrange
        checkpoint_id = "checkpoint_005"
        original_workflow_id = "workflow_abc_123"
        executor._current_workflow_id = original_workflow_id
        
        # Create checkpoint
        await executor.checkpoint_state(checkpoint_id)
        
        # Modify workflow ID after checkpoint
        executor._current_workflow_id = "different_workflow_xyz"
        
        # Act - restore from checkpoint
        context = await executor.restore_from_checkpoint(checkpoint_id)
        
        # Assert
        assert executor._current_workflow_id == original_workflow_id
        assert context["workflow_id"] == original_workflow_id

    @pytest.mark.asyncio
    async def test_restore_from_checkpoint_returns_context(self, executor, state_manager):
        """Test that restore returns execution context with workflow state."""
        # Arrange
        checkpoint_id = "checkpoint_006"
        
        await state_manager.set("context_key1", "context_value1", StateScope.WORKFLOW)
        await state_manager.set("context_key2", "context_value2", StateScope.WORKFLOW)
        
        await executor.checkpoint_state(checkpoint_id)
        
        # Act
        context = await executor.restore_from_checkpoint(checkpoint_id)
        
        # Assert
        assert "workflow_id" in context
        assert "completed_steps" in context
        assert "workflow_state" in context
        assert context["workflow_state"]["context_key1"] == "context_value1"
        assert context["workflow_state"]["context_key2"] == "context_value2"

    @pytest.mark.asyncio
    async def test_restore_from_nonexistent_checkpoint_raises_error(self, executor):
        """Test that restoring from non-existent checkpoint raises error."""
        # Arrange
        nonexistent_checkpoint_id = "nonexistent_checkpoint"
        
        # Act & Assert
        with pytest.raises(ValueError, match="Snapshot .* not found"):
            await executor.restore_from_checkpoint(nonexistent_checkpoint_id)

    @pytest.mark.asyncio
    async def test_checkpoint_with_empty_completed_steps(self, executor):
        """Test checkpoint with no completed steps."""
        # Arrange
        checkpoint_id = "checkpoint_007"
        executor._completed_steps = set()
        
        # Act
        await executor.checkpoint_state(checkpoint_id)
        
        # Assert
        metadata = await executor.state_manager.get(
            f"checkpoint_metadata_{checkpoint_id}",
            StateScope.WORKFLOW
        )
        assert metadata["completed_steps"] == []

    @pytest.mark.asyncio
    async def test_multiple_checkpoints_independent(self, executor, state_manager):
        """Test that multiple checkpoints are independent."""
        # Arrange
        checkpoint_id_1 = "checkpoint_008"
        checkpoint_id_2 = "checkpoint_009"
        
        # Create first checkpoint
        executor._completed_steps = {"step1", "step2"}
        await state_manager.set("key", "value1", StateScope.WORKFLOW)
        await executor.checkpoint_state(checkpoint_id_1)
        
        # Modify state and create second checkpoint
        executor._completed_steps = {"step1", "step2", "step3", "step4"}
        await state_manager.set("key", "value2", StateScope.WORKFLOW)
        await executor.checkpoint_state(checkpoint_id_2)
        
        # Act - restore first checkpoint
        await executor.restore_from_checkpoint(checkpoint_id_1)
        
        # Assert - verify first checkpoint state
        assert executor._completed_steps == {"step1", "step2"}
        value = await state_manager.get("key", StateScope.WORKFLOW)
        assert value == "value1"
        
        # Act - restore second checkpoint
        await executor.restore_from_checkpoint(checkpoint_id_2)
        
        # Assert - verify second checkpoint state
        assert executor._completed_steps == {"step1", "step2", "step3", "step4"}
        value = await state_manager.get("key", StateScope.WORKFLOW)
        assert value == "value2"



class TestWorkflowExecutorProviderSwitching:
    """Tests for WorkflowExecutor provider switching."""

    @pytest.fixture
    def state_manager(self):
        """Create a state manager for testing."""
        backend = InMemoryStateBackend()
        return StateManager(backend=backend)

    @pytest.fixture
    def event_bus(self):
        """Create an event bus for testing."""
        return EventBus()

    @pytest.fixture
    def mock_old_provider(self):
        """Create a mock old provider."""
        from agentlegatus.core.graph import PortableExecutionGraph, PEGNode
        
        provider = Mock()
        provider.export_state = Mock(return_value={
            "state_key1": "state_value1",
            "state_key2": "state_value2",
        })
        
        # Create a simple portable graph
        graph = PortableExecutionGraph()
        graph.add_node(PEGNode(
            node_id="node1",
            node_type="agent",
            config={"model": "gpt-4"},
        ))
        provider.to_portable_graph = Mock(return_value=graph)
        
        return provider

    @pytest.fixture
    def mock_new_provider(self):
        """Create a mock new provider."""
        provider = Mock()
        provider.import_state = Mock()
        provider.from_portable_graph = Mock(return_value={"workflow": "new_format"})
        return provider

    @pytest.fixture
    def executor(self, mock_old_provider, state_manager, event_bus):
        """Create a workflow executor for testing."""
        mock_tool_registry = Mock()
        
        executor = WorkflowExecutor(
            provider=mock_old_provider,
            state_manager=state_manager,
            tool_registry=mock_tool_registry,
            event_bus=event_bus,
        )
        
        executor._current_workflow_id = "test_workflow_123"
        
        return executor

    @pytest.mark.asyncio
    async def test_switch_provider_exports_state(
        self, executor, mock_old_provider, mock_new_provider
    ):
        """Test that switch_provider exports state from old provider."""
        # Act
        await executor.switch_provider(mock_new_provider)
        
        # Assert
        mock_old_provider.export_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_switch_provider_converts_to_portable_graph(
        self, executor, mock_old_provider, mock_new_provider, state_manager
    ):
        """Test that switch_provider converts workflow to portable graph."""
        # Arrange - store a workflow in state
        workflow = {"steps": ["step1", "step2"]}
        await state_manager.set("current_workflow", workflow, StateScope.WORKFLOW)
        
        # Act
        await executor.switch_provider(mock_new_provider)
        
        # Assert
        mock_old_provider.to_portable_graph.assert_called_once_with(workflow)

    @pytest.mark.asyncio
    async def test_switch_provider_validates_portable_graph(
        self, executor, mock_old_provider, mock_new_provider, state_manager
    ):
        """Test that switch_provider validates the portable graph."""
        # Arrange - create an invalid graph (with cycle)
        from agentlegatus.core.graph import PortableExecutionGraph, PEGNode, PEGEdge
        
        invalid_graph = PortableExecutionGraph()
        invalid_graph.add_node(PEGNode(node_id="node1", node_type="agent", config={}))
        invalid_graph.add_node(PEGNode(node_id="node2", node_type="agent", config={}))
        invalid_graph.add_edge(PEGEdge(source="node1", target="node2"))
        invalid_graph.add_edge(PEGEdge(source="node2", target="node1"))  # Creates cycle
        
        mock_old_provider.to_portable_graph = Mock(return_value=invalid_graph)
        
        # Store a workflow so to_portable_graph gets called
        workflow = {"steps": ["step1"]}
        await state_manager.set("current_workflow", workflow, StateScope.WORKFLOW)
        
        # Act & Assert
        with pytest.raises(ValueError, match="Portable graph validation failed"):
            await executor.switch_provider(mock_new_provider)

    @pytest.mark.asyncio
    async def test_switch_provider_imports_state_to_new_provider(
        self, executor, mock_old_provider, mock_new_provider
    ):
        """Test that switch_provider imports state into new provider."""
        # Arrange
        exported_state = {"state_key1": "state_value1", "state_key2": "state_value2"}
        mock_old_provider.export_state = Mock(return_value=exported_state)
        
        # Act
        await executor.switch_provider(mock_new_provider)
        
        # Assert
        mock_new_provider.import_state.assert_called_once_with(exported_state)

    @pytest.mark.asyncio
    async def test_switch_provider_converts_from_portable_graph(
        self, executor, mock_old_provider, mock_new_provider
    ):
        """Test that switch_provider converts portable graph to new provider format."""
        # Act
        await executor.switch_provider(mock_new_provider)
        
        # Assert
        mock_new_provider.from_portable_graph.assert_called_once()
        call_args = mock_new_provider.from_portable_graph.call_args[0][0]
        assert hasattr(call_args, 'nodes')
        assert hasattr(call_args, 'edges')

    @pytest.mark.asyncio
    async def test_switch_provider_updates_state_manager(
        self, executor, mock_new_provider, state_manager
    ):
        """Test that switch_provider updates StateManager with new workflow."""
        # Arrange
        new_workflow = {"workflow": "new_format", "steps": ["a", "b"]}
        mock_new_provider.from_portable_graph = Mock(return_value=new_workflow)
        
        # Act
        await executor.switch_provider(mock_new_provider)
        
        # Assert
        stored_workflow = await state_manager.get("current_workflow", StateScope.WORKFLOW)
        assert stored_workflow == new_workflow

    @pytest.mark.asyncio
    async def test_switch_provider_updates_executor_provider(
        self, executor, mock_old_provider, mock_new_provider
    ):
        """Test that switch_provider updates the executor's provider reference."""
        # Arrange
        assert executor.provider == mock_old_provider
        
        # Act
        await executor.switch_provider(mock_new_provider)
        
        # Assert
        assert executor.provider == mock_new_provider

    @pytest.mark.asyncio
    async def test_switch_provider_emits_event(
        self, executor, mock_new_provider, event_bus
    ):
        """Test that switch_provider emits ProviderSwitched event."""
        # Arrange
        events = []
        
        async def capture_event(event):
            events.append(event)
        
        from agentlegatus.core.event_bus import EventType
        event_bus.subscribe(EventType.PROVIDER_SWITCHED, capture_event)
        
        # Act
        await executor.switch_provider(mock_new_provider)
        
        # Assert
        assert len(events) == 1
        event = events[0]
        assert event.event_type == EventType.PROVIDER_SWITCHED
        assert event.source == "WorkflowExecutor"
        assert "old_provider" in event.data
        assert "new_provider" in event.data
        assert event.data["workflow_id"] == "test_workflow_123"

    @pytest.mark.asyncio
    async def test_switch_provider_rollback_on_failure(
        self, executor, mock_old_provider, mock_new_provider
    ):
        """Test that switch_provider rolls back to old provider on failure."""
        # Arrange
        mock_new_provider.import_state = Mock(side_effect=Exception("Import failed"))
        
        # Act & Assert
        with pytest.raises(Exception, match="Import failed"):
            await executor.switch_provider(mock_new_provider)
        
        # Verify rollback - old provider is still active
        assert executor.provider == mock_old_provider

    @pytest.mark.asyncio
    async def test_switch_provider_with_no_current_workflow(
        self, executor, mock_old_provider, mock_new_provider
    ):
        """Test switch_provider when no current workflow exists."""
        # Arrange - ensure no workflow is stored
        # (default state has no workflow)
        
        # Mock to_portable_graph should not be called when no workflow exists
        from agentlegatus.core.graph import PortableExecutionGraph
        empty_graph = PortableExecutionGraph()
        mock_old_provider.to_portable_graph = Mock(return_value=empty_graph)
        
        # Act
        await executor.switch_provider(mock_new_provider)
        
        # Assert - should succeed with empty graph
        assert executor.provider == mock_new_provider
        # to_portable_graph should not be called when no workflow exists
        mock_old_provider.to_portable_graph.assert_not_called()

    @pytest.mark.asyncio
    async def test_switch_provider_preserves_workflow_id(
        self, executor, mock_new_provider
    ):
        """Test that switch_provider preserves workflow ID."""
        # Arrange
        original_workflow_id = "workflow_xyz_789"
        executor._current_workflow_id = original_workflow_id
        
        # Act
        await executor.switch_provider(mock_new_provider)
        
        # Assert
        assert executor._current_workflow_id == original_workflow_id

    @pytest.mark.asyncio
    async def test_switch_provider_event_contains_graph_info(
        self, executor, mock_new_provider, event_bus
    ):
        """Test that ProviderSwitched event contains graph information."""
        # Arrange
        events = []
        
        async def capture_event(event):
            events.append(event)
        
        from agentlegatus.core.event_bus import EventType
        event_bus.subscribe(EventType.PROVIDER_SWITCHED, capture_event)
        
        # Act
        await executor.switch_provider(mock_new_provider)
        
        # Assert
        event = events[0]
        assert "graph_nodes" in event.data
        assert "graph_edges" in event.data
        assert "state_keys" in event.data
        assert isinstance(event.data["state_keys"], list)


class TestWorkflowExecutorExecuteGraph:
    """Tests for WorkflowExecutor.execute_graph()."""

    @pytest.fixture
    def state_manager(self):
        backend = InMemoryStateBackend()
        return StateManager(backend=backend)

    @pytest.fixture
    def event_bus(self):
        return EventBus()

    @pytest.fixture
    def mock_provider(self):
        provider = Mock()
        provider.execute_agent = AsyncMock(return_value={"output": "ok"})
        return provider

    @pytest.fixture
    def executor(self, mock_provider, state_manager, event_bus):
        return WorkflowExecutor(
            provider=mock_provider,
            state_manager=state_manager,
            tool_registry=Mock(),
            event_bus=event_bus,
        )

    def _make_graph(self, nodes, edges=None):
        """Helper to build a PortableExecutionGraph."""
        from agentlegatus.core.graph import PEGEdge, PEGNode, PortableExecutionGraph

        graph = PortableExecutionGraph()
        for n in nodes:
            graph.add_node(
                PEGNode(
                    node_id=n["id"],
                    node_type=n.get("type", "agent"),
                    config=n.get("config", {}),
                    inputs=n.get("inputs", []),
                    outputs=n.get("outputs", []),
                )
            )
        for e in edges or []:
            graph.add_edge(PEGEdge(source=e[0], target=e[1]))
        return graph

    @pytest.mark.asyncio
    async def test_execute_empty_graph(self, executor):
        """Executing an empty graph returns empty results."""
        from agentlegatus.core.graph import PortableExecutionGraph

        graph = PortableExecutionGraph()
        result = await executor.execute_graph(graph, {})
        assert result == {}

    @pytest.mark.asyncio
    async def test_execute_single_node(self, executor, mock_provider):
        """Single-node graph executes the node and returns its result."""
        graph = self._make_graph([{"id": "a"}])
        result = await executor.execute_graph(graph, {})
        assert "a" in result
        mock_provider.execute_agent.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_execute_respects_dependency_order(self, executor, mock_provider):
        """Nodes execute in topological order respecting edges."""
        call_order = []
        original = mock_provider.execute_agent

        async def track_call(agent, input_data, state):
            call_order.append(input_data["node_id"])
            return {"output": input_data["node_id"]}

        mock_provider.execute_agent = AsyncMock(side_effect=track_call)

        graph = self._make_graph(
            [{"id": "a"}, {"id": "b", "inputs": ["a"]}, {"id": "c", "inputs": ["b"]}],
            [("a", "b"), ("b", "c")],
        )
        await executor.execute_graph(graph, {})
        assert call_order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_execute_passes_prior_results_as_inputs(self, executor, mock_provider):
        """Node inputs are populated from prior node results."""
        captured_inputs = {}

        async def capture(agent, input_data, state):
            captured_inputs[input_data["node_id"]] = input_data.get("inputs")
            return f"result_{input_data['node_id']}"

        mock_provider.execute_agent = AsyncMock(side_effect=capture)

        graph = self._make_graph(
            [{"id": "a"}, {"id": "b", "inputs": ["a"]}],
            [("a", "b")],
        )
        await executor.execute_graph(graph, {})
        assert captured_inputs["b"] == {"a": "result_a"}

    @pytest.mark.asyncio
    async def test_execute_initializes_state(self, executor, state_manager):
        """Initial state is written to the state manager before execution."""
        from agentlegatus.core.graph import PortableExecutionGraph

        graph = PortableExecutionGraph()
        await executor.execute_graph(graph, {"foo": "bar"})
        val = await state_manager.get("foo", scope=StateScope.WORKFLOW)
        assert val == "bar"

    @pytest.mark.asyncio
    async def test_execute_emits_step_events(self, executor, event_bus):
        """StepStarted and StepCompleted events are emitted for each node."""
        from agentlegatus.core.event_bus import EventType

        events = []

        async def capture(event):
            events.append(event)

        event_bus.subscribe(EventType.STEP_STARTED, capture)
        event_bus.subscribe(EventType.STEP_COMPLETED, capture)

        graph = self._make_graph([{"id": "x"}])
        await executor.execute_graph(graph, {})

        types = [e.event_type for e in events]
        assert EventType.STEP_STARTED in types
        assert EventType.STEP_COMPLETED in types

    @pytest.mark.asyncio
    async def test_execute_emits_step_failed_on_error(self, executor, mock_provider, event_bus):
        """StepFailed event is emitted when a node raises."""
        from agentlegatus.core.event_bus import EventType

        mock_provider.execute_agent = AsyncMock(side_effect=RuntimeError("boom"))

        events = []

        async def capture(event):
            events.append(event)

        event_bus.subscribe(EventType.STEP_FAILED, capture)

        graph = self._make_graph([{"id": "x"}])
        with pytest.raises(RuntimeError, match="boom"):
            await executor.execute_graph(graph, {})

        assert len(events) == 1
        assert events[0].data["error"] == "boom"

    @pytest.mark.asyncio
    async def test_execute_invalid_graph_raises(self, executor):
        """Executing an invalid graph (with cycles) raises ValueError."""
        from agentlegatus.core.graph import PEGEdge, PEGNode, PortableExecutionGraph

        graph = PortableExecutionGraph()
        graph.nodes["a"] = PEGNode(node_id="a", node_type="agent", config={}, inputs=[], outputs=[])
        graph.nodes["b"] = PEGNode(node_id="b", node_type="agent", config={}, inputs=[], outputs=[])
        graph.edges = [PEGEdge(source="a", target="b"), PEGEdge(source="b", target="a")]

        with pytest.raises(ValueError, match="Graph validation failed"):
            await executor.execute_graph(graph, {})

    @pytest.mark.asyncio
    async def test_execute_timeout_enforcement(self, executor, mock_provider):
        """A node with a timeout that exceeds it raises TimeoutError."""
        import asyncio

        async def slow_agent(agent, input_data, state):
            await asyncio.sleep(10)
            return "done"

        mock_provider.execute_agent = AsyncMock(side_effect=slow_agent)

        graph = self._make_graph([{"id": "slow", "config": {"timeout": 0.05}}])
        with pytest.raises(asyncio.TimeoutError):
            await executor.execute_graph(graph, {})

    @pytest.mark.asyncio
    async def test_execute_no_timeout_when_not_configured(self, executor, mock_provider):
        """Nodes without a timeout config execute without timeout enforcement."""
        async def normal_agent(agent, input_data, state):
            return "ok"

        mock_provider.execute_agent = AsyncMock(side_effect=normal_agent)

        graph = self._make_graph([{"id": "a", "config": {}}])
        result = await executor.execute_graph(graph, {})
        assert result["a"] == "ok"

    @pytest.mark.asyncio
    async def test_execute_stores_results_in_step_scope(self, executor, state_manager, mock_provider):
        """Each node result is stored in the STEP scope of the state manager."""
        mock_provider.execute_agent = AsyncMock(return_value="step_result")

        graph = self._make_graph([{"id": "n1"}])
        await executor.execute_graph(graph, {})

        stored = await state_manager.get("result_n1", scope=StateScope.STEP)
        assert stored == "step_result"
