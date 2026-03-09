"""Workflow executor for executing workflow steps."""

from typing import Any, Dict, Optional

from agentlegatus.core.event_bus import Event, EventBus, EventType
from agentlegatus.core.state import StateManager, StateScope
from agentlegatus.core.workflow import WorkflowStep
from agentlegatus.utils.logging import get_logger

logger = get_logger(__name__)


class WorkflowExecutor:
    """Executes workflow steps using provider abstraction."""

    def __init__(
        self,
        provider: Any,  # BaseProvider - using Any to avoid circular import
        state_manager: StateManager,
        tool_registry: Any,  # ToolRegistry - using Any to avoid circular import
        event_bus: EventBus,
    ):
        """
        Initialize executor with provider and dependencies.

        Args:
            provider: Provider instance for executing agents
            state_manager: State manager for workflow state
            tool_registry: Tool registry for tool invocations
            event_bus: Event bus for emitting events
        """
        self.provider = provider
        self.state_manager = state_manager
        self.tool_registry = tool_registry
        self.event_bus = event_bus
        self._current_workflow_id: Optional[str] = None
        self._completed_steps: set[str] = set()

    async def execute_step(
        self, step: WorkflowStep, context: Dict[str, Any]
    ) -> Any:
        """
        Execute a single workflow step.

        Args:
            step: Workflow step to execute
            context: Execution context

        Returns:
            Step execution result
        """
        # TODO: Implement in task 11.1
        raise NotImplementedError("execute_step will be implemented in task 11.1")

    async def execute_graph(
        self, graph: Any, initial_state: Dict[str, Any]  # PortableExecutionGraph
    ) -> Dict[str, Any]:
        """
        Execute a complete portable execution graph.

        Args:
            graph: Portable execution graph to execute
            initial_state: Initial state for execution

        Returns:
            Final execution state
        """
        # TODO: Implement in task 11.5
        raise NotImplementedError("execute_graph will be implemented in task 11.5")

    async def switch_provider(self, new_provider: Any) -> None:  # BaseProvider
        """
        Switch to a different provider at runtime.

        This method performs the following steps:
        1. Export state from current provider
        2. Convert current workflow to PortableExecutionGraph
        3. Validate the portable graph
        4. Import state into new provider
        5. Convert portable graph to new provider's format
        6. Emit ProviderSwitched event

        Args:
            new_provider: New provider instance to switch to

        Raises:
            ValueError: If portable graph validation fails
            Exception: If state export/import or graph conversion fails

        Requirements:
            - 5.1: Export state from current provider
            - 5.2: Convert workflow to PortableExecutionGraph
            - 5.3: Validate portable graph
            - 5.4: Instantiate new provider
            - 5.5: Import exported state into new provider
            - 5.6: Convert PortableExecutionGraph to new provider's format
            - 5.7: Update StateManager with new workflow definition
            - 5.8: Emit ProviderSwitched event
        """
        logger.info(
            f"Switching provider from {type(self.provider).__name__} "
            f"to {type(new_provider).__name__}"
        )

        old_provider = self.provider
        old_provider_name = type(old_provider).__name__
        new_provider_name = type(new_provider).__name__

        try:
            # Step 1: Export state from current provider (Requirement 5.1)
            logger.debug("Exporting state from current provider")
            exported_state = old_provider.export_state()
            logger.debug(f"Exported state keys: {list(exported_state.keys())}")

            # Step 2: Convert current workflow to PortableExecutionGraph (Requirement 5.2)
            # Note: We need to get the current workflow from the provider
            # For now, we'll create a minimal workflow representation
            logger.debug("Converting workflow to PortableExecutionGraph")

            # Get current workflow from state manager if available
            current_workflow = await self.state_manager.get(
                key="current_workflow",
                scope=StateScope.WORKFLOW,
            )

            if current_workflow is not None:
                portable_graph = old_provider.to_portable_graph(current_workflow)
            else:
                # If no workflow is stored, create an empty portable graph
                # This handles the case where we're switching before any workflow execution
                from agentlegatus.core.graph import PortableExecutionGraph
                portable_graph = PortableExecutionGraph()
                logger.warning("No current workflow found, using empty portable graph")

            # Step 3: Validate portable graph (Requirement 5.3)
            logger.debug("Validating portable graph")
            is_valid, errors = portable_graph.validate()

            if not is_valid:
                error_msg = f"Portable graph validation failed: {', '.join(errors)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.debug("Portable graph validation successful")

            # Step 4: Import state into new provider (Requirement 5.5)
            logger.debug("Importing state into new provider")
            new_provider.import_state(exported_state)

            # Step 5: Convert portable graph to new provider's format (Requirement 5.6)
            logger.debug("Converting portable graph to new provider format")
            new_workflow = new_provider.from_portable_graph(portable_graph)

            # Step 6: Update StateManager with new workflow definition (Requirement 5.7)
            logger.debug("Updating StateManager with new workflow")
            await self.state_manager.set(
                key="current_workflow",
                value=new_workflow,
                scope=StateScope.WORKFLOW,
            )

            # Update the executor's provider reference
            self.provider = new_provider

            # Step 7: Emit ProviderSwitched event (Requirement 5.8)
            logger.info(
                f"Provider switch successful: {old_provider_name} -> {new_provider_name}"
            )

            await self.event_bus.emit(
                Event(
                    event_type=EventType.PROVIDER_SWITCHED,
                    timestamp=__import__("datetime").datetime.now(),
                    source="WorkflowExecutor",
                    data={
                        "old_provider": old_provider_name,
                        "new_provider": new_provider_name,
                        "workflow_id": self._current_workflow_id,
                        "state_keys": list(exported_state.keys()),
                        "graph_nodes": len(portable_graph.nodes),
                        "graph_edges": len(portable_graph.edges),
                    },
                )
            )

        except Exception as e:
            logger.error(
                f"Provider switch failed: {old_provider_name} -> {new_provider_name}. "
                f"Error: {e}",
                exc_info=True,
            )
            # Rollback: Keep the old provider
            self.provider = old_provider
            raise

    async def checkpoint_state(self, checkpoint_id: str) -> None:
        """
        Create a state checkpoint for recovery.

        This method saves the current execution state including:
        - All state scopes (workflow, step, agent, global)
        - Completed steps tracking
        - Current workflow ID

        Args:
            checkpoint_id: Unique identifier for the checkpoint

        Requirements:
            - 21.1: Save current execution state with checkpoint ID
            - 21.7: Create snapshot of all state scopes
        """
        logger.info(f"Creating checkpoint: {checkpoint_id}")

        try:
            # Store checkpoint metadata in workflow scope BEFORE creating snapshot
            # This ensures the metadata is included in the snapshot
            checkpoint_metadata = {
                "workflow_id": self._current_workflow_id,
                "completed_steps": list(self._completed_steps),
                "checkpoint_id": checkpoint_id,
            }

            await self.state_manager.set(
                key=f"checkpoint_metadata_{checkpoint_id}",
                value=checkpoint_metadata,
                scope=StateScope.WORKFLOW,
            )

            # Create snapshot of all state scopes using StateManager
            # This captures the checkpoint metadata we just stored
            await self.state_manager.create_snapshot(checkpoint_id)

            logger.info(
                f"Checkpoint {checkpoint_id} created successfully. "
                f"Completed steps: {len(self._completed_steps)}"
            )

        except Exception as e:
            logger.error(f"Failed to create checkpoint {checkpoint_id}: {e}", exc_info=True)
            raise

    async def restore_from_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Restore execution from a checkpoint.

        This method restores the execution state from a checkpoint including:
        - All state scopes (workflow, step, agent, global)
        - Completed steps tracking
        - Current workflow ID

        Args:
            checkpoint_id: Unique identifier of the checkpoint to restore

        Returns:
            Restored execution context containing workflow state

        Requirements:
            - 21.2: Load saved state and resume execution
            - 21.5: Skip already completed steps
            - 21.6: Restore StateManager to checkpointed state
        """
        logger.info(f"Restoring from checkpoint: {checkpoint_id}")

        try:
            # Restore snapshot of all state scopes using StateManager
            await self.state_manager.restore_snapshot(checkpoint_id)

            # Retrieve checkpoint metadata
            checkpoint_metadata = await self.state_manager.get(
                key=f"checkpoint_metadata_{checkpoint_id}",
                scope=StateScope.WORKFLOW,
            )

            if checkpoint_metadata is None:
                raise ValueError(
                    f"Checkpoint metadata not found for checkpoint_id: {checkpoint_id}"
                )

            # Restore executor state
            self._current_workflow_id = checkpoint_metadata.get("workflow_id")
            self._completed_steps = set(checkpoint_metadata.get("completed_steps", []))

            logger.info(
                f"Checkpoint {checkpoint_id} restored successfully. "
                f"Workflow ID: {self._current_workflow_id}, "
                f"Completed steps: {len(self._completed_steps)}"
            )

            # Get all workflow state to return as context
            workflow_state = await self.state_manager.get_all(scope=StateScope.WORKFLOW)

            return {
                "workflow_id": self._current_workflow_id,
                "completed_steps": list(self._completed_steps),
                "workflow_state": workflow_state,
            }

        except Exception as e:
            logger.error(
                f"Failed to restore from checkpoint {checkpoint_id}: {e}", exc_info=True
            )
            raise
