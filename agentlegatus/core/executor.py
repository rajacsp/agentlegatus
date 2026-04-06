"""Workflow executor for executing workflow steps."""

import asyncio
from collections import deque
from datetime import datetime
from typing import Any

from agentlegatus.core.event_bus import Event, EventBus, EventType
from agentlegatus.core.graph import PEGNode, PortableExecutionGraph
from agentlegatus.core.state import StateManager, StateScope
from agentlegatus.core.workflow import WorkflowStep
from agentlegatus.exceptions import ProviderSwitchError
from agentlegatus.utils.logging import get_logger, log_error

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
        self._current_workflow_id: str | None = None
        self._completed_steps: set[str] = set()

    async def execute_step(self, step: WorkflowStep, context: dict[str, Any]) -> Any:
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
        self, graph: PortableExecutionGraph, initial_state: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute a complete portable execution graph.

        Uses Kahn's algorithm to identify independent nodes and runs them
        concurrently via ``asyncio.gather``, falling back to sequential
        execution when dependencies exist.  Per-node timeouts are enforced
        via ``asyncio.wait_for()``.

        Args:
            graph: Portable execution graph to execute
            initial_state: Initial state for execution

        Returns:
            Final execution state including all node results

        Raises:
            ValueError: If graph validation fails
            asyncio.TimeoutError: If a step exceeds its configured timeout
        """
        # Validate graph before execution
        is_valid, errors = graph.validate()
        if not is_valid:
            raise ValueError(f"Graph validation failed: {', '.join(errors)}")

        # Initialize state from initial_state
        for key, value in initial_state.items():
            await self.state_manager.set(key=key, value=value, scope=StateScope.WORKFLOW)

        # Build in-degree map for Kahn's parallel scheduling
        in_degree: dict[str, int] = dict.fromkeys(graph.nodes, 0)
        for edge in graph.edges:
            in_degree[edge.target] += 1

        results: dict[str, Any] = {}
        remaining = set(graph.nodes.keys())

        while remaining:
            # Collect all nodes whose dependencies are satisfied
            ready = [nid for nid in remaining if in_degree[nid] == 0]
            if not ready:
                raise RuntimeError("Deadlock: no nodes are ready but some remain")

            # Execute ready nodes concurrently
            async def _run_node(node_id: str) -> None:
                node = graph.get_node(node_id)
                if node is None:
                    return

                await self.event_bus.emit(
                    Event(
                        event_type=EventType.STEP_STARTED,
                        timestamp=datetime.now(),
                        source="WorkflowExecutor",
                        data={"node_id": node_id, "node_type": node.node_type},
                    )
                )

                try:
                    result = await self._execute_node_with_timeout(node, results)
                    results[node_id] = result

                    await self.state_manager.set(
                        key=f"result_{node_id}", value=result, scope=StateScope.STEP
                    )

                    await self.event_bus.emit(
                        Event(
                            event_type=EventType.STEP_COMPLETED,
                            timestamp=datetime.now(),
                            source="WorkflowExecutor",
                            data={"node_id": node_id, "result": result},
                        )
                    )
                except Exception as e:
                    log_error(
                        logger,
                        f"Graph node '{node_id}' execution failed",
                        e,
                        node_id=node_id,
                        node_type=node.node_type,
                        workflow_id=self._current_workflow_id,
                    )

                    await self.event_bus.emit(
                        Event(
                            event_type=EventType.STEP_FAILED,
                            timestamp=datetime.now(),
                            source="WorkflowExecutor",
                            data={
                                "node_id": node_id,
                                "error": str(e),
                                "error_type": type(e).__name__,
                            },
                        )
                    )
                    raise

            if len(ready) == 1:
                await _run_node(ready[0])
            else:
                await asyncio.gather(*(_run_node(nid) for nid in ready))

            # Update in-degrees for successors
            for nid in ready:
                remaining.discard(nid)
                for successor in graph.get_successors(nid):
                    in_degree[successor] -= 1

        return results

    async def _execute_node_with_timeout(self, node: PEGNode, prior_results: dict[str, Any]) -> Any:
        """
        Execute a single graph node, enforcing its timeout if configured.

        Args:
            node: The PEG node to execute
            prior_results: Results from previously executed nodes

        Returns:
            Node execution result
        """
        timeout = node.config.get("timeout")

        context = {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "config": node.config,
            "inputs": {inp: prior_results.get(inp) for inp in node.inputs},
        }

        coro = self.provider.execute_agent(
            agent=node.config.get("agent"),
            input_data=context,
            state=None,
        )

        if timeout is not None and timeout > 0:
            return await asyncio.wait_for(coro, timeout=timeout)
        return await coro

    @staticmethod
    def _topological_sort(graph: PortableExecutionGraph) -> list[str]:
        """
        Kahn's algorithm for topological sort of the graph.

        Args:
            graph: The portable execution graph

        Returns:
            List of node IDs in topological order
        """
        in_degree: dict[str, int] = dict.fromkeys(graph.nodes, 0)
        for edge in graph.edges:
            in_degree[edge.target] += 1

        queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order: list[str] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for successor in graph.get_successors(nid):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        return order

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
        exported_state = {}

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
            logger.info(f"Provider switch successful: {old_provider_name} -> {new_provider_name}")

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
            log_error(
                logger,
                f"Provider switch failed: {old_provider_name} -> {new_provider_name}",
                e,
                old_provider=old_provider_name,
                new_provider=new_provider_name,
                workflow_id=self._current_workflow_id,
            )
            # Rollback: Keep the old provider (Requirement 15.8)
            self.provider = old_provider
            # Re-import original state to ensure consistency
            try:
                old_provider.import_state(exported_state)
            except Exception:
                pass  # best-effort rollback of state

            raise ProviderSwitchError(
                old_provider=old_provider_name,
                new_provider=new_provider_name,
                reason=str(e),
                original_error=e,
            ) from e

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
            log_error(
                logger,
                f"Failed to create checkpoint {checkpoint_id}",
                e,
                checkpoint_id=checkpoint_id,
                workflow_id=self._current_workflow_id,
            )
            raise

    async def restore_from_checkpoint(self, checkpoint_id: str) -> dict[str, Any]:
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
            log_error(
                logger,
                f"Failed to restore from checkpoint {checkpoint_id}",
                e,
                checkpoint_id=checkpoint_id,
                workflow_id=self._current_workflow_id,
            )
            raise
