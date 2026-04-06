"""Legatus top-level orchestrator for multi-agent workflows.

The Legatus sits at the top of the Roman military hierarchy and manages
the complete workflow lifecycle: start, monitor, cancel, and timeout
enforcement. It coordinates one or more Centurion controllers and emits
global workflow events through the EventBus.
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any

from agentlegatus.core.event_bus import Event, EventBus, EventType
from agentlegatus.core.executor import WorkflowExecutor
from agentlegatus.core.state import StateManager, StateScope
from agentlegatus.core.workflow import (
    WorkflowDefinition,
    WorkflowResult,
    WorkflowStatus,
)
from agentlegatus.exceptions import WorkflowTimeoutError
from agentlegatus.hierarchy.centurion import Centurion
from agentlegatus.security.rate_limiter import RateLimitConfig, RateLimiter
from agentlegatus.utils.logging import get_logger

logger = get_logger(__name__)


class Legatus:
    """Top-level orchestrator for multi-agent workflows.

    The Legatus manages the complete lifecycle of workflow execution,
    coordinating Centurion controllers, emitting events, collecting
    metrics, and enforcing timeouts and cancellation.
    """

    def __init__(self, config: dict[str, Any], event_bus: EventBus) -> None:
        """Initialize Legatus with configuration and event bus.

        Args:
            config: Configuration dictionary for the orchestrator
            event_bus: Event bus for emitting workflow events
        """
        self.config = config
        self.event_bus = event_bus

        self._centurions: dict[str, Centurion] = {}
        self._workflow_statuses: dict[str, WorkflowStatus] = {}
        self._running_tasks: dict[str, asyncio.Task] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._rate_limiter = RateLimiter()

    def configure_rate_limit(self, workflow_id: str, config: RateLimitConfig) -> None:
        """Set a rate limit for a workflow.

        Args:
            workflow_id: Workflow identifier to rate-limit
            config: Rate limit configuration (max_requests, window_seconds)
        """
        self._rate_limiter.configure(workflow_id, config)

    async def add_centurion(self, centurion: Centurion) -> None:
        """Register a Centurion controller.

        Args:
            centurion: Centurion instance to register
        """
        self._centurions[centurion.name] = centurion
        logger.debug(f"Legatus registered centurion '{centurion.name}'")

    def get_status(self, workflow_id: str) -> WorkflowStatus:
        """Get current status of a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Current workflow status

        Raises:
            KeyError: If workflow_id is not found
        """
        if workflow_id not in self._workflow_statuses:
            raise KeyError(f"Workflow '{workflow_id}' not found")
        return self._workflow_statuses[workflow_id]

    async def execute_workflow(
        self,
        workflow_def: WorkflowDefinition,
        initial_state: dict[str, Any] | None = None,
        executor: WorkflowExecutor | None = None,
        state_manager: StateManager | None = None,
    ) -> WorkflowResult:
        """Execute a complete workflow with Centurion coordination.

        This is the main entry point for workflow execution. It:
        1. Validates the workflow definition
        2. Initialises execution context with a trace_id
        3. Emits WorkflowStarted event
        4. Delegates to a Centurion for step orchestration
        5. Collects execution metrics
        6. Emits WorkflowCompleted or WorkflowFailed event
        7. Returns a WorkflowResult

        If the workflow definition specifies a ``timeout``, execution is
        wrapped in ``asyncio.wait_for`` so that it is cancelled when the
        deadline is exceeded.

        Args:
            workflow_def: Validated workflow definition
            initial_state: Optional initial state for the workflow
            executor: Optional WorkflowExecutor for step execution
            state_manager: Optional StateManager for workflow state

        Returns:
            WorkflowResult with status, output, metrics, and execution_time
        """
        workflow_id = workflow_def.workflow_id
        trace_id = str(uuid.uuid4())
        start_time = time.monotonic()
        start_dt = datetime.now()

        # Enforce rate limit before proceeding
        self._rate_limiter.acquire(workflow_id)

        # Mark workflow as running
        self._workflow_statuses[workflow_id] = WorkflowStatus.RUNNING
        cancel_event = asyncio.Event()
        self._cancel_events[workflow_id] = cancel_event

        # Track the current task so cancel_workflow can interrupt it
        current_task = asyncio.current_task()
        if current_task is not None:
            self._running_tasks[workflow_id] = current_task

        # Emit WorkflowStarted event
        await self.event_bus.emit(
            Event(
                event_type=EventType.WORKFLOW_STARTED,
                timestamp=start_dt,
                source="legatus",
                data={
                    "workflow_id": workflow_id,
                    "name": workflow_def.name,
                    "provider": workflow_def.provider,
                    "strategy": workflow_def.execution_strategy.value,
                    "step_count": len(workflow_def.steps),
                },
                trace_id=trace_id,
            )
        )

        # Initialise state manager with initial state
        if state_manager and initial_state:
            for key, value in initial_state.items():
                await state_manager.set(key, value, scope=StateScope.WORKFLOW)

        # Select centurion — use the first registered one, or create a
        # default centurion matching the workflow's execution strategy.
        centurion = self._select_centurion(workflow_def)

        try:
            # Build the coroutine for the actual orchestration
            coro = centurion.orchestrate(
                workflow_def,
                state_manager,
                executor,
            )

            # Enforce timeout if configured
            if workflow_def.timeout is not None and workflow_def.timeout > 0:
                result_state = await self._execute_with_timeout(
                    coro, workflow_def.timeout, workflow_id, executor
                )
            else:
                result_state = await coro

            # Successful completion
            execution_time = time.monotonic() - start_time
            self._workflow_statuses[workflow_id] = WorkflowStatus.COMPLETED

            metrics = self._build_metrics(
                workflow_id,
                trace_id,
                workflow_def.provider,
                start_dt,
                execution_time,
                result_state,
            )

            await self.event_bus.emit(
                Event(
                    event_type=EventType.WORKFLOW_COMPLETED,
                    timestamp=datetime.now(),
                    source="legatus",
                    data={
                        "workflow_id": workflow_id,
                        "execution_time": execution_time,
                        "metrics": metrics,
                    },
                    trace_id=trace_id,
                )
            )

            return WorkflowResult(
                status=WorkflowStatus.COMPLETED,
                output=result_state,
                metrics=metrics,
                execution_time=execution_time,
            )

        except (asyncio.CancelledError, WorkflowTimeoutError):
            # Workflow was cancelled (either explicitly or via timeout)
            execution_time = time.monotonic() - start_time
            status = self._workflow_statuses.get(workflow_id, WorkflowStatus.CANCELLED)
            self._workflow_statuses[workflow_id] = status

            metrics = self._build_metrics(
                workflow_id,
                trace_id,
                workflow_def.provider,
                start_dt,
                execution_time,
                {},
            )

            return WorkflowResult(
                status=status,
                output=None,
                metrics=metrics,
                execution_time=execution_time,
            )

        except Exception as exc:
            execution_time = time.monotonic() - start_time
            self._workflow_statuses[workflow_id] = WorkflowStatus.FAILED

            metrics = self._build_metrics(
                workflow_id,
                trace_id,
                workflow_def.provider,
                start_dt,
                execution_time,
                {},
            )

            await self.event_bus.emit(
                Event(
                    event_type=EventType.WORKFLOW_FAILED,
                    timestamp=datetime.now(),
                    source="legatus",
                    data={
                        "workflow_id": workflow_id,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "execution_time": execution_time,
                    },
                    trace_id=trace_id,
                )
            )

            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                output=None,
                metrics=metrics,
                execution_time=execution_time,
                error=exc,
            )

        finally:
            self._cancel_events.pop(workflow_id, None)
            self._running_tasks.pop(workflow_id, None)

    async def cancel_workflow(
        self,
        workflow_id: str,
        executor: WorkflowExecutor | None = None,
    ) -> bool:
        """Cancel a running workflow.

        Cancellation performs the following:
        1. Creates a checkpoint of the current state (if executor provided)
        2. Cancels the running asyncio task
        3. Emits a WorkflowCancelled event
        4. Cleans up resources

        Args:
            workflow_id: Identifier of the workflow to cancel
            executor: Optional executor for checkpointing state

        Returns:
            True if the workflow was cancelled, False if not found or
            not running
        """
        status = self._workflow_statuses.get(workflow_id)
        if status != WorkflowStatus.RUNNING:
            logger.warning(
                f"Cannot cancel workflow '{workflow_id}': "
                f"status is {status.value if status else 'unknown'}"
            )
            return False

        logger.info(f"Cancelling workflow '{workflow_id}'")

        # Create checkpoint before cancellation
        if executor is not None:
            try:
                checkpoint_id = f"cancel_{workflow_id}_{int(time.time())}"
                await executor.checkpoint_state(checkpoint_id)
                logger.info(f"Checkpoint '{checkpoint_id}' created before cancellation")
            except Exception as e:
                logger.warning(f"Failed to create checkpoint before cancellation: {e}")

        # Cancel the running task
        task = self._running_tasks.get(workflow_id)
        if task is not None and not task.done():
            task.cancel()

        # Signal cancellation via event
        cancel_event = self._cancel_events.get(workflow_id)
        if cancel_event is not None:
            cancel_event.set()

        # Update status
        self._workflow_statuses[workflow_id] = WorkflowStatus.CANCELLED

        # Emit WorkflowCancelled event
        await self.event_bus.emit(
            Event(
                event_type=EventType.WORKFLOW_CANCELLED,
                timestamp=datetime.now(),
                source="legatus",
                data={
                    "workflow_id": workflow_id,
                    "reason": "user_requested",
                },
            )
        )

        # Clean up
        self._cancel_events.pop(workflow_id, None)
        self._running_tasks.pop(workflow_id, None)

        logger.info(f"Workflow '{workflow_id}' cancelled")
        return True

    # ------------------------------------------------------------------
    # Timeout enforcement (Requirement 28.1, 28.4)
    # ------------------------------------------------------------------

    async def _execute_with_timeout(
        self,
        coro: Any,
        timeout: float,
        workflow_id: str,
        executor: WorkflowExecutor | None = None,
    ) -> dict[str, Any]:
        """Execute a coroutine with timeout enforcement.

        If the timeout is exceeded, a checkpoint is created (if possible)
        and the workflow is marked as CANCELLED.

        Args:
            coro: Coroutine to execute
            timeout: Timeout in seconds
            workflow_id: Workflow identifier for status tracking
            executor: Optional executor for checkpointing

        Returns:
            Result from the coroutine

        Raises:
            asyncio.CancelledError: When timeout is exceeded
        """
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Workflow '{workflow_id}' exceeded timeout of {timeout}s")

            # Create checkpoint before timeout cancellation (Req 15.9)
            if executor is not None:
                try:
                    checkpoint_id = f"timeout_{workflow_id}_{int(time.time())}"
                    await executor.checkpoint_state(checkpoint_id)
                    logger.info(f"Checkpoint '{checkpoint_id}' created before timeout cancellation")
                except Exception as e:
                    logger.warning(f"Failed to create checkpoint on timeout: {e}")

            self._workflow_statuses[workflow_id] = WorkflowStatus.CANCELLED

            # Emit cancellation event with timeout reason
            await self.event_bus.emit(
                Event(
                    event_type=EventType.WORKFLOW_CANCELLED,
                    timestamp=datetime.now(),
                    source="legatus",
                    data={
                        "workflow_id": workflow_id,
                        "reason": "timeout",
                        "timeout": timeout,
                    },
                )
            )

            raise WorkflowTimeoutError(
                workflow_id=workflow_id,
                timeout=timeout,
            ) from None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_centurion(self, workflow_def: WorkflowDefinition) -> Centurion:
        """Select or create a Centurion for the workflow.

        If centurions have been registered, the first one is used.
        Otherwise a default centurion is created matching the workflow's
        execution strategy.

        Args:
            workflow_def: Workflow definition

        Returns:
            Centurion instance
        """
        if self._centurions:
            return next(iter(self._centurions.values()))

        # Create a default centurion with the workflow's strategy
        return Centurion(
            name=f"default_{workflow_def.workflow_id}",
            strategy=workflow_def.execution_strategy,
            event_bus=self.event_bus,
        )

    @staticmethod
    def _build_metrics(
        workflow_id: str,
        trace_id: str,
        provider: str,
        start_dt: datetime,
        execution_time: float,
        result_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a metrics dictionary for the WorkflowResult.

        Args:
            workflow_id: Workflow identifier
            trace_id: Trace identifier
            provider: Provider name
            start_dt: Execution start datetime
            execution_time: Total execution time in seconds
            result_state: Final workflow state

        Returns:
            Metrics dictionary
        """
        return {
            "workflow_id": workflow_id,
            "trace_id": trace_id,
            "provider": provider,
            "start_time": start_dt.isoformat(),
            "duration": execution_time,
            "cost": result_state.get("total_cost", 0.0),
            "token_usage": {
                "input_tokens": result_state.get("input_tokens", 0),
                "output_tokens": result_state.get("output_tokens", 0),
            },
        }
