"""Centurion workflow controller for managing execution flow."""

import asyncio
from collections import deque
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from agentlegatus.core.event_bus import Event, EventBus, EventType
from agentlegatus.core.executor import WorkflowExecutor
from agentlegatus.core.state import StateManager, StateScope
from agentlegatus.core.workflow import (
    ExecutionStrategy,
    WorkflowDefinition,
    WorkflowStep,
)
from agentlegatus.hierarchy.cohort import Cohort
from agentlegatus.utils.logging import get_logger
from agentlegatus.utils.retry import execute_with_retry

logger = get_logger(__name__)


class Centurion:
    """Workflow controller managing execution flow.

    The Centurion sits between the Legatus (top-level orchestrator) and
    Cohorts (agent groups) in the Roman military hierarchy. It controls
    how workflow steps are executed — sequentially, in parallel, or
    conditionally — and coordinates with the WorkflowExecutor for
    actual step execution.
    """

    def __init__(
        self,
        name: str,
        strategy: ExecutionStrategy,
        event_bus: EventBus,
    ) -> None:
        """Initialize Centurion with execution strategy.

        Args:
            name: Human-readable name for this centurion
            strategy: Execution strategy (SEQUENTIAL, PARALLEL, CONDITIONAL)
            event_bus: Event bus for emitting step events
        """
        self.name = name
        self.strategy = strategy
        self.event_bus = event_bus

        self._cohorts: Dict[str, Cohort] = {}
        self._executor: Optional[WorkflowExecutor] = None

    async def add_cohort(self, cohort: Cohort) -> None:
        """Register a Cohort of agents.

        Args:
            cohort: Cohort instance to register
        """
        self._cohorts[cohort.name] = cohort
        logger.debug(f"Centurion '{self.name}' registered cohort '{cohort.name}'")


    def build_execution_plan(
        self, steps: List[WorkflowStep]
    ) -> List[WorkflowStep]:
        """Build an execution plan using topological sort.

        Validates that the step dependency graph forms a valid DAG (no
        cycles) and returns steps in dependency order.

        Args:
            steps: List of workflow steps with dependency information

        Returns:
            Steps ordered by topological sort of the dependency graph

        Raises:
            ValueError: If the dependency graph contains cycles or
                        references non-existent steps
        """
        step_map: Dict[str, WorkflowStep] = {s.step_id: s for s in steps}
        all_ids = set(step_map.keys())

        # Validate all dependencies reference existing steps
        for step in steps:
            for dep in step.depends_on:
                if dep not in all_ids:
                    raise ValueError(
                        f"Step '{step.step_id}' depends on non-existent step '{dep}'"
                    )

        # Kahn's algorithm for topological sort
        in_degree: Dict[str, int] = {sid: 0 for sid in all_ids}
        # Build adjacency: dep -> step (dep must run before step)
        adjacency: Dict[str, List[str]] = {sid: [] for sid in all_ids}

        for step in steps:
            for dep in step.depends_on:
                adjacency[dep].append(step.step_id)
                in_degree[step.step_id] += 1

        queue: deque[str] = deque(
            sid for sid, deg in in_degree.items() if deg == 0
        )
        order: List[str] = []

        while queue:
            sid = queue.popleft()
            order.append(sid)
            for successor in adjacency[sid]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        if len(order) != len(all_ids):
            raise ValueError(
                "Workflow step dependencies contain a cycle — "
                "steps do not form a valid DAG"
            )

        return [step_map[sid] for sid in order]

    # ------------------------------------------------------------------
    # Sequential execution
    # ------------------------------------------------------------------

    async def execute_sequential(
        self,
        execution_plan: List[WorkflowStep],
        executor: WorkflowExecutor,
        state_manager: StateManager,
    ) -> Dict[str, Any]:
        """Execute workflow steps one at a time in dependency order.

        Args:
            execution_plan: Steps in topological order
            executor: WorkflowExecutor for running individual steps
            state_manager: State manager for reading/writing state

        Returns:
            Final workflow state dictionary

        Raises:
            Exception: Propagates any step execution failure
        """
        for step in execution_plan:
            await self._execute_single_step(step, executor, state_manager)

        return await state_manager.get_all(scope=StateScope.WORKFLOW)

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------

    async def execute_parallel(
        self,
        execution_plan: List[WorkflowStep],
        executor: WorkflowExecutor,
        state_manager: StateManager,
        max_concurrency: int = 0,
    ) -> Dict[str, Any]:
        """Execute independent steps concurrently, respecting dependencies.

        Steps whose dependencies are all satisfied are launched together
        via ``asyncio.gather``.  A concurrency limit can be set via
        *max_concurrency* (0 means unlimited).

        Args:
            execution_plan: Steps in topological order
            executor: WorkflowExecutor for running individual steps
            state_manager: State manager for reading/writing state
            max_concurrency: Maximum concurrent steps (0 = unlimited)

        Returns:
            Final workflow state dictionary

        Raises:
            Exception: Propagates the first step failure encountered
        """
        completed: set[str] = set()
        step_map: Dict[str, WorkflowStep] = {s.step_id: s for s in execution_plan}
        remaining = set(step_map.keys())

        while remaining:
            # Identify steps whose dependencies are all completed
            ready = [
                step_map[sid]
                for sid in remaining
                if all(dep in completed for dep in step_map[sid].depends_on)
            ]

            if not ready:
                raise RuntimeError(
                    "Deadlock detected: no steps are ready but some remain"
                )

            # Apply concurrency limit
            if max_concurrency > 0:
                ready = ready[:max_concurrency]

            # Launch ready steps concurrently
            tasks = [
                self._execute_single_step(step, executor, state_manager)
                for step in ready
            ]

            try:
                await asyncio.gather(*tasks)
            except Exception:
                # Cancel remaining parallel tasks on first failure
                # asyncio.gather already propagates the first exception
                raise

            for step in ready:
                completed.add(step.step_id)
                remaining.discard(step.step_id)

        return await state_manager.get_all(scope=StateScope.WORKFLOW)

    # ------------------------------------------------------------------
    # Conditional execution
    # ------------------------------------------------------------------

    async def execute_conditional(
        self,
        execution_plan: List[WorkflowStep],
        executor: WorkflowExecutor,
        state_manager: StateManager,
    ) -> Dict[str, Any]:
        """Execute steps with conditional evaluation.

        For each step, if a ``condition`` callable is present in the
        step config, it is evaluated against the current workflow state.
        Steps whose condition returns ``False`` are skipped.

        Args:
            execution_plan: Steps in topological order
            executor: WorkflowExecutor for running individual steps
            state_manager: State manager for reading/writing state

        Returns:
            Final workflow state dictionary

        Raises:
            Exception: Propagates condition evaluation or step execution
                       failures
        """
        for step in execution_plan:
            condition = step.config.get("condition")

            if condition is not None:
                workflow_state = await state_manager.get_all(
                    scope=StateScope.WORKFLOW
                )
                should_execute = await self.evaluate_condition(
                    condition, workflow_state
                )

                if not should_execute:
                    logger.info(
                        f"Step '{step.step_id}' skipped — condition evaluated to False"
                    )
                    await state_manager.set(
                        f"step_{step.step_id}_status",
                        "skipped",
                        scope=StateScope.WORKFLOW,
                    )
                    continue

            await self._execute_single_step(step, executor, state_manager)

        return await state_manager.get_all(scope=StateScope.WORKFLOW)

    async def evaluate_condition(
        self,
        condition: Callable[[Dict[str, Any]], bool],
        state: Dict[str, Any],
    ) -> bool:
        """Evaluate a conditional branching expression.

        Args:
            condition: Callable that receives the current workflow state
                       and returns a boolean
            state: Current workflow state dictionary

        Returns:
            True if the step should execute, False to skip

        Raises:
            Exception: Propagates any exception raised by the condition
        """
        result = condition(state)
        # Support both sync and async condition callables
        if asyncio.iscoroutine(result):
            result = await result
        return bool(result)


    # ------------------------------------------------------------------
    # Orchestrate (main entry point)
    # ------------------------------------------------------------------

    async def orchestrate(
        self,
        workflow_def: WorkflowDefinition,
        state_manager: StateManager,
        executor: Optional[WorkflowExecutor] = None,
    ) -> Dict[str, Any]:
        """Orchestrate workflow execution across strategies.

        This is the main entry point called by the Legatus. It builds
        an execution plan, selects the appropriate strategy, and
        coordinates the full workflow run.

        Args:
            workflow_def: Validated workflow definition
            state_manager: Initialized state manager
            executor: Optional WorkflowExecutor; if not provided, the
                      previously set ``_executor`` is used

        Returns:
            Final workflow state dictionary

        Raises:
            ValueError: If no executor is available or strategy is unknown
            Exception: Propagates step execution failures
        """
        if executor is not None:
            self._executor = executor
        if self._executor is None:
            raise ValueError(
                "No WorkflowExecutor available — pass one to orchestrate() "
                "or set it before calling"
            )

        logger.info(
            f"Centurion '{self.name}' orchestrating workflow "
            f"'{workflow_def.workflow_id}' with strategy {self.strategy.value}"
        )

        # Step 1: Build execution plan (topological sort)
        execution_plan = self.build_execution_plan(workflow_def.steps)

        # Step 2: Store initial state
        for key, value in workflow_def.initial_state.items():
            await state_manager.set(key, value, scope=StateScope.WORKFLOW)

        # Step 3: Dispatch to the appropriate strategy
        try:
            if self.strategy == ExecutionStrategy.SEQUENTIAL:
                result = await self.execute_sequential(
                    execution_plan, self._executor, state_manager
                )
            elif self.strategy == ExecutionStrategy.PARALLEL:
                max_conc = workflow_def.metadata.get("max_concurrency", 0)
                result = await self.execute_parallel(
                    execution_plan, self._executor, state_manager,
                    max_concurrency=max_conc,
                )
            elif self.strategy == ExecutionStrategy.CONDITIONAL:
                result = await self.execute_conditional(
                    execution_plan, self._executor, state_manager
                )
            else:
                raise ValueError(
                    f"Unknown execution strategy: {self.strategy}"
                )
        except Exception as e:
            logger.error(
                f"Centurion '{self.name}' workflow "
                f"'{workflow_def.workflow_id}' failed: {e}"
            )
            raise

        logger.info(
            f"Centurion '{self.name}' completed workflow "
            f"'{workflow_def.workflow_id}'"
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute_single_step(
        self,
        step: WorkflowStep,
        executor: WorkflowExecutor,
        state_manager: StateManager,
    ) -> Any:
        """Execute one step with event emission, state update, and retry.

        Args:
            step: The workflow step to execute
            executor: WorkflowExecutor instance
            state_manager: State manager for reading/writing state

        Returns:
            Step execution result

        Raises:
            Exception: Propagates step execution failure after retries
        """
        # Emit StepStarted
        await self.event_bus.emit(
            Event(
                event_type=EventType.STEP_STARTED,
                timestamp=datetime.now(),
                source=f"centurion:{self.name}",
                data={"step_id": step.step_id, "step_type": step.step_type},
            )
        )

        context = await state_manager.get_all(scope=StateScope.WORKFLOW)

        try:
            # Execute with retry if a retry policy is configured
            if step.retry_policy is not None:
                result = await execute_with_retry(
                    executor.execute_step,
                    step,
                    context,
                    retry_policy=step.retry_policy,
                    operation_name=f"step:{step.step_id}",
                )
            else:
                result = await executor.execute_step(step, context)

            # Update state with result
            await state_manager.set(
                f"step_{step.step_id}_result",
                result,
                scope=StateScope.WORKFLOW,
            )
            await state_manager.set(
                f"step_{step.step_id}_status",
                "completed",
                scope=StateScope.WORKFLOW,
            )

            # Emit StepCompleted
            await self.event_bus.emit(
                Event(
                    event_type=EventType.STEP_COMPLETED,
                    timestamp=datetime.now(),
                    source=f"centurion:{self.name}",
                    data={"step_id": step.step_id, "result": result},
                )
            )

            return result

        except Exception as e:
            # Emit StepFailed
            await self.event_bus.emit(
                Event(
                    event_type=EventType.STEP_FAILED,
                    timestamp=datetime.now(),
                    source=f"centurion:{self.name}",
                    data={
                        "step_id": step.step_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
            )
            raise
