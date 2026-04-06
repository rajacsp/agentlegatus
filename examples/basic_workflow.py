#!/usr/bin/env python3
"""Example 1: Basic Sequential Workflow Execution.

Demonstrates how to define a simple multi-step workflow and execute it
through the Legatus orchestrator using the MockProvider.
"""

import asyncio

from agentlegatus.core.event_bus import EventBus
from agentlegatus.core.executor import WorkflowExecutor
from agentlegatus.core.state import InMemoryStateBackend, StateManager
from agentlegatus.core.workflow import (
    ExecutionStrategy,
    RetryPolicy,
    WorkflowDefinition,
    WorkflowStep,
)
from agentlegatus.hierarchy.centurion import Centurion
from agentlegatus.hierarchy.legatus import Legatus
from agentlegatus.providers.mock import MockProvider
from agentlegatus.providers.registry import ProviderRegistry
from agentlegatus.tools.registry import ToolRegistry


async def main():
    # --- 1. Set up infrastructure ---
    event_bus = EventBus()
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend=backend, event_bus=event_bus)
    provider = MockProvider(config={"name": "mock"})
    tool_registry = ToolRegistry()

    # Register the provider
    registry = ProviderRegistry()
    registry.register_provider("mock", MockProvider)

    # Create the workflow executor
    executor = WorkflowExecutor(
        provider=provider,
        state_manager=state_manager,
        tool_registry=tool_registry,
        event_bus=event_bus,
    )

    # --- 2. Define a workflow ---
    workflow = WorkflowDefinition(
        workflow_id="data-pipeline-001",
        name="Data Processing Pipeline",
        description="Fetch, transform, and store data",
        version="1.0.0",
        provider="mock",
        execution_strategy=ExecutionStrategy.SEQUENTIAL,
        timeout=120.0,
        initial_state={"source_url": "https://api.example.com/data"},
        steps=[
            WorkflowStep(
                step_id="fetch",
                step_type="agent",
                config={"agent_id": "fetcher", "model": "mock-model"},
            ),
            WorkflowStep(
                step_id="transform",
                step_type="agent",
                config={"agent_id": "transformer", "model": "mock-model"},
                depends_on=["fetch"],
                retry_policy=RetryPolicy(max_attempts=3, initial_delay=1.0),
            ),
            WorkflowStep(
                step_id="store",
                step_type="agent",
                config={"agent_id": "storer", "model": "mock-model"},
                depends_on=["transform"],
            ),
        ],
    )

    # Validate before running
    is_valid, errors = workflow.validate()
    print(f"Workflow valid: {is_valid}")
    if errors:
        print(f"  Errors: {errors}")
        return

    # --- 3. Create the Legatus orchestrator and execute ---
    legatus = Legatus(config={"name": "example"}, event_bus=event_bus)

    result = await legatus.execute_workflow(
        workflow_def=workflow,
        initial_state=workflow.initial_state,
        executor=executor,
        state_manager=state_manager,
    )

    # --- 4. Inspect results ---
    print(f"\nWorkflow status : {result.status.value}")
    print(f"Execution time  : {result.execution_time:.3f}s")
    print(f"Output          : {result.output}")
    if result.metrics:
        print(f"Metrics         : {result.metrics}")

    # Check events that were emitted
    history = event_bus.get_event_history()
    print(f"\nEvents emitted  : {len(history)}")
    for evt in history:
        print(f"  [{evt.event_type.value}] {evt.data}")


if __name__ == "__main__":
    asyncio.run(main())
