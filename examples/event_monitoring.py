#!/usr/bin/env python3
"""Example 4: Event-Driven Monitoring.

Demonstrates subscribing to EventBus events to monitor workflow execution
in real time, including filtering event history.
"""

import asyncio
from datetime import datetime, timedelta

from agentlegatus.core.event_bus import Event, EventBus, EventType
from agentlegatus.core.executor import WorkflowExecutor
from agentlegatus.core.state import InMemoryStateBackend, StateManager
from agentlegatus.core.workflow import (
    ExecutionStrategy,
    WorkflowDefinition,
    WorkflowStep,
)
from agentlegatus.hierarchy.legatus import Legatus
from agentlegatus.providers.mock import MockProvider
from agentlegatus.tools.registry import ToolRegistry


# --- Custom event handlers ---

async def log_workflow_started(event: Event):
    data = event.data
    print(f"[MONITOR] Workflow '{data.get('name')}' started "
          f"(id={data.get('workflow_id')}, strategy={data.get('strategy')})")


async def log_step_event(event: Event):
    data = event.data
    step = data.get("step_id", "?")
    if event.event_type == EventType.STEP_STARTED:
        print(f"[MONITOR]   Step '{step}' started")
    elif event.event_type == EventType.STEP_COMPLETED:
        print(f"[MONITOR]   Step '{step}' completed")
    elif event.event_type == EventType.STEP_FAILED:
        print(f"[MONITOR]   Step '{step}' FAILED — {data.get('error')}")


async def log_workflow_ended(event: Event):
    data = event.data
    status = data.get("status", event.event_type.value)
    print(f"[MONITOR] Workflow ended — {status}")


async def log_state_change(event: Event):
    data = event.data
    print(f"[MONITOR]   State change: {data.get('key')} "
          f"(scope={data.get('scope')}, op={data.get('operation')})")


async def main():
    # --- 1. Set up infrastructure ---
    event_bus = EventBus()
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend=backend, event_bus=event_bus)
    provider = MockProvider(config={"name": "mock"})
    tool_registry = ToolRegistry()

    executor = WorkflowExecutor(
        provider=provider,
        state_manager=state_manager,
        tool_registry=tool_registry,
        event_bus=event_bus,
    )

    # --- 2. Subscribe to events ---
    sub_ids = []
    sub_ids.append(event_bus.subscribe(EventType.WORKFLOW_STARTED, log_workflow_started))
    sub_ids.append(event_bus.subscribe(EventType.STEP_STARTED, log_step_event))
    sub_ids.append(event_bus.subscribe(EventType.STEP_COMPLETED, log_step_event))
    sub_ids.append(event_bus.subscribe(EventType.STEP_FAILED, log_step_event))
    sub_ids.append(event_bus.subscribe(EventType.WORKFLOW_COMPLETED, log_workflow_ended))
    sub_ids.append(event_bus.subscribe(EventType.WORKFLOW_FAILED, log_workflow_ended))
    sub_ids.append(event_bus.subscribe(EventType.STATE_UPDATED, log_state_change))

    # --- 3. Execute a workflow ---
    workflow = WorkflowDefinition(
        workflow_id="monitored-wf-001",
        name="Monitored Pipeline",
        description="A workflow with real-time event monitoring",
        version="1.0.0",
        provider="mock",
        execution_strategy=ExecutionStrategy.SEQUENTIAL,
        steps=[
            WorkflowStep(step_id="ingest", step_type="agent",
                         config={"agent_id": "ingestor", "model": "mock-model"}),
            WorkflowStep(step_id="analyse", step_type="agent",
                         config={"agent_id": "analyser", "model": "mock-model"},
                         depends_on=["ingest"]),
        ],
    )

    legatus = Legatus(config={"name": "example"}, event_bus=event_bus)
    print("=== Executing workflow with live monitoring ===\n")

    result = await legatus.execute_workflow(
        workflow_def=workflow,
        executor=executor,
        state_manager=state_manager,
    )
    # Let fire-and-forget handlers finish
    await asyncio.sleep(0.1)

    print(f"\nFinal status: {result.status.value}")

    # --- 4. Query event history ---
    print("\n=== Event history (last 5) ===")
    recent = event_bus.get_event_history(limit=5)
    for evt in recent:
        print(f"  {evt.event_type.value} @ {evt.timestamp:%H:%M:%S}")

    # Filter by type
    step_events = event_bus.get_event_history(event_type=EventType.STEP_COMPLETED)
    print(f"\nTotal STEP_COMPLETED events: {len(step_events)}")

    # --- 5. Unsubscribe ---
    for sid in sub_ids:
        event_bus.unsubscribe(sid)
    print("\nAll handlers unsubscribed.")


if __name__ == "__main__":
    asyncio.run(main())
