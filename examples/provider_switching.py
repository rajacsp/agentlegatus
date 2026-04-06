#!/usr/bin/env python3
"""Example 2: Provider Switching with State Preservation.

Demonstrates switching from MockProvider to LangGraphProvider at runtime
while preserving workflow state across the transition.
"""

import asyncio

from agentlegatus.core.event_bus import EventBus, EventType
from agentlegatus.core.executor import WorkflowExecutor
from agentlegatus.core.state import InMemoryStateBackend, StateManager, StateScope
from agentlegatus.providers.langgraph_provider import LangGraphProvider
from agentlegatus.providers.mock import MockProvider
from agentlegatus.providers.registry import ProviderRegistry
from agentlegatus.tools.registry import ToolRegistry


async def main():
    # --- 1. Set up infrastructure ---
    event_bus = EventBus()
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend=backend, event_bus=event_bus)
    tool_registry = ToolRegistry()

    # Register both providers
    registry = ProviderRegistry()
    registry.register_provider("mock", MockProvider)
    registry.register_provider("langgraph", LangGraphProvider)

    # Start with MockProvider
    mock_provider = MockProvider(config={"name": "mock"})
    executor = WorkflowExecutor(
        provider=mock_provider,
        state_manager=state_manager,
        tool_registry=tool_registry,
        event_bus=event_bus,
    )

    # --- 2. Populate some state on the mock provider ---
    mock_provider._state["user_query"] = "Summarise quarterly report"
    mock_provider._state["model_version"] = "v2"
    await state_manager.set("progress", "step-2-complete", scope=StateScope.WORKFLOW)

    print("=== Before switch ===")
    print(f"Provider        : {type(executor.provider).__name__}")
    print(f"Provider state  : {mock_provider.export_state()['state']}")
    sm_progress = await state_manager.get("progress", scope=StateScope.WORKFLOW)
    print(f"StateManager    : progress={sm_progress}")

    # --- 3. Switch to LangGraphProvider ---
    new_provider = LangGraphProvider(config={"name": "langgraph"})

    # Listen for the switch event
    switch_events = []

    async def on_switch(event):
        switch_events.append(event)

    event_bus.subscribe(EventType.PROVIDER_SWITCHED, on_switch)

    await executor.switch_provider(new_provider)
    # Give fire-and-forget handlers a moment to run
    await asyncio.sleep(0.05)

    # --- 4. Verify state was preserved ---
    print("\n=== After switch ===")
    print(f"Provider        : {type(executor.provider).__name__}")
    exported = new_provider.export_state()
    print(f"Provider state  : {exported.get('state', {})}")
    sm_progress = await state_manager.get("progress", scope=StateScope.WORKFLOW)
    print(f"StateManager    : progress={sm_progress}")

    if switch_events:
        evt = switch_events[0]
        print(f"\nSwitch event    : {evt.data}")

    print("\nProvider switch completed — state preserved across providers.")


if __name__ == "__main__":
    asyncio.run(main())
