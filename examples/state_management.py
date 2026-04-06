#!/usr/bin/env python3
"""Example 6: State Management Across Scopes.

Demonstrates get/set/update/delete operations, scope isolation,
and snapshot/restore functionality using the StateManager.
"""

import asyncio

from agentlegatus.core.event_bus import EventBus
from agentlegatus.core.state import InMemoryStateBackend, StateManager, StateScope


async def main():
    event_bus = EventBus()
    backend = InMemoryStateBackend()
    state_manager = StateManager(
        backend=backend,
        default_scope_id="wf-001",
        event_bus=event_bus,
    )

    # --- 1. Basic get / set ---
    await state_manager.set("counter", 0, scope=StateScope.WORKFLOW)
    await state_manager.set("model", "gpt-4", scope=StateScope.AGENT, scope_id="agent-1")
    await state_manager.set("global_flag", True, scope=StateScope.GLOBAL)

    counter = await state_manager.get("counter", scope=StateScope.WORKFLOW)
    model = await state_manager.get("model", scope=StateScope.AGENT, scope_id="agent-1")
    flag = await state_manager.get("global_flag", scope=StateScope.GLOBAL)
    print(f"counter={counter}, model={model}, global_flag={flag}")

    # --- 2. Atomic update ---
    new_val = await state_manager.update(
        "counter",
        updater=lambda v: (v or 0) + 5,
        scope=StateScope.WORKFLOW,
    )
    print(f"counter after update: {new_val}")

    # --- 3. Scope isolation ---
    await state_manager.set("shared_key", "workflow-value", scope=StateScope.WORKFLOW)
    await state_manager.set("shared_key", "step-value", scope=StateScope.STEP, scope_id="step-1")

    wf_val = await state_manager.get("shared_key", scope=StateScope.WORKFLOW)
    step_val = await state_manager.get("shared_key", scope=StateScope.STEP, scope_id="step-1")
    print(f"\nScope isolation — WORKFLOW: {wf_val}, STEP: {step_val}")

    # --- 4. get_all for a scope ---
    all_wf = await state_manager.get_all(scope=StateScope.WORKFLOW)
    print(f"All WORKFLOW state: {all_wf}")

    # --- 5. Snapshot and restore ---
    await state_manager.create_snapshot("snap-before-experiment")
    print("\nSnapshot 'snap-before-experiment' created")

    # Mutate state
    await state_manager.set("counter", 999, scope=StateScope.WORKFLOW)
    await state_manager.set("extra", "temporary", scope=StateScope.WORKFLOW)
    mutated = await state_manager.get_all(scope=StateScope.WORKFLOW)
    print(f"After mutation: {mutated}")

    # Restore
    await state_manager.restore_snapshot("snap-before-experiment")
    restored = await state_manager.get_all(scope=StateScope.WORKFLOW)
    print(f"After restore : {restored}")

    # --- 6. Delete and clear ---
    deleted = await state_manager.delete("shared_key", scope=StateScope.STEP, scope_id="step-1")
    print(f"\nDeleted step shared_key: {deleted}")

    await state_manager.clear_scope(StateScope.GLOBAL)
    global_after = await state_manager.get_all(scope=StateScope.GLOBAL)
    print(f"GLOBAL scope after clear: {global_after}")

    # --- 7. List snapshots ---
    snapshots = await state_manager.list_snapshots()
    print(f"\nAvailable snapshots: {snapshots}")


if __name__ == "__main__":
    asyncio.run(main())
