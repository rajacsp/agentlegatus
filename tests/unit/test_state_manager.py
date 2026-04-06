"""Unit tests for StateManager.

Covers:
- get() – returns value for key/scope, returns default for missing key, returns None default
- set() – stores value with key and scope, overwrites existing value
- update() – applies updater function atomically, passes None for non-existent key
- delete() – removes value and returns True, returns False for non-existent key
- get_all() – returns all key-value pairs for a scope
- clear_scope() – removes all state in a scope
- Scope isolation – operations in one scope don't affect other scopes
- create_snapshot() / restore_snapshot() – saves and restores state correctly
- StateUpdated event emission on modifications (if EventBus is provided)
- Atomic updates – concurrent updates don't corrupt state
"""

import asyncio
from datetime import datetime

import pytest

from agentlegatus.core.event_bus import Event, EventBus, EventType
from agentlegatus.core.state import (
    InMemoryStateBackend,
    StateManager,
    StateScope,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def backend():
    return InMemoryStateBackend()


@pytest.fixture
def manager(backend):
    return StateManager(backend=backend, default_scope_id="test-scope")


@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def manager_with_bus(backend, event_bus):
    return StateManager(backend=backend, default_scope_id="test-scope", event_bus=event_bus)



# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


class TestGet:
    @pytest.mark.asyncio
    async def test_get_returns_value_after_set(self, manager):
        await manager.set("key1", "value1")
        result = await manager.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_returns_default_for_missing_key(self, manager):
        result = await manager.get("missing", default="fallback")
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing_key_no_default(self, manager):
        result = await manager.get("missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_with_explicit_scope_and_scope_id(self, manager):
        await manager.set("k", 42, scope=StateScope.AGENT, scope_id="agent-1")
        result = await manager.get("k", scope=StateScope.AGENT, scope_id="agent-1")
        assert result == 42

    @pytest.mark.asyncio
    async def test_get_complex_value(self, manager):
        data = {"nested": [1, 2, 3], "flag": True}
        await manager.set("complex", data)
        result = await manager.get("complex")
        assert result == data


# ---------------------------------------------------------------------------
# set()
# ---------------------------------------------------------------------------


class TestSet:
    @pytest.mark.asyncio
    async def test_set_stores_value(self, manager):
        await manager.set("x", 100)
        assert await manager.get("x") == 100

    @pytest.mark.asyncio
    async def test_set_overwrites_existing_value(self, manager):
        await manager.set("x", "old")
        await manager.set("x", "new")
        assert await manager.get("x") == "new"

    @pytest.mark.asyncio
    async def test_set_different_scopes(self, manager):
        await manager.set("k", "workflow_val", scope=StateScope.WORKFLOW)
        await manager.set("k", "step_val", scope=StateScope.STEP)
        assert await manager.get("k", scope=StateScope.WORKFLOW) == "workflow_val"
        assert await manager.get("k", scope=StateScope.STEP) == "step_val"

    @pytest.mark.asyncio
    async def test_set_none_value(self, manager):
        """Setting None is stored; get returns default because backend returns None."""
        await manager.set("k", None)
        # The backend stores None, and get() treats None from backend as 'not found'
        result = await manager.get("k", default="fallback")
        assert result == "fallback"


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------


class TestUpdate:
    @pytest.mark.asyncio
    async def test_update_applies_function(self, manager):
        await manager.set("counter", 10)
        result = await manager.update("counter", lambda v: v + 5)
        assert result == 15
        assert await manager.get("counter") == 15

    @pytest.mark.asyncio
    async def test_update_passes_none_for_nonexistent_key(self, manager):
        result = await manager.update("new_key", lambda v: "created" if v is None else v)
        assert result == "created"
        assert await manager.get("new_key") == "created"

    @pytest.mark.asyncio
    async def test_update_with_scope(self, manager):
        await manager.set("val", [1], scope=StateScope.STEP, scope_id="s1")
        result = await manager.update(
            "val", lambda v: v + [2], scope=StateScope.STEP, scope_id="s1"
        )
        assert result == [1, 2]

    @pytest.mark.asyncio
    async def test_update_returns_new_value(self, manager):
        await manager.set("k", "hello")
        result = await manager.update("k", lambda v: v.upper())
        assert result == "HELLO"


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_existing_returns_true(self, manager):
        await manager.set("k", "v")
        assert await manager.delete("k") is True

    @pytest.mark.asyncio
    async def test_delete_removes_value(self, manager):
        await manager.set("k", "v")
        await manager.delete("k")
        assert await manager.get("k") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, manager):
        assert await manager.delete("nope") is False

    @pytest.mark.asyncio
    async def test_delete_with_scope(self, manager):
        await manager.set("k", 1, scope=StateScope.AGENT, scope_id="a1")
        assert await manager.delete("k", scope=StateScope.AGENT, scope_id="a1") is True
        assert await manager.get("k", scope=StateScope.AGENT, scope_id="a1") is None


# ---------------------------------------------------------------------------
# get_all()
# ---------------------------------------------------------------------------


class TestGetAll:
    @pytest.mark.asyncio
    async def test_get_all_returns_all_pairs(self, manager):
        await manager.set("a", 1)
        await manager.set("b", 2)
        await manager.set("c", 3)
        result = await manager.get_all()
        assert result == {"a": 1, "b": 2, "c": 3}

    @pytest.mark.asyncio
    async def test_get_all_empty_scope(self, manager):
        result = await manager.get_all(scope=StateScope.AGENT, scope_id="empty")
        assert result == {}

    @pytest.mark.asyncio
    async def test_get_all_returns_copy(self, manager):
        await manager.set("k", "v")
        result = await manager.get_all()
        result["k"] = "modified"
        assert await manager.get("k") == "v"


# ---------------------------------------------------------------------------
# clear_scope()
# ---------------------------------------------------------------------------


class TestClearScope:
    @pytest.mark.asyncio
    async def test_clear_scope_removes_all_state(self, manager):
        await manager.set("a", 1)
        await manager.set("b", 2)
        await manager.clear_scope(StateScope.WORKFLOW)
        result = await manager.get_all()
        assert result == {}

    @pytest.mark.asyncio
    async def test_clear_scope_does_not_affect_other_scopes(self, manager):
        await manager.set("k", "workflow", scope=StateScope.WORKFLOW)
        await manager.set("k", "step", scope=StateScope.STEP)
        await manager.clear_scope(StateScope.WORKFLOW)
        assert await manager.get("k", scope=StateScope.STEP) == "step"

    @pytest.mark.asyncio
    async def test_clear_scope_on_empty_scope_no_error(self, manager):
        await manager.clear_scope(StateScope.GLOBAL, scope_id="nonexistent")


# ---------------------------------------------------------------------------
# Scope isolation
# ---------------------------------------------------------------------------


class TestScopeIsolation:
    @pytest.mark.asyncio
    async def test_different_scopes_are_isolated(self, manager):
        """Same key in different scopes holds independent values."""
        await manager.set("key", "workflow", scope=StateScope.WORKFLOW)
        await manager.set("key", "step", scope=StateScope.STEP)
        await manager.set("key", "agent", scope=StateScope.AGENT)
        await manager.set("key", "global", scope=StateScope.GLOBAL)

        assert await manager.get("key", scope=StateScope.WORKFLOW) == "workflow"
        assert await manager.get("key", scope=StateScope.STEP) == "step"
        assert await manager.get("key", scope=StateScope.AGENT) == "agent"
        assert await manager.get("key", scope=StateScope.GLOBAL) == "global"

    @pytest.mark.asyncio
    async def test_delete_in_one_scope_does_not_affect_others(self, manager):
        await manager.set("key", "w", scope=StateScope.WORKFLOW)
        await manager.set("key", "s", scope=StateScope.STEP)
        await manager.delete("key", scope=StateScope.WORKFLOW)
        assert await manager.get("key", scope=StateScope.WORKFLOW) is None
        assert await manager.get("key", scope=StateScope.STEP) == "s"

    @pytest.mark.asyncio
    async def test_clear_scope_does_not_affect_other_scopes(self, manager):
        await manager.set("a", 1, scope=StateScope.WORKFLOW)
        await manager.set("b", 2, scope=StateScope.AGENT)
        await manager.clear_scope(StateScope.WORKFLOW)
        assert await manager.get_all(scope=StateScope.WORKFLOW) == {}
        assert await manager.get("b", scope=StateScope.AGENT) == 2

    @pytest.mark.asyncio
    async def test_different_scope_ids_are_isolated(self, manager):
        """Same scope type but different scope_ids are independent."""
        await manager.set("k", "id1", scope=StateScope.STEP, scope_id="step-1")
        await manager.set("k", "id2", scope=StateScope.STEP, scope_id="step-2")
        assert await manager.get("k", scope=StateScope.STEP, scope_id="step-1") == "id1"
        assert await manager.get("k", scope=StateScope.STEP, scope_id="step-2") == "id2"


# ---------------------------------------------------------------------------
# create_snapshot() / restore_snapshot()
# ---------------------------------------------------------------------------


class TestSnapshotRestore:
    @pytest.mark.asyncio
    async def test_snapshot_and_restore_preserves_state(self, manager):
        await manager.set("a", 1)
        await manager.set("b", 2)
        await manager.create_snapshot("snap1")

        # Modify state after snapshot
        await manager.set("a", 999)
        await manager.delete("b")

        # Restore
        await manager.restore_snapshot("snap1")
        assert await manager.get("a") == 1
        assert await manager.get("b") == 2

    @pytest.mark.asyncio
    async def test_restore_nonexistent_snapshot_raises(self, manager):
        with pytest.raises(ValueError, match="not found"):
            await manager.restore_snapshot("nonexistent")

    @pytest.mark.asyncio
    async def test_snapshot_is_independent_of_later_changes(self, manager):
        await manager.set("k", "original")
        await manager.create_snapshot("snap")
        await manager.set("k", "changed")

        await manager.restore_snapshot("snap")
        assert await manager.get("k") == "original"

    @pytest.mark.asyncio
    async def test_multiple_snapshots(self, manager):
        await manager.set("k", "v1")
        await manager.create_snapshot("s1")

        await manager.set("k", "v2")
        await manager.create_snapshot("s2")

        await manager.restore_snapshot("s1")
        assert await manager.get("k") == "v1"

        await manager.restore_snapshot("s2")
        assert await manager.get("k") == "v2"

    @pytest.mark.asyncio
    async def test_list_snapshots(self, manager):
        await manager.create_snapshot("snap-a")
        await manager.create_snapshot("snap-b")
        snapshots = await manager.list_snapshots()
        assert "snap-a" in snapshots
        assert "snap-b" in snapshots

    @pytest.mark.asyncio
    async def test_snapshot_with_explicit_scope(self, manager):
        await manager.set("k", "agent_val", scope=StateScope.AGENT, scope_id="a1")
        await manager.create_snapshot("snap", scope=StateScope.AGENT, scope_id="a1")

        await manager.set("k", "modified", scope=StateScope.AGENT, scope_id="a1")
        await manager.restore_snapshot("snap", scope=StateScope.AGENT, scope_id="a1")
        assert await manager.get("k", scope=StateScope.AGENT, scope_id="a1") == "agent_val"


# ---------------------------------------------------------------------------
# StateUpdated event emission
# ---------------------------------------------------------------------------


class TestEventEmission:
    @pytest.mark.asyncio
    async def test_set_emits_state_updated(self, manager_with_bus, event_bus):
        events_received = []

        async def handler(event: Event):
            events_received.append(event)

        event_bus.subscribe(EventType.STATE_UPDATED, handler)
        await manager_with_bus.set("k", "v")
        await asyncio.sleep(0.05)

        assert len(events_received) == 1
        assert events_received[0].event_type == EventType.STATE_UPDATED
        assert events_received[0].data["operation"] == "set"
        assert events_received[0].data["key"] == "k"

    @pytest.mark.asyncio
    async def test_update_emits_state_updated(self, manager_with_bus, event_bus):
        events_received = []

        async def handler(event: Event):
            events_received.append(event)

        event_bus.subscribe(EventType.STATE_UPDATED, handler)
        await manager_with_bus.set("k", 1)
        await manager_with_bus.update("k", lambda v: v + 1)
        await asyncio.sleep(0.05)

        # set emits once, update emits once (set inside update + update itself)
        update_events = [e for e in events_received if e.data.get("operation") == "update"]
        assert len(update_events) == 1

    @pytest.mark.asyncio
    async def test_delete_emits_state_updated_when_key_exists(self, manager_with_bus, event_bus):
        events_received = []

        async def handler(event: Event):
            events_received.append(event)

        event_bus.subscribe(EventType.STATE_UPDATED, handler)
        await manager_with_bus.set("k", "v")
        await manager_with_bus.delete("k")
        await asyncio.sleep(0.05)

        delete_events = [e for e in events_received if e.data.get("operation") == "delete"]
        assert len(delete_events) == 1

    @pytest.mark.asyncio
    async def test_delete_does_not_emit_when_key_missing(self, manager_with_bus, event_bus):
        events_received = []

        async def handler(event: Event):
            events_received.append(event)

        event_bus.subscribe(EventType.STATE_UPDATED, handler)
        await manager_with_bus.delete("nonexistent")
        await asyncio.sleep(0.05)

        delete_events = [e for e in events_received if e.data.get("operation") == "delete"]
        assert len(delete_events) == 0

    @pytest.mark.asyncio
    async def test_clear_scope_emits_state_updated(self, manager_with_bus, event_bus):
        events_received = []

        async def handler(event: Event):
            events_received.append(event)

        event_bus.subscribe(EventType.STATE_UPDATED, handler)
        await manager_with_bus.clear_scope(StateScope.WORKFLOW)
        await asyncio.sleep(0.05)

        clear_events = [e for e in events_received if e.data.get("operation") == "clear_scope"]
        assert len(clear_events) == 1

    @pytest.mark.asyncio
    async def test_restore_snapshot_emits_state_updated(self, manager_with_bus, event_bus):
        events_received = []

        async def handler(event: Event):
            events_received.append(event)

        await manager_with_bus.set("k", "v")
        await manager_with_bus.create_snapshot("snap")

        event_bus.subscribe(EventType.STATE_UPDATED, handler)
        await manager_with_bus.restore_snapshot("snap")
        await asyncio.sleep(0.05)

        restore_events = [
            e for e in events_received if e.data.get("operation") == "restore_snapshot"
        ]
        assert len(restore_events) == 1
        assert restore_events[0].data["snapshot_id"] == "snap"

    @pytest.mark.asyncio
    async def test_no_events_without_event_bus(self, manager):
        """Operations succeed without event_bus (no errors)."""
        await manager.set("k", "v")
        await manager.update("k", lambda v: v)
        await manager.delete("k")
        await manager.clear_scope(StateScope.WORKFLOW)

    @pytest.mark.asyncio
    async def test_event_contains_scope_info(self, manager_with_bus, event_bus):
        events_received = []

        async def handler(event: Event):
            events_received.append(event)

        event_bus.subscribe(EventType.STATE_UPDATED, handler)
        await manager_with_bus.set("k", "v", scope=StateScope.AGENT, scope_id="agent-x")
        await asyncio.sleep(0.05)

        assert events_received[0].data["scope"] == "agent"
        assert events_received[0].data["scope_id"] == "agent-x"


# ---------------------------------------------------------------------------
# Atomic updates (concurrent)
# ---------------------------------------------------------------------------


class TestAtomicUpdates:
    @pytest.mark.asyncio
    async def test_concurrent_sets_do_not_lose_data(self, manager):
        """Multiple concurrent set() calls on different keys all persist."""
        keys = [f"key-{i}" for i in range(20)]

        async def set_key(k):
            await manager.set(k, k)

        await asyncio.gather(*[set_key(k) for k in keys])

        for k in keys:
            assert await manager.get(k) == k

    @pytest.mark.asyncio
    async def test_concurrent_updates_on_same_key(self, manager):
        """Concurrent update() calls on the same key all execute without error.

        Since InMemoryStateBackend is single-threaded via asyncio, updates
        are effectively serialized. We verify the final value is consistent
        with some number of increments having been applied.
        """
        await manager.set("counter", 0)

        async def increment():
            await manager.update("counter", lambda v: v + 1)

        await asyncio.gather(*[increment() for _ in range(50)])

        result = await manager.get("counter")
        assert result == 50

    @pytest.mark.asyncio
    async def test_concurrent_operations_across_scopes(self, manager):
        """Concurrent operations on different scopes don't interfere."""

        async def workflow_ops():
            await manager.set("k", "w", scope=StateScope.WORKFLOW)

        async def step_ops():
            await manager.set("k", "s", scope=StateScope.STEP)

        async def agent_ops():
            await manager.set("k", "a", scope=StateScope.AGENT)

        await asyncio.gather(workflow_ops(), step_ops(), agent_ops())

        assert await manager.get("k", scope=StateScope.WORKFLOW) == "w"
        assert await manager.get("k", scope=StateScope.STEP) == "s"
        assert await manager.get("k", scope=StateScope.AGENT) == "a"
