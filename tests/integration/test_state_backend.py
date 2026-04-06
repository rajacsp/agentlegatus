"""Integration test: state backend consistency.

Runs the same state operations against every available backend and
verifies identical behaviour. Redis and Postgres tests are skipped
when the corresponding services are not reachable.
"""

import pytest

from agentlegatus.core.event_bus import EventBus
from agentlegatus.core.state import InMemoryStateBackend, StateManager, StateScope


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _run_state_consistency_suite(backend):
    """Exercise core StateManager operations against the given backend."""
    sm = StateManager(backend=backend, event_bus=EventBus())
    scope = StateScope.WORKFLOW

    # set / get
    await sm.set("key1", "value1", scope=scope)
    assert await sm.get("key1", scope=scope) == "value1"

    # get with default
    assert await sm.get("missing", scope=scope, default="fallback") == "fallback"

    # update
    await sm.set("counter", 10, scope=scope)
    result = await sm.update("counter", lambda v: (v or 0) + 5, scope=scope)
    assert result == 15
    assert await sm.get("counter", scope=scope) == 15

    # delete
    deleted = await sm.delete("key1", scope=scope)
    assert deleted is True
    assert await sm.get("key1", scope=scope) is None

    # delete non-existent
    deleted = await sm.delete("nope", scope=scope)
    assert deleted is False

    # get_all
    await sm.set("a", 1, scope=scope)
    await sm.set("b", 2, scope=scope)
    all_state = await sm.get_all(scope=scope)
    assert all_state["a"] == 1
    assert all_state["b"] == 2

    # clear_scope
    await sm.clear_scope(scope)
    all_state = await sm.get_all(scope=scope)
    assert len(all_state) == 0

    # snapshot / restore
    await sm.set("x", 100, scope=scope)
    await sm.set("y", 200, scope=scope)
    await sm.create_snapshot("snap1")
    await sm.set("x", 999, scope=scope)
    assert await sm.get("x", scope=scope) == 999
    await sm.restore_snapshot("snap1")
    assert await sm.get("x", scope=scope) == 100
    assert await sm.get("y", scope=scope) == 200


# ===================================================================
# In-memory backend (always available)
# ===================================================================


class TestInMemoryStateBackend:
    @pytest.mark.asyncio
    async def test_consistency(self):
        backend = InMemoryStateBackend()
        await _run_state_consistency_suite(backend)

    @pytest.mark.asyncio
    async def test_scope_isolation(self):
        """Values in different scopes should not interfere."""
        backend = InMemoryStateBackend()
        sm = StateManager(backend=backend, event_bus=EventBus())

        await sm.set("shared_key", "workflow_val", scope=StateScope.WORKFLOW)
        await sm.set("shared_key", "step_val", scope=StateScope.STEP)
        await sm.set("shared_key", "agent_val", scope=StateScope.AGENT)

        assert await sm.get("shared_key", scope=StateScope.WORKFLOW) == "workflow_val"
        assert await sm.get("shared_key", scope=StateScope.STEP) == "step_val"
        assert await sm.get("shared_key", scope=StateScope.AGENT) == "agent_val"

    @pytest.mark.asyncio
    async def test_complex_values(self):
        """Backend should handle dicts, lists, and nested structures."""
        backend = InMemoryStateBackend()
        sm = StateManager(backend=backend, event_bus=EventBus())

        complex_val = {"nested": {"list": [1, 2, 3], "flag": True}}
        await sm.set("complex", complex_val, scope=StateScope.WORKFLOW)
        retrieved = await sm.get("complex", scope=StateScope.WORKFLOW)
        assert retrieved == complex_val


# ===================================================================
# Redis backend (skipped when Redis is not available)
# ===================================================================

def _redis_available() -> bool:
    try:
        import redis
        r = redis.Redis()
        r.ping()
        r.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _redis_available(), reason="Redis not reachable")
class TestRedisStateBackend:
    @pytest.mark.asyncio
    async def test_consistency(self):
        from agentlegatus.core.redis_backend import RedisStateBackend

        backend = RedisStateBackend(key_prefix="agentlegatus:test_integ")
        try:
            await _run_state_consistency_suite(backend)
        finally:
            await backend.close()


# ===================================================================
# Postgres backend (skipped when Postgres is not available)
# ===================================================================

def _postgres_available() -> bool:
    try:
        import asyncpg
        import asyncio

        async def _check():
            conn = await asyncpg.connect("postgresql://localhost:5432/agentlegatus")
            await conn.close()

        asyncio.get_event_loop().run_until_complete(_check())
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _postgres_available(), reason="Postgres not reachable")
class TestPostgresStateBackend:
    @pytest.mark.asyncio
    async def test_consistency(self):
        from agentlegatus.core.postgres_backend import PostgresStateBackend

        backend = PostgresStateBackend()
        try:
            await backend.initialize()
            await _run_state_consistency_suite(backend)
        finally:
            await backend.close()
