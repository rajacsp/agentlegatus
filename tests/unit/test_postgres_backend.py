"""Unit tests for PostgresStateBackend using mocked asyncpg pool."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentlegatus.core.state import StateScope


class FakeConnection:
    """Fake asyncpg connection supporting async context manager."""

    def __init__(self):
        self.fetchrow = AsyncMock(return_value=None)
        self.fetch = AsyncMock(return_value=[])
        self.execute = AsyncMock(return_value="DELETE 0")
        self._tx = AsyncMock()

    def transaction(self):
        return self._tx


class FakePool:
    """Fake asyncpg pool that yields a FakeConnection."""

    def __init__(self, conn: FakeConnection):
        self._conn = conn

    def acquire(self):
        return _FakeAcquire(self._conn)

    async def close(self):
        pass


class _FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def fake_conn():
    return FakeConnection()


@pytest.fixture
def postgres_backend(fake_conn):
    with patch.dict("sys.modules", {"asyncpg": MagicMock()}):
        from agentlegatus.core.postgres_backend import PostgresStateBackend

        backend = PostgresStateBackend.__new__(PostgresStateBackend)
        backend._dsn = "postgresql://localhost/test"
        backend._pool_min_size = 1
        backend._pool_max_size = 2
        backend._pool = FakePool(fake_conn)

        from agentlegatus.core.workflow import RetryPolicy
        backend._retry_policy = RetryPolicy(max_attempts=1, initial_delay=0.01)
        return backend


class TestPostgresGet:
    async def test_get_returns_none_when_missing(self, postgres_backend, fake_conn):
        fake_conn.fetchrow.return_value = None
        result = await postgres_backend.get("key1", StateScope.WORKFLOW, "wf1")
        assert result is None

    async def test_get_returns_deserialized_value(self, postgres_backend, fake_conn):
        fake_conn.fetchrow.return_value = {"value": json.dumps({"x": 42})}
        result = await postgres_backend.get("key1", StateScope.WORKFLOW, "wf1")
        assert result == {"x": 42}


class TestPostgresSet:
    async def test_set_calls_execute(self, postgres_backend, fake_conn):
        await postgres_backend.set("key1", {"a": 1}, StateScope.WORKFLOW, "wf1")
        fake_conn.execute.assert_called_once()
        args = fake_conn.execute.call_args[0]
        assert "INSERT INTO agentlegatus_state" in args[0]
        assert args[1] == "workflow"
        assert args[2] == "wf1"
        assert args[3] == "key1"


class TestPostgresDelete:
    async def test_delete_returns_true_when_existed(self, postgres_backend, fake_conn):
        fake_conn.execute.return_value = "DELETE 1"
        result = await postgres_backend.delete("key1", StateScope.WORKFLOW, "wf1")
        assert result is True

    async def test_delete_returns_false_when_missing(self, postgres_backend, fake_conn):
        fake_conn.execute.return_value = "DELETE 0"
        result = await postgres_backend.delete("key1", StateScope.WORKFLOW, "wf1")
        assert result is False


class TestPostgresGetAll:
    async def test_get_all_returns_dict(self, postgres_backend, fake_conn):
        fake_conn.fetch.return_value = [
            {"key": "a", "value": json.dumps(1)},
            {"key": "b", "value": json.dumps("hello")},
        ]
        result = await postgres_backend.get_all(StateScope.WORKFLOW, "wf1")
        assert result == {"a": 1, "b": "hello"}


class TestPostgresClearScope:
    async def test_clear_scope_calls_delete(self, postgres_backend, fake_conn):
        await postgres_backend.clear_scope(StateScope.WORKFLOW, "wf1")
        fake_conn.execute.assert_called_once()
        assert "DELETE FROM agentlegatus_state" in fake_conn.execute.call_args[0][0]


class TestPostgresSnapshots:
    async def test_list_snapshots_empty(self, postgres_backend, fake_conn):
        fake_conn.fetch.return_value = []
        result = await postgres_backend.list_snapshots()
        assert result == []

    async def test_list_snapshots_returns_ids(self, postgres_backend, fake_conn):
        fake_conn.fetch.return_value = [
            {"snapshot_id": "snap1"},
            {"snapshot_id": "snap2"},
        ]
        result = await postgres_backend.list_snapshots()
        assert result == ["snap1", "snap2"]

    async def test_restore_snapshot_raises_when_missing(self, postgres_backend, fake_conn):
        fake_conn.fetchrow.return_value = None
        with pytest.raises(ValueError, match="not found"):
            await postgres_backend.restore_snapshot("missing", StateScope.WORKFLOW, "wf1")
