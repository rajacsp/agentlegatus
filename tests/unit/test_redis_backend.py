"""Unit tests for RedisStateBackend using mocked redis client."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentlegatus.core.state import StateScope


@pytest.fixture
def mock_redis_client():
    """Create a mock async Redis client."""
    client = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock()
    client.delete = AsyncMock(return_value=1)
    client.sadd = AsyncMock()
    client.smembers = AsyncMock(return_value=set())
    return client


@pytest.fixture
def redis_backend(mock_redis_client):
    """Create a RedisStateBackend with mocked internals."""
    with patch.dict("sys.modules", {"redis": MagicMock(), "redis.asyncio": MagicMock()}):
        from agentlegatus.core.redis_backend import RedisStateBackend

        backend = RedisStateBackend.__new__(RedisStateBackend)
        backend._url = "redis://localhost:6379/0"
        backend._key_prefix = "test:state"
        backend._pool_size = 10
        backend._client = mock_redis_client

        from agentlegatus.core.workflow import RetryPolicy
        backend._retry_policy = RetryPolicy(max_attempts=1, initial_delay=0.01)
        return backend


class TestRedisStateBackendGet:
    async def test_get_returns_none_when_missing(self, redis_backend, mock_redis_client):
        mock_redis_client.get.return_value = None
        result = await redis_backend.get("key1", StateScope.WORKFLOW, "wf1")
        assert result is None

    async def test_get_returns_deserialized_value(self, redis_backend, mock_redis_client):
        mock_redis_client.get.return_value = json.dumps({"hello": "world"})
        result = await redis_backend.get("key1", StateScope.WORKFLOW, "wf1")
        assert result == {"hello": "world"}


class TestRedisStateBackendSet:
    async def test_set_serializes_and_stores(self, redis_backend, mock_redis_client):
        await redis_backend.set("key1", {"a": 1}, StateScope.WORKFLOW, "wf1")
        mock_redis_client.set.assert_called_once()
        call_args = mock_redis_client.set.call_args
        assert json.loads(call_args[0][1]) == {"a": 1}

    async def test_set_with_ttl(self, redis_backend, mock_redis_client):
        await redis_backend.set("key1", "val", StateScope.STEP, "s1", ttl=60)
        call_kwargs = mock_redis_client.set.call_args
        assert call_kwargs.kwargs.get("ex") == 60 or call_kwargs[1].get("ex") == 60


class TestRedisStateBackendDelete:
    async def test_delete_returns_true_when_existed(self, redis_backend, mock_redis_client):
        mock_redis_client.delete.return_value = 1
        result = await redis_backend.delete("key1", StateScope.WORKFLOW, "wf1")
        assert result is True

    async def test_delete_returns_false_when_missing(self, redis_backend, mock_redis_client):
        mock_redis_client.delete.return_value = 0
        result = await redis_backend.delete("key1", StateScope.WORKFLOW, "wf1")
        assert result is False


class TestRedisStateBackendGetAll:
    async def test_get_all_returns_all_keys(self, redis_backend, mock_redis_client):
        prefix = "test:state:workflow:wf1:"

        async def fake_scan_iter(match=None):
            for k in [f"{prefix}a", f"{prefix}b"]:
                yield k

        mock_redis_client.scan_iter = fake_scan_iter
        mock_redis_client.get.side_effect = [json.dumps(1), json.dumps(2)]

        result = await redis_backend.get_all(StateScope.WORKFLOW, "wf1")
        assert result == {"a": 1, "b": 2}


class TestRedisStateBackendClearScope:
    async def test_clear_scope_deletes_matching_keys(self, redis_backend, mock_redis_client):
        prefix = "test:state:workflow:wf1:"

        async def fake_scan_iter(match=None):
            for k in [f"{prefix}a", f"{prefix}b"]:
                yield k

        mock_redis_client.scan_iter = fake_scan_iter
        await redis_backend.clear_scope(StateScope.WORKFLOW, "wf1")
        mock_redis_client.delete.assert_called_once()


class TestRedisStateBackendSnapshots:
    async def test_list_snapshots_empty(self, redis_backend, mock_redis_client):
        mock_redis_client.smembers.return_value = set()
        result = await redis_backend.list_snapshots()
        assert result == []

    async def test_list_snapshots_returns_sorted(self, redis_backend, mock_redis_client):
        mock_redis_client.smembers.return_value = {"snap2", "snap1"}
        result = await redis_backend.list_snapshots()
        assert result == ["snap1", "snap2"]
