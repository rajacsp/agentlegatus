"""Redis state backend implementation with connection pooling and retry logic."""

import json
import logging
from typing import Any

from agentlegatus.core.state import StateBackend, StateScope
from agentlegatus.core.workflow import RetryPolicy
from agentlegatus.utils.retry import execute_with_retry

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None  # type: ignore[assignment]


class RedisStateBackend(StateBackend):
    """Redis-backed state storage with connection pooling and retry support."""

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        key_prefix: str = "agentlegatus:state",
        retry_policy: RetryPolicy | None = None,
        pool_size: int = 10,
    ):
        """
        Initialize Redis state backend.

        Args:
            url: Redis connection URL
            key_prefix: Prefix for all Redis keys
            retry_policy: Retry policy for connection failures
            pool_size: Max connections in the pool
        """
        if aioredis is None:
            raise ImportError(
                "redis package is required for RedisStateBackend. "
                "Install with: pip install 'agentlegatus[redis]'"
            )
        self._url = url
        self._key_prefix = key_prefix
        self._retry_policy = retry_policy or RetryPolicy(
            max_attempts=3, initial_delay=0.5, backoff_multiplier=2.0, max_delay=5.0
        )
        self._pool_size = pool_size
        self._client: aioredis.Redis | None = None  # type: ignore[name-defined]

    async def _get_client(self) -> "aioredis.Redis":  # type: ignore[name-defined]
        """Get or create the Redis client with connection pooling."""
        if self._client is None:
            pool = aioredis.ConnectionPool.from_url(
                self._url, max_connections=self._pool_size, decode_responses=True
            )
            self._client = aioredis.Redis(connection_pool=pool)
        return self._client

    async def close(self) -> None:
        """Close the Redis connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _make_key(self, scope: StateScope, scope_id: str, key: str) -> str:
        """Build a namespaced Redis key."""
        return f"{self._key_prefix}:{scope.value}:{scope_id}:{key}"

    def _scope_pattern(self, scope: StateScope, scope_id: str) -> str:
        """Build a glob pattern matching all keys in a scope."""
        return f"{self._key_prefix}:{scope.value}:{scope_id}:*"

    def _snapshot_key(self, snapshot_id: str, scope: StateScope, scope_id: str) -> str:
        """Build a Redis key for snapshot data."""
        return f"{self._key_prefix}:__snapshots__:{snapshot_id}:{scope.value}:{scope_id}"

    def _snapshot_registry_key(self) -> str:
        """Key for the set that tracks all snapshot IDs."""
        return f"{self._key_prefix}:__snapshot_ids__"

    # ---- serialisation helpers ----

    @staticmethod
    def _serialize(value: Any) -> str:
        return json.dumps(value)

    @staticmethod
    def _deserialize(raw: str | None) -> Any:
        if raw is None:
            return None
        return json.loads(raw)

    # ---- StateBackend interface ----

    async def get(self, key: str, scope: StateScope, scope_id: str) -> Any | None:
        async def _op() -> Any | None:
            client = await self._get_client()
            raw = await client.get(self._make_key(scope, scope_id, key))
            return self._deserialize(raw)

        return await execute_with_retry(
            _op, retry_policy=self._retry_policy, operation_name="redis_get"
        )

    async def set(
        self,
        key: str,
        value: Any,
        scope: StateScope,
        scope_id: str,
        ttl: int | None = None,
    ) -> None:
        async def _op() -> None:
            client = await self._get_client()
            rkey = self._make_key(scope, scope_id, key)
            await client.set(rkey, self._serialize(value), ex=ttl)

        await execute_with_retry(_op, retry_policy=self._retry_policy, operation_name="redis_set")

    async def delete(self, key: str, scope: StateScope, scope_id: str) -> bool:
        async def _op() -> bool:
            client = await self._get_client()
            removed = await client.delete(self._make_key(scope, scope_id, key))
            return removed > 0

        return await execute_with_retry(
            _op, retry_policy=self._retry_policy, operation_name="redis_delete"
        )

    async def get_all(self, scope: StateScope, scope_id: str) -> dict[str, Any]:
        async def _op() -> dict[str, Any]:
            client = await self._get_client()
            pattern = self._scope_pattern(scope, scope_id)
            prefix = f"{self._key_prefix}:{scope.value}:{scope_id}:"
            result: dict[str, Any] = {}
            async for rkey in client.scan_iter(match=pattern):
                short_key = (
                    rkey.removeprefix(prefix)
                    if isinstance(rkey, str)
                    else rkey.decode().removeprefix(prefix)
                )
                raw = await client.get(rkey)
                result[short_key] = self._deserialize(raw)
            return result

        return await execute_with_retry(
            _op, retry_policy=self._retry_policy, operation_name="redis_get_all"
        )

    async def clear_scope(self, scope: StateScope, scope_id: str) -> None:
        async def _op() -> None:
            client = await self._get_client()
            pattern = self._scope_pattern(scope, scope_id)
            keys = [k async for k in client.scan_iter(match=pattern)]
            if keys:
                await client.delete(*keys)

        await execute_with_retry(
            _op, retry_policy=self._retry_policy, operation_name="redis_clear_scope"
        )

    async def create_snapshot(self, snapshot_id: str, scope: StateScope, scope_id: str) -> None:
        async def _op() -> None:
            client = await self._get_client()
            data = await self.get_all(scope, scope_id)
            snap_key = self._snapshot_key(snapshot_id, scope, scope_id)
            await client.set(snap_key, self._serialize(data))
            await client.sadd(self._snapshot_registry_key(), snapshot_id)

        await execute_with_retry(
            _op, retry_policy=self._retry_policy, operation_name="redis_create_snapshot"
        )

    async def restore_snapshot(self, snapshot_id: str, scope: StateScope, scope_id: str) -> None:
        async def _op() -> None:
            client = await self._get_client()
            snap_key = self._snapshot_key(snapshot_id, scope, scope_id)
            raw = await client.get(snap_key)
            if raw is None:
                raise ValueError(f"Snapshot '{snapshot_id}' not found")
            data: dict[str, Any] = self._deserialize(raw)

            # Clear current scope then restore
            await self.clear_scope(scope, scope_id)
            for k, v in data.items():
                await client.set(self._make_key(scope, scope_id, k), self._serialize(v))

        await execute_with_retry(
            _op, retry_policy=self._retry_policy, operation_name="redis_restore_snapshot"
        )

    async def list_snapshots(self) -> list[str]:
        async def _op() -> list[str]:
            client = await self._get_client()
            members = await client.smembers(self._snapshot_registry_key())
            return sorted(m if isinstance(m, str) else m.decode() for m in members)

        return await execute_with_retry(
            _op, retry_policy=self._retry_policy, operation_name="redis_list_snapshots"
        )
