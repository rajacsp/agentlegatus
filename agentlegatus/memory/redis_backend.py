"""Redis memory backend implementation with TTL and memory type isolation."""

import json
import logging
import time
from typing import Any

from agentlegatus.memory.base import MemoryBackend, MemoryType

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None  # type: ignore[assignment]


class RedisMemoryBackend(MemoryBackend):
    """Redis-backed memory storage with TTL support and memory type isolation.

    Each memory type is isolated using key prefixes of the form
    ``{key_prefix}:{memory_type.value}:{key}``.  Short-term entries
    honour the TTL stored in metadata so they expire automatically in
    Redis.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        key_prefix: str = "agentlegatus:memory",
        pool_size: int = 10,
    ) -> None:
        if aioredis is None:
            raise ImportError(
                "redis package is required for RedisMemoryBackend. "
                "Install with: pip install 'agentlegatus[redis]'"
            )
        self._url = url
        self._key_prefix = key_prefix
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

    # ---- key helpers ----

    def _make_key(self, memory_type: MemoryType, key: str) -> str:
        """Build a namespaced Redis key: prefix:type:key."""
        return f"{self._key_prefix}:{memory_type.value}:{key}"

    def _type_pattern(self, memory_type: MemoryType) -> str:
        """Glob pattern matching all keys for a memory type."""
        return f"{self._key_prefix}:{memory_type.value}:*"

    def _strip_prefix(self, memory_type: MemoryType, rkey: str) -> str:
        """Remove the prefix portion to recover the original key."""
        prefix = f"{self._key_prefix}:{memory_type.value}:"
        return rkey.removeprefix(prefix) if isinstance(rkey, str) else rkey

    # ---- serialisation ----

    @staticmethod
    def _serialize(entry: dict[str, Any]) -> str:
        return json.dumps(entry)

    @staticmethod
    def _deserialize(raw: str | None) -> dict[str, Any] | None:
        if raw is None:
            return None
        return json.loads(raw)

    # ---- MemoryBackend interface ----

    async def store(
        self,
        key: str,
        value: Any,
        memory_type: MemoryType,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        client = await self._get_client()
        rkey = self._make_key(memory_type, key)
        entry = {
            "value": value,
            "metadata": metadata,
            "timestamp": time.time(),
        }
        ttl = (metadata or {}).get("ttl")
        if ttl is not None and isinstance(ttl, (int, float)) and ttl > 0:
            await client.set(rkey, self._serialize(entry), ex=int(ttl))
        else:
            await client.set(rkey, self._serialize(entry))

    async def retrieve(
        self,
        query: str,
        memory_type: MemoryType,
        limit: int = 10,
    ) -> list[Any]:
        client = await self._get_client()
        pattern = self._type_pattern(memory_type)
        entries: list[dict[str, Any]] = []

        async for rkey in client.scan_iter(match=pattern):
            raw = await client.get(rkey)
            entry = self._deserialize(raw)
            if entry is None:
                continue
            short_key = self._strip_prefix(memory_type, rkey)
            # Simple substring match on key when query is non-empty
            if query and query not in short_key:
                continue
            entries.append(entry)

        # Sort by most-recent first
        entries.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
        return [e["value"] for e in entries[:limit]]

    async def delete(self, key: str, memory_type: MemoryType) -> bool:
        client = await self._get_client()
        removed = await client.delete(self._make_key(memory_type, key))
        return removed > 0

    async def clear(self, memory_type: MemoryType) -> None:
        client = await self._get_client()
        pattern = self._type_pattern(memory_type)
        keys = [k async for k in client.scan_iter(match=pattern)]
        if keys:
            await client.delete(*keys)
