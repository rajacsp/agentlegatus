"""Memory backend abstract base class and in-memory implementation."""

import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class MemoryType(Enum):
    """Types of memory storage."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class MemoryBackend(ABC):
    """Abstract base class for memory backends."""

    @abstractmethod
    async def store(
        self,
        key: str,
        value: Any,
        memory_type: MemoryType,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store data in memory."""
        pass

    @abstractmethod
    async def retrieve(self, query: str, memory_type: MemoryType, limit: int = 10) -> list[Any]:
        """Retrieve data from memory."""
        pass

    @abstractmethod
    async def delete(self, key: str, memory_type: MemoryType) -> bool:
        """Delete data from memory."""
        pass

    @abstractmethod
    async def clear(self, memory_type: MemoryType) -> None:
        """Clear all data of a memory type."""
        pass


class InMemoryMemoryBackend(MemoryBackend):
    """Dictionary-based in-memory backend for development and testing.

    Each memory type is stored in its own isolated dict, keyed by
    ``MemoryType``.  Entries carry a timestamp so ``retrieve`` can
    return the most recent items and ``store_short_term`` TTL
    expiration can be honoured.
    """

    def __init__(self) -> None:
        # {MemoryType: {key: {"value": ..., "metadata": ..., "timestamp": float}}}
        self._storage: dict[MemoryType, dict[str, dict[str, Any]]] = {mt: {} for mt in MemoryType}

    def _get_type_store(self, memory_type: MemoryType) -> dict[str, dict[str, Any]]:
        return self._storage[memory_type]

    def _is_expired(self, entry: dict[str, Any]) -> bool:
        """Check if an entry has exceeded its TTL."""
        meta = entry.get("metadata") or {}
        ttl = meta.get("ttl")
        if ttl is None:
            return False
        return (time.monotonic() - entry["timestamp"]) > ttl

    async def store(
        self,
        key: str,
        value: Any,
        memory_type: MemoryType,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        store = self._get_type_store(memory_type)
        store[key] = {
            "value": value,
            "metadata": metadata,
            "timestamp": time.monotonic(),
        }

    async def retrieve(
        self,
        query: str,
        memory_type: MemoryType,
        limit: int = 10,
    ) -> list[Any]:
        store = self._get_type_store(memory_type)
        # Filter out expired entries, then sort by most-recent first
        live = [(k, e) for k, e in store.items() if not self._is_expired(e)]
        live.sort(key=lambda pair: pair[1]["timestamp"], reverse=True)

        # If a non-empty query is given, do a simple substring match on key
        if query:
            live = [(k, e) for k, e in live if query in k]

        return [e["value"] for _, e in live[:limit]]

    async def delete(self, key: str, memory_type: MemoryType) -> bool:
        store = self._get_type_store(memory_type)
        if key in store:
            del store[key]
            return True
        return False

    async def clear(self, memory_type: MemoryType) -> None:
        self._storage[memory_type] = {}
