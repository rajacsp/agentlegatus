"""Memory manager for unified memory operations."""

from typing import Any

from agentlegatus.memory.base import MemoryBackend, MemoryType


class MemoryManager:
    """Unified memory management.

    Wraps a ``MemoryBackend`` and exposes convenience methods that map
    to the four supported memory types (SHORT_TERM, LONG_TERM,
    EPISODIC, SEMANTIC).  All operations are isolated by memory type
    as required by Requirement 30.
    """

    def __init__(self, backend: MemoryBackend) -> None:
        """Initialize memory manager with backend.

        Args:
            backend: Memory backend implementation.
        """
        self.backend = backend

    # ------------------------------------------------------------------
    # Store helpers
    # ------------------------------------------------------------------

    async def store_short_term(
        self,
        key: str,
        value: Any,
        ttl: int | None = 3600,
    ) -> None:
        """Store short-term memory with optional TTL.

        Args:
            key: Memory key.
            value: Value to store.
            ttl: Time-to-live in seconds (``None`` means no expiry).
        """
        metadata: dict[str, Any] | None = {"ttl": ttl} if ttl is not None else None
        await self.backend.store(key, value, MemoryType.SHORT_TERM, metadata)

    async def store_long_term(
        self,
        key: str,
        value: Any,
        embedding: list[float] | None = None,
    ) -> None:
        """Store long-term memory with optional embedding.

        Args:
            key: Memory key.
            value: Value to store.
            embedding: Optional embedding vector for semantic search.
        """
        metadata: dict[str, Any] | None = (
            {"embedding": embedding} if embedding is not None else None
        )
        await self.backend.store(key, value, MemoryType.LONG_TERM, metadata)

    async def store(
        self,
        key: str,
        value: Any,
        memory_type: MemoryType,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Generic store that delegates directly to the backend.

        Args:
            key: Memory key.
            value: Value to store.
            memory_type: Type of memory.
            metadata: Optional metadata dict.
        """
        await self.backend.store(key, value, memory_type, metadata)

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Perform semantic search in long-term memory.

        Args:
            query: Search query.
            limit: Maximum number of results.
            threshold: Similarity threshold (reserved for vector backends).

        Returns:
            List of matching memory items.
        """
        return await self.backend.retrieve(query, MemoryType.LONG_TERM, limit)

    async def get_recent(
        self,
        memory_type: MemoryType,
        limit: int = 10,
    ) -> list[Any]:
        """Get most recent memories of the given type.

        Args:
            memory_type: Type of memory to retrieve.
            limit: Maximum number of results.

        Returns:
            List of recent memory values, newest first.
        """
        return await self.backend.retrieve("", memory_type, limit)

    # ------------------------------------------------------------------
    # Delete / clear helpers
    # ------------------------------------------------------------------

    async def delete(self, key: str, memory_type: MemoryType) -> bool:
        """Delete a memory entry.

        Args:
            key: Memory key.
            memory_type: Type of memory.

        Returns:
            ``True`` if the entry existed and was removed.
        """
        return await self.backend.delete(key, memory_type)

    async def clear(self, memory_type: MemoryType) -> None:
        """Clear all entries of a given memory type.

        Args:
            memory_type: Type of memory to clear.
        """
        await self.backend.clear(memory_type)
