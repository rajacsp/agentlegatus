"""Memory manager for unified memory operations."""

from typing import Any, Dict, List, Optional

from agentlegatus.memory.base import MemoryBackend, MemoryType


class MemoryManager:
    """Unified memory management."""
    
    def __init__(self, backend: MemoryBackend):
        """Initialize memory manager with backend.
        
        Args:
            backend: Memory backend implementation
        """
        self.backend = backend
    
    async def store_short_term(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = 3600
    ) -> None:
        """Store short-term memory with TTL.
        
        Args:
            key: Memory key
            value: Value to store
            ttl: Time-to-live in seconds
        """
        metadata = {"ttl": ttl} if ttl else None
        await self.backend.store(key, value, MemoryType.SHORT_TERM, metadata)
    
    async def store_long_term(
        self,
        key: str,
        value: Any,
        embedding: Optional[List[float]] = None
    ) -> None:
        """Store long-term memory with optional embedding.
        
        Args:
            key: Memory key
            value: Value to store
            embedding: Optional embedding vector for semantic search
        """
        metadata = {"embedding": embedding} if embedding else None
        await self.backend.store(key, value, MemoryType.LONG_TERM, metadata)
    
    async def semantic_search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Perform semantic search in long-term memory.
        
        Args:
            query: Search query
            limit: Maximum number of results
            threshold: Similarity threshold
            
        Returns:
            List of matching memory items
        """
        results = await self.backend.retrieve(query, MemoryType.LONG_TERM, limit)
        return results
    
    async def get_recent(
        self,
        memory_type: MemoryType,
        limit: int = 10
    ) -> List[Any]:
        """Get most recent memories.
        
        Args:
            memory_type: Type of memory to retrieve
            limit: Maximum number of results
            
        Returns:
            List of recent memory items
        """
        return await self.backend.retrieve("", memory_type, limit)
