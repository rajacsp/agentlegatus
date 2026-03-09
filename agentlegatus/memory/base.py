"""Memory backend abstract base class."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional


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
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store data in memory."""
        pass
    
    @abstractmethod
    async def retrieve(
        self,
        query: str,
        memory_type: MemoryType,
        limit: int = 10
    ) -> List[Any]:
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
