"""Vector store memory backend using ChromaDB for semantic search."""

import logging
import time
from typing import Any

from agentlegatus.memory.base import MemoryBackend, MemoryType

logger = logging.getLogger(__name__)

try:
    import chromadb
except ImportError:
    chromadb = None  # type: ignore[assignment]


class VectorStoreMemoryBackend(MemoryBackend):
    """ChromaDB-backed memory storage with semantic search support.

    Each memory type maps to a separate ChromaDB collection, providing
    full isolation.  Entries that include an ``embedding`` in metadata
    are stored with that vector; otherwise ChromaDB's default embedding
    function is used (if configured) or the entry is stored as a plain
    document.

    Semantic search is performed via ChromaDB's ``query`` API which
    returns results ranked by cosine similarity.
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_prefix: str = "agentlegatus_memory",
        client: Any | None = None,
    ) -> None:
        """
        Args:
            persist_directory: Path for persistent ChromaDB storage.
                If ``None``, an ephemeral in-memory client is used.
            collection_prefix: Prefix for ChromaDB collection names.
            client: Optional pre-configured ChromaDB client instance.
        """
        if chromadb is None:
            raise ImportError(
                "chromadb package is required for VectorStoreMemoryBackend. "
                "Install with: pip install 'agentlegatus[vector]'"
            )
        self._collection_prefix = collection_prefix

        if client is not None:
            self._client = client
        elif persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.EphemeralClient()

        # Cache collection handles keyed by MemoryType
        self._collections: dict[MemoryType, Any] = {}

    def _get_collection(self, memory_type: MemoryType) -> Any:
        """Get or create the ChromaDB collection for a memory type."""
        if memory_type not in self._collections:
            name = f"{self._collection_prefix}_{memory_type.value}"
            self._collections[memory_type] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[memory_type]

    @staticmethod
    def _make_id(key: str) -> str:
        """ChromaDB requires string IDs; use the key directly."""
        return key

    # ---- serialisation helpers ----

    @staticmethod
    def _serialize_value(value: Any) -> str:
        """Encode value as a JSON string stored in the document field."""
        import json

        return json.dumps({"v": value})

    @staticmethod
    def _deserialize_value(doc: str) -> Any:
        import json

        return json.loads(doc)["v"]

    # ---- MemoryBackend interface ----

    async def store(
        self,
        key: str,
        value: Any,
        memory_type: MemoryType,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        collection = self._get_collection(memory_type)
        doc = self._serialize_value(value)
        entry_meta: dict[str, Any] = {"timestamp": time.time()}

        embedding = (metadata or {}).get("embedding")
        ttl = (metadata or {}).get("ttl")
        if ttl is not None:
            entry_meta["ttl"] = float(ttl)

        kwargs: dict[str, Any] = {
            "ids": [self._make_id(key)],
            "documents": [doc],
            "metadatas": [entry_meta],
        }
        if embedding is not None:
            kwargs["embeddings"] = [embedding]

        collection.upsert(**kwargs)

    async def retrieve(
        self,
        query: str,
        memory_type: MemoryType,
        limit: int = 10,
    ) -> list[Any]:
        collection = self._get_collection(memory_type)
        count = collection.count()
        if count == 0:
            return []

        effective_limit = min(limit, count)

        if query:
            # Semantic search via ChromaDB query
            results = collection.query(
                query_texts=[query],
                n_results=effective_limit,
            )
            docs = results.get("documents", [[]])[0]
        else:
            # No query — return most recent entries
            results = collection.get(limit=effective_limit)
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])
            # Sort by timestamp descending
            paired = list(zip(docs, metas, strict=False))
            paired.sort(key=lambda p: p[1].get("timestamp", 0), reverse=True)
            docs = [d for d, _ in paired]

        return [self._deserialize_value(d) for d in docs]

    async def delete(self, key: str, memory_type: MemoryType) -> bool:
        collection = self._get_collection(memory_type)
        cid = self._make_id(key)
        existing = collection.get(ids=[cid])
        if not existing["ids"]:
            return False
        collection.delete(ids=[cid])
        return True

    async def clear(self, memory_type: MemoryType) -> None:
        name = f"{self._collection_prefix}_{memory_type.value}"
        try:
            self._client.delete_collection(name=name)
        except Exception:
            pass
        # Remove cached handle so it gets re-created on next access
        self._collections.pop(memory_type, None)
