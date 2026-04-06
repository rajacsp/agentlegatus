"""Postgres state backend implementation using asyncpg with connection pooling."""

import json
import logging
from typing import Any

from agentlegatus.core.state import StateBackend, StateScope
from agentlegatus.core.workflow import RetryPolicy
from agentlegatus.utils.retry import execute_with_retry

logger = logging.getLogger(__name__)

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore[assignment]


# SQL for table creation
_CREATE_STATE_TABLE = """
CREATE TABLE IF NOT EXISTS agentlegatus_state (
    scope      TEXT NOT NULL,
    scope_id   TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (scope, scope_id, key)
);
"""

_CREATE_SNAPSHOT_TABLE = """
CREATE TABLE IF NOT EXISTS agentlegatus_snapshots (
    snapshot_id TEXT NOT NULL,
    scope       TEXT NOT NULL,
    scope_id    TEXT NOT NULL,
    data        JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (snapshot_id, scope, scope_id)
);
"""


class PostgresStateBackend(StateBackend):
    """Postgres-backed state storage using asyncpg with connection pooling."""

    def __init__(
        self,
        dsn: str = "postgresql://localhost:5432/agentlegatus",
        pool_min_size: int = 2,
        pool_max_size: int = 10,
        retry_policy: RetryPolicy | None = None,
    ):
        """
        Initialize Postgres state backend.

        Args:
            dsn: Postgres connection DSN
            pool_min_size: Minimum connections in the pool
            pool_max_size: Maximum connections in the pool
            retry_policy: Retry policy for connection failures
        """
        if asyncpg is None:
            raise ImportError(
                "asyncpg package is required for PostgresStateBackend. "
                "Install with: pip install 'agentlegatus[postgres]'"
            )
        self._dsn = dsn
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._retry_policy = retry_policy or RetryPolicy(
            max_attempts=3, initial_delay=0.5, backoff_multiplier=2.0, max_delay=5.0
        )
        self._pool: asyncpg.Pool | None = None  # type: ignore[name-defined]

    async def _get_pool(self) -> "asyncpg.Pool":  # type: ignore[name-defined]
        """Get or create the connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=self._pool_min_size,
                max_size=self._pool_max_size,
            )
        return self._pool

    async def initialize(self) -> None:
        """Create required tables if they don't exist."""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(_CREATE_STATE_TABLE)
            await conn.execute(_CREATE_SNAPSHOT_TABLE)

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # ---- StateBackend interface ----

    async def get(self, key: str, scope: StateScope, scope_id: str) -> Any | None:
        async def _op() -> Any | None:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT value FROM agentlegatus_state "
                    "WHERE scope = $1 AND scope_id = $2 AND key = $3",
                    scope.value,
                    scope_id,
                    key,
                )
            if row is None:
                return None
            return json.loads(row["value"])

        return await execute_with_retry(
            _op, retry_policy=self._retry_policy, operation_name="pg_get"
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
            pool = await self._get_pool()
            serialized = json.dumps(value)
            async with pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO agentlegatus_state (scope, scope_id, key, value, updated_at) "
                    "VALUES ($1, $2, $3, $4::jsonb, now()) "
                    "ON CONFLICT (scope, scope_id, key) "
                    "DO UPDATE SET value = $4::jsonb, updated_at = now()",
                    scope.value,
                    scope_id,
                    key,
                    serialized,
                )

        await execute_with_retry(_op, retry_policy=self._retry_policy, operation_name="pg_set")

    async def delete(self, key: str, scope: StateScope, scope_id: str) -> bool:
        async def _op() -> bool:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM agentlegatus_state "
                    "WHERE scope = $1 AND scope_id = $2 AND key = $3",
                    scope.value,
                    scope_id,
                    key,
                )
            return result == "DELETE 1"

        return await execute_with_retry(
            _op, retry_policy=self._retry_policy, operation_name="pg_delete"
        )

    async def get_all(self, scope: StateScope, scope_id: str) -> dict[str, Any]:
        async def _op() -> dict[str, Any]:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT key, value FROM agentlegatus_state "
                    "WHERE scope = $1 AND scope_id = $2",
                    scope.value,
                    scope_id,
                )
            return {row["key"]: json.loads(row["value"]) for row in rows}

        return await execute_with_retry(
            _op, retry_policy=self._retry_policy, operation_name="pg_get_all"
        )

    async def clear_scope(self, scope: StateScope, scope_id: str) -> None:
        async def _op() -> None:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                await conn.execute(
                    "DELETE FROM agentlegatus_state " "WHERE scope = $1 AND scope_id = $2",
                    scope.value,
                    scope_id,
                )

        await execute_with_retry(
            _op, retry_policy=self._retry_policy, operation_name="pg_clear_scope"
        )

    async def create_snapshot(self, snapshot_id: str, scope: StateScope, scope_id: str) -> None:
        async def _op() -> None:
            pool = await self._get_pool()
            data = await self.get_all(scope, scope_id)
            serialized = json.dumps(data)
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        "INSERT INTO agentlegatus_snapshots (snapshot_id, scope, scope_id, data) "
                        "VALUES ($1, $2, $3, $4::jsonb) "
                        "ON CONFLICT (snapshot_id, scope, scope_id) "
                        "DO UPDATE SET data = $4::jsonb, created_at = now()",
                        snapshot_id,
                        scope.value,
                        scope_id,
                        serialized,
                    )

        await execute_with_retry(
            _op, retry_policy=self._retry_policy, operation_name="pg_create_snapshot"
        )

    async def restore_snapshot(self, snapshot_id: str, scope: StateScope, scope_id: str) -> None:
        async def _op() -> None:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT data FROM agentlegatus_snapshots "
                    "WHERE snapshot_id = $1 AND scope = $2 AND scope_id = $3",
                    snapshot_id,
                    scope.value,
                    scope_id,
                )
            if row is None:
                raise ValueError(f"Snapshot '{snapshot_id}' not found")

            data: dict[str, Any] = json.loads(row["data"])

            async with pool.acquire() as conn:
                async with conn.transaction():
                    # Clear current scope
                    await conn.execute(
                        "DELETE FROM agentlegatus_state " "WHERE scope = $1 AND scope_id = $2",
                        scope.value,
                        scope_id,
                    )
                    # Restore all keys
                    for k, v in data.items():
                        await conn.execute(
                            "INSERT INTO agentlegatus_state (scope, scope_id, key, value) "
                            "VALUES ($1, $2, $3, $4::jsonb)",
                            scope.value,
                            scope_id,
                            k,
                            json.dumps(v),
                        )

        await execute_with_retry(
            _op, retry_policy=self._retry_policy, operation_name="pg_restore_snapshot"
        )

    async def list_snapshots(self) -> list[str]:
        async def _op() -> list[str]:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT DISTINCT snapshot_id FROM agentlegatus_snapshots "
                    "ORDER BY snapshot_id"
                )
            return [row["snapshot_id"] for row in rows]

        return await execute_with_retry(
            _op, retry_policy=self._retry_policy, operation_name="pg_list_snapshots"
        )
