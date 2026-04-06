"""Error recovery mechanisms for AgentLegatus.

Provides:
- Reconnection logic for state backend failures (Req 15.4)
- Rollback for failed provider switches (Req 15.8)
- State preservation on workflow timeout (Req 15.9)
- Fallback to in-memory storage when backends unavailable (Req 15.5)

Requirements: 15.4, 15.5, 15.8, 15.9
"""

from typing import Any

from agentlegatus.core.state import InMemoryStateBackend, StateBackend, StateManager, StateScope
from agentlegatus.core.workflow import RetryPolicy
from agentlegatus.exceptions import (
    ProviderSwitchError,
    StateBackendUnavailableError,
)
from agentlegatus.utils.logging import get_logger, log_error

logger = get_logger(__name__)


class ResilientStateManager(StateManager):
    """StateManager with automatic reconnection and in-memory fallback.

    Wraps a primary backend and falls back to an in-memory backend when
    the primary is unavailable.  Reconnection to the primary is attempted
    on every subsequent operation so that the system self-heals once the
    backend comes back online.

    Requirements:
        15.4 - reconnection logic for state backend failures
        15.5 - fallback to in-memory storage when backends unavailable
    """

    def __init__(
        self,
        backend: StateBackend,
        default_scope_id: str = "default",
        event_bus: Any | None = None,
        reconnect_policy: RetryPolicy | None = None,
    ):
        super().__init__(backend, default_scope_id, event_bus)
        self._primary_backend = backend
        self._fallback_backend = InMemoryStateBackend()
        self._using_fallback = False
        self._reconnect_policy = reconnect_policy or RetryPolicy(
            max_attempts=3,
            initial_delay=0.5,
            backoff_multiplier=2.0,
            max_delay=5.0,
        )

    @property
    def is_using_fallback(self) -> bool:
        """Return True if currently using the in-memory fallback backend."""
        return self._using_fallback

    async def _try_reconnect(self) -> bool:
        """Attempt to reconnect to the primary backend.

        Returns:
            True if reconnection succeeded, False otherwise.
        """
        if not self._using_fallback:
            return True

        try:
            # Probe the primary backend with a lightweight operation
            await self._primary_backend.get_all(StateScope.GLOBAL, "__probe__")
            # If we get here, the primary is back
            self.backend = self._primary_backend
            self._using_fallback = False
            logger.info("Reconnected to primary state backend")
            return True
        except Exception:
            return False

    async def _execute_with_fallback(self, operation_name: str, coro_factory):
        """Execute a backend operation with fallback on failure.

        Args:
            operation_name: Name of the operation for logging.
            coro_factory: Callable that returns a coroutine for the operation.
                          Called once for the primary attempt and once for fallback.
        """
        # If already on fallback, try reconnecting first
        if self._using_fallback:
            await self._try_reconnect()

        try:
            return await coro_factory()
        except Exception as exc:
            if self._using_fallback:
                # Already on fallback and it failed — re-raise
                raise

            # Primary backend failed — switch to fallback
            backend_type = type(self._primary_backend).__name__
            logger.warning(
                f"Primary state backend ({backend_type}) failed during "
                f"'{operation_name}': {exc}. Falling back to in-memory storage."
            )
            self.backend = self._fallback_backend
            self._using_fallback = True

            # Retry the operation on the fallback backend
            try:
                return await coro_factory()
            except Exception as fallback_exc:
                raise StateBackendUnavailableError(
                    backend_type=backend_type,
                    reason=f"Both primary and fallback failed: {fallback_exc}",
                    original_error=exc,
                ) from fallback_exc

    # ------------------------------------------------------------------
    # Override StateManager methods to add resilience
    # ------------------------------------------------------------------

    async def get(
        self,
        key: str,
        scope: StateScope = StateScope.WORKFLOW,
        scope_id: str | None = None,
        default: Any = None,
    ) -> Any:
        scope_id = scope_id or self.default_scope_id

        async def _op():
            value = await self.backend.get(key, scope, scope_id)
            return value if value is not None else default

        return await self._execute_with_fallback("get", _op)

    async def set(
        self,
        key: str,
        value: Any,
        scope: StateScope = StateScope.WORKFLOW,
        scope_id: str | None = None,
        ttl: int | None = None,
    ) -> None:
        scope_id = scope_id or self.default_scope_id

        async def _op():
            await self.backend.set(key, value, scope, scope_id, ttl)

        await self._execute_with_fallback("set", _op)

        # Emit state updated event (same as parent)
        if self.event_bus:
            from datetime import datetime

            from agentlegatus.core.event_bus import Event, EventType

            await self.event_bus.emit(
                Event(
                    event_type=EventType.STATE_UPDATED,
                    timestamp=datetime.now(),
                    source="ResilientStateManager",
                    data={
                        "key": key,
                        "scope": scope.value,
                        "scope_id": scope_id,
                        "operation": "set",
                        "using_fallback": self._using_fallback,
                    },
                )
            )

    async def get_all(
        self,
        scope: StateScope = StateScope.WORKFLOW,
        scope_id: str | None = None,
    ) -> dict[str, Any]:
        scope_id = scope_id or self.default_scope_id

        async def _op():
            return await self.backend.get_all(scope, scope_id)

        return await self._execute_with_fallback("get_all", _op)

    async def delete(
        self,
        key: str,
        scope: StateScope = StateScope.WORKFLOW,
        scope_id: str | None = None,
    ) -> bool:
        scope_id = scope_id or self.default_scope_id

        async def _op():
            return await self.backend.delete(key, scope, scope_id)

        return await self._execute_with_fallback("delete", _op)

    async def clear_scope(
        self,
        scope: StateScope,
        scope_id: str | None = None,
    ) -> None:
        scope_id = scope_id or self.default_scope_id

        async def _op():
            await self.backend.clear_scope(scope, scope_id)

        await self._execute_with_fallback("clear_scope", _op)

    async def create_snapshot(
        self,
        snapshot_id: str,
        scope: StateScope = StateScope.WORKFLOW,
        scope_id: str | None = None,
    ) -> None:
        scope_id = scope_id or self.default_scope_id

        async def _op():
            await self.backend.create_snapshot(snapshot_id, scope, scope_id)

        await self._execute_with_fallback("create_snapshot", _op)

    async def restore_snapshot(
        self,
        snapshot_id: str,
        scope: StateScope = StateScope.WORKFLOW,
        scope_id: str | None = None,
    ) -> None:
        scope_id = scope_id or self.default_scope_id

        async def _op():
            await self.backend.restore_snapshot(snapshot_id, scope, scope_id)

        await self._execute_with_fallback("restore_snapshot", _op)


async def rollback_provider_switch(
    executor: Any,
    old_provider: Any,
    new_provider_name: str,
    exported_state: dict[str, Any] | None = None,
) -> None:
    """Rollback a failed provider switch.

    Restores the executor to use the old provider and re-imports the
    previously exported state if available.

    Requirement 15.8: Rollback to previous provider and preserve state.

    Args:
        executor: WorkflowExecutor instance.
        old_provider: The provider to restore.
        new_provider_name: Name of the provider that failed (for logging).
        exported_state: State that was exported before the switch attempt.

    Raises:
        ProviderSwitchError: If rollback itself fails.
    """
    old_name = type(old_provider).__name__
    logger.info(
        f"Rolling back provider switch: restoring '{old_name}' "
        f"(failed target: '{new_provider_name}')"
    )

    try:
        executor.provider = old_provider
        if exported_state is not None:
            old_provider.import_state(exported_state)
        logger.info(f"Rollback to '{old_name}' successful")
    except Exception as exc:
        log_error(
            logger,
            "Rollback failed — executor may be in inconsistent state",
            exc,
            old_provider=old_name,
            new_provider=new_provider_name,
        )
        raise ProviderSwitchError(
            old_provider=old_name,
            new_provider=new_provider_name,
            reason=f"Rollback failed: {exc}",
            original_error=exc,
        ) from exc


async def preserve_state_on_timeout(
    executor: Any,
    workflow_id: str,
) -> str | None:
    """Create a checkpoint when a workflow times out.

    Requirement 15.9: Preserve state on workflow timeout.

    Args:
        executor: WorkflowExecutor instance.
        workflow_id: Identifier of the timed-out workflow.

    Returns:
        The checkpoint ID if successful, None otherwise.
    """
    import time

    checkpoint_id = f"timeout_{workflow_id}_{int(time.time())}"
    try:
        await executor.checkpoint_state(checkpoint_id)
        logger.info(
            f"State preserved for timed-out workflow '{workflow_id}' "
            f"as checkpoint '{checkpoint_id}'"
        )
        return checkpoint_id
    except Exception as exc:
        log_error(
            logger,
            f"Failed to preserve state on timeout for workflow '{workflow_id}'",
            exc,
            workflow_id=workflow_id,
        )
        return None
