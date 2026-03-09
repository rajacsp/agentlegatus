"""State management abstractions and implementations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional

from agentlegatus.core.event_bus import Event, EventBus, EventType


class StateScope(Enum):
    """Scope levels for state isolation."""

    WORKFLOW = "workflow"
    STEP = "step"
    AGENT = "agent"
    GLOBAL = "global"


class StateBackend(ABC):
    """Abstract base class for state storage implementations."""

    @abstractmethod
    async def get(
        self, key: str, scope: StateScope, scope_id: str
    ) -> Optional[Any]:
        """
        Get state value by key and scope.

        Args:
            key: State key
            scope: State scope level
            scope_id: Identifier for the scope (e.g., workflow_id, step_id)

        Returns:
            State value or None if not found
        """
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        scope: StateScope,
        scope_id: str,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Set state value with optional TTL.

        Args:
            key: State key
            value: State value
            scope: State scope level
            scope_id: Identifier for the scope
            ttl: Time-to-live in seconds (optional)
        """
        pass

    @abstractmethod
    async def delete(self, key: str, scope: StateScope, scope_id: str) -> bool:
        """
        Delete state value.

        Args:
            key: State key
            scope: State scope level
            scope_id: Identifier for the scope

        Returns:
            True if value existed and was deleted, False otherwise
        """
        pass

    @abstractmethod
    async def get_all(self, scope: StateScope, scope_id: str) -> Dict[str, Any]:
        """
        Get all state values for a scope.

        Args:
            scope: State scope level
            scope_id: Identifier for the scope

        Returns:
            Dictionary of all key-value pairs in the scope
        """
        pass

    @abstractmethod
    async def clear_scope(self, scope: StateScope, scope_id: str) -> None:
        """
        Clear all state in a scope.

        Args:
            scope: State scope level
            scope_id: Identifier for the scope
        """
        pass

    @abstractmethod
    async def create_snapshot(
        self, snapshot_id: str, scope: StateScope, scope_id: str
    ) -> None:
        """
        Create a snapshot of current state.

        Args:
            snapshot_id: Unique identifier for the snapshot
            scope: State scope level
            scope_id: Identifier for the scope
        """
        pass

    @abstractmethod
    async def restore_snapshot(
        self, snapshot_id: str, scope: StateScope, scope_id: str
    ) -> None:
        """
        Restore state from a snapshot.

        Args:
            snapshot_id: Unique identifier for the snapshot
            scope: State scope level
            scope_id: Identifier for the scope
        """
        pass

    @abstractmethod
    async def list_snapshots(self) -> list[str]:
        """
        List all available snapshot IDs.

        Returns:
            List of snapshot IDs
        """
        pass



class StateManager:
    """Unified state management across providers."""

    def __init__(
        self,
        backend: StateBackend,
        default_scope_id: str = "default",
        event_bus: Optional[EventBus] = None,
    ):
        """
        Initialize state manager with backend.

        Args:
            backend: State storage backend
            default_scope_id: Default scope identifier
            event_bus: Optional event bus for state change events
        """
        self.backend = backend
        self.default_scope_id = default_scope_id
        self.event_bus = event_bus

    async def get(
        self,
        key: str,
        scope: StateScope = StateScope.WORKFLOW,
        scope_id: Optional[str] = None,
        default: Any = None,
    ) -> Any:
        """
        Get state value by key and scope.

        Args:
            key: State key
            scope: State scope level
            scope_id: Identifier for the scope (uses default if None)
            default: Default value if key not found

        Returns:
            State value or default if not found
        """
        scope_id = scope_id or self.default_scope_id
        value = await self.backend.get(key, scope, scope_id)
        return value if value is not None else default

    async def set(
        self,
        key: str,
        value: Any,
        scope: StateScope = StateScope.WORKFLOW,
        scope_id: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Set state value with optional TTL.

        Args:
            key: State key
            value: State value
            scope: State scope level
            scope_id: Identifier for the scope (uses default if None)
            ttl: Time-to-live in seconds (optional)
        """
        scope_id = scope_id or self.default_scope_id
        await self.backend.set(key, value, scope, scope_id, ttl)

        # Emit state updated event
        if self.event_bus:
            await self.event_bus.emit(
                Event(
                    event_type=EventType.STATE_UPDATED,
                    timestamp=datetime.now(),
                    source="StateManager",
                    data={
                        "key": key,
                        "scope": scope.value,
                        "scope_id": scope_id,
                        "operation": "set",
                    },
                )
            )

    async def update(
        self,
        key: str,
        updater: Callable[[Any], Any],
        scope: StateScope = StateScope.WORKFLOW,
        scope_id: Optional[str] = None,
    ) -> Any:
        """
        Update state value using updater function atomically.

        Args:
            key: State key
            updater: Function that takes current value and returns new value
            scope: State scope level
            scope_id: Identifier for the scope (uses default if None)

        Returns:
            Updated value
        """
        scope_id = scope_id or self.default_scope_id

        # Get current value
        current_value = await self.backend.get(key, scope, scope_id)

        # Apply updater function
        new_value = updater(current_value)

        # Set new value
        await self.backend.set(key, new_value, scope, scope_id)

        # Emit state updated event
        if self.event_bus:
            await self.event_bus.emit(
                Event(
                    event_type=EventType.STATE_UPDATED,
                    timestamp=datetime.now(),
                    source="StateManager",
                    data={
                        "key": key,
                        "scope": scope.value,
                        "scope_id": scope_id,
                        "operation": "update",
                    },
                )
            )

        return new_value

    async def delete(
        self,
        key: str,
        scope: StateScope = StateScope.WORKFLOW,
        scope_id: Optional[str] = None,
    ) -> bool:
        """
        Delete state value.

        Args:
            key: State key
            scope: State scope level
            scope_id: Identifier for the scope (uses default if None)

        Returns:
            True if value existed and was deleted, False otherwise
        """
        scope_id = scope_id or self.default_scope_id
        deleted = await self.backend.delete(key, scope, scope_id)

        # Emit state updated event if deleted
        if deleted and self.event_bus:
            await self.event_bus.emit(
                Event(
                    event_type=EventType.STATE_UPDATED,
                    timestamp=datetime.now(),
                    source="StateManager",
                    data={
                        "key": key,
                        "scope": scope.value,
                        "scope_id": scope_id,
                        "operation": "delete",
                    },
                )
            )

        return deleted

    async def get_all(
        self,
        scope: StateScope = StateScope.WORKFLOW,
        scope_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get all state values for a scope.

        Args:
            scope: State scope level
            scope_id: Identifier for the scope (uses default if None)

        Returns:
            Dictionary of all key-value pairs in the scope
        """
        scope_id = scope_id or self.default_scope_id
        return await self.backend.get_all(scope, scope_id)

    async def clear_scope(
        self,
        scope: StateScope,
        scope_id: Optional[str] = None,
    ) -> None:
        """
        Clear all state in a scope.

        Args:
            scope: State scope level
            scope_id: Identifier for the scope (uses default if None)
        """
        scope_id = scope_id or self.default_scope_id
        await self.backend.clear_scope(scope, scope_id)

        # Emit state updated event
        if self.event_bus:
            await self.event_bus.emit(
                Event(
                    event_type=EventType.STATE_UPDATED,
                    timestamp=datetime.now(),
                    source="StateManager",
                    data={
                        "scope": scope.value,
                        "scope_id": scope_id,
                        "operation": "clear_scope",
                    },
                )
            )

    async def create_snapshot(
        self,
        snapshot_id: str,
        scope: StateScope = StateScope.WORKFLOW,
        scope_id: Optional[str] = None,
    ) -> None:
        """
        Create a snapshot of current state.

        Args:
            snapshot_id: Unique identifier for the snapshot
            scope: State scope level
            scope_id: Identifier for the scope (uses default if None)
        """
        scope_id = scope_id or self.default_scope_id
        await self.backend.create_snapshot(snapshot_id, scope, scope_id)

    async def restore_snapshot(
        self,
        snapshot_id: str,
        scope: StateScope = StateScope.WORKFLOW,
        scope_id: Optional[str] = None,
    ) -> None:
        """
        Restore state from a snapshot.

        Args:
            snapshot_id: Unique identifier for the snapshot
            scope: State scope level
            scope_id: Identifier for the scope (uses default if None)
        """
        scope_id = scope_id or self.default_scope_id
        await self.backend.restore_snapshot(snapshot_id, scope, scope_id)

        # Emit state updated event
        if self.event_bus:
            await self.event_bus.emit(
                Event(
                    event_type=EventType.STATE_UPDATED,
                    timestamp=datetime.now(),
                    source="StateManager",
                    data={
                        "scope": scope.value,
                        "scope_id": scope_id,
                        "operation": "restore_snapshot",
                        "snapshot_id": snapshot_id,
                    },
                )
            )

    async def list_snapshots(self) -> list[str]:
        """
        List all available snapshot IDs.

        Returns:
            List of snapshot IDs
        """
        return await self.backend.list_snapshots()



class InMemoryStateBackend(StateBackend):
    """In-memory state backend for development and testing."""

    def __init__(self):
        """Initialize in-memory backend."""
        # Structure: {scope: {scope_id: {key: value}}}
        self._storage: Dict[StateScope, Dict[str, Dict[str, Any]]] = {
            StateScope.WORKFLOW: {},
            StateScope.STEP: {},
            StateScope.AGENT: {},
            StateScope.GLOBAL: {},
        }
        # Structure: {snapshot_id: {scope: {scope_id: {key: value}}}}
        self._snapshots: Dict[str, Dict[StateScope, Dict[str, Dict[str, Any]]]] = {}

    def _get_scope_storage(
        self, scope: StateScope, scope_id: str
    ) -> Dict[str, Any]:
        """
        Get or create storage for a specific scope and scope_id.

        Args:
            scope: State scope level
            scope_id: Identifier for the scope

        Returns:
            Dictionary for the scope storage
        """
        if scope_id not in self._storage[scope]:
            self._storage[scope][scope_id] = {}
        return self._storage[scope][scope_id]

    async def get(
        self, key: str, scope: StateScope, scope_id: str
    ) -> Optional[Any]:
        """Get state value by key and scope."""
        storage = self._get_scope_storage(scope, scope_id)
        return storage.get(key)

    async def set(
        self,
        key: str,
        value: Any,
        scope: StateScope,
        scope_id: str,
        ttl: Optional[int] = None,
    ) -> None:
        """Set state value with optional TTL."""
        storage = self._get_scope_storage(scope, scope_id)
        storage[key] = value
        # Note: TTL not implemented in in-memory backend for simplicity

    async def delete(self, key: str, scope: StateScope, scope_id: str) -> bool:
        """Delete state value."""
        storage = self._get_scope_storage(scope, scope_id)
        if key in storage:
            del storage[key]
            return True
        return False

    async def get_all(self, scope: StateScope, scope_id: str) -> Dict[str, Any]:
        """Get all state values for a scope."""
        storage = self._get_scope_storage(scope, scope_id)
        return storage.copy()

    async def clear_scope(self, scope: StateScope, scope_id: str) -> None:
        """Clear all state in a scope."""
        if scope_id in self._storage[scope]:
            self._storage[scope][scope_id].clear()

    async def create_snapshot(
        self, snapshot_id: str, scope: StateScope, scope_id: str
    ) -> None:
        """Create a snapshot of current state."""
        if snapshot_id not in self._snapshots:
            self._snapshots[snapshot_id] = {
                StateScope.WORKFLOW: {},
                StateScope.STEP: {},
                StateScope.AGENT: {},
                StateScope.GLOBAL: {},
            }

        # Deep copy the current state for the specified scope
        storage = self._get_scope_storage(scope, scope_id)
        self._snapshots[snapshot_id][scope][scope_id] = {
            k: v for k, v in storage.items()
        }

    async def restore_snapshot(
        self, snapshot_id: str, scope: StateScope, scope_id: str
    ) -> None:
        """Restore state from a snapshot."""
        if snapshot_id not in self._snapshots:
            raise ValueError(f"Snapshot '{snapshot_id}' not found")

        if scope_id not in self._snapshots[snapshot_id][scope]:
            raise ValueError(
                f"Snapshot '{snapshot_id}' does not contain data for scope '{scope.value}' with id '{scope_id}'"
            )

        # Restore the snapshot data
        snapshot_data = self._snapshots[snapshot_id][scope][scope_id]
        self._storage[scope][scope_id] = {k: v for k, v in snapshot_data.items()}

    async def list_snapshots(self) -> list[str]:
        """List all available snapshot IDs."""
        return list(self._snapshots.keys())
