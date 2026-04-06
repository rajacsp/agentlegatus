"""Unit tests for error recovery mechanisms."""

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentlegatus.core.event_bus import EventBus
from agentlegatus.core.recovery import ResilientStateManager
from agentlegatus.core.state import InMemoryStateBackend, StateBackend, StateScope
from agentlegatus.exceptions import StateBackendUnavailableError


class FailingBackend(StateBackend):
    """Backend that fails on demand for testing."""

    def __init__(self, fail_after: int = 0):
        self._call_count = 0
        self._fail_after = fail_after  # 0 = fail immediately
        self._recovered = False

    async def get(self, key, scope, scope_id):
        self._call_count += 1
        if not self._recovered and self._call_count > self._fail_after:
            raise ConnectionError("backend unavailable")
        return f"value_{key}"

    async def set(self, key, value, scope, scope_id, ttl=None):
        self._call_count += 1
        if not self._recovered and self._call_count > self._fail_after:
            raise ConnectionError("backend unavailable")

    async def delete(self, key, scope, scope_id):
        self._call_count += 1
        if not self._recovered and self._call_count > self._fail_after:
            raise ConnectionError("backend unavailable")
        return True

    async def get_all(self, scope, scope_id):
        self._call_count += 1
        if not self._recovered and self._call_count > self._fail_after:
            raise ConnectionError("backend unavailable")
        return {}

    async def clear_scope(self, scope, scope_id):
        self._call_count += 1
        if not self._recovered and self._call_count > self._fail_after:
            raise ConnectionError("backend unavailable")

    async def create_snapshot(self, snapshot_id, scope, scope_id):
        self._call_count += 1
        if not self._recovered and self._call_count > self._fail_after:
            raise ConnectionError("backend unavailable")

    async def restore_snapshot(self, snapshot_id, scope, scope_id):
        self._call_count += 1
        if not self._recovered and self._call_count > self._fail_after:
            raise ConnectionError("backend unavailable")

    async def list_snapshots(self):
        return []


class TestResilientStateManager:
    """Tests for ResilientStateManager fallback and reconnection."""

    @pytest.mark.asyncio
    async def test_normal_operation_uses_primary(self):
        """When primary is healthy, it is used directly."""
        primary = InMemoryStateBackend()
        mgr = ResilientStateManager(primary)

        await mgr.set("k", "v", StateScope.WORKFLOW)
        result = await mgr.get("k", StateScope.WORKFLOW)
        assert result == "v"
        assert not mgr.is_using_fallback

    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self):
        """When primary fails, operations fall back to in-memory."""
        failing = FailingBackend(fail_after=0)
        mgr = ResilientStateManager(failing)

        # This should trigger fallback
        await mgr.set("k", "v", StateScope.WORKFLOW)
        assert mgr.is_using_fallback

        # Subsequent reads should work via fallback
        result = await mgr.get("k", StateScope.WORKFLOW)
        assert result == "v"

    @pytest.mark.asyncio
    async def test_reconnect_after_recovery(self):
        """When primary recovers, operations switch back."""
        failing = FailingBackend(fail_after=0)
        mgr = ResilientStateManager(failing)

        # Trigger fallback
        await mgr.set("k", "v", StateScope.WORKFLOW)
        assert mgr.is_using_fallback

        # Simulate primary recovery
        failing._recovered = True

        # Next operation should attempt reconnect
        result = await mgr.get_all(StateScope.WORKFLOW)
        assert not mgr.is_using_fallback

    @pytest.mark.asyncio
    async def test_get_all_with_fallback(self):
        """get_all works through fallback."""
        failing = FailingBackend(fail_after=0)
        mgr = ResilientStateManager(failing)

        await mgr.set("a", 1, StateScope.WORKFLOW)
        await mgr.set("b", 2, StateScope.WORKFLOW)
        result = await mgr.get_all(StateScope.WORKFLOW)
        assert result == {"a": 1, "b": 2}
        assert mgr.is_using_fallback

    @pytest.mark.asyncio
    async def test_delete_with_fallback(self):
        """delete works through fallback."""
        failing = FailingBackend(fail_after=0)
        mgr = ResilientStateManager(failing)

        await mgr.set("k", "v", StateScope.WORKFLOW)
        deleted = await mgr.delete("k", StateScope.WORKFLOW)
        assert deleted is True

    @pytest.mark.asyncio
    async def test_clear_scope_with_fallback(self):
        """clear_scope works through fallback."""
        failing = FailingBackend(fail_after=0)
        mgr = ResilientStateManager(failing)

        await mgr.set("k", "v", StateScope.WORKFLOW)
        await mgr.clear_scope(StateScope.WORKFLOW)
        result = await mgr.get_all(StateScope.WORKFLOW)
        assert result == {}

    @pytest.mark.asyncio
    async def test_snapshot_with_fallback(self):
        """create_snapshot and restore_snapshot work through fallback."""
        failing = FailingBackend(fail_after=0)
        mgr = ResilientStateManager(failing)

        await mgr.set("k", "v", StateScope.WORKFLOW)
        await mgr.create_snapshot("snap1", StateScope.WORKFLOW)
        await mgr.set("k", "changed", StateScope.WORKFLOW)
        await mgr.restore_snapshot("snap1", StateScope.WORKFLOW)
        result = await mgr.get("k", StateScope.WORKFLOW)
        assert result == "v"

    @pytest.mark.asyncio
    async def test_default_value_on_missing_key(self):
        """get returns default when key is missing."""
        primary = InMemoryStateBackend()
        mgr = ResilientStateManager(primary)

        result = await mgr.get("missing", StateScope.WORKFLOW, default="fallback")
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_event_emission_on_set(self):
        """set emits STATE_UPDATED event when event_bus is provided."""
        primary = InMemoryStateBackend()
        event_bus = EventBus()
        mgr = ResilientStateManager(primary, event_bus=event_bus)

        await mgr.set("k", "v", StateScope.WORKFLOW)

        history = event_bus.get_event_history()
        assert len(history) >= 1
        assert any(e.data.get("key") == "k" for e in history)
