"""Unit tests for EventBus.

Covers:
- subscribe() – registering handlers, returns subscription ID
- unsubscribe() – removing handlers by subscription ID, returns True/False
- emit() – emitting events invokes all subscribed handlers
- emit_and_wait() – emitting and waiting for all handlers to complete
- Event history tracking – get_event_history() with filtering
- Handler isolation – exception in one handler doesn't affect others
- Multiple handlers for same event type
- No handlers for an event type (emit should not fail)
- clear_history() clears event history
- correlation_id and trace_id preservation in events
"""

import asyncio
from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from agentlegatus.core.event_bus import Event, EventBus, EventType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    event_type: EventType = EventType.WORKFLOW_STARTED,
    source: str = "test",
    data: dict | None = None,
    correlation_id: str | None = None,
    trace_id: str | None = None,
    timestamp: datetime | None = None,
) -> Event:
    return Event(
        event_type=event_type,
        timestamp=timestamp or datetime.now(tz=None),
        source=source,
        data=data or {},
        correlation_id=correlation_id,
        trace_id=trace_id,
    )


# ---------------------------------------------------------------------------
# subscribe()
# ---------------------------------------------------------------------------


class TestSubscribe:
    def test_subscribe_returns_string_id(self):
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass

        sub_id = bus.subscribe(EventType.WORKFLOW_STARTED, handler)
        assert isinstance(sub_id, str)
        assert len(sub_id) > 0

    def test_subscribe_returns_unique_ids(self):
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass

        ids = {bus.subscribe(EventType.WORKFLOW_STARTED, handler) for _ in range(50)}
        assert len(ids) == 50

    def test_subscribe_multiple_event_types(self):
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass

        id1 = bus.subscribe(EventType.WORKFLOW_STARTED, handler)
        id2 = bus.subscribe(EventType.WORKFLOW_COMPLETED, handler)
        assert id1 != id2


# ---------------------------------------------------------------------------
# unsubscribe()
# ---------------------------------------------------------------------------


class TestUnsubscribe:
    def test_unsubscribe_existing_returns_true(self):
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass

        sub_id = bus.subscribe(EventType.WORKFLOW_STARTED, handler)
        assert bus.unsubscribe(sub_id) is True

    def test_unsubscribe_nonexistent_returns_false(self):
        bus = EventBus()
        assert bus.unsubscribe("nonexistent-id") is False

    def test_unsubscribe_same_id_twice_returns_false_second_time(self):
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass

        sub_id = bus.subscribe(EventType.WORKFLOW_STARTED, handler)
        assert bus.unsubscribe(sub_id) is True
        assert bus.unsubscribe(sub_id) is False

    @pytest.mark.asyncio
    async def test_unsubscribed_handler_not_invoked(self):
        bus = EventBus()
        called = []

        async def handler(event: Event) -> None:
            called.append(event)

        sub_id = bus.subscribe(EventType.WORKFLOW_STARTED, handler)
        bus.unsubscribe(sub_id)

        await bus.emit(_make_event(EventType.WORKFLOW_STARTED))
        # Give fire-and-forget tasks a chance to run
        await asyncio.sleep(0.05)
        assert len(called) == 0


# ---------------------------------------------------------------------------
# emit()
# ---------------------------------------------------------------------------


class TestEmit:
    @pytest.mark.asyncio
    async def test_emit_invokes_subscribed_handler(self):
        bus = EventBus()
        received = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.STEP_STARTED, handler)
        event = _make_event(EventType.STEP_STARTED)
        await bus.emit(event)
        await asyncio.sleep(0.05)

        assert len(received) == 1
        assert received[0] is event

    @pytest.mark.asyncio
    async def test_emit_does_not_invoke_handler_for_different_type(self):
        bus = EventBus()
        received = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.STEP_STARTED, handler)
        await bus.emit(_make_event(EventType.WORKFLOW_COMPLETED))
        await asyncio.sleep(0.05)

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_emit_no_handlers_does_not_fail(self):
        bus = EventBus()
        # Should not raise
        await bus.emit(_make_event(EventType.TOOL_INVOKED))

    @pytest.mark.asyncio
    async def test_emit_multiple_handlers_same_event_type(self):
        bus = EventBus()
        results_a = []
        results_b = []

        async def handler_a(event: Event) -> None:
            results_a.append(event.source)

        async def handler_b(event: Event) -> None:
            results_b.append(event.source)

        bus.subscribe(EventType.AGENT_CREATED, handler_a)
        bus.subscribe(EventType.AGENT_CREATED, handler_b)

        await bus.emit(_make_event(EventType.AGENT_CREATED, source="src"))
        await asyncio.sleep(0.05)

        assert results_a == ["src"]
        assert results_b == ["src"]


# ---------------------------------------------------------------------------
# emit_and_wait()
# ---------------------------------------------------------------------------


class TestEmitAndWait:
    @pytest.mark.asyncio
    async def test_emit_and_wait_returns_handler_results(self):
        bus = EventBus()

        async def handler(event: Event):
            return "ok"

        bus.subscribe(EventType.WORKFLOW_COMPLETED, handler)
        results = await bus.emit_and_wait(_make_event(EventType.WORKFLOW_COMPLETED))
        assert results == ["ok"]

    @pytest.mark.asyncio
    async def test_emit_and_wait_multiple_handlers(self):
        bus = EventBus()

        async def handler_a(event: Event):
            return 1

        async def handler_b(event: Event):
            return 2

        bus.subscribe(EventType.WORKFLOW_COMPLETED, handler_a)
        bus.subscribe(EventType.WORKFLOW_COMPLETED, handler_b)

        results = await bus.emit_and_wait(_make_event(EventType.WORKFLOW_COMPLETED))
        assert sorted(results) == [1, 2]

    @pytest.mark.asyncio
    async def test_emit_and_wait_no_handlers_returns_empty(self):
        bus = EventBus()
        results = await bus.emit_and_wait(_make_event(EventType.WORKFLOW_FAILED))
        assert results == []

    @pytest.mark.asyncio
    async def test_emit_and_wait_handler_exception_returns_none(self):
        bus = EventBus()

        async def bad_handler(event: Event):
            raise ValueError("boom")

        async def good_handler(event: Event):
            return "fine"

        bus.subscribe(EventType.STEP_COMPLETED, bad_handler)
        bus.subscribe(EventType.STEP_COMPLETED, good_handler)

        results = await bus.emit_and_wait(_make_event(EventType.STEP_COMPLETED))
        assert None in results
        assert "fine" in results

    @pytest.mark.asyncio
    async def test_emit_and_wait_timeout(self):
        bus = EventBus()

        async def slow_handler(event: Event):
            await asyncio.sleep(10)
            return "late"

        bus.subscribe(EventType.STEP_FAILED, slow_handler)
        results = await bus.emit_and_wait(
            _make_event(EventType.STEP_FAILED), timeout=0.1
        )
        # On timeout the implementation returns []
        assert results == []


# ---------------------------------------------------------------------------
# Event history tracking
# ---------------------------------------------------------------------------


class TestEventHistory:
    @pytest.mark.asyncio
    async def test_emit_adds_to_history(self):
        bus = EventBus()
        event = _make_event(EventType.WORKFLOW_STARTED)
        await bus.emit(event)

        history = bus.get_event_history()
        assert len(history) == 1
        assert history[0] is event

    @pytest.mark.asyncio
    async def test_emit_and_wait_adds_to_history(self):
        bus = EventBus()
        event = _make_event(EventType.WORKFLOW_COMPLETED)
        await bus.emit_and_wait(event)

        history = bus.get_event_history()
        assert len(history) == 1
        assert history[0] is event

    @pytest.mark.asyncio
    async def test_history_chronological_order(self):
        bus = EventBus()
        events = []
        for i in range(5):
            e = _make_event(EventType.STEP_STARTED, source=f"step-{i}")
            events.append(e)
            await bus.emit(e)

        history = bus.get_event_history()
        assert [e.source for e in history] == [f"step-{i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_history_filter_by_event_type(self):
        bus = EventBus()
        await bus.emit(_make_event(EventType.WORKFLOW_STARTED))
        await bus.emit(_make_event(EventType.STEP_STARTED))
        await bus.emit(_make_event(EventType.WORKFLOW_COMPLETED))

        history = bus.get_event_history(event_type=EventType.STEP_STARTED)
        assert len(history) == 1
        assert history[0].event_type == EventType.STEP_STARTED

    @pytest.mark.asyncio
    async def test_history_filter_by_since(self):
        bus = EventBus()
        old_time = datetime(2020, 1, 1)
        recent_time = datetime(2025, 1, 1)

        await bus.emit(_make_event(EventType.STEP_STARTED, timestamp=old_time))
        await bus.emit(_make_event(EventType.STEP_COMPLETED, timestamp=recent_time))

        cutoff = datetime(2024, 1, 1)
        history = bus.get_event_history(since=cutoff)
        assert len(history) == 1
        assert history[0].timestamp == recent_time

    @pytest.mark.asyncio
    async def test_history_limit(self):
        bus = EventBus()
        for i in range(10):
            await bus.emit(_make_event(EventType.STEP_STARTED, source=f"s-{i}"))

        history = bus.get_event_history(limit=3)
        assert len(history) == 3
        # Should return the most recent 3
        assert [e.source for e in history] == ["s-7", "s-8", "s-9"]

    @pytest.mark.asyncio
    async def test_history_combined_filters(self):
        bus = EventBus()
        t1 = datetime(2024, 1, 1)
        t2 = datetime(2025, 1, 1)
        t3 = datetime(2025, 6, 1)

        await bus.emit(_make_event(EventType.STEP_STARTED, timestamp=t1))
        await bus.emit(_make_event(EventType.STEP_STARTED, timestamp=t2))
        await bus.emit(_make_event(EventType.STEP_COMPLETED, timestamp=t3))
        await bus.emit(_make_event(EventType.STEP_STARTED, timestamp=t3))

        history = bus.get_event_history(
            event_type=EventType.STEP_STARTED,
            since=datetime(2024, 6, 1),
            limit=1,
        )
        assert len(history) == 1
        assert history[0].timestamp == t3

    @pytest.mark.asyncio
    async def test_history_max_size_trimming(self):
        bus = EventBus()
        bus._max_history_size = 5

        for i in range(10):
            await bus.emit(_make_event(EventType.STEP_STARTED, source=f"s-{i}"))

        history = bus.get_event_history()
        assert len(history) == 5
        # Should keep the most recent 5
        assert [e.source for e in history] == [f"s-{i}" for i in range(5, 10)]


# ---------------------------------------------------------------------------
# Handler isolation
# ---------------------------------------------------------------------------


class TestHandlerIsolation:
    @pytest.mark.asyncio
    async def test_exception_in_handler_does_not_affect_others_emit(self):
        bus = EventBus()
        results = []

        async def bad_handler(event: Event) -> None:
            raise RuntimeError("handler crash")

        async def good_handler(event: Event) -> None:
            results.append("ok")

        bus.subscribe(EventType.AGENT_EXECUTING, bad_handler)
        bus.subscribe(EventType.AGENT_EXECUTING, good_handler)

        await bus.emit(_make_event(EventType.AGENT_EXECUTING))
        await asyncio.sleep(0.05)

        assert results == ["ok"]

    @pytest.mark.asyncio
    async def test_exception_in_handler_does_not_affect_others_emit_and_wait(self):
        bus = EventBus()

        async def bad_handler(event: Event):
            raise RuntimeError("handler crash")

        async def good_handler(event: Event):
            return 42

        bus.subscribe(EventType.AGENT_COMPLETED, bad_handler)
        bus.subscribe(EventType.AGENT_COMPLETED, good_handler)

        results = await bus.emit_and_wait(_make_event(EventType.AGENT_COMPLETED))
        assert None in results
        assert 42 in results


# ---------------------------------------------------------------------------
# clear_history()
# ---------------------------------------------------------------------------


class TestClearHistory:
    @pytest.mark.asyncio
    async def test_clear_history_empties_history(self):
        bus = EventBus()
        await bus.emit(_make_event(EventType.WORKFLOW_STARTED))
        await bus.emit(_make_event(EventType.WORKFLOW_COMPLETED))

        assert len(bus.get_event_history()) == 2
        bus.clear_history()
        assert len(bus.get_event_history()) == 0

    @pytest.mark.asyncio
    async def test_clear_history_does_not_affect_subscriptions(self):
        bus = EventBus()
        received = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe(EventType.STEP_STARTED, handler)
        await bus.emit(_make_event(EventType.STEP_STARTED))
        await asyncio.sleep(0.05)

        bus.clear_history()

        await bus.emit(_make_event(EventType.STEP_STARTED))
        await asyncio.sleep(0.05)

        # Handler should still be active after clear_history
        assert len(received) == 2


# ---------------------------------------------------------------------------
# correlation_id and trace_id preservation
# ---------------------------------------------------------------------------


class TestCorrelationAndTraceId:
    @pytest.mark.asyncio
    async def test_correlation_id_preserved_in_handler(self):
        bus = EventBus()
        captured = []

        async def handler(event: Event) -> None:
            captured.append(event.correlation_id)

        bus.subscribe(EventType.WORKFLOW_STARTED, handler)
        await bus.emit(
            _make_event(EventType.WORKFLOW_STARTED, correlation_id="corr-123")
        )
        await asyncio.sleep(0.05)

        assert captured == ["corr-123"]

    @pytest.mark.asyncio
    async def test_trace_id_preserved_in_handler(self):
        bus = EventBus()
        captured = []

        async def handler(event: Event) -> None:
            captured.append(event.trace_id)

        bus.subscribe(EventType.WORKFLOW_STARTED, handler)
        await bus.emit(
            _make_event(EventType.WORKFLOW_STARTED, trace_id="trace-abc")
        )
        await asyncio.sleep(0.05)

        assert captured == ["trace-abc"]

    @pytest.mark.asyncio
    async def test_correlation_and_trace_preserved_in_history(self):
        bus = EventBus()
        await bus.emit(
            _make_event(
                EventType.STEP_COMPLETED,
                correlation_id="c-1",
                trace_id="t-1",
            )
        )

        history = bus.get_event_history()
        assert history[0].correlation_id == "c-1"
        assert history[0].trace_id == "t-1"

    @pytest.mark.asyncio
    async def test_none_correlation_and_trace_by_default(self):
        bus = EventBus()
        await bus.emit(_make_event(EventType.STATE_UPDATED))

        history = bus.get_event_history()
        assert history[0].correlation_id is None
        assert history[0].trace_id is None
