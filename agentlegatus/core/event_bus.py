"""Event bus for event-driven architecture."""

import asyncio
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from agentlegatus.utils.logging import get_logger

logger = get_logger(__name__)


class EventType(Enum):
    """Types of events in the system."""

    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_CANCELLED = "workflow.cancelled"
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    AGENT_CREATED = "agent.created"
    AGENT_EXECUTING = "agent.executing"
    AGENT_COMPLETED = "agent.completed"
    TOOL_INVOKED = "tool.invoked"
    STATE_UPDATED = "state.updated"
    PROVIDER_SWITCHED = "provider.switched"


@dataclass
class Event:
    """Event data structure."""

    event_type: EventType
    timestamp: datetime
    source: str
    data: dict[str, Any]
    correlation_id: str | None = None
    trace_id: str | None = None


EventHandler = Callable[[Event], Awaitable[None]]


class EventBus:
    """Unified event bus for system-wide events."""

    def __init__(self) -> None:
        """Initialize event bus."""
        self._handlers: dict[EventType, list[tuple[str, EventHandler]]] = defaultdict(list)
        self._event_history: list[Event] = []
        self._max_history_size = 1000

    def subscribe(self, event_type: EventType, handler: EventHandler) -> str:
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Async function to handle the event

        Returns:
            Subscription ID for unsubscribing
        """
        subscription_id = str(uuid.uuid4())
        self._handlers[event_type].append((subscription_id, handler))
        logger.debug(f"Subscribed to {event_type.value} with ID {subscription_id}")
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: Subscription ID returned from subscribe()

        Returns:
            True if subscription was found and removed, False otherwise
        """
        for event_type, handlers in self._handlers.items():
            for i, (sub_id, _) in enumerate(handlers):
                if sub_id == subscription_id:
                    handlers.pop(i)
                    logger.debug(f"Unsubscribed {subscription_id} from {event_type.value}")
                    return True
        return False

    async def emit(self, event: Event) -> None:
        """
        Emit an event to all subscribers.

        Handlers are invoked concurrently via ``asyncio.gather`` with
        error isolation — a failing handler does not prevent other
        handlers from executing.

        Args:
            event: Event to emit
        """
        # Add to history
        self._event_history.append(event)

        # Trim history if needed
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size :]

        # Get handlers for this event type
        handlers = self._handlers.get(event.event_type, [])

        if not handlers:
            logger.debug(f"No handlers for event {event.event_type.value}")
            return

        # Invoke all handlers concurrently with error isolation
        await asyncio.gather(
            *(self._invoke_handler(sub_id, handler, event) for sub_id, handler in handlers),
        )

    async def _invoke_handler(
        self, subscription_id: str, handler: EventHandler, event: Event
    ) -> None:
        """
        Invoke a handler with error isolation.

        Args:
            subscription_id: Subscription ID
            handler: Handler function
            event: Event to pass to handler
        """
        try:
            await handler(event)
        except Exception as e:
            logger.error(
                f"Error in event handler {subscription_id} for {event.event_type.value}: {e}",
                exc_info=True,
            )

    async def emit_and_wait(self, event: Event, timeout: float = 30.0) -> list[Any]:
        """
        Emit event and wait for all handlers to complete.

        Args:
            event: Event to emit
            timeout: Maximum time to wait for handlers

        Returns:
            List of handler results
        """
        # Add to history
        self._event_history.append(event)

        # Trim history if needed
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size :]

        # Get handlers for this event type
        handlers = self._handlers.get(event.event_type, [])

        if not handlers:
            return []

        # Invoke all handlers and wait for completion
        tasks = [self._invoke_handler_with_result(handler, event) for _, handler in handlers]

        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
            return results
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for handlers of {event.event_type.value}")
            return []

    async def _invoke_handler_with_result(self, handler: EventHandler, event: Event) -> Any:
        """
        Invoke a handler and return its result.

        Args:
            handler: Handler function
            event: Event to pass to handler

        Returns:
            Handler result or None if error
        """
        try:
            return await handler(event)
        except Exception as e:
            logger.error(f"Error in event handler for {event.event_type.value}: {e}")
            return None

    def get_event_history(
        self,
        event_type: EventType | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """
        Get event history with optional filtering.

        Args:
            event_type: Filter by event type
            since: Filter events after this timestamp
            limit: Maximum number of events to return

        Returns:
            List of events matching filters
        """
        events = self._event_history

        # Filter by event type
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]

        # Filter by timestamp
        if since is not None:
            events = [e for e in events if e.timestamp >= since]

        # Apply limit (return most recent)
        if len(events) > limit:
            events = events[-limit:]

        return events

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
        logger.debug("Event history cleared")
