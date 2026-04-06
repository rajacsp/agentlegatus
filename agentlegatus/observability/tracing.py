"""OpenTelemetry integration for distributed tracing.

Provides span creation, trace propagation, and trace export.
Gracefully degrades when opentelemetry packages are not installed.

Requirements: 16.7, 16.8
"""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.trace import StatusCode

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as OTLPGrpcExporter,
    )

    _HAS_OTLP_GRPC = True
except ImportError:
    _HAS_OTLP_GRPC = False

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as OTLPHttpExporter,
    )

    _HAS_OTLP_HTTP = True
except ImportError:
    _HAS_OTLP_HTTP = False


class TracingManager:
    """Manages OpenTelemetry tracing for AgentLegatus.

    When OpenTelemetry is not installed, all operations are no-ops.
    """

    def __init__(
        self,
        service_name: str = "agentlegatus",
        exporter: Any | None = None,
        use_batch_processor: bool = True,
    ) -> None:
        """Initialize tracing manager.

        Args:
            service_name: Service name for traces
            exporter: Optional SpanExporter instance. Defaults to ConsoleSpanExporter.
            use_batch_processor: Use BatchSpanProcessor (True) or SimpleSpanProcessor (False)
        """
        self.service_name = service_name
        self._enabled = _HAS_OTEL
        self._provider: Any | None = None
        self._tracer: Any | None = None

        if self._enabled:
            self._setup_provider(exporter, use_batch_processor)

    @classmethod
    def create_with_otlp(
        cls,
        service_name: str = "agentlegatus",
        endpoint: str | None = None,
        protocol: str = "grpc",
        use_batch_processor: bool = True,
        headers: dict[str, str] | None = None,
    ) -> "TracingManager":
        """Create a TracingManager with OTLP exporter.

        Args:
            service_name: Service name for traces
            endpoint: OTLP collector endpoint. Defaults to localhost:4317 (grpc)
                      or localhost:4318 (http).
            protocol: Export protocol — "grpc" or "http"
            use_batch_processor: Use BatchSpanProcessor (True) or SimpleSpanProcessor
            headers: Optional headers for the OTLP exporter

        Returns:
            Configured TracingManager instance

        Raises:
            ImportError: If the required OTLP exporter package is not installed
        """
        exporter = cls._create_otlp_exporter(endpoint, protocol, headers)
        return cls(
            service_name=service_name,
            exporter=exporter,
            use_batch_processor=use_batch_processor,
        )

    @staticmethod
    def _create_otlp_exporter(
        endpoint: str | None,
        protocol: str,
        headers: dict[str, str] | None,
    ) -> Any:
        """Create an OTLP exporter based on protocol.

        Args:
            endpoint: Collector endpoint
            protocol: "grpc" or "http"
            headers: Optional headers

        Returns:
            OTLP SpanExporter instance

        Raises:
            ImportError: If the required package is not installed
            ValueError: If protocol is not supported
        """
        kwargs: dict[str, Any] = {}
        if endpoint:
            kwargs["endpoint"] = endpoint
        if headers:
            kwargs["headers"] = headers

        if protocol == "grpc":
            if not _HAS_OTLP_GRPC:
                raise ImportError(
                    "opentelemetry-exporter-otlp-proto-grpc is required for gRPC export. "
                    "Install with: pip install opentelemetry-exporter-otlp-proto-grpc"
                )
            return OTLPGrpcExporter(**kwargs)

        if protocol == "http":
            if not _HAS_OTLP_HTTP:
                raise ImportError(
                    "opentelemetry-exporter-otlp-proto-http is required for HTTP export. "
                    "Install with: pip install opentelemetry-exporter-otlp-proto-http"
                )
            return OTLPHttpExporter(**kwargs)

        raise ValueError(f"Unsupported OTLP protocol: {protocol}. Use 'grpc' or 'http'.")

    def _setup_provider(self, exporter: Any | None, use_batch_processor: bool) -> None:
        """Set up TracerProvider with exporter.

        Args:
            exporter: SpanExporter instance
            use_batch_processor: Whether to use batch processing
        """
        resource = Resource.create({"service.name": self.service_name})
        self._provider = TracerProvider(resource=resource)

        if exporter is None:
            exporter = ConsoleSpanExporter()

        if use_batch_processor:
            processor = BatchSpanProcessor(exporter)
        else:
            processor = SimpleSpanProcessor(exporter)

        self._provider.add_span_processor(processor)
        trace.set_tracer_provider(self._provider)
        self._tracer = trace.get_tracer(self.service_name)

    @property
    def enabled(self) -> bool:
        """Whether OpenTelemetry tracing is enabled."""
        return self._enabled

    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        trace_id: str | None = None,
        correlation_id: str | None = None,
    ) -> Generator[Any | None, None, None]:
        """Start a new trace span.

        Args:
            name: Span name
            attributes: Optional span attributes
            trace_id: Optional trace ID for correlation
            correlation_id: Optional correlation ID

        Yields:
            The span object, or None if tracing is disabled
        """
        if not self._enabled or self._tracer is None:
            yield None
            return

        span_attributes = attributes or {}
        if trace_id:
            span_attributes["trace_id"] = trace_id
        if correlation_id:
            span_attributes["correlation_id"] = correlation_id

        with self._tracer.start_as_current_span(name, attributes=span_attributes) as span:
            yield span

    def start_workflow_span(
        self,
        workflow_id: str,
        execution_id: str,
        trace_id: str | None = None,
    ) -> Any:
        """Start a span for workflow execution.

        Args:
            workflow_id: Workflow identifier
            execution_id: Execution identifier
            trace_id: Optional trace ID

        Returns:
            Context manager yielding the span
        """
        return self.start_span(
            name=f"workflow.{workflow_id}",
            attributes={
                "workflow.id": workflow_id,
                "execution.id": execution_id,
            },
            trace_id=trace_id,
        )

    def start_step_span(
        self,
        step_id: str,
        workflow_id: str,
        execution_id: str,
        trace_id: str | None = None,
    ) -> Any:
        """Start a span for step execution.

        Args:
            step_id: Step identifier
            workflow_id: Workflow identifier
            execution_id: Execution identifier
            trace_id: Optional trace ID

        Returns:
            Context manager yielding the span
        """
        return self.start_span(
            name=f"step.{step_id}",
            attributes={
                "step.id": step_id,
                "workflow.id": workflow_id,
                "execution.id": execution_id,
            },
            trace_id=trace_id,
        )

    def start_event_span(
        self,
        event_type: str,
        source: str,
        trace_id: str | None = None,
        correlation_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Any:
        """Start a span for an EventBus event.

        Args:
            event_type: Event type value string
            source: Event source
            trace_id: Trace ID from the event
            correlation_id: Correlation ID from the event
            attributes: Additional span attributes

        Returns:
            Context manager yielding the span
        """
        span_attrs: dict[str, Any] = {
            "event.type": event_type,
            "event.source": source,
        }
        if attributes:
            span_attrs.update(attributes)

        return self.start_span(
            name=f"event.{event_type}",
            attributes=span_attrs,
            trace_id=trace_id,
            correlation_id=correlation_id,
        )

    def record_error(self, span: Any | None, error: Exception) -> None:
        """Record an error on a span.

        Args:
            span: The span to record the error on
            error: The exception to record
        """
        if not self._enabled or span is None:
            return

        span.set_status(StatusCode.ERROR, str(error))
        span.record_exception(error)

    def shutdown(self) -> None:
        """Shut down the tracer provider, flushing pending spans."""
        if self._enabled and self._provider is not None:
            self._provider.shutdown()

    def get_current_trace_id(self) -> str | None:
        """Get the current trace ID from the active span.

        Returns:
            Hex trace ID string, or None if no active span
        """
        if not self._enabled:
            return None

        current_span = trace.get_current_span()
        ctx = current_span.get_span_context()
        if ctx and ctx.trace_id:
            return format(ctx.trace_id, "032x")
        return None

    def get_current_span_id(self) -> str | None:
        """Get the current span ID from the active span.

        Returns:
            Hex span ID string, or None if no active span
        """
        if not self._enabled:
            return None

        current_span = trace.get_current_span()
        ctx = current_span.get_span_context()
        if ctx and ctx.span_id:
            return format(ctx.span_id, "016x")
        return None

    def export_traces_as_otel(self, spans: list[Any] | None = None) -> list[dict[str, Any]]:
        """Export trace data in OpenTelemetry-compatible dict format.

        This produces a list of span dictionaries following the OTLP JSON
        structure, useful for serialisation or forwarding to collectors
        that accept JSON payloads.

        Args:
            spans: Optional list of ReadableSpan objects. If None, returns
                   an empty list (spans are normally exported via the
                   configured SpanProcessor/Exporter pipeline).

        Returns:
            List of span dictionaries in OpenTelemetry format
        """
        if not self._enabled or not spans:
            return []

        exported: list[dict[str, Any]] = []
        for span in spans:
            ctx = span.get_span_context()
            exported.append(
                {
                    "traceId": format(ctx.trace_id, "032x") if ctx else None,
                    "spanId": format(ctx.span_id, "016x") if ctx else None,
                    "name": span.name,
                    "kind": span.kind.name if hasattr(span, "kind") else "INTERNAL",
                    "startTimeUnixNano": span.start_time,
                    "endTimeUnixNano": span.end_time,
                    "attributes": dict(span.attributes) if span.attributes else {},
                    "status": {
                        "code": span.status.status_code.name if span.status else "UNSET",
                        "message": span.status.description if span.status else "",
                    },
                    "events": [
                        {
                            "name": evt.name,
                            "timeUnixNano": evt.timestamp,
                            "attributes": dict(evt.attributes) if evt.attributes else {},
                        }
                        for evt in (span.events or [])
                    ],
                }
            )
        return exported


class EventBusTracingBridge:
    """Bridges EventBus events to OpenTelemetry spans.

    Subscribes to all EventBus event types and creates corresponding
    OpenTelemetry spans, propagating trace_id and correlation_id from
    each event into span attributes.

    Requirements: 16.7, 16.8
    """

    def __init__(
        self,
        event_bus: Any,
        tracing_manager: TracingManager,
    ) -> None:
        """Initialize the bridge.

        Args:
            event_bus: EventBus instance to subscribe to
            tracing_manager: TracingManager for creating spans
        """
        self._event_bus = event_bus
        self._tracing = tracing_manager
        self._subscription_ids: list[str] = []
        self._active_spans: dict[str, Any] = {}

    def attach(self) -> None:
        """Subscribe to all event types on the EventBus.

        Call this once after construction to start bridging events to spans.
        """
        from agentlegatus.core.event_bus import EventType

        for event_type in EventType:
            sub_id = self._event_bus.subscribe(event_type, self._on_event)
            self._subscription_ids.append(sub_id)

    def detach(self) -> None:
        """Unsubscribe from all event types."""
        for sub_id in self._subscription_ids:
            self._event_bus.unsubscribe(sub_id)
        self._subscription_ids.clear()

    async def _on_event(self, event: Any) -> None:
        """Handle an EventBus event by creating an OpenTelemetry span.

        The span carries the event's trace_id and correlation_id as
        attributes, ensuring distributed trace context is propagated.

        Args:
            event: Event instance from the EventBus
        """
        if not self._tracing.enabled:
            return

        event_type_value = event.event_type.value
        data = event.data or {}

        # Build attributes from event data (only string/int/float/bool values)
        attributes: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, (str, int, float, bool)):
                attributes[f"event.data.{key}"] = value

        attributes["event.timestamp"] = event.timestamp.isoformat()

        with self._tracing.start_event_span(
            event_type=event_type_value,
            source=event.source,
            trace_id=event.trace_id,
            correlation_id=event.correlation_id,
            attributes=attributes,
        ):
            pass  # Span is created and immediately closed — records the event
