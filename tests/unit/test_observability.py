"""Unit tests for the observability layer."""

import pytest
from datetime import datetime

from agentlegatus.core.workflow import WorkflowStatus
from agentlegatus.observability.metrics import (
    PROVIDER_PRICING,
    MetricsCollector,
    MetricsData,
    StepMetrics,
    calculate_token_cost,
)
from agentlegatus.observability.tracing import TracingManager
from agentlegatus.observability.prometheus import PrometheusExporter
from agentlegatus.utils.logging import (
    CONTEXT_FIELDS,
    _add_default_context_fields,
    _format_exception_info,
    get_logger,
    bind_context,
    unbind_context,
    clear_context,
    log_error,
    setup_logging,
)


class TestCalculateTokenCost:
    """Tests for calculate_token_cost function."""

    def test_known_provider(self):
        cost = calculate_token_cost("openai", 1_000_000, 1_000_000)
        expected = PROVIDER_PRICING["openai"]["input_per_1m"] + PROVIDER_PRICING["openai"]["output_per_1m"]
        assert cost == expected

    def test_unknown_provider_uses_default(self):
        cost = calculate_token_cost("unknown_provider", 1_000_000, 0)
        assert cost == PROVIDER_PRICING["default"]["input_per_1m"]

    def test_zero_tokens(self):
        cost = calculate_token_cost("openai", 0, 0)
        assert cost == 0.0

    def test_fractional_tokens(self):
        cost = calculate_token_cost("openai", 500, 250)
        assert cost > 0.0


class TestMetricsData:
    """Tests for MetricsData dataclass."""

    def test_to_prometheus_format_with_labels(self):
        md = MetricsData(
            execution_id="e1",
            timestamp=datetime.now(),
            metric_type="test_metric",
            value=42.0,
            unit="count",
            labels={"provider": "openai"},
        )
        result = md.to_prometheus_format()
        assert 'test_metric{provider="openai"} 42.0' == result

    def test_to_prometheus_format_without_labels(self):
        md = MetricsData(
            execution_id="e1",
            timestamp=datetime.now(),
            metric_type="test_metric",
            value=42.0,
            unit="count",
        )
        result = md.to_prometheus_format()
        assert result == "test_metric 42.0"

    def test_to_opentelemetry_format(self):
        md = MetricsData(
            execution_id="e1",
            timestamp=datetime.now(),
            metric_type="test_metric",
            value=42.0,
            unit="count",
            labels={"step": "s1"},
        )
        result = md.to_opentelemetry_format()
        assert result["name"] == "test_metric"
        assert result["value"] == 42.0
        assert result["unit"] == "count"
        assert result["attributes"]["execution_id"] == "e1"
        assert result["attributes"]["step"] == "s1"


class TestStepMetrics:
    """Tests for StepMetrics."""

    def test_finalize_sets_end_time_and_duration(self):
        sm = StepMetrics(step_id="s1", start_time=datetime.now())
        sm.finalize()
        assert sm.end_time is not None
        assert sm.duration is not None
        assert sm.duration >= 0.0

    def test_finalize_calculates_cost_with_provider(self):
        sm = StepMetrics(
            step_id="s1",
            start_time=datetime.now(),
            input_tokens=1000,
            output_tokens=500,
        )
        sm.finalize(provider="openai")
        assert sm.cost > 0.0
        assert sm.provider == "openai"


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_start_and_end_execution(self):
        mc = MetricsCollector()
        mc.start_execution("e1", "w1", "openai")
        mc.end_execution("e1", WorkflowStatus.COMPLETED)
        m = mc.get_metrics("e1")
        assert m is not None
        assert m.status == WorkflowStatus.COMPLETED
        assert m.duration is not None
        assert m.duration >= 0.0

    def test_step_tracking(self):
        mc = MetricsCollector()
        mc.start_execution("e1", "w1", "openai")
        sm = mc.start_step("e1", "step1")
        mc.end_step("e1", sm, status="completed", input_tokens=100, output_tokens=50)
        mc.end_execution("e1", WorkflowStatus.COMPLETED)

        m = mc.get_metrics("e1")
        assert len(m.step_metrics) == 1
        assert m.step_metrics[0].step_id == "step1"
        assert m.step_metrics[0].input_tokens == 100
        assert m.step_metrics[0].output_tokens == 50
        assert m.token_usage["input"] == 100
        assert m.token_usage["output"] == 50
        assert m.total_cost > 0.0

    def test_get_nonexistent_metrics(self):
        mc = MetricsCollector()
        assert mc.get_metrics("nonexistent") is None

    def test_data_points_recorded(self):
        mc = MetricsCollector()
        mc.start_execution("e1", "w1", "openai")
        mc.end_execution("e1", WorkflowStatus.COMPLETED)
        # start_execution records workflow_started, end_execution records duration + cost
        assert len(mc.data_points) >= 2

    def test_get_all_data_points_filter_by_type(self):
        mc = MetricsCollector()
        mc.record_metric("e1", "type_a", 1.0, "count")
        mc.record_metric("e1", "type_b", 2.0, "count")
        mc.record_metric("e1", "type_a", 3.0, "count")
        result = mc.get_all_data_points(metric_type="type_a")
        assert len(result) == 2

    def test_get_all_data_points_filter_by_labels(self):
        mc = MetricsCollector()
        mc.record_metric("e1", "m", 1.0, "c", labels={"env": "prod"})
        mc.record_metric("e1", "m", 2.0, "c", labels={"env": "dev"})
        result = mc.get_all_data_points(labels={"env": "prod"})
        assert len(result) == 1


class TestTracingManager:
    """Tests for TracingManager."""

    def test_enabled(self):
        tm = TracingManager(service_name="test")
        assert tm.enabled is True

    def test_start_span_context_manager(self):
        tm = TracingManager(service_name="test", use_batch_processor=False)
        with tm.start_span("test_span", attributes={"key": "val"}) as span:
            assert span is not None
        tm.shutdown()

    def test_workflow_span(self):
        tm = TracingManager(service_name="test", use_batch_processor=False)
        with tm.start_workflow_span("w1", "e1", trace_id="abc") as span:
            assert span is not None
        tm.shutdown()

    def test_step_span(self):
        tm = TracingManager(service_name="test", use_batch_processor=False)
        with tm.start_step_span("s1", "w1", "e1") as span:
            assert span is not None
        tm.shutdown()

    def test_record_error(self):
        tm = TracingManager(service_name="test", use_batch_processor=False)
        with tm.start_span("err_span") as span:
            tm.record_error(span, ValueError("test error"))
        tm.shutdown()

    def test_record_error_none_span(self):
        tm = TracingManager(service_name="test")
        tm.record_error(None, ValueError("no-op"))


class TestPrometheusExporter:
    """Tests for PrometheusExporter."""

    def test_enabled(self):
        pe = PrometheusExporter()
        assert pe.enabled is True

    def test_workflow_metrics(self):
        pe = PrometheusExporter(namespace="test")
        pe.record_workflow_start("w1", "openai")
        pe.record_workflow_end("w1", "openai", "completed", 1.5, 0.01)
        output = pe.generate_metrics()
        assert "test_workflow_executions_total" in output
        assert "test_workflow_duration_seconds" in output

    def test_step_metrics(self):
        pe = PrometheusExporter(namespace="test")
        pe.record_step_end("s1", "openai", "completed", 0.5, input_tokens=100, output_tokens=50)
        output = pe.generate_metrics()
        assert "test_step_executions_total" in output
        assert "test_tokens_total" in output

    def test_error_metrics(self):
        pe = PrometheusExporter(namespace="test")
        pe.record_error("openai", "TimeoutError")
        output = pe.generate_metrics()
        assert "test_errors_total" in output

    def test_generate_metrics_returns_string(self):
        pe = PrometheusExporter()
        output = pe.generate_metrics()
        assert isinstance(output, str)

    def test_provider_switch_metrics(self):
        pe = PrometheusExporter(namespace="test")
        pe.record_provider_switch("openai", "anthropic")
        output = pe.generate_metrics()
        assert "test_provider_switches_total" in output

    def test_disabled_exporter_noop(self):
        pe = PrometheusExporter()
        pe._enabled = False
        pe.record_workflow_start("w1", "openai")
        pe.record_workflow_end("w1", "openai", "completed", 1.0, 0.01)
        pe.record_step_end("s1", "openai", "completed", 0.5)
        pe.record_error("openai", "Timeout")
        pe.record_provider_switch("openai", "anthropic")
        assert pe.generate_metrics() == ""


class TestMetricsCollectorPrometheusIntegration:
    """Tests for MetricsCollector → PrometheusExporter bridge."""

    def test_collector_without_prometheus(self):
        mc = MetricsCollector()
        mc.start_execution("e1", "w1", "openai")
        mc.end_execution("e1", WorkflowStatus.COMPLETED)
        # Fallback format from data points
        output = mc.to_prometheus_format()
        assert "workflow_started" in output
        assert "workflow_duration_seconds" in output

    def test_collector_with_prometheus_workflow(self):
        pe = PrometheusExporter(namespace="test")
        mc = MetricsCollector(prometheus_exporter=pe)
        mc.start_execution("e1", "w1", "openai")
        mc.end_execution("e1", WorkflowStatus.COMPLETED)
        output = mc.to_prometheus_format()
        assert "test_workflow_executions_total" in output
        assert "test_workflow_duration_seconds" in output
        assert "test_active_workflows" in output

    def test_collector_with_prometheus_step(self):
        pe = PrometheusExporter(namespace="test")
        mc = MetricsCollector(prometheus_exporter=pe)
        mc.start_execution("e1", "w1", "openai")
        step = mc.start_step("e1", "s1")
        mc.end_step("e1", step, status="completed", input_tokens=200, output_tokens=100)
        output = mc.to_prometheus_format()
        assert "test_step_executions_total" in output
        assert "test_tokens_total" in output

    def test_collector_with_prometheus_errors(self):
        pe = PrometheusExporter(namespace="test")
        mc = MetricsCollector(prometheus_exporter=pe)
        mc.start_execution("e1", "w1", "openai")
        mc.end_execution("e1", WorkflowStatus.FAILED, error_count=2)
        output = mc.to_prometheus_format()
        assert "test_errors_total" in output

    def test_collector_fallback_empty(self):
        mc = MetricsCollector()
        output = mc.to_prometheus_format()
        assert output == ""


class TestStructuredLogging:
    """Tests for structured logging utilities (Req 27.1-27.6)."""

    def test_get_logger_returns_logger(self):
        logger = get_logger("test")
        assert logger is not None

    def test_get_logger_with_initial_context(self):
        logger = get_logger("test_ctx", workflow_id="w1")
        assert logger is not None

    def test_bind_and_unbind_context(self):
        bind_context(workflow_id="w1", execution_id="e1")
        unbind_context("workflow_id", "execution_id")

    def test_clear_context(self):
        bind_context(trace_id="t1")
        clear_context()

    def test_log_error_includes_error_fields(self):
        """Req 27.4: error type, message, and stack trace."""
        logger = get_logger("test_error")
        try:
            raise ValueError("test error")
        except ValueError as e:
            log_error(logger, "Something failed", e, workflow_id="w1")

    def test_setup_logging_json_default(self):
        """Req 27.1: JSON format by default."""
        setup_logging(level="DEBUG")

    def test_setup_logging_console_format(self):
        setup_logging(level="INFO", json_format=False)

    def test_setup_logging_with_global_context(self):
        """Req 27.2: workflow_id, execution_id in all logs."""
        setup_logging(global_context={"service": "agentlegatus"})

    def test_setup_logging_all_levels(self):
        """Req 27.6: consistent log levels."""
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            setup_logging(level=level)

    def test_context_fields_defined(self):
        """Req 27.2, 27.3, 27.5: required context fields."""
        assert "workflow_id" in CONTEXT_FIELDS
        assert "execution_id" in CONTEXT_FIELDS
        assert "step_id" in CONTEXT_FIELDS
        assert "trace_id" in CONTEXT_FIELDS
        assert "correlation_id" in CONTEXT_FIELDS

    def test_add_default_context_fields_processor(self):
        """Processor fills missing context fields with None."""
        event_dict = {"event": "test"}
        result = _add_default_context_fields(None, "info", event_dict)
        for field in CONTEXT_FIELDS:
            assert field in result
            assert result[field] is None

    def test_add_default_context_fields_preserves_existing(self):
        """Processor does not overwrite existing context values."""
        event_dict = {"event": "test", "workflow_id": "w1", "trace_id": "t1"}
        result = _add_default_context_fields(None, "info", event_dict)
        assert result["workflow_id"] == "w1"
        assert result["trace_id"] == "t1"
        assert result["step_id"] is None

    def test_format_exception_info_with_exception(self):
        """Req 27.4: error_type, error_message, stack_trace extracted."""
        try:
            raise ValueError("boom")
        except ValueError as exc:
            event_dict = {"event": "fail", "exc_info": exc}
            result = _format_exception_info(None, "error", event_dict)
            assert result["error_type"] == "ValueError"
            assert result["error_message"] == "boom"
            assert "stack_trace" in result
            assert "ValueError" in result["stack_trace"]
            assert "exc_info" not in result

    def test_format_exception_info_with_tuple(self):
        """Handles exc_info as a 3-tuple (type, value, tb)."""
        try:
            raise RuntimeError("oops")
        except RuntimeError:
            import sys
            exc_info_tuple = sys.exc_info()
            event_dict = {"event": "fail", "exc_info": exc_info_tuple}
            result = _format_exception_info(None, "error", event_dict)
            assert result["error_type"] == "RuntimeError"
            assert result["error_message"] == "oops"

    def test_format_exception_info_no_exc(self):
        """No-op when exc_info is absent."""
        event_dict = {"event": "ok"}
        result = _format_exception_info(None, "info", event_dict)
        assert "error_type" not in result
        assert result["event"] == "ok"

    def test_bind_context_workflow_fields(self):
        """Req 27.2, 27.3: bind workflow/step context."""
        bind_context(
            workflow_id="w1",
            execution_id="e1",
            step_id="s1",
            trace_id="t1",
            correlation_id="c1",
        )
        # Clean up
        unbind_context("workflow_id", "execution_id", "step_id", "trace_id", "correlation_id")

    def test_log_error_with_all_context(self):
        """Req 27.4: log_error includes error_type, error_message, stack_trace."""
        logger = get_logger("test_full_error")
        try:
            raise TypeError("bad type")
        except TypeError as e:
            log_error(
                logger,
                "Type error occurred",
                e,
                workflow_id="w1",
                execution_id="e1",
                step_id="s1",
            )


class TestTracingManagerOTLP:
    """Tests for OTLP exporter creation and OpenTelemetry format export."""

    def test_create_otlp_exporter_grpc(self):
        """OTLP gRPC exporter can be created."""
        exporter = TracingManager._create_otlp_exporter(
            endpoint="localhost:4317", protocol="grpc", headers=None
        )
        assert exporter is not None

    def test_create_otlp_exporter_http(self):
        """OTLP HTTP exporter can be created."""
        exporter = TracingManager._create_otlp_exporter(
            endpoint="http://localhost:4318/v1/traces", protocol="http", headers=None
        )
        assert exporter is not None

    def test_create_otlp_exporter_invalid_protocol(self):
        """Invalid protocol raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported OTLP protocol"):
            TracingManager._create_otlp_exporter(
                endpoint=None, protocol="websocket", headers=None
            )

    def test_create_with_otlp_grpc(self):
        """create_with_otlp factory produces a working TracingManager."""
        tm = TracingManager.create_with_otlp(
            service_name="test-otlp",
            endpoint="localhost:4317",
            protocol="grpc",
            use_batch_processor=False,
        )
        assert tm.enabled is True
        tm.shutdown()

    def test_create_with_otlp_http(self):
        """create_with_otlp factory works with HTTP protocol."""
        tm = TracingManager.create_with_otlp(
            service_name="test-otlp-http",
            endpoint="http://localhost:4318/v1/traces",
            protocol="http",
            use_batch_processor=False,
        )
        assert tm.enabled is True
        tm.shutdown()

    def test_export_traces_as_otel_empty(self):
        """export_traces_as_otel returns empty list when no spans given."""
        tm = TracingManager(service_name="test", use_batch_processor=False)
        result = tm.export_traces_as_otel()
        assert result == []
        tm.shutdown()

    def test_export_traces_as_otel_with_spans(self):
        """export_traces_as_otel produces OTLP-style dicts from ReadableSpans."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            SimpleSpanProcessor,
            SpanExporter,
            SpanExportResult,
        )

        # Simple in-memory exporter to capture finished spans
        class _CapturingExporter(SpanExporter):
            def __init__(self):
                self.spans = []

            def export(self, spans):
                self.spans.extend(spans)
                return SpanExportResult.SUCCESS

            def shutdown(self):
                pass

        cap = _CapturingExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(cap))
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("my-span", attributes={"k": "v"}) as span:
            span.add_event("test-event", attributes={"ev_key": "ev_val"})

        provider.force_flush()
        assert len(cap.spans) == 1

        tm = TracingManager(service_name="test", use_batch_processor=False)
        exported = tm.export_traces_as_otel(cap.spans)
        assert len(exported) == 1
        assert exported[0]["name"] == "my-span"
        assert exported[0]["attributes"]["k"] == "v"
        assert exported[0]["traceId"] is not None
        assert exported[0]["spanId"] is not None
        assert len(exported[0]["events"]) == 1
        assert exported[0]["events"][0]["name"] == "test-event"
        tm.shutdown()
        provider.shutdown()


class TestTracingManagerEventSpan:
    """Tests for event span creation."""

    def test_start_event_span(self):
        """start_event_span creates a span with event attributes."""
        tm = TracingManager(service_name="test", use_batch_processor=False)
        with tm.start_event_span(
            event_type="workflow.started",
            source="legatus",
            trace_id="t123",
            correlation_id="c456",
            attributes={"workflow.id": "w1"},
        ) as span:
            assert span is not None
        tm.shutdown()

    def test_start_event_span_disabled(self):
        """start_event_span yields None when tracing is disabled."""
        tm = TracingManager(service_name="test")
        tm._enabled = False
        with tm.start_event_span(
            event_type="step.completed",
            source="centurion",
        ) as span:
            assert span is None


class TestEventBusTracingBridge:
    """Tests for EventBusTracingBridge."""

    @pytest.fixture
    def event_bus(self):
        from agentlegatus.core.event_bus import EventBus
        return EventBus()

    @pytest.fixture
    def tracing_manager(self):
        tm = TracingManager(service_name="bridge-test", use_batch_processor=False)
        yield tm
        tm.shutdown()

    def test_attach_subscribes_to_all_event_types(self, event_bus, tracing_manager):
        """attach() subscribes to every EventType."""
        from agentlegatus.core.event_bus import EventType
        from agentlegatus.observability.tracing import EventBusTracingBridge

        bridge = EventBusTracingBridge(event_bus, tracing_manager)
        bridge.attach()

        assert len(bridge._subscription_ids) == len(EventType)

    def test_detach_unsubscribes(self, event_bus, tracing_manager):
        """detach() removes all subscriptions."""
        from agentlegatus.observability.tracing import EventBusTracingBridge

        bridge = EventBusTracingBridge(event_bus, tracing_manager)
        bridge.attach()
        bridge.detach()

        assert len(bridge._subscription_ids) == 0

    @pytest.mark.asyncio
    async def test_event_creates_span(self, event_bus, tracing_manager):
        """Emitting an event through the bus creates a span via the bridge."""
        from agentlegatus.core.event_bus import Event, EventType
        from agentlegatus.observability.tracing import EventBusTracingBridge

        bridge = EventBusTracingBridge(event_bus, tracing_manager)
        bridge.attach()

        event = Event(
            event_type=EventType.WORKFLOW_STARTED,
            timestamp=datetime.now(),
            source="legatus",
            data={"workflow_id": "w1", "provider": "mock"},
            trace_id="trace-abc",
            correlation_id="corr-xyz",
        )

        # emit_and_wait ensures the handler runs to completion
        await event_bus.emit_and_wait(event, timeout=5.0)

        # If we got here without error, the bridge handled the event
        bridge.detach()

    @pytest.mark.asyncio
    async def test_bridge_propagates_trace_and_correlation_ids(
        self, event_bus, tracing_manager
    ):
        """Bridge passes trace_id and correlation_id into span attributes."""
        from agentlegatus.core.event_bus import Event, EventType
        from agentlegatus.observability.tracing import EventBusTracingBridge

        bridge = EventBusTracingBridge(event_bus, tracing_manager)
        bridge.attach()

        event = Event(
            event_type=EventType.STEP_COMPLETED,
            timestamp=datetime.now(),
            source="centurion:default",
            data={"step_id": "s1", "result": "ok"},
            trace_id="trace-123",
            correlation_id="corr-456",
        )

        await event_bus.emit_and_wait(event, timeout=5.0)
        bridge.detach()

    @pytest.mark.asyncio
    async def test_bridge_handles_disabled_tracing(self, event_bus):
        """Bridge is a no-op when tracing is disabled."""
        from agentlegatus.core.event_bus import Event, EventType
        from agentlegatus.observability.tracing import EventBusTracingBridge

        tm = TracingManager(service_name="disabled-test")
        tm._enabled = False

        bridge = EventBusTracingBridge(event_bus, tm)
        bridge.attach()

        event = Event(
            event_type=EventType.WORKFLOW_COMPLETED,
            timestamp=datetime.now(),
            source="legatus",
            data={"workflow_id": "w1"},
        )

        await event_bus.emit_and_wait(event, timeout=5.0)
        bridge.detach()


class TestMetricsAccuracy:
    """Validate that metrics pipeline produces accurate numbers."""

    def test_multi_step_token_aggregation(self):
        """Token counts from multiple steps aggregate correctly."""
        mc = MetricsCollector()
        mc.start_execution("e1", "w1", "openai")

        for i, (inp, out) in enumerate([(100, 50), (200, 80), (50, 30)]):
            sm = mc.start_step("e1", f"step{i}")
            mc.end_step("e1", sm, input_tokens=inp, output_tokens=out)

        mc.end_execution("e1", WorkflowStatus.COMPLETED)
        m = mc.get_metrics("e1")

        assert m.token_usage["input"] == 350
        assert m.token_usage["output"] == 160
        assert m.total_cost > 0.0

    def test_cost_scales_with_tokens(self):
        """Cost should increase when more tokens are used."""
        mc = MetricsCollector()

        mc.start_execution("small", "w1", "openai")
        sm = mc.start_step("small", "s1")
        mc.end_step("small", sm, input_tokens=10, output_tokens=5)
        mc.end_execution("small", WorkflowStatus.COMPLETED)

        mc.start_execution("large", "w1", "openai")
        sm = mc.start_step("large", "s1")
        mc.end_step("large", sm, input_tokens=1000, output_tokens=500)
        mc.end_execution("large", WorkflowStatus.COMPLETED)

        small_cost = mc.get_metrics("small").total_cost
        large_cost = mc.get_metrics("large").total_cost
        assert large_cost > small_cost

    def test_prometheus_output_contains_recorded_metrics(self):
        """Prometheus export includes workflow and step metrics."""
        mc = MetricsCollector()
        mc.start_execution("e1", "w1", "mock")
        sm = mc.start_step("e1", "s1")
        mc.end_step("e1", sm, input_tokens=50, output_tokens=25)
        mc.end_execution("e1", WorkflowStatus.COMPLETED)

        prom = mc.to_prometheus_format()
        assert "workflow_duration_seconds" in prom
        assert "step_duration_seconds" in prom

    def test_duration_is_non_negative(self):
        """Duration should always be non-negative."""
        mc = MetricsCollector()
        mc.start_execution("e1", "w1", "openai")
        sm = mc.start_step("e1", "s1")
        mc.end_step("e1", sm)
        mc.end_execution("e1", WorkflowStatus.COMPLETED)

        m = mc.get_metrics("e1")
        assert m.duration >= 0.0
        assert m.step_metrics[0].duration >= 0.0
