"""Metrics data models and collection."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agentlegatus.core.workflow import WorkflowStatus

if TYPE_CHECKING:
    from agentlegatus.observability.prometheus import PrometheusExporter

# Provider pricing per 1M tokens (input, output) in USD
PROVIDER_PRICING: dict[str, dict[str, float]] = {
    "openai": {"input_per_1m": 2.50, "output_per_1m": 10.00},
    "anthropic": {"input_per_1m": 3.00, "output_per_1m": 15.00},
    "google": {"input_per_1m": 1.25, "output_per_1m": 5.00},
    "aws": {"input_per_1m": 2.00, "output_per_1m": 8.00},
    "azure": {"input_per_1m": 2.50, "output_per_1m": 10.00},
    "default": {"input_per_1m": 2.00, "output_per_1m": 8.00},
}


def calculate_token_cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost based on provider pricing and token counts.

    Args:
        provider: Provider name (looked up in PROVIDER_PRICING)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    pricing = PROVIDER_PRICING.get(provider, PROVIDER_PRICING["default"])
    input_cost = (input_tokens / 1_000_000) * pricing["input_per_1m"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_1m"]
    return input_cost + output_cost


@dataclass
class MetricsData:
    """Individual metric data point."""

    execution_id: str
    timestamp: datetime
    metric_type: str
    value: float
    unit: str
    labels: dict[str, str] = field(default_factory=dict)

    def to_prometheus_format(self) -> str:
        """Convert to Prometheus exposition format.

        Returns:
            Metric in Prometheus format
        """
        labels_str = ",".join([f'{k}="{v}"' for k, v in self.labels.items()])
        if labels_str:
            return f"{self.metric_type}{{{labels_str}}} {self.value}"
        return f"{self.metric_type} {self.value}"

    def to_opentelemetry_format(self) -> dict[str, Any]:
        """Convert to OpenTelemetry format.

        Returns:
            Metric in OpenTelemetry format
        """
        return {
            "name": self.metric_type,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "attributes": {
                "execution_id": self.execution_id,
                **self.labels,
            },
        }


@dataclass
class StepMetrics:
    """Metrics for a single workflow step."""

    step_id: str
    start_time: datetime
    end_time: datetime | None = None
    duration: float | None = None
    status: str = "running"
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    tool_calls: int = 0
    provider: str = ""

    def finalize(self, provider: str = "") -> None:
        """Finalize metrics when step completes.

        Args:
            provider: Provider name for cost calculation
        """
        if self.end_time is None:
            self.end_time = datetime.now()
        if self.duration is None:
            self.duration = (self.end_time - self.start_time).total_seconds()
        if provider:
            self.provider = provider
        if self.provider and self.cost == 0.0:
            self.cost = calculate_token_cost(self.provider, self.input_tokens, self.output_tokens)


@dataclass
class ExecutionMetrics:
    """Metrics for complete workflow execution."""

    execution_id: str
    workflow_id: str
    provider: str
    start_time: datetime
    end_time: datetime | None = None
    duration: float | None = None
    status: WorkflowStatus = WorkflowStatus.RUNNING
    total_cost: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    step_metrics: list[StepMetrics] = field(default_factory=list)
    error_count: int = 0
    retry_count: int = 0

    def calculate_cost(self) -> float:
        """Calculate total execution cost from step metrics.

        Returns:
            Total cost in dollars
        """
        self.total_cost = sum(step.cost for step in self.step_metrics)
        return self.total_cost

    def aggregate_tokens(self) -> dict[str, int]:
        """Aggregate token usage across all steps.

        Returns:
            Dictionary with input, output, and total token counts
        """
        total_input = sum(s.input_tokens for s in self.step_metrics)
        total_output = sum(s.output_tokens for s in self.step_metrics)
        self.token_usage = {
            "input": total_input,
            "output": total_output,
            "total": total_input + total_output,
        }
        return self.token_usage

    def get_summary(self) -> dict[str, Any]:
        """Get execution summary.

        Returns:
            Dictionary with execution summary
        """
        if self.end_time and self.duration is None:
            self.duration = (self.end_time - self.start_time).total_seconds()

        return {
            "execution_id": self.execution_id,
            "workflow_id": self.workflow_id,
            "provider": self.provider,
            "status": self.status.value,
            "duration": self.duration,
            "total_cost": self.total_cost,
            "token_usage": self.token_usage,
            "step_count": len(self.step_metrics),
            "error_count": self.error_count,
            "retry_count": self.retry_count,
        }


@dataclass
class BenchmarkMetrics:
    """Metrics for provider benchmark."""

    provider_name: str
    execution_time: float
    total_cost: float
    token_usage: dict[str, int]
    success_rate: float
    error_count: int
    latency_p50: float
    latency_p95: float
    latency_p99: float
    custom_metrics: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collector for execution metrics.

    Tracks workflow-level and step-level metrics including execution time,
    cost, and token usage. Supports provider-based cost calculation.
    Optionally bridges to a PrometheusExporter for Prometheus exposition format.
    """

    def __init__(
        self,
        prometheus_exporter: PrometheusExporter | None = None,
    ) -> None:
        """Initialize metrics collector.

        Args:
            prometheus_exporter: Optional PrometheusExporter for Prometheus format export.
                If not provided, Prometheus export is not available via this collector.
        """
        self.metrics: dict[str, ExecutionMetrics] = {}
        self.data_points: list[MetricsData] = []
        self._step_timers: dict[str, float] = {}
        self._prometheus: PrometheusExporter | None = prometheus_exporter

    def start_execution(
        self, execution_id: str, workflow_id: str, provider: str
    ) -> ExecutionMetrics:
        """Start tracking metrics for an execution.

        Args:
            execution_id: Unique execution identifier
            workflow_id: Workflow identifier
            provider: Provider name

        Returns:
            ExecutionMetrics instance
        """
        metrics = ExecutionMetrics(
            execution_id=execution_id,
            workflow_id=workflow_id,
            provider=provider,
            start_time=datetime.now(),
        )
        self.metrics[execution_id] = metrics

        self.record_metric(
            execution_id=execution_id,
            metric_type="workflow_started",
            value=1.0,
            unit="count",
            labels={"workflow_id": workflow_id, "provider": provider},
        )

        if self._prometheus:
            self._prometheus.record_workflow_start(workflow_id, provider)

        return metrics

    def end_execution(
        self, execution_id: str, status: WorkflowStatus, error_count: int = 0
    ) -> None:
        """End tracking for an execution.

        Args:
            execution_id: Execution identifier
            status: Final workflow status
            error_count: Number of errors encountered
        """
        if execution_id in self.metrics:
            metrics = self.metrics[execution_id]
            metrics.end_time = datetime.now()
            metrics.duration = (metrics.end_time - metrics.start_time).total_seconds()
            metrics.status = status
            metrics.error_count = error_count
            metrics.calculate_cost()
            metrics.aggregate_tokens()

            self.record_metric(
                execution_id=execution_id,
                metric_type="workflow_duration_seconds",
                value=metrics.duration,
                unit="seconds",
                labels={
                    "workflow_id": metrics.workflow_id,
                    "provider": metrics.provider,
                    "status": status.value,
                },
            )
            self.record_metric(
                execution_id=execution_id,
                metric_type="workflow_cost_dollars",
                value=metrics.total_cost,
                unit="dollars",
                labels={
                    "workflow_id": metrics.workflow_id,
                    "provider": metrics.provider,
                },
            )

            if self._prometheus:
                self._prometheus.record_workflow_end(
                    workflow_id=metrics.workflow_id,
                    provider=metrics.provider,
                    status=status.value,
                    duration=metrics.duration or 0.0,
                    cost=metrics.total_cost,
                )
                if metrics.error_count > 0:
                    self._prometheus.record_error(
                        provider=metrics.provider,
                        error_type="workflow_error",
                    )

    def start_step(self, execution_id: str, step_id: str) -> StepMetrics | None:
        """Start tracking metrics for a step.

        Args:
            execution_id: Execution identifier
            step_id: Step identifier

        Returns:
            StepMetrics instance if execution exists, None otherwise
        """
        timer_key = f"{execution_id}:{step_id}"
        self._step_timers[timer_key] = time.monotonic()

        step_metrics = StepMetrics(
            step_id=step_id,
            start_time=datetime.now(),
        )
        if execution_id in self.metrics:
            exec_metrics = self.metrics[execution_id]
            step_metrics.provider = exec_metrics.provider
            return step_metrics
        return step_metrics

    def end_step(
        self,
        execution_id: str,
        step_metrics: StepMetrics,
        status: str = "completed",
        input_tokens: int = 0,
        output_tokens: int = 0,
        tool_calls: int = 0,
    ) -> None:
        """End tracking for a step and add to execution metrics.

        Args:
            execution_id: Execution identifier
            step_metrics: StepMetrics to finalize
            status: Step completion status
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            tool_calls: Number of tool calls made
        """
        step_metrics.status = status
        step_metrics.input_tokens = input_tokens
        step_metrics.output_tokens = output_tokens
        step_metrics.tool_calls = tool_calls

        provider = ""
        if execution_id in self.metrics:
            provider = self.metrics[execution_id].provider

        step_metrics.finalize(provider=provider)
        self.add_step_metrics(execution_id, step_metrics)

        self.record_metric(
            execution_id=execution_id,
            metric_type="step_duration_seconds",
            value=step_metrics.duration or 0.0,
            unit="seconds",
            labels={
                "step_id": step_metrics.step_id,
                "status": status,
                "provider": provider,
            },
        )
        if input_tokens > 0 or output_tokens > 0:
            self.record_metric(
                execution_id=execution_id,
                metric_type="step_tokens_total",
                value=float(input_tokens + output_tokens),
                unit="tokens",
                labels={
                    "step_id": step_metrics.step_id,
                    "provider": provider,
                },
            )

        # Clean up timer
        timer_key = f"{execution_id}:{step_metrics.step_id}"
        self._step_timers.pop(timer_key, None)

        if self._prometheus and provider:
            self._prometheus.record_step_end(
                step_id=step_metrics.step_id,
                provider=provider,
                status=status,
                duration=step_metrics.duration or 0.0,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

    def add_step_metrics(self, execution_id: str, step_metrics: StepMetrics) -> None:
        """Add step metrics to execution.

        Args:
            execution_id: Execution identifier
            step_metrics: Step metrics to add
        """
        if execution_id in self.metrics:
            self.metrics[execution_id].step_metrics.append(step_metrics)

    def record_metric(
        self,
        execution_id: str,
        metric_type: str,
        value: float,
        unit: str,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a metric data point.

        Args:
            execution_id: Execution identifier
            metric_type: Type of metric
            value: Metric value
            unit: Unit of measurement
            labels: Optional labels for the metric
        """
        metric = MetricsData(
            execution_id=execution_id,
            timestamp=datetime.now(),
            metric_type=metric_type,
            value=value,
            unit=unit,
            labels=labels or {},
        )
        self.data_points.append(metric)

    def get_metrics(self, execution_id: str) -> ExecutionMetrics | None:
        """Get metrics for an execution.

        Args:
            execution_id: Execution identifier

        Returns:
            ExecutionMetrics if found, None otherwise
        """
        return self.metrics.get(execution_id)

    def get_all_data_points(
        self,
        metric_type: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> list[MetricsData]:
        """Get all recorded data points with optional filtering.

        Args:
            metric_type: Filter by metric type
            labels: Filter by label key-value pairs

        Returns:
            List of matching MetricsData
        """
        result = self.data_points
        if metric_type:
            result = [d for d in result if d.metric_type == metric_type]
        if labels:
            result = [d for d in result if all(d.labels.get(k) == v for k, v in labels.items())]
        return result

    def to_prometheus_format(self) -> str:
        """Export all collected metrics in Prometheus exposition format.

        If a PrometheusExporter is attached, delegates to it for proper
        Prometheus client library output. Otherwise, falls back to
        rendering individual MetricsData points.

        Returns:
            Metrics string in Prometheus text exposition format
        """
        if self._prometheus and self._prometheus.enabled:
            return self._prometheus.generate_metrics()

        # Fallback: render data points manually
        lines: list[str] = []
        for dp in self.data_points:
            lines.append(dp.to_prometheus_format())
        return "\n".join(lines) + "\n" if lines else ""
