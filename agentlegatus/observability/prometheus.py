"""Prometheus metrics export for AgentLegatus.

Provides Prometheus-compatible metrics using prometheus_client when available.
Gracefully degrades when prometheus-client is not installed.
"""

from typing import Any

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False


class PrometheusExporter:
    """Exports metrics in Prometheus exposition format.

    When prometheus-client is not installed, all operations are no-ops
    and generate_metrics() returns an empty string.
    """

    def __init__(self, namespace: str = "agentlegatus") -> None:
        """Initialize Prometheus exporter.

        Args:
            namespace: Metric namespace prefix
        """
        self._enabled = _HAS_PROMETHEUS
        self._namespace = namespace
        self._registry: Any | None = None

        if self._enabled:
            self._setup_metrics()

    def _setup_metrics(self) -> None:
        """Set up Prometheus metric instruments."""
        self._registry = CollectorRegistry()

        self._workflow_total = Counter(
            f"{self._namespace}_workflow_executions_total",
            "Total number of workflow executions",
            labelnames=["workflow_id", "provider", "status"],
            registry=self._registry,
        )

        self._workflow_duration = Histogram(
            f"{self._namespace}_workflow_duration_seconds",
            "Workflow execution duration in seconds",
            labelnames=["workflow_id", "provider"],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
            registry=self._registry,
        )

        self._workflow_cost = Counter(
            f"{self._namespace}_workflow_cost_dollars_total",
            "Total workflow cost in dollars",
            labelnames=["workflow_id", "provider"],
            registry=self._registry,
        )

        self._step_total = Counter(
            f"{self._namespace}_step_executions_total",
            "Total number of step executions",
            labelnames=["step_id", "provider", "status"],
            registry=self._registry,
        )

        self._step_duration = Histogram(
            f"{self._namespace}_step_duration_seconds",
            "Step execution duration in seconds",
            labelnames=["step_id", "provider"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
            registry=self._registry,
        )

        self._tokens_total = Counter(
            f"{self._namespace}_tokens_total",
            "Total tokens used",
            labelnames=["provider", "direction"],
            registry=self._registry,
        )

        self._active_workflows = Gauge(
            f"{self._namespace}_active_workflows",
            "Number of currently active workflows",
            labelnames=["provider"],
            registry=self._registry,
        )

        self._errors_total = Counter(
            f"{self._namespace}_errors_total",
            "Total number of errors",
            labelnames=["provider", "error_type"],
            registry=self._registry,
        )

        self._provider_switches_total = Counter(
            f"{self._namespace}_provider_switches_total",
            "Total number of provider switches",
            labelnames=["from_provider", "to_provider"],
            registry=self._registry,
        )

    @property
    def enabled(self) -> bool:
        """Whether Prometheus export is enabled."""
        return self._enabled

    def record_workflow_start(self, workflow_id: str, provider: str) -> None:
        """Record a workflow execution start.

        Args:
            workflow_id: Workflow identifier
            provider: Provider name
        """
        if not self._enabled:
            return
        self._active_workflows.labels(provider=provider).inc()

    def record_workflow_end(
        self,
        workflow_id: str,
        provider: str,
        status: str,
        duration: float,
        cost: float,
    ) -> None:
        """Record a workflow execution completion.

        Args:
            workflow_id: Workflow identifier
            provider: Provider name
            status: Final status (completed, failed, cancelled)
            duration: Execution duration in seconds
            cost: Total cost in dollars
        """
        if not self._enabled:
            return
        self._workflow_total.labels(workflow_id=workflow_id, provider=provider, status=status).inc()
        self._workflow_duration.labels(workflow_id=workflow_id, provider=provider).observe(duration)
        self._workflow_cost.labels(workflow_id=workflow_id, provider=provider).inc(cost)
        self._active_workflows.labels(provider=provider).dec()

    def record_step_end(
        self,
        step_id: str,
        provider: str,
        status: str,
        duration: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record a step execution completion.

        Args:
            step_id: Step identifier
            provider: Provider name
            status: Step status
            duration: Step duration in seconds
            input_tokens: Input tokens used
            output_tokens: Output tokens used
        """
        if not self._enabled:
            return
        self._step_total.labels(step_id=step_id, provider=provider, status=status).inc()
        self._step_duration.labels(step_id=step_id, provider=provider).observe(duration)

        if input_tokens > 0:
            self._tokens_total.labels(provider=provider, direction="input").inc(input_tokens)
        if output_tokens > 0:
            self._tokens_total.labels(provider=provider, direction="output").inc(output_tokens)

    def record_error(self, provider: str, error_type: str) -> None:
        """Record an error occurrence.

        Args:
            provider: Provider name
            error_type: Type of error
        """
        if not self._enabled:
            return
        self._errors_total.labels(provider=provider, error_type=error_type).inc()

    def record_provider_switch(self, from_provider: str, to_provider: str) -> None:
        """Record a provider switch event.

        Args:
            from_provider: Source provider name
            to_provider: Target provider name
        """
        if not self._enabled:
            return
        self._provider_switches_total.labels(
            from_provider=from_provider, to_provider=to_provider
        ).inc()

    def generate_metrics(self) -> str:
        """Generate metrics in Prometheus exposition format.

        Returns:
            Metrics string in Prometheus text format, or empty string if disabled
        """
        if not self._enabled or self._registry is None:
            return ""
        return generate_latest(self._registry).decode("utf-8")
