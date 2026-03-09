"""Metrics data models and collection."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from agentlegatus.core.workflow import WorkflowStatus


@dataclass
class MetricsData:
    """Individual metric data point."""

    execution_id: str
    timestamp: datetime
    metric_type: str
    value: float
    unit: str
    labels: Dict[str, str] = field(default_factory=dict)

    def to_prometheus_format(self) -> str:
        """
        Convert to Prometheus exposition format.
        
        Returns:
            Metric in Prometheus format
        """
        labels_str = ",".join([f'{k}="{v}"' for k, v in self.labels.items()])
        if labels_str:
            return f"{self.metric_type}{{{labels_str}}} {self.value}"
        return f"{self.metric_type} {self.value}"

    def to_opentelemetry_format(self) -> Dict[str, Any]:
        """
        Convert to OpenTelemetry format.
        
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
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    status: str = "running"
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    tool_calls: int = 0

    def finalize(self) -> None:
        """Finalize metrics when step completes."""
        if self.end_time is None:
            self.end_time = datetime.now()
        if self.duration is None:
            self.duration = (self.end_time - self.start_time).total_seconds()


@dataclass
class ExecutionMetrics:
    """Metrics for complete workflow execution."""

    execution_id: str
    workflow_id: str
    provider: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    status: WorkflowStatus = WorkflowStatus.RUNNING
    total_cost: float = 0.0
    token_usage: Dict[str, int] = field(default_factory=dict)
    step_metrics: List[StepMetrics] = field(default_factory=list)
    error_count: int = 0
    retry_count: int = 0

    def calculate_cost(self) -> float:
        """
        Calculate total execution cost from step metrics.
        
        Returns:
            Total cost in dollars
        """
        self.total_cost = sum(step.cost for step in self.step_metrics)
        return self.total_cost

    def get_summary(self) -> Dict[str, Any]:
        """
        Get execution summary.
        
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
    token_usage: Dict[str, int]
    success_rate: float
    error_count: int
    latency_p50: float
    latency_p95: float
    latency_p99: float
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collector for execution metrics."""

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self.metrics: Dict[str, ExecutionMetrics] = {}
        self.data_points: List[MetricsData] = []

    def start_execution(
        self, execution_id: str, workflow_id: str, provider: str
    ) -> ExecutionMetrics:
        """
        Start tracking metrics for an execution.
        
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
        return metrics

    def end_execution(
        self, execution_id: str, status: WorkflowStatus, error_count: int = 0
    ) -> None:
        """
        End tracking for an execution.
        
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

    def add_step_metrics(self, execution_id: str, step_metrics: StepMetrics) -> None:
        """
        Add step metrics to execution.
        
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
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Record a metric data point.
        
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

    def get_metrics(self, execution_id: str) -> Optional[ExecutionMetrics]:
        """
        Get metrics for an execution.
        
        Args:
            execution_id: Execution identifier
            
        Returns:
            ExecutionMetrics if found, None otherwise
        """
        return self.metrics.get(execution_id)
