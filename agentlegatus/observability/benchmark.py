"""Benchmark engine for comparing provider performance."""

import asyncio
import json
import time
from typing import Any

from agentlegatus.core.event_bus import EventBus
from agentlegatus.core.graph import PortableExecutionGraph
from agentlegatus.core.state import InMemoryStateBackend, StateManager
from agentlegatus.observability.metrics import BenchmarkMetrics, MetricsCollector
from agentlegatus.providers.registry import ProviderRegistry
from agentlegatus.tools.registry import ToolRegistry
from agentlegatus.utils.logging import get_logger

logger = get_logger(__name__)


class BenchmarkEngine:
    """Engine for benchmarking providers.

    Executes identical workflows across multiple providers and compares
    performance, cost, and quality metrics.
    """

    def __init__(
        self,
        provider_registry: ProviderRegistry,
        metrics_collector: MetricsCollector,
    ) -> None:
        """Initialize benchmark engine.

        Args:
            provider_registry: Registry for accessing providers
            metrics_collector: Collector for execution metrics
        """
        self.provider_registry = provider_registry
        self.metrics_collector = metrics_collector

    async def run_benchmark(
        self,
        workflow: PortableExecutionGraph,
        providers: list[str],
        iterations: int = 10,
        parallel: bool = False,
        initial_state: dict[str, Any] | None = None,
    ) -> dict[str, BenchmarkMetrics]:
        """Run benchmark across multiple providers.

        Args:
            workflow: Portable execution graph to benchmark
            providers: List of provider names to benchmark
            iterations: Number of iterations per provider
            parallel: Whether to run providers concurrently
            initial_state: Initial state for each iteration

        Returns:
            Dictionary mapping provider names to their BenchmarkMetrics
        """
        logger.info(
            f"Starting benchmark: {len(providers)} providers, "
            f"{iterations} iterations, parallel={parallel}"
        )

        initial_state = initial_state or {}

        if parallel:
            tasks = [
                self._benchmark_provider(provider_name, workflow, iterations, initial_state)
                for provider_name in providers
            ]
            results_list = await asyncio.gather(*tasks)
            results = dict(zip(providers, results_list, strict=False))
        else:
            results: dict[str, BenchmarkMetrics] = {}
            for provider_name in providers:
                results[provider_name] = await self._benchmark_provider(
                    provider_name, workflow, iterations, initial_state
                )

        logger.info("Benchmark complete")
        return results

    async def _benchmark_provider(
        self,
        provider_name: str,
        workflow: PortableExecutionGraph,
        iterations: int,
        initial_state: dict[str, Any],
    ) -> BenchmarkMetrics:
        """Benchmark a single provider over multiple iterations.

        Args:
            provider_name: Name of the provider to benchmark
            workflow: Portable execution graph to execute
            iterations: Number of iterations
            initial_state: Initial state for each iteration

        Returns:
            BenchmarkMetrics for this provider
        """
        provider = self.provider_registry.get_provider(provider_name)

        latencies: list[float] = []
        total_cost = 0.0
        total_input_tokens = 0
        total_output_tokens = 0
        error_count = 0
        success_count = 0

        for i in range(iterations):
            logger.debug(f"Provider {provider_name}: iteration {i + 1}/{iterations}")

            # Reset state between iterations using fresh backend
            state_backend = InMemoryStateBackend()
            event_bus = EventBus()
            state_manager = StateManager(backend=state_backend, event_bus=event_bus)
            tool_registry = ToolRegistry()

            from agentlegatus.core.executor import WorkflowExecutor

            executor = WorkflowExecutor(
                provider=provider,
                state_manager=state_manager,
                tool_registry=tool_registry,
                event_bus=event_bus,
            )

            # Use identical initial state for all iterations
            iteration_state = initial_state.copy()

            start_time = time.monotonic()
            try:
                await executor.execute_graph(workflow, iteration_state)
                elapsed = time.monotonic() - start_time
                latencies.append(elapsed)
                success_count += 1
            except Exception as e:
                elapsed = time.monotonic() - start_time
                latencies.append(elapsed)
                error_count += 1
                logger.warning(f"Provider {provider_name} iteration {i + 1} failed: {e}")

            # Collect token/cost metrics from the metrics collector
            exec_id = f"benchmark_{provider_name}_{i}"
            exec_metrics = self.metrics_collector.get_metrics(exec_id)
            if exec_metrics:
                total_cost += exec_metrics.total_cost
                total_input_tokens += exec_metrics.token_usage.get("input", 0)
                total_output_tokens += exec_metrics.token_usage.get("output", 0)

        success_rate = success_count / iterations if iterations > 0 else 0.0

        # Calculate latency percentiles
        p50, p95, p99 = self._calculate_percentiles(latencies)

        return BenchmarkMetrics(
            provider_name=provider_name,
            execution_time=sum(latencies),
            total_cost=total_cost,
            token_usage={
                "input": total_input_tokens,
                "output": total_output_tokens,
                "total": total_input_tokens + total_output_tokens,
            },
            success_rate=success_rate,
            error_count=error_count,
            latency_p50=p50,
            latency_p95=p95,
            latency_p99=p99,
        )

    @staticmethod
    def _calculate_percentiles(
        latencies: list[float],
    ) -> tuple[float, float, float]:
        """Calculate p50, p95, p99 latency percentiles.

        Args:
            latencies: List of latency measurements

        Returns:
            Tuple of (p50, p95, p99)
        """
        if not latencies:
            return 0.0, 0.0, 0.0

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            idx = (p / 100.0) * (n - 1)
            lower = int(idx)
            upper = min(lower + 1, n - 1)
            frac = idx - lower
            return sorted_latencies[lower] * (1 - frac) + sorted_latencies[upper] * frac

        return percentile(50), percentile(95), percentile(99)

    def generate_report(
        self,
        results: dict[str, BenchmarkMetrics],
        format: str = "table",
    ) -> str:
        """Generate benchmark report.

        Args:
            results: Benchmark results by provider name
            format: Report format ("table" or "json")

        Returns:
            Formatted report string
        """
        if format == "json":
            return self._generate_json_report(results)
        return self._generate_table_report(results)

    @staticmethod
    def _generate_json_report(results: dict[str, BenchmarkMetrics]) -> str:
        """Generate JSON format report."""
        data = {}
        for name, metrics in results.items():
            data[name] = {
                "execution_time": metrics.execution_time,
                "total_cost": metrics.total_cost,
                "token_usage": metrics.token_usage,
                "success_rate": metrics.success_rate,
                "error_count": metrics.error_count,
                "latency_p50": metrics.latency_p50,
                "latency_p95": metrics.latency_p95,
                "latency_p99": metrics.latency_p99,
                "custom_metrics": metrics.custom_metrics,
            }
        return json.dumps(data, indent=2)

    @staticmethod
    def _generate_table_report(results: dict[str, BenchmarkMetrics]) -> str:
        """Generate table format report."""
        if not results:
            return "No benchmark results."

        headers = [
            "Provider",
            "Time(s)",
            "Cost($)",
            "Tokens",
            "Success%",
            "Errors",
            "P50(s)",
            "P95(s)",
            "P99(s)",
        ]

        rows = []
        for name, m in results.items():
            rows.append(
                [
                    name,
                    f"{m.execution_time:.3f}",
                    f"{m.total_cost:.4f}",
                    str(m.token_usage.get("total", 0)),
                    f"{m.success_rate * 100:.1f}",
                    str(m.error_count),
                    f"{m.latency_p50:.3f}",
                    f"{m.latency_p95:.3f}",
                    f"{m.latency_p99:.3f}",
                ]
            )

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        # Build table
        sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
        header_line = "|" + "|".join(f" {h:<{col_widths[i]}} " for i, h in enumerate(headers)) + "|"

        lines = [sep, header_line, sep]
        for row in rows:
            line = "|" + "|".join(f" {cell:<{col_widths[i]}} " for i, cell in enumerate(row)) + "|"
            lines.append(line)
        lines.append(sep)

        return "\n".join(lines)

    def compare_providers(
        self,
        results: dict[str, BenchmarkMetrics],
        metric: str = "execution_time",
    ) -> list[tuple[str, float]]:
        """Compare providers by a specific metric, ranked best to worst.

        Args:
            results: Benchmark results by provider name
            metric: Metric to compare by (e.g. "execution_time", "total_cost",
                    "success_rate", "latency_p50", "latency_p95", "latency_p99",
                    "error_count")

        Returns:
            List of (provider_name, metric_value) tuples sorted best-first.
            For success_rate, higher is better; for all others, lower is better.
        """
        ranked: list[tuple[str, float]] = []
        for name, m in results.items():
            value = getattr(m, metric, None)
            if value is None:
                continue
            ranked.append((name, float(value)))

        # Higher is better for success_rate, lower is better for everything else
        reverse = metric == "success_rate"
        ranked.sort(key=lambda x: x[1], reverse=reverse)
        return ranked
