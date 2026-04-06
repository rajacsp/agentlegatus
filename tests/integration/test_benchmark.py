"""Integration test: benchmark engine.

Runs the BenchmarkEngine across multiple mock providers and verifies
metrics collection, report generation, and provider comparison.
"""

import json

import pytest

from agentlegatus.core.graph import PEGEdge, PEGNode, PortableExecutionGraph
from agentlegatus.observability.benchmark import BenchmarkEngine
from agentlegatus.observability.metrics import BenchmarkMetrics, MetricsCollector
from agentlegatus.providers.mock import MockProvider
from agentlegatus.providers.registry import ProviderRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register_provider("mock_a", MockProvider)
    registry.register_provider("mock_b", MockProvider)
    return registry


def _make_graph() -> PortableExecutionGraph:
    graph = PortableExecutionGraph()
    graph.add_node(PEGNode(node_id="n1", node_type="agent", config={"agent": {"agent_id": "a1"}}, inputs=[], outputs=[]))
    graph.add_node(PEGNode(node_id="n2", node_type="agent", config={"agent": {"agent_id": "a2"}}, inputs=[], outputs=[]))
    graph.add_edge(PEGEdge(source="n1", target="n2"))
    return graph


# ===================================================================
# Tests
# ===================================================================


class TestBenchmarkIntegration:
    """Run benchmarks across mock providers and verify metrics."""

    @pytest.mark.asyncio
    async def test_sequential_benchmark_returns_metrics_for_all_providers(self):
        """Each provider should have a BenchmarkMetrics entry."""
        registry = _make_registry()
        collector = MetricsCollector()
        engine = BenchmarkEngine(registry, collector)

        results = await engine.run_benchmark(
            workflow=_make_graph(),
            providers=["mock_a", "mock_b"],
            iterations=3,
            parallel=False,
        )

        assert set(results.keys()) == {"mock_a", "mock_b"}
        for name, m in results.items():
            assert isinstance(m, BenchmarkMetrics)
            assert m.provider_name == name
            assert m.success_rate == 1.0
            assert m.error_count == 0
            assert m.execution_time > 0

    @pytest.mark.asyncio
    async def test_parallel_benchmark(self):
        """Parallel mode should produce the same structure as sequential."""
        registry = _make_registry()
        collector = MetricsCollector()
        engine = BenchmarkEngine(registry, collector)

        results = await engine.run_benchmark(
            workflow=_make_graph(),
            providers=["mock_a", "mock_b"],
            iterations=2,
            parallel=True,
        )

        assert set(results.keys()) == {"mock_a", "mock_b"}
        for m in results.values():
            assert m.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_latency_percentiles_are_populated(self):
        """p50, p95, p99 should be non-negative after a benchmark run."""
        registry = _make_registry()
        collector = MetricsCollector()
        engine = BenchmarkEngine(registry, collector)

        results = await engine.run_benchmark(
            workflow=_make_graph(),
            providers=["mock_a"],
            iterations=5,
        )

        m = results["mock_a"]
        assert m.latency_p50 >= 0
        assert m.latency_p95 >= 0
        assert m.latency_p99 >= 0
        assert m.latency_p50 <= m.latency_p95 <= m.latency_p99

    @pytest.mark.asyncio
    async def test_generate_json_report(self):
        """JSON report should be valid JSON with expected keys."""
        registry = _make_registry()
        collector = MetricsCollector()
        engine = BenchmarkEngine(registry, collector)

        results = await engine.run_benchmark(
            workflow=_make_graph(),
            providers=["mock_a"],
            iterations=2,
        )

        report = engine.generate_report(results, format="json")
        data = json.loads(report)
        assert "mock_a" in data
        assert "execution_time" in data["mock_a"]
        assert "success_rate" in data["mock_a"]

    @pytest.mark.asyncio
    async def test_generate_table_report(self):
        """Table report should contain provider names and header row."""
        registry = _make_registry()
        collector = MetricsCollector()
        engine = BenchmarkEngine(registry, collector)

        results = await engine.run_benchmark(
            workflow=_make_graph(),
            providers=["mock_a", "mock_b"],
            iterations=2,
        )

        report = engine.generate_report(results, format="table")
        assert "mock_a" in report
        assert "mock_b" in report
        assert "Provider" in report

    @pytest.mark.asyncio
    async def test_compare_providers_by_execution_time(self):
        """compare_providers should return a sorted ranking."""
        registry = _make_registry()
        collector = MetricsCollector()
        engine = BenchmarkEngine(registry, collector)

        results = await engine.run_benchmark(
            workflow=_make_graph(),
            providers=["mock_a", "mock_b"],
            iterations=2,
        )

        ranking = engine.compare_providers(results, metric="execution_time")
        assert len(ranking) == 2
        # Should be sorted ascending (lower is better)
        assert ranking[0][1] <= ranking[1][1]
