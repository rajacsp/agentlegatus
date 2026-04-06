"""Unit tests for BenchmarkEngine."""

import asyncio
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

def _agent_config() -> dict:
    """Return a node config that satisfies MockProvider.execute_agent."""
    return {"agent": {"agent_id": "bench-agent", "name": "bench-agent"}}


def _simple_graph() -> PortableExecutionGraph:
    """Return a minimal valid PEG with a single node."""
    graph = PortableExecutionGraph()
    graph.add_node(
        PEGNode(node_id="step1", node_type="agent", config=_agent_config())
    )
    return graph


def _two_node_graph() -> PortableExecutionGraph:
    """Return a PEG with two sequential nodes."""
    graph = PortableExecutionGraph()
    graph.add_node(
        PEGNode(node_id="a", node_type="agent", config=_agent_config())
    )
    graph.add_node(
        PEGNode(
            node_id="b", node_type="agent", config=_agent_config(), inputs=["a"]
        )
    )
    graph.add_edge(PEGEdge(source="a", target="b"))
    return graph


def _make_registry(*provider_names: str) -> ProviderRegistry:
    """Create a ProviderRegistry with MockProvider registered under each name."""
    registry = ProviderRegistry()
    for name in provider_names:
        registry.register_provider(name, MockProvider)
    return registry


def _make_engine(
    *provider_names: str,
) -> tuple[BenchmarkEngine, ProviderRegistry, MetricsCollector]:
    """Convenience: build engine + registry + collector."""
    registry = _make_registry(*provider_names)
    collector = MetricsCollector()
    engine = BenchmarkEngine(provider_registry=registry, metrics_collector=collector)
    return engine, registry, collector


# ---------------------------------------------------------------------------
# Benchmark Execution
# ---------------------------------------------------------------------------

class TestBenchmarkExecution:
    """Tests for run_benchmark() execution behaviour."""

    @pytest.mark.asyncio
    async def test_run_benchmark_executes_each_provider(self):
        """run_benchmark() returns results keyed by every specified provider."""
        engine, _, _ = _make_engine("mock_a", "mock_b", "mock_c")
        results = await engine.run_benchmark(
            workflow=_simple_graph(),
            providers=["mock_a", "mock_b", "mock_c"],
            iterations=1,
        )
        assert set(results.keys()) == {"mock_a", "mock_b", "mock_c"}
        for name, metrics in results.items():
            assert isinstance(metrics, BenchmarkMetrics)
            assert metrics.provider_name == name

    @pytest.mark.asyncio
    async def test_run_benchmark_respects_iteration_count(self):
        """Each provider is executed exactly `iterations` times."""
        engine, registry, _ = _make_engine("mock")
        iterations = 5
        results = await engine.run_benchmark(
            workflow=_simple_graph(),
            providers=["mock"],
            iterations=iterations,
        )
        metrics = results["mock"]
        # success_count + error_count == iterations
        assert (metrics.success_rate * iterations) + metrics.error_count == iterations

    @pytest.mark.asyncio
    async def test_run_benchmark_parallel_true(self):
        """parallel=True still produces correct results for all providers."""
        engine, _, _ = _make_engine("p1", "p2")
        results = await engine.run_benchmark(
            workflow=_simple_graph(),
            providers=["p1", "p2"],
            iterations=2,
            parallel=True,
        )
        assert "p1" in results and "p2" in results
        assert results["p1"].success_rate == 1.0
        assert results["p2"].success_rate == 1.0

    @pytest.mark.asyncio
    async def test_run_benchmark_parallel_false(self):
        """parallel=False still produces correct results for all providers."""
        engine, _, _ = _make_engine("s1", "s2")
        results = await engine.run_benchmark(
            workflow=_simple_graph(),
            providers=["s1", "s2"],
            iterations=2,
            parallel=False,
        )
        assert "s1" in results and "s2" in results
        assert results["s1"].success_rate == 1.0
        assert results["s2"].success_rate == 1.0


    @pytest.mark.asyncio
    async def test_run_benchmark_resets_state_between_iterations(self):
        """Each iteration starts with a fresh state manager (state isolation)."""
        engine, _, _ = _make_engine("mock")
        initial_state = {"counter": 0}
        results = await engine.run_benchmark(
            workflow=_simple_graph(),
            providers=["mock"],
            iterations=3,
            initial_state=initial_state,
        )
        # All iterations should succeed because state is reset each time
        assert results["mock"].success_rate == 1.0
        assert results["mock"].error_count == 0

    @pytest.mark.asyncio
    async def test_run_benchmark_handles_iteration_failures(self):
        """Failures increment error_count and execution continues."""
        registry = ProviderRegistry()
        registry.register_provider("failing", MockProvider)
        collector = MetricsCollector()
        engine = BenchmarkEngine(
            provider_registry=registry, metrics_collector=collector
        )

        # Create a graph that will cause failures by using an invalid graph
        # We'll monkey-patch the provider's execute_agent to fail on odd iterations
        call_count = {"n": 0}
        original_benchmark = engine._benchmark_provider

        async def patched_benchmark(provider_name, workflow, iterations, initial_state):
            """Wrap _benchmark_provider to inject failures via provider patching."""
            provider = registry.get_provider(provider_name)
            original_execute = provider.execute_agent

            async def failing_execute(agent, input_data, state=None):
                call_count["n"] += 1
                if call_count["n"] % 2 == 0:
                    raise RuntimeError("Simulated failure")
                return await original_execute(agent, input_data, state)

            provider.execute_agent = failing_execute
            return await original_benchmark(provider_name, workflow, iterations, initial_state)

        engine._benchmark_provider = patched_benchmark

        results = await engine.run_benchmark(
            workflow=_simple_graph(),
            providers=["failing"],
            iterations=4,
        )
        metrics = results["failing"]
        # Some iterations should have failed, some succeeded
        assert metrics.error_count > 0
        assert metrics.success_rate < 1.0
        # Total should still be 4 iterations
        success_count = int(metrics.success_rate * 4)
        assert success_count + metrics.error_count == 4


# ---------------------------------------------------------------------------
# Metrics Collection
# ---------------------------------------------------------------------------

class TestMetricsCollection:
    """Tests for metrics collected during benchmarking."""

    @pytest.mark.asyncio
    async def test_collects_execution_time(self):
        """BenchmarkMetrics.execution_time is positive after benchmark."""
        engine, _, _ = _make_engine("mock")
        results = await engine.run_benchmark(
            workflow=_simple_graph(),
            providers=["mock"],
            iterations=3,
        )
        assert results["mock"].execution_time > 0.0

    @pytest.mark.asyncio
    async def test_collects_token_usage(self):
        """BenchmarkMetrics.token_usage contains input/output/total keys."""
        engine, _, _ = _make_engine("mock")
        results = await engine.run_benchmark(
            workflow=_simple_graph(),
            providers=["mock"],
            iterations=2,
        )
        tu = results["mock"].token_usage
        assert "input" in tu
        assert "output" in tu
        assert "total" in tu
        assert tu["total"] == tu["input"] + tu["output"]

    @pytest.mark.asyncio
    async def test_success_rate_all_pass(self):
        """success_rate is 1.0 when all iterations succeed."""
        engine, _, _ = _make_engine("mock")
        results = await engine.run_benchmark(
            workflow=_simple_graph(),
            providers=["mock"],
            iterations=5,
        )
        assert results["mock"].success_rate == 1.0

    @pytest.mark.asyncio
    async def test_success_rate_with_failures(self):
        """success_rate reflects the fraction of successful iterations."""
        registry = ProviderRegistry()
        registry.register_provider("flaky", MockProvider)
        collector = MetricsCollector()
        engine = BenchmarkEngine(
            provider_registry=registry, metrics_collector=collector
        )

        call_count = {"n": 0}
        original_benchmark = engine._benchmark_provider

        async def patched_benchmark(provider_name, workflow, iterations, initial_state):
            provider = registry.get_provider(provider_name)
            original_execute = provider.execute_agent

            async def sometimes_fail(agent, input_data, state=None):
                call_count["n"] += 1
                if call_count["n"] <= 2:
                    raise RuntimeError("fail")
                return await original_execute(agent, input_data, state)

            provider.execute_agent = sometimes_fail
            return await original_benchmark(provider_name, workflow, iterations, initial_state)

        engine._benchmark_provider = patched_benchmark

        results = await engine.run_benchmark(
            workflow=_simple_graph(),
            providers=["flaky"],
            iterations=4,
        )
        # First 2 fail, last 2 succeed → 50%
        assert results["flaky"].success_rate == pytest.approx(0.5)
        assert results["flaky"].error_count == 2

    @pytest.mark.asyncio
    async def test_latency_percentiles_calculated(self):
        """p50, p95, p99 are populated and non-negative."""
        engine, _, _ = _make_engine("mock")
        results = await engine.run_benchmark(
            workflow=_simple_graph(),
            providers=["mock"],
            iterations=10,
        )
        m = results["mock"]
        assert m.latency_p50 >= 0.0
        assert m.latency_p95 >= 0.0
        assert m.latency_p99 >= 0.0
        # p50 <= p95 <= p99
        assert m.latency_p50 <= m.latency_p95
        assert m.latency_p95 <= m.latency_p99


# ---------------------------------------------------------------------------
# Percentile calculation (static method)
# ---------------------------------------------------------------------------

class TestCalculatePercentiles:
    """Tests for BenchmarkEngine._calculate_percentiles."""

    def test_empty_latencies(self):
        p50, p95, p99 = BenchmarkEngine._calculate_percentiles([])
        assert (p50, p95, p99) == (0.0, 0.0, 0.0)

    def test_single_value(self):
        p50, p95, p99 = BenchmarkEngine._calculate_percentiles([1.0])
        assert p50 == 1.0
        assert p95 == 1.0
        assert p99 == 1.0

    def test_known_distribution(self):
        latencies = list(range(1, 101))  # 1..100
        p50, p95, p99 = BenchmarkEngine._calculate_percentiles(latencies)
        assert 49 <= p50 <= 51
        assert 94 <= p95 <= 96
        assert 98 <= p99 <= 100

    def test_ordering_invariant(self):
        """Percentiles satisfy p50 <= p95 <= p99."""
        import random
        latencies = [random.uniform(0.01, 5.0) for _ in range(50)]
        p50, p95, p99 = BenchmarkEngine._calculate_percentiles(latencies)
        assert p50 <= p95 <= p99


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

class TestReportGeneration:
    """Tests for generate_report() and compare_providers()."""

    @pytest.fixture
    def sample_results(self) -> dict[str, BenchmarkMetrics]:
        return {
            "provider_a": BenchmarkMetrics(
                provider_name="provider_a",
                execution_time=10.0,
                total_cost=0.05,
                token_usage={"input": 1000, "output": 500, "total": 1500},
                success_rate=1.0,
                error_count=0,
                latency_p50=0.8,
                latency_p95=1.2,
                latency_p99=1.5,
            ),
            "provider_b": BenchmarkMetrics(
                provider_name="provider_b",
                execution_time=15.0,
                total_cost=0.03,
                token_usage={"input": 800, "output": 400, "total": 1200},
                success_rate=0.9,
                error_count=1,
                latency_p50=1.2,
                latency_p95=2.0,
                latency_p99=2.5,
            ),
        }

    def test_generate_table_report(self, sample_results):
        """Table format returns a formatted string with headers and rows."""
        engine, _, _ = _make_engine()
        report = engine.generate_report(sample_results, format="table")
        assert isinstance(report, str)
        assert "Provider" in report
        assert "provider_a" in report
        assert "provider_b" in report
        # Table separators
        assert "+" in report
        assert "|" in report

    def test_generate_json_report(self, sample_results):
        """JSON format returns valid JSON with all provider keys."""
        engine, _, _ = _make_engine()
        report = engine.generate_report(sample_results, format="json")
        data = json.loads(report)
        assert "provider_a" in data
        assert "provider_b" in data
        assert data["provider_a"]["execution_time"] == 10.0
        assert data["provider_b"]["success_rate"] == 0.9

    def test_json_report_contains_all_fields(self, sample_results):
        """JSON report includes all expected metric fields."""
        engine, _, _ = _make_engine()
        report = engine.generate_report(sample_results, format="json")
        data = json.loads(report)
        expected_keys = {
            "execution_time",
            "total_cost",
            "token_usage",
            "success_rate",
            "error_count",
            "latency_p50",
            "latency_p95",
            "latency_p99",
            "custom_metrics",
        }
        for provider_data in data.values():
            assert set(provider_data.keys()) == expected_keys

    def test_table_report_empty_results(self):
        """Table report handles empty results gracefully."""
        engine, _, _ = _make_engine()
        report = engine.generate_report({}, format="table")
        assert "No benchmark results" in report

    def test_compare_providers_by_execution_time(self, sample_results):
        """compare_providers ranks by execution_time (lower is better)."""
        engine, _, _ = _make_engine()
        ranked = engine.compare_providers(sample_results, metric="execution_time")
        assert len(ranked) == 2
        # provider_a (10.0) should come before provider_b (15.0)
        assert ranked[0][0] == "provider_a"
        assert ranked[1][0] == "provider_b"

    def test_compare_providers_by_success_rate(self, sample_results):
        """compare_providers ranks by success_rate (higher is better)."""
        engine, _, _ = _make_engine()
        ranked = engine.compare_providers(sample_results, metric="success_rate")
        # provider_a (1.0) should come before provider_b (0.9)
        assert ranked[0][0] == "provider_a"
        assert ranked[0][1] == 1.0
        assert ranked[1][0] == "provider_b"
        assert ranked[1][1] == 0.9

    def test_compare_providers_by_total_cost(self, sample_results):
        """compare_providers ranks by total_cost (lower is better)."""
        engine, _, _ = _make_engine()
        ranked = engine.compare_providers(sample_results, metric="total_cost")
        # provider_b (0.03) should come before provider_a (0.05)
        assert ranked[0][0] == "provider_b"
        assert ranked[1][0] == "provider_a"

    def test_compare_providers_by_latency_p50(self, sample_results):
        """compare_providers ranks by latency_p50 (lower is better)."""
        engine, _, _ = _make_engine()
        ranked = engine.compare_providers(sample_results, metric="latency_p50")
        assert ranked[0][0] == "provider_a"
        assert ranked[0][1] == 0.8

    def test_compare_providers_by_error_count(self, sample_results):
        """compare_providers ranks by error_count (lower is better)."""
        engine, _, _ = _make_engine()
        ranked = engine.compare_providers(sample_results, metric="error_count")
        assert ranked[0][0] == "provider_a"
        assert ranked[0][1] == 0

    def test_compare_providers_empty_results(self):
        """compare_providers returns empty list for empty results."""
        engine, _, _ = _make_engine()
        ranked = engine.compare_providers({}, metric="execution_time")
        assert ranked == []
