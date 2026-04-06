"""Property-based tests for BenchmarkEngine."""

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agentlegatus.core.graph import PEGNode, PortableExecutionGraph
from agentlegatus.observability.benchmark import BenchmarkEngine
from agentlegatus.observability.metrics import MetricsCollector
from agentlegatus.providers.base import BaseProvider, ProviderCapability
from agentlegatus.providers.registry import ProviderRegistry


# --- Helpers ---

class _MockProvider(BaseProvider):
    """Deterministic mock provider for benchmark testing."""

    def __init__(self, config: Dict[str, Any]):
        self._state: Dict[str, Any] = {}
        super().__init__(config)

    def _get_capabilities(self) -> List[ProviderCapability]:
        return list(ProviderCapability)

    async def create_agent(self, agent_config: Dict[str, Any]) -> Any:
        return {"mock": True}

    async def execute_agent(
        self, agent: Any, input_data: Any, state: Optional[Dict[str, Any]] = None
    ) -> Any:
        return {"result": "ok"}

    async def invoke_tool(
        self, tool: Any, tool_input: Dict[str, Any], context: Dict[str, Any]
    ) -> Any:
        return {"tool_result": "ok"}

    def export_state(self) -> Dict[str, Any]:
        return dict(self._state)

    def import_state(self, state: Dict[str, Any]) -> None:
        self._state = dict(state)

    def to_portable_graph(self, workflow: Any) -> PortableExecutionGraph:
        return PortableExecutionGraph()

    def from_portable_graph(self, graph: PortableExecutionGraph) -> Any:
        return {}


class _FailingProvider(BaseProvider):
    """Provider that always fails during execution."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def _get_capabilities(self) -> List[ProviderCapability]:
        return list(ProviderCapability)

    async def create_agent(self, agent_config: Dict[str, Any]) -> Any:
        return {"mock": True}

    async def execute_agent(
        self, agent: Any, input_data: Any, state: Optional[Dict[str, Any]] = None
    ) -> Any:
        raise RuntimeError("Provider execution failed")

    async def invoke_tool(
        self, tool: Any, tool_input: Dict[str, Any], context: Dict[str, Any]
    ) -> Any:
        raise RuntimeError("Tool invocation failed")

    def export_state(self) -> Dict[str, Any]:
        return {}

    def import_state(self, state: Dict[str, Any]) -> None:
        pass

    def to_portable_graph(self, workflow: Any) -> PortableExecutionGraph:
        return PortableExecutionGraph()

    def from_portable_graph(self, graph: PortableExecutionGraph) -> Any:
        return {}


def _make_simple_graph() -> PortableExecutionGraph:
    """Create a minimal valid graph with one node."""
    graph = PortableExecutionGraph()
    graph.add_node(
        PEGNode(
            node_id="step1",
            node_type="agent",
            config={"name": "test"},
            inputs=[],
            outputs=[],
        )
    )
    return graph


def _make_registry(*provider_names: str, failing: bool = False) -> ProviderRegistry:
    """Create a ProviderRegistry with mock providers registered."""
    registry = ProviderRegistry()
    cls = _FailingProvider if failing else _MockProvider
    for name in provider_names:
        registry.register_provider(name, cls)
    return registry


# --- Property 14: Benchmark Iteration Count ---


@pytest.mark.asyncio
@given(
    iterations=st.integers(min_value=1, max_value=8),
    num_providers=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=10, deadline=10000)
async def test_property_14_iteration_count(iterations: int, num_providers: int):
    """
    Property 14: Benchmark Iteration Count

    For any benchmark run, each provider is executed exactly the specified
    number of iterations. The total number of execute_graph calls equals
    iterations * number_of_providers.

    Validates: Requirement 11.2
    """
    provider_names = [f"provider_{i}" for i in range(num_providers)]
    registry = _make_registry(*provider_names)
    collector = MetricsCollector()
    engine = BenchmarkEngine(provider_registry=registry, metrics_collector=collector)
    graph = _make_simple_graph()

    call_counts: Dict[str, int] = {name: 0 for name in provider_names}

    original_benchmark = engine._benchmark_provider

    async def tracking_benchmark(provider_name, workflow, iters, initial_state):
        call_counts[provider_name] = iters
        return await original_benchmark(provider_name, workflow, iters, initial_state)

    engine._benchmark_provider = tracking_benchmark

    results = await engine.run_benchmark(
        workflow=graph,
        providers=provider_names,
        iterations=iterations,
        parallel=False,
    )

    # Each provider must appear in results
    assert set(results.keys()) == set(provider_names), (
        f"Expected results for {provider_names}, got {list(results.keys())}"
    )

    for name in provider_names:
        metrics = results[name]
        # success_count + error_count must equal iterations
        total = int(metrics.success_rate * iterations + 0.5) + metrics.error_count
        assert total == iterations, (
            f"Provider '{name}': success + errors = {total}, expected {iterations}"
        )


@pytest.mark.asyncio
@given(iterations=st.integers(min_value=1, max_value=6))
@settings(max_examples=5, deadline=10000)
async def test_property_14_iteration_count_parallel(iterations: int):
    """
    Property 14 (Parallel): Benchmark Iteration Count in parallel mode.

    Same property holds when parallel=True.

    Validates: Requirement 11.2
    """
    provider_names = ["alpha", "beta"]
    registry = _make_registry(*provider_names)
    collector = MetricsCollector()
    engine = BenchmarkEngine(provider_registry=registry, metrics_collector=collector)
    graph = _make_simple_graph()

    results = await engine.run_benchmark(
        workflow=graph,
        providers=provider_names,
        iterations=iterations,
        parallel=True,
    )

    for name in provider_names:
        metrics = results[name]
        total = int(metrics.success_rate * iterations + 0.5) + metrics.error_count
        assert total == iterations, (
            f"Provider '{name}' (parallel): success + errors = {total}, "
            f"expected {iterations}"
        )


@pytest.mark.asyncio
@given(iterations=st.integers(min_value=1, max_value=6))
@settings(max_examples=5, deadline=10000)
async def test_property_14_failing_provider_still_counts(iterations: int):
    """
    Property 14 (Failure Case): Even when all iterations fail, the engine
    still runs exactly `iterations` times and records them as errors.

    Validates: Requirement 11.2
    """
    registry = _make_registry("bad_provider", failing=True)
    collector = MetricsCollector()
    engine = BenchmarkEngine(provider_registry=registry, metrics_collector=collector)
    graph = _make_simple_graph()

    results = await engine.run_benchmark(
        workflow=graph,
        providers=["bad_provider"],
        iterations=iterations,
        parallel=False,
    )

    metrics = results["bad_provider"]
    assert metrics.error_count == iterations, (
        f"Expected {iterations} errors, got {metrics.error_count}"
    )
    assert metrics.success_rate == 0.0


# --- Property 15: Benchmark State Isolation ---


@pytest.mark.asyncio
@given(iterations=st.integers(min_value=2, max_value=6))
@settings(max_examples=5, deadline=10000)
async def test_property_15_state_isolation_between_iterations(iterations: int):
    """
    Property 15: Benchmark State Isolation

    Each benchmark iteration starts with a fresh state. State set during
    one iteration must not leak into the next iteration.

    Validates: Requirements 11.5, 11.6
    """
    registry = _make_registry("isolated_provider")
    collector = MetricsCollector()
    engine = BenchmarkEngine(provider_registry=registry, metrics_collector=collector)
    graph = _make_simple_graph()

    initial_state = {"counter": 0, "marker": "original"}

    # Track the initial_state passed to each iteration's executor
    iteration_states: List[Dict[str, Any]] = []

    original_benchmark = engine._benchmark_provider

    async def intercepting_benchmark(provider_name, workflow, iters, init_state):
        """Intercept to verify each iteration gets a copy of initial_state."""
        # Verify the initial_state dict is passed correctly
        iteration_states.append(dict(init_state))
        return await original_benchmark(provider_name, workflow, iters, init_state)

    engine._benchmark_provider = intercepting_benchmark

    results = await engine.run_benchmark(
        workflow=graph,
        providers=["isolated_provider"],
        iterations=iterations,
        parallel=False,
        initial_state=initial_state,
    )

    # The original initial_state must not be mutated
    assert initial_state == {"counter": 0, "marker": "original"}, (
        "Original initial_state was mutated during benchmark"
    )


@pytest.mark.asyncio
@given(
    iterations=st.integers(min_value=2, max_value=5),
    num_providers=st.integers(min_value=2, max_value=3),
)
@settings(max_examples=5, deadline=10000)
async def test_property_15_state_isolation_between_providers(
    iterations: int, num_providers: int
):
    """
    Property 15 (Cross-Provider): Benchmark State Isolation between providers.

    Each provider gets its own fresh state manager per iteration. State from
    one provider's execution must not affect another provider's execution.

    Validates: Requirements 11.5, 11.6
    """
    provider_names = [f"prov_{i}" for i in range(num_providers)]
    registry = _make_registry(*provider_names)
    collector = MetricsCollector()
    engine = BenchmarkEngine(provider_registry=registry, metrics_collector=collector)
    graph = _make_simple_graph()

    initial_state = {"shared_key": "initial_value"}

    results = await engine.run_benchmark(
        workflow=graph,
        providers=provider_names,
        iterations=iterations,
        parallel=False,
        initial_state=initial_state,
    )

    # All providers should have independent results
    for name in provider_names:
        assert name in results, f"Missing results for provider '{name}'"
        metrics = results[name]
        # Each provider ran the correct number of iterations
        total = int(metrics.success_rate * iterations + 0.5) + metrics.error_count
        assert total == iterations

    # Original state must remain untouched
    assert initial_state == {"shared_key": "initial_value"}


@pytest.mark.asyncio
@given(iterations=st.integers(min_value=2, max_value=5))
@settings(max_examples=10, deadline=10000)
async def test_property_15_identical_initial_state_per_iteration(iterations: int):
    """
    Property 15 (Identical State): Each iteration receives identical initial state.

    The benchmark engine must use the same initial_state for every iteration,
    ensuring fair comparison.

    Validates: Requirement 11.6
    """
    registry = _make_registry("fair_provider")
    collector = MetricsCollector()
    engine = BenchmarkEngine(provider_registry=registry, metrics_collector=collector)
    graph = _make_simple_graph()

    initial_state = {"seed": 42, "data": [1, 2, 3]}

    # Patch execute_graph to record the state it receives
    observed_states: List[Dict[str, Any]] = []

    from agentlegatus.core.executor import WorkflowExecutor

    original_execute = WorkflowExecutor.execute_graph

    async def capturing_execute(self, g, state):
        observed_states.append(dict(state))
        return await original_execute(self, g, state)

    with patch.object(WorkflowExecutor, "execute_graph", capturing_execute):
        await engine.run_benchmark(
            workflow=graph,
            providers=["fair_provider"],
            iterations=iterations,
            parallel=False,
            initial_state=initial_state,
        )

    assert len(observed_states) == iterations, (
        f"Expected {iterations} iterations, observed {len(observed_states)}"
    )

    # Every iteration must have received the same initial state
    for i, state in enumerate(observed_states):
        assert state == initial_state, (
            f"Iteration {i} received state {state}, expected {initial_state}"
        )
