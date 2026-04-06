#!/usr/bin/env python3
"""Example 3: Benchmarking Across Multiple Providers.

Demonstrates running the same workflow on MockProvider and LangGraphProvider,
collecting metrics, and generating a comparison report.
"""

import asyncio

from agentlegatus.core.graph import PEGEdge, PEGNode, PortableExecutionGraph
from agentlegatus.observability.benchmark import BenchmarkEngine
from agentlegatus.observability.metrics import MetricsCollector
from agentlegatus.providers.langgraph import LangGraphProvider
from agentlegatus.providers.mock import MockProvider
from agentlegatus.providers.registry import ProviderRegistry


async def main():
    # --- 1. Register providers ---
    registry = ProviderRegistry()
    registry.register_provider("mock", MockProvider)
    registry.register_provider("langgraph", LangGraphProvider)

    metrics_collector = MetricsCollector()

    engine = BenchmarkEngine(
        provider_registry=registry,
        metrics_collector=metrics_collector,
    )

    # --- 2. Build a portable workflow graph ---
    graph = PortableExecutionGraph()
    graph.metadata = {"workflow_id": "bench-wf-001", "name": "Benchmark Workflow"}

    graph.add_node(PEGNode(
        node_id="start",
        node_type="agent",
        config={"agent_id": "starter", "model": "mock-model"},
        inputs=[],
        outputs=["intermediate"],
    ))
    graph.add_node(PEGNode(
        node_id="process",
        node_type="agent",
        config={"agent_id": "processor", "model": "mock-model"},
        inputs=["intermediate"],
        outputs=["final"],
    ))
    graph.add_edge(PEGEdge(source="start", target="process"))

    # --- 3. Run benchmark (sequential, 5 iterations per provider) ---
    print("Running benchmark …")
    results = await engine.run_benchmark(
        workflow=graph,
        providers=["mock", "langgraph"],
        iterations=5,
        parallel=False,
    )

    # --- 4. Display results ---
    table_report = engine.generate_report(results, format="table")
    print(table_report)

    json_report = engine.generate_report(results, format="json")
    print("\nJSON report (truncated):")
    print(json_report[:500])

    # --- 5. Compare providers by execution time ---
    ranking = engine.compare_providers(results, metric="execution_time")
    print("\nProvider ranking by execution time:")
    for name, value in ranking:
        print(f"  {name}: {value:.4f}s")


if __name__ == "__main__":
    asyncio.run(main())
