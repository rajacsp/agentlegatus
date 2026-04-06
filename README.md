# AgentLegatus

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/rajacsp/agentlegatus)

Vendor-agnostic agent framework abstraction layer — Terraform for AI Agents.

Switch between AI agent frameworks (LangGraph, AutoGen, CrewAI, Google ADK, AWS Strands, Microsoft Agent Framework) with minimal configuration changes. AgentLegatus provides a unified API, a Roman military hierarchy for orchestrating multi-agent workflows, and built-in benchmarking to compare providers side by side.

## Quick Start

```bash
# Install core
pip install -e .

# Install with all optional backends
pip install -e ".[all]"
```

### Define and run a workflow

```python
import asyncio
from agentlegatus.core.event_bus import EventBus
from agentlegatus.core.executor import WorkflowExecutor
from agentlegatus.core.state import InMemoryStateBackend, StateManager
from agentlegatus.core.workflow import (
    ExecutionStrategy, WorkflowDefinition, WorkflowStep,
)
from agentlegatus.hierarchy.legatus import Legatus
from agentlegatus.providers.mock import MockProvider
from agentlegatus.tools.registry import ToolRegistry

async def main():
    event_bus = EventBus()
    state_manager = StateManager(InMemoryStateBackend(), event_bus=event_bus)
    provider = MockProvider(config={})
    executor = WorkflowExecutor(provider, state_manager, ToolRegistry(), event_bus)

    workflow = WorkflowDefinition(
        workflow_id="hello",
        name="Hello Workflow",
        version="1.0.0",
        provider="mock",
        execution_strategy=ExecutionStrategy.SEQUENTIAL,
        steps=[
            WorkflowStep(step_id="greet", step_type="agent", config={"agent_id": "greeter"}),
        ],
    )

    legatus = Legatus(config={}, event_bus=event_bus)
    result = await legatus.execute_workflow(workflow, executor=executor, state_manager=state_manager)
    print(f"Status: {result.status.value}, Time: {result.execution_time:.3f}s")

asyncio.run(main())
```

### CLI

```bash
legatus init --provider mock
legatus apply workflow.yaml
legatus apply workflow.yaml --dry-run
legatus plan workflow.yaml
legatus benchmark workflow.yaml --providers mock,langgraph --iterations 5
legatus switch langgraph
legatus providers
legatus status <workflow-id>
legatus cancel <workflow-id>
```

## Architecture

```
CLI (legatus)
    └── Legatus (Orchestrator)
            ├── EventBus ──► Observability (OpenTelemetry, Prometheus)
            ├── StateManager (in-memory / Redis / Postgres)
            └── Centurion (Workflow Controller)
                    ├── Sequential / Parallel / Conditional execution
                    └── Cohort (Agent Group)
                            └── Agent (Worker)
                                    ├── ToolRegistry
                                    └── MemoryManager
```

## Features

- **Provider Abstraction** — `BaseProvider` interface with runtime switching and state migration via Portable Execution Graphs
- **Roman Hierarchy** — Legatus → Centurion → Cohort → Agent, with sequential, parallel, and conditional execution strategies
- **Event-Driven** — Unified `EventBus` with subscription, history, correlation/trace ID propagation
- **State Management** — Scoped state (workflow/step/agent/global) with snapshot/restore, backed by in-memory, Redis, or Postgres
- **Tool Registry** — Register tools once, auto-convert to OpenAI/Anthropic formats, cached per provider
- **Memory Abstraction** — Short-term (TTL), long-term, episodic, and semantic memory types with Redis and vector store backends
- **Benchmark Engine** — Run identical workflows across providers, compare latency (p50/p95/p99), cost, tokens, and success rate
- **Observability** — OpenTelemetry tracing, Prometheus metrics export, structured logging via structlog
- **Security** — Input sanitization, path traversal prevention, PII detection/redaction, rate limiting, audit logging, HTTPS with cert validation
- **Checkpoint & Recovery** — Checkpoint workflow state for resumption after failures or timeouts
- **Retry Logic** — Configurable exponential backoff with max delay capping

## Optional Dependencies

```bash
pip install -e ".[langgraph]"      # LangGraph provider
pip install -e ".[redis]"          # Redis state/memory backend
pip install -e ".[postgres]"       # Postgres state backend
pip install -e ".[vector]"         # ChromaDB vector memory
pip install -e ".[observability]"  # OpenTelemetry + Prometheus + structlog
pip install -e ".[dev]"            # black, ruff, mypy, isort
pip install -e ".[test]"           # pytest, hypothesis, coverage
```

## Configuration

AgentLegatus supports YAML/JSON config files with environment variable overrides:

```yaml
default_provider: mock
providers:
  - name: mock
  - name: openai
    api_key: ${OPENAI_API_KEY}
state:
  backend: memory
memory:
  backend: memory
observability:
  enable_tracing: false
  enable_prometheus: false
```

Load via code or CLI:

```python
from agentlegatus.config.loader import ConfigLoader
config = ConfigLoader.load("agentlegatus.yaml")
```

## Examples

See the `examples/` directory:

| File | Description |
|------|-------------|
| `basic_workflow.py` | Sequential workflow execution |
| `provider_switching.py` | Runtime provider switching with state preservation |
| `benchmark.py` | Cross-provider benchmarking |
| `event_monitoring.py` | Event subscription and real-time monitoring |
| `custom_tools.py` | Tool creation and registration |
| `state_management.py` | State operations, snapshots, and restore |

## Development

```bash
# Install dev + test dependencies
pip install -e ".[dev,test]"

# Run all tests (936 tests, 88% coverage)
pytest

# Run by category
pytest tests/unit/
pytest tests/integration/
pytest tests/property/

# Coverage report
pytest --cov=agentlegatus --cov-report=html

# Linting
ruff check agentlegatus/
black --check agentlegatus/
```

## Requirements

- Python 3.10+
- Core: click, pydantic, httpx, rich, pyyaml

## License

MIT
