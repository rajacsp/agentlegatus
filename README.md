# AgentLegatus

A vendor-agnostic agent framework abstraction layer that enables developers to switch between different AI agent frameworks with minimal configuration changes.

## Overview

AgentLegatus provides a unified API across multiple agent frameworks (Microsoft Agent Framework, Google ADK, AWS Strands, LangGraph, AutoGen, CrewAI) while maintaining framework-specific optimizations. It implements a Roman military hierarchy architecture (Legatus → Centurion → Cohort → Agent) for orchestrating complex multi-agent workflows.

## Features

- **Provider Abstraction**: Switch between agent frameworks with a single configuration change
- **Event-Driven Architecture**: Unified event bus for monitoring and reactive behaviors
- **State Management**: Unified state management across different framework state models
- **Tool Registry**: Normalize tool invocation across different framework tool systems
- **Memory Abstraction**: Unified memory interface supporting multiple backends (Redis, Postgres, vector stores)
- **Benchmark Engine**: Compare performance, cost, and quality metrics across providers
- **Portable Execution Graph (PEG)**: Framework-agnostic workflow definition format
- **Property-Based Testing**: Comprehensive test suite with formal correctness properties

## Installation

```bash
pip install -e .
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run property-based tests
pytest tests/property/

# Run with coverage
pytest --cov=agentlegatus --cov-report=html
```

### Test Coverage

The project includes comprehensive testing:
- **Unit Tests**: Test individual components and methods
- **Property-Based Tests**: Validate universal correctness properties using Hypothesis
- **Integration Tests**: Test end-to-end workflows

Current property tests validate:
- Event temporal ordering and handler isolation
- State round-trip consistency and scope isolation
- Tool registry and input validation
- Graph serialization and cycle detection
- Provider state and graph conversion round-trips
- Retry logic with exponential backoff
- Cohort capacity enforcement

## Architecture

```
CLI Interface
    ↓
Legatus (Orchestrator)
    ↓
Centurion (Workflow Controller)
    ↓
Cohort (Agent Group)
    ↓
Agent (Worker)
```

## Requirements

- Python 3.10+
- See `pyproject.toml` for dependencies

## License

[Add license information]

## Contributing

[Add contributing guidelines]
