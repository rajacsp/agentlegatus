# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-04-05

### Added

- `AgentScopeProvider` — new provider adapter for the AgentScope framework with lazy import to avoid package shadowing
  - Single-agent execution via `ReActAgent` and `UserAgent`
  - Multi-agent support with three strategies: `SEQUENTIAL` (sequential_pipeline), `FANOUT` (fanout_pipeline), and `DISCUSSION` (MsgHub group chat)
  - `create_agents()` for batch agent creation, `create_group()` / `execute_group()` for multi-agent orchestration
  - `MultiAgentStrategy` enum exported from providers package
- `LangGraphProvider` rewritten to use actual LangGraph API
  - `StateGraph` with `add_node()`, `add_edge(START, ...)`, `compile()` for graph construction
  - `compiled_graph.ainvoke()` for async execution
  - `create_react_agent()` from `langgraph.prebuilt` for tool-calling agents
  - `build_graph()` convenience method for custom multi-node workflows with conditional edges
  - `HUMAN_IN_LOOP` capability added
- `agentscope` optional dependency group in pyproject.toml

### Changed

- Renamed `langgraph.py` → `langgraph_provider.py` to avoid shadowing the `langgraph` package on import
- Renamed `agentscope.py` → `agentscope_provider.py` to avoid shadowing the `agentscope` package on import
- Updated all import references across `__init__.py`, `cli/main.py`, and example scripts
- Updated project URLs from `agentlegatus/agentlegatus` to `rajacsp/agentlegatus`
- Added `langchain_core` and `langchain_openai` to mypy ignore list

## [0.1.0] - 2026-04-05

### Added

- Core data models: `WorkflowDefinition`, `WorkflowStep`, `RetryPolicy`, `WorkflowResult`, `WorkflowStatus`, `ExecutionContext`, `AgentConfig`, `ProviderConfig`
- `EventBus` with subscribe/unsubscribe, emit/emit_and_wait, event history with filtering, handler isolation, and correlation/trace ID propagation
- `StateManager` with scoped state (workflow/step/agent/global), atomic updates, snapshot/restore, and `StateUpdated` event emission
- `InMemoryStateBackend` for development and testing
- `RedisStateBackend` with connection pooling and retry logic
- `PostgresStateBackend` using asyncpg with connection pooling and transactions
- `ToolRegistry` with register/unregister, OpenAI/Anthropic format conversion, provider-format caching, and input validation
- `Tool` abstraction with parameter validation and multi-format export
- `PortableExecutionGraph` (PEG) with node/edge management, cycle detection via DFS, JSON serialization round-trip, and graph validation
- `BaseProvider` abstract class with capability declaration, state export/import, and portable graph conversion
- `ProviderRegistry` with registration, discovery, instance caching, and capability introspection
- `MockProvider` for deterministic testing with full capability support
- `LangGraphProvider` adapter implementation
- `WorkflowExecutor` with parallel graph execution (Kahn's algorithm), per-node timeout enforcement, provider switching with rollback, and checkpoint/restore
- `Centurion` workflow controller with sequential, parallel, and conditional execution strategies, topological sort planning, and concurrency limits
- `Cohort` agent group with round-robin, load-balanced, broadcast, and leader-follower coordination strategies
- `Agent` worker with capability-based tool invocation and memory operations
- `Legatus` top-level orchestrator with workflow lifecycle management, timeout enforcement, cancellation with checkpointing, and rate limiting
- `MemoryManager` with short-term (TTL), long-term (embedding), semantic search, and recent retrieval
- `InMemoryMemoryBackend`, `RedisMemoryBackend`, and `VectorStoreMemoryBackend` (ChromaDB)
- `BenchmarkEngine` with parallel/sequential provider benchmarking, latency percentiles (p50/p95/p99), cost tracking, and table/JSON report generation
- `MetricsCollector` with workflow and step-level metrics, token counting, and cost calculation
- `PrometheusExporter` with workflow/step/token/error counters and histograms
- `TracingManager` with OpenTelemetry integration, OTLP gRPC/HTTP export, and `EventBusTracingBridge`
- Structured logging via structlog with consistent context fields (workflow_id, execution_id, step_id, trace_id)
- `ResilientStateManager` with automatic fallback to in-memory backend on connection failures
- CLI (`legatus`) with init, apply, plan, benchmark, switch, providers, status, and cancel commands
- `ConfigLoader` supporting YAML/JSON files, environment variable overrides, and secrets management
- Input validation and sanitization (path traversal prevention, injection detection)
- `AccessController` with scope-based access control
- `AuditLogger` for state modification audit trails
- `PIIDetector` with email, phone, and SSN pattern detection and redaction
- `RateLimiter` with per-workflow rate limiting
- `SecureHTTPClient` with HTTPS certificate validation and custom CA support
- Custom exception hierarchy: `ProviderNotFoundError`, `ProviderSwitchError`, `CapabilityNotSupportedError`, `StateBackendUnavailableError`, `MemoryOperationError`, `WorkflowValidationError`, `WorkflowTimeoutError`
- 814 unit tests, 26 integration tests, 96 property-based tests (Hypothesis)
- 88% test coverage
- 6 example scripts: basic workflow, provider switching, benchmarking, event monitoring, custom tools, state management

### Fixed

- `EventBus.emit()` now uses `asyncio.gather` instead of fire-and-forget `create_task` for reliable handler execution

### Changed

- Modernized all type annotations from `typing.Dict`/`List`/`Optional` to built-in `dict`/`list`/`X | None` (Python 3.10+)
- Sorted all imports via ruff isort rules
- Updated pyproject.toml ruff config to use `[tool.ruff.lint]` section format

[Unreleased]: https://github.com/rajacsp/agentlegatus/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/rajacsp/agentlegatus/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/rajacsp/agentlegatus/releases/tag/v0.1.0
