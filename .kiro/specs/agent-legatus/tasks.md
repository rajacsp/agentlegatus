# Implementation Plan: AgentLegatus

## Overview

This implementation plan breaks down the AgentLegatus framework into discrete, manageable coding tasks. The framework is a vendor-agnostic agent orchestration system with a Roman military hierarchy architecture (Legatus → Centurion → Cohort → Agent). Implementation follows a bottom-up approach: core abstractions first, then hierarchy components, followed by provider implementations, and finally CLI and observability layers.

Target: ~2000 lines of core code with minimal dependencies, Python 3.10+, async/await throughout.

## Tasks

- [x] 1. Set up project structure and core infrastructure
  - Create Python package structure following design document layout
  - Set up pyproject.toml with core dependencies (click, pydantic, httpx, asyncio)
  - Configure development tools (pytest, black, ruff, mypy)
  - Create __init__.py files for all modules
  - Set up basic logging configuration
  - _Requirements: Project Structure section_

- [x] 2. Implement core data models and enumerations
  - [x] 2.1 Create workflow data models
    - Implement WorkflowDefinition, WorkflowStep, RetryPolicy dataclasses
    - Implement WorkflowStatus, WorkflowResult enumerations and dataclasses
    - Add validation methods to WorkflowDefinition
    - _Requirements: 1.1, 1.3, 1.4, 1.5, 14.1-14.5_
  
  - [ ]* 2.2 Write property test for workflow data models
    - **Property 27: Workflow Validation Completeness**
    - **Validates: Requirements 14.1-14.5**
  
  - [x] 2.3 Create execution context and configuration models
    - Implement ExecutionContext, AgentConfig, ProviderConfig dataclasses
    - Add validation methods and from_env() class method
    - _Requirements: 14.6-14.11, 19.1-19.9_
  
  - [x] 2.4 Create metrics data models
    - Implement MetricsData, ExecutionMetrics, StepMetrics, BenchmarkMetrics dataclasses
    - Add to_prometheus_format() and to_opentelemetry_format() methods
    - _Requirements: 1.8, 16.1-16.5, 26.1-26.5_


- [x] 3. Implement EventBus for event-driven architecture
  - [x] 3.1 Create Event and EventType classes
    - Implement Event dataclass with event_type, timestamp, source, data, correlation_id, trace_id
    - Implement EventType enumeration with all event types
    - _Requirements: 7.1, 7.8_
  
  - [x] 3.2 Implement EventBus core functionality
    - Implement subscribe() and unsubscribe() methods
    - Implement emit() and emit_and_wait() methods with async handler invocation
    - Implement event history tracking with get_event_history() and filtering
    - Add handler isolation (exceptions in one handler don't affect others)
    - _Requirements: 7.1-7.7_
  
  - [x] 3.3 Write property tests for EventBus
    - **Property 4: Event Temporal Ordering**
    - **Property 20: Event Handler Isolation**
    - **Property 21: Event History Completeness**
    - **Property 29: Unsubscribe Effectiveness**
    - **Validates: Requirements 7.3, 7.4, 7.5, 7.6**

- [x] 4. Implement StateManager and state backends
  - [x] 4.1 Create StateBackend abstract base class
    - Define abstract interface for state storage
    - Implement StateScope enumeration (WORKFLOW, STEP, AGENT, GLOBAL)
    - _Requirements: 8.12_
  
  - [x] 4.2 Implement StateManager core functionality
    - Implement get(), set(), update(), delete() methods with scope support
    - Implement get_all() and clear_scope() methods
    - Implement create_snapshot() and restore_snapshot() methods
    - Add StateUpdated event emission on modifications
    - Ensure atomic updates and scope isolation
    - _Requirements: 8.1-8.12_
  
  - [x] 4.3 Write property tests for StateManager
    - **Property 3: State Round-Trip Consistency**
    - **Property 22: State Scope Isolation**
    - **Property 23: State Snapshot Round-Trip**
    - **Property 30: State Update Atomicity**
    - **Validates: Requirements 8.1, 8.2, 8.4, 8.9, 8.10, 8.12**
  
  - [x] 4.4 Implement in-memory StateBackend
    - Create simple dictionary-based backend for development/testing
    - Implement all StateBackend abstract methods
    - _Requirements: 8.1-8.12_

- [x] 5. Implement ToolRegistry and Tool abstraction
  - [x] 5.1 Create Tool class and ToolParameter model
    - Implement ToolParameter with name, type, description, required, default
    - Implement Tool class with name, description, parameters, handler
    - Add validate_input() method
    - _Requirements: 9.5, 9.6_
  
  - [x] 5.2 Implement tool format converters
    - Implement to_openai_format() method
    - Implement to_anthropic_format() method
    - _Requirements: 9.8, 9.9_
  
  - [x] 5.3 Implement ToolRegistry
    - Implement register_tool(), get_tool(), list_tools() methods
    - Implement get_tools_for_provider() with format conversion
    - Implement unregister_tool() method
    - _Requirements: 9.1-9.4_
  
  - [x] 5.4 Write property tests for Tool and ToolRegistry
    - **Property 10: Tool Registry Round-Trip**
    - **Property 11: Tool Input Validation Consistency**
    - **Validates: Requirements 9.1, 9.2, 9.5, 9.6, 9.7**

- [x] 6. Implement PortableExecutionGraph (PEG)
  - [x] 6.1 Create PEGNode and PEGEdge dataclasses
    - Implement PEGNode with node_id, node_type, config, inputs, outputs
    - Implement PEGEdge with source, target, optional condition
    - _Requirements: 12.1, 12.2_
  
  - [x] 6.2 Implement PortableExecutionGraph core functionality
    - Implement add_node(), add_edge(), remove_node() methods
    - Implement get_node(), get_successors(), get_predecessors() methods
    - _Requirements: 12.1-12.6_
  
  - [x] 6.3 Implement graph validation
    - Implement validate() method with cycle detection using DFS
    - Add validation for invalid node references
    - Add validation for unique node IDs
    - Return (is_valid, errors) tuple
    - _Requirements: 12.7, 12.8, 29.1-29.5_
  
  - [x] 6.4 Implement graph serialization
    - Implement to_dict(), from_dict() class method
    - Implement to_json(), from_json() class method
    - _Requirements: 12.9, 12.10_
  
  - [x] 6.5 Write property tests for PortableExecutionGraph
    - **Property 16: Graph Serialization Round-Trip**
    - **Property 17: Graph Node Removal Completeness**
    - **Property 25: Graph Cycle Detection**
    - **Property 26: Graph Reference Validation**
    - **Validates: Requirements 12.3, 12.7, 12.8, 12.9, 12.10, 29.1, 29.2**


- [x] 7. Implement BaseProvider and ProviderRegistry
  - [x] 7.1 Create BaseProvider abstract class
    - Define abstract methods: create_agent(), execute_agent(), invoke_tool()
    - Define abstract methods: export_state(), import_state()
    - Define abstract methods: to_portable_graph(), from_portable_graph()
    - Implement _get_capabilities() abstract method
    - Implement supports_capability() concrete method
    - Define ProviderCapability enumeration
    - _Requirements: 3.1-3.8, 24.1-24.5_
  
  - [x] 7.2 Implement ProviderRegistry
    - Implement register_provider(), get_provider(), list_providers() methods
    - Implement get_provider_info() and unregister_provider() methods
    - Add provider instance caching
    - _Requirements: 4.1-4.6_
  
  - [x] 7.3 Write property tests for ProviderRegistry
    - **Property 18: Provider Registry Caching**
    - **Property 19: Provider Registry Completeness**
    - **Validates: Requirements 4.1, 4.4, 4.6**

- [x] 8. Implement retry logic utilities
  - [x] 8.1 Create retry utility functions
    - Implement execute_with_retry() function with exponential backoff
    - Support RetryPolicy configuration (max_attempts, backoff_multiplier, initial_delay, max_delay)
    - Add logging for each retry attempt
    - _Requirements: 6.1-6.7_
  
  - [x] 8.2 Write property tests for retry logic
    - **Property 7: Retry Attempt Limit**
    - **Property 8: Retry Exponential Backoff**
    - **Validates: Requirements 6.1-6.5**

- [x] 9. Implement Agent class
  - [x] 9.1 Create Agent class with capabilities
    - Implement __init__ with agent_id, name, capabilities, provider
    - Implement AgentCapability enumeration (TOOL_USE, MEMORY, PLANNING, REFLECTION)
    - Implement get_status() method
    - _Requirements: 18.1-18.9_
  
  - [x] 9.2 Implement Agent execution and tool invocation
    - Implement run() method that delegates to provider
    - Implement invoke_tool() method using ToolRegistry
    - _Requirements: 18.2, 18.6_
  
  - [x] 9.3 Implement Agent memory operations
    - Implement store_memory() method using MemoryManager
    - Implement retrieve_memory() method using MemoryManager
    - _Requirements: 18.3, 18.7, 18.8_

- [x] 10. Implement Cohort class
  - [x] 10.1 Create Cohort class with coordination strategies
    - Implement __init__ with name, strategy, max_agents
    - Implement CohortStrategy enumeration (ROUND_ROBIN, LOAD_BALANCED, BROADCAST, LEADER_FOLLOWER)
    - Implement add_agent() and remove_agent() methods with capacity enforcement
    - _Requirements: 17.1-17.3_
  
  - [x] 10.2 Implement Cohort task execution strategies
    - Implement execute_task() with strategy routing
    - Implement ROUND_ROBIN strategy
    - Implement LOAD_BALANCED strategy
    - Implement BROADCAST strategy
    - Implement LEADER_FOLLOWER strategy
    - _Requirements: 17.4-17.7_
  
  - [x] 10.3 Implement Cohort communication
    - Implement broadcast_message() method
    - Implement get_available_agents() method
    - _Requirements: 17.8, 17.9_
  
  - [x] 10.4 Write property tests for Cohort
    - **Property 24: Cohort Capacity Enforcement**
    - **Validates: Requirements 17.1, 17.2**

- [x] 11. Implement WorkflowExecutor
  - [ ] 11.1 Create WorkflowExecutor class
    - Implement __init__ with provider, state_manager, tool_registry, event_bus
    - Implement execute_step() method with retry logic integration
    - Add step metrics collection (duration, tokens, cost)
    - Emit StepStarted and StepCompleted/StepFailed events
    - _Requirements: 2.6-2.9, 6.1-6.7_
  
  - [x] 11.2 Implement checkpoint and recovery
    - Implement checkpoint_state() method
    - Implement restore_from_checkpoint() method
    - _Requirements: 21.1-21.7_
  
  - [x] 11.3 Implement provider switching
    - Implement switch_provider() method
    - Export state from current provider
    - Convert workflow to PortableExecutionGraph
    - Validate portable graph
    - Import state to new provider
    - Emit ProviderSwitched event
    - _Requirements: 5.1-5.8_
  
  - [x] 11.4 Write property tests for WorkflowExecutor
    - **Property 5: Provider State Round-Trip**
    - **Property 6: Portable Graph Round-Trip**
    - **Validates: Requirements 3.5, 3.6, 3.7, 3.8**
  
  - [x] 11.5 Implement execute_graph() method
    - Execute complete PortableExecutionGraph
    - Handle timeout enforcement at step level
    - _Requirements: 28.2, 28.3, 28.5_


- [ ] 12. Implement Centurion class
  - [-] 12.1 Create Centurion class with execution strategies
    - Implement __init__ with name, strategy, event_bus
    - Implement ExecutionStrategy enumeration (SEQUENTIAL, PARALLEL, CONDITIONAL)
    - Implement add_cohort() method
    - _Requirements: 2.1-2.4_
  
  - [x] 12.2 Implement execution plan building
    - Implement build_execution_plan() using topological sort
    - Validate DAG structure (no cycles)
    - Resolve step dependencies
    - _Requirements: 2.1, 2.5_
  
  - [x] 12.3 Implement sequential execution strategy
    - Implement execute_sequential() method
    - Execute steps one at a time in dependency order
    - Update state after each step completion
    - Emit events for each step
    - _Requirements: 2.2, 2.6-2.9_
  
  - [x] 12.4 Implement parallel execution strategy
    - Implement execute_parallel() method
    - Use asyncio.gather() for independent steps
    - Respect concurrency limits
    - Handle partial failures
    - _Requirements: 2.3, 22.1-22.5_
  
  - [x] 12.5 Implement conditional execution strategy
    - Implement execute_conditional() method
    - Implement evaluate_condition() method
    - Skip steps when condition evaluates to False
    - _Requirements: 2.4, 23.1-23.5_
  
  - [x] 12.6 Implement orchestrate() main method
    - Coordinate workflow execution across strategies
    - Integrate with WorkflowExecutor
    - Handle workflow-level error recovery
    - _Requirements: 2.1-2.9_
  
  - [ ]* 12.7 Write property tests for Centurion
    - **Property 9: Dependency Execution Order**
    - **Validates: Requirements 2.5**

- [ ] 13. Implement Legatus orchestrator
  - [ ] 13.1 Create Legatus class
    - Implement __init__ with config and event_bus
    - Implement add_centurion() method
    - Implement get_status() method
    - _Requirements: 1.1-1.8_
  
  - [ ] 13.2 Implement workflow execution
    - Implement execute_workflow() method
    - Initialize execution context with trace_id
    - Emit WorkflowStarted event
    - Coordinate Centurion execution
    - Collect execution metrics
    - Emit WorkflowCompleted or WorkflowFailed event
    - Return WorkflowResult with status, output, metrics, execution_time
    - _Requirements: 1.1-1.8_
  
  - [ ] 13.3 Implement workflow cancellation
    - Implement cancel_workflow() method
    - Stop all running steps
    - Create checkpoint before cancellation
    - Emit WorkflowCancelled event
    - Clean up resources
    - _Requirements: 1.5, 25.1-25.5_
  
  - [ ] 13.4 Implement workflow timeout enforcement
    - Add timeout checking in execute_workflow()
    - Cancel workflow when timeout exceeded
    - Create checkpoint before timeout cancellation
    - _Requirements: 28.1, 28.4_
  
  - [ ]* 13.5 Write property tests for Legatus
    - **Property 1: Workflow Execution Completeness**
    - **Property 2: Workflow Event Lifecycle**
    - **Validates: Requirements 1.1-1.6**

- [ ] 14. Checkpoint - Core framework complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 15. Implement MemoryManager and memory backends
  - [ ] 15.1 Create MemoryBackend abstract class
    - Define abstract methods: store(), retrieve(), delete(), clear()
    - Implement MemoryType enumeration (SHORT_TERM, LONG_TERM, EPISODIC, SEMANTIC)
    - _Requirements: 10.5-10.9, 30.1-30.4_
  
  - [ ] 15.2 Implement MemoryManager
    - Implement __init__ with backend
    - Implement store_short_term() with TTL support
    - Implement store_long_term() with optional embeddings
    - Implement semantic_search() method
    - Implement get_recent() method
    - _Requirements: 10.1-10.4_
  
  - [ ] 15.3 Implement in-memory MemoryBackend
    - Create dictionary-based backend for development/testing
    - Implement all MemoryBackend abstract methods
    - Ensure memory type isolation
    - _Requirements: 10.5-10.9, 30.1-30.4_
  
  - [ ]* 15.4 Write property tests for MemoryManager
    - **Property 12: Memory Type Isolation**
    - **Property 13: Memory Backend Round-Trip**
    - **Validates: Requirements 10.5, 10.6, 10.9, 30.1-30.3**

- [ ] 16. Implement BenchmarkEngine
  - [ ] 16.1 Create BenchmarkEngine class
    - Implement __init__ with provider_registry and metrics_collector
    - _Requirements: 11.1-11.11_
  
  - [ ] 16.2 Implement benchmark execution
    - Implement run_benchmark() method with parallel/sequential modes
    - Implement benchmark_provider() helper for single provider
    - Reset state between iterations
    - Use identical initial state for all iterations
    - Collect metrics: execution_time, cost, token_usage, success_rate, error_count
    - Calculate latency percentiles (p50, p95, p99)
    - _Requirements: 11.1-11.9_
  
  - [ ] 16.3 Implement benchmark reporting
    - Implement generate_report() with multiple formats (table, JSON)
    - Implement compare_providers() method
    - _Requirements: 11.10, 11.11_
  
  - [ ]* 16.4 Write property tests for BenchmarkEngine
    - **Property 14: Benchmark Iteration Count**
    - **Property 15: Benchmark State Isolation**
    - **Validates: Requirements 11.2, 11.5, 11.6**


- [ ] 17. Implement observability layer
  - [ ] 17.1 Create MetricsCollector class
    - Implement metrics collection for workflows and steps
    - Track execution time, cost, token usage
    - Calculate costs based on provider pricing
    - _Requirements: 16.1-16.5, 26.1-26.5_
  
  - [ ] 17.2 Implement structured logging
    - Set up structlog with JSON formatting
    - Include workflow_id, execution_id, step_id, trace_id in all logs
    - Use consistent log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Include error type, message, and stack trace for errors
    - _Requirements: 27.1-27.6_
  
  - [ ] 17.3 Implement OpenTelemetry integration
    - Set up OpenTelemetry tracing
    - Propagate trace_id and correlation_id through events
    - Export traces in OpenTelemetry format
    - _Requirements: 16.7, 16.8_
  
  - [ ] 17.4 Implement Prometheus metrics export
    - Export metrics in Prometheus exposition format
    - Include workflow, step, and provider metrics
    - _Requirements: 16.6_

- [ ] 18. Implement provider adapters (minimal implementations)
  - [ ] 18.1 Implement LangGraph provider
    - Create LangGraphProvider class extending BaseProvider
    - Implement all abstract methods
    - Declare supported capabilities
    - Implement state export/import
    - Implement portable graph conversion
    - _Requirements: 3.1-3.8_
  
  - [ ] 18.2 Implement mock provider for testing
    - Create MockProvider class extending BaseProvider
    - Implement deterministic behavior for testing
    - Support all capabilities
    - _Requirements: 3.1-3.8_
  
  - [ ]* 18.3 Write property tests for provider implementations
    - **Property 28: Provider Capability Enforcement**
    - **Validates: Requirements 24.4**

- [ ] 19. Implement additional state backends
  - [ ] 19.1 Implement Redis StateBackend
    - Create RedisStateBackend class
    - Implement connection pooling
    - Handle connection failures with retry
    - Support TTL for state values
    - _Requirements: 8.1-8.12, 15.4_
  
  - [ ] 19.2 Implement Postgres StateBackend
    - Create PostgresStateBackend class
    - Use asyncpg for async operations
    - Implement schema for state storage
    - Support transactions for atomic updates
    - _Requirements: 8.1-8.12, 15.4_

- [ ] 20. Implement additional memory backends
  - [ ] 20.1 Implement Redis MemoryBackend
    - Create RedisMemoryBackend class
    - Support TTL for short-term memory
    - Implement memory type isolation using key prefixes
    - _Requirements: 10.5-10.9, 30.1-30.4_
  
  - [ ] 20.2 Implement vector store MemoryBackend
    - Create VectorStoreMemoryBackend class
    - Integrate with ChromaDB or similar
    - Support semantic search with embeddings
    - _Requirements: 10.2, 10.3_

- [ ] 21. Implement CLI interface
  - [ ] 21.1 Create CLI command structure
    - Set up Click CLI framework
    - Create main cli() group
    - Add rich formatting for output
    - _Requirements: 13.1-13.10_
  
  - [ ] 21.2 Implement init command
    - Implement init() command with --provider and --config options
    - Initialize AgentLegatus with specified provider
    - Create default configuration file
    - _Requirements: 13.1_
  
  - [ ] 21.3 Implement apply command
    - Implement apply() command with workflow_file argument
    - Support --provider override and --dry-run flag
    - Execute workflow or validate only
    - Display execution results
    - _Requirements: 13.2, 13.3_
  
  - [ ] 21.4 Implement plan command
    - Implement plan() command
    - Display execution plan without running
    - Show step dependencies and order
    - _Requirements: 13.4_
  
  - [ ] 21.5 Implement benchmark command
    - Implement benchmark() command
    - Support --providers and --iterations options
    - Display benchmark results in table format
    - _Requirements: 13.5_
  
  - [ ] 21.6 Implement utility commands
    - Implement switch() command for provider switching
    - Implement providers() command to list available providers
    - Implement status() command to get workflow status
    - Implement cancel() command to cancel running workflow
    - _Requirements: 13.6-13.9_
  
  - [ ] 21.7 Implement error handling in CLI
    - Display clear error messages for all failures
    - Exit with non-zero status on errors
    - _Requirements: 13.10_

- [ ] 22. Implement configuration management
  - [ ] 22.1 Create configuration loader
    - Support YAML and JSON configuration files
    - Implement schema validation using Pydantic
    - _Requirements: 19.1-19.3_
  
  - [ ] 22.2 Implement environment variable overrides
    - Load configuration from environment variables
    - Override file configuration with env values
    - Support loading API keys from environment
    - _Requirements: 19.4, 19.6_
  
  - [ ] 22.3 Implement secrets management integration
    - Support loading from secrets management systems
    - Never log or expose sensitive values
    - _Requirements: 19.7, 19.8, 20.1, 20.2_
  
  - [ ] 22.4 Add configuration defaults
    - Provide sensible defaults for all optional values
    - Validate required configuration values
    - _Requirements: 19.5, 19.9_


- [ ] 23. Implement security features
  - [ ] 23.1 Implement input validation and sanitization
    - Validate all user inputs before execution
    - Sanitize file paths to prevent path traversal
    - Prevent injection attacks in workflow definitions
    - _Requirements: 20.4, 20.5_
  
  - [ ] 23.2 Implement secure API communication
    - Use HTTPS with certificate validation for all external calls
    - Support custom CA certificates
    - _Requirements: 20.3_
  
  - [ ] 23.3 Implement access control and audit logging
    - Enforce scope-based access control for state
    - Create audit trail for state modifications
    - Support PII detection and redaction
    - _Requirements: 20.7, 20.8, 20.9_
  
  - [ ] 23.4 Implement rate limiting
    - Add rate limiting per workflow
    - Enforce rate limits when configured
    - _Requirements: 20.10_

- [ ] 24. Implement error handling and recovery
  - [ ] 24.1 Create custom exception hierarchy
    - Define ProviderNotFoundError with available providers list
    - Define StateBackendUnavailableError
    - Define WorkflowValidationError
    - Define CapabilityNotSupportedError
    - Define ProviderSwitchError
    - Define MemoryOperationError
    - Define TimeoutError
    - _Requirements: 15.1-15.10_
  
  - [ ] 24.2 Implement error recovery mechanisms
    - Add reconnection logic for state backend failures
    - Implement rollback for failed provider switches
    - Preserve state on workflow timeout
    - Fallback to in-memory storage when backends unavailable
    - _Requirements: 15.4, 15.5, 15.8, 15.9_
  
  - [ ] 24.3 Implement comprehensive error logging
    - Log errors with full context (workflow_id, step_id, stack trace)
    - Use structured logging for all errors
    - _Requirements: 16.10_

- [ ] 25. Checkpoint - All core features implemented
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 26. Create example workflows and usage demonstrations
  - [ ] 26.1 Create basic workflow example
    - Demonstrate simple sequential workflow execution
    - Show workflow definition and execution
    - _Requirements: Example 1_
  
  - [ ] 26.2 Create provider switching example
    - Demonstrate switching between providers
    - Show state preservation across switch
    - _Requirements: Example 2_
  
  - [ ] 26.3 Create benchmark example
    - Demonstrate benchmarking across multiple providers
    - Show metrics comparison and reporting
    - _Requirements: Example 3_
  
  - [ ] 26.4 Create event-driven monitoring example
    - Demonstrate event subscription and handling
    - Show real-time workflow monitoring
    - _Requirements: Example 4_
  
  - [ ] 26.5 Create custom tool registration example
    - Demonstrate tool creation and registration
    - Show tool usage in agents
    - _Requirements: Example 5_
  
  - [ ] 26.6 Create state management example
    - Demonstrate state operations across scopes
    - Show snapshot and restore functionality
    - _Requirements: Example 6_

- [ ] 27. Write comprehensive unit tests
  - [ ] 27.1 Write unit tests for data models
    - Test WorkflowDefinition validation
    - Test AgentConfig validation
    - Test ExecutionContext methods
    - Test metrics data models
  
  - [ ] 27.2 Write unit tests for EventBus
    - Test subscription and unsubscription
    - Test event emission and handler invocation
    - Test event history tracking
    - Test handler isolation
  
  - [ ] 27.3 Write unit tests for StateManager
    - Test get/set/update/delete operations
    - Test scope isolation
    - Test snapshot/restore
    - Test atomic updates
  
  - [ ] 27.4 Write unit tests for ToolRegistry
    - Test tool registration and retrieval
    - Test format conversion
    - Test input validation
  
  - [ ] 27.5 Write unit tests for PortableExecutionGraph
    - Test graph construction
    - Test graph validation
    - Test serialization/deserialization
  
  - [ ] 27.6 Write unit tests for hierarchy components
    - Test Agent execution and capabilities
    - Test Cohort strategies
    - Test Centurion orchestration
    - Test Legatus workflow execution
  
  - [ ] 27.7 Write unit tests for WorkflowExecutor
    - Test step execution
    - Test checkpoint/restore
    - Test provider switching
  
  - [ ] 27.8 Write unit tests for BenchmarkEngine
    - Test benchmark execution
    - Test metrics collection
    - Test report generation
  
  - [ ] 27.9 Write unit tests for CLI commands
    - Test all CLI commands
    - Test error handling
    - Test output formatting

- [ ] 28. Write integration tests
  - [ ] 28.1 Write end-to-end workflow execution test
    - Test complete workflow from CLI to result
    - Use mock provider for deterministic testing
    - Verify state persistence and event emission
  
  - [ ] 28.2 Write provider switching integration test
    - Execute workflow with one provider
    - Switch to another provider
    - Verify state migration and workflow continuation
  
  - [ ] 28.3 Write benchmark integration test
    - Run benchmark across multiple mock providers
    - Verify metrics collection and reporting
  
  - [ ] 28.4 Write state backend integration test
    - Test with in-memory, Redis, and Postgres backends
    - Verify state consistency across backends
  
  - [ ] 28.5 Write event-driven workflow integration test
    - Register event handlers
    - Execute workflow and verify events
    - Test event ordering and correlation

- [ ] 29. Final checkpoint - Complete implementation
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 30. Performance optimization and final polish
  - [ ] 30.1 Optimize async execution
    - Review and optimize asyncio.gather() usage
    - Implement connection pooling where needed
    - Add caching for frequently accessed data
  
  - [ ] 30.2 Add performance monitoring
    - Verify OpenTelemetry integration works correctly
    - Test Prometheus metrics export
    - Validate metrics accuracy
  
  - [ ] 30.3 Code quality improvements
    - Run black, ruff, mypy on entire codebase
    - Fix any type hints issues
    - Ensure 85%+ test coverage
    - Add docstrings to all public APIs
  
  - [ ] 30.4 Final validation
    - Run all unit tests
    - Run all integration tests
    - Run all property-based tests
    - Verify all 30 requirements are met
    - Verify all 30 correctness properties hold

## Notes

- Tasks marked with `*` are optional property-based tests that can be skipped for faster MVP
- Each task references specific requirements for traceability
- Property tests validate universal correctness properties from the design document
- Checkpoints ensure incremental validation at key milestones
- Implementation follows bottom-up approach: core abstractions → hierarchy → providers → CLI
- Target is ~2000 lines of core code with minimal dependencies
- All code uses Python 3.10+ with async/await throughout
- Testing strategy includes unit tests, integration tests, and property-based tests
