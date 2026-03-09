"""Workflow data models and definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class WorkflowStatus(Enum):
    """Status of workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionStrategy(Enum):
    """Strategy for executing workflow steps."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    backoff_multiplier: float = 2.0
    initial_delay: float = 1.0
    max_delay: float = 60.0

    def validate(self) -> tuple[bool, List[str]]:
        """Validate retry policy configuration."""
        errors = []
        
        if self.max_attempts < 1:
            errors.append("max_attempts must be at least 1")
        if self.backoff_multiplier < 1.0:
            errors.append("backoff_multiplier must be at least 1.0")
        if self.initial_delay <= 0:
            errors.append("initial_delay must be positive")
        if self.max_delay <= 0:
            errors.append("max_delay must be positive")
        if self.initial_delay > self.max_delay:
            errors.append("initial_delay cannot exceed max_delay")
            
        return len(errors) == 0, errors


@dataclass
class WorkflowStep:
    """Individual step in a workflow."""

    step_id: str
    step_type: str  # "agent", "cohort", "condition", "loop"
    config: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_policy: Optional[RetryPolicy] = None

    def validate(self, all_step_ids: List[str]) -> tuple[bool, List[str]]:
        """
        Validate workflow step configuration.
        
        Args:
            all_step_ids: List of all step IDs in the workflow
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        if not self.step_id:
            errors.append("step_id cannot be empty")
        if not self.step_type:
            errors.append("step_type cannot be empty")
        if self.timeout is not None and self.timeout <= 0:
            errors.append(f"timeout must be positive, got {self.timeout}")
            
        # Validate dependencies reference existing steps
        for dep in self.depends_on:
            if dep not in all_step_ids:
                errors.append(f"dependency '{dep}' does not reference an existing step")
                
        # Validate retry policy if present
        if self.retry_policy:
            policy_valid, policy_errors = self.retry_policy.validate()
            if not policy_valid:
                errors.extend(policy_errors)
                
        return len(errors) == 0, errors


@dataclass
class WorkflowDefinition:
    """Complete workflow specification."""

    workflow_id: str
    name: str
    description: str
    version: str
    provider: str
    steps: List[WorkflowStep]
    initial_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[float] = None
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL

    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate workflow definition.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Validate required fields
        if not self.workflow_id:
            errors.append("workflow_id cannot be empty")
        if not self.name:
            errors.append("name cannot be empty")
        if not self.provider:
            errors.append("provider cannot be empty")
        if not self.steps:
            errors.append("workflow must have at least one step")
        if self.timeout is not None and self.timeout <= 0:
            errors.append(f"timeout must be positive, got {self.timeout}")
            
        # Validate step IDs are unique
        step_ids = [step.step_id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            duplicates = [sid for sid in step_ids if step_ids.count(sid) > 1]
            errors.append(f"duplicate step IDs found: {set(duplicates)}")
            
        # Validate each step
        for step in self.steps:
            step_valid, step_errors = step.validate(step_ids)
            if not step_valid:
                errors.extend([f"Step {step.step_id}: {err}" for err in step_errors])
                
        # Validate DAG structure (no cycles)
        if not errors:  # Only check cycles if basic validation passed
            has_cycle, cycle_errors = self._check_for_cycles()
            if has_cycle:
                errors.extend(cycle_errors)
                
        return len(errors) == 0, errors

    def _check_for_cycles(self) -> tuple[bool, List[str]]:
        """
        Check if workflow steps form a DAG (no cycles).
        
        Returns:
            Tuple of (has_cycle, error_messages)
        """
        # Build adjacency list
        graph: Dict[str, List[str]] = {step.step_id: step.depends_on for step in self.steps}
        
        # Track visited nodes and recursion stack
        visited = set()
        rec_stack = set()
        errors = []
        
        def dfs(node: str, path: List[str]) -> bool:
            """DFS to detect cycles."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    errors.append(f"cycle detected: {' -> '.join(cycle)}")
                    return True
                    
            path.pop()
            rec_stack.remove(node)
            return False
        
        # Check each node
        for step in self.steps:
            if step.step_id not in visited:
                if dfs(step.step_id, []):
                    return True, errors
                    
        return False, []


@dataclass
class WorkflowResult:
    """Result of workflow execution."""

    status: WorkflowStatus
    output: Any
    metrics: Dict[str, Any]
    execution_time: float
    error: Optional[Exception] = None
