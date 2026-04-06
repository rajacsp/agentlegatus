"""Scope-based access control for state operations.

Requirements: 20.7
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from agentlegatus.core.state import StateScope


class Operation(Enum):
    """Operations that can be performed on state."""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"


class AccessDeniedError(Exception):
    """Raised when a caller lacks permission for the requested operation."""

    def __init__(self, caller_id: str, scope: StateScope, operation: Operation) -> None:
        self.caller_id = caller_id
        self.scope = scope
        self.operation = operation
        super().__init__(
            f"Access denied: caller '{caller_id}' cannot {operation.value} "
            f"in scope '{scope.value}'"
        )


@dataclass
class AccessPolicy:
    """Defines what scopes and operations a caller is allowed."""

    allowed_scopes: set[StateScope] = field(default_factory=set)
    read_only: bool = False


class AccessController:
    """Enforces scope-based access control for state operations."""

    def __init__(self) -> None:
        self._policies: dict[str, AccessPolicy] = {}

    def register_policy(self, caller_id: str, policy: AccessPolicy) -> None:
        """Register an access policy for a caller."""
        self._policies[caller_id] = policy

    def check_access(self, caller_id: str, scope: StateScope, operation: Operation) -> bool:
        """Check whether *caller_id* may perform *operation* in *scope*.

        Returns True if allowed, False otherwise.
        """
        policy = self._policies.get(caller_id)
        if policy is None:
            return False
        if scope not in policy.allowed_scopes:
            return False
        if policy.read_only and operation != Operation.READ:
            return False
        return True

    def enforce_access(self, caller_id: str, scope: StateScope, operation: Operation) -> None:
        """Like check_access but raises AccessDeniedError on failure."""
        if not self.check_access(caller_id, scope, operation):
            raise AccessDeniedError(caller_id, scope, operation)
