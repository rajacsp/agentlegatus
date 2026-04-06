"""Audit logging for state modifications.

Requirements: 20.8
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agentlegatus.core.state import StateScope


@dataclass
class AuditEntry:
    """Single audit trail record for a state modification."""

    timestamp: datetime
    caller_id: str
    operation: str
    scope: StateScope
    key: str
    success: bool = True
    old_value: Any | None = None
    new_value: Any | None = None


class AuditLogger:
    """Records and queries an audit trail of state modifications."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def log_access(
        self,
        caller_id: str,
        operation: str,
        scope: StateScope,
        key: str,
        success: bool = True,
        old_value: Any | None = None,
        new_value: Any | None = None,
    ) -> AuditEntry:
        """Record an audit entry and return it."""
        entry = AuditEntry(
            timestamp=datetime.now(),
            caller_id=caller_id,
            operation=operation,
            scope=scope,
            key=key,
            success=success,
            old_value=old_value,
            new_value=new_value,
        )
        self._entries.append(entry)
        return entry

    def get_audit_trail(
        self,
        scope: StateScope | None = None,
        caller_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Retrieve audit entries with optional filtering.

        Entries are returned in chronological order (oldest first),
        capped at *limit*.
        """
        results = self._entries
        if scope is not None:
            results = [e for e in results if e.scope == scope]
        if caller_id is not None:
            results = [e for e in results if e.caller_id == caller_id]
        return results[:limit]

    def clear(self) -> None:
        """Clear the entire audit trail."""
        self._entries.clear()
