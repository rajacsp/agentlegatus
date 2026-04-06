"""Unit tests for audit logging.

Requirements: 20.8
"""

from datetime import datetime

from agentlegatus.core.state import StateScope
from agentlegatus.security.audit import AuditEntry, AuditLogger


class TestAuditLogger:

    def setup_method(self):
        self.logger = AuditLogger()

    def test_log_access_creates_entry(self):
        entry = self.logger.log_access(
            caller_id="agent-1",
            operation="write",
            scope=StateScope.WORKFLOW,
            key="counter",
            old_value=0,
            new_value=1,
        )
        assert isinstance(entry, AuditEntry)
        assert entry.caller_id == "agent-1"
        assert entry.operation == "write"
        assert entry.scope == StateScope.WORKFLOW
        assert entry.key == "counter"
        assert entry.old_value == 0
        assert entry.new_value == 1
        assert entry.success is True
        assert isinstance(entry.timestamp, datetime)

    def test_log_access_failure(self):
        entry = self.logger.log_access(
            caller_id="agent-2",
            operation="read",
            scope=StateScope.GLOBAL,
            key="secret",
            success=False,
        )
        assert entry.success is False

    def test_get_audit_trail_returns_all(self):
        self.logger.log_access("a", "read", StateScope.WORKFLOW, "k1")
        self.logger.log_access("b", "write", StateScope.STEP, "k2")
        trail = self.logger.get_audit_trail()
        assert len(trail) == 2

    def test_filter_by_scope(self):
        self.logger.log_access("a", "read", StateScope.WORKFLOW, "k1")
        self.logger.log_access("a", "write", StateScope.STEP, "k2")
        self.logger.log_access("a", "read", StateScope.WORKFLOW, "k3")
        trail = self.logger.get_audit_trail(scope=StateScope.WORKFLOW)
        assert len(trail) == 2
        assert all(e.scope == StateScope.WORKFLOW for e in trail)

    def test_filter_by_caller_id(self):
        self.logger.log_access("alice", "read", StateScope.WORKFLOW, "k1")
        self.logger.log_access("bob", "write", StateScope.WORKFLOW, "k2")
        trail = self.logger.get_audit_trail(caller_id="bob")
        assert len(trail) == 1
        assert trail[0].caller_id == "bob"

    def test_filter_by_scope_and_caller(self):
        self.logger.log_access("alice", "read", StateScope.WORKFLOW, "k1")
        self.logger.log_access("alice", "write", StateScope.STEP, "k2")
        self.logger.log_access("bob", "read", StateScope.WORKFLOW, "k3")
        trail = self.logger.get_audit_trail(scope=StateScope.WORKFLOW, caller_id="alice")
        assert len(trail) == 1
        assert trail[0].key == "k1"

    def test_limit(self):
        for i in range(10):
            self.logger.log_access("a", "read", StateScope.WORKFLOW, f"k{i}")
        trail = self.logger.get_audit_trail(limit=3)
        assert len(trail) == 3

    def test_chronological_order(self):
        self.logger.log_access("a", "read", StateScope.WORKFLOW, "first")
        self.logger.log_access("a", "read", StateScope.WORKFLOW, "second")
        trail = self.logger.get_audit_trail()
        assert trail[0].key == "first"
        assert trail[1].key == "second"
        assert trail[0].timestamp <= trail[1].timestamp

    def test_clear(self):
        self.logger.log_access("a", "read", StateScope.WORKFLOW, "k1")
        assert len(self.logger.get_audit_trail()) == 1
        self.logger.clear()
        assert len(self.logger.get_audit_trail()) == 0
