"""Unit tests for scope-based access control.

Requirements: 20.7
"""

import pytest

from agentlegatus.core.state import StateScope
from agentlegatus.security.access_control import (
    AccessController,
    AccessDeniedError,
    AccessPolicy,
    Operation,
)


class TestAccessPolicy:

    def test_default_policy_empty(self):
        policy = AccessPolicy()
        assert policy.allowed_scopes == set()
        assert policy.read_only is False

    def test_policy_with_scopes(self):
        policy = AccessPolicy(
            allowed_scopes={StateScope.WORKFLOW, StateScope.STEP},
            read_only=True,
        )
        assert StateScope.WORKFLOW in policy.allowed_scopes
        assert policy.read_only is True


class TestAccessController:

    def setup_method(self):
        self.ctrl = AccessController()

    def test_unregistered_caller_denied(self):
        assert self.ctrl.check_access("unknown", StateScope.WORKFLOW, Operation.READ) is False

    def test_registered_caller_allowed(self):
        self.ctrl.register_policy(
            "agent-1",
            AccessPolicy(allowed_scopes={StateScope.WORKFLOW}),
        )
        assert self.ctrl.check_access("agent-1", StateScope.WORKFLOW, Operation.READ) is True
        assert self.ctrl.check_access("agent-1", StateScope.WORKFLOW, Operation.WRITE) is True

    def test_scope_not_in_policy_denied(self):
        self.ctrl.register_policy(
            "agent-1",
            AccessPolicy(allowed_scopes={StateScope.STEP}),
        )
        assert self.ctrl.check_access("agent-1", StateScope.GLOBAL, Operation.READ) is False

    def test_read_only_blocks_write(self):
        self.ctrl.register_policy(
            "reader",
            AccessPolicy(allowed_scopes={StateScope.WORKFLOW}, read_only=True),
        )
        assert self.ctrl.check_access("reader", StateScope.WORKFLOW, Operation.READ) is True
        assert self.ctrl.check_access("reader", StateScope.WORKFLOW, Operation.WRITE) is False
        assert self.ctrl.check_access("reader", StateScope.WORKFLOW, Operation.DELETE) is False

    def test_enforce_access_raises(self):
        with pytest.raises(AccessDeniedError, match="Access denied"):
            self.ctrl.enforce_access("nobody", StateScope.WORKFLOW, Operation.READ)

    def test_enforce_access_passes(self):
        self.ctrl.register_policy(
            "admin",
            AccessPolicy(allowed_scopes={StateScope.GLOBAL}),
        )
        # Should not raise
        self.ctrl.enforce_access("admin", StateScope.GLOBAL, Operation.WRITE)

    def test_overwrite_policy(self):
        self.ctrl.register_policy(
            "agent-1",
            AccessPolicy(allowed_scopes={StateScope.WORKFLOW}),
        )
        assert self.ctrl.check_access("agent-1", StateScope.WORKFLOW, Operation.WRITE) is True

        # Overwrite with read-only
        self.ctrl.register_policy(
            "agent-1",
            AccessPolicy(allowed_scopes={StateScope.WORKFLOW}, read_only=True),
        )
        assert self.ctrl.check_access("agent-1", StateScope.WORKFLOW, Operation.WRITE) is False


class TestAccessDeniedError:

    def test_error_attributes(self):
        err = AccessDeniedError("caller-1", StateScope.AGENT, Operation.DELETE)
        assert err.caller_id == "caller-1"
        assert err.scope == StateScope.AGENT
        assert err.operation == Operation.DELETE
        assert "caller-1" in str(err)
        assert "delete" in str(err)
        assert "agent" in str(err)
