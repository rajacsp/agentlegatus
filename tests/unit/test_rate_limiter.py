"""Unit tests for per-workflow rate limiting.

Requirements: 20.10
"""

import time

import pytest

from agentlegatus.security.rate_limiter import (
    RateLimitConfig,
    RateLimitExceededError,
    RateLimiter,
)


class TestRateLimiter:

    def setup_method(self):
        self.limiter = RateLimiter()

    # -- basic rate limiting --------------------------------------------------

    def test_allows_up_to_max_requests(self):
        self.limiter.configure("wf-1", RateLimitConfig(max_requests=3, window_seconds=10))
        assert self.limiter.check("wf-1") is True
        assert self.limiter.check("wf-1") is True
        assert self.limiter.check("wf-1") is True

    def test_denies_after_max_requests(self):
        self.limiter.configure("wf-1", RateLimitConfig(max_requests=2, window_seconds=10))
        assert self.limiter.check("wf-1") is True
        assert self.limiter.check("wf-1") is True
        assert self.limiter.check("wf-1") is False

    # -- window expiration ----------------------------------------------------

    def test_allows_again_after_window_expires(self):
        self.limiter.configure("wf-1", RateLimitConfig(max_requests=1, window_seconds=0.05))
        assert self.limiter.check("wf-1") is True
        assert self.limiter.check("wf-1") is False
        time.sleep(0.06)
        assert self.limiter.check("wf-1") is True

    # -- per-workflow isolation -----------------------------------------------

    def test_workflows_are_isolated(self):
        self.limiter.configure("wf-a", RateLimitConfig(max_requests=1, window_seconds=10))
        self.limiter.configure("wf-b", RateLimitConfig(max_requests=1, window_seconds=10))
        assert self.limiter.check("wf-a") is True
        assert self.limiter.check("wf-a") is False
        # wf-b should still be allowed
        assert self.limiter.check("wf-b") is True

    # -- unconfigured workflow ------------------------------------------------

    def test_unconfigured_workflow_always_allowed(self):
        for _ in range(100):
            assert self.limiter.check("no-config") is True

    # -- reset ----------------------------------------------------------------

    def test_reset_clears_counter(self):
        self.limiter.configure("wf-1", RateLimitConfig(max_requests=1, window_seconds=10))
        assert self.limiter.check("wf-1") is True
        assert self.limiter.check("wf-1") is False
        self.limiter.reset("wf-1")
        assert self.limiter.check("wf-1") is True

    def test_reset_noop_for_unknown_workflow(self):
        # Should not raise
        self.limiter.reset("nonexistent")

    # -- acquire --------------------------------------------------------------

    def test_acquire_succeeds_when_allowed(self):
        self.limiter.configure("wf-1", RateLimitConfig(max_requests=1, window_seconds=10))
        self.limiter.acquire("wf-1")  # should not raise

    def test_acquire_raises_when_denied(self):
        self.limiter.configure("wf-1", RateLimitConfig(max_requests=1, window_seconds=10))
        self.limiter.acquire("wf-1")
        with pytest.raises(RateLimitExceededError, match="wf-1"):
            self.limiter.acquire("wf-1")


class TestRateLimitExceededError:

    def test_error_attributes(self):
        err = RateLimitExceededError("wf-42")
        assert err.workflow_id == "wf-42"
        assert "wf-42" in str(err)
