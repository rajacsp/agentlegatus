"""Rate limiting per workflow.

Requirements: 20.10
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting a workflow."""

    max_requests: int
    window_seconds: float


class RateLimitExceededError(Exception):
    """Raised when a workflow exceeds its configured rate limit."""

    def __init__(self, workflow_id: str) -> None:
        self.workflow_id = workflow_id
        super().__init__(f"Rate limit exceeded for workflow '{workflow_id}'")


class RateLimiter:
    """Sliding-window rate limiter enforced per workflow."""

    def __init__(self) -> None:
        self._configs: dict[str, RateLimitConfig] = {}
        self._timestamps: dict[str, list[float]] = defaultdict(list)

    def configure(self, workflow_id: str, config: RateLimitConfig) -> None:
        """Set the rate limit for *workflow_id*."""
        self._configs[workflow_id] = config

    def check(self, workflow_id: str) -> bool:
        """Return True if a request is allowed, False if rate-limited.

        Unconfigured workflows are always allowed.
        """
        config = self._configs.get(workflow_id)
        if config is None:
            return True

        now = time.monotonic()
        cutoff = now - config.window_seconds
        timestamps = self._timestamps[workflow_id]

        # Remove expired timestamps
        self._timestamps[workflow_id] = [t for t in timestamps if t > cutoff]
        timestamps = self._timestamps[workflow_id]

        if len(timestamps) >= config.max_requests:
            return False

        timestamps.append(now)
        return True

    def acquire(self, workflow_id: str) -> None:
        """Like :meth:`check` but raises on denial."""
        if not self.check(workflow_id):
            raise RateLimitExceededError(workflow_id)

    def reset(self, workflow_id: str) -> None:
        """Reset the request counter for *workflow_id*."""
        self._timestamps.pop(workflow_id, None)
