"""Security module for input validation, sanitization, secure HTTP, access control, audit, PII, and rate limiting.

Requirements: 20.3, 20.4, 20.5, 20.7, 20.8, 20.9, 20.10
"""

from agentlegatus.security.access_control import (
    AccessController,
    AccessDeniedError,
    AccessPolicy,
    Operation,
)
from agentlegatus.security.audit import (
    AuditEntry,
    AuditLogger,
)
from agentlegatus.security.http_client import (
    InsecureURLError,
    SecureHTTPClient,
    create_secure_client,
    validate_url,
)
from agentlegatus.security.pii import (
    PIIDetector,
)
from agentlegatus.security.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    RateLimitExceededError,
)
from agentlegatus.security.sanitization import (
    SanitizationError,
    detect_injection,
    is_safe_identifier,
    sanitize_file_path,
    sanitize_string_input,
)
from agentlegatus.security.validation import (
    ValidationError,
    validate_agent_config,
    validate_workflow_definition,
)

__all__ = [
    "AccessController",
    "AccessDeniedError",
    "AccessPolicy",
    "AuditEntry",
    "AuditLogger",
    "InsecureURLError",
    "Operation",
    "PIIDetector",
    "RateLimitConfig",
    "RateLimitExceededError",
    "RateLimiter",
    "SanitizationError",
    "SecureHTTPClient",
    "ValidationError",
    "create_secure_client",
    "detect_injection",
    "is_safe_identifier",
    "sanitize_file_path",
    "sanitize_string_input",
    "validate_agent_config",
    "validate_url",
    "validate_workflow_definition",
]
