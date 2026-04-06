"""Structured logging for AgentLegatus.

Provides structured JSON logging via structlog with automatic context
propagation for workflow_id, execution_id, step_id, and trace_id.
Falls back to stdlib logging when structlog is not installed.

Requirements: 27.1-27.6
"""

import logging
import sys
import traceback
from typing import Any, Union

try:
    import structlog

    _HAS_STRUCTLOG = True
except ImportError:
    _HAS_STRUCTLOG = False


# Module-level context for stdlib fallback
_global_context: dict[str, Any] = {}

# Context fields that should appear in every log entry
CONTEXT_FIELDS: list[str] = [
    "workflow_id",
    "execution_id",
    "step_id",
    "trace_id",
    "correlation_id",
]


def _add_default_context_fields(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor that ensures context fields are always present.

    Missing context fields are set to None so log entries have a
    consistent schema for downstream parsing (Req 27.2, 27.3, 27.5).
    """
    for field in CONTEXT_FIELDS:
        event_dict.setdefault(field, None)
    return event_dict


def _format_exception_info(
    logger: Any, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Structlog processor that formats exception info consistently (Req 27.4).

    When exc_info is present, extracts error_type, error_message, and
    stack_trace into top-level fields for easy parsing.
    """
    exc_info = event_dict.pop("exc_info", None)
    if exc_info:
        if isinstance(exc_info, BaseException):
            exc = exc_info
        elif isinstance(exc_info, tuple):
            exc = exc_info[1]
        else:
            exc = None

        if exc is not None:
            event_dict.setdefault("error_type", type(exc).__name__)
            event_dict.setdefault("error_message", str(exc))
            event_dict.setdefault(
                "stack_trace",
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            )
    return event_dict


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    global_context: dict[str, Any] | None = None,
) -> None:
    """Set up structured logging for AgentLegatus.

    Configures structlog with JSON formatting by default (Req 27.1).
    Falls back to stdlib logging when structlog is not installed.

    Args:
        level: Log level — DEBUG, INFO, WARNING, ERROR, CRITICAL (Req 27.6)
        json_format: Use JSON output. Defaults to True per Req 27.1.
        global_context: Optional fields added to every log entry.
    """
    global _global_context
    if global_context:
        _global_context.update(global_context)

    log_level = getattr(logging, level.upper(), logging.INFO)

    if _HAS_STRUCTLOG:
        _setup_structlog(log_level, json_format)
    else:
        _setup_stdlib_logging(log_level)


def _setup_structlog(log_level: int, json_format: bool) -> None:
    """Configure structlog with JSON formatting and context processors."""
    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_default_context_fields,
        _format_exception_info,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def _setup_stdlib_logging(log_level: int) -> None:
    """Configure stdlib logging as fallback."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(
    name: str, **initial_context: Any
) -> Union["structlog.stdlib.BoundLogger", logging.Logger]:
    """Get a logger instance with optional initial context.

    Args:
        name: Logger name (typically ``__name__``)
        **initial_context: Fields bound to this logger instance

    Returns:
        structlog BoundLogger when structlog is available, else stdlib Logger.
    """
    context = {**_global_context, **initial_context}

    if _HAS_STRUCTLOG:
        logger = structlog.get_logger(name)
        if context:
            logger = logger.bind(**context)
        return logger

    return logging.getLogger(name)


def bind_context(**kwargs: Any) -> None:
    """Bind context variables to all subsequent log entries.

    Common fields: workflow_id, execution_id, step_id, trace_id, correlation_id

    Args:
        **kwargs: Context key-value pairs to bind.
    """
    if _HAS_STRUCTLOG:
        structlog.contextvars.bind_contextvars(**kwargs)
    else:
        _global_context.update(kwargs)


def unbind_context(*keys: str) -> None:
    """Remove context variables from subsequent log entries.

    Args:
        *keys: Context keys to remove.
    """
    if _HAS_STRUCTLOG:
        structlog.contextvars.unbind_contextvars(*keys)
    else:
        for key in keys:
            _global_context.pop(key, None)


def clear_context() -> None:
    """Clear all bound context variables."""
    if _HAS_STRUCTLOG:
        structlog.contextvars.clear_contextvars()
    else:
        _global_context.clear()


def log_error(
    logger: Any,
    message: str,
    error: Exception,
    **extra: Any,
) -> None:
    """Log an error with full context including stack trace (Req 27.4).

    Args:
        logger: Logger instance (structlog or stdlib)
        message: Error description
        error: The exception
        **extra: Additional context (workflow_id, step_id, etc.)
    """
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "stack_trace": traceback.format_exc(),
        **extra,
    }

    if _HAS_STRUCTLOG and hasattr(logger, "error"):
        logger.error(message, **error_info)
    else:
        logger.error(f"{message}: {error_info}", exc_info=True)
