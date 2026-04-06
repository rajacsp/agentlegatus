"""Input validation for workflow definitions and agent configurations.

Requirements: 20.4, 20.5
"""

from __future__ import annotations

from typing import Any

from agentlegatus.security.sanitization import (
    SanitizationError,
    detect_injection,
    is_safe_identifier,
    sanitize_string_input,
)


class ValidationError(Exception):
    """Raised when input validation fails.

    Attributes:
        errors: List of specific validation failure messages.
    """

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"Validation errors: {'; '.join(errors)}")


def _check_string_fields(
    data: dict[str, Any],
    field_names: list[str],
    errors: list[str],
    max_length: int = 1024,
) -> None:
    """Validate and sanitize string fields in *data*, appending issues to *errors*."""
    for name in field_names:
        val = data.get(name)
        if val is None:
            continue
        if not isinstance(val, str):
            errors.append(f"Field '{name}' must be a string, got {type(val).__name__}")
            continue
        try:
            sanitize_string_input(val, max_length=max_length)
        except SanitizationError as exc:
            errors.append(f"Field '{name}': {exc.message}")


def _check_identifier_fields(
    data: dict[str, Any],
    field_names: list[str],
    errors: list[str],
) -> None:
    """Validate that fields are safe identifiers."""
    for name in field_names:
        val = data.get(name)
        if val is None:
            continue
        if not isinstance(val, str):
            errors.append(f"Field '{name}' must be a string, got {type(val).__name__}")
            continue
        if not is_safe_identifier(val):
            errors.append(
                f"Field '{name}' contains unsafe characters; "
                "only alphanumeric, hyphens, and underscores are allowed"
            )


def validate_workflow_definition(data: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a workflow definition dictionary for injection and type safety.

    Checks identifier fields for unsafe characters and string fields for
    injection patterns (Req 20.4).

    Args:
        data: Raw workflow definition dictionary.

    Returns:
        Tuple of (is_valid, error_messages).
    """
    errors: list[str] = []

    if not isinstance(data, dict):
        return False, ["Workflow definition must be a dictionary"]

    # Identifier fields — must be safe identifiers
    _check_identifier_fields(data, ["workflow_id", "provider"], errors)

    # Free-text string fields — check for injection
    _check_string_fields(data, ["name", "description"], errors)

    # Validate steps list
    steps = data.get("steps")
    if steps is not None:
        if not isinstance(steps, list):
            errors.append("'steps' must be a list")
        else:
            for idx, step in enumerate(steps):
                if not isinstance(step, dict):
                    errors.append(f"Step {idx}: must be a dictionary")
                    continue
                _check_identifier_fields(step, ["step_id", "step_type"], errors)

                # Check for injection in step config values
                config = step.get("config")
                if isinstance(config, dict):
                    _scan_dict_for_injection(config, f"steps[{idx}].config", errors)

    # Scan metadata for injection
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        _scan_dict_for_injection(metadata, "metadata", errors)

    return len(errors) == 0, errors


def validate_agent_config(config: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate an agent configuration dictionary for injection and type safety.

    Args:
        config: Raw agent configuration dictionary.

    Returns:
        Tuple of (is_valid, error_messages).
    """
    errors: list[str] = []

    if not isinstance(config, dict):
        return False, ["Agent config must be a dictionary"]

    _check_identifier_fields(config, ["agent_id"], errors)
    _check_string_fields(config, ["name", "system_prompt"], errors)

    # Validate model field
    model = config.get("model")
    if model is not None and isinstance(model, str):
        if not is_safe_identifier(model):
            errors.append("Field 'model' contains unsafe characters")

    # Validate tools list entries
    tools = config.get("tools")
    if tools is not None:
        if not isinstance(tools, list):
            errors.append("'tools' must be a list")
        else:
            for i, tool in enumerate(tools):
                if not isinstance(tool, str):
                    errors.append(f"tools[{i}] must be a string")
                elif not is_safe_identifier(tool):
                    errors.append(f"tools[{i}] contains unsafe characters")

    return len(errors) == 0, errors


def _scan_dict_for_injection(
    data: dict[str, Any],
    prefix: str,
    errors: list[str],
) -> None:
    """Recursively scan dictionary string values for injection patterns."""
    for key, value in data.items():
        if isinstance(value, str):
            pattern = detect_injection(value)
            if pattern is not None:
                errors.append(f"{prefix}.{key}: potential injection pattern detected")
        elif isinstance(value, dict):
            _scan_dict_for_injection(value, f"{prefix}.{key}", errors)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, str):
                    pattern = detect_injection(item)
                    if pattern is not None:
                        errors.append(f"{prefix}.{key}[{i}]: potential injection pattern detected")
                elif isinstance(item, dict):
                    _scan_dict_for_injection(item, f"{prefix}.{key}[{i}]", errors)
