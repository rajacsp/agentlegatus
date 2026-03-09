"""Retry utilities with exponential backoff."""

import asyncio
import logging
from typing import Any, Callable, Optional, TypeVar, Union

from agentlegatus.core.workflow import RetryPolicy

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def execute_with_retry(
    func: Callable[..., Any],
    *args: Any,
    retry_policy: Optional[RetryPolicy] = None,
    operation_name: str = "operation",
    **kwargs: Any,
) -> Any:
    """
    Execute a function with retry logic and exponential backoff.

    Args:
        func: The function to execute (can be sync or async)
        *args: Positional arguments to pass to the function
        retry_policy: Retry configuration. If None, uses default policy
        operation_name: Name of the operation for logging purposes
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function execution

    Raises:
        The last exception if all retry attempts are exhausted
    """
    # Use default retry policy if none provided
    if retry_policy is None:
        retry_policy = RetryPolicy()

    # Validate retry policy
    is_valid, errors = retry_policy.validate()
    if not is_valid:
        raise ValueError(f"Invalid retry policy: {', '.join(errors)}")

    last_exception: Optional[Exception] = None
    delay = retry_policy.initial_delay

    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            # Execute the function (handle both sync and async)
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Success - log if this was a retry
            if attempt > 1:
                logger.info(
                    f"{operation_name} succeeded on attempt {attempt}/{retry_policy.max_attempts}"
                )

            return result

        except Exception as e:
            last_exception = e

            # Log the failure
            if attempt < retry_policy.max_attempts:
                logger.warning(
                    f"{operation_name} failed on attempt {attempt}/{retry_policy.max_attempts}: "
                    f"{type(e).__name__}: {str(e)}. Retrying in {delay:.2f}s..."
                )

                # Wait before retrying
                await asyncio.sleep(delay)

                # Calculate next delay with exponential backoff
                delay = min(
                    delay * retry_policy.backoff_multiplier,
                    retry_policy.max_delay,
                )
            else:
                # Final attempt failed
                logger.error(
                    f"{operation_name} failed after {retry_policy.max_attempts} attempts: "
                    f"{type(e).__name__}: {str(e)}"
                )

    # All retries exhausted - raise the last exception
    if last_exception:
        raise last_exception
    else:
        # This should never happen, but handle it gracefully
        raise RuntimeError(f"{operation_name} failed with no exception captured")


def execute_with_retry_sync(
    func: Callable[..., T],
    *args: Any,
    retry_policy: Optional[RetryPolicy] = None,
    operation_name: str = "operation",
    **kwargs: Any,
) -> T:
    """
    Synchronous wrapper for execute_with_retry.

    Args:
        func: The synchronous function to execute
        *args: Positional arguments to pass to the function
        retry_policy: Retry configuration. If None, uses default policy
        operation_name: Name of the operation for logging purposes
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function execution

    Raises:
        The last exception if all retry attempts are exhausted
    """
    # Use default retry policy if none provided
    if retry_policy is None:
        retry_policy = RetryPolicy()

    # Validate retry policy
    is_valid, errors = retry_policy.validate()
    if not is_valid:
        raise ValueError(f"Invalid retry policy: {', '.join(errors)}")

    last_exception: Optional[Exception] = None
    delay = retry_policy.initial_delay

    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            result = func(*args, **kwargs)

            # Success - log if this was a retry
            if attempt > 1:
                logger.info(
                    f"{operation_name} succeeded on attempt {attempt}/{retry_policy.max_attempts}"
                )

            return result

        except Exception as e:
            last_exception = e

            # Log the failure
            if attempt < retry_policy.max_attempts:
                logger.warning(
                    f"{operation_name} failed on attempt {attempt}/{retry_policy.max_attempts}: "
                    f"{type(e).__name__}: {str(e)}. Retrying in {delay:.2f}s..."
                )

                # Wait before retrying (synchronous sleep)
                import time
                time.sleep(delay)

                # Calculate next delay with exponential backoff
                delay = min(
                    delay * retry_policy.backoff_multiplier,
                    retry_policy.max_delay,
                )
            else:
                # Final attempt failed
                logger.error(
                    f"{operation_name} failed after {retry_policy.max_attempts} attempts: "
                    f"{type(e).__name__}: {str(e)}"
                )

    # All retries exhausted - raise the last exception
    if last_exception:
        raise last_exception
    else:
        # This should never happen, but handle it gracefully
        raise RuntimeError(f"{operation_name} failed with no exception captured")
