"""Property-based tests for retry logic."""

import asyncio
import time
from typing import List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agentlegatus.core.workflow import RetryPolicy
from agentlegatus.utils.retry import execute_with_retry, execute_with_retry_sync


# Helper strategies
@st.composite
def retry_policy_strategy(draw):
    """Generate random valid RetryPolicy instances with small delays for testing."""
    max_attempts = draw(st.integers(min_value=1, max_value=5))
    backoff_multiplier = draw(st.floats(min_value=1.0, max_value=3.0))
    initial_delay = draw(st.floats(min_value=0.001, max_value=0.05))
    max_delay = draw(st.floats(min_value=initial_delay, max_value=0.2))
    
    return RetryPolicy(
        max_attempts=max_attempts,
        backoff_multiplier=backoff_multiplier,
        initial_delay=initial_delay,
        max_delay=max_delay,
    )


# Property 7: Retry Attempt Limit
@pytest.mark.asyncio
@given(
    retry_policy_strategy(),
    st.integers(min_value=1, max_value=20),
)
@settings(max_examples=20, deadline=3000)
async def test_property_7_retry_attempt_limit(
    retry_policy: RetryPolicy, failure_count: int
):
    """
    Property 7: Retry Attempt Limit
    
    For any operation with a RetryPolicy, the operation is attempted 
    at most max_attempts times before raising an exception.
    
    Validates: Requirements 6.1, 6.5
    """
    attempt_count = 0
    
    async def failing_operation():
        nonlocal attempt_count
        attempt_count += 1
        raise ValueError(f"Attempt {attempt_count} failed")
    
    # Execute with retry and expect failure
    with pytest.raises(ValueError) as exc_info:
        await execute_with_retry(
            failing_operation,
            retry_policy=retry_policy,
            operation_name="test_operation",
        )
    
    # Verify the operation was attempted exactly max_attempts times
    assert attempt_count == retry_policy.max_attempts, (
        f"Operation should be attempted exactly {retry_policy.max_attempts} times, "
        f"but was attempted {attempt_count} times"
    )
    
    # Verify the last exception was raised
    assert f"Attempt {retry_policy.max_attempts} failed" in str(exc_info.value)


@pytest.mark.asyncio
@given(
    retry_policy_strategy(),
    st.integers(min_value=1, max_value=5),
)
@settings(max_examples=20, deadline=3000)
async def test_property_7_retry_success_before_limit(
    retry_policy: RetryPolicy, success_on_attempt: int
):
    """
    Property 7 (Success Case): Retry Attempt Limit - Success Before Limit
    
    For any operation with a RetryPolicy, if the operation succeeds 
    before max_attempts, no further attempts are made.
    
    Validates: Requirements 6.6
    """
    # Ensure success happens within max_attempts
    success_on_attempt = min(success_on_attempt, retry_policy.max_attempts)
    
    attempt_count = 0
    
    async def eventually_succeeding_operation():
        nonlocal attempt_count
        attempt_count += 1
        
        if attempt_count < success_on_attempt:
            raise ValueError(f"Attempt {attempt_count} failed")
        
        return f"Success on attempt {attempt_count}"
    
    # Execute with retry
    result = await execute_with_retry(
        eventually_succeeding_operation,
        retry_policy=retry_policy,
        operation_name="test_operation",
    )
    
    # Verify the operation succeeded on the expected attempt
    assert attempt_count == success_on_attempt, (
        f"Operation should succeed on attempt {success_on_attempt}, "
        f"but succeeded on attempt {attempt_count}"
    )
    
    assert result == f"Success on attempt {success_on_attempt}"


# Property 8: Retry Exponential Backoff
@pytest.mark.asyncio
@given(retry_policy_strategy())
@settings(max_examples=15, deadline=5000)
async def test_property_8_retry_exponential_backoff(retry_policy: RetryPolicy):
    """
    Property 8: Retry Exponential Backoff
    
    For any operation with a RetryPolicy, the delay between retry attempts 
    increases exponentially by backoff_multiplier, capped at max_delay.
    
    Validates: Requirements 6.2, 6.3, 6.4
    """
    # Skip if max_attempts is 1 (no retries)
    if retry_policy.max_attempts == 1:
        return
    
    attempt_times: List[float] = []
    
    async def failing_operation():
        attempt_times.append(time.time())
        raise ValueError("Operation failed")
    
    # Execute with retry and expect failure
    with pytest.raises(ValueError):
        await execute_with_retry(
            failing_operation,
            retry_policy=retry_policy,
            operation_name="test_operation",
        )
    
    # Verify we have the expected number of attempts
    assert len(attempt_times) == retry_policy.max_attempts
    
    # Calculate actual delays between attempts
    actual_delays = [
        attempt_times[i + 1] - attempt_times[i]
        for i in range(len(attempt_times) - 1)
    ]
    
    # Calculate expected delays
    expected_delays = []
    delay = retry_policy.initial_delay
    for i in range(retry_policy.max_attempts - 1):
        expected_delays.append(delay)
        # Calculate next delay with exponential backoff
        delay = min(
            delay * retry_policy.backoff_multiplier,
            retry_policy.max_delay,
        )
    
    # Verify delays match expected pattern (with tolerance for timing variance)
    tolerance = 0.05  # 50ms tolerance for timing variance
    for i, (actual, expected) in enumerate(zip(actual_delays, expected_delays)):
        assert abs(actual - expected) < tolerance, (
            f"Delay {i + 1} should be approximately {expected:.3f}s, "
            f"but was {actual:.3f}s (difference: {abs(actual - expected):.3f}s)"
        )


@pytest.mark.asyncio
@given(
    st.integers(min_value=2, max_value=5),
    st.floats(min_value=2.0, max_value=3.0),
    st.floats(min_value=0.001, max_value=0.05),
    st.floats(min_value=0.1, max_value=0.3),
)
@settings(max_examples=15, deadline=5000)
async def test_property_8_backoff_respects_max_delay(
    max_attempts: int,
    backoff_multiplier: float,
    initial_delay: float,
    max_delay: float,
):
    """
    Property 8 (Max Delay): Retry Exponential Backoff - Max Delay Cap
    
    For any operation with a RetryPolicy, the delay between retry attempts 
    never exceeds max_delay, even with exponential backoff.
    
    Validates: Requirements 6.4
    """
    # Ensure max_delay is at least initial_delay
    max_delay = max(max_delay, initial_delay)
    
    retry_policy = RetryPolicy(
        max_attempts=max_attempts,
        backoff_multiplier=backoff_multiplier,
        initial_delay=initial_delay,
        max_delay=max_delay,
    )
    
    attempt_times: List[float] = []
    
    async def failing_operation():
        attempt_times.append(time.time())
        raise ValueError("Operation failed")
    
    # Execute with retry and expect failure
    with pytest.raises(ValueError):
        await execute_with_retry(
            failing_operation,
            retry_policy=retry_policy,
            operation_name="test_operation",
        )
    
    # Calculate actual delays between attempts
    actual_delays = [
        attempt_times[i + 1] - attempt_times[i]
        for i in range(len(attempt_times) - 1)
    ]
    
    # Verify no delay exceeds max_delay (with tolerance)
    tolerance = 0.05  # 50ms tolerance
    for i, delay in enumerate(actual_delays):
        assert delay <= max_delay + tolerance, (
            f"Delay {i + 1} should not exceed max_delay ({max_delay:.3f}s), "
            f"but was {delay:.3f}s"
        )


# Synchronous version tests
@given(retry_policy_strategy())
@settings(max_examples=15, deadline=3000)
def test_property_7_retry_attempt_limit_sync(retry_policy: RetryPolicy):
    """
    Property 7 (Sync): Retry Attempt Limit - Synchronous Version
    
    For any synchronous operation with a RetryPolicy, the operation is 
    attempted at most max_attempts times before raising an exception.
    
    Validates: Requirements 6.1, 6.5
    """
    attempt_count = 0
    
    def failing_operation():
        nonlocal attempt_count
        attempt_count += 1
        raise ValueError(f"Attempt {attempt_count} failed")
    
    # Execute with retry and expect failure
    with pytest.raises(ValueError) as exc_info:
        execute_with_retry_sync(
            failing_operation,
            retry_policy=retry_policy,
            operation_name="test_operation",
        )
    
    # Verify the operation was attempted exactly max_attempts times
    assert attempt_count == retry_policy.max_attempts, (
        f"Operation should be attempted exactly {retry_policy.max_attempts} times, "
        f"but was attempted {attempt_count} times"
    )
    
    # Verify the last exception was raised
    assert f"Attempt {retry_policy.max_attempts} failed" in str(exc_info.value)


@given(retry_policy_strategy())
@settings(max_examples=10, deadline=5000)
def test_property_8_retry_exponential_backoff_sync(retry_policy: RetryPolicy):
    """
    Property 8 (Sync): Retry Exponential Backoff - Synchronous Version
    
    For any synchronous operation with a RetryPolicy, the delay between 
    retry attempts increases exponentially by backoff_multiplier, 
    capped at max_delay.
    
    Validates: Requirements 6.2, 6.3, 6.4
    """
    # Skip if max_attempts is 1 (no retries)
    if retry_policy.max_attempts == 1:
        return
    
    attempt_times: List[float] = []
    
    def failing_operation():
        attempt_times.append(time.time())
        raise ValueError("Operation failed")
    
    # Execute with retry and expect failure
    with pytest.raises(ValueError):
        execute_with_retry_sync(
            failing_operation,
            retry_policy=retry_policy,
            operation_name="test_operation",
        )
    
    # Verify we have the expected number of attempts
    assert len(attempt_times) == retry_policy.max_attempts
    
    # Calculate actual delays between attempts
    actual_delays = [
        attempt_times[i + 1] - attempt_times[i]
        for i in range(len(attempt_times) - 1)
    ]
    
    # Calculate expected delays
    expected_delays = []
    delay = retry_policy.initial_delay
    for i in range(retry_policy.max_attempts - 1):
        expected_delays.append(delay)
        # Calculate next delay with exponential backoff
        delay = min(
            delay * retry_policy.backoff_multiplier,
            retry_policy.max_delay,
        )
    
    # Verify delays match expected pattern (with tolerance for timing variance)
    tolerance = 0.05  # 50ms tolerance for timing variance
    for i, (actual, expected) in enumerate(zip(actual_delays, expected_delays)):
        assert abs(actual - expected) < tolerance, (
            f"Delay {i + 1} should be approximately {expected:.3f}s, "
            f"but was {actual:.3f}s (difference: {abs(actual - expected):.3f}s)"
        )


# Edge case tests
@pytest.mark.asyncio
async def test_retry_with_default_policy():
    """Test retry with default RetryPolicy."""
    attempt_count = 0
    
    async def failing_operation():
        nonlocal attempt_count
        attempt_count += 1
        raise ValueError("Operation failed")
    
    # Execute with default retry policy
    with pytest.raises(ValueError):
        await execute_with_retry(
            failing_operation,
            operation_name="test_operation",
        )
    
    # Default policy has max_attempts=3
    assert attempt_count == 3


@pytest.mark.asyncio
async def test_retry_with_invalid_policy():
    """Test retry with invalid RetryPolicy raises ValueError."""
    invalid_policy = RetryPolicy(
        max_attempts=0,  # Invalid: must be at least 1
        backoff_multiplier=2.0,
        initial_delay=1.0,
        max_delay=60.0,
    )
    
    async def dummy_operation():
        return "success"
    
    # Should raise ValueError due to invalid policy
    with pytest.raises(ValueError) as exc_info:
        await execute_with_retry(
            dummy_operation,
            retry_policy=invalid_policy,
            operation_name="test_operation",
        )
    
    assert "Invalid retry policy" in str(exc_info.value)


@pytest.mark.asyncio
async def test_retry_immediate_success():
    """Test that successful operation on first attempt doesn't retry."""
    attempt_count = 0
    
    async def successful_operation():
        nonlocal attempt_count
        attempt_count += 1
        return "success"
    
    retry_policy = RetryPolicy(max_attempts=5)
    
    result = await execute_with_retry(
        successful_operation,
        retry_policy=retry_policy,
        operation_name="test_operation",
    )
    
    assert result == "success"
    assert attempt_count == 1, "Successful operation should only be attempted once"
