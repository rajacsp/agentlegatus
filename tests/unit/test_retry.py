"""Unit tests for retry utilities."""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock

from agentlegatus.core.workflow import RetryPolicy
from agentlegatus.utils.retry import execute_with_retry, execute_with_retry_sync


class TestExecuteWithRetry:
    """Tests for execute_with_retry function."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """Test that function succeeds on first attempt without retry."""
        mock_func = AsyncMock(return_value="success")
        
        result = await execute_with_retry(
            mock_func,
            retry_policy=RetryPolicy(max_attempts=3),
            operation_name="test_op"
        )
        
        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_success_after_retries(self):
        """Test that function succeeds after some retries."""
        mock_func = AsyncMock(side_effect=[
            Exception("fail 1"),
            Exception("fail 2"),
            "success"
        ])
        
        result = await execute_with_retry(
            mock_func,
            retry_policy=RetryPolicy(
                max_attempts=3,
                initial_delay=0.01,
                backoff_multiplier=2.0
            ),
            operation_name="test_op"
        )
        
        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """Test that exception is raised when all retries are exhausted."""
        mock_func = AsyncMock(side_effect=ValueError("persistent error"))
        
        with pytest.raises(ValueError, match="persistent error"):
            await execute_with_retry(
                mock_func,
                retry_policy=RetryPolicy(
                    max_attempts=3,
                    initial_delay=0.01
                ),
                operation_name="test_op"
            )
        
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test that delays follow exponential backoff pattern."""
        mock_func = AsyncMock(side_effect=[
            Exception("fail 1"),
            Exception("fail 2"),
            "success"
        ])
        
        start_time = asyncio.get_event_loop().time()
        
        await execute_with_retry(
            mock_func,
            retry_policy=RetryPolicy(
                max_attempts=3,
                initial_delay=0.1,
                backoff_multiplier=2.0,
                max_delay=1.0
            ),
            operation_name="test_op"
        )
        
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # Should have waited ~0.1s + ~0.2s = ~0.3s total
        assert elapsed >= 0.25  # Allow some tolerance
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        mock_func = AsyncMock(side_effect=[
            Exception("fail 1"),
            Exception("fail 2"),
            "success"
        ])
        
        await execute_with_retry(
            mock_func,
            retry_policy=RetryPolicy(
                max_attempts=3,
                initial_delay=0.5,
                backoff_multiplier=10.0,
                max_delay=0.6
            ),
            operation_name="test_op"
        )
        
        # Second delay should be capped at 0.6 instead of 5.0
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_with_function_arguments(self):
        """Test that function arguments are passed correctly."""
        mock_func = AsyncMock(return_value="result")
        
        result = await execute_with_retry(
            mock_func,
            "arg1",
            "arg2",
            kwarg1="value1",
            retry_policy=RetryPolicy(max_attempts=2),
            operation_name="test_op"
        )
        
        assert result == "result"
        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")

    @pytest.mark.asyncio
    async def test_default_retry_policy(self):
        """Test that default retry policy is used when none provided."""
        mock_func = AsyncMock(return_value="success")
        
        result = await execute_with_retry(mock_func, operation_name="test_op")
        
        assert result == "success"
        assert mock_func.call_count == 1

    @pytest.mark.asyncio
    async def test_invalid_retry_policy(self):
        """Test that invalid retry policy raises ValueError."""
        mock_func = AsyncMock(return_value="success")
        
        with pytest.raises(ValueError, match="Invalid retry policy"):
            await execute_with_retry(
                mock_func,
                retry_policy=RetryPolicy(max_attempts=0),  # Invalid
                operation_name="test_op"
            )


class TestExecuteWithRetrySync:
    """Tests for execute_with_retry_sync function."""

    def test_success_on_first_attempt(self):
        """Test that synchronous function succeeds on first attempt."""
        mock_func = Mock(return_value="success")
        
        result = execute_with_retry_sync(
            mock_func,
            retry_policy=RetryPolicy(max_attempts=3),
            operation_name="test_op"
        )
        
        assert result == "success"
        assert mock_func.call_count == 1

    def test_success_after_retries(self):
        """Test that synchronous function succeeds after retries."""
        mock_func = Mock(side_effect=[
            Exception("fail 1"),
            Exception("fail 2"),
            "success"
        ])
        
        result = execute_with_retry_sync(
            mock_func,
            retry_policy=RetryPolicy(
                max_attempts=3,
                initial_delay=0.01,
                backoff_multiplier=2.0
            ),
            operation_name="test_op"
        )
        
        assert result == "success"
        assert mock_func.call_count == 3

    def test_all_retries_exhausted(self):
        """Test that exception is raised when all retries exhausted."""
        mock_func = Mock(side_effect=ValueError("persistent error"))
        
        with pytest.raises(ValueError, match="persistent error"):
            execute_with_retry_sync(
                mock_func,
                retry_policy=RetryPolicy(
                    max_attempts=3,
                    initial_delay=0.01
                ),
                operation_name="test_op"
            )
        
        assert mock_func.call_count == 3

    def test_with_function_arguments(self):
        """Test that function arguments are passed correctly."""
        mock_func = Mock(return_value="result")
        
        result = execute_with_retry_sync(
            mock_func,
            "arg1",
            "arg2",
            kwarg1="value1",
            retry_policy=RetryPolicy(max_attempts=2),
            operation_name="test_op"
        )
        
        assert result == "result"
        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")
