"""Property-based tests for MemoryManager and MemoryBackend."""

import asyncio
from typing import Any, Dict, List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agentlegatus.memory.base import InMemoryMemoryBackend, MemoryBackend, MemoryType
from agentlegatus.memory.manager import MemoryManager


# Helper strategies
@st.composite
def memory_key_strategy(draw):
    """Generate valid memory keys."""
    return draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="_-."
    )))


@st.composite
def memory_value_strategy(draw):
    """Generate various memory values."""
    return draw(st.one_of(
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=100),
        st.booleans(),
        st.lists(st.integers(), max_size=10),
        st.dictionaries(st.text(max_size=20), st.integers(), max_size=10),
    ))


@st.composite
def memory_type_strategy(draw):
    """Generate random MemoryType values."""
    return draw(st.sampled_from(list(MemoryType)))


# Property 12: Memory Type Isolation
@pytest.mark.asyncio
@given(
    memory_key_strategy(),
    memory_value_strategy(),
    st.lists(memory_type_strategy(), min_size=2, max_size=4, unique=True),
)
@settings(max_examples=10, deadline=2000)
async def test_property_12_memory_type_isolation(
    key: str, value: Any, memory_types: List[MemoryType]
):
    """
    Property 12: Memory Type Isolation

    For any key, value, and two different memory types, storing in one
    type and retrieving from another type returns empty results.

    Validates: Requirements 10.9, 30.1, 30.2, 30.3
    """
    backend = InMemoryMemoryBackend()
    manager = MemoryManager(backend)

    target_type = memory_types[0]

    # Store value in target memory type
    await manager.store(key, value, target_type)

    # Retrieve from target type should find the value
    results = await manager.get_recent(target_type, limit=100)
    assert value in results, (
        f"Value should be retrievable from target type {target_type.value}"
    )

    # Retrieve from other types should NOT find the value
    for other_type in memory_types[1:]:
        other_results = await manager.get_recent(other_type, limit=100)
        assert value not in other_results, (
            f"Value stored in {target_type.value} should not appear in "
            f"{other_type.value}. Got {other_results}"
        )


# Property 12 (continued): Clearing one type does not affect others
@pytest.mark.asyncio
@given(
    memory_key_strategy(),
    memory_value_strategy(),
    st.lists(memory_type_strategy(), min_size=2, max_size=4, unique=True),
)
@settings(max_examples=10, deadline=2000)
async def test_property_12_clear_isolation(
    key: str, value: Any, memory_types: List[MemoryType]
):
    """
    Property 12 (supplement): Clearing one memory type does not affect others.

    Validates: Requirements 30.3
    """
    backend = InMemoryMemoryBackend()
    manager = MemoryManager(backend)

    # Store value in ALL memory types
    for mt in memory_types:
        await manager.store(key, value, mt)

    # Clear only the first type
    cleared_type = memory_types[0]
    await manager.clear(cleared_type)

    # Cleared type should be empty
    cleared_results = await manager.get_recent(cleared_type, limit=100)
    assert len(cleared_results) == 0, (
        f"Cleared type {cleared_type.value} should have no entries"
    )

    # Other types should still have their values
    for other_type in memory_types[1:]:
        other_results = await manager.get_recent(other_type, limit=100)
        assert value in other_results, (
            f"Type {other_type.value} should still have its value after "
            f"clearing {cleared_type.value}"
        )


# Property 13: Memory Backend Round-Trip
@pytest.mark.asyncio
@given(
    memory_key_strategy(),
    memory_value_strategy(),
    memory_type_strategy(),
)
@settings(max_examples=10, deadline=2000)
async def test_property_13_memory_backend_round_trip(
    key: str, value: Any, memory_type: MemoryType
):
    """
    Property 13: Memory Backend Round-Trip

    For any memory backend, key, value, and memory type, storing then
    retrieving returns the same value.

    Validates: Requirements 10.5, 10.6
    """
    backend = InMemoryMemoryBackend()
    manager = MemoryManager(backend)

    # Store value
    await manager.store(key, value, memory_type)

    # Retrieve using key as query (InMemoryMemoryBackend matches on key substring)
    results = await backend.retrieve(key, memory_type, limit=10)

    assert len(results) >= 1, (
        f"Should retrieve at least one result for key '{key}' "
        f"in type {memory_type.value}"
    )
    assert results[0] == value, (
        f"Retrieved value should match stored value. "
        f"Expected {value!r}, got {results[0]!r}"
    )


# Additional: store_short_term round-trip
@pytest.mark.asyncio
@given(
    memory_key_strategy(),
    memory_value_strategy(),
)
@settings(max_examples=10, deadline=2000)
async def test_store_short_term_round_trip(key: str, value: Any):
    """
    Test that store_short_term stores in SHORT_TERM type and is retrievable.

    Validates: Requirements 10.1
    """
    backend = InMemoryMemoryBackend()
    manager = MemoryManager(backend)

    await manager.store_short_term(key, value)

    results = await backend.retrieve(key, MemoryType.SHORT_TERM, limit=10)
    assert len(results) >= 1, "Should retrieve short-term memory"
    assert results[0] == value

    # Should NOT appear in long-term
    lt_results = await backend.retrieve(key, MemoryType.LONG_TERM, limit=10)
    assert value not in lt_results, "Short-term memory should not leak to long-term"


# Additional: store_long_term round-trip
@pytest.mark.asyncio
@given(
    memory_key_strategy(),
    memory_value_strategy(),
)
@settings(max_examples=10, deadline=2000)
async def test_store_long_term_round_trip(key: str, value: Any):
    """
    Test that store_long_term stores in LONG_TERM type and is retrievable.

    Validates: Requirements 10.2
    """
    backend = InMemoryMemoryBackend()
    manager = MemoryManager(backend)

    await manager.store_long_term(key, value)

    results = await backend.retrieve(key, MemoryType.LONG_TERM, limit=10)
    assert len(results) >= 1, "Should retrieve long-term memory"
    assert results[0] == value

    # Should NOT appear in short-term
    st_results = await backend.retrieve(key, MemoryType.SHORT_TERM, limit=10)
    assert value not in st_results, "Long-term memory should not leak to short-term"


# Additional: delete returns correct boolean
@pytest.mark.asyncio
@given(
    memory_key_strategy(),
    memory_value_strategy(),
    memory_type_strategy(),
)
@settings(max_examples=10, deadline=2000)
async def test_memory_delete_return_value(
    key: str, value: Any, memory_type: MemoryType
):
    """
    Test that delete returns True when key exists and False when it doesn't.

    Validates: Requirements 10.7
    """
    backend = InMemoryMemoryBackend()
    manager = MemoryManager(backend)

    # Delete non-existent key
    result = await manager.delete(key, memory_type)
    assert result is False, "Delete should return False for non-existent key"

    # Store then delete
    await manager.store(key, value, memory_type)
    result = await manager.delete(key, memory_type)
    assert result is True, "Delete should return True for existing key"

    # Verify key is gone
    results = await backend.retrieve(key, memory_type, limit=10)
    assert value not in results, "Value should not exist after deletion"


# Additional: get_recent returns newest first
@pytest.mark.asyncio
@given(
    st.lists(
        st.tuples(memory_key_strategy(), memory_value_strategy()),
        min_size=2,
        max_size=10,
        unique_by=lambda pair: pair[0],
    ),
    memory_type_strategy(),
)
@settings(max_examples=10, deadline=2000)
async def test_get_recent_ordering(
    kv_pairs: List[tuple], memory_type: MemoryType
):
    """
    Test that get_recent returns memories in most-recent-first order.

    Validates: Requirements 10.4
    """
    backend = InMemoryMemoryBackend()
    manager = MemoryManager(backend)

    # Store values in order
    for key, value in kv_pairs:
        await manager.store(key, value, memory_type)

    results = await manager.get_recent(memory_type, limit=len(kv_pairs))

    # The last stored value should be first in results
    last_key, last_value = kv_pairs[-1]
    assert results[0] == last_value, (
        f"Most recently stored value should be first. "
        f"Expected {last_value!r}, got {results[0]!r}"
    )
