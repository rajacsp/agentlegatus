"""Property-based tests for StateManager."""

import asyncio
from typing import Any, Dict, List

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agentlegatus.core.state import (
    InMemoryStateBackend,
    StateBackend,
    StateManager,
    StateScope,
)


# Helper strategies
@st.composite
def state_key_strategy(draw):
    """Generate valid state keys."""
    return draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="_-."
    )))


@st.composite
def state_value_strategy(draw):
    """Generate various state values."""
    return draw(st.one_of(
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=100),
        st.booleans(),
        st.lists(st.integers(), max_size=10),
        st.dictionaries(st.text(max_size=20), st.integers(), max_size=10),
        st.none(),
    ))


@st.composite
def scope_strategy(draw):
    """Generate random StateScope values."""
    return draw(st.sampled_from(list(StateScope)))


@st.composite
def scope_id_strategy(draw):
    """Generate valid scope IDs."""
    return draw(st.text(min_size=1, max_size=30, alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="_-"
    )))


# Property 3: State Round-Trip Consistency
@pytest.mark.asyncio
@given(
    state_key_strategy(),
    state_value_strategy(),
    scope_strategy(),
    scope_id_strategy(),
)
@settings(max_examples=100, deadline=2000)
async def test_property_3_state_round_trip_consistency(
    key: str, value: Any, scope: StateScope, scope_id: str
):
    """
    Property 3: State Round-Trip Consistency
    
    For any state value V stored with key K in scope S, 
    retrieving K from S returns V.
    
    Validates: Requirements 8.1, 8.2
    """
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend, default_scope_id=scope_id)
    
    # Set state value
    await state_manager.set(key, value, scope=scope, scope_id=scope_id)
    
    # Get state value
    retrieved_value = await state_manager.get(key, scope=scope, scope_id=scope_id)
    
    # Verify round-trip consistency
    assert retrieved_value == value, (
        f"Retrieved value should match stored value. "
        f"Expected {value}, got {retrieved_value}"
    )


# Property 22: State Scope Isolation
@pytest.mark.asyncio
@given(
    state_key_strategy(),
    state_value_strategy(),
    st.lists(scope_strategy(), min_size=2, max_size=4, unique=True),
    scope_id_strategy(),
)
@settings(max_examples=100, deadline=2000)
async def test_property_22_state_scope_isolation(
    key: str, value: Any, scopes: List[StateScope], scope_id: str
):
    """
    Property 22: State Scope Isolation
    
    For any key K, setting K in scope S1 does not affect the value 
    of K in scope S2 where S1 ≠ S2.
    
    Validates: Requirements 8.12
    """
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend, default_scope_id=scope_id)
    
    # Set value in first scope
    target_scope = scopes[0]
    await state_manager.set(key, value, scope=target_scope, scope_id=scope_id)
    
    # Verify value exists in target scope
    retrieved = await state_manager.get(key, scope=target_scope, scope_id=scope_id)
    assert retrieved == value, "Value should exist in target scope"
    
    # Verify value does NOT exist in other scopes
    for other_scope in scopes[1:]:
        other_value = await state_manager.get(
            key, scope=other_scope, scope_id=scope_id, default="NOT_FOUND"
        )
        assert other_value == "NOT_FOUND", (
            f"Key '{key}' should not exist in scope {other_scope.value}. "
            f"Expected 'NOT_FOUND', got {other_value}"
        )


# Property 23: State Snapshot Round-Trip
@pytest.mark.asyncio
@given(
    st.dictionaries(
        state_key_strategy(),
        state_value_strategy(),
        min_size=1,
        max_size=10
    ),
    scope_strategy(),
    scope_id_strategy(),
    st.text(min_size=1, max_size=30, alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters="_-"
    )),
)
@settings(max_examples=100, deadline=2000)
async def test_property_23_state_snapshot_round_trip(
    state_data: Dict[str, Any],
    scope: StateScope,
    scope_id: str,
    snapshot_id: str,
):
    """
    Property 23: State Snapshot Round-Trip
    
    For any state S in scope SC, creating a snapshot SN and then 
    restoring SN returns the state to S.
    
    Validates: Requirements 8.9, 8.10
    """
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend, default_scope_id=scope_id)
    
    # Set initial state
    for key, value in state_data.items():
        await state_manager.set(key, value, scope=scope, scope_id=scope_id)
    
    # Create snapshot
    await state_manager.create_snapshot(snapshot_id, scope=scope, scope_id=scope_id)
    
    # Modify state (add new keys, modify existing ones, delete some)
    for key in list(state_data.keys())[:len(state_data) // 2]:
        await state_manager.delete(key, scope=scope, scope_id=scope_id)
    
    await state_manager.set("new_key_1", "new_value_1", scope=scope, scope_id=scope_id)
    await state_manager.set("new_key_2", "new_value_2", scope=scope, scope_id=scope_id)
    
    # Verify state has changed
    modified_state = await state_manager.get_all(scope=scope, scope_id=scope_id)
    assert modified_state != state_data, "State should have been modified"
    
    # Restore snapshot
    await state_manager.restore_snapshot(snapshot_id, scope=scope, scope_id=scope_id)
    
    # Verify state matches original
    restored_state = await state_manager.get_all(scope=scope, scope_id=scope_id)
    assert restored_state == state_data, (
        f"Restored state should match original state. "
        f"Expected {state_data}, got {restored_state}"
    )


# Property 30: State Update Atomicity
@pytest.mark.asyncio
@given(
    state_key_strategy(),
    st.integers(min_value=0, max_value=100),
    st.integers(min_value=1, max_value=10),
    scope_strategy(),
    scope_id_strategy(),
)
@settings(max_examples=100, deadline=3000)
async def test_property_30_state_update_atomicity(
    key: str,
    initial_value: int,
    increment: int,
    scope: StateScope,
    scope_id: str,
):
    """
    Property 30: State Update Atomicity
    
    For any state update operation using an updater function, 
    the update is atomic (read-modify-write happens without 
    interference).
    
    Validates: Requirements 8.4
    """
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend, default_scope_id=scope_id)
    
    # Set initial value
    await state_manager.set(key, initial_value, scope=scope, scope_id=scope_id)
    
    # Define updater function
    def increment_updater(current_value: Any) -> int:
        if current_value is None:
            return increment
        return current_value + increment
    
    # Perform atomic update
    updated_value = await state_manager.update(
        key, increment_updater, scope=scope, scope_id=scope_id
    )
    
    # Verify update was applied correctly
    expected_value = initial_value + increment
    assert updated_value == expected_value, (
        f"Updated value should be {expected_value}, got {updated_value}"
    )
    
    # Verify value persisted correctly
    retrieved_value = await state_manager.get(key, scope=scope, scope_id=scope_id)
    assert retrieved_value == expected_value, (
        f"Retrieved value should match updated value. "
        f"Expected {expected_value}, got {retrieved_value}"
    )


# Additional test: State Update with None initial value
@pytest.mark.asyncio
@given(
    state_key_strategy(),
    state_value_strategy(),
    scope_strategy(),
    scope_id_strategy(),
)
@settings(max_examples=50, deadline=2000)
async def test_state_update_with_none_initial_value(
    key: str, default_value: Any, scope: StateScope, scope_id: str
):
    """
    Test that update works correctly when key doesn't exist 
    (updater receives None).
    
    Validates: Requirements 8.5
    """
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend, default_scope_id=scope_id)
    
    # Define updater that handles None
    def updater(current_value: Any) -> Any:
        if current_value is None:
            return default_value
        return current_value
    
    # Update non-existent key
    updated_value = await state_manager.update(
        key, updater, scope=scope, scope_id=scope_id
    )
    
    # Verify default value was set
    assert updated_value == default_value, (
        f"Updated value should be default value {default_value}, got {updated_value}"
    )
    
    # Verify value persisted
    retrieved_value = await state_manager.get(key, scope=scope, scope_id=scope_id)
    assert retrieved_value == default_value


# Additional test: State deletion returns correct boolean
@pytest.mark.asyncio
@given(
    state_key_strategy(),
    state_value_strategy(),
    scope_strategy(),
    scope_id_strategy(),
)
@settings(max_examples=50, deadline=2000)
async def test_state_delete_return_value(
    key: str, value: Any, scope: StateScope, scope_id: str
):
    """
    Test that delete returns True when key exists and False when it doesn't.
    
    Validates: Requirements 8.6
    """
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend, default_scope_id=scope_id)
    
    # Delete non-existent key
    result = await state_manager.delete(key, scope=scope, scope_id=scope_id)
    assert result is False, "Delete should return False for non-existent key"
    
    # Set value
    await state_manager.set(key, value, scope=scope, scope_id=scope_id)
    
    # Delete existing key
    result = await state_manager.delete(key, scope=scope, scope_id=scope_id)
    assert result is True, "Delete should return True for existing key"
    
    # Verify key is gone
    retrieved = await state_manager.get(
        key, scope=scope, scope_id=scope_id, default="NOT_FOUND"
    )
    assert retrieved == "NOT_FOUND", "Key should not exist after deletion"


# Additional test: Get all returns all keys in scope
@pytest.mark.asyncio
@given(
    st.dictionaries(
        state_key_strategy(),
        state_value_strategy(),
        min_size=1,
        max_size=20
    ),
    scope_strategy(),
    scope_id_strategy(),
)
@settings(max_examples=50, deadline=2000)
async def test_get_all_returns_all_keys(
    state_data: Dict[str, Any], scope: StateScope, scope_id: str
):
    """
    Test that get_all returns all key-value pairs in a scope.
    
    Validates: Requirements 8.7
    """
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend, default_scope_id=scope_id)
    
    # Set all state values
    for key, value in state_data.items():
        await state_manager.set(key, value, scope=scope, scope_id=scope_id)
    
    # Get all state
    all_state = await state_manager.get_all(scope=scope, scope_id=scope_id)
    
    # Verify all keys are present
    assert set(all_state.keys()) == set(state_data.keys()), (
        "get_all should return all keys"
    )
    
    # Verify all values match
    for key, expected_value in state_data.items():
        assert all_state[key] == expected_value, (
            f"Value for key '{key}' should match. "
            f"Expected {expected_value}, got {all_state[key]}"
        )


# Additional test: Clear scope removes all keys
@pytest.mark.asyncio
@given(
    st.dictionaries(
        state_key_strategy(),
        state_value_strategy(),
        min_size=1,
        max_size=20
    ),
    scope_strategy(),
    scope_id_strategy(),
)
@settings(max_examples=50, deadline=2000)
async def test_clear_scope_removes_all_keys(
    state_data: Dict[str, Any], scope: StateScope, scope_id: str
):
    """
    Test that clear_scope removes all keys in a scope.
    
    Validates: Requirements 8.8
    """
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend, default_scope_id=scope_id)
    
    # Set all state values
    for key, value in state_data.items():
        await state_manager.set(key, value, scope=scope, scope_id=scope_id)
    
    # Verify state exists
    all_state = await state_manager.get_all(scope=scope, scope_id=scope_id)
    assert len(all_state) == len(state_data), "State should exist before clear"
    
    # Clear scope
    await state_manager.clear_scope(scope=scope, scope_id=scope_id)
    
    # Verify all keys are gone
    all_state_after = await state_manager.get_all(scope=scope, scope_id=scope_id)
    assert len(all_state_after) == 0, "All keys should be removed after clear_scope"
    
    # Verify individual keys return default
    for key in state_data.keys():
        retrieved = await state_manager.get(
            key, scope=scope, scope_id=scope_id, default="NOT_FOUND"
        )
        assert retrieved == "NOT_FOUND", (
            f"Key '{key}' should not exist after clear_scope"
        )


# Additional test: Scope ID isolation
@pytest.mark.asyncio
@given(
    state_key_strategy(),
    state_value_strategy(),
    scope_strategy(),
    st.lists(scope_id_strategy(), min_size=2, max_size=5, unique=True),
)
@settings(max_examples=50, deadline=2000)
async def test_scope_id_isolation(
    key: str, value: Any, scope: StateScope, scope_ids: List[str]
):
    """
    Test that different scope IDs within the same scope are isolated.
    
    Validates: Requirements 8.12
    """
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend)
    
    # Set value in first scope_id
    target_scope_id = scope_ids[0]
    await state_manager.set(key, value, scope=scope, scope_id=target_scope_id)
    
    # Verify value exists in target scope_id
    retrieved = await state_manager.get(key, scope=scope, scope_id=target_scope_id)
    assert retrieved == value, "Value should exist in target scope_id"
    
    # Verify value does NOT exist in other scope_ids
    for other_scope_id in scope_ids[1:]:
        other_value = await state_manager.get(
            key, scope=scope, scope_id=other_scope_id, default="NOT_FOUND"
        )
        assert other_value == "NOT_FOUND", (
            f"Key '{key}' should not exist in scope_id '{other_scope_id}'. "
            f"Expected 'NOT_FOUND', got {other_value}"
        )


# Additional test: Multiple snapshots
@pytest.mark.asyncio
@given(
    st.dictionaries(
        state_key_strategy(),
        state_value_strategy(),
        min_size=1,
        max_size=10
    ),
    scope_strategy(),
    scope_id_strategy(),
    st.lists(
        st.text(min_size=1, max_size=20, alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters="_-"
        )),
        min_size=2,
        max_size=5,
        unique=True
    ),
)
@settings(max_examples=50, deadline=2000)
async def test_multiple_snapshots(
    initial_state: Dict[str, Any],
    scope: StateScope,
    scope_id: str,
    snapshot_ids: List[str],
):
    """
    Test that multiple snapshots can be created and restored independently.
    
    Validates: Requirements 8.9, 8.10
    """
    backend = InMemoryStateBackend()
    state_manager = StateManager(backend, default_scope_id=scope_id)
    
    # Set initial state
    for key, value in initial_state.items():
        await state_manager.set(key, value, scope=scope, scope_id=scope_id)
    
    # Create first snapshot
    await state_manager.create_snapshot(
        snapshot_ids[0], scope=scope, scope_id=scope_id
    )
    
    # Modify state
    await state_manager.set("modified_key", "modified_value", scope=scope, scope_id=scope_id)
    
    # Create second snapshot
    await state_manager.create_snapshot(
        snapshot_ids[1], scope=scope, scope_id=scope_id
    )
    
    # Modify state again
    await state_manager.clear_scope(scope=scope, scope_id=scope_id)
    
    # Restore second snapshot
    await state_manager.restore_snapshot(
        snapshot_ids[1], scope=scope, scope_id=scope_id
    )
    
    # Verify second snapshot state
    state_after_second = await state_manager.get_all(scope=scope, scope_id=scope_id)
    assert "modified_key" in state_after_second
    assert state_after_second["modified_key"] == "modified_value"
    
    # Restore first snapshot
    await state_manager.restore_snapshot(
        snapshot_ids[0], scope=scope, scope_id=scope_id
    )
    
    # Verify first snapshot state (original state)
    state_after_first = await state_manager.get_all(scope=scope, scope_id=scope_id)
    assert state_after_first == initial_state
    assert "modified_key" not in state_after_first
