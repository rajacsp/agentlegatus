"""Property-based tests for ProviderRegistry."""

from typing import Any, Dict, List, Optional

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agentlegatus.providers.base import BaseProvider, ProviderCapability
from agentlegatus.providers.registry import ProviderRegistry, ProviderNotFoundError
from agentlegatus.tools.tool import Tool


# Mock provider classes for testing
class MockProviderA(BaseProvider):
    """Mock provider A for testing."""
    
    def _get_capabilities(self) -> List[ProviderCapability]:
        return [
            ProviderCapability.TOOL_CALLING,
            ProviderCapability.STATE_PERSISTENCE,
        ]
    
    async def create_agent(self, agent_config: Dict[str, Any]) -> Any:
        return {"agent_id": "mock_agent_a", "config": agent_config}
    
    async def execute_agent(
        self,
        agent: Any,
        input_data: Any,
        state: Optional[Dict[str, Any]] = None
    ) -> Any:
        return {"result": "mock_result_a", "input": input_data}
    
    async def invoke_tool(
        self,
        tool: Tool,
        tool_input: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        return {"tool": tool.name, "output": "mock_output_a"}
    
    def export_state(self) -> Dict[str, Any]:
        return {"state": "exported_a"}
    
    def import_state(self, state: Dict[str, Any]) -> None:
        pass
    
    def to_portable_graph(self, workflow: Any) -> Any:
        return {"graph": "portable_a"}
    
    def from_portable_graph(self, graph: Any) -> Any:
        return {"workflow": "provider_specific_a"}


class MockProviderB(BaseProvider):
    """Mock provider B for testing."""
    
    def _get_capabilities(self) -> List[ProviderCapability]:
        return [
            ProviderCapability.STREAMING,
            ProviderCapability.PARALLEL_EXECUTION,
        ]
    
    async def create_agent(self, agent_config: Dict[str, Any]) -> Any:
        return {"agent_id": "mock_agent_b", "config": agent_config}
    
    async def execute_agent(
        self,
        agent: Any,
        input_data: Any,
        state: Optional[Dict[str, Any]] = None
    ) -> Any:
        return {"result": "mock_result_b", "input": input_data}
    
    async def invoke_tool(
        self,
        tool: Tool,
        tool_input: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        return {"tool": tool.name, "output": "mock_output_b"}
    
    def export_state(self) -> Dict[str, Any]:
        return {"state": "exported_b"}
    
    def import_state(self, state: Dict[str, Any]) -> None:
        pass
    
    def to_portable_graph(self, workflow: Any) -> Any:
        return {"graph": "portable_b"}
    
    def from_portable_graph(self, graph: Any) -> Any:
        return {"workflow": "provider_specific_b"}


class MockProviderC(BaseProvider):
    """Mock provider C for testing."""
    
    def _get_capabilities(self) -> List[ProviderCapability]:
        return [
            ProviderCapability.HUMAN_IN_LOOP,
        ]
    
    async def create_agent(self, agent_config: Dict[str, Any]) -> Any:
        return {"agent_id": "mock_agent_c", "config": agent_config}
    
    async def execute_agent(
        self,
        agent: Any,
        input_data: Any,
        state: Optional[Dict[str, Any]] = None
    ) -> Any:
        return {"result": "mock_result_c", "input": input_data}
    
    async def invoke_tool(
        self,
        tool: Tool,
        tool_input: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        return {"tool": tool.name, "output": "mock_output_c"}
    
    def export_state(self) -> Dict[str, Any]:
        return {"state": "exported_c"}
    
    def import_state(self, state: Dict[str, Any]) -> None:
        pass
    
    def to_portable_graph(self, workflow: Any) -> Any:
        return {"graph": "portable_c"}
    
    def from_portable_graph(self, graph: Any) -> Any:
        return {"workflow": "provider_specific_c"}


# Helper strategies
@st.composite
def provider_name_strategy(draw):
    """Generate valid provider names."""
    return draw(st.text(
        min_size=1,
        max_size=30,
        alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd'),
            whitelist_characters='_-'
        )
    ))


@st.composite
def provider_config_strategy(draw):
    """Generate provider configuration dictionaries."""
    # Generate a simple config with common fields
    config = {}
    
    # Optionally add api_key
    if draw(st.booleans()):
        config["api_key"] = draw(st.text(min_size=10, max_size=50))
    
    # Optionally add model
    if draw(st.booleans()):
        config["model"] = draw(st.sampled_from([
            "gpt-4", "gpt-3.5-turbo", "claude-3", "gemini-pro"
        ]))
    
    # Optionally add temperature
    if draw(st.booleans()):
        config["temperature"] = draw(st.floats(min_value=0.0, max_value=2.0))
    
    # Optionally add max_tokens
    if draw(st.booleans()):
        config["max_tokens"] = draw(st.integers(min_value=1, max_value=4096))
    
    return config


@st.composite
def provider_class_strategy(draw):
    """Generate provider classes."""
    return draw(st.sampled_from([MockProviderA, MockProviderB, MockProviderC]))


# Property 18: Provider Registry Caching
@given(
    provider_name=provider_name_strategy(),
    provider_class=provider_class_strategy(),
    config=provider_config_strategy()
)
@settings(max_examples=50, deadline=2000)
def test_property_18_provider_registry_caching(
    provider_name: str,
    provider_class: type,
    config: Dict[str, Any]
):
    """
    Property 18: Provider Registry Caching
    
    For any provider name and configuration, requesting the same provider 
    twice returns the same instance.
    
    Validates: Requirements 4.6
    """
    registry = ProviderRegistry()
    
    # Register the provider
    registry.register_provider(provider_name, provider_class)
    
    # Get provider instance twice with the same config
    provider1 = registry.get_provider(provider_name, config)
    provider2 = registry.get_provider(provider_name, config)
    
    # Should return the exact same instance (identity check)
    assert provider1 is provider2, (
        f"Provider '{provider_name}' with config {config} should return "
        "the same instance on repeated calls"
    )
    
    # Verify it's the correct type
    assert isinstance(provider1, provider_class), (
        f"Provider should be instance of {provider_class.__name__}"
    )
    
    # Verify config is preserved
    assert provider1.config == config, "Provider config should match"
    assert provider2.config == config, "Provider config should match"


# Property 18 Extension: Different configs create different instances
@given(
    provider_name=provider_name_strategy(),
    provider_class=provider_class_strategy(),
    config1=provider_config_strategy(),
    config2=provider_config_strategy()
)
@settings(max_examples=50, deadline=2000)
def test_property_18_different_configs_different_instances(
    provider_name: str,
    provider_class: type,
    config1: Dict[str, Any],
    config2: Dict[str, Any]
):
    """
    Property 18 Extension: Different configurations create different instances.
    
    For any provider name with different configurations, requesting the 
    provider returns different instances.
    
    Validates: Requirements 4.6
    """
    # Ensure configs are actually different
    assume(config1 != config2)
    
    registry = ProviderRegistry()
    
    # Register the provider
    registry.register_provider(provider_name, provider_class)
    
    # Get provider instances with different configs
    provider1 = registry.get_provider(provider_name, config1)
    provider2 = registry.get_provider(provider_name, config2)
    
    # Should return different instances
    assert provider1 is not provider2, (
        f"Provider '{provider_name}' with different configs should return "
        "different instances"
    )
    
    # Verify both are correct type
    assert isinstance(provider1, provider_class)
    assert isinstance(provider2, provider_class)
    
    # Verify configs are preserved correctly
    assert provider1.config == config1
    assert provider2.config == config2


# Property 19: Provider Registry Completeness
@given(
    st.lists(
        st.tuples(provider_name_strategy(), provider_class_strategy()),
        min_size=1,
        max_size=20,
        unique_by=lambda x: x[0]  # Unique by provider name
    )
)
@settings(max_examples=50, deadline=2000)
def test_property_19_provider_registry_completeness(
    providers: List[tuple[str, type]]
):
    """
    Property 19: Provider Registry Completeness
    
    For any registered provider, the provider name appears in the list 
    of all providers.
    
    Validates: Requirements 4.1, 4.4
    """
    registry = ProviderRegistry()
    
    # Register all providers
    for provider_name, provider_class in providers:
        registry.register_provider(provider_name, provider_class)
    
    # Get list of all providers
    provider_list = registry.list_providers()
    
    # Verify all registered providers appear in the list
    for provider_name, _ in providers:
        assert provider_name in provider_list, (
            f"Registered provider '{provider_name}' should appear in list_providers()"
        )
    
    # Verify the count matches
    assert len(provider_list) == len(providers), (
        f"list_providers() should return exactly {len(providers)} providers, "
        f"got {len(provider_list)}"
    )
    
    # Verify no extra providers in the list
    expected_names = {name for name, _ in providers}
    actual_names = set(provider_list)
    assert actual_names == expected_names, (
        "list_providers() should return exactly the registered provider names"
    )


# Property 19 Extension: Unregistered providers don't appear in list
@given(
    registered_providers=st.lists(
        st.tuples(provider_name_strategy(), provider_class_strategy()),
        min_size=1,
        max_size=10,
        unique_by=lambda x: x[0]
    ),
    unregistered_name=provider_name_strategy()
)
@settings(max_examples=50, deadline=2000)
def test_property_19_unregistered_providers_not_in_list(
    registered_providers: List[tuple[str, type]],
    unregistered_name: str
):
    """
    Property 19 Extension: Unregistered providers don't appear in list.
    
    For any provider name that was not registered, it should not appear 
    in the list of providers.
    
    Validates: Requirements 4.1, 4.4
    """
    # Ensure unregistered_name is not in registered providers
    registered_names = {name for name, _ in registered_providers}
    assume(unregistered_name not in registered_names)
    
    registry = ProviderRegistry()
    
    # Register providers
    for provider_name, provider_class in registered_providers:
        registry.register_provider(provider_name, provider_class)
    
    # Get list of providers
    provider_list = registry.list_providers()
    
    # Verify unregistered provider is not in the list
    assert unregistered_name not in provider_list, (
        f"Unregistered provider '{unregistered_name}' should not appear in list_providers()"
    )


# Additional test: Provider registry operations consistency
@given(
    providers=st.lists(
        st.tuples(provider_name_strategy(), provider_class_strategy()),
        min_size=2,
        max_size=10,
        unique_by=lambda x: x[0]
    ),
    unregister_index=st.integers(min_value=0, max_value=9)
)
@settings(max_examples=30, deadline=2000)
def test_provider_registry_unregister_consistency(
    providers: List[tuple[str, type]],
    unregister_index: int
):
    """
    Test that unregistering a provider removes it from the list and cache.
    
    Validates: Requirements 4.1, 4.4, 4.6
    """
    unregister_index = unregister_index % len(providers)
    
    registry = ProviderRegistry()
    
    # Register all providers
    for provider_name, provider_class in providers:
        registry.register_provider(provider_name, provider_class)
    
    # Get instances to populate cache
    for provider_name, _ in providers:
        registry.get_provider(provider_name, {"test": "config"})
    
    # Verify all providers are listed
    assert len(registry.list_providers()) == len(providers)
    
    # Unregister one provider
    name_to_unregister, _ = providers[unregister_index]
    result = registry.unregister_provider(name_to_unregister)
    
    assert result is True, "Unregister should return True for existing provider"
    
    # Verify provider is removed from list
    provider_list = registry.list_providers()
    assert name_to_unregister not in provider_list, (
        f"Unregistered provider '{name_to_unregister}' should not appear in list"
    )
    assert len(provider_list) == len(providers) - 1, (
        "Provider count should decrease by 1"
    )
    
    # Verify other providers are still listed
    for i, (provider_name, _) in enumerate(providers):
        if i != unregister_index:
            assert provider_name in provider_list, (
                f"Provider '{provider_name}' should still be listed"
            )
    
    # Verify getting unregistered provider raises error
    with pytest.raises(ProviderNotFoundError):
        registry.get_provider(name_to_unregister)


# Additional test: Provider info consistency
@given(
    provider_name=provider_name_strategy(),
    provider_class=provider_class_strategy()
)
@settings(max_examples=30, deadline=2000)
def test_provider_info_consistency(
    provider_name: str,
    provider_class: type
):
    """
    Test that provider info is consistent with registration.
    
    Validates: Requirements 4.1, 4.4
    """
    registry = ProviderRegistry()
    
    # Register provider
    registry.register_provider(provider_name, provider_class)
    
    # Get provider info
    info = registry.get_provider_info(provider_name)
    
    # Verify info consistency
    assert info["name"] == provider_name, "Info name should match registered name"
    assert info["class"] == provider_class.__name__, "Info class should match"
    assert "capabilities" in info, "Info should include capabilities"
    assert "module" in info, "Info should include module"
    assert isinstance(info["capabilities"], list), "Capabilities should be a list"
    
    # Verify provider appears in list
    assert provider_name in registry.list_providers(), (
        "Provider with info should appear in list"
    )


# Additional test: Provider not found error includes available providers
@given(
    registered_providers=st.lists(
        st.tuples(provider_name_strategy(), provider_class_strategy()),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x[0]
    ),
    nonexistent_name=provider_name_strategy()
)
@settings(max_examples=30, deadline=2000)
def test_provider_not_found_error_includes_available(
    registered_providers: List[tuple[str, type]],
    nonexistent_name: str
):
    """
    Test that ProviderNotFoundError includes list of available providers.
    
    Validates: Requirements 4.1, 4.4
    """
    # Ensure nonexistent_name is not registered
    registered_names = {name for name, _ in registered_providers}
    assume(nonexistent_name not in registered_names)
    
    registry = ProviderRegistry()
    
    # Register providers
    for provider_name, provider_class in registered_providers:
        registry.register_provider(provider_name, provider_class)
    
    # Try to get nonexistent provider
    with pytest.raises(ProviderNotFoundError) as exc_info:
        registry.get_provider(nonexistent_name)
    
    error = exc_info.value
    
    # Verify error includes the nonexistent name
    assert error.provider_name == nonexistent_name
    
    # Verify error includes available providers (order doesn't matter)
    assert set(error.available_providers) == registered_names
    
    # Verify error message contains useful information
    error_message = str(error)
    assert nonexistent_name in error_message
    for registered_name, _ in registered_providers:
        assert registered_name in error_message


# Additional test: Empty registry behavior
def test_empty_registry_behavior():
    """
    Test that empty registry behaves correctly.
    
    Validates: Requirements 4.1, 4.4
    """
    registry = ProviderRegistry()
    
    # Empty registry should return empty list
    assert registry.list_providers() == []
    
    # Getting provider from empty registry should raise error
    with pytest.raises(ProviderNotFoundError) as exc_info:
        registry.get_provider("any_provider")
    
    # Error should indicate no providers available
    assert exc_info.value.available_providers == []
    
    # Getting info from empty registry should raise error
    with pytest.raises(ProviderNotFoundError):
        registry.get_provider_info("any_provider")
    
    # Unregistering from empty registry should return False
    assert registry.unregister_provider("any_provider") is False
