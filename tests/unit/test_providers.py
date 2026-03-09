"""Unit tests for BaseProvider and ProviderRegistry."""

import pytest
from typing import Any, Dict, List, Optional

from agentlegatus.providers.base import BaseProvider, ProviderCapability
from agentlegatus.providers.registry import ProviderRegistry, ProviderNotFoundError
from agentlegatus.tools.tool import Tool


class MockProvider(BaseProvider):
    """Mock provider for testing."""
    
    def _get_capabilities(self) -> List[ProviderCapability]:
        return [
            ProviderCapability.TOOL_CALLING,
            ProviderCapability.STATE_PERSISTENCE,
        ]
    
    async def create_agent(self, agent_config: Dict[str, Any]) -> Any:
        return {"agent_id": "mock_agent", "config": agent_config}
    
    async def execute_agent(
        self,
        agent: Any,
        input_data: Any,
        state: Optional[Dict[str, Any]] = None
    ) -> Any:
        return {"result": "mock_result", "input": input_data}
    
    async def invoke_tool(
        self,
        tool: Tool,
        tool_input: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        return {"tool": tool.name, "output": "mock_output"}
    
    def export_state(self) -> Dict[str, Any]:
        return {"state": "exported"}
    
    def import_state(self, state: Dict[str, Any]) -> None:
        pass
    
    def to_portable_graph(self, workflow: Any) -> Any:
        return {"graph": "portable"}
    
    def from_portable_graph(self, graph: Any) -> Any:
        return {"workflow": "provider_specific"}


class TestBaseProvider:
    """Tests for BaseProvider abstract class."""
    
    def test_provider_initialization(self):
        """Test that provider initializes with config and capabilities."""
        config = {"api_key": "test_key", "model": "test_model"}
        provider = MockProvider(config)
        
        assert provider.config == config
        assert len(provider.capabilities) == 2
        assert ProviderCapability.TOOL_CALLING in provider.capabilities
        assert ProviderCapability.STATE_PERSISTENCE in provider.capabilities
    
    def test_supports_capability_true(self):
        """Test supports_capability returns True for supported capabilities."""
        provider = MockProvider({})
        
        assert provider.supports_capability(ProviderCapability.TOOL_CALLING) is True
        assert provider.supports_capability(ProviderCapability.STATE_PERSISTENCE) is True
    
    def test_supports_capability_false(self):
        """Test supports_capability returns False for unsupported capabilities."""
        provider = MockProvider({})
        
        assert provider.supports_capability(ProviderCapability.STREAMING) is False
        assert provider.supports_capability(ProviderCapability.PARALLEL_EXECUTION) is False
        assert provider.supports_capability(ProviderCapability.HUMAN_IN_LOOP) is False


class TestProviderRegistry:
    """Tests for ProviderRegistry."""
    
    def test_register_provider(self):
        """Test registering a provider."""
        registry = ProviderRegistry()
        registry.register_provider("mock", MockProvider)
        
        assert "mock" in registry.list_providers()
    
    def test_register_invalid_provider(self):
        """Test registering a non-provider class raises TypeError."""
        registry = ProviderRegistry()
        
        with pytest.raises(TypeError, match="must inherit from BaseProvider"):
            registry.register_provider("invalid", str)  # type: ignore
    
    def test_get_provider(self):
        """Test getting a provider instance."""
        registry = ProviderRegistry()
        registry.register_provider("mock", MockProvider)
        
        config = {"api_key": "test"}
        provider = registry.get_provider("mock", config)
        
        assert isinstance(provider, MockProvider)
        assert provider.config == config
    
    def test_get_provider_not_found(self):
        """Test getting unregistered provider raises ProviderNotFoundError."""
        registry = ProviderRegistry()
        registry.register_provider("mock", MockProvider)
        
        with pytest.raises(ProviderNotFoundError) as exc_info:
            registry.get_provider("nonexistent")
        
        assert "nonexistent" in str(exc_info.value)
        assert "mock" in str(exc_info.value)
    
    def test_get_provider_caching(self):
        """Test that provider instances are cached."""
        registry = ProviderRegistry()
        registry.register_provider("mock", MockProvider)
        
        config = {"api_key": "test"}
        provider1 = registry.get_provider("mock", config)
        provider2 = registry.get_provider("mock", config)
        
        # Should return the same instance
        assert provider1 is provider2
    
    def test_get_provider_different_configs(self):
        """Test that different configs create different instances."""
        registry = ProviderRegistry()
        registry.register_provider("mock", MockProvider)
        
        provider1 = registry.get_provider("mock", {"api_key": "key1"})
        provider2 = registry.get_provider("mock", {"api_key": "key2"})
        
        # Should return different instances
        assert provider1 is not provider2
    
    def test_list_providers(self):
        """Test listing all registered providers."""
        registry = ProviderRegistry()
        registry.register_provider("mock1", MockProvider)
        registry.register_provider("mock2", MockProvider)
        
        providers = registry.list_providers()
        
        assert len(providers) == 2
        assert "mock1" in providers
        assert "mock2" in providers
    
    def test_get_provider_info(self):
        """Test getting provider metadata."""
        registry = ProviderRegistry()
        registry.register_provider("mock", MockProvider)
        
        info = registry.get_provider_info("mock")
        
        assert info["name"] == "mock"
        assert info["class"] == "MockProvider"
        assert "tool_calling" in info["capabilities"]
        assert "state_persistence" in info["capabilities"]
        assert "module" in info
    
    def test_get_provider_info_not_found(self):
        """Test getting info for unregistered provider raises error."""
        registry = ProviderRegistry()
        
        with pytest.raises(ProviderNotFoundError):
            registry.get_provider_info("nonexistent")
    
    def test_unregister_provider(self):
        """Test unregistering a provider."""
        registry = ProviderRegistry()
        registry.register_provider("mock", MockProvider)
        
        # Get an instance to cache it
        registry.get_provider("mock", {"api_key": "test"})
        
        # Unregister
        result = registry.unregister_provider("mock")
        
        assert result is True
        assert "mock" not in registry.list_providers()
        
        # Should raise error when trying to get it
        with pytest.raises(ProviderNotFoundError):
            registry.get_provider("mock")
    
    def test_unregister_nonexistent_provider(self):
        """Test unregistering a provider that doesn't exist."""
        registry = ProviderRegistry()
        
        result = registry.unregister_provider("nonexistent")
        
        assert result is False
