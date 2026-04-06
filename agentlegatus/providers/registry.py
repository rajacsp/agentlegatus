"""Provider registry for managing provider implementations."""

from typing import Any

# Import from centralized exceptions module; re-export for backward compat
from agentlegatus.exceptions import ProviderNotFoundError  # noqa: F401
from agentlegatus.providers.base import BaseProvider


class ProviderRegistry:
    """Registry for managing provider implementations.

    The registry maintains a mapping of provider names to provider classes
    and caches provider instances to avoid redundant instantiation.
    """

    def __init__(self):
        """Initialize provider registry."""
        self._providers: dict[str, type[BaseProvider]] = {}
        self._instances: dict[str, BaseProvider] = {}

    def register_provider(self, name: str, provider_class: type[BaseProvider]) -> None:
        """Register a provider implementation.

        Args:
            name: Unique name for the provider
            provider_class: Provider class that inherits from BaseProvider

        Raises:
            TypeError: If provider_class is not a subclass of BaseProvider
        """
        if not issubclass(provider_class, BaseProvider):
            raise TypeError(
                f"Provider class must inherit from BaseProvider, " f"got {provider_class.__name__}"
            )

        self._providers[name] = provider_class

    def get_provider(self, name: str, config: dict[str, Any] | None = None) -> BaseProvider:
        """Get or create a provider instance.

        If a provider instance with the same name and config already exists,
        it will be returned from cache. Otherwise, a new instance is created.

        Args:
            name: Name of the provider to get
            config: Optional configuration for the provider

        Returns:
            Provider instance

        Raises:
            ProviderNotFoundError: If the provider is not registered
        """
        if name not in self._providers:
            raise ProviderNotFoundError(name, self.list_providers())

        # Create cache key from name and config
        cache_key = self._get_cache_key(name, config)

        # Return cached instance if available
        if cache_key in self._instances:
            return self._instances[cache_key]

        # Create new instance
        provider_class = self._providers[name]
        config = config or {}
        instance = provider_class(config)

        # Cache the instance
        self._instances[cache_key] = instance

        return instance

    def list_providers(self) -> list[str]:
        """List all registered provider names.

        Returns:
            List of provider names
        """
        return list(self._providers.keys())

    def get_provider_info(self, name: str) -> dict[str, Any]:
        """Get provider metadata and capabilities.

        Args:
            name: Name of the provider

        Returns:
            Dictionary containing provider metadata including:
            - name: Provider name
            - class: Provider class name
            - capabilities: List of supported capabilities

        Raises:
            ProviderNotFoundError: If the provider is not registered
        """
        if name not in self._providers:
            raise ProviderNotFoundError(name, self.list_providers())

        provider_class = self._providers[name]

        # Create a temporary instance to get capabilities
        temp_instance = provider_class({})

        return {
            "name": name,
            "class": provider_class.__name__,
            "capabilities": [cap.value for cap in temp_instance.capabilities],
            "module": provider_class.__module__,
        }

    def unregister_provider(self, name: str) -> bool:
        """Unregister a provider.

        This removes the provider from the registry and clears any cached
        instances associated with it.

        Args:
            name: Name of the provider to unregister

        Returns:
            True if the provider was unregistered, False if it wasn't registered
        """
        if name not in self._providers:
            return False

        # Remove provider class
        del self._providers[name]

        # Remove all cached instances for this provider
        keys_to_remove = [key for key in self._instances.keys() if key.startswith(f"{name}:")]
        for key in keys_to_remove:
            del self._instances[key]

        return True

    def _get_cache_key(self, name: str, config: dict[str, Any] | None) -> str:
        """Generate cache key for provider instance.

        Args:
            name: Provider name
            config: Provider configuration

        Returns:
            Cache key string
        """
        if config is None:
            return f"{name}:default"

        # Create a simple hash of the config for caching
        # Note: This is a simple implementation. For production,
        # consider using a more robust hashing mechanism
        config_str = str(sorted(config.items()))
        return f"{name}:{hash(config_str)}"
