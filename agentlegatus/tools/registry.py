"""Tool registry for managing and discovering tools."""

from typing import Any

from agentlegatus.tools.tool import Tool


class ToolRegistry:
    """Registry for managing tools across providers.

    Maintains a mapping of tool names to ``Tool`` instances and provides
    format conversion for different provider APIs.  Converted formats are
    cached per provider to avoid redundant serialisation on repeated calls.
    """

    def __init__(self) -> None:
        """Initialize tool registry."""
        self._tools: dict[str, Tool] = {}
        # Cache: provider_name -> list of converted tool dicts.
        # Invalidated whenever the tool set changes.
        self._format_cache: dict[str, list[dict[str, Any]]] = {}

    def _invalidate_cache(self) -> None:
        """Clear the provider-format cache after tool set changes."""
        self._format_cache.clear()

    def register_tool(self, tool: Tool) -> None:
        """Register a tool.

        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool
        self._invalidate_cache()

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance if found, None otherwise
        """
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_tools_for_provider(
        self,
        provider_name: str,
    ) -> list[dict[str, Any]]:
        """Get tools in provider-specific format.

        Results are cached per *provider_name* and invalidated when tools
        are registered or unregistered.

        Args:
            provider_name: Provider name (e.g., 'openai', 'anthropic')

        Returns:
            List of tools in provider-specific format
        """
        cache_key = provider_name.lower()
        if cache_key in self._format_cache:
            return self._format_cache[cache_key]

        tools: list[dict[str, Any]] = []
        for tool in self._tools.values():
            if cache_key in ("openai", "microsoft", "azure"):
                tools.append(tool.to_openai_format())
            elif cache_key in ("anthropic", "claude"):
                tools.append(tool.to_anthropic_format())
            else:
                tools.append(tool.to_openai_format())

        self._format_cache[cache_key] = tools
        return tools

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool.

        Args:
            name: Tool name to unregister

        Returns:
            True if tool was unregistered, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            self._invalidate_cache()
            return True
        return False
