"""Tool registry for managing and discovering tools."""

from typing import Dict, List, Optional
from agentlegatus.tools.tool import Tool


class ToolRegistry:
    """Registry for managing tools across providers."""
    
    def __init__(self):
        """Initialize tool registry."""
        self._tools: Dict[str, Tool] = {}
    
    def register_tool(self, tool: Tool) -> None:
        """Register a tool.
        
        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance if found, None otherwise
        """
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_tools_for_provider(
        self,
        provider_name: str
    ) -> List[Dict[str, any]]:
        """Get tools in provider-specific format.
        
        Args:
            provider_name: Provider name (e.g., 'openai', 'anthropic')
            
        Returns:
            List of tools in provider-specific format
        """
        tools = []
        
        for tool in self._tools.values():
            if provider_name.lower() in ['openai', 'microsoft', 'azure']:
                tools.append(tool.to_openai_format())
            elif provider_name.lower() in ['anthropic', 'claude']:
                tools.append(tool.to_anthropic_format())
            else:
                # Default to OpenAI format for unknown providers
                tools.append(tool.to_openai_format())
        
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
            return True
        return False
