"""Tool abstraction layer."""

from agentlegatus.tools.tool import Tool, ToolParameter
from agentlegatus.tools.registry import ToolRegistry

__all__ = [
    "Tool",
    "ToolParameter",
    "ToolRegistry",
]
