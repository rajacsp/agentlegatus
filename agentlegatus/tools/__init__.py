"""Tool abstraction layer."""

from agentlegatus.tools.registry import ToolRegistry
from agentlegatus.tools.tool import Tool, ToolParameter

__all__ = [
    "Tool",
    "ToolParameter",
    "ToolRegistry",
]
