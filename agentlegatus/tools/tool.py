"""Tool abstraction for unified tool invocation across providers."""

from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Parameter definition for a tool."""
    
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (e.g., 'string', 'number', 'boolean', 'object', 'array')")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    default: Optional[Any] = Field(default=None, description="Default value if not provided")


class Tool:
    """Unified tool abstraction for cross-provider tool invocation."""
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        handler: Callable[[Dict[str, Any]], Any]
    ):
        """Initialize tool with metadata and handler.
        
        Args:
            name: Tool name
            description: Tool description
            parameters: List of tool parameters
            handler: Callable that executes the tool logic
        """
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
    
    async def invoke(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Invoke the tool with input data.
        
        Args:
            input_data: Input data for the tool
            context: Optional execution context
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If input validation fails
        """
        if not self.validate_input(input_data):
            raise ValueError(f"Invalid input for tool '{self.name}'")
        
        # Add context if handler accepts it
        import inspect
        sig = inspect.signature(self.handler)
        if 'context' in sig.parameters:
            return await self.handler(input_data, context=context)
        else:
            return await self.handler(input_data)
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate tool input against parameter definitions.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in input_data:
                return False
        
        # Check for unknown parameters
        param_names = {p.name for p in self.parameters}
        for key in input_data.keys():
            if key not in param_names:
                return False
        
        return True
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function calling format.
        
        Returns:
            Tool definition in OpenAI format
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.default is not None:
                properties[param.name]["default"] = param.default
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert tool to Anthropic tool format.
        
        Returns:
            Tool definition in Anthropic format
        """
        input_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param in self.parameters:
            input_schema["properties"][param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.default is not None:
                input_schema["properties"][param.name]["default"] = param.default
            
            if param.required:
                input_schema["required"].append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": input_schema
        }
