"""Property-based tests for Tool and ToolRegistry."""

import asyncio
from typing import Any, Dict, List

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agentlegatus.tools.tool import Tool, ToolParameter
from agentlegatus.tools.registry import ToolRegistry


# Helper strategies
@st.composite
def tool_parameter_strategy(draw):
    """Generate random ToolParameter instances."""
    name = draw(st.text(min_size=1, max_size=30, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'), 
        whitelist_characters='_'
    )))
    param_type = draw(st.sampled_from(['string', 'number', 'boolean', 'object', 'array']))
    description = draw(st.text(min_size=1, max_size=100))
    required = draw(st.booleans())
    default = draw(st.one_of(
        st.none(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=50),
        st.booleans()
    ))
    
    return ToolParameter(
        name=name,
        type=param_type,
        description=description,
        required=required,
        default=default
    )


@st.composite
def tool_strategy(draw):
    """Generate random Tool instances."""
    name = draw(st.text(min_size=1, max_size=50, alphabet=st.characters(
        whitelist_categories=('Lu', 'Ll', 'Nd'), 
        whitelist_characters='_-'
    )))
    description = draw(st.text(min_size=1, max_size=200))
    parameters = draw(st.lists(tool_parameter_strategy(), min_size=0, max_size=10, unique_by=lambda p: p.name))
    
    # Create a simple async handler
    async def handler(input_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": "success", "input": input_data}
    
    return Tool(
        name=name,
        description=description,
        parameters=parameters,
        handler=handler
    )


# Property 10: Tool Registry Round-Trip
@given(st.lists(tool_strategy(), min_size=1, max_size=20, unique_by=lambda t: t.name))
@settings(max_examples=10, deadline=2000)
def test_property_10_tool_registry_round_trip(tools: List[Tool]):
    """
    Property 10: Tool Registry Round-Trip
    
    For any set of tools registered in the ToolRegistry, 
    retrieving each tool by name returns the exact same tool instance.
    
    Validates: Requirements 9.1, 9.2
    """
    registry = ToolRegistry()
    
    # Register all tools
    for tool in tools:
        registry.register_tool(tool)
    
    # Verify each tool can be retrieved
    for tool in tools:
        retrieved_tool = registry.get_tool(tool.name)
        
        assert retrieved_tool is not None, f"Tool '{tool.name}' should be retrievable"
        assert retrieved_tool.name == tool.name, "Tool name should match"
        assert retrieved_tool.description == tool.description, "Tool description should match"
        assert len(retrieved_tool.parameters) == len(tool.parameters), "Parameter count should match"
        
        # Verify parameters match
        for orig_param, retrieved_param in zip(tool.parameters, retrieved_tool.parameters):
            assert orig_param.name == retrieved_param.name
            assert orig_param.type == retrieved_param.type
            assert orig_param.description == retrieved_param.description
            assert orig_param.required == retrieved_param.required
            assert orig_param.default == retrieved_param.default
    
    # Verify list_tools returns all registered tool names
    tool_names = registry.list_tools()
    assert len(tool_names) == len(tools), "All tools should be listed"
    assert set(tool_names) == {t.name for t in tools}, "Tool names should match"


# Property 11: Tool Input Validation Consistency
@given(
    tool_strategy(),
    st.data()
)
@settings(max_examples=10, deadline=2000)
def test_property_11_tool_input_validation_consistency(tool: Tool, data):
    """
    Property 11: Tool Input Validation Consistency
    
    For any tool with defined parameters:
    1. Input containing all required parameters passes validation
    2. Input missing required parameters fails validation
    3. Input with unknown parameters fails validation
    4. Input with only optional parameters passes validation if no required params exist
    
    Validates: Requirements 9.5, 9.6, 9.7
    """
    # Test 1: Valid input with all required parameters
    required_params = [p for p in tool.parameters if p.required]
    optional_params = [p for p in tool.parameters if not p.required]
    
    if required_params:
        # Build valid input with all required parameters
        valid_input = {}
        for param in required_params:
            if param.type == 'string':
                valid_input[param.name] = data.draw(st.text(max_size=50))
            elif param.type == 'number':
                valid_input[param.name] = data.draw(st.integers())
            elif param.type == 'boolean':
                valid_input[param.name] = data.draw(st.booleans())
            else:
                valid_input[param.name] = "test_value"
        
        assert tool.validate_input(valid_input) is True, (
            "Input with all required parameters should pass validation"
        )
    
    # Test 2: Invalid input missing required parameters
    if required_params:
        # Pick a random required parameter to omit
        param_to_omit = data.draw(st.sampled_from(required_params))
        
        invalid_input = {}
        for param in required_params:
            if param.name != param_to_omit.name:
                if param.type == 'string':
                    invalid_input[param.name] = data.draw(st.text(max_size=50))
                elif param.type == 'number':
                    invalid_input[param.name] = data.draw(st.integers())
                elif param.type == 'boolean':
                    invalid_input[param.name] = data.draw(st.booleans())
                else:
                    invalid_input[param.name] = "test_value"
        
        assert tool.validate_input(invalid_input) is False, (
            f"Input missing required parameter '{param_to_omit.name}' should fail validation"
        )
    
    # Test 3: Invalid input with unknown parameters
    if tool.parameters:
        # Create input with an unknown parameter
        valid_input = {}
        for param in required_params:
            if param.type == 'string':
                valid_input[param.name] = data.draw(st.text(max_size=50))
            elif param.type == 'number':
                valid_input[param.name] = data.draw(st.integers())
            elif param.type == 'boolean':
                valid_input[param.name] = data.draw(st.booleans())
            else:
                valid_input[param.name] = "test_value"
        
        # Add unknown parameter
        unknown_param_name = data.draw(st.text(min_size=1, max_size=20))
        assume(unknown_param_name not in {p.name for p in tool.parameters})
        
        invalid_input = {**valid_input, unknown_param_name: "unknown_value"}
        
        assert tool.validate_input(invalid_input) is False, (
            f"Input with unknown parameter '{unknown_param_name}' should fail validation"
        )
    
    # Test 4: Valid input with only optional parameters (if no required params)
    if not required_params and optional_params:
        # Build input with some optional parameters
        optional_input = {}
        for param in optional_params[:min(3, len(optional_params))]:
            if param.type == 'string':
                optional_input[param.name] = data.draw(st.text(max_size=50))
            elif param.type == 'number':
                optional_input[param.name] = data.draw(st.integers())
            elif param.type == 'boolean':
                optional_input[param.name] = data.draw(st.booleans())
            else:
                optional_input[param.name] = "test_value"
        
        assert tool.validate_input(optional_input) is True, (
            "Input with only optional parameters should pass validation when no required params exist"
        )
    
    # Test 5: Empty input is valid only if no required parameters
    empty_input = {}
    if required_params:
        assert tool.validate_input(empty_input) is False, (
            "Empty input should fail validation when required parameters exist"
        )
    else:
        assert tool.validate_input(empty_input) is True, (
            "Empty input should pass validation when no required parameters exist"
        )


# Additional test: Tool format conversion consistency
@given(tool_strategy())
@settings(max_examples=10, deadline=2000)
def test_tool_format_conversion_consistency(tool: Tool):
    """
    Test that tool format conversions preserve essential information.
    
    Validates: Requirements 9.8, 9.9
    """
    # Test OpenAI format conversion
    openai_format = tool.to_openai_format()
    
    assert openai_format["type"] == "function"
    assert openai_format["function"]["name"] == tool.name
    assert openai_format["function"]["description"] == tool.description
    assert "parameters" in openai_format["function"]
    assert openai_format["function"]["parameters"]["type"] == "object"
    
    # Verify all parameters are present
    properties = openai_format["function"]["parameters"]["properties"]
    assert len(properties) == len(tool.parameters)
    
    for param in tool.parameters:
        assert param.name in properties
        assert properties[param.name]["type"] == param.type
        assert properties[param.name]["description"] == param.description
    
    # Verify required parameters
    required_params = [p.name for p in tool.parameters if p.required]
    assert set(openai_format["function"]["parameters"]["required"]) == set(required_params)
    
    # Test Anthropic format conversion
    anthropic_format = tool.to_anthropic_format()
    
    assert anthropic_format["name"] == tool.name
    assert anthropic_format["description"] == tool.description
    assert "input_schema" in anthropic_format
    assert anthropic_format["input_schema"]["type"] == "object"
    
    # Verify all parameters are present
    properties = anthropic_format["input_schema"]["properties"]
    assert len(properties) == len(tool.parameters)
    
    for param in tool.parameters:
        assert param.name in properties
        assert properties[param.name]["type"] == param.type
        assert properties[param.name]["description"] == param.description
    
    # Verify required parameters
    assert set(anthropic_format["input_schema"]["required"]) == set(required_params)


# Additional test: Tool registry operations
@given(
    st.lists(tool_strategy(), min_size=2, max_size=10, unique_by=lambda t: t.name),
    st.integers(min_value=0, max_value=9)
)
@settings(max_examples=10, deadline=2000)
def test_tool_registry_operations(tools: List[Tool], unregister_index: int):
    """
    Test tool registry operations including unregister.
    
    Validates: Requirements 9.1, 9.2, 9.3, 9.4
    """
    unregister_index = unregister_index % len(tools)
    
    registry = ToolRegistry()
    
    # Register all tools
    for tool in tools:
        registry.register_tool(tool)
    
    # Verify all tools are registered
    assert len(registry.list_tools()) == len(tools)
    
    # Unregister one tool
    tool_to_unregister = tools[unregister_index]
    result = registry.unregister_tool(tool_to_unregister.name)
    
    assert result is True, "Unregister should return True for existing tool"
    assert len(registry.list_tools()) == len(tools) - 1, "Tool count should decrease"
    assert tool_to_unregister.name not in registry.list_tools(), "Unregistered tool should not be listed"
    assert registry.get_tool(tool_to_unregister.name) is None, "Unregistered tool should not be retrievable"
    
    # Verify other tools are still registered
    for i, tool in enumerate(tools):
        if i != unregister_index:
            assert tool.name in registry.list_tools()
            assert registry.get_tool(tool.name) is not None
    
    # Unregister non-existent tool
    result = registry.unregister_tool("non_existent_tool")
    assert result is False, "Unregister should return False for non-existent tool"


# Additional test: Tool registry provider format conversion
@given(
    st.lists(tool_strategy(), min_size=1, max_size=5, unique_by=lambda t: t.name),
    st.sampled_from(['openai', 'anthropic', 'microsoft', 'azure', 'claude', 'unknown'])
)
@settings(max_examples=10, deadline=2000)
def test_tool_registry_provider_format(tools: List[Tool], provider_name: str):
    """
    Test that tool registry converts tools to provider-specific formats.
    
    Validates: Requirements 9.4, 9.8, 9.9
    """
    registry = ToolRegistry()
    
    # Register all tools
    for tool in tools:
        registry.register_tool(tool)
    
    # Get tools in provider format
    provider_tools = registry.get_tools_for_provider(provider_name)
    
    assert len(provider_tools) == len(tools), "All tools should be converted"
    
    # Verify format based on provider
    for tool_def in provider_tools:
        if provider_name.lower() in ['openai', 'microsoft', 'azure', 'unknown']:
            # Should be OpenAI format
            assert "type" in tool_def
            assert tool_def["type"] == "function"
            assert "function" in tool_def
        elif provider_name.lower() in ['anthropic', 'claude']:
            # Should be Anthropic format
            assert "name" in tool_def
            assert "description" in tool_def
            assert "input_schema" in tool_def


# Additional test: Tool invocation with valid input
@pytest.mark.asyncio
@given(tool_strategy())
@settings(max_examples=10, deadline=2000)
async def test_tool_invocation_with_valid_input(tool: Tool):
    """
    Test that tool invocation works with valid input.
    
    Validates: Requirements 9.5, 9.6
    """
    # Build valid input
    valid_input = {}
    for param in tool.parameters:
        if param.required:
            if param.type == 'string':
                valid_input[param.name] = "test_value"
            elif param.type == 'number':
                valid_input[param.name] = 42
            elif param.type == 'boolean':
                valid_input[param.name] = True
            else:
                valid_input[param.name] = "test"
    
    # Invoke tool
    result = await tool.invoke(valid_input)
    
    # Verify result (our test handler returns a dict with result and input)
    assert result is not None
    assert isinstance(result, dict)
    assert result["result"] == "success"
    assert result["input"] == valid_input


# Additional test: Tool invocation with invalid input
@pytest.mark.asyncio
@given(tool_strategy())
@settings(max_examples=10, deadline=2000)
async def test_tool_invocation_with_invalid_input(tool: Tool):
    """
    Test that tool invocation raises ValueError with invalid input.
    
    Validates: Requirements 9.6, 9.7
    """
    # Only test if tool has required parameters
    required_params = [p for p in tool.parameters if p.required]
    
    if required_params:
        # Build invalid input (missing required parameters)
        invalid_input = {}
        
        # Invoke tool and expect ValueError
        with pytest.raises(ValueError, match=f"Invalid input for tool '{tool.name}'"):
            await tool.invoke(invalid_input)
