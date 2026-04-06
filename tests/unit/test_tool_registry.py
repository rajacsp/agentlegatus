"""Unit tests for Tool and ToolRegistry.

Covers:
- Tool creation with name, description, parameters, handler
- Tool.validate_input() – valid inputs pass, missing required params fail, unknown params fail
- Tool.invoke() – calls handler with input data, returns result
- Tool.to_openai_format() – returns correct OpenAI function calling schema
- Tool.to_anthropic_format() – returns correct Anthropic tool schema
- ToolRegistry.register_tool() – stores tool by name
- ToolRegistry.get_tool() – returns tool by name, returns None for missing
- ToolRegistry.list_tools() – returns all registered tool names
- ToolRegistry.get_tools_for_provider() – converts tools to provider format
- ToolRegistry.unregister_tool() – removes tool, returns True/False
"""

import pytest

from agentlegatus.tools.tool import Tool, ToolParameter
from agentlegatus.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _echo_handler(input_data):
    """Simple handler that echoes input."""
    return input_data


async def _add_handler(input_data):
    """Handler that adds two numbers."""
    return input_data["a"] + input_data["b"]


async def _context_handler(input_data, context=None):
    """Handler that uses context."""
    return {"input": input_data, "context": context}


def _make_params(*specs):
    """Build a list of ToolParameter from (name, type, desc, required, default) tuples."""
    params = []
    for spec in specs:
        name, typ, desc = spec[0], spec[1], spec[2]
        required = spec[3] if len(spec) > 3 else True
        default = spec[4] if len(spec) > 4 else None
        params.append(ToolParameter(name=name, type=typ, description=desc, required=required, default=default))
    return params


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_params():
    return _make_params(
        ("query", "string", "Search query"),
        ("limit", "number", "Max results", False, 10),
    )


@pytest.fixture
def simple_tool(simple_params):
    return Tool(
        name="search",
        description="Search for items",
        parameters=simple_params,
        handler=_echo_handler,
    )


@pytest.fixture
def add_tool():
    params = _make_params(
        ("a", "number", "First number"),
        ("b", "number", "Second number"),
    )
    return Tool(name="add", description="Add two numbers", parameters=params, handler=_add_handler)


@pytest.fixture
def registry():
    return ToolRegistry()


# ---------------------------------------------------------------------------
# Tool creation
# ---------------------------------------------------------------------------


class TestToolCreation:
    def test_tool_has_name(self, simple_tool):
        assert simple_tool.name == "search"

    def test_tool_has_description(self, simple_tool):
        assert simple_tool.description == "Search for items"

    def test_tool_has_parameters(self, simple_tool):
        assert len(simple_tool.parameters) == 2
        assert simple_tool.parameters[0].name == "query"
        assert simple_tool.parameters[1].name == "limit"

    def test_tool_has_handler(self, simple_tool):
        assert simple_tool.handler is _echo_handler

    def test_parameter_defaults(self, simple_tool):
        query_param = simple_tool.parameters[0]
        assert query_param.required is True
        assert query_param.default is None

        limit_param = simple_tool.parameters[1]
        assert limit_param.required is False
        assert limit_param.default == 10


# ---------------------------------------------------------------------------
# Tool.validate_input()
# ---------------------------------------------------------------------------


class TestValidateInput:
    def test_valid_input_all_params(self, simple_tool):
        assert simple_tool.validate_input({"query": "hello", "limit": 5}) is True

    def test_valid_input_required_only(self, simple_tool):
        assert simple_tool.validate_input({"query": "hello"}) is True

    def test_missing_required_param_returns_false(self, simple_tool):
        assert simple_tool.validate_input({"limit": 5}) is False

    def test_empty_input_missing_required(self, simple_tool):
        assert simple_tool.validate_input({}) is False

    def test_unknown_param_returns_false(self, simple_tool):
        assert simple_tool.validate_input({"query": "hello", "unknown": "x"}) is False

    def test_all_optional_params_empty_input_valid(self):
        params = _make_params(("opt1", "string", "Optional", False))
        tool = Tool(name="t", description="d", parameters=params, handler=_echo_handler)
        assert tool.validate_input({}) is True

    def test_no_params_empty_input_valid(self):
        tool = Tool(name="t", description="d", parameters=[], handler=_echo_handler)
        assert tool.validate_input({}) is True

    def test_no_params_extra_input_invalid(self):
        tool = Tool(name="t", description="d", parameters=[], handler=_echo_handler)
        assert tool.validate_input({"extra": 1}) is False


# ---------------------------------------------------------------------------
# Tool.invoke()
# ---------------------------------------------------------------------------


class TestInvoke:
    @pytest.mark.asyncio
    async def test_invoke_calls_handler(self, simple_tool):
        result = await simple_tool.invoke({"query": "hello"})
        assert result == {"query": "hello"}

    @pytest.mark.asyncio
    async def test_invoke_returns_computed_result(self, add_tool):
        result = await add_tool.invoke({"a": 3, "b": 7})
        assert result == 10

    @pytest.mark.asyncio
    async def test_invoke_raises_on_invalid_input(self, simple_tool):
        with pytest.raises(ValueError, match="Invalid input"):
            await simple_tool.invoke({"bad_key": "value"})

    @pytest.mark.asyncio
    async def test_invoke_passes_context_when_handler_accepts_it(self):
        params = _make_params(("x", "string", "Input"))
        tool = Tool(name="ctx", description="d", parameters=params, handler=_context_handler)
        result = await tool.invoke({"x": "val"}, context={"user": "alice"})
        assert result["input"] == {"x": "val"}
        assert result["context"] == {"user": "alice"}

    @pytest.mark.asyncio
    async def test_invoke_without_context_on_context_handler(self):
        params = _make_params(("x", "string", "Input"))
        tool = Tool(name="ctx", description="d", parameters=params, handler=_context_handler)
        result = await tool.invoke({"x": "val"})
        assert result["context"] is None


# ---------------------------------------------------------------------------
# Tool.to_openai_format()
# ---------------------------------------------------------------------------


class TestToOpenAIFormat:
    def test_top_level_structure(self, simple_tool):
        fmt = simple_tool.to_openai_format()
        assert fmt["type"] == "function"
        assert "function" in fmt

    def test_function_name_and_description(self, simple_tool):
        func = simple_tool.to_openai_format()["function"]
        assert func["name"] == "search"
        assert func["description"] == "Search for items"

    def test_parameters_object_type(self, simple_tool):
        params = simple_tool.to_openai_format()["function"]["parameters"]
        assert params["type"] == "object"

    def test_properties_contain_all_params(self, simple_tool):
        props = simple_tool.to_openai_format()["function"]["parameters"]["properties"]
        assert "query" in props
        assert "limit" in props

    def test_property_type_and_description(self, simple_tool):
        props = simple_tool.to_openai_format()["function"]["parameters"]["properties"]
        assert props["query"]["type"] == "string"
        assert props["query"]["description"] == "Search query"

    def test_required_list_contains_only_required(self, simple_tool):
        required = simple_tool.to_openai_format()["function"]["parameters"]["required"]
        assert "query" in required
        assert "limit" not in required

    def test_default_value_included(self, simple_tool):
        props = simple_tool.to_openai_format()["function"]["parameters"]["properties"]
        assert props["limit"]["default"] == 10

    def test_no_default_when_none(self, simple_tool):
        props = simple_tool.to_openai_format()["function"]["parameters"]["properties"]
        assert "default" not in props["query"]

    def test_empty_params_tool(self):
        tool = Tool(name="noop", description="No-op", parameters=[], handler=_echo_handler)
        fmt = tool.to_openai_format()
        params = fmt["function"]["parameters"]
        assert params["properties"] == {}
        assert params["required"] == []


# ---------------------------------------------------------------------------
# Tool.to_anthropic_format()
# ---------------------------------------------------------------------------


class TestToAnthropicFormat:
    def test_top_level_keys(self, simple_tool):
        fmt = simple_tool.to_anthropic_format()
        assert fmt["name"] == "search"
        assert fmt["description"] == "Search for items"
        assert "input_schema" in fmt

    def test_input_schema_type(self, simple_tool):
        schema = simple_tool.to_anthropic_format()["input_schema"]
        assert schema["type"] == "object"

    def test_properties_contain_all_params(self, simple_tool):
        props = simple_tool.to_anthropic_format()["input_schema"]["properties"]
        assert "query" in props
        assert "limit" in props

    def test_property_type_and_description(self, simple_tool):
        props = simple_tool.to_anthropic_format()["input_schema"]["properties"]
        assert props["query"]["type"] == "string"
        assert props["query"]["description"] == "Search query"

    def test_required_list(self, simple_tool):
        required = simple_tool.to_anthropic_format()["input_schema"]["required"]
        assert "query" in required
        assert "limit" not in required

    def test_default_value_included(self, simple_tool):
        props = simple_tool.to_anthropic_format()["input_schema"]["properties"]
        assert props["limit"]["default"] == 10

    def test_empty_params_tool(self):
        tool = Tool(name="noop", description="No-op", parameters=[], handler=_echo_handler)
        fmt = tool.to_anthropic_format()
        assert fmt["input_schema"]["properties"] == {}
        assert fmt["input_schema"]["required"] == []


# ---------------------------------------------------------------------------
# ToolRegistry.register_tool()
# ---------------------------------------------------------------------------


class TestRegisterTool:
    def test_register_stores_tool(self, registry, simple_tool):
        registry.register_tool(simple_tool)
        assert registry.get_tool("search") is simple_tool

    def test_register_overwrites_existing(self, registry, simple_tool):
        registry.register_tool(simple_tool)
        replacement = Tool(name="search", description="New", parameters=[], handler=_echo_handler)
        registry.register_tool(replacement)
        assert registry.get_tool("search") is replacement

    def test_register_multiple_tools(self, registry, simple_tool, add_tool):
        registry.register_tool(simple_tool)
        registry.register_tool(add_tool)
        assert registry.get_tool("search") is simple_tool
        assert registry.get_tool("add") is add_tool


# ---------------------------------------------------------------------------
# ToolRegistry.get_tool()
# ---------------------------------------------------------------------------


class TestGetTool:
    def test_get_existing_tool(self, registry, simple_tool):
        registry.register_tool(simple_tool)
        assert registry.get_tool("search") is simple_tool

    def test_get_missing_tool_returns_none(self, registry):
        assert registry.get_tool("nonexistent") is None

    def test_get_after_unregister_returns_none(self, registry, simple_tool):
        registry.register_tool(simple_tool)
        registry.unregister_tool("search")
        assert registry.get_tool("search") is None


# ---------------------------------------------------------------------------
# ToolRegistry.list_tools()
# ---------------------------------------------------------------------------


class TestListTools:
    def test_empty_registry(self, registry):
        assert registry.list_tools() == []

    def test_lists_registered_names(self, registry, simple_tool, add_tool):
        registry.register_tool(simple_tool)
        registry.register_tool(add_tool)
        names = registry.list_tools()
        assert set(names) == {"search", "add"}

    def test_list_after_unregister(self, registry, simple_tool, add_tool):
        registry.register_tool(simple_tool)
        registry.register_tool(add_tool)
        registry.unregister_tool("search")
        assert registry.list_tools() == ["add"]


# ---------------------------------------------------------------------------
# ToolRegistry.get_tools_for_provider()
# ---------------------------------------------------------------------------


class TestGetToolsForProvider:
    def test_openai_format(self, registry, simple_tool):
        registry.register_tool(simple_tool)
        tools = registry.get_tools_for_provider("openai")
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "search"

    def test_anthropic_format(self, registry, simple_tool):
        registry.register_tool(simple_tool)
        tools = registry.get_tools_for_provider("anthropic")
        assert len(tools) == 1
        assert tools[0]["name"] == "search"
        assert "input_schema" in tools[0]

    def test_azure_uses_openai_format(self, registry, simple_tool):
        registry.register_tool(simple_tool)
        tools = registry.get_tools_for_provider("azure")
        assert len(tools) == 1
        assert tools[0]["type"] == "function"

    def test_claude_uses_anthropic_format(self, registry, simple_tool):
        registry.register_tool(simple_tool)
        tools = registry.get_tools_for_provider("claude")
        assert len(tools) == 1
        assert tools[0]["name"] == "search"
        assert "input_schema" in tools[0]

    def test_unknown_provider_defaults_to_openai(self, registry, simple_tool):
        registry.register_tool(simple_tool)
        tools = registry.get_tools_for_provider("some_unknown")
        assert len(tools) == 1
        assert tools[0]["type"] == "function"

    def test_empty_registry_returns_empty(self, registry):
        assert registry.get_tools_for_provider("openai") == []

    def test_multiple_tools_converted(self, registry, simple_tool, add_tool):
        registry.register_tool(simple_tool)
        registry.register_tool(add_tool)
        tools = registry.get_tools_for_provider("openai")
        assert len(tools) == 2
        names = {t["function"]["name"] for t in tools}
        assert names == {"search", "add"}


# ---------------------------------------------------------------------------
# ToolRegistry.unregister_tool()
# ---------------------------------------------------------------------------


class TestUnregisterTool:
    def test_unregister_existing_returns_true(self, registry, simple_tool):
        registry.register_tool(simple_tool)
        assert registry.unregister_tool("search") is True

    def test_unregister_removes_tool(self, registry, simple_tool):
        registry.register_tool(simple_tool)
        registry.unregister_tool("search")
        assert registry.get_tool("search") is None

    def test_unregister_missing_returns_false(self, registry):
        assert registry.unregister_tool("nonexistent") is False

    def test_unregister_does_not_affect_other_tools(self, registry, simple_tool, add_tool):
        registry.register_tool(simple_tool)
        registry.register_tool(add_tool)
        registry.unregister_tool("search")
        assert registry.get_tool("add") is add_tool

    def test_double_unregister_returns_false(self, registry, simple_tool):
        registry.register_tool(simple_tool)
        registry.unregister_tool("search")
        assert registry.unregister_tool("search") is False
