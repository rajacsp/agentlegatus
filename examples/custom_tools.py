#!/usr/bin/env python3
"""Example 5: Custom Tool Registration and Usage.

Demonstrates creating tools with parameter definitions, registering them
in the ToolRegistry, validating inputs, invoking them, and converting
to provider-specific formats (OpenAI / Anthropic).
"""

import asyncio
from typing import Any, Dict

from agentlegatus.tools.registry import ToolRegistry
from agentlegatus.tools.tool import Tool, ToolParameter


# --- 1. Define tool handlers ---

async def calculator_handler(input_data: Dict[str, Any]) -> Any:
    """Simple arithmetic calculator."""
    a = input_data["a"]
    b = input_data["b"]
    op = input_data.get("operation", "add")
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "division by zero",
    }
    return {"result": ops.get(op, "unknown operation")}


async def weather_handler(input_data: Dict[str, Any]) -> Any:
    """Simulated weather lookup."""
    city = input_data["city"]
    return {"city": city, "temperature": 22, "unit": "celsius", "condition": "sunny"}


async def main():
    # --- 2. Create Tool instances ---
    calculator = Tool(
        name="calculator",
        description="Perform basic arithmetic operations",
        parameters=[
            ToolParameter(name="a", type="number", description="First operand"),
            ToolParameter(name="b", type="number", description="Second operand"),
            ToolParameter(
                name="operation", type="string",
                description="Operation: add, subtract, multiply, divide",
                required=False, default="add",
            ),
        ],
        handler=calculator_handler,
    )

    weather = Tool(
        name="weather_lookup",
        description="Get current weather for a city",
        parameters=[
            ToolParameter(name="city", type="string", description="City name"),
        ],
        handler=weather_handler,
    )

    # --- 3. Register tools ---
    registry = ToolRegistry()
    registry.register_tool(calculator)
    registry.register_tool(weather)

    print(f"Registered tools: {registry.list_tools()}")

    # --- 4. Validate and invoke ---
    calc_input = {"a": 10, "b": 3, "operation": "multiply"}
    print(f"\nCalculator valid : {calculator.validate_input(calc_input)}")
    calc_result = await calculator.invoke(calc_input)
    print(f"Calculator result: {calc_result}")

    bad_input = {"x": 1}  # missing required params
    print(f"Bad input valid  : {calculator.validate_input(bad_input)}")

    weather_result = await weather.invoke({"city": "Tokyo"})
    print(f"Weather result   : {weather_result}")

    # --- 5. Convert to provider formats ---
    print("\n=== OpenAI format ===")
    for fmt in registry.get_tools_for_provider("openai"):
        print(f"  {fmt['function']['name']}: {fmt['function']['description']}")

    print("\n=== Anthropic format ===")
    for fmt in registry.get_tools_for_provider("anthropic"):
        print(f"  {fmt['name']}: {fmt['description']}")

    # --- 6. Unregister ---
    registry.unregister_tool("weather_lookup")
    print(f"\nAfter unregister: {registry.list_tools()}")


if __name__ == "__main__":
    asyncio.run(main())
