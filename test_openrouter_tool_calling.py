#!/usr/bin/env python3
"""Test script for OpenRouter tool calling implementation."""

from openrouter_client import Client

def get_weather(location: str) -> str:
    """Get the weather for a location (mock function)."""
    return f"The weather in {location} is sunny and 25°C"

def calculate(expression: str) -> str:
    """Calculate a mathematical expression (mock function)."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

def main():
    """Test OpenRouter tool calling."""
    client = Client(role="You are a helpful assistant that can use tools to help users.")
    
    # Test 1: Basic tool calling
    print("=== Test 1: Basic Tool Calling ===")
    try:
        response = client.invoke(
            "What's the weather like in Paris?",
            tools=[get_weather]
        )
        print(f"Response: {response.get('response', 'No response')}")
        print(f"Tool calls made: {bool(response.get('tool_calls'))}")
        print()
    except Exception as e:
        print(f"Error in Test 1: {e}")
        print()
    
    # Test 2: Multiple tool calls
    print("=== Test 2: Multiple Tool Calls ===")
    try:
        response = client.invoke(
            "What's the weather in London and what's 15 + 27?",
            tools=[get_weather, calculate]
        )
        print(f"Response: {response.get('response', 'No response')}")
        print(f"Tool calls made: {bool(response.get('tool_calls'))}")
        print()
    except Exception as e:
        print(f"Error in Test 2: {e}")
        print()
    
    # Test 3: Agentic loop
    print("=== Test 3: Agentic Loop ===")
    try:
        response = client.invoke_with_tools_loop(
            "Calculate 5 * 8, then tell me the weather in Tokyo",
            tools=[calculate, get_weather],
            max_iterations=5
        )
        print(f"Final response: {response.get('response', 'No response')}")
        print(f"Total iterations: {response.get('total_iterations', 1)}")
        print()
    except Exception as e:
        print(f"Error in Test 3: {e}")
        print()
    
    # Test 4: Structured output
    print("=== Test 4: Structured Output ===")
    try:
        schema = {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "temperature": {"type": "number"},
                "conditions": {"type": "string"}
            },
            "required": ["location", "temperature", "conditions"]
        }
        
        response = client.invoke(
            "Give me weather data for New York in the specified format",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "weather_data",
                    "strict": True,
                    "schema": schema
                }
            }
        )
        print(f"Structured response: {response.get('response', 'No response')}")
        print()
    except Exception as e:
        print(f"Error in Test 4: {e}")
        print()

if __name__ == "__main__":
    main()
