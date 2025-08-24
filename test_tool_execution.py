#!/usr/bin/env python3
"""
Test script to verify tool call execution functionality
"""

from ollama_client import OllamaClient


def test_tool_execution():
    """Test that tool execution works correctly"""
    
    def get_weather(city: str, unit: str = "C") -> str:
        """Get weather information for a city.

        Args:
            city (str): The name of the city.
            unit (str): Temperature unit ('C' for Celsius, 'F' for Fahrenheit).
        """
        return f"Weather in {city}: 24°{unit}, sunny"

    def calculate_sum(a: int, b: int) -> int:
        """Calculate the sum of two numbers.
        
        Args:
            a (int): First number
            b (int): Second number
        """
        return a + b

    # Create a test agent
    agent = OllamaClient(
        role='Test Agent for Tool Execution',
        agent_name='ToolTestBot'
    )
    
    print("=== Testing Tool Call Execution ===")
    
    # Test the _execute_tool_calls method directly
    print("\n1. Testing _execute_tool_calls method:")
    
    # Simulate tool calls from the model
    mock_tool_calls = [
        {
            "function": {
                "name": "get_weather",
                "arguments": {"city": "New York", "unit": "F"}
            }
        },
        {
            "function": {
                "name": "calculate_sum", 
                "arguments": {"a": 5, "b": 3}
            }
        }
    ]
    
    available_tools = [get_weather, calculate_sum]
    
    results = agent._execute_tool_calls(mock_tool_calls, available_tools)
    print(f"Tool execution results: {results}")
    
    # Test storing results
    print("\n2. Testing tool call results storage:")
    agent.tool_call_results.update(results)
    print(f"Stored tool call results: {agent.get_tool_call_results()}")
    
    # Test building agent context with tool results
    print("\n3. Testing agent context building with tool results:")
    agent._build_agent_context(
        agent_response="I've executed the requested tools.",
        tool_results=results
    )
    print(f"Agent context with tool results:")
    print(agent.get_agent_context())
    
    # Test with invalid tool call
    print("\n4. Testing error handling:")
    invalid_tool_calls = [
        {
            "function": {
                "name": "nonexistent_tool",
                "arguments": {"param": "value"}
            }
        }
    ]
    
    error_results = agent._execute_tool_calls(invalid_tool_calls, available_tools)
    print(f"Error handling results: {error_results}")
    
    # Test with malformed arguments
    print("\n5. Testing malformed arguments:")
    malformed_tool_calls = [
        {
            "function": {
                "name": "calculate_sum",
                "arguments": '{"a": 5, "b": "not_a_number"}'  # String arguments
            }
        }
    ]
    
    malformed_results = agent._execute_tool_calls(malformed_tool_calls, available_tools)
    print(f"Malformed arguments results: {malformed_results}")
    
    print("\n✅ All tool execution tests completed!")
    print("\nNote: The invoke methods will use this functionality when")
    print("the Ollama model returns tool_calls in the response.")


if __name__ == "__main__":
    test_tool_execution()
