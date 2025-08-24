#!/usr/bin/env python3
"""
Test script to simulate the invoke flow and check agent context
"""

from ollama_client import OllamaClient


def mock_invoke_flow():
    """Simulate the invoke flow without actually calling the API"""
    
    print("=== Simulating Invoke Flow ===")
    
    # Create an agent exactly like in the main example
    agent = OllamaClient(
        role="Python programming consultant", 
        agent_name="PythonExpert"
    )
    
    print(f"Agent created:")
    print(f"  - agent_name: '{agent.agent_name}'")
    print(f"  - role: '{agent.role}'")
    print(f"  - Initial context: '{agent.only_this_agent_context}'")
    
    # Simulate what happens in the invoke method
    print(f"\nSimulating invoke method execution...")
    
    # Simulate the response processing
    assistant_response = "Python is a high-level programming language known for its simplicity and readability."
    tools = None  # No tools in this example
    
    # This is what happens in the invoke method
    tool_results = {}
    if tools:
        tool_results = {func.__name__: "invoked" for func in tools}
    
    print(f"Before _build_agent_context:")
    print(f"  - assistant_response: '{assistant_response[:50]}...'")
    print(f"  - tool_results: {tool_results}")
    print(f"  - agent_name: '{agent.agent_name}'")
    print(f"  - agent context: '{agent.only_this_agent_context}'")
    
    # Call _build_agent_context like the invoke method does
    agent._build_agent_context(assistant_response, tool_results)
    
    print(f"\nAfter _build_agent_context:")
    print(f"  - agent context: '{agent.only_this_agent_context}'")
    
    if agent.only_this_agent_context:
        print(f"\n✅ Agent context successfully built!")
        print(f"Formatted context:")
        print(agent.only_this_agent_context)
    else:
        print(f"\n❌ Agent context is still empty!")
    
    # Test the get_agent_context method
    context_via_getter = agent.get_agent_context()
    print(f"\nContext via get_agent_context(): '{context_via_getter}'")
    
    return agent


def test_empty_agent_name_scenario():
    """Test what happens when agent_name is empty"""
    print("\n=== Testing Empty Agent Name Scenario ===")
    
    # Test with empty agent name
    agent_empty = OllamaClient(
        role="Test role",
        agent_name=""  # Empty name
    )
    
    print(f"Agent with empty name:")
    print(f"  - agent_name: '{agent_empty.agent_name}'")
    print(f"  - bool(agent_name): {bool(agent_empty.agent_name)}")
    
    agent_empty._build_agent_context("Test response", {"tool": "result"})
    print(f"  - context after build: '{agent_empty.only_this_agent_context}'")
    
    # Test with None agent name
    agent_none = OllamaClient(
        role="Test role",
        agent_name=None  # None name
    )
    
    print(f"\nAgent with None name:")
    print(f"  - agent_name: {agent_none.agent_name}")
    print(f"  - bool(agent_name): {bool(agent_none.agent_name)}")
    
    agent_none._build_agent_context("Test response", {"tool": "result"})
    print(f"  - context after build: '{agent_none.only_this_agent_context}'")


if __name__ == "__main__":
    mock_invoke_flow()
    test_empty_agent_name_scenario()
    print("\n=== All tests completed ===")
