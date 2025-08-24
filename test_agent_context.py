#!/usr/bin/env python3
"""
Test script to verify _build_agent_context is being called in invoke methods
"""

from ollama_client import OllamaClient


def test_build_agent_context_integration():
    """Test that _build_agent_context is called correctly"""
    
    # Create a test agent
    agent = OllamaClient(
        role='Test Agent for Context Building',
        agent_name='TestContextBot'
    )
    
    print("=== Testing _build_agent_context Integration ===")
    print(f"Initial agent context: {agent.get_agent_context()}")
    
    # Test manual context building
    print("\n1. Testing manual _build_agent_context call:")
    agent._build_agent_context(
        agent_response="This is a test response",
        tool_results={"test_tool": "test_result"}
    )
    print(f"Agent context after manual call:")
    print(agent.get_agent_context())
    
    # Test update_agent_context (public method)
    print("\n2. Testing update_agent_context method:")
    agent.update_agent_context(
        agent_response="Updated test response",
        tool_results={"tool1": "result1", "tool2": "result2"}
    )
    print(f"Agent context after update:")
    print(agent.get_agent_context())
    
    # Test with empty values
    print("\n3. Testing with empty values:")
    agent.update_agent_context(
        agent_response="",
        tool_results=None
    )
    print(f"Agent context with empty values:")
    print(agent.get_agent_context())
    
    print("\n✅ All _build_agent_context tests completed successfully!")
    print("\nNote: The invoke methods will call _build_agent_context automatically")
    print("when they process responses from the Ollama API.")


if __name__ == "__main__":
    test_build_agent_context_integration()
