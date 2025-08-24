#!/usr/bin/env python3
"""
Test script to verify that _build_agent_context is working correctly
"""

from ollama_client import OllamaClient


def test_agent_context_building():
    """Test that agent context is being built correctly"""
    
    print("=== Testing Agent Context Building ===")
    
    # Create an agent with proper parameters
    agent = OllamaClient(
        role="Test Agent for Context Building",
        agent_name="TestContextBot"
    )
    
    print(f"Agent name: '{agent.agent_name}'")
    print(f"Agent role: '{agent.role}'")
    print(f"Initial context: '{agent.only_this_agent_context}'")
    
    # Test manual context building
    print("\n1. Testing manual _build_agent_context call:")
    agent._build_agent_context(
        agent_response="This is a test response",
        tool_results={"test_tool": "test_result"}
    )
    print(f"Context after manual call:")
    print(repr(agent.only_this_agent_context))
    print(f"Formatted context:")
    print(agent.only_this_agent_context)
    
    # Test update_agent_context (public method)
    print("\n2. Testing update_agent_context method:")
    agent.update_agent_context(
        agent_response="Updated test response",
        tool_results={"tool1": "result1", "tool2": "result2"}
    )
    print(f"Context after update:")
    print(agent.only_this_agent_context)
    
    # Test with empty agent name to verify the guard
    print("\n3. Testing with empty agent name:")
    agent_no_name = OllamaClient(
        role="Test Agent without name",
        agent_name=""  # Empty agent name
    )
    print(f"Agent with no name - agent_name: '{agent_no_name.agent_name}'")
    agent_no_name._build_agent_context("Test response", {"tool": "result"})
    print(f"Context with no agent name: '{agent_no_name.only_this_agent_context}'")
    
    print("\n✅ All agent context tests completed!")


if __name__ == "__main__":
    test_agent_context_building()
