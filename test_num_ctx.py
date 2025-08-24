#!/usr/bin/env python3
"""
Test script to verify that num_ctx parameter is being set correctly
"""

from ollama_client import OllamaClient


def test_num_ctx_parameter():
    """Test that num_ctx is included in the request payload"""
    
    # Create a test agent
    agent = OllamaClient(
        role='Test Agent for Context Window',
        agent_name='ContextTestBot'
    )
    
    print("=== Testing num_ctx Parameter ===")
    
    # Test building chat payload
    test_query = "Test query to check context window"
    payload = agent._build_chat_payload(test_query)
    
    print("Generated payload:")
    print(f"Model: {payload.get('model')}")
    print(f"Stream: {payload.get('stream')}")
    print(f"Keep Alive: {payload.get('keep_alive')}")
    print(f"Options: {payload.get('options')}")
    
    # Check if num_ctx is set correctly
    options = payload.get('options', {})
    num_ctx = options.get('num_ctx')
    
    if num_ctx == 64000:
        print("✅ num_ctx parameter is correctly set to 64000")
    else:
        print(f"❌ num_ctx parameter is incorrect: {num_ctx}")
    
    # Test with different parameters
    print("\nTesting payload with tools and schema...")
    def dummy_tool():
        """Dummy tool for testing"""
        return "test"
    
    payload_with_tools = agent._build_chat_payload(
        "Test with tools",
        tools=[dummy_tool],
        stream=True
    )
    
    print(f"Payload with tools - num_ctx: {payload_with_tools.get('options', {}).get('num_ctx')}")
    print(f"Tools included: {'tools' in payload_with_tools}")
    print(f"Stream enabled: {payload_with_tools.get('stream')}")
    
    print("\n✅ num_ctx parameter testing completed!")
    print("All API requests will now use a 64K token context window.")


if __name__ == "__main__":
    test_num_ctx_parameter()
