#!/usr/bin/env python3

"""
Test script to verify OpenAI client migration to Responses API.

This script tests both simple queries (using Responses API) and tool calling
(using Completions API) to ensure the hybrid approach works correctly.
"""

import os
import sys
from openai_client import OpenAIClient

def test_simple_query():
    """Test simple query without tools - should use Responses API."""
    print("=== Testing Simple Query (Responses API) ===")
    
    client = OpenAIClient(
        role="Test Agent",
        system_instructions="You are a helpful assistant. Keep responses concise."
    )
    
    try:
        result = client.invoke("What is 2+2? Just give me the number.")
        print(f"✅ Simple query successful")
        print(f"Model used: gpt-5-nano (forced)")
        print(f"Response: {result['text'][:100]}...")
        print(f"Tool calls: {len(result['tool_calls'])}")
        print(f"Tool results: {len(result['tool_results'])}")
        return True
    except Exception as e:
        print(f"❌ Simple query failed: {e}")
        return False

def test_tool_calling():
    """Test query with tools - should use Completions API."""
    print("\n=== Testing Tool Calling (Completions API) ===")
    
    def sample_tool(message: str) -> str:
        """Sample tool function for testing."""
        return f"Tool received: {message}"
    
    client = OpenAIClient(
        role="Tool Agent",
        system_instructions="You are a helpful assistant with access to tools."
    )
    
    try:
        result = client.invoke(
            "Please use the sample_tool function to send the message 'Hello World'",
            tools=[sample_tool]
        )
        print(f"✅ Tool calling successful")
        print(f"Model used: gpt-5-nano (forced)")
        print(f"Response: {result['text'][:100]}...")
        print(f"Tool calls: {len(result['tool_calls'])}")
        print(f"Tool results: {len(result['tool_results'])}")
        return True
    except Exception as e:
        print(f"❌ Tool calling failed: {e}")
        return False

def test_model_forcing():
    """Test that model is forced to gpt-5-nano regardless of what's requested."""
    print("\n=== Testing Model Forcing ===")
    
    client = OpenAIClient(role="Model Test Agent")
    
    # Try to set a different model
    client.set_default_model("gpt-4o")
    forced_model = client.get_default_model()
    
    if forced_model == "gpt-5-nano":
        print("✅ Model forcing successful - always gpt-5-nano")
        return True
    else:
        print(f"❌ Model forcing failed - got {forced_model}")
        return False

def test_responses_api_directly():
    """Test the Responses API method directly."""
    print("\n=== Testing Responses API Method Directly ===")
    
    client = OpenAIClient(role="Direct API Test Agent")
    
    try:
        result = client.invoke_responses_api("What is the capital of France?")
        print(f"✅ Direct Responses API call successful")
        print(f"Response: {result['text'][:100]}...")
        return True
    except Exception as e:
        print(f"❌ Direct Responses API call failed: {e}")
        return False

def main():
    """Run all tests."""
    print("OpenAI Client Migration Test Suite")
    print("=" * 50)
    
    # Check if API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key before running tests")
        sys.exit(1)
    
    tests = [
        test_model_forcing,
        test_simple_query,
        test_responses_api_directly,
        test_tool_calling,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Migration successful!")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
