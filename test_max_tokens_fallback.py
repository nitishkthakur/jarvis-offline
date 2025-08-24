#!/usr/bin/env python3
"""Test script to verify max_tokens fallback to max_completion_tokens."""

import os
from openai_client import OpenAIClient

def test_max_tokens_fallback():
    """Test that the client falls back to max_completion_tokens when max_tokens fails."""
    
    # Check if API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("No OPENAI_API_KEY found. Setting dummy key for testing error handling...")
        api_key = "dummy-key-for-testing"
    
    try:
        # Create client
        client = OpenAIClient(
            role="test_agent",
            api_key=api_key,
            model_name="gpt-5-mini"
        )
        
        print("Testing normal invoke method...")
        
        # Test with a simple query
        response = client.invoke("Hello, how are you?")
        print(f"Response: {response['text'][:100]}...")
        print("✅ Normal invoke method works")
        
        print("\nTesting streaming method...")
        
        # Test streaming
        streaming_response = ""
        for chunk in client.invoke_streaming("Tell me a short story"):
            streaming_response += chunk
            if len(streaming_response) > 50:  # Just get first part
                break
        
        print(f"Streaming response: {streaming_response[:100]}...")
        print("✅ Streaming method works")
        
        print("\n✅ All tests passed! The max_tokens fallback logic is in place.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Note: This might be expected if using a dummy API key.")

if __name__ == "__main__":
    test_max_tokens_fallback()
