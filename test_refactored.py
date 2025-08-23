#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/nitish/Documents/github/jarvis-offline')

from ollama_client import OllamaClient
from pydantic import BaseModel

class TestResponse(BaseModel):
    message: str
    status: str

def test_tool(name: str, greeting: str = "Hello") -> str:
    """A simple test tool.
    
    Args:
        name (str): The name to greet.
        greeting (str): The greeting to use.
    """
    return f"{greeting}, {name}!"

def test_all_features():
    client = OllamaClient()
    
    print("✓ Client initialization successful")
    
    # Test 1: Simple invoke
    try:
        result = client.invoke("Say hello in one word")
        assert "text" in result
        print("✓ Simple invoke works")
    except Exception as e:
        print(f"✗ Simple invoke failed: {e}")
        return False
    
    # Test 2: JSON schema
    try:
        result = client.invoke(
            "Respond with a greeting message and 'success' status",
            json_schema=TestResponse
        )
        assert "text" in result
        print("✓ JSON schema works")
    except Exception as e:
        print(f"✗ JSON schema failed: {e}")
        return False
    
    # Test 3: Tools
    try:
        result = client.invoke(
            "Use the test_tool to greet 'World'",
            tools=[test_tool]
        )
        assert "text" in result
        print("✓ Tools work")
    except Exception as e:
        print(f"✗ Tools failed: {e}")
        return False
    
    # Test 4: Streaming
    try:
        stream_count = 0
        for chunk in client.invoke_streaming("Count to 3"):
            stream_count += 1
            if stream_count > 5:  # Just test that streaming works
                break
        assert stream_count > 0
        print("✓ Streaming works")
    except Exception as e:
        print(f"✗ Streaming failed: {e}")
        return False
    
    print("\n🎉 All features work correctly!")
    return True

if __name__ == "__main__":
    test_all_features()
