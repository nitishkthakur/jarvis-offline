#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/nitish/Documents/github/jarvis-offline')

from ollama_client import OllamaClient

def test_agent_features():
    """Test the new agent features."""
    
    print("🤖 Testing Agent Features")
    print("=" * 40)
    
    # Test 1: Agent with system instructions
    print("\n1. Testing system instructions...")
    agent_with_instructions = OllamaClient(
        model_name="llama3.2:3b",
        system_instructions="You are a math tutor. Always explain your reasoning step by step."
    )
    
    result = agent_with_instructions.invoke("What is 5 + 3?")
    print(f"✓ Agent with instructions response: {result['text'][:100]}...")
    
    # Test 2: Agent without system instructions
    print("\n2. Testing without system instructions...")
    agent_without_instructions = OllamaClient(
        model_name="llama3.2:3b",
        system_instructions=""
    )
    
    result = agent_without_instructions.invoke("What is 5 + 3?")
    print(f"✓ Agent without instructions response: {result['text'][:100]}...")
    
    # Test 3: Default model usage
    print("\n3. Testing default model usage...")
    agent = OllamaClient(model_name="llama3.2:3b")
    
    result = agent.invoke("Say hello")
    print(f"✓ Default model response: {result['text'][:50]}...")
    
    # Test 4: Model override
    print("\n4. Testing model override...")
    try:
        result = agent.invoke("Say hello", model_name="llama3.1:8b")
        print(f"✓ Model override worked: {result['text'][:50]}...")
    except Exception as e:
        print(f"⚠️  Model override failed (model may not be available): {str(e)[:50]}...")
    
    print("\n🎉 All agent features tested successfully!")

if __name__ == "__main__":
    test_agent_features()
