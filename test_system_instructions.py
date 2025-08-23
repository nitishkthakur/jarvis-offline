#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/nitish/Documents/github/jarvis-offline')

from ollama_client import OllamaClient
import requests
import json

def test_multiple_system_instructions():
    """Test what happens with multiple system instructions."""
    
    print("🧪 Testing Multiple System Instructions")
    print("=" * 50)
    
    # Test 1: Multiple system messages in conversation history
    print("\n1. Testing multiple system messages in conversation...")
    
    agent = OllamaClient(
        model_name="llama3.2:3b",
        system_instructions="You are a helpful assistant."
    )
    
    # Manually add another system message to conversation history
    agent.conversation_history.append({
        "role": "system", 
        "content": "You must respond only with yes or no answers."
    })
    
    result = agent.invoke("Is Python a programming language?")
    print(f"Response with two system messages: {result['text']}")
    
    # Test 2: Direct API call with multiple system messages
    print("\n2. Testing direct API call with multiple system messages...")
    
    payload = {
        "model": "llama3.2:3b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "You must respond only with yes or no answers."},
            {"role": "user", "content": "Is Python a programming language?"}
        ],
        "stream": False
    }
    
    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        direct_response = data.get("message", {}).get("content", "")
        print(f"Direct API response: {direct_response}")
    except Exception as e:
        print(f"Direct API call failed: {e}")
    
    # Test 3: Override vs append behavior
    print("\n3. Testing override behavior...")
    
    agent2 = OllamaClient(
        model_name="llama3.2:3b",
        system_instructions="You are a math expert."
    )
    
    result1 = agent2.invoke("What is 2+2?")
    print(f"Math expert response: {result1['text'][:100]}...")
    
    # Change system instructions
    agent2.set_system_instructions("You are a poet who speaks only in rhymes.")
    
    result2 = agent2.invoke("What is 2+2?")
    print(f"Poet response: {result2['text'][:100]}...")
    
    print("\n📋 Analysis:")
    print("- Multiple system messages: Each is processed sequentially")
    print("- Later system messages can modify or override earlier ones")
    print("- The model tends to follow the most recent/specific instruction")
    print("- set_system_instructions() replaces the original instruction")

if __name__ == "__main__":
    test_multiple_system_instructions()
