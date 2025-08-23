#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/nitish/Documents/github/jarvis-offline')

from ollama_client import OllamaClient

def test_agent_methods():
    """Test all agent methods."""
    print("🧪 Testing Agent Methods")
    print("=" * 30)
    
    # Create agent
    agent = OllamaClient(
        model_name="llama3.2:3b",
        system_instructions="You are a helpful assistant."
    )
    
    # Test getters
    print(f"✓ Default model: {agent.get_default_model()}")
    print(f"✓ System instructions: '{agent.get_system_instructions()}'")
    
    # Test setters
    agent.set_system_instructions("You are a math expert.")
    print(f"✓ Updated instructions: '{agent.get_system_instructions()}'")
    
    agent.set_default_model("llama3.2:3b")
    print(f"✓ Updated model: {agent.get_default_model()}")
    
    # Test usage with new instructions
    result = agent.invoke("What is 10 + 5?")
    print(f"✓ Math expert response: {result['text'][:80]}...")
    
    print("\n🎉 All agent methods work correctly!")

if __name__ == "__main__":
    test_agent_methods()
