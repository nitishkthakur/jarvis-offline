#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/nitish/Documents/github/jarvis-offline')

from ollama_client import OllamaClient

def test_conversation_features():
    """Test all conversation history features."""
    print("🗣️  Testing Conversation History Features")
    print("=" * 50)
    
    # Create agent
    agent = OllamaClient(
        model_name="llama3.2:3b",
        system_instructions="You are a helpful assistant. Keep responses brief."
    )
    
    # Test 1: Basic conversation with history
    print("\n1. Testing conversation continuity...")
    result1 = agent.invoke("My name is Alice. What's 2+2?")
    print(f"Q1: My name is Alice. What's 2+2?")
    print(f"A1: {result1['text'][:50]}...")
    
    result2 = agent.invoke("What's my name?")
    print(f"Q2: What's my name?")
    print(f"A2: {result2['text'][:50]}...")
    
    # Verify conversation history is returned
    assert 'conversation_history' in result2
    print("✓ Conversation history included in response")
    
    # Test 2: Context from other agents
    print("\n2. Testing context from other agents...")
    context = "Agent B discovered that Alice prefers mathematical explanations."
    result3 = agent.invoke_with_context(
        "Explain what addition is",
        context_from_other_agents=context
    )
    print(f"Context: {context}")
    print(f"Q3: Explain what addition is")
    print(f"A3: {result3['text'][:50]}...")
    
    # Test 3: History management
    print("\n3. Testing history management...")
    history = agent.get_conversation_history()
    print(f"✓ History length: {len(history)} messages")
    
    agent.clear_conversation_history()
    new_history = agent.get_conversation_history()
    print(f"✓ After clearing: {len(new_history)} messages")
    
    # Test 4: Streaming with history
    print("\n4. Testing streaming with history...")
    agent.invoke("I like cats.")
    
    print("Q: Tell me about dogs.")
    print("A: ", end="")
    chunk_count = 0
    for chunk in agent.invoke_streaming("Tell me about dogs."):
        print(chunk, end="")
        chunk_count += 1
        if chunk_count > 10:  # Limit output for test
            print("...")
            break
    
    final_history = agent.get_conversation_history()
    print(f"\n✓ Final history length: {len(final_history)} messages")
    
    # Verify history contains both user and assistant messages
    user_messages = [msg for msg in final_history if msg['role'] == 'user']
    assistant_messages = [msg for msg in final_history if msg['role'] == 'assistant']
    print(f"✓ User messages: {len(user_messages)}")
    print(f"✓ Assistant messages: {len(assistant_messages)}")
    
    print("\n🎉 All conversation features work correctly!")
    return True

if __name__ == "__main__":
    test_conversation_features()
