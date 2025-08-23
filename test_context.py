#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/nitish/Documents/github/jarvis-offline')

from ollama_client import OllamaClient

def test_context_integration():
    """Test the new context integration with conversation history."""
    
    print("🧪 Testing Context Integration with Conversation History")
    print("=" * 60)
    
    # Create agent with system instructions
    agent = OllamaClient(
        model_name="llama3.2:3b",
        system_instructions="You are a helpful coding assistant. Always be concise."
    )
    
    print("\n1. Initial conversation:")
    result1 = agent.invoke("Hello, what can you help me with?")
    print(f"Q: Hello, what can you help me with?")
    print(f"A: {result1['text'][:80]}...")
    
    print(f"\nHistory after first interaction: {len(agent.get_conversation_history())} messages")
    
    print("\n2. Adding context from other agents (conversation history):")
    other_agent_context = [
        {"role": "user", "content": "I'm working on a machine learning project"},
        {"role": "assistant", "content": "Great! Are you using Python for your ML project?"},
        {"role": "user", "content": "Yes, but I'm having trouble with data preprocessing"},
        {"role": "assistant", "content": "For data preprocessing in Python, I recommend using pandas and scikit-learn"}
    ]
    
    print("Adding context:")
    for msg in other_agent_context:
        print(f"  {msg['role']}: {msg['content'][:50]}...")
    
    result2 = agent.invoke_with_context(
        "What libraries should I use for this?",
        context_from_other_agents=other_agent_context
    )
    
    print(f"\nQ: What libraries should I use for this?")
    print(f"A: {result2['text'][:100]}...")
    
    print(f"\nHistory after adding context: {len(agent.get_conversation_history())} messages")
    
    print("\n3. Full conversation history:")
    history = agent.get_conversation_history()
    for i, msg in enumerate(history):
        role = msg['role'].upper()
        content = msg['content'][:60] + "..." if len(msg['content']) > 60 else msg['content']
        print(f"  {i+1:2d}. {role:9}: {content}")
    
    print("\n4. Testing system instructions remain unchanged:")
    print(f"System instructions: '{agent.get_system_instructions()}'")
    
    print("\n5. Continuing conversation (should have full context):")
    result3 = agent.invoke("What about visualization libraries?")
    print(f"Q: What about visualization libraries?")
    print(f"A: {result3['text'][:100]}...")
    
    print(f"\n✅ Context integration test completed!")
    print(f"Final history length: {len(agent.get_conversation_history())} messages")

if __name__ == "__main__":
    test_context_integration()
