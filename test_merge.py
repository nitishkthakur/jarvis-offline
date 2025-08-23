#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/nitish/Documents/github/jarvis-offline')

from ollama_client import OllamaClient

def test_agent_merging():
    """Test merging conversation history between agents."""
    
    print("🤖 Testing Agent Conversation Merging")
    print("=" * 40)
    
    # Agent 1: Code reviewer
    agent1 = OllamaClient(
        model_name="llama3.2:3b",
        system_instructions="You are a code reviewer. Focus on code quality."
    )
    
    # Agent 2: Security expert  
    agent2 = OllamaClient(
        model_name="llama3.2:3b",
        system_instructions="You are a security expert. Focus on security vulnerabilities."
    )
    
    print("\n1. Agent 1 (Code Reviewer) conversation:")
    result1 = agent1.invoke("Review this code: print('Hello World')")
    print(f"Q: Review this code: print('Hello World')")
    print(f"A1: {result1['text'][:80]}...")
    
    print("\n2. Agent 2 (Security Expert) conversation:")
    result2 = agent2.invoke("Are there any security issues with print statements?")
    print(f"Q: Are there any security issues with print statements?")
    print(f"A2: {result2['text'][:80]}...")
    
    print("\n3. Creating Agent 3 that merges both conversations:")
    agent3 = OllamaClient(
        model_name="llama3.2:3b",
        system_instructions="You are a senior developer. Provide comprehensive advice."
    )
    
    # Merge conversations from both agents
    agent3.merge_conversation_from_agent(agent1)
    agent3.merge_conversation_from_agent(agent2)
    
    print(f"\nAgent 3 conversation history length: {len(agent3.get_conversation_history())} messages")
    
    result3 = agent3.invoke("Based on the previous discussions, what's your overall assessment?")
    print(f"\nQ: Based on the previous discussions, what's your overall assessment?")
    print(f"A3: {result3['text'][:100]}...")
    
    print(f"\n4. Full merged conversation history:")
    for i, msg in enumerate(agent3.get_conversation_history()):
        role = msg['role'].upper()
        content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
        print(f"  {i+1}. {role:9}: {content}")
    
    print(f"\n✅ Agent merging test completed!")

if __name__ == "__main__":
    test_agent_merging()
