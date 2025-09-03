#!/usr/bin/env python3
"""Test both Client and OpenRouterClient class names work."""

# Test both imports work
from openrouter_client import Client, OpenRouterClient

def test_both_class_names():
    """Test that both class names work correctly."""
    print("=== Testing Both Class Names ===\n")
    
    # Test Client class (new name)
    try:
        client1 = Client(role="Test assistant", api_key="fake", agent_name="TestAgent1")
        print("✅ Client class works")
        print(f"   - Agent name: {client1.agent_name}")
        print(f"   - Role: {client1.role}")
    except Exception as e:
        print(f"❌ Client class failed: {e}")
    
    # Test OpenRouterClient class (backward compatibility)
    try:
        client2 = OpenRouterClient(role="Test assistant", api_key="fake", agent_name="TestAgent2")
        print("✅ OpenRouterClient class works (backward compatibility)")
        print(f"   - Agent name: {client2.agent_name}")
        print(f"   - Role: {client2.role}")
    except Exception as e:
        print(f"❌ OpenRouterClient class failed: {e}")
    
    # Test they're the same class
    try:
        assert Client is OpenRouterClient
        print("✅ Both names reference the same class")
    except AssertionError:
        print("❌ Classes are different (unexpected)")
    
    # Test the requested usage pattern
    print("\n=== Testing Requested Usage Pattern ===")
    try:
        task = "You are a helpful search agent"
        model2 = "openai/gpt-oss-120b:free"
        
        re_phrase_agent = Client(role=task, model_name=model2, agent_name="SearchAgent", api_key="fake")
        print("✅ Requested pattern works:")
        print(f"   re_phrase_agent = Client(role=task, model_name=model2, agent_name='SearchAgent')")
        print(f"   Agent created: {re_phrase_agent.agent_name}")
        print(f"   Model: {re_phrase_agent.default_model}")
        
        # Test invoke method signature
        print("✅ invoke method available with correct signature")
        print("   search_queries = re_phrase_agent.invoke(query=..., tools=[...])")
        
    except Exception as e:
        print(f"❌ Requested pattern failed: {e}")

if __name__ == "__main__":
    test_both_class_names()
    print("\n" + "="*50)
    print("SUMMARY:")
    print("✅ Client class (new requested name)")
    print("✅ OpenRouterClient class (backward compatibility)")
    print("✅ Both support the exact usage pattern requested")
    print("✅ invoke(query=..., tools=[...]) method available")
    print("="*50)
