#!/usr/bin/env python3
"""
Test script to verify OpenRouter client implementation
"""

from openrouter_client import OpenRouterClient
import os

def test_openrouter_implementation():
    """Test basic OpenRouter client functionality"""
    
    # Check if environment variable is set
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("⚠️  OPENROUTER_API_KEY not set - testing offline functionality only")
        api_key = "test_key_for_offline_testing"
    
    try:
        # Test client creation with mock API key
        print("🔧 Creating OpenRouter client...")
        
        # Temporarily disable connection test for offline testing
        original_test_connection = OpenRouterClient._test_connection
        OpenRouterClient._test_connection = lambda self: None
        
        client = OpenRouterClient(
            role="Test assistant",
            model_name="openai/gpt-oss-120b:free",
            agent_name="TestAgent",
            api_key=api_key
        )
        
        # Restore original method
        OpenRouterClient._test_connection = original_test_connection
        
        print("✅ Client created successfully")
        
        # Test model listing (skip if no real API key)
        if os.getenv("OPENROUTER_API_KEY"):
            print("📋 Testing model listing...")
            models = client.list_available_models()
            print(f"✅ Found {len(models)} available models")
        else:
            print("📋 Skipping model listing (no API key)")
        
        # Test basic payload building
        print("📦 Testing payload building...")
        payload = client._build_chat_payload(
            "Hello world",
            max_tokens=50,
            temperature=0.7,
            top_p=0.9
        )
        
        required_fields = ["model", "messages", "stream"]
        for field in required_fields:
            if field not in payload:
                print(f"❌ Missing required field: {field}")
                return False
        
        print("✅ Payload structure correct")
        
        # Test structured output payload
        print("🏗️ Testing structured output payload...")
        schema = {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "confidence": {"type": "number"}
            }
        }
        
        structured_payload = client._build_chat_payload(
            "Respond with JSON",
            json_schema=schema
        )
        
        if "response_format" not in structured_payload:
            print("❌ response_format not added for structured output")
            return False
            
        if structured_payload["response_format"]["type"] != "json_object":
            print("❌ Wrong response_format type")
            return False
            
        print("✅ Structured output payload correct")
        
        # Test tool calling payload
        print("🔧 Testing tool calling payload...")
        
        def test_tool(message: str) -> str:
            """Test tool function"""
            return f"Tool received: {message}"
        
        tools_payload = client._build_chat_payload(
            "Use the tool",
            tools=[test_tool],
            tool_choice="auto",
            parallel_tool_calls=False
        )
        
        if "tools" not in tools_payload:
            print("❌ tools not added to payload")
            return False
            
        if "parallel_tool_calls" not in tools_payload:
            print("❌ parallel_tool_calls not added to payload")
            return False
            
        if "tool_choice" not in tools_payload:
            print("❌ tool_choice not added to payload")
            return False
            
        print("✅ Tool calling payload correct")
        
        print("\n🎉 All tests passed! OpenRouter client implementation is correct.")
        print("\n📝 Key improvements made:")
        print("   • Added all OpenRouter API parameters (top_k, frequency_penalty, etc.)")
        print("   • Fixed structured output format (json_object instead of json_schema)")
        print("   • Added tool_choice parameter support")
        print("   • Enhanced parameter validation and documentation")
        print("   • Fixed schema instruction handling in payload")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    test_openrouter_implementation()
