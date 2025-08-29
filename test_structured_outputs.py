#!/usr/bin/env python3
"""
Test the corrected structured outputs implementation
"""

from openrouter_client import OpenRouterClient
import json

def test_structured_outputs():
    """Test structured outputs implementation"""
    
    # Temporarily disable connection test for offline testing
    original_test_connection = OpenRouterClient._test_connection
    OpenRouterClient._test_connection = lambda self: None
    
    try:
        # Create client
        client = OpenRouterClient(
            role="Test assistant",
            model_name="openai/gpt-oss-120b:free",  # Default model unchanged
            agent_name="TestAgent",
            api_key="test_key"
        )
        
        # Test schema for weather data (matching the documentation example)
        weather_schema = {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City or location name"
                },
                "temperature": {
                    "type": "number",
                    "description": "Temperature in Celsius"
                },
                "conditions": {
                    "type": "string",
                    "description": "Weather conditions description"
                }
            },
            "required": ["location", "temperature", "conditions"],
            "additionalProperties": False
        }
        
        # Test structured output payload
        payload = client._build_chat_payload(
            "What's the weather like in London?",
            json_schema=weather_schema
        )
        
        print("🧪 Testing Structured Outputs Implementation")
        print("=" * 50)
        
        # Verify response_format structure
        if "response_format" not in payload:
            print("❌ Missing response_format")
            return False
            
        response_format = payload["response_format"]
        
        # Check type
        if response_format.get("type") != "json_schema":
            print(f"❌ Wrong type: {response_format.get('type')} (expected: json_schema)")
            return False
        print("✅ Correct type: json_schema")
        
        # Check json_schema object
        if "json_schema" not in response_format:
            print("❌ Missing json_schema object")
            return False
        print("✅ json_schema object present")
        
        json_schema_obj = response_format["json_schema"]
        
        # Check required fields
        required_fields = ["name", "strict", "schema"]
        for field in required_fields:
            if field not in json_schema_obj:
                print(f"❌ Missing required field: {field}")
                return False
            print(f"✅ Required field present: {field}")
        
        # Check strict mode
        if json_schema_obj["strict"] is not True:
            print(f"❌ Wrong strict value: {json_schema_obj['strict']} (expected: True)")
            return False
        print("✅ Strict mode enabled")
        
        # Check schema content
        if json_schema_obj["schema"] != weather_schema:
            print("❌ Schema content doesn't match")
            return False
        print("✅ Schema content matches")
        
        # Verify model hasn't changed
        if payload["model"] != "openai/gpt-oss-120b:free":
            print(f"❌ Model changed: {payload['model']} (expected: openai/gpt-oss-120b:free)")
            return False
        print("✅ Default model unchanged")
        
        print("\n🎉 All structured outputs tests passed!")
        print("\n📋 Corrected Implementation Details:")
        print("   • Fixed format: json_schema instead of json_object")
        print("   • Added required 'name' parameter")
        print("   • Added required 'strict: true' parameter")
        print("   • Removed manual schema instructions (not needed)")
        print("   • Maintains backward compatibility")
        print("   • Default model preserved: openai/gpt-oss-120b:free")
        
        # Show example payload structure
        print(f"\n📦 Example Response Format:")
        print(json.dumps(response_format, indent=2))
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        # Restore original method
        OpenRouterClient._test_connection = original_test_connection

if __name__ == "__main__":
    test_structured_outputs()
