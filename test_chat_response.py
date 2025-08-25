#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Test the OllamaClient and ChatResponse creation
from ollama_client import OllamaClient
from pydantic import BaseModel
from typing import Optional
import time

class ChatResponse(BaseModel):
    response: str
    agent: str
    timestamp: float
    token_count: Optional[int] = None
    processing_time: Optional[float] = None

def estimate_tokens(text: str) -> int:
    """Estimate token count for a given text."""
    return int(len(text.split()) * 1.3)  # Rough estimation

def test_non_streaming():
    print("Testing non-streaming API components...")
    
    try:
        # Initialize client
        client = OllamaClient(role="chat_assistant")
        print("✓ Client initialized successfully")
        
        # Test invoke
        response_data = client.invoke(
            query="Hello, this is a test",
            model_name="qwen3:8b",
        )
        print(f"✓ Client invoke successful: {type(response_data)}")
        print(f"Response keys: {response_data.keys() if isinstance(response_data, dict) else 'Not a dict'}")
        
        # Extract response
        response = response_data.get("text", "") if isinstance(response_data, dict) else str(response_data)
        print(f"✓ Response extracted: '{response[:50]}...'")
        
        # Test token estimation
        token_count = estimate_tokens(response)
        print(f"✓ Token count: {token_count} (type: {type(token_count)})")
        
        # Test ChatResponse creation
        chat_response = ChatResponse(
            response=response,
            agent="general",
            timestamp=time.time(),
            token_count=token_count,
            processing_time=1.5
        )
        print("✓ ChatResponse created successfully")
        print(f"ChatResponse: {chat_response.model_dump()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_non_streaming()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
