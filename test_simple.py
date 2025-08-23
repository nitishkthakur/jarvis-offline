#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/nitish/Documents/github/jarvis-offline')

from ollama_client import OllamaClient

if __name__ == "__main__":
    client = OllamaClient()
    
    # Simple test without schema or tools
    print("Testing simple invoke without schema or tools...")
    try:
        result = client.invoke(
            query="Hello, how are you?",
            json_schema=None,
            tools=None,
            model_name="llama3.2:3b",
        )
        print("SUCCESS: Simple test worked!")
        print("Response:", result["text"])
    except Exception as e:
        print("ERROR: Simple test failed:", e)
    
    print("\n" + "="*50 + "\n")
    
    # Test with JSON schema only
    print("Testing with JSON schema...")
    try:
        from pydantic import BaseModel
        
        class Answer(BaseModel):
            summary: str
            points: list[str]
        
        result = client.invoke(
            query="Summarize Python in two bullet points.",
            json_schema=Answer,
            tools=None,
            model_name="llama3.2:3b",
        )
        print("SUCCESS: JSON schema test worked!")
        print("Response:", result["text"])
    except Exception as e:
        print("ERROR: JSON schema test failed:", e)
