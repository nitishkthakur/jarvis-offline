#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/nitish/Documents/github/jarvis-offline')

from ollama_client import OllamaClient
from pydantic import BaseModel
import json

class Answer(BaseModel):
    summary: str
    points: list[str]

if __name__ == "__main__":
    client = OllamaClient()
    
    # Test what payload is generated
    print("Testing payload generation...")
    
    payload = client._chat_payload(
        query="Summarize Python in two bullet points.",
        json_schema=Answer,
        tools=None,
        model_name="llama3.2:3b",
        stream=False
    )
    
    print("Generated payload:")
    print(json.dumps(payload, indent=2))
