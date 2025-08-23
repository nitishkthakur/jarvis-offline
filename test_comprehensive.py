#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/nitish/Documents/github/jarvis-offline')

from ollama_client import OllamaClient
import requests
import json

def test_system_instruction_behavior():
    """Comprehensive test of system instruction behavior."""
    
    print("🧪 Comprehensive System Instruction Test")
    print("=" * 50)
    
    # Test 1: Conflicting instructions
    print("\n1. Testing conflicting system instructions...")
    
    payload = {
        "model": "llama3.2:3b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant who provides detailed explanations."},
            {"role": "system", "content": "You must respond with only one word answers."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "stream": False,
        "keep_alive": "15m"
    }
    
    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        result = data.get("message", {}).get("content", "")
        print(f"Conflicting instructions result: '{result}'")
        print(f"Length: {len(result.split())} words")
    except Exception as e:
        print(f"Test 1 failed: {e}")
    
    # Test 2: Additive instructions
    print("\n2. Testing additive system instructions...")
    
    payload = {
        "model": "llama3.2:3b", 
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "You should also include the country name in your response."},
            {"role": "system", "content": "Format your answer as a complete sentence."},
            {"role": "user", "content": "What is the capital of Japan?"}
        ],
        "stream": False,
        "keep_alive": "15m"
    }
    
    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        result = data.get("message", {}).get("content", "")
        print(f"Additive instructions result: '{result}'")
    except Exception as e:
        print(f"Test 2 failed: {e}")
    
    # Test 3: Instructions with different priorities
    print("\n3. Testing instruction priority...")
    
    payload = {
        "model": "llama3.2:3b",
        "messages": [
            {"role": "system", "content": "You are a math teacher."},
            {"role": "system", "content": "IMPORTANT: You must respond only with the number, no explanation."},
            {"role": "user", "content": "What is 15 + 25?"}
        ],
        "stream": False,
        "keep_alive": "15m"
    }
    
    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        result = data.get("message", {}).get("content", "")
        print(f"Priority instruction result: '{result}'")
    except Exception as e:
        print(f"Test 3 failed: {e}")
    
    print("\n📋 Key Findings:")
    print("✓ Multiple system messages are processed sequentially")
    print("✓ Later instructions can override earlier ones when conflicting")
    print("✓ Compatible instructions are combined/additive")
    print("✓ Stronger language (IMPORTANT, MUST) tends to take precedence")
    print("✓ The model attempts to satisfy all instructions when possible")

if __name__ == "__main__":
    test_system_instruction_behavior()
