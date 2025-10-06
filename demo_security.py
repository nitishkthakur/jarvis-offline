#!/usr/bin/env python3
"""
Demonstration of the security features implemented in the Jarvis Offline project.

This script shows how the security utilities protect against secret exposure.
"""

import os
from security_utils import (
    secure_print, 
    validate_env_vars, 
    SecretsDetector, 
    SecureLogger,
    mask_secrets
)

def demo_secret_detection():
    """Demonstrate secret detection capabilities."""
    secure_print("\n🔍 Secret Detection Demo")
    secure_print("=" * 50)
    
    detector = SecretsDetector()
    
    # Test various types of secrets
    test_cases = [
        "OPENAI_API_KEY=sk-1234567890abcdef1234567890abcdef",
        "My password is super_secret_password_123456",
        "Bearer token: bearer_token_xyz789abc123def456",
        "api_key = 'demo_key'",  # This should be ignored (safe value)
        "TAVILY_API_KEY=your_tavily_api_key_here",  # Template value - ignored
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        secure_print(f"\nTest {i}: Original text")
        print(f"  {test_text}")  # Use regular print to show original
        
        detected = detector.detect_secrets(test_text)
        if detected:
            secure_print(f"  ⚠️  Detected: {list(detected.keys())}")
        else:
            secure_print(f"  ✅ No secrets detected (safe/template value)")
        
        masked = detector.mask_secrets(test_text)
        secure_print(f"  Masked: {masked}")

def demo_secure_printing():
    """Demonstrate secure printing functionality."""
    secure_print("\n🔒 Secure Printing Demo")
    secure_print("=" * 50)
    
    # Simulate API keys
    fake_openai_key = "sk-1234567890abcdef1234567890abcdef"
    fake_tavily_key = "tavily_key_abcdef123456789"
    
    secure_print("Regular text prints normally")
    secure_print("API key gets masked:", fake_openai_key)
    secure_print("Multiple secrets:", fake_openai_key, "and", fake_tavily_key)
    
    # Show comparison with regular print
    print("\nComparison - Regular print() exposes secrets:")
    print(f"API key: {fake_openai_key}")
    
    secure_print("Secure print() masks secrets:")
    secure_print(f"API key: {fake_openai_key}")

def demo_environment_validation():
    """Demonstrate environment variable validation."""
    secure_print("\n🌍 Environment Validation Demo")
    secure_print("=" * 50)
    
    # Test with non-existent variables
    secure_print("Testing with non-existent variables:")
    validate_env_vars(['NONEXISTENT_KEY_1', 'NONEXISTENT_KEY_2'], warn_only=True)
    
    # Set a template value for testing
    os.environ['TEST_TEMPLATE_KEY'] = 'your_api_key_here'
    secure_print("\nTesting with template value:")
    validate_env_vars(['TEST_TEMPLATE_KEY'], warn_only=True)
    
    # Clean up
    del os.environ['TEST_TEMPLATE_KEY']

def demo_secure_logging():
    """Demonstrate secure logging capabilities."""
    secure_print("\n📝 Secure Logging Demo")
    secure_print("=" * 50)
    
    logger = SecureLogger('demo_app')
    
    fake_api_key = "sk-1234567890abcdef1234567890abcdef"
    
    secure_print("Logging with secrets - check console output:")
    logger.info(f"Application started with API key: {fake_api_key}")
    logger.warning(f"Rate limit warning for key: {fake_api_key}")
    logger.error(f"Authentication failed for key: {fake_api_key}")

def demo_repository_scanning():
    """Demonstrate repository scanning."""
    secure_print("\n📂 Repository Scanning Demo")
    secure_print("=" * 50)
    
    detector = SecretsDetector()
    
    # Scan current directory (limited to a few file types for demo)
    secure_print("Scanning repository for secrets...")
    results = detector.scan_directory(
        directory=".",
        patterns=["*.py"],
        exclude_dirs=['.git', '__pycache__', '.venv']
    )
    
    secure_print(f"Files scanned: {results['files_scanned']}")
    secure_print(f"Files with potential secrets: {results['files_with_secrets']}")
    secure_print(f"Files skipped: {results['files_skipped']}")
    
    if results['files_with_secrets'] > 0:
        secure_print("\nFiles with potential secrets (showing first 3):")
        count = 0
        for result in results['results']:
            if result.get('has_secrets') and count < 3:
                secure_print(f"  📁 {result['file']}")
                for category in result['secrets']:
                    secure_print(f"    - {category}: {len(result['secrets'][category])} detections")
                count += 1

def main():
    """Main demonstration function."""
    secure_print("🛡️  JARVIS OFFLINE - SECURITY IMPLEMENTATION DEMO")
    secure_print("=" * 60)
    secure_print("This demo shows how the security utilities protect against")
    secure_print("accidental exposure of sensitive data like API keys and tokens.")
    
    # Run all demos
    demo_secret_detection()
    demo_secure_printing()
    demo_environment_validation()
    demo_secure_logging()
    demo_repository_scanning()
    
    secure_print("\n✅ Security Demo Complete!")
    secure_print("=" * 60)
    secure_print("Key Features Demonstrated:")
    secure_print("• Automatic secret detection and masking")
    secure_print("• Secure printing that protects sensitive data")
    secure_print("• Environment variable validation")
    secure_print("• Secure logging with automatic masking")
    secure_print("• Repository-wide secret scanning")
    secure_print("\nFor more information, see SECURITY.md")

if __name__ == "__main__":
    main()