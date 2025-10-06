#!/usr/bin/env python3
"""
Tests for security utilities module.
"""

import os
import tempfile
import unittest
from pathlib import Path
from security_utils import SecretsDetector, secure_print, validate_env_vars, mask_secrets

class TestSecretsDetector(unittest.TestCase):
    """Test cases for SecretsDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = SecretsDetector()
    
    def test_detect_api_keys(self):
        """Test detection of API keys."""
        test_text = "OPENAI_API_KEY=sk-1234567890abcdef1234567890abcdef"
        detected = self.detector.detect_secrets(test_text)
        
        self.assertIn('api_key', detected)
        self.assertGreater(len(detected['api_key']), 0)
        
        # Check that at least one detection found the full key
        values = [d['value'] for d in detected['api_key']]
        self.assertIn('sk-1234567890abcdef1234567890abcdef', values)
    
    def test_ignore_safe_values(self):
        """Test that safe/template values are ignored."""
        test_text = """
        api_key = "demo_key"
        TAVILY_API_KEY=your_tavily_api_key_here
        test_key = "fake_key_for_testing"
        """
        detected = self.detector.detect_secrets(test_text)
        
        # Should not detect safe values
        self.assertEqual(len(detected), 0)
    
    def test_mask_secrets(self):
        """Test secret masking functionality."""
        test_text = "My API key is sk-1234567890abcdef1234567890abcdef"
        masked = self.detector.mask_secrets(test_text)
        
        # Should mask the secret but keep last 4 characters
        self.assertIn('cdef', masked)
        self.assertNotIn('sk-1234567890abcdef1234567890abcdef', masked)
    
    def test_scan_file(self):
        """Test file scanning functionality."""
        # Create a temporary file with secrets
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('API_KEY = "sk-1234567890abcdef1234567890abcdef"\n')
            f.write('print("Hello world")\n')
            temp_file = f.name
        
        try:
            result = self.detector.scan_file(temp_file)
            
            self.assertFalse(result['skipped'])
            self.assertTrue(result['has_secrets'])
            self.assertIn('api_key', result['secrets'])
        finally:
            os.unlink(temp_file)
    
    def test_secure_print_masking(self):
        """Test that secure_print masks secrets."""
        # This test captures stdout to verify masking works
        import io
        import sys
        
        captured_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            secure_print("API Key:", "sk-1234567890abcdef1234567890abcdef")
            output = captured_output.getvalue()
            
            # Should contain masked version, not original
            self.assertNotIn('sk-1234567890abcdef1234567890abcdef', output)
            self.assertIn('***', output)  # Should contain masking characters
        finally:
            sys.stdout = original_stdout

class TestValidateEnvVars(unittest.TestCase):
    """Test cases for environment variable validation."""
    
    def test_missing_env_var(self):
        """Test validation of missing environment variable."""
        # Test with a variable that shouldn't exist
        results = validate_env_vars(['NONEXISTENT_TEST_VAR_12345'], warn_only=True)
        self.assertFalse(results['NONEXISTENT_TEST_VAR_12345'])
    
    def test_template_value_detection(self):
        """Test detection of template/placeholder values."""
        # Set a template value
        os.environ['TEST_API_KEY'] = 'your_api_key_here'
        
        try:
            results = validate_env_vars(['TEST_API_KEY'], warn_only=True)
            self.assertFalse(results['TEST_API_KEY'])
        finally:
            # Clean up
            del os.environ['TEST_API_KEY']

class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def test_mask_secrets_function(self):
        """Test the global mask_secrets function."""
        text = "API Key: sk-1234567890abcdef1234567890abcdef"
        masked = mask_secrets(text)
        
        self.assertNotIn('sk-1234567890abcdef1234567890abcdef', masked)
        self.assertIn('***', masked)

if __name__ == '__main__':
    # Run the tests
    print("🧪 Running Security Utilities Tests")
    print("=" * 50)
    
    unittest.main(verbosity=2)