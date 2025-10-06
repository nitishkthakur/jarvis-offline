#!/usr/bin/env python3
"""
Security utilities for detecting and masking secrets in the Jarvis Offline project.

This module provides:
- Pattern-based secrets detection
- Automatic masking of sensitive data in outputs
- Environment variable validation
- Secure logging utilities
- Repository scanning for potential secrets
"""

import re
import os
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# Common secret patterns
SECRET_PATTERNS = {
    'api_key': [
        r'\b[Aa][Pp][Ii]_?[Kk][Ee][Yy]\s*[:=]\s*["\']?([A-Za-z0-9_\-]{20,})["\']?',
        r'\b[Aa][Pp][Ii][Kk][Ee][Yy]\s*[:=]\s*["\']?([A-Za-z0-9_\-]{20,})["\']?',
        r'OPENAI_API_KEY\s*[:=]\s*["\']?([A-Za-z0-9_\-]{20,})["\']?',
        r'OPENROUTER_API_KEY\s*[:=]\s*["\']?([A-Za-z0-9_\-]{20,})["\']?',
        r'TAVILY_API_KEY\s*[:=]\s*["\']?([A-Za-z0-9_\-]{20,})["\']?',
        r'\bsk-[A-Za-z0-9]{32,}',  # OpenAI-style keys
        r'\b[A-Za-z0-9]{32,}\b',  # Generic long alphanumeric strings
    ],
    'secret_key': [
        r'\b[Ss][Ee][Cc][Rr][Ee][Tt]_?[Kk][Ee][Yy]\s*[:=]\s*["\']?([A-Za-z0-9_\-]{20,})["\']?',
        r'SECRET_KEY\s*[:=]\s*["\']?([A-Za-z0-9_\-]{20,})["\']?',
    ],
    'token': [
        r'\b[Tt][Oo][Kk][Ee][Nn]\s*[:=]\s*["\']?([A-Za-z0-9_\-]{20,})["\']?',
        r'ACCESS_TOKEN\s*[:=]\s*["\']?([A-Za-z0-9_\-]{20,})["\']?',
        r'BEARER_TOKEN\s*[:=]\s*["\']?([A-Za-z0-9_\-]{20,})["\']?',
    ],
    'password': [
        r'\b[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd]\s*[:=]\s*["\']?([A-Za-z0-9_\-!@#$%^&*()]{8,})["\']?',
        r'PASSWORD\s*[:=]\s*["\']?([A-Za-z0-9_\-!@#$%^&*()]{8,})["\']?',
    ],
}

# Known safe/template values that should not be considered secrets
SAFE_VALUES = {
    'your_api_key_here',
    'your_tavily_api_key_here',
    'your-api-key-here',
    'your-openai-api-key',
    'demo_key',
    'fake_key_for_testing',
    'test_key',
    'fake',
    'test_key_for_offline_testing',
    'fake_for_demo',
    'fake_key_for_testing',
    'example_key',
    'placeholder_key',
}

class SecretsDetector:
    """Detects and manages secrets in text and files."""
    
    def __init__(self, custom_patterns: Optional[Dict[str, List[str]]] = None):
        """Initialize the secrets detector.
        
        Args:
            custom_patterns: Additional patterns to detect beyond the defaults
        """
        self.patterns = SECRET_PATTERNS.copy()
        if custom_patterns:
            for category, patterns in custom_patterns.items():
                if category in self.patterns:
                    self.patterns[category].extend(patterns)
                else:
                    self.patterns[category] = patterns
        
        # Compile regex patterns for better performance
        self.compiled_patterns = {}
        for category, patterns in self.patterns.items():
            self.compiled_patterns[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def detect_secrets(self, text: str, include_safe_values: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Detect potential secrets in text.
        
        Args:
            text: Text to scan for secrets
            include_safe_values: Whether to include known safe/template values
            
        Returns:
            Dictionary with categories as keys and lists of detected secrets as values
        """
        detected = {}
        
        for category, compiled_patterns in self.compiled_patterns.items():
            detected[category] = []
            
            for pattern in compiled_patterns:
                for match in pattern.finditer(text):
                    # Handle patterns with and without capture groups
                    if match.groups():
                        secret_value = match.group(1)
                        full_match = match.group(0)
                    else:
                        secret_value = match.group(0)
                        full_match = match.group(0)
                    
                    # Skip safe/template values unless explicitly requested
                    if not include_safe_values and secret_value.lower() in SAFE_VALUES:
                        continue
                    
                    detected[category].append({
                        'value': secret_value,
                        'start': match.start(),
                        'end': match.end(),
                        'full_match': full_match,
                        'pattern': pattern.pattern
                    })
        
        # Remove empty categories
        return {k: v for k, v in detected.items() if v}
    
    def mask_secrets(self, text: str, mask_char: str = '*', keep_chars: int = 4) -> str:
        """Mask detected secrets in text.
        
        Args:
            text: Text to mask secrets in
            mask_char: Character to use for masking
            keep_chars: Number of characters to keep visible at the end
            
        Returns:
            Text with secrets masked
        """
        masked_text = text
        detected = self.detect_secrets(text)
        
        # Sort by position (reverse order to maintain positions)
        all_secrets = []
        for category, secrets in detected.items():
            for secret in secrets:
                all_secrets.append(secret)
        
        all_secrets.sort(key=lambda x: x['start'], reverse=True)
        
        for secret in all_secrets:
            value = secret['value']
            if len(value) <= keep_chars:
                # If value is too short, mask most of it but keep 1 char
                masked_value = mask_char * (len(value) - 1) + value[-1:]
            else:
                # Keep last few characters visible
                masked_value = mask_char * (len(value) - keep_chars) + value[-keep_chars:]
            
            # Replace in text
            masked_text = masked_text[:secret['start']] + \
                         secret['full_match'].replace(value, masked_value) + \
                         masked_text[secret['end']:]
        
        return masked_text
    
    def scan_file(self, file_path: Union[str, Path], exclude_extensions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Scan a file for potential secrets.
        
        Args:
            file_path: Path to the file to scan
            exclude_extensions: File extensions to skip (e.g., ['.pyc', '.log'])
            
        Returns:
            Dictionary with file info and detected secrets
        """
        file_path = Path(file_path)
        
        if exclude_extensions and file_path.suffix in exclude_extensions:
            return {'file': str(file_path), 'skipped': True, 'reason': 'excluded_extension'}
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            detected = self.detect_secrets(content)
            
            return {
                'file': str(file_path),
                'size': file_path.stat().st_size,
                'secrets': detected,
                'has_secrets': bool(detected),
                'skipped': False
            }
        
        except Exception as e:
            return {
                'file': str(file_path),
                'error': str(e),
                'skipped': True,
                'reason': 'read_error'
            }
    
    def scan_directory(self, directory: Union[str, Path], 
                      patterns: Optional[List[str]] = None,
                      exclude_dirs: Optional[List[str]] = None,
                      exclude_extensions: Optional[List[str]] = None) -> Dict[str, Any]:
        """Scan a directory for potential secrets.
        
        Args:
            directory: Directory to scan
            patterns: File patterns to include (e.g., ['*.py', '*.js'])
            exclude_dirs: Directory names to exclude (e.g., ['.git', '__pycache__'])
            exclude_extensions: File extensions to skip
            
        Returns:
            Dictionary with scan results
        """
        directory = Path(directory)
        
        if exclude_dirs is None:
            exclude_dirs = ['.git', '__pycache__', '.venv', 'venv', 'node_modules', '.pytest_cache']
        
        if exclude_extensions is None:
            exclude_extensions = ['.pyc', '.pyo', '.log', '.tmp', '.cache']
        
        results = {
            'directory': str(directory),
            'files_scanned': 0,
            'files_with_secrets': 0,
            'files_skipped': 0,
            'results': []
        }
        
        def should_skip_dir(dir_name: str) -> bool:
            return dir_name in exclude_dirs or dir_name.startswith('.')
        
        for root, dirs, files in os.walk(directory):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not should_skip_dir(d)]
            
            for file in files:
                file_path = Path(root) / file
                
                # Apply file patterns if specified
                if patterns and not any(file_path.match(pattern) for pattern in patterns):
                    continue
                
                result = self.scan_file(file_path, exclude_extensions)
                results['results'].append(result)
                
                if result.get('skipped'):
                    results['files_skipped'] += 1
                else:
                    results['files_scanned'] += 1
                    if result.get('has_secrets'):
                        results['files_with_secrets'] += 1
        
        return results

class SecureLogger:
    """Logger that automatically masks secrets in log messages."""
    
    def __init__(self, name: str, detector: Optional[SecretsDetector] = None):
        """Initialize secure logger.
        
        Args:
            name: Logger name
            detector: SecretsDetector instance to use
        """
        self.logger = logging.getLogger(name)
        self.detector = detector or SecretsDetector()
    
    def _mask_message(self, message: str) -> str:
        """Mask secrets in a log message."""
        return self.detector.mask_secrets(message)
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message with secrets masked."""
        self.logger.debug(self._mask_message(message), *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message with secrets masked."""
        self.logger.info(self._mask_message(message), *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message with secrets masked."""
        self.logger.warning(self._mask_message(message), *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message with secrets masked."""
        self.logger.error(self._mask_message(message), *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message with secrets masked."""
        self.logger.critical(self._mask_message(message), *args, **kwargs)

def secure_print(*args, detector: Optional[SecretsDetector] = None, **kwargs):
    """Print function that automatically masks secrets.
    
    Args:
        *args: Arguments to print
        detector: SecretsDetector instance to use
        **kwargs: Keyword arguments for print function
    """
    if detector is None:
        detector = SecretsDetector()
    
    # Join all args into a single string for detection, then mask
    combined_text = ' '.join(str(arg) for arg in args)
    masked_text = detector.mask_secrets(combined_text)
    
    print(masked_text, **kwargs)

def validate_env_vars(required_vars: List[str], warn_only: bool = True) -> Dict[str, bool]:
    """Validate that required environment variables are set and not using template values.
    
    Args:
        required_vars: List of environment variable names to check
        warn_only: If True, only warn about issues; if False, raise exceptions
        
    Returns:
        Dictionary mapping variable names to validation status
    """
    results = {}
    detector = SecretsDetector()
    
    for var_name in required_vars:
        value = os.getenv(var_name)
        
        if not value:
            results[var_name] = False
            message = f"Environment variable {var_name} is not set"
            if warn_only:
                secure_print(f"⚠️  {message}", detector=detector)
            else:
                raise ValueError(message)
        elif value.lower() in SAFE_VALUES:
            results[var_name] = False
            message = f"Environment variable {var_name} appears to be using a template/placeholder value"
            if warn_only:
                secure_print(f"⚠️  {message}", detector=detector)
            else:
                raise ValueError(message)
        else:
            results[var_name] = True
            secure_print(f"✅ Environment variable {var_name} is properly configured", detector=detector)
    
    return results

# Global detector instance for convenience
_global_detector = SecretsDetector()

def mask_secrets(text: str) -> str:
    """Convenience function to mask secrets using global detector."""
    return _global_detector.mask_secrets(text)

def detect_secrets(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """Convenience function to detect secrets using global detector."""
    return _global_detector.detect_secrets(text)

if __name__ == "__main__":
    # Demo and testing
    detector = SecretsDetector()
    
    # Test secret detection
    test_text = """
    OPENAI_API_KEY=sk-1234567890abcdef1234567890abcdef
    api_key = "demo_key"
    TAVILY_API_KEY=your_tavily_api_key_here
    secret_key = "super_secret_password_123456"
    """
    
    print("🔍 Testing secret detection:")
    detected = detector.detect_secrets(test_text)
    for category, secrets in detected.items():
        print(f"\n{category.upper()}:")
        for secret in secrets:
            print(f"  - Found: {secret['full_match']}")
    
    print("\n🎭 Testing secret masking:")
    masked = detector.mask_secrets(test_text)
    print(masked)
    
    print("\n🔒 Testing secure print:")
    secure_print("API Key:", "sk-1234567890abcdef1234567890abcdef")
    
    print("\n✅ Security utilities demo complete!")