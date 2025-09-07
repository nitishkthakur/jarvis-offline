# Security Implementation Guide

This document describes the security enhancements implemented in the Jarvis Offline project to detect and mask secrets.

## Overview

The security implementation provides comprehensive protection against accidental exposure of sensitive data such as API keys, tokens, passwords, and other secrets in code, logs, and output.

## Components

### 1. Security Utilities (`security_utils.py`)

The main security module that provides:

- **SecretsDetector**: Pattern-based detection of common secrets
- **SecureLogger**: Logger that automatically masks secrets
- **secure_print()**: Print function with automatic secret masking
- **validate_env_vars()**: Environment variable validation
- **Repository scanning**: Scan files and directories for potential secrets

### 2. Repository Scanner (`scan_secrets.py`)

A command-line tool to scan the entire repository for potential secrets:

```bash
# Scan current directory
python scan_secrets.py

# Scan with verbose output
python scan_secrets.py --verbose

# Output to JSON file
python scan_secrets.py --format json --output secrets_report.json
```

### 3. Test Suite (`test_security_utils.py`)

Comprehensive tests for all security functionality to ensure reliability.

## Secret Detection Patterns

The system detects the following types of secrets:

### API Keys
- OpenAI API keys (`sk-...`)
- Generic API keys with environment variable patterns
- Custom API key patterns

### Tokens
- Access tokens
- Bearer tokens
- Generic token patterns

### Passwords
- Password fields and variables
- Secret keys

### Safe Values
The system ignores known safe/template values:
- `demo_key`
- `your_api_key_here`
- `fake_key_for_testing`
- `test_key`
- And other common placeholder values

## Usage Examples

### Basic Secret Masking

```python
from security_utils import secure_print, mask_secrets

# Instead of print()
secure_print("API Key:", api_key)

# Manual masking
masked_text = mask_secrets("API Key: sk-1234567890abcdef")
print(masked_text)  # Output: API Key: **********************cdef
```

### Environment Variable Validation

```python
from security_utils import validate_env_vars

# Validate required environment variables
results = validate_env_vars(['OPENAI_API_KEY', 'TAVILY_API_KEY'])
if all(results.values()):
    print("All API keys are properly configured")
```

### Secure Logging

```python
from security_utils import SecureLogger

logger = SecureLogger('my_app')
logger.info(f"Using API key: {api_key}")  # Automatically masked
```

### Repository Scanning

```python
from security_utils import SecretsDetector

detector = SecretsDetector()
results = detector.scan_directory('.')

if results['files_with_secrets'] > 0:
    print("⚠️ Potential secrets found!")
```

## Files Updated

The following files have been updated to use secure printing and remove hardcoded secrets:

1. **`tools.py`**: Now uses `secure_print` and `validate_env_vars`
2. **`openai_client.py`**: Secure API key handling and imports
3. **`openrouter_client.py`**: Secure imports
4. **`exact_pattern_example.py`**: Replaced hardcoded keys with environment variables
5. **`deep_research_tools.py`**: Secure printing and validation

## Best Practices

### For Developers

1. **Always use environment variables** for sensitive data:
   ```python
   api_key = os.getenv('OPENAI_API_KEY')
   ```

2. **Use secure_print()** instead of print() for any output that might contain secrets:
   ```python
   from security_utils import secure_print
   secure_print(f"Processing with key: {api_key}")
   ```

3. **Validate environment variables** at startup:
   ```python
   from security_utils import validate_env_vars
   validate_env_vars(['REQUIRED_API_KEY'], warn_only=False)
   ```

4. **Use SecureLogger** for application logging:
   ```python
   from security_utils import SecureLogger
   logger = SecureLogger(__name__)
   logger.info("Application started")
   ```

### For Testing

1. **Use placeholder values** that are in the safe values list:
   ```python
   test_client = Client(api_key="demo_key")  # Safe value
   ```

2. **Never commit real secrets** to version control

3. **Run repository scans** regularly:
   ```bash
   python scan_secrets.py --verbose
   ```

## Environment Setup

### Required Environment Variables

Copy `.env.template` to `.env` and configure:

```bash
cp .env.template .env
```

Edit `.env` with your actual API keys:
```
TAVILY_API_KEY=your_actual_tavily_api_key
OPENAI_API_KEY=your_actual_openai_api_key
OPENROUTER_API_KEY=your_actual_openrouter_api_key
```

### Verification

Verify your setup:
```python
from security_utils import validate_env_vars
validate_env_vars(['TAVILY_API_KEY', 'OPENAI_API_KEY'])
```

## Security Features

1. **Pattern Recognition**: Detects secrets using regex patterns
2. **Automatic Masking**: Masks detected secrets in output (keeps last 4 characters visible)
3. **Safe Value Detection**: Ignores known demo/template values
4. **Environment Validation**: Checks for missing or template environment variables
5. **Repository Scanning**: Scans entire codebase for potential secrets
6. **Test Coverage**: Comprehensive test suite ensures reliability

## Maintenance

### Adding New Secret Patterns

To add detection for new types of secrets:

```python
custom_patterns = {
    'new_secret_type': [
        r'NEW_SECRET\s*[:=]\s*["\']?([A-Za-z0-9_\-]{20,})["\']?'
    ]
}

detector = SecretsDetector(custom_patterns)
```

### Adding Safe Values

To add new safe/template values:

```python
# In security_utils.py, add to SAFE_VALUES set
SAFE_VALUES.add('new_template_value')
```

## Troubleshooting

### False Positives

If the scanner detects false positives:

1. Check if the value should be added to `SAFE_VALUES`
2. Adjust regex patterns if needed
3. Use the `include_safe_values=False` parameter to exclude template values

### Missing Detections

If real secrets are not detected:

1. Review and improve regex patterns
2. Add new patterns for specific secret formats
3. Test with the pattern before deployment

## Integration with CI/CD

Add to your CI pipeline:

```bash
# Fail if secrets are detected
python scan_secrets.py
if [ $? -eq 1 ]; then
    echo "❌ Potential secrets detected! Please review."
    exit 1
fi
```

This security implementation provides comprehensive protection while maintaining usability and minimizing false positives.