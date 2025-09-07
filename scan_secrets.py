#!/usr/bin/env python3
"""
Repository secrets scanner for the Jarvis Offline project.

This script scans the entire repository for potential secrets and generates a report.
"""

import argparse
import json
import sys
from pathlib import Path
from security_utils import SecretsDetector, secure_print

def scan_repository(directory: str = ".", output_format: str = "text", verbose: bool = False) -> dict:
    """Scan the repository for secrets.
    
    Args:
        directory: Directory to scan (default: current directory)
        output_format: Output format ('text', 'json')
        verbose: Show verbose output
        
    Returns:
        Dictionary with scan results
    """
    detector = SecretsDetector()
    
    # Define file patterns to scan
    patterns = ['*.py', '*.js', '*.sh', '*.env*', '*.txt', '*.md', '*.yaml', '*.yml']
    
    # Scan the repository
    results = detector.scan_directory(
        directory=directory,
        patterns=patterns,
        exclude_dirs=['.git', '__pycache__', '.venv', 'venv', 'node_modules', '.pytest_cache'],
        exclude_extensions=['.pyc', '.pyo', '.log', '.tmp', '.cache']
    )
    
    return results

def print_results(results: dict, verbose: bool = False):
    """Print scan results in human-readable format."""
    secure_print(f"\n🔍 Repository Secrets Scan Results")
    secure_print(f"=" * 50)
    secure_print(f"Directory: {results['directory']}")
    secure_print(f"Files scanned: {results['files_scanned']}")
    secure_print(f"Files with potential secrets: {results['files_with_secrets']}")
    secure_print(f"Files skipped: {results['files_skipped']}")
    
    if results['files_with_secrets'] > 0:
        secure_print(f"\n⚠️  Files with potential secrets:")
        secure_print("-" * 40)
        
        for result in results['results']:
            if result.get('has_secrets'):
                secure_print(f"\n📁 {result['file']}")
                
                for category, secrets in result['secrets'].items():
                    if secrets:
                        secure_print(f"  {category.upper()}:")
                        for secret in secrets:
                            if verbose:
                                secure_print(f"    - Pattern: {secret['pattern']}")
                                secure_print(f"      Match: {secret['full_match']}")
                                secure_print(f"      Position: {secret['start']}-{secret['end']}")
                            else:
                                secure_print(f"    - {secret['full_match']}")
    
    if verbose and results['files_skipped'] > 0:
        secure_print(f"\n📋 Skipped files:")
        for result in results['results']:
            if result.get('skipped'):
                reason = result.get('reason', 'unknown')
                secure_print(f"  {result['file']} ({reason})")
    
    if results['files_with_secrets'] == 0:
        secure_print(f"\n✅ No potential secrets detected!")
    else:
        secure_print(f"\n⚠️  Please review the detected items and ensure they are not real secrets.")

def main():
    """Main entry point for the secrets scanner."""
    parser = argparse.ArgumentParser(
        description="Scan repository for potential secrets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scan_secrets.py                    # Scan current directory
  python scan_secrets.py --verbose          # Verbose output
  python scan_secrets.py --format json      # JSON output
  python scan_secrets.py /path/to/repo      # Scan specific directory
        """
    )
    
    parser.add_argument(
        'directory',
        nargs='?',
        default='.',
        help='Directory to scan (default: current directory)'
    )
    
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose output'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file (default: stdout)'
    )
    
    args = parser.parse_args()
    
    try:
        # Scan the repository
        results = scan_repository(args.directory, args.format, args.verbose)
        
        # Output results
        if args.format == 'json':
            output = json.dumps(results, indent=2)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(output)
                secure_print(f"Results written to {args.output}")
            else:
                print(output)
        else:
            if args.output:
                # Redirect stdout to file for text output
                original_stdout = sys.stdout
                with open(args.output, 'w') as f:
                    sys.stdout = f
                    print_results(results, args.verbose)
                sys.stdout = original_stdout
                secure_print(f"Results written to {args.output}")
            else:
                print_results(results, args.verbose)
        
        # Exit with appropriate code
        exit_code = 1 if results['files_with_secrets'] > 0 else 0
        sys.exit(exit_code)
        
    except Exception as e:
        secure_print(f"❌ Error scanning repository: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()