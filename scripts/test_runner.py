#!/usr/bin/env python3
"""
Test runner script for the Persona Classifier project.
"""

import subprocess
import sys
import os

def run_tests(test_type="all", coverage=True, verbose=True):
    """
    Run tests with specified options.
    
    Args:
        test_type (str): Type of tests to run ('unit', 'integration', 'e2e', 'all')
        coverage (bool): Whether to run with coverage
        verbose (bool): Whether to run in verbose mode
    """
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html"])
    
    # Add verbose flag
    if verbose:
        cmd.append("-v")
    
    # Add test type filter
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "e2e":
        cmd.extend(["-m", "e2e"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    elif test_type != "all":
        print(f"Unknown test type: {test_type}")
        print("Available types: unit, integration, e2e, fast, all")
        return False
    
    # Add tests directory
    cmd.append("tests/")
    
    print(f"Running tests: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("-" * 50)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 50)
        print(f"❌ Tests failed with exit code: {e.returncode}")
        return False

def run_linting():
    """Run code linting checks."""
    print("Running linting checks...")
    
    # Check if flake8 is available
    try:
        result = subprocess.run(["flake8", "src/", "tests/"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Linting passed!")
            return True
        else:
            print("❌ Linting failed:")
            print(result.stdout)
            return False
    except FileNotFoundError:
        print("⚠️  flake8 not found. Install with: pip install flake8")
        return True

def run_type_checking():
    """Run type checking with mypy."""
    print("Running type checking...")
    
    # Check if mypy is available
    try:
        result = subprocess.run(["mypy", "src/"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Type checking passed!")
            return True
        else:
            print("❌ Type checking failed:")
            print(result.stdout)
            return False
    except FileNotFoundError:
        print("⚠️  mypy not found. Install with: pip install mypy")
        return True

def main():
    """Main function to run tests and checks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for Persona Classifier")
    parser.add_argument("--type", choices=["unit", "integration", "e2e", "fast", "all"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--no-coverage", action="store_true", 
                       help="Disable coverage reporting")
    parser.add_argument("--lint", action="store_true", 
                       help="Run linting checks")
    parser.add_argument("--type-check", action="store_true", 
                       help="Run type checking")
    parser.add_argument("--all-checks", action="store_true", 
                       help="Run all checks (tests, linting, type checking)")
    
    args = parser.parse_args()
    
    success = True
    
    # Run tests
    if not run_tests(args.type, not args.no_coverage):
        success = False
    
    # Run additional checks if requested
    if args.lint or args.all_checks:
        if not run_linting():
            success = False
    
    if args.type_check or args.all_checks:
        if not run_type_checking():
            success = False
    
    if success:
        print("\n🎉 All checks passed!")
        sys.exit(0)
    else:
        print("\n💥 Some checks failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 