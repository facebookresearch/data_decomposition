#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e  # Exit on any error

# Script to run all tests for the data_decomposition project

echo "=========================================="
echo "Running Data Decomposition Test Suite"
echo "=========================================="

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Change to project root directory
cd "$PROJECT_ROOT"

# Check if running in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Warning: No virtual environment detected."
    echo "Consider creating and activating a virtual environment:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate  # On Linux/Mac"
    echo "  # or"
    echo "  venv\\Scripts\\activate     # On Windows"
    echo
fi

# Check if dependencies are installed
echo "Checking dependencies..."
python3 -c "
import sys
required_packages = ['numpy', 'pandas', 'sklearn', 'scipy', 'statsmodels', 'testslide']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f'Missing packages: {missing_packages}')
    print('Install them with: pip install -r requirements.txt')
    sys.exit(1)
else:
    print('All dependencies are installed.')
" || {
    echo "Please install the required dependencies:"
    echo "  pip install -r requirements.txt"
    exit 1
}

# Add src directory to Python path so imports work correctly
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

echo
echo "Running tests..."
echo "=================="

# Test discovery and execution options
TEST_OPTIONS=""
if [[ "$1" == "--verbose" || "$1" == "-v" ]]; then
    TEST_OPTIONS="--verbose"
    echo "Running in verbose mode..."
fi

# Run tests using testslide's test runner
# Note: testslide expects individual .py files, not directories
if command -v testslide >/dev/null 2>&1; then
    echo "Using testslide test runner..."
    testslide tests/test_*.py $TEST_OPTIONS
elif python3 -c "import testslide" 2>/dev/null; then
    echo "Using testslide via python module..."
    python3 -m testslide tests/test_*.py $TEST_OPTIONS
else
    echo "testslide not found. Trying with pytest as fallback..."
    if command -v pytest >/dev/null 2>&1; then
        pytest tests/ -v $TEST_OPTIONS
    elif python3 -c "import pytest" 2>/dev/null; then
        python3 -m pytest tests/ -v $TEST_OPTIONS
    else
        echo "Neither testslide nor pytest found. Running tests individually..."

        # Individual test execution as fallback
        test_files=(
            "tests/test_aipw.py"
            "tests/test_data.py"
            "tests/test_outcome.py"
            "tests/test_propensity.py"
        )

        failed_tests=()

        for test_file in "${test_files[@]}"; do
            if [[ -f "$test_file" ]]; then
                echo "Running $test_file..."
                if python3 -m unittest discover -s "$(dirname "$test_file")" -p "$(basename "$test_file")" $TEST_OPTIONS; then
                    echo "✓ $test_file passed"
                else
                    echo "✗ $test_file failed"
                    failed_tests+=("$test_file")
                fi
            else
                echo "Warning: $test_file not found"
            fi
        done

        if [[ ${#failed_tests[@]} -gt 0 ]]; then
            echo
            echo "Failed tests:"
            for test in "${failed_tests[@]}"; do
                echo "  - $test"
            done
            exit 1
        fi
    fi
fi

echo
echo "=========================================="
echo "All tests completed successfully! ✓"
echo "=========================================="
