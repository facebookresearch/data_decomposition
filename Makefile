# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Makefile for data_decomposition project

.PHONY: test test-verbose install clean help dev-install format lint type-check check-all git-clean git-setup

# Default target
help:
	@echo "Data Decomposition Project - Available Commands:"
	@echo ""
	@echo "  test              Run all tests"
	@echo "  test-verbose      Run all tests with verbose output"
	@echo "  install           Install project dependencies"
	@echo "  dev-install       Install development dependencies"
	@echo "  clean             Clean up generated files"
	@echo "  format            Format code using black"
	@echo "  lint              Run linting checks"
	@echo "  type-check        Run type checking"
	@echo "  check-all         Run tests, linting, and type checking"
	@echo "  git-clean         Clean up git repository from ignored files"
	@echo "  git-setup         Setup git repository (clean + add files)"
	@echo "  help              Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make test         # Run the test suite"
	@echo "  make install      # Install dependencies"
	@echo "  make check-all    # Full validation pipeline"
	@echo "  make git-setup    # Setup git repository"

# Test commands
test:
	@echo "Running tests..."
	@./run_tests.sh

test-verbose:
	@echo "Running tests with verbose output..."
	@./run_tests.sh --verbose

# Installation commands
install:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt

dev-install: install
	@echo "Installing development dependencies..."
	@pip install black flake8 mypy pytest pytest-cov

# Code quality commands
format:
	@echo "Formatting code with black..."
	@python -m black . --line-length 88 --target-version py38

lint:
	@echo "Running linting checks..."
	@python -m flake8 . --max-line-length=88 --ignore=E203,W503,E501

type-check:
	@echo "Running type checks..."
	@python -m mypy . --ignore-missing-imports

# Comprehensive check
check-all: format lint type-check test
	@echo "All checks completed successfully!"

# Clean up
clean:
	@echo "Cleaning up generated files..."
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name "*.pyd" -delete 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".tox" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -name ".coverage" -delete 2>/dev/null || true
	@find . -name "coverage.xml" -delete 2>/dev/null || true
	@rm -rf build/ dist/ 2>/dev/null || true
	@echo "Clean up completed."

# Quick quality checks (without tests)
quick-check: format lint type-check
	@echo "Quick quality checks completed!"

# Test with coverage
test-coverage:
	@echo "Running tests with coverage..."
	@PYTHONPATH=src python -m pytest tests/ --cov=src/data_decomposition --cov-report=html --cov-report=term

# Run specific test file
test-file:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make test-file FILE=test_filename.py"; \
		echo "Example: make test-file FILE=test_aipw.py"; \
	else \
		echo "Running tests for $(FILE)..."; \
		PYTHONPATH=src python -m pytest tests/$(FILE) -v; \
	fi

# Setup development environment
setup-dev: dev-install
	@echo "Setting up development environment..."
	@echo "Dependencies installed."
	@echo "Run 'make test' to verify everything works."

# Git commands
git-clean:
	@echo "Running git cleanup script..."
	@./git_clean.sh

git-setup: git-clean
	@echo "Setting up git repository..."
	@echo "Repository cleaned and ready for commit."

# Version display
version:
	@echo "Data Decomposition Project"
	@echo "MIT License - Copyright (c) Meta Platforms, Inc. and affiliates."
