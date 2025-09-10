#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Git cleanup script for the data_decomposition project
# This script removes cached files that should be ignored

set -e  # Exit on any error

echo "=========================================="
echo "Git Repository Cleanup"
echo "=========================================="

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Change to project root directory
cd "$PROJECT_ROOT"

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "Not a git repository. Initializing git repository..."
    git init
    echo "Git repository initialized."
fi

echo "Cleaning up cached files that should be ignored..."

# Remove files from git index that should be ignored
echo "Removing __pycache__ directories from git..."
git rm -r --cached . 2>/dev/null || true
find . -type d -name "__pycache__" -exec git rm -r --cached {} + 2>/dev/null || true

echo "Removing compiled Python files from git..."
find . -name "*.pyc" -exec git rm --cached {} + 2>/dev/null || true
find . -name "*.pyo" -exec git rm --cached {} + 2>/dev/null || true
find . -name "*.pyd" -exec git rm --cached {} + 2>/dev/null || true

echo "Removing IDE and editor files from git..."
git rm -r --cached .vscode/ 2>/dev/null || true
git rm -r --cached .idea/ 2>/dev/null || true
find . -name "*.swp" -exec git rm --cached {} + 2>/dev/null || true
find . -name "*.swo" -exec git rm --cached {} + 2>/dev/null || true

echo "Removing test artifacts from git..."
git rm -r --cached .pytest_cache/ 2>/dev/null || true
git rm -r --cached .mypy_cache/ 2>/dev/null || true
git rm -r --cached htmlcov/ 2>/dev/null || true
git rm --cached .coverage 2>/dev/null || true
git rm --cached coverage.xml 2>/dev/null || true

echo "Removing build artifacts from git..."
git rm -r --cached build/ 2>/dev/null || true
git rm -r --cached dist/ 2>/dev/null || true
git rm -r --cached *.egg-info/ 2>/dev/null || true

echo "Removing environment files from git..."
git rm -r --cached venv/ 2>/dev/null || true
git rm -r --cached env/ 2>/dev/null || true
git rm --cached .env 2>/dev/null || true

# Clean up the working directory as well
echo "Cleaning up working directory..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
find . -name "*.pyd" -delete 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

# Add all remaining files to git (respecting .gitignore)
echo "Adding files to git (respecting .gitignore)..."
git add .

echo "=========================================="
echo "Git cleanup completed!"
echo ""
echo "Summary of what was done:"
echo "- Removed __pycache__ directories and *.pyc files from git index"
echo "- Removed IDE configuration files from git index"
echo "- Removed test artifacts from git index"
echo "- Removed build artifacts from git index"
echo "- Cleaned up working directory"
echo "- Added remaining files respecting .gitignore"
echo ""
echo "Next steps:"
echo "1. Review changes: git status"
echo "2. Commit if satisfied: git commit -m 'Clean up repository and add .gitignore'"
echo "=========================================="
