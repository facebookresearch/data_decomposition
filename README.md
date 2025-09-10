# Data Decomposition beyond Splitting for Causal Inference

A self-contained library for causal inference data decomposition methods, including AIPW estimators, data generators, and various propensity and outcome models.

Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

## Quick Start

```bash
# Git setup
make git-clean                    # Clean git repository
make git-setup                    # Setup git repository

# Testing
make test                         # Run all tests
make test-verbose                 # Run tests with verbose output
make test-file FILE=test_aipw.py  # Run specific test file
```

## Usage

```python
from data_decomposition import AIPW, Splitting, BinomialGaussian
from data_decomposition import LogisticPropensityModel, OLSOutcomeModel

# Generate synthetic data
generator = BinomialGaussian(...)
data = generator.generate(num_samples=1000, treatment_effect=2.0)

# Split data
splitter = Splitting(train_ratio=0.6)
train_test_data = splitter.decompose(data)

# Estimate treatment effect
estimator = AIPW()
estimator.fit(train_test_data, LogisticPropensityModel(), OLSOutcomeModel())
ate = estimator.ate()
```

## Project Structure

- `src/data_decomposition/` - Main source code
- `tests/` - Test files
- `requirements.txt` - Dependencies
- `setup.py` - Package configuration
- `Makefile` - Build automation
- `.gitignore` - Git ignore rules
