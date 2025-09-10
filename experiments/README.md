# Treatment Effect Estimation Experiments

This folder contains experiments comparing data splitting vs thinning fission (data decomposition) for causal inference across different data generation processes (DGPs).

## Structure

```
experiments/
├── configs/                          # YAML configuration files
│   ├── linear_homoskedastic.yaml     # Linear Gaussian (homoskedastic)
│   ├── linear_heteroskedastic.yaml   # Linear Gaussian (heteroskedastic)
│   ├── athey_imbens_nonlinear.yaml   # Athey & Imbens (2016) DGP
│   └── polynomial_nonlinear.yaml     # Polynomial (quadratic) DGP
├── scripts/                          # Experiment scripts
│   └── treatment_only.py             # Main treatment experiment runner
├── utils.py                          # Shared utilities and functions
├── results/                          # Generated results (gitignored)
├── plots/                            # Generated plots (gitignored)
└── README.md                         # This file
```

## Running Experiments

```bash
# Run all DGPs (100 repetitions each)
make experiments

# Quick test (10 repetitions per DGP)
make experiments-quick

# Run single DGP
make experiments-single DGP=linear_homoskedastic
make experiments-single DGP=athey_imbens

# Run directly with options
cd /home/dsin/projects/data_decomposition
PYTHONPATH=src python3 experiments/scripts/treatment_only.py --quick
PYTHONPATH=src python3 experiments/scripts/treatment_only.py --dgps linear_homoskedastic polynomial
PYTHONPATH=src python3 experiments/scripts/treatment_only.py --dims 10 20 30
```

## Data Generation Processes (DGPs)

All DGPs compare **data splitting** vs **thinning fission** for treatment effect estimation:

1. **`linear_homoskedastic`** - Linear Gaussian with constant noise variance
2. **`linear_heteroskedastic`** - Linear Gaussian with beta-prime heteroskedastic noise
3. **`athey_imbens`** - Non-linear treatment effects (Athey & Imbens, 2016)
4. **`polynomial`** - Non-linear quadratic outcome model

## Configuration

Each DGP is configured via YAML files in `configs/` specifying:
- Data generation parameters (sample size, covariates, noise)
- DGP-specific model settings
- Experiment repetitions
- Plotting preferences

Results are saved as pickle files in `results/` and plots as PDFs in `plots/`.

## Available Commands

```bash
# Show available DGPs
PYTHONPATH=src python3 experiments/scripts/treatment_only.py --help

# Run specific DGPs only
PYTHONPATH=src python3 experiments/scripts/treatment_only.py --dgps linear_homoskedastic polynomial

# Run single DGP
PYTHONPATH=src python3 experiments/scripts/treatment_only.py --single athey_imbens
```
