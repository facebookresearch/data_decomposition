#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Treatment Effect Estimation Experiment

This script runs experiments comparing data splitting vs thinning fission (data decomposition)
for causal inference across different data generation processes (DGPs):

1. Linear Gaussian (Homoskedastic)
2. Linear Gaussian (Heteroskedastic)
3. Non-linear (Athey & Imbens, 2016)
4. Non-linear (Quadratic)

Each DGP compares:
- Traditional data splitting approach
- Thinning fission (data decomposition) approach
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add the experiments directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import (
    generate_athey_imbens_data,
    generate_linear_heteroskedastic_data,
    generate_linear_homoskedastic_data,
    generate_polynomial_data,
    get_ratio_se,
    load_config,
    plot_experiment_results,
    plot_se_ratios,
    run_experiment_from_config,
)


# Map DGP names to their config files and data generation functions
DGP_CONFIGS = {
    "linear_homoskedastic": {
        "config_file": "linear_homoskedastic.yaml",
        "generate_data_func": generate_linear_homoskedastic_data,
        "description": "Linear Gaussian with homoskedastic noise",
    },
    "linear_heteroskedastic": {
        "config_file": "linear_heteroskedastic.yaml",
        "generate_data_func": generate_linear_heteroskedastic_data,
        "description": "Linear Gaussian with heteroskedastic noise",
    },
    "athey_imbens": {
        "config_file": "athey_imbens_nonlinear.yaml",
        "generate_data_func": generate_athey_imbens_data,
        "description": "Non-linear (Athey & Imbens, 2016)",
    },
    "polynomial": {
        "config_file": "polynomial_nonlinear.yaml",
        "generate_data_func": generate_polynomial_data,
        "description": "Non-linear (Quadratic)",
    },
}


def run_treatment_experiment(
    dgps: Optional[List[str]] = None,
    dim_covariates: Optional[List[int]] = None,
    save_results: bool = True,
    create_plots: bool = True,
    quick_mode: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Run treatment effect estimation experiments across different DGPs.

    Args:
        dgps: List of DGPs to run. If None, runs all DGPs
        dim_covariates: List of covariate dimensions to test
        save_results: Whether to save results to disk
        create_plots: Whether to create plots
        quick_mode: If True, reduce repetitions for faster execution

    Returns:
        Dictionary of experiment results DataFrames by DGP
    """
    if dgps is None:
        dgps = list(DGP_CONFIGS.keys())

    if dim_covariates is None:
        dim_covariates = [10, 20, 30, 40, 50]

    # Validate DGP names
    invalid_dgps = [dgp for dgp in dgps if dgp not in DGP_CONFIGS]
    if invalid_dgps:
        raise ValueError(
            f"Invalid DGP names: {invalid_dgps}. Available: {list(DGP_CONFIGS.keys())}"
        )

    configs_dir = Path(__file__).parent.parent / "configs"
    results_dict = {}

    print("=== Treatment Effect Estimation Experiment ===")
    print(f"Running DGPs: {dgps}")
    if quick_mode:
        print("Running in QUICK MODE (fewer repetitions)")

    for dgp_name in dgps:
        dgp_config = DGP_CONFIGS[dgp_name]
        config_path = configs_dir / dgp_config["config_file"]

        if not config_path.exists():
            print(
                f"Warning: Config file {config_path} not found, skipping {dgp_name}..."
            )
            continue

        print(f"\n--- Running {dgp_name}: {dgp_config['description']} ---")

        # Modify repetitions for quick mode
        if quick_mode:
            # Load config, modify repetitions, save temporarily
            config = load_config(str(config_path))
            config["repetitions"] = 10  # Reduce repetitions for quick mode

            # Use a temporary path with modified config
            temp_config_path = config_path.with_name(
                f"temp_{dgp_config['config_file']}"
            )
            import yaml

            with open(temp_config_path, "w") as f:
                yaml.dump(config, f)

            # Run experiment with modified config
            results = run_experiment_from_config(
                config_path=str(temp_config_path),
                generate_data_func=dgp_config["generate_data_func"],
                dim_covariates=dim_covariates,
                save_results=save_results,
            )

            # Clean up temp file
            temp_config_path.unlink()
        else:
            # Run with original config
            results = run_experiment_from_config(
                config_path=str(config_path),
                generate_data_func=dgp_config["generate_data_func"],
                dim_covariates=dim_covariates,
                save_results=save_results,
            )

        results_dict[dgp_name] = results

    if create_plots and results_dict:
        create_treatment_plots(results_dict)

    return results_dict


def create_treatment_plots(results_dict: Dict[str, pd.DataFrame]) -> None:
    """Create and save all treatment experiment plots."""
    plots_dir = Path(__file__).parent.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    configs_dir = Path(__file__).parent.parent / "configs"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot individual DGPs
    for dgp_name, results in results_dict.items():
        dgp_config = DGP_CONFIGS[dgp_name]
        config_path = configs_dir / dgp_config["config_file"]

        if config_path.exists():
            config = load_config(str(config_path))

            savepath = plots_dir / f"treatment_{dgp_name}_{timestamp}.pdf"
            plot_experiment_results(
                expt_results=results, config=config, savepath=str(savepath)
            )
            print(f"Plot saved: {savepath}")

    # Create combined SE ratio plot across all DGPs
    combined_ratio_se = []
    for dgp_name, results in results_dict.items():
        dgp_config = DGP_CONFIGS[dgp_name]
        config_path = configs_dir / dgp_config["config_file"]

        if config_path.exists():
            config = load_config(str(config_path))

            expt_ratio_se = get_ratio_se(results)
            expt_ratio_se = expt_ratio_se.assign(
                expt=dgp_name, expt_name=config["name"]
            )
            combined_ratio_se.append(expt_ratio_se)

    if combined_ratio_se:
        combined_ratio_se = pd.concat(combined_ratio_se, ignore_index=True)

        savepath = plots_dir / f"treatment_se_ratios_{timestamp}.pdf"
        plot_se_ratios(combined_ratio_se, savepath=str(savepath))
        print(f"SE ratios plot saved: {savepath}")


def run_single_dgp(
    dgp_name: str,
    dim_covariates: Optional[List[int]] = None,
    save_results: bool = True,
    create_plot: bool = True,
) -> pd.DataFrame:
    """
    Run experiment for a single DGP.

    Args:
        dgp_name: Name of the DGP to run
        dim_covariates: List of covariate dimensions to test
        save_results: Whether to save results to disk
        create_plot: Whether to create plot

    Returns:
        DataFrame with experiment results
    """
    if dgp_name not in DGP_CONFIGS:
        raise ValueError(
            f"Invalid DGP name: {dgp_name}. Available: {list(DGP_CONFIGS.keys())}"
        )

    results_dict = run_treatment_experiment(
        dgps=[dgp_name],
        dim_covariates=dim_covariates,
        save_results=save_results,
        create_plots=create_plot,
        quick_mode=False,
    )

    return results_dict[dgp_name]


def main():
    """Main function to run treatment experiments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run treatment effect estimation experiments"
    )
    parser.add_argument(
        "--dgps",
        nargs="+",
        choices=list(DGP_CONFIGS.keys()),
        default=list(DGP_CONFIGS.keys()),
        help="DGPs to run",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run in quick mode (fewer repetitions)"
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip creating plots")
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[10, 20, 30, 40, 50],
        help="Covariate dimensions to test",
    )
    parser.add_argument(
        "--single", choices=list(DGP_CONFIGS.keys()), help="Run only a single DGP"
    )

    args = parser.parse_args()

    # If single DGP requested, override dgps
    if args.single:
        args.dgps = [args.single]

    # Print available DGPs
    print("\nAvailable DGPs:")
    for dgp_name, dgp_info in DGP_CONFIGS.items():
        selected = "✓" if dgp_name in args.dgps else " "
        print(f"  [{selected}] {dgp_name}: {dgp_info['description']}")

    # Run experiments
    results = run_treatment_experiment(
        dgps=args.dgps,
        dim_covariates=args.dims,
        save_results=True,
        create_plots=not args.no_plots,
        quick_mode=args.quick,
    )

    print("\n=== Treatment Experiments completed! ===")
    print("Results saved to experiments/results/")
    if not args.no_plots:
        print("Plots saved to experiments/plots/")

    # Print summary
    print(f"\nRan {len(results)} DGPs:")
    for dgp_name in results.keys():
        print(f"  - {dgp_name}: {DGP_CONFIGS[dgp_name]['description']}")

    return results


if __name__ == "__main__":
    main()
