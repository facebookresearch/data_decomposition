# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for experiment configuration and shared functionality.
"""

import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

# Import from our local package
from data_decomposition.aipw import AIPW
from data_decomposition.data_decomposer import Splitting, ThinningFission
from data_decomposition.data_generator import (
    BinomialGaussian,
    HeteroGaussian,
    HomoGaussian,
)
from data_decomposition.outcome import DGPLinearGaussianOutcomeModel, OLSOutcomeModel
from data_decomposition.propensity import (
    DGPSigmoidGaussianPropensityModel,
    LogisticPropensityModel,
)
from data_decomposition.types import Dataset
from tqdm import tqdm

from matplotlib.patches import Patch

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_single_experiment(config: Dict[str, Any], generate_data_func) -> pd.DataFrame:
    """
    Run a single experiment comparing data splitting vs thinning fission.

    Args:
        config: Experiment configuration dictionary
        generate_data_func: Function to generate synthetic data

    Returns:
        DataFrame with results for both methods
    """

    def single_split(full_data: Dataset) -> Dict[str, Any]:
        """Run AIPW with traditional data splitting."""
        splitter = Splitting(
            train_ratio=0.5, shuffle=False, stratify_by_treatment=False
        )
        training_inference_data = splitter.decompose(full_data)

        estimator = AIPW()
        propensity_model = LogisticPropensityModel(
            maxiter=1000, fit_method="bfgs", clip_for_stability=1e-3
        )
        outcome_model = OLSOutcomeModel()

        estimator.fit(training_inference_data, propensity_model, outcome_model)

        ate_result = estimator.ate()
        return {
            "method": "single_split",
            "ate": ate_result.ate,
            "ate_se": ate_result.ate_se,
        }

    def thinning_fission(full_data: Dataset, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run AIPW with thinning fission (data decomposition)."""
        thinner = ThinningFission(
            std_outcome_noise=config["std_thinning_noise"],
            train_information_parameter=config["thinning_train_information_parameter"],
            decompose_treatment=False,
            treatment_noise=None,
        )
        training_inference_data = thinner.decompose(full_data)

        estimator = AIPW()
        propensity_model = LogisticPropensityModel()
        outcome_model = OLSOutcomeModel()

        estimator.fit(training_inference_data, propensity_model, outcome_model)

        ate_result = estimator.ate()
        return {
            "method": "thinning",
            "ate": ate_result.ate,
            "ate_se": ate_result.ate_se,
        }

    # Generate data and run both methods
    full_data = generate_data_func(config)
    results = pd.DataFrame(
        [
            single_split(full_data),
            thinning_fission(full_data, config),
        ]
    )
    return results


def get_ratio_se(expt_results: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ratio of standard errors between thinning and single split methods.

    Args:
        expt_results: DataFrame with experiment results

    Returns:
        DataFrame with SE ratios by dimension
    """
    ate_se_mean = (
        expt_results[["method", "dim_covariates", "ate_se"]]
        .groupby(["method", "dim_covariates"])
        .mean()
        .reset_index()
    )

    single_split_se = ate_se_mean[ate_se_mean.method == "single_split"].set_index(
        "dim_covariates"
    )["ate_se"]
    thinning_se = ate_se_mean[ate_se_mean.method == "thinning"].set_index(
        "dim_covariates"
    )["ate_se"]

    ratio_se = (thinning_se / single_split_se).reset_index()
    ratio_se.columns = ["dim_covariates", "ate_se_ratio"]
    return ratio_se


def run_experiment_from_config(
    config_path: str,
    generate_data_func,
    dim_covariates: List[int] = None,
    save_results: bool = True,
    results_dir: str = None,
) -> pd.DataFrame:
    """
    Run a complete experiment from a configuration file.

    Args:
        config_path: Path to YAML configuration file
        generate_data_func: Function to generate data for this experiment
        dim_covariates: List of covariate dimensions to test
        save_results: Whether to save results to disk
        results_dir: Directory to save results

    Returns:
        DataFrame with experiment results
    """
    config = load_config(config_path)

    if dim_covariates is None:
        dim_covariates = [20, 40, 60, 80, 100]#[10, 20, 30, 40, 50]

    if results_dir is None:
        results_dir = Path(config_path).parent.parent / "results"
    else:
        results_dir = Path(results_dir)

    results_dir.mkdir(exist_ok=True)

    print(f"\n=== Running {config['name']} ===")
    print(f"Description: {config['description']}")

    results_list = []

    for dim in dim_covariates:
        config["dim_covariates"] = dim

        for run_number in tqdm(
            range(config["repetitions"]),
            desc=f"{config['name']}, dim_covariates={dim}",
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = run_single_experiment(config, generate_data_func)

            results["run_number"] = run_number + 1
            results["dim_covariates"] = dim
            results_list.append(results)

    expt_results = pd.concat(results_list, ignore_index=True)

    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = Path(config_path).stem
        filepath = results_dir / f"{config_name}_results_{timestamp}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(expt_results, f)
        print(f"Saved results to {filepath}")

    return expt_results


def plot_experiment_results(
    expt_results: pd.DataFrame,
    config: Dict[str, Any],
    savepath: str = None,
) -> None:
    """Plot boxplot comparison of experiment results."""
    plot_settings = config.get("plot_settings", {})
    fontsize = plot_settings.get("fontsize", 15)
    dpi = plot_settings.get("dpi", 300)
    max_covariates = plot_settings.get("max_covariates_for_plot", 50) # 40

    # Filter results for plotting
    plot_results = expt_results.copy()
    plot_results = plot_results.loc[
        plot_results["dim_covariates"] <= max_covariates
    ].reset_index(drop=True)
    plot_results["ate_norm"] = plot_results["ate"] / config["true_ate"]

    plt.figure(figsize=(12, 8))

    sns.set_style("whitegrid")
    box = sns.boxplot(
        data=plot_results,
        x="dim_covariates",
        y="ate_norm",
        hue="method",
        palette="pastel",
    )

    # Add hatching patterns
    hatches = ["//", "\\\\", "||"]
    num_categories = len(plot_results["method"].unique())


    handles = []
    for i, patch in enumerate(box.patches):   
        hatch = hatches[(i // 2) // num_categories]
        #hatch = hatches[i // (len(box.patches) // num_categories)]
        patch.set_hatch(hatch)
          
            
    handles = []
    methods = plot_results["method"].unique()
    palette = sns.color_palette("pastel", len(methods))
    
    for color, hatch, method in zip(palette, hatches, methods):
        patch = Patch(facecolor=color, edgecolor="black", hatch=hatch, label=method)
        handles.append(patch)


    # Style the plot
    for spine in plt.gca().spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2)

    plt.axhline(1, color="green", linewidth=0.8, linestyle="--")
    plt.axhline(0, color="red", linewidth=0.8, linestyle="--")
    plt.xlabel("#Covariates", fontsize=fontsize)
    plt.ylabel("AIPW Estimated ATE / True ATE", fontsize=fontsize)
    plt.title(config["name"], fontsize=fontsize + 2)
    

        
    #handles, _ = box.get_legend_handles_labels()

    plt.legend(
        handles=handles,
        title="Method",
        labels=["Data Splitting", "Data Decomposition"],
        fontsize=fontsize,
        title_fontsize=fontsize,
    )
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_se_ratios(
    combined_ratio_se: pd.DataFrame,
    savepath: str = None,
    fontsize: int = 20,
    dpi: int = 600,
) -> None:
    """Plot standard error ratios across experiments."""
    plt.figure(figsize=(12, 8))

    markers = ["o", "D", "^", "v"]
    sns.lineplot(
        data=combined_ratio_se,
        x="dim_covariates",
        y="ate_se_ratio",
        hue="expt_name",
        style="expt_name",
        markers=markers,
        markersize=10,
        dashes=False,
        palette="tab10",
    )

    plt.xlabel("#Covariates", fontsize=fontsize)
    plt.ylabel(
        "Ratio: Data Decomposition SE / Data Splitting SE", fontsize=fontsize * 0.8
    )
    plt.title("Standard Error Comparison Across Experiments", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    for spine in plt.gca().spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2)

    plt.legend(
        title="Data Generation Setup",
        fontsize=fontsize * 0.8,
        title_fontsize=fontsize * 0.8,
    )

    if savepath:
        plt.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close()


# Data generation functions for different experiments


def generate_linear_homoskedastic_data(config: Dict[str, Any]) -> Dataset:
    """Generate data for Linear Gaussian (Homoskedastic) experiment."""
    outcome_dgp_model = DGPLinearGaussianOutcomeModel(
        sparsity_factor=config["outcome_model"]["sparsity_factor"],
        std_coefficients=config["outcome_model"]["std_coefficients"],
    )
    propensity_dgp_model = DGPSigmoidGaussianPropensityModel(
        sparsity_factor=config["propensity_model"]["sparsity_factor"],
        std_coefficients=config["propensity_model"]["std_coefficients"],
        clip_for_stability=config["propensity_model"]["clip_for_stability"],
    )
    noise_model = HomoGaussian(std_noise=config["std_dgp_noise"])

    generator = BinomialGaussian(
        outcome_model=outcome_dgp_model,
        propensity_model=propensity_dgp_model,
        outcome_noise_model=noise_model,
        dim_covariates=config["dim_covariates"],
        std_covariates=config["std_covariates"],
    )

    full_data = generator.generate(
        num_samples=config["num_samples"],
        treatment_effect=config["true_ate"],
        seed=None,
    )
    return full_data


def generate_linear_heteroskedastic_data(config: Dict[str, Any]) -> Dataset:
    """Generate data for Linear Gaussian (Heteroskedastic) experiment."""
    outcome_dgp_model = DGPLinearGaussianOutcomeModel(
        sparsity_factor=config["outcome_model"]["sparsity_factor"],
        std_coefficients=config["outcome_model"]["std_coefficients"],
    )
    propensity_dgp_model = DGPSigmoidGaussianPropensityModel(
        sparsity_factor=config["propensity_model"]["sparsity_factor"],
        std_coefficients=config["propensity_model"]["std_coefficients"],
        clip_for_stability=config["propensity_model"]["clip_for_stability"],
    )
    noise_model = HeteroGaussian(
        std_noise=config["std_dgp_noise"],
        betaprime_a=config["noise_model"]["betaprime_a"],
        betaprime_b=config["noise_model"]["betaprime_b"],
    )

    generator = BinomialGaussian(
        outcome_model=outcome_dgp_model,
        propensity_model=propensity_dgp_model,
        outcome_noise_model=noise_model,
        dim_covariates=config["dim_covariates"],
        std_covariates=config["std_covariates"],
    )

    full_data = generator.generate(
        num_samples=config["num_samples"],
        treatment_effect=config["true_ate"],
        seed=None,
    )
    return full_data


def generate_athey_imbens_data(config: Dict[str, Any]) -> Dataset:
    """Generate data for Athey & Imbens (2016) non-linear experiment."""

    def eta(x: List[float], dgp: int = 1) -> float:
        """Mean effect function for Athey & Imbens DGP."""
        if dgp == 1:  # x needs dimension 2
            return x[0] / 2 + x[1]
        elif dgp == 2:  # x needs dimension 6
            return (x[0] + x[1]) / 2 + (x[2] + x[3] + x[4])
        else:
            return 0.0

    def kappa(x: List[float], dgp: int = 1, ate: float = 0.5) -> float:
        """Treatment effect function for Athey & Imbens DGP."""
        if dgp == 1:
            return (1 + x[0]) * ate
        elif dgp == 2:
            return (1 + x[0]) * ate * (x[0] > 0) + x[1] * (x[1] > 0)
        else:
            return 0.0

    def model_outcome(
        eta,
        kappa,
        x: List[float],
        w: float,
        ate: float = 0.5,
        var: float = 0.01,
        dgp: int = 2,
    ) -> float:
        """Model the response for the Athey & Imbens DGP."""
        epsilon = np.random.normal(0, var**0.5)
        y = eta(x, dgp) + (2 * w - 1) * (kappa(x, dgp, ate)) / 2 + epsilon
        return float(y)

    dim = config["dim_covariates"]
    num_samples = config["num_samples"]
    treatment_effect = config["true_ate"]
    std_noise = config["std_dgp_noise"] ** 2
    cov_noise = config["std_covariates"]
    dgp_type = config.get("dgp_type", 2)

    X = (
        np.random.multivariate_normal(
            np.zeros(dim),
            np.eye(dim),
            size=num_samples,
        )
        * cov_noise
    )
    A = np.random.binomial(1, 0.5, num_samples)
    Y = [
        model_outcome(
            eta=eta,
            kappa=kappa,
            x=X[i],
            w=A[i],
            ate=treatment_effect,
            var=std_noise,
            dgp=dgp_type,
        )
        for i in range(num_samples)
    ]

    return Dataset(X=X, A=A, Y=np.array(Y))


def generate_polynomial_data(config: Dict[str, Any]) -> Dataset:
    """Generate data for polynomial non-linear experiment."""
    dim = config["dim_covariates"]
    num_samples = config["num_samples"]
    treatment_effect = config["true_ate"]
    std_noise = config["std_dgp_noise"]
    cov_noise = config["std_covariates"]

    X = (
        np.random.multivariate_normal(
            np.zeros(dim),
            np.eye(dim),
            size=num_samples,
        )
        * cov_noise
    )
    A = np.random.binomial(1, 0.5, num_samples)
    beta = np.random.normal(0, 1, dim)
    Y = (
        (X @ beta) ** 2
        + A * treatment_effect
        + std_noise * np.random.normal(0, 1, num_samples)
    )
    return Dataset(X=X, A=A, Y=Y)
