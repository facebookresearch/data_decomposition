# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
from numpy.typing import NDArray
from scipy.stats import betaprime

from .types import (
    DataGenerator,
    Dataset,
    OutcomeModel,
    OutcomeNoiseModel,
    PropensityModel,
)


class HomoGaussian(OutcomeNoiseModel):
    """
    Homoscedastic Gaussian noise generator.

    Generates i.i.d. Gaussian noise with constant variance (homoscedastic).
    """

    def __init__(self, std_noise: float) -> None:
        """
        Initialize the homoscedastic Gaussian noise generator.

        Parameters
        ----------
        std_noise : float
            Standard deviation of the Gaussian noise
        """
        self.std_noise = std_noise

    def sample(self, size: int, seed: int | None = None) -> NDArray:
        """
        Sample from the homoscedastic Gaussian noise distribution.

        Parameters
        ----------
        size : int
            Number of samples to draw
        seed : int | None
            Random seed for reproducible sampling

        Returns
        -------
        NDArray
            Sampled noise values from N(0, std_noise^2)
        """
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(0, self.std_noise, size)


class HeteroGaussian(OutcomeNoiseModel):
    """
    Heteroscedastic Gaussian noise generator.

    Generates non-i.i.d. noise using Beta-prime distribution to create
    heteroscedasticity (varying variance).
    """

    def __init__(
        self, std_noise: float, betaprime_a: float, betaprime_b: float
    ) -> None:
        """
        Initialize the heteroscedastic Gaussian noise generator.

        Parameters
        ----------
        std_noise : float
            Base standard deviation for the noise
        betaprime_a : float
            Alpha parameter for the Beta-prime distribution
        betaprime_b : float
            Beta parameter for the Beta-prime distribution
        """
        self.std_noise = std_noise
        self.betaprime_a = betaprime_a
        self.betaprime_b = betaprime_b

    def sample(self, size: int, seed: int | None = None) -> NDArray:
        """
        Sample from the heteroscedastic noise distribution.

        Uses Beta-prime distribution to create varying variance structure.

        Parameters
        ----------
        size : int
            Number of samples to draw
        seed : int | None
            Random seed for reproducible sampling

        Returns
        -------
        NDArray
            Sampled noise values with heteroscedastic structure
        """
        if seed is not None:
            np.random.seed(seed)
        # Cast to float array to ensure proper type handling
        betaprime_sample = betaprime.rvs(
            a=self.betaprime_a, b=self.betaprime_b, size=size
        )
        return np.array(betaprime_sample) * self.std_noise


class BinomialGaussian(DataGenerator):
    """
    Data generator using binomial treatment assignment, Gaussian features and configurable noise.

    Generates data with:
    - Features from multivariate Gaussian distribution
    - Treatment assignments from binomial distribution using propensity scores
    - Outcomes from linear model with treatment effect and noise
    """

    def __init__(
        self,
        outcome_model: OutcomeModel,
        propensity_model: PropensityModel,
        outcome_noise_model: OutcomeNoiseModel,
        dim_covariates: int,
        std_covariates: float,
    ) -> None:
        """
        Initialize the BinomialGaussian data generator.

        Parameters
        ----------
        outcome_model : OutcomeModel
            Outcome model to use for generating baseline outcomes
        propensity_model : PropensityModel
            Propensity model to use for generating treatment assignments
        outcome_noise_model : OutcomeNoiseModel
            Outcome noise model to use for generating noise
        dim_covariates : int
            Dimension of covariates (d in the logic)
        std_covariates : float
            Standard deviation for covariates (cov_sd in the logic)
        """
        self.outcome_model = outcome_model
        self.propensity_model = propensity_model
        self.outcome_noise_model = outcome_noise_model
        self.dim_covariates = dim_covariates
        self.std_covariates = std_covariates

    def generate(
        self,
        num_samples: int,
        treatment_effect: float,
        seed: int | None = None,
    ) -> Dataset:
        """
        Generate synthetic data with binomial treatment assignment, Gaussian features and configurable noise.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate (n in the logic)
        treatment_effect : float
            True treatment effect to embed in the data (true_ate in the logic)
        seed : int | None
            Random seed for reproducible data generation

        Returns
        -------
        Dataset
            Generated dataset containing features (X), treatment assignments (A), and outcomes (Y)
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate features: X = np.random.multivariate_normal(np.zeros(d), np.eye(d), size=n) * cov_sd
        X = (
            np.random.multivariate_normal(
                np.zeros(self.dim_covariates),
                np.eye(self.dim_covariates),
                size=num_samples,
            )
            * self.std_covariates
        )

        # Create temporary dataset to get propensity scores
        temp_data = Dataset(
            X=X,
            A=np.zeros(num_samples),  # Placeholder, not used for propensity prediction
            Y=np.zeros(num_samples),  # Placeholder, not used for propensity prediction
        )
        self.propensity_model.fit(temp_data)
        propensity_scores = self.propensity_model.predict(temp_data)

        # Generate treatment assignments: A = np.random.binomial(1, propensity_scores)
        A = np.random.binomial(1, propensity_scores)

        # Generate baseline outcomes using outcome model for control group
        control_data = Dataset(X=X, A=np.zeros(num_samples), Y=np.zeros(num_samples))
        self.outcome_model.fit(control_data)
        baseline_outcomes = self.outcome_model.predict(control_data)

        # Generate noise based on outcome noise model
        noise = self.outcome_noise_model.sample(num_samples, seed)

        # Generate outcomes: Y = baseline + A * treatment_effect + noise
        Y = baseline_outcomes + A * treatment_effect + noise

        return Dataset(X=X, A=A, Y=Y)
