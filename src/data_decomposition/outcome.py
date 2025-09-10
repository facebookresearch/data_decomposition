# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

from .types import Dataset, OutcomeModel


class OLSOutcomeModel(OutcomeModel):
    """Simple OLS outcome model without regularization."""

    def __init__(self) -> None:
        """Initialize an OLSOutcomeModel instance."""
        self._fitted_model: Any = None

    def fit(self, data: Dataset) -> None:
        """
        Fit the OLS outcome model.

        Parameters
        ----------
        data : Dataset
            Training data containing features (X), treatment assignments (A), and outcomes (Y)
        """
        # Concatenate treatment and features
        AX = np.concatenate((data.A.reshape(-1, 1), data.X), axis=1)

        # Fit OLS model
        outcome_model = sm.OLS(data.Y, AX)
        self._fitted_model = outcome_model.fit()

    def predict(self, data: Dataset) -> NDArray:
        """
        Predict outcomes using the fitted model.

        Parameters
        ----------
        data : Dataset
            Data containing features (X) and treatment assignments (A) for prediction

        Returns
        -------
        NDArray
            Predicted outcomes
        """
        if self._fitted_model is None:
            raise ValueError("Model must be fitted before prediction")

        # Concatenate treatment and features
        AX = np.concatenate((data.A.reshape(-1, 1), data.X), axis=1)
        predictions = self._fitted_model.predict(AX)
        # Ensure return type is NDArray
        return np.asarray(predictions)


class LassoOutcomeModel(OutcomeModel):
    """Lasso outcome model with cross-validation for alpha selection."""

    def __init__(self, alpha_grid: NDArray | None = None, cv_folds: int = 2) -> None:
        """
        Initialize a LassoOutcomeModel instance.

        Parameters
        ----------
        alpha_grid : NDArray, optional
            Grid of alpha values for cross-validation, by default np.linspace(0.01, 1, 25)
        cv_folds : int, optional
            Number of cross-validation folds, by default 2
        """
        if alpha_grid is None:
            alpha_grid = np.linspace(0.01, 1, 25)
        self._alpha_grid = alpha_grid
        self.cv_folds = cv_folds
        self._fitted_model: Any = None

    def fit(self, data: Dataset) -> None:
        """
        Fit the Lasso outcome model using cross-validation for alpha selection.

        Parameters
        ----------
        data : Dataset
            Training data containing features (X), treatment assignments (A), and outcomes (Y)
        """
        # Concatenate treatment and features
        AX = np.concatenate((data.A.reshape(-1, 1), data.X), axis=1)

        # Cross-validation to select best alpha
        if self._alpha_grid is None:
            raise ValueError("Alpha grid must be provided")

        neg_mse_scores = pd.Series(
            [
                cross_val_score(
                    Lasso(alpha=alpha),
                    AX,
                    data.Y,
                    scoring="neg_mean_squared_error",
                    cv=self.cv_folds,
                ).mean()
                for alpha in self._alpha_grid
            ],
            index=self._alpha_grid,
        )

        best_alpha = neg_mse_scores.index[neg_mse_scores.argmax()]

        # Fit model with best alpha using statsmodels
        outcome_model = sm.OLS(data.Y, AX)
        self._fitted_model = outcome_model.fit_regularized(alpha=best_alpha, L1_wt=1.0)

    def predict(self, data: Dataset) -> NDArray:
        """
        Predict outcomes using the fitted model.

        Parameters
        ----------
        data : Dataset
            Data containing features (X) and treatment assignments (A) for prediction

        Returns
        -------
        NDArray
            Predicted outcomes
        """
        if self._fitted_model is None:
            raise ValueError("Model must be fitted before prediction")

        # Concatenate treatment and features
        AX = np.concatenate((data.A.reshape(-1, 1), data.X), axis=1)
        return self._fitted_model.predict(AX)


class DGPLinearGaussianOutcomeModel(OutcomeModel):
    """Data generating process for linear Gaussian outcome model with sparse coefficients."""

    def __init__(
        self, sparsity_factor: float = 0.0, std_coefficients: float = 1.0
    ) -> None:
        """
        Initialize a DGPLinearGaussianOutcomeModel instance.

        Parameters
        ----------
        sparsity_factor : float, optional
            Fraction of coefficients to set to zero (0.0 <= sparsity_factor < 1.0), by default 0.0
        std_coefficients : float, optional
            Standard deviation for coefficient generation, by default 1.0
        """
        assert (
            0.0 <= sparsity_factor < 1.0
        ), f"sparsity_factor must be between 0.0 (inclusive) and 1.0 (exclusive), got {sparsity_factor}"
        self.sparsity_factor = sparsity_factor
        self.std_coefficients = std_coefficients
        self._coefficients: NDArray | None = None

    def fit(self, data: Dataset) -> None:
        """
        Fit the outcome model by generating random coefficients.

        Parameters
        ----------
        data : Dataset
            Training data containing features (X) - used to determine dimensionality
        """
        dim_covariates = data.X.shape[1]

        # Generate random coefficients
        coefficients = np.random.randn(dim_covariates) * self.std_coefficients

        # Set the latter fraction sparsity_factor of coefficients to zeros
        if self.sparsity_factor > 0.0:
            num_zeros = int(np.ceil(dim_covariates * self.sparsity_factor))
            assert num_zeros < dim_covariates, "Cannot set all coefficients to zero"
            if num_zeros > 0:
                # Set the last num_zeros coefficients to zero
                coefficients[-num_zeros:] = 0.0

        self._coefficients = coefficients

    def predict(self, data: Dataset) -> NDArray:
        """
        Predict outcomes using the generated coefficients.

        Parameters
        ----------
        data : Dataset
            Data containing features (X) and treatment assignments (A) for prediction

        Returns
        -------
        NDArray
            Predicted outcomes (linear combination of features with coefficients)
        """
        if self._coefficients is None:
            raise ValueError("Model must be fitted before prediction")

        # Return linear combination of features with coefficients
        return data.X @ self._coefficients
