# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Union

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from .types import Dataset, PropensityModel


def sigmoid(z: Union[float, NDArray]) -> Union[float, NDArray]:
    """
    Compute the sigmoid function.

    Parameters
    ----------
    z : float or NDArray
        The input value or array of values for which to compute the sigmoid.

    Returns
    -------
    float or NDArray
        The sigmoid of the input value(s), which is a value between 0 and 1.
    """

    return 1 / (1 + np.exp(-z))


class ConstantPropensityModel(PropensityModel):
    """A propensity model that always predicts a (known) constant propensity score."""

    def __init__(self, propensity_score: float = 0.5) -> None:
        """
        Initialize a constant propensity model.

        Parameters
        ----------
        propensity_score : float, optional
            Constant propensity score to predict, by default 0.5
        """
        assert (
            0.0 < propensity_score < 1.0
        ), f"Propensity score must be between 0 and 1: got {propensity_score}"
        self._propensity_score = propensity_score

    def fit(self, data: Dataset) -> None:
        """
        Fit the propensity model to training data.

        Parameters
        ----------
        data : Dataset
            Training data containing features (X) and treatment assignments (A)
        """
        pass

    def predict(self, data: Dataset) -> NDArray:
        """
        Predict propensity scores for given features.

        Parameters
        ----------
        data : Dataset
            Data containing features (X) to predict propensity scores for

        Returns
        -------
        NDArray
            Predicted propensity scores
        """
        return np.full(data.X.shape[0], self._propensity_score)


class AveragePropensityModel(PropensityModel):
    """A propensity model that always predicts the average propensity score."""

    def __init__(self) -> None:
        """
        Initialize a constant propensity model.
        """
        self._propensity_score: float | None = None

    def fit(self, data: Dataset) -> None:
        """
        Fit the propensity model to training data.

        Parameters
        ----------
        data : Dataset
            Training data containing features (X) and treatment assignments (A)
        """
        propensity_score = float(np.mean(data.A))
        assert (
            0.0 < propensity_score < 1.0
        ), f"Propensity score must be between 0 and 1: got {propensity_score}"
        self._propensity_score = propensity_score

    def predict(self, data: Dataset) -> NDArray:
        """
        Predict propensity scores for given features.

        Parameters
        ----------
        data : Dataset
            Data containing features (X) to predict propensity scores for

        Returns
        -------
        NDArray
            Predicted propensity scores
        """
        return np.full(data.X.shape[0], self._propensity_score)


class LogisticPropensityModel(PropensityModel):
    """A propensity model that uses logistic regression."""

    def __init__(
        self,
        maxiter: int = 1000,
        fit_method: str = "bfgs",
        clip_for_stability: float = 1e-3,
    ) -> None:
        """
        Initialize a logistic propensity model.

        Parameters
        ----------
        maxiter : int, optional
            Maximum number of iterations for logistic regression, by default 1000
        fit_method : str, optional
            Optimization method for logistic regression, by default 'bfgs'
        clip_for_stability : float, optional
            Constant for clipping propensity scores to avoid extreme values, by default 1e-3
        """
        self.maxiter = maxiter
        self.fit_method = fit_method
        self.clip_for_stability = clip_for_stability
        self._fitted_model: Any = None

    def fit(self, data: Dataset) -> None:
        """
        Fit the propensity model to training data using logistic regression.

        Parameters
        ----------
        data : Dataset
            Training data containing features (X) and treatment assignments (A)
        """
        logistic_reg = sm.Logit(data.A, data.X).fit(
            maxiter=self.maxiter, method=self.fit_method, disp=False
        )
        self._fitted_model = logistic_reg

    def predict(self, data: Dataset) -> NDArray:
        """
        Predict propensity scores for given features.

        Parameters
        ----------
        data : Dataset
            Data containing features (X) to predict propensity scores for

        Returns
        -------
        NDArray
            Predicted propensity scores
        """
        if self._fitted_model is None:
            raise ValueError("Model must be fitted before prediction")

        predictions = self._fitted_model.predict(data.X)
        return np.clip(
            np.asarray(predictions),
            self.clip_for_stability,
            1 - self.clip_for_stability,
        )


class LassoLogisticPropensityModel(PropensityModel):
    """A propensity model that uses logistic regression with L1 regularization (Lasso)."""

    def __init__(
        self,
        alpha_grid: NDArray | None = None,
        cv_folds: int = 2,
        clip_for_stability: float = 1e-3,
        penalty: str = "l1",
        solver: str = "liblinear",
    ) -> None:
        """
        Initialize a Lasso logistic propensity model.

        Parameters
        ----------
        alpha_grid : NDArray, optional
            Grid of regularization parameters (C values) for cross-validation, by default np.linspace(0.01, 1, 25)
        cv_folds : int, optional
            Number of cross-validation folds, by default 2
        clip_for_stability : float, optional
            Constant for clipping propensity scores to avoid extreme values, by default 1e-3
        penalty : str, optional
            The penalty term for LogisticRegression, by default "l1"
        solver : str, optional
            The solver for LogisticRegression, by default "liblinear"
        """
        if alpha_grid is None:
            alpha_grid = np.linspace(0.01, 1, 25)
        self.alpha_grid = alpha_grid
        self.cv_folds = cv_folds
        self.clip_for_stability = clip_for_stability
        self.penalty = penalty
        self.solver = solver
        self._fitted_model: Any = None

    def fit(self, data: Dataset) -> None:
        """
        Fit the propensity model to training data using Lasso logistic regression.

        Parameters
        ----------
        data : Dataset
            Training data containing features (X) and treatment assignments (A)
        """
        # Cross-validation to find best regularization parameter
        if self.alpha_grid is None:
            raise ValueError("Alpha grid must be provided")

        neg_mse_df = pd.Series(
            [
                cross_val_score(
                    LogisticRegression(
                        penalty=self.penalty, C=float(x), solver=self.solver
                    ),
                    data.X,
                    data.A,
                    scoring="accuracy",
                    cv=self.cv_folds,
                ).mean()
                for x in self.alpha_grid
            ],
            index=self.alpha_grid,
        )
        cv_reg_param = neg_mse_df.index[neg_mse_df.argmax()]

        # Fit model with best regularization parameter
        logistic_reg = LogisticRegression(
            penalty=self.penalty, C=cv_reg_param, solver=self.solver
        )
        logistic_reg.fit(data.X, data.A)
        self._fitted_model = logistic_reg

    def predict(self, data: Dataset) -> NDArray:
        """
        Predict propensity scores for given features.

        Parameters
        ----------
        data : Dataset
            Data containing features (X) to predict propensity scores for

        Returns
        -------
        NDArray
            Predicted propensity scores
        """
        if self._fitted_model is None:
            raise ValueError("Model must be fitted before prediction")

        # Predict probabilities
        propensity_scores = self._fitted_model.predict_proba(data.X)[:, 1]

        # Clip for numerical stability
        return np.clip(
            propensity_scores,
            self.clip_for_stability,
            1 - self.clip_for_stability,
        )


class SigmoidPropensityModel(PropensityModel):
    """A propensity model that uses sigmoid function with predefined weights."""

    def __init__(
        self,
        sigmoid_weight: NDArray,
        clip_for_stability: float = 1e-3,
    ) -> None:
        """
        Initialize a sigmoid propensity model.

        Parameters
        ----------
        sigmoid_weight : NDArray
            Weights for the sigmoid function
        clip_for_stability : float, optional
            Constant for clipping propensity scores to avoid extreme values, by default 1e-3
        """
        self.sigmoid_weight = sigmoid_weight
        self.clip_for_stability = clip_for_stability

    def fit(self, data: Dataset) -> None:
        """
        Fit the propensity model to training data.

        Parameters
        ----------
        data : Dataset
            Training data containing features (X) and treatment assignments (A)

        Note
        ----
        This method is empty as the sigmoid weights are predefined.
        """
        pass

    def predict(self, data: Dataset) -> NDArray:
        """
        Predict propensity scores for given features using sigmoid function.

        Parameters
        ----------
        data : Dataset
            Data containing features (X) to predict propensity scores for

        Returns
        -------
        NDArray
            Predicted propensity scores
        """
        propensity_scores = sigmoid(data.X @ self.sigmoid_weight)
        propensity_scores = np.clip(
            propensity_scores, self.clip_for_stability, 1 - self.clip_for_stability
        )
        return propensity_scores


class DGPSigmoidGaussianPropensityModel(PropensityModel):
    """Data generating process for sigmoid Gaussian propensity model with sparse coefficients."""

    def __init__(
        self,
        sparsity_factor: float = 0.0,
        std_coefficients: float = 1.0,
        clip_for_stability: float = 1e-3,
    ) -> None:
        """
        Initialize a DGPSigmoidGaussianPropensityModel instance.

        Parameters
        ----------
        sparsity_factor : float, optional
            Fraction of coefficients to set to zero (0.0 <= sparsity_factor < 1.0), by default 0.0
        std_coefficients : float, optional
            Standard deviation for coefficient generation, by default 1.0
        clip_for_stability : float, optional
            Constant for clipping propensity scores to avoid extreme values, by default 1e-3
        """
        assert (
            0.0 <= sparsity_factor < 1.0
        ), f"sparsity_factor must be between 0.0 (inclusive) and 1.0 (exclusive), got {sparsity_factor}"
        self.sparsity_factor = sparsity_factor
        self.std_coefficients = std_coefficients
        self.clip_for_stability = clip_for_stability
        self._coefficients: NDArray | None = None

    def fit(self, data: Dataset) -> None:
        """
        Fit the propensity model by generating random coefficients.

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
            if num_zeros > 0:
                # Set the last num_zeros coefficients to zero
                coefficients[-num_zeros:] = 0.0

        self._coefficients = coefficients

    def predict(self, data: Dataset) -> NDArray:
        """
        Predict propensity scores for given features using sigmoid function.

        Parameters
        ----------
        data : Dataset
            Data containing features (X) to predict propensity scores for

        Returns
        -------
        NDArray
            Predicted propensity scores
        """
        if self._coefficients is None:
            raise ValueError("Model must be fitted before prediction")

        # Use sigmoid function with linear combination of features
        propensity_scores = sigmoid(data.X @ self._coefficients)
        propensity_scores = np.clip(
            propensity_scores, self.clip_for_stability, 1 - self.clip_for_stability
        )
        return propensity_scores
