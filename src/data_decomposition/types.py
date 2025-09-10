# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import ABC, abstractmethod
from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class Dataset:
    """Data container for a dataset."""

    X: NDArray
    A: NDArray
    Y: NDArray


@dataclass
class TrainingInferenceData:
    """Data container for estimation with training and inference splits."""

    train: Dataset
    inf: Dataset


@dataclass
class AverageTreatmentEffect:
    """Container for average treatment effect estimates."""

    ate: float
    ate_se: float


class PropensityModel(ABC):
    """Abstract base class for propensity score models."""

    @abstractmethod
    def fit(self, data: Dataset) -> None:
        """
        Fit the propensity model to training data.

        Parameters
        ----------
        data : Dataset
            Training data containing features (X) and treatment assignments (A)
        """
        pass

    @abstractmethod
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
        pass


class OutcomeModel(ABC):
    """Abstract base class for outcome models."""

    @abstractmethod
    def fit(self, data: Dataset) -> None:
        """
        Fit the outcome model to training data.

        Parameters
        ----------
        data : Dataset
            Training data containing features (X), treatment assignments (A), and outcomes (Y)
        """
        pass

    @abstractmethod
    def predict(self, data: Dataset) -> NDArray:
        """
        Predict outcomes for given treatment-feature combinations.

        Parameters
        ----------
        data : Dataset
            Data containing features (X) and treatment assignments (A) for prediction

        Returns
        -------
        NDArray
            Predicted outcomes
        """
        pass


class DataGenerator(ABC):
    """Abstract base class for data generators."""

    @abstractmethod
    def generate(
        self,
        num_samples: int,
        treatment_effect: float,
        seed: int | None = None,
    ) -> Dataset:
        """
        Generate synthetic data for causal inference experiments.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate
        treatment_effect : float
            True treatment effect to embed in the data
        seed : int | None
            Random seed for reproducible data generation

        Returns
        -------
        Dataset
            Generated dataset containing features (X), treatment assignments (A), and outcomes (Y)
        """
        pass


class OutcomeNoiseModel(ABC):
    """Abstract base class for outcome noise models."""

    @abstractmethod
    def sample(self, size: int, seed: int | None = None) -> NDArray:
        """
        Sample from the noise distribution.

        Parameters
        ----------
        size : int
            Number of samples to draw
        seed : int | None
            Random seed for reproducible sampling

        Returns
        -------
        NDArray
            Sampled noise values
        """
        pass


class DataDecomposer(ABC):
    """Abstract base class for data decomposers that split data into training and inference sets."""

    @abstractmethod
    def decompose(self, data: Dataset) -> TrainingInferenceData:
        """
        Decompose a dataset into training and inference splits.

        Parameters
        ----------
        data : Dataset
            Complete dataset to decompose

        Returns
        -------
        TrainingInferenceData
            Data split into training and inference sets
        """
        pass
