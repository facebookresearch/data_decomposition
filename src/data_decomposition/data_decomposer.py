# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np

from .types import DataDecomposer, Dataset, TrainingInferenceData


class Splitting(DataDecomposer):
    """
    Data decomposer that splits data into training and inference sets.

    Supports stratified splitting by treatment assignment and optional shuffling.
    """

    def __init__(
        self,
        train_ratio: float = 0.5,
        shuffle: bool = False,
        stratify_by_treatment: bool = False,
    ) -> None:
        """
        Initialize a Splitting data decomposer.

        Parameters
        ----------
        train_ratio : float, optional
            Ratio of data to be used for training, by default 0.5
        shuffle : bool, optional
            Whether to shuffle the data before splitting, by default False
        stratify_by_treatment : bool, optional
            Whether to stratify the data by treatment, by default False
        """
        super().__init__()
        assert (
            0.0 < train_ratio < 1.0
        ), f"train_ratio must be between 0 and 1, got {train_ratio}"
        self.train_ratio = train_ratio
        self.shuffle = shuffle
        self.stratify_by_treatment = stratify_by_treatment

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
        X, A, Y = data.X, data.A, data.Y
        N = X.shape[0]

        # Shuffle data if requested
        if self.shuffle:
            idx = np.random.permutation(N)
            X = X[idx]
            A = A[idx]
            Y = Y[idx]

        if self.stratify_by_treatment:
            # Stratified splitting by treatment
            treated_indices = np.where(A == 1)[0]
            control_indices = np.where(A == 0)[0]

            n_treated_train = int(len(treated_indices) * self.train_ratio)
            n_control_train = int(len(control_indices) * self.train_ratio)

            treated_train_idx = treated_indices[:n_treated_train]
            treated_inference_idx = treated_indices[n_treated_train:]

            control_train_idx = control_indices[:n_control_train]
            control_inference_idx = control_indices[n_control_train:]

            train_idx = np.concatenate([treated_train_idx, control_train_idx])
            inference_idx = np.concatenate(
                [treated_inference_idx, control_inference_idx]
            )

            X_t, A_t, Y_t = X[train_idx], A[train_idx], Y[train_idx]
            X_v, A_v, Y_v = X[inference_idx], A[inference_idx], Y[inference_idx]
        else:
            # Simple splitting without stratification
            n_per_split = int(N * self.train_ratio)
            X_t = X[:n_per_split, :]
            A_t = A[:n_per_split]
            Y_t = Y[:n_per_split]
            X_v = X[n_per_split:, :]
            A_v = A[n_per_split:]
            Y_v = Y[n_per_split:]

        # Create Dataset objects for training and inference
        train_data = Dataset(X=X_t, A=A_t, Y=Y_t)
        inference_data = Dataset(X=X_v, A=A_v, Y=Y_v)

        return TrainingInferenceData(train=train_data, inf=inference_data)


class ThinningFission(DataDecomposer):
    """
    Data decomposer that applies thinning/fission operations to create training and inference sets.

    Uses additive/subtractive noise for outcomes and optionally flips treatment assignments
    to create synthetic train/inference splits while preserving statistical relationships.
    """

    def __init__(
        self,
        std_outcome_noise: float = 1.0,
        train_information_parameter: float = 1.0,
        decompose_treatment: bool = False,
        treatment_noise: float | None = None,
    ) -> None:
        """
        Initialize a ThinningFission data decomposer.

        Parameters
        ----------
        std_outcome_noise : float, optional
            Noise standard deviation to be added/subtracted to/from Y, by default 1.0
        train_information_parameter : float, optional
            Coefficient for fission operation, by default 1.0
        decompose_treatment : bool, optional
            Whether to perform fission on treatment, by default False
        treatment_noise : float | None, optional
            Epsilon for fission treatment (assumes Bernoulli treatment), by default None
        """
        super().__init__()

        if not decompose_treatment and treatment_noise is not None:
            raise ValueError(
                "treatment_noise must be None when decompose_treatment is False"
            )

        if decompose_treatment and treatment_noise is None:
            raise ValueError(
                "treatment_noise must be provided when decompose_treatment is True"
            )

        self.std_outcome_noise = std_outcome_noise
        self.train_information_parameter = train_information_parameter
        self.decompose_treatment = decompose_treatment
        self.treatment_noise = treatment_noise

    def decompose(self, data: Dataset) -> TrainingInferenceData:
        """
        Decompose a dataset using fission operations.

        Parameters
        ----------
        data : Dataset
            Complete dataset to decompose

        Returns
        -------
        TrainingInferenceData
            Data split using fission operations
        """
        X, A, Y = data.X, data.A, data.Y

        # Generate noise for outcome fission
        Z = np.random.randn(X.shape[0]) * self.std_outcome_noise
        Y_t = Y + self.train_information_parameter * Z
        Y_v = Y - (1 / self.train_information_parameter) * Z

        if not self.decompose_treatment:
            # Only fission on outcomes, keep treatments the same
            train_data = Dataset(X=X, A=A, Y=Y_t)
            inference_data = Dataset(X=X, A=A, Y=Y_v)
        else:
            # Fission on both outcomes and treatments
            Z_treatment = np.random.binomial(1, self.treatment_noise, size=X.shape[0])
            A_t = (
                A * (1 - Z_treatment) + (1 - A) * Z_treatment
            )  # Flip treatment assignments
            A_v = A  # Keep original treatment assignments for inference

            train_data = Dataset(X=X, A=A_t, Y=Y_t)
            inference_data = Dataset(X=X, A=A_v, Y=Y_v)

        return TrainingInferenceData(train=train_data, inf=inference_data)
