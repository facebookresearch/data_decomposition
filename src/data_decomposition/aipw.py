# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import numpy as np
from numpy.typing import NDArray

from .types import (
    AverageTreatmentEffect,
    Dataset,
    OutcomeModel,
    PropensityModel,
    TrainingInferenceData,
)


class AIPW:
    """Augmented Inverse Probability Weighting (AIPW) estimator."""

    def __init__(self):
        self._ate = None

    def fit(
        self,
        data: TrainingInferenceData,
        propensity_model: PropensityModel,
        outcome_model: OutcomeModel,
        cross_fit: bool = False,
    ) -> None:
        """
        Fit the AIPW estimator.

        Parameters
        ----------
        data : TrainingInferenceData
            Training and inference data
        propensity_model : PropensityModel
            Propensity score model
        outcome_model : OutcomeModel
            Outcome model
        cross_fit : bool, optional
            Whether to use cross-fitting, by default False
        """
        # Fit models on training data
        propensity_model.fit(data.train)
        outcome_model.fit(data.train)
        # Calculate scores for inference data
        scores_inf = self._estimate_scores(data.inf, outcome_model, propensity_model)
        if cross_fit:
            # Cross-fitting AIPW
            # Fit models on inference data, estimate on training data
            propensity_model.fit(data.inf)
            outcome_model.fit(data.inf)

            scores_train = self._estimate_scores(
                data.train,
                outcome_model,
                propensity_model,
            )
            # Combine estimates
            scores = np.concatenate([scores_inf, scores_train])
        else:
            # Standard AIPW
            scores = scores_inf

        # Calculate ATE estimate
        ate_hat = scores.mean()
        # TODO: adjust for degrees of freedom in standard error
        ate_se = scores.std() / np.sqrt(len(scores))
        self._ate = AverageTreatmentEffect(ate=ate_hat, ate_se=ate_se)

    def _estimate_scores(
        self,
        data: Dataset,
        outcome_model: OutcomeModel,
        propensity_model: PropensityModel,
    ) -> NDArray:
        """Estimate AIPW scores."""
        # Create data for treated potential outcomes
        treated_data = Dataset(
            X=data.X,
            A=np.ones(data.X.shape[0]),
            Y=data.Y,  # Not used in prediction but needed for Dataset structure
        )

        # Create data for control potential outcomes
        control_data = Dataset(
            X=data.X,
            A=np.zeros(data.X.shape[0]),
            Y=data.Y,  # Not used in prediction but needed for Dataset structure
        )

        # Predict potential outcomes
        treated_outcome = outcome_model.predict(treated_data)
        control_outcome = outcome_model.predict(control_data)

        # Get propensity scores
        prop_scores = propensity_model.predict(data)

        # Calculate AIPW scores
        scores = (
            (data.A / prop_scores) * (data.Y - treated_outcome)
            + treated_outcome
            - (
                ((1 - data.A) / (1 - prop_scores)) * (data.Y - control_outcome)
                + control_outcome
            )
        )

        return scores

    def ate(self) -> AverageTreatmentEffect:
        """
        Get the average treatment effect estimate.

        Returns
        -------
        AverageTreatmentEffect
            The estimated ATE and its standard error

        Raises
        ------
        ValueError
            If the model has not been fitted yet
        """
        if self._ate is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self._ate
