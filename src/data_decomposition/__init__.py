# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Data Decomposition Library

A self-contained library for causal inference data decomposition methods,
including AIPW estimators, data generators, and various propensity and outcome models.
"""

__version__ = "0.1.0"
__author__ = "Meta Platforms, Inc."
__license__ = "MIT"

# Import main components for easy access
from .aipw import AIPW
from .data_decomposer import Splitting, ThinningFission
from .data_generator import BinomialGaussian, HeteroGaussian, HomoGaussian
from .outcome import DGPLinearGaussianOutcomeModel, LassoOutcomeModel, OLSOutcomeModel
from .propensity import (
    AveragePropensityModel,
    ConstantPropensityModel,
    DGPSigmoidGaussianPropensityModel,
    LassoLogisticPropensityModel,
    LogisticPropensityModel,
    SigmoidPropensityModel,
)
from .types import (
    AverageTreatmentEffect,
    DataDecomposer,
    DataGenerator,
    Dataset,
    OutcomeModel,
    OutcomeNoiseModel,
    PropensityModel,
    TrainingInferenceData,
)

__all__ = [
    # Core types
    "Dataset",
    "TrainingInferenceData",
    "AverageTreatmentEffect",
    # Abstract base classes
    "PropensityModel",
    "OutcomeModel",
    "DataGenerator",
    "OutcomeNoiseModel",
    "DataDecomposer",
    # Main estimator
    "AIPW",
    # Data decomposers
    "Splitting",
    "ThinningFission",
    # Data generators and noise models
    "BinomialGaussian",
    "HomoGaussian",
    "HeteroGaussian",
    # Outcome models
    "OLSOutcomeModel",
    "LassoOutcomeModel",
    "DGPLinearGaussianOutcomeModel",
    # Propensity models
    "ConstantPropensityModel",
    "AveragePropensityModel",
    "LogisticPropensityModel",
    "LassoLogisticPropensityModel",
    "SigmoidPropensityModel",
    "DGPSigmoidGaussianPropensityModel",
]
