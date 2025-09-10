# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import testslide

from data_decomposition.aipw import AIPW
from data_decomposition.data_decomposer import Splitting, ThinningFission
from data_decomposition.data_generator import BinomialGaussian, HomoGaussian
from data_decomposition.outcome import (
    DGPLinearGaussianOutcomeModel,
    LassoOutcomeModel,
    OLSOutcomeModel,
)
from data_decomposition.propensity import (
    AveragePropensityModel,
    ConstantPropensityModel,
    DGPSigmoidGaussianPropensityModel,
    LogisticPropensityModel,
)
from data_decomposition.types import Dataset


class TestAIPWBasicFunctionality(testslide.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)
        self.num_samples = 200
        self.dim_covariates = 5
        self.true_ate = 2.0

        # Generate synthetic data with known treatment effect
        outcome_model = DGPLinearGaussianOutcomeModel()
        propensity_model = DGPSigmoidGaussianPropensityModel()
        noise_model = HomoGaussian(std_noise=1.0)

        generator = BinomialGaussian(
            outcome_model=outcome_model,
            propensity_model=propensity_model,
            outcome_noise_model=noise_model,
            dim_covariates=self.dim_covariates,
            std_covariates=1.0,
        )

        self.full_data = generator.generate(
            num_samples=self.num_samples,
            treatment_effect=self.true_ate,
        )

        # Split data for AIPW
        splitter = Splitting(train_ratio=0.6, stratify_by_treatment=True)
        self.training_inference_data = splitter.decompose(self.full_data)

    def test_aipw_basic_estimation(self) -> None:
        """Test basic AIPW estimation functionality."""
        estimator = AIPW()
        propensity_model = LogisticPropensityModel()
        outcome_model = OLSOutcomeModel()

        estimator.fit(
            self.training_inference_data,
            propensity_model,
            outcome_model,
            cross_fit=False,
        )

        ate = estimator.ate()

        # Check that we get a reasonable estimate
        self.assertIsNotNone(ate)
        self.assertGreater(ate.ate_se, 0)  # Standard error should be positive

        # Estimate should be in a reasonable range around true ATE
        self.assertGreater(ate.ate, self.true_ate - 2.0)
        self.assertLess(ate.ate, self.true_ate + 2.0)

    def test_aipw_cross_fitting(self) -> None:
        """Test AIPW with cross-fitting."""
        estimator = AIPW()
        propensity_model = LogisticPropensityModel()
        outcome_model = OLSOutcomeModel()

        estimator.fit(
            self.training_inference_data,
            propensity_model,
            outcome_model,
            cross_fit=True,
        )
        ate = estimator.ate()

        # Check basic properties
        self.assertIsNotNone(ate)
        self.assertGreater(ate.ate_se, 0)

        # Should be in reasonable range
        self.assertGreater(ate.ate, self.true_ate - 3.0)
        self.assertLess(ate.ate, self.true_ate + 3.0)

    def test_aipw_different_model_combinations(self) -> None:
        """Test AIPW with different propensity and outcome model combinations."""
        estimator = AIPW()

        # Test different combinations
        combinations = [
            (ConstantPropensityModel(0.5), OLSOutcomeModel()),
            (AveragePropensityModel(), LassoOutcomeModel()),
            (LogisticPropensityModel(), OLSOutcomeModel()),
        ]

        estimates = []
        for prop_model, outcome_model in combinations:
            estimator_test = AIPW()
            estimator_test.fit(
                self.training_inference_data, prop_model, outcome_model, cross_fit=False
            )
            ate = estimator_test.ate()
            estimates.append(ate.ate)

        # All estimates should be in reasonable range
        for estimate in estimates:
            self.assertGreater(estimate, self.true_ate - 3.0)
            self.assertLess(estimate, self.true_ate + 3.0)

    def test_aipw_deterministic_with_seed(self) -> None:
        """Test that AIPW is deterministic given same models and data."""
        # Create two identical estimators
        np.random.seed(123)
        estimator1 = AIPW()
        propensity_model1 = LogisticPropensityModel()
        outcome_model1 = OLSOutcomeModel()

        estimator1.fit(
            self.training_inference_data,
            propensity_model1,
            outcome_model1,
            cross_fit=False,
        )
        ate1 = estimator1.ate()

        np.random.seed(123)
        estimator2 = AIPW()
        propensity_model2 = LogisticPropensityModel()
        outcome_model2 = OLSOutcomeModel()

        estimator2.fit(
            self.training_inference_data,
            propensity_model2,
            outcome_model2,
            cross_fit=False,
        )
        ate2 = estimator2.ate()

        # Should get identical results
        self.assertAlmostEqual(ate1.ate, ate2.ate, places=10)
        self.assertAlmostEqual(ate1.ate_se, ate2.ate_se, places=10)

    def test_aipw_must_fit_before_ate(self) -> None:
        """Test that ate fails if estimator not fitted."""
        estimator = AIPW()

        with self.assertRaises(ValueError):
            estimator.ate()


class TestAIPWWithFission(testslide.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)
        self.num_samples = 200
        self.dim_covariates = 5
        self.true_ate = 1.5

        # Generate synthetic data
        outcome_model = DGPLinearGaussianOutcomeModel()
        propensity_model = DGPSigmoidGaussianPropensityModel()
        noise_model = HomoGaussian(std_noise=0.5)

        generator = BinomialGaussian(
            outcome_model=outcome_model,
            propensity_model=propensity_model,
            outcome_noise_model=noise_model,
            dim_covariates=self.dim_covariates,
            std_covariates=1.0,
        )

        self.full_data = generator.generate(
            num_samples=self.num_samples,
            treatment_effect=self.true_ate,
        )

    def test_aipw_with_thinning_fission(self) -> None:
        """Test AIPW with thinning fission data decomposition."""
        # Use thinning fission for data splitting
        fission_splitter = ThinningFission(
            std_outcome_noise=0.5,
            train_information_parameter=1.0,
            decompose_treatment=False,
        )
        training_inference_data = fission_splitter.decompose(self.full_data)

        estimator = AIPW()
        propensity_model = LogisticPropensityModel()
        outcome_model = OLSOutcomeModel()

        estimator.fit(
            training_inference_data, propensity_model, outcome_model, cross_fit=False
        )

        ate = estimator.ate()

        # Should get reasonable estimate even with fission
        self.assertIsNotNone(ate)
        self.assertGreater(ate.ate_se, 0)

        # May be less accurate than regular splitting but should be in ballpark
        self.assertGreater(ate.ate, self.true_ate - 4.0)
        self.assertLess(ate.ate, self.true_ate + 4.0)

    def test_aipw_with_treatment_fission(self) -> None:
        """Test AIPW with treatment fission."""
        fission_splitter = ThinningFission(
            std_outcome_noise=0.5,
            train_information_parameter=1.0,
            decompose_treatment=True,
            treatment_noise=0.3,
        )
        training_inference_data = fission_splitter.decompose(self.full_data)

        estimator = AIPW()
        propensity_model = LogisticPropensityModel()
        outcome_model = OLSOutcomeModel()

        estimator.fit(
            training_inference_data, propensity_model, outcome_model, cross_fit=False
        )

        ate = estimator.ate()

        # With treatment fission, estimation might be more challenging
        self.assertIsNotNone(ate)
        self.assertGreater(ate.ate_se, 0)

    def test_aipw_cross_fit_vs_standard(self) -> None:
        """Compare cross-fitting vs standard AIPW."""
        splitter = Splitting(train_ratio=0.6, stratify_by_treatment=True)
        training_inference_data = splitter.decompose(self.full_data)

        # Standard AIPW
        estimator_standard = AIPW()
        propensity_model_std = LogisticPropensityModel()
        outcome_model_std = OLSOutcomeModel()

        estimator_standard.fit(
            training_inference_data,
            propensity_model_std,
            outcome_model_std,
            cross_fit=False,
        )
        ate_standard = estimator_standard.ate()

        # Cross-fitted AIPW
        estimator_crossfit = AIPW()
        propensity_model_cf = LogisticPropensityModel()
        outcome_model_cf = OLSOutcomeModel()

        estimator_crossfit.fit(
            training_inference_data,
            propensity_model_cf,
            outcome_model_cf,
            cross_fit=True,
        )
        ate_crossfit = estimator_crossfit.ate()

        # Both should give reasonable estimates
        self.assertGreater(ate_standard.ate, self.true_ate - 3.0)
        self.assertLess(ate_standard.ate, self.true_ate + 3.0)

        self.assertGreater(ate_crossfit.ate, self.true_ate - 3.0)
        self.assertLess(ate_crossfit.ate, self.true_ate + 3.0)

        # Standard errors should be reasonable
        self.assertGreater(ate_standard.ate_se, 0)
        self.assertGreater(ate_crossfit.ate_se, 0)


class TestAIPWRobustness(testslide.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)

    def test_aipw_small_sample_size(self) -> None:
        """Test AIPW with small sample sizes."""
        num_samples = 50  # Small sample
        outcome_model = DGPLinearGaussianOutcomeModel()
        propensity_model = DGPSigmoidGaussianPropensityModel()
        noise_model = HomoGaussian(std_noise=1.0)

        generator = BinomialGaussian(
            outcome_model=outcome_model,
            propensity_model=propensity_model,
            outcome_noise_model=noise_model,
            dim_covariates=3,
            std_covariates=1.0,
        )

        data = generator.generate(
            num_samples=num_samples,
            treatment_effect=2.0,
        )

        splitter = Splitting(train_ratio=0.6)
        training_inference_data = splitter.decompose(data)

        estimator = AIPW()
        propensity_model_test = ConstantPropensityModel(
            0.5
        )  # Simple model for small sample
        outcome_model_test = OLSOutcomeModel()

        estimator.fit(
            training_inference_data,
            propensity_model_test,
            outcome_model_test,
            cross_fit=False,
        )

        ate = estimator.ate()

        # Should complete without error and give some estimate
        self.assertIsNotNone(ate)
        self.assertGreater(ate.ate_se, 0)

    def test_aipw_extreme_propensity_scores(self) -> None:
        """Test AIPW robustness to extreme propensity scores."""
        # Create data with some extreme propensity scores
        X = np.random.randn(100, 3)
        # Create very imbalanced treatment assignment
        A = np.concatenate([np.ones(10), np.zeros(90)])  # Only 10% treated
        Y = X[:, 0] + 2 * A + np.random.randn(100) * 0.5

        data = Dataset(X=X, A=A, Y=Y)

        splitter = Splitting(train_ratio=0.6, stratify_by_treatment=True)
        training_inference_data = splitter.decompose(data)

        estimator = AIPW()
        # Use average propensity which will be very low
        propensity_model = AveragePropensityModel()
        outcome_model = OLSOutcomeModel()

        estimator.fit(
            training_inference_data, propensity_model, outcome_model, cross_fit=False
        )

        ate = estimator.ate()

        # Should complete without error
        self.assertIsNotNone(ate)
        self.assertGreater(ate.ate_se, 0)

    def test_aipw_high_dimensional_features(self) -> None:
        """Test AIPW with high-dimensional features."""
        num_samples = 200
        dim_covariates = 20  # High dimensional

        # Use sparse models for high-dimensional case
        outcome_model = DGPLinearGaussianOutcomeModel(
            sparsity_factor=0.8
        )  # Most coefficients zero
        propensity_model = DGPSigmoidGaussianPropensityModel(sparsity_factor=0.8)
        noise_model = HomoGaussian(std_noise=1.0)

        generator = BinomialGaussian(
            outcome_model=outcome_model,
            propensity_model=propensity_model,
            outcome_noise_model=noise_model,
            dim_covariates=dim_covariates,
            std_covariates=1.0,
        )

        data = generator.generate(
            num_samples=num_samples,
            treatment_effect=1.5,
        )

        splitter = Splitting(train_ratio=0.6)
        training_inference_data = splitter.decompose(data)

        estimator = AIPW()
        # Use regularized models for high-dimensional case
        propensity_model_test = LogisticPropensityModel()
        outcome_model_test = LassoOutcomeModel()  # Use Lasso for sparsity

        estimator.fit(
            training_inference_data,
            propensity_model_test,
            outcome_model_test,
            cross_fit=False,
        )

        ate = estimator.ate()

        # Should complete without error
        self.assertIsNotNone(ate)
        self.assertGreater(ate.ate_se, 0)

        # Should get reasonable estimate
        self.assertGreater(ate.ate, 1.5 - 4.0)
        self.assertLess(ate.ate, 1.5 + 4.0)

    def test_aipw_no_treatment_variation(self) -> None:
        """Test AIPW behavior when there's insufficient treatment variation."""
        # Create data with very little treatment variation in training
        X = np.random.randn(100, 5)
        A = np.concatenate([np.ones(2), np.zeros(98)])  # Only 2 treated units
        Y = X[:, 0] + 2 * A + np.random.randn(100) * 0.5

        data = Dataset(X=X, A=A, Y=Y)

        # Force a specific split to ensure training has minimal treatment variation
        train_data = Dataset(
            X=X[:60], A=A[:60], Y=Y[:60]
        )  # First 60 samples (likely includes treated)
        inf_data = Dataset(X=X[60:], A=A[60:], Y=Y[60:])

        from data_decomposition.types import TrainingInferenceData

        training_inference_data = TrainingInferenceData(train=train_data, inf=inf_data)

        estimator = AIPW()
        propensity_model = AveragePropensityModel()
        outcome_model = OLSOutcomeModel()

        # This might fail or give unreliable results, but should handle gracefully
        try:
            estimator.fit(
                training_inference_data,
                propensity_model,
                outcome_model,
                cross_fit=False,
            )
            ate = estimator.ate()

            # If it succeeds, should at least return valid ATE object
            self.assertIsNotNone(ate)
            self.assertGreater(ate.ate_se, 0)

        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            # It's acceptable for this to fail gracefully with extreme data
            pass
