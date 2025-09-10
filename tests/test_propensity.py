# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import testslide

from data_decomposition.propensity import (
    AveragePropensityModel,
    ConstantPropensityModel,
    DGPSigmoidGaussianPropensityModel,
    LassoLogisticPropensityModel,
    LogisticPropensityModel,
    SigmoidPropensityModel,
)
from data_decomposition.types import Dataset


class TestConstantPropensityModel(testslide.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)
        self.num_samples = 100
        self.dim_covariates = 5

        X = np.random.randn(self.num_samples, self.dim_covariates)
        A = np.random.binomial(1, 0.5, self.num_samples)
        Y = np.random.randn(self.num_samples)
        self.data = Dataset(X=X, A=A, Y=Y)

    def test_constant_basic_functionality(self) -> None:
        """Test basic constant propensity functionality."""
        propensity_score = 0.3
        model = ConstantPropensityModel(propensity_score)

        # Fit (does nothing)
        model.fit(self.data)

        # Test prediction
        predictions = model.predict(self.data)
        self.assertEqual(predictions.shape, (self.num_samples,))
        self.assertTrue(np.allclose(predictions, propensity_score))

    def test_constant_edge_cases(self) -> None:
        """Test edge cases for constant propensity."""
        # Test near boundaries
        model_low = ConstantPropensityModel(0.001)
        model_low.fit(self.data)
        preds_low = model_low.predict(self.data)
        self.assertTrue(np.allclose(preds_low, 0.001))

        model_high = ConstantPropensityModel(0.999)
        model_high.fit(self.data)
        preds_high = model_high.predict(self.data)
        self.assertTrue(np.allclose(preds_high, 0.999))

    def test_constant_invalid_values(self) -> None:
        """Test that invalid propensity scores are rejected."""
        with self.assertRaises(AssertionError):
            ConstantPropensityModel(0.0)

        with self.assertRaises(AssertionError):
            ConstantPropensityModel(1.0)

        with self.assertRaises(AssertionError):
            ConstantPropensityModel(-0.1)

        with self.assertRaises(AssertionError):
            ConstantPropensityModel(1.1)


class TestAveragePropensityModel(testslide.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)
        self.num_samples = 100
        self.dim_covariates = 5

        X = np.random.randn(self.num_samples, self.dim_covariates)
        # Create data with known treatment proportion
        A = np.random.binomial(1, 0.6, self.num_samples)  # 60% treatment rate
        Y = np.random.randn(self.num_samples)
        self.data = Dataset(X=X, A=A, Y=Y)

    def test_average_basic_functionality(self) -> None:
        """Test basic average propensity functionality."""
        model = AveragePropensityModel()
        model.fit(self.data)

        predictions = model.predict(self.data)
        self.assertEqual(predictions.shape, (self.num_samples,))

        # Should predict the empirical treatment rate
        expected_rate = np.mean(self.data.A)
        self.assertTrue(np.allclose(predictions, expected_rate))

    def test_average_different_treatment_rates(self) -> None:
        """Test average propensity with different treatment rates."""
        # Create data with different treatment rates
        X = np.random.randn(100, 5)
        A_low = np.concatenate([np.ones(20), np.zeros(80)])  # 20% treatment
        A_high = np.concatenate([np.ones(80), np.zeros(20)])  # 80% treatment
        Y = np.random.randn(100)

        data_low = Dataset(X=X, A=A_low, Y=Y)
        data_high = Dataset(X=X, A=A_high, Y=Y)

        model_low = AveragePropensityModel()
        model_low.fit(data_low)
        preds_low = model_low.predict(data_low)

        model_high = AveragePropensityModel()
        model_high.fit(data_high)
        preds_high = model_high.predict(data_high)

        self.assertAlmostEqual(preds_low[0], 0.2, places=2)
        self.assertAlmostEqual(preds_high[0], 0.8, places=2)

    def test_average_extreme_cases(self) -> None:
        """Test edge cases for average propensity."""
        X = np.random.randn(100, 5)
        Y = np.random.randn(100)

        # Very low treatment rate
        A_extreme_low = np.concatenate([np.ones(1), np.zeros(99)])
        data_extreme_low = Dataset(X=X, A=A_extreme_low, Y=Y)

        model = AveragePropensityModel()
        model.fit(data_extreme_low)
        preds = model.predict(data_extreme_low)
        self.assertAlmostEqual(preds[0], 0.01, places=3)


class TestLogisticPropensityModel(testslide.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)
        self.num_samples = 200
        self.dim_covariates = 5

        # Create data with logistic relationship
        X = np.random.randn(self.num_samples, self.dim_covariates)
        # True logistic model: logit(p) = X[0] + 0.5*X[1]
        logits = X[:, 0] + 0.5 * X[:, 1]
        true_probs = 1 / (1 + np.exp(-logits))
        A = np.random.binomial(1, true_probs, self.num_samples)
        Y = np.random.randn(self.num_samples)
        self.data = Dataset(X=X, A=A, Y=Y)

    def test_logistic_basic_functionality(self) -> None:
        """Test basic logistic propensity functionality."""
        model = LogisticPropensityModel()
        model.fit(self.data)

        predictions = model.predict(self.data)
        self.assertEqual(predictions.shape, (self.num_samples,))

        # Check that predictions are in valid range
        self.assertTrue(np.all(predictions > 0))
        self.assertTrue(np.all(predictions < 1))

    def test_logistic_parameter_recovery(self) -> None:
        """Test that logistic model can recover parameters."""
        model = LogisticPropensityModel()
        model.fit(self.data)

        predictions = model.predict(self.data)

        # Check that model makes reasonable predictions
        # Higher X[0] should generally lead to higher propensity
        high_x0_mask = self.data.X[:, 0] > 1
        low_x0_mask = self.data.X[:, 0] < -1

        if np.sum(high_x0_mask) > 0 and np.sum(low_x0_mask) > 0:
            high_x0_preds = np.mean(predictions[high_x0_mask])
            low_x0_preds = np.mean(predictions[low_x0_mask])
            self.assertGreater(high_x0_preds, low_x0_preds)

    def test_logistic_clipping(self) -> None:
        """Test that predictions are clipped for stability."""
        model = LogisticPropensityModel(clip_for_stability=0.01)
        model.fit(self.data)

        predictions = model.predict(self.data)

        # Check clipping bounds
        self.assertTrue(np.all(predictions >= 0.01))
        self.assertTrue(np.all(predictions <= 0.99))

    def test_logistic_different_parameters(self) -> None:
        """Test logistic model with different parameters."""
        model1 = LogisticPropensityModel(maxiter=500, fit_method="bfgs")
        model2 = LogisticPropensityModel(maxiter=1000, fit_method="bfgs")

        model1.fit(self.data)
        model2.fit(self.data)

        preds1 = model1.predict(self.data)
        preds2 = model2.predict(self.data)

        # Should be similar but may have small differences
        self.assertTrue(np.allclose(preds1, preds2, atol=0.1))

    def test_logistic_must_fit_before_predict(self) -> None:
        """Test that prediction fails if model not fitted."""
        model = LogisticPropensityModel()

        with self.assertRaises(ValueError):
            model.predict(self.data)


class TestLassoLogisticPropensityModel(testslide.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)
        self.num_samples = 200
        self.dim_covariates = 10

        # Create sparse logistic relationship (only first 2 features matter)
        X = np.random.randn(self.num_samples, self.dim_covariates)
        logits = X[:, 0] + 0.5 * X[:, 1]  # Only first two features matter
        true_probs = 1 / (1 + np.exp(-logits))
        A = np.random.binomial(1, true_probs, self.num_samples)
        Y = np.random.randn(self.num_samples)
        self.data = Dataset(X=X, A=A, Y=Y)

    def test_lasso_logistic_basic_functionality(self) -> None:
        """Test basic Lasso logistic functionality."""
        model = LassoLogisticPropensityModel()
        model.fit(self.data)

        predictions = model.predict(self.data)
        self.assertEqual(predictions.shape, (self.num_samples,))

        # Check that predictions are in valid range
        self.assertTrue(np.all(predictions > 0))
        self.assertTrue(np.all(predictions < 1))

    def test_lasso_logistic_regularization(self) -> None:
        """Test different regularization parameters."""
        # High regularization (small C)
        model_high_reg = LassoLogisticPropensityModel(
            alpha_grid=np.array([0.001, 0.01])
        )
        model_high_reg.fit(self.data)
        preds_high_reg = model_high_reg.predict(self.data)

        # Low regularization (large C)
        model_low_reg = LassoLogisticPropensityModel(alpha_grid=np.array([10.0, 100.0]))
        model_low_reg.fit(self.data)
        preds_low_reg = model_low_reg.predict(self.data)

        # Predictions should be different
        self.assertFalse(np.allclose(preds_high_reg, preds_low_reg, atol=0.01))

    def test_lasso_logistic_cv_folds(self) -> None:
        """Test with different CV fold numbers."""
        model = LassoLogisticPropensityModel(cv_folds=3)
        model.fit(self.data)

        predictions = model.predict(self.data)
        self.assertEqual(predictions.shape, (self.num_samples,))

    def test_lasso_logistic_solver_penalty(self) -> None:
        """Test different solver and penalty combinations."""
        model = LassoLogisticPropensityModel(penalty="l1", solver="liblinear")
        model.fit(self.data)

        predictions = model.predict(self.data)
        self.assertEqual(predictions.shape, (self.num_samples,))

    def test_lasso_logistic_must_fit_before_predict(self) -> None:
        """Test that prediction fails if model not fitted."""
        model = LassoLogisticPropensityModel()

        with self.assertRaises(ValueError):
            model.predict(self.data)


class TestSigmoidPropensityModel(testslide.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)
        self.num_samples = 100
        self.dim_covariates = 5

        X = np.random.randn(self.num_samples, self.dim_covariates)
        A = np.random.binomial(1, 0.5, self.num_samples)
        Y = np.random.randn(self.num_samples)
        self.data = Dataset(X=X, A=A, Y=Y)

        # Define sigmoid weights
        self.sigmoid_weight = np.array([1.0, 0.5, -0.3, 0.0, 0.2])

    def test_sigmoid_basic_functionality(self) -> None:
        """Test basic sigmoid propensity functionality."""
        model = SigmoidPropensityModel(self.sigmoid_weight)

        # Fit (does nothing)
        model.fit(self.data)

        predictions = model.predict(self.data)
        self.assertEqual(predictions.shape, (self.num_samples,))

        # Check that predictions are in valid range
        self.assertTrue(np.all(predictions > 0))
        self.assertTrue(np.all(predictions < 1))

    def test_sigmoid_deterministic(self) -> None:
        """Test that sigmoid predictions are deterministic."""
        model = SigmoidPropensityModel(self.sigmoid_weight)
        model.fit(self.data)

        preds1 = model.predict(self.data)
        preds2 = model.predict(self.data)

        self.assertTrue(np.allclose(preds1, preds2))

    def test_sigmoid_clipping(self) -> None:
        """Test sigmoid clipping."""
        model = SigmoidPropensityModel(self.sigmoid_weight, clip_for_stability=0.05)
        model.fit(self.data)

        predictions = model.predict(self.data)

        # Check clipping bounds
        self.assertTrue(np.all(predictions >= 0.05))
        self.assertTrue(np.all(predictions <= 0.95))

    def test_sigmoid_different_weights(self) -> None:
        """Test sigmoid with different weight vectors."""
        weight1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        weight2 = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

        model1 = SigmoidPropensityModel(weight1)
        model2 = SigmoidPropensityModel(weight2)

        model1.fit(self.data)
        model2.fit(self.data)

        preds1 = model1.predict(self.data)
        preds2 = model2.predict(self.data)

        # Should be different (unless X[:, 0] == X[:, 1] which is unlikely)
        self.assertFalse(np.allclose(preds1, preds2))


class TestDGPSigmoidGaussianPropensityModel(testslide.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)
        self.num_samples = 100
        self.dim_covariates = 10

        X = np.random.randn(self.num_samples, self.dim_covariates)
        A = np.random.binomial(1, 0.5, self.num_samples)
        Y = np.random.randn(self.num_samples)
        self.data = Dataset(X=X, A=A, Y=Y)

    def test_dgp_sigmoid_basic_functionality(self) -> None:
        """Test basic DGP sigmoid functionality."""
        model = DGPSigmoidGaussianPropensityModel()
        model.fit(self.data)

        predictions = model.predict(self.data)
        self.assertEqual(predictions.shape, (self.num_samples,))

        # Check that predictions are in valid range
        self.assertTrue(np.all(predictions > 0))
        self.assertTrue(np.all(predictions < 1))

    def test_dgp_sigmoid_sparsity_factor(self) -> None:
        """Test sparsity factor in DGP sigmoid model."""
        # No sparsity
        model_dense = DGPSigmoidGaussianPropensityModel(sparsity_factor=0.0)
        model_dense.fit(self.data)

        # 50% sparsity
        model_sparse = DGPSigmoidGaussianPropensityModel(sparsity_factor=0.5)
        model_sparse.fit(self.data)

        # Check that sparse model has some zero coefficients
        num_zeros = np.sum(model_sparse._coefficients == 0)
        expected_zeros = int(np.ceil(self.dim_covariates * 0.5))
        self.assertEqual(num_zeros, expected_zeros)

        # Dense model should have no zeros (with high probability)
        num_zeros_dense = np.sum(np.abs(model_dense._coefficients) < 1e-10)
        self.assertEqual(num_zeros_dense, 0)

    def test_dgp_sigmoid_coefficient_distribution(self) -> None:
        """Test coefficient distribution in DGP sigmoid model."""
        model = DGPSigmoidGaussianPropensityModel(std_coefficients=2.0)
        model.fit(self.data)

        # Check that coefficients have roughly the right scale
        self.assertIsNotNone(model._coefficients)
        assert model._coefficients is not None  # Type narrowing for pyre
        non_zero_coefs = model._coefficients[model._coefficients != 0]
        if len(non_zero_coefs) > 0:
            empirical_std = np.std(non_zero_coefs)
            # Should be roughly around 2.0, allow some variation
            self.assertGreater(empirical_std, 1.0)
            self.assertLess(empirical_std, 4.0)

    def test_dgp_sigmoid_deterministic_with_seed(self) -> None:
        """Test that DGP sigmoid is deterministic given a seed."""
        # Fit two models with same seed
        np.random.seed(123)
        model1 = DGPSigmoidGaussianPropensityModel()
        model1.fit(self.data)

        np.random.seed(123)
        model2 = DGPSigmoidGaussianPropensityModel()
        model2.fit(self.data)

        # Coefficients should be identical
        self.assertIsNotNone(model1._coefficients)
        self.assertIsNotNone(model2._coefficients)
        assert model1._coefficients is not None  # Type narrowing for pyre
        assert model2._coefficients is not None  # Type narrowing for pyre
        # pyre-ignore[6]: Type checker incorrectly infers Optional[NDArray] from _coefficients
        self.assertTrue(np.allclose(model1._coefficients, model2._coefficients))

        # Predictions should be identical
        preds1 = model1.predict(self.data)
        preds2 = model2.predict(self.data)
        self.assertTrue(np.allclose(preds1, preds2))

    def test_dgp_sigmoid_clipping(self) -> None:
        """Test clipping in DGP sigmoid model."""
        model = DGPSigmoidGaussianPropensityModel(clip_for_stability=0.01)
        model.fit(self.data)

        predictions = model.predict(self.data)

        # Check clipping bounds
        self.assertTrue(np.all(predictions >= 0.01))
        self.assertTrue(np.all(predictions <= 0.99))

    def test_dgp_sigmoid_edge_cases(self) -> None:
        """Test edge cases for DGP sigmoid model."""
        # Test boundary sparsity factors
        model = DGPSigmoidGaussianPropensityModel(sparsity_factor=0.99)
        model.fit(self.data)

        # Should have almost all zeros
        num_zeros = np.sum(model._coefficients == 0)
        expected_zeros = int(np.ceil(self.dim_covariates * 0.99))
        self.assertEqual(num_zeros, expected_zeros)

        # Test invalid sparsity factor
        with self.assertRaises(AssertionError):
            DGPSigmoidGaussianPropensityModel(sparsity_factor=1.0)

        with self.assertRaises(AssertionError):
            DGPSigmoidGaussianPropensityModel(sparsity_factor=-0.1)

    def test_dgp_sigmoid_must_fit_before_predict(self) -> None:
        """Test that prediction fails if model not fitted."""
        model = DGPSigmoidGaussianPropensityModel()

        with self.assertRaises(ValueError):
            model.predict(self.data)

    def test_dgp_sigmoid_different_dimensions(self) -> None:
        """Test DGP sigmoid with different data dimensions."""
        # Test with different dimension
        X_small = np.random.randn(50, 3)
        data_small = Dataset(
            X=X_small, A=np.random.binomial(1, 0.5, 50), Y=np.random.randn(50)
        )

        model = DGPSigmoidGaussianPropensityModel()
        model.fit(data_small)

        # Coefficients should match data dimension
        self.assertIsNotNone(model._coefficients)
        assert model._coefficients is not None  # Type narrowing for pyre
        self.assertEqual(len(model._coefficients), 3)

        predictions = model.predict(data_small)
        self.assertEqual(predictions.shape, (50,))
