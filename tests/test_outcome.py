# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import testslide
from data_decomposition.outcome import (
    DGPLinearGaussianOutcomeModel,
    LassoOutcomeModel,
    OLSOutcomeModel,
)
from data_decomposition.types import Dataset


class TestOLSOutcomeModel(testslide.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)
        self.num_samples = 100
        self.dim_covariates = 5

        # Create test data with known linear relationship
        X = np.random.randn(self.num_samples, self.dim_covariates)
        A = np.random.binomial(1, 0.5, self.num_samples)
        # Linear relationship: Y = 2*A + sum(X) + noise
        Y = 2 * A + np.sum(X, axis=1) + 0.1 * np.random.randn(self.num_samples)
        # Ensure Y is NDArray type
        Y = np.asarray(Y)
        self.data = Dataset(X=X, A=A, Y=Y)

    def test_ols_basic_functionality(self) -> None:
        """Test basic OLS fitting and prediction."""
        model = OLSOutcomeModel()
        model.fit(self.data)

        # Test prediction
        predictions = model.predict(self.data)
        self.assertEqual(predictions.shape, (self.num_samples,))

        # Check that predictions are reasonable (R^2 should be high for this linear case)
        residuals = self.data.Y - predictions
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.data.Y - np.mean(self.data.Y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        self.assertGreater(r_squared, 0.8)  # Should have good fit for linear data

    def test_ols_prediction_shapes(self) -> None:
        """Test that predictions have correct shapes."""
        model = OLSOutcomeModel()
        model.fit(self.data)

        # Test prediction on different data sizes
        test_data = Dataset(
            X=np.random.randn(50, self.dim_covariates),
            A=np.random.binomial(1, 0.5, 50),
            Y=np.zeros(50),  # Y not used for prediction
        )
        predictions = model.predict(test_data)
        self.assertEqual(predictions.shape, (50,))

    def test_ols_treatment_effect_recovery(self) -> None:
        """Test that OLS can recover treatment effects."""
        model = OLSOutcomeModel()
        model.fit(self.data)

        # Create test data where treatment is the only difference
        X_test = np.zeros((100, self.dim_covariates))
        treated_data = Dataset(X=X_test, A=np.ones(100), Y=np.zeros(100))
        control_data = Dataset(X=X_test, A=np.zeros(100), Y=np.zeros(100))

        treated_preds = model.predict(treated_data)
        control_preds = model.predict(control_data)

        estimated_effect = np.mean(treated_preds - control_preds)
        # Should be close to the true effect of 2
        self.assertAlmostEqual(estimated_effect, 2.0, delta=0.5)

    def test_ols_must_fit_before_predict(self) -> None:
        """Test that prediction fails if model not fitted."""
        model = OLSOutcomeModel()

        with self.assertRaises(ValueError):
            model.predict(self.data)


class TestLassoOutcomeModel(testslide.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)
        self.num_samples = 100
        self.dim_covariates = 10

        # Create test data with sparse relationship
        X = np.random.randn(self.num_samples, self.dim_covariates)
        A = np.random.binomial(1, 0.5, self.num_samples)
        # Only first 3 features matter: Y = A + X[:, 0] + 0.5*X[:, 1] + 0.25*X[:, 2] + noise
        Y = (
            A
            + X[:, 0]
            + 0.5 * X[:, 1]
            + 0.25 * X[:, 2]
            + 0.1 * np.random.randn(self.num_samples)
        )
        self.data = Dataset(X=X, A=A, Y=Y)

    def test_lasso_basic_functionality(self) -> None:
        """Test basic Lasso fitting and prediction."""
        model = LassoOutcomeModel()
        model.fit(self.data)

        # Test prediction
        predictions = model.predict(self.data)
        self.assertEqual(predictions.shape, (self.num_samples,))

    def test_lasso_sparsity(self) -> None:
        """Test that Lasso can handle sparse relationships."""
        model = LassoOutcomeModel(alpha_grid=np.logspace(-3, 1, 10))
        model.fit(self.data)

        predictions = model.predict(self.data)

        # Check that predictions are reasonable
        residuals = self.data.Y - predictions
        mse = np.mean(residuals**2)
        self.assertLess(mse, 1.0)  # Should have reasonable MSE

    def test_lasso_different_alpha_grids(self) -> None:
        """Test Lasso with different regularization grids."""
        # High regularization
        model_high = LassoOutcomeModel(alpha_grid=np.array([10.0, 100.0]))
        model_high.fit(self.data)
        preds_high = model_high.predict(self.data)

        # Low regularization
        model_low = LassoOutcomeModel(alpha_grid=np.array([0.001, 0.01]))
        model_low.fit(self.data)
        preds_low = model_low.predict(self.data)

        # Predictions should be different
        self.assertFalse(np.allclose(preds_high, preds_low))

    def test_lasso_cv_folds(self) -> None:
        """Test Lasso with different CV fold numbers."""
        model = LassoOutcomeModel(cv_folds=3)
        model.fit(self.data)

        predictions = model.predict(self.data)
        self.assertEqual(predictions.shape, (self.num_samples,))

    def test_lasso_must_fit_before_predict(self) -> None:
        """Test that prediction fails if model not fitted."""
        model = LassoOutcomeModel()

        with self.assertRaises(ValueError):
            model.predict(self.data)


class TestDGPLinearGaussianOutcomeModel(testslide.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)
        self.num_samples = 100
        self.dim_covariates = 10

        # Create test data
        X = np.random.randn(self.num_samples, self.dim_covariates)
        A = np.random.binomial(1, 0.5, self.num_samples)
        Y = np.random.randn(self.num_samples)
        self.data = Dataset(X=X, A=A, Y=Y)

    def test_dgp_basic_functionality(self) -> None:
        """Test basic DGP model functionality."""
        model = DGPLinearGaussianOutcomeModel()
        model.fit(self.data)

        # Test prediction
        predictions = model.predict(self.data)
        self.assertEqual(predictions.shape, (self.num_samples,))

    def test_dgp_sparsity_factor(self) -> None:
        """Test that sparsity factor works correctly."""
        # No sparsity
        model_dense = DGPLinearGaussianOutcomeModel(sparsity_factor=0.0)
        model_dense.fit(self.data)

        # 50% sparsity
        model_sparse = DGPLinearGaussianOutcomeModel(sparsity_factor=0.5)
        model_sparse.fit(self.data)

        # Check that sparse model has some zero coefficients
        num_zeros = np.sum(model_sparse._coefficients == 0)
        expected_zeros = int(np.ceil(self.dim_covariates * 0.5))
        self.assertEqual(num_zeros, expected_zeros)

        # Dense model should have no zeros (with high probability)
        num_zeros_dense = np.sum(np.abs(model_dense._coefficients) < 1e-10)
        self.assertEqual(num_zeros_dense, 0)

    def test_dgp_coefficient_distribution(self) -> None:
        """Test that coefficients follow expected distribution."""
        model = DGPLinearGaussianOutcomeModel(std_coefficients=2.0)
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

    def test_dgp_deterministic_with_seed(self) -> None:
        """Test that DGP is deterministic given a seed."""
        # Fit two models with same seed
        np.random.seed(123)
        model1 = DGPLinearGaussianOutcomeModel()
        model1.fit(self.data)

        np.random.seed(123)
        model2 = DGPLinearGaussianOutcomeModel()
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

    def test_dgp_edge_cases(self) -> None:
        """Test edge cases for DGP model."""
        # Test boundary sparsity factors
        model = DGPLinearGaussianOutcomeModel(sparsity_factor=0.9)
        model.fit(self.data)

        # Should have almost all zeros
        num_zeros = np.sum(model._coefficients == 0)
        expected_zeros = int(np.ceil(self.dim_covariates * 0.9))
        self.assertEqual(num_zeros, expected_zeros)

        # Test invalid sparsity factor
        with self.assertRaises(AssertionError):
            DGPLinearGaussianOutcomeModel(sparsity_factor=1.0)

        with self.assertRaises(AssertionError):
            DGPLinearGaussianOutcomeModel(sparsity_factor=-0.1)

    def test_dgp_must_fit_before_predict(self) -> None:
        """Test that prediction fails if model not fitted."""
        model = DGPLinearGaussianOutcomeModel()

        with self.assertRaises(ValueError):
            model.predict(self.data)

    def test_dgp_different_data_dimensions(self) -> None:
        """Test DGP with different data dimensions."""
        # Test with different dimension
        X_small = np.random.randn(50, 3)
        data_small = Dataset(
            X=X_small, A=np.random.binomial(1, 0.5, 50), Y=np.random.randn(50)
        )

        model = DGPLinearGaussianOutcomeModel()
        model.fit(data_small)

        # Coefficients should match data dimension
        self.assertIsNotNone(model._coefficients)
        assert model._coefficients is not None  # Type narrowing for pyre
        self.assertEqual(len(model._coefficients), 3)

        predictions = model.predict(data_small)
        self.assertEqual(predictions.shape, (50,))
