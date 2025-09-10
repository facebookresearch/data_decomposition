# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import testslide

from data_decomposition.data_decomposer import Splitting, ThinningFission
from data_decomposition.data_generator import (
    BinomialGaussian,
    HeteroGaussian,
    HomoGaussian,
)
from data_decomposition.outcome import DGPLinearGaussianOutcomeModel
from data_decomposition.propensity import DGPSigmoidGaussianPropensityModel
from data_decomposition.types import Dataset


class TestOutcomeNoiseModel(testslide.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)
        self.sample_size = 100
        self.std_noise = 1.5

    def test_homo_gaussian_basic(self) -> None:
        """Test basic HomoGaussian noise generation."""
        noise_model = HomoGaussian(std_noise=self.std_noise)
        noise = noise_model.sample(size=self.sample_size)

        # Check shape
        self.assertEqual(noise.shape, (self.sample_size,))

        # Check that it's approximately normal with correct std
        # (allowing for some random variation)
        sample_std = np.std(noise)
        self.assertAlmostEqual(sample_std, self.std_noise, delta=0.3)

        # Check mean is approximately zero
        sample_mean = np.mean(noise)
        self.assertAlmostEqual(sample_mean, 0.0, delta=0.3)

    def test_homo_gaussian_reproducibility(self) -> None:
        """Test that HomoGaussian is reproducible with seed."""
        noise_model = HomoGaussian(std_noise=self.std_noise)

        noise1 = noise_model.sample(size=self.sample_size, seed=123)
        noise2 = noise_model.sample(size=self.sample_size, seed=123)

        # Should be identical with same seed
        self.assertTrue(np.array_equal(noise1, noise2))

    def test_homo_gaussian_different_seeds(self) -> None:
        """Test that HomoGaussian produces different results with different seeds."""
        noise_model = HomoGaussian(std_noise=self.std_noise)

        noise1 = noise_model.sample(size=self.sample_size, seed=123)
        noise2 = noise_model.sample(size=self.sample_size, seed=456)

        # Should be different with different seeds
        self.assertFalse(np.array_equal(noise1, noise2))

    def test_hetero_gaussian_basic(self) -> None:
        """Test basic HeteroGaussian noise generation."""
        noise_model = HeteroGaussian(
            std_noise=self.std_noise, betaprime_a=2.0, betaprime_b=5.0
        )
        noise = noise_model.sample(size=self.sample_size)

        # Check shape
        self.assertEqual(noise.shape, (self.sample_size,))

        # Check that all values are non-negative (beta-prime * std_noise)
        self.assertTrue(np.all(noise >= 0))

    def test_hetero_gaussian_reproducibility(self) -> None:
        """Test that HeteroGaussian is reproducible with seed."""
        noise_model = HeteroGaussian(
            std_noise=self.std_noise, betaprime_a=2.0, betaprime_b=5.0
        )

        noise1 = noise_model.sample(size=self.sample_size, seed=123)
        noise2 = noise_model.sample(size=self.sample_size, seed=123)

        # Should be identical with same seed
        self.assertTrue(np.array_equal(noise1, noise2))

    def test_hetero_gaussian_different_parameters(self) -> None:
        """Test that different beta-prime parameters produce different distributions."""
        noise_model1 = HeteroGaussian(
            std_noise=self.std_noise, betaprime_a=1.0, betaprime_b=1.0
        )
        noise_model2 = HeteroGaussian(
            std_noise=self.std_noise, betaprime_a=5.0, betaprime_b=1.0
        )

        # Use same seed to isolate parameter effect
        noise1 = noise_model1.sample(size=self.sample_size, seed=42)
        noise2 = noise_model2.sample(size=self.sample_size, seed=42)

        # Should produce different distributions
        self.assertFalse(np.array_equal(noise1, noise2))

        # Different parameters should lead to different statistics
        self.assertNotAlmostEqual(np.mean(noise1), np.mean(noise2), delta=0.1)

    def test_homo_vs_hetero_different_patterns(self) -> None:
        """Test that HomoGaussian and HeteroGaussian produce different noise patterns."""
        homo_model = HomoGaussian(std_noise=self.std_noise)
        hetero_model = HeteroGaussian(
            std_noise=self.std_noise, betaprime_a=2.0, betaprime_b=5.0
        )

        # Generate noise with different seeds to ensure difference
        homo_noise = homo_model.sample(size=self.sample_size, seed=42)
        hetero_noise = hetero_model.sample(size=self.sample_size, seed=42)

        # Should be different
        self.assertFalse(np.array_equal(homo_noise, hetero_noise))

        # HomoGaussian should have negative values, HeteroGaussian should not
        self.assertTrue(np.any(homo_noise < 0))
        self.assertTrue(np.all(hetero_noise >= 0))


class TestBinomialGaussian(testslide.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)
        self.num_samples = 100
        self.dim_covariates = 5
        self.treatment_effect = 2.0
        self.std_covariates = 1.0

        # Create models
        self.outcome_model = DGPLinearGaussianOutcomeModel()
        self.propensity_model = DGPSigmoidGaussianPropensityModel()
        self.homo_noise_model = HomoGaussian(std_noise=0.5)
        self.hetero_noise_model = HeteroGaussian(
            std_noise=0.5, betaprime_a=2.0, betaprime_b=5.0
        )

    def test_binomial_gaussian_homo_initialization(self) -> None:
        """Test BinomialGaussian initialization with HomoGaussian noise."""
        data_gen = BinomialGaussian(
            outcome_model=self.outcome_model,
            propensity_model=self.propensity_model,
            outcome_noise_model=self.homo_noise_model,
            dim_covariates=self.dim_covariates,
            std_covariates=self.std_covariates,
        )

        # Check that attributes are set correctly
        self.assertEqual(data_gen.dim_covariates, self.dim_covariates)
        self.assertEqual(data_gen.std_covariates, self.std_covariates)
        self.assertIs(data_gen.outcome_model, self.outcome_model)
        self.assertIs(data_gen.propensity_model, self.propensity_model)
        self.assertIs(data_gen.outcome_noise_model, self.homo_noise_model)

    def test_binomial_gaussian_hetero_initialization(self) -> None:
        """Test BinomialGaussian initialization with HeteroGaussian noise."""
        data_gen = BinomialGaussian(
            outcome_model=self.outcome_model,
            propensity_model=self.propensity_model,
            outcome_noise_model=self.hetero_noise_model,
            dim_covariates=self.dim_covariates,
            std_covariates=self.std_covariates,
        )

        # Check that hetero noise model is correctly stored
        self.assertIs(data_gen.outcome_noise_model, self.hetero_noise_model)

    def test_binomial_gaussian_generate_basic(self) -> None:
        """Test basic data generation with BinomialGaussian."""
        data_gen = BinomialGaussian(
            outcome_model=self.outcome_model,
            propensity_model=self.propensity_model,
            outcome_noise_model=self.homo_noise_model,
            dim_covariates=self.dim_covariates,
            std_covariates=self.std_covariates,
        )

        data = data_gen.generate(
            num_samples=self.num_samples,
            treatment_effect=self.treatment_effect,
            seed=42,
        )

        # Check data shapes
        self.assertEqual(data.X.shape, (self.num_samples, self.dim_covariates))
        self.assertEqual(data.A.shape, (self.num_samples,))
        self.assertEqual(data.Y.shape, (self.num_samples,))

        # Check treatment assignments are binary
        self.assertTrue(np.all(np.isin(data.A, [0, 1])))

        # Check that we have both treated and control units (with high probability)
        self.assertGreater(np.sum(data.A), 0)
        self.assertLess(np.sum(data.A), self.num_samples)

    def test_binomial_gaussian_reproducibility(self) -> None:
        """Test that BinomialGaussian is reproducible with seed."""
        data_gen = BinomialGaussian(
            outcome_model=self.outcome_model,
            propensity_model=self.propensity_model,
            outcome_noise_model=self.homo_noise_model,
            dim_covariates=self.dim_covariates,
            std_covariates=self.std_covariates,
        )

        # Generate data twice with same seed
        data1 = data_gen.generate(
            num_samples=self.num_samples,
            treatment_effect=self.treatment_effect,
            seed=123,
        )
        data2 = data_gen.generate(
            num_samples=self.num_samples,
            treatment_effect=self.treatment_effect,
            seed=123,
        )

        # Check that data is identical
        self.assertTrue(np.allclose(data1.X, data2.X))
        self.assertTrue(np.array_equal(data1.A, data2.A))
        self.assertTrue(np.allclose(data1.Y, data2.Y))

    def test_binomial_gaussian_different_seeds(self) -> None:
        """Test that different seeds produce different data."""
        data_gen = BinomialGaussian(
            outcome_model=self.outcome_model,
            propensity_model=self.propensity_model,
            outcome_noise_model=self.homo_noise_model,
            dim_covariates=self.dim_covariates,
            std_covariates=self.std_covariates,
        )

        data1 = data_gen.generate(
            num_samples=self.num_samples,
            treatment_effect=self.treatment_effect,
            seed=123,
        )
        data2 = data_gen.generate(
            num_samples=self.num_samples,
            treatment_effect=self.treatment_effect,
            seed=456,
        )

        # Data should be different with different seeds
        self.assertFalse(np.allclose(data1.X, data2.X))
        self.assertFalse(np.array_equal(data1.A, data2.A))
        self.assertFalse(np.allclose(data1.Y, data2.Y))

    def test_binomial_gaussian_homo_vs_hetero_noise(self) -> None:
        """Test that HomoGaussian and HeteroGaussian produce different outcomes."""
        # Data gen with homoscedastic noise
        data_gen_homo = BinomialGaussian(
            outcome_model=self.outcome_model,
            propensity_model=self.propensity_model,
            outcome_noise_model=self.homo_noise_model,
            dim_covariates=self.dim_covariates,
            std_covariates=self.std_covariates,
        )

        # Data gen with heteroscedastic noise
        data_gen_hetero = BinomialGaussian(
            outcome_model=self.outcome_model,
            propensity_model=self.propensity_model,
            outcome_noise_model=self.hetero_noise_model,
            dim_covariates=self.dim_covariates,
            std_covariates=self.std_covariates,
        )

        # Generate data with same seed to isolate noise effect
        data_homo = data_gen_homo.generate(
            num_samples=self.num_samples,
            treatment_effect=self.treatment_effect,
            seed=42,
        )
        data_hetero = data_gen_hetero.generate(
            num_samples=self.num_samples,
            treatment_effect=self.treatment_effect,
            seed=42,
        )

        # Features should be similar (same generation process and seed)
        self.assertTrue(np.allclose(data_homo.X, data_hetero.X))

        # Treatment assignments should be similar (same propensity model)
        self.assertTrue(np.array_equal(data_homo.A, data_hetero.A))

        # Outcomes should differ due to different noise models
        self.assertFalse(np.allclose(data_homo.Y, data_hetero.Y))

    def test_binomial_gaussian_treatment_effect(self) -> None:
        """Test that treatment effect is properly embedded in the data."""
        data_gen = BinomialGaussian(
            outcome_model=self.outcome_model,
            propensity_model=self.propensity_model,
            outcome_noise_model=self.homo_noise_model,
            dim_covariates=self.dim_covariates,
            std_covariates=self.std_covariates,
        )

        # Generate data with different treatment effects
        data_small_effect = data_gen.generate(
            num_samples=self.num_samples,
            treatment_effect=0.5,
            seed=42,
        )
        data_large_effect = data_gen.generate(
            num_samples=self.num_samples,
            treatment_effect=5.0,
            seed=42,
        )

        # Features and treatment assignments should be identical
        self.assertTrue(np.allclose(data_small_effect.X, data_large_effect.X))
        self.assertTrue(np.array_equal(data_small_effect.A, data_large_effect.A))

        # Outcomes should differ
        self.assertFalse(np.allclose(data_small_effect.Y, data_large_effect.Y))

        # The difference should be related to the treatment effect difference
        treated_mask = data_small_effect.A == 1
        if np.sum(treated_mask) > 0:  # If we have treated units
            outcome_diff = np.mean(
                data_large_effect.Y[treated_mask] - data_small_effect.Y[treated_mask]
            )
            expected_diff = 5.0 - 0.5  # difference in treatment effects
            # Allow some tolerance due to noise
            self.assertAlmostEqual(outcome_diff, expected_diff, delta=1.0)

    def test_binomial_gaussian_covariate_dimensions(self) -> None:
        """Test that different covariate dimensions work correctly."""
        for dim in [1, 3, 10]:
            data_gen = BinomialGaussian(
                outcome_model=self.outcome_model,
                propensity_model=self.propensity_model,
                outcome_noise_model=self.homo_noise_model,
                dim_covariates=dim,
                std_covariates=self.std_covariates,
            )

            data = data_gen.generate(
                num_samples=self.num_samples,
                treatment_effect=self.treatment_effect,
                seed=42,
            )

            # Check that feature dimension is correct
            self.assertEqual(data.X.shape, (self.num_samples, dim))


class TestDataSplitting(testslide.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)
        self.num_samples = 100
        self.dim_covariates = 5

        # Create simple test data
        X = np.random.randn(self.num_samples, self.dim_covariates)
        A = np.random.binomial(1, 0.5, self.num_samples)
        Y = np.random.randn(self.num_samples)
        self.data = Dataset(X=X, A=A, Y=Y)

    def test_splitting_basic(self) -> None:
        """Test basic splitting functionality."""
        splitter = Splitting(train_ratio=0.6)
        split_data = splitter.decompose(self.data)

        # Check that we have training and inference data
        self.assertEqual(len(split_data.train.X), 60)
        self.assertEqual(len(split_data.inf.X), 40)

        # Check shapes are consistent
        self.assertEqual(split_data.train.X.shape[1], self.dim_covariates)
        self.assertEqual(split_data.inf.X.shape[1], self.dim_covariates)

    def test_splitting_with_stratification(self) -> None:
        """Test stratified splitting by treatment."""
        splitter = Splitting(train_ratio=0.6, stratify_by_treatment=True)
        split_data = splitter.decompose(self.data)

        # Check that both training and inference have both treatment groups
        self.assertGreater(np.sum(split_data.train.A), 0)
        self.assertLess(np.sum(split_data.train.A), len(split_data.train.A))
        self.assertGreater(np.sum(split_data.inf.A), 0)
        self.assertLess(np.sum(split_data.inf.A), len(split_data.inf.A))

    def test_splitting_no_stratification(self) -> None:
        """Test splitting without stratification."""
        splitter = Splitting(train_ratio=0.6, stratify_by_treatment=False)
        split_data = splitter.decompose(self.data)

        # Check sizes
        self.assertEqual(len(split_data.train.X), 60)
        self.assertEqual(len(split_data.inf.X), 40)

    def test_splitting_with_shuffle(self) -> None:
        """Test that shuffling produces different splits."""
        splitter1 = Splitting(train_ratio=0.6, shuffle=False)
        splitter2 = Splitting(train_ratio=0.6, shuffle=True)

        np.random.seed(42)
        split_data1 = splitter1.decompose(self.data)

        np.random.seed(42)
        split_data2 = splitter2.decompose(self.data)

        # With shuffle=False, should be identical to original order
        # With shuffle=True, should be different (with high probability)
        are_identical = np.array_equal(split_data1.train.X, split_data2.train.X)
        self.assertFalse(are_identical)  # Should be different with shuffle

    def test_splitting_edge_cases(self) -> None:
        """Test edge cases for splitting."""
        # Test extreme ratios
        splitter = Splitting(train_ratio=0.01)
        split_data = splitter.decompose(self.data)
        self.assertEqual(len(split_data.train.X), 1)
        self.assertEqual(len(split_data.inf.X), 99)

        splitter = Splitting(train_ratio=0.99)
        split_data = splitter.decompose(self.data)
        self.assertEqual(len(split_data.train.X), 99)
        self.assertEqual(len(split_data.inf.X), 1)

        # Test invalid ratios
        with self.assertRaises(AssertionError):
            Splitting(train_ratio=0.0)

        with self.assertRaises(AssertionError):
            Splitting(train_ratio=1.0)


class TestThinningFission(testslide.TestCase):
    def setUp(self) -> None:
        super().setUp()
        np.random.seed(42)
        self.num_samples = 100
        self.dim_covariates = 5

        # Create simple test data
        X = np.random.randn(self.num_samples, self.dim_covariates)
        A = np.random.binomial(1, 0.5, self.num_samples)
        Y = np.random.randn(self.num_samples)
        self.data = Dataset(X=X, A=A, Y=Y)

    def test_thinning_fission_outcome_only(self) -> None:
        """Test thinning fission with outcomes only."""
        fission = ThinningFission(
            std_outcome_noise=1.0,
            train_information_parameter=1.0,
            decompose_treatment=False,
        )
        split_data = fission.decompose(self.data)

        # Check that features and treatments are identical
        self.assertTrue(np.array_equal(split_data.train.X, self.data.X))
        self.assertTrue(np.array_equal(split_data.inf.X, self.data.X))
        self.assertTrue(np.array_equal(split_data.train.A, self.data.A))
        self.assertTrue(np.array_equal(split_data.inf.A, self.data.A))

        # Check that outcomes are different but related
        self.assertFalse(np.array_equal(split_data.train.Y, self.data.Y))
        self.assertFalse(np.array_equal(split_data.inf.Y, self.data.Y))
        self.assertFalse(np.array_equal(split_data.train.Y, split_data.inf.Y))

    def test_thinning_fission_with_treatment(self) -> None:
        """Test thinning fission with treatment decomposition."""
        fission = ThinningFission(
            std_outcome_noise=1.0,
            train_information_parameter=1.0,
            decompose_treatment=True,
            treatment_noise=0.3,
        )
        split_data = fission.decompose(self.data)

        # Check that features are identical
        self.assertTrue(np.array_equal(split_data.train.X, self.data.X))
        self.assertTrue(np.array_equal(split_data.inf.X, self.data.X))

        # Check that inference treatments are identical to original
        self.assertTrue(np.array_equal(split_data.inf.A, self.data.A))

        # Check that training treatments might be different (but not necessarily)
        # At least check they're valid binary treatments
        self.assertTrue(np.all(np.isin(split_data.train.A, [0, 1])))

    def test_thinning_fission_deterministic(self) -> None:
        """Test that thinning fission is deterministic given a seed."""
        fission = ThinningFission(
            std_outcome_noise=1.0, train_information_parameter=1.0
        )

        # Generate splits twice with same seed
        np.random.seed(123)
        split_data1 = fission.decompose(self.data)

        np.random.seed(123)
        split_data2 = fission.decompose(self.data)

        # Check that splits are identical
        self.assertTrue(np.allclose(split_data1.train.Y, split_data2.train.Y))
        self.assertTrue(np.allclose(split_data1.inf.Y, split_data2.inf.Y))

    def test_thinning_fission_information_parameter(self) -> None:
        """Test different information parameters."""
        # Higher parameter should give training data more information
        fission_high = ThinningFission(
            std_outcome_noise=1.0, train_information_parameter=2.0
        )
        fission_low = ThinningFission(
            std_outcome_noise=1.0, train_information_parameter=0.5
        )

        np.random.seed(42)
        split_high = fission_high.decompose(self.data)

        np.random.seed(42)
        split_low = fission_low.decompose(self.data)

        # Check that the splits are different
        self.assertFalse(np.allclose(split_high.train.Y, split_low.train.Y))
        self.assertFalse(np.allclose(split_high.inf.Y, split_low.inf.Y))
