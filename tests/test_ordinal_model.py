"""
Unit tests for OrderedLogistic model with entropy-based stopping.

Tests cover:
1. Model creation and structure
2. Category probability computation
3. Entropy computation
4. Full entropy CI computation
5. Threshold probability computation
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '/home/ubuntu/optstop')

from optstop.ordinal_model import (
    _create_orderedlogistic_model,
    _compute_category_probabilities,
    _compute_entropy,
    _ordinal_entropy_ci_adaptive,
    _compute_threshold_probability
)


class TestCreateOrderedLogisticModel:
    """Tests for _create_orderedlogistic_model()"""

    def test_model_creation_basic(self):
        """Test basic model creation"""
        model = _create_orderedlogistic_model(n_categories=11, n_items=10)
        assert model is not None
        assert 'mu_group' in model.named_vars
        assert 'sigma_group' in model.named_vars
        assert 'cutpoints' in model.named_vars
        assert 'eta' in model.named_vars

    def test_model_cutpoints_count(self):
        """Test correct number of cutpoints (n_categories - 1)"""
        model = _create_orderedlogistic_model(n_categories=11, n_items=5)
        # 11 categories → 10 cutpoints
        cutpoints_shape = model.named_vars['cutpoints'].eval().shape
        assert cutpoints_shape == (10,)

    def test_model_different_categories(self):
        """Test model with different category counts"""
        for n_cat in [5, 7, 11]:
            model = _create_orderedlogistic_model(n_categories=n_cat, n_items=10)
            cutpoints_shape = model.named_vars['cutpoints'].eval().shape
            assert cutpoints_shape == (n_cat - 1,)

    def test_model_prior_customization(self):
        """Test custom prior parameters"""
        model = _create_orderedlogistic_model(
            n_categories=11,
            n_items=10,
            mu_group_prior=(1.0, 3.0),
            sigma_group_prior=2.0
        )
        assert model is not None


class TestComputeCategoryProbabilities:
    """Tests for _compute_category_probabilities()"""

    def test_probability_computation_shape(self):
        """Test output shape is correct"""
        n_samples = 100
        n_items = 5
        n_categories = 11
        n_cutpoints = n_categories - 1

        eta = np.random.randn(n_samples, n_items)
        cutpoints = np.sort(np.random.randn(n_samples, n_cutpoints), axis=1)

        probs = _compute_category_probabilities(eta, cutpoints, n_categories)

        assert probs.shape == (n_samples, n_items, n_categories)

    def test_probabilities_sum_to_one(self):
        """Test probabilities sum to 1 for each sample and item"""
        n_samples = 50
        n_items = 3
        n_categories = 5

        eta = np.random.randn(n_samples, n_items)
        cutpoints = np.sort(np.random.randn(n_samples, n_categories - 1), axis=1)

        probs = _compute_category_probabilities(eta, cutpoints, n_categories)

        # Check sums (should be 1.0 for each sample and item)
        sums = probs.sum(axis=2)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_probabilities_non_negative(self):
        """Test all probabilities are non-negative"""
        n_samples = 50
        n_items = 3
        n_categories = 5

        eta = np.random.randn(n_samples, n_items)
        cutpoints = np.sort(np.random.randn(n_samples, n_categories - 1), axis=1)

        probs = _compute_category_probabilities(eta, cutpoints, n_categories)

        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_single_item_case(self):
        """Test with single item (1D eta)"""
        n_samples = 50
        n_categories = 11

        eta = np.random.randn(n_samples)  # 1D
        cutpoints = np.sort(np.random.randn(n_samples, n_categories - 1), axis=1)

        probs = _compute_category_probabilities(eta, cutpoints, n_categories)

        assert probs.shape == (n_samples, 1, n_categories)

    def test_extreme_eta_values(self):
        """Test with extreme eta values (very high or low ability)"""
        n_samples = 10
        n_categories = 11

        # Very high ability → should concentrate on high categories
        eta_high = np.full((n_samples, 1), 10.0)
        cutpoints = np.sort(np.random.randn(n_samples, n_categories - 1), axis=1)

        probs_high = _compute_category_probabilities(eta_high, cutpoints, n_categories)

        # Most probability should be in upper categories
        assert probs_high[:, :, -3:].sum() > 0.8

        # Very low ability → should concentrate on low categories
        eta_low = np.full((n_samples, 1), -10.0)
        probs_low = _compute_category_probabilities(eta_low, cutpoints, n_categories)

        # Most probability should be in lower categories
        assert probs_low[:, :, :3].sum() > 0.8


class TestComputeEntropy:
    """Tests for _compute_entropy()"""

    def test_entropy_uniform_distribution(self):
        """Test entropy is maximum for uniform distribution"""
        n_categories = 11
        probs = np.ones((1, n_categories)) / n_categories

        entropy = _compute_entropy(probs)

        # Maximum entropy for uniform distribution
        max_entropy = np.log2(n_categories)
        np.testing.assert_allclose(entropy, max_entropy, atol=0.01)

    def test_entropy_deterministic_distribution(self):
        """Test entropy is ~0 for deterministic distribution"""
        n_categories = 11
        probs = np.zeros((1, n_categories))
        probs[0, 5] = 1.0  # All probability in one category

        entropy = _compute_entropy(probs)

        # Entropy should be near 0 (with epsilon for numerical stability)
        assert entropy[0] < 0.01

    def test_entropy_peaked_distribution(self):
        """Test entropy for peaked distribution (high confidence)"""
        probs = np.array([[0.9, 0.05, 0.03, 0.02, 0, 0, 0, 0, 0, 0, 0]])

        entropy = _compute_entropy(probs)

        # Entropy should be low (< 1.0 for peaked distribution)
        assert entropy[0] < 1.0

    def test_entropy_bimodal_distribution(self):
        """Test entropy for bimodal distribution"""
        probs = np.array([[0.45, 0, 0, 0, 0, 0, 0, 0, 0.45, 0.1, 0]])

        entropy = _compute_entropy(probs)

        # Entropy should be moderate (bimodal has more uncertainty than unimodal)
        assert 1.0 < entropy[0] < 2.0

    def test_entropy_batch_processing(self):
        """Test entropy computation on batch of distributions"""
        n_samples = 100
        n_categories = 11

        # Random probability distributions
        probs = np.random.dirichlet(np.ones(n_categories), size=n_samples)

        entropy = _compute_entropy(probs)

        assert entropy.shape == (n_samples,)
        assert np.all(entropy >= 0)
        assert np.all(entropy <= np.log2(n_categories) + 0.1)  # Allow small numerical error

    def test_entropy_zero_probability_handling(self):
        """Test entropy computation with zero probabilities"""
        probs = np.array([[0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        entropy = _compute_entropy(probs)

        # Should handle zeros gracefully (no NaN or inf)
        assert np.isfinite(entropy[0])
        np.testing.assert_allclose(entropy, 1.0, atol=0.01)  # -0.5*log2(0.5) - 0.5*log2(0.5) = 1


class TestOrdinalEntropyCIAdaptive:
    """Tests for _ordinal_entropy_ci_adaptive()"""

    def test_basic_functionality(self):
        """Test basic entropy CI computation"""
        scores = np.array([5, 6, 7, 7, 8, 7, 6, 7, 7, 8])

        lo, hi, width, diagnostics = _ordinal_entropy_ci_adaptive(
            scores,
            ordinal_max_score=10,
            cred_level=0.95,
            n_samples=500,
            n_tune=250
        )

        # Check outputs
        assert 0 <= lo < hi
        assert width == pytest.approx(hi - lo, abs=1e-6)
        assert 'entropy_median' in diagnostics
        assert 'entropy_samples' in diagnostics
        assert 'probs_median' in diagnostics

    def test_ci_bounds_ordering(self):
        """Test CI bounds are properly ordered"""
        scores = np.array([3, 4, 5, 6, 7, 8, 9])

        lo, hi, width, diagnostics = _ordinal_entropy_ci_adaptive(
            scores,
            ordinal_max_score=10,
            n_samples=500,
            n_tune=250
        )

        assert lo < hi
        assert lo >= 0
        assert hi <= np.log2(11) + 0.5  # Maximum entropy for 11 categories (with buffer)

    def test_peaked_vs_diffuse_scores(self):
        """Test entropy is lower for peaked scores vs diffuse scores"""
        # Peaked scores (low entropy)
        scores_peaked = np.array([7] * 20)

        lo_peaked, hi_peaked, width_peaked, diag_peaked = _ordinal_entropy_ci_adaptive(
            scores_peaked,
            ordinal_max_score=10,
            n_samples=500,
            n_tune=250
        )

        # Diffuse scores (high entropy)
        scores_diffuse = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2)

        lo_diffuse, hi_diffuse, width_diffuse, diag_diffuse = _ordinal_entropy_ci_adaptive(
            scores_diffuse,
            ordinal_max_score=10,
            n_samples=500,
            n_tune=250
        )

        # Peaked should have lower median entropy than diffuse
        assert diag_peaked['entropy_median'] < diag_diffuse['entropy_median']

    def test_conservatism_effect(self):
        """Test conservatism increases effective width for low performance"""
        scores = np.array([0, 1, 1, 2, 2, 3])  # Low performance

        # No conservatism
        _, _, width_no_cons, _ = _ordinal_entropy_ci_adaptive(
            scores,
            ordinal_max_score=10,
            conservatism=1.0,
            low_perf_threshold=0.5,
            n_samples=500,
            n_tune=250
        )

        # High conservatism
        _, _, width_cons, _ = _ordinal_entropy_ci_adaptive(
            scores,
            ordinal_max_score=10,
            conservatism=2.0,
            low_perf_threshold=0.5,
            n_samples=500,
            n_tune=250
        )

        # Conservative should have wider effective width
        assert width_cons > width_no_cons

    def test_invalid_scores_rejection(self):
        """Test invalid scores are rejected"""
        # Scores out of range
        with pytest.raises(ValueError, match="Scores must be in"):
            _ordinal_entropy_ci_adaptive(
                np.array([0, 5, 12, 8]),
                ordinal_max_score=10
            )

        # Negative scores
        with pytest.raises(ValueError, match="Scores must be in"):
            _ordinal_entropy_ci_adaptive(
                np.array([-1, 5, 7, 8]),
                ordinal_max_score=10
            )

    def test_empty_scores_rejection(self):
        """Test empty scores array is rejected"""
        with pytest.raises(ValueError, match="Cannot compute entropy CI with zero scores"):
            _ordinal_entropy_ci_adaptive(
                np.array([]),
                ordinal_max_score=10
            )

    def test_model_caching(self):
        """Test model caching functionality"""
        scores1 = np.array([5, 6, 7, 7, 8])
        scores2 = np.array([5, 6, 7, 7, 8, 7, 6, 7, 7, 8])

        model_cache = {}

        # First call (creates model)
        lo1, hi1, width1, diag1 = _ordinal_entropy_ci_adaptive(
            scores1,
            ordinal_max_score=10,
            n_samples=500,
            n_tune=250,
            model_cache=model_cache
        )

        assert 'model' in model_cache
        assert model_cache['n_items_last'] == len(scores1)

        # Second call (reuses model)
        lo2, hi2, width2, diag2 = _ordinal_entropy_ci_adaptive(
            scores2,
            ordinal_max_score=10,
            n_samples=500,
            n_tune=250,
            model_cache=model_cache
        )

        assert model_cache['n_items_last'] == len(scores2)

    def test_convergence_diagnostics(self):
        """Test convergence diagnostics are returned"""
        scores = np.array([5, 6, 7, 7, 8, 7, 6, 7, 7, 8])

        lo, hi, width, diagnostics = _ordinal_entropy_ci_adaptive(
            scores,
            ordinal_max_score=10,
            n_samples=1000,
            n_tune=500
        )

        assert 'rhat_max' in diagnostics
        assert 'ess_min' in diagnostics
        assert 'trace' in diagnostics

        # R-hat should be close to 1 for convergence
        assert diagnostics['rhat_max'] < 1.1


class TestComputeThresholdProbability:
    """Tests for _compute_threshold_probability()"""

    def test_threshold_probability_basic(self):
        """Test basic threshold probability computation"""
        probs = np.array([[0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0, 0, 0]])

        p_above = _compute_threshold_probability(probs, threshold=7)

        # P(Score ≥ 7) = sum of probs for categories 7, 8, 9, 10
        expected = 0.02 + 0 + 0 + 0
        np.testing.assert_allclose(p_above, expected, atol=1e-6)

    def test_threshold_probability_all_above(self):
        """Test when all probability is above threshold"""
        probs = np.array([[0, 0, 0, 0, 0, 0, 0, 0.5, 0.3, 0.2, 0]])

        p_above = _compute_threshold_probability(probs, threshold=7)

        # All non-zero probability is in categories 7, 8, 9
        np.testing.assert_allclose(p_above, 1.0, atol=1e-6)

    def test_threshold_probability_none_above(self):
        """Test when no probability is above threshold"""
        probs = np.array([[0.3, 0.3, 0.2, 0.2, 0, 0, 0, 0, 0, 0, 0]])

        p_above = _compute_threshold_probability(probs, threshold=7)

        np.testing.assert_allclose(p_above, 0.0, atol=1e-6)

    def test_threshold_probability_batch(self):
        """Test threshold probability on batch of distributions"""
        n_samples = 100
        n_categories = 11

        probs = np.random.dirichlet(np.ones(n_categories), size=n_samples)

        p_above = _compute_threshold_probability(probs, threshold=5)

        assert p_above.shape == (n_samples,)
        assert np.all(p_above >= 0)
        assert np.all(p_above <= 1)

    def test_threshold_at_boundaries(self):
        """Test threshold at category boundaries"""
        probs = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0]])

        # Threshold = 0 → all categories
        p_above_0 = _compute_threshold_probability(probs, threshold=0)
        np.testing.assert_allclose(p_above_0, 1.0, atol=1e-6)

        # Threshold = 10 → only last category
        p_above_10 = _compute_threshold_probability(probs, threshold=10)
        np.testing.assert_allclose(p_above_10, 0.0, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
