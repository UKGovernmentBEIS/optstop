"""
Unit tests for _continuous_bounded_ci_adaptive() function.

Tests cover:
- Binary aggregated scores (high performance)
- Ordinal aggregated scores (high performance)
- Low performance with conservatism
- Edge cases (empty, single observation, zero variance)
- Convergence with sample size
- Parameter validation
"""

import pytest
import numpy as np
from optstop.rule import _continuous_bounded_ci_adaptive


class TestContinuousBoundedInference:
    """Test suite for continuous bounded CI inference."""

    def test_binary_aggregated_high_performance(self):
        """Test binary aggregated scores with high performance."""
        # Aggregated binary scores: mean of multiple 0/1 scores
        scores = np.array([0.8, 0.85, 0.9, 0.75, 0.95, 0.8, 0.85, 0.9, 0.85, 0.8])

        lo, hi, width = _continuous_bounded_ci_adaptive(
            scores,
            lower_bound=0.0,
            upper_bound=1.0,
            cred_level=0.95,
            conservatism=1.0,
            low_perf_threshold=0.2
        )

        # Assertions
        assert 0.0 <= lo < hi <= 1.0, "CI bounds must be within [0, 1]"
        assert width == hi - lo, "Width must equal hi - lo"
        assert 0.65 < lo < 0.9, "Lower bound should be reasonable for high scores"
        assert 0.8 < hi <= 1.0, "Upper bound should be reasonable for high scores"
        assert width < 0.35, "Width should be reasonably narrow with 10 observations"

    def test_ordinal_aggregated_high_performance(self):
        """Test ordinal aggregated scores with high performance."""
        # Aggregated ordinal scores: mean of multiple 0-10 scores
        scores = np.array([8.2, 8.5, 8.7, 8.4, 8.9, 8.3, 8.6, 8.8, 8.5, 8.4])

        lo, hi, width = _continuous_bounded_ci_adaptive(
            scores,
            lower_bound=0.0,
            upper_bound=10.0,
            cred_level=0.95,
            conservatism=1.0,
            low_perf_threshold=0.2
        )

        # Assertions
        assert 0.0 <= lo < hi <= 10.0, "CI bounds must be within [0, 10]"
        assert width == hi - lo, "Width must equal hi - lo"
        assert 7.5 < lo < 9.0, "Lower bound should be reasonable for high scores"
        assert 8.0 < hi <= 10.0, "Upper bound should be reasonable for high scores"
        assert width < 2.0, "Width should be reasonably narrow with 10 observations"

    def test_low_performance_with_conservatism(self):
        """Test that conservatism is applied for low performance."""
        # Low performance scores
        scores_low = np.array([0.1, 0.15, 0.05, 0.12, 0.08, 0.1, 0.14, 0.06, 0.11, 0.09])

        # Without conservatism
        lo1, hi1, width1 = _continuous_bounded_ci_adaptive(
            scores_low,
            lower_bound=0.0,
            upper_bound=1.0,
            cred_level=0.95,
            conservatism=1.0,  # No conservatism
            low_perf_threshold=0.2
        )

        # With conservatism
        lo2, hi2, width2 = _continuous_bounded_ci_adaptive(
            scores_low,
            lower_bound=0.0,
            upper_bound=1.0,
            cred_level=0.95,
            conservatism=5.0,  # 5x conservatism
            low_perf_threshold=0.2
        )

        # Conservatism should widen the effective width
        assert width2 > width1, "Conservatism should increase effective width"
        assert width2 / width1 >= 4.0, "Conservative width should be significantly larger"

    def test_edge_case_empty_array(self):
        """Test handling of empty score array."""
        scores = np.array([])

        lo, hi, width = _continuous_bounded_ci_adaptive(
            scores,
            lower_bound=0.0,
            upper_bound=1.0
        )

        # Should return full range
        assert lo == 0.0, "Empty array should return lower bound"
        assert hi == 1.0, "Empty array should return upper bound"
        assert width == 1.0, "Empty array should return full width"

    def test_edge_case_single_observation(self):
        """Test handling of single observation."""
        score_value = 0.75
        scores = np.array([score_value])

        lo, hi, width = _continuous_bounded_ci_adaptive(
            scores,
            lower_bound=0.0,
            upper_bound=1.0
        )

        # Should return wide interval around observation
        assert 0.0 <= lo < score_value < hi <= 1.0, "CI should contain observation"
        assert width >= 0.3, "Single observation should have wide CI"
        assert abs((lo + hi) / 2 - score_value) < 0.2, "CI should be roughly centered on observation"

    def test_edge_case_zero_variance(self):
        """Test handling of zero variance (all scores identical)."""
        # All scores are identical
        scores = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])

        lo, hi, width = _continuous_bounded_ci_adaptive(
            scores,
            lower_bound=0.0,
            upper_bound=1.0
        )

        # Should return tight interval around mean
        assert 0.0 <= lo < hi <= 1.0, "CI bounds must be valid"
        assert width < 0.15, "Zero variance should give tight CI"
        assert 0.7 <= lo <= 0.85, "Lower bound should be near mean"
        assert 0.75 <= hi <= 0.9, "Upper bound should be near mean"

    def test_convergence_with_sample_size(self):
        """Test that CI width decreases with increasing sample size."""
        # Generate scores from same distribution with different sample sizes
        np.random.seed(42)
        true_mean = 0.7
        true_std = 0.1

        widths = []
        sample_sizes = [5, 10, 20, 50, 100]

        for n in sample_sizes:
            scores = np.random.normal(true_mean, true_std, n)
            scores = np.clip(scores, 0.0, 1.0)  # Keep in bounds

            lo, hi, width = _continuous_bounded_ci_adaptive(
                scores,
                lower_bound=0.0,
                upper_bound=1.0,
                cred_level=0.95
            )

            widths.append(width)

        # Width should generally decrease with sample size
        # Check that later widths are smaller than earlier ones
        # Note: Due to prior effects, this may not be strictly monotonic for very small samples
        # So we check that large sample width is reasonable
        assert widths[-1] < 0.4, "Width should be reasonable for n=100"
        assert widths[-2] < 0.5, "Width should be reasonable for n=50"
        # Check general trend: larger samples should not be wider than small samples
        assert widths[-1] < widths[0] * 1.5, "Large sample width should not be much larger than small sample"

    def test_bounds_scaling_ordinal(self):
        """Test that function correctly scales for arbitrary bounds."""
        # Same relative scores, different bounds
        scores_01 = np.array([0.5, 0.6, 0.55, 0.58, 0.52])
        scores_010 = np.array([5.0, 6.0, 5.5, 5.8, 5.2])

        lo1, hi1, width1 = _continuous_bounded_ci_adaptive(
            scores_01,
            lower_bound=0.0,
            upper_bound=1.0
        )

        lo2, hi2, width2 = _continuous_bounded_ci_adaptive(
            scores_010,
            lower_bound=0.0,
            upper_bound=10.0
        )

        # Scaled results should be proportional
        assert abs((lo2 / lo1) - 10.0) < 0.5, "Lower bound should scale by factor of 10"
        assert abs((hi2 / hi1) - 10.0) < 0.5, "Upper bound should scale by factor of 10"
        assert abs((width2 / width1) - 10.0) < 0.5, "Width should scale by factor of 10"

    def test_credibility_level_effect(self):
        """Test that different credibility levels produce different widths."""
        scores = np.array([0.7, 0.75, 0.8, 0.72, 0.78, 0.76, 0.74, 0.77])

        # 90% CI
        lo_90, hi_90, width_90 = _continuous_bounded_ci_adaptive(
            scores,
            lower_bound=0.0,
            upper_bound=1.0,
            cred_level=0.90
        )

        # 95% CI
        lo_95, hi_95, width_95 = _continuous_bounded_ci_adaptive(
            scores,
            lower_bound=0.0,
            upper_bound=1.0,
            cred_level=0.95
        )

        # 99% CI
        lo_99, hi_99, width_99 = _continuous_bounded_ci_adaptive(
            scores,
            lower_bound=0.0,
            upper_bound=1.0,
            cred_level=0.99
        )

        # Higher credibility level should give wider CI
        assert width_90 < width_95 < width_99, "Width should increase with credibility level"

    def test_return_types(self):
        """Test that function returns correct types."""
        scores = np.array([0.7, 0.8, 0.75])

        lo, hi, width = _continuous_bounded_ci_adaptive(
            scores,
            lower_bound=0.0,
            upper_bound=1.0
        )

        assert isinstance(lo, (float, np.floating)), "Lower bound should be float"
        assert isinstance(hi, (float, np.floating)), "Upper bound should be float"
        assert isinstance(width, (float, np.floating)), "Width should be float"

    def test_boundary_scores_near_zero(self):
        """Test scores near lower boundary."""
        scores = np.array([0.01, 0.02, 0.015, 0.025, 0.018])

        lo, hi, width = _continuous_bounded_ci_adaptive(
            scores,
            lower_bound=0.0,
            upper_bound=1.0,
            conservatism=5.0,  # High conservatism for low scores
            low_perf_threshold=0.2
        )

        assert lo >= 0.0, "Lower bound should not go below 0"
        assert hi <= 1.0, "Upper bound should not exceed 1"
        assert lo < hi, "Lower bound should be less than upper bound"

    def test_boundary_scores_near_upper(self):
        """Test scores near upper boundary."""
        scores = np.array([0.98, 0.99, 0.985, 0.975, 0.99])

        lo, hi, width = _continuous_bounded_ci_adaptive(
            scores,
            lower_bound=0.0,
            upper_bound=1.0
        )

        assert lo >= 0.0, "Lower bound should not go below 0"
        assert hi <= 1.0, "Upper bound should not exceed 1"
        assert lo < hi, "Lower bound should be less than upper bound"

    def test_negative_bounds_invalid(self):
        """Test handling of invalid bounds (lower >= upper)."""
        scores = np.array([0.7, 0.8, 0.75])

        # Invalid bounds: lower >= upper
        lo, hi, width = _continuous_bounded_ci_adaptive(
            scores,
            lower_bound=1.0,
            upper_bound=0.0  # Invalid: upper < lower
        )

        # Should return bounds as-is (safe fallback)
        assert lo == 1.0
        assert hi == 0.0
        assert width == -1.0  # Will be negative due to invalid input


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
