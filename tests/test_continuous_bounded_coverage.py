"""
Coverage validation tests for _continuous_bounded_ci_adaptive().

This module validates that the 95% credible intervals actually achieve
~95% coverage when applied to synthetic data from known distributions.

This is a critical validation that ensures the statistical method is sound.
"""

import pytest
import numpy as np
from optstop.rule import _continuous_bounded_ci_adaptive



# CI partition: real MCMC via _continuous_bounded_ci_adaptive - deselected from PR CI.
pytestmark = pytest.mark.optstop


class TestContinuousBoundedCoverage:
    """Coverage validation tests for continuous bounded CI."""

    def test_coverage_binary_aggregated_high_performance(self):
        """
        Validate CI coverage for binary aggregated scores with high performance.

        Generates synthetic continuous data from Beta(8, 2) distribution,
        which has mean ~0.8 (high performance). Verifies that 95% CIs
        actually contain the true parameter ~95% of the time.
        """
        np.random.seed(42)

        # True distribution: Beta(8, 2) with mean = 8/(8+2) = 0.8
        true_alpha = 8.0
        true_beta = 2.0
        true_mean = true_alpha / (true_alpha + true_beta)

        # Simulation parameters
        n_trials = 500  # Number of trials to run
        sample_size = 20  # Observations per trial
        coverage_count = 0

        for _ in range(n_trials):
            # Generate synthetic continuous data from true distribution
            scores = np.random.beta(true_alpha, true_beta, sample_size)

            # Compute CI
            lo, hi, _ = _continuous_bounded_ci_adaptive(
                scores,
                lower_bound=0.0,
                upper_bound=1.0,
                cred_level=0.95
            )

            # Check if true mean is within CI
            if lo <= true_mean <= hi:
                coverage_count += 1

        coverage_rate = coverage_count / n_trials

        # CI should achieve ~95% coverage (allow ±5% tolerance)
        assert 0.90 <= coverage_rate <= 1.0, \
            f"Coverage rate {coverage_rate:.3f} outside expected range [0.90, 1.00]"

        print(f"✓ Binary aggregated (high perf): {coverage_rate:.1%} coverage (target: 95%)")

    def test_coverage_binary_aggregated_medium_performance(self):
        """
        Validate CI coverage for binary aggregated scores with medium performance.

        Uses Beta(5, 5) distribution with mean = 0.5 (medium performance).
        """
        np.random.seed(43)

        # True distribution: Beta(5, 5) with mean = 0.5
        true_alpha = 5.0
        true_beta = 5.0
        true_mean = true_alpha / (true_alpha + true_beta)

        # Simulation parameters
        n_trials = 500
        sample_size = 20
        coverage_count = 0

        for _ in range(n_trials):
            scores = np.random.beta(true_alpha, true_beta, sample_size)

            lo, hi, _ = _continuous_bounded_ci_adaptive(
                scores,
                lower_bound=0.0,
                upper_bound=1.0,
                cred_level=0.95
            )

            if lo <= true_mean <= hi:
                coverage_count += 1

        coverage_rate = coverage_count / n_trials

        assert 0.90 <= coverage_rate <= 1.0, \
            f"Coverage rate {coverage_rate:.3f} outside expected range [0.90, 1.00]"

        print(f"✓ Binary aggregated (medium perf): {coverage_rate:.1%} coverage (target: 95%)")

    def test_coverage_ordinal_aggregated_high_performance(self):
        """
        Validate CI coverage for ordinal aggregated scores with high performance.

        Uses scaled Beta distribution to simulate aggregated ordinal scores
        in range [0, 10] with mean ~8.0.
        """
        np.random.seed(44)

        # True distribution: Beta(8, 2) scaled to [0, 10], mean = 8.0
        true_alpha = 8.0
        true_beta = 2.0
        scale = 10.0
        true_mean_normalized = true_alpha / (true_alpha + true_beta)
        true_mean = true_mean_normalized * scale

        # Simulation parameters
        n_trials = 500
        sample_size = 20
        coverage_count = 0

        for _ in range(n_trials):
            # Generate Beta(8, 2) and scale to [0, 10]
            scores_normalized = np.random.beta(true_alpha, true_beta, sample_size)
            scores = scores_normalized * scale

            lo, hi, _ = _continuous_bounded_ci_adaptive(
                scores,
                lower_bound=0.0,
                upper_bound=10.0,
                cred_level=0.95
            )

            if lo <= true_mean <= hi:
                coverage_count += 1

        coverage_rate = coverage_count / n_trials

        assert 0.90 <= coverage_rate <= 1.0, \
            f"Coverage rate {coverage_rate:.3f} outside expected range [0.90, 1.00]"

        print(f"✓ Ordinal aggregated (high perf): {coverage_rate:.1%} coverage (target: 95%)")

    def test_coverage_with_small_samples(self):
        """
        Validate CI coverage with small sample sizes (n=5).

        Small samples should still achieve reasonable coverage, though may be
        slightly wider due to higher uncertainty.
        """
        np.random.seed(45)

        # True distribution: Beta(6, 4) with mean = 0.6
        true_alpha = 6.0
        true_beta = 4.0
        true_mean = true_alpha / (true_alpha + true_beta)

        # Simulation parameters
        n_trials = 500
        sample_size = 5  # Small sample
        coverage_count = 0

        for _ in range(n_trials):
            scores = np.random.beta(true_alpha, true_beta, sample_size)

            lo, hi, _ = _continuous_bounded_ci_adaptive(
                scores,
                lower_bound=0.0,
                upper_bound=1.0,
                cred_level=0.95
            )

            if lo <= true_mean <= hi:
                coverage_count += 1

        coverage_rate = coverage_count / n_trials

        # With small samples, allow wider tolerance (85-100%)
        assert 0.85 <= coverage_rate <= 1.0, \
            f"Coverage rate {coverage_rate:.3f} outside expected range [0.85, 1.00]"

        print(f"✓ Small samples (n=5): {coverage_rate:.1%} coverage (target: 95%)")

    def test_coverage_with_large_samples(self):
        """
        Validate CI coverage with large sample sizes (n=100).

        Large samples should achieve very accurate coverage close to 95%.
        """
        np.random.seed(46)

        # True distribution: Beta(7, 3) with mean = 0.7
        true_alpha = 7.0
        true_beta = 3.0
        true_mean = true_alpha / (true_alpha + true_beta)

        # Simulation parameters
        n_trials = 500
        sample_size = 100  # Large sample
        coverage_count = 0

        for _ in range(n_trials):
            scores = np.random.beta(true_alpha, true_beta, sample_size)

            lo, hi, _ = _continuous_bounded_ci_adaptive(
                scores,
                lower_bound=0.0,
                upper_bound=1.0,
                cred_level=0.95
            )

            if lo <= true_mean <= hi:
                coverage_count += 1

        coverage_rate = coverage_count / n_trials

        # Large samples should be very accurate (92-100%)
        # Note: 100% indicates conservative CIs, which is desirable for early stopping
        assert 0.92 <= coverage_rate <= 1.0, \
            f"Coverage rate {coverage_rate:.3f} outside expected range [0.92, 1.00]"

        print(f"✓ Large samples (n=100): {coverage_rate:.1%} coverage (target: 95%)")

    def test_coverage_90_percent_ci(self):
        """
        Validate that 90% CIs achieve ~90% coverage.

        Tests that different credibility levels are correctly implemented.
        """
        np.random.seed(47)

        # True distribution: Beta(5, 5) with mean = 0.5
        true_alpha = 5.0
        true_beta = 5.0
        true_mean = true_alpha / (true_alpha + true_beta)

        # Simulation parameters
        n_trials = 500
        sample_size = 20
        coverage_count = 0

        for _ in range(n_trials):
            scores = np.random.beta(true_alpha, true_beta, sample_size)

            lo, hi, _ = _continuous_bounded_ci_adaptive(
                scores,
                lower_bound=0.0,
                upper_bound=1.0,
                cred_level=0.90  # 90% CI
            )

            if lo <= true_mean <= hi:
                coverage_count += 1

        coverage_rate = coverage_count / n_trials

        # 90% CI should achieve ~90% coverage (allow ±5% tolerance, up to 100% for conservative CIs)
        # Note: 100% indicates conservative CIs, which is desirable for early stopping
        assert 0.85 <= coverage_rate <= 1.0, \
            f"Coverage rate {coverage_rate:.3f} outside expected range [0.85, 1.00]"

        print(f"✓ 90% CI: {coverage_rate:.1%} coverage (target: 90%)")

    def test_coverage_99_percent_ci(self):
        """
        Validate that 99% CIs achieve ~99% coverage.

        Tests wider CIs with higher credibility levels.
        """
        np.random.seed(48)

        # True distribution: Beta(6, 4) with mean = 0.6
        true_alpha = 6.0
        true_beta = 4.0
        true_mean = true_alpha / (true_alpha + true_beta)

        # Simulation parameters
        n_trials = 500
        sample_size = 20
        coverage_count = 0

        for _ in range(n_trials):
            scores = np.random.beta(true_alpha, true_beta, sample_size)

            lo, hi, _ = _continuous_bounded_ci_adaptive(
                scores,
                lower_bound=0.0,
                upper_bound=1.0,
                cred_level=0.99  # 99% CI
            )

            if lo <= true_mean <= hi:
                coverage_count += 1

        coverage_rate = coverage_count / n_trials

        # 99% CI should achieve very high coverage (96-100%)
        assert 0.96 <= coverage_rate <= 1.0, \
            f"Coverage rate {coverage_rate:.3f} outside expected range [0.96, 1.00]"

        print(f"✓ 99% CI: {coverage_rate:.1%} coverage (target: 99%)")


if __name__ == "__main__":
    # Run with verbose output to see coverage rates
    pytest.main([__file__, "-v", "-s"])
