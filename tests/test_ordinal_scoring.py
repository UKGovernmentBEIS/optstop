"""
Tests for ordinal scoring functionality in optstop.

This module tests the ordinal utilities including:
- Bootstrap CI computation
- Score type determination
- Score validation
- Integration with main optimal stopping functions
"""

import pytest
import numpy as np
import pandas as pd
from optstop.ordinal_utils import (
    _ordinal_ci_adaptive,
    determine_score_type,
    validate_ordinal_scores
)


class TestDetermineScoreType:
    """Tests for score type determination."""

    def test_binary_default(self):
        """Test that binary is returned when no ordinal_tasks specified."""
        assert determine_score_type("subject1-task1", None) == 'binary'
        assert determine_score_type("subject1-task1", []) == 'binary'

    def test_ordinal_single_match(self):
        """Test ordinal detection with single pattern match."""
        assert determine_score_type("subject1-likert_scale", ['likert']) == 'ordinal'
        assert determine_score_type("1-2_rating_task", ['rating']) == 'ordinal'

    def test_ordinal_multiple_patterns(self):
        """Test ordinal detection with multiple patterns."""
        ordinal_tasks = ['likert', 'rating', 'difficulty']
        assert determine_score_type("subject1-likert", ordinal_tasks) == 'ordinal'
        assert determine_score_type("subject1-rating_task", ordinal_tasks) == 'ordinal'
        assert determine_score_type("subject1-difficulty_scale", ordinal_tasks) == 'ordinal'

    def test_binary_no_match(self):
        """Test binary when no pattern matches."""
        ordinal_tasks = ['likert', 'rating']
        assert determine_score_type("subject1-accuracy", ordinal_tasks) == 'binary'
        assert determine_score_type("1-1", ordinal_tasks) == 'binary'

    def test_case_insensitive(self):
        """Test that matching is case-insensitive."""
        assert determine_score_type("Subject1-LIKERT", ['likert']) == 'ordinal'
        assert determine_score_type("subject1-likert", ['LIKERT']) == 'ordinal'
        assert determine_score_type("SuBjEcT1-LiKeRt", ['LiKeRt']) == 'ordinal'

    def test_substring_matching(self):
        """Test that substrings are matched correctly."""
        # Should match partial strings
        assert determine_score_type("subject1-likert_scale_5pt", ['likert']) == 'ordinal'
        assert determine_score_type("pre_rating_post", ['rating']) == 'ordinal'

        # Should not match if substring not present
        assert determine_score_type("subject1-like", ['likert']) == 'binary'


class TestValidateOrdinalScores:
    """Tests for ordinal score validation."""

    def test_valid_scores(self):
        """Test validation passes for valid scores."""
        scores = np.array([0, 3, 5, 7, 10])
        validate_ordinal_scores(scores, ordinal_max_score=10, grouping_name="test")
        # Should not raise

    def test_negative_scores(self):
        """Test that negative scores raise ValueError."""
        scores = np.array([-1, 0, 5, 10])
        with pytest.raises(ValueError, match="negative scores"):
            validate_ordinal_scores(scores, ordinal_max_score=10, grouping_name="test")

    def test_scores_exceed_max(self):
        """Test that scores exceeding max raise ValueError."""
        scores = np.array([0, 5, 10, 11])
        with pytest.raises(ValueError, match="exceeding ordinal_max_score"):
            validate_ordinal_scores(scores, ordinal_max_score=10, grouping_name="test")

    def test_empty_scores(self):
        """Test that empty scores array does not raise."""
        scores = np.array([])
        validate_ordinal_scores(scores, ordinal_max_score=10, grouping_name="test")
        # Should not raise, just warn

    def test_nan_scores(self):
        """Test that NaN values are handled correctly."""
        scores = np.array([np.nan, 0, 5, 10])
        validate_ordinal_scores(scores, ordinal_max_score=10, grouping_name="test")
        # Should not raise

    def test_all_nan_scores(self):
        """Test that all-NaN scores do not raise."""
        scores = np.array([np.nan, np.nan, np.nan])
        validate_ordinal_scores(scores, ordinal_max_score=10, grouping_name="test")
        # Should not raise, just warn

    def test_non_integer_scores(self):
        """Test that non-integer scores trigger warning but don't raise."""
        scores = np.array([0.5, 3.7, 7.2, 10.0])
        # Should warn but not raise
        validate_ordinal_scores(scores, ordinal_max_score=10, grouping_name="test")


class TestOrdinalCIAdaptive:
    """Tests for Bayesian bootstrap CI computation."""

    def test_basic_ci_computation(self):
        """Test basic CI computation for ordinal scores."""
        scores = np.array([5, 6, 7, 8, 9, 5, 6, 7, 8, 9])
        lo, hi, width = _ordinal_ci_adaptive(
            scores,
            ordinal_max_score=10,
            cred_level=0.95,
            conservatism=1.0
        )

        # Check that CI is computed
        assert 0 <= lo <= hi <= 1
        assert width == hi - lo

        # Check that mean is within CI
        mean_scaled = np.mean(scores) / 10
        assert lo <= mean_scaled <= hi

    def test_single_score(self):
        """Test CI computation with single score."""
        scores = np.array([7])
        lo, hi, width = _ordinal_ci_adaptive(
            scores,
            ordinal_max_score=10,
            cred_level=0.95
        )

        # Should return wide interval
        assert 0 <= lo <= hi <= 1
        assert width > 0

    def test_empty_scores(self):
        """Test CI computation with empty scores."""
        scores = np.array([])
        lo, hi, width = _ordinal_ci_adaptive(
            scores,
            ordinal_max_score=10,
            cred_level=0.95
        )

        # Should return maximum uncertainty
        assert lo == 0.0
        assert hi == 1.0
        assert width == 1.0

    def test_conservatism_applied(self):
        """Test that conservatism is applied for low performance."""
        # Low performance scores
        low_scores = np.array([0, 1, 0, 1, 2, 1, 0, 1])

        lo_conservative, hi_conservative, width_conservative = _ordinal_ci_adaptive(
            low_scores,
            ordinal_max_score=10,
            cred_level=0.95,
            conservatism=5.0,
            low_perf_threshold=0.2
        )

        lo_normal, hi_normal, width_normal = _ordinal_ci_adaptive(
            low_scores,
            ordinal_max_score=10,
            cred_level=0.95,
            conservatism=1.0,
            low_perf_threshold=0.2
        )

        # Conservative width should be larger
        assert width_conservative > width_normal

    def test_high_performance_no_conservatism(self):
        """Test that conservatism is NOT applied for high performance."""
        # High performance scores
        high_scores = np.array([8, 9, 10, 9, 8, 9, 10, 9])

        lo_conservative, hi_conservative, width_conservative = _ordinal_ci_adaptive(
            high_scores,
            ordinal_max_score=10,
            cred_level=0.95,
            conservatism=5.0,
            low_perf_threshold=0.2
        )

        lo_normal, hi_normal, width_normal = _ordinal_ci_adaptive(
            high_scores,
            ordinal_max_score=10,
            cred_level=0.95,
            conservatism=1.0,
            low_perf_threshold=0.2
        )

        # Widths should be similar (no conservatism applied)
        assert np.isclose(width_conservative, width_normal, rtol=0.01)

    def test_different_max_scores(self):
        """Test scaling with different maximum scores."""
        # 0-5 scale
        scores_5 = np.array([2, 3, 4, 3, 2, 3, 4, 3])
        lo_5, hi_5, width_5 = _ordinal_ci_adaptive(
            scores_5,
            ordinal_max_score=5,
            cred_level=0.95
        )

        # 0-10 scale (doubled scores)
        scores_10 = scores_5 * 2
        lo_10, hi_10, width_10 = _ordinal_ci_adaptive(
            scores_10,
            ordinal_max_score=10,
            cred_level=0.95
        )

        # Scaled CIs should be similar
        assert np.isclose(lo_5, lo_10, atol=0.05)
        assert np.isclose(hi_5, hi_10, atol=0.05)

    def test_ci_bounds_valid(self):
        """Test that CI bounds are always in [0,1]."""
        # Try various score patterns
        score_patterns = [
            np.array([0, 0, 0, 1, 1, 2]),  # Very low
            np.array([8, 9, 10, 9, 10, 10]),  # Very high
            np.array([0, 5, 10, 0, 5, 10]),  # Bimodal
            np.array([5, 5, 5, 5, 5]),  # Constant
        ]

        for scores in score_patterns:
            lo, hi, width = _ordinal_ci_adaptive(
                scores,
                ordinal_max_score=10,
                cred_level=0.95
            )

            assert 0 <= lo <= 1, f"Lower bound out of range: {lo}"
            assert 0 <= hi <= 1, f"Upper bound out of range: {hi}"
            assert lo <= hi, f"Lower bound exceeds upper bound: {lo} > {hi}"

    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        scores = np.array([3, 5, 7, 6, 4, 5, 6, 7])

        np.random.seed(42)
        lo1, hi1, width1 = _ordinal_ci_adaptive(
            scores,
            ordinal_max_score=10,
            cred_level=0.95,
            n_bootstrap=1000
        )

        np.random.seed(42)
        lo2, hi2, width2 = _ordinal_ci_adaptive(
            scores,
            ordinal_max_score=10,
            cred_level=0.95,
            n_bootstrap=1000
        )

        assert lo1 == lo2
        assert hi1 == hi2
        assert width1 == width2
