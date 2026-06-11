"""
Unit tests for enhanced determine_score_type() function.

Tests Phase 1 implementation of continuous score detection and routing.
"""

import pytest
from optstop.ordinal_utils import determine_score_type


class TestDetermineScoreTypePhase1:
    """Test suite for enhanced determine_score_type() with aggregation support."""

    def test_discrete_binary_default(self):
        """Test discrete binary scoring (default case)."""
        score_type, bounds = determine_score_type(
            "subject1-accuracy",
            ordinal_tasks=None,
            is_aggregated=False
        )

        assert score_type == 'binary'
        assert bounds == {'lower': 0.0, 'upper': 1.0}

    def test_discrete_binary_explicit(self):
        """Test discrete binary scoring with explicit parameters."""
        score_type, bounds = determine_score_type(
            "model_gpt4-task_math",
            ordinal_tasks=['rating', 'likert'],
            is_aggregated=False,
            upper_bound=1.0
        )

        assert score_type == 'binary'
        assert bounds == {'lower': 0.0, 'upper': 1.0}

    def test_continuous_binary_aggregated(self):
        """Test continuous binary scoring (aggregated with mean/median)."""
        score_type, bounds = determine_score_type(
            "subject1-accuracy",
            ordinal_tasks=None,
            is_aggregated=True,
            upper_bound=1.0
        )

        assert score_type == 'continuous_01'
        assert bounds == {'lower': 0.0, 'upper': 1.0}

    def test_discrete_ordinal(self):
        """Test discrete ordinal scoring."""
        score_type, bounds = determine_score_type(
            "subject1-likert_scale",
            ordinal_tasks=['likert'],
            is_aggregated=False,
            upper_bound=10.0
        )

        assert score_type == 'ordinal'
        assert bounds == {'lower': 0.0, 'upper': 10.0}

    def test_continuous_ordinal_aggregated(self):
        """Test continuous ordinal scoring (aggregated with mean/median)."""
        score_type, bounds = determine_score_type(
            "subject1-likert_scale",
            ordinal_tasks=['likert'],
            is_aggregated=True,
            upper_bound=10.0
        )

        assert score_type == 'continuous_bounded'
        assert bounds == {'lower': 0.0, 'upper': 10.0}

    def test_ordinal_substring_matching_case_insensitive(self):
        """Test that ordinal pattern matching is case-insensitive."""
        # Lowercase pattern, uppercase in name
        score_type1, bounds1 = determine_score_type(
            "task_CONFIDENCE_rating",
            ordinal_tasks=['confidence'],
            is_aggregated=False,
            upper_bound=10.0
        )

        # Uppercase pattern, lowercase in name
        score_type2, bounds2 = determine_score_type(
            "task_confidence_rating",
            ordinal_tasks=['CONFIDENCE'],
            is_aggregated=False,
            upper_bound=10.0
        )

        assert score_type1 == 'ordinal'
        assert score_type2 == 'ordinal'

    def test_ordinal_multiple_patterns(self):
        """Test ordinal detection with multiple patterns."""
        # First pattern matches
        score_type1, _ = determine_score_type(
            "task_rating_test",
            ordinal_tasks=['rating', 'likert', 'confidence'],
            is_aggregated=False
        )

        # Second pattern matches
        score_type2, _ = determine_score_type(
            "task_likert_test",
            ordinal_tasks=['rating', 'likert', 'confidence'],
            is_aggregated=False
        )

        # Third pattern matches
        score_type3, _ = determine_score_type(
            "task_confidence_test",
            ordinal_tasks=['rating', 'likert', 'confidence'],
            is_aggregated=False
        )

        # No pattern matches
        score_type4, _ = determine_score_type(
            "task_accuracy_test",
            ordinal_tasks=['rating', 'likert', 'confidence'],
            is_aggregated=False
        )

        assert score_type1 == 'ordinal'
        assert score_type2 == 'ordinal'
        assert score_type3 == 'ordinal'
        assert score_type4 == 'binary'

    def test_bounds_with_different_upper_bound(self):
        """Test that upper_bound parameter is correctly used in bounds."""
        # Test with max score 5
        score_type1, bounds1 = determine_score_type(
            "task_rating",
            ordinal_tasks=['rating'],
            is_aggregated=False,
            upper_bound=5.0
        )

        # Test with max score 100
        score_type2, bounds2 = determine_score_type(
            "task_rating",
            ordinal_tasks=['rating'],
            is_aggregated=True,
            upper_bound=100.0
        )

        assert bounds1 == {'lower': 0.0, 'upper': 5.0}
        assert bounds2 == {'lower': 0.0, 'upper': 100.0}

    def test_aggregated_flag_overrides_discrete_inference(self):
        """Test that is_aggregated=True always returns continuous type."""
        # Binary task
        score_type_binary, _ = determine_score_type(
            "accuracy",
            ordinal_tasks=None,
            is_aggregated=True
        )

        # Ordinal task
        score_type_ordinal, _ = determine_score_type(
            "rating",
            ordinal_tasks=['rating'],
            is_aggregated=True,
            upper_bound=10.0
        )

        assert score_type_binary == 'continuous_01'
        assert score_type_ordinal == 'continuous_bounded'

    def test_empty_ordinal_tasks_list(self):
        """Test behavior with empty ordinal_tasks list."""
        score_type, bounds = determine_score_type(
            "task_rating",
            ordinal_tasks=[],  # Empty list
            is_aggregated=False
        )

        assert score_type == 'binary'
        assert bounds == {'lower': 0.0, 'upper': 1.0}

    def test_backward_compatibility_no_aggregation_params(self):
        """Test backward compatibility: default values should work."""
        # This simulates old code that doesn't pass new parameters
        score_type, bounds = determine_score_type(
            "task_accuracy"
            # Not passing ordinal_tasks, is_aggregated, or upper_bound
        )

        assert score_type == 'binary'
        assert bounds == {'lower': 0.0, 'upper': 1.0}

    def test_return_type_is_tuple(self):
        """Test that return type is always a tuple of (str, dict)."""
        result = determine_score_type("test")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], dict)
        assert 'lower' in result[1]
        assert 'upper' in result[1]

    def test_bounds_always_have_lower_and_upper(self):
        """Test that bounds dict always has 'lower' and 'upper' keys."""
        test_cases = [
            ('binary_task', None, False, 1.0),
            ('binary_task', None, True, 1.0),
            ('task_rating', ['rating'], False, 10.0),  # Name matches pattern
            ('task_rating', ['rating'], True, 10.0),   # Name matches pattern
        ]

        for name, ordinal_tasks, is_aggregated, upper_bound in test_cases:
            _, bounds = determine_score_type(
                name,
                ordinal_tasks=ordinal_tasks,
                is_aggregated=is_aggregated,
                upper_bound=upper_bound
            )

            assert 'lower' in bounds
            assert 'upper' in bounds
            assert isinstance(bounds['lower'], float)
            assert isinstance(bounds['upper'], float)
            assert bounds['lower'] == 0.0
            assert bounds['upper'] == upper_bound


class TestDetermineScoreTypeEdgeCases:
    """Test edge cases and error handling."""

    def test_none_grouping_name(self):
        """Test behavior with None as grouping name."""
        # Should not crash, should default to binary
        score_type, bounds = determine_score_type(
            None,
            ordinal_tasks=['rating']
        )

        # None doesn't match any pattern
        assert score_type == 'binary'
        assert bounds == {'lower': 0.0, 'upper': 1.0}

    def test_none_grouping_name_with_aggregation(self):
        """Test that None grouping_name works with aggregation."""
        # Should not crash, should return continuous_01
        score_type, bounds = determine_score_type(
            None,
            ordinal_tasks=['rating'],
            is_aggregated=True,
            upper_bound=10.0
        )

        # None doesn't match ordinal pattern, so returns continuous_01
        assert score_type == 'continuous_01'
        assert bounds == {'lower': 0.0, 'upper': 1.0}

    def test_empty_string_grouping_name(self):
        """Test behavior with empty string as grouping name."""
        score_type, bounds = determine_score_type(
            "",
            ordinal_tasks=['rating']
        )

        # Empty string doesn't match any pattern
        assert score_type == 'binary'

    def test_special_characters_in_grouping_name(self):
        """Test that special characters don't break pattern matching."""
        score_type, bounds = determine_score_type(
            "task_rating-v2.0_test@2024",
            ordinal_tasks=['rating'],
            is_aggregated=False
        )

        assert score_type == 'ordinal'

    def test_very_large_upper_bound(self):
        """Test with very large upper_bound values."""
        score_type, bounds = determine_score_type(
            "rating",
            ordinal_tasks=['rating'],
            is_aggregated=True,
            upper_bound=1e6
        )

        assert score_type == 'continuous_bounded'
        assert bounds['upper'] == 1e6

    def test_numeric_grouping_name(self):
        """Test that numeric grouping names are handled correctly."""
        # Integer grouping name (e.g., from numeric grouping IDs)
        score_type1, bounds1 = determine_score_type(
            12345,
            ordinal_tasks=['rating']
        )

        # Float grouping name
        score_type2, bounds2 = determine_score_type(
            123.45,
            ordinal_tasks=['rating']
        )

        # Should convert to string and not match ordinal pattern
        assert score_type1 == 'binary'
        assert score_type2 == 'binary'

    def test_numeric_grouping_name_matching_pattern(self):
        """Test numeric grouping name that contains ordinal pattern after str conversion."""
        # Unlikely but possible: grouping ID that contains "rating" when stringified
        # This tests the str() conversion works correctly
        score_type, bounds = determine_score_type(
            "task_123_rating_456",  # Mix of numbers and text
            ordinal_tasks=['rating']
        )

        assert score_type == 'ordinal'


class TestDetermineScoreTypeRealWorldScenarios:
    """Test real-world usage scenarios from inspect_ai integration."""

    def test_inspect_ai_binary_task_no_aggregation(self):
        """Test typical inspect_ai binary task without aggregation."""
        score_type, bounds = determine_score_type(
            "gpt-4-math_accuracy",
            ordinal_tasks=None,
            is_aggregated=False,
            upper_bound=1.0
        )

        assert score_type == 'binary'

    def test_inspect_ai_binary_task_with_mean_aggregation(self):
        """Test inspect_ai binary task with mean aggregation (3 scorers)."""
        score_type, bounds = determine_score_type(
            "gpt-4-math_accuracy",
            ordinal_tasks=None,
            is_aggregated=True,  # User set score_agg='mean'
            upper_bound=1.0
        )

        assert score_type == 'continuous_01'

    def test_inspect_ai_ordinal_task_no_aggregation(self):
        """Test inspect_ai ordinal task (confidence rating) without aggregation."""
        score_type, bounds = determine_score_type(
            "claude-3-confidence_rating",
            ordinal_tasks=['confidence', 'rating'],
            is_aggregated=False,
            upper_bound=10.0
        )

        assert score_type == 'ordinal'

    def test_inspect_ai_ordinal_task_with_median_aggregation(self):
        """Test inspect_ai ordinal task with median aggregation."""
        score_type, bounds = determine_score_type(
            "claude-3-confidence_rating",
            ordinal_tasks=['confidence', 'rating'],
            is_aggregated=True,  # User set score_agg='median'
            upper_bound=10.0
        )

        assert score_type == 'continuous_bounded'

    def test_inspect_ai_mixed_groupings(self):
        """Test that different groupings route correctly in mixed evaluation."""
        # Binary task
        binary_type, _ = determine_score_type(
            "model_a-task_math",
            ordinal_tasks=['confidence'],
            is_aggregated=False
        )

        # Ordinal task (same model)
        ordinal_type, _ = determine_score_type(
            "model_a-task_confidence",
            ordinal_tasks=['confidence'],
            is_aggregated=False,
            upper_bound=10.0
        )

        assert binary_type == 'binary'
        assert ordinal_type == 'ordinal'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
