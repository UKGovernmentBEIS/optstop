#!/usr/bin/env python3
"""
Verification script for Issue #1 fix: None grouping_name handling

This script tests that determine_score_type() correctly handles:
1. None as grouping_name
2. Empty string as grouping_name
3. Numeric grouping names (int, float)
4. All combinations with/without aggregation

Run this script to verify the fix works before running full test suite.
"""

# Add parent directory to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from optstop.ordinal_utils import determine_score_type


def test_none_handling():
    """Test that None grouping_name doesn't crash."""
    print("Testing None grouping_name handling...")

    # Test 1: None without aggregation
    try:
        score_type, bounds = determine_score_type(
            None,
            ordinal_tasks=['rating'],
            is_aggregated=False
        )
        assert score_type == 'binary'
        assert bounds == {'lower': 0.0, 'upper': 1.0}
        print("  ✓ Test 1: None without aggregation -> 'binary' ✓")
    except Exception as e:
        print(f"  ✗ Test 1 FAILED: {e}")
        return False

    # Test 2: None with aggregation
    try:
        score_type, bounds = determine_score_type(
            None,
            ordinal_tasks=['rating'],
            is_aggregated=True,
            upper_bound=10.0
        )
        assert score_type == 'continuous_01'
        assert bounds == {'lower': 0.0, 'upper': 1.0}
        print("  ✓ Test 2: None with aggregation -> 'continuous_01' ✓")
    except Exception as e:
        print(f"  ✗ Test 2 FAILED: {e}")
        return False

    # Test 3: Empty string
    try:
        score_type, bounds = determine_score_type(
            "",
            ordinal_tasks=['rating'],
            is_aggregated=False
        )
        assert score_type == 'binary'
        print("  ✓ Test 3: Empty string -> 'binary' ✓")
    except Exception as e:
        print(f"  ✗ Test 3 FAILED: {e}")
        return False

    # Test 4: Integer grouping name
    try:
        score_type, bounds = determine_score_type(
            12345,
            ordinal_tasks=['rating'],
            is_aggregated=False
        )
        assert score_type == 'binary'
        print("  ✓ Test 4: Integer (12345) -> 'binary' ✓")
    except Exception as e:
        print(f"  ✗ Test 4 FAILED: {e}")
        return False

    # Test 5: Float grouping name
    try:
        score_type, bounds = determine_score_type(
            123.45,
            ordinal_tasks=['rating'],
            is_aggregated=False
        )
        assert score_type == 'binary'
        print("  ✓ Test 5: Float (123.45) -> 'binary' ✓")
    except Exception as e:
        print(f"  ✗ Test 5 FAILED: {e}")
        return False

    return True


def test_normal_behavior():
    """Test that normal behavior still works correctly."""
    print("\nTesting normal behavior (regression check)...")

    # Test 1: Normal string, binary
    try:
        score_type, bounds = determine_score_type(
            "model_gpt4-task_accuracy",
            ordinal_tasks=None,
            is_aggregated=False
        )
        assert score_type == 'binary'
        print("  ✓ Test 1: Normal binary string ✓")
    except Exception as e:
        print(f"  ✗ Test 1 FAILED: {e}")
        return False

    # Test 2: Ordinal pattern match
    try:
        score_type, bounds = determine_score_type(
            "model_gpt4-task_rating",
            ordinal_tasks=['rating', 'likert'],
            is_aggregated=False,
            upper_bound=10.0
        )
        assert score_type == 'ordinal'
        assert bounds == {'lower': 0.0, 'upper': 10.0}
        print("  ✓ Test 2: Ordinal pattern matching ✓")
    except Exception as e:
        print(f"  ✗ Test 2 FAILED: {e}")
        return False

    # Test 3: Continuous with aggregation
    try:
        score_type, bounds = determine_score_type(
            "model_gpt4-task_accuracy",
            ordinal_tasks=None,
            is_aggregated=True,
            upper_bound=1.0
        )
        assert score_type == 'continuous_01'
        assert bounds == {'lower': 0.0, 'upper': 1.0}
        print("  ✓ Test 3: Continuous binary (aggregated) ✓")
    except Exception as e:
        print(f"  ✗ Test 3 FAILED: {e}")
        return False

    # Test 4: Continuous bounded (ordinal aggregated)
    try:
        score_type, bounds = determine_score_type(
            "model_gpt4-task_rating",
            ordinal_tasks=['rating'],
            is_aggregated=True,
            upper_bound=10.0
        )
        assert score_type == 'continuous_bounded'
        assert bounds == {'lower': 0.0, 'upper': 10.0}
        print("  ✓ Test 4: Continuous bounded (ordinal aggregated) ✓")
    except Exception as e:
        print(f"  ✗ Test 4 FAILED: {e}")
        return False

    return True


def main():
    """Run all verification tests."""
    print("=" * 70)
    print("Issue #1 Fix Verification")
    print("Testing None grouping_name handling")
    print("=" * 70)
    print()

    # Run tests
    none_tests_passed = test_none_handling()
    normal_tests_passed = test_normal_behavior()

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if none_tests_passed and normal_tests_passed:
        print("✅ ALL TESTS PASSED")
        print()
        print("Issue #1 is FIXED:")
        print("  • None grouping_name no longer causes AttributeError")
        print("  • Empty string handled correctly")
        print("  • Numeric grouping names converted to string")
        print("  • Normal behavior unchanged (backward compatible)")
        print()
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print()
        if not none_tests_passed:
            print("  • None handling tests failed")
        if not normal_tests_passed:
            print("  • Normal behavior tests failed (regression)")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
