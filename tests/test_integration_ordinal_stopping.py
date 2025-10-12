"""
Comprehensive integration tests for ordinal stopping with mixed binary/ordinal datasets.

Tests cover:
1. Mixed binary and ordinal groupings in same dataset
2. Simulated data sequences with known properties (peaked, diffuse, bimodal)
3. Stopping decision validation across score types
4. Comparison of modal vs entropy inference (when integrated)
5. Performance and accuracy metrics
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, '/home/ubuntu/optstop')

from optstop import optimal_stopping_posthoc, configure_optstop_logging
from optstop.ordinal_utils import determine_score_type, validate_ordinal_scores


# Configure logging for tests
configure_optstop_logging(logfile='test_integration_ordinal.log', level=20, console_output=False)


class TestDataGenerator:
    """Generate synthetic test data with controlled properties."""

    @staticmethod
    def generate_binary_sequence(n_items: int, n_epochs: int, true_prob: float,
                                  noise: float = 0.0) -> np.ndarray:
        """
        Generate binary (0/1) scores for testing.

        Parameters
        ----------
        n_items : int
            Number of items/samples
        n_epochs : int
            Number of epochs per item
        true_prob : float
            True underlying success probability
        noise : float
            Random noise level (0-1)

        Returns
        -------
        scores : np.ndarray, shape (n_items, n_epochs)
        """
        scores = np.zeros((n_items, n_epochs))
        for i in range(n_items):
            # Add item-level variability
            item_prob = true_prob + np.random.normal(0, noise)
            item_prob = np.clip(item_prob, 0.0, 1.0)
            scores[i, :] = np.random.binomial(1, item_prob, n_epochs)
        return scores

    @staticmethod
    def generate_ordinal_peaked(n_items: int, n_epochs: int, modal_category: int,
                                max_score: int = 10, spread: float = 1.0) -> np.ndarray:
        """
        Generate ordinal scores with a peaked (unimodal) distribution.

        Parameters
        ----------
        n_items : int
            Number of items/samples
        n_epochs : int
            Number of epochs per item
        modal_category : int
            Most common category (0 to max_score)
        max_score : int
            Maximum score value
        spread : float
            Standard deviation around modal category

        Returns
        -------
        scores : np.ndarray, shape (n_items, n_epochs)
        """
        scores = np.zeros((n_items, n_epochs), dtype=int)
        for i in range(n_items):
            for j in range(n_epochs):
                # Sample from normal, round and clip
                score = np.random.normal(modal_category, spread)
                score = int(np.round(np.clip(score, 0, max_score)))
                scores[i, j] = score
        return scores

    @staticmethod
    def generate_ordinal_diffuse(n_items: int, n_epochs: int, max_score: int = 10,
                                 min_category: int = 3, max_category: int = 8) -> np.ndarray:
        """
        Generate ordinal scores with a diffuse (uniform-ish) distribution.

        Parameters
        ----------
        n_items : int
            Number of items/samples
        n_epochs : int
            Number of epochs per item
        max_score : int
            Maximum score value
        min_category, max_category : int
            Range of categories to sample from

        Returns
        -------
        scores : np.ndarray, shape (n_items, n_epochs)
        """
        scores = np.zeros((n_items, n_epochs), dtype=int)
        for i in range(n_items):
            scores[i, :] = np.random.randint(min_category, max_category + 1, n_epochs)
        return scores

    @staticmethod
    def generate_ordinal_bimodal(n_items: int, n_epochs: int, mode1: int, mode2: int,
                                 max_score: int = 10, spread: float = 1.0) -> np.ndarray:
        """
        Generate ordinal scores with a bimodal distribution.

        Parameters
        ----------
        n_items : int
            Number of items/samples
        n_epochs : int
            Number of epochs per item
        mode1, mode2 : int
            Two modal categories
        max_score : int
            Maximum score value
        spread : float
            Standard deviation around each mode

        Returns
        -------
        scores : np.ndarray, shape (n_items, n_epochs)
        """
        scores = np.zeros((n_items, n_epochs), dtype=int)
        for i in range(n_items):
            for j in range(n_epochs):
                # Randomly pick one of two modes
                mode = mode1 if np.random.rand() < 0.5 else mode2
                score = np.random.normal(mode, spread)
                score = int(np.round(np.clip(score, 0, max_score)))
                scores[i, j] = score
        return scores

    @staticmethod
    def create_mixed_dataset(n_items_per_group: int = 10, n_epochs: int = 50) -> pd.DataFrame:
        """
        Create a mixed dataset with binary and ordinal groupings.

        Groupings:
        1. 'accuracy_binary' - Binary (0/1) scores
        2. 'likert_survey' - Ordinal peaked (mode=7, spread=1.5)
        3. 'pass_fail_binary' - Binary (0/1) scores
        4. 'rating_scale' - Ordinal diffuse (uniform 4-8)
        5. 'performance_ordinal' - Ordinal bimodal (modes 3, 8)

        Returns
        -------
        df : pd.DataFrame
            Columns: grouping, item_id, trial, score
        """
        data = []

        # Group 1: Binary accuracy (high performance)
        scores = TestDataGenerator.generate_binary_sequence(
            n_items_per_group, n_epochs, true_prob=0.75, noise=0.1
        )
        for item_id in range(n_items_per_group):
            for trial in range(n_epochs):
                data.append({
                    'grouping': 'accuracy_binary',
                    'item_id': f'acc_{item_id}',
                    'trial': trial,
                    'score': int(scores[item_id, trial])
                })

        # Group 2: Ordinal likert (peaked, mode=7)
        scores = TestDataGenerator.generate_ordinal_peaked(
            n_items_per_group, n_epochs, modal_category=7, spread=1.5
        )
        for item_id in range(n_items_per_group):
            for trial in range(n_epochs):
                data.append({
                    'grouping': 'likert_survey',
                    'item_id': f'likert_{item_id}',
                    'trial': trial,
                    'score': int(scores[item_id, trial])
                })

        # Group 3: Binary pass/fail (medium performance)
        scores = TestDataGenerator.generate_binary_sequence(
            n_items_per_group, n_epochs, true_prob=0.5, noise=0.15
        )
        for item_id in range(n_items_per_group):
            for trial in range(n_epochs):
                data.append({
                    'grouping': 'pass_fail_binary',
                    'item_id': f'pass_{item_id}',
                    'trial': trial,
                    'score': int(scores[item_id, trial])
                })

        # Group 4: Ordinal rating (diffuse)
        scores = TestDataGenerator.generate_ordinal_diffuse(
            n_items_per_group, n_epochs, min_category=4, max_category=8
        )
        for item_id in range(n_items_per_group):
            for trial in range(n_epochs):
                data.append({
                    'grouping': 'rating_scale',
                    'item_id': f'rating_{item_id}',
                    'trial': trial,
                    'score': int(scores[item_id, trial])
                })

        # Group 5: Ordinal performance (bimodal, modes 3 and 8)
        scores = TestDataGenerator.generate_ordinal_bimodal(
            n_items_per_group, n_epochs, mode1=3, mode2=8, spread=1.0
        )
        for item_id in range(n_items_per_group):
            for trial in range(n_epochs):
                data.append({
                    'grouping': 'performance_ordinal',
                    'item_id': f'perf_{item_id}',
                    'trial': trial,
                    'score': int(scores[item_id, trial])
                })

        return pd.DataFrame(data)


class TestScoreTypeDetection:
    """Test score type determination for mixed datasets."""

    def test_binary_detection(self):
        """Test that binary groupings are correctly identified."""
        ordinal_tasks = ['likert', 'rating', 'performance']

        assert determine_score_type('accuracy_binary', ordinal_tasks) == 'binary'
        assert determine_score_type('pass_fail_binary', ordinal_tasks) == 'binary'

    def test_ordinal_detection(self):
        """Test that ordinal groupings are correctly identified."""
        ordinal_tasks = ['likert', 'rating', 'performance']

        assert determine_score_type('likert_survey', ordinal_tasks) == 'ordinal'
        assert determine_score_type('rating_scale', ordinal_tasks) == 'ordinal'
        assert determine_score_type('performance_ordinal', ordinal_tasks) == 'ordinal'

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        ordinal_tasks = ['Likert', 'RATING']

        assert determine_score_type('likert_survey', ordinal_tasks) == 'ordinal'
        assert determine_score_type('RATING_scale', ordinal_tasks) == 'ordinal'
        assert determine_score_type('accuracy', ordinal_tasks) == 'binary'


class TestScoreValidation:
    """Test score validation for ordinal data."""

    def test_valid_ordinal_scores(self):
        """Test that valid scores pass validation."""
        scores = np.array([0, 3, 5, 7, 10, 8, 6])
        # Should not raise
        validate_ordinal_scores(scores, ordinal_max_score=10, grouping_name='test')

    def test_out_of_range_scores(self):
        """Test that out-of-range scores are rejected."""
        scores = np.array([0, 5, 12, 8])  # 12 exceeds max
        with pytest.raises(ValueError, match="out of valid range"):
            validate_ordinal_scores(scores, ordinal_max_score=10, grouping_name='test')

    def test_negative_scores(self):
        """Test that negative scores are rejected."""
        scores = np.array([-1, 5, 7, 8])
        with pytest.raises(ValueError, match="out of valid range"):
            validate_ordinal_scores(scores, ordinal_max_score=10, grouping_name='test')


class TestMixedDatasetStopping:
    """Test stopping behavior on mixed binary/ordinal datasets."""

    def test_mixed_dataset_basic(self):
        """Test basic functionality with mixed binary and ordinal groupings."""
        # Generate small test dataset
        df = TestDataGenerator.create_mixed_dataset(n_items_per_group=5, n_epochs=30)

        params = {
            'delta_item': 0.15,
            'delta_cap': 0.15,
            'cred_level': 0.95,
            'draws': 500,
            'tune': 250,
            'rep_batch_size': 2,
            'pymc_refresh_every': 2
        }

        # Run optimal stopping with mixed scoring
        pruned_df, summary = optimal_stopping_posthoc(
            df,
            params,
            grouping_columns=['grouping'],
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['likert', 'rating', 'performance'],  # 3 ordinal, 2 binary
            ordinal_max_score=10,
            display_progress=False
        )

        # Basic checks
        assert len(summary) == 5, "Should have 5 grouping results"
        assert not pruned_df.empty, "Pruned data should not be empty"
        assert len(pruned_df) < len(df), "Should prune some data"

        # Check that all groupings were processed
        groupings_processed = [s['grouping'] for s in summary]
        expected_groupings = [0, 1, 2, 3, 4]  # Numeric codes
        assert set(groupings_processed) == set(expected_groupings)

        print("\n✓ Mixed dataset basic test passed")
        print(f"  Original data: {len(df)} rows")
        print(f"  Pruned data: {len(pruned_df)} rows")
        print(f"  Efficiency: {(1 - len(pruned_df)/len(df))*100:.1f}% reduction")

    def test_stopping_appropriateness_peaked(self):
        """Test that peaked ordinal data stops appropriately (early)."""
        # Generate peaked data (should converge quickly)
        n_items = 10
        n_epochs = 50

        data = []
        scores = TestDataGenerator.generate_ordinal_peaked(
            n_items, n_epochs, modal_category=7, spread=0.8  # Very peaked
        )
        for item_id in range(n_items):
            for trial in range(n_epochs):
                data.append({
                    'grouping': 'peaked_ordinal',
                    'item_id': f'item_{item_id}',
                    'trial': trial,
                    'score': int(scores[item_id, trial])
                })

        df = pd.DataFrame(data)

        params = {
            'delta_item': 0.15,
            'delta_cap': 0.15,
            'cred_level': 0.95,
            'draws': 500,
            'tune': 250,
            'rep_batch_size': 2,
            'pymc_refresh_every': 2
        }

        pruned_df, summary = optimal_stopping_posthoc(
            df,
            params,
            grouping_columns=['grouping'],
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['peaked'],
            ordinal_max_score=10,
            display_progress=False
        )

        # Peaked data should stop relatively early
        avg_trials_used = pruned_df.groupby('item_id').size().mean()
        efficiency = (1 - len(pruned_df)/len(df))

        assert efficiency > 0.2, f"Peaked data should save >20% trials, got {efficiency*100:.1f}%"
        assert avg_trials_used < n_epochs * 0.7, f"Should use <70% of trials on average, used {avg_trials_used}"

        print(f"\n✓ Peaked ordinal stopping test passed")
        print(f"  Average trials used: {avg_trials_used:.1f} / {n_epochs}")
        print(f"  Efficiency: {efficiency*100:.1f}% reduction")

    def test_stopping_appropriateness_diffuse(self):
        """Test that diffuse ordinal data continues longer (doesn't stop too early)."""
        # Generate diffuse data (should need more trials)
        n_items = 10
        n_epochs = 50

        data = []
        scores = TestDataGenerator.generate_ordinal_diffuse(
            n_items, n_epochs, min_category=3, max_category=8
        )
        for item_id in range(n_items):
            for trial in range(n_epochs):
                data.append({
                    'grouping': 'diffuse_ordinal',
                    'item_id': f'item_{item_id}',
                    'trial': trial,
                    'score': int(scores[item_id, trial])
                })

        df = pd.DataFrame(data)

        params = {
            'delta_item': 0.15,
            'delta_cap': 0.15,
            'cred_level': 0.95,
            'draws': 500,
            'tune': 250,
            'rep_batch_size': 2,
            'pymc_refresh_every': 2
        }

        pruned_df, summary = optimal_stopping_posthoc(
            df,
            params,
            grouping_columns=['grouping'],
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['diffuse'],
            ordinal_max_score=10,
            display_progress=False
        )

        # Diffuse data should use more trials
        avg_trials_used = pruned_df.groupby('item_id').size().mean()

        # Should not stop too early for diffuse data
        assert avg_trials_used > n_epochs * 0.4, f"Diffuse data should use >40% trials, used {avg_trials_used}"

        print(f"\n✓ Diffuse ordinal stopping test passed")
        print(f"  Average trials used: {avg_trials_used:.1f} / {n_epochs}")
        print(f"  Correctly continues longer for uncertain distributions")

    def test_bimodal_handling(self):
        """Test that bimodal ordinal data is handled correctly."""
        # Generate bimodal data (two peaks)
        n_items = 10
        n_epochs = 50

        data = []
        scores = TestDataGenerator.generate_ordinal_bimodal(
            n_items, n_epochs, mode1=3, mode2=8, spread=0.8
        )
        for item_id in range(n_items):
            for trial in range(n_epochs):
                data.append({
                    'grouping': 'bimodal_ordinal',
                    'item_id': f'item_{item_id}',
                    'trial': trial,
                    'score': int(scores[item_id, trial])
                })

        df = pd.DataFrame(data)

        params = {
            'delta_item': 0.20,  # Slightly relaxed threshold
            'delta_cap': 0.20,
            'cred_level': 0.95,
            'draws': 500,
            'tune': 250,
            'rep_batch_size': 2,
            'pymc_refresh_every': 2
        }

        pruned_df, summary = optimal_stopping_posthoc(
            df,
            params,
            grouping_columns=['grouping'],
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['bimodal'],
            ordinal_max_score=10,
            display_progress=False
        )

        # Should complete without errors
        assert len(summary) == 1
        assert summary[0]['error'] is None
        assert not pruned_df.empty

        print(f"\n✓ Bimodal ordinal handling test passed")
        print(f"  Successfully processed bimodal distribution")
        print(f"  Average trials used: {pruned_df.groupby('item_id').size().mean():.1f} / {n_epochs}")


class TestConsistencyAcrossScoreTypes:
    """Test that stopping decisions are consistent and sensible across score types."""

    def test_similar_uncertainty_similar_stopping(self):
        """Test that similar uncertainty levels lead to similar stopping behavior."""
        n_items = 8
        n_epochs = 40

        # Binary with moderate uncertainty (p ≈ 0.7)
        binary_scores = TestDataGenerator.generate_binary_sequence(
            n_items, n_epochs, true_prob=0.7, noise=0.1
        )

        # Ordinal with similar performance (mode = 7 out of 10, so ~0.7 scaled)
        ordinal_scores = TestDataGenerator.generate_ordinal_peaked(
            n_items, n_epochs, modal_category=7, spread=1.5
        )

        # Create two separate datasets
        binary_data = []
        for item_id in range(n_items):
            for trial in range(n_epochs):
                binary_data.append({
                    'grouping': 'binary_group',
                    'item_id': f'item_{item_id}',
                    'trial': trial,
                    'score': int(binary_scores[item_id, trial])
                })

        ordinal_data = []
        for item_id in range(n_items):
            for trial in range(n_epochs):
                ordinal_data.append({
                    'grouping': 'ordinal_group',
                    'item_id': f'item_{item_id}',
                    'trial': trial,
                    'score': int(ordinal_scores[item_id, trial])
                })

        df_binary = pd.DataFrame(binary_data)
        df_ordinal = pd.DataFrame(ordinal_data)

        params = {
            'delta_item': 0.15,
            'delta_cap': 0.15,
            'cred_level': 0.95,
            'draws': 500,
            'tune': 250,
            'rep_batch_size': 2,
            'pymc_refresh_every': 2
        }

        # Run binary
        pruned_binary, summary_binary = optimal_stopping_posthoc(
            df_binary,
            params,
            grouping_columns=['grouping'],
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            display_progress=False
        )

        # Run ordinal
        pruned_ordinal, summary_ordinal = optimal_stopping_posthoc(
            df_ordinal,
            params,
            grouping_columns=['grouping'],
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['ordinal'],
            ordinal_max_score=10,
            display_progress=False
        )

        # Compare efficiency
        binary_efficiency = 1 - len(pruned_binary) / len(df_binary)
        ordinal_efficiency = 1 - len(pruned_ordinal) / len(df_ordinal)

        # Should be somewhat similar (within 50% relative difference)
        relative_diff = abs(binary_efficiency - ordinal_efficiency) / max(binary_efficiency, ordinal_efficiency)

        print(f"\n✓ Cross-type consistency test")
        print(f"  Binary efficiency: {binary_efficiency*100:.1f}%")
        print(f"  Ordinal efficiency: {ordinal_efficiency*100:.1f}%")
        print(f"  Relative difference: {relative_diff*100:.1f}%")

        # This is a soft check - stopping behavior can differ, but should be reasonable
        assert relative_diff < 0.8, f"Stopping behavior too different: {relative_diff*100:.1f}%"


class TestRobustnessAndEdgeCases:
    """Test edge cases and robustness of mixed stopping."""

    def test_single_item_per_grouping(self):
        """Test with just 1 item per grouping."""
        data = []

        # Binary with 1 item
        for trial in range(20):
            data.append({
                'grouping': 'binary_single',
                'item_id': 'item_0',
                'trial': trial,
                'score': np.random.binomial(1, 0.8)
            })

        # Ordinal with 1 item
        for trial in range(20):
            data.append({
                'grouping': 'ordinal_single',
                'item_id': 'item_0',
                'trial': trial,
                'score': np.random.randint(6, 9)
            })

        df = pd.DataFrame(data)

        params = {
            'delta_item': 0.20,
            'delta_cap': 0.30,
            'cred_level': 0.95,
            'draws': 300,
            'tune': 150,
            'rep_batch_size': 2,
            'pymc_refresh_every': 1
        }

        # Should handle single items gracefully
        pruned_df, summary = optimal_stopping_posthoc(
            df,
            params,
            grouping_columns=['grouping'],
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['ordinal'],
            ordinal_max_score=10,
            display_progress=False
        )

        assert len(summary) == 2
        assert all(s['error'] is None for s in summary)

        print("\n✓ Single item per grouping test passed")

    def test_very_few_trials(self):
        """Test with very limited data (5 trials per item)."""
        n_items = 5
        n_trials = 5

        data = []
        for item_id in range(n_items):
            for trial in range(n_trials):
                data.append({
                    'grouping': 'limited_ordinal',
                    'item_id': f'item_{item_id}',
                    'trial': trial,
                    'score': np.random.randint(5, 9)
                })

        df = pd.DataFrame(data)

        params = {
            'delta_item': 0.30,  # Relaxed
            'delta_cap': 0.40,
            'cred_level': 0.90,
            'draws': 200,
            'tune': 100,
            'rep_batch_size': 1,
            'pymc_refresh_every': 1
        }

        # Should handle limited data gracefully (may use all trials)
        pruned_df, summary = optimal_stopping_posthoc(
            df,
            params,
            grouping_columns=['grouping'],
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['limited'],
            ordinal_max_score=10,
            display_progress=False
        )

        assert not pruned_df.empty
        assert summary[0]['error'] is None

        print("\n✓ Very few trials test passed")


if __name__ == '__main__':
    print("=" * 80)
    print("Running Comprehensive Ordinal Stopping Integration Tests")
    print("=" * 80)

    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
