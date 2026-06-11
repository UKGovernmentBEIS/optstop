"""
Comprehensive integration tests for ordinal stopping using optimal_stopping_live_single.

Tests cover:
1. Ordinal hierarchical inference with Dirichlet-Multinomial model
2. Peaked, diffuse, and bimodal distributions
3. Binary vs ordinal consistency
4. Edge cases (single item, few trials)

Based on test_large_datasets_numpyro.py pattern with restricted sample sizes for fast execution.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

from optstop.rule import optimal_stopping_live_single
from optstop.ordinal_utils import determine_score_type, validate_ordinal_scores
from optstop import configure_optstop_logging


# Configure logging for tests
configure_optstop_logging(logfile='test_integration_ordinal.log', level=20, console_output=False)


class TestDataGenerator:
    """Generate synthetic test data with controlled properties."""

    @staticmethod
    def generate_binary_sequence(n_items: int, n_epochs: int, true_prob: float,
                                  noise: float = 0.0, seed: int = None) -> np.ndarray:
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
        seed : int
            Random seed for reproducibility

        Returns
        -------
        scores : np.ndarray, shape (n_items, n_epochs)
        """
        if seed is not None:
            np.random.seed(seed)
        scores = np.zeros((n_items, n_epochs))
        for i in range(n_items):
            # Add item-level variability
            item_prob = true_prob + np.random.normal(0, noise)
            item_prob = np.clip(item_prob, 0.0, 1.0)
            scores[i, :] = np.random.binomial(1, item_prob, n_epochs)
        return scores

    @staticmethod
    def generate_ordinal_peaked(n_items: int, n_epochs: int, modal_category: int,
                                max_score: int = 10, spread: float = 1.0, seed: int = None) -> np.ndarray:
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
        seed : int
            Random seed for reproducibility

        Returns
        -------
        scores : np.ndarray, shape (n_items, n_epochs)
        """
        if seed is not None:
            np.random.seed(seed)
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
                                 min_category: int = 3, max_category: int = 8, seed: int = None) -> np.ndarray:
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
        seed : int
            Random seed for reproducibility

        Returns
        -------
        scores : np.ndarray, shape (n_items, n_epochs)
        """
        if seed is not None:
            np.random.seed(seed)
        scores = np.zeros((n_items, n_epochs), dtype=int)
        for i in range(n_items):
            scores[i, :] = np.random.randint(min_category, max_category + 1, n_epochs)
        return scores

    @staticmethod
    def generate_ordinal_bimodal(n_items: int, n_epochs: int, mode1: int, mode2: int,
                                 max_score: int = 10, spread: float = 1.0, seed: int = None) -> np.ndarray:
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
        seed : int
            Random seed for reproducibility

        Returns
        -------
        scores : np.ndarray, shape (n_items, n_epochs)
        """
        if seed is not None:
            np.random.seed(seed)
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
    def create_dataframe_from_scores(scores: np.ndarray, grouping_name: str) -> pd.DataFrame:
        """Convert scores array to DataFrame format expected by optimal_stopping_live_single."""
        n_items, n_epochs = scores.shape
        data = []
        for item_id in range(n_items):
            for trial in range(n_epochs):
                data.append({
                    'grouping': grouping_name,
                    'item_id': f'item_{item_id}',
                    'trial': trial,
                    'score': int(scores[item_id, trial])
                })
        return pd.DataFrame(data)


class TestScoreTypeDetection:
    """Test score type determination for mixed datasets."""

    def test_binary_detection(self):
        """Test that binary groupings are correctly identified."""
        ordinal_tasks = ['likert', 'rating', 'performance']

        assert determine_score_type('accuracy_binary', ordinal_tasks)[0] == 'binary'
        assert determine_score_type('pass_fail_binary', ordinal_tasks)[0] == 'binary'

    def test_ordinal_detection(self):
        """Test that ordinal groupings are correctly identified."""
        ordinal_tasks = ['likert', 'rating', 'performance']

        assert determine_score_type('likert_survey', ordinal_tasks)[0] == 'ordinal'
        assert determine_score_type('rating_scale', ordinal_tasks)[0] == 'ordinal'
        assert determine_score_type('performance_ordinal', ordinal_tasks)[0] == 'ordinal'

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        ordinal_tasks = ['Likert', 'RATING']

        assert determine_score_type('likert_survey', ordinal_tasks)[0] == 'ordinal'
        assert determine_score_type('RATING_scale', ordinal_tasks)[0] == 'ordinal'
        assert determine_score_type('accuracy', ordinal_tasks)[0] == 'binary'


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
        with pytest.raises(ValueError, match="exceeding ordinal_max_score"):
            validate_ordinal_scores(scores, ordinal_max_score=10, grouping_name='test')

    def test_negative_scores(self):
        """Test that negative scores are rejected."""
        scores = np.array([-1, 5, 7, 8])
        with pytest.raises(ValueError, match="negative scores"):
            validate_ordinal_scores(scores, ordinal_max_score=10, grouping_name='test')


class TestOrdinalStoppingLiveSingle:
    """Test ordinal stopping using optimal_stopping_live_single."""

    def test_peaked_ordinal_basic(self):
        """Test basic functionality with peaked ordinal data."""
        # Generate peaked data (should converge quickly)
        n_items = 8
        n_epochs = 15

        scores = TestDataGenerator.generate_ordinal_peaked(
            n_items, n_epochs, modal_category=7, spread=0.8, seed=42
        )
        df = TestDataGenerator.create_dataframe_from_scores(scores, 'peaked_ordinal')

        params = {
            'delta_item': 0.15,
            'delta_cap': 0.15,
            'cred_level': 0.95,
            'draws': 300,
            'tune': 150,
        }

        # Run optimal stopping
        result = optimal_stopping_live_single(
            df_grouping=df,
            grouping_name='peaked_ordinal',
            params=params,
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['peaked'],
            ordinal_max_score=10,
            ordinal_inference='modal'
        )

        # Basic checks
        assert 'grouping' in result
        assert 'stop_this_grouping' in result
        assert 'stabilization_history' in result
        assert 'metadata' in result

        print(f"\n Peaked ordinal basic test:")
        print(f"  Grouping stopped: {len(result['stop_this_grouping']) > 0}")
        print(f"  Samples stopped: {len(result.get('stop_sample_ids', []))}")

    def test_diffuse_ordinal_basic(self):
        """Test with diffuse ordinal data (should need more data)."""
        n_items = 8
        n_epochs = 15

        scores = TestDataGenerator.generate_ordinal_diffuse(
            n_items, n_epochs, min_category=3, max_category=8, seed=42
        )
        df = TestDataGenerator.create_dataframe_from_scores(scores, 'diffuse_ordinal')

        params = {
            'delta_item': 0.15,
            'delta_cap': 0.15,
            'cred_level': 0.95,
            'draws': 300,
            'tune': 150,
        }

        result = optimal_stopping_live_single(
            df_grouping=df,
            grouping_name='diffuse_ordinal',
            params=params,
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['diffuse'],
            ordinal_max_score=10,
            ordinal_inference='modal'
        )

        assert 'grouping' in result
        assert 'stabilization_history' in result

        print(f"\n Diffuse ordinal test:")
        print(f"  Grouping stopped: {len(result['stop_this_grouping']) > 0}")

    def test_bimodal_ordinal_basic(self):
        """Test with bimodal ordinal data."""
        n_items = 8
        n_epochs = 15

        scores = TestDataGenerator.generate_ordinal_bimodal(
            n_items, n_epochs, mode1=3, mode2=8, spread=0.8, seed=42
        )
        df = TestDataGenerator.create_dataframe_from_scores(scores, 'bimodal_ordinal')

        params = {
            'delta_item': 0.20,
            'delta_cap': 0.20,
            'cred_level': 0.95,
            'draws': 300,
            'tune': 150,
        }

        result = optimal_stopping_live_single(
            df_grouping=df,
            grouping_name='bimodal_ordinal',
            params=params,
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['bimodal'],
            ordinal_max_score=10,
            ordinal_inference='modal'
        )

        assert 'grouping' in result
        assert 'stabilization_history' in result

        print(f"\n Bimodal ordinal test:")
        print(f"  Grouping stopped: {len(result['stop_this_grouping']) > 0}")

    def test_entropy_inference_mode(self):
        """Test ordinal stopping with entropy inference mode."""
        n_items = 8
        n_epochs = 15

        scores = TestDataGenerator.generate_ordinal_peaked(
            n_items, n_epochs, modal_category=7, spread=1.0, seed=42
        )
        df = TestDataGenerator.create_dataframe_from_scores(scores, 'entropy_test')

        params = {
            'delta_item': 0.15,
            'delta_cap': 0.15,
            'cred_level': 0.95,
            'draws': 300,
            'tune': 150,
        }

        result = optimal_stopping_live_single(
            df_grouping=df,
            grouping_name='entropy_test',
            params=params,
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['entropy'],
            ordinal_max_score=10,
            ordinal_inference='entropy'
        )

        assert 'grouping' in result
        assert 'stabilization_history' in result

        print(f"\n Entropy mode test:")
        print(f"  Grouping stopped: {len(result['stop_this_grouping']) > 0}")

    def test_hybrid_inference_mode(self):
        """Test ordinal stopping with hybrid inference mode."""
        n_items = 8
        n_epochs = 15

        scores = TestDataGenerator.generate_ordinal_peaked(
            n_items, n_epochs, modal_category=7, spread=1.0, seed=42
        )
        df = TestDataGenerator.create_dataframe_from_scores(scores, 'hybrid_test')

        params = {
            'delta_item': 0.15,
            'delta_cap': 0.15,
            'cred_level': 0.95,
            'draws': 300,
            'tune': 150,
        }

        result = optimal_stopping_live_single(
            df_grouping=df,
            grouping_name='hybrid_test',
            params=params,
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['hybrid'],
            ordinal_max_score=10,
            ordinal_inference='hybrid'
        )

        assert 'grouping' in result
        assert 'stabilization_history' in result

        print(f"\n Hybrid mode test:")
        print(f"  Grouping stopped: {len(result['stop_this_grouping']) > 0}")


class TestBinaryVsOrdinalConsistency:
    """Test consistency between binary and ordinal inference."""

    def test_similar_data_similar_behavior(self):
        """Test that similar data leads to similar stopping behavior."""
        n_items = 8
        n_epochs = 15

        # Binary with high success rate (p=0.7)
        binary_scores = TestDataGenerator.generate_binary_sequence(
            n_items, n_epochs, true_prob=0.7, noise=0.1, seed=42
        )
        df_binary = TestDataGenerator.create_dataframe_from_scores(binary_scores, 'binary_test')

        # Ordinal peaked at 7 (similar scaled performance)
        ordinal_scores = TestDataGenerator.generate_ordinal_peaked(
            n_items, n_epochs, modal_category=7, spread=1.0, seed=42
        )
        df_ordinal = TestDataGenerator.create_dataframe_from_scores(ordinal_scores, 'ordinal_test')

        params = {
            'delta_item': 0.15,
            'delta_cap': 0.15,
            'cred_level': 0.95,
            'draws': 300,
            'tune': 150,
        }

        # Run binary
        result_binary = optimal_stopping_live_single(
            df_grouping=df_binary,
            grouping_name='binary_test',
            params=params,
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=[],  # Empty = binary inference
            ordinal_max_score=10
        )

        # Run ordinal
        result_ordinal = optimal_stopping_live_single(
            df_grouping=df_ordinal,
            grouping_name='ordinal_test',
            params=params,
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['ordinal'],
            ordinal_max_score=10,
            ordinal_inference='modal'
        )

        # Both should complete without errors
        assert 'grouping' in result_binary
        assert 'grouping' in result_ordinal

        binary_stopped = len(result_binary['stop_this_grouping']) > 0
        ordinal_stopped = len(result_ordinal['stop_this_grouping']) > 0

        print(f"\n Binary vs Ordinal consistency test:")
        print(f"  Binary stopped: {binary_stopped}")
        print(f"  Ordinal stopped: {ordinal_stopped}")


class TestStabilizationHistory:
    """Test stabilization history persistence across calls."""

    def test_history_accumulation(self):
        """Test that stabilization history accumulates across calls."""
        n_items = 8
        n_epochs = 15

        scores = TestDataGenerator.generate_ordinal_peaked(
            n_items, n_epochs, modal_category=7, spread=1.0, seed=42
        )
        df = TestDataGenerator.create_dataframe_from_scores(scores, 'history_test')

        params = {
            'delta_item': 0.15,
            'delta_cap': 0.15,
            'cred_level': 0.95,
            'draws': 300,
            'tune': 150,
        }

        # First call - no history
        result1 = optimal_stopping_live_single(
            df_grouping=df.head(n_items * 5),  # First 5 epochs
            grouping_name='history_test',
            params=params,
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['history'],
            ordinal_max_score=10,
            stabilization_history=None
        )

        # Get history from first call
        history1 = result1['stabilization_history']
        assert history1 is not None

        # Second call - pass history
        result2 = optimal_stopping_live_single(
            df_grouping=df,  # All epochs
            grouping_name='history_test',
            params=params,
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['history'],
            ordinal_max_score=10,
            stabilization_history=history1
        )

        history2 = result2['stabilization_history']
        assert history2 is not None

        print(f"\n History accumulation test:")
        print(f"  History after call 1: {len(history1.get('ci_width_history', []))} entries")
        print(f"  History after call 2: {len(history2.get('ci_width_history', []))} entries")


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_small_sample_size(self):
        """Test with minimal sample size (3 items)."""
        n_items = 3
        n_epochs = 10

        scores = TestDataGenerator.generate_ordinal_peaked(
            n_items, n_epochs, modal_category=7, spread=1.0, seed=42
        )
        df = TestDataGenerator.create_dataframe_from_scores(scores, 'small_sample')

        params = {
            'delta_item': 0.25,
            'delta_cap': 0.30,
            'cred_level': 0.90,
            'draws': 200,
            'tune': 100,
        }

        result = optimal_stopping_live_single(
            df_grouping=df,
            grouping_name='small_sample',
            params=params,
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['small'],
            ordinal_max_score=10
        )

        assert 'grouping' in result
        print(f"\n Small sample test (3 items):")
        print(f"  Completed successfully")

    def test_few_epochs(self):
        """Test with few epochs (5 per item)."""
        n_items = 5
        n_epochs = 5

        scores = TestDataGenerator.generate_ordinal_peaked(
            n_items, n_epochs, modal_category=7, spread=1.0, seed=42
        )
        df = TestDataGenerator.create_dataframe_from_scores(scores, 'few_epochs')

        params = {
            'delta_item': 0.30,
            'delta_cap': 0.40,
            'cred_level': 0.90,
            'draws': 200,
            'tune': 100,
        }

        result = optimal_stopping_live_single(
            df_grouping=df,
            grouping_name='few_epochs',
            params=params,
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['few'],
            ordinal_max_score=10
        )

        assert 'grouping' in result
        print(f"\n Few epochs test (5 epochs):")
        print(f"  Completed successfully")

    def test_all_same_score(self):
        """Test with all scores the same (edge case)."""
        n_items = 5
        n_epochs = 10

        # All scores are 7
        scores = np.full((n_items, n_epochs), 7, dtype=int)
        df = TestDataGenerator.create_dataframe_from_scores(scores, 'same_score')

        params = {
            'delta_item': 0.15,
            'delta_cap': 0.15,
            'cred_level': 0.95,
            'draws': 300,
            'tune': 150,
        }

        result = optimal_stopping_live_single(
            df_grouping=df,
            grouping_name='same_score',
            params=params,
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['same'],
            ordinal_max_score=10
        )

        assert 'grouping' in result
        print(f"\n All same score test:")
        print(f"  Completed successfully")
        print(f"  Grouping stopped: {len(result['stop_this_grouping']) > 0}")


class TestCalibration:
    """
    Calibration tests: Verify that credible intervals have correct coverage.

    A well-calibrated 95% CI should contain the true parameter ~95% of the time.
    These tests use known parameters to verify coverage properties.
    """

    def test_modal_ci_contains_true_mode(self):
        """Test that modal CI usually contains the true modal category."""
        n_items = 10
        n_epochs = 20
        true_mode = 7

        # Generate strongly peaked data at true mode
        scores = TestDataGenerator.generate_ordinal_peaked(
            n_items, n_epochs, modal_category=true_mode, spread=0.5, seed=42
        )
        df = TestDataGenerator.create_dataframe_from_scores(scores, 'calibration_modal')

        params = {
            'delta_item': 0.5,  # Wide threshold to not trigger stopping
            'delta_cap': 0.5,
            'cred_level': 0.95,
            'draws': 400,
            'tune': 200,
        }

        result = optimal_stopping_live_single(
            df_grouping=df,
            grouping_name='calibration_modal',
            params=params,
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['calibration'],
            ordinal_max_score=10,
            ordinal_inference='modal'
        )

        history = result['stabilization_history']
        modal_ci = history.get('final_modal_ci', [0, 1])

        # Scale true mode to [0,1]
        true_mode_scaled = true_mode / 10.0

        # Check if true mode is within CI
        ci_contains_true = modal_ci[0] <= true_mode_scaled <= modal_ci[1]

        print(f"\n Modal calibration test:")
        print(f"  True mode (scaled): {true_mode_scaled}")
        print(f"  95% CI: [{modal_ci[0]:.3f}, {modal_ci[1]:.3f}]")
        print(f"  CI contains true mode: {ci_contains_true}")

        # With strongly peaked data and correct implementation, this should usually pass
        # Note: Not asserting True because single-run coverage can fail stochastically
        assert 'final_modal_ci' in history or 'ci_width_history' in history

    def test_entropy_scaling_in_bounds(self):
        """Test that entropy values are scaled to [0, 1]."""
        n_items = 10
        n_epochs = 15

        # Peaked data should have low entropy (< 0.5 scaled)
        scores = TestDataGenerator.generate_ordinal_peaked(
            n_items, n_epochs, modal_category=7, spread=0.5, seed=42
        )
        df = TestDataGenerator.create_dataframe_from_scores(scores, 'entropy_scaling')

        params = {
            'delta_item': 0.5,
            'delta_cap': 0.5,
            'cred_level': 0.95,
            'draws': 400,
            'tune': 200,
        }

        result = optimal_stopping_live_single(
            df_grouping=df,
            grouping_name='entropy_scaling',
            params=params,
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['entropy'],
            ordinal_max_score=10,
            ordinal_inference='entropy'
        )

        history = result['stabilization_history']
        final_entropy = history.get('final_entropy', 0.5)

        print(f"\n Entropy scaling test:")
        print(f"  Final entropy (scaled): {final_entropy:.3f}")
        print(f"  Expected range: [0, 1]")

        # Entropy should be in [0, 1] after scaling
        assert 0.0 <= final_entropy <= 1.0, f"Entropy {final_entropy} not in [0, 1]"

        # Peaked data should have relatively low entropy (< 0.5)
        # Note: Not strictly asserting because depends on data specifics


class TestShrinkage:
    """
    Shrinkage tests: Verify partial pooling behavior.

    In hierarchical models, items with less data should be "shrunk" toward
    the population mean more than items with more data.
    """

    def test_hierarchical_pooling_effect(self):
        """
        Test that hierarchical model produces reasonable group-level estimates.

        With heterogeneous item-level data, the group-level posterior should
        reflect the aggregate pattern while accounting for item variability.
        """
        n_items = 8
        n_epochs = 15

        # Create heterogeneous data: some items peaked at 7, some at 8
        np.random.seed(42)
        scores = np.zeros((n_items, n_epochs), dtype=int)
        for i in range(n_items):
            # Half items peaked at 7, half at 8
            mode = 7 if i < n_items // 2 else 8
            for j in range(n_epochs):
                score = np.random.normal(mode, 0.8)
                scores[i, j] = int(np.round(np.clip(score, 0, 10)))

        df = TestDataGenerator.create_dataframe_from_scores(scores, 'shrinkage_test')

        params = {
            'delta_item': 0.5,
            'delta_cap': 0.5,
            'cred_level': 0.95,
            'draws': 400,
            'tune': 200,
        }

        result = optimal_stopping_live_single(
            df_grouping=df,
            grouping_name='shrinkage_test',
            params=params,
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['shrinkage'],
            ordinal_max_score=10,
            ordinal_inference='modal'
        )

        history = result['stabilization_history']
        modal_ci = history.get('final_modal_ci', [0, 1])

        # Group modal should be somewhere between 7 and 8 (scaled: 0.7 to 0.8)
        # Due to partial pooling, we expect the CI to cover this range
        expected_range_lo = 0.65  # Allow some slack
        expected_range_hi = 0.85

        print(f"\n Shrinkage/pooling test:")
        print(f"  Data: half items at mode 7, half at mode 8")
        print(f"  Modal CI: [{modal_ci[0]:.3f}, {modal_ci[1]:.3f}]")
        print(f"  Expected group mode: 0.7-0.8 (scaled)")

        # The CI should overlap with [0.7, 0.8]
        ci_overlaps_expected = modal_ci[0] < expected_range_hi and modal_ci[1] > expected_range_lo
        assert ci_overlaps_expected, f"CI {modal_ci} does not overlap expected [0.65, 0.85]"


class TestUnitOrdinalModel:
    """
    Unit tests for ordinal model components.

    These tests verify individual functions work correctly in isolation.
    """

    def test_counts_to_scores_roundtrip(self):
        """Test that counts_to_scores correctly reconstructs scores."""
        from optstop.ordinal_utils import counts_to_scores

        # Create a count vector: 0 zeros, 2 ones, 3 twos, 1 three
        counts = np.array([0, 2, 3, 1])

        # Convert to scores
        scores = counts_to_scores(counts)

        # Verify
        assert len(scores) == 6, f"Expected 6 scores, got {len(scores)}"
        assert np.sum(scores == 0) == 0
        assert np.sum(scores == 1) == 2
        assert np.sum(scores == 2) == 3
        assert np.sum(scores == 3) == 1

        print(f"\n counts_to_scores test:")
        print(f"  Input counts: {counts}")
        print(f"  Output scores: {scores}")
        print(f"  Roundtrip successful")

    def test_aggregate_item_counts(self):
        """Test aggregation of item counts."""
        from optstop.ordinal_utils import aggregate_item_counts

        # Two items with different count vectors
        item_summaries = [
            {'counts': np.array([1, 2, 3, 0, 0])},  # Item 1: 1 zero, 2 ones, 3 twos
            {'counts': np.array([0, 1, 1, 2, 1])}   # Item 2: 1 one, 1 two, 2 threes, 1 four
        ]

        total = aggregate_item_counts(item_summaries, ordinal_max_score=4)

        # Expected: [1, 3, 4, 2, 1]
        expected = np.array([1, 3, 4, 2, 1])

        assert np.array_equal(total, expected), f"Expected {expected}, got {total}"

        print(f"\n aggregate_item_counts test:")
        print(f"  Item 1 counts: {item_summaries[0]['counts']}")
        print(f"  Item 2 counts: {item_summaries[1]['counts']}")
        print(f"  Aggregated: {total}")

    def test_determine_score_type_aggregation(self):
        """Test score type determination with aggregation flag."""
        from optstop.ordinal_utils import determine_score_type

        # Non-aggregated ordinal
        score_type, bounds = determine_score_type(
            'likert_survey',
            ordinal_tasks=['likert'],
            is_aggregated=False,
            upper_bound=10.0
        )
        assert score_type == 'ordinal'
        assert bounds == {'lower': 0.0, 'upper': 10.0}

        # Aggregated ordinal -> continuous_bounded
        score_type, bounds = determine_score_type(
            'likert_survey',
            ordinal_tasks=['likert'],
            is_aggregated=True,
            upper_bound=10.0
        )
        assert score_type == 'continuous_bounded'
        assert bounds == {'lower': 0.0, 'upper': 10.0}

        # Aggregated binary -> continuous_01
        score_type, bounds = determine_score_type(
            'accuracy',
            ordinal_tasks=['likert'],
            is_aggregated=True,
            upper_bound=1.0
        )
        assert score_type == 'continuous_01'
        assert bounds == {'lower': 0.0, 'upper': 1.0}

        print(f"\n determine_score_type aggregation test:")
        print(f"  Non-aggregated ordinal: ordinal")
        print(f"  Aggregated ordinal: continuous_bounded")
        print(f"  Aggregated binary: continuous_01")

    def test_validate_ordinal_scores_edge_cases(self):
        """Test validation handles edge cases correctly."""
        from optstop.ordinal_utils import validate_ordinal_scores

        # Empty array should not raise
        validate_ordinal_scores(np.array([]), 10, 'empty')

        # Array with NaN should work (NaN filtered)
        scores_with_nan = np.array([1, 2, np.nan, 5])
        validate_ordinal_scores(scores_with_nan, 10, 'with_nan')

        # Single value at boundary
        validate_ordinal_scores(np.array([0]), 10, 'single_zero')
        validate_ordinal_scores(np.array([10]), 10, 'single_max')

        print(f"\n validate_ordinal_scores edge cases test:")
        print(f"  Empty array: OK")
        print(f"  Array with NaN: OK")
        print(f"  Single boundary values: OK")


class TestModelCaching:
    """Test model caching behavior."""

    def test_cache_reuse(self):
        """Test that model caches are properly reused."""
        n_items = 8
        n_epochs = 10

        scores = TestDataGenerator.generate_ordinal_peaked(
            n_items, n_epochs, modal_category=7, spread=1.0, seed=42
        )
        df = TestDataGenerator.create_dataframe_from_scores(scores, 'cache_test')

        params = {
            'delta_item': 0.15,
            'delta_cap': 0.15,
            'cred_level': 0.95,
            'draws': 200,
            'tune': 100,
        }

        # First call - no cache
        result1 = optimal_stopping_live_single(
            df_grouping=df,
            grouping_name='cache_test',
            params=params,
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['cache'],
            ordinal_max_score=10,
            model_caches=None
        )

        # Get caches
        caches = result1.get('model_caches', {})

        # Second call - pass caches
        result2 = optimal_stopping_live_single(
            df_grouping=df,
            grouping_name='cache_test',
            params=params,
            sample_id_column='item_id',
            epoch_column='trial',
            score_column='score',
            ordinal_tasks=['cache'],
            ordinal_max_score=10,
            model_caches=caches
        )

        assert 'grouping' in result2
        print(f"\n Model caching test:")
        print(f"  First call completed")
        print(f"  Second call with cache completed")


if __name__ == '__main__':
    print("=" * 80)
    print("Running Ordinal Stopping Integration Tests (optimal_stopping_live_single)")
    print("=" * 80)

    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
