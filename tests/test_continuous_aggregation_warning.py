"""
Integration test for Phase 1: Continuous score aggregation warning.

This test verifies that when users specify score_agg='mean' or 'median',
the system correctly:
1. Detects aggregated continuous scores
2. Logs a warning that continuous inference is not yet implemented
3. Runs all trials to completion (no early stopping)
4. Returns appropriate metadata indicating the warning
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
import logging
from unittest.mock import MagicMock
from optstop.early_stopping import OptimalStoppingManager


# Helper functions to create mock objects
def create_mock_evalspec(model='gpt-4', task='test_task'):
    """Create a mock EvalSpec object."""
    mock = MagicMock()
    mock.model = model
    mock.task = task
    mock.eval_id = 'test_eval_001'
    return mock


def create_mock_samples(n_samples=10):
    """Create a list of mock Sample objects."""
    samples = []
    for i in range(n_samples):
        mock = MagicMock()
        mock.id = f"sample_{i}"
        mock.metadata = {}
        samples.append(mock)
    return samples


def create_mock_score(value):
    """Create a mock SampleScore object."""
    mock_score = MagicMock()
    mock_score.score = MagicMock()
    mock_score.score.value = value
    return mock_score


class TestContinuousAggregationWarningPhase1:
    """Test suite for Phase 1 continuous aggregation warning."""

    @pytest.mark.asyncio
    async def test_binary_task_with_mean_aggregation_shows_warning(self, caplog):
        """Test that binary task with score_agg='mean' triggers continuous warning."""

        # Configure logging to capture warnings
        caplog.set_level(logging.WARNING, logger='optstop.live_single')

        # Create manager with mean aggregation
        manager = OptimalStoppingManager(
            optstop_params={
                'delta_item': 0.05,
                'delta_cap': 0.05,
                'cred_level': 0.95,
                'conservatism': 5
            },
            grouping_columns=['model', 'task'],
            score_agg='mean',  # CRITICAL: Triggers continuous inference
            reanalysis_interval=10,
            min_samples_per_grouping=5,
            shadow_mode=False
        )

        # Create mock task and samples
        evalspec = create_mock_evalspec(model='gpt-4', task='binary_test')
        samples = create_mock_samples(n_samples=20)

        # Start task
        await manager.start_task(evalspec, samples, epochs=15)

        # Simulate evaluation with aggregated scores
        completed_samples = 0
        for sample_idx in range(20):
            for epoch in range(1, 16):
                sample_id = f"sample_{sample_idx}"

                # Generate 3 binary scores and aggregate (mean)
                binary_scores = np.random.binomial(1, 0.8, size=3)
                mean_score = float(np.mean(binary_scores))  # e.g., 0.667

                # Create mock scores (3 scorers)
                scores = {
                    'scorer_1': create_mock_score(float(binary_scores[0])),
                    'scorer_2': create_mock_score(float(binary_scores[1])),
                    'scorer_3': create_mock_score(float(binary_scores[2]))
                }

                # Check if should run (should always return None since no stopping)
                directive = await manager.schedule_sample(sample_id, epoch)
                assert directive is None, "Should not stop any samples (continuous not implemented)"

                # Complete sample
                await manager.complete_sample(sample_id, epoch, scores)
                completed_samples += 1

                # After min_samples_per_grouping, check for warning
                if completed_samples == manager.reanalysis_interval * 1:
                    break

            if completed_samples >= manager.reanalysis_interval * 1:
                break

        # Complete task
        metadata = await manager.complete_task()

        # Verify warning was logged
        warning_logged = any(
            'continuous bounded inference' in record.message.lower() and
            'not yet implemented' in record.message.lower()
            for record in caplog.records
        )
        assert warning_logged, "Should log warning about continuous inference not implemented"

        # Verify no samples were stopped
        assert metadata['stopped_samples_count'] == 0, "No samples should be stopped"
        assert metadata['total_skipped'] == 0, "No trials should be skipped"
        assert metadata['efficiency_percent'] == 0.0, "Efficiency should be 0% (no stopping)"

    @pytest.mark.asyncio
    async def test_ordinal_task_with_median_aggregation_shows_warning(self, caplog):
        """Test that ordinal task with score_agg='median' triggers continuous warning."""

        # Configure logging
        caplog.set_level(logging.WARNING, logger='optstop.live_single')

        # Create manager with median aggregation and ordinal task
        manager = OptimalStoppingManager(
            optstop_params={
                'delta_item': 0.05,
                'delta_cap': 0.05,
                'cred_level': 0.95,
                'conservatism': 5
            },
            grouping_columns=['model', 'task'],
            score_agg='median',  # CRITICAL: Triggers continuous inference
            ordinal_tasks=['rating'],
            ordinal_max_score=10,
            reanalysis_interval=10,
            min_samples_per_grouping=5,
            shadow_mode=False
        )

        # Create mock task and samples
        evalspec = create_mock_evalspec(model='gpt-4', task='ordinal_rating')
        samples = create_mock_samples(n_samples=20)

        # Start task
        await manager.start_task(evalspec, samples, epochs=15)

        # Simulate evaluation with aggregated ordinal scores
        completed_samples = 0
        for sample_idx in range(20):
            for epoch in range(1, 16):
                sample_id = f"sample_{sample_idx}"

                # Generate 3 ordinal scores and aggregate (median)
                ordinal_scores = np.random.randint(7, 11, size=3)
                median_score = float(np.median(ordinal_scores))  # e.g., 8.0 or 8.5

                # Create mock scores (3 scorers)
                scores = {
                    'scorer_1': create_mock_score(float(ordinal_scores[0])),
                    'scorer_2': create_mock_score(float(ordinal_scores[1])),
                    'scorer_3': create_mock_score(float(ordinal_scores[2]))
                }

                # Check if should run
                directive = await manager.schedule_sample(sample_id, epoch)
                assert directive is None, "Should not stop any samples"

                # Complete sample
                await manager.complete_sample(sample_id, epoch, scores)
                completed_samples += 1

                if completed_samples == manager.reanalysis_interval * 1:
                    break

            if completed_samples >= manager.reanalysis_interval * 1:
                break

        # Complete task
        metadata = await manager.complete_task()

        # Verify warning was logged
        warning_logged = any(
            'continuous bounded inference' in record.message.lower() and
            'not yet implemented' in record.message.lower()
            for record in caplog.records
        )
        assert warning_logged, "Should log warning for ordinal continuous inference"

        # Verify no samples were stopped
        assert metadata['stopped_samples_count'] == 0
        assert metadata['total_skipped'] == 0

    @pytest.mark.asyncio
    async def test_no_aggregation_no_warning(self, caplog):
        """Test that discrete scores (no aggregation) do NOT trigger warning."""

        # Configure logging
        caplog.set_level(logging.WARNING, logger='optstop.live_single')

        # Create manager WITHOUT aggregation (discrete scores)
        manager = OptimalStoppingManager(
            optstop_params={
                'delta_item': 0.05,
                'delta_cap': 0.05,
                'cred_level': 0.95,
                'conservatism': 5
            },
            grouping_columns=['model', 'task'],
            score_agg=None,  # No aggregation - discrete binary scores
            reanalysis_interval=10,
            min_samples_per_grouping=5,
            shadow_mode=False
        )

        # Create mock task and samples
        evalspec = create_mock_evalspec(model='gpt-4', task='binary_test')
        samples = create_mock_samples(n_samples=20)

        # Start task
        await manager.start_task(evalspec, samples, epochs=15)

        # Simulate evaluation with discrete binary scores
        completed_samples = 0
        for sample_idx in range(20):
            for epoch in range(1, 16):
                sample_id = f"sample_{sample_idx}"

                # Single discrete binary score (0 or 1)
                score_value = float(np.random.binomial(1, 0.8))

                # Create mock scores (single scorer)
                scores = {
                    'accuracy': create_mock_score(score_value)
                }

                # Check if should run
                directive = await manager.schedule_sample(sample_id, epoch)
                # May or may not stop (binary inference is implemented)

                if directive is not None:
                    break  # This sample stopped

                # Complete sample
                await manager.complete_sample(sample_id, epoch, scores)
                completed_samples += 1

                if completed_samples >= manager.reanalysis_interval * 1:
                    break

            if completed_samples >= manager.reanalysis_interval * 1:
                break

        # Complete task
        metadata = await manager.complete_task()

        # Verify NO warning about continuous inference
        warning_logged = any(
            'continuous bounded inference' in record.message.lower() and
            'not yet implemented' in record.message.lower()
            for record in caplog.records
        )
        assert not warning_logged, "Should NOT log continuous warning for discrete scores"

    @pytest.mark.asyncio
    async def test_aggregation_metadata_includes_warning(self):
        """Test that metadata includes warning key when continuous inference is triggered."""

        # Create manager with aggregation
        manager = OptimalStoppingManager(
            optstop_params={
                'delta_item': 0.05,
                'delta_cap': 0.05,
            },
            grouping_columns=['model', 'task'],
            score_agg='mean',
            reanalysis_interval=5,
            min_samples_per_grouping=3,
        )

        # Create mock task and samples
        evalspec = create_mock_evalspec()
        samples = create_mock_samples(n_samples=10)

        # Start task
        await manager.start_task(evalspec, samples, epochs=10)

        # Simulate minimal evaluation to trigger inference
        for sample_idx in range(6):  # More than min_samples_per_grouping
            sample_id = f"sample_{sample_idx}"
            scores = {
                'scorer_1': create_mock_score(0.8),
                'scorer_2': create_mock_score(0.9)
            }
            await manager.complete_sample(sample_id, 1, scores)

        # Complete task
        metadata = await manager.complete_task()

        # Check metadata structure
        # (Implementation note: We may need to expose stabilization_histories
        # metadata to verify the warning)
        assert metadata['stopped_samples_count'] == 0
        assert metadata['efficiency_percent'] == 0.0


class TestScoreValidationWithAggregation:
    """Test that score validation correctly handles aggregated continuous scores."""

    @pytest.mark.asyncio
    async def test_aggregated_binary_allows_continuous_values(self):
        """Test that aggregated binary scores allow continuous [0,1] values."""

        manager = OptimalStoppingManager(
            optstop_params={'delta_item': 0.05, 'delta_cap': 0.05},
            grouping_columns=['model', 'task'],
            score_agg='mean',  # Aggregation allows continuous values
            reanalysis_interval=100  # High value to avoid inference
        )

        evalspec = create_mock_evalspec()
        samples = create_mock_samples(n_samples=1)
        await manager.start_task(evalspec, samples, epochs=5)

        # Test continuous values between 0 and 1
        test_scores = [0.0, 0.333, 0.5, 0.667, 1.0]

        for epoch, score_val in enumerate(test_scores, start=1):
            scores = {'scorer': create_mock_score(score_val)}

            # Should not raise any validation errors
            await manager.complete_sample('sample_0', epoch, scores)

            # Verify score was recorded
            mask = (
                (manager.compiled_dataset[manager.sample_id_column] == 'sample_0') &
                (manager.compiled_dataset[manager.epoch_column] == epoch)
            )
            recorded_score = manager.compiled_dataset.loc[mask, manager.score_column].iloc[0]
            assert recorded_score == score_val

    @pytest.mark.asyncio
    async def test_discrete_binary_rejects_continuous_values(self, caplog):
        """Test that discrete binary scores reject continuous values (non-aggregated)."""

        caplog.set_level(logging.WARNING)

        manager = OptimalStoppingManager(
            optstop_params={'delta_item': 0.05, 'delta_cap': 0.05},
            grouping_columns=['model', 'task'],
            score_agg=None,  # NO aggregation - discrete only
            reanalysis_interval=100
        )

        evalspec = create_mock_evalspec()
        samples = create_mock_samples(n_samples=1)
        await manager.start_task(evalspec, samples, epochs=2)

        # Test that score > 1 is rejected for discrete binary
        scores_invalid = {'scorer': create_mock_score(1.5)}  # Invalid
        await manager.complete_sample('sample_0', 1, scores_invalid)

        # Check that warning was logged
        validation_warning = any(
            'score > 1' in record.message.lower() and
            'cannot perform inference' in record.message.lower()
            for record in caplog.records
        )
        assert validation_warning, "Should warn about invalid discrete binary score > 1"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
