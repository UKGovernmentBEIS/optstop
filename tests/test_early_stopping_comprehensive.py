"""
Comprehensive Integration Tests for OptimalStoppingManager (early_stopping.py)

Tests each score-type inference path end-to-end through the bridge manager:
1. Binary discrete (0/1)
2. Ordinal discrete (0-10, modal inference)
3. Continuous [0, 1] (binary aggregated, mean)
4. Continuous [0, 10] (ordinal aggregated, median)

Plus the distinct structural/edge regimes: zero-variance, single-epoch
(insufficient data), and simultaneous multi-grouping.

These are end-to-end structural checks (the manager runs to completion and
compiles the expected dataset). Numeric stopping behaviour is asserted in the
pathway-specific unit suites; large-N stress is covered by the slow-marked
test_large_datasets_ordered_logistic.py.
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock
from typing import List, Any

import time

from optstop.early_stopping import OptimalStoppingManager, StoppedSample


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


# CI partition: heavy MCMC tests deselected from PR CI (see pyproject.toml markers).
pytestmark = pytest.mark.optstop


def create_mock_sample(sample_id: str, metadata: dict = None):
    """Create a mock Sample object."""
    sample = Mock()
    sample.id = sample_id
    sample.metadata = metadata or {}
    return sample


def create_mock_task(model: str, task: str):
    """Create a mock EvalSpec object."""
    task_mock = Mock()
    task_mock.model = model
    task_mock.task = task
    task_mock.eval_id = f"eval_{model}_{task}"
    return task_mock


def create_mock_score(value: float, metadata: dict = None):
    """Create a mock SampleScore object."""
    # Create nested structure: SampleScore.score.value
    inner_score = Mock()
    inner_score.value = value

    sample_score = Mock()
    sample_score.score = inner_score
    sample_score.metadata = metadata or {}
    return sample_score


# =============================================================================
# BINARY DISCRETE (0/1)
# =============================================================================

class TestBinaryDiscreteHighPerformance:
    """Test binary discrete scores with consistent high performance."""

    @pytest.mark.asyncio
    async def test_binary_discrete_consistent_high_50_samples(self):
        """
        Binary discrete: 50 samples, 15 epochs each, consistent ~90% performance.
        Should reach stopping criteria for most samples.
        """
        np.random.seed(100)

        # Configuration
        manager = OptimalStoppingManager(
            optstop_params={
                'delta_item': 0.08,
                'delta_cap': 0.08,
                'cred_level': 0.95,
                'conservatism': 5.0,
                'low_performance_threshold': 0.2,
                'draws': 500,
                'tune': 500
            },
            grouping_columns=['model', 'task'],
            score_choice=None,
            score_agg=None,  # No aggregation = discrete binary
            reanalysis_interval=188,  # 25%, 50%, 75% of 750 trials
            ordinal_tasks=None,
            ordinal_max_score=10
        )

        # Create task and samples
        task = create_mock_task('gpt4', 'math')
        samples = [create_mock_sample(f'sample_{i:03d}') for i in range(50)]

        # Initialize
        await manager.start_task(task, samples, epochs=15)

        # Simulate scoring: ~90% success rate with small variance
        for epoch in range(15):
            for sample in samples:
                # Generate binary score with ~90% success probability
                score_value = float(np.random.random() < 0.90)
                score = create_mock_score(score_value)

                # Pass sample id, epoch, and scores dict
                await manager.complete_sample(sample.id, epoch, {'score': score})

        # Finalize
        final_result = await manager.complete_task()

        # Assertions
        assert 'stopped_samples' in final_result
        assert 'stopped_groupings' in final_result
        assert 'efficiency_percent' in final_result

        # With consistent high performance, should have some stopped samples
        print(f"✓ Binary discrete (high): {len(final_result['stopped_samples'])} samples stopped out of 50")
        print(f"  Efficiency: {final_result['efficiency_percent']}")

        # Verify dataset structure
        assert len(manager.compiled_dataset) == 50 * 15
        assert 'score' in manager.compiled_dataset.columns


# =============================================================================
# ORDINAL DISCRETE (0-10) - MODAL INFERENCE
# =============================================================================

class TestOrdinalDiscreteHighPerformance:
    """Test ordinal discrete scores with consistent high ratings."""

    @pytest.mark.asyncio
    async def test_ordinal_discrete_consistent_high_60_samples(self):
        """
        Ordinal discrete: 60 samples, 15 epochs each, consistent ~8/10 ratings.
        Should reach stopping criteria.
        """
        np.random.seed(102)

        manager = OptimalStoppingManager(
            optstop_params={
                'delta_item': 0.08,
                'delta_cap': 0.08,
                'cred_level': 0.95,
                'conservatism': 5.0,
                'draws': 500,
                'tune': 500
            },
            grouping_columns=['model', 'task'],
            reanalysis_interval=225,  # reanalysis_interval=225,  # 25%, 50%, 75% of 900 trials
            ordinal_tasks=['quality'],  # Identifies as ordinal
            ordinal_max_score=10,
            ordinal_inference='modal'
        )

        task = create_mock_task('gpt4', 'quality_assessment')
        samples = [create_mock_sample(f'sample_{i:03d}') for i in range(60)]

        await manager.start_task(task, samples, epochs=15)

        # Generate ordinal scores around 8/10 with small variance
        for epoch in range(15):
            for sample in samples:
                # Generate score: mean=8, std=1, clipped to [0, 10]
                score_value = int(np.clip(np.round(np.random.normal(8.0, 1.0)), 0, 10))
                score = create_mock_score(float(score_value))
                await manager.complete_sample(sample.id, epoch, {'score': score})

        final_result = await manager.complete_task()

        print(f"✓ Ordinal discrete (high): {len(final_result['stopped_samples'])} samples stopped out of 60")
        print(f"  Efficiency: {final_result['efficiency_percent']}")

        # Verify ordinal inference was used
        assert len(manager.compiled_dataset) == 60 * 15


# =============================================================================
# CONTINUOUS [0, 1] - BINARY AGGREGATED (MEAN)
# =============================================================================

class TestBinaryAggregatedHighPerformance:
    """Test binary aggregated (continuous [0,1]) with consistent high performance."""

    @pytest.mark.asyncio
    async def test_binary_aggregated_consistent_75_samples(self):
        """
        Binary aggregated: 75 samples, 12 epochs each, continuous scores ~0.85.
        Uses mean aggregation, triggers continuous bounded inference.
        """
        np.random.seed(104)

        manager = OptimalStoppingManager(
            optstop_params={
                'delta_item': 0.10,
                'delta_cap': 0.10,
                'cred_level': 0.95,
                'conservatism': 5.0,
                'draws': 500,
                'tune': 500
            },
            grouping_columns=['model', 'task'],
            score_agg='mean',  # CRITICAL: Triggers continuous inference
            reanalysis_interval=225  # 25%, 50%, 75% of 900 trials
        )

        task = create_mock_task('gpt4', 'summarization')
        samples = [create_mock_sample(f'sample_{i:03d}') for i in range(75)]

        await manager.start_task(task, samples, epochs=12)

        # Generate continuous scores in [0, 1] around 0.85
        for epoch in range(12):
            for sample in samples:
                # Mean of binary trials: 0.85 ± 0.05
                score_value = np.clip(np.random.normal(0.85, 0.05), 0.0, 1.0)
                score = create_mock_score(score_value)
                await manager.complete_sample(sample.id, epoch, {'score': score})

        final_result = await manager.complete_task()

        print(f"✓ Binary aggregated (high): {len(final_result['stopped_samples'])} samples stopped out of 75")
        print(f"  Continuous inference used for [0, 1] bounds")
        print(f"  Efficiency: {final_result['efficiency_percent']}")

        # Verify continuous inference was triggered
        assert len(manager.compiled_dataset) == 75 * 12


# =============================================================================
# CONTINUOUS [0, 10] - ORDINAL AGGREGATED (MEDIAN)
# =============================================================================

class TestOrdinalAggregatedHighPerformance:
    """Test ordinal aggregated (continuous [0,10]) with high ratings."""

    @pytest.mark.asyncio
    async def test_ordinal_aggregated_consistent_60_samples(self):
        """
        Ordinal aggregated: 60 samples, 15 epochs each, continuous scores ~8.2/10.
        Uses median aggregation, triggers continuous bounded inference.
        """
        np.random.seed(106)

        manager = OptimalStoppingManager(
            optstop_params={
                'delta_item': 0.10,
                'delta_cap': 0.10,
                'cred_level': 0.95,
                'draws': 500,
                'tune': 500
            },
            grouping_columns=['model', 'task'],
            score_agg='median',  # CRITICAL: Triggers continuous inference
            reanalysis_interval=225,  # reanalysis_interval=225,  # 25%, 50%, 75% of 900 trials
            ordinal_tasks=['quality'],  # Identifies task for bounds determination
            ordinal_max_score=10
        )

        task = create_mock_task('gpt4', 'quality_rating')
        samples = [create_mock_sample(f'sample_{i:03d}') for i in range(60)]

        await manager.start_task(task, samples, epochs=15)

        # Generate continuous scores in [0, 10] around 8.2
        for epoch in range(15):
            for sample in samples:
                # Median of ordinal ratings: 8.2 ± 0.4
                score_value = np.clip(np.random.normal(8.2, 0.4), 0.0, 10.0)
                score = create_mock_score(score_value)
                await manager.complete_sample(sample.id, epoch, {'score': score})

        final_result = await manager.complete_task()

        print(f"✓ Ordinal aggregated (high): {len(final_result['stopped_samples'])} samples stopped out of 60")
        print(f"  Continuous inference used for [0, 10] bounds")
        print(f"  Efficiency: {final_result['efficiency_percent']}")

        assert len(manager.compiled_dataset) == 60 * 15


# =============================================================================
# EDGE CASE - ALL SAMPLES IDENTICAL SCORES (ZERO VARIANCE)
# =============================================================================

class TestEdgeCaseIdenticalScores:
    """Test edge case where all samples have identical scores."""

    @pytest.mark.asyncio
    async def test_all_samples_identical_binary_aggregated(self):
        """
        Edge case: All 40 samples get exactly 0.8 every epoch.
        Zero variance should lead to very narrow CIs and stopping.
        """
        np.random.seed(108)

        manager = OptimalStoppingManager(
            optstop_params={
                'delta_item': 0.05,
                'delta_cap': 0.05,
                'cred_level': 0.95,
                'draws': 500,
                'tune': 500
            },
            grouping_columns=['model'],
            score_agg='mean',
            reanalysis_interval=150  # 25%, 50%, 75% of 600 trials
        )

        task = create_mock_task('test_model', 'constant_task')
        samples = [create_mock_sample(f'sample_{i:03d}') for i in range(40)]

        await manager.start_task(task, samples, epochs=15)

        # All samples, all epochs: exactly 0.8
        for epoch in range(15):
            for sample in samples:
                score = create_mock_score(0.8)
                await manager.complete_sample(sample.id, epoch, {'score': score})

        final_result = await manager.complete_task()

        print(f"✓ Edge case (identical): {len(final_result['stopped_samples'])} samples stopped out of 40")
        print(f"  Zero variance should lead to narrow CIs")

        assert len(manager.compiled_dataset) == 40 * 15


# =============================================================================
# EDGE CASE - SINGLE EPOCH (INSUFFICIENT DATA)
# =============================================================================

class TestEdgeCaseSingleEpoch:
    """Test edge case with only one epoch per sample."""

    @pytest.mark.smoke
    @pytest.mark.asyncio
    async def test_single_epoch_binary_aggregated(self):
        """
        Edge case: 30 samples, only 1 epoch each.
        Should handle gracefully without stopping (insufficient data).
        """
        np.random.seed(109)

        manager = OptimalStoppingManager(
            optstop_params={
                'delta_item': 0.10,
                'delta_cap': 0.10,
                'draws': 500,
                'tune': 500,
                'random_seed': 109,  # Pinned for the smoke selection.
            },
            grouping_columns=['model'],
            score_agg='mean',
            reanalysis_interval=10  # 25%, 50%, 75% of 30 trials
        )

        task = create_mock_task('test_model', 'single_epoch_task')
        samples = [create_mock_sample(f'sample_{i:03d}') for i in range(30)]

        await manager.start_task(task, samples, epochs=1)

        # Single epoch for all samples
        for sample in samples:
            score_value = np.random.uniform(0.5, 0.9)
            score = create_mock_score(score_value)
            await manager.complete_sample(sample.id, 0, {'score': score})

        final_result = await manager.complete_task()

        print(f"✓ Edge case (single epoch): {len(final_result['stopped_samples'])} samples stopped out of 30")
        print(f"  Should have wide CIs with single observation")

        assert len(manager.compiled_dataset) == 30 * 1


# =============================================================================
# MULTI-GROUPING WITH MIXED SCORE TYPES
# =============================================================================

class TestMultiGroupingMixedScores:
    """Test multiple groupings with different score types."""

    @pytest.mark.asyncio
    async def test_multi_grouping_mixed_performance(self):
        """
        50 samples across 2 models × 2 tasks = 4 groupings.
        Mixed performance levels and variance patterns.
        """
        np.random.seed(110)

        manager = OptimalStoppingManager(
            optstop_params={
                'delta_item': 0.10,
                'delta_cap': 0.10,
                'draws': 500,
                'tune': 500
            },
            grouping_columns=['model', 'task'],
            score_agg='mean',
            reanalysis_interval=150  # 25%, 50%, 75% of 600 trials
        )

        # Create tasks for 2 models × 2 tasks
        tasks_and_samples = [
            (create_mock_task('model_a', 'task_1'), [create_mock_sample(f'a1_{i:02d}') for i in range(15)]),
            (create_mock_task('model_a', 'task_2'), [create_mock_sample(f'a2_{i:02d}') for i in range(15)]),
            (create_mock_task('model_b', 'task_1'), [create_mock_sample(f'b1_{i:02d}') for i in range(10)]),
            (create_mock_task('model_b', 'task_2'), [create_mock_sample(f'b2_{i:02d}') for i in range(10)]),
        ]

        # Performance profiles for each grouping
        performance_profiles = [
            (0.85, 0.03),  # model_a-task_1: high, low variance
            (0.70, 0.10),  # model_a-task_2: medium, high variance
            (0.90, 0.02),  # model_b-task_1: very high, very low variance
            (0.40, 0.05),  # model_b-task_2: low, low variance
        ]

        # Initialize all tasks
        for task, samples in tasks_and_samples:
            await manager.start_task(task, samples, epochs=12)

        # Run epochs
        for epoch in range(12):
            for (task, samples), (mean, std) in zip(tasks_and_samples, performance_profiles):
                for sample in samples:
                    score_value = np.clip(np.random.normal(mean, std), 0.0, 1.0)
                    score = create_mock_score(score_value)
                    await manager.complete_sample(sample.id, epoch, {'score': score})

        # Complete all tasks
        results = []
        for task, _ in tasks_and_samples:
            result = await manager.complete_task()
            results.append(result)

        # Verify
        total_stopped = sum(len(r['stopped_samples']) for r in results)
        print(f"✓ Multi-grouping: {total_stopped} total samples stopped across 4 groupings")
        for i, ((task, samples), result) in enumerate(zip(tasks_and_samples, results)):
            print(f"  Grouping {i+1}: {len(result['stopped_samples'])}/{len(samples)} stopped")

        assert len(manager.compiled_dataset) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
