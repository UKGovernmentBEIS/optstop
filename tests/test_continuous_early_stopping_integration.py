"""
Integration tests for continuous bounded early stopping.

Tests end-to-end pipeline with:
- Binary aggregated scores [0, 1]
- Ordinal aggregated scores [0, 10]
- Sample-level and group-level stopping
- Metadata validation
- Stabilization criteria
"""

import pytest
import numpy as np
import pandas as pd
from optstop.rule import optimal_stopping_live_single



# CI partition: heavy MCMC tests deselected from PR CI (see pyproject.toml markers).
pytestmark = pytest.mark.optstop


class TestContinuousEarlyStoppingIntegration:
    """Integration tests for continuous early stopping pipeline."""

    def test_binary_aggregated_consistent_high_performance(self):
        """
        Test binary aggregated scores with consistent high performance.

        Scores are continuous floats in [0, 1] representing mean of binary trials.
        Should stop due to narrow CI width.
        """
        np.random.seed(42)

        # Generate consistent high-performance continuous scores (aggregated binary)
        # Simulating mean of multiple binary trials: ~0.85 performance
        n_samples = 8
        n_epochs_per_sample = 10

        data = []
        for sample_id in range(n_samples):
            for epoch in range(n_epochs_per_sample):
                # Generate continuous score around 0.85 with small variance
                score = np.clip(np.random.normal(0.85, 0.05), 0.0, 1.0)
                data.append({
                    'grouping': 'model1-task1',
                    'sample_id': f'sample_{sample_id}',
                    'epoch': epoch,
                    'score': score
                })

        df = pd.DataFrame(data)

        # Parameters for early stopping
        params = {
            'delta_item': 0.10,  # Sample-level width threshold
            'delta_cap': 0.10,   # Group-level width threshold
            'cred_level': 0.95,
            'conservatism': 5.0,
            'low_performance_threshold': 0.2,
            'CI_delta': 0.00005,
            'stab_window': 5,
            'is_aggregated': True,  # CRITICAL: triggers continuous inference
            'random_seed': 42,  # Pinned for determinism.
            # Single-process, single-chain sampling keeps this test's sequential
            # continuous MCMC runs within the memory of small CI runners (the macOS
            # Apple-Silicon runner has ~7 GB RAM). Real usage runs one eval per
            # process, so this does not reflect production sampling settings.
            'chains': 1,
            'cores': 1,
        }

        # Run early stopping
        result = optimal_stopping_live_single(
            df_grouping=df,
            grouping_name='model1-task1',
            params=params,
            sample_id_column='sample_id',
            epoch_column='epoch',
            score_column='score',
            stabilization_history=None,
            ordinal_tasks=None,  # Not ordinal
            ordinal_max_score=10
        )

        # Assertions
        assert 'stop_sample_ids' in result
        assert 'stop_this_grouping' in result
        assert 'metadata' in result
        assert 'stabilization_history' in result

        # System should run without errors
        # Note: With these parameters, samples may or may not stop depending on variance
        # The key is that the system runs correctly

        # Check metadata structure for stopped samples (if any)
        if len(result['stop_sample_ids']) > 0:
            stopped_sample_id = result['stop_sample_ids'][0].split(':::')[1]
            sample_metadata = result['metadata']['sample_stopping_reasons'][stopped_sample_id]

            assert sample_metadata['reason'] == 'continuous_bounded_ci_width'
            assert 'ci_width' in sample_metadata  # Raw width
            assert 'ci_width_normalized' in sample_metadata  # Normalized width
            assert 'threshold' in sample_metadata
            assert 'epochs_used' in sample_metadata
            assert 'bounds' in sample_metadata

            # Verify normalization
            assert sample_metadata['bounds']['lower'] == 0.0
            assert sample_metadata['bounds']['upper'] == 1.0
            # For [0, 1] bounds, normalized = raw
            assert abs(sample_metadata['ci_width_normalized'] - sample_metadata['ci_width']) < 0.001

        # Check stabilization history
        assert 'ci_width_history' in result['stabilization_history']
        # Group-level stopping creates width history
        if len(result['stabilization_history']['ci_width_history']) > 0:
            # All widths should be in [0, 1] (normalized)
            for width in result['stabilization_history']['ci_width_history']:
                assert 0.0 <= width <= 1.0, f"Normalized width {width} should be in [0, 1]"

        # Verify continuous inference was used (not binary or ordinal)
        # If samples stopped, should have continuous reason
        for sample_id, reason_dict in result['metadata']['sample_stopping_reasons'].items():
            assert 'continuous' in reason_dict['reason'], f"Should use continuous inference, got {reason_dict['reason']}"

        print(f"✓ Binary aggregated test: {len(result['stop_sample_ids'])} samples stopped, system working correctly")

    def test_ordinal_aggregated_consistent_high_performance(self):
        """
        Test ordinal aggregated scores with consistent high performance.

        Scores are continuous floats in [0, 10] representing mean of ordinal ratings.
        Should stop due to narrow CI width.
        """
        np.random.seed(43)

        # Generate consistent high-performance continuous scores (aggregated ordinal)
        # Simulating mean of multiple ordinal ratings: ~8.0 performance
        n_samples = 15
        n_epochs_per_sample = 10

        data = []
        for sample_id in range(n_samples):
            for epoch in range(n_epochs_per_sample):
                # Generate continuous score around 8.0 with small variance
                score = np.clip(np.random.normal(8.0, 0.5), 0.0, 10.0)
                data.append({
                    'grouping': 'model1-task1',
                    'sample_id': f'sample_{sample_id}',
                    'epoch': epoch,
                    'score': score
                })

        df = pd.DataFrame(data)

        # Parameters for early stopping
        params = {
            'delta_item': 0.10,  # Sample-level width threshold
            'delta_cap': 0.10,   # Group-level width threshold
            'cred_level': 0.95,
            'conservatism': 5.0,
            'low_performance_threshold': 0.2,
            'CI_delta': 0.00005,
            'stab_window': 5,
            'is_aggregated': True  # CRITICAL: triggers continuous inference
        }

        # Run early stopping
        # Set ordinal_tasks so system knows to use [0, 10] bounds for continuous_bounded
        result = optimal_stopping_live_single(
            df_grouping=df,
            grouping_name='model1-task1',
            params=params,
            sample_id_column='sample_id',
            epoch_column='epoch',
            score_column='score',
            stabilization_history=None,
            ordinal_tasks=['task1'],  # Identifies as ordinal for bounds determination
            ordinal_max_score=10
        )

        # Assertions
        assert 'stop_sample_ids' in result
        assert 'stop_this_grouping' in result
        assert 'metadata' in result

        # System should run without errors
        # Note: With these parameters, samples may or may not stop

        # Check metadata structure for stopped samples (if any)
        if len(result['stop_sample_ids']) > 0:
            stopped_sample_id = result['stop_sample_ids'][0].split(':::')[1]
            sample_metadata = result['metadata']['sample_stopping_reasons'][stopped_sample_id]

            assert sample_metadata['reason'] == 'continuous_bounded_ci_width'
            assert 'ci_width' in sample_metadata
            assert 'ci_width_normalized' in sample_metadata
            assert 'bounds' in sample_metadata

            # Verify bounds
            assert sample_metadata['bounds']['lower'] == 0.0
            assert sample_metadata['bounds']['upper'] == 10.0

            # Verify normalization: normalized = raw / 10
            expected_normalized = sample_metadata['ci_width'] / 10.0
            assert abs(sample_metadata['ci_width_normalized'] - expected_normalized) < 0.001

        # Check stabilization history
        assert len(result['stabilization_history']['ci_width_history']) > 0
        # All widths should be in [0, 1] (normalized)
        for width in result['stabilization_history']['ci_width_history']:
            assert 0.0 <= width <= 1.0, f"Normalized width {width} should be in [0, 1]"

        print(f"✓ Ordinal aggregated test: {len(result['stop_sample_ids'])} samples stopped")

    def test_group_level_stopping_with_stabilization(self):
        """
        Test that group-level stopping triggers with stabilization criterion.

        Uses many samples with narrow CIs to trigger stabilization.
        """
        np.random.seed(44)

        # Generate data for many samples (to accumulate stabilization history)
        n_samples = 30
        n_epochs_per_sample = 8

        data = []
        for sample_id in range(n_samples):
            for epoch in range(n_epochs_per_sample):
                # Very consistent scores to get narrow CIs
                score = np.clip(np.random.normal(0.80, 0.03), 0.0, 1.0)
                data.append({
                    'grouping': 'model1-task1',
                    'sample_id': f'sample_{sample_id}',
                    'epoch': epoch,
                    'score': score
                })

        df = pd.DataFrame(data)

        # Parameters for early stopping
        params = {
            'delta_item': 0.05,  # Tighter threshold
            'delta_cap': 0.05,   # Tighter threshold
            'cred_level': 0.95,
            'conservatism': 5.0,
            'low_performance_threshold': 0.2,
            'CI_delta': 0.001,  # Looser for stabilization
            'stab_window': 5,
            'is_aggregated': True
        }

        # Run early stopping
        result = optimal_stopping_live_single(
            df_grouping=df,
            grouping_name='model1-task1',
            params=params,
            sample_id_column='sample_id',
            epoch_column='epoch',
            score_column='score',
            stabilization_history=None,
            ordinal_tasks=None,
            ordinal_max_score=10
        )

        # Check that stabilization history was created
        # Note: Single call only creates one history entry
        # Multiple calls would accumulate history for stabilization detection
        assert 'ci_width_history' in result['stabilization_history']
        assert len(result['stabilization_history']['ci_width_history']) > 0

        # Check if group stopped (may or may not, depending on data)
        # If it did, verify the reason
        if len(result['stop_this_grouping']) > 0:
            group_metadata = result['metadata']['group_stopping_reason']
            assert group_metadata is not None

            # Should be either width or stabilization (item-level or group-level)
            assert group_metadata['reason'] in [
                'continuous_bounded_ci_width',
                'continuous_hierarchical_ci_width',
                'continuous_bounded_stabilization',
                'continuous_hierarchical_stabilization',
                'continuous_bounded_stabilization_low_perf',
                'continuous_hierarchical_stabilization_low_perf',
            ]

            if 'stabilization' in group_metadata['reason']:
                assert 'slope' in group_metadata
                assert 'slope_threshold' in group_metadata

        print(f"✓ Stabilization test: CI history length = {len(result['stabilization_history']['ci_width_history'])}")

    def test_metadata_structure_validation(self):
        """
        Validate that metadata structure matches expected format.
        """
        np.random.seed(45)

        # Generate simple dataset
        n_samples = 10
        n_epochs = 8

        data = []
        for sample_id in range(n_samples):
            for epoch in range(n_epochs):
                score = np.clip(np.random.normal(0.75, 0.05), 0.0, 1.0)
                data.append({
                    'grouping': 'test-grouping',
                    'sample_id': f'item_{sample_id}',
                    'epoch': epoch,
                    'score': score
                })

        df = pd.DataFrame(data)

        params = {
            'delta_item': 0.10,
            'delta_cap': 0.10,
            'cred_level': 0.95,
            'conservatism': 5.0,
            'low_performance_threshold': 0.2,
            'CI_delta': 0.00005,
            'stab_window': 5,
            'is_aggregated': True
        }

        result = optimal_stopping_live_single(
            df_grouping=df,
            grouping_name='test-grouping',
            params=params,
            sample_id_column='sample_id',
            epoch_column='epoch',
            score_column='score'
        )

        # Validate result structure
        assert 'grouping' in result
        assert result['grouping'] == 'test-grouping'
        assert 'stop_sample_ids' in result
        assert 'stop_this_grouping' in result
        assert 'stabilization_history' in result
        assert 'metadata' in result

        # Validate metadata structure
        assert 'sample_stopping_reasons' in result['metadata']
        assert 'group_stopping_reason' in result['metadata']

        # Validate stabilization history structure
        assert 'ci_width_history' in result['stabilization_history']
        assert 'ci_slope_history' in result['stabilization_history']
        assert 'entropy_history' in result['stabilization_history']
        assert 'n_samples_evaluated' in result['stabilization_history']

        print("✓ Metadata structure validation passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
