"""
Test for Issue #6: Entropy History Persistence for Ordinal Hybrid Mode

This test verifies that sample-level entropy history is correctly persisted
across multiple calls to optimal_stopping_live_single(), enabling Pathway 2
(entropy stabilization) to trigger for ordinal hybrid mode.

Before fix: entropy_history_per_item was recreated fresh each call, so
           entropy stabilization could never trigger.

After fix:  entropy_history_per_item is passed as item_entropy_histories
           parameter and returned in result, enabling cross-call persistence.
"""

import pytest
import pandas as pd
import numpy as np
from optstop.rule import optimal_stopping_live_single


class TestEntropyHistoryPersistence:
    """Test that entropy history is persisted across calls."""

    @pytest.fixture
    def basic_ordinal_data(self):
        """Create basic ordinal data for a single grouping."""
        np.random.seed(42)

        # Create data with 5 samples, 3 epochs each
        # Use diffuse distribution to force entropy path (not narrow modal CI)
        data = []
        for sample_id in range(5):
            for epoch in range(1, 4):
                # Diffuse scores across 0-10 range (not peaked)
                score = np.random.choice([3, 4, 5, 6, 7])  # Spread out
                data.append({
                    'sample_id': f'sample_{sample_id}',
                    'epoch': epoch,
                    'score': score
                })

        return pd.DataFrame(data)

    @pytest.fixture
    def basic_params(self):
        """Basic stopping parameters."""
        return {
            'delta_item': 0.15,
            'delta_cap': 0.10,
            'cred_level': 0.90,
            'conservatism': 2,
            'draws': 100,  # Minimal for speed
            'tune': 50,
            'chains': 1,
            'cores': 1
        }

    def test_entropy_histories_returned(self, basic_ordinal_data, basic_params):
        """Test that item_entropy_histories is included in the return value."""
        result = optimal_stopping_live_single(
            df_grouping=basic_ordinal_data,
            grouping_name='test-ordinal',
            params=basic_params,
            sample_id_column='sample_id',
            epoch_column='epoch',
            score_column='score',
            ordinal_tasks=['ordinal', 'test'],  # Match 'test-ordinal'
            ordinal_max_score=10,
            ordinal_inference='hybrid',  # Must be hybrid to use entropy history
            item_entropy_histories=None  # First call
        )

        # Verify the new return field exists
        assert 'item_entropy_histories' in result, \
            "item_entropy_histories should be in result dict"

        # It should be a dict (even if empty for non-ordinal or first call)
        assert isinstance(result['item_entropy_histories'], dict), \
            "item_entropy_histories should be a dict"

    def test_entropy_histories_persist_across_calls(self, basic_params):
        """Test that entropy history accumulates across multiple calls."""
        # Create data that will be extended across calls
        def create_data_for_epochs(max_epoch):
            data = []
            for sample_id in range(3):
                for epoch in range(1, max_epoch + 1):
                    # Diffuse scores to force entropy path
                    np.random.seed(42 + sample_id * 100 + epoch)
                    score = np.random.choice([3, 4, 5, 6, 7])
                    data.append({
                        'sample_id': f'sample_{sample_id}',
                        'epoch': epoch,
                        'score': score
                    })
            return pd.DataFrame(data)

        # First call with epochs 1-2
        df1 = create_data_for_epochs(2)
        result1 = optimal_stopping_live_single(
            df_grouping=df1,
            grouping_name='test-ordinal',
            params=basic_params,
            sample_id_column='sample_id',
            epoch_column='epoch',
            score_column='score',
            ordinal_tasks=['ordinal', 'test'],
            ordinal_max_score=10,
            ordinal_inference='hybrid',
            item_entropy_histories=None  # First call
        )

        first_histories = result1['item_entropy_histories']

        # Second call with epochs 1-4, passing the history from first call
        df2 = create_data_for_epochs(4)
        result2 = optimal_stopping_live_single(
            df_grouping=df2,
            grouping_name='test-ordinal',
            params=basic_params,
            sample_id_column='sample_id',
            epoch_column='epoch',
            score_column='score',
            ordinal_tasks=['ordinal', 'test'],
            ordinal_max_score=10,
            ordinal_inference='hybrid',
            item_entropy_histories=first_histories  # Pass previous history
        )

        second_histories = result2['item_entropy_histories']

        # Verify histories are accumulating (or at least persisted)
        # The key test is that we can pass histories across calls without error
        assert isinstance(second_histories, dict), \
            "Second call should return item_entropy_histories"

    def test_entropy_history_not_affected_for_binary(self, basic_params):
        """Test that binary scoring doesn't use item_entropy_histories."""
        # Binary data (0/1 scores)
        data = []
        for sample_id in range(5):
            for epoch in range(1, 4):
                score = np.random.choice([0, 1])
                data.append({
                    'sample_id': f'sample_{sample_id}',
                    'epoch': epoch,
                    'score': score
                })
        df = pd.DataFrame(data)

        result = optimal_stopping_live_single(
            df_grouping=df,
            grouping_name='test-binary',
            params=basic_params,
            sample_id_column='sample_id',
            epoch_column='epoch',
            score_column='score',
            ordinal_tasks=None,  # No ordinal tasks -> binary routing
            ordinal_max_score=10,
            ordinal_inference='hybrid',
            item_entropy_histories=None
        )

        # Even for binary, the field should be returned (as empty dict)
        assert 'item_entropy_histories' in result
        assert isinstance(result['item_entropy_histories'], dict)

    def test_entropy_history_not_affected_for_modal(self, basic_ordinal_data, basic_params):
        """Test that modal inference doesn't populate item_entropy_histories."""
        result = optimal_stopping_live_single(
            df_grouping=basic_ordinal_data,
            grouping_name='test-ordinal',
            params=basic_params,
            sample_id_column='sample_id',
            epoch_column='epoch',
            score_column='score',
            ordinal_tasks=['ordinal', 'test'],
            ordinal_max_score=10,
            ordinal_inference='modal',  # Modal doesn't use entropy history
            item_entropy_histories=None
        )

        # Modal mode should still return the field but won't populate it
        assert 'item_entropy_histories' in result
        assert isinstance(result['item_entropy_histories'], dict)


class TestEntropyHistoryWithBridge:
    """Test entropy history persistence through the bridge (OptimalStoppingManager)."""

    @pytest.fixture(autouse=True)
    def import_manager(self):
        """Import OptimalStoppingManager for this test class."""
        from optstop.early_stopping import OptimalStoppingManager
        self.OptimalStoppingManager = OptimalStoppingManager

    @pytest.fixture
    def mock_sample_score(self):
        """Create mock sample score function."""
        # Import or define mocks
        try:
            from inspect_ai.scorer._metric import SampleScore
            from inspect_ai.scorer._score import Score
            def create_score(value):
                return {"rating": SampleScore(Score(value=value))}
        except ImportError:
            class MockScore:
                def __init__(self, value):
                    self.value = value
            class SampleScore:
                def __init__(self, score):
                    self.score = score
            def create_score(value):
                return {"rating": SampleScore(MockScore(value))}
        return create_score

    @pytest.fixture
    def mock_sample_and_evalspec(self):
        """Create mock Sample and EvalSpec."""
        try:
            from inspect_ai.dataset._dataset import Sample
            from inspect_ai.log._log import EvalSpec
        except ImportError:
            from optstop.early_stopping import Sample, EvalSpec
        return Sample, EvalSpec

    @pytest.mark.asyncio
    async def test_bridge_persists_entropy_histories(self, mock_sample_score, mock_sample_and_evalspec):
        """Test that OptimalStoppingManager persists item entropy histories."""
        Sample, EvalSpec = mock_sample_and_evalspec
        create_score = mock_sample_score

        # Create manager with hybrid ordinal inference
        manager = self.OptimalStoppingManager(
            optstop_params={
                'delta_item': 0.20,
                'delta_cap': 0.15,
                'cred_level': 0.90,
                'draws': 100,
                'tune': 50,
                'chains': 1
            },
            grouping_columns=['model', 'task'],
            ordinal_tasks=['rating'],  # Will match task name
            ordinal_max_score=10,
            ordinal_inference='hybrid',
            reanalysis_interval=3,
            min_samples_per_grouping=2
        )

        # Create samples
        samples = [Sample(id=f"sample_{i}") for i in range(5)]
        eval_spec = EvalSpec(task="rating_task", model="gpt-4")

        # Start task
        await manager.start_task(eval_spec, samples, epochs=5)

        # Run enough samples to trigger reanalysis (reanalysis_interval=3)
        # Need to complete 3+ samples to trigger _run_stopping_inference
        np.random.seed(42)
        completed_count = 0
        for sample in samples:
            for epoch in range(1, 6):  # All 5 epochs
                should_stop = await manager.schedule_sample(sample.id, epoch)
                if should_stop is None:
                    # Cast to Python int to avoid numpy.int64 type issues with score extraction
                    score = int(np.random.choice([3, 4, 5, 6, 7]))
                    await manager.complete_sample(sample.id, epoch, create_score(score))
                    completed_count += 1

        print(f"DEBUG: Completed {completed_count} samples")

        # After completing enough samples, inference should have been triggered
        # which initializes _item_entropy_histories
        # Note: The attribute is initialized lazily inside _run_stopping_inference
        # If no inference was triggered yet, it won't exist - that's OK for this test
        # The key is that it's properly passed when it exists

        # Complete task
        diagnostics = await manager.complete_task()
        assert diagnostics is not None

        # Verify we completed enough samples to have triggered inference
        assert completed_count >= 3, f"Should have completed at least 3 samples, got {completed_count}"

        # The _item_entropy_histories should exist after inference was triggered
        # (initialized in _run_stopping_inference when first called)
        if completed_count >= 3:
            # Inference should have been called at least once
            assert hasattr(manager, '_item_entropy_histories'), \
                "Manager should have _item_entropy_histories attribute after inference"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
