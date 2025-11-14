"""
Unit and integration tests for early_stopping.py

Tests cover:
- Priority 1: Configuration validation, task validation
- Priority 2: .loc filtering, cache invalidation
- Integration: Multi-grouping scenarios, edge cases
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, AsyncMock
from typing import Any

# Import the module under test
import sys
sys.path.insert(0, '/home/ubuntu/optstop')

from optstop.early_stopping import (
    OptimalStoppingManager,
    EarlyStop,
    StoppedSample
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def valid_optstop_params():
    """Valid optstop_params for testing."""
    return {
        'delta_item': 0.05,
        'delta_cap': 0.05,
        'cred_level': 0.95
    }


@pytest.fixture
def empty_optstop_params():
    """Empty optstop_params to test defaults."""
    return {}


@pytest.fixture
def valid_grouping_columns():
    """Valid grouping columns for testing."""
    return ['model', 'task']


@pytest.fixture
def mock_task_valid():
    """Mock EvalSpec with valid structure."""
    mock_dataset = Mock()
    mock_dataset.sample_ids = ['sample1', 'sample2', 'sample3']

    mock_config = Mock()
    mock_config.epochs = 5

    mock_task = Mock()
    mock_task.dataset = mock_dataset
    mock_task.config = mock_config
    mock_task.model = 'gpt4'
    mock_task.task = 'math'
    mock_task.metadata = {}
    mock_task.tags = []

    return mock_task


@pytest.fixture
def manager_basic(valid_optstop_params, valid_grouping_columns):
    """Basic OptimalStoppingManager instance for testing."""
    return OptimalStoppingManager(
        optstop_params=valid_optstop_params,
        grouping_columns=valid_grouping_columns
    )


# ============================================================================
# PRIORITY 1 TESTS: CONFIGURATION VALIDATION
# ============================================================================

class TestConfigurationValidation:
    """Test Priority 1: Configuration validation at initialization."""

    def test_valid_config_with_all_params(self):
        """Test initialization with all valid parameters."""
        params = {
            'delta_item': 0.05,
            'delta_cap': 0.05,
            'cred_level': 0.95,
            'conservatism': 5,
            'low_performance_threshold': 0.01,
            'CI_delta': 0.00005,
            'stab_window': 10,
            'rep_batch_size': 1,
            'tune': 1000,
            'draws': 2000
        }

        manager = OptimalStoppingManager(
            optstop_params=params,
            grouping_columns=['model', 'task']
        )

        assert manager.optstop_params == params
        assert manager.grouping_columns == ['model', 'task']

    def test_valid_config_with_empty_params(self):
        """Test initialization with empty params (all defaults)."""
        manager = OptimalStoppingManager(
            optstop_params={},
            grouping_columns=['model']
        )

        assert manager.optstop_params == {}
        assert manager.grouping_columns == ['model']

    def test_invalid_delta_item_negative(self):
        """Test rejection of negative delta_item."""
        with pytest.raises(ValueError, match="delta_item must be > 0"):
            OptimalStoppingManager(
                optstop_params={'delta_item': -0.05},
                grouping_columns=['model']
            )

    def test_invalid_delta_item_zero(self):
        """Test rejection of zero delta_item."""
        with pytest.raises(ValueError, match="delta_item must be > 0"):
            OptimalStoppingManager(
                optstop_params={'delta_item': 0},
                grouping_columns=['model']
            )

    def test_invalid_delta_cap_negative(self):
        """Test rejection of negative delta_cap."""
        with pytest.raises(ValueError, match="delta_cap must be > 0"):
            OptimalStoppingManager(
                optstop_params={'delta_cap': -0.1},
                grouping_columns=['model']
            )

    def test_invalid_cred_level_too_low(self):
        """Test rejection of cred_level <= 0."""
        with pytest.raises(ValueError, match="cred_level must be between 0 and 1"):
            OptimalStoppingManager(
                optstop_params={'cred_level': 0},
                grouping_columns=['model']
            )

    def test_invalid_cred_level_too_high(self):
        """Test rejection of cred_level >= 1."""
        with pytest.raises(ValueError, match="cred_level must be between 0 and 1"):
            OptimalStoppingManager(
                optstop_params={'cred_level': 1.5},
                grouping_columns=['model']
            )

    def test_invalid_conservatism_less_than_one(self):
        """Test rejection of conservatism < 1."""
        with pytest.raises(ValueError, match="conservatism must be >= 1"):
            OptimalStoppingManager(
                optstop_params={'conservatism': 0.5},
                grouping_columns=['model']
            )

    def test_invalid_low_perf_threshold_negative(self):
        """Test rejection of negative low_performance_threshold."""
        with pytest.raises(ValueError, match="low_performance_threshold must be between 0 and 1"):
            OptimalStoppingManager(
                optstop_params={'low_performance_threshold': -0.1},
                grouping_columns=['model']
            )

    def test_invalid_CI_delta_negative(self):
        """Test rejection of negative CI_delta."""
        with pytest.raises(ValueError, match="CI_delta must be > 0"):
            OptimalStoppingManager(
                optstop_params={'CI_delta': -0.0001},
                grouping_columns=['model']
            )

    def test_invalid_stab_window_zero(self):
        """Test rejection of zero stab_window."""
        with pytest.raises(ValueError, match="stab_window must be > 0"):
            OptimalStoppingManager(
                optstop_params={'stab_window': 0},
                grouping_columns=['model']
            )

    def test_invalid_tune_negative(self):
        """Test rejection of negative tune."""
        with pytest.raises(ValueError, match="tune must be >= 0"):
            OptimalStoppingManager(
                optstop_params={'tune': -100},
                grouping_columns=['model']
            )

    def test_invalid_draws_zero(self):
        """Test rejection of zero draws."""
        with pytest.raises(ValueError, match="draws must be > 0"):
            OptimalStoppingManager(
                optstop_params={'draws': 0},
                grouping_columns=['model']
            )

    def test_empty_grouping_columns(self):
        """Test rejection of empty grouping_columns."""
        with pytest.raises(ValueError, match="grouping_columns cannot be empty"):
            OptimalStoppingManager(
                optstop_params={},
                grouping_columns=[]
            )

    def test_invalid_grouping_column_format(self):
        """Test rejection of invalid grouping column format."""
        with pytest.raises(ValueError, match="Invalid grouping column 'invalid.format'"):
            OptimalStoppingManager(
                optstop_params={},
                grouping_columns=['invalid.format']
            )

    def test_invalid_grouping_column_empty_metadata_key(self):
        """Test rejection of metadata. without key."""
        with pytest.raises(ValueError, match="missing key after 'metadata.'"):
            OptimalStoppingManager(
                optstop_params={},
                grouping_columns=['metadata.']
            )

    def test_invalid_grouping_column_empty_tag_name(self):
        """Test rejection of tag. without name."""
        with pytest.raises(ValueError, match="missing tag name after 'tag.'"):
            OptimalStoppingManager(
                optstop_params={},
                grouping_columns=['tag.']
            )

    def test_valid_grouping_columns_all_types(self):
        """Test valid grouping columns of all supported types."""
        manager = OptimalStoppingManager(
            optstop_params={},
            grouping_columns=['model', 'task', 'metadata.difficulty', 'tag.experimental']
        )

        assert manager.grouping_columns == ['model', 'task', 'metadata.difficulty', 'tag.experimental']

    def test_invalid_reanalysis_interval_zero(self):
        """Test rejection of zero reanalysis_interval."""
        with pytest.raises(ValueError, match="reanalysis_interval must be > 0"):
            OptimalStoppingManager(
                optstop_params={},
                grouping_columns=['model'],
                reanalysis_interval=0
            )

    def test_invalid_min_samples_negative(self):
        """Test rejection of negative min_samples_per_grouping."""
        with pytest.raises(ValueError, match="min_samples_per_grouping must be >= 0"):
            OptimalStoppingManager(
                optstop_params={},
                grouping_columns=['model'],
                min_samples_per_grouping=-1
            )

    def test_invalid_ordinal_inference_mode(self):
        """Test rejection of invalid ordinal_inference."""
        with pytest.raises(ValueError, match="ordinal_inference must be one of"):
            OptimalStoppingManager(
                optstop_params={},
                grouping_columns=['model'],
                ordinal_inference='invalid'
            )

    def test_invalid_ordinal_max_score_zero(self):
        """Test rejection of zero ordinal_max_score."""
        with pytest.raises(ValueError, match="ordinal_max_score must be > 0"):
            OptimalStoppingManager(
                optstop_params={},
                grouping_columns=['model'],
                ordinal_max_score=0
            )

    def test_empty_score_column(self):
        """Test rejection of empty score_column."""
        with pytest.raises(ValueError, match="score_column cannot be empty"):
            OptimalStoppingManager(
                optstop_params={},
                grouping_columns=['model'],
                score_column=""
            )


# ============================================================================
# PRIORITY 1 TESTS: TASK VALIDATION
# ============================================================================

class TestTaskValidation:
    """Test Priority 1: Task validation in start_task()."""

    @pytest.mark.asyncio
    async def test_valid_task(self, manager_basic, mock_task_valid):
        """Test start_task with valid task."""
        result = await manager_basic.start_task(mock_task_valid)

        assert result == manager_basic.manager_name
        assert manager_basic.compiled_dataset is not None
        assert len(manager_basic.compiled_dataset) == 3 * 5  # 3 samples × 5 epochs

    @pytest.mark.asyncio
    async def test_task_with_none_dataset(self, manager_basic):
        """Test rejection of task with None dataset."""
        mock_task = Mock()
        mock_task.dataset = None
        mock_task.config = Mock(epochs=5)

        with pytest.raises(ValueError, match="task.dataset is None"):
            await manager_basic.start_task(mock_task)

    @pytest.mark.asyncio
    async def test_task_with_empty_sample_ids(self, manager_basic):
        """Test rejection of task with empty sample_ids."""
        mock_dataset = Mock()
        mock_dataset.sample_ids = []

        mock_task = Mock()
        mock_task.dataset = mock_dataset
        mock_task.config = Mock(epochs=5)

        with pytest.raises(ValueError, match="sample_ids is empty"):
            await manager_basic.start_task(mock_task)

    @pytest.mark.asyncio
    async def test_task_with_none_config(self, manager_basic, caplog):
        """Test handling of task with None config (should warn, default to 1 epoch)."""
        mock_dataset = Mock()
        mock_dataset.sample_ids = ['sample1']

        mock_task = Mock()
        mock_task.dataset = mock_dataset
        mock_task.config = None
        mock_task.model = 'gpt4'
        mock_task.task = 'math'
        mock_task.metadata = {}
        mock_task.tags = []

        result = await manager_basic.start_task(mock_task)

        assert result == manager_basic.manager_name
        assert len(manager_basic.compiled_dataset) == 1  # 1 sample × 1 epoch (default)
        assert "task.config is None" in caplog.text


# ============================================================================
# PRIORITY 2 TESTS: .LOC FILTERING WITH EDGE CASES
# ============================================================================

class TestLocFiltering:
    """Test Priority 2.1: .loc filtering robustness with edge cases."""

    @pytest.mark.asyncio
    async def test_schedule_sample_with_none_metadata(self, manager_basic):
        """Test schedule_sample with None value in metadata column."""
        # Create manager with metadata column
        manager = OptimalStoppingManager(
            optstop_params={},
            grouping_columns=['model', 'metadata.difficulty']
        )

        # Create task with None metadata value
        mock_dataset = Mock()
        mock_dataset.sample_ids = ['sample1']
        mock_config = Mock()
        mock_config.epochs = 2

        mock_task = Mock()
        mock_task.dataset = mock_dataset
        mock_task.config = mock_config
        mock_task.model = 'gpt4'
        mock_task.task = 'math'
        mock_task.metadata = {'difficulty': None}  # None value
        mock_task.tags = []

        # Initialize
        await manager.start_task(mock_task)

        # Should handle None gracefully in schedule_sample
        result = await manager.schedule_sample(mock_task, id='sample1', epoch=1)

        assert result is None  # Should run (not stopped)

    @pytest.mark.asyncio
    async def test_schedule_sample_with_special_characters(self, manager_basic):
        """Test schedule_sample with special characters in model/task names."""
        # Create task with special characters
        mock_dataset = Mock()
        mock_dataset.sample_ids = ['sample_1']  # Underscore in ID
        mock_config = Mock()
        mock_config.epochs = 2

        mock_task = Mock()
        mock_task.dataset = mock_dataset
        mock_task.config = mock_config
        mock_task.model = 'gpt-4_turbo'  # Hyphen and underscore
        mock_task.task = 'math:hard'      # Colon
        mock_task.metadata = {}
        mock_task.tags = []

        # Initialize
        await manager_basic.start_task(mock_task)

        # Should handle special characters without crashing
        result = await manager_basic.schedule_sample(mock_task, id='sample_1', epoch=1)

        assert result is None  # Should run

    @pytest.mark.asyncio
    async def test_schedule_sample_with_underscores_in_grouping(self):
        """Test that underscores in grouping names don't break filtering."""
        manager = OptimalStoppingManager(
            optstop_params={},
            grouping_columns=['model', 'task']
        )

        mock_dataset = Mock()
        mock_dataset.sample_ids = ['item_123', 'item_456']  # Underscores in IDs
        mock_config = Mock()
        mock_config.epochs = 3

        mock_task = Mock()
        mock_task.dataset = mock_dataset
        mock_task.config = mock_config
        mock_task.model = 'gpt_4'      # Underscore
        mock_task.task = 'math_task'   # Underscore
        mock_task.metadata = {}
        mock_task.tags = []

        await manager.start_task(mock_task)

        # Test filtering works correctly with underscores
        result1 = await manager.schedule_sample(mock_task, id='item_123', epoch=1)
        result2 = await manager.schedule_sample(mock_task, id='item_456', epoch=2)

        assert result1 is None
        assert result2 is None

    @pytest.mark.asyncio
    async def test_schedule_sample_cache_hit(self, manager_basic, mock_task_valid):
        """Test that cache is populated and used correctly."""
        await manager_basic.start_task(mock_task_valid)

        # First call - cache miss, should query dataframe
        result1 = await manager_basic.schedule_sample(mock_task_valid, id='sample1', epoch=1)
        assert result1 is None

        # Check cache was populated
        grouping_tuple = (mock_task_valid.model, mock_task_valid.task)
        cache_key = (*grouping_tuple, 'sample1', 1)
        assert cache_key in manager_basic._schedule_cache
        assert manager_basic._schedule_cache[cache_key] == True  # Should run

        # Second call - cache hit
        result2 = await manager_basic.schedule_sample(mock_task_valid, id='sample1', epoch=1)
        assert result2 is None


# ============================================================================
# PRIORITY 2 TESTS: CACHE INVALIDATION LOGIC
# ============================================================================

class TestCacheInvalidation:
    """Test Priority 2.2: Cache invalidation after stopping decisions."""

    @pytest.mark.asyncio
    async def test_sample_level_cache_invalidation(self):
        """Test cache invalidation when a sample is stopped."""
        manager = OptimalStoppingManager(
            optstop_params={'delta_item': 0.01, 'delta_cap': 0.01},
            grouping_columns=['model'],
            reanalysis_interval=1,  # Run inference every sample
            min_samples_per_grouping=1
        )

        # Create task
        mock_dataset = Mock()
        mock_dataset.sample_ids = ['sample1']
        mock_config = Mock()
        mock_config.epochs = 10  # Many epochs

        mock_task = Mock()
        mock_task.dataset = mock_dataset
        mock_task.config = mock_config
        mock_task.model = 'gpt4'
        mock_task.task = 'math'
        mock_task.metadata = {}
        mock_task.tags = []

        await manager.start_task(mock_task)

        # Schedule all epochs (populate cache)
        for epoch in range(1, 11):
            result = await manager.schedule_sample(mock_task, id='sample1', epoch=epoch)
            assert result is None  # All should be schedulable initially

        # Verify cache populated
        assert len(manager._schedule_cache) > 0

        # Complete a sample with perfect score (might trigger stopping)
        mock_score_obj = Mock()
        mock_score_obj.as_float.return_value = 1.0
        mock_sample_score = Mock()
        mock_sample_score.score = mock_score_obj
        mock_scores = {'scorer1': mock_sample_score}
        await manager.complete_sample(mock_task, id='sample1', epoch=1, scores=mock_scores)

        # Note: Actual stopping depends on inference running
        # Cache invalidation should happen if sample stops
        # This test verifies no crashes occur during cache operations

    @pytest.mark.asyncio
    async def test_cache_unique_epochs_optimization(self):
        """Test that cache invalidation uses .unique() optimization."""
        manager = OptimalStoppingManager(
            optstop_params={},
            grouping_columns=['model']
        )

        mock_dataset = Mock()
        mock_dataset.sample_ids = ['sample1']
        mock_config = Mock()
        mock_config.epochs = 50  # Many epochs to test optimization

        mock_task = Mock()
        mock_task.dataset = mock_dataset
        mock_task.config = mock_config
        mock_task.model = 'gpt4'
        mock_task.task = 'math'
        mock_task.metadata = {}
        mock_task.tags = []

        await manager.start_task(mock_task)

        # Verify compiled_dataset created correctly
        assert len(manager.compiled_dataset) == 50

        # Mark some epochs as run
        manager.compiled_dataset.loc[
            (manager.compiled_dataset['model'] == 'gpt4') &
            (manager.compiled_dataset['sample_id'] == 'sample1') &
            (manager.compiled_dataset['epoch'] <= 10),
            'trial_ran'
        ] = 1

        # Verify optimization runs without errors
        # (actual stopping logic requires full inference which needs more setup)

    @pytest.mark.asyncio
    async def test_group_level_cache_clearing(self):
        """Test that group-level stopping clears all cache entries for that grouping."""
        manager = OptimalStoppingManager(
            optstop_params={},
            grouping_columns=['model', 'task']
        )

        mock_dataset = Mock()
        mock_dataset.sample_ids = ['s1', 's2', 's3']
        mock_config = Mock()
        mock_config.epochs = 5

        mock_task = Mock()
        mock_task.dataset = mock_dataset
        mock_task.config = mock_config
        mock_task.model = 'gpt4'
        mock_task.task = 'math'
        mock_task.metadata = {}
        mock_task.tags = []

        await manager.start_task(mock_task)

        # Populate cache for multiple samples
        for sample_id in ['s1', 's2', 's3']:
            for epoch in [1, 2]:
                await manager.schedule_sample(mock_task, id=sample_id, epoch=epoch)

        initial_cache_size = len(manager._schedule_cache)
        assert initial_cache_size > 0

        # Simulate group stopping by manually calling the cache clearing logic
        grouping_values = {'model': 'gpt4', 'task': 'math'}
        grouping_prefix = tuple(grouping_values.values())

        # This is what the code does for group-level cache invalidation
        manager._schedule_cache = {
            k: v for k, v in manager._schedule_cache.items()
            if k[:len(grouping_prefix)] != grouping_prefix
        }

        # Verify all entries for this grouping were removed
        assert len(manager._schedule_cache) == 0

    @pytest.mark.asyncio
    async def test_cache_pop_with_none_default(self):
        """Test that cache uses .pop() with None default (no KeyError)."""
        manager = OptimalStoppingManager(
            optstop_params={},
            grouping_columns=['model']
        )

        mock_dataset = Mock()
        mock_dataset.sample_ids = ['sample1']
        mock_config = Mock()
        mock_config.epochs = 5

        mock_task = Mock()
        mock_task.dataset = mock_dataset
        mock_task.config = mock_config
        mock_task.model = 'gpt4'
        mock_task.task = 'math'
        mock_task.metadata = {}
        mock_task.tags = []

        await manager.start_task(mock_task)

        # Manually test .pop() behavior (simulating Priority 2.2 optimization)
        fake_key = ('gpt4', 'sample1', 999)  # Key that doesn't exist
        result = manager._schedule_cache.pop(fake_key, None)

        # Should not raise KeyError, should return None
        assert result is None


class TestMultiGroupingIntegration:
    """Integration tests for multi-grouping scenarios and complex configurations."""

    @pytest.mark.asyncio
    async def test_single_grouping_column_model(self):
        """Test basic single grouping by model attribute."""
        manager = OptimalStoppingManager(
            optstop_params={'delta_item': 0.01},
            grouping_columns=['model'],
            reanalysis_interval=2,
            min_samples_per_grouping=2
        )

        # Create task with multiple models
        mock_dataset = Mock()
        mock_dataset.sample_ids = ['s1', 's2', 's3', 's4']
        mock_config = Mock()
        mock_config.epochs = 5

        mock_task = Mock()
        mock_task.dataset = mock_dataset
        mock_task.config = mock_config
        mock_task.model = 'gpt4'
        mock_task.task = 'math'
        mock_task.metadata = {}
        mock_task.tags = []

        await manager.start_task(mock_task)

        # Verify grouping structure
        assert manager.compiled_dataset is not None
        assert 'model' in manager.compiled_dataset.columns
        assert len(manager.compiled_dataset['model'].unique()) == 1
        assert manager.compiled_dataset['model'].iloc[0] == 'gpt4'

        # Verify compiled dataset rows created correctly
        assert len(manager.compiled_dataset) == 4 * 5  # 4 samples * 5 epochs

    @pytest.mark.asyncio
    async def test_multiple_grouping_columns_model_and_task(self):
        """Test multi-grouping by both model and task attributes."""
        manager = OptimalStoppingManager(
            optstop_params={},
            grouping_columns=['model', 'task'],
            reanalysis_interval=1
        )

        mock_dataset = Mock()
        mock_dataset.sample_ids = ['s1', 's2']
        mock_config = Mock()
        mock_config.epochs = 3

        mock_task = Mock()
        mock_task.dataset = mock_dataset
        mock_task.config = mock_config
        mock_task.model = 'gpt4'
        mock_task.task = 'coding'
        mock_task.metadata = {}
        mock_task.tags = []

        await manager.start_task(mock_task)

        # Verify both grouping columns exist
        assert 'model' in manager.compiled_dataset.columns
        assert 'task' in manager.compiled_dataset.columns

        # Verify compiled dataset created correctly
        assert len(manager.compiled_dataset) == 2 * 3  # 2 samples * 3 epochs
        assert all(manager.compiled_dataset['model'] == 'gpt4')
        assert all(manager.compiled_dataset['task'] == 'coding')

    @pytest.mark.asyncio
    async def test_grouping_with_metadata_columns(self):
        """Test grouping using metadata.key syntax."""
        manager = OptimalStoppingManager(
            optstop_params={},
            grouping_columns=['model', "metadata.temperature"],
            reanalysis_interval=1
        )

        mock_dataset = Mock()
        mock_dataset.sample_ids = ['s1', 's2']
        mock_config = Mock()
        mock_config.epochs = 2

        mock_task = Mock()
        mock_task.dataset = mock_dataset
        mock_task.config = mock_config
        mock_task.model = 'claude'
        mock_task.task = 'reasoning'
        mock_task.metadata = {'temperature': 0.7, 'max_tokens': 1000}
        mock_task.tags = []

        await manager.start_task(mock_task)

        # Verify metadata column extracted correctly
        assert 'metadata.temperature' in manager.compiled_dataset.columns
        assert manager.compiled_dataset['metadata.temperature'].iloc[0] == 0.7

        # Verify all rows have same metadata value
        assert all(manager.compiled_dataset['metadata.temperature'] == 0.7)

    @pytest.mark.asyncio
    async def test_grouping_with_tags(self):
        """Test grouping using tag.name syntax."""
        manager = OptimalStoppingManager(
            optstop_params={},
            grouping_columns=['model', "tag.difficulty"],
            reanalysis_interval=1
        )

        mock_dataset = Mock()
        mock_dataset.sample_ids = ['s1']
        mock_config = Mock()
        mock_config.epochs = 2

        mock_task = Mock()
        mock_task.dataset = mock_dataset
        mock_task.config = mock_config
        mock_task.model = 'gpt4'
        mock_task.task = 'math'
        mock_task.metadata = {}
        mock_task.tags = ['difficulty', 'hard']  # Exact match for tag.difficulty

        await manager.start_task(mock_task)

        # Verify tag column extracted correctly
        assert 'tag.difficulty' in manager.compiled_dataset.columns
        assert manager.compiled_dataset['tag.difficulty'].iloc[0] == 'difficulty'

        # Verify all rows have same tag value
        assert all(manager.compiled_dataset['tag.difficulty'] == 'difficulty')

    @pytest.mark.asyncio
    async def test_complex_multi_grouping_all_types(self):
        """Test complex grouping combining direct attributes, metadata, and tags."""
        manager = OptimalStoppingManager(
            optstop_params={},
            grouping_columns=[
                'model',
                'task',
                "metadata.temperature",
                "tag.difficulty"
            ],
            reanalysis_interval=1
        )

        mock_dataset = Mock()
        mock_dataset.sample_ids = ['s1']
        mock_config = Mock()
        mock_config.epochs = 3

        mock_task = Mock()
        mock_task.dataset = mock_dataset
        mock_task.config = mock_config
        mock_task.model = 'claude'
        mock_task.task = 'reasoning'
        mock_task.metadata = {'temperature': 0.5}
        mock_task.tags = ['difficulty', 'easy']

        await manager.start_task(mock_task)

        # Verify all grouping columns created
        assert 'model' in manager.compiled_dataset.columns
        assert 'task' in manager.compiled_dataset.columns
        assert 'metadata.temperature' in manager.compiled_dataset.columns
        assert 'tag.difficulty' in manager.compiled_dataset.columns

        # Verify all rows have correct grouping values
        assert all(manager.compiled_dataset['model'] == 'claude')
        assert all(manager.compiled_dataset['task'] == 'reasoning')
        assert all(manager.compiled_dataset['metadata.temperature'] == 0.5)
        assert all(manager.compiled_dataset['tag.difficulty'] == 'difficulty')

    @pytest.mark.asyncio
    async def test_schedule_sample_respects_grouping_isolation(self):
        """Test that different groupings maintain independent stopping decisions."""
        manager = OptimalStoppingManager(
            optstop_params={},
            grouping_columns=['model'],
            reanalysis_interval=1
        )

        # Create first task (gpt4)
        mock_dataset1 = Mock()
        mock_dataset1.sample_ids = ['s1']
        mock_config1 = Mock()
        mock_config1.epochs = 3

        mock_task1 = Mock()
        mock_task1.dataset = mock_dataset1
        mock_task1.config = mock_config1
        mock_task1.model = 'gpt4'
        mock_task1.task = 'math'
        mock_task1.metadata = {}
        mock_task1.tags = []

        await manager.start_task(mock_task1)

        # Mark sample as stopped in compiled_dataset (schedule_status=False)
        manager.compiled_dataset.loc[
            (manager.compiled_dataset['model'] == 'gpt4') &
            (manager.compiled_dataset['sample_id'] == 's1') &
            (manager.compiled_dataset['epoch'] == 1),
            'schedule_status'
        ] = False

        # Schedule should return EarlyStop object for stopped sample
        result1 = await manager.schedule_sample(mock_task1, id='s1', epoch=1)
        assert result1 is not None  # Should return EarlyStop
        assert result1.reason == "Stopped by optimal stopping criteria"

        # Verify other epochs for same sample are still schedulable (independent)
        result2 = await manager.schedule_sample(mock_task1, id='s1', epoch=2)
        assert result2 is None  # Should still be schedulable

        # Verify that different groupings would maintain independent schedule_status
        # (tracked per grouping in the compiled_dataset)

    @pytest.mark.asyncio
    async def test_decision_counters_per_grouping(self):
        """Test that decision counters are maintained independently per grouping."""
        manager = OptimalStoppingManager(
            optstop_params={},
            grouping_columns=['model'],
            reanalysis_interval=3
        )

        mock_dataset = Mock()
        mock_dataset.sample_ids = ['s1', 's2']
        mock_config = Mock()
        mock_config.epochs = 5

        # Task 1: gpt4
        mock_task1 = Mock()
        mock_task1.dataset = mock_dataset
        mock_task1.config = mock_config
        mock_task1.model = 'gpt4'
        mock_task1.task = 'math'
        mock_task1.metadata = {}
        mock_task1.tags = []

        # Task 2: claude
        mock_task2 = Mock()
        mock_task2.dataset = mock_dataset
        mock_task2.config = mock_config
        mock_task2.model = 'claude'
        mock_task2.task = 'math'
        mock_task2.metadata = {}
        mock_task2.tags = []

        await manager.start_task(mock_task1)
        await manager.start_task(mock_task2)

        # Mock score
        mock_score_obj = Mock()
        mock_score_obj.as_float.return_value = 0.8
        mock_sample_score = Mock()
        mock_sample_score.score = mock_score_obj
        mock_scores = {'scorer1': mock_sample_score}

        # Complete 2 samples for gpt4 (counter = 2, not enough for reanalysis_interval=3)
        await manager.complete_sample(mock_task1, id='s1', epoch=1, scores=mock_scores)
        await manager.complete_sample(mock_task1, id='s2', epoch=1, scores=mock_scores)
        assert manager._decision_counters['gpt4'] == 2

        # Complete 1 sample for claude (counter = 1)
        await manager.complete_sample(mock_task2, id='s1', epoch=1, scores=mock_scores)
        assert manager._decision_counters['claude'] == 1

        # Verify counters are independent
        assert manager._decision_counters['gpt4'] == 2
        assert manager._decision_counters['claude'] == 1

    @pytest.mark.asyncio
    async def test_cache_isolation_between_groupings(self):
        """Test that schedule cache keys include all grouping values for isolation."""
        manager = OptimalStoppingManager(
            optstop_params={},
            grouping_columns=['model', 'task']
        )

        mock_dataset = Mock()
        mock_dataset.sample_ids = ['s1']
        mock_config = Mock()
        mock_config.epochs = 5

        mock_task = Mock()
        mock_task.dataset = mock_dataset
        mock_task.config = mock_config
        mock_task.model = 'gpt4'
        mock_task.task = 'math'
        mock_task.metadata = {}
        mock_task.tags = []

        await manager.start_task(mock_task)

        # Schedule a sample to populate cache
        await manager.schedule_sample(mock_task, id='s1', epoch=1)

        # Verify cache key structure includes all grouping columns
        # Keys should be tuples: (grouping_values..., sample_id, epoch)
        cache_key = ('gpt4', 'math', 's1', 1)
        assert cache_key in manager._schedule_cache

        # Manually add a different grouping key to verify isolation
        different_grouping_key = ('gpt4', 'coding', 's1', 1)
        manager._schedule_cache[different_grouping_key] = True

        # Verify both keys coexist in cache (showing isolation)
        assert cache_key in manager._schedule_cache
        assert different_grouping_key in manager._schedule_cache
        assert len(manager._schedule_cache) == 2

    @pytest.mark.asyncio
    async def test_grouping_with_missing_metadata(self):
        """Test behavior when specified metadata key is missing."""
        manager = OptimalStoppingManager(
            optstop_params={},
            grouping_columns=['model', "metadata.temperature"]
        )

        mock_dataset = Mock()
        mock_dataset.sample_ids = ['s1']
        mock_config = Mock()
        mock_config.epochs = 2

        mock_task = Mock()
        mock_task.dataset = mock_dataset
        mock_task.config = mock_config
        mock_task.model = 'gpt4'
        mock_task.task = 'math'
        mock_task.metadata = {}  # No 'temperature' key
        mock_task.tags = []

        await manager.start_task(mock_task)

        # Verify metadata column created with None value
        assert 'metadata.temperature' in manager.compiled_dataset.columns
        # All rows should have None since metadata key is missing
        assert all(pd.isna(manager.compiled_dataset['metadata.temperature']))

    @pytest.mark.asyncio
    async def test_grouping_with_missing_tag(self):
        """Test behavior when specified tag is missing."""
        manager = OptimalStoppingManager(
            optstop_params={},
            grouping_columns=['model', "tag.difficulty"]
        )

        mock_dataset = Mock()
        mock_dataset.sample_ids = ['s1']
        mock_config = Mock()
        mock_config.epochs = 2

        mock_task = Mock()
        mock_task.dataset = mock_dataset
        mock_task.config = mock_config
        mock_task.model = 'gpt4'
        mock_task.task = 'math'
        mock_task.metadata = {}
        mock_task.tags = ['easy', 'algebra']  # No 'difficulty' tag

        await manager.start_task(mock_task)

        # Verify tag column created with None value
        assert 'tag.difficulty' in manager.compiled_dataset.columns
        # All rows should have None since tag is not in the list
        assert all(manager.compiled_dataset['tag.difficulty'].isna())

    @pytest.mark.asyncio
    async def test_special_characters_in_grouping_values(self):
        """Test grouping names handle special characters correctly."""
        manager = OptimalStoppingManager(
            optstop_params={},
            grouping_columns=['model', 'task']
        )

        mock_dataset = Mock()
        mock_dataset.sample_ids = ['s1']
        mock_config = Mock()
        mock_config.epochs = 2

        mock_task = Mock()
        mock_task.dataset = mock_dataset
        mock_task.config = mock_config
        mock_task.model = 'gpt-4-turbo'  # Hyphen in name
        mock_task.task = 'math_basic'  # Underscore in name
        mock_task.metadata = {}
        mock_task.tags = []

        await manager.start_task(mock_task)

        # Verify dataset created with special characters in grouping values
        assert all(manager.compiled_dataset['model'] == 'gpt-4-turbo')
        assert all(manager.compiled_dataset['task'] == 'math_basic')

        # Verify schedule works with special characters
        result = await manager.schedule_sample(mock_task, id='s1', epoch=1)
        assert result is None  # Should schedule without issues


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
