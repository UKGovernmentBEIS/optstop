"""
Simple test to verify OptimalStoppingManager basic functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock

import sys
sys.path.insert(0, '/home/ubuntu/optstop')

from optstop.early_stopping import OptimalStoppingManager


def create_mock_sample(sample_id: str):
    """Create a mock Sample object."""
    sample = Mock()
    sample.id = sample_id
    sample.metadata = {}
    return sample


def create_mock_task(model: str, task: str):
    """Create a mock EvalSpec object."""
    task_mock = Mock()
    task_mock.model = model
    task_mock.task = task
    task_mock.eval_id = f"eval_{model}_{task}"
    return task_mock


def create_mock_score(value: float):
    """Create a mock SampleScore object."""
    # Create nested structure: SampleScore.score.value
    inner_score = Mock()
    inner_score.value = value

    sample_score = Mock()
    sample_score.score = inner_score
    sample_score.metadata = {}
    return sample_score


@pytest.mark.asyncio
async def test_simple_binary_discrete():
    """Simple test with 5 samples, 3 epochs each."""
    np.random.seed(42)

    # Configuration
    manager = OptimalStoppingManager(
        optstop_params={
            'delta_item': 0.15,
            'delta_cap': 0.15,
            'cred_level': 0.95,
            'conservatism': 5.0,
        },
        grouping_columns=['model'],
        score_agg=None,  # Discrete binary
        reanalysis_interval=2,  # Run inference every 2 samples
    )

    # Create task and samples
    task = create_mock_task('test_model', 'test_task')
    samples = [create_mock_sample(f'sample_{i}') for i in range(5)]

    print("\\n1. Starting task...")
    await manager.start_task(task, samples, epochs=3)
    print(f"   Dataset initialized: {len(manager.compiled_dataset)} rows")

    # Run 3 epochs
    print("\\n2. Running epochs...")
    for epoch in range(3):
        for sample in samples:
            # Binary score: 1 with 80% probability
            score_value = float(np.random.random() < 0.80)
            score = create_mock_score(score_value)

            print(f"   Sample {sample.id}, epoch {epoch}, score={score_value}")
            await manager.complete_sample(sample.id, epoch, {'score': score})

    # Complete task
    print("\\n3. Completing task...")
    final_result = await manager.complete_task()

    print("\\n4. Results:")
    print(f"   Stopped samples: {len(final_result['stopped_samples'])}")
    print(f"   Stopped groupings: {len(final_result['stopped_groupings'])}")

    # Basic assertions
    assert 'stopped_samples' in final_result
    assert 'stopped_groupings' in final_result
    assert len(manager.compiled_dataset) == 5 * 3

    print("\\n✅ Test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
