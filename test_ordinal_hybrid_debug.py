"""
Debug script for Test 4 - Ordinal Discrete Bimodal with Hybrid Inference

This script runs a simplified version of Test 4 to diagnose the parameter
passing issue with MCMC sampling (draws/tune).

Expected behavior:
- Should use draws=500, tune=500 (from optstop_params)
- Should complete in ~5-10 minutes with 20 samples

If it takes much longer, parameters are not being passed correctly.
"""

import asyncio
import pandas as pd
import numpy as np
import logging
import sys
from unittest.mock import Mock

# Add to path
sys.path.insert(0, '/home/ubuntu/optstop')

from optstop.early_stopping import OptimalStoppingManager

# Configure logging to see warnings
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_ordinal_hybrid_debug.log', mode='w')
    ]
)

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
    inner_score = Mock()
    inner_score.value = value

    sample_score = Mock()
    sample_score.score = inner_score
    sample_score.metadata = metadata or {}
    return sample_score


async def test_ordinal_hybrid_debug():
    """
    Simplified Test 4: Ordinal discrete with bimodal distribution and hybrid inference.

    Uses only 20 samples (instead of 80) to complete faster while still triggering
    the hybrid inference pathway.
    """
    print("\n" + "="*80)
    print("DIAGNOSTIC TEST: Ordinal Discrete Bimodal with Hybrid Inference")
    print("="*80)

    np.random.seed(104)  # Different seed for different distribution

    # Configuration
    manager = OptimalStoppingManager(
        optstop_params={
            'delta_item': 0.12,
            'delta_cap': 0.12,
            'cred_level': 0.95,
            'conservatism': 5.0,
            'low_performance_threshold': 0.2,
            'draws': 500,        # ← DIAGNOSTIC: Should see this in logs
            'tune': 500          # ← DIAGNOSTIC: Should see this in logs
        },
        grouping_columns=['task'],
        score_choice=None,
        score_agg=None,  # No aggregation = discrete ordinal
        reanalysis_interval=50,  # Run inference at 50%, 100% (100 trials)
        ordinal_tasks=['rating'],
        ordinal_max_score=10,
        ordinal_inference='hybrid'  # ← This triggers _ordinal_entropy_ci_adaptive
    )

    # Create task and samples
    task = create_mock_task('gpt4', 'rating')
    num_samples = 20  # Reduced from 80
    num_epochs = 5    # Reduced from 10
    samples = [create_mock_sample(f'sample_{i}', {'task': 'rating'}) for i in range(num_samples)]

    print(f"\nTest configuration:")
    print(f"  - Samples: {num_samples}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Total trials: {num_samples * num_epochs}")
    print(f"  - Inference at: trials 50, 100")
    print(f"  - draws=500, tune=500 (should be logged)")
    print(f"  - Expected time: 5-10 minutes")
    print()

    # Initialize
    await manager.start_task(task, samples, num_epochs)

    # Simulate bimodal distribution: half high (~8/10), half low (~3/10)
    # This should trigger hybrid inference pathway
    trial_count = 0

    for epoch in range(1, num_epochs + 1):
        for i, sample in enumerate(samples):
            trial_count += 1

            # Check if should schedule
            early_stop = await manager.schedule_sample(sample.id, epoch)
            if early_stop:
                continue

            # Generate bimodal scores: half high, half low
            if i < num_samples // 2:
                # High performance group (~8/10)
                score = np.random.choice([7, 8, 9], p=[0.2, 0.5, 0.3])
            else:
                # Low performance group (~3/10)
                score = np.random.choice([2, 3, 4], p=[0.3, 0.4, 0.3])

            scores_dict = {'accuracy': create_mock_score(score)}
            await manager.complete_sample(sample.id, epoch, scores_dict)

            # Progress indicator
            if trial_count % 25 == 0:
                print(f"  Completed {trial_count}/{num_samples * num_epochs} trials...")

    # Complete task
    metadata = await manager.complete_task()

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print(f"Efficiency: {metadata['efficiency_percent']:.1f}%")
    print(f"Trials ran: {metadata['total_ran']}/{metadata['total_planned_trials']}")
    print(f"Stopped samples: {metadata['stopped_samples_count']}")
    print(f"Stopped groupings: {metadata['stopped_groupings_count']}")
    print("\nCheck test_ordinal_hybrid_debug.log for parameter diagnostics!")
    print("Look for lines starting with '🔍 _ordinal_entropy_ci_adaptive called:'")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\nStarting diagnostic test...")
    print("This will create test_ordinal_hybrid_debug.log with detailed parameter tracking.\n")

    asyncio.run(test_ordinal_hybrid_debug())
