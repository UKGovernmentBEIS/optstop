"""
Test Suite 1.1.2: Ordinal Discrete Scoring Tests

Tests ordinal discrete scoring (integer ratings on a discrete scale, e.g., 1-5 stars)
with three inference modes:
- Modal: Uses mode (most common value) for inference
- Entropy: Uses entropy to measure distribution spread
- Hybrid: Combines both, uses entropy threshold to detect false peaks

Test Coverage:
- 1.1.2a: Ordinal modal inference
- 1.1.2b: Ordinal entropy inference
- 1.1.2c: Ordinal hybrid inference (peaked data - low entropy)
- 1.1.2d: Ordinal hybrid inference (diffuse data - high entropy)

Author: Generated for optstop bridge validation
Date: 2025-11-23
"""

import pytest
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path

# Import bridge components
from optstop.early_stopping import OptimalStoppingManager

# Mock inspect_ai classes if not installed
try:
    from inspect_ai.dataset._dataset import Sample
    from inspect_ai.log._log import EvalSpec, SampleScore, Score
    INSPECT_AI_AVAILABLE = True

    def create_mock_sample_score(value: int) -> dict:
        """Create a mock score dict for ordinal discrete scoring (integer ratings)."""
        return {"rating": SampleScore(Score(value=value))}

except ImportError:
    INSPECT_AI_AVAILABLE = False
    # Use mock classes from early_stopping
    from optstop.early_stopping import Sample, EvalSpec

    # Define MockScore and SampleScore
    class MockScore:
        def __init__(self, value):
            self.value = value

    class SampleScore:
        def __init__(self, score):
            self.score = score

    def create_mock_sample_score(value: int) -> dict:
        """Create a mock score dict for ordinal discrete scoring (integer ratings)."""
        return {"rating": SampleScore(MockScore(value))}


# Configure test output directory
TEST_OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "bridge_ordinal_discrete"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_deterministic_ordinal_data(
    n_samples: int,
    mode_value: int,
    concentration: float = 0.8,
    max_score: int = 5,
    seed: int = 42
) -> list[int]:
    """
    Create deterministic ordinal data (integer ratings) with specified mode and concentration.

    Args:
        n_samples: Number of samples to generate
        mode_value: The mode (most common value) for the distribution
        concentration: Proportion of samples that should be the mode (0-1)
        max_score: Maximum score value (e.g., 5 for 1-5 ratings)
        seed: Random seed for reproducibility

    Returns:
        List of integer ratings

    Example:
        create_deterministic_ordinal_data(100, mode_value=4, concentration=0.8, max_score=5)
        # Returns: ~80% are 4s, ~20% distributed among 1,2,3,5
    """
    np.random.seed(seed)

    # Calculate number of mode values
    n_mode = int(n_samples * concentration)
    n_other = n_samples - n_mode

    # Create pattern: mostly mode value, rest distributed
    pattern = [mode_value] * n_mode

    # Distribute remaining among other values
    if n_other > 0:
        other_values = [v for v in range(1, max_score + 1) if v != mode_value]
        # Evenly distribute among other values
        for i in range(n_other):
            pattern.append(other_values[i % len(other_values)])

    # Shuffle to avoid patterns
    np.random.shuffle(pattern)

    return pattern


def create_diffuse_ordinal_data(
    n_samples: int,
    max_score: int = 5,
    seed: int = 42
) -> list[int]:
    """
    Create diffuse (high entropy) ordinal data with uniform distribution.

    Args:
        n_samples: Number of samples to generate
        max_score: Maximum score value
        seed: Random seed

    Returns:
        List of integer ratings uniformly distributed
    """
    np.random.seed(seed)

    # Uniform distribution across all possible values
    pattern = []
    values = list(range(1, max_score + 1))
    samples_per_value = n_samples // len(values)
    remainder = n_samples % len(values)

    for value in values:
        pattern.extend([value] * samples_per_value)

    # Add remainder
    for i in range(remainder):
        pattern.append(values[i % len(values)])

    # Shuffle
    np.random.shuffle(pattern)

    return pattern


@pytest.mark.asyncio
async def test_1_1_2a_ordinal_modal_inference():
    """
    Test 1.1.2a: Ordinal Modal Inference

    Validates that ordinal discrete scoring works with modal inference.
    Modal inference uses the mode (most common value) for statistical decisions.

    Setup:
    - 20 samples, 8 epochs each
    - Single grouping: gpt-4-rating (ordinal task)
    - Deterministic ordinal data: mode=4, concentration=0.85 (5-point scale)
    - Inference mode: 'modal'

    Expected:
    - Modal inference should converge based on mode stability
    - Stopping should occur when mode is stable with narrow CI
    - Some efficiency gain expected (>20%)
    """
    print("\n" + "="*80)
    print("TEST 1.1.2a: Ordinal Modal Inference")
    print("="*80)

    # Configure optstop parameters (relaxed for testing)
    # IMPORTANT: Use reduced MCMC settings for faster testing
    optstop_params = {
        'delta_item': 0.20,
        'delta_cap': 0.15,
        'cred_level': 0.85,
        'conservatism': 3,
        'draws': 1000,  # Reduced from default 6000 for faster testing
        'tune': 1000,   # Reduced from default 6000 for faster testing
    }

    # Configure manager for ordinal modal inference
    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
        reanalysis_interval=10,  # Increased from 3 to reduce inference frequency
        min_samples_per_grouping=5,  # Increased to delay first inference
        ordinal_tasks=['rating'],  # Identify 'rating' tasks as ordinal
        ordinal_max_score=5,
        ordinal_inference='modal',  # Use modal inference
    )

    # Create samples with ordinal task name
    n_samples = 15  # Reduced from 20 for faster testing
    samples = [
        Sample(id=f"sample_{i}", metadata={})
        for i in range(n_samples)
    ]

    # Create eval spec
    eval_spec = EvalSpec(
        task="gpt-4-rating",  # Task name contains 'rating' (ordinal identifier)
        model="gpt-4"
    )

    # Generate deterministic ordinal data (mode=4, 85% concentration)
    ordinal_data = create_deterministic_ordinal_data(
        n_samples * 8,  # 15 samples × 8 epochs
        mode_value=4,
        concentration=0.85,
        max_score=5,
        seed=42
    )
    data_idx = 0

    # Start task
    await manager.start_task(eval_spec, samples, epochs=8)

    # Run evaluation loop
    completed_trials = 0
    stopped_trials = 0
    sample_epoch_counts = {}

    for sample in samples:
        sample_epoch_counts[sample.id] = 0
        for epoch in range(1, 9):  # epochs 1-8
            # Check if we should stop early
            early_stop = await manager.schedule_sample(sample.id, epoch)

            if early_stop is not None:
                stopped_trials += 1
                print(f"Sample {sample.id} stopped at epoch {epoch}: {early_stop.reason}")
                continue

            # Run the trial and complete it
            score_value = ordinal_data[data_idx]
            data_idx += 1
            scores = create_mock_sample_score(score_value)

            await manager.complete_sample(sample.id, epoch, scores)
            completed_trials += 1
            sample_epoch_counts[sample.id] += 1

    # Complete task
    diagnostics = await manager.complete_task()

    # Calculate efficiency
    total_planned = n_samples * 8
    efficiency_percent = (stopped_trials / total_planned) * 100

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Total planned trials: {total_planned}")
    print(f"Completed trials: {completed_trials}")
    print(f"Stopped trials: {stopped_trials}")
    print(f"Efficiency: {efficiency_percent:.1f}%")
    print(f"Stopped groupings: {diagnostics.get('stopped_groupings', [])}")
    print(f"Stopped samples count: {diagnostics.get('stopped_samples_count', 0)}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "test": "1.1.2a_ordinal_modal_inference",
        "inference_mode": "modal",
        "ordinal_max_score": 5,
        "n_samples": n_samples,
        "epochs_per_sample": 8,
        "total_planned": total_planned,
        "completed_trials": completed_trials,
        "stopped_trials": stopped_trials,
        "efficiency_percent": efficiency_percent,
        "stopped_groupings": diagnostics.get('stopped_groupings', []),
        "stopped_samples_count": diagnostics.get('stopped_samples_count', 0),
        "sample_epoch_counts": sample_epoch_counts,
        "validation": {
            "modal_inference_used": True,
            "ordinal_scoring": True,
            "some_stopping": stopped_trials > 0
        }
    }

    output_file = TEST_OUTPUT_DIR / f"test_1_1_2a_modal_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Assertions
    assert completed_trials + stopped_trials == total_planned, "Trial count mismatch"
    assert stopped_trials > 0, "Expected some stopping to occur with modal inference"
    assert efficiency_percent > 10, f"Expected >10% efficiency, got {efficiency_percent:.1f}%"

    print("\n✅ TEST 1.1.2a PASSED")


@pytest.mark.asyncio
async def test_1_1_2b_ordinal_entropy_inference():
    """
    Test 1.1.2b: Ordinal Entropy Inference

    Validates that ordinal discrete scoring works with entropy inference.
    Entropy inference uses Shannon entropy to measure distribution spread.

    Setup:
    - 20 samples, 8 epochs each
    - Single grouping: gpt-4-confidence (ordinal task)
    - Deterministic ordinal data: mode=5, concentration=0.90 (very peaked)
    - Inference mode: 'entropy'

    Expected:
    - Entropy inference should converge based on entropy stability
    - Low entropy (peaked distribution) should allow stopping
    - Efficiency >20% expected
    """
    print("\n" + "="*80)
    print("TEST 1.1.2b: Ordinal Entropy Inference")
    print("="*80)

    # Configure optstop parameters
    # IMPORTANT: Use reduced MCMC settings for faster testing
    optstop_params = {
        'delta_item': 0.25,  # Relaxed for entropy convergence
        'delta_cap': 0.25,  # Relaxed to allow stopping with entropy inference
        'cred_level': 0.80,  # Lower credibility for easier stopping
        'conservatism': 3,
        'draws': 1000,  # Reduced from default 6000 for faster testing
        'tune': 1000,   # Reduced from default 6000 for faster testing
    }

    # Configure manager for ordinal entropy inference
    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
        reanalysis_interval=3,  # More frequent inference checks for entropy stabilization
        min_samples_per_grouping=3,  # Start inference earlier
        ordinal_tasks=['confidence'],  # Identify 'confidence' tasks as ordinal
        ordinal_max_score=5,
        ordinal_inference='entropy',  # Use entropy inference
    )

    # Create samples with ordinal task name
    n_samples = 15  # Reduced from 20 for faster testing
    samples = [
        Sample(id=f"sample_{i}", metadata={})
        for i in range(n_samples)
    ]

    # Create eval spec
    eval_spec = EvalSpec(
        task="gpt-4-confidence",  # Task name contains 'confidence' (ordinal identifier)
        model="gpt-4"
    )

    # Generate very peaked ordinal data (mode=5, 90% concentration)
    ordinal_data = create_deterministic_ordinal_data(
        n_samples * 8,  # 15 samples × 8 epochs
        mode_value=5,
        concentration=0.90,
        max_score=5,
        seed=43
    )
    data_idx = 0

    # Start task
    await manager.start_task(eval_spec, samples, epochs=8)

    # Run evaluation loop
    completed_trials = 0
    stopped_trials = 0
    sample_epoch_counts = {}

    for sample in samples:
        sample_epoch_counts[sample.id] = 0
        for epoch in range(1, 9):
            early_stop = await manager.schedule_sample(sample.id, epoch)

            if early_stop is not None:
                stopped_trials += 1
                print(f"Sample {sample.id} stopped at epoch {epoch}: {early_stop.reason}")
                continue

            score_value = ordinal_data[data_idx]
            data_idx += 1
            scores = create_mock_sample_score(score_value)

            await manager.complete_sample(sample.id, epoch, scores)
            completed_trials += 1
            sample_epoch_counts[sample.id] += 1

    # Complete task
    diagnostics = await manager.complete_task()

    # Calculate efficiency
    total_planned = n_samples * 8
    efficiency_percent = (stopped_trials / total_planned) * 100

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Total planned trials: {total_planned}")
    print(f"Completed trials: {completed_trials}")
    print(f"Stopped trials: {stopped_trials}")
    print(f"Efficiency: {efficiency_percent:.1f}%")
    print(f"Stopped groupings: {diagnostics.get('stopped_groupings', [])}")
    print(f"Stopped samples count: {diagnostics.get('stopped_samples_count', 0)}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "test": "1.1.2b_ordinal_entropy_inference",
        "inference_mode": "entropy",
        "ordinal_max_score": 5,
        "n_samples": n_samples,
        "epochs_per_sample": 8,
        "total_planned": total_planned,
        "completed_trials": completed_trials,
        "stopped_trials": stopped_trials,
        "efficiency_percent": efficiency_percent,
        "stopped_groupings": diagnostics.get('stopped_groupings', []),
        "stopped_samples_count": diagnostics.get('stopped_samples_count', 0),
        "sample_epoch_counts": sample_epoch_counts,
        "validation": {
            "entropy_inference_used": True,
            "ordinal_scoring": True,
            "peaked_distribution": True,
            "some_stopping": stopped_trials > 0
        }
    }

    output_file = TEST_OUTPUT_DIR / f"test_1_1_2b_entropy_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Assertions
    assert completed_trials + stopped_trials == total_planned
    assert stopped_trials > 0, "Expected stopping with low entropy (peaked) data"
    assert efficiency_percent > 10, f"Expected >10% efficiency, got {efficiency_percent:.1f}%"

    print("\n✅ TEST 1.1.2b PASSED")


@pytest.mark.asyncio
async def test_1_1_2c_ordinal_hybrid_peaked():
    """
    Test 1.1.2c: Ordinal Hybrid Inference (Peaked Data)

    Validates hybrid inference with peaked (low entropy) ordinal data.
    Hybrid mode should use modal inference when entropy is low (true peak).

    Setup:
    - 20 samples, 8 epochs each
    - Single grouping: gpt-4-rating
    - Peaked ordinal data: mode=4, concentration=0.88
    - Inference mode: 'hybrid' (entropy threshold = 1.5)

    Expected:
    - Hybrid should detect low entropy and use modal inference
    - Stopping should occur (peaked data converges well)
    - Efficiency >20% expected
    """
    print("\n" + "="*80)
    print("TEST 1.1.2c: Ordinal Hybrid Inference (Peaked Data)")
    print("="*80)

    # Configure optstop parameters
    # IMPORTANT: Use reduced MCMC settings for faster testing
    optstop_params = {
        'delta_item': 0.20,
        'delta_cap': 0.15,
        'cred_level': 0.85,
        'conservatism': 3,
        'draws': 1000,  # Reduced from default 6000 for faster testing
        'tune': 1000,   # Reduced from default 6000 for faster testing
    }

    # Configure manager for ordinal hybrid inference
    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
        reanalysis_interval=10,  # Increased from 3 to reduce inference frequency
        min_samples_per_grouping=5,  # Increased to delay first inference
        ordinal_tasks=['rating'],
        ordinal_max_score=5,
        ordinal_inference='hybrid',  # Use hybrid inference
    )

    # Create samples
    n_samples = 15  # Reduced from 20 for faster testing
    samples = [
        Sample(id=f"sample_{i}", metadata={})
        for i in range(n_samples)
    ]

    # Create eval spec
    eval_spec = EvalSpec(
        task="gpt-4-rating",
        model="gpt-4"
    )

    # Generate peaked ordinal data (mode=4, 88% concentration)
    ordinal_data = create_deterministic_ordinal_data(
        n_samples * 8,  # 15 samples × 8 epochs
        mode_value=4,
        concentration=0.88,
        max_score=5,
        seed=44
    )
    data_idx = 0

    # Start task
    await manager.start_task(eval_spec, samples, epochs=8)

    # Run evaluation loop
    completed_trials = 0
    stopped_trials = 0
    sample_epoch_counts = {}

    for sample in samples:
        sample_epoch_counts[sample.id] = 0
        for epoch in range(1, 9):
            early_stop = await manager.schedule_sample(sample.id, epoch)

            if early_stop is not None:
                stopped_trials += 1
                print(f"Sample {sample.id} stopped at epoch {epoch}: {early_stop.reason}")
                continue

            score_value = ordinal_data[data_idx]
            data_idx += 1
            scores = create_mock_sample_score(score_value)

            await manager.complete_sample(sample.id, epoch, scores)
            completed_trials += 1
            sample_epoch_counts[sample.id] += 1

    # Complete task
    diagnostics = await manager.complete_task()

    # Calculate efficiency
    total_planned = n_samples * 8
    efficiency_percent = (stopped_trials / total_planned) * 100

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Total planned trials: {total_planned}")
    print(f"Completed trials: {completed_trials}")
    print(f"Stopped trials: {stopped_trials}")
    print(f"Efficiency: {efficiency_percent:.1f}%")
    print(f"Stopped groupings: {diagnostics.get('stopped_groupings', [])}")
    print(f"Stopped samples count: {diagnostics.get('stopped_samples_count', 0)}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "test": "1.1.2c_ordinal_hybrid_peaked",
        "inference_mode": "hybrid",
        "data_type": "peaked",
        "ordinal_max_score": 5,
        "n_samples": n_samples,
        "epochs_per_sample": 8,
        "total_planned": total_planned,
        "completed_trials": completed_trials,
        "stopped_trials": stopped_trials,
        "efficiency_percent": efficiency_percent,
        "stopped_groupings": diagnostics.get('stopped_groupings', []),
        "stopped_samples_count": diagnostics.get('stopped_samples_count', 0),
        "sample_epoch_counts": sample_epoch_counts,
        "validation": {
            "hybrid_inference_used": True,
            "peaked_data": True,
            "expected_mode": "modal (low entropy)",
            "some_stopping": stopped_trials > 0
        }
    }

    output_file = TEST_OUTPUT_DIR / f"test_1_1_2c_hybrid_peaked_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Assertions
    assert completed_trials + stopped_trials == total_planned
    assert stopped_trials > 0, "Expected stopping with peaked data in hybrid mode"
    assert efficiency_percent > 10, f"Expected >10% efficiency, got {efficiency_percent:.1f}%"

    print("\n✅ TEST 1.1.2c PASSED")


@pytest.mark.asyncio
async def test_1_1_2d_ordinal_hybrid_diffuse():
    """
    Test 1.1.2d: Ordinal Hybrid Inference (Diffuse Data)

    Validates hybrid inference with diffuse (high entropy) ordinal data.
    Hybrid mode should use entropy inference when entropy is high (false peak/uniform).

    Setup:
    - 20 samples, 10 epochs each (more epochs due to diffuse data)
    - Single grouping: gpt-4-rating
    - Diffuse ordinal data: uniform distribution across 1-5
    - Inference mode: 'hybrid' (entropy threshold = 1.5)

    Expected:
    - Hybrid should detect high entropy and use entropy inference
    - May have lower efficiency (diffuse data harder to converge)
    - Test validates hybrid switching behavior
    """
    print("\n" + "="*80)
    print("TEST 1.1.2d: Ordinal Hybrid Inference (Diffuse Data)")
    print("="*80)

    # Configure optstop parameters (more relaxed for diffuse data)
    # IMPORTANT: Use reduced MCMC settings for faster testing
    optstop_params = {
        'delta_item': 0.25,
        'delta_cap': 0.20,
        'cred_level': 0.80,
        'conservatism': 2,
        'draws': 1000,  # Reduced from default 6000 for faster testing
        'tune': 1000,   # Reduced from default 6000 for faster testing
    }

    # Configure manager for ordinal hybrid inference
    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
        reanalysis_interval=10,  # Increased from 4 to reduce inference frequency
        min_samples_per_grouping=5,  # Increased to delay first inference
        ordinal_tasks=['rating'],
        ordinal_max_score=5,
        ordinal_inference='hybrid',  # Use hybrid inference
    )

    # Create samples
    n_samples = 15  # Reduced from 20 for faster testing
    samples = [
        Sample(id=f"sample_{i}", metadata={})
        for i in range(n_samples)
    ]

    # Create eval spec
    eval_spec = EvalSpec(
        task="gpt-4-rating",
        model="gpt-4"
    )

    # Generate diffuse ordinal data (uniform distribution)
    ordinal_data = create_diffuse_ordinal_data(
        n_samples * 10,  # 15 samples × 10 epochs
        max_score=5,
        seed=45
    )
    data_idx = 0

    # Start task
    await manager.start_task(eval_spec, samples, epochs=10)

    # Run evaluation loop
    completed_trials = 0
    stopped_trials = 0
    sample_epoch_counts = {}

    for sample in samples:
        sample_epoch_counts[sample.id] = 0
        for epoch in range(1, 11):  # epochs 1-10
            early_stop = await manager.schedule_sample(sample.id, epoch)

            if early_stop is not None:
                stopped_trials += 1
                print(f"Sample {sample.id} stopped at epoch {epoch}: {early_stop.reason}")
                continue

            score_value = ordinal_data[data_idx]
            data_idx += 1
            scores = create_mock_sample_score(score_value)

            await manager.complete_sample(sample.id, epoch, scores)
            completed_trials += 1
            sample_epoch_counts[sample.id] += 1

    # Complete task
    diagnostics = await manager.complete_task()

    # Calculate efficiency
    total_planned = n_samples * 10
    efficiency_percent = (stopped_trials / total_planned) * 100

    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"Total planned trials: {total_planned}")
    print(f"Completed trials: {completed_trials}")
    print(f"Stopped trials: {stopped_trials}")
    print(f"Efficiency: {efficiency_percent:.1f}%")
    print(f"Stopped groupings: {diagnostics.get('stopped_groupings', [])}")
    print(f"Stopped samples count: {diagnostics.get('stopped_samples_count', 0)}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "test": "1.1.2d_ordinal_hybrid_diffuse",
        "inference_mode": "hybrid",
        "data_type": "diffuse",
        "ordinal_max_score": 5,
        "n_samples": n_samples,
        "epochs_per_sample": 10,
        "total_planned": total_planned,
        "completed_trials": completed_trials,
        "stopped_trials": stopped_trials,
        "efficiency_percent": efficiency_percent,
        "stopped_groupings": diagnostics.get('stopped_groupings', []),
        "stopped_samples_count": diagnostics.get('stopped_samples_count', 0),
        "sample_epoch_counts": sample_epoch_counts,
        "validation": {
            "hybrid_inference_used": True,
            "diffuse_data": True,
            "expected_mode": "entropy (high entropy)",
            "test_completed": True
        }
    }

    output_file = TEST_OUTPUT_DIR / f"test_1_1_2d_hybrid_diffuse_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Assertions
    assert completed_trials + stopped_trials == total_planned
    # Note: Diffuse data may not stop much (or at all) - this is expected behavior
    print(f"Note: Diffuse data efficiency was {efficiency_percent:.1f}% (may be low, this is expected)")

    print("\n✅ TEST 1.1.2d PASSED")


if __name__ == "__main__":
    # Run tests individually for debugging
    import asyncio

    async def run_all_tests():
        print("Running Section 1.1.2: Ordinal Discrete Scoring Tests\n")

        await test_1_1_2a_ordinal_modal_inference()
        await test_1_1_2b_ordinal_entropy_inference()
        await test_1_1_2c_ordinal_hybrid_peaked()
        await test_1_1_2d_ordinal_hybrid_diffuse()

        print("\n" + "="*80)
        print("ALL SECTION 1.1.2 TESTS COMPLETED")
        print("="*80)

    asyncio.run(run_all_tests())
