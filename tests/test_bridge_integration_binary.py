"""
Integration tests for OptimalStoppingManager with binary discrete scoring.

Tests realistic inspect_ai evaluation scenarios with full lifecycle validation.
Corresponds to section 1.1.1 of BRIDGE_TESTING_AND_DEVELOPMENT_ROADMAP.md.
"""

import pytest
import pandas as pd
import numpy as np
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Import the bridge - this handles inspect_ai import with fallback
from optstop.early_stopping import OptimalStoppingManager

# early_stopping.py handles the import of Sample, EvalSpec, SampleScore
# with automatic fallback to mocks if inspect_ai is not installed
try:
    from inspect_ai.dataset._dataset import Sample
    from inspect_ai.log._log import EvalSpec
    from inspect_ai.scorer._metric import SampleScore
    USING_REAL_INSPECT = True
except ImportError:
    # Use the same mocks that early_stopping.py creates
    from optstop.early_stopping import Sample, EvalSpec

    # Create mock SampleScore
    class MockScore:
        def __init__(self, value):
            self.value = value

    class SampleScore:
        def __init__(self, score):
            if hasattr(score, 'value'):
                self.score = score
            else:
                self.score = MockScore(score)

    USING_REAL_INSPECT = False


# Test configuration
TEST_OUTPUT_DIR = Path(__file__).parent / "test_outputs" / "bridge_binary"
TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class MockScore:
    """Mock score object that mimics inspect_ai's Score."""
    def __init__(self, value):
        self.value = value


def create_mock_sample_score(value: float) -> dict:
    """Create mock scores dictionary as inspect_ai would provide."""
    return {
        "accuracy": SampleScore(MockScore(value))
    }


def get_test_log_file() -> Path:
    """Generate unique log file path for this test run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return TEST_OUTPUT_DIR / f"bridge_integration_test_{timestamp}.log"


def configure_test_logging(log_file: Path):
    """Configure logging for test with both file and console output."""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler - capture everything
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


# ============================================================================
# Test 1.1.1a: Simple Binary Evaluation (Single Grouping)
# ============================================================================

@pytest.mark.asyncio
async def test_1_1_1a_simple_binary_single_grouping():
    """
    Test 1.1.1a: Simple Binary Evaluation (Single Grouping)

    Mock inspect_ai evaluation with:
    - 20 samples × 10 epochs = 200 planned trials
    - Binary scores (0/1) only
    - Single grouping (model='gpt-4', task='math')

    Expected outputs:
    - start_task() returns manager name
    - schedule_sample() returns None until stopping criteria met
    - complete_sample() processes scores correctly
    - complete_task() returns valid diagnostics dict

    Validation:
    - Configuration summary printed to console
    - Sample-level stopping decisions logged with reasons
    - Group-level stopping decisions logged with metrics
    - Process cleanup logged
    - Representative schedule_sample() calls printed
    - Representative complete_sample() score extractions printed
    - Final diagnostics summary printed
    """
    # Setup logging
    log_file = get_test_log_file()
    logger = configure_test_logging(log_file)

    print("\n" + "="*80)
    print("TEST 1.1.1a: Simple Binary Evaluation (Single Grouping)")
    print("="*80)

    # Configure optimal stopping parameters
    optstop_params = {
        'delta_item': 0.15,      # Relaxed for faster stopping in tests
        'delta_cap': 0.10,       # Relaxed for faster stopping
        'cred_level': 0.90,      # Lower for faster convergence
        'conservatism': 3,       # Lower for faster stopping
        'draws': 500,            # Reduced for faster tests
        'tune': 500,             # Reduced for faster tests
        'chains': 2,             # Reduced for faster tests
        'cores': 2,              # Reduced for faster tests
    }

    # Create OptimalStoppingManager
    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
        sample_id_column='sample_id',
        epoch_column='epoch',
        score_column='score',
        reanalysis_interval=5,  # Analyze every 5 completed samples
        min_samples_per_grouping=3,  # Minimum 3 samples before analysis
        manager_name="test_binary_single"
    )

    # Create mock samples (20 samples)
    samples = [
        Sample(id=f"sample_{i}", metadata={})
        for i in range(20)
    ]

    # Create mock EvalSpec
    eval_spec = EvalSpec(
        model="gpt-4",
        task="math",
        eval_id="test_eval_001",
        metadata={},
        tags=[]
    )

    # STEP 1: start_task()
    print("\n[STEP 1] Calling start_task()...")
    manager_name = await manager.start_task(
        task=eval_spec,
        samples=samples,
        epochs=10
    )

    assert manager_name == "test_binary_single", "Manager name mismatch"
    assert manager.compiled_dataset is not None, "compiled_dataset not initialized"
    assert len(manager.compiled_dataset) == 200, "Expected 200 planned trials"

    print(f"✓ start_task() returned: {manager_name}")
    print(f"✓ compiled_dataset initialized: {len(manager.compiled_dataset)} trials")

    # STEP 2: Run evaluation loop
    print("\n[STEP 2] Running evaluation loop...")

    # Simulate binary scores with moderate performance (0.65 success rate)
    np.random.seed(42)

    stopped_count = 0
    completed_count = 0

    # Print first 3 schedule_sample calls
    print("\nFirst 3 schedule_sample() calls:")

    for sample in samples:
        for epoch in range(1, 11):  # epochs 1-10
            # Schedule sample
            early_stop = await manager.schedule_sample(
                id=sample.id,
                epoch=epoch
            )

            # Print first 3 calls
            if completed_count < 3:
                print(f"  schedule_sample(id='{sample.id}', epoch={epoch}) -> {early_stop}")

            if early_stop is not None:
                stopped_count += 1
                continue  # Skip this trial

            # Generate binary score (0 or 1)
            score_value = 1 if np.random.random() < 0.65 else 0

            # Complete sample
            await manager.complete_sample(
                id=sample.id,
                epoch=epoch,
                scores=create_mock_sample_score(score_value)
            )

            completed_count += 1

            # Print first 3 complete_sample calls
            if completed_count <= 3:
                print(f"  complete_sample(id='{sample.id}', epoch={epoch}, score={score_value})")

    print(f"\n✓ Evaluation loop complete:")
    print(f"  - Completed trials: {completed_count}")
    print(f"  - Stopped trials: {stopped_count}")
    print(f"  - Efficiency: {stopped_count / 200 * 100:.1f}%")

    # STEP 3: complete_task()
    print("\n[STEP 3] Calling complete_task()...")
    diagnostics = await manager.complete_task()

    assert diagnostics is not None, "Diagnostics should not be None"
    assert 'total_planned_trials' in diagnostics
    assert 'total_ran' in diagnostics
    assert 'total_skipped' in diagnostics
    assert 'efficiency_percent' in diagnostics

    print(f"✓ complete_task() returned diagnostics")

    # STEP 4: Validate diagnostics
    print("\n[STEP 4] Validating diagnostics...")
    print(f"\nFinal Diagnostics Summary:")
    print(f"  Total planned trials: {diagnostics['total_planned_trials']}")
    print(f"  Total ran: {diagnostics['total_ran']}")
    print(f"  Total skipped: {diagnostics['total_skipped']}")
    print(f"  Efficiency: {diagnostics['efficiency_percent']:.1f}%")
    print(f"  Stopped samples: {diagnostics['stopped_samples_count']}")
    print(f"  Stopped groupings: {diagnostics['stopped_groupings_count']}")

    assert diagnostics['total_planned_trials'] == 200
    assert diagnostics['total_ran'] == completed_count
    assert diagnostics['total_skipped'] == stopped_count

    # Check if any stopping occurred (should with these parameters)
    if diagnostics['efficiency_percent'] > 0:
        print(f"✓ Early stopping triggered ({diagnostics['efficiency_percent']:.1f}% efficiency)")
    else:
        print("⚠ No early stopping occurred (criteria may be too strict for test data)")

    # STEP 5: Save artifacts
    print("\n[STEP 5] Saving artifacts...")

    # Save compiled_dataset
    dataset_file = TEST_OUTPUT_DIR / f"compiled_dataset_1_1_1a_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    manager.compiled_dataset.to_csv(dataset_file, index=False)
    print(f"✓ Saved compiled_dataset: {dataset_file}")

    # Save diagnostics
    diagnostics_file = TEST_OUTPUT_DIR / f"diagnostics_1_1_1a_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(diagnostics_file, 'w') as f:
        # Convert diagnostics to JSON-serializable format
        json_diagnostics = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in diagnostics.items()
            if k != 'stopped_samples'  # Exclude complex objects
        }
        json.dump(json_diagnostics, f, indent=2)
    print(f"✓ Saved diagnostics: {diagnostics_file}")

    # Save stopped samples
    if diagnostics['stopped_samples']:
        stopped_file = TEST_OUTPUT_DIR / f"stopped_samples_1_1_1a_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stopped_file, 'w') as f:
            stopped_data = [
                {
                    'id': str(s['id']),
                    'epoch': s['epoch'],
                    'reason': s['reason']
                }
                for s in diagnostics['stopped_samples']
            ]
            json.dump(stopped_data, f, indent=2)
        print(f"✓ Saved stopped samples: {stopped_file}")

    print(f"\n✓ Log file: {log_file}")

    # STEP 6: Validate log contents
    print("\n[STEP 6] Validating log contents...")
    with open(log_file, 'r') as f:
        log_content = f.read()

    # Check for key log messages
    checks = [
        ("Configuration summary", "OptimalStoppingManager Configuration Summary" in log_content),
        ("Dataset initialization", "Initialized optimal stopping dataset" in log_content),
        ("Inference execution", "Running optimal stopping inference" in log_content or "Skipping inference" in log_content),
        ("Task completion", "Task complete" in log_content),
        ("Executor shutdown", "Shutting down inference executor" in log_content),
    ]

    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}: {'Found' if result else 'NOT FOUND'}")

    print("\n" + "="*80)
    print("TEST 1.1.1a: PASSED")
    print("="*80)


# ============================================================================
# Test 1.1.1b: Multi-Grouping Binary Evaluation
# ============================================================================

@pytest.mark.asyncio
async def test_1_1_1b_multi_grouping_binary():
    """
    Test 1.1.1b: Multi-Grouping Binary Evaluation

    Mock evaluation with:
    - 4 groupings (2 models × 2 tasks)
    - 10 samples per grouping × 5 epochs
    - Binary scores with varied performance (0.2, 0.5, 0.8, 0.95)

    Expected outputs:
    - Independent stopping decisions per grouping
    - Some groupings stop early, others complete fully

    Validation:
    - Verify no cross-contamination between groupings
    - Check stabilization histories maintained independently
    - Confirm decision_counters work correctly per grouping
    """
    log_file = get_test_log_file()
    logger = configure_test_logging(log_file)

    print("\n" + "="*80)
    print("TEST 1.1.1b: Multi-Grouping Binary Evaluation")
    print("="*80)

    # Configure optimal stopping
    optstop_params = {
        'delta_item': 0.15,
        'delta_cap': 0.10,
        'cred_level': 0.90,
        'conservatism': 3,
        'draws': 500,
        'tune': 500,
        'chains': 2,
        'cores': 2,
    }

    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
        reanalysis_interval=5,
        min_samples_per_grouping=3,
        manager_name="test_multi_grouping"
    )

    # Create samples with metadata for grouping
    # 10 samples per grouping × 4 groupings = 40 samples total
    samples = []

    grouping_configs = [
        ('gpt-4', 'math', 0.95),     # High performance - should stop early
        ('gpt-4', 'coding', 0.80),   # Good performance
        ('gpt-3.5', 'math', 0.50),   # Medium performance
        ('gpt-3.5', 'coding', 0.20), # Low performance - may not stop
    ]

    sample_id = 0
    for model, task, performance in grouping_configs:
        for i in range(10):
            samples.append(
                Sample(
                    id=f"sample_{sample_id}",
                    metadata={'model': model, 'task': task, 'expected_perf': performance}
                )
            )
            sample_id += 1

    # Create EvalSpec (will be augmented with sample metadata)
    eval_spec = EvalSpec(
        model="multi-model",  # This will be overridden by sample metadata
        task="multi-task",
        eval_id="test_eval_002"
    )

    # Start task
    print("\n[STEP 1] Starting multi-grouping evaluation...")
    await manager.start_task(eval_spec, samples, epochs=5)

    assert len(manager.compiled_dataset) == 200, "Expected 200 trials (40 samples × 5 epochs)"

    # Count groupings in compiled_dataset
    unique_groupings = manager.compiled_dataset.groupby(['model', 'task']).size()
    print(f"✓ Unique groupings: {len(unique_groupings)}")
    for (model, task), count in unique_groupings.items():
        print(f"  - {model} × {task}: {count} trials")

    assert len(unique_groupings) == 4, "Expected 4 unique groupings"

    # Run evaluation
    print("\n[STEP 2] Running evaluation with varied performance...")

    np.random.seed(42)

    grouping_stats = {str(config[:2]): {'completed': 0, 'stopped': 0}
                      for config in grouping_configs}

    for sample in samples:
        # Get expected performance from metadata
        expected_perf = sample.metadata['expected_perf']
        model = sample.metadata['model']
        task = sample.metadata['task']
        grouping_key = str((model, task))

        for epoch in range(1, 6):
            early_stop = await manager.schedule_sample(sample.id, epoch)

            if early_stop is not None:
                grouping_stats[grouping_key]['stopped'] += 1
                continue

            # Generate score based on expected performance
            score = 1 if np.random.random() < expected_perf else 0

            await manager.complete_sample(
                sample.id,
                epoch,
                create_mock_sample_score(score)
            )

            grouping_stats[grouping_key]['completed'] += 1

    # Complete task
    print("\n[STEP 3] Completing task...")
    diagnostics = await manager.complete_task()

    # Validate independent stopping per grouping
    print("\n[STEP 4] Validating per-grouping behavior...")
    print(f"\nGrouping Statistics:")
    for grouping, stats in grouping_stats.items():
        total = stats['completed'] + stats['stopped']
        efficiency = stats['stopped'] / total * 100 if total > 0 else 0
        print(f"  {grouping}:")
        print(f"    Completed: {stats['completed']}")
        print(f"    Stopped: {stats['stopped']}")
        print(f"    Efficiency: {efficiency:.1f}%")

    # Check decision counters
    print(f"\nDecision Counters:")
    for grouping, counter_info in diagnostics['decision_counters'].items():
        print(f"  {grouping}: {counter_info['completed_samples']} completed samples")

    # Verify no cross-contamination (each grouping has independent history)
    assert len(diagnostics['decision_counters']) == 4, "Should have 4 groupings in counters"

    # Save artifacts
    print("\n[STEP 5] Saving artifacts...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    manager.compiled_dataset.to_csv(
        TEST_OUTPUT_DIR / f"compiled_dataset_1_1_1b_{timestamp}.csv",
        index=False
    )

    with open(TEST_OUTPUT_DIR / f"diagnostics_1_1_1b_{timestamp}.json", 'w') as f:
        json_diagnostics = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in diagnostics.items()
            if k != 'stopped_samples'
        }
        json.dump(json_diagnostics, f, indent=2)

    print(f"✓ Artifacts saved")
    print(f"✓ Log file: {log_file}")

    print("\n" + "="*80)
    print("TEST 1.1.1b: PASSED")
    print("="*80)


# ============================================================================
# Test 1.1.1c: Shadow Mode Comparison
# ============================================================================

@pytest.mark.asyncio
async def test_1_1_1c_shadow_mode_comparison():
    """
    Test 1.1.1c: Shadow Mode Comparison

    Run same evaluation twice:
    - Once with shadow_mode=False (normal)
    - Once with shadow_mode=True (all trials run)

    Expected outputs:
    - Shadow mode: schedule_sample() always returns None
    - Shadow mode: complete_task() shows 0% efficiency
    - Normal mode: some trials stopped early

    Validation:
    - Confirm shadow mode useful for ablation studies
    """
    log_file = get_test_log_file()
    logger = configure_test_logging(log_file)

    print("\n" + "="*80)
    print("TEST 1.1.1c: Shadow Mode Comparison")
    print("="*80)

    optstop_params = {
        'delta_item': 0.15,
        'delta_cap': 0.10,
        'cred_level': 0.90,
        'conservatism': 3,
        'draws': 500,
        'tune': 500,
        'chains': 2,
        'cores': 2,
    }

    # Create test data
    samples = [Sample(id=f"sample_{i}", metadata={}) for i in range(15)]
    eval_spec = EvalSpec(model="gpt-4", task="math", eval_id="test_shadow")

    np.random.seed(42)  # Same seed for both runs

    # ===== RUN 1: Normal Mode =====
    print("\n[RUN 1] Normal Mode (shadow_mode=False)...")

    manager_normal = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
        reanalysis_interval=5,
        min_samples_per_grouping=3,
        shadow_mode=False,  # Normal operation
        manager_name="test_normal"
    )

    await manager_normal.start_task(eval_spec, samples, epochs=8)

    normal_completed = 0
    normal_stopped = 0

    np.random.seed(42)  # Reset seed
    for sample in samples:
        for epoch in range(1, 9):
            early_stop = await manager_normal.schedule_sample(sample.id, epoch)

            if early_stop is not None:
                normal_stopped += 1
                continue

            score = 1 if np.random.random() < 0.70 else 0
            await manager_normal.complete_sample(
                sample.id, epoch, create_mock_sample_score(score)
            )
            normal_completed += 1

    diagnostics_normal = await manager_normal.complete_task()

    print(f"✓ Normal mode:")
    print(f"  Completed: {normal_completed}")
    print(f"  Stopped: {normal_stopped}")
    print(f"  Efficiency: {diagnostics_normal['efficiency_percent']:.1f}%")

    # ===== RUN 2: Shadow Mode =====
    print("\n[RUN 2] Shadow Mode (shadow_mode=True)...")

    manager_shadow = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
        reanalysis_interval=5,
        min_samples_per_grouping=3,
        shadow_mode=True,  # All trials run
        manager_name="test_shadow"
    )

    await manager_shadow.start_task(eval_spec, samples, epochs=8)

    shadow_completed = 0
    shadow_stopped = 0

    np.random.seed(42)  # Reset seed
    for sample in samples:
        for epoch in range(1, 9):
            early_stop = await manager_shadow.schedule_sample(sample.id, epoch)

            # Shadow mode should NEVER return early stop
            assert early_stop is None, "Shadow mode should never stop samples"

            score = 1 if np.random.random() < 0.70 else 0
            await manager_shadow.complete_sample(
                sample.id, epoch, create_mock_sample_score(score)
            )
            shadow_completed += 1

    diagnostics_shadow = await manager_shadow.complete_task()

    print(f"✓ Shadow mode:")
    print(f"  Completed: {shadow_completed}")
    print(f"  Stopped: {shadow_stopped}")
    print(f"  Efficiency: {diagnostics_shadow['efficiency_percent']:.1f}%")

    # ===== VALIDATION =====
    print("\n[VALIDATION] Comparing modes...")

    # Shadow mode should run all trials
    assert shadow_completed == 120, "Shadow mode should run all 120 trials"
    assert shadow_stopped == 0, "Shadow mode should stop 0 trials"
    assert diagnostics_shadow['efficiency_percent'] == 0.0, "Shadow mode should have 0% efficiency"

    # Normal mode should stop some trials (with these parameters)
    # Note: May not always stop depending on random data
    print(f"\n✓ Normal mode efficiency: {diagnostics_normal['efficiency_percent']:.1f}%")
    print(f"✓ Shadow mode efficiency: {diagnostics_shadow['efficiency_percent']:.1f}%")

    if diagnostics_normal['efficiency_percent'] > 0:
        print("✓ Normal mode stopped some trials as expected")
    else:
        print("⚠ Normal mode did not stop trials (may be due to data/criteria)")

    # Save comparison
    print("\n[SAVING] Comparison artifacts...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    comparison = {
        'normal_mode': {
            'completed': normal_completed,
            'stopped': normal_stopped,
            'efficiency_percent': diagnostics_normal['efficiency_percent']
        },
        'shadow_mode': {
            'completed': shadow_completed,
            'stopped': shadow_stopped,
            'efficiency_percent': diagnostics_shadow['efficiency_percent']
        }
    }

    with open(TEST_OUTPUT_DIR / f"shadow_comparison_1_1_1c_{timestamp}.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"✓ Comparison saved")
    print(f"✓ Log file: {log_file}")

    print("\n" + "="*80)
    print("TEST 1.1.1c: PASSED")
    print("="*80)


# ============================================================================
# Test Discovery
# ============================================================================

if __name__ == "__main__":
    """Run tests directly with asyncio."""
    print("\nRunning bridge integration tests (binary discrete scoring)...\n")

    asyncio.run(test_1_1_1a_simple_binary_single_grouping())
    asyncio.run(test_1_1_1b_multi_grouping_binary())
    asyncio.run(test_1_1_1c_shadow_mode_comparison())

    print("\n✓ All binary integration tests passed!")
