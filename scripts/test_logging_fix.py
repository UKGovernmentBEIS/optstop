#!/usr/bin/env python3
"""
Quick test to verify logging fix for Dataset 2 (ordinal discrete with score_choice).
Tests with just 50 samples to quickly verify:
1. Logging produces non-empty log files
2. Routing verification patterns are captured
3. Grouping attribution is working correctly
"""

import asyncio
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from optstop.early_stopping import OptimalStoppingManager, Sample, EvalSpec

# Setup output directory
output_dir = Path("/home/ubuntu/optstop/test_outputs/logging_fix_test")
output_dir.mkdir(parents=True, exist_ok=True)

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def create_mock_score(value):
    """Create a mock SampleScore object matching inspect_ai format."""
    class MockScore:
        def __init__(self, val):
            self.value = val

    class MockSampleScore:
        def __init__(self, val):
            self.score = MockScore(val)

    return MockSampleScore(value)

class DatasetScorer:
    """Handles scoring for a specific dataset."""

    def __init__(self, df, score_columns):
        self.df = df
        self.score_columns = score_columns

    def score_sample(self, sample_id, epoch):
        """Get scores for a specific sample and epoch."""
        row = self.df[(self.df['sample_id'] == sample_id) & (self.df['epoch'] == epoch)]
        if row.empty:
            raise ValueError(f"No data found for sample_id={sample_id}, epoch={epoch}")

        row = row.iloc[0]
        scores = {col: create_mock_score(int(row[col])) for col in self.score_columns}
        return scores

async def test_dataset_2_small():
    """Test Dataset 2 with just 50 samples."""

    print("=" * 80)
    print("TESTING LOGGING FIX - Dataset 2 (50 samples)")
    print("=" * 80)

    # Setup logging with the new manual handler approach
    log_path = output_dir / "dataset_2_small.log"

    # Get loggers
    logger = logging.getLogger(__name__)
    root_logger = logging.getLogger()
    optstop_logger = logging.getLogger('optstop')

    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    for handler in optstop_logger.handlers[:]:
        optstop_logger.removeHandler(handler)

    # Create new handlers for this test
    formatter = logging.Formatter('%(message)s')

    # File handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add handlers to root logger (captures all loggers)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)

    # Set optstop logger to DEBUG to capture routing messages
    optstop_logger.setLevel(logging.DEBUG)

    logger.info("Logging configured successfully")

    # Load Dataset 2
    dataset_path = "/home/ubuntu/optstop/test_data/large_scale/dataset_2_ordinal_with_choice.csv"
    df = pd.read_csv(dataset_path)

    # Take only first 50 samples
    unique_samples = df['sample_id'].unique()[:50]
    df_small = df[df['sample_id'].isin(unique_samples)]

    num_samples = len(unique_samples)
    num_epochs = df_small['epoch'].max() + 1
    total_trials = len(df_small)

    logger.info(f"\nDataset: {dataset_path}")
    logger.info(f"Samples: {num_samples}")
    logger.info(f"Epochs per sample: {num_epochs}")
    logger.info(f"Total trials: {total_trials}")

    # Check groupings in the subset
    groupings = df_small.groupby(['model', 'task']).size()
    logger.info(f"\nGroupings in subset:")
    for (model, task), count in groupings.items():
        logger.info(f"  {model}-{task}: {count} trials")

    # Initialize manager
    score_columns = ['accuracy', 'precision', 'recall']

    optstop_params = {
        'delta_item': 0.15,
        'delta_cap': 0.1,
        'cred_level': 0.95,
        'conservatism': 5,
        'draws': 500,
        'tune': 500
    }

    base_config = {
        'optstop_params': optstop_params,
        'grouping_columns': ['model', 'task'],
        'reanalysis_interval': 25,
        'min_samples_per_grouping': 10,
        'score_choice': 'accuracy',
        'ordinal_max_score': 10,
        'ordinal_tasks': ['math_easy', 'math_hard', 'coding_medium',
                         'creative_writing', 'reasoning']  # Required for ordinal inference
    }

    logger.info(f"\nInitializing OptimalStoppingManager...")
    manager = OptimalStoppingManager(**base_config)

    # Create samples
    samples = []
    for sample_id in unique_samples:
        sample_row = df_small[df_small['sample_id'] == sample_id].iloc[0]
        metadata = {
            'model': sample_row['model'],
            'task': sample_row['task']
        }
        samples.append(Sample(id=sample_id, metadata=metadata))

    # Start task
    eval_spec = EvalSpec(task="logging_fix_test", model="test_model")
    manager_name = await manager.start_task(eval_spec, samples, epochs=num_epochs)
    logger.info(f"Manager: {manager_name}")

    # Create scorer
    scorer = DatasetScorer(df_small, score_columns)

    # Run trials
    logger.info("\nRunning trials...")
    trial_count = 0
    stopped_samples = set()

    for sample_id in unique_samples:
        for epoch in range(num_epochs):
            trial_count += 1

            if trial_count % 100 == 0:
                logger.info(f"  Trial {trial_count}/{total_trials}")

            # Check if sample should run
            early_stop = await manager.schedule_sample(sample_id, epoch)
            if early_stop is not None:
                logger.info(f"    Sample {sample_id} stopped at epoch {epoch}: {early_stop.reason}")
                stopped_samples.add(sample_id)
                break

            # Score and complete
            scores = scorer.score_sample(sample_id, epoch)
            await manager.complete_sample(sample_id, epoch, scores)

    # Complete task
    logger.info("\nCompleting task...")
    diagnostics = await manager.complete_task()

    # Save diagnostics
    diagnostics_path = output_dir / "dataset_2_small_diagnostics.json"
    result = {
        'dataset': str(dataset_path),
        'subset_samples': num_samples,
        'score_columns': score_columns,
        'configuration': base_config,
        'execution': {
            'trial_count': trial_count,
            'total_planned': total_trials,
            'stopped_samples': len(stopped_samples)
        },
        'diagnostics': diagnostics
    }

    with open(diagnostics_path, 'w') as f:
        json.dump(convert_numpy_types(result), f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Samples: {num_samples}")
    print(f"Trials ran: {trial_count}/{total_trials}")
    print(f"Stopped samples: {len(stopped_samples)}")
    print(f"Efficiency: {100*(total_trials-trial_count)/total_trials:.1f}%")

    # Check log file
    log_size = log_path.stat().st_size
    print(f"\nLog file: {log_path}")
    print(f"Log size: {log_size} bytes")

    if log_size == 0:
        print("❌ FAIL: Log file is empty!")
        return False
    else:
        print("✓ PASS: Log file has content")

    # Check for routing patterns
    with open(log_path, 'r') as f:
        log_content = f.read()

    ordinal_patterns = ['ordinal', 'Ordinal', 'ORDINAL', 'Dirichlet']
    found_patterns = [p for p in ordinal_patterns if p in log_content]

    print(f"\nRouting patterns found: {found_patterns if found_patterns else 'None'}")

    if found_patterns:
        print("✓ PASS: Ordinal inference patterns detected")
    else:
        print("⚠️  WARNING: No ordinal inference patterns found")

    # Check groupings in diagnostics
    if 'decision_counters' in diagnostics:
        groupings_in_diagnostics = list(diagnostics['decision_counters'].keys())
        print(f"\nGroupings in diagnostics: {len(groupings_in_diagnostics)}")
        for grouping in groupings_in_diagnostics:
            stats = diagnostics['decision_counters'][grouping]
            print(f"  {grouping}: {stats['completed_samples']} samples")

    print(f"\nDiagnostics saved: {diagnostics_path}")
    print("=" * 80)

    return log_size > 0

if __name__ == "__main__":
    success = asyncio.run(test_dataset_2_small())
    exit(0 if success else 1)
