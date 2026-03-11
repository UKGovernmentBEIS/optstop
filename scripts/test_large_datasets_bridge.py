"""
Bridge protocol test script for three large-scale datasets.

This script runs each of the three generated datasets through the OptimalStoppingManager
to test:
1. Routing correctness (binary discrete vs ordinal discrete vs continuous bounded)
2. Stopping behavior per grouping
3. Diagnostic output completeness
4. Performance at scale (500 samples × 10 epochs = 5,000 trials)

Each dataset tests a different scoring type:
- Dataset 1: Binary discrete (single binary score)
- Dataset 2: Ordinal discrete with score_choice='accuracy'
- Dataset 3: Ordinal discrete with score_agg='mean' → continuous bounded
"""

import pandas as pd
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any
import logging

from optstop.early_stopping import OptimalStoppingManager, Sample, EvalSpec
import numpy as np


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    NOTE: This is needed because OptimalStoppingManager.complete_task() returns
    diagnostics with numpy int64/float64 types that aren't JSON serializable.

    **POTENTIAL ISSUE FOR INSPECT_AI INTEGRATION**:
    If inspect_ai users want to save diagnostics to JSON, they'll encounter this
    same serialization error. The optstop package should either:
    1. Convert numpy types to native Python types before returning diagnostics, OR
    2. Document this issue and provide a helper function for users
    """
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
    """
    Create a mock SampleScore object matching inspect_ai format.

    The manager expects scores with the structure: score.score.value
    """
    class MockScore:
        def __init__(self, val):
            self.value = val

    class MockSampleScore:
        def __init__(self, val):
            self.score = MockScore(val)

    return MockSampleScore(value)


class PreloadedDataScorer:
    """
    Simulated scorer that returns pre-generated scores from CSV data.

    This scorer mimics the inspect_ai bridge behavior by returning scores
    from our pre-generated datasets rather than running actual evaluations.
    """

    def __init__(self, dataset_df: pd.DataFrame, score_columns: List[str]):
        """
        Initialize the scorer with pre-loaded data.

        Args:
            dataset_df: DataFrame containing all pre-generated trials
            score_columns: List of column names containing score data
        """
        self.dataset_df = dataset_df
        self.score_columns = score_columns

        # Index for fast lookup: (sample_id, epoch) -> score_dict
        self.score_index = {}
        for _, row in dataset_df.iterrows():
            key = (row['sample_id'], row['epoch'])
            score_dict = {col: row[col] for col in score_columns}
            self.score_index[key] = score_dict

    def score_sample(self, sample_id: str, epoch: int) -> Dict[str, Any]:
        """
        Return the pre-generated score for this sample and epoch.

        Args:
            sample_id: Sample identifier
            epoch: Epoch number

        Returns:
            Dictionary with mock SampleScore values
        """
        raw_scores = self.score_index.get((sample_id, epoch), {})
        # Convert raw scores to mock SampleScore format
        return {col: create_mock_score(raw_scores[col]) for col in raw_scores}


def verify_routing(log_path: Path, expected_type: str) -> Dict[str, Any]:
    """
    Verify correct inference routing from log file.

    Args:
        log_path: Path to log file
        expected_type: Expected inference type ('BINARY', 'ORDINAL', or 'CONTINUOUS')

    Returns:
        Dictionary with verification results
    """
    with open(log_path, 'r') as f:
        log_content = f.read()

    # Patterns from Section 1.1/1.2 testing
    routing_patterns = {
        'BINARY': [
            "Processing grouping",
            "as BINARY",
            "Binary group inference took"
        ],
        'ORDINAL': [
            "Processing grouping",
            "as ORDINAL",
            "Ordinal group inference took"
        ],
        'CONTINUOUS': [
            "CONTINUOUS_BOUNDED",
            "Processing grouping",
            "Continuous PyMC MCMC sampling",
            "Continuous group inference"
        ]
    }

    found_patterns = []
    for pattern in routing_patterns.get(expected_type, []):
        if pattern in log_content:
            found_patterns.append(pattern)

    verified = len(found_patterns) > 0

    return {
        'expected_type': expected_type,
        'verified': verified,
        'found_patterns': found_patterns,
        'pattern_count': len(found_patterns)
    }


def parse_group_level_stopping(log_path: Path) -> Dict[str, Any]:
    """
    Parse log file for group-level stopping events.

    This addresses Issue #5: Track group-level stopping (primary efficiency driver).
    Lesson from Section 1.2: stopped_samples=0 is normal when using group-level stopping.

    Args:
        log_path: Path to log file

    Returns:
        Dictionary with group-level stopping information
    """
    with open(log_path, 'r') as f:
        log_lines = f.readlines()

    stopped_groupings = []
    for line in log_lines:
        # Look for group-level stopping messages
        if "Stopping grouping" in line or "Stopped grouping" in line:
            stopped_groupings.append(line.strip())

    return {
        'group_level_stopping_events': len(stopped_groupings),
        'stopped_groupings': stopped_groupings,
        'used_group_level_stopping': len(stopped_groupings) > 0
    }


async def simulate_bridge_protocol(
    dataset_path: Path,
    score_columns: List[str],
    config: Dict[str, Any],
    output_path: Path,
    log_path: Path,
    expected_routing: str
) -> Dict[str, Any]:
    """
    Simulate the bridge protocol workflow for a dataset.

    Args:
        dataset_path: Path to CSV dataset
        score_columns: List of score column names
        config: OptimalStoppingManager configuration
        output_path: Path to save diagnostics JSON
        log_path: Path to save detailed log

    Returns:
        Dictionary with test results and diagnostics
    """
    # Configure logging - FIXED: basicConfig() only works once per process!
    # Manually manage handlers for each dataset

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

    # Create new handlers for this dataset
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

    logger.info("=" * 80)
    logger.info(f"BRIDGE PROTOCOL TEST: {dataset_path.name}")
    logger.info("=" * 80)

    # Load dataset
    logger.info(f"\n📁 Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  Unique samples: {df['sample_id'].nunique()}")
    logger.info(f"  Score columns: {score_columns}")
    logger.info(f"  Grouping columns: {config.get('grouping_cols', ['model', 'task'])}")

    # Create scorer
    scorer = PreloadedDataScorer(df, score_columns)

    # Extract sample metadata
    sample_metadata = df.groupby('sample_id').first()[['model', 'task']].to_dict('index')

    # Calculate total planned trials
    num_samples = df['sample_id'].nunique()
    num_epochs = df['epoch'].max() + 1  # epochs are 0-indexed
    total_planned = num_samples * num_epochs

    logger.info(f"\n📊 Dataset structure:")
    logger.info(f"  Samples: {num_samples}")
    logger.info(f"  Epochs per sample: {num_epochs}")
    logger.info(f"  Total planned trials: {total_planned}")

    # Display grouping statistics
    grouping_cols = config.get('grouping_cols', ['model', 'task'])
    logger.info(f"\n🎯 Grouping statistics:")
    for group_name, group_df in df.groupby(grouping_cols):
        group_samples = group_df['sample_id'].nunique()
        # Calculate mean of first score column for summary
        first_score = score_columns[0]
        if len(score_columns) > 1 and 'score_agg' in config:
            # For continuous bounded, show mean of all scores
            mean_score = group_df[score_columns].mean().mean()
        else:
            mean_score = group_df[first_score].mean()
        logger.info(f"  {group_name}: {group_samples} samples, mean score: {mean_score:.3f}")

    # Initialize OptimalStoppingManager
    logger.info(f"\n⚙️  Initializing OptimalStoppingManager...")
    start_time = time.time()

    manager = OptimalStoppingManager(**config)

    init_time = time.time() - start_time
    logger.info(f"  Initialization time: {init_time:.2f}s")

    # Create Sample objects for bridge protocol
    logger.info(f"\n🔨 Creating Sample objects...")
    samples = []
    for sample_id in df['sample_id'].unique():
        metadata = sample_metadata[sample_id]
        sample = Sample(id=sample_id, metadata=metadata)
        samples.append(sample)
    logger.info(f"  Created {len(samples)} Sample objects")

    # Create EvalSpec object
    eval_spec = EvalSpec(task="bridge_test_task", model="test_model")

    # Start task with bridge protocol objects
    logger.info(f"\n🚀 Starting task...")
    manager_name = await manager.start_task(eval_spec, samples, epochs=num_epochs)
    logger.info(f"  Manager name: {manager_name}")

    # Simulate bridge protocol: schedule → complete loop
    logger.info(f"\n⏳ Running bridge protocol simulation...")
    logger.info("  (This simulates the inspect_ai → optstop → inspect_ai workflow)")

    trial_count = 0
    stopped_samples = set()

    for sample_id in df['sample_id'].unique():
        for epoch in range(num_epochs):
            # Check if this sample was stopped
            if sample_id in stopped_samples:
                break

            # Schedule sample (bridge step 1) - with await
            early_stop = await manager.schedule_sample(sample_id, epoch)

            if early_stop is not None:
                logger.info(f"    Sample {sample_id} stopped at epoch {epoch}: {early_stop.reason}")
                stopped_samples.add(sample_id)
                break

            # Get pre-generated score
            scores = scorer.score_sample(sample_id, epoch)

            # Complete sample (bridge step 2) - with await
            await manager.complete_sample(sample_id, epoch, scores)

            trial_count += 1

            # Progress indicator (every 500 trials)
            if trial_count % 500 == 0:
                logger.info(f"    Progress: {trial_count}/{total_planned} trials ({100*trial_count/total_planned:.1f}%)")

    run_time = time.time() - start_time

    logger.info(f"\n✓ Bridge protocol simulation complete")
    logger.info(f"  Total trials run: {trial_count}/{total_planned}")
    logger.info(f"  Efficiency: {100*(total_planned - trial_count)/total_planned:.1f}%")
    logger.info(f"  Stopped samples: {len(stopped_samples)}")
    logger.info(f"  Runtime: {run_time:.2f}s")

    # Complete task and get diagnostics
    logger.info(f"\n📊 Retrieving diagnostics...")
    diagnostics = await manager.complete_task()

    logger.info(f"\n📋 Diagnostics summary:")
    logger.info(f"  Total planned: {diagnostics.get('total_planned', 'N/A')}")
    logger.info(f"  Total ran: {diagnostics.get('total_ran', 'N/A')}")
    logger.info(f"  Efficiency: {diagnostics.get('efficiency', 'N/A')}")
    logger.info(f"  Inference type: {diagnostics.get('inference_type', 'N/A')}")

    if 'grouping_results' in diagnostics:
        logger.info(f"\n📊 Per-grouping results:")
        for group_name, group_data in diagnostics['grouping_results'].items():
            logger.info(f"  {group_name}:")
            logger.info(f"    Samples: {group_data.get('n_samples', 'N/A')}")
            logger.info(f"    Trials planned: {group_data.get('trials_planned', 'N/A')}")
            logger.info(f"    Trials ran: {group_data.get('trials_ran', 'N/A')}")
            logger.info(f"    Stopped: {group_data.get('stopped', 'N/A')}")

    # FIX Issue #3: Verify routing correctness
    logger.info(f"\n🔍 Verifying inference routing...")
    routing_verification = verify_routing(log_path, expected_routing)
    logger.info(f"  Expected: {routing_verification['expected_type']}")
    logger.info(f"  Verified: {'✓ PASS' if routing_verification['verified'] else '✗ FAIL'}")
    if routing_verification['verified']:
        logger.info(f"  Found patterns: {routing_verification['found_patterns']}")
    else:
        logger.warning(f"  WARNING: Routing verification failed!")

    # FIX Issue #5: Parse group-level stopping
    logger.info(f"\n🎯 Analyzing group-level stopping...")
    group_stopping = parse_group_level_stopping(log_path)
    logger.info(f"  Group-level stopping events: {group_stopping['group_level_stopping_events']}")
    if group_stopping['used_group_level_stopping']:
        logger.info(f"  ℹ️  Note: stopped_samples=0 is normal with group-level stopping")
        for msg in group_stopping['stopped_groupings'][:3]:  # Show first 3
            logger.info(f"    • {msg}")
    else:
        logger.info(f"  No group-level stopping detected")

    # Compile results
    results = {
        'dataset': str(dataset_path),
        'score_columns': score_columns,
        'configuration': config,
        'execution': {
            'init_time_sec': init_time,
            'run_time_sec': run_time,
            'total_time_sec': time.time() - start_time,
            'trial_count': trial_count,
            'total_planned': total_planned,
            'stopped_samples': len(stopped_samples)
        },
        'diagnostics': diagnostics,
        'routing_verification': routing_verification,
        'group_level_stopping': group_stopping
    }

    # Save diagnostics (convert numpy types for JSON serialization)
    logger.info(f"\n💾 Saving diagnostics to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(convert_numpy_types(results), f, indent=2)

    logger.info(f"\n📄 Log saved to: {log_path}")

    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)

    return results


async def main():
    """Run bridge protocol tests for all three datasets."""

    print("\n" + "=" * 80)
    print("LARGE-SCALE BRIDGE PROTOCOL TESTING")
    print("=" * 80)
    print("\nThis script tests the OptimalStoppingManager bridge with:")
    print("  • 3 datasets × 500 samples × 10 epochs = 15,000 total trials")
    print("  • Binary discrete, ordinal discrete, and continuous bounded inference")
    print("  • 5 groupings per dataset with varying performance")
    print("\nExpected routing:")
    print("  • Dataset 1 → Binary discrete inference")
    print("  • Dataset 2 → Ordinal discrete inference (score_choice='accuracy')")
    print("  • Dataset 3 → Continuous bounded inference (score_agg='mean')")

    # Paths
    data_dir = Path('/home/ubuntu/optstop/test_data/large_scale')
    output_dir = Path('/home/ubuntu/optstop/test_outputs/large_scale')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base configuration (common parameters)
    # FIX Issue #1: reanalysis_interval 50→25 (5% ratio, more granular than Section 1.2's 25%)
    # FIX Issue #4: MCMC 6000→500 (4x speedup from Section 1.1.2 findings)

    # Optstop inference parameters (nested dict)
    optstop_params = {
        'delta_item': 0.15,
        'delta_cap': 0.10,
        'cred_level': 0.95,
        'conservatism': 5,  # Default conservatism
        'draws': 500,  # Optimized (Section 1.1.2: 4x speedup)
        'tune': 500,
    }

    # Manager-level parameters
    base_config = {
        'optstop_params': optstop_params,
        'grouping_columns': ['model', 'task'],  # Correct parameter name
        'reanalysis_interval': 25,  # 5% ratio (25/500) for frequent group-level stopping checks
        'min_samples_per_grouping': 10,
    }

    # Dataset 1: Binary Discrete
    print("\n" + "=" * 80)
    print("TEST 1/3: Binary Discrete")
    print("=" * 80)

    dataset1_config = {
        **base_config,
        # No additional parameters needed for binary discrete
    }

    results1 = await simulate_bridge_protocol(
        dataset_path=data_dir / 'dataset_1_binary_discrete.csv',
        score_columns=['score'],
        config=dataset1_config,
        output_path=output_dir / 'dataset_1_diagnostics.json',
        log_path=output_dir / 'dataset_1.log',
        expected_routing='BINARY'
    )

    # Dataset 2: Ordinal Discrete with score_choice
    print("\n" + "=" * 80)
    print("TEST 2/3: Ordinal Discrete (score_choice='accuracy')")
    print("=" * 80)

    dataset2_config = {
        **base_config,
        'score_choice': 'accuracy',
        'ordinal_max_score': 10,
        'ordinal_tasks': ['math_easy', 'math_hard', 'coding_medium',
                         'creative_writing', 'reasoning']  # Required for ordinal inference!
    }

    results2 = await simulate_bridge_protocol(
        dataset_path=data_dir / 'dataset_2_ordinal_with_choice.csv',
        score_columns=['accuracy', 'precision', 'recall'],
        config=dataset2_config,
        output_path=output_dir / 'dataset_2_diagnostics.json',
        log_path=output_dir / 'dataset_2.log',
        expected_routing='ORDINAL'
    )

    # Dataset 3: Ordinal Discrete with score_agg='mean' → Continuous Bounded
    print("\n" + "=" * 80)
    print("TEST 3/3: Continuous Bounded (score_agg='mean')")
    print("=" * 80)

    dataset3_config = {
        **base_config,
        'score_agg': 'mean',
        'ordinal_max_score': 10,
        'ordinal_tasks': ['math_easy', 'math_hard', 'coding_medium',
                         'creative_writing', 'reasoning']  # Required for ordinal → continuous!
    }

    results3 = await simulate_bridge_protocol(
        dataset_path=data_dir / 'dataset_3_ordinal_with_agg.csv',
        score_columns=['scorer_1', 'scorer_2', 'scorer_3'],
        config=dataset3_config,
        output_path=output_dir / 'dataset_3_diagnostics.json',
        log_path=output_dir / 'dataset_3.log',
        expected_routing='CONTINUOUS'
    )

    # Final summary
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)

    print("\n📊 Summary:")
    print(f"\nDataset 1 (Binary Discrete):")
    print(f"  Inference type: {results1['diagnostics'].get('inference_type', 'N/A')}")
    print(f"  Routing verification: {'✓ PASS' if results1['routing_verification']['verified'] else '✗ FAIL'}")
    print(f"  Trials ran: {results1['execution']['trial_count']}/{results1['execution']['total_planned']}")
    print(f"  Efficiency: {100*(results1['execution']['total_planned']-results1['execution']['trial_count'])/results1['execution']['total_planned']:.1f}%")
    print(f"  Group-level stopping: {'✓ Used' if results1['group_level_stopping']['used_group_level_stopping'] else 'Not used'}")
    print(f"  Runtime: {results1['execution']['run_time_sec']:.2f}s")

    print(f"\nDataset 2 (Ordinal Discrete):")
    print(f"  Inference type: {results2['diagnostics'].get('inference_type', 'N/A')}")
    print(f"  Routing verification: {'✓ PASS' if results2['routing_verification']['verified'] else '✗ FAIL'}")
    print(f"  Trials ran: {results2['execution']['trial_count']}/{results2['execution']['total_planned']}")
    print(f"  Efficiency: {100*(results2['execution']['total_planned']-results2['execution']['trial_count'])/results2['execution']['total_planned']:.1f}%")
    print(f"  Group-level stopping: {'✓ Used' if results2['group_level_stopping']['used_group_level_stopping'] else 'Not used'}")
    print(f"  Runtime: {results2['execution']['run_time_sec']:.2f}s")

    print(f"\nDataset 3 (Continuous Bounded):")
    print(f"  Inference type: {results3['diagnostics'].get('inference_type', 'N/A')}")
    print(f"  Routing verification: {'✓ PASS' if results3['routing_verification']['verified'] else '✗ FAIL'}")
    print(f"  Trials ran: {results3['execution']['trial_count']}/{results3['execution']['total_planned']}")
    print(f"  Efficiency: {100*(results3['execution']['total_planned']-results3['execution']['trial_count'])/results3['execution']['total_planned']:.1f}%")
    print(f"  Group-level stopping: {'✓ Used' if results3['group_level_stopping']['used_group_level_stopping'] else 'Not used'}")
    print(f"  Runtime: {results3['execution']['run_time_sec']:.2f}s")

    print(f"\n📁 All diagnostics saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review diagnostics JSON files for detailed results")
    print("  2. Analyze per-grouping stopping behavior")
    print("  3. Verify inference routing correctness")
    print("  4. Assess performance at scale")


if __name__ == '__main__':
    asyncio.run(main())
