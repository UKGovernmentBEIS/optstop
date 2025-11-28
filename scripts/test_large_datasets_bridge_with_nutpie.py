"""
Enhanced bridge protocol test script with nutpie tracking and performance comparison.

This script extends test_large_datasets_bridge.py to:
1. Track nutpie detection and usage
2. Capture per-inference timing
3. Compare performance with previous runs
4. Use ordinal_inference='hybrid' for dataset 2
5. Generate comparison reports

Usage:
    python scripts/test_large_datasets_bridge_with_nutpie.py
"""

import pandas as pd
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any
import logging
import shutil
from datetime import datetime

from optstop.early_stopping import OptimalStoppingManager, Sample, EvalSpec
from optstop import gpu_utils
import numpy as np


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


class PreloadedDataScorer:
    """Simulated scorer that returns pre-generated scores from CSV data."""

    def __init__(self, dataset_df: pd.DataFrame, score_columns: List[str]):
        self.dataset_df = dataset_df
        self.score_columns = score_columns

        # Index for fast lookup: (sample_id, epoch) -> score_dict
        self.score_index = {}
        for _, row in dataset_df.iterrows():
            key = (row['sample_id'], row['epoch'])
            score_dict = {col: row[col] for col in score_columns}
            self.score_index[key] = score_dict

    def score_sample(self, sample_id: str, epoch: int) -> Dict[str, Any]:
        """Return the pre-generated score for this sample and epoch."""
        raw_scores = self.score_index.get((sample_id, epoch), {})
        return {col: create_mock_score(raw_scores[col]) for col in raw_scores}


def check_nutpie_environment() -> Dict[str, Any]:
    """
    Check nutpie availability and version.

    Returns:
        Dictionary with nutpie detection results
    """
    nutpie_available, nutpie_version = gpu_utils.check_nutpie_available()

    result = {
        'nutpie_available': nutpie_available,
        'nutpie_version': nutpie_version,
        'detection_time': datetime.now().isoformat()
    }

    return result


def parse_inference_timing(log_path: Path) -> Dict[str, Any]:
    """
    Parse log file for inference timing information.

    Returns:
        Dictionary with inference timing statistics
    """
    with open(log_path, 'r') as f:
        log_content = f.read()

    # Parse timing lines (e.g., "Binary group inference took 12.34s")
    timing_patterns = {
        'binary': r'Binary group inference took ([\d.]+)s',
        'ordinal': r'Ordinal group inference took ([\d.]+)s',
        'continuous': r'Continuous group inference took ([\d.]+)s',
        'modal': r'Modal inference took ([\d.]+)s',
        'entropy': r'Entropy inference took ([\d.]+)s',
    }

    import re
    timing_data = {}

    for pathway, pattern in timing_patterns.items():
        matches = re.findall(pattern, log_content)
        if matches:
            times = [float(m) for m in matches]
            timing_data[pathway] = {
                'count': len(times),
                'total_sec': sum(times),
                'mean_sec': sum(times) / len(times),
                'min_sec': min(times),
                'max_sec': max(times),
                'times': times[:10]  # Save first 10 for inspection
            }

    return timing_data


def parse_sampler_selection(log_path: Path) -> Dict[str, Any]:
    """
    Parse log file for sampler selection (nutpie vs pymc).

    Returns:
        Dictionary with sampler selection information
    """
    with open(log_path, 'r') as f:
        log_content = f.read()

    # Look for sampler selection messages (check both patterns)
    nutpie_used = ('Selected NUTS sampler: nutpie' in log_content or
                   'Configured sampling for CPU: nutpie' in log_content)
    pymc_used = ('Selected NUTS sampler: pymc' in log_content or
                 'Configured sampling for CPU: pymc' in log_content)
    numpyro_used = ('Selected NUTS sampler: numpyro' in log_content or
                    'GPU acceleration with JAX/numpyro' in log_content)

    # Extract version if available
    nutpie_version = None
    if nutpie_used:
        import re
        match = re.search(r'nutpie.*?v([\d.]+)', log_content)
        if match:
            nutpie_version = match.group(1)

    return {
        'nutpie_used': nutpie_used,
        'pymc_used': pymc_used,
        'numpyro_used': numpyro_used,
        'nutpie_version': nutpie_version,
        'sampler': 'nutpie' if nutpie_used else ('numpyro' if numpyro_used else 'pymc')
    }


def compare_with_baseline(current_results: Dict[str, Any], baseline_path: Path) -> Dict[str, Any]:
    """
    Compare current results with baseline from previous run.

    Returns:
        Dictionary with comparison statistics
    """
    if not baseline_path.exists():
        return {'comparison_available': False, 'reason': 'No baseline file found'}

    with open(baseline_path, 'r') as f:
        baseline = json.load(f)

    current_exec = current_results['execution']
    baseline_exec = baseline['execution']

    comparison = {
        'comparison_available': True,
        'baseline_date': baseline.get('test_date', 'unknown'),
        'current_date': current_results.get('test_date', 'unknown'),
        'runtime_comparison': {
            'baseline_sec': baseline_exec['run_time_sec'],
            'current_sec': current_exec['run_time_sec'],
            'speedup': baseline_exec['run_time_sec'] / current_exec['run_time_sec'] if current_exec['run_time_sec'] > 0 else 0,
            'time_saved_sec': baseline_exec['run_time_sec'] - current_exec['run_time_sec'],
            'percent_change': 100 * (baseline_exec['run_time_sec'] - current_exec['run_time_sec']) / baseline_exec['run_time_sec']
        },
        'efficiency_comparison': {
            'baseline_trials_ran': baseline_exec['trial_count'],
            'current_trials_ran': current_exec['trial_count'],
            'trial_difference': baseline_exec['trial_count'] - current_exec['trial_count']
        },
        'nutpie_comparison': {
            'baseline_nutpie': baseline.get('nutpie_environment', {}).get('nutpie_available', False),
            'current_nutpie': current_results.get('nutpie_environment', {}).get('nutpie_available', False),
            'baseline_sampler': baseline.get('sampler_selection', {}).get('sampler', 'unknown'),
            'current_sampler': current_results.get('sampler_selection', {}).get('sampler', 'unknown')
        }
    }

    return comparison


def verify_routing(log_path: Path, expected_type: str) -> Dict[str, Any]:
    """Verify correct inference routing from log file."""
    with open(log_path, 'r') as f:
        log_content = f.read()

    routing_patterns = {
        'BINARY': ["Processing grouping", "as BINARY", "Binary group inference took"],
        'ORDINAL': ["Processing grouping", "as ORDINAL", "Ordinal group inference took"],
        'CONTINUOUS': ["CONTINUOUS_BOUNDED", "Processing grouping", "Continuous PyMC MCMC sampling"]
    }

    found_patterns = []
    for pattern in routing_patterns.get(expected_type, []):
        if pattern in log_content:
            found_patterns.append(pattern)

    return {
        'expected_type': expected_type,
        'verified': len(found_patterns) > 0,
        'found_patterns': found_patterns,
        'pattern_count': len(found_patterns)
    }


async def simulate_bridge_protocol(
    dataset_path: Path,
    score_columns: List[str],
    config: Dict[str, Any],
    output_path: Path,
    log_path: Path,
    expected_routing: str,
    baseline_path: Path = None
) -> Dict[str, Any]:
    """
    Simulate the bridge protocol workflow with nutpie tracking.
    """
    # Configure logging
    logger = logging.getLogger(__name__)
    root_logger = logging.getLogger()
    optstop_logger = logging.getLogger('optstop')
    gpu_logger = logging.getLogger('optstop.gpu_utils')

    # Clear handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    for handler in optstop_logger.handlers[:]:
        optstop_logger.removeHandler(handler)

    # Create handlers
    formatter = logging.Formatter('%(message)s')

    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.DEBUG)
    optstop_logger.setLevel(logging.DEBUG)
    gpu_logger.setLevel(logging.INFO)  # Ensure we capture sampler selection logs

    logger.info("=" * 80)
    logger.info(f"BRIDGE PROTOCOL TEST (WITH NUTPIE TRACKING): {dataset_path.name}")
    logger.info("=" * 80)

    # Check nutpie environment BEFORE test
    logger.info("\n🔍 Checking nutpie environment...")
    nutpie_env = check_nutpie_environment()
    logger.info(f"  Nutpie available: {nutpie_env['nutpie_available']}")
    if nutpie_env['nutpie_available']:
        logger.info(f"  Nutpie version: {nutpie_env['nutpie_version']}")
        logger.info(f"  ✅ Expected: ~2-5× CPU speedup")
    else:
        logger.info(f"  ⚠️  Nutpie NOT available - will use PyMC default sampler")
        logger.info(f"  To install: pip install nutpie")

    # Load dataset
    logger.info(f"\n📁 Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  Unique samples: {df['sample_id'].nunique()}")
    logger.info(f"  Score columns: {score_columns}")

    # Create scorer
    scorer = PreloadedDataScorer(df, score_columns)

    # Extract sample metadata
    sample_metadata = df.groupby('sample_id').first()[['model', 'task']].to_dict('index')

    # Calculate totals
    num_samples = df['sample_id'].nunique()
    num_epochs = df['epoch'].max() + 1
    total_planned = num_samples * num_epochs

    logger.info(f"\n📊 Dataset structure:")
    logger.info(f"  Samples: {num_samples}")
    logger.info(f"  Epochs per sample: {num_epochs}")
    logger.info(f"  Total planned trials: {total_planned}")

    # Display grouping statistics
    grouping_cols = config.get('grouping_columns', ['model', 'task'])
    logger.info(f"\n🎯 Grouping statistics:")
    for group_name, group_df in df.groupby(grouping_cols):
        group_samples = group_df['sample_id'].nunique()
        first_score = score_columns[0]
        if len(score_columns) > 1 and 'score_agg' in config:
            mean_score = group_df[score_columns].mean().mean()
        else:
            mean_score = group_df[first_score].mean()
        logger.info(f"  {group_name}: {group_samples} samples, mean score: {mean_score:.3f}")

    # Initialize manager
    logger.info(f"\n⚙️  Initializing OptimalStoppingManager...")
    logger.info(f"  Configuration:")
    logger.info(f"    Grouping: {config.get('grouping_columns')}")
    logger.info(f"    Reanalysis interval: {config.get('reanalysis_interval')}")
    logger.info(f"    MCMC draws: {config['optstop_params']['draws']}")
    logger.info(f"    MCMC tune: {config['optstop_params']['tune']}")
    if 'ordinal_inference' in config:
        logger.info(f"    Ordinal inference mode: {config['ordinal_inference']}")

    start_time = time.time()
    manager = OptimalStoppingManager(**config)
    init_time = time.time() - start_time
    logger.info(f"  Initialization time: {init_time:.2f}s")

    # Create samples
    logger.info(f"\n🔨 Creating Sample objects...")
    samples = []
    for sample_id in df['sample_id'].unique():
        metadata = sample_metadata[sample_id]
        sample = Sample(id=sample_id, metadata=metadata)
        samples.append(sample)
    logger.info(f"  Created {len(samples)} Sample objects")

    # Create eval spec
    eval_spec = EvalSpec(task="bridge_test_task", model="test_model")

    # Start task
    logger.info(f"\n🚀 Starting task...")
    manager_name = await manager.start_task(eval_spec, samples, epochs=num_epochs)
    logger.info(f"  Manager name: {manager_name}")

    # Run simulation
    logger.info(f"\n⏳ Running bridge protocol simulation...")
    logger.info("  (This simulates the inspect_ai → optstop → inspect_ai workflow)")

    trial_count = 0
    stopped_samples = set()

    for sample_id in df['sample_id'].unique():
        for epoch in range(num_epochs):
            if sample_id in stopped_samples:
                break

            early_stop = await manager.schedule_sample(sample_id, epoch)

            if early_stop is not None:
                logger.info(f"    Sample {sample_id} stopped at epoch {epoch}: {early_stop.reason}")
                stopped_samples.add(sample_id)
                break

            scores = scorer.score_sample(sample_id, epoch)
            await manager.complete_sample(sample_id, epoch, scores)
            trial_count += 1

            if trial_count % 500 == 0:
                logger.info(f"    Progress: {trial_count}/{total_planned} trials ({100*trial_count/total_planned:.1f}%)")

    run_time = time.time() - start_time

    logger.info(f"\n✓ Bridge protocol simulation complete")
    logger.info(f"  Total trials run: {trial_count}/{total_planned}")
    logger.info(f"  Efficiency: {100*(total_planned - trial_count)/total_planned:.1f}%")
    logger.info(f"  Stopped samples: {len(stopped_samples)}")
    logger.info(f"  Runtime: {run_time:.2f}s ({run_time/60:.1f} min)")

    # Get diagnostics
    logger.info(f"\n📊 Retrieving diagnostics...")
    diagnostics = await manager.complete_task()

    # Parse additional diagnostics from log
    logger.info(f"\n🔍 Analyzing logs for nutpie usage...")
    sampler_selection = parse_sampler_selection(log_path)
    logger.info(f"  Sampler used: {sampler_selection['sampler']}")
    if sampler_selection['nutpie_used']:
        logger.info(f"  ✅ Nutpie WAS used (version: {sampler_selection['nutpie_version']})")
    else:
        logger.info(f"  ℹ️  Nutpie was NOT used (sampler: {sampler_selection['sampler']})")

    logger.info(f"\n⏱️  Parsing inference timing...")
    inference_timing = parse_inference_timing(log_path)
    for pathway, timing in inference_timing.items():
        logger.info(f"  {pathway.capitalize()}: {timing['count']} calls, mean={timing['mean_sec']:.2f}s, total={timing['total_sec']:.1f}s")

    # Routing verification
    logger.info(f"\n🔍 Verifying inference routing...")
    routing_verification = verify_routing(log_path, expected_routing)
    logger.info(f"  Expected: {routing_verification['expected_type']}")
    logger.info(f"  Verified: {'✓ PASS' if routing_verification['verified'] else '✗ FAIL'}")

    # Compile results
    results = {
        'test_date': datetime.now().isoformat(),
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
        'nutpie_environment': nutpie_env,
        'sampler_selection': sampler_selection,
        'inference_timing': inference_timing
    }

    # Compare with baseline if available
    if baseline_path and baseline_path.exists():
        logger.info(f"\n📊 Comparing with baseline: {baseline_path}")
        comparison = compare_with_baseline(results, baseline_path)
        results['comparison'] = comparison

        if comparison['comparison_available']:
            logger.info(f"  Baseline runtime: {comparison['runtime_comparison']['baseline_sec']:.1f}s")
            logger.info(f"  Current runtime: {comparison['runtime_comparison']['current_sec']:.1f}s")
            logger.info(f"  Speedup: {comparison['runtime_comparison']['speedup']:.2f}×")
            logger.info(f"  Time saved: {comparison['runtime_comparison']['time_saved_sec']:.1f}s ({comparison['runtime_comparison']['time_saved_sec']/60:.1f} min)")
            logger.info(f"  Baseline sampler: {comparison['nutpie_comparison']['baseline_sampler']}")
            logger.info(f"  Current sampler: {comparison['nutpie_comparison']['current_sampler']}")

    # Save results
    logger.info(f"\n💾 Saving diagnostics to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(convert_numpy_types(results), f, indent=2)

    logger.info(f"\n📄 Log saved to: {log_path}")
    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)

    return results


async def main():
    """Run enhanced bridge protocol tests with nutpie tracking."""

    print("\n" + "=" * 80)
    print("LARGE-SCALE BRIDGE PROTOCOL TESTING (WITH NUTPIE TRACKING)")
    print("=" * 80)
    print("\nEnhancements:")
    print("  ✓ Nutpie detection and usage tracking")
    print("  ✓ Per-inference timing capture")
    print("  ✓ Performance comparison with baseline")
    print("  ✓ Ordinal inference mode: hybrid (for dataset 2)")

    # Check nutpie availability upfront
    nutpie_env = check_nutpie_environment()
    print(f"\n🔍 Pre-flight nutpie check:")
    print(f"  Nutpie available: {nutpie_env['nutpie_available']}")
    if nutpie_env['nutpie_available']:
        print(f"  Nutpie version: {nutpie_env['nutpie_version']}")
        print(f"  ✅ Tests will use nutpie for CPU sampling")
    else:
        print(f"  ⚠️  Nutpie NOT available")
        print(f"  Tests will use PyMC default sampler (slower)")
        print(f"  To install nutpie: pip install nutpie")

    # Paths
    data_dir = Path('/home/ubuntu/optstop/test_data/large_scale')
    output_dir = Path('/home/ubuntu/optstop/test_outputs/large_scale')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Backup previous results
    backup_dir = output_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Backing up previous results to: {backup_dir}")
    for file in output_dir.glob('dataset_*.json'):
        shutil.copy(file, backup_dir / file.name)
        print(f"  ✓ Backed up: {file.name}")

    # Base configuration
    optstop_params = {
        'delta_item': 0.15,
        'delta_cap': 0.10,
        'cred_level': 0.95,
        'conservatism': 10,
        'draws': 500,
        'tune': 500,
    }

    base_config = {
        'optstop_params': optstop_params,
        'grouping_columns': ['model', 'task'],
        'reanalysis_interval': 25,
        'min_samples_per_grouping': 10,
    }

    # Dataset 1: Binary Discrete
    print("\n" + "=" * 80)
    print("TEST 1/3: Binary Discrete")
    print("=" * 80)

    dataset1_config = {**base_config}

    results1 = await simulate_bridge_protocol(
        dataset_path=data_dir / 'dataset_1_binary_discrete.csv',
        score_columns=['score'],
        config=dataset1_config,
        output_path=output_dir / 'dataset_1_diagnostics.json',
        log_path=output_dir / 'dataset_1.log',
        expected_routing='BINARY',
        baseline_path=backup_dir / 'dataset_1_diagnostics.json'
    )

    # Dataset 2: Ordinal Discrete with HYBRID mode
    print("\n" + "=" * 80)
    print("TEST 2/3: Ordinal Discrete (HYBRID mode with nutpie)")
    print("=" * 80)

    dataset2_config = {
        **base_config,
        'score_choice': 'accuracy',
        'ordinal_max_score': 10,
        'ordinal_tasks': ['math_easy', 'math_hard', 'coding_medium',
                         'creative_writing', 'reasoning'],
        'ordinal_inference': 'hybrid'  # HYBRID MODE
    }

    results2 = await simulate_bridge_protocol(
        dataset_path=data_dir / 'dataset_2_ordinal_with_choice.csv',
        score_columns=['accuracy', 'precision', 'recall'],
        config=dataset2_config,
        output_path=output_dir / 'dataset_2_diagnostics.json',
        log_path=output_dir / 'dataset_2.log',
        expected_routing='ORDINAL',
        baseline_path=backup_dir / 'dataset_2_diagnostics.json'
    )

    # Dataset 3: Continuous Bounded
    print("\n" + "=" * 80)
    print("TEST 3/3: Continuous Bounded")
    print("=" * 80)

    dataset3_config = {
        **base_config,
        'score_agg': 'mean',
        'ordinal_max_score': 10,
        'ordinal_tasks': ['math_easy', 'math_hard', 'coding_medium',
                         'creative_writing', 'reasoning']
    }

    results3 = await simulate_bridge_protocol(
        dataset_path=data_dir / 'dataset_3_ordinal_with_agg.csv',
        score_columns=['scorer_1', 'scorer_2', 'scorer_3'],
        config=dataset3_config,
        output_path=output_dir / 'dataset_3_diagnostics.json',
        log_path=output_dir / 'dataset_3.log',
        expected_routing='CONTINUOUS',
        baseline_path=backup_dir / 'dataset_3_diagnostics.json'
    )

    # Final summary with nutpie comparison
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE - PERFORMANCE SUMMARY")
    print("=" * 80)

    for i, (results, name) in enumerate([(results1, "Binary"), (results2, "Ordinal Hybrid"), (results3, "Continuous")], 1):
        print(f"\nDataset {i} ({name}):")
        print(f"  Runtime: {results['execution']['run_time_sec']:.1f}s ({results['execution']['run_time_sec']/60:.1f} min)")
        print(f"  Sampler: {results['sampler_selection']['sampler']}")
        print(f"  Nutpie used: {'✅ YES' if results['sampler_selection']['nutpie_used'] else '❌ NO'}")
        print(f"  Trials: {results['execution']['trial_count']}/{results['execution']['total_planned']}")
        print(f"  Efficiency: {100*(results['execution']['total_planned']-results['execution']['trial_count'])/results['execution']['total_planned']:.1f}%")

        if 'comparison' in results and results['comparison']['comparison_available']:
            comp = results['comparison']
            print(f"  📊 Comparison with baseline:")
            print(f"    Baseline runtime: {comp['runtime_comparison']['baseline_sec']/60:.1f} min")
            print(f"    Current runtime: {comp['runtime_comparison']['current_sec']/60:.1f} min")
            print(f"    Speedup: {comp['runtime_comparison']['speedup']:.2f}×")
            print(f"    Time saved: {comp['runtime_comparison']['time_saved_sec']/60:.1f} min ({comp['runtime_comparison']['percent_change']:.1f}%)")
            print(f"    Baseline sampler: {comp['nutpie_comparison']['baseline_sampler']}")

    print(f"\n📁 All diagnostics saved to: {output_dir}")
    print(f"📁 Previous results backed up to: {backup_dir}")
    print("\n✅ Testing complete with nutpie tracking!")


if __name__ == '__main__':
    asyncio.run(main())
