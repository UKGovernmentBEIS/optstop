"""
Comprehensive logging test script for verifying logging cleanup changes.

This script captures THREE distinct output streams:
1. logger.info() messages - Standard logging output
2. trace_message() calls - inspect_ai integration traces (mocked for local testing)
3. JSON diagnostics - Complete output that users would receive

FULL-SCALE TEST: Uses all 500 samples per dataset (15,000 total trials).

Expected runtimes (with ordinal_inference='modal'):
- Dataset 1 (Binary): ~8-10 min
- Dataset 2 (Ordinal modal): ~8-10 min (modal is fast, no entropy-based inference)
- Dataset 3 (Continuous): ~8-10 min
"""

import pandas as pd
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from io import StringIO
import sys

import numpy as np


# ============================================================================
# TRACE MESSAGE MOCK - Captures trace_message() calls for verification
# ============================================================================

class TraceMessageCollector:
    """
    Mock collector for trace_message() calls.

    In production, trace_message() sends messages to inspect_ai's trace system.
    This mock captures all calls so we can verify what would be traced.
    """

    def __init__(self):
        self.traces: List[Dict[str, Any]] = []
        self._original_trace_message = None

    def mock_trace_message(self, logger, component: str, message: str):
        """Mock trace_message that captures calls and also logs them."""
        timestamp = datetime.now().isoformat()
        self.traces.append({
            'timestamp': timestamp,
            'component': component,
            'message': message,
            'logger_name': logger.name if hasattr(logger, 'name') else str(logger)
        })
        # Also log it so it appears in standard output
        logger.info(f"[TRACE:{component}] {message}")

    def install(self):
        """Install the mock trace_message in all relevant modules."""
        import optstop.early_stopping as es_module
        import optstop.ordinal_utils as ou_module

        # Save originals
        self._es_original = getattr(es_module, 'trace_message', None)
        self._ou_original = getattr(ou_module, 'trace_message', None)

        # Install mocks
        es_module.trace_message = self.mock_trace_message
        ou_module.trace_message = self.mock_trace_message

    def uninstall(self):
        """Restore original trace_message functions."""
        import optstop.early_stopping as es_module
        import optstop.ordinal_utils as ou_module

        if self._es_original:
            es_module.trace_message = self._es_original
        if self._ou_original:
            ou_module.trace_message = self._ou_original

    def get_traces(self) -> List[Dict[str, Any]]:
        """Return all captured traces."""
        return self.traces.copy()

    def clear(self):
        """Clear captured traces."""
        self.traces = []

    def get_summary(self) -> Dict[str, Any]:
        """Return summary of captured traces."""
        by_component = {}
        for trace in self.traces:
            comp = trace['component']
            if comp not in by_component:
                by_component[comp] = []
            by_component[comp].append(trace['message'])

        return {
            'total_traces': len(self.traces),
            'by_component': {k: len(v) for k, v in by_component.items()},
            'components': list(by_component.keys()),
            'all_traces': self.traces
        }


# ============================================================================
# LOGGING CAPTURE - Captures logger.info() and other log messages
# ============================================================================

class LogCapture:
    """
    Captures log messages at different levels for verification.
    """

    def __init__(self):
        self.logs: Dict[str, List[str]] = {
            'DEBUG': [],
            'INFO': [],
            'WARNING': [],
            'ERROR': [],
            'CRITICAL': []
        }
        self.handler = None

    def install(self, logger_name: Optional[str] = None):
        """Install log capture handler."""

        class CaptureHandler(logging.Handler):
            def __init__(self, capture):
                super().__init__()
                self.capture = capture

            def emit(self, record):
                level = record.levelname
                msg = self.format(record)
                if level in self.capture.logs:
                    self.capture.logs[level].append(msg)

        self.handler = CaptureHandler(self)
        self.handler.setLevel(logging.DEBUG)
        self.handler.setFormatter(logging.Formatter('%(name)s - %(message)s'))

        if logger_name:
            logger = logging.getLogger(logger_name)
        else:
            logger = logging.getLogger()

        logger.addHandler(self.handler)

    def uninstall(self, logger_name: Optional[str] = None):
        """Remove log capture handler."""
        if self.handler:
            if logger_name:
                logger = logging.getLogger(logger_name)
            else:
                logger = logging.getLogger()
            logger.removeHandler(self.handler)

    def get_all_logs(self) -> Dict[str, List[str]]:
        """Return all captured logs by level."""
        return self.logs.copy()

    def get_summary(self) -> Dict[str, int]:
        """Return count of logs by level."""
        return {level: len(msgs) for level, msgs in self.logs.items()}

    def clear(self):
        """Clear captured logs."""
        for level in self.logs:
            self.logs[level] = []


# ============================================================================
# TEST UTILITIES
# ============================================================================

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
        self.score_index = {}
        for _, row in dataset_df.iterrows():
            key = (row['sample_id'], row['epoch'])
            score_dict = {col: row[col] for col in score_columns}
            self.score_index[key] = score_dict

    def score_sample(self, sample_id: str, epoch: int) -> Dict[str, Any]:
        raw_scores = self.score_index.get((sample_id, epoch), {})
        return {col: create_mock_score(raw_scores[col]) for col in raw_scores}


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

async def run_logging_test(
    dataset_path: Path,
    score_columns: List[str],
    config: Dict[str, Any],
    expected_routing: str,
    max_samples: Optional[int] = None,  # None = use all samples
    trace_collector: Optional[TraceMessageCollector] = None,
    log_capture: Optional[LogCapture] = None
) -> Dict[str, Any]:
    """
    Run a single dataset test capturing all output streams.

    Args:
        dataset_path: Path to CSV dataset
        score_columns: List of score column names
        config: OptimalStoppingManager configuration
        expected_routing: Expected inference type
        max_samples: Maximum samples to use (None = use all samples)
        trace_collector: TraceMessageCollector instance
        log_capture: LogCapture instance

    Returns:
        Dictionary with all captured outputs
    """
    from optstop.early_stopping import OptimalStoppingManager, Sample, EvalSpec

    logger = logging.getLogger('test_logging')

    # Load dataset
    logger.info(f"Loading dataset: {dataset_path.name}")
    df = pd.read_csv(dataset_path)

    # Optionally subset to max_samples for faster testing
    if max_samples is not None:
        unique_samples = df['sample_id'].unique()[:max_samples]
        df = df[df['sample_id'].isin(unique_samples)]
        logger.info(f"Using {len(unique_samples)} samples (subset for testing)")
    else:
        unique_samples = df['sample_id'].unique()
        logger.info(f"Using all {len(unique_samples)} samples (full-scale test)")

    # Create scorer
    scorer = PreloadedDataScorer(df, score_columns)

    # Extract sample metadata
    sample_metadata = df.groupby('sample_id').first()[['model', 'task']].to_dict('index')

    # Calculate totals
    num_samples = df['sample_id'].nunique()
    num_epochs = df['epoch'].max()
    total_planned = num_samples * num_epochs

    logger.info(f"Samples: {num_samples}, Epochs: {num_epochs}, Total planned: {total_planned}")

    # Initialize manager
    start_time = time.time()
    manager = OptimalStoppingManager(**config)
    init_time = time.time() - start_time

    # Create Sample objects
    samples = []
    for sample_id in df['sample_id'].unique():
        metadata = sample_metadata[sample_id]
        sample = Sample(id=sample_id, metadata=metadata)
        samples.append(sample)

    # Create EvalSpec
    eval_spec = EvalSpec(task="logging_test_task", model="test_model")

    # Start task
    manager_name = await manager.start_task(eval_spec, samples, epochs=num_epochs)
    logger.info(f"Manager started: {manager_name}")

    # Run bridge protocol
    trial_count = 0
    stopped_samples = set()
    last_progress_time = time.time()
    progress_interval = 60  # Log progress every 60 seconds

    for sample_id in df['sample_id'].unique():
        for epoch in range(1, num_epochs + 1):
            if sample_id in stopped_samples:
                break

            early_stop = await manager.schedule_sample(sample_id, epoch)

            if early_stop is not None:
                stopped_samples.add(sample_id)
                break

            scores = scorer.score_sample(sample_id, epoch)
            await manager.complete_sample(sample_id, epoch, scores)
            trial_count += 1

            # Progress logging every minute
            current_time = time.time()
            if current_time - last_progress_time >= progress_interval:
                elapsed = current_time - start_time
                pct_complete = 100 * trial_count / total_planned
                logger.info(f"Progress: {trial_count}/{total_planned} trials ({pct_complete:.1f}%) - {elapsed/60:.1f} min elapsed")
                last_progress_time = current_time

    run_time = time.time() - start_time

    # Get diagnostics (the JSON output users would see)
    diagnostics = await manager.complete_task()

    # Compile results
    results = {
        'dataset': dataset_path.name,
        'expected_routing': expected_routing,
        'execution': {
            'init_time_sec': init_time,
            'run_time_sec': run_time,
            'trial_count': trial_count,
            'total_planned': total_planned,
            'efficiency_pct': 100 * (total_planned - trial_count) / total_planned if total_planned > 0 else 0,
            'stopped_samples': len(stopped_samples)
        },
        'diagnostics': convert_numpy_types(diagnostics)
    }

    return results


async def main():
    """Run comprehensive logging tests for all three datasets."""

    print("\n" + "=" * 80)
    print("COMPREHENSIVE LOGGING OUTPUT TEST - FULL SCALE")
    print("=" * 80)
    print("\nThis test captures THREE output streams:")
    print("  1. logger.info() messages - Standard logging")
    print("  2. trace_message() calls - inspect_ai traces (mocked)")
    print("  3. JSON diagnostics - User-facing output")
    print("\nFULL-SCALE TEST: 500 samples × 10 epochs = 5,000 trials per dataset")
    print("\nExpected runtimes (with ordinal_inference='modal'):")
    print("  - Dataset 1 (Binary): ~8-10 min")
    print("  - Dataset 2 (Ordinal modal): ~8-10 min")
    print("  - Dataset 3 (Continuous): ~8-10 min")
    print("=" * 80)

    # Paths
    data_dir = Path('/home/ubuntu/optstop/test_data/large_scale')
    output_dir = Path('/home/ubuntu/optstop/test_outputs/logging_test')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Initialize collectors
    trace_collector = TraceMessageCollector()
    log_capture = LogCapture()

    # Install mocks
    trace_collector.install()
    log_capture.install('optstop')

    # Configure logging to also go to file
    log_file = output_dir / f'full_test_{timestamp}.log'
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.DEBUG)

    # Also set optstop loggers to DEBUG
    logging.getLogger('optstop').setLevel(logging.DEBUG)

    # Console handler for progress
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(console_handler)

    test_logger = logging.getLogger('test_logging')

    # Base configuration
    optstop_params = {
        'delta_item': 0.15,
        'delta_cap': 0.10,
        'cred_level': 0.95,
        'conservatism': 10,
        'draws': 500,
        'tune': 500,
        'entropy_convergence_threshold': 0.05,  # Stricter than default 0.10 (absolute entropy CI width on [0,1] scale)
    }

    base_config = {
        'optstop_params': optstop_params,
        'grouping_columns': ['model', 'task'],
        'reanalysis_interval': 25,
        'min_samples_per_grouping': 10,
        'random_seed': 42,
    }

    all_results = {}

    # =========================================================================
    # Dataset 1: Binary Discrete
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 1/3: Binary Discrete")
    print("-" * 40)

    # Reset GPU logging flags so each dataset logs its GPU status once
    from optstop.gpu_utils import reset_logging_flags
    reset_logging_flags()

    trace_collector.clear()
    log_capture.clear()

    dataset1_config = {**base_config}

    results1 = await run_logging_test(
        dataset_path=data_dir / 'dataset_1_binary_discrete.csv',
        score_columns=['score'],
        config=dataset1_config,
        expected_routing='BINARY',
        trace_collector=trace_collector,
        log_capture=log_capture
    )

    results1['traces'] = trace_collector.get_summary()
    results1['log_summary'] = log_capture.get_summary()
    results1['log_info_messages'] = log_capture.logs['INFO'].copy()
    all_results['dataset_1_binary'] = results1

    print(f"  ✓ Completed: {results1['execution']['trial_count']}/{results1['execution']['total_planned']} trials")
    print(f"  ✓ Efficiency: {results1['execution']['efficiency_pct']:.1f}%")
    print(f"  ✓ Traces captured: {results1['traces']['total_traces']}")
    print(f"  ✓ Log messages: {sum(results1['log_summary'].values())}")

    # =========================================================================
    # Dataset 2: Ordinal Discrete
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 2/3: Ordinal Discrete")
    print("-" * 40)

    # Reset GPU logging flags for this dataset
    reset_logging_flags()

    trace_collector.clear()
    log_capture.clear()

    dataset2_config = {
        **base_config,
        'score_choice': 'accuracy',
        'ordinal_max_score': 10,
        'ordinal_tasks': ['math_easy', 'math_hard', 'coding_medium',
                         'creative_writing', 'reasoning'],
        'ordinal_inference': 'modal'  # Use modal (fast) instead of hybrid for this test
    }

    results2 = await run_logging_test(
        dataset_path=data_dir / 'dataset_2_ordinal_with_choice.csv',
        score_columns=['accuracy', 'precision', 'recall'],
        config=dataset2_config,
        expected_routing='ORDINAL',
        trace_collector=trace_collector,
        log_capture=log_capture
    )

    results2['traces'] = trace_collector.get_summary()
    results2['log_summary'] = log_capture.get_summary()
    results2['log_info_messages'] = log_capture.logs['INFO'].copy()
    all_results['dataset_2_ordinal'] = results2

    print(f"  ✓ Completed: {results2['execution']['trial_count']}/{results2['execution']['total_planned']} trials")
    print(f"  ✓ Efficiency: {results2['execution']['efficiency_pct']:.1f}%")
    print(f"  ✓ Traces captured: {results2['traces']['total_traces']}")
    print(f"  ✓ Log messages: {sum(results2['log_summary'].values())}")

    # =========================================================================
    # Dataset 3: Continuous Bounded
    # =========================================================================
    print("\n" + "-" * 40)
    print("TEST 3/3: Continuous Bounded")
    print("-" * 40)

    # Reset GPU logging flags for this dataset
    reset_logging_flags()

    trace_collector.clear()
    log_capture.clear()

    dataset3_config = {
        **base_config,
        'score_agg': 'mean',
        'ordinal_max_score': 10,
        'ordinal_tasks': ['math_easy', 'math_hard', 'coding_medium',
                         'creative_writing', 'reasoning']
    }

    results3 = await run_logging_test(
        dataset_path=data_dir / 'dataset_3_ordinal_with_agg.csv',
        score_columns=['scorer_1', 'scorer_2', 'scorer_3'],
        config=dataset3_config,
        expected_routing='CONTINUOUS',
        trace_collector=trace_collector,
        log_capture=log_capture
    )

    results3['traces'] = trace_collector.get_summary()
    results3['log_summary'] = log_capture.get_summary()
    results3['log_info_messages'] = log_capture.logs['INFO'].copy()
    all_results['dataset_3_continuous'] = results3

    print(f"  ✓ Completed: {results3['execution']['trial_count']}/{results3['execution']['total_planned']} trials")
    print(f"  ✓ Efficiency: {results3['execution']['efficiency_pct']:.1f}%")
    print(f"  ✓ Traces captured: {results3['traces']['total_traces']}")
    print(f"  ✓ Log messages: {sum(results3['log_summary'].values())}")

    # =========================================================================
    # Cleanup and save results
    # =========================================================================
    trace_collector.uninstall()
    log_capture.uninstall('optstop')

    # Save comprehensive results
    output_file = output_dir / f'logging_test_results_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # =========================================================================
    # Print summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    print("\n📊 EXECUTION SUMMARY:")
    print("-" * 40)
    for name, result in all_results.items():
        print(f"\n{name}:")
        print(f"  Trials: {result['execution']['trial_count']}/{result['execution']['total_planned']}")
        print(f"  Efficiency: {result['execution']['efficiency_pct']:.1f}%")
        print(f"  Runtime: {result['execution']['run_time_sec']:.1f}s")

    print("\n📝 LOGGING SUMMARY (logger.info messages):")
    print("-" * 40)
    for name, result in all_results.items():
        print(f"\n{name}:")
        print(f"  Total log messages by level:")
        for level, count in result['log_summary'].items():
            if count > 0:
                print(f"    {level}: {count}")

    print("\n🔍 TRACE MESSAGE SUMMARY (trace_message calls):")
    print("-" * 40)
    for name, result in all_results.items():
        print(f"\n{name}:")
        print(f"  Total traces: {result['traces']['total_traces']}")
        if result['traces']['by_component']:
            print(f"  By component:")
            for comp, count in result['traces']['by_component'].items():
                print(f"    {comp}: {count}")

    print("\n📁 OUTPUT FILES:")
    print("-" * 40)
    print(f"  Full results: {output_file}")
    print(f"  Full log: {log_file}")

    print("\n" + "=" * 80)
    print("To inspect detailed outputs:")
    print(f"  cat {output_file} | jq '.dataset_1_binary.log_info_messages[:10]'")
    print(f"  cat {output_file} | jq '.dataset_1_binary.traces'")
    print(f"  cat {output_file} | jq '.dataset_1_binary.diagnostics'")
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())
