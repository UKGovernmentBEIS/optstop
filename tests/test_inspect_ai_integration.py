"""
Single-task integration test simulating realistic inspect_ai usage.

This test simulates how inspect_ai would actually use the EarlyStopping protocol:
- ONE start_task() call with all samples
- Multiple schedule_sample() and complete_sample() calls
- ONE complete_task() call at the end

This produces accurate complete_task() diagnostics that can be shared with
inspect_ai developers for log viewer integration.

OUTPUT FILES (when --output-dir is specified):
- diagnostics.json      - The exact JSON returned by complete_task()
- optstop_logs_*.log    - Logger output from optstop modules
- console_output_*.log  - Console output (config summary, progress)
- trial_data.csv        - All trial data for analysis
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from unittest.mock import Mock
import numpy as np
import pandas as pd

from optstop.early_stopping import OptimalStoppingManager


# =============================================================================
# Logging Setup
# =============================================================================

class TeeWriter:
    """Write to both a file and the original stream (stdout/stderr)."""

    def __init__(self, file_path: Path, original_stream):
        self.file = open(file_path, 'w')
        self.original = original_stream

    def write(self, text):
        self.original.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.original.flush()
        self.file.flush()

    def close(self):
        self.file.close()


class LoggingContext:
    """Context manager for logging setup and teardown."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else None
        self.file_handler: Optional[logging.FileHandler] = None
        self.tee_writer: Optional[TeeWriter] = None
        self.original_stdout = None
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def setup(self) -> 'LoggingContext':
        """Configure logging to capture output as an inspect_ai user would see it."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        logging.basicConfig(
            level=logging.DEBUG,
            format=log_format,
            handlers=[logging.StreamHandler(sys.stdout)]
        )

        # Set levels matching what inspect_ai users see
        logging.getLogger('optstop').setLevel(logging.INFO)
        logging.getLogger('optstop.early_stopping').setLevel(logging.INFO)
        logging.getLogger('optstop.rule').setLevel(logging.INFO)
        logging.getLogger('optstop.ordinal_model').setLevel(logging.INFO)
        logging.getLogger('optstop.ordinal_utils').setLevel(logging.INFO)
        logging.getLogger('optstop.gpu_utils').setLevel(logging.INFO)
        logging.getLogger('pymc').setLevel(logging.WARNING)
        logging.getLogger('pytensor').setLevel(logging.WARNING)
        logging.getLogger('filelock').setLevel(logging.WARNING)

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # File handler for logger output
            log_file = self.output_dir / f'optstop_logs_{self.timestamp}.log'
            self.file_handler = logging.FileHandler(log_file, mode='w')
            self.file_handler.setLevel(logging.INFO)
            self.file_handler.setFormatter(logging.Formatter(log_format))

            for logger_name in ['optstop', 'optstop.early_stopping', 'optstop.rule',
                               'optstop.ordinal_model', 'optstop.ordinal_utils',
                               'optstop.gpu_utils']:
                logging.getLogger(logger_name).addHandler(self.file_handler)

            # Tee stdout to capture print() output
            console_file = self.output_dir / f'console_output_{self.timestamp}.log'
            self.original_stdout = sys.stdout
            self.tee_writer = TeeWriter(console_file, sys.stdout)
            sys.stdout = self.tee_writer

            print(f"\n📝 Logger output: {log_file}")
            print(f"📝 Console output: {console_file}")

        return self

    def teardown(self):
        """Clean up logging handlers and restore stdout."""
        if self.original_stdout:
            sys.stdout = self.original_stdout
            if self.tee_writer:
                self.tee_writer.close()

        if self.file_handler:
            self.file_handler.close()
            for logger_name in ['optstop', 'optstop.early_stopping', 'optstop.rule',
                               'optstop.ordinal_model', 'optstop.ordinal_utils',
                               'optstop.gpu_utils']:
                logging.getLogger(logger_name).removeHandler(self.file_handler)


logger = logging.getLogger(__name__)


# =============================================================================
# Test Configuration
# =============================================================================

class SingleTaskConfig:
    """Configuration for a single-task evaluation simulating inspect_ai usage.

    This test has THREE GROUPINGS with different score consistency levels:
    - high_consistency: Very consistent scores → should stop early (samples AND grouping)
    - medium_consistency: Moderately consistent → some samples stop, grouping may stop
    - low_consistency: High variance → fewer/no early stops

    This allows demonstration of early stopping at different points in the logs.
    """

    # Single model, but three different subtasks (groupings)
    MODEL = "claude-3-sonnet"

    # Three subtasks with different consistency characteristics
    # NOTE: epoch_variance is kept LOW relative to sample_variance to enable item-level stopping
    # (individual samples show consistent behavior across epochs, but vary between samples)
    SUBTASKS = {
        'arithmetic_basic': {
            'description': 'Basic arithmetic - very consistent within-sample',
            'num_samples': 50,
            'base_rate': 0.90,        # High success rate
            'sample_variance': 0.15,  # Moderate variance BETWEEN samples
            'epoch_variance': 0.01,   # Very low variance WITHIN sample (consistent per-sample)
            'consistency': 'high'
        },
        'algebra_intermediate': {
            'description': 'Intermediate algebra - moderately consistent within-sample',
            'num_samples': 50,
            'base_rate': 0.70,        # Medium success rate
            'sample_variance': 0.20,  # Higher variance BETWEEN samples
            'epoch_variance': 0.02,   # Low variance WITHIN sample
            'consistency': 'medium'
        },
        'calculus_advanced': {
            'description': 'Advanced calculus - somewhat consistent within-sample',
            'num_samples': 50,
            'base_rate': 0.45,        # Lower success rate
            'sample_variance': 0.30,  # High variance BETWEEN samples
            'epoch_variance': 0.05,   # Moderate variance WITHIN sample
            'consistency': 'low'
        }
    }

    # Total: 150 samples, 40 epochs each = 6000 planned trials
    MAX_EPOCHS = 40

    @classmethod
    def total_samples(cls) -> int:
        return sum(s['num_samples'] for s in cls.SUBTASKS.values())

    @classmethod
    def total_planned_trials(cls) -> int:
        return cls.total_samples() * cls.MAX_EPOCHS


class ScoreGenerator:
    """Generate realistic mock scores with configurable consistency per subtask."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._sample_base_rates: dict[str, float] = {}

    def _get_sample_base_rate(self, sample_id: str, subtask: str) -> float:
        """Get consistent base rate for a sample (set once, reused across epochs)."""
        key = f"{subtask}:{sample_id}"
        if key not in self._sample_base_rates:
            config = SingleTaskConfig.SUBTASKS[subtask]
            # Sample's inherent ability varies around subtask base rate
            sample_rate = config['base_rate'] + self.rng.uniform(
                -config['sample_variance'],
                config['sample_variance']
            )
            self._sample_base_rates[key] = np.clip(sample_rate, 0.05, 0.95)
        return self._sample_base_rates[key]

    def generate_score(self, sample_id: str, epoch: int, subtask: str) -> float:
        """Generate binary score (0 or 1) for a sample.

        High consistency subtasks: scores cluster tightly around sample's base rate
        Low consistency subtasks: scores vary widely even for same sample
        """
        config = SingleTaskConfig.SUBTASKS[subtask]
        sample_base = self._get_sample_base_rate(sample_id, subtask)

        # Add epoch-level noise (more noise = less consistent)
        epoch_noise = self.rng.uniform(
            -config['epoch_variance'],
            config['epoch_variance']
        )

        final_rate = np.clip(sample_base + epoch_noise, 0.05, 0.95)
        return 1.0 if self.rng.random() < final_rate else 0.0


def convert_numpy_types(obj: Any) -> Any:
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


def print_section(title: str, char: str = "="):
    """Print a section header."""
    print(f"\n{char*80}")
    print(title)
    print(char*80)


# =============================================================================
# Main Test Function
# =============================================================================

async def run_single_task_test(
    output_dir: Optional[Path] = None,
    random_seed: int = 42,
) -> tuple:
    """
    Run a single-task evaluation simulating realistic inspect_ai usage.

    This test:
    1. Calls start_task() ONCE with all samples (across 3 groupings)
    2. Runs schedule_sample() and complete_sample() for each trial
    3. Calls complete_task() ONCE at the end

    The three groupings have different consistency levels to demonstrate
    early stopping at different points:
    - high_consistency: Should stop early (both samples and grouping)
    - medium_consistency: Some samples stop, grouping may stop
    - low_consistency: Fewer/no early stops due to high variance

    Args:
        output_dir: Optional directory to save output files
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (manager, diagnostics, trial_data_df)
    """
    start_time = time.time()

    print_section("MULTI-GROUPING INTEGRATION TEST")
    print("Simulating realistic inspect_ai EarlyStopping protocol usage")
    print("Testing THREE groupings with different consistency levels")
    print(f"\nRandom seed: {random_seed}")

    config = SingleTaskConfig()
    score_generator = ScoreGenerator(seed=random_seed)

    # =========================================================================
    # Configuration Summary
    # =========================================================================
    print_section("CONFIGURATION", "-")

    print(f"\nEvaluation Setup:")
    print(f"  Model: {config.MODEL}")
    print(f"  Total samples: {config.total_samples()}")
    print(f"  Epochs per sample: {config.MAX_EPOCHS}")
    print(f"  Total planned trials: {config.total_planned_trials()}")

    print("\nGroupings (subtasks) with different consistency levels:")
    for subtask_name, subtask_config in config.SUBTASKS.items():
        print(f"\n  {subtask_name} ({subtask_config['consistency']} consistency):")
        print(f"    Description: {subtask_config['description']}")
        print(f"    Samples: {subtask_config['num_samples']}")
        print(f"    Base success rate: {subtask_config['base_rate']}")
        print(f"    Sample variance: ±{subtask_config['sample_variance']}")
        print(f"    Epoch variance: ±{subtask_config['epoch_variance']}")

    # =========================================================================
    # Initialize OptimalStoppingManager
    # =========================================================================
    print_section("INITIALIZING OPTIMAL STOPPING MANAGER", "-")

    optstop_params = {
        'delta_item': 0.15,       # CI width threshold for individual samples (relaxed to trigger more easily)
        'delta_cap': 0.02,        # CI width threshold for grouping (very strict - delays grouping stops)
        'cred_level': 0.95,       # 95% credible intervals
        'conservatism': 2.0,      # Conservatism factor for low performance
        'low_performance_threshold': 0.2,
        'CI_delta': 0.000001,     # Very strict slope threshold (delays slope-based stopping)
        'tune': 500,              # MCMC tuning
        'draws': 500,             # MCMC samples
        'chains': 2,
        'cores': 1,
    }

    print("\nOptimal Stopping Parameters:")
    for param, value in optstop_params.items():
        print(f"  {param}: {value}")

    # Initialize manager
    # Grouping by model and subtask - creates 3 groupings (one per subtask)
    # Note: 'metadata.subtask' references the 'subtask' key in sample.metadata
    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'metadata.subtask'],
        reanalysis_interval=5,      # Run inference every 5 completed samples
        min_samples_per_grouping=5, # Minimum samples before first inference
        ordinal_tasks=None,         # Binary scoring only
        gpu_ids=None,               # CPU-only
        manager_name='inspect_ai_integration_test',
        shadow_mode=False,
        random_seed=random_seed,
    )

    print(f"\nManager initialized: {manager.manager_name}")
    print(f"Grouping columns: {manager.grouping_columns}")

    # =========================================================================
    # Create Mock EvalSpec and Samples (simulating inspect_ai)
    # =========================================================================
    print_section("PREPARING EVALUATION", "-")

    # Create samples with subtask assignments in metadata
    # This creates the three groupings based on subtask
    mock_samples = []
    sample_subtask_map = {}  # Track which subtask each sample belongs to

    sample_counter = 0
    for subtask_name, subtask_config in config.SUBTASKS.items():
        for i in range(subtask_config['num_samples']):
            sample_id = f"{subtask_name}_{i:03d}"
            sample_subtask_map[sample_id] = subtask_name

            sample = Mock()
            sample.id = sample_id
            sample.metadata = {
                "subtask": subtask_name,  # This becomes the grouping key
                "consistency": subtask_config['consistency'],
                "base_rate": subtask_config['base_rate']
            }
            mock_samples.append(sample)
            sample_counter += 1

    # Create mock EvalSpec (what inspect_ai passes to start_task)
    mock_eval_spec = Mock()
    mock_eval_spec.model = config.MODEL
    mock_eval_spec.task = "math_benchmark"
    mock_eval_spec.eval_id = f"eval_math_benchmark_{random_seed}"
    mock_eval_spec.metadata = {"test_type": "multi_grouping_integration"}
    mock_eval_spec.tags = ["early_stopping_test", "multi_grouping"]

    print(f"Created {len(mock_samples)} samples across {len(config.SUBTASKS)} groupings:")
    for subtask_name, subtask_config in config.SUBTASKS.items():
        print(f"  {subtask_name}: {subtask_config['num_samples']} samples")

    # =========================================================================
    # START_TASK - Single call as inspect_ai would do
    # =========================================================================
    print_section("PROTOCOL: start_task()", "-")

    manager_name = await manager.start_task(
        task=mock_eval_spec,
        samples=mock_samples,
        epochs=config.MAX_EPOCHS
    )

    print(f"Manager name returned: {manager_name}")

    # =========================================================================
    # SCHEDULE_SAMPLE + COMPLETE_SAMPLE - For each trial
    # =========================================================================
    print_section("PROTOCOL: schedule_sample() + complete_sample()", "-")

    total_scheduled = 0
    total_stopped = 0
    total_completed = 0
    trial_data = []

    # Track per-grouping statistics
    grouping_stats = {subtask: {'scheduled': 0, 'stopped': 0, 'completed': 0}
                      for subtask in config.SUBTASKS.keys()}

    # Track progress
    last_progress_time = time.time()
    progress_interval = 30  # Print progress every 30 seconds

    # Get all sample IDs in order
    all_sample_ids = [s.id for s in mock_samples]

    # Iterate through epochs (outer) then samples (inner)
    # This matches typical inspect_ai execution order
    for epoch in range(1, config.MAX_EPOCHS + 1):
        epoch_scheduled = 0
        epoch_stopped = 0

        for sample_id in all_sample_ids:
            subtask = sample_subtask_map[sample_id]

            # Check if sample should be scheduled
            early_stop = await manager.schedule_sample(id=sample_id, epoch=epoch)

            if early_stop is not None:
                # Sample was stopped early
                total_stopped += 1
                epoch_stopped += 1
                grouping_stats[subtask]['stopped'] += 1
                trial_data.append({
                    'sample_id': sample_id,
                    'subtask': subtask,
                    'epoch': epoch,
                    'score': None,
                    'scheduled': False,
                    'stopped': True,
                    'stop_reason': early_stop.reason
                })
                continue

            # Sample is scheduled - generate and submit score
            total_scheduled += 1
            epoch_scheduled += 1
            grouping_stats[subtask]['scheduled'] += 1

            score = score_generator.generate_score(sample_id, epoch, subtask)

            # Create mock score object (as inspect_ai would provide)
            mock_score_obj = Mock()
            mock_score_obj.value = score
            mock_sample_score = Mock()
            mock_sample_score.score = mock_score_obj
            mock_scores = {'accuracy': mock_sample_score}

            # Complete the sample
            await manager.complete_sample(
                id=sample_id,
                epoch=epoch,
                scores=mock_scores
            )
            total_completed += 1
            grouping_stats[subtask]['completed'] += 1

            trial_data.append({
                'sample_id': sample_id,
                'subtask': subtask,
                'epoch': epoch,
                'score': score,
                'scheduled': True,
                'stopped': False,
                'stop_reason': None
            })

        # Progress logging
        current_time = time.time()
        if current_time - last_progress_time >= progress_interval:
            elapsed = current_time - start_time
            print(f"  Progress: Epoch {epoch}/{config.MAX_EPOCHS}, "
                  f"Scheduled: {total_scheduled}, Stopped: {total_stopped}, "
                  f"Elapsed: {elapsed:.1f}s")
            last_progress_time = current_time

        # Summary for this epoch
        print(f"  Epoch {epoch}: scheduled={epoch_scheduled}, stopped={epoch_stopped}")

    print("\nTrial Summary:")
    print(f"  Total scheduled: {total_scheduled}")
    print(f"  Total stopped: {total_stopped}")
    print(f"  Total completed: {total_completed}")

    print("\nPer-Grouping Summary:")
    for subtask, stats in grouping_stats.items():
        consistency = config.SUBTASKS[subtask]['consistency']
        print(f"  {subtask} ({consistency}):")
        print(f"    Scheduled: {stats['scheduled']}, Stopped: {stats['stopped']}")

    # =========================================================================
    # COMPLETE_TASK - Single call to get diagnostics
    # =========================================================================
    print_section("PROTOCOL: complete_task()", "-")

    diagnostics = await manager.complete_task()

    print("\nDiagnostics returned by complete_task():")
    print(f"  Manager: {diagnostics.get('manager')}")
    print(f"  Random seed: {diagnostics.get('random_seed')}")
    print(f"  Total planned: {diagnostics.get('total_planned_trials')}")
    print(f"  Total ran: {diagnostics.get('total_ran')}")
    print(f"  Total skipped: {diagnostics.get('total_skipped')}")
    print(f"  Efficiency: {diagnostics.get('efficiency_percent')}%")
    print(f"  Stopped samples count: {diagnostics.get('stopped_samples_count')}")
    print(f"  Stopped groupings: {diagnostics.get('stopped_groupings')}")

    # Show stopped samples per grouping
    stopped_per_grouping = diagnostics.get('stopped_samples_per_grouping', {})
    if stopped_per_grouping:
        print("\n  Stopped samples per grouping:")
        for grouping, count in stopped_per_grouping.items():
            print(f"    {grouping}: {count} samples")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print_section("FINAL SUMMARY")

    total_time = time.time() - start_time
    trial_df = pd.DataFrame(trial_data)

    print(f"\nExecution Time: {total_time:.1f}s")
    print(f"\nTrial Statistics:")
    print(f"  Planned: {config.total_planned_trials()}")
    print(f"  Ran: {total_completed}")
    print(f"  Stopped: {total_stopped}")

    # Per-grouping breakdown
    print("\nPer-Grouping Breakdown:")
    for subtask in config.SUBTASKS.keys():
        subtask_data = trial_df[trial_df['subtask'] == subtask]
        scheduled = subtask_data[subtask_data['scheduled']]
        stopped = subtask_data[subtask_data['stopped']]
        consistency = config.SUBTASKS[subtask]['consistency']

        print(f"\n  {subtask} ({consistency} consistency):")
        print(f"    Trials ran: {len(scheduled)}")
        print(f"    Trials stopped: {len(stopped)}")
        if len(scheduled) > 0:
            mean_score = scheduled['score'].mean()
            print(f"    Mean score: {mean_score:.3f}")

    if total_completed > 0:
        completed_scores = trial_df[trial_df['scheduled']]['score']
        print(f"\nOverall Score Statistics (completed trials):")
        print(f"  Mean: {completed_scores.mean():.3f}")
        print(f"  Std: {completed_scores.std():.3f}")

    # =========================================================================
    # Save Outputs
    # =========================================================================
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the exact diagnostics JSON (most important for inspect_ai)
        diagnostics_path = output_dir / 'diagnostics.json'
        with open(diagnostics_path, 'w') as f:
            json.dump(convert_numpy_types(diagnostics), f, indent=2)
        print(f"\n✓ Diagnostics saved to: {diagnostics_path}")

        # Save trial data
        trial_data_path = output_dir / 'trial_data.csv'
        trial_df.to_csv(trial_data_path, index=False)
        print(f"✓ Trial data saved to: {trial_data_path}")

        # Print output files summary
        print("\n" + "-"*40)
        print("OUTPUT FILES FOR INSPECT_AI DEVELOPER")
        print("-"*40)
        print("  diagnostics.json      - Exact output from complete_task()")
        print("  optstop_logs_*.log    - Logger output during evaluation")
        print("  console_output_*.log  - Console output (config summary)")
        print("  trial_data.csv        - All trial data")
        print("-"*40)

    print_section("TEST COMPLETE")
    print(f"Total execution time: {total_time:.1f}s")

    return manager, diagnostics, trial_df


# =============================================================================
# Entry Points
# =============================================================================

def test_inspect_ai_integration():
    """Pytest entry point."""
    manager, diagnostics, trial_df = asyncio.run(run_single_task_test())

    # Basic assertions
    assert manager is not None
    assert diagnostics is not None
    assert len(trial_df) > 0

    # Verify diagnostics structure
    assert 'manager' in diagnostics
    assert 'random_seed' in diagnostics
    assert 'total_planned_trials' in diagnostics
    assert 'total_ran' in diagnostics
    assert 'total_skipped' in diagnostics
    assert 'efficiency_percent' in diagnostics
    assert 'stopped_samples' in diagnostics
    assert 'stabilization_histories' in diagnostics

    print("\n✓ All assertions passed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Single-task integration test simulating inspect_ai usage'
    )
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save output files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()

    # Set up logging context
    logging_ctx = LoggingContext(args.output_dir)
    logging_ctx.setup()

    try:
        manager, diagnostics, trial_df = asyncio.run(
            run_single_task_test(
                output_dir=args.output_dir,
                random_seed=args.seed,
            )
        )
    finally:
        logging_ctx.teardown()
