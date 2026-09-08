"""
Continuous inference integration test simulating realistic inspect_ai usage with ordinal scores.

This test is a variant of test_inspect_ai_integration.py that tests the CONTINUOUS INFERENCE
pathway instead of binary inference. This is triggered when:
- User has ordinal scores (e.g., 0-10 scale)
- User specifies score_agg='mean' to aggregate multiple scores

The key differences from the binary test:
1. ordinal_tasks is set to match subtask names (enables ordinal scoring)
2. score_agg='mean' is set (triggers continuous inference via aggregation)
3. Scores are generated in ordinal range [0, ordinal_max_score]
4. Multiple scores per sample are aggregated to produce continuous [0, ordinal_max_score] values

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

class ContinuousTaskConfig:
    """Configuration for continuous inference test with ordinal scores.

    This test has THREE GROUPINGS with different score consistency levels.
    Each generates ORDINAL scores (0-10 scale) with MULTIPLE scores per sample
    that are aggregated using mean.

    The aggregation of ordinal scores produces continuous values in [0, ordinal_max_score],
    which triggers the continuous bounded inference pathway.
    """

    # Single model, but three different subtasks (groupings)
    MODEL = "claude-3-sonnet"

    # Ordinal scoring configuration
    ORDINAL_MAX_SCORE = 10  # Scores range from 0 to 10
    NUM_SCORES_PER_SAMPLE = 5  # Number of ordinal scores to aggregate per sample

    # Three subtasks with different consistency characteristics
    # NOTE: epoch_variance is kept LOW relative to sample_variance to enable item-level stopping
    SUBTASKS = {
        'rubric_basic': {
            'description': 'Basic rubric scoring - very consistent within-sample',
            'num_samples': 50,
            'base_score': 8.0,        # High base score (out of 10)
            'sample_variance': 1.5,   # Moderate variance BETWEEN samples
            'epoch_variance': 0.1,    # Very low variance WITHIN sample (consistent per-sample)
            'score_variance': 0.5,    # Variance between individual scores in aggregation
            'consistency': 'high'
        },
        'rubric_intermediate': {
            'description': 'Intermediate rubric scoring - moderately consistent within-sample',
            'num_samples': 50,
            'base_score': 6.0,        # Medium base score
            'sample_variance': 2.0,   # Higher variance BETWEEN samples
            'epoch_variance': 0.2,    # Low variance WITHIN sample
            'score_variance': 1.0,    # Moderate variance between individual scores
            'consistency': 'medium'
        },
        'rubric_advanced': {
            'description': 'Advanced rubric scoring - somewhat consistent within-sample',
            'num_samples': 50,
            'base_score': 4.5,        # Lower base score
            'sample_variance': 3.0,   # High variance BETWEEN samples
            'epoch_variance': 0.5,    # Moderate variance WITHIN sample
            'score_variance': 1.5,    # Higher variance between individual scores
            'consistency': 'low'
        }
    }

    # Total: 150 samples, 10 epochs each = 1500 planned trials
    MAX_EPOCHS = 10

    @classmethod
    def total_samples(cls) -> int:
        return sum(s['num_samples'] for s in cls.SUBTASKS.values())

    @classmethod
    def total_planned_trials(cls) -> int:
        return cls.total_samples() * cls.MAX_EPOCHS


class OrdinalScoreGenerator:
    """Generate realistic ordinal scores with configurable consistency per subtask.

    This generates MULTIPLE ordinal scores per sample (simulating rubric subscores)
    that will be aggregated to produce continuous values.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._sample_base_scores: dict[str, float] = {}
        self.ordinal_max = ContinuousTaskConfig.ORDINAL_MAX_SCORE
        self.num_scores = ContinuousTaskConfig.NUM_SCORES_PER_SAMPLE

    def _get_sample_base_score(self, sample_id: str, subtask: str) -> float:
        """Get consistent base score for a sample (set once, reused across epochs)."""
        key = f"{subtask}:{sample_id}"
        if key not in self._sample_base_scores:
            config = ContinuousTaskConfig.SUBTASKS[subtask]
            # Sample's inherent ability varies around subtask base score
            sample_score = config['base_score'] + self.rng.uniform(
                -config['sample_variance'],
                config['sample_variance']
            )
            self._sample_base_scores[key] = np.clip(sample_score, 0.5, self.ordinal_max - 0.5)
        return self._sample_base_scores[key]

    def generate_scores(self, sample_id: str, epoch: int, subtask: str) -> dict[str, float]:
        """Generate multiple ordinal scores for a sample, to be aggregated.

        Returns a dictionary of score names -> ordinal values (integers 0-10).
        These simulate rubric subscores that would be aggregated to get a mean score.

        High consistency subtasks: scores cluster tightly around sample's base score
        Low consistency subtasks: scores vary widely even for same sample
        """
        config = ContinuousTaskConfig.SUBTASKS[subtask]
        sample_base = self._get_sample_base_score(sample_id, subtask)

        # Add epoch-level noise (more noise = less consistent across epochs)
        epoch_noise = self.rng.uniform(
            -config['epoch_variance'],
            config['epoch_variance']
        )
        epoch_adjusted_base = sample_base + epoch_noise

        # Generate multiple ordinal scores around the epoch-adjusted base
        scores = {}
        for i in range(self.num_scores):
            # Each subscore varies around the base
            score_noise = self.rng.normal(0, config['score_variance'])
            raw_score = epoch_adjusted_base + score_noise
            # Clip to valid ordinal range and round to integer
            ordinal_score = int(np.clip(round(raw_score), 0, self.ordinal_max))
            scores[f'rubric_criterion_{i+1}'] = ordinal_score

        return scores


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

async def run_continuous_inference_test(
    output_dir: Optional[Path] = None,
    random_seed: int = 42,
) -> tuple:
    """
    Run a continuous inference test simulating inspect_ai usage with ordinal scores.

    This test demonstrates the CONTINUOUS BOUNDED inference pathway:
    1. Scores are ordinal (0-10 scale)
    2. Multiple scores per sample are aggregated using mean
    3. The aggregated scores are continuous values in [0, ordinal_max_score]
    4. Continuous bounded normal inference is used instead of binary Beta-Binomial

    The three groupings have different consistency levels to demonstrate
    early stopping at different points.

    Args:
        output_dir: Optional directory to save output files
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (manager, diagnostics, trial_data_df)
    """
    start_time = time.time()

    print_section("CONTINUOUS INFERENCE INTEGRATION TEST")
    print("Testing CONTINUOUS bounded inference pathway")
    print("Using ordinal scores (0-10) with score_agg='mean'")
    print("Testing THREE groupings with different consistency levels")
    print(f"\nRandom seed: {random_seed}")

    config = ContinuousTaskConfig()
    score_generator = OrdinalScoreGenerator(seed=random_seed)

    # =========================================================================
    # Configuration Summary
    # =========================================================================
    print_section("CONFIGURATION", "-")

    print(f"\nEvaluation Setup:")
    print(f"  Model: {config.MODEL}")
    print(f"  Total samples: {config.total_samples()}")
    print(f"  Epochs per sample: {config.MAX_EPOCHS}")
    print(f"  Total planned trials: {config.total_planned_trials()}")

    print(f"\nOrdinal Scoring Configuration:")
    print(f"  Score range: 0 to {config.ORDINAL_MAX_SCORE}")
    print(f"  Scores per sample: {config.NUM_SCORES_PER_SAMPLE}")
    print(f"  Aggregation method: mean")
    print(f"  Inference type: continuous_bounded (Normal model)")

    print("\nGroupings (subtasks) with different consistency levels:")
    for subtask_name, subtask_config in config.SUBTASKS.items():
        print(f"\n  {subtask_name} ({subtask_config['consistency']} consistency):")
        print(f"    Description: {subtask_config['description']}")
        print(f"    Samples: {subtask_config['num_samples']}")
        print(f"    Base score: {subtask_config['base_score']}/{config.ORDINAL_MAX_SCORE}")
        print(f"    Sample variance: +/-{subtask_config['sample_variance']}")
        print(f"    Epoch variance: +/-{subtask_config['epoch_variance']}")
        print(f"    Score variance: +/-{subtask_config['score_variance']}")

    # =========================================================================
    # Initialize OptimalStoppingManager
    # =========================================================================
    print_section("INITIALIZING OPTIMAL STOPPING MANAGER", "-")

    optstop_params = {
        'delta_item': 0.05,       # CI width threshold for individual samples (stricter for continuous)
        'delta_cap': 0.05,        # CI width threshold for grouping (same as item-level)
        'cred_level': 0.95,       # 95% credible intervals
        'conservatism': 2.0,      # Conservatism factor for low performance
        'low_performance_threshold': 0.2,
        'CI_delta': 0.0005,       # Relaxed slope threshold for faster grouping-level stopping
        'tune': 500,              # MCMC tuning
        'draws': 500,             # MCMC samples
        'chains': 2,
        'cores': 1,
    }

    print("\nOptimal Stopping Parameters:")
    for param, value in optstop_params.items():
        print(f"  {param}: {value}")

    # Get list of subtask names for ordinal_tasks
    # This tells the manager which tasks use ordinal scoring
    ordinal_task_names = list(config.SUBTASKS.keys())

    print(f"\nOrdinal Tasks Configuration:")
    print(f"  ordinal_tasks: {ordinal_task_names}")
    print(f"  ordinal_max_score: {config.ORDINAL_MAX_SCORE}")
    print(f"  score_agg: 'mean' (triggers continuous inference)")

    # Initialize manager with ordinal configuration and score aggregation
    # - ordinal_tasks: list of task names using ordinal scoring
    # - ordinal_max_score: maximum ordinal score value
    # - score_agg='mean': aggregate multiple scores using mean
    #
    # The combination of ordinal_tasks + score_agg='mean' triggers:
    #   determine_score_type() -> 'continuous_bounded' with bounds [0, ordinal_max_score]
    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'metadata.subtask'],
        reanalysis_interval=5,      # Run inference every 5 completed samples
        min_samples_per_grouping=5, # Minimum samples before first inference
        ordinal_tasks=ordinal_task_names,  # Task names using ordinal scoring
        ordinal_max_score=config.ORDINAL_MAX_SCORE,  # Maximum ordinal score
        gpu_ids=None,               # CPU-only
        manager_name='continuous_inference_test',
        shadow_mode=False,
        score_agg='mean',           # Aggregate scores using mean -> triggers continuous inference
        random_seed=random_seed,
    )

    print(f"\nManager initialized: {manager.manager_name}")
    print(f"Grouping columns: {manager.grouping_columns}")

    # =========================================================================
    # Create Mock EvalSpec and Samples (simulating inspect_ai)
    # =========================================================================
    print_section("PREPARING EVALUATION", "-")

    # Create samples with subtask assignments in metadata
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
                "base_score": subtask_config['base_score']
            }
            mock_samples.append(sample)
            sample_counter += 1

    # Create mock EvalSpec (what inspect_ai passes to start_task)
    mock_eval_spec = Mock()
    mock_eval_spec.model = config.MODEL
    mock_eval_spec.task = "rubric_benchmark"
    mock_eval_spec.eval_id = f"eval_rubric_benchmark_{random_seed}"
    mock_eval_spec.metadata = {"test_type": "continuous_inference_integration"}
    mock_eval_spec.tags = ["early_stopping_test", "continuous_inference", "ordinal_scores"]

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
                    'raw_scores': None,
                    'aggregated_score': None,
                    'scheduled': False,
                    'stopped': True,
                    'stop_reason': early_stop.reason
                })
                continue

            # Sample is scheduled - generate and submit scores
            total_scheduled += 1
            epoch_scheduled += 1
            grouping_stats[subtask]['scheduled'] += 1

            # Generate multiple ordinal scores (will be aggregated by manager)
            raw_scores = score_generator.generate_scores(sample_id, epoch, subtask)

            # Create mock score objects (as inspect_ai would provide)
            # Each score is an ordinal value that will be aggregated
            mock_scores = {}
            for score_name, score_value in raw_scores.items():
                mock_score_obj = Mock()
                mock_score_obj.value = score_value  # Ordinal integer
                mock_sample_score = Mock()
                mock_sample_score.score = mock_score_obj
                mock_scores[score_name] = mock_sample_score

            # Complete the sample - manager will aggregate scores using mean
            await manager.complete_sample(
                id=sample_id,
                epoch=epoch,
                scores=mock_scores
            )
            total_completed += 1
            grouping_stats[subtask]['completed'] += 1

            # Calculate aggregated score for logging
            aggregated_score = np.mean(list(raw_scores.values()))

            trial_data.append({
                'sample_id': sample_id,
                'subtask': subtask,
                'epoch': epoch,
                'raw_scores': str(raw_scores),
                'aggregated_score': aggregated_score,
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
            mean_score = scheduled['aggregated_score'].mean()
            print(f"    Mean aggregated score: {mean_score:.3f}/{config.ORDINAL_MAX_SCORE}")

    if total_completed > 0:
        completed_scores = trial_df[trial_df['scheduled']]['aggregated_score']
        print(f"\nOverall Score Statistics (completed trials):")
        print(f"  Mean: {completed_scores.mean():.3f}/{config.ORDINAL_MAX_SCORE}")
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

def test_continuous_inference_integration():
    """Pytest entry point."""
    manager, diagnostics, trial_df = asyncio.run(run_continuous_inference_test())

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
        description='Continuous inference integration test simulating inspect_ai usage with ordinal scores'
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
            run_continuous_inference_test(
                output_dir=args.output_dir,
                random_seed=args.seed,
            )
        )
    finally:
        logging_ctx.teardown()
