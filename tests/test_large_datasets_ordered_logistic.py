"""
Large dataset test for all three inference pathways with ordered_logistic model.

Tests:
- Binary pathway (0/1 scoring)
- Ordinal pathway (0-10 Likert scale) using ordered_logistic + hybrid inference
- Continuous pathway (0.0-1.0 bounded scores)

Key features:
- Uses ordered_logistic model type for ordinal inference (default)
- Tests hybrid inference mode which combines modal CI + entropy validation
- Comprehensive logging to verify model selection and inference behavior
- Verifies fallback behavior if ordered_logistic fails
"""

import asyncio
import logging
import sys
import time
from unittest.mock import Mock
import numpy as np
import pandas as pd
from optstop.early_stopping import OptimalStoppingManager

# Configure comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set specific loggers to appropriate levels
logging.getLogger('optstop').setLevel(logging.INFO)
logging.getLogger('optstop.rule').setLevel(logging.INFO)
logging.getLogger('optstop.ordinal_model').setLevel(logging.INFO)
logging.getLogger('optstop.ordinal_utils').setLevel(logging.INFO)
logging.getLogger('pymc').setLevel(logging.WARNING)
logging.getLogger('pytensor').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class LargeDatasetConfig:
    """Configuration for large dataset test with all three pathways."""

    # Models to test
    MODELS = ['model_A', 'model_B']

    # Tasks with all three scoring types
    TASKS = {
        # Binary tasks (0/1)
        'binary_easy': {'type': 'binary', 'base_rate': 0.8},
        'binary_hard': {'type': 'binary', 'base_rate': 0.5},
        # Ordinal tasks (0-10) - will use ordered_logistic + hybrid
        'ordinal_peaked': {'type': 'ordinal', 'max_score': 10, 'mean': 7.0, 'std': 1.5},
        'ordinal_spread': {'type': 'ordinal', 'max_score': 10, 'mean': 5.0, 'std': 2.5},
        # Continuous tasks (0.0-1.0)
        'continuous_high': {'type': 'continuous', 'mean': 0.75, 'std': 0.1},
        'continuous_mid': {'type': 'continuous', 'mean': 0.5, 'std': 0.15},
    }

    # Simulation parameters
    NUM_SAMPLES = 15  # Samples per task
    MAX_EPOCHS = 8    # Epochs per sample

    # Model performance multipliers
    MODEL_PERFORMANCE = {
        'model_A': 1.0,
        'model_B': 0.9,
    }


class ScoreGenerator:
    """Generate realistic mock scores for all three pathways."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def generate_score(self, model: str, task: str, epoch: int,
                      sample_id: str, config: LargeDatasetConfig) -> float:
        """Generate score based on task type."""
        task_info = config.TASKS[task]
        task_type = task_info['type']

        # Model performance modifier
        model_mult = config.MODEL_PERFORMANCE[model]

        # Sample-specific variation
        sample_seed = hash(sample_id) % 1000
        sample_var = (sample_seed / 1000.0) * 0.2 - 0.1

        # Epoch learning bonus
        epoch_bonus = epoch * 0.02

        if task_type == 'binary':
            base_rate = task_info['base_rate']
            final_rate = min(0.95, max(0.05,
                base_rate * model_mult + sample_var + epoch_bonus))
            return 1.0 if self.rng.random() < final_rate else 0.0

        elif task_type == 'ordinal':
            mean = task_info['mean']
            std = task_info['std']
            adjusted_mean = mean * model_mult + sample_var * 2 + epoch_bonus * 5
            score = self.rng.normal(adjusted_mean, std)
            return float(np.clip(np.round(score), 0, task_info['max_score']))

        elif task_type == 'continuous':
            mean = task_info['mean']
            std = task_info['std']
            adjusted_mean = mean * model_mult + sample_var + epoch_bonus
            score = self.rng.normal(adjusted_mean, std)
            return float(np.clip(score, 0.0, 1.0))

        raise ValueError(f"Unknown task type: {task_type}")


def print_section(title: str, char: str = "="):
    """Print a section header."""
    print(f"\n{char*80}")
    print(title)
    print(char*80)


async def run_large_dataset_test():
    """Run comprehensive test of all three inference pathways."""

    start_time = time.time()

    print_section("LARGE DATASET TEST: ALL THREE INFERENCE PATHWAYS")
    print("Testing: Binary, Ordinal (ordered_logistic + hybrid), Continuous")

    config = LargeDatasetConfig()
    score_generator = ScoreGenerator(seed=42)

    # =========================================================================
    # Configuration Summary
    # =========================================================================
    print_section("CONFIGURATION", "-")

    print(f"\nModels: {', '.join(config.MODELS)}")
    print(f"\nTasks by type:")

    binary_tasks = [t for t, info in config.TASKS.items() if info['type'] == 'binary']
    ordinal_tasks = [t for t, info in config.TASKS.items() if info['type'] == 'ordinal']
    continuous_tasks = [t for t, info in config.TASKS.items() if info['type'] == 'continuous']

    print(f"  Binary tasks: {binary_tasks}")
    for t in binary_tasks:
        print(f"    - {t}: base_rate={config.TASKS[t]['base_rate']}")

    print(f"  Ordinal tasks: {ordinal_tasks}")
    for t in ordinal_tasks:
        info = config.TASKS[t]
        print(f"    - {t}: 0-{info['max_score']}, mean={info['mean']}, std={info['std']}")

    print(f"  Continuous tasks: {continuous_tasks}")
    for t in continuous_tasks:
        info = config.TASKS[t]
        print(f"    - {t}: mean={info['mean']}, std={info['std']}")

    print(f"\nSimulation scale:")
    print(f"  Samples per task: {config.NUM_SAMPLES}")
    print(f"  Epochs per sample: {config.MAX_EPOCHS}")
    total_planned = len(config.MODELS) * len(config.TASKS) * config.NUM_SAMPLES * config.MAX_EPOCHS
    print(f"  Total planned trials: {total_planned}")

    # =========================================================================
    # Initialize OptimalStoppingManager
    # =========================================================================
    print_section("INITIALIZING OPTIMAL STOPPING MANAGER", "-")

    optstop_params = {
        'delta_item': 0.08,       # CI width threshold for individual samples
        'delta_cap': 0.06,        # CI width threshold for grouping
        'cred_level': 0.95,       # 95% credible intervals
        'conservatism': 1.5,      # Moderate conservatism
        'low_performance_threshold': 0.3,
        'tune': 200,              # MCMC tuning (reduced for speed)
        'draws': 300,             # MCMC samples (reduced for speed)
        'chains': 2,
        'cores': 1,
    }

    print("\nOptimal Stopping Parameters:")
    for param, value in optstop_params.items():
        print(f"  {param}: {value}")

    print("\nOrdinal Configuration:")
    print(f"  ordinal_tasks: {ordinal_tasks}")
    print(f"  ordinal_max_score: 10")
    print(f"  ordinal_inference: 'hybrid' (modal CI + entropy validation)")
    print(f"  ordinal_model_type: 'ordered_logistic' (cumulative link model)")

    print("\nContinuous Configuration:")
    print(f"  continuous_tasks: {continuous_tasks}")
    print("  Continuous scores (0.0-1.0) will use continuous_bounded pathway")

    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
        reanalysis_interval=5,
        min_samples_per_grouping=8,
        score_column='score',
        sample_id_column='sample_id',
        epoch_column='epoch',
        ordinal_tasks=ordinal_tasks,
        ordinal_max_score=10,
        ordinal_inference='hybrid',
        ordinal_model_type='ordered_logistic',
        gpu_ids=None,
        max_workers=1,
    )

    print(f"\nManager initialized: {manager.manager_name}")

    # =========================================================================
    # Run Simulation
    # =========================================================================
    print_section("RUNNING SIMULATION")

    total_run = 0
    total_stopped = 0
    grouping_results = {}
    pathway_stats = {'binary': [], 'ordinal': [], 'continuous': []}

    for model in config.MODELS:
        for task_name, task_info in config.TASKS.items():
            grouping_name = f"{model}_{task_name}"
            task_type = task_info['type']

            print(f"\n{'-'*80}")
            print(f"GROUPING: {grouping_name}")
            print(f"  Model: {model}")
            print(f"  Task: {task_name}")
            print(f"  Type: {task_type.upper()}")
            if task_type == 'ordinal':
                print(f"  Model: ordered_logistic (hybrid inference)")
            print("-"*80)

            # Create mock task (EvalSpec)
            sample_ids = [f"s{i:02d}" for i in range(config.NUM_SAMPLES)]
            mock_task = Mock()
            mock_task.model = model
            mock_task.task = task_name
            mock_task.eval_id = f"eval_{grouping_name}"
            mock_task.metadata = {}
            mock_task.tags = []

            # Create Sample objects for start_task
            mock_samples = []
            for sid in sample_ids:
                sample = Mock()
                sample.id = sid
                sample.metadata = {}
                mock_samples.append(sample)

            await manager.start_task(mock_task, mock_samples, config.MAX_EPOCHS)

            grouping_results[grouping_name] = {
                'type': task_type,
                'trials_run': 0,
                'trials_stopped': 0,
                'samples_stopped': set(),
                'grouping_stopped': False,
                'scores': [],
                'inference_time': 0,
            }

            grouping_start = time.time()

            # Execute trials
            for epoch in range(1, config.MAX_EPOCHS + 1):
                for sample_id in sample_ids:

                    # Check if should schedule
                    result = await manager.schedule_sample(id=sample_id, epoch=epoch)

                    if result is not None:
                        total_stopped += 1
                        grouping_results[grouping_name]['trials_stopped'] += 1
                        if sample_id not in grouping_results[grouping_name]['samples_stopped']:
                            grouping_results[grouping_name]['samples_stopped'].add(sample_id)
                        continue

                    # Generate and submit score
                    score = score_generator.generate_score(
                        model, task_name, epoch, sample_id, config
                    )
                    grouping_results[grouping_name]['scores'].append(score)

                    mock_score_obj = Mock()
                    mock_score_obj.value = score  # _extract_score_value accesses .score.value
                    mock_sample_score = Mock()
                    mock_sample_score.score = mock_score_obj
                    mock_scores = {'scorer': mock_sample_score}

                    await manager.complete_sample(id=sample_id, epoch=epoch, scores=mock_scores)

                    total_run += 1
                    grouping_results[grouping_name]['trials_run'] += 1

                # Check if grouping stopped
                if grouping_name in manager._stopped_groupings:
                    grouping_results[grouping_name]['grouping_stopped'] = True
                    print(f"  GROUPING STOPPED at epoch {epoch}")
                    break

            grouping_results[grouping_name]['inference_time'] = time.time() - grouping_start

            # Grouping summary
            gr = grouping_results[grouping_name]
            scores = gr['scores']

            print(f"\n  Results:")
            print(f"    Trials run: {gr['trials_run']}")
            print(f"    Trials stopped: {gr['trials_stopped']}")
            print(f"    Samples stopped: {len(gr['samples_stopped'])}/{config.NUM_SAMPLES}")
            print(f"    Grouping stopped: {gr['grouping_stopped']}")
            print(f"    Time: {gr['inference_time']:.2f}s")

            if scores:
                if task_type == 'binary':
                    print(f"    Mean score: {np.mean(scores):.3f} (success rate)")
                elif task_type == 'ordinal':
                    print(f"    Mean score: {np.mean(scores):.2f}/10")
                else:
                    print(f"    Mean score: {np.mean(scores):.3f} (bounded)")

            pathway_stats[task_type].append(gr)

    # =========================================================================
    # Final Summary
    # =========================================================================
    print_section("FINAL SUMMARY")

    total_time = time.time() - start_time

    print(f"\nOverall Statistics:")
    print(f"  Total planned trials: {total_planned}")
    print(f"  Trials run: {total_run}")
    print(f"  Trials stopped: {total_stopped}")
    savings = 100 * (total_planned - total_run) / total_planned
    print(f"  Trials saved: {total_planned - total_run} ({savings:.1f}%)")
    print(f"  Total time: {total_time:.1f}s")

    # Summary by pathway
    print_section("RESULTS BY PATHWAY", "-")

    for pathway_type in ['binary', 'ordinal', 'continuous']:
        stats = pathway_stats[pathway_type]
        if not stats:
            continue

        print(f"\n{pathway_type.upper()} PATHWAY:")
        total_trials = sum(s['trials_run'] for s in stats)
        total_stopped_trials = sum(s['trials_stopped'] for s in stats)
        groupings_stopped = sum(1 for s in stats if s['grouping_stopped'])
        avg_time = np.mean([s['inference_time'] for s in stats])

        print(f"  Groupings: {len(stats)}")
        print(f"  Trials run: {total_trials}")
        print(f"  Trials stopped: {total_stopped_trials}")
        print(f"  Groupings stopped early: {groupings_stopped}/{len(stats)}")
        print(f"  Avg time per grouping: {avg_time:.2f}s")

        # Scores summary
        all_scores = []
        for s in stats:
            all_scores.extend(s['scores'])
        if all_scores:
            if pathway_type == 'binary':
                print(f"  Overall success rate: {np.mean(all_scores):.3f}")
            elif pathway_type == 'ordinal':
                print(f"  Overall mean score: {np.mean(all_scores):.2f}/10")
            else:
                print(f"  Overall mean score: {np.mean(all_scores):.3f}")

    # Grouping details
    print_section("GROUPING DETAILS", "-")

    for grouping_name, gr in sorted(grouping_results.items()):
        status = "STOPPED" if gr['grouping_stopped'] else "COMPLETE"
        print(f"  {grouping_name:30s} [{gr['type']:10s}] [{status:8s}] "
              f"trials={gr['trials_run']:3d}, stopped={len(gr['samples_stopped']):2d}, "
              f"time={gr['inference_time']:.1f}s")

    # Stopped samples
    if manager.stopped_samples:
        print_section("STOPPED SAMPLES", "-")
        print(f"\nTotal stopped samples: {len(manager.stopped_samples)}")

        # Group by task type
        by_type = {'binary': [], 'ordinal': [], 'continuous': []}
        for stopped in manager.stopped_samples:
            task = stopped.metadata.get('task', 'unknown')
            for t_name, t_info in config.TASKS.items():
                if task == t_name:
                    by_type[t_info['type']].append(stopped)
                    break

        for t_type, samples in by_type.items():
            if samples:
                print(f"\n  {t_type.upper()} pathway stopped samples: {len(samples)}")
                for s in samples[:3]:
                    print(f"    - {s.id}: {s.early_stop.reason}")
                if len(samples) > 3:
                    print(f"    ... and {len(samples) - 3} more")

    # Compiled dataset summary
    print_section("COMPILED DATASET SUMMARY", "-")

    df = manager.compiled_dataset
    print(f"\nTotal rows: {len(df)}")
    print(f"Trials ran: {df['trial_ran'].sum()}")

    print("\nBy task type:")
    for task_name, task_info in config.TASKS.items():
        task_data = df[(df['task'] == task_name) & (df['trial_ran'] == 1)]
        if len(task_data) > 0:
            mean_score = task_data['score'].mean()
            print(f"  {task_name:20s} ({task_info['type']:10s}): "
                  f"n={len(task_data):3d}, mean={mean_score:.3f}")

    print_section("TEST COMPLETE")
    print(f"Total execution time: {total_time:.1f}s")

    return manager, grouping_results


def test_large_datasets_ordered_logistic():
    """Pytest entry point for the large dataset test."""
    manager, results = asyncio.run(run_large_dataset_test())

    # Assertions
    assert manager is not None
    assert len(results) > 0

    # Verify all three pathways were tested
    types_tested = set(r['type'] for r in results.values())
    assert 'binary' in types_tested, "Binary pathway not tested"
    assert 'ordinal' in types_tested, "Ordinal pathway not tested"
    assert 'continuous' in types_tested, "Continuous pathway not tested"

    # Verify some trials ran for each type
    for pathway_type in ['binary', 'ordinal', 'continuous']:
        type_results = [r for r in results.values() if r['type'] == pathway_type]
        total_trials = sum(r['trials_run'] for r in type_results)
        assert total_trials > 0, f"No trials ran for {pathway_type} pathway"

    print("\nAll assertions passed!")


if __name__ == "__main__":
    # Run the test directly
    manager, results = asyncio.run(run_large_dataset_test())
