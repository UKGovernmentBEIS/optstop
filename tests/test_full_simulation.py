"""
Full end-to-end simulation test for OptimalStoppingManager.

This test simulates a complete evaluation run with:
- 4 models tested across 5 tasks
- 3 binary scoring tasks (0/1) + 2 ordinal scoring tasks (0-10)
- 50 sample_ids per task
- 15 epochs per sample_id
- Realistic score generation with performance patterns
- Full optimal stopping inference execution
"""

import asyncio
import logging
import sys
from unittest.mock import Mock
import numpy as np
import pandas as pd
from optstop.early_stopping import OptimalStoppingManager

# Configure logging to show detailed progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SimulationConfig:
    """Configuration for the simulation test."""

    # Models to test
    MODELS = ['gpt-4', 'claude-3', 'gemini-pro', 'llama-70b']

    # Tasks with their scoring types
    TASKS = {
        'math': {'type': 'binary', 'difficulty': 0.7},
        'coding': {'type': 'binary', 'difficulty': 0.6},
        'reasoning': {'type': 'binary', 'difficulty': 0.8},
        'creative_writing': {'type': 'ordinal', 'max_score': 10, 'mean_score': 6.5},
        'translation': {'type': 'ordinal', 'max_score': 10, 'mean_score': 7.0},
    }

    # Simulation parameters
    NUM_SAMPLES = 50  # Samples per task
    MAX_EPOCHS = 15   # Maximum epochs per sample

    # Model performance profiles (base success rates for binary tasks)
    MODEL_PERFORMANCE = {
        'gpt-4': 0.75,
        'claude-3': 0.72,
        'gemini-pro': 0.68,
        'llama-70b': 0.65,
    }


class ScoreGenerator:
    """Generate realistic mock scores with learning patterns."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def generate_binary_score(self, model: str, task: str, epoch: int,
                             sample_id: str, config: SimulationConfig) -> float:
        """Generate binary score (0 or 1) with learning curve."""
        # Base performance for model
        base_rate = config.MODEL_PERFORMANCE[model]

        # Task difficulty adjustment
        task_difficulty = config.TASKS[task]['difficulty']
        adjusted_rate = base_rate * (1.0 - task_difficulty + 0.5)

        # Learning curve: improve slightly with more epochs
        learning_bonus = min(0.15, epoch * 0.01)
        final_rate = min(0.95, adjusted_rate + learning_bonus)

        # Add some sample-specific variation (some samples are harder)
        sample_seed = hash(sample_id) % 100
        sample_difficulty = (sample_seed / 100.0) * 0.2 - 0.1  # -0.1 to +0.1
        final_rate = max(0.05, min(0.95, final_rate + sample_difficulty))

        # Generate score
        return 1.0 if self.rng.random() < final_rate else 0.0

    def generate_ordinal_score(self, model: str, task: str, epoch: int,
                               sample_id: str, config: SimulationConfig) -> float:
        """Generate ordinal score (0-10) with learning curve."""
        task_info = config.TASKS[task]
        max_score = task_info['max_score']
        mean_score = task_info['mean_score']

        # Base performance for model (scale to ordinal range)
        base_rate = config.MODEL_PERFORMANCE[model]
        base_mean = mean_score * base_rate / 0.7  # Normalize to expected performance

        # Learning curve: improve with more epochs
        learning_bonus = min(1.5, epoch * 0.1)
        adjusted_mean = min(max_score - 1, base_mean + learning_bonus)

        # Add sample-specific variation
        sample_seed = hash(sample_id) % 100
        sample_adjustment = (sample_seed / 100.0) * 2 - 1  # -1 to +1
        final_mean = max(1, min(max_score - 1, adjusted_mean + sample_adjustment))

        # Generate score from normal distribution, clipped to valid range
        std_dev = 1.5
        score = self.rng.normal(final_mean, std_dev)
        return float(np.clip(np.round(score), 0, max_score))


async def run_full_simulation():
    """Run the complete simulation test."""

    print("\n" + "="*80)
    print("OPTIMAL STOPPING SIMULATION TEST")
    print("="*80)

    config = SimulationConfig()
    score_generator = ScoreGenerator(seed=42)

    # =========================================================================
    # STEP 1: Print Configuration
    # =========================================================================
    print("\n" + "-"*80)
    print("SIMULATION CONFIGURATION")
    print("-"*80)
    print(f"\nModels: {', '.join(config.MODELS)}")
    print(f"Number of models: {len(config.MODELS)}")
    print(f"\nTasks:")
    for task_name, task_info in config.TASKS.items():
        if task_info['type'] == 'binary':
            print(f"  - {task_name}: Binary scoring (difficulty: {task_info['difficulty']})")
        else:
            print(f"  - {task_name}: Ordinal scoring (0-{task_info['max_score']}, mean: {task_info['mean_score']})")

    print(f"\nSamples per task: {config.NUM_SAMPLES}")
    print(f"Maximum epochs per sample: {config.MAX_EPOCHS}")
    print(f"Total planned trials: {len(config.MODELS)} models × {len(config.TASKS)} tasks × {config.NUM_SAMPLES} samples × {config.MAX_EPOCHS} epochs = {len(config.MODELS) * len(config.TASKS) * config.NUM_SAMPLES * config.MAX_EPOCHS:,}")

    # =========================================================================
    # STEP 2: Initialize OptimalStoppingManager
    # =========================================================================
    print("\n" + "-"*80)
    print("INITIALIZING OPTIMAL STOPPING MANAGER")
    print("-"*80)

    # User-specified parameters
    optstop_params = {
        'delta_item': 0.05,      # Stop individual samples at 95% CI
        'delta_cap': 0.03,       # Stop entire grouping at 97% CI
        'cred_level': 0.95,      # 95% credible interval
        'conservatism': 2.0,     # Conservative stopping (wait for stable evidence)
        'low_performance_threshold': 0.3,  # Stop low performers early
        'CI_delta': 0.0001,      # Stabilization threshold
        'stab_window': 5,        # Stabilization window
        'tune': 500,             # MCMC tuning steps
        'draws': 1000,           # MCMC samples
    }

    print("\nOptimal Stopping Parameters:")
    for param, value in optstop_params.items():
        print(f"  {param}: {value}")

    print("\nGrouping Configuration:")
    print("  - Grouping by: ['model', 'task']")
    print("  - Total unique groupings: 4 models × 5 tasks = 20 groupings")
    print("  - Reanalysis interval: 10 (run inference every 10 completed samples)")
    print("  - Minimum samples per grouping: 15 (require at least 15 samples before stopping)")

    # Identify ordinal tasks
    ordinal_tasks = [name for name, info in config.TASKS.items() if info['type'] == 'ordinal']
    print(f"\nOrdinal tasks: {ordinal_tasks}")
    print(f"Ordinal max score: 10")

    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
        reanalysis_interval=10,  # Run inference every 10 samples
        min_samples_per_grouping=15,  # Need at least 15 samples before stopping
        score_column='score',
        sample_id_column='sample_id',
        epoch_column='epoch',
        ordinal_tasks=ordinal_tasks,
        ordinal_max_score=10,
        ordinal_inference='modal',  # Use modal inference for ordinal tasks
        gpu_ids=None,  # Use CPU for this test
        max_workers=1,
    )

    print(f"\nManager initialized: {manager.manager_name}")

    # =========================================================================
    # STEP 3: Run Simulation for Each Model-Task Combination
    # =========================================================================
    print("\n" + "="*80)
    print("STARTING SIMULATION")
    print("="*80)

    total_trials_run = 0
    total_trials_scheduled = 0
    total_trials_stopped = 0
    grouping_stats = {}

    # Process each model-task combination
    for model_idx, model in enumerate(config.MODELS):
        for task_idx, (task_name, task_info) in enumerate(config.TASKS.items()):

            grouping_name = f"{model}_{task_name}"
            print(f"\n{'='*80}")
            print(f"GROUPING {model_idx * len(config.TASKS) + task_idx + 1}/20: {grouping_name}")
            print(f"{'='*80}")
            print(f"Model: {model}")
            print(f"Task: {task_name} ({task_info['type']} scoring)")

            # Create mock task
            mock_dataset = Mock()
            mock_dataset.sample_ids = [f"sample_{i:03d}" for i in range(config.NUM_SAMPLES)]

            mock_config = Mock()
            mock_config.epochs = config.MAX_EPOCHS

            mock_task = Mock()
            mock_task.dataset = mock_dataset
            mock_task.config = mock_config
            mock_task.model = model
            mock_task.task = task_name
            mock_task.metadata = {}
            mock_task.tags = []

            # Initialize task
            await manager.start_task(mock_task)
            print(f"\n✓ Task initialized: {len(mock_dataset.sample_ids)} samples × {config.MAX_EPOCHS} epochs = {len(mock_dataset.sample_ids) * config.MAX_EPOCHS} planned trials")

            # Track grouping statistics
            grouping_stats[grouping_name] = {
                'trials_run': 0,
                'trials_stopped': 0,
                'samples_stopped': set(),
                'grouping_stopped': False,
                'final_epoch_reached': 0,
            }

            # Simulate trial execution
            trials_this_grouping = 0
            stopped_this_grouping = 0

            # Process trials epoch by epoch, sample by sample
            for epoch in range(1, config.MAX_EPOCHS + 1):
                for sample_id in mock_dataset.sample_ids:

                    # Check if we should schedule this trial
                    schedule_result = await manager.schedule_sample(mock_task, id=sample_id, epoch=epoch)
                    total_trials_scheduled += 1

                    if schedule_result is not None:
                        # Trial is stopped
                        total_trials_stopped += 1
                        stopped_this_grouping += 1
                        if sample_id not in grouping_stats[grouping_name]['samples_stopped']:
                            grouping_stats[grouping_name]['samples_stopped'].add(sample_id)
                        continue

                    # Trial should run - generate score
                    if task_info['type'] == 'binary':
                        score = score_generator.generate_binary_score(
                            model, task_name, epoch, sample_id, config
                        )
                    else:
                        score = score_generator.generate_ordinal_score(
                            model, task_name, epoch, sample_id, config
                        )

                    # Create mock score object
                    mock_score_obj = Mock()
                    mock_score_obj.as_float.return_value = score
                    mock_sample_score = Mock()
                    mock_sample_score.score = mock_score_obj
                    mock_scores = {'scorer': mock_sample_score}

                    # Complete the sample (this may trigger inference)
                    await manager.complete_sample(mock_task, id=sample_id, epoch=epoch, scores=mock_scores)

                    total_trials_run += 1
                    trials_this_grouping += 1
                    grouping_stats[grouping_name]['trials_run'] += 1

                    # Print progress every 100 trials
                    if trials_this_grouping % 100 == 0:
                        print(f"  Progress: {trials_this_grouping} trials completed, {stopped_this_grouping} stopped")

                # Check if grouping was stopped
                if grouping_name in manager._stopped_groupings:
                    grouping_stats[grouping_name]['grouping_stopped'] = True
                    grouping_stats[grouping_name]['final_epoch_reached'] = epoch
                    print(f"\n  ⚠ GROUPING STOPPED at epoch {epoch}")
                    print(f"    Reason: Entire grouping stopped by optimal stopping criteria")
                    break

            grouping_stats[grouping_name]['trials_stopped'] = stopped_this_grouping

            # Print grouping summary
            print(f"\n  Grouping Summary:")
            print(f"    Trials run: {trials_this_grouping}")
            print(f"    Trials stopped: {stopped_this_grouping}")
            print(f"    Samples stopped: {len(grouping_stats[grouping_name]['samples_stopped'])}/{config.NUM_SAMPLES}")
            print(f"    Grouping stopped: {grouping_stats[grouping_name]['grouping_stopped']}")
            if grouping_stats[grouping_name]['grouping_stopped']:
                print(f"    Stopped at epoch: {grouping_stats[grouping_name]['final_epoch_reached']}/{config.MAX_EPOCHS}")

    # =========================================================================
    # STEP 4: Print Final Summary
    # =========================================================================
    print("\n" + "="*80)
    print("SIMULATION COMPLETE - FINAL SUMMARY")
    print("="*80)

    print(f"\nOverall Statistics:")
    print(f"  Total trials planned: {len(config.MODELS) * len(config.TASKS) * config.NUM_SAMPLES * config.MAX_EPOCHS:,}")
    print(f"  Total trials scheduled: {total_trials_scheduled:,}")
    print(f"  Total trials run: {total_trials_run:,}")
    print(f"  Total trials stopped: {total_trials_stopped:,}")
    print(f"  Trials saved: {total_trials_scheduled - total_trials_run:,} ({100 * (total_trials_scheduled - total_trials_run) / total_trials_scheduled:.1f}%)")

    print(f"\nGrouping Statistics:")
    print(f"  Total groupings: {len(grouping_stats)}")
    groupings_fully_stopped = sum(1 for stats in grouping_stats.values() if stats['grouping_stopped'])
    print(f"  Groupings stopped early: {groupings_fully_stopped}/{len(grouping_stats)}")

    print(f"\nStopped Samples by Grouping:")
    for grouping_name, stats in grouping_stats.items():
        if len(stats['samples_stopped']) > 0 or stats['grouping_stopped']:
            status = "STOPPED" if stats['grouping_stopped'] else "PARTIAL"
            print(f"  {grouping_name:30s} [{status}]: {len(stats['samples_stopped']):2d} samples stopped, {stats['trials_run']:4d} trials run")

    # Print stopped samples details from manager
    if manager.stopped_samples:
        print(f"\nStopped Samples Details (from manager.stopped_samples):")
        print(f"  Total stopped samples: {len(manager.stopped_samples)}")

        # Group by grouping
        by_grouping = {}
        for stopped_sample in manager.stopped_samples:
            grouping_key = f"{stopped_sample.metadata.get('model', 'unknown')}_{stopped_sample.metadata.get('task', 'unknown')}"
            if grouping_key not in by_grouping:
                by_grouping[grouping_key] = []
            by_grouping[grouping_key].append(stopped_sample)

        for grouping_key, samples in by_grouping.items():
            print(f"\n  {grouping_key}:")
            for sample in samples[:5]:  # Show first 5
                print(f"    - {sample.id}: {sample.early_stop.reason}")
            if len(samples) > 5:
                print(f"    ... and {len(samples) - 5} more")

    # Print compiled dataset sample
    print(f"\nCompiled Dataset Summary:")
    print(f"  Total rows: {len(manager.compiled_dataset):,}")
    print(f"  Trials ran: {manager.compiled_dataset['trial_ran'].sum():,}")
    print(f"  Trials scheduled (status=False): {(~manager.compiled_dataset['schedule_status']).sum():,}")

    # Calculate statistics by task type
    print(f"\nPerformance by Task Type:")
    for task_name, task_info in config.TASKS.items():
        task_data = manager.compiled_dataset[manager.compiled_dataset['task'] == task_name]
        completed_data = task_data[task_data['trial_ran'] == 1]

        if len(completed_data) > 0:
            mean_score = completed_data['score'].mean()
            if task_info['type'] == 'binary':
                print(f"  {task_name:20s} (binary):  Success rate = {mean_score:.3f} ({len(completed_data)} trials)")
            else:
                print(f"  {task_name:20s} (ordinal): Mean score = {mean_score:.2f}/10 ({len(completed_data)} trials)")

    print("\n" + "="*80)
    print("SIMULATION TEST COMPLETE")
    print("="*80 + "\n")

    return manager, grouping_stats


if __name__ == "__main__":
    # Run the simulation
    manager, stats = asyncio.run(run_full_simulation())
