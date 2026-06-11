"""
Quick end-to-end simulation test (reduced scale for faster execution).

Demonstrates full optimal stopping workflow with:
- 2 models tested across 2 tasks
- 1 binary scoring task + 1 ordinal scoring task
- 20 sample_ids per task (reduced from 50)
- 10 epochs per sample_id (reduced from 15)
"""

import asyncio
import logging
import sys
from unittest.mock import Mock
import numpy as np
from optstop.early_stopping import OptimalStoppingManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class QuickSimConfig:
    """Quick simulation configuration."""
    MODELS = ['gpt-4', 'claude-3']
    TASKS = {
        'math': {'type': 'binary', 'difficulty': 0.7},
        'creative_writing': {'type': 'ordinal', 'max_score': 10, 'mean_score': 6.5},
    }
    NUM_SAMPLES = 20  # Reduced for speed
    MAX_EPOCHS = 10   # Reduced for speed
    MODEL_PERFORMANCE = {'gpt-4': 0.75, 'claude-3': 0.70}


class ScoreGenerator:
    """Generate realistic mock scores."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def generate_binary_score(self, model, task, epoch, sample_id, config):
        base_rate = config.MODEL_PERFORMANCE[model]
        task_difficulty = config.TASKS[task]['difficulty']
        adjusted_rate = base_rate * (1.0 - task_difficulty + 0.5)
        learning_bonus = min(0.15, epoch * 0.015)
        final_rate = min(0.95, adjusted_rate + learning_bonus)
        sample_seed = hash(sample_id) % 100
        sample_difficulty = (sample_seed / 100.0) * 0.2 - 0.1
        final_rate = max(0.05, min(0.95, final_rate + sample_difficulty))
        return 1.0 if self.rng.random() < final_rate else 0.0

    def generate_ordinal_score(self, model, task, epoch, sample_id, config):
        task_info = config.TASKS[task]
        max_score = task_info['max_score']
        mean_score = task_info['mean_score']
        base_rate = config.MODEL_PERFORMANCE[model]
        base_mean = mean_score * base_rate / 0.7
        learning_bonus = min(1.5, epoch * 0.15)
        adjusted_mean = min(max_score - 1, base_mean + learning_bonus)
        sample_seed = hash(sample_id) % 100
        sample_adjustment = (sample_seed / 100.0) * 2 - 1
        final_mean = max(1, min(max_score - 1, adjusted_mean + sample_adjustment))
        std_dev = 1.5
        score = self.rng.normal(final_mean, std_dev)
        return float(np.clip(np.round(score), 0, max_score))


async def run_quick_simulation():
    """Run quick simulation test."""

    print("\n" + "="*80)
    print("QUICK OPTIMAL STOPPING SIMULATION")
    print("="*80)

    config = QuickSimConfig()
    score_generator = ScoreGenerator(seed=42)

    # Configuration summary
    print(f"\n{'='*80}")
    print("CONFIGURATION")
    print("="*80)
    print(f"Models: {', '.join(config.MODELS)}")
    print(f"Tasks: {', '.join(config.TASKS.keys())}")
    print(f"  - math: Binary (0/1)")
    print(f"  - creative_writing: Ordinal (0-10)")
    print(f"Samples per task: {config.NUM_SAMPLES}")
    print(f"Epochs per sample: {config.MAX_EPOCHS}")
    total_planned = len(config.MODELS) * len(config.TASKS) * config.NUM_SAMPLES * config.MAX_EPOCHS
    print(f"Total planned trials: {total_planned}")

    # Initialize manager
    print(f"\n{'='*80}")
    print("INITIALIZING MANAGER")
    print("="*80)

    optstop_params = {
        'delta_item': 0.05,
        'delta_cap': 0.03,
        'cred_level': 0.95,
        'conservatism': 1.5,  # Less conservative for quicker stops
        'tune': 300,  # Fewer MCMC steps for speed
        'draws': 500,
    }

    print("Parameters:")
    for k, v in optstop_params.items():
        print(f"  {k}: {v}")

    ordinal_tasks = ['creative_writing']
    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
        reanalysis_interval=5,  # Run inference every 5 samples
        min_samples_per_grouping=10,  # Need at least 10 samples
        ordinal_tasks=ordinal_tasks,
        ordinal_max_score=10,
        ordinal_inference='modal',
        gpu_ids=None,
        max_workers=1,
    )
    print(f"\n✓ Manager initialized: {manager.manager_name}")
    print(f"  Grouping: ['model', 'task']")
    print(f"  Reanalysis interval: 5")
    print(f"  Min samples: 10")

    # Run simulation
    print(f"\n{'='*80}")
    print("RUNNING SIMULATION")
    print("="*80)

    total_run = 0
    total_stopped = 0
    grouping_results = {}

    for model in config.MODELS:
        for task_name, task_info in config.TASKS.items():
            grouping_name = f"{model}_{task_name}"
            print(f"\n{'-'*80}")
            print(f"Grouping: {grouping_name}")
            print(f"  Model: {model}, Task: {task_name} ({task_info['type']})")
            print("-"*80)

            # Create mock task
            mock_dataset = Mock()
            mock_dataset.sample_ids = [f"s{i:02d}" for i in range(config.NUM_SAMPLES)]
            mock_config = Mock()
            mock_config.epochs = config.MAX_EPOCHS
            mock_task = Mock()
            mock_task.dataset = mock_dataset
            mock_task.config = mock_config
            mock_task.model = model
            mock_task.task = task_name
            mock_task.metadata = {}
            mock_task.tags = []

            await manager.start_task(mock_task)

            grouping_results[grouping_name] = {
                'trials_run': 0,
                'trials_stopped': 0,
                'samples_stopped': set(),
                'grouping_stopped': False,
            }

            # Execute trials
            for epoch in range(1, config.MAX_EPOCHS + 1):
                for sample_id in mock_dataset.sample_ids:

                    # Check if should schedule
                    result = await manager.schedule_sample(mock_task, id=sample_id, epoch=epoch)

                    if result is not None:
                        total_stopped += 1
                        grouping_results[grouping_name]['trials_stopped'] += 1
                        if sample_id not in grouping_results[grouping_name]['samples_stopped']:
                            grouping_results[grouping_name]['samples_stopped'].add(sample_id)
                        continue

                    # Generate and submit score
                    if task_info['type'] == 'binary':
                        score = score_generator.generate_binary_score(
                            model, task_name, epoch, sample_id, config
                        )
                    else:
                        score = score_generator.generate_ordinal_score(
                            model, task_name, epoch, sample_id, config
                        )

                    mock_score_obj = Mock()
                    mock_score_obj.as_float.return_value = score
                    mock_sample_score = Mock()
                    mock_sample_score.score = mock_score_obj
                    mock_scores = {'scorer': mock_sample_score}

                    await manager.complete_sample(mock_task, id=sample_id, epoch=epoch, scores=mock_scores)

                    total_run += 1
                    grouping_results[grouping_name]['trials_run'] += 1

                # Check if grouping stopped
                if grouping_name in manager._stopped_groupings:
                    grouping_results[grouping_name]['grouping_stopped'] = True
                    print(f"  → GROUPING STOPPED at epoch {epoch}")
                    break

            # Grouping summary
            gr = grouping_results[grouping_name]
            print(f"  Results:")
            print(f"    Trials run: {gr['trials_run']}")
            print(f"    Trials stopped: {gr['trials_stopped']}")
            print(f"    Samples stopped: {len(gr['samples_stopped'])}/{config.NUM_SAMPLES}")
            print(f"    Grouping stopped: {gr['grouping_stopped']}")

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\nOverall Statistics:")
    print(f"  Total planned: {total_planned}")
    print(f"  Trials run: {total_run}")
    print(f"  Trials stopped: {total_stopped}")
    print(f"  Trials saved: {total_planned - total_run} ({100*(total_planned-total_run)/total_planned:.1f}%)")

    print(f"\nGrouping Results:")
    for name, gr in grouping_results.items():
        status = "STOPPED" if gr['grouping_stopped'] else "COMPLETE"
        print(f"  {name:30s} [{status}] - {gr['trials_run']} trials, {len(gr['samples_stopped'])} samples stopped")

    # Stopped samples details
    if manager.stopped_samples:
        print(f"\nStopped Samples: {len(manager.stopped_samples)} total")
        for stopped in manager.stopped_samples[:10]:
            print(f"  - {stopped.id}: {stopped.early_stop.reason}")
        if len(manager.stopped_samples) > 10:
            print(f"  ... and {len(manager.stopped_samples) - 10} more")

    # Performance by task
    print(f"\nPerformance by Task:")
    for task_name, task_info in config.TASKS.items():
        task_data = manager.compiled_dataset[
            (manager.compiled_dataset['task'] == task_name) &
            (manager.compiled_dataset['trial_ran'] == 1)
        ]
        if len(task_data) > 0:
            mean_score = task_data['score'].mean()
            if task_info['type'] == 'binary':
                print(f"  {task_name:20s} (binary):  Success rate = {mean_score:.3f} (n={len(task_data)})")
            else:
                print(f"  {task_name:20s} (ordinal): Mean score = {mean_score:.2f}/10 (n={len(task_data)})")

    print(f"\n{'='*80}")
    print("SIMULATION COMPLETE")
    print("="*80 + "\n")

    return manager, grouping_results


if __name__ == "__main__":
    manager, results = asyncio.run(run_quick_simulation())
