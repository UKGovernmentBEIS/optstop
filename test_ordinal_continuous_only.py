#!/usr/bin/env python3
"""
Test continuous float ordinal inference with corrected configuration.

This test suite focuses ONLY on ordinal tasks with continuous float scores,
applying the fix discovered from discrete integer testing:
- n_groupings=2 ensures 'task' is in grouping_columns
- ordinal_inference='modal' for fast, production-ready inference
- Continuous scores generated with small Gaussian noise around base values

Purpose: Compare continuous float vs discrete integer ordinal inference performance.
"""

import sys
import os
import time
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from enum import Enum
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optstop.early_stopping import OptimalStoppingManager
from mock_inspect_early_stop import EarlyStop, StoppedSample, EarlyStopping


# Create mock classes for Sample, SampleScore, and EvalSpec
class MockScore:
    """Mock score value object."""
    def __init__(self, value):
        self.value = value


class SampleScore:
    """Mock SampleScore object."""
    def __init__(self, value):
        self.score = MockScore(value)


class Sample:
    """Mock Sample object."""
    def __init__(self, id: str, metadata: dict):
        self.id = id
        self.metadata = metadata


class EvalSpec:
    """Mock EvalSpec object."""
    def __init__(self, model: str, task: str, eval_id: str):
        self.model = model
        self.task = task
        self.eval_id = eval_id


# ============================================================================
# PERFORMANCE PATTERNS
# ============================================================================

class PerformancePattern(str, Enum):
    """Performance pattern for score generation."""
    CONSISTENT_GOOD = "consistent_good"
    CONSISTENT_BAD = "consistent_bad"
    INCONSISTENT = "inconsistent"
    IMPROVING = "improving"


# ============================================================================
# CONTINUOUS FLOAT SCORE GENERATOR
# ============================================================================

class ContinuousOrdinalGenerator:
    """
    Generate CONTINUOUS FLOAT ordinal scores (e.g., 8.8, 9.1, 9.2).
    This matches the V2 test approach with Gaussian noise.
    """

    @staticmethod
    def generate_ordinal_scores(
        n_samples: int,
        n_epochs: int,
        pattern: str,
        max_score: int = 10,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate continuous ordinal scores with specified performance pattern.

        Returns DataFrame with columns: sample_id, epoch, score
        All scores are CONTINUOUS FLOATS (e.g., 8.6, 9.1, 9.2, 1.1, 0.8).
        """
        if seed is not None:
            np.random.seed(seed)

        data = []

        for sample_id in range(1, n_samples + 1):
            for epoch in range(1, n_epochs + 1):
                if pattern == PerformancePattern.CONSISTENT_GOOD:
                    # EXTREMELY consistent: scores tightly clustered around 9
                    base_score = 9.0
                    noise = np.random.normal(0, 0.2)  # Very small variance
                    score = np.clip(base_score + noise, 0, max_score)

                elif pattern == PerformancePattern.CONSISTENT_BAD:
                    # EXTREMELY consistent: scores tightly clustered around 1
                    base_score = 1.0
                    noise = np.random.normal(0, 0.2)  # Very small variance
                    score = np.clip(base_score + noise, 0, max_score)

                elif pattern == PerformancePattern.INCONSISTENT:
                    # Wide range of scores across samples
                    if sample_id % 3 == 0:  # Some items very good
                        base = 8.5
                        noise = np.random.normal(0, 0.8)
                    elif sample_id % 3 == 1:  # Some items very bad
                        base = 1.5
                        noise = np.random.normal(0, 0.8)
                    else:  # Some items medium
                        base = 5.0
                        noise = np.random.normal(0, 1.2)
                    score = np.clip(base + noise, 0, max_score)

                elif pattern == PerformancePattern.IMPROVING:
                    # Scores improve with epochs
                    base_score = 3.0 + (epoch / n_epochs) * 6.0  # 3.0 -> 9.0
                    noise = np.random.normal(0, 0.3)
                    score = np.clip(base_score + noise, 0, max_score)

                else:
                    raise ValueError(f"Unknown pattern: {pattern}")

                data.append({
                    'sample_id': f'sample_{sample_id}',
                    'epoch': epoch,
                    'score': score  # CONTINUOUS FLOAT
                })

        return pd.DataFrame(data)


# ============================================================================
# MOCK OBJECTS
# ============================================================================

def create_mock_evalspec(model: str = "test_model", task: str = "test_task", eval_id: str = "test_eval") -> EvalSpec:
    """Create a minimal mock EvalSpec."""
    return EvalSpec(model=model, task=task, eval_id=eval_id)


def create_mock_scores(score_value: float) -> Dict[str, SampleScore]:
    """Create mock scores dictionary."""
    return {'scorer_1': SampleScore(score_value)}


# ============================================================================
# TEST SCENARIO
# ============================================================================

class TestScenario:
    """Container for test scenario configuration."""

    def __init__(
        self,
        name: str,
        description: str,
        n_samples: int,
        n_epochs: int,
        performance_pattern: str,
        reanalysis_interval: int = 10,
    ):
        self.name = name
        self.description = description
        self.n_samples = n_samples
        self.n_epochs = n_epochs
        self.performance_pattern = performance_pattern
        self.reanalysis_interval = reanalysis_interval


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_scenario(scenario: TestScenario) -> Dict[str, Any]:
    """Run a single test scenario and return results."""

    print("\n" + "="*80)
    print(f"TEST: {scenario.name}")
    print("="*80)
    print(f"Description: {scenario.description}")
    print(f"Samples: {scenario.n_samples}, Epochs: {scenario.n_epochs}")
    print(f"Pattern: {scenario.performance_pattern}")
    print("-"*80)

    # Generate continuous float scores
    print("📊 Generating continuous ordinal scores...")
    df = ContinuousOrdinalGenerator.generate_ordinal_scores(
        n_samples=scenario.n_samples,
        n_epochs=scenario.n_epochs,
        pattern=scenario.performance_pattern,
        max_score=10,
        seed=42
    )

    print(f"   Generated {len(df)} trial rows")
    print(f"   Score range: [{df['score'].min():.2f}, {df['score'].max():.2f}]")
    print(f"   Score mean: {df['score'].mean():.2f}")
    print(f"   Score std: {df['score'].std():.2f}")
    print(f"   Score dtype: {df['score'].dtype}")
    print(f"   Sample scores: {df['score'].head(10).tolist()}")

    # Configure manager with FIXED settings
    print("\n⚙️  Configuring manager...")
    print("   Ordinal inference: modal")
    print("   Ordinal max score: 10")
    print("   Ordinal tasks: ['ordinal', 'test_task', 'rating']")
    print("   Grouping columns: ['model', 'task']  # FIX: Includes 'task'")

    optstop_params = {
        'delta_item': 0.05,
        'delta_cap': 0.05,
        'cred_level': 0.95,
        'conservatism': 5,
        'draws': 1000,
        'tune': 1000,
        'chains': 2,
        'cores': 2
    }

    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],  # FIX: Added 'task'
        reanalysis_interval=scenario.reanalysis_interval,
        min_samples_per_grouping=5,
        ordinal_tasks=['ordinal', 'test_task', 'rating'],
        ordinal_max_score=10,
        ordinal_inference='modal',  # Fast modal inference
        shadow_mode=False
    )

    # Run start_task
    print("\n🚀 Starting task...")
    evalspec = create_mock_evalspec(task="test_task")  # FIX: Matches ordinal_tasks
    start_time = time.time()
    manager.start_task(evalspec)

    # Simulate evaluation
    print("\n▶️  Running evaluation simulation...")

    trials_completed = 0
    trials_skipped = 0
    first_skip_logged = False

    for _, row in df.iterrows():
        sample_id = row['sample_id']
        epoch = row['epoch']
        score = row['score']

        # Check if should skip
        should_skip = manager.should_skip(sample_id, epoch)

        if should_skip:
            trials_skipped += 1
            if not first_skip_logged:
                print(f"   ⏭️  First skip: {sample_id} epoch {epoch}")
                first_skip_logged = True
        else:
            # Record score
            scores_dict = create_mock_scores(score)
            manager.record_score(sample_id, epoch, scores_dict, evalspec)
            trials_completed += 1

            # Progress update every 100 trials
            if trials_completed % 100 == 0:
                total = trials_completed + trials_skipped
                pct = int(100 * total / len(df))
                print(f"   Progress: {pct}% (completed: {trials_completed}, skipped: {trials_skipped})")

    # Complete task
    print("\n✅ Completing task...")
    summary = manager.complete_task(evalspec)

    execution_time = time.time() - start_time

    # Calculate efficiency
    total_planned = len(df)
    efficiency = (trials_skipped / total_planned * 100) if total_planned > 0 else 0

    # Extract stopping details
    stopped_samples = []
    stopped_groupings = 0

    if hasattr(manager, 'stopped_samples'):
        for sample_id, reason in manager.stopped_samples.items():
            stopped_samples.append({'sample_id': sample_id, 'reason': reason})

    if hasattr(manager, 'stopped_groupings'):
        stopped_groupings = len(manager.stopped_groupings)

    n_stopped_samples = len(stopped_samples)

    # Print results
    print("\n" + "="*80)
    print(f"RESULTS: {scenario.name}")
    print("="*80)
    print(f"⏱️  Execution time: {execution_time:.2f}s")
    print(f"📊 Trials planned: {total_planned}")
    print(f"✓  Trials completed: {trials_completed}")
    print(f"⏭️  Trials skipped: {trials_skipped}")
    print(f"📈 Efficiency gain: {efficiency:.1f}%")
    print(f"🛑 Stopped samples: {n_stopped_samples}/{scenario.n_samples}")
    print(f"🎯 Stopped groupings: {stopped_groupings}")
    print("="*80)

    return {
        'scenario': scenario.name,
        'description': scenario.description,
        'pattern': scenario.performance_pattern,
        'n_samples': scenario.n_samples,
        'n_epochs': scenario.n_epochs,
        'trials_planned': total_planned,
        'trials_completed': trials_completed,
        'trials_skipped': trials_skipped,
        'efficiency_percent': efficiency,
        'stopped_samples': n_stopped_samples,
        'stopped_groupings': stopped_groupings,
        'execution_time': execution_time,
        'score_min': float(df['score'].min()),
        'score_max': float(df['score'].max()),
        'score_mean': float(df['score'].mean()),
        'score_std': float(df['score'].std()),
    }


# ============================================================================
# MAIN TEST SUITE
# ============================================================================

def main():
    """Run all continuous float ordinal test scenarios."""

    print("="*80)
    print("CONTINUOUS FLOAT ORDINAL TESTING (CORRECTED CONFIGURATION)")
    print("="*80)
    print("\nFocus: Testing ordinal inference with CONTINUOUS FLOAT scores")
    print("- Scores are continuous floats (e.g., 8.8, 9.1, 9.2, 1.1, 0.8)")
    print("- Modal inference for speed")
    print("- CORRECTED: n_groupings=2 ensures 'task' in grouping_columns")
    print("="*80)

    # Create test scenarios
    scenarios = []

    # Modal inference with different patterns
    scenarios.append(TestScenario(
        name="modal_continuous_consistent_good",
        description="Modal inference with EXTREMELY consistent high scores (9.0 ± 0.2)",
        n_samples=50,
        n_epochs=20,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        reanalysis_interval=10
    ))

    scenarios.append(TestScenario(
        name="modal_continuous_consistent_bad",
        description="Modal inference with EXTREMELY consistent low scores (1.0 ± 0.2)",
        n_samples=50,
        n_epochs=20,
        performance_pattern=PerformancePattern.CONSISTENT_BAD,
        reanalysis_interval=10
    ))

    scenarios.append(TestScenario(
        name="modal_continuous_inconsistent",
        description="Modal inference with high variance across samples",
        n_samples=50,
        n_epochs=15,
        performance_pattern=PerformancePattern.INCONSISTENT,
        reanalysis_interval=10
    ))

    scenarios.append(TestScenario(
        name="modal_continuous_improving",
        description="Modal inference with scores improving over epochs",
        n_samples=50,
        n_epochs=20,
        performance_pattern=PerformancePattern.IMPROVING,
        reanalysis_interval=10
    ))

    print(f"\n📋 Created {len(scenarios)} test scenarios")
    print(f"Estimated runtime: ~{len(scenarios) * 10} seconds\n")

    # Run all scenarios
    print("="*80)
    print("CONTINUOUS FLOAT ORDINAL TESTING")
    print("="*80)
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Focus: Continuous float scores with modal inference")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    results = []
    start_time = time.time()

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*80}")
        print(f"SCENARIO {i}/{len(scenarios)}")
        print(f"{'='*80}")

        try:
            result = run_scenario(scenario)
            result['status'] = 'success'
            results.append(result)
        except Exception as e:
            print(f"\n❌ FAILED: {scenario.name}")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

            results.append({
                'scenario': scenario.name,
                'status': 'failed',
                'error': str(e)
            })

    total_time = time.time() - start_time

    # Generate summary report
    print("\n" + "="*80)
    print("CONTINUOUS FLOAT ORDINAL TEST REPORT")
    print("="*80)
    print(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total scenarios: {len(scenarios)}")
    print("="*80)

    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') == 'failed']

    print(f"\n📊 SUMMARY")
    print("-"*80)
    print(f"✓ Successful: {len(successful)}/{len(scenarios)}")
    print(f"✗ Failed: {len(failed)}/{len(scenarios)}")

    if successful:
        print(f"\n📈 EFFICIENCY")
        print("-"*80)
        avg_eff = sum(r['efficiency_percent'] for r in successful) / len(successful)
        print(f"Average efficiency: {avg_eff:.1f}%")

        with_stopping = [r for r in successful if r['stopped_samples'] > 0]
        print(f"Scenarios with stopping: {len(with_stopping)}/{len(successful)}")

        print(f"\n⏱️  TIMING")
        print("-"*80)
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        avg_time = sum(r['execution_time'] for r in successful) / len(successful)
        print(f"Average per scenario: {avg_time:.1f}s")

        print(f"\n📊 RESULTS BY SCENARIO")
        print("-"*80)
        for r in successful:
            print(f"  {r['scenario']:40s}: {r['efficiency_percent']:5.1f}% in {r['execution_time']:6.1f}s")

    if failed:
        print(f"\n❌ FAILED SCENARIOS")
        print("-"*80)
        for r in failed:
            print(f"  {r['scenario']}: {r['error']}")

    print("\n" + "="*80)
    print("END OF REPORT")
    print("="*80)

    # Save results to JSON
    output_file = 'test_ordinal_continuous_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_scenarios': len(scenarios),
            'successful': len(successful),
            'failed': len(failed),
            'total_time': total_time,
            'results': results
        }, f, indent=2)

    print(f"\n✅ Tests complete!")
    print(f"📁 Results saved to: {os.path.abspath(output_file)}")


if __name__ == '__main__':
    main()
