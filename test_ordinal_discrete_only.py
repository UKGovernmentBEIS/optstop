"""
Focused Ordinal Testing - Discrete Integer Scores Only

This test suite focuses exclusively on ordinal tasks with:
- Discrete integer scores (1, 2, 3, 4, ..., max_score)
- No floating point values
- Mode or max aggregation for multiple scores
- Testing all three ordinal inference modes: modal, entropy, hybrid
- EXTREMELY consistent patterns to trigger stopping

Goal: Validate ordinal inference works correctly with discrete scores
before considering continuous score support.
"""

import asyncio
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
import os

# Add project to path
sys.path.insert(0, '/home/ubuntu/optstop')

# Import the module under test
from optstop.early_stopping import OptimalStoppingManager

# Import mock classes
sys.path.insert(0, '/home/ubuntu/optstop')
from mock_inspect_early_stop import EarlyStop, StoppedSample, EarlyStopping

# Mock classes
class MockScore:
    def __init__(self, value):
        self.value = value

class SampleScore:
    def __init__(self, value):
        self.score = MockScore(value)

class Sample:
    def __init__(self, id: str, metadata: dict):
        self.id = id
        self.metadata = metadata

class EvalSpec:
    def __init__(self, model: str, task: str, eval_id: str):
        self.model = model
        self.task = task
        self.eval_id = eval_id


# ============================================================================
# DISCRETE ORDINAL DATA GENERATOR
# ============================================================================

class DiscreteOrdinalGenerator:
    """Generate discrete integer ordinal scores only (no floats)."""

    @staticmethod
    def generate_ordinal_scores(
        n_samples: int,
        n_epochs: int,
        pattern: str,
        max_score: int = 10,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate DISCRETE INTEGER ordinal scores with specified pattern.

        NO FLOATS - only integers 0, 1, 2, ..., max_score
        """
        if seed is not None:
            np.random.seed(seed)

        data = []

        for sample_id in range(1, n_samples + 1):
            for epoch in range(1, n_epochs + 1):
                if pattern == "consistent_good":
                    # EXTREMELY consistent: 95% choose 9, 5% choose 10
                    score = np.random.choice([9, 10], p=[0.95, 0.05])

                elif pattern == "consistent_bad":
                    # EXTREMELY consistent: 95% choose 1, 5% choose 0
                    score = np.random.choice([0, 1], p=[0.05, 0.95])

                elif pattern == "consistent_mid":
                    # EXTREMELY consistent: 95% choose 5, 5% choose 4 or 6
                    score = np.random.choice([4, 5, 6], p=[0.025, 0.95, 0.025])

                elif pattern == "inconsistent":
                    # High variance across samples
                    if sample_id % 3 == 0:  # Some samples very good
                        score = np.random.choice([8, 9, 10])
                    elif sample_id % 3 == 1:  # Some samples very bad
                        score = np.random.choice([0, 1, 2])
                    else:  # Some samples medium
                        score = np.random.choice([4, 5, 6])

                elif pattern == "improving":
                    # Scores improve with epochs
                    # Early: 2-4, Mid: 5-7, Late: 8-10
                    if epoch < n_epochs / 3:
                        score = np.random.choice([2, 3, 4])
                    elif epoch < 2 * n_epochs / 3:
                        score = np.random.choice([5, 6, 7])
                    else:
                        score = np.random.choice([8, 9, 10])

                elif pattern == "declining":
                    # Scores decline with epochs
                    if epoch < n_epochs / 3:
                        score = np.random.choice([8, 9, 10])
                    elif epoch < 2 * n_epochs / 3:
                        score = np.random.choice([5, 6, 7])
                    else:
                        score = np.random.choice([2, 3, 4])

                else:
                    raise ValueError(f"Unknown pattern: {pattern}")

                data.append({
                    'sample_id': f'sample_{sample_id}',
                    'epoch': epoch,
                    'score': int(score)  # ENSURE INTEGER
                })

        return pd.DataFrame(data)


# ============================================================================
# MOCK HELPERS
# ============================================================================

def create_mock_samples(n_samples: int) -> List[Sample]:
    """Create mock Sample objects."""
    return [Sample(id=f'sample_{i}', metadata={}) for i in range(1, n_samples + 1)]


def create_mock_evalspec(task: str = "ordinal_task") -> EvalSpec:
    """Create mock EvalSpec object."""
    return EvalSpec(model="test_model", task=task, eval_id=f"test_ordinal")


def create_mock_scores(
    score_value: int,
    score_choice: Optional[str] = None,
    score_agg: Optional[str] = None,
    n_scores: int = 1
) -> Dict[str, SampleScore]:
    """
    Create mock scores dictionary with DISCRETE INTEGER values only.

    If n_scores > 1, creates multiple scores with small integer variation.
    """
    scores = {}

    if score_choice:
        # Single score with specific key
        scores[score_choice] = SampleScore(int(score_value))
    elif n_scores == 1:
        # Single default score
        scores['scorer_1'] = SampleScore(int(score_value))
    else:
        # Multiple scores with small INTEGER variation
        # Add ±1 variation, ensure stays in valid range [0, 10]
        for i in range(n_scores):
            variation = np.random.choice([-1, 0, 1])
            varied_score = int(np.clip(score_value + variation, 0, 10))
            scores[f'scorer_{i+1}'] = SampleScore(varied_score)

    return scores


# ============================================================================
# TEST SCENARIO
# ============================================================================

class OrdinalTestScenario:
    """Container for ordinal test scenario configuration."""

    def __init__(
        self,
        name: str,
        description: str,
        n_samples: int,
        n_epochs: int,
        performance_pattern: str,
        ordinal_inference: str,  # 'modal', 'entropy', 'hybrid'
        score_mode: str = 'default',  # 'default', 'choice', 'mode', 'max'
        n_scores: int = 1,
        reanalysis_interval: int = 10,
        ordinal_max_score: int = 10
    ):
        self.name = name
        self.description = description
        self.n_samples = n_samples
        self.n_epochs = n_epochs
        self.performance_pattern = performance_pattern
        self.ordinal_inference = ordinal_inference
        self.score_mode = score_mode
        self.n_scores = n_scores
        self.reanalysis_interval = reanalysis_interval
        self.ordinal_max_score = ordinal_max_score


# ============================================================================
# TEST EXECUTOR
# ============================================================================

class OrdinalTestExecutor:
    """Execute ordinal test scenarios."""

    def __init__(self):
        self.results = []

    async def run_scenario(self, scenario: OrdinalTestScenario) -> Dict[str, Any]:
        """Run a single ordinal test scenario."""
        print("\n" + "="*80)
        print(f"TEST: {scenario.name}")
        print("="*80)
        print(f"Description: {scenario.description}")
        print(f"Samples: {scenario.n_samples}, Epochs: {scenario.n_epochs}")
        print(f"Pattern: {scenario.performance_pattern}")
        print(f"Inference mode: {scenario.ordinal_inference}")
        print(f"Score mode: {scenario.score_mode} (n_scores={scenario.n_scores})")
        print("-"*80)

        start_time = time.time()

        try:
            # Generate discrete integer ordinal scores
            print(f"📊 Generating discrete ordinal scores...")
            df = DiscreteOrdinalGenerator.generate_ordinal_scores(
                n_samples=scenario.n_samples,
                n_epochs=scenario.n_epochs,
                pattern=scenario.performance_pattern,
                max_score=scenario.ordinal_max_score,
                seed=42
            )

            print(f"   Generated {len(df)} trial rows")
            print(f"   Score range: [{df['score'].min()}, {df['score'].max()}]")
            print(f"   Score mean: {df['score'].mean():.2f}")
            print(f"   Score std: {df['score'].std():.2f}")
            print(f"   Score dtype: {df['score'].dtype}")  # Should be int
            print(f"   Unique scores: {sorted(df['score'].unique())}")

            # Configure manager
            print(f"\n⚙️  Configuring manager...")

            # Configure score extraction
            score_choice = None
            score_agg = None
            if scenario.score_mode == 'choice':
                score_choice = 'scorer_1'
            elif scenario.score_mode in ['mode', 'max', 'mean', 'median']:
                score_agg = scenario.score_mode

            manager = OptimalStoppingManager(
                optstop_params={
                    'delta_item': 0.05,
                    'delta_cap': 0.05,
                    'cred_level': 0.95,
                    'conservatism': 5,
                    'draws': 1000,
                    'tune': 1000,
                    'chains': 2,
                    'cores': 2
                },
                grouping_columns=['model', 'task'],  # FIX: Added 'task' to enable ordinal detection
                reanalysis_interval=scenario.reanalysis_interval,
                min_samples_per_grouping=5,
                ordinal_tasks=['ordinal', 'test_task', 'rating'],  # Match task names
                ordinal_max_score=scenario.ordinal_max_score,
                ordinal_inference=scenario.ordinal_inference,
                shadow_mode=False,
                score_choice=score_choice,
                score_agg=score_agg
            )

            print(f"   Ordinal inference: {scenario.ordinal_inference}")
            print(f"   Ordinal max score: {scenario.ordinal_max_score}")
            print(f"   Ordinal tasks: {manager.ordinal_tasks}")

            # Create samples and start task
            samples = create_mock_samples(scenario.n_samples)
            evalspec = create_mock_evalspec(task="test_task")  # FIX: Changed from "ordinal_task" to match ordinal_tasks list

            print(f"\n🚀 Starting task...")
            await manager.start_task(evalspec, samples, scenario.n_epochs)

            # Run evaluation simulation
            print(f"\n▶️  Running evaluation simulation...")
            completed_trials = 0
            skipped_trials = 0
            stopped_samples = set()

            for idx, row in df.iterrows():
                sample_id = row['sample_id']
                epoch = row['epoch']
                score = row['score']

                # Check if should schedule
                early_stop = await manager.schedule_sample(sample_id, epoch)

                if early_stop is not None:
                    skipped_trials += 1
                    stopped_samples.add(sample_id)
                    if skipped_trials == 1:
                        print(f"   ⏭️  First skip: {sample_id} epoch {epoch}")
                    continue

                # Create mock scores (DISCRETE INTEGERS)
                scores = create_mock_scores(
                    score_value=int(score),
                    score_choice=score_choice,
                    score_agg=score_agg,
                    n_scores=scenario.n_scores
                )

                # Complete sample
                await manager.complete_sample(sample_id, epoch, scores)
                completed_trials += 1

                # Progress updates
                if completed_trials % max(1, len(df) // 10) == 0:
                    progress_pct = (idx + 1) / len(df) * 100
                    print(f"   Progress: {progress_pct:.0f}% "
                          f"(completed: {completed_trials}, skipped: {skipped_trials})")

            # Complete task
            print(f"\n✅ Completing task...")
            metadata = await manager.complete_task()

            elapsed_time = time.time() - start_time

            # Calculate efficiency
            total_planned = len(df)
            total_ran = metadata['total_ran']
            total_skipped = metadata['total_skipped']
            efficiency_pct = metadata['efficiency_percent']

            # Print results
            print(f"\n" + "="*80)
            print(f"RESULTS: {scenario.name}")
            print("="*80)
            print(f"⏱️  Execution time: {elapsed_time:.2f}s")
            print(f"📊 Trials planned: {total_planned}")
            print(f"✓  Trials completed: {total_ran}")
            print(f"⏭️  Trials skipped: {total_skipped}")
            print(f"📈 Efficiency gain: {efficiency_pct:.1f}%")
            print(f"🛑 Stopped samples: {len(stopped_samples)}/{scenario.n_samples}")
            print(f"🎯 Stopped groupings: {len(metadata['stopped_groupings'])}")

            # Extract stopping details
            if metadata.get('stabilization_histories'):
                print(f"\n🔍 Stopping details:")
                for grouping, history in metadata['stabilization_histories'].items():
                    n_samples = history.get('n_samples_evaluated', 0)
                    ci_width = history.get('final_ci_width')
                    slope = history.get('final_slope')
                    n_checks = history.get('n_group_checks', 0)
                    print(f"   {grouping}:")
                    print(f"      Samples evaluated: {n_samples}")
                    if ci_width is not None:
                        print(f"      Final CI width: {ci_width:.6f}")
                    if slope is not None:
                        print(f"      Final slope: {slope:.8f}")
                    print(f"      Group checks: {n_checks}")

            if metadata.get('stopped_samples'):
                print(f"\n📋 Stopped samples (first 5):")
                for sample in metadata['stopped_samples'][:5]:
                    print(f"   - {sample['id']}: {sample['reason']}")

            print("="*80)

            result = {
                'scenario_name': scenario.name,
                'n_samples': scenario.n_samples,
                'n_epochs': scenario.n_epochs,
                'pattern': scenario.performance_pattern,
                'inference_mode': scenario.ordinal_inference,
                'score_mode': scenario.score_mode,
                'n_scores': scenario.n_scores,
                'execution_time': elapsed_time,
                'efficiency_percent': efficiency_pct,
                'stopped_samples_count': len(stopped_samples),
                'stopped_groupings_count': len(metadata['stopped_groupings']),
                'success': True,
                'error': None
            }

            return result

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

            elapsed_time = time.time() - start_time

            return {
                'scenario_name': scenario.name,
                'execution_time': elapsed_time,
                'success': False,
                'error': str(e)
            }

    async def run_all_scenarios(self, scenarios: List[OrdinalTestScenario]) -> List[Dict[str, Any]]:
        """Run all ordinal test scenarios."""
        print("\n" + "="*80)
        print("ORDINAL DISCRETE INTEGER TESTING")
        print("="*80)
        print(f"Total scenarios: {len(scenarios)}")
        print(f"Focus: Discrete integer scores with mode/max aggregation")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        results = []
        for idx, scenario in enumerate(scenarios, 1):
            print(f"\n\n{'█'*80}")
            print(f"SCENARIO {idx}/{len(scenarios)}")
            print(f"{'█'*80}")

            result = await self.run_scenario(scenario)
            results.append(result)
            self.results.append(result)

            await asyncio.sleep(0.5)

        return results


# ============================================================================
# TEST SUITE DEFINITION
# ============================================================================

def create_ordinal_test_suite() -> List[OrdinalTestScenario]:
    """
    Create focused ordinal test suite with discrete integer scores.

    Tests all 3 inference modes (modal, entropy, hybrid) with:
    - Consistent patterns (should trigger stopping)
    - Inconsistent patterns (should not trigger stopping)
    - Different score aggregation modes (mode, max)
    """
    scenarios = []

    # ========================================================================
    # MODAL INFERENCE MODE (Fast, no MCMC)
    # ========================================================================

    scenarios.append(OrdinalTestScenario(
        name="modal_consistent_good",
        description="Modal inference with EXTREMELY consistent high scores (95% choose 9)",
        n_samples=50,
        n_epochs=20,
        performance_pattern="consistent_good",
        ordinal_inference='modal',
        score_mode='default',
        n_scores=1
    ))

    scenarios.append(OrdinalTestScenario(
        name="modal_consistent_bad",
        description="Modal inference with EXTREMELY consistent low scores (95% choose 1)",
        n_samples=50,
        n_epochs=20,
        performance_pattern="consistent_bad",
        ordinal_inference='modal',
        score_mode='default',
        n_scores=1
    ))

    scenarios.append(OrdinalTestScenario(
        name="modal_consistent_mid",
        description="Modal inference with EXTREMELY consistent mid scores (95% choose 5)",
        n_samples=50,
        n_epochs=20,
        performance_pattern="consistent_mid",
        ordinal_inference='modal',
        score_mode='default',
        n_scores=1
    ))

    scenarios.append(OrdinalTestScenario(
        name="modal_inconsistent",
        description="Modal inference with high variance across samples",
        n_samples=50,
        n_epochs=15,
        performance_pattern="inconsistent",
        ordinal_inference='modal',
        score_mode='default',
        n_scores=1
    ))

    # ========================================================================
    # ENTROPY INFERENCE MODE (Bootstrap-based)
    # ========================================================================

    scenarios.append(OrdinalTestScenario(
        name="entropy_consistent_good",
        description="Entropy inference with EXTREMELY consistent high scores",
        n_samples=50,
        n_epochs=20,
        performance_pattern="consistent_good",
        ordinal_inference='entropy',
        score_mode='default',
        n_scores=1
    ))

    scenarios.append(OrdinalTestScenario(
        name="entropy_inconsistent",
        description="Entropy inference with high variance (high entropy)",
        n_samples=50,
        n_epochs=15,
        performance_pattern="inconsistent",
        ordinal_inference='entropy',
        score_mode='default',
        n_scores=1
    ))

    # ========================================================================
    # HYBRID INFERENCE MODE (Adaptive: entropy check → modal or MCMC)
    # ========================================================================

    scenarios.append(OrdinalTestScenario(
        name="hybrid_consistent_good",
        description="Hybrid inference with EXTREMELY consistent high scores (low entropy → modal)",
        n_samples=50,
        n_epochs=20,
        performance_pattern="consistent_good",
        ordinal_inference='hybrid',
        score_mode='default',
        n_scores=1
    ))

    scenarios.append(OrdinalTestScenario(
        name="hybrid_inconsistent",
        description="Hybrid inference with high variance (high entropy → MCMC)",
        n_samples=50,
        n_epochs=15,
        performance_pattern="inconsistent",
        ordinal_inference='hybrid',
        score_mode='default',
        n_scores=1
    ))

    # ========================================================================
    # SCORE AGGREGATION MODES (Multiple scores with mode/max)
    # ========================================================================

    scenarios.append(OrdinalTestScenario(
        name="modal_agg_mode",
        description="Modal inference with mode aggregation (3 scores)",
        n_samples=50,
        n_epochs=15,
        performance_pattern="consistent_good",
        ordinal_inference='modal',
        score_mode='mode',
        n_scores=3
    ))

    scenarios.append(OrdinalTestScenario(
        name="modal_agg_max",
        description="Modal inference with max aggregation (3 scores)",
        n_samples=50,
        n_epochs=15,
        performance_pattern="consistent_good",
        ordinal_inference='modal',
        score_mode='max',
        n_scores=3
    ))

    # ========================================================================
    # PERFORMANCE PATTERNS (Improving/Declining)
    # ========================================================================

    scenarios.append(OrdinalTestScenario(
        name="modal_improving",
        description="Modal inference with improving scores over epochs",
        n_samples=50,
        n_epochs=20,
        performance_pattern="improving",
        ordinal_inference='modal',
        score_mode='default',
        n_scores=1
    ))

    scenarios.append(OrdinalTestScenario(
        name="entropy_declining",
        description="Entropy inference with declining scores over epochs",
        n_samples=50,
        n_epochs=20,
        performance_pattern="declining",
        ordinal_inference='entropy',
        score_mode='default',
        n_scores=1
    ))

    return scenarios


# ============================================================================
# RESULTS REPORTING
# ============================================================================

def generate_report(results: List[Dict[str, Any]]):
    """Generate comprehensive test report."""
    print("\n\n" + "="*80)
    print("ORDINAL DISCRETE INTEGER TEST REPORT")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total scenarios: {len(results)}")
    print("="*80)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"\n📊 SUMMARY")
    print("-"*80)
    print(f"✓ Successful: {len(successful)}/{len(results)}")
    print(f"✗ Failed: {len(failed)}/{len(results)}")

    if successful:
        avg_efficiency = sum(r['efficiency_percent'] for r in successful) / len(successful)
        total_time = sum(r['execution_time'] for r in successful)

        print(f"\n📈 EFFICIENCY")
        print("-"*80)
        print(f"Average efficiency: {avg_efficiency:.1f}%")
        print(f"Scenarios with stopping: {len([r for r in successful if r['efficiency_percent'] > 0])}/{len(successful)}")

        print(f"\n⏱️  TIMING")
        print("-"*80)
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Average per scenario: {total_time/len(successful):.1f}s")

        # Breakdown by inference mode
        print(f"\n🔬 RESULTS BY INFERENCE MODE")
        print("-"*80)

        for mode in ['modal', 'entropy', 'hybrid']:
            mode_results = [r for r in successful if r['inference_mode'] == mode]
            if mode_results:
                avg_eff = sum(r['efficiency_percent'] for r in mode_results) / len(mode_results)
                avg_time = sum(r['execution_time'] for r in mode_results) / len(mode_results)
                with_stopping = len([r for r in mode_results if r['efficiency_percent'] > 0])

                print(f"\n{mode.upper()} mode ({len(mode_results)} scenarios):")
                print(f"  Average efficiency: {avg_eff:.1f}%")
                print(f"  Scenarios with stopping: {with_stopping}/{len(mode_results)}")
                print(f"  Average time: {avg_time:.1f}s")

                for r in mode_results:
                    print(f"    {r['scenario_name']:30s}: {r['efficiency_percent']:5.1f}% in {r['execution_time']:6.1f}s")

    if failed:
        print(f"\n❌ FAILED SCENARIOS")
        print("-"*80)
        for r in failed:
            print(f"  - {r['scenario_name']}: {r['error']}")

    print("\n" + "="*80)
    print("END OF REPORT")
    print("="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main test execution."""
    print("\n" + "█"*80)
    print("ORDINAL DISCRETE INTEGER TESTING")
    print("█"*80)
    print("\nFocus: Testing ordinal inference with DISCRETE INTEGER scores only")
    print("- No floating point values")
    print("- Mode/max aggregation for multiple scores")
    print("- All three inference modes: modal, entropy, hybrid")
    print("- EXTREMELY consistent patterns to trigger stopping")
    print("█"*80)

    try:
        # Create test suite
        scenarios = create_ordinal_test_suite()
        print(f"\n📋 Created {len(scenarios)} test scenarios")
        print(f"Estimated runtime: ~{len(scenarios) * 30} seconds\n")

        # Create executor
        executor = OrdinalTestExecutor()

        # Run all scenarios with timeout (30 min)
        timeout_seconds = 1800

        try:
            results = await asyncio.wait_for(
                executor.run_all_scenarios(scenarios),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            print(f"\n⏱️  TIMEOUT after {timeout_seconds}s")
            print(f"Completed {len(executor.results)}/{len(scenarios)} scenarios")
            results = executor.results

        # Generate report
        generate_report(results)

        # Save results
        results_file = "/home/ubuntu/optstop/test_ordinal_discrete_results.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Tests complete!")
        print(f"📁 Results saved to: {results_file}")

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
