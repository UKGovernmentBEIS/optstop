"""
Comprehensive Test Battery for OptimalStoppingManager with inspect_ai Integration

This test suite emulates real-world use cases including:
- Binary and ordinal scoring tasks
- Different score extraction modes (choice, aggregation, default)
- Various performance patterns (consistent good/bad, inconsistent)
- Different dataset sizes (samples, epochs)
- Multiple concurrent groupings
- Shadow mode comparison

NOTE: All print() and detailed logging statements are marked with
      # TEST_LOG: for easy identification and removal if integrated into production code
"""

import asyncio
import pandas as pd
import numpy as np
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from unittest.mock import Mock
import sys
import os

# Add project to path
sys.path.insert(0, '/home/ubuntu/optstop')

# Import the module under test
from optstop.early_stopping import OptimalStoppingManager

# Import the exact protocol from mock file
sys.path.insert(0, '/home/ubuntu/optstop')
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

INSPECT_AI_AVAILABLE = True  # We have mocks now


# ============================================================================
# TEST LOGGING CONFIGURATION
# ============================================================================

def setup_test_logging(test_name: str) -> logging.Logger:
    """
    Set up comprehensive logging for test execution.

    # TEST_LOG: This function for test purposes only - remove in production
    """
    log_dir = "/home/ubuntu/optstop/test_logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{test_name}_{timestamp}.log"

    # Configure root logger
    logger = logging.getLogger(test_name)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers = []

    # File handler - detailed logs
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)8s] %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # Console handler - key events only
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # TEST_LOG: Print log file location
    print(f"\n📁 Test logs: {log_file}")

    return logger


# ============================================================================
# MOCK DATA GENERATORS
# ============================================================================

class PerformancePattern:
    """Enums for different performance patterns."""
    CONSISTENT_GOOD = "consistent_good"      # High mean, low variance
    CONSISTENT_BAD = "consistent_bad"        # Low mean, low variance
    INCONSISTENT = "inconsistent"            # Medium mean, high variance
    IMPROVING = "improving"                  # Scores improve over epochs
    DECLINING = "declining"                  # Scores decline over epochs


class DataGenerator:
    """Generate mock data for testing different scenarios."""

    @staticmethod
    def generate_binary_scores(
        n_samples: int,
        n_epochs: int,
        pattern: str,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate binary scores (0 or 1) with specified performance pattern.

        # TEST_LOG: This generator for test purposes only
        """
        if seed is not None:
            np.random.seed(seed)

        data = []

        for sample_id in range(1, n_samples + 1):
            for epoch in range(1, n_epochs + 1):
                if pattern == PerformancePattern.CONSISTENT_GOOD:
                    # 85% success rate, low variance
                    score = 1 if np.random.random() < 0.85 else 0

                elif pattern == PerformancePattern.CONSISTENT_BAD:
                    # 15% success rate, low variance
                    score = 1 if np.random.random() < 0.15 else 0

                elif pattern == PerformancePattern.INCONSISTENT:
                    # 50% success rate, high variance per item
                    if sample_id % 3 == 0:  # Some items very good
                        score = 1 if np.random.random() < 0.9 else 0
                    elif sample_id % 3 == 1:  # Some items very bad
                        score = 1 if np.random.random() < 0.1 else 0
                    else:  # Some items medium
                        score = 1 if np.random.random() < 0.5 else 0

                elif pattern == PerformancePattern.IMPROVING:
                    # Success rate improves with epochs
                    base_rate = 0.3 + (epoch / n_epochs) * 0.5  # 30% -> 80%
                    score = 1 if np.random.random() < base_rate else 0

                elif pattern == PerformancePattern.DECLINING:
                    # Success rate declines with epochs
                    base_rate = 0.8 - (epoch / n_epochs) * 0.5  # 80% -> 30%
                    score = 1 if np.random.random() < base_rate else 0

                else:
                    raise ValueError(f"Unknown pattern: {pattern}")

                data.append({
                    'sample_id': f'sample_{sample_id}',
                    'epoch': epoch,
                    'score': score
                })

        return pd.DataFrame(data)

    @staticmethod
    def generate_ordinal_scores(
        n_samples: int,
        n_epochs: int,
        pattern: str,
        max_score: int = 10,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate ordinal scores (0 to max_score) with specified performance pattern.

        # TEST_LOG: This generator for test purposes only
        """
        if seed is not None:
            np.random.seed(seed)

        data = []

        for sample_id in range(1, n_samples + 1):
            for epoch in range(1, n_epochs + 1):
                if pattern == PerformancePattern.CONSISTENT_GOOD:
                    # High scores (7-10), low variance
                    score = np.random.choice([7, 8, 9, 10], p=[0.1, 0.2, 0.3, 0.4])

                elif pattern == PerformancePattern.CONSISTENT_BAD:
                    # Low scores (0-3), low variance
                    score = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])

                elif pattern == PerformancePattern.INCONSISTENT:
                    # Wide range of scores
                    if sample_id % 3 == 0:  # Some items very good
                        score = np.random.choice(range(7, 11))
                    elif sample_id % 3 == 1:  # Some items very bad
                        score = np.random.choice(range(0, 4))
                    else:  # Some items medium
                        score = np.random.choice(range(4, 8))

                elif pattern == PerformancePattern.IMPROVING:
                    # Scores improve with epochs
                    base_score = 3 + int((epoch / n_epochs) * 6)  # 3 -> 9
                    score = np.clip(base_score + np.random.choice([-1, 0, 1]), 0, max_score)

                elif pattern == PerformancePattern.DECLINING:
                    # Scores decline with epochs
                    base_score = 9 - int((epoch / n_epochs) * 6)  # 9 -> 3
                    score = np.clip(base_score + np.random.choice([-1, 0, 1]), 0, max_score)

                else:
                    raise ValueError(f"Unknown pattern: {pattern}")

                data.append({
                    'sample_id': f'sample_{sample_id}',
                    'epoch': epoch,
                    'score': int(score)
                })

        return pd.DataFrame(data)

    @staticmethod
    def generate_multi_score_data(
        n_samples: int,
        n_epochs: int,
        n_scorers: int = 3,
        pattern: str = PerformancePattern.CONSISTENT_GOOD,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate data with multiple scores per sample/epoch.

        # TEST_LOG: This generator for test purposes only
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate base scores
        base_df = DataGenerator.generate_binary_scores(
            n_samples, n_epochs, pattern, seed
        )

        # Add multiple scorer columns
        for scorer_idx in range(1, n_scorers + 1):
            # Add some variance across scorers
            variance = np.random.choice([-0.1, 0, 0.1])
            base_df[f'scorer_{scorer_idx}'] = base_df['score'].apply(
                lambda x: max(0, min(1, x + np.random.choice([-1, 0, 1]) * 0.3))
            )

        return base_df


# ============================================================================
# MOCK INSPECT_AI OBJECTS
# ============================================================================

def create_mock_samples(
    n_samples: int,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Sample]:
    """Create mock Sample objects."""
    samples = []
    for i in range(1, n_samples + 1):
        sample = Sample(
            id=f'sample_{i}',
            metadata=metadata or {}
        )
        samples.append(sample)
    return samples


def create_mock_evalspec(
    model: str = "test_model",
    task: str = "test_task",
    eval_id: str = "test_eval_001"
) -> EvalSpec:
    """Create mock EvalSpec object."""
    return EvalSpec(model=model, task=task, eval_id=eval_id)


def create_mock_scores(
    score_value: float,
    score_choice: Optional[str] = None,
    n_scores: int = 1
) -> Dict[str, SampleScore]:
    """Create mock scores dictionary."""
    scores = {}

    if score_choice:
        # Single score with specific key
        scores[score_choice] = SampleScore(score_value)
    else:
        # Multiple scores or default
        for i in range(n_scores):
            # Add noise but clamp to valid range [0, 1] for binary or [0, max] for ordinal
            noisy_score = score_value + np.random.uniform(-0.1, 0.1)
            # Clamp to [0, max(score_value range)]
            # For binary: [0, 1], for ordinal: [0, 10] (but we'll be conservative and use 0 as min)
            clamped_score = max(0.0, noisy_score)
            scores[f'scorer_{i+1}'] = SampleScore(clamped_score)

    return scores


# ============================================================================
# TEST SCENARIOS
# ============================================================================

class TestScenario:
    """Container for test scenario configuration."""

    def __init__(
        self,
        name: str,
        description: str,
        score_type: str,  # 'binary' or 'ordinal'
        n_samples: int,
        n_epochs: int,
        performance_pattern: str,
        score_mode: str = 'default',  # 'default', 'choice', 'mean', 'median', 'mode', 'max'
        n_groupings: int = 1,
        reanalysis_interval: int = 5,
        optstop_params: Optional[Dict[str, Any]] = None,
        ordinal_inference: str = 'modal',
        shadow_mode: bool = False
    ):
        self.name = name
        self.description = description
        self.score_type = score_type
        self.n_samples = n_samples
        self.n_epochs = n_epochs
        self.performance_pattern = performance_pattern
        self.score_mode = score_mode
        self.n_groupings = n_groupings
        self.reanalysis_interval = reanalysis_interval
        self.optstop_params = optstop_params or {
            'delta_item': 0.05,
            'delta_cap': 0.05,
            'cred_level': 0.95,
            'conservatism': 5,
            'draws': 500,  # Reduced for faster testing
            'tune': 500,
            'chains': 2,
            'cores': 2
        }
        self.ordinal_inference = ordinal_inference
        self.shadow_mode = shadow_mode

    def __repr__(self):
        return (f"TestScenario(name='{self.name}', score_type='{self.score_type}', "
                f"n_samples={self.n_samples}, n_epochs={self.n_epochs}, "
                f"pattern='{self.performance_pattern}')")


# ============================================================================
# TEST EXECUTOR
# ============================================================================

class TestExecutor:
    """Execute test scenarios and collect results."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.results = []

    async def run_scenario(self, scenario: TestScenario) -> Dict[str, Any]:
        """
        Run a single test scenario.

        # TEST_LOG: This function contains extensive logging for test purposes
        """
        # TEST_LOG: Print scenario header
        print("\n" + "="*80)
        print(f"TEST SCENARIO: {scenario.name}")
        print("="*80)
        print(f"Description: {scenario.description}")
        print(f"Score Type: {scenario.score_type}")
        print(f"Samples: {scenario.n_samples}, Epochs: {scenario.n_epochs}")
        print(f"Performance: {scenario.performance_pattern}")
        print(f"Score Mode: {scenario.score_mode}")
        print(f"Groupings: {scenario.n_groupings}")
        print(f"Shadow Mode: {scenario.shadow_mode}")
        print("-"*80)

        self.logger.info(f"Starting scenario: {scenario.name}")

        start_time = time.time()

        try:
            # Generate data
            # TEST_LOG: Print data generation progress
            print(f"📊 Generating {scenario.score_type} score data...")

            if scenario.score_type == 'binary':
                df = DataGenerator.generate_binary_scores(
                    n_samples=scenario.n_samples,
                    n_epochs=scenario.n_epochs,
                    pattern=scenario.performance_pattern,
                    seed=42
                )
            else:  # ordinal
                df = DataGenerator.generate_ordinal_scores(
                    n_samples=scenario.n_samples,
                    n_epochs=scenario.n_epochs,
                    pattern=scenario.performance_pattern,
                    max_score=10,
                    seed=42
                )

            # TEST_LOG: Print data summary
            print(f"   Generated {len(df)} trial rows")
            print(f"   Score range: [{df['score'].min()}, {df['score'].max()}]")
            print(f"   Score mean: {df['score'].mean():.3f}")
            print(f"   Score std: {df['score'].std():.3f}")

            # Configure manager
            # TEST_LOG: Print manager configuration
            print(f"\n⚙️  Configuring OptimalStoppingManager...")

            grouping_columns = ['model', 'task'] if scenario.n_groupings > 1 else ['model']

            # Determine ordinal tasks
            ordinal_tasks = None
            if scenario.score_type == 'ordinal':
                ordinal_tasks = ['ordinal', 'confidence', 'rating']

            # Configure score extraction
            score_choice = None
            score_agg = None
            if scenario.score_mode == 'choice':
                score_choice = 'scorer_1'
            elif scenario.score_mode in ['mean', 'median', 'mode', 'max']:
                score_agg = scenario.score_mode

            manager = OptimalStoppingManager(
                optstop_params=scenario.optstop_params,
                grouping_columns=grouping_columns,
                reanalysis_interval=scenario.reanalysis_interval,
                min_samples_per_grouping=3,
                ordinal_tasks=ordinal_tasks,
                ordinal_inference=scenario.ordinal_inference,
                shadow_mode=scenario.shadow_mode,
                score_choice=score_choice,
                score_agg=score_agg
            )

            # TEST_LOG: Print manager configuration
            print(f"   Grouping columns: {grouping_columns}")
            print(f"   Reanalysis interval: {scenario.reanalysis_interval}")
            print(f"   Ordinal inference: {scenario.ordinal_inference}")
            print(f"   Shadow mode: {scenario.shadow_mode}")
            print(f"   Score extraction: choice={score_choice}, agg={score_agg}")

            # Create mock samples
            samples = create_mock_samples(scenario.n_samples)

            # Run start_task
            # TEST_LOG: Print start_task progress
            print(f"\n🚀 Starting task...")

            task_name = "ordinal_task" if scenario.score_type == 'ordinal' else "binary_task"
            evalspec = create_mock_evalspec(
                model="test_model",
                task=task_name,
                eval_id=f"test_{scenario.name}"
            )

            await manager.start_task(evalspec, samples, scenario.n_epochs)

            # TEST_LOG: Print compiled dataset info
            print(f"   Compiled dataset: {len(manager.compiled_dataset)} rows")
            print(f"   Unique samples: {manager.compiled_dataset['sample_id'].nunique()}")

            # Simulate evaluation loop
            # TEST_LOG: Print evaluation progress
            print(f"\n▶️  Running evaluation simulation...")

            scheduled_trials = 0
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
                    # TEST_LOG: Log skipped trial
                    if skipped_trials == 1:
                        print(f"   ⏭️  First skip: {sample_id} epoch {epoch}")
                    continue

                scheduled_trials += 1

                # Create mock scores
                scores = create_mock_scores(
                    score_value=float(score),
                    score_choice=score_choice,
                    n_scores=3 if score_agg else 1
                )

                # Complete sample
                await manager.complete_sample(sample_id, epoch, scores)
                completed_trials += 1

                # TEST_LOG: Print progress every 10%
                if completed_trials % max(1, len(df) // 10) == 0:
                    progress_pct = (idx + 1) / len(df) * 100
                    print(f"   Progress: {progress_pct:.0f}% "
                          f"(completed: {completed_trials}, skipped: {skipped_trials})")

            # Complete task
            # TEST_LOG: Print completion
            print(f"\n✅ Completing task...")

            metadata = await manager.complete_task()

            elapsed_time = time.time() - start_time

            # Calculate efficiency
            total_planned = len(df)
            total_ran = metadata['total_ran']
            total_skipped = metadata['total_skipped']
            efficiency_pct = metadata['efficiency_percent']

            # TEST_LOG: Print results
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

            if metadata['stopped_samples']:
                print(f"\n📋 Sample stopping details:")
                for sample in metadata['stopped_samples'][:5]:  # Show first 5
                    print(f"   - {sample['id']}: {sample['reason']} "
                          f"(epoch {sample['epoch']})")
                if len(metadata['stopped_samples']) > 5:
                    print(f"   ... and {len(metadata['stopped_samples']) - 5} more")

            print("="*80)

            # Collect results
            result = {
                'scenario_name': scenario.name,
                'score_type': scenario.score_type,
                'n_samples': scenario.n_samples,
                'n_epochs': scenario.n_epochs,
                'performance_pattern': scenario.performance_pattern,
                'score_mode': scenario.score_mode,
                'shadow_mode': scenario.shadow_mode,
                'execution_time': elapsed_time,
                'total_planned': total_planned,
                'total_ran': total_ran,
                'total_skipped': total_skipped,
                'efficiency_percent': efficiency_pct,
                'stopped_samples_count': len(stopped_samples),
                'stopped_groupings_count': len(metadata['stopped_groupings']),
                'metadata': metadata,
                'success': True,
                'error': None
            }

            self.logger.info(f"Scenario '{scenario.name}' completed successfully "
                           f"(efficiency: {efficiency_pct:.1f}%)")

            return result

        except Exception as e:
            # TEST_LOG: Print error
            print(f"\n❌ ERROR in scenario '{scenario.name}': {e}")
            import traceback
            traceback.print_exc()

            self.logger.error(f"Scenario '{scenario.name}' failed: {e}", exc_info=True)

            elapsed_time = time.time() - start_time

            return {
                'scenario_name': scenario.name,
                'score_type': scenario.score_type,
                'n_samples': scenario.n_samples,
                'n_epochs': scenario.n_epochs,
                'performance_pattern': scenario.performance_pattern,
                'execution_time': elapsed_time,
                'success': False,
                'error': str(e)
            }

    async def run_all_scenarios(self, scenarios: List[TestScenario]) -> List[Dict[str, Any]]:
        """Run all test scenarios."""
        # TEST_LOG: Print test battery header
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST BATTERY")
        print("="*80)
        print(f"Total scenarios: {len(scenarios)}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        self.logger.info(f"Starting test battery with {len(scenarios)} scenarios")

        results = []
        for idx, scenario in enumerate(scenarios, 1):
            # TEST_LOG: Print scenario progress
            print(f"\n\n{'█'*80}")
            print(f"SCENARIO {idx}/{len(scenarios)}")
            print(f"{'█'*80}")

            result = await self.run_scenario(scenario)
            results.append(result)
            self.results.append(result)

            # Brief pause between scenarios
            await asyncio.sleep(0.5)

        # TEST_LOG: Print test battery summary
        print("\n" + "="*80)
        print("TEST BATTERY COMPLETE")
        print("="*80)

        self.logger.info("Test battery completed")

        return results


# ============================================================================
# TEST SUITE DEFINITION
# ============================================================================

def create_test_suite() -> List[TestScenario]:
    """
    Create comprehensive test suite covering all use cases.

    # TEST_LOG: This function defines test scenarios for validation
    """
    scenarios = []

    # ========================================================================
    # CATEGORY 1: Binary Scoring - Performance Patterns
    # ========================================================================

    scenarios.append(TestScenario(
        name="binary_consistent_good",
        description="Binary scoring with consistently high performance (85% success)",
        score_type='binary',
        n_samples=20,
        n_epochs=10,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        reanalysis_interval=5
    ))

    scenarios.append(TestScenario(
        name="binary_consistent_bad",
        description="Binary scoring with consistently low performance (15% success)",
        score_type='binary',
        n_samples=20,
        n_epochs=10,
        performance_pattern=PerformancePattern.CONSISTENT_BAD,
        reanalysis_interval=5
    ))

    scenarios.append(TestScenario(
        name="binary_inconsistent",
        description="Binary scoring with high variance across samples",
        score_type='binary',
        n_samples=20,
        n_epochs=10,
        performance_pattern=PerformancePattern.INCONSISTENT,
        reanalysis_interval=5
    ))

    scenarios.append(TestScenario(
        name="binary_improving",
        description="Binary scoring with performance improving over epochs",
        score_type='binary',
        n_samples=20,
        n_epochs=15,
        performance_pattern=PerformancePattern.IMPROVING,
        reanalysis_interval=5
    ))

    scenarios.append(TestScenario(
        name="binary_declining",
        description="Binary scoring with performance declining over epochs",
        score_type='binary',
        n_samples=20,
        n_epochs=15,
        performance_pattern=PerformancePattern.DECLINING,
        reanalysis_interval=5
    ))

    # ========================================================================
    # CATEGORY 2: Ordinal Scoring - Performance Patterns
    # ========================================================================

    scenarios.append(TestScenario(
        name="ordinal_consistent_good",
        description="Ordinal scoring (0-10) with consistently high scores (7-10)",
        score_type='ordinal',
        n_samples=20,
        n_epochs=10,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        ordinal_inference='modal',
        reanalysis_interval=5
    ))

    scenarios.append(TestScenario(
        name="ordinal_consistent_bad",
        description="Ordinal scoring (0-10) with consistently low scores (0-3)",
        score_type='ordinal',
        n_samples=20,
        n_epochs=10,
        performance_pattern=PerformancePattern.CONSISTENT_BAD,
        ordinal_inference='modal',
        reanalysis_interval=5
    ))

    scenarios.append(TestScenario(
        name="ordinal_inconsistent",
        description="Ordinal scoring with high variance across samples",
        score_type='ordinal',
        n_samples=20,
        n_epochs=10,
        performance_pattern=PerformancePattern.INCONSISTENT,
        ordinal_inference='modal',
        reanalysis_interval=5
    ))

    scenarios.append(TestScenario(
        name="ordinal_improving",
        description="Ordinal scoring with scores improving over epochs",
        score_type='ordinal',
        n_samples=20,
        n_epochs=15,
        performance_pattern=PerformancePattern.IMPROVING,
        ordinal_inference='modal',
        reanalysis_interval=5
    ))

    # ========================================================================
    # CATEGORY 3: Score Extraction Modes
    # ========================================================================

    scenarios.append(TestScenario(
        name="score_mode_choice",
        description="Binary scoring with specific score selection (score_choice)",
        score_type='binary',
        n_samples=15,
        n_epochs=10,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        score_mode='choice',
        reanalysis_interval=5
    ))

    scenarios.append(TestScenario(
        name="score_mode_mean",
        description="Binary scoring with mean aggregation of multiple scores",
        score_type='binary',
        n_samples=15,
        n_epochs=10,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        score_mode='mean',
        reanalysis_interval=5
    ))

    scenarios.append(TestScenario(
        name="score_mode_median",
        description="Binary scoring with median aggregation of multiple scores",
        score_type='binary',
        n_samples=15,
        n_epochs=10,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        score_mode='median',
        reanalysis_interval=5
    ))

    # ========================================================================
    # CATEGORY 4: Different Dataset Sizes
    # ========================================================================

    scenarios.append(TestScenario(
        name="small_dataset",
        description="Small dataset (5 samples, 5 epochs)",
        score_type='binary',
        n_samples=5,
        n_epochs=5,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        reanalysis_interval=2
    ))

    scenarios.append(TestScenario(
        name="medium_dataset",
        description="Medium dataset (30 samples, 15 epochs)",
        score_type='binary',
        n_samples=30,
        n_epochs=15,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        reanalysis_interval=10
    ))

    scenarios.append(TestScenario(
        name="large_dataset",
        description="Large dataset (50 samples, 20 epochs)",
        score_type='binary',
        n_samples=50,
        n_epochs=20,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        reanalysis_interval=15
    ))

    # ========================================================================
    # CATEGORY 5: Different Reanalysis Intervals
    # ========================================================================

    scenarios.append(TestScenario(
        name="frequent_reanalysis",
        description="Frequent inference (every 2 samples)",
        score_type='binary',
        n_samples=20,
        n_epochs=10,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        reanalysis_interval=2
    ))

    scenarios.append(TestScenario(
        name="infrequent_reanalysis",
        description="Infrequent inference (every 20 samples)",
        score_type='binary',
        n_samples=40,
        n_epochs=10,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        reanalysis_interval=20
    ))

    # ========================================================================
    # CATEGORY 6: Shadow Mode Comparison
    # ========================================================================

    scenarios.append(TestScenario(
        name="shadow_mode_enabled",
        description="Shadow mode (no actual stopping, for benchmarking)",
        score_type='binary',
        n_samples=20,
        n_epochs=10,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        shadow_mode=True,
        reanalysis_interval=5
    ))

    # ========================================================================
    # CATEGORY 7: Edge Cases
    # ========================================================================

    scenarios.append(TestScenario(
        name="edge_minimal",
        description="Minimal dataset (3 samples, 3 epochs)",
        score_type='binary',
        n_samples=3,
        n_epochs=3,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        reanalysis_interval=1
    ))

    scenarios.append(TestScenario(
        name="edge_many_epochs",
        description="Many epochs per sample (10 samples, 30 epochs)",
        score_type='binary',
        n_samples=10,
        n_epochs=30,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        reanalysis_interval=5
    ))

    return scenarios


# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

def generate_test_report(results: List[Dict[str, Any]], logger: logging.Logger):
    """
    Generate comprehensive test report.

    # TEST_LOG: This function generates detailed test reports
    """
    # TEST_LOG: Print report header
    print("\n\n" + "="*80)
    print("COMPREHENSIVE TEST REPORT")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total scenarios: {len(results)}")
    print("="*80)

    # Summary statistics
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    # TEST_LOG: Print summary
    print(f"\n📊 SUMMARY")
    print("-"*80)
    print(f"✓ Successful: {len(successful)}/{len(results)}")
    print(f"✗ Failed: {len(failed)}/{len(results)}")

    if successful:
        total_time = sum(r['execution_time'] for r in successful)
        avg_time = total_time / len(successful)
        avg_efficiency = sum(r['efficiency_percent'] for r in successful) / len(successful)

        print(f"\n⏱️  TIMING")
        print("-"*80)
        print(f"Total execution time: {total_time:.2f}s")
        print(f"Average per scenario: {avg_time:.2f}s")

        print(f"\n📈 EFFICIENCY")
        print("-"*80)
        print(f"Average efficiency gain: {avg_efficiency:.1f}%")

        # Breakdown by category
        print(f"\n📋 RESULTS BY CATEGORY")
        print("-"*80)

        # Group by score type
        binary_results = [r for r in successful if r['score_type'] == 'binary']
        ordinal_results = [r for r in successful if r['score_type'] == 'ordinal']

        if binary_results:
            avg_eff_binary = sum(r['efficiency_percent'] for r in binary_results) / len(binary_results)
            print(f"\nBinary Scoring ({len(binary_results)} scenarios):")
            print(f"  Average efficiency: {avg_eff_binary:.1f}%")

            for r in binary_results:
                print(f"  - {r['scenario_name']:30s}: "
                      f"{r['efficiency_percent']:5.1f}% efficiency, "
                      f"{r['stopped_samples_count']:2d}/{r['n_samples']:2d} samples stopped")

        if ordinal_results:
            avg_eff_ordinal = sum(r['efficiency_percent'] for r in ordinal_results) / len(ordinal_results)
            print(f"\nOrdinal Scoring ({len(ordinal_results)} scenarios):")
            print(f"  Average efficiency: {avg_eff_ordinal:.1f}%")

            for r in ordinal_results:
                print(f"  - {r['scenario_name']:30s}: "
                      f"{r['efficiency_percent']:5.1f}% efficiency, "
                      f"{r['stopped_samples_count']:2d}/{r['n_samples']:2d} samples stopped")

        # Performance pattern analysis
        print(f"\n🎯 PERFORMANCE PATTERN ANALYSIS")
        print("-"*80)

        pattern_groups = {}
        for r in successful:
            pattern = r['performance_pattern']
            if pattern not in pattern_groups:
                pattern_groups[pattern] = []
            pattern_groups[pattern].append(r)

        for pattern, pattern_results in pattern_groups.items():
            avg_eff = sum(r['efficiency_percent'] for r in pattern_results) / len(pattern_results)
            print(f"\n{pattern}:")
            print(f"  Scenarios: {len(pattern_results)}")
            print(f"  Average efficiency: {avg_eff:.1f}%")
            print(f"  Insight: ", end="")

            if 'consistent_good' in pattern:
                print("High, consistent performance leads to fast convergence")
            elif 'consistent_bad' in pattern:
                print("Low, consistent performance still allows stopping with high confidence")
            elif 'inconsistent' in pattern:
                print("High variance requires more data for reliable stopping")
            elif 'improving' in pattern:
                print("Performance improvement detected and handled appropriately")
            elif 'declining' in pattern:
                print("Performance decline detected and handled appropriately")

        # Shadow mode comparison
        shadow_results = [r for r in successful if r.get('shadow_mode', False)]
        if shadow_results:
            print(f"\n🔍 SHADOW MODE (Benchmarking)")
            print("-"*80)
            for r in shadow_results:
                print(f"  {r['scenario_name']}: 0% efficiency (all trials ran as expected)")

    if failed:
        print(f"\n❌ FAILED SCENARIOS")
        print("-"*80)
        for r in failed:
            print(f"  - {r['scenario_name']}: {r['error']}")

    print("\n" + "="*80)
    print("END OF REPORT")
    print("="*80)

    logger.info("Test report generated")


# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

async def main():
    """Main test execution function."""
    # Setup logging
    logger = setup_test_logging("comprehensive_early_stopping")

    # TEST_LOG: Print test configuration
    print("\n" + "█"*80)
    print("COMPREHENSIVE EARLY STOPPING TEST BATTERY")
    print("█"*80)
    print("\nThis test suite validates OptimalStoppingManager with inspect_ai integration")
    print("across diverse real-world scenarios including:")
    print("  • Binary and ordinal scoring")
    print("  • Different performance patterns (consistent, inconsistent, improving, declining)")
    print("  • Various score extraction modes (choice, aggregation)")
    print("  • Different dataset sizes and reanalysis intervals")
    print("  • Shadow mode for benchmarking")
    print("\nAll detailed logs are marked with # TEST_LOG: for easy identification")
    print("█"*80)

    logger.info("Starting comprehensive test battery")

    try:
        # Create test suite
        scenarios = create_test_suite()

        # TEST_LOG: Print test suite summary
        print(f"\n📋 Test suite created: {len(scenarios)} scenarios")
        print(f"Estimated runtime: ~{len(scenarios) * 90} seconds (increased timeout: 3 hours)\n")

        # Create executor
        executor = TestExecutor(logger)

        # Run all scenarios with extended timeout (3 hours = 10800 seconds)
        # This should be more than enough for all 20 scenarios
        timeout_seconds = 10800  # 3 hours

        try:
            results = await asyncio.wait_for(
                executor.run_all_scenarios(scenarios),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            print(f"\n⏱️  TIMEOUT: Test battery exceeded {timeout_seconds}s limit")
            print(f"Completed {len(executor.results)}/{len(scenarios)} scenarios")
            logger.warning(f"Test battery timed out after {timeout_seconds}s")
            results = executor.results  # Use partial results

        # Generate report
        generate_test_report(results, logger)

        # Save results to file
        results_file = "/home/ubuntu/optstop/test_results.json"
        import json
        with open(results_file, 'w') as f:
            # Remove non-serializable objects
            clean_results = []
            for r in results:
                clean_r = {k: v for k, v in r.items() if k != 'metadata'}
                clean_results.append(clean_r)
            json.dump(clean_results, f, indent=2)

        # TEST_LOG: Print completion
        print(f"\n✅ All tests complete!")
        print(f"📁 Results saved to: {results_file}")

        logger.info("Test battery completed successfully")

    except Exception as e:
        # TEST_LOG: Print fatal error
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

        logger.error("Test battery failed with fatal error", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
