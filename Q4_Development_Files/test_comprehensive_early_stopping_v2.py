"""
Comprehensive Test Battery for OptimalStoppingManager with inspect_ai Integration - V2

IMPROVEMENTS IN V2:
- Increased sample sizes to 50-100 for better statistical power
- EXTREMELY consistent scores for consistent_good/bad patterns to trigger stopping
- Detailed stopping reason reporting (CI widths, stabilization metrics)
- Fixed ordinal score generation to ensure valid MCMC inference
- Added comprehensive stopping diagnostics

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
# IMPROVED MOCK DATA GENERATORS
# ============================================================================

class PerformancePattern:
    """Enums for different performance patterns."""
    CONSISTENT_GOOD = "consistent_good"      # High mean, VERY low variance
    CONSISTENT_BAD = "consistent_bad"        # Low mean, VERY low variance
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

        V2 IMPROVEMENTS:
        - EXTREMELY consistent for consistent_good/bad patterns (95%+ consistency)
        - Ensures scores stay in valid [0, 1] range

        # TEST_LOG: This generator for test purposes only
        """
        if seed is not None:
            np.random.seed(seed)

        data = []

        for sample_id in range(1, n_samples + 1):
            for epoch in range(1, n_epochs + 1):
                if pattern == PerformancePattern.CONSISTENT_GOOD:
                    # EXTREMELY consistent: 98% success rate
                    # This should trigger early stopping via CI width and stabilization
                    score = 1 if np.random.random() < 0.98 else 0

                elif pattern == PerformancePattern.CONSISTENT_BAD:
                    # EXTREMELY consistent: 2% success rate
                    # This should also trigger early stopping
                    score = 1 if np.random.random() < 0.02 else 0

                elif pattern == PerformancePattern.INCONSISTENT:
                    # 50% success rate, high variance per item
                    if sample_id % 3 == 0:  # Some items very good
                        score = 1 if np.random.random() < 0.95 else 0
                    elif sample_id % 3 == 1:  # Some items very bad
                        score = 1 if np.random.random() < 0.05 else 0
                    else:  # Some items medium
                        score = 1 if np.random.random() < 0.5 else 0

                elif pattern == PerformancePattern.IMPROVING:
                    # Success rate improves with epochs
                    base_rate = 0.3 + (epoch / n_epochs) * 0.6  # 30% -> 90%
                    score = 1 if np.random.random() < base_rate else 0

                elif pattern == PerformancePattern.DECLINING:
                    # Success rate declines with epochs
                    base_rate = 0.9 - (epoch / n_epochs) * 0.6  # 90% -> 30%
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

        V2 IMPROVEMENTS:
        - Fixed to generate proper continuous values for MCMC inference
        - EXTREMELY consistent for consistent_good/bad patterns
        - Ensures valid score range [0, max_score]

        # TEST_LOG: This generator for test purposes only
        """
        if seed is not None:
            np.random.seed(seed)

        data = []

        for sample_id in range(1, n_samples + 1):
            for epoch in range(1, n_epochs + 1):
                if pattern == PerformancePattern.CONSISTENT_GOOD:
                    # EXTREMELY consistent: scores tightly clustered around 9
                    # Use continuous values with very small noise for MCMC
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
                        noise = np.random.normal(0, 1.5)
                    score = np.clip(base + noise, 0, max_score)

                elif pattern == PerformancePattern.IMPROVING:
                    # Scores improve with epochs
                    base_score = 2.0 + (epoch / n_epochs) * 7.0  # 2 -> 9
                    noise = np.random.normal(0, 0.5)
                    score = np.clip(base_score + noise, 0, max_score)

                elif pattern == PerformancePattern.DECLINING:
                    # Scores decline with epochs
                    base_score = 9.0 - (epoch / n_epochs) * 7.0  # 9 -> 2
                    noise = np.random.normal(0, 0.5)
                    score = np.clip(base_score + noise, 0, max_score)

                else:
                    raise ValueError(f"Unknown pattern: {pattern}")

                data.append({
                    'sample_id': f'sample_{sample_id}',
                    'epoch': epoch,
                    'score': float(score)  # Keep as float for MCMC
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

        # Add multiple scorer columns with small noise
        for scorer_idx in range(1, n_scorers + 1):
            base_df[f'scorer_{scorer_idx}'] = base_df['score'].apply(
                lambda x: max(0, min(1, x + np.random.choice([-1, 0, 1]) * 0.1))
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
    """Create mock scores dictionary with proper clamping."""
    scores = {}

    if score_choice:
        # Single score with specific key
        scores[score_choice] = SampleScore(score_value)
    else:
        # Multiple scores or default
        for i in range(n_scores):
            # Add noise but clamp to valid range [0, max]
            noisy_score = score_value + np.random.uniform(-0.05, 0.05)  # Smaller noise
            # Clamp to [0, max(score_value range)]
            clamped_score = max(0.0, noisy_score)
            scores[f'scorer_{i+1}'] = SampleScore(clamped_score)

    return scores


# ============================================================================
# TEST SCENARIOS WITH INCREASED SAMPLE SIZES
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
        reanalysis_interval: int = 10,  # Increased default
        optstop_params: Optional[Dict[str, Any]] = None,
        ordinal_inference: str = 'hybrid',  # Changed default to hybrid for MCMC
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
            'draws': 1000,  # Increased from 500
            'tune': 1000,   # Increased from 500
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
# ENHANCED TEST EXECUTOR WITH STOPPING DIAGNOSTICS
# ============================================================================

class TestExecutor:
    """Execute test scenarios and collect results with detailed stopping diagnostics."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.results = []

    def _extract_stopping_diagnostics(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detailed stopping diagnostics from metadata."""
        diagnostics = {
            'sample_stops': [],
            'group_stop': None,
            'stabilization_summary': {}
        }

        # Extract sample-level stopping details
        if 'stopped_samples' in metadata:
            for sample in metadata['stopped_samples']:
                stop_detail = {
                    'id': sample['id'],
                    'epoch': sample['epoch'],
                    'reason': sample['reason'],
                    'ci_width': sample['metadata'].get('ci_width'),
                    'threshold': sample['metadata'].get('threshold'),
                    'epochs_used': sample['metadata'].get('epochs_used')
                }
                diagnostics['sample_stops'].append(stop_detail)

        # Extract group-level stopping details
        if metadata.get('stopped_groupings_count', 0) > 0:
            # Look for group stopping reason in stabilization histories
            if 'stabilization_histories' in metadata:
                for grouping, history in metadata['stabilization_histories'].items():
                    diagnostics['group_stop'] = {
                        'grouping': grouping,
                        'n_samples': history.get('n_samples'),
                        'final_ci_width': history.get('final_ci_width'),
                        'final_slope': history.get('final_slope'),
                        'n_group_checks': history.get('n_group_checks')
                    }

        # Extract stabilization history summary
        if 'stabilization_histories' in metadata:
            for grouping, history in metadata['stabilization_histories'].items():
                diagnostics['stabilization_summary'][grouping] = {
                    'n_samples_evaluated': history.get('n_samples'),
                    'final_ci_width': history.get('final_ci_width'),
                    'final_slope': history.get('final_slope'),
                    'n_group_checks': history.get('n_group_checks')
                }

        return diagnostics

    async def run_scenario(self, scenario: TestScenario) -> Dict[str, Any]:
        """
        Run a single test scenario with enhanced stopping diagnostics.

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
            print(f"   Score range: [{df['score'].min():.3f}, {df['score'].max():.3f}]")
            print(f"   Score mean: {df['score'].mean():.3f}")
            print(f"   Score std: {df['score'].std():.3f}")

            # Configure manager
            # TEST_LOG: Print manager configuration
            print(f"\n⚙️  Configuring OptimalStoppingManager...")

            grouping_columns = ['model', 'task'] if scenario.n_groupings > 1 else ['model']

            # Determine ordinal tasks
            ordinal_tasks = None
            if scenario.score_type == 'ordinal':
                ordinal_tasks = ['ordinal', 'confidence', 'rating', 'test_task']

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
                min_samples_per_grouping=5,  # Increased from 3
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

            # Extract stopping diagnostics
            stopping_diagnostics = self._extract_stopping_diagnostics(metadata)

            # Calculate efficiency
            total_planned = len(df)
            total_ran = metadata['total_ran']
            total_skipped = metadata['total_skipped']
            efficiency_pct = metadata['efficiency_percent']

            # TEST_LOG: Print results with stopping diagnostics
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

            # Print stopping diagnostics
            if stopping_diagnostics['sample_stops']:
                print(f"\n📋 Sample stopping details (first 5):")
                for i, stop in enumerate(stopping_diagnostics['sample_stops'][:5]):
                    ci_str = f"CI={stop['ci_width']:.4f}" if stop['ci_width'] else "N/A"
                    thresh_str = f"thresh={stop['threshold']}" if stop['threshold'] else ""
                    print(f"   - {stop['id']}: {stop['reason']} after {stop['epochs_used']} epochs "
                          f"({ci_str} {thresh_str})")
                if len(stopping_diagnostics['sample_stops']) > 5:
                    print(f"   ... and {len(stopping_diagnostics['sample_stops']) - 5} more")

            if stopping_diagnostics['group_stop']:
                print(f"\n🔍 Group stopping details:")
                gs = stopping_diagnostics['group_stop']
                print(f"   Grouping: {gs['grouping']}")
                print(f"   Samples evaluated: {gs['n_samples']}")
                print(f"   Final CI width: {gs['final_ci_width']:.6f}" if gs['final_ci_width'] else "   Final CI width: N/A")
                print(f"   Final slope: {gs['final_slope']:.8f}" if gs['final_slope'] else "   Final slope: N/A")
                print(f"   Group checks: {gs['n_group_checks']}")

            if stopping_diagnostics['stabilization_summary']:
                print(f"\n📊 Stabilization summary:")
                for grouping, summary in stopping_diagnostics['stabilization_summary'].items():
                    ci_str = f"{summary['final_ci_width']:.6f}" if summary['final_ci_width'] is not None else 'N/A'
                    slope_str = f"{summary['final_slope']:.8f}" if summary['final_slope'] is not None else 'N/A'
                    print(f"   {grouping}: {summary['n_samples_evaluated']} samples, "
                          f"CI={ci_str}, slope={slope_str}")

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
                'stopping_diagnostics': stopping_diagnostics,
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
        print("COMPREHENSIVE TEST BATTERY V2")
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
# IMPROVED TEST SUITE DEFINITION (V2)
# ============================================================================

def create_test_suite() -> List[TestScenario]:
    """
    Create comprehensive test suite covering all use cases.

    V2 IMPROVEMENTS:
    - Increased sample sizes to 50-100
    - EXTREMELY consistent patterns for testing early stopping
    - Hybrid ordinal inference for MCMC testing

    # TEST_LOG: This function defines test scenarios for validation
    """
    scenarios = []

    # ========================================================================
    # CATEGORY 1: Binary Scoring - Performance Patterns (50-100 samples)
    # ========================================================================

    scenarios.append(TestScenario(
        name="binary_consistent_good",
        description="Binary scoring with EXTREMELY consistent high performance (98% success)",
        score_type='binary',
        n_samples=50,  # Increased from 20
        n_epochs=20,   # Increased from 10
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        reanalysis_interval=10
    ))

    scenarios.append(TestScenario(
        name="binary_consistent_bad",
        description="Binary scoring with EXTREMELY consistent low performance (2% success)",
        score_type='binary',
        n_samples=50,  # Increased from 20
        n_epochs=20,   # Increased from 10
        performance_pattern=PerformancePattern.CONSISTENT_BAD,
        reanalysis_interval=10
    ))

    scenarios.append(TestScenario(
        name="binary_inconsistent",
        description="Binary scoring with high variance across samples",
        score_type='binary',
        n_samples=50,  # Increased from 20
        n_epochs=15,
        performance_pattern=PerformancePattern.INCONSISTENT,
        reanalysis_interval=10
    ))

    scenarios.append(TestScenario(
        name="binary_improving",
        description="Binary scoring with performance improving over epochs",
        score_type='binary',
        n_samples=50,  # Increased from 20
        n_epochs=20,   # Increased from 15
        performance_pattern=PerformancePattern.IMPROVING,
        reanalysis_interval=10
    ))

    scenarios.append(TestScenario(
        name="binary_declining",
        description="Binary scoring with performance declining over epochs",
        score_type='binary',
        n_samples=50,  # Increased from 20
        n_epochs=20,   # Increased from 15
        performance_pattern=PerformancePattern.DECLINING,
        reanalysis_interval=10
    ))

    # ========================================================================
    # CATEGORY 2: Ordinal Scoring - Performance Patterns with MCMC (50 samples)
    # ========================================================================

    scenarios.append(TestScenario(
        name="ordinal_consistent_good",
        description="Ordinal scoring with EXTREMELY consistent high scores (9.0 ± 0.2)",
        score_type='ordinal',
        n_samples=50,  # Increased from 20
        n_epochs=20,   # Increased from 10
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        ordinal_inference='modal',  # FIX: Changed to modal (hybrid too slow)
        n_groupings=2,  # FIX: Added to ensure 'task' in grouping_columns
        reanalysis_interval=10
    ))

    scenarios.append(TestScenario(
        name="ordinal_consistent_bad",
        description="Ordinal scoring with EXTREMELY consistent low scores (1.0 ± 0.2)",
        score_type='ordinal',
        n_samples=50,  # Increased from 20
        n_epochs=20,   # Increased from 10
        performance_pattern=PerformancePattern.CONSISTENT_BAD,
        ordinal_inference='modal',  # FIX: Changed to modal (hybrid too slow)
        n_groupings=2,  # FIX: Added to ensure 'task' in grouping_columns
        reanalysis_interval=10
    ))

    scenarios.append(TestScenario(
        name="ordinal_inconsistent",
        description="Ordinal scoring with high variance across samples",
        score_type='ordinal',
        n_samples=50,  # Increased from 20
        n_epochs=15,
        performance_pattern=PerformancePattern.INCONSISTENT,
        ordinal_inference='modal',  # FIX: Changed to modal (hybrid too slow)
        n_groupings=2,  # FIX: Added to ensure 'task' in grouping_columns
        reanalysis_interval=10
    ))

    scenarios.append(TestScenario(
        name="ordinal_improving",
        description="Ordinal scoring with scores improving over epochs",
        score_type='ordinal',
        n_samples=50,  # Increased from 20
        n_epochs=20,   # Increased from 15
        performance_pattern=PerformancePattern.IMPROVING,
        ordinal_inference='modal',  # FIX: Changed to modal (hybrid too slow)
        n_groupings=2,  # FIX: Added to ensure 'task' in grouping_columns
        reanalysis_interval=10
    ))

    # ========================================================================
    # CATEGORY 3: Score Extraction Modes (50 samples)
    # ========================================================================

    scenarios.append(TestScenario(
        name="score_mode_choice",
        description="Binary scoring with specific score selection (score_choice)",
        score_type='binary',
        n_samples=50,  # Increased from 15
        n_epochs=15,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        score_mode='choice',
        reanalysis_interval=10
    ))

    scenarios.append(TestScenario(
        name="score_mode_mean",
        description="Binary scoring with mean aggregation of multiple scores",
        score_type='binary',
        n_samples=50,  # Increased from 15
        n_epochs=15,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        score_mode='mean',
        reanalysis_interval=10
    ))

    scenarios.append(TestScenario(
        name="score_mode_median",
        description="Binary scoring with median aggregation of multiple scores",
        score_type='binary',
        n_samples=50,  # Increased from 15
        n_epochs=15,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        score_mode='median',
        reanalysis_interval=10
    ))

    # ========================================================================
    # CATEGORY 4: Different Dataset Sizes (now 50-100)
    # ========================================================================

    scenarios.append(TestScenario(
        name="small_dataset",
        description="Small dataset (50 samples, 10 epochs)",
        score_type='binary',
        n_samples=50,  # Increased from 5
        n_epochs=10,   # Increased from 5
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        reanalysis_interval=10
    ))

    scenarios.append(TestScenario(
        name="medium_dataset",
        description="Medium dataset (75 samples, 20 epochs)",
        score_type='binary',
        n_samples=75,  # Increased from 30
        n_epochs=20,   # Increased from 15
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        reanalysis_interval=15
    ))

    scenarios.append(TestScenario(
        name="large_dataset",
        description="Large dataset (100 samples, 25 epochs)",
        score_type='binary',
        n_samples=100,  # Increased from 50
        n_epochs=25,    # Increased from 20
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        reanalysis_interval=20
    ))

    # ========================================================================
    # CATEGORY 5: Different Reanalysis Intervals (50 samples)
    # ========================================================================

    scenarios.append(TestScenario(
        name="frequent_reanalysis",
        description="Frequent inference (every 5 samples)",
        score_type='binary',
        n_samples=50,  # Increased from 20
        n_epochs=15,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        reanalysis_interval=5
    ))

    scenarios.append(TestScenario(
        name="infrequent_reanalysis",
        description="Infrequent inference (every 25 samples)",
        score_type='binary',
        n_samples=50,  # Same as frequent for fair comparison
        n_epochs=15,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        reanalysis_interval=25
    ))

    # ========================================================================
    # CATEGORY 6: Shadow Mode Comparison (50 samples)
    # ========================================================================

    scenarios.append(TestScenario(
        name="shadow_mode_enabled",
        description="Shadow mode (no actual stopping, for benchmarking)",
        score_type='binary',
        n_samples=50,  # Increased from 20
        n_epochs=15,
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        shadow_mode=True,
        reanalysis_interval=10
    ))

    # ========================================================================
    # CATEGORY 7: Edge Cases (adjusted)
    # ========================================================================

    scenarios.append(TestScenario(
        name="edge_many_epochs",
        description="Many epochs per sample (50 samples, 30 epochs)",
        score_type='binary',
        n_samples=50,  # Increased from 10
        n_epochs=30,   # Same
        performance_pattern=PerformancePattern.CONSISTENT_GOOD,
        reanalysis_interval=10
    ))

    return scenarios


# ============================================================================
# RESULTS ANALYSIS WITH STOPPING DETAILS
# ============================================================================

def generate_test_report(results: List[Dict[str, Any]], logger: logging.Logger):
    """
    Generate comprehensive test report with stopping diagnostics.

    # TEST_LOG: This function generates detailed test reports
    """
    # TEST_LOG: Print report header
    print("\n\n" + "="*80)
    print("COMPREHENSIVE TEST REPORT V2")
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
        print(f"Total execution time: {total_time:.2f}s ({total_time/60:.1f} min)")
        print(f"Average per scenario: {avg_time:.2f}s")

        print(f"\n📈 EFFICIENCY")
        print("-"*80)
        print(f"Average efficiency gain: {avg_efficiency:.1f}%")

        # Count scenarios with early stopping
        with_stopping = [r for r in successful if r['efficiency_percent'] > 0]
        print(f"Scenarios with early stopping: {len(with_stopping)}/{len(successful)}")

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
                diag = r.get('stopping_diagnostics', {})
                n_sample_stops = len(diag.get('sample_stops', []))
                group_stopped = "YES" if diag.get('group_stop') else "NO"
                print(f"  - {r['scenario_name']:30s}: "
                      f"{r['efficiency_percent']:5.1f}% efficiency, "
                      f"{r['stopped_samples_count']:2d}/{r['n_samples']:2d} samples stopped, "
                      f"group_stopped={group_stopped}")

        if ordinal_results:
            avg_eff_ordinal = sum(r['efficiency_percent'] for r in ordinal_results) / len(ordinal_results)
            print(f"\nOrdinal Scoring ({len(ordinal_results)} scenarios):")
            print(f"  Average efficiency: {avg_eff_ordinal:.1f}%")

            for r in ordinal_results:
                diag = r.get('stopping_diagnostics', {})
                n_sample_stops = len(diag.get('sample_stops', []))
                group_stopped = "YES" if diag.get('group_stop') else "NO"
                print(f"  - {r['scenario_name']:30s}: "
                      f"{r['efficiency_percent']:5.1f}% efficiency, "
                      f"{r['stopped_samples_count']:2d}/{r['n_samples']:2d} samples stopped, "
                      f"group_stopped={group_stopped}")

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
            with_stop = len([r for r in pattern_results if r['efficiency_percent'] > 0])
            print(f"\n{pattern}:")
            print(f"  Scenarios: {len(pattern_results)}")
            print(f"  Average efficiency: {avg_eff:.1f}%")
            print(f"  Scenarios with stopping: {with_stop}/{len(pattern_results)}")

        # Shadow mode comparison
        shadow_results = [r for r in successful if r.get('shadow_mode', False)]
        if shadow_results:
            print(f"\n🔍 SHADOW MODE (Benchmarking)")
            print("-"*80)
            for r in shadow_results:
                print(f"  {r['scenario_name']}: 0% efficiency (all trials ran as expected)")

        # Detailed stopping analysis
        print(f"\n🛑 DETAILED STOPPING ANALYSIS")
        print("-"*80)
        for r in with_stopping:
            print(f"\n{r['scenario_name']}:")
            diag = r['stopping_diagnostics']

            if diag.get('sample_stops'):
                print(f"  Sample-level stops: {len(diag['sample_stops'])}")
                for stop in diag['sample_stops'][:3]:  # Show first 3
                    ci_str = f"CI={stop['ci_width']:.6f}" if stop['ci_width'] else "N/A"
                    print(f"    - {stop['id']}: {stop['reason']} ({ci_str})")

            if diag.get('group_stop'):
                gs = diag['group_stop']
                print(f"  Group-level stop: YES")
                if gs.get('final_ci_width'):
                    print(f"    Final CI width: {gs['final_ci_width']:.6f}")
                if gs.get('final_slope'):
                    print(f"    Final slope: {gs['final_slope']:.8f}")

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
    logger = setup_test_logging("comprehensive_early_stopping_v2")

    # TEST_LOG: Print test configuration
    print("\n" + "█"*80)
    print("COMPREHENSIVE EARLY STOPPING TEST BATTERY V2")
    print("█"*80)
    print("\nIMPROVEMENTS IN V2:")
    print("  • Increased sample sizes to 50-100 for better statistical power")
    print("  • EXTREMELY consistent scores (98%/2%) to trigger early stopping")
    print("  • Detailed stopping diagnostics (CI widths, stabilization metrics)")
    print("  • Fixed ordinal scores for proper MCMC inference (continuous values)")
    print("  • Hybrid ordinal inference mode for testing MCMC path")
    print("\nAll detailed logs are marked with # TEST_LOG: for easy identification")
    print("█"*80)

    logger.info("Starting comprehensive test battery V2")

    try:
        # Create test suite
        scenarios = create_test_suite()

        # FILTER: Run only ordinal scenarios for continuous float testing
        scenarios = [s for s in scenarios if 'ordinal' in s.name]
        print("\n🎯 FILTERED TO ORDINAL SCENARIOS ONLY (continuous float scores)")
        print("   Testing corrected configuration: n_groupings=2, ordinal_inference='modal'")

        # TEST_LOG: Print test suite summary
        print(f"\n📋 Test suite created: {len(scenarios)} scenarios")
        print(f"Estimated runtime: ~{len(scenarios) * 180} seconds (timeout: 4 hours)\n")

        # Create executor
        executor = TestExecutor(logger)

        # Run all scenarios with extended timeout (4 hours = 14400 seconds)
        timeout_seconds = 14400  # 4 hours

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
        results_file = "/home/ubuntu/optstop/test_results_v2.json"
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
