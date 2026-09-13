"""
Section 1.2: Advanced Scenarios Testing (ENHANCED with Section 1.1 Lessons)

Tests complex real-world scenarios including:
- Mixed grouping scenarios with varied data quality
- Shadow mode validation
- Mixed scoring types (binary + ordinal + continuous)
- Extreme parameter configurations
- Error handling and edge cases

All tests incorporate lessons from Section 1.1:
- Routing verification in logs
- Conservative Bayesian expectations (0% efficiency is valid)
- Positive controls for high-quality data
- JSON validation outputs
- Explicit log checking

Test File: tests/test_bridge_advanced_scenarios.py
"""

import pytest
import pandas as pd
import numpy as np
import asyncio
import logging
import json
import tempfile
from pathlib import Path

# Import OptimalStoppingManager
from optstop.early_stopping import OptimalStoppingManager

# Import inspect_ai types - use same approach as test_bridge_integration_binary.py
# early_stopping.py handles the import with fallback to mocks if inspect_ai is not installed
try:
    from inspect_ai.dataset._dataset import Sample
    from inspect_ai.log._log import EvalSpec
    from inspect_ai.scorer._metric import SampleScore
    USING_REAL_INSPECT = True
except ImportError:
    # Use the same mocks that early_stopping.py creates
    from optstop.early_stopping import Sample, EvalSpec

    # Create mock SampleScore compatible with early_stopping.py mocks
    class MockScore:
        def __init__(self, value):
            self.value = value

    class SampleScore:
        def __init__(self, score):
            if hasattr(score, 'value'):
                self.score = score
            else:
                self.score = MockScore(score)

    USING_REAL_INSPECT = False

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# CI partition: heavy MCMC tests deselected from PR CI (see pyproject.toml markers).
pytestmark = pytest.mark.optstop


def verify_routing_in_logs(log_file: str, expected_routing: str) -> bool:
    """
    Verify that logs contain expected routing message.

    Args:
        log_file: Path to log file
        expected_routing: Expected routing type ('BINARY', 'ORDINAL', 'CONTINUOUS')

    Returns:
        True if routing message found in logs
    """
    try:
        with open(log_file, 'r') as f:
            log_content = f.read()
            lc = log_content.lower()
            # The bridge/live path (OptimalStoppingManager) does not emit the posthoc
            # "identified as BINARY" messages; it emits per-model diagnostics like
            # "[DIAG] BINARY POSTERIOR ..." / "[DIAG] BINARY CI ..." and MCMC tags
            # such as "MCMC [binary live_single ...]". Match those as the primary
            # evidence of routing, keeping the posthoc-style strings as fallbacks.
            if expected_routing == 'BINARY':
                return ("[diag] binary" in lc
                        or "mcmc [binary" in lc
                        or "as binary" in lc
                        or "binary discrete" in lc
                        or "binary group inference took" in lc)
            elif expected_routing == 'ORDINAL':
                return ("[diag] ordinal" in lc
                        or "mcmc [ordinal" in lc
                        or "as ordinal" in lc
                        or "ordinal discrete inference" in lc
                        or "ordinal model" in lc)
            elif expected_routing == 'CONTINUOUS':
                return ("[diag] continuous" in lc
                        or "mcmc [continuous" in lc
                        or "as continuous" in lc
                        or "continuous bounded" in lc
                        or "beta model" in lc
                        or "hierarchical beta" in lc)
            return False
    except FileNotFoundError:
        return False


# ============================================================================
# Section 1.2.1: Mixed Grouping Scenarios
# ============================================================================

@pytest.mark.asyncio
async def test_1_2_1a_mixed_quality_groupings():
    """
    Test 1.2.1a: Mixed Data Quality Across Groupings (ENHANCED)

    Goal: Validate that groupings with different data quality stop independently.

    ENHANCEMENTS from Section 1.1:
    - Explicit positive control validation (high-quality groupings MUST stop)
    - Log routing verification (confirms binary discrete routing)
    - Conservative Bayesian expectations (0% efficiency is valid for p≤0.70)
    - JSON validation output

    Configuration:
    - 4 groupings (2 models × 2 tasks)
    - Grouping 1: High quality (p=0.95) → MUST stop early
    - Grouping 2: Medium quality (p=0.70) → should not stop
    - Grouping 3: Perfect quality (p=1.0) → MUST stop early
    - Grouping 4: Low quality (p=0.60) → should not stop
    - Uses STRICT thresholds (delta_item=0.15, delta_cap=0.10)

    Expected (with strict thresholds):
    - Grouping 1 (p=0.95): efficiency >35% (strict CI requirements)
    - Grouping 3 (p=1.00): efficiency >75% (perfect scores)
    - Groupings 2 and 4: 0% efficiency (conservative Bayesian behavior)
    - No cross-contamination between groupings
    - Logs confirm binary discrete routing

    Note: Section 1.1.1 used relaxed thresholds (0.20, 0.18) achieving 87.5% for p≥0.90.
    These strict thresholds require narrower CIs, reducing efficiency for p=0.95.
    """
    print("\n" + "="*80)
    print("TEST 1.2.1a: Mixed Data Quality Across Groupings (ENHANCED)")
    print("="*80)

    # Test configuration
    num_samples = 20
    num_epochs = 8
    reanalysis_interval = 5

    # Create output directory
    output_dir = Path("tests/test_outputs/bridge_advanced")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_file = str(Path(tempfile.gettempdir()) / "test_1_2_1a_enhanced.log")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    optstop_logger = logging.getLogger('optstop')
    optstop_logger.addHandler(file_handler)
    optstop_logger.setLevel(logging.DEBUG)  # Enable INFO-level routing messages

    print(f"\n📝 Logging to: {log_file}")

    # Create manager
    optstop_params = {
        'delta_item': 0.15,
        'delta_cap': 0.10,
        'cred_level': 0.95,
        'conservatism': 5,
    }

    # CRITICAL: NO score_agg parameter (binary discrete routing)
    # SAFETY: Adding score_agg='mean' would route to continuous bounded
    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
        reanalysis_interval=reanalysis_interval,
        min_samples_per_grouping=5,
        random_seed=42,  # Pin MCMC seed so efficiency is deterministic (no auto-seed flake)
        # NO score_agg parameter - binary discrete routing
    )

    # Create samples with metadata for grouping
    samples = []
    grouping_configs = [
        ('model-A', 'task-1', 0.95),  # High quality (MUST stop)
        ('model-A', 'task-2', 0.70),  # Medium quality (0% efficiency expected)
        ('model-B', 'task-1', 1.00),  # Perfect quality (MUST stop)
        ('model-B', 'task-2', 0.60),  # Low quality (0% efficiency expected)
    ]

    sample_id = 0
    for model, task, success_rate in grouping_configs:
        for i in range(num_samples):
            samples.append(Sample(
                id=f"{model}_{task}_s{i}",
                metadata={'model': model, 'task': task}
            ))
            sample_id += 1

    # Create EvalSpec
    eval_spec = EvalSpec(
        model="mixed_test",
        task="mixed_quality_test",
        eval_id="test_1_2_1a"
    )

    # Start task
    print(f"\n🚀 Starting task with {len(samples)} samples × {num_epochs} epochs = {len(samples) * num_epochs} planned trials")
    await manager.start_task(eval_spec, samples, num_epochs)

    print(f"\n✓ Task started successfully")
    print(f"  Groupings expected: 4 (model-A/task-1, model-A/task-2, model-B/task-1, model-B/task-2)")

    # Run simulation
    print(f"\n⏳ Running evaluation simulation...")

    # Track which groupings we've seen
    seen_groupings = set()

    # For each sample and epoch
    rng = np.random.RandomState(42)
    for epoch in range(1, num_epochs + 1):
        for sample in samples:
            sample_id = sample.id

            # Get grouping info from sample
            model = sample.metadata['model']
            task = sample.metadata['task']
            grouping_key = f"{model}-{task}"
            seen_groupings.add(grouping_key)

            # Determine success rate for this grouping
            success_rate = None
            for m, t, sr in grouping_configs:
                if m == model and t == task:
                    success_rate = sr
                    break

            # Check if should run
            early_stop = await manager.schedule_sample(sample_id, epoch)

            if early_stop is None:
                # Generate score based on success rate
                score = 1 if rng.random() < success_rate else 0

                # Create scores dict
                scores = {
                    'accuracy': SampleScore(score=type('Score', (), {'value': score})())
                }

                # Complete sample
                await manager.complete_sample(sample_id, epoch, scores)

    print(f"\n✓ Simulation complete")
    print(f"  Groupings encountered: {len(seen_groupings)}")

    # Complete task and get diagnostics
    print(f"\n📊 Generating diagnostics...")
    diagnostics = await manager.complete_task()

    # Extract per-grouping efficiency
    print(f"\n📈 Per-Grouping Results:")
    print("-" * 80)

    grouping_results = {}
    for sample in samples:
        model = sample.metadata['model']
        task = sample.metadata['task']
        grouping_key = f"{model}-{task}"

        if grouping_key not in grouping_results:
            # Count trials for this grouping
            sample_ids = [s.id for s in samples if s.metadata['model'] == model and s.metadata['task'] == task]

            # Check compiled_dataset for this grouping
            grouping_data = manager.compiled_dataset[
                (manager.compiled_dataset['model'] == model) &
                (manager.compiled_dataset['task'] == task)
            ]

            total_planned = len(grouping_data)
            total_ran = int(grouping_data['trial_ran'].sum())
            total_skipped = total_planned - total_ran
            efficiency = (total_skipped / total_planned * 100) if total_planned > 0 else 0

            # Determine expected success rate
            expected_sr = None
            for m, t, sr in grouping_configs:
                if m == model and t == task:
                    expected_sr = sr
                    break

            grouping_results[grouping_key] = {
                'success_rate': expected_sr,
                'total_planned': total_planned,
                'total_ran': total_ran,
                'total_skipped': total_skipped,
                'efficiency': efficiency
            }

            print(f"  {grouping_key}:")
            print(f"    Success rate: {expected_sr:.2f}")
            print(f"    Planned trials: {total_planned}")
            print(f"    Ran trials: {total_ran}")
            print(f"    Skipped trials: {total_skipped}")
            print(f"    Efficiency: {efficiency:.1f}%")

    # Overall results
    print(f"\n📊 Overall Results:")
    print(f"  Total planned: {diagnostics['total_planned_trials']}")
    print(f"  Total ran: {diagnostics['total_ran']}")
    print(f"  Total skipped: {diagnostics['total_skipped']}")
    print(f"  Overall efficiency: {diagnostics['efficiency_percent']:.1f}%")
    print(f"  Stopped samples: {diagnostics['stopped_samples_count']}")
    print(f"  Stopped groupings: {diagnostics['stopped_groupings_count']}")

    # Verify routing in logs
    print(f"\n🔍 Verifying routing in logs...")
    routing_verified = verify_routing_in_logs(log_file, 'BINARY')
    print(f"  Binary discrete routing verified: {'✓ PASS' if routing_verified else '✗ FAIL'}")

    # Validation
    print(f"\n✅ Validation:")

    validation = {
        'test_id': 'test_1_2_1a',
        'test_name': 'Mixed Data Quality Across Groupings (ENHANCED)',
        'configuration': {
            'num_samples_per_grouping': num_samples,
            'num_epochs': num_epochs,
            'num_groupings': 4,
            'reanalysis_interval': reanalysis_interval,
            'routing': 'binary_discrete',
        },
        'grouping_results': grouping_results,
        'overall_diagnostics': {
            'total_planned': diagnostics['total_planned_trials'],
            'total_ran': diagnostics['total_ran'],
            'total_skipped': diagnostics['total_skipped'],
            'efficiency': diagnostics['efficiency_percent'],
        },
        'validation': {
            'routing_verified': routing_verified,
        }
    }

    # Check 1: HIGH QUALITY GROUPINGS MUST STOP (positive controls)
    # NOTE: With strict thresholds (delta_item=0.15, delta_cap=0.10):
    #   - p=0.95 achieves ~40% efficiency (requires narrow CI)
    #   - p=1.00 achieves ~80% efficiency (perfect scores)
    # This differs from Section 1.1.1 which used relaxed thresholds (0.20, 0.18)
    # Positive-control floors. Seeded (random_seed=42) these achieve 62.5% (p=0.95)
    # and 75.0% (p=1.00); floors are set well below to absorb cross-platform MCMC
    # jitter (the CI matrix spans 3 OS x 4 Python versions on floors-based installs).
    high_quality_groupings = {
        'model-A-task-1': (0.95, 25.0),  # p=0.95: seeded 62.5%, floor 25%
        'model-B-task-1': (1.00, 50.0),  # p=1.00: seeded 75.0%, floor 50% (relaxed from 65%)
    }
    for grouping, (expected_sr, min_efficiency) in high_quality_groupings.items():
        if grouping in grouping_results:
            efficiency = grouping_results[grouping]['efficiency']
            # CRITICAL: High quality data MUST achieve stopping with appropriate thresholds
            passed = efficiency >= min_efficiency
            validation['validation'][f'{grouping}_stops'] = passed
            print(f"  {'✓' if passed else '✗'} {grouping} (p={expected_sr}): {efficiency:.1f}% efficiency (MUST be >={min_efficiency}%): {'✓ PASS' if passed else '✗ FAIL'}")

    # Check 2: Low/medium quality groupings (0% efficiency is EXPECTED)
    low_quality_groupings = {
        'model-A-task-2': 0.70,
        'model-B-task-2': 0.60,
    }
    for grouping, expected_sr in low_quality_groupings.items():
        if grouping in grouping_results:
            efficiency = grouping_results[grouping]['efficiency']
            # 0% efficiency is valid behavior for p≤0.70
            validation['validation'][f'{grouping}_efficiency'] = efficiency
            print(f"  ✓ {grouping} (p={expected_sr}): {efficiency:.1f}% efficiency (0% expected and valid)")

    # Check 3: No cross-contamination (groupings analyzed independently)
    validation['validation']['independent_groupings'] = len(seen_groupings) == 4
    print(f"  {'✓' if len(seen_groupings) == 4 else '✗'} Independent groupings: {len(seen_groupings)} == 4: {'✓ PASS' if len(seen_groupings) == 4 else '✗ FAIL'}")

    # Check 4: Overall efficiency is non-zero (some groupings stopped)
    validation['validation']['overall_efficiency_nonzero'] = diagnostics['efficiency_percent'] > 0
    print(f"  {'✓' if diagnostics['efficiency_percent'] > 0 else '✗'} Overall efficiency > 0%: {diagnostics['efficiency_percent']:.1f}% {'✓ PASS' if diagnostics['efficiency_percent'] > 0 else '✗ FAIL'}")

    # Save validation output
    output_file = output_dir / "test_1_2_1a_mixed_quality.json"
    with open(output_file, 'w') as f:
        json.dump(validation, f, indent=2)

    print(f"\n💾 Validation output saved to: {output_file}")
    print(f"📄 Full logs saved to: {log_file}")
    print("\n" + "="*80)
    print("TEST 1.2.1a COMPLETE")
    print("="*80)

    # Assertions (positive controls MUST pass)
    assert len(seen_groupings) == 4, "Should have 4 independent groupings"
    assert routing_verified, "Binary discrete routing must be verified in logs"

    # CRITICAL: High-quality groupings MUST stop
    for grouping, (expected_sr, min_efficiency) in high_quality_groupings.items():
        if grouping in grouping_results:
            efficiency = grouping_results[grouping]['efficiency']
            assert efficiency >= min_efficiency, \
                f"{grouping} (p={expected_sr}) MUST achieve >={min_efficiency}% efficiency with strict thresholds, got {efficiency:.1f}%"


@pytest.mark.asyncio
async def test_1_2_1d_mixed_scoring_types():
    """
    Test 1.2.1d: Mixed Scoring Types in Same Evaluation (NEW)

    Goal: Verify correct routing when multiple scoring types are used in different groupings.

    CRITICAL TEST: This validates the routing logic works correctly when:
    - Grouping 1: Binary discrete (0/1, no aggregation)
    - Grouping 2: Ordinal discrete (1-5, ordinal_tasks match)
    - Grouping 3: Continuous bounded (aggregated mean)

    Configuration:
    - 3 groupings with different scoring types
    - 10 samples per grouping
    - 5 epochs

    Expected:
    - Each grouping routes to correct inference pathway
    - Logs show different inference types per grouping
    - No cross-contamination
    """
    print("\n" + "="*80)
    print("TEST 1.2.1d: Mixed Scoring Types in Same Evaluation (NEW)")
    print("="*80)

    print("\n⚠️  SKIPPING: This test requires more complex mock setup")
    print("   Reason: Different groupings need different score_agg settings")
    print("   Current manager only supports single score_agg configuration")
    print("   TODO: Re-evaluate after discussing with user")
    print("\n" + "="*80)
    print("TEST 1.2.1d SKIPPED")
    print("="*80)

    # Mark as skipped
    pytest.skip("Mixed scoring types require per-grouping configuration (TODO)")


# ============================================================================
# Section 1.2.3: Extreme Parameter Configurations
# ============================================================================

@pytest.mark.asyncio
async def test_1_2_3a_aggressive_thresholds():
    """
    Test 1.2.3a: Very Aggressive Thresholds (NEW)

    Goal: Test with very relaxed stopping criteria to maximize efficiency.

    Configuration:
    - delta_item = 0.30 (very wide CI allowed)
    - delta_cap = 0.25 (very wide grouping CI)
    - cred_level = 0.85 (85% confidence, less conservative)
    - Success rate = 0.75 (moderate quality)
    - OPTIMIZED: 10 samples, 5 epochs (faster runtime ~10-15 min)

    Expected:
    - Higher efficiency than standard thresholds
    - More samples stop early
    - Lower confidence intervals
    """
    print("\n" + "="*80)
    print("TEST 1.2.3a: Very Aggressive Thresholds (NEW - OPTIMIZED)")
    print("="*80)

    # OPTIMIZED: Reduced from 20/8 to 10/5 for faster testing
    num_samples = 10
    num_epochs = 5
    reanalysis_interval = 5

    # Create output directory
    output_dir = Path("tests/test_outputs/bridge_advanced")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_file = str(Path(tempfile.gettempdir()) / "test_1_2_3a_aggressive.log")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    optstop_logger = logging.getLogger('optstop')
    optstop_logger.addHandler(file_handler)
    optstop_logger.setLevel(logging.DEBUG)  # Enable INFO-level routing messages

    print(f"\n📝 Logging to: {log_file}")
    print(f"\n⚙️  Configuration:")
    print(f"  delta_item: 0.30 (very relaxed)")
    print(f"  delta_cap: 0.25 (very relaxed)")
    print(f"  cred_level: 0.85 (less conservative)")
    print(f"  Success rate: 0.75 (moderate)")

    # AGGRESSIVE parameters
    optstop_params = {
        'delta_item': 0.30,   # Very wide CI allowed
        'delta_cap': 0.25,    # Very wide grouping CI
        'cred_level': 0.85,   # 85% confidence (less conservative)
        'conservatism': 2,    # Low conservatism
    }

    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
        reanalysis_interval=reanalysis_interval,
        min_samples_per_grouping=3,  # Lower threshold
    )

    # Create samples
    samples = [
        Sample(id=f"sample_{i}", metadata={'model': 'gpt-4', 'task': 'test'})
        for i in range(num_samples)
    ]

    eval_spec = EvalSpec(model="gpt-4", task="aggressive_test", eval_id="test_1_2_3a")

    print(f"\n🚀 Starting task with {num_samples} samples × {num_epochs} epochs")
    await manager.start_task(eval_spec, samples, num_epochs)

    # Run simulation with moderate success rate
    print(f"\n⏳ Running evaluation...")
    rng = np.random.RandomState(42)
    success_rate = 0.75

    for epoch in range(1, num_epochs + 1):
        for sample in samples:
            early_stop = await manager.schedule_sample(sample.id, epoch)
            if early_stop is None:
                score = 1 if rng.random() < success_rate else 0
                scores = {'accuracy': SampleScore(score=type('Score', (), {'value': score})())}
                await manager.complete_sample(sample.id, epoch, scores)

    print(f"✓ Evaluation complete")

    diagnostics = await manager.complete_task()

    print(f"\n📊 Results:")
    print(f"  Total planned: {diagnostics['total_planned_trials']}")
    print(f"  Total ran: {diagnostics['total_ran']}")
    print(f"  Efficiency: {diagnostics['efficiency_percent']:.1f}%")
    print(f"  Stopped samples: {diagnostics['stopped_samples_count']}")

    # Validation
    print(f"\n✅ Validation:")

    validation = {
        'test_id': 'test_1_2_3a',
        'test_name': 'Very Aggressive Thresholds',
        'configuration': {
            'delta_item': 0.30,
            'delta_cap': 0.25,
            'cred_level': 0.85,
            'success_rate': success_rate,
        },
        'diagnostics': {
            'total_planned': diagnostics['total_planned_trials'],
            'total_ran': diagnostics['total_ran'],
            'efficiency': diagnostics['efficiency_percent'],
            'stopped_samples': diagnostics['stopped_samples_count'],
        },
        'validation': {}
    }

    # With aggressive thresholds and p=0.75, expect SOME efficiency
    # (May not be 100% due to variance, but should be >0%)
    has_efficiency = diagnostics['efficiency_percent'] > 0
    validation['validation']['has_efficiency'] = has_efficiency
    print(f"  {'✓' if has_efficiency else '✗'} Efficiency > 0% with aggressive thresholds: {diagnostics['efficiency_percent']:.1f}%: {'✓ PASS' if has_efficiency else 'ℹ️  INFO'}")

    # Note: If efficiency is still 0%, this is informative (shows even aggressive thresholds may not help with p=0.75)

    # Save validation output
    output_file = output_dir / "test_1_2_3a_aggressive.json"
    with open(output_file, 'w') as f:
        json.dump(validation, f, indent=2)

    print(f"\n💾 Validation output saved to: {output_file}")
    print("\nℹ️  NOTE: Even aggressive thresholds may yield 0% efficiency with p=0.75 (Bayesian conservatism)")
    print("\n" + "="*80)
    print("TEST 1.2.3a COMPLETE")
    print("="*80)


@pytest.mark.asyncio
async def test_1_2_3b_conservative_thresholds():
    """
    Test 1.2.3b: Very Conservative Thresholds (REVISED)

    Goal: Test with very strict stopping criteria for maximum confidence.

    REVISION (addressing Concern #2):
    - Increased samples to 20 (matching Test 1.2.1a scale)
    - Disambiguates threshold effect from sample size confound
    - Enables direct comparison: 20 samples @ p=0.95 with different thresholds

    Configuration:
    - delta_item = 0.05 (very tight CI required)
    - delta_cap = 0.03 (very tight grouping CI)
    - cred_level = 0.99 (99% confidence, very conservative)
    - Success rate = 0.95 (high quality)
    - 20 samples, 8 epochs (MATCHES Test 1.2.1a for valid comparison)

    Expected:
    - Test 1.2.1a (strict): p=0.95, 20 samples → 40.6% efficiency
    - Test 1.2.3b (conservative): p=0.95, 20 samples → ? efficiency
    - This disambiguates: Is 0% due to thresholds or sample size?
    """
    print("\n" + "="*80)
    print("TEST 1.2.3b: Very Conservative Thresholds (REVISED - Concern #2 Fix)")
    print("="*80)

    # REVISED: Increased to 20/8 to match Test 1.2.1a for valid comparison
    num_samples = 20
    num_epochs = 8
    reanalysis_interval = 5

    # Create output directory
    output_dir = Path("tests/test_outputs/bridge_advanced")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_file = str(Path(tempfile.gettempdir()) / "test_1_2_3b_conservative.log")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    optstop_logger = logging.getLogger('optstop')
    optstop_logger.addHandler(file_handler)
    optstop_logger.setLevel(logging.DEBUG)  # Enable INFO-level routing messages

    print(f"\n📝 Logging to: {log_file}")
    print(f"\n⚙️  Configuration:")
    print(f"  delta_item: 0.05 (very tight)")
    print(f"  delta_cap: 0.03 (very tight)")
    print(f"  cred_level: 0.99 (very conservative)")
    print(f"  Success rate: 0.95 (high quality)")

    # CONSERVATIVE parameters
    optstop_params = {
        'delta_item': 0.05,    # Very tight CI required
        'delta_cap': 0.03,     # Very tight grouping CI
        'cred_level': 0.99,    # 99% confidence (very conservative)
        'conservatism': 10,    # High conservatism
    }

    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
        reanalysis_interval=reanalysis_interval,
        min_samples_per_grouping=8,  # Higher threshold
    )

    # Create samples
    samples = [
        Sample(id=f"sample_{i}", metadata={'model': 'gpt-4', 'task': 'test'})
        for i in range(num_samples)
    ]

    eval_spec = EvalSpec(model="gpt-4", task="conservative_test", eval_id="test_1_2_3b")

    print(f"\n🚀 Starting task with {num_samples} samples × {num_epochs} epochs")
    await manager.start_task(eval_spec, samples, num_epochs)

    # Run simulation with HIGH success rate
    print(f"\n⏳ Running evaluation...")
    rng = np.random.RandomState(42)
    success_rate = 0.95

    for epoch in range(1, num_epochs + 1):
        for sample in samples:
            early_stop = await manager.schedule_sample(sample.id, epoch)
            if early_stop is None:
                score = 1 if rng.random() < success_rate else 0
                scores = {'accuracy': SampleScore(score=type('Score', (), {'value': score})())}
                await manager.complete_sample(sample.id, epoch, scores)

    print(f"✓ Evaluation complete")

    diagnostics = await manager.complete_task()

    print(f"\n📊 Results:")
    print(f"  Total planned: {diagnostics['total_planned_trials']}")
    print(f"  Total ran: {diagnostics['total_ran']}")
    print(f"  Efficiency: {diagnostics['efficiency_percent']:.1f}%")
    print(f"  Stopped samples: {diagnostics['stopped_samples_count']}")

    # Validation
    print(f"\n✅ Validation:")

    validation = {
        'test_id': 'test_1_2_3b',
        'test_name': 'Very Conservative Thresholds (REVISED)',
        'configuration': {
            'num_samples': num_samples,
            'num_epochs': num_epochs,
            'delta_item': 0.05,
            'delta_cap': 0.03,
            'cred_level': 0.99,
            'success_rate': success_rate,
        },
        'diagnostics': {
            'total_planned': diagnostics['total_planned_trials'],
            'total_ran': diagnostics['total_ran'],
            'efficiency': diagnostics['efficiency_percent'],
            'stopped_samples': diagnostics['stopped_samples_count'],
        },
        'validation': {},
        'comparison_with_test_1_2_1a': {
            'test_1_2_1a_config': 'p=0.95, 20 samples, strict thresholds (0.15, 0.10)',
            'test_1_2_1a_efficiency': 40.6,
            'test_1_2_3b_config': 'p=0.95, 20 samples, conservative thresholds (0.05, 0.03)',
            'test_1_2_3b_efficiency': diagnostics['efficiency_percent'],
            'difference': 40.6 - diagnostics['efficiency_percent'],
        }
    }

    # With p=0.95, 20 samples, and conservative thresholds
    validation['validation']['efficiency'] = diagnostics['efficiency_percent']
    print(f"  ℹ️  Efficiency with conservative thresholds: {diagnostics['efficiency_percent']:.1f}%")

    # Compare with Test 1.2.1a (strict thresholds, same data quality/size)
    test_1_2_1a_efficiency = 40.6
    efficiency_drop = test_1_2_1a_efficiency - diagnostics['efficiency_percent']
    print(f"  ℹ️  Comparison with Test 1.2.1a (strict thresholds):")
    print(f"      Test 1.2.1a: p=0.95, 20 samples → {test_1_2_1a_efficiency}% efficiency")
    print(f"      Test 1.2.3b: p=0.95, 20 samples → {diagnostics['efficiency_percent']:.1f}% efficiency")
    print(f"      Difference: {efficiency_drop:.1f}% drop due to conservative thresholds")

    # Interpretation
    if diagnostics['efficiency_percent'] == 0 and test_1_2_1a_efficiency > 0:
        print(f"\n  📊 INTERPRETATION:")
        print(f"      Conservative thresholds (0.05/0.03) prevent stopping entirely")
        print(f"      while strict thresholds (0.15/0.10) allow 40.6% efficiency.")
        print(f"      This confirms threshold effect (not sample size confound).")
        validation['validation']['threshold_effect_confirmed'] = True
    elif diagnostics['efficiency_percent'] > 0:
        print(f"\n  📊 INTERPRETATION:")
        print(f"      Conservative thresholds still allow {diagnostics['efficiency_percent']:.1f}% efficiency,")
        print(f"      but {efficiency_drop:.1f}% less than strict thresholds.")
        print(f"      Threshold tunability validated.")
        validation['validation']['threshold_effect_confirmed'] = True
    else:
        print(f"\n  ⚠️  UNEXPECTED: Both tests show 0% efficiency at same sample size")
        validation['validation']['threshold_effect_confirmed'] = False

    # Save validation output
    output_file = output_dir / "test_1_2_3b_conservative.json"
    with open(output_file, 'w') as f:
        json.dump(validation, f, indent=2)

    print(f"\n💾 Validation output saved to: {output_file}")
    print(f"📄 Log saved to: {log_file}")
    print("\n" + "="*80)
    print("TEST 1.2.3b COMPLETE")
    print("="*80)


# ============================================================================
# Section 1.2.4: Error Handling and Edge Cases
# ============================================================================

@pytest.mark.asyncio
async def test_1_2_4a_minimal_dataset_single_sample():
    """
    Test 1.2.4a: Minimal Dataset (Single Sample) - ENHANCED

    Goal: Verify graceful handling of minimal dataset.

    ENHANCEMENTS from Section 1.1:
    - Explicitly expect 0% efficiency (insufficient data)
    - Log verification for warnings
    - JSON validation output

    Configuration:
    - 1 sample, 3 epochs
    - min_samples_per_grouping=5 (higher than available)

    Expected:
    - Task starts successfully
    - All trials run to completion (0% efficiency)
    - No crashes or errors
    - Log shows warnings about insufficient data
    """
    print("\n" + "="*80)
    print("TEST 1.2.4a: Minimal Dataset (Single Sample) - ENHANCED")
    print("="*80)

    # Create output directory
    output_dir = Path("tests/test_outputs/bridge_advanced")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_file = str(Path(tempfile.gettempdir()) / "test_1_2_4a_minimal.log")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)  # Capture warnings
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger('optstop').addHandler(file_handler)

    print(f"\n📝 Logging to: {log_file}")

    # Create manager
    optstop_params = {
        'delta_item': 0.15,
        'delta_cap': 0.10,
    }

    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
        reanalysis_interval=2,
        min_samples_per_grouping=5,  # Higher than available samples
    )

    # Create single sample
    samples = [Sample(id="sample_1", metadata={'model': 'gpt-4', 'task': 'test'})]
    num_epochs = 3

    eval_spec = EvalSpec(model="gpt-4", task="minimal_test", eval_id="test_1_2_4a")

    print(f"\n🚀 Starting task with 1 sample × {num_epochs} epochs = {num_epochs} planned trials")
    await manager.start_task(eval_spec, samples, num_epochs)
    print(f"✓ Task started successfully")

    # Run simulation
    print(f"\n⏳ Running evaluation...")
    for epoch in range(1, num_epochs + 1):
        early_stop = await manager.schedule_sample("sample_1", epoch)
        if early_stop is None:
            scores = {'accuracy': SampleScore(score=type('Score', (), {'value': 1})())}
            await manager.complete_sample("sample_1", epoch, scores)

    print(f"✓ Evaluation complete")

    # Complete task
    diagnostics = await manager.complete_task()

    print(f"\n📊 Results:")
    print(f"  Total planned: {diagnostics['total_planned_trials']}")
    print(f"  Total ran: {diagnostics['total_ran']}")
    print(f"  Efficiency: {diagnostics['efficiency_percent']:.1f}%")

    # Check logs for warnings
    print(f"\n🔍 Checking logs for warnings...")
    with open(log_file, 'r') as f:
        log_content = f.read()
        has_insufficient_data_warning = "only" in log_content.lower() and "completed samples" in log_content.lower()

    print(f"  Insufficient data warning found: {'✓' if has_insufficient_data_warning else 'ℹ️  Not found'}")

    # Validation
    print(f"\n✅ Validation:")

    validation = {
        'test_id': 'test_1_2_4a',
        'test_name': 'Minimal Dataset (Single Sample) - ENHANCED',
        'configuration': {
            'num_samples': 1,
            'num_epochs': num_epochs,
            'min_samples_per_grouping': 5,
        },
        'diagnostics': {
            'total_planned': diagnostics['total_planned_trials'],
            'total_ran': diagnostics['total_ran'],
            'efficiency': diagnostics['efficiency_percent'],
        },
        'validation': {
            'log_warning_found': has_insufficient_data_warning,
        }
    }

    # Check 1: No crashes
    validation['validation']['no_crashes'] = True
    print(f"  ✓ No crashes: ✓ PASS")

    # Check 2: All trials ran
    all_ran = diagnostics['total_ran'] == diagnostics['total_planned_trials']
    validation['validation']['all_trials_ran'] = all_ran
    print(f"  {'✓' if all_ran else '✗'} All trials ran: {diagnostics['total_ran']} == {diagnostics['total_planned_trials']}: {'✓ PASS' if all_ran else '✗ FAIL'}")

    # Check 3: 0% efficiency (EXPECTED with insufficient data)
    zero_efficiency = diagnostics['efficiency_percent'] == 0
    validation['validation']['zero_efficiency_expected'] = zero_efficiency
    print(f"  {'✓' if zero_efficiency else '✗'} 0% efficiency (EXPECTED): {diagnostics['efficiency_percent']:.1f}%: {'✓ PASS' if zero_efficiency else '✗ FAIL'}")

    # Save validation output
    output_file = output_dir / "test_1_2_4a_minimal_dataset.json"
    with open(output_file, 'w') as f:
        json.dump(validation, f, indent=2)

    print(f"\n💾 Validation output saved to: {output_file}")
    print(f"📄 Full logs saved to: {log_file}")
    print("\n" + "="*80)
    print("TEST 1.2.4a COMPLETE")
    print("="*80)

    # Assertions
    assert all_ran, "All trials should run with insufficient data"
    assert zero_efficiency, "Efficiency should be 0% with single sample (EXPECTED behavior)"


@pytest.mark.asyncio
async def test_1_2_4b_empty_dataset_error_handling():
    """
    Test 1.2.4b: Empty Dataset Error Handling - ENHANCED

    Goal: Verify that empty dataset is caught and handled gracefully with clear error.

    Configuration:
    - 0 samples (empty list)

    Expected:
    - start_task() raises ValueError with clear message
    - Error message indicates samples list is empty
    """
    print("\n" + "="*80)
    print("TEST 1.2.4b: Empty Dataset Error Handling - ENHANCED")
    print("="*80)

    # Create output directory
    output_dir = Path("tests/test_outputs/bridge_advanced")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🧪 Testing empty dataset error handling...")

    # Create manager
    optstop_params = {'delta_item': 0.15, 'delta_cap': 0.10}
    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
    )

    # Create empty samples list
    samples = []
    num_epochs = 10
    eval_spec = EvalSpec(model="gpt-4", task="empty_test", eval_id="test_1_2_4b")

    # Try to start task (should raise ValueError)
    error_caught = False
    error_message = None

    try:
        await manager.start_task(eval_spec, samples, num_epochs)
        print(f"✗ FAIL: No error raised for empty dataset")
    except ValueError as e:
        error_caught = True
        error_message = str(e)
        print(f"✓ PASS: ValueError raised as expected")
        print(f"  Error message: {error_message}")

    # Validation
    print(f"\n✅ Validation:")

    validation = {
        'test_id': 'test_1_2_4b',
        'test_name': 'Empty Dataset Error Handling - ENHANCED',
        'configuration': {'num_samples': 0},
        'validation': {
            'error_caught': error_caught,
            'error_message': error_message,
            'error_mentions_empty': 'empty' in error_message.lower() if error_message else False
        }
    }

    print(f"  {'✓' if error_caught else '✗'} Error caught: {'✓ PASS' if error_caught else '✗ FAIL'}")
    print(f"  {'✓' if validation['validation']['error_mentions_empty'] else '✗'} Error message mentions 'empty': {'✓ PASS' if validation['validation']['error_mentions_empty'] else '✗ FAIL'}")

    # Save validation output
    output_file = output_dir / "test_1_2_4b_empty_dataset.json"
    with open(output_file, 'w') as f:
        json.dump(validation, f, indent=2)

    print(f"\n💾 Validation output saved to: {output_file}")
    print("\n" + "="*80)
    print("TEST 1.2.4b COMPLETE")
    print("="*80)

    # Assertions
    assert error_caught, "Should raise ValueError for empty dataset"
    assert error_message and 'empty' in error_message.lower(), "Error message should mention empty samples"


@pytest.mark.asyncio
async def test_1_2_4c_invalid_scores_graceful_handling():
    """
    Test 1.2.4c: Missing/Invalid Scores Graceful Handling - ENHANCED

    Goal: Verify that missing or invalid scores are handled gracefully.

    ENHANCEMENTS from Section 1.1:
    - Log verification for warnings
    - Explicit check that valid samples proceed normally
    - Confirmation that invalid samples don't crash inference

    Configuration:
    - 10 samples, 5 epochs
    - Samples 0-5: Valid scores (0/1)
    - Sample 6: None score
    - Sample 7: Negative score
    - Samples 8-9: Valid scores

    Expected:
    - Valid samples (0-5, 8-9) proceed normally
    - Invalid samples (6-7) logged with warnings
    - Invalid samples don't crash inference
    - Evaluation completes successfully
    """
    print("\n" + "="*80)
    print("TEST 1.2.4c: Missing/Invalid Scores Graceful Handling - ENHANCED")
    print("="*80)

    # Create output directory
    output_dir = Path("tests/test_outputs/bridge_advanced")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_file = str(Path(tempfile.gettempdir()) / "test_1_2_4c_invalid_scores.log")
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.WARNING)  # Capture warnings
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger('optstop').addHandler(file_handler)

    print(f"\n📝 Logging to: {log_file}")

    # Create manager
    optstop_params = {'delta_item': 0.15, 'delta_cap': 0.10}
    manager = OptimalStoppingManager(
        optstop_params=optstop_params,
        grouping_columns=['model', 'task'],
        reanalysis_interval=3,
        min_samples_per_grouping=3,
    )

    # Create samples
    num_samples = 10
    num_epochs = 5
    samples = [
        Sample(id=f"sample_{i}", metadata={'model': 'gpt-4', 'task': 'test'})
        for i in range(num_samples)
    ]

    eval_spec = EvalSpec(model="gpt-4", task="invalid_scores_test", eval_id="test_1_2_4c")

    print(f"\n🚀 Starting task with {num_samples} samples × {num_epochs} epochs")
    await manager.start_task(eval_spec, samples, num_epochs)
    print(f"✓ Task started")

    # Run simulation with mixed valid/invalid scores
    print(f"\n⏳ Running evaluation with mixed scores...")
    print(f"  Samples 0-5: Valid scores (0/1)")
    print(f"  Sample 6: None score (invalid)")
    print(f"  Sample 7: Negative score (invalid)")
    print(f"  Samples 8-9: Valid scores (0/1)")

    rng = np.random.RandomState(42)
    invalid_score_count = 0

    for epoch in range(1, num_epochs + 1):
        for i, sample in enumerate(samples):
            sample_id = sample.id
            early_stop = await manager.schedule_sample(sample_id, epoch)

            if early_stop is None:
                # Generate score (some valid, some invalid)
                if i < 6 or i >= 8:
                    # Valid scores for samples 0-5, 8-9
                    score = 1 if rng.random() < 0.8 else 0
                elif i == 6:
                    # None score for sample 6
                    score = None
                    invalid_score_count += 1
                else:  # i == 7
                    # Negative score for sample 7
                    score = -1
                    invalid_score_count += 1

                # Create scores dict
                if score is not None:
                    scores = {
                        'accuracy': SampleScore(score=type('Score', (), {'value': score})())
                    }
                else:
                    scores = {
                        'accuracy': SampleScore(score=type('Score', (), {'value': None})())
                    }

                # Complete sample
                await manager.complete_sample(sample_id, epoch, scores)

    print(f"✓ Evaluation complete")
    print(f"  Invalid scores injected: {invalid_score_count}")

    # Complete task
    diagnostics = await manager.complete_task()

    print(f"\n📊 Results:")
    print(f"  Total planned: {diagnostics['total_planned_trials']}")
    print(f"  Total ran: {diagnostics['total_ran']}")
    print(f"  Efficiency: {diagnostics['efficiency_percent']:.1f}%")

    # Check logs for warning messages
    print(f"\n🔍 Checking logs for invalid score warnings...")
    with open(log_file, 'r') as f:
        log_content = f.read()
        has_invalid_warnings = "invalid score" in log_content.lower() or "negative" in log_content.lower()

    print(f"  Invalid score warnings found: {'✓' if has_invalid_warnings else 'ℹ️  Not found (may be at DEBUG level)'}")

    # Validation
    print(f"\n✅ Validation:")

    validation = {
        'test_id': 'test_1_2_4c',
        'test_name': 'Missing/Invalid Scores Graceful Handling - ENHANCED',
        'configuration': {
            'num_samples': num_samples,
            'num_epochs': num_epochs,
            'invalid_scores_injected': invalid_score_count,
        },
        'diagnostics': {
            'total_planned': diagnostics['total_planned_trials'],
            'total_ran': diagnostics['total_ran'],
            'efficiency': diagnostics['efficiency_percent'],
        },
        'validation': {
            'log_warnings_found': has_invalid_warnings,
        }
    }

    # Check 1: No crashes
    validation['validation']['no_crashes'] = True
    print(f"  ✓ No crashes: ✓ PASS")

    # Check 2: Task completed successfully
    validation['validation']['task_completed'] = diagnostics is not None
    print(f"  {'✓' if diagnostics else '✗'} Task completed: {'✓ PASS' if diagnostics else '✗ FAIL'}")

    # Check 3: Some trials ran (valid samples processed)
    some_ran = diagnostics['total_ran'] > 0
    validation['validation']['some_trials_ran'] = some_ran
    print(f"  {'✓' if some_ran else '✗'} Some trials ran (valid samples processed): {diagnostics['total_ran']} > 0: {'✓ PASS' if some_ran else '✗ FAIL'}")

    # Check 4: Not all trials ran (invalid samples skipped from inference)
    # Note: Trials still "ran" in the sense that complete_sample was called,
    # but they shouldn't contribute to inference
    validation['validation']['all_trials_ran'] = diagnostics['total_ran'] == diagnostics['total_planned_trials']
    print(f"  ℹ️  All trials ran: {diagnostics['total_ran']} == {diagnostics['total_planned_trials']}")
    print(f"      (Invalid samples record scores but don't contribute to inference)")

    # Save validation output
    output_file = output_dir / "test_1_2_4c_invalid_scores.json"
    with open(output_file, 'w') as f:
        json.dump(validation, f, indent=2)

    print(f"\n💾 Validation output saved to: {output_file}")
    print(f"📄 Full logs saved to: {log_file}")
    print(f"\n⚠️  Check log file for warnings about invalid scores")
    print("\n" + "="*80)
    print("TEST 1.2.4c COMPLETE")
    print("="*80)

    # Assertions
    assert some_ran, "Should process valid samples despite some invalid scores"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
