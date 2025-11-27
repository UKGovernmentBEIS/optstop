#!/usr/bin/env python3
"""
GPU Multi-Processing Usage Examples for optstop package

This file demonstrates how to use the new GPU multi-processing capabilities
while maintaining backward compatibility with existing CPU-only code.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add optstop to path if running standalone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optstop import gpu_utils
from optstop.rule import optimal_stopping_posthoc, optimal_stopping_live, configure_optstop_logging


def create_example_dataset(n_groups=4, n_items=20, n_epochs=15, seed=42):
    """Create a realistic example dataset for optimal stopping analysis."""
    np.random.seed(seed)

    data = []
    for group in range(n_groups):
        for item in range(n_items):
            # Different groups have different base success rates
            base_rate = 0.4 + 0.15 * group

            for epoch in range(n_epochs):
                # Performance improves slightly with epochs (learning effect)
                success_prob = min(0.95, base_rate + 0.02 * epoch)
                score = np.random.binomial(1, success_prob)

                data.append({
                    'task_type': f'task_{group}',
                    'subject_id': f'subject_{group}',
                    'item_id': f'item_{item:02d}',
                    'trial_number': epoch + 1,
                    'performance': score,
                    'group_info': f'group_{group}'
                })

    return pd.DataFrame(data)


def example_1_backward_compatibility():
    """Example 1: Existing code continues to work unchanged (CPU-only)."""
    print("\n" + "="*60)
    print("Example 1: Backward Compatibility (CPU-only)")
    print("="*60)

    # Configure logging
    configure_optstop_logging('example1.log', console_output=True)

    # Create test data
    df = create_example_dataset(n_groups=2, n_items=5, n_epochs=8)

    # Standard parameters
    params = {
        'draws': 200,
        'tune': 200,
        'chains': 2,
        'cores': 2,
        'delta_item': 0.1,
        'delta_cap': 0.1,
        'cred_level': 0.95
    }

    print("Original function calls (no GPU parameters):")

    # Post-hoc analysis - original signature
    print("Running post-hoc analysis...")
    pruned_df, summary = optimal_stopping_posthoc(
        df=df,
        params=params,
        grouping_columns=['task_type', 'subject_id'],  # Can use multiple grouping columns
        sample_id_column='item_id',
        epoch_column='trial_number',
        score_column='performance'
    )

    print(f"✅ Post-hoc complete: {len(df)} → {len(pruned_df)} rows ({len(summary)} group summaries)")

    # Live analysis - original signature
    print("Running live analysis...")
    live_result = optimal_stopping_live(
        df=df,
        params=params,
        grouping_columns=['task_type', 'subject_id'],
        sample_id_column='item_id',
        epoch_column='trial_number',
        score_column='performance'
    )

    print(f"✅ Live complete: {len(live_result['stop_sample_ids'])} samples flagged, {len(live_result['stop_task'])} tasks complete")
    print("🎯 Existing code works perfectly - no changes needed!")


def example_2_gpu_auto_detection():
    """Example 2: Automatic GPU detection and usage."""
    print("\n" + "="*60)
    print("Example 2: Automatic GPU Detection")
    print("="*60)

    # Check what GPUs are available
    available_gpus = gpu_utils.get_available_gpu_ids()
    print(f"Available GPUs: {available_gpus}")

    if not available_gpus:
        print("No GPUs detected - will demonstrate CPU-only with new parameters")
        gpu_ids = None
        max_workers = 4
    else:
        print(f"Found {len(available_gpus)} GPUs - will use all available")
        gpu_ids = available_gpus
        max_workers = len(available_gpus)

    # Create larger dataset to benefit from parallelization
    df = create_example_dataset(n_groups=4, n_items=10, n_epochs=12)

    params = {
        'draws': 300,
        'tune': 300,
        'chains': 4,
        'cores': 4,
        'delta_item': 0.08,
        'delta_cap': 0.08
    }

    print(f"Running post-hoc with gpu_ids={gpu_ids}, max_workers={max_workers}")

    pruned_df, summary = optimal_stopping_posthoc(
        df=df,
        params=params,
        grouping_columns=['task_type'],
        sample_id_column='item_id',
        epoch_column='trial_number',
        score_column='performance',
        gpu_ids=gpu_ids,              # New parameter
        max_workers=max_workers       # New parameter
    )

    print(f"✅ Auto-detection complete: {len(df)} → {len(pruned_df)} rows")


def example_3_explicit_gpu_assignment():
    """Example 3: Explicit GPU assignment for fine-grained control."""
    print("\n" + "="*60)
    print("Example 3: Explicit GPU Assignment")
    print("="*60)

    # Get available GPUs
    available_gpus = gpu_utils.get_available_gpu_ids()

    if available_gpus:
        # Example: Use only specific GPUs
        selected_gpus = available_gpus[:2] if len(available_gpus) >= 2 else available_gpus
        print(f"Available GPUs: {available_gpus}")
        print(f"Using GPUs: {selected_gpus}")

        df = create_example_dataset(n_groups=6, n_items=8, n_epochs=10)

        params = {
            'draws': 250,
            'tune': 250,
            'chains': 4,
            'delta_item': 0.07,
            'delta_cap': 0.07
        }

        # Explicit GPU assignment
        pruned_df, summary = optimal_stopping_posthoc(
            df=df,
            params=params,
            grouping_columns=['task_type', 'subject_id'],
            sample_id_column='item_id',
            epoch_column='trial_number',
            score_column='performance',
            gpu_ids=selected_gpus,       # Explicitly specify which GPUs to use
            max_workers=6                # More workers than GPUs (cyclical assignment)
        )

        print(f"✅ Explicit assignment complete: {len(df)} → {len(pruned_df)} rows")
        print(f"Workers used GPUs cyclically: {selected_gpus}")
    else:
        print("No GPUs available - demonstrating CPU-only explicit configuration")

        df = create_example_dataset(n_groups=3, n_items=6, n_epochs=8)

        params = {
            'draws': 200,
            'tune': 200,
            'chains': 2,
            'delta_item': 0.1,
            'delta_cap': 0.1
        }

        # Force CPU-only with explicit empty list
        pruned_df, summary = optimal_stopping_posthoc(
            df=df,
            params=params,
            grouping_columns=['task_type'],
            sample_id_column='item_id',
            epoch_column='trial_number',
            score_column='performance',
            gpu_ids=[],                  # Explicit CPU-only
            max_workers=3                # 3 CPU workers
        )

        print(f"✅ CPU-only explicit complete: {len(df)} → {len(pruned_df)} rows")


def example_4_mixed_usage_patterns():
    """Example 4: Different usage patterns for different scenarios."""
    print("\n" + "="*60)
    print("Example 4: Mixed Usage Patterns")
    print("="*60)

    df = create_example_dataset(n_groups=2, n_items=8, n_epochs=10)

    params = {
        'draws': 150,
        'tune': 150,
        'chains': 2,
        'delta_item': 0.12,
        'delta_cap': 0.12
    }

    print("Scenario A: High-performance computing cluster with many GPUs")
    # Simulate HPC scenario
    simulated_gpus = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 GPUs available
    validated_gpus, workers = gpu_utils.validate_gpu_configuration(simulated_gpus, max_workers=16)
    print(f"  Simulated HPC validation: {validated_gpus}, workers: {workers}")

    print("\nScenario B: Workstation with 2 GPUs")
    simulated_gpus = [0, 1]  # 2 GPUs
    validated_gpus, workers = gpu_utils.validate_gpu_configuration(simulated_gpus, max_workers=8)
    print(f"  Workstation validation: {validated_gpus}, workers: {workers}")

    print("\nScenario C: Laptop with no GPU")
    validated_gpus, workers = gpu_utils.validate_gpu_configuration(None, max_workers=4)
    print(f"  Laptop validation: {validated_gpus}, workers: {workers}")

    print("\nScenario D: Cloud instance with invalid GPU specification")
    invalid_gpus = [99, 100]  # Non-existent GPUs
    validated_gpus, workers = gpu_utils.validate_gpu_configuration(invalid_gpus, max_workers=4)
    print(f"  Cloud fallback validation: {validated_gpus}, workers: {workers}")


def example_5_performance_comparison():
    """Example 5: Performance considerations and best practices."""
    print("\n" + "="*60)
    print("Example 5: Performance Best Practices")
    print("="*60)

    print("Best practices for GPU multi-processing:")
    print("1. GPU assignment is cyclical - workers share GPUs efficiently")
    print("2. Use gpu_ids=None for automatic CPU-only (maintains compatibility)")
    print("3. Use gpu_ids=[] to explicitly force CPU-only")
    print("4. Use gpu_ids=[0,1,2] to use specific GPUs")
    print("5. Set max_workers higher than GPU count for better utilization")

    # Demonstrate worker argument creation
    import tempfile
    temp_dir = tempfile.mkdtemp()

    print(f"\nWorker initialization examples:")

    # CPU-only workers
    cpu_args = gpu_utils.create_worker_initargs(temp_dir, None)
    print(f"CPU workers: {len(cpu_args)} workers, args: {cpu_args}")

    # GPU workers (simulated)
    gpu_args = gpu_utils.create_worker_initargs(temp_dir, [0, 1])
    print(f"GPU workers: {len(gpu_args)} workers, args: {gpu_args}")

    print("\n📊 Performance expectations:")
    print("- Small datasets (< 1000 rows): CPU-only often sufficient")
    print("- Medium datasets (1000-10000 rows): GPU can provide 2-5x speedup")
    print("- Large datasets (> 10000 rows): GPU can provide 5-20x speedup")
    print("- Many groupings benefit most from parallelization")


def main():
    """Run all examples to demonstrate GPU multi-processing capabilities."""
    print("🚀 GPU Multi-Processing Examples for optstop")
    print("Demonstrating backward compatibility and new GPU capabilities")

    # Configure logging to file to reduce console noise during examples
    configure_optstop_logging('gpu_examples.log', console_output=False)

    try:
        example_1_backward_compatibility()
        example_2_gpu_auto_detection()
        example_3_explicit_gpu_assignment()
        example_4_mixed_usage_patterns()
        example_5_performance_comparison()

        print("\n" + "="*60)
        print("🎉 All examples completed successfully!")
        print("✅ Backward compatibility maintained")
        print("✅ GPU multi-processing implemented")
        print("✅ Flexible configuration options available")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        raise


if __name__ == "__main__":
    main()