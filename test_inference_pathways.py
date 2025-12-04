#!/usr/bin/env python3
"""
Test script to verify all three inference pathways (binary, ordinal, continuous)
work correctly with optimal_stopping_posthoc, optimal_stopping_live, and convergence_posthoc.
"""

import numpy as np
import pandas as pd
import warnings
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from optstop.rule import (
    optimal_stopping_posthoc,
    optimal_stopping_live,
    configure_optstop_logging
)
from optstop.convergence import convergence_posthoc

# Configure logging to file only
configure_optstop_logging('test_inference_pathways.log', console_output=False)


def create_test_data():
    """Create test datasets for binary, ordinal, and continuous scoring."""
    np.random.seed(42)

    # Parameters
    n_items = 10  # Small for fast testing
    n_epochs = 15

    data_rows = []

    # === BINARY TASK ===
    # Binary scores (0/1) - simulating a pass/fail scenario
    for item_id in range(n_items):
        true_prob = np.random.uniform(0.6, 0.9)  # High success probability
        for epoch in range(n_epochs):
            score = int(np.random.random() < true_prob)
            data_rows.append({
                'task': 'binary_task',
                'item_id': f'item_{item_id}',
                'epoch': epoch,
                'score': score
            })

    # === ORDINAL TASK ===
    # Ordinal scores (0-10) - simulating a Likert-like scale
    for item_id in range(n_items):
        # Most responses cluster around 7-8
        for epoch in range(n_epochs):
            score = int(np.clip(np.random.normal(7.5, 1.5), 0, 10))
            data_rows.append({
                'task': 'ordinal_rating',
                'item_id': f'item_{item_id}',
                'epoch': epoch,
                'score': score
            })

    # === CONTINUOUS TASK ===
    # Continuous bounded scores (0-1 range, pre-aggregated means)
    for item_id in range(n_items):
        # Mean scores around 0.75
        for epoch in range(n_epochs):
            score = np.clip(np.random.beta(8, 3), 0.01, 0.99)  # Beta distribution centered around 0.73
            data_rows.append({
                'task': 'continuous_mean',
                'item_id': f'item_{item_id}',
                'epoch': epoch,
                'score': float(score)
            })

    df = pd.DataFrame(data_rows)
    return df


def test_optimal_stopping_posthoc(df):
    """Test optimal_stopping_posthoc with all three pathways."""
    print("\n" + "="*70)
    print("TESTING optimal_stopping_posthoc")
    print("="*70)

    params = {
        'delta_item': 0.15,  # Relaxed for faster convergence
        'delta_cap': 0.15,
        'draws': 500,  # Minimal for testing
        'tune': 500,
        'chains': 2,
        'cores': 2,
        'stab_window': 5,
        'rep_batch_size': 1,
        'pymc_refresh_every': 5,
    }

    results = {}

    # Test each pathway
    for task_name, ordinal_tasks, continuous_tasks, expected_type in [
        ('binary_task', None, None, 'binary'),
        ('ordinal_rating', ['ordinal'], None, 'ordinal'),
        ('continuous_mean', None, ['continuous'], 'continuous'),
    ]:
        print(f"\n--- Testing {expected_type.upper()} pathway ({task_name}) ---")

        # Filter to just this task
        df_task = df[df['task'] == task_name].copy()

        try:
            pruned_df, summary = optimal_stopping_posthoc(
                df_task,
                params=params,
                grouping_columns='task',
                sample_id_column='item_id',
                epoch_column='epoch',
                score_column='score',
                display_progress=False,
                generate_diagnostics=False,
                ordinal_tasks=ordinal_tasks,
                ordinal_max_score=10,
                ordinal_inference='modal',
                ordinal_model_type='ordered_logistic',
                continuous_tasks=continuous_tasks,
            )

            original_rows = len(df_task)
            pruned_rows = len(pruned_df)
            reduction = 100 * (1 - pruned_rows / original_rows) if original_rows > 0 else 0

            # CRITICAL: Check if result is empty due to worker failure
            # A 100% reduction with 0 rows is suspicious - check summary for actual processing
            if pruned_rows == 0 and len(summary) > 0:
                # Check if summary indicates actual stopping or worker failure
                summary_entry = summary[0] if summary else {}
                if 'error' in str(summary_entry).lower() or summary_entry.get('stop_reason') is None:
                    raise ValueError(f"Worker failed - empty result with suspicious summary: {summary_entry}")

            print(f"  SUCCESS: {expected_type} pathway worked")
            print(f"  Original rows: {original_rows}, Pruned rows: {pruned_rows} ({reduction:.1f}% reduction)")
            print(f"  Summary entries: {len(summary)}")
            if summary:
                print(f"  First summary: {summary[0]}")

            results[expected_type] = {
                'status': 'SUCCESS',
                'original': original_rows,
                'pruned': pruned_rows,
                'reduction': reduction
            }

        except Exception as e:
            print(f"  FAILED: {expected_type} pathway error: {e}")
            import traceback
            traceback.print_exc()
            results[expected_type] = {'status': 'FAILED', 'error': str(e)}

    return results


def test_optimal_stopping_live(df):
    """Test optimal_stopping_live with all three pathways."""
    print("\n" + "="*70)
    print("TESTING optimal_stopping_live")
    print("="*70)

    params = {
        'delta_item': 0.15,
        'delta_cap': 0.15,
        'draws': 500,
        'tune': 500,
        'chains': 2,
        'cores': 2,
        'stab_window': 5,
        'rep_batch_size': 1,
        'pymc_refresh_every': 5,
    }

    results = {}

    # Test each pathway
    for task_name, ordinal_tasks, continuous_tasks, expected_type in [
        ('binary_task', None, None, 'binary'),
        ('ordinal_rating', ['ordinal'], None, 'ordinal'),
        ('continuous_mean', None, ['continuous'], 'continuous'),
    ]:
        print(f"\n--- Testing {expected_type.upper()} pathway ({task_name}) ---")

        # Filter to just this task
        df_task = df[df['task'] == task_name].copy()

        try:
            result = optimal_stopping_live(
                df_task,
                params=params,
                grouping_columns='task',
                sample_id_column='item_id',
                epoch_column='epoch',
                score_column='score',
                display_progress=False,
                ordinal_tasks=ordinal_tasks,
                ordinal_max_score=10,
                ordinal_inference='modal',
                ordinal_model_type='ordered_logistic',
                continuous_tasks=continuous_tasks,
            )

            n_stop_samples = len(result.get('stop_sample_ids', []))
            n_stop_tasks = len(result.get('stop_task', []))

            print(f"  SUCCESS: {expected_type} pathway worked")
            print(f"  Samples to stop: {n_stop_samples}")
            print(f"  Tasks to stop: {n_stop_tasks}")

            results[expected_type] = {
                'status': 'SUCCESS',
                'stop_samples': n_stop_samples,
                'stop_tasks': n_stop_tasks
            }

        except Exception as e:
            print(f"  FAILED: {expected_type} pathway error: {e}")
            import traceback
            traceback.print_exc()
            results[expected_type] = {'status': 'FAILED', 'error': str(e)}

    return results


def test_convergence_posthoc(df):
    """Test convergence_posthoc with all three pathways."""
    print("\n" + "="*70)
    print("TESTING convergence_posthoc")
    print("="*70)

    params = {
        'delta_item': 0.15,
        'delta_cap': 0.15,
        'draws': 500,
        'tune': 500,
        'chains': 2,
        'cores': 2,
        'stab_window': 5,
        'rep_batch_size': 1,
        'pymc_refresh_every': 5,
        'item_seqs': 2,  # Minimal for testing
        'epoch_seqs': 2,
    }

    results = {}

    # Test each pathway
    for task_name, ordinal_tasks, continuous_tasks, expected_type in [
        ('binary_task', None, None, 'binary'),
        ('ordinal_rating', ['ordinal'], None, 'ordinal'),
        ('continuous_mean', None, ['continuous'], 'continuous'),
    ]:
        print(f"\n--- Testing {expected_type.upper()} pathway ({task_name}) ---")

        # Filter to just this task
        df_task = df[df['task'] == task_name].copy()

        try:
            result_df = convergence_posthoc(
                df_task,
                params=params,
                grouping_columns='task',
                sample_id_column='item_id',
                epoch_column='epoch',
                score_column='score',
                display_progress=False,
                generate_diagnostics=False,
                ordinal_tasks=ordinal_tasks,
                ordinal_max_score=10,
                ordinal_inference='modal',
                ordinal_model_type='ordered_logistic',
                continuous_tasks=continuous_tasks,
            )

            n_rows = len(result_df)
            has_theta_ci = 'theta_ci_width' in result_df.columns and not result_df['theta_ci_width'].isna().all()
            has_performance = 'task_performance' in result_df.columns

            print(f"  SUCCESS: {expected_type} pathway worked")
            print(f"  Result rows: {n_rows}")
            print(f"  Has theta CI: {has_theta_ci}")
            print(f"  Has performance: {has_performance}")
            if has_theta_ci and n_rows > 0:
                theta_width = result_df['theta_ci_width'].iloc[0]
                print(f"  Theta CI width: {theta_width:.4f}" if theta_width else "  Theta CI width: N/A")

            results[expected_type] = {
                'status': 'SUCCESS',
                'rows': n_rows,
                'has_theta_ci': has_theta_ci
            }

        except Exception as e:
            print(f"  FAILED: {expected_type} pathway error: {e}")
            import traceback
            traceback.print_exc()
            results[expected_type] = {'status': 'FAILED', 'error': str(e)}

    return results


def main():
    print("="*70)
    print("INFERENCE PATHWAY TESTING")
    print("Testing binary, ordinal, and continuous scoring pathways")
    print("="*70)

    # Create test data
    print("\nCreating test data...")
    df = create_test_data()
    print(f"Created {len(df)} rows of test data:")
    print(f"  - Binary task: {len(df[df['task'] == 'binary_task'])} rows")
    print(f"  - Ordinal task: {len(df[df['task'] == 'ordinal_rating'])} rows")
    print(f"  - Continuous task: {len(df[df['task'] == 'continuous_mean'])} rows")

    # Run tests
    all_results = {}

    all_results['posthoc'] = test_optimal_stopping_posthoc(df)
    all_results['live'] = test_optimal_stopping_live(df)
    all_results['convergence'] = test_convergence_posthoc(df)

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    total_tests = 0
    passed_tests = 0

    for func_name, func_results in all_results.items():
        print(f"\n{func_name}:")
        for pathway, result in func_results.items():
            total_tests += 1
            status = result.get('status', 'UNKNOWN')
            if status == 'SUCCESS':
                passed_tests += 1
                print(f"  {pathway}: PASS")
            else:
                print(f"  {pathway}: FAIL - {result.get('error', 'Unknown error')}")

    print(f"\n{'='*70}")
    print(f"TOTAL: {passed_tests}/{total_tests} tests passed")
    print(f"{'='*70}")

    return passed_tests == total_tests


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
