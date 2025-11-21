#!/usr/bin/env python3
"""
Analyze timing results from test suite to compare binary/ordinal/continuous performance.
"""

import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

def parse_timing_line(line: str) -> Tuple[str, str, float, int, str]:
    """
    Parse a TIMING_TEST line.
    Returns: (test_type, operation, time_seconds, n_items, score_type)
    """
    # Extract timing value
    time_match = re.search(r'took (\d+\.\d+)s', line)
    if not time_match:
        return None

    time_s = float(time_match.group(1))

    # Extract number of items
    items_match = re.search(r'for (\d+) items', line)
    n_items = int(items_match.group(1)) if items_match else 0

    # Extract number of observations (continuous only)
    obs_match = re.search(r'(\d+) observations', line)
    n_obs = int(obs_match.group(1)) if obs_match else 0

    # Extract score type
    score_type = None
    if '(binary)' in line:
        score_type = 'binary'
    elif '(ordinal)' in line:
        score_type = 'ordinal'
    elif '(continuous_01)' in line or '(continuous_bounded)' in line:
        score_type = 'continuous'

    # Determine operation type
    operation = None
    if 'Binary group inference' in line:
        operation = 'binary_group_inference'
    elif 'Ordinal group inference' in line:
        operation = 'ordinal_group_inference'
        # Extract mode
        mode_match = re.search(r'mode=(\w+)', line)
        if mode_match:
            operation += f"_{mode_match.group(1)}"
    elif 'Continuous data aggregation' in line:
        operation = 'continuous_aggregation'
    elif 'Continuous PyMC MCMC sampling' in line:
        operation = 'continuous_mcmc'
    elif 'Continuous group inference TOTAL' in line:
        operation = 'continuous_group_total'
    elif 'optimal_stopping_live_single TOTAL' in line:
        operation = 'function_total'

    return (score_type, operation, time_s, n_items, n_obs)

def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        with open(filename, 'r') as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    # Parse all timing lines
    timings = defaultdict(list)  # operation -> list of (time, n_items, n_obs)

    for line in lines:
        if 'TIMING_TEST' not in line:
            continue

        result = parse_timing_line(line)
        if result:
            score_type, operation, time_s, n_items, n_obs = result
            if operation:
                key = f"{score_type}_{operation}" if score_type else operation
                timings[key].append((time_s, n_items, n_obs))

    # Print summary
    print("=" * 80)
    print("TIMING ANALYSIS SUMMARY")
    print("=" * 80)
    print()

    # Group by operation type
    for operation_type in ['group_inference', 'aggregation', 'mcmc', 'total']:
        matching_ops = [k for k in timings.keys() if operation_type in k]
        if not matching_ops:
            continue

        print(f"\n{'='*40}")
        print(f"{operation_type.upper()} TIMING")
        print(f"{'='*40}")

        for key in sorted(matching_ops):
            times = [t[0] for t in timings[key]]
            items = [t[1] for t in timings[key] if t[1] > 0]
            obs = [t[2] for t in timings[key] if t[2] > 0]

            if not times:
                continue

            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            avg_items = sum(items) / len(items) if items else 0
            avg_obs = sum(obs) / len(obs) if obs else 0

            print(f"\n{key}:")
            print(f"  Samples: {len(times)}")
            print(f"  Avg time: {avg_time:.3f}s")
            print(f"  Range: [{min_time:.3f}s - {max_time:.3f}s]")
            if avg_items > 0:
                print(f"  Avg items: {avg_items:.1f}")
                print(f"  Time per item: {avg_time/avg_items*1000:.1f}ms")
            if avg_obs > 0:
                print(f"  Avg observations: {avg_obs:.1f}")

    # Comparison table
    print(f"\n\n{'='*80}")
    print("CROSS-TYPE COMPARISON (Group-Level Inference)")
    print(f"{'='*80}")
    print()
    print(f"{'Score Type':<20} {'Avg Time':>12} {'Min/Max':>20} {'Samples':>10}")
    print("-" * 80)

    for prefix in ['binary', 'ordinal', 'continuous']:
        group_keys = [k for k in timings.keys() if k.startswith(prefix) and 'group' in k and 'total' not in k]

        if not group_keys:
            continue

        all_times = []
        for key in group_keys:
            all_times.extend([t[0] for t in timings[key]])

        if all_times:
            avg_time = sum(all_times) / len(all_times)
            min_time = min(all_times)
            max_time = max(all_times)

            print(f"{prefix:<20} {avg_time:>10.3f}s  [{min_time:>5.2f}s - {max_time:>5.2f}s]  {len(all_times):>8}")

    print()
    print("=" * 80)

if __name__ == '__main__':
    main()
