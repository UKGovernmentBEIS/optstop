"""
Generate a detailed comparison report between baseline and nutpie test runs.

Usage:
    python scripts/generate_comparison_report.py
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd


def load_latest_backup(output_dir: Path) -> Path:
    """Find the most recent backup directory."""
    backup_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('backup_')])
    if not backup_dirs:
        raise FileNotFoundError("No backup directories found")
    return backup_dirs[-1]


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f} min"
    else:
        return f"{seconds/3600:.2f} hours"


def generate_comparison_report(output_dir: Path):
    """Generate comparison report between baseline and current runs."""

    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON REPORT: Baseline vs. Nutpie")
    print("=" * 80)

    # Load results
    current_results = {}
    for i in [1, 2, 3]:
        path = output_dir / f'dataset_{i}_diagnostics.json'
        if path.exists():
            with open(path, 'r') as f:
                current_results[i] = json.load(f)

    # Try to find baseline
    try:
        baseline_dir = load_latest_backup(output_dir)
        print(f"\n📁 Baseline directory: {baseline_dir}")

        baseline_results = {}
        for i in [1, 2, 3]:
            path = baseline_dir / f'dataset_{i}_diagnostics.json'
            if path.exists():
                with open(path, 'r') as f:
                    baseline_results[i] = json.load(f)

        print(f"📁 Current results: {output_dir}")
    except FileNotFoundError:
        print("\n⚠️  No baseline found. Showing current results only.")
        baseline_results = {}

    # Dataset names
    dataset_names = {
        1: "Binary Discrete",
        2: "Ordinal Discrete (Hybrid)",
        3: "Continuous Bounded"
    }

    # Generate comparison for each dataset
    comparison_data = []

    for i in [1, 2, 3]:
        print(f"\n{'=' * 80}")
        print(f"Dataset {i}: {dataset_names[i]}")
        print("=" * 80)

        if i not in current_results:
            print("  ❌ Current results not found")
            continue

        current = current_results[i]

        # Current run info
        print(f"\n📊 Current Run:")
        print(f"  Date: {current.get('test_date', 'unknown')}")
        print(f"  Runtime: {format_time(current['execution']['run_time_sec'])}")
        print(f"  Trials ran: {current['execution']['trial_count']}/{current['execution']['total_planned']}")
        print(f"  Efficiency: {100*(current['execution']['total_planned']-current['execution']['trial_count'])/current['execution']['total_planned']:.1f}%")
        print(f"  Sampler: {current.get('sampler_selection', {}).get('sampler', 'unknown')}")

        nutpie_used = current.get('sampler_selection', {}).get('nutpie_used', False)
        print(f"  Nutpie: {'✅ YES' if nutpie_used else '❌ NO'}")

        if nutpie_used:
            nutpie_version = current.get('sampler_selection', {}).get('nutpie_version', 'unknown')
            print(f"  Nutpie version: {nutpie_version}")

        # Inference timing breakdown
        if 'inference_timing' in current and current['inference_timing']:
            print(f"\n⏱️  Inference Timing Breakdown:")
            for pathway, timing in current['inference_timing'].items():
                print(f"    {pathway.capitalize()}: {timing['count']} calls, mean={timing['mean_sec']:.2f}s, total={format_time(timing['total_sec'])}")

        # Comparison with baseline
        if i in baseline_results:
            baseline = baseline_results[i]

            print(f"\n📊 Baseline Run:")
            print(f"  Date: {baseline.get('test_date', 'unknown')}")
            print(f"  Runtime: {format_time(baseline['execution']['run_time_sec'])}")
            print(f"  Sampler: {baseline.get('sampler_selection', {}).get('sampler', 'unknown')}")

            # Calculate speedup
            baseline_time = baseline['execution']['run_time_sec']
            current_time = current['execution']['run_time_sec']

            speedup = baseline_time / current_time if current_time > 0 else 0
            time_saved = baseline_time - current_time
            percent_change = 100 * time_saved / baseline_time

            print(f"\n🚀 Performance Comparison:")
            print(f"  Speedup: {speedup:.2f}×")
            print(f"  Time saved: {format_time(time_saved)} ({percent_change:.1f}%)")

            if speedup > 1.5:
                print(f"  ✅ Significant improvement!")
            elif speedup > 1.1:
                print(f"  ✓ Moderate improvement")
            elif speedup > 0.9:
                print(f"  ≈ Similar performance")
            else:
                print(f"  ⚠️  Performance regression")

            # Add to comparison data
            comparison_data.append({
                'Dataset': f"{i}. {dataset_names[i]}",
                'Baseline Time': format_time(baseline_time),
                'Current Time': format_time(current_time),
                'Speedup': f"{speedup:.2f}×",
                'Time Saved': format_time(time_saved),
                'Baseline Sampler': baseline.get('sampler_selection', {}).get('sampler', 'unknown'),
                'Current Sampler': current.get('sampler_selection', {}).get('sampler', 'unknown'),
                'Nutpie Used': '✅' if nutpie_used else '❌'
            })
        else:
            print(f"\n  ℹ️  No baseline available for comparison")

    # Overall summary table
    if comparison_data:
        print(f"\n{'=' * 80}")
        print("OVERALL SUMMARY TABLE")
        print("=" * 80)
        print()

        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))

        # Total time saved
        total_baseline = sum([baseline_results[i]['execution']['run_time_sec'] for i in [1, 2, 3] if i in baseline_results])
        total_current = sum([current_results[i]['execution']['run_time_sec'] for i in [1, 2, 3] if i in current_results])
        total_saved = total_baseline - total_current
        overall_speedup = total_baseline / total_current if total_current > 0 else 0

        print(f"\n{'=' * 80}")
        print("TOTAL ACROSS ALL DATASETS")
        print("=" * 80)
        print(f"  Baseline total: {format_time(total_baseline)}")
        print(f"  Current total: {format_time(total_current)}")
        print(f"  Overall speedup: {overall_speedup:.2f}×")
        print(f"  Total time saved: {format_time(total_saved)}")

    # Recommendations
    print(f"\n{'=' * 80}")
    print("RECOMMENDATIONS")
    print("=" * 80)

    if not any(current_results.get(i, {}).get('sampler_selection', {}).get('nutpie_used', False) for i in [1, 2, 3]):
        print("\n⚠️  Nutpie was NOT used in this run.")
        print("   To enable nutpie for faster sampling:")
        print("     pip install nutpie")
        print("   Expected speedup: 2-5× for CPU sampling")
    else:
        print("\n✅ Nutpie was successfully used in this run!")
        print("   CPU sampling benefited from Rust-based NUTS sampler")

    # Check for Dataset 2 (ordinal hybrid) specifically
    if 2 in current_results:
        dataset2 = current_results[2]
        ordinal_mode = dataset2.get('configuration', {}).get('ordinal_inference', 'unknown')
        runtime = dataset2['execution']['run_time_sec']

        print(f"\n📋 Dataset 2 (Ordinal) Notes:")
        print(f"   Ordinal inference mode: {ordinal_mode}")
        print(f"   Runtime: {format_time(runtime)}")

        if runtime > 3600:  # > 1 hour
            print(f"   ⚠️  Long runtime detected")
            print(f"   Consider:")
            print(f"     • Switch to ordinal_inference='modal' for production")
            print(f"     • Modal mode: ~0.1s vs. hybrid: ~minutes-hours")
            print(f"     • See BRIDGE_API_REFERENCE.md for guidance")

    print(f"\n{'=' * 80}")
    print("Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()


def main():
    output_dir = Path('/home/ubuntu/optstop/test_outputs/large_scale')

    if not output_dir.exists():
        print(f"❌ Output directory not found: {output_dir}")
        print("   Have you run the tests yet?")
        return

    generate_comparison_report(output_dir)


if __name__ == '__main__':
    main()
