"""
Cause 4 Investigation: Excessive Reanalysis with Growing Datasets

Test hypothesis: Later inference calls take much longer because they process
MORE accumulated data, causing exponential time growth.

Test 4 configuration:
- reanalysis_interval = 200
- total_trials = 800
- Inference runs at trials: 200, 400, 600, 800

Expected pattern if this is the cause:
  Inference 1 (200 trials):   Quick
  Inference 2 (400 trials):   2x longer
  Inference 3 (600 trials):   3x longer
  Inference 4 (800 trials):   4x longer → HANGS?
"""

import sys
import time
import numpy as np
import logging

# Add to path
sys.path.insert(0, '/home/ubuntu/optstop')

from optstop.ordinal_model import _ordinal_entropy_ci_adaptive

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s'
)

print("="*80)
print("CAUSE 4: DATA GROWTH INVESTIGATION")
print("="*80)

print("\nHypothesis: MCMC sampling time grows with dataset size")
print("Testing with bimodal ordinal data (like Test 4)")

# Generate bimodal data (half high, half low)
np.random.seed(42)

def generate_bimodal_scores(n):
    """Generate n bimodal ordinal scores."""
    half = n // 2
    high = np.random.choice([7, 8, 9], size=half)
    low = np.random.choice([2, 3, 4], size=n - half)
    scores = np.concatenate([high, low])
    np.random.shuffle(scores)
    return scores

# Test with different dataset sizes (simulating Test 4 checkpoints)
test_sizes = [
    (200, "Inference 1 (trial 200)"),
    (400, "Inference 2 (trial 400)"),
    (600, "Inference 3 (trial 600)"),
    (800, "Inference 4 (trial 800)")
]

compute_kwargs = {
    'draws': 500,
    'tune': 500,
    'chains': 2,
    'cores': 2
}

results = []

print("\n" + "-"*80)
print("Running inference with increasing dataset sizes...")
print("-"*80)

for size, label in test_sizes:
    # Generate bimodal scores for this size
    scores = generate_bimodal_scores(size)

    print(f"\n{label}:")
    print(f"  Dataset size: {size} trials")
    print(f"  Score distribution: mean={np.mean(scores):.2f}, std={np.std(scores):.2f}")

    # Time the inference
    start_time = time.time()

    try:
        lo, hi, width, diagnostics = _ordinal_entropy_ci_adaptive(
            scores,
            ordinal_max_score=10,
            cred_level=0.95,
            conservatism=1.0,
            low_perf_threshold=0.2,
            n_samples=6000,  # Will be overridden
            n_tune=6000,     # Will be overridden
            model_cache=None,
            compute_kwargs=compute_kwargs
        )

        elapsed = time.time() - start_time

        print(f"  ✅ Completed in {elapsed:.1f} seconds")
        print(f"  Entropy CI: [{lo:.3f}, {hi:.3f}], width: {width:.3f}")

        # Check for convergence warnings
        if 'error' in diagnostics:
            print(f"  ⚠️  Error: {diagnostics['error']}")

        results.append({
            'size': size,
            'time': elapsed,
            'label': label,
            'entropy_width': width
        })

    except KeyboardInterrupt:
        print(f"  ❌ INTERRUPTED after {time.time() - start_time:.1f} seconds")
        print(f"  This suggests the function was taking too long!")
        results.append({
            'size': size,
            'time': time.time() - start_time,
            'label': label + " (INTERRUPTED)",
            'entropy_width': None
        })
        break
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  ❌ ERROR after {elapsed:.1f} seconds: {e}")
        results.append({
            'size': size,
            'time': elapsed,
            'label': label + " (ERROR)",
            'entropy_width': None
        })

# Analyze results
print("\n" + "="*80)
print("TIMING ANALYSIS")
print("="*80)

if len(results) >= 2:
    print("\nTiming progression:")
    print(f"{'Size':<10} {'Time (s)':<12} {'Relative':<12} {'Description':<30}")
    print("-"*70)

    base_time = results[0]['time']
    for r in results:
        relative = r['time'] / base_time
        print(f"{r['size']:<10} {r['time']:<12.1f} {relative:<12.1f}x  {r['label']:<30}")

    # Check for exponential growth
    if len(results) >= 3:
        time_ratios = [results[i]['time'] / results[i-1]['time']
                       for i in range(1, len(results))]
        avg_ratio = np.mean(time_ratios)

        print(f"\nAverage time growth ratio: {avg_ratio:.2f}x per doubling")

        if avg_ratio > 1.8:
            print("\n⚠️  FINDING: Near-linear or super-linear time growth!")
            print("   Time increases proportionally (or worse) with dataset size")
            print("   This confirms the data growth hypothesis")
            print("\n   For Test 4 with 800 trials:")
            print(f"   Expected time for inference 4: ~{results[0]['time'] * 4:.0f}s")
            if results[0]['time'] * 4 > 180:
                print(f"   ⚠️  This exceeds reasonable timeout (>3 minutes)")
        elif avg_ratio > 1.2:
            print("\n⚠️  FINDING: Moderate time growth with dataset size")
            print("   Time increases somewhat with more data")
        else:
            print("\n✅ Time growth is sub-linear (good)")
            print("   MCMC sampling scales reasonably")

else:
    print("\n❌ Insufficient data to analyze timing progression")

# Check if this explains Test 4 hang
print("\n" + "="*80)
print("CONCLUSION: Does this explain Test 4 hang?")
print("="*80)

if results:
    inference_4_time = results[-1]['time'] if results[-1]['size'] == 800 else None

    if inference_4_time and inference_4_time > 120:
        print(f"\n✅ YES - This likely explains the hang!")
        print(f"   Inference on 800 trials took {inference_4_time:.0f}s (>{120}s)")
        print(f"   With 4 chains and real test overhead, could easily exceed 5-10 minutes")
    elif inference_4_time and inference_4_time > 60:
        print(f"\n⚠️  POSSIBLY - Getting close to problematic")
        print(f"   Inference on 800 trials took {inference_4_time:.0f}s")
        print(f"   Combined with test overhead, could cause perceived hang")
    elif inference_4_time:
        print(f"\n❌ NO - Times are reasonable")
        print(f"   Inference on 800 trials took only {inference_4_time:.0f}s")
        print(f"   Data growth is NOT the primary cause")
    else:
        print(f"\n⚠️  UNCLEAR - Did not complete all tests")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

if results and results[-1]['time'] > 60:
    print("\n1. Reduce dataset size for inference:")
    print("   - Limit to last N trials instead of all accumulated data")
    print("   - Example: max_trials_for_inference = 400")
    print("\n2. Increase reanalysis_interval:")
    print("   - Current: 200 (4 calls for 800 trials)")
    print("   - Suggested: 400 (2 calls for 800 trials)")
    print("\n3. Use sampling instead of full dataset:")
    print("   - Randomly sample N trials if dataset > threshold")
    print("\n4. Reduce MCMC samples for large datasets:")
    print("   - Adaptive: draws=500 for <200 trials, draws=300 for >500 trials")
else:
    print("\nData growth does not appear to be the primary issue.")
    print("Continue investigation with other hypotheses.")

print("="*80)
