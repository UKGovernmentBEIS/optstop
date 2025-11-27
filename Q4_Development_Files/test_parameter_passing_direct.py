"""
Direct test of parameter passing to _ordinal_entropy_ci_adaptive

This bypasses the OptimalStoppingManager and directly tests if compute_kwargs
are being passed correctly to the MCMC sampling function.
"""

import sys
import logging
import numpy as np

# Add to path
sys.path.insert(0, '/home/ubuntu/optstop')

from optstop.ordinal_model import _ordinal_entropy_ci_adaptive

# Configure logging to see warnings
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def test_parameter_passing():
    """Test that compute_kwargs are passed correctly to PyMC sampling."""

    print("\n" + "="*80)
    print("DIRECT PARAMETER PASSING TEST")
    print("="*80)

    # Generate bimodal ordinal scores
    np.random.seed(42)
    scores_high = np.random.choice([7, 8, 9], size=15)
    scores_low = np.random.choice([2, 3, 4], size=15)
    scores = np.concatenate([scores_high, scores_low])
    np.random.shuffle(scores)

    print(f"\nTest data: {len(scores)} ordinal scores (0-10 scale)")
    print(f"Distribution: {np.bincount(scores, minlength=11)}")
    print(f"Mean: {np.mean(scores):.2f}")

    # Call with explicit compute_kwargs
    compute_kwargs = {
        'draws': 500,
        'tune': 500,
        'chains': 2,  # Reduce chains to speed up test
        'cores': 2
    }

    print(f"\n🔍 Calling _ordinal_entropy_ci_adaptive with:")
    print(f"   compute_kwargs = {compute_kwargs}")
    print(f"\nExpected behavior:")
    print(f"   - Should see diagnostic logs showing received parameters")
    print(f"   - Final draws should be 500 (not 6000)")
    print(f"   - Final tune should be 500 (not 6000)")
    print(f"   - Should complete in ~30-60 seconds")
    print("\n" + "-"*80)

    import time
    start_time = time.time()

    try:
        lo, hi, width, diagnostics = _ordinal_entropy_ci_adaptive(
            scores,
            ordinal_max_score=10,
            cred_level=0.95,
            conservatism=1.0,
            low_perf_threshold=0.2,
            n_samples=6000,  # Default value (should be overridden)
            n_tune=6000,     # Default value (should be overridden)
            model_cache=None,
            compute_kwargs=compute_kwargs  # Should override defaults
        )

        elapsed = time.time() - start_time

        print("-"*80)
        print(f"\n✅ SUCCESS! Completed in {elapsed:.1f} seconds")
        print(f"\nResults:")
        print(f"   Entropy CI: [{lo:.3f}, {hi:.3f}]")
        print(f"   Width: {width:.3f}")
        print(f"   Median entropy: {diagnostics.get('entropy_median', 'N/A'):.3f}")

        # Check timing to infer if correct parameters were used
        if elapsed < 90:
            print(f"\n✅ Timing suggests parameters were CORRECTLY passed (500/500)")
            print(f"   Expected: 30-60s, Actual: {elapsed:.1f}s")
        else:
            print(f"\n❌ Timing suggests parameters were NOT passed (using 6000/6000)")
            print(f"   Expected: 30-60s, Actual: {elapsed:.1f}s")
            print(f"   Running with 6000 draws would take 5-10 minutes")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n❌ ERROR after {elapsed:.1f} seconds:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()

    print("="*80 + "\n")


if __name__ == "__main__":
    test_parameter_passing()
