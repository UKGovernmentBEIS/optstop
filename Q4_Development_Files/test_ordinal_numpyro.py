"""
Test that ordinal inference works with numpyro (no nutpie)
"""
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from optstop.ordinal_model import _ordinal_entropy_ci_adaptive

print("="*80)
print("Testing ordinal inference with numpyro (no nutpie)")
print("="*80)

# Small test dataset
np.random.seed(42)
scores = np.array([5, 6, 7, 7, 8, 7, 6, 7, 7, 8])

print(f"\nTest data: {len(scores)} scores")
print(f"Scores: {scores}")

# Test with nuts_sampler='numpyro' (as set by gpu_utils for ordinal)
print("\n" + "-"*80)
print("Calling _ordinal_entropy_ci_adaptive with nuts_sampler='numpyro'")
print("Expected: Should sample successfully using numpyro")
print("-"*80 + "\n")

try:
    lo, hi, width, diagnostics = _ordinal_entropy_ci_adaptive(
        scores=scores,
        ordinal_max_score=10,
        cred_level=0.95,
        n_samples=100,  # Small for speed
        n_tune=100,
        compute_kwargs={
            'nuts_sampler': 'numpyro',  # This is what gpu_utils now sets
            'chains': 2,
            'cores': 2,
            'progressbar': False
        }
    )

    print("\n" + "="*80)
    print("✓ SUCCESS: Ordinal sampling with numpyro")
    print("="*80)
    print(f"Entropy CI: [{lo:.3f}, {hi:.3f}]")
    print(f"CI Width: {width:.3f}")

    if 'error' in diagnostics:
        print(f"⚠ Warning: Error in diagnostics: {diagnostics['error']}")
    else:
        print(f"✓ No errors - numpyro working correctly")

except Exception as e:
    print("\n" + "="*80)
    print(f"✗ FAILED: {type(e).__name__}: {e}")
    print("="*80)
    import traceback
    traceback.print_exc()
