"""
Test the nutpie -> numpyro fallback logic in ordinal_model.py
"""
import numpy as np
import logging

# Set up logging to see the fallback messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from optstop.ordinal_model import _ordinal_entropy_ci_adaptive

print("="*80)
print("Testing nutpie -> numpyro fallback for OrderedLogistic")
print("="*80)

# Small test dataset
np.random.seed(42)
scores = np.array([5, 6, 7, 7, 8, 7, 6, 7, 7, 8])

print(f"\nTest data: {len(scores)} scores")
print(f"Scores: {scores}")

# Test with nuts_sampler='nutpie' (should auto-fallback to numpyro)
print("\n" + "-"*80)
print("Calling _ordinal_entropy_ci_adaptive with nuts_sampler='nutpie'")
print("Expected: Should log fallback to numpyro")
print("-"*80 + "\n")

try:
    lo, hi, width, diagnostics = _ordinal_entropy_ci_adaptive(
        scores=scores,
        ordinal_max_score=10,
        cred_level=0.95,
        n_samples=100,  # Small for speed
        n_tune=100,
        compute_kwargs={
            'nuts_sampler': 'nutpie',  # This should trigger fallback
            'chains': 2,
            'cores': 2,
            'progressbar': False
        }
    )

    print("\n" + "="*80)
    print("✓ SUCCESS: Sampling completed with fallback")
    print("="*80)
    print(f"Entropy CI: [{lo:.3f}, {hi:.3f}]")
    print(f"CI Width: {width:.3f}")

    if 'error' in diagnostics:
        print(f"⚠ Warning: Error in diagnostics: {diagnostics['error']}")
    else:
        print(f"✓ No errors in diagnostics")

except Exception as e:
    print("\n" + "="*80)
    print(f"✗ FAILED: {type(e).__name__}: {e}")
    print("="*80)
    import traceback
    traceback.print_exc()
