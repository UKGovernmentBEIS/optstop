"""
Test what happens with ordinal inference when numpyro is NOT available
"""
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from optstop.ordinal_model import _ordinal_entropy_ci_adaptive

print("="*80)
print("Testing ordinal inference WITHOUT numpyro (PyMC default)")
print("="*80)

# Small test dataset
np.random.seed(42)
scores = np.array([5, 6, 7, 7, 8, 7, 6, 7, 7, 8])

print(f"\nTest data: {len(scores)} scores")
print(f"Scores: {scores}")

# Test with NO nuts_sampler (simulates numpyro not installed)
print("\n" + "-"*80)
print("Calling _ordinal_entropy_ci_adaptive with no nuts_sampler")
print("Expected: Should use PyMC default")
print("-"*80 + "\n")

try:
    lo, hi, width, diagnostics = _ordinal_entropy_ci_adaptive(
        scores=scores,
        ordinal_max_score=10,
        cred_level=0.95,
        n_samples=100,  # Small for speed
        n_tune=100,
        compute_kwargs={
            # No nuts_sampler parameter = PyMC default
            'chains': 2,
            'cores': 2,
            'progressbar': False
        }
    )

    print("\n" + "="*80)
    print("✓ SUCCESS: OrderedLogistic works with PyMC default")
    print("="*80)
    print(f"Entropy CI: [{lo:.3f}, {hi:.3f}]")
    print(f"CI Width: {width:.3f}")

    if 'error' in diagnostics:
        print(f"⚠ Warning: Error in diagnostics: {diagnostics['error']}")
    else:
        print(f"✓ No errors - PyMC default can handle OrderedLogistic")
        print(f"✓ This means ordinal works even without numpyro installed!")

except Exception as e:
    print("\n" + "="*80)
    print(f"✗ FAILED: {type(e).__name__}: {e}")
    print("="*80)
    print(f"OrderedLogistic REQUIRES numpyro or PyMC will fail")
    import traceback
    traceback.print_exc()
