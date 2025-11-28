"""
Test script to verify numpyro works with OrderedLogistic models.

This tests whether PyMC can compile OrderedLogistic models to JAX backend
and sample using numpyro (with CPU).
"""
import numpy as np
import pymc as pm
import time
import sys

# Import the ordinal model creation function
from optstop.ordinal_model import _create_orderedlogistic_model


def test_orderedlogistic_numpyro():
    """Test OrderedLogistic sampling with numpyro sampler."""
    print("="*80)
    print("Testing OrderedLogistic with numpyro sampler (JAX CPU)")
    print("="*80)

    # Small test dataset
    np.random.seed(42)
    n_categories = 11
    n_items = 1
    scores = np.array([5, 6, 7, 7, 8, 7, 6, 7, 7, 8], dtype='int64')

    print(f"\nTest data: {len(scores)} scores, {n_categories} categories")
    print(f"Scores: {scores}")

    # Create model
    print("\n1. Creating OrderedLogistic model...")
    model = _create_orderedlogistic_model(n_categories=n_categories, n_items=n_items)

    # Update data
    with model:
        pm.set_data({"scores": scores})

    print("   Model created successfully ✓")

    # Test 1: PyMC default sampler (baseline)
    print("\n2. Testing PyMC default sampler (baseline)...")
    try:
        start = time.time()
        with model:
            trace_pymc = pm.sample(
                draws=100,
                tune=100,
                chains=2,
                cores=2,
                progressbar=False,
                random_seed=42
            )
        elapsed_pymc = time.time() - start
        print(f"   ✓ PyMC default: {elapsed_pymc:.2f}s")
        print(f"     - Samples: {trace_pymc.posterior.sizes}")
        pymc_success = True
    except Exception as e:
        print(f"   ✗ PyMC default failed: {e}")
        pymc_success = False
        elapsed_pymc = None

    # Test 2: numpyro sampler
    print("\n3. Testing numpyro sampler (JAX backend)...")
    try:
        start = time.time()
        with model:
            trace_numpyro = pm.sample(
                nuts_sampler='numpyro',
                draws=100,
                tune=100,
                chains=2,
                progressbar=False,
                random_seed=42
            )
        elapsed_numpyro = time.time() - start
        print(f"   ✓ numpyro: {elapsed_numpyro:.2f}s")
        print(f"     - Samples: {trace_numpyro.posterior.sizes}")
        numpyro_success = True
    except Exception as e:
        print(f"   ✗ numpyro failed: {type(e).__name__}: {e}")
        numpyro_success = False
        elapsed_numpyro = None

    # Summary
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    print(f"PyMC default:  {'✓ SUCCESS' if pymc_success else '✗ FAILED'}" +
          (f" ({elapsed_pymc:.2f}s)" if elapsed_pymc else ""))
    print(f"numpyro:       {'✓ SUCCESS' if numpyro_success else '✗ FAILED'}" +
          (f" ({elapsed_numpyro:.2f}s)" if elapsed_numpyro else ""))

    if pymc_success and numpyro_success:
        speedup = elapsed_pymc / elapsed_numpyro
        print(f"\nSpeedup: {speedup:.2f}x")
        if speedup > 1.2:
            print(f"✓ numpyro is faster than PyMC default")
        else:
            print(f"⚠ numpyro not significantly faster (but works)")

    print("\n" + "="*80)
    if numpyro_success:
        print("CONCLUSION: ✓ numpyro works with OrderedLogistic models")
        print("            Ready to implement fallback logic")
        return 0
    else:
        print("CONCLUSION: ✗ numpyro does NOT work with OrderedLogistic")
        print("            Need to use PyMC default for ordinal models")
        return 1


if __name__ == "__main__":
    sys.exit(test_orderedlogistic_numpyro())
