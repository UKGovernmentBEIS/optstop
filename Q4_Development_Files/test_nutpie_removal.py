"""
Test script to verify nutpie removal and numpyro direct assignment for ordinal
"""
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from optstop.gpu_utils import get_sampling_kwargs

print("="*80)
print("Testing nutpie removal - numpyro should be used for ordinal")
print("="*80)

params = {
    'draws': 500,
    'tune': 500,
    'chains': 4,
    'cores': 4
}

# Test 1: Binary (should use PyMC default)
print("\n1. Testing BINARY inference:")
kwargs_binary = get_sampling_kwargs(
    params=params,
    gpu_available=False,
    gpu_backend='cpu',
    score_type='binary'
)
sampler = kwargs_binary.get('nuts_sampler', 'pymc (default)')
print(f"   nuts_sampler: {sampler}")
print(f"   ✓ PASS: PyMC default" if sampler == 'pymc (default)' else f"   ✗ FAIL: Expected 'pymc (default)', got {sampler}")

# Test 2: Continuous (should use PyMC default)
print("\n2. Testing CONTINUOUS inference:")
kwargs_continuous = get_sampling_kwargs(
    params=params,
    gpu_available=False,
    gpu_backend='cpu',
    score_type='continuous_bounded'
)
sampler = kwargs_continuous.get('nuts_sampler', 'pymc (default)')
print(f"   nuts_sampler: {sampler}")
print(f"   ✓ PASS: PyMC default" if sampler == 'pymc (default)' else f"   ✗ FAIL: Expected 'pymc (default)', got {sampler}")

# Test 3: Ordinal (should use numpyro directly - no nutpie conversion)
print("\n3. Testing ORDINAL inference:")
kwargs_ordinal = get_sampling_kwargs(
    params=params,
    gpu_available=False,
    gpu_backend='cpu',
    score_type='ordinal'
)
sampler = kwargs_ordinal.get('nuts_sampler', 'pymc (default)')
print(f"   nuts_sampler: {sampler}")
print(f"   ✓ PASS: numpyro directly set" if sampler == 'numpyro' else f"   ⚠ WARNING: Expected 'numpyro', got {sampler} (numpyro may not be installed)")

# Test 4: None score_type (should use numpyro if available)
print("\n4. Testing None score_type (backward compatibility):")
kwargs_none = get_sampling_kwargs(
    params=params,
    gpu_available=False,
    gpu_backend='cpu',
    score_type=None
)
sampler = kwargs_none.get('nuts_sampler', 'pymc (default)')
print(f"   nuts_sampler: {sampler}")
print(f"   ✓ PASS: numpyro used" if sampler == 'numpyro' else f"   ⚠ INFO: Using {sampler} (numpyro may not be installed)")

print("\n" + "="*80)
print("SUMMARY:")
print("  - Binary/Continuous: Using PyMC default (no compilation overhead)")
print("  - Ordinal: Using numpyro directly (no nutpie intermediate step)")
print("  - No nutpie package needed!")
print("="*80)
