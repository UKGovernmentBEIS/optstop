"""
Test script to verify nutpie is excluded for binary/continuous but allowed for ordinal
"""
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from optstop.gpu_utils import get_sampling_kwargs

print("="*80)
print("Testing nutpie exclusion for binary/continuous inference")
print("="*80)

params = {
    'draws': 500,
    'tune': 500,
    'chains': 4,
    'cores': 4
}

# Test 1: Binary (should exclude nutpie)
print("\n1. Testing BINARY inference:")
kwargs_binary = get_sampling_kwargs(
    params=params,
    gpu_available=False,
    gpu_backend='cpu',
    score_type='binary'
)
print(f"   nuts_sampler: {kwargs_binary.get('nuts_sampler', 'NOT SET (using PyMC default)')}")
expected = 'NOT SET (using PyMC default)'
actual = kwargs_binary.get('nuts_sampler', 'NOT SET (using PyMC default)')
print(f"   ✓ PASS: nutpie excluded" if actual == expected else f"   ✗ FAIL: Expected {expected}, got {actual}")

# Test 2: Continuous (should exclude nutpie)
print("\n2. Testing CONTINUOUS inference:")
kwargs_continuous = get_sampling_kwargs(
    params=params,
    gpu_available=False,
    gpu_backend='cpu',
    score_type='continuous_bounded'
)
print(f"   nuts_sampler: {kwargs_continuous.get('nuts_sampler', 'NOT SET (using PyMC default)')}")
expected = 'NOT SET (using PyMC default)'
actual = kwargs_continuous.get('nuts_sampler', 'NOT SET (using PyMC default)')
print(f"   ✓ PASS: nutpie excluded" if actual == expected else f"   ✗ FAIL: Expected {expected}, got {actual}")

# Test 3: Ordinal (should allow numpyro fallback, but nutpie if numpyro unavailable)
print("\n3. Testing ORDINAL inference:")
kwargs_ordinal = get_sampling_kwargs(
    params=params,
    gpu_available=False,
    gpu_backend='cpu',
    score_type='ordinal'
)
print(f"   nuts_sampler: {kwargs_ordinal.get('nuts_sampler', 'NOT SET (using PyMC default)')}")
# For ordinal, nutpie will be set (but ordinal_model.py will override to numpyro)
expected = 'nutpie'
actual = kwargs_ordinal.get('nuts_sampler', 'NOT SET (using PyMC default)')
print(f"   ✓ PASS: nutpie allowed (will fallback to numpyro in ordinal_model.py)" if actual == expected else f"   ✗ FAIL: Expected {expected}, got {actual}")

# Test 4: None (should allow nutpie - backward compatibility)
print("\n4. Testing None score_type (backward compatibility):")
kwargs_none = get_sampling_kwargs(
    params=params,
    gpu_available=False,
    gpu_backend='cpu',
    score_type=None
)
print(f"   nuts_sampler: {kwargs_none.get('nuts_sampler', 'NOT SET (using PyMC default)')}")
expected = 'nutpie'
actual = kwargs_none.get('nuts_sampler', 'NOT SET (using PyMC default)')
print(f"   ✓ PASS: nutpie allowed" if actual == expected else f"   ✗ FAIL: Expected {expected}, got {actual}")

print("\n" + "="*80)
print("Test complete!")
print("="*80)
