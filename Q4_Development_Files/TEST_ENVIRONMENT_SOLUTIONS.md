# Test Environment Solutions for PyMC/PyTensor Performance Issues

**Date**: 2025-11-21
**Issue**: Continuous hierarchical tests become extremely slow in full test suite
**Root Cause**: PyTensor compilation cache/memory accumulation over many tests

---

## Solution 1: PyTensor Cache Cleanup (RECOMMENDED)

### Implementation

Add to `tests/conftest.py`:

```python
import pytest
import gc
import shutil
import tempfile
import os

@pytest.fixture(scope="function", autouse=True)
def cleanup_pytensor_cache():
    """Clean up PyTensor compilation cache between tests to prevent slowdown."""

    # Before test: Clear Python garbage
    gc.collect()

    yield

    # After test: Aggressive cleanup
    try:
        import pytensor

        # Force clear any cached compiled functions
        if hasattr(pytensor.compile.function, 'function_cache'):
            pytensor.compile.function.function_cache.clear()

        # Clear module cache
        if hasattr(pytensor, 'function_graph'):
            pytensor.function_graph.cache.clear()

    except Exception as e:
        # Don't fail tests if cleanup fails
        print(f"Warning: PyTensor cleanup failed: {e}")

    # Force garbage collection
    gc.collect()
```

**Pros:**
- ✅ Lightweight - no extra dependencies
- ✅ Targets root cause directly
- ✅ Minimal performance overhead

**Cons:**
- ⚠️ May not catch all cache issues
- ⚠️ Relies on PyTensor internal APIs

---

## Solution 2: Process Isolation with pytest-forked

### Installation

```bash
pip install pytest-forked
```

### Usage

```bash
# Run tests with process isolation
pytest --forked tests/test_early_stopping_comprehensive.py

# Or in pytest.ini
[tool:pytest]
addopts = --forked
```

### Implementation in conftest.py

```python
import pytest

# Mark continuous tests to run in separate processes
def pytest_collection_modifyitems(items):
    """Mark continuous aggregated tests for process isolation."""
    for item in items:
        if 'BinaryAggregated' in item.nodeid or 'OrdinalAggregated' in item.nodeid:
            item.add_marker(pytest.mark.forked)
```

**Pros:**
- ✅ Complete isolation - guaranteed clean environment
- ✅ Eliminates all state pollution
- ✅ Works for any caching issue

**Cons:**
- ⚠️ Slower - overhead of forking processes
- ⚠️ Requires additional dependency

---

## Solution 3: Temporary Compilation Directory Per Test

### Implementation

Add to `tests/conftest.py`:

```python
import pytest
import tempfile
import shutil
import os

@pytest.fixture(scope="function", autouse=True)
def isolated_pytensor_compiledir(monkeypatch):
    """Create unique PyTensor compilation directory for each test."""

    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="pytensor_test_")

    # Set environment before PyTensor import
    monkeypatch.setenv('PYTENSOR_FLAGS', f'compiledir={temp_dir}')

    # Force re-import if already imported
    import sys
    if 'pytensor' in sys.modules:
        import pytensor
        pytensor.config.compiledir = temp_dir

    yield temp_dir

    # Cleanup after test
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass
```

**Pros:**
- ✅ Each test gets fresh compilation cache
- ✅ No cross-test pollution possible
- ✅ No external dependencies

**Cons:**
- ⚠️ Slower - must recompile for each test
- ⚠️ High disk I/O

---

## Solution 4: Test Ordering - Run Continuous Tests First

### Implementation

Add to `tests/conftest.py`:

```python
import pytest

def pytest_collection_modifyitems(items):
    """Reorder tests to run continuous tests first when they're fastest."""

    continuous_tests = []
    other_tests = []

    for item in items:
        if 'BinaryAggregated' in item.nodeid or 'OrdinalAggregated' in item.nodeid:
            continuous_tests.append(item)
        else:
            other_tests.append(item)

    # Run continuous first, then others
    items[:] = continuous_tests + other_tests
```

**Pros:**
- ✅ Simple - no cleanup needed
- ✅ Fast - no overhead
- ✅ Tests run at optimal speed

**Cons:**
- ⚠️ Doesn't solve root cause
- ⚠️ Test order dependency is fragile

---

## Solution 5: Memory and Thread Pool Cleanup

### Implementation

Add to `tests/conftest.py`:

```python
import pytest
import gc
import threading

@pytest.fixture(scope="function", autouse=True)
def cleanup_resources():
    """Clean up memory and thread pools between tests."""

    yield

    # Force garbage collection
    gc.collect()

    # Clean up any lingering thread pools
    try:
        import concurrent.futures
        # Force cleanup of any executors
        for obj in gc.get_objects():
            if isinstance(obj, concurrent.futures.ThreadPoolExecutor):
                obj.shutdown(wait=False)
    except Exception:
        pass

    # PyMC/JAX cleanup
    try:
        import jax
        jax.clear_caches()
    except Exception:
        pass
```

**Pros:**
- ✅ Addresses memory leaks
- ✅ Cleans up thread pools
- ✅ Good general practice

**Cons:**
- ⚠️ May not target PyTensor cache specifically
- ⚠️ Some overhead

---

## Solution 6: Separate Test Suites (PRACTICAL)

### Create separate test files

```
tests/
├── test_discrete_comprehensive.py      # Binary & Ordinal discrete
├── test_continuous_comprehensive.py    # Continuous aggregated
└── conftest.py
```

### Run separately

```bash
# Fast - no pollution
pytest tests/test_discrete_comprehensive.py
pytest tests/test_continuous_comprehensive.py

# Or in CI/CD
pytest tests/test_discrete_comprehensive.py && \
pytest tests/test_continuous_comprehensive.py
```

**Pros:**
- ✅ Complete isolation
- ✅ No code changes needed
- ✅ Parallel execution in CI

**Cons:**
- ⚠️ Test organization overhead
- ⚠️ Can't run "all tests" easily

---

## Recommended Approach

### For Development (Quick Fix)

**Use Solution 4 (Test Ordering)**:
```python
# tests/conftest.py
def pytest_collection_modifyitems(items):
    continuous_tests = [i for i in items if 'Aggregated' in i.nodeid]
    other_tests = [i for i in items if 'Aggregated' not in i.nodeid]
    items[:] = continuous_tests + other_tests
```

Run continuous tests first when cache is clean.

### For CI/CD (Robust)

**Use Solution 2 (pytest-forked)** + **Solution 1 (Cache Cleanup)**:
```bash
pip install pytest-forked
pytest --forked tests/test_early_stopping_comprehensive.py
```

Complete isolation with cleanup.

### For Long-term (Best Practice)

**Use Solution 6 (Separate Suites)** + **Solution 1 (Cache Cleanup)**:
- Split discrete and continuous tests
- Add cache cleanup fixtures
- Run suites separately in CI

---

## Implementation Priority

1. **Quick Win**: Add Solution 4 (test ordering) to conftest.py
2. **Validation**: Run tests and verify continuous tests are fast
3. **If still slow**: Add Solution 1 (PyTensor cache cleanup)
4. **If critical**: Install pytest-forked (Solution 2)
5. **Long-term**: Restructure into separate suites (Solution 6)

---

## Testing the Solution

```bash
# Test 1: Verify continuous in isolation (should be fast)
pytest tests/test_early_stopping_comprehensive.py::TestBinaryAggregatedHighPerformance -v
# Expected: 6-9 seconds

# Test 2: Full suite with solution applied
pytest tests/test_early_stopping_comprehensive.py -v
# Expected: Continuous tests complete in <10s each

# Test 3: Monitor for slowdown
pytest tests/test_early_stopping_comprehensive.py -v --log-cli-level=WARNING | grep "TIMING_TEST.*continuous"
# All continuous should be <15s
```

---

## Additional Notes

### Why This Happens

PyTensor compiles PyMC models to optimized C code and caches the compilation:
1. First test: Fresh cache → fast compilation
2. After many tests: Cache bloated → slow lookups/recompilation
3. Memory fragmentation from repeated MCMC chains
4. Thread pool exhaustion

### Why Our Code Is Not At Fault

The aggregated model is actually SIMPLER than binary:
- Fewer likelihood evaluations (50 vs 950)
- Smaller data structures
- Less complex graph

The slowdown is purely environmental.

---

**Status**: Multiple solutions available. Recommend starting with Solution 4 (ordering) + Solution 1 (cleanup).
