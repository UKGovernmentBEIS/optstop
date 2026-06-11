"""
Pytest configuration for optimal stopping tests.

Includes solutions for PyTensor/PyMC cache pollution issues.
"""

import pytest
import gc


def _reset_pymc_model_context():
    """Clear any leftover entries on PyMC's model-context stack.

    PyMC tracks active ``with pm.Model()`` contexts on ``MODEL_MANAGER``, a
    ``threading.local`` stack (append on enter, pop on exit). The inspect_ai
    bridge runs inference inside a ``ThreadPoolExecutor``; when many tests share
    one interpreter, a stray enter/exit imbalance can leave the stack desynced,
    after which every later model build raises ``TypeError: No model on context
    stack``. Forcing the stack empty between tests isolates each test from the
    previous one's residue. Best-effort and version-tolerant - never fails a test.
    """
    try:
        from pymc.model.core import MODEL_MANAGER
        MODEL_MANAGER.active_contexts.clear()
    except Exception:
        pass


def pytest_collection_modifyitems(items):
    """
    Reorder tests to run continuous aggregated tests first.

    RATIONALE:
    Continuous hierarchical tests become slow when run after many other tests
    due to PyTensor compilation cache accumulation. Running them first ensures
    they execute in a clean environment with optimal performance.

    This is a pragmatic solution that doesn't require code changes or external
    dependencies while ensuring all tests complete in reasonable time.
    """
    continuous_tests = []
    other_tests = []

    for item in items:
        # Identify continuous aggregated tests (binary_aggregated, ordinal_aggregated)
        if 'BinaryAggregated' in item.nodeid or 'OrdinalAggregated' in item.nodeid:
            continuous_tests.append(item)
        else:
            other_tests.append(item)

    # Reorder: continuous first, then others
    items[:] = continuous_tests + other_tests

    # Optional: Print test order for verification
    if items and hasattr(items[0].config.option, 'verbose') and items[0].config.option.verbose > 0:
        print(f"\n📋 Test execution order: {len(continuous_tests)} continuous, {len(other_tests)} discrete")


@pytest.fixture(scope="function", autouse=True)
def cleanup_pytensor_cache():
    """
    Clean up PyTensor compilation cache and memory between tests.

    This fixture runs automatically after each test to prevent cache accumulation
    and memory fragmentation that can slow down PyMC MCMC sampling.
    """
    # Before test: Clear Python garbage and any leftover PyMC model context
    gc.collect()
    _reset_pymc_model_context()

    yield

    # After test: drop any PyMC model context the test left on the stack
    _reset_pymc_model_context()

    # After test: Aggressive cleanup
    try:
        import pytensor

        # Clear any cached compiled functions
        if hasattr(pytensor.compile, 'function') and hasattr(pytensor.compile.function, 'function_cache'):
            pytensor.compile.function.function_cache.clear()

        # Clear function graph cache if available
        if hasattr(pytensor, 'graph') and hasattr(pytensor.graph, 'cache'):
            pytensor.graph.cache.clear()

    except (ImportError, AttributeError) as e:
        # Don't fail tests if cleanup fails - PyTensor internals may change
        pass

    # Force garbage collection to free memory
    gc.collect()


@pytest.fixture(scope="function")
def cleanup_jax_cache():
    """
    Optional: Clear JAX compilation cache.

    Use this fixture explicitly in tests that need extra cleanup:
        def test_something(cleanup_jax_cache):
            ...
    """
    yield

    try:
        import jax
        jax.clear_caches()
    except ImportError:
        pass
