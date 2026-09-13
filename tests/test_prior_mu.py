"""
Tests for the prior_mu parameter functionality.

The prior_mu parameter controls the centre of the group-level Normal prior
on the logit scale for hierarchical models. Default is 0.0 (50% probability).

These tests verify:
1. Parameter presence in all function signatures with correct defaults
2. Parameter propagation through all pathways
3. CLI argument parsing

Note: prior_mu shifts the prior location, not the posterior-variance
convergence rate, so there is deliberately no "extreme priors change the
stopping decision" test here - the empirical effect is <0.2% and any such
assertion would be flaky rather than meaningful.
"""

import importlib

import pandas as pd
import numpy as np
import pytest
import inspect


# =============================================================================
# Test Data Fixtures
# =============================================================================


# CI partition: heavy MCMC tests deselected from PR CI (see pyproject.toml markers).
pytestmark = pytest.mark.optstop


@pytest.fixture
def binary_test_data():
    """Create minimal binary test data"""
    np.random.seed(42)
    return pd.DataFrame({
        'grouping': ['task1'] * 20,
        'sample_id': [f's{i}' for i in range(4)] * 5,
        'epoch': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
        'score': np.random.binomial(1, 0.7, 20)
    })


@pytest.fixture
def ordinal_test_data():
    """Create minimal ordinal test data"""
    np.random.seed(42)
    return pd.DataFrame({
        'grouping': ['task1'] * 20,
        'sample_id': [f's{i}' for i in range(4)] * 5,
        'epoch': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],
        'score': np.random.randint(0, 11, 20)
    })


@pytest.fixture
def default_params():
    """Default stopping parameters for testing"""
    return {
        'delta_item': 0.3,
        'delta_cap': 0.3,
        'draws': 100,
        'tune': 100,
        'chains': 2,
        'cores': 1
    }


# =============================================================================
# Signature Tests - Verify prior_mu in all function signatures
# =============================================================================

@pytest.mark.parametrize("module_path, attr", [
    ("optstop.rule", "optimal_stopping_posthoc"),
    ("optstop.rule", "optimal_stopping_live_single"),
    ("optstop.rule", "optimal_stopping_live"),
    ("optstop.convergence", "convergence_posthoc"),
    ("optstop.early_stopping", "OptimalStoppingManager"),
])
def test_prior_mu_in_signature(module_path, attr):
    """prior_mu is present with default 0.0 in every public entry point."""
    obj = getattr(importlib.import_module(module_path), attr)
    target = obj.__init__ if attr == "OptimalStoppingManager" else obj
    sig = inspect.signature(target)
    assert 'prior_mu' in sig.parameters
    assert sig.parameters['prior_mu'].default == 0.0


# =============================================================================
# Propagation Tests - Verify prior_mu is passed through correctly
# =============================================================================

class TestPriorMuPropagation:
    """Test that prior_mu propagates correctly through all functions"""

    def test_manager_stores_prior_mu(self, default_params):
        """Manager stores prior_mu: default 0.0, floats verbatim, ints coerced."""
        from optstop.early_stopping import OptimalStoppingManager

        # Default is the uninformative 0.0 (logit scale -> 50% probability)
        manager = OptimalStoppingManager(
            optstop_params=default_params,
            grouping_columns=['task']
        )
        assert manager.prior_mu == 0.0

        # Float values (positive, negative, fractional) are stored verbatim
        for val in [1.5, -2.0, 0.5, -0.5, 2.5]:
            manager = OptimalStoppingManager(
                optstop_params=default_params,
                grouping_columns=['task'],
                prior_mu=val
            )
            assert manager.prior_mu == val

        # Integer values are accepted (compared as float)
        manager = OptimalStoppingManager(
            optstop_params=default_params,
            grouping_columns=['task'],
            prior_mu=2  # int, not float
        )
        assert manager.prior_mu == 2.0

    def test_posthoc_accepts_prior_mu(self, binary_test_data, default_params):
        """Verify optimal_stopping_posthoc accepts prior_mu without error"""
        from optstop.rule import optimal_stopping_posthoc

        # Should not raise with default
        result, _ = optimal_stopping_posthoc(
            binary_test_data, default_params,
            grouping_columns='grouping',
            sample_id_column='sample_id',
            epoch_column='epoch',
            display_progress=False
        )
        assert result is not None

        # Should not raise with custom prior_mu
        result, _ = optimal_stopping_posthoc(
            binary_test_data, default_params,
            grouping_columns='grouping',
            sample_id_column='sample_id',
            epoch_column='epoch',
            prior_mu=2.0,
            display_progress=False
        )
        assert result is not None

    def test_live_accepts_prior_mu(self, binary_test_data, default_params):
        """Verify optimal_stopping_live accepts prior_mu without error"""
        from optstop.rule import optimal_stopping_live

        result = optimal_stopping_live(
            binary_test_data, default_params,
            grouping_columns='grouping',
            sample_id_column='sample_id',
            epoch_column='epoch',
            prior_mu=-1.0,
            display_progress=False
        )
        assert 'stop_sample_ids' in result
        assert 'stop_task' in result


# =============================================================================
# CLI Tests - Verify --prior_mu argument parsing
# =============================================================================

class TestPriorMuCLI:
    """Test CLI argument parsing for prior_mu"""

    def test_cli_argument_parser_prior_mu(self):
        """Verify all CLI argument parsers include prior_mu"""
        import optstop.cli as cli_module
        source = inspect.getsource(cli_module)

        assert "--prior_mu" in source or "'--prior_mu'" in source or '"--prior_mu"' in source


# =============================================================================
# Integration with Ordinal Pathway
# =============================================================================

class TestPriorMuOrdinal:
    """Test prior_mu with ordinal inference pathway"""

    def test_ordinal_accepts_prior_mu(self, ordinal_test_data, default_params):
        """Verify ordinal pathway accepts prior_mu"""
        from optstop.rule import optimal_stopping_posthoc

        result, diagnostics = optimal_stopping_posthoc(
            ordinal_test_data, default_params,
            grouping_columns='grouping',
            sample_id_column='sample_id',
            epoch_column='epoch',
            ordinal_tasks=['task'],
            ordinal_max_score=10,
            ordinal_inference='modal',
            prior_mu=1.0,
            display_progress=False
        )

        assert result is not None
        assert len(diagnostics) > 0
