"""Tests for MCMC failure handling and diagnostic logging.

Verifies that:
1. _log_mcmc_diagnostics correctly extracts and logs diagnostics
2. pm.sample() failures are caught gracefully at all sites
3. Failure fallbacks produce conservative (non-stopping) behaviour
"""

import pytest
import logging
import numpy as np
from unittest.mock import MagicMock, patch


# --- Tests for _log_mcmc_diagnostics ---

class TestLogMcmcDiagnostics:
    """Test the diagnostic logging helper function."""

    def _make_mock_trace(self, n_divergences=0, ess_min=500.0, rhat_max=1.001):
        """Create a mock InferenceData-like object."""
        trace = MagicMock()
        # sample_stats.diverging
        diverging = MagicMock()
        diverging.values.sum.return_value = n_divergences
        trace.sample_stats = {'diverging': diverging}
        return trace

    def test_clean_trace_logs_debug(self, caplog):
        """Clean trace (no divergences, good ESS) should log at DEBUG level."""
        from optstop.rule import _log_mcmc_diagnostics

        trace = self._make_mock_trace(n_divergences=0)
        logger = logging.getLogger('test.clean')

        with patch('optstop.rule.az') as mock_az:
            mock_ess = MagicMock()
            mock_ess.to_array.return_value.min.return_value.item.return_value = 500.0
            mock_az.ess.return_value = mock_ess

            mock_rhat = MagicMock()
            mock_rhat.to_array.return_value.max.return_value.item.return_value = 1.001
            mock_az.rhat.return_value = mock_rhat

            with caplog.at_level(logging.DEBUG, logger='test.clean'):
                _log_mcmc_diagnostics(trace, logger, "test_context")

            assert any("divergences=0" in r.message and r.levelno == logging.DEBUG for r in caplog.records)

    def test_divergent_trace_logs_warning(self, caplog):
        """Trace with divergences should log at WARNING level."""
        from optstop.rule import _log_mcmc_diagnostics

        trace = self._make_mock_trace(n_divergences=50)
        logger = logging.getLogger('test.divergent')

        with patch('optstop.rule.az') as mock_az:
            mock_ess = MagicMock()
            mock_ess.to_array.return_value.min.return_value.item.return_value = 500.0
            mock_az.ess.return_value = mock_ess

            mock_rhat = MagicMock()
            mock_rhat.to_array.return_value.max.return_value.item.return_value = 1.001
            mock_az.rhat.return_value = mock_rhat

            with caplog.at_level(logging.WARNING, logger='test.divergent'):
                _log_mcmc_diagnostics(trace, logger, "test_context")

            assert any("divergences=50" in r.message and r.levelno == logging.WARNING for r in caplog.records)

    def test_low_ess_logs_warning(self, caplog):
        """Trace with low ESS should log at WARNING level."""
        from optstop.rule import _log_mcmc_diagnostics

        trace = self._make_mock_trace(n_divergences=0)
        logger = logging.getLogger('test.low_ess')

        with patch('optstop.rule.az') as mock_az:
            mock_ess = MagicMock()
            mock_ess.to_array.return_value.min.return_value.item.return_value = 50.0  # Below 100
            mock_az.ess.return_value = mock_ess

            mock_rhat = MagicMock()
            mock_rhat.to_array.return_value.max.return_value.item.return_value = 1.001
            mock_az.rhat.return_value = mock_rhat

            with caplog.at_level(logging.WARNING, logger='test.low_ess'):
                _log_mcmc_diagnostics(trace, logger, "test_context")

            assert any("ess_min=50" in r.message and r.levelno == logging.WARNING for r in caplog.records)

    def test_exception_swallowed(self):
        """Corrupt trace should not raise - diagnostics must never crash."""
        from optstop.rule import _log_mcmc_diagnostics

        logger = logging.getLogger('test.corrupt')
        # Pass something that will fail on attribute access
        _log_mcmc_diagnostics("not_a_trace", logger, "test_context")
        # Should not raise


# --- Tests for sampling failure fallbacks ---

class TestSamplingFailureFallbacks:
    """Test that pm.sample() failures produce conservative behaviour."""

    def test_ordinal_modal_sampling_failure(self):
        """ordinal_utils._ordinal_ci_hierarchical_modal returns max uncertainty on failure."""
        from optstop.ordinal_utils import _ordinal_ci_hierarchical_modal

        # Create minimal valid inputs
        item_counts = np.array([[5, 3, 2, 0, 0]], dtype=np.int64)
        item_ns = np.array([10], dtype=np.int64)

        # pm is imported locally inside the function, so we mock pymc at the import level
        mock_pm = MagicMock()
        mock_pm.sample.side_effect = RuntimeError("Test sampling failure")

        mock_model = MagicMock()
        mock_model.__enter__ = MagicMock(return_value=mock_model)
        mock_model.__exit__ = MagicMock(return_value=False)

        with patch.dict('sys.modules', {'pymc': mock_pm}):
            result = _ordinal_ci_hierarchical_modal(
                item_counts=item_counts,
                item_ns=item_ns,
                ordinal_max_score=4,
                model_cache={'model': mock_model, 'model_n_items': 1},
            )

        # Should return max-uncertainty tuple
        assert result == (0.0, 1.0, 1.0)

    def test_ordinal_entropy_sampling_failure(self):
        """ordinal_utils._ordinal_ci_hierarchical_entropy returns max uncertainty on failure."""
        from optstop.ordinal_utils import _ordinal_ci_hierarchical_entropy

        item_counts = np.array([[5, 3, 2, 0, 0]], dtype=np.int64)
        item_ns = np.array([10], dtype=np.int64)

        mock_pm = MagicMock()
        mock_pm.sample.side_effect = RuntimeError("Test sampling failure")

        mock_model = MagicMock()
        mock_model.__enter__ = MagicMock(return_value=mock_model)
        mock_model.__exit__ = MagicMock(return_value=False)

        with patch.dict('sys.modules', {'pymc': mock_pm}):
            result = _ordinal_ci_hierarchical_entropy(
                item_counts=item_counts,
                item_ns=item_ns,
                ordinal_max_score=4,
                model_cache={'model': mock_model, 'model_n_items': 1},
            )

        # Should return 4-tuple with max uncertainty
        assert len(result) == 4
        lo, hi, width, diagnostics = result
        assert lo == 0.0
        assert hi == 1.0
        assert width == 1.0
        assert 'error' in diagnostics
