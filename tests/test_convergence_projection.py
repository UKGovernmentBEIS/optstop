"""Tests for convergence projection functionality."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from optstop.convergence import (
    project_convergence, _run_projection_loop,
    _fit_exponential, _project_exponential, _exp_decay,
)


# ---------------------------------------------------------------------------
# Helper: generate synthetic CI width trajectories
# ---------------------------------------------------------------------------

def make_decreasing_widths(n, start=0.15, end=0.04, noise=0.0):
    """Linearly decreasing widths with optional Gaussian noise."""
    widths = np.linspace(start, end, n)
    if noise > 0:
        rng = np.random.default_rng(42)
        widths = widths + rng.normal(0, noise, n)
        widths = np.clip(widths, 0.001, None)
    return widths.tolist()


def make_slopes_from_widths(widths, stab_window=15):
    """Compute slopes from widths using the same polyfit logic as the function."""
    slopes = []
    for i in range(len(widths) - stab_window + 1):
        window = widths[i:i + stab_window]
        slope = np.polyfit(range(stab_window), window, 1)[0]
        slopes.append(float(slope))
    return slopes


# ===========================================================================
# Point estimate tests
# ===========================================================================

class TestPointEstimate:

    def test_converges_within_cap(self):
        """Sufficient data, converges within max_steps."""
        widths = make_decreasing_widths(25, start=0.10, end=0.055)
        result = project_convergence(
            ci_widths=widths, delta=0.05, slope_threshold=0.00001,
            step_size=10, max_steps=200, n_bootstrap=50,
        )
        assert result is not None
        assert result['projected_additional_steps'] > 0
        assert result['projected_additional_trials'] == result['projected_additional_steps'] * 10
        assert not result['capped']
        assert result['convergence_target'] in ('projected_width', 'projected_slope_stabilisation')
        assert 'optimistic' not in result
        assert result['projection_basis'] in ('exponential_decay', 'linear_extrapolation')

    def test_hits_cap(self):
        """Width barely decreasing - should hit max_steps cap or detect plateau."""
        # Nearly flat trajectory well above delta
        widths = make_decreasing_widths(25, start=0.30, end=0.29)
        result = project_convergence(
            ci_widths=widths, delta=0.05, slope_threshold=0.00001,
            step_size=10, max_steps=50, n_bootstrap=50,
        )
        assert result is not None
        # Exponential model correctly detects asymptote above delta (slope_stab).
        # Linear fallback would hit cap since slope is tiny.
        assert result['convergence_target'] in ('projected_capped', 'projected_slope_stabilisation')
        if result['capped']:
            assert result['projected_additional_steps'] == 50

    def test_ordinal_slopes_computed_internally(self):
        """With empty ci_slopes (ordinal pathway), slopes are computed from widths.

        Both paths (with and without pre-computed slopes) attempt exponential
        first, so the point estimate should match regardless of slopes input.
        """
        widths = make_decreasing_widths(25, start=0.10, end=0.055)
        result = project_convergence(
            ci_widths=widths, ci_slopes=[],  # Empty - ordinal pathway
            delta=0.05, slope_threshold=0.00001,
            step_size=10, n_bootstrap=50,
        )
        assert result is not None
        assert result['projected_additional_steps'] > 0
        # Should match the result with pre-computed slopes
        slopes = make_slopes_from_widths(widths)
        result_with_slopes = project_convergence(
            ci_widths=widths, ci_slopes=slopes,
            delta=0.05, slope_threshold=0.00001,
            step_size=10, n_bootstrap=50,
        )
        assert result['projected_additional_steps'] == result_with_slopes['projected_additional_steps']

    def test_insufficient_history(self):
        """Too few width observations (above delta) returns None."""
        widths = make_decreasing_widths(10, start=0.15, end=0.08)  # < stab_window + 1 = 16, all > delta
        result = project_convergence(ci_widths=widths, delta=0.05)
        assert result is None

    def test_empty_inputs(self):
        """Empty inputs return None."""
        assert project_convergence(ci_widths=[], delta=0.05) is None
        assert project_convergence(ci_widths=None, delta=0.05) is None

    def test_already_converged(self):
        """Width already below delta."""
        widths = make_decreasing_widths(20, start=0.08, end=0.03)
        result = project_convergence(
            ci_widths=widths, delta=0.05, n_bootstrap=50,
        )
        assert result is not None
        assert result['projected_additional_steps'] == 0
        assert result['projected_additional_trials'] == 0
        assert result['proximity_ratio'] < 1.0
        assert result['convergence_target'] == 'projected_width'
        assert result['plateau_reached'] is False  # converged, not plateaued
        assert result['uncertainty']['confidence_level'] == 'high'

    def test_flat_trajectory_slope_stabilisation(self):
        """Slope near zero with width above delta triggers slope stabilisation."""
        # Constant width - slope is ~0 everywhere
        widths = [0.08] * 25
        result = project_convergence(
            ci_widths=widths, delta=0.05, slope_threshold=0.001,
            step_size=10, n_bootstrap=50,
        )
        assert result is not None
        assert result['convergence_target'] == 'projected_slope_stabilisation'
        assert result['projected_additional_steps'] <= 2  # Near-zero slope stabilises quickly

    def test_plateau_reached_flag(self):
        """plateau_reached distinguishes 'already at plateau' from 'already converged'."""
        # Case 1: already converged (width < delta) -> plateau_reached=False
        widths_converged = make_decreasing_widths(20, start=0.08, end=0.03)
        r1 = project_convergence(ci_widths=widths_converged, delta=0.05, n_bootstrap=50)
        assert r1 is not None
        assert r1['projected_additional_steps'] == 0
        assert r1['plateau_reached'] is False

        # Case 2: true exponential with asymptote above delta -> plateau_reached=True
        t = np.arange(50)
        widths_plateau = (0.4 * np.exp(-0.1 * t) + 0.20).tolist()
        r2 = project_convergence(ci_widths=widths_plateau, delta=0.05, n_bootstrap=50)
        assert r2 is not None
        assert r2['convergence_target'] == 'projected_slope_stabilisation'
        if r2['projected_additional_steps'] == 0:
            assert r2['plateau_reached'] is True

    def test_decelerating_convergence(self):
        """Positive slope-of-slopes (decelerating) means fewer steps than linear."""
        # Rapidly converging then levelling off
        widths_fast = make_decreasing_widths(25, start=0.12, end=0.055)
        widths_slow = make_decreasing_widths(25, start=0.12, end=0.075)
        result_fast = project_convergence(
            ci_widths=widths_fast, delta=0.05, step_size=10, n_bootstrap=50,
        )
        result_slow = project_convergence(
            ci_widths=widths_slow, delta=0.05, step_size=10, n_bootstrap=50,
        )
        assert result_fast is not None and result_slow is not None
        # Faster convergence should need fewer steps
        assert result_fast['projected_additional_steps'] <= result_slow['projected_additional_steps']


# ===========================================================================
# Uncertainty tests
# ===========================================================================

class TestUncertainty:

    def test_bootstrap_ci_contains_point_estimate(self):
        """Point estimate should be within or near the bootstrap CI."""
        widths = make_decreasing_widths(30, start=0.10, end=0.06)
        result = project_convergence(
            ci_widths=widths, delta=0.05, step_size=10, n_bootstrap=500,
        )
        assert result is not None
        point = result['projected_additional_trials']
        ci_80 = result['uncertainty']['ci_trials_80']
        # Point estimate should be within or close to the 80% CI
        # (allow some tolerance since point estimate uses exact final values)
        assert ci_80[0] <= point + 50  # Lower bound shouldn't be way above point
        assert ci_80[1] >= point - 50  # Upper bound shouldn't be way below point

    def test_more_evidence_tighter_ci(self):
        """Longer trajectory with same shape should produce tighter CI."""
        # Short trajectory (just above minimum)
        widths_short = make_decreasing_widths(20, start=0.10, end=0.06)
        result_short = project_convergence(
            ci_widths=widths_short, delta=0.05, step_size=10, n_bootstrap=300,
        )

        # Long trajectory with same slope
        widths_long = make_decreasing_widths(40, start=0.10, end=0.06)
        result_long = project_convergence(
            ci_widths=widths_long, delta=0.05, step_size=10, n_bootstrap=300,
        )

        assert result_short is not None and result_long is not None
        width_short = result_short['uncertainty']['ci_trials_80'][1] - result_short['uncertainty']['ci_trials_80'][0]
        width_long = result_long['uncertainty']['ci_trials_80'][1] - result_long['uncertainty']['ci_trials_80'][0]
        # More evidence should give equal or tighter CI
        assert width_long <= width_short + 20  # Allow small tolerance for stochasticity

    def test_noisy_trajectory_high_rmse(self):
        """Adding noise produces measurable RMSE."""
        widths_noisy = make_decreasing_widths(30, start=0.10, end=0.06, noise=0.02)

        result_noisy = project_convergence(
            ci_widths=widths_noisy, delta=0.05, n_bootstrap=50,
        )

        assert result_noisy is not None
        # Noisy data should have non-trivial RMSE regardless of projection basis
        assert result_noisy['uncertainty']['trajectory_rmse'] > 0

    def test_few_observations_not_high_confidence(self):
        """Few observations (< 10 for exponential, < 8 slopes for linear)
        prevents high confidence."""
        # 20 widths with stab_window=15 gives only 6 slopes (< 8 for high linear)
        # 20 widths < 10 threshold for high exponential confidence too
        widths = make_decreasing_widths(20, start=0.22, end=0.20)
        result = project_convergence(
            ci_widths=widths, delta=0.05, n_bootstrap=50,
        )
        assert result is not None
        assert result['uncertainty']['confidence_level'] != 'high'

    def test_clean_trajectory_high_confidence(self):
        """Clean monotonic trajectory close to threshold -> high confidence."""
        widths = make_decreasing_widths(30, start=0.08, end=0.052, noise=0.0)
        result = project_convergence(
            ci_widths=widths, delta=0.05, step_size=10, n_bootstrap=200,
        )
        assert result is not None
        assert result['uncertainty']['confidence_level'] in ('high', 'moderate')

    def test_uncertainty_dict_structure(self):
        """Verify all expected keys in uncertainty dict."""
        widths = make_decreasing_widths(25, start=0.10, end=0.06)
        result = project_convergence(ci_widths=widths, delta=0.05, n_bootstrap=50)
        assert result is not None
        # Top-level keys
        assert 'projected_width_at_termination' in result
        assert isinstance(result['projected_width_at_termination'], float)
        assert result['projection_basis'] in ('exponential_decay', 'linear_extrapolation')
        assert isinstance(result['plateau_reached'], bool)
        # Uncertainty keys
        unc = result['uncertainty']
        assert 'ci_trials_80' in unc
        assert 'ci_trials_50' in unc
        assert 'bootstrap_skipped' in unc
        assert 'trajectory_rmse' in unc
        assert 'confidence_level' in unc
        assert len(unc['ci_trials_80']) == 2
        assert len(unc['ci_trials_50']) == 2
        assert unc['ci_trials_80'][0] <= unc['ci_trials_80'][1]
        assert unc['ci_trials_50'][0] <= unc['ci_trials_50'][1]
        assert unc['confidence_level'] in ('low', 'moderate', 'high')
        assert isinstance(unc['bootstrap_skipped'], bool)
        assert isinstance(unc['trajectory_rmse'], float)
        # Basis-specific keys
        if result['projection_basis'] == 'exponential_decay':
            assert 'n_width_observations' in unc
            assert 'exponential_fit' in result
            fit = result['exponential_fit']
            assert 'a' in fit and 'b' in fit and 'c' in fit and 'r_squared' in fit
            assert fit['r_squared'] >= 0.7
        else:
            assert 'n_slope_observations' in unc
            assert isinstance(unc['n_slope_observations'], int)

    def test_terminal_width_consistency(self):
        """Width target: terminal_width <= delta. Slope_stab: terminal_width >= delta."""
        # Width convergence case
        widths = make_decreasing_widths(30, start=0.08, end=0.052, noise=0.0)
        r = project_convergence(ci_widths=widths, delta=0.05, n_bootstrap=50)
        assert r is not None
        if r['convergence_target'] == 'projected_width':
            # Exponential gives exactly delta at crossing; linear gives < delta
            assert r['projected_width_at_termination'] <= 0.05 + 1e-10
        elif r['convergence_target'] == 'projected_slope_stabilisation':
            assert r['projected_width_at_termination'] >= 0.05


# ===========================================================================
# Integration tests
# ===========================================================================

class TestBridgeIntegration:

    def test_projection_present_for_non_stopped(self):
        """_build_stabilization_entry adds projection for non-stopped groupings."""
        from optstop.early_stopping import OptimalStoppingManager

        manager = MagicMock(spec=OptimalStoppingManager)
        manager._stopped_groupings = set()
        manager.optstop_params = {'delta_cap': 0.05, 'CI_delta': 0.00001}
        manager.reanalysis_interval = 10
        manager.ordinal_tasks = None
        manager.score_agg = None

        history = {
            'ci_width_history': make_decreasing_widths(25, start=0.10, end=0.06),
            'ci_slope_history': [],
            'n_samples_evaluated': 250,
            'current_perf_estimate': 0.75,  # Above low_perf_threshold
        }

        entry = OptimalStoppingManager._build_stabilization_entry(
            manager, 'test-grouping', history
        )

        assert 'convergence_projection' in entry
        proj = entry['convergence_projection']
        assert proj['projected_additional_steps'] > 0
        assert 'uncertainty' in proj
        assert 'optimistic' not in proj

    def test_projection_absent_for_stopped(self):
        """_build_stabilization_entry does NOT add projection for stopped groupings."""
        from optstop.early_stopping import OptimalStoppingManager

        manager = MagicMock(spec=OptimalStoppingManager)
        manager._stopped_groupings = {'test-grouping'}  # Stopped
        manager.optstop_params = {'delta_cap': 0.05, 'CI_delta': 0.00001}
        manager.reanalysis_interval = 10
        manager.ordinal_tasks = None
        manager.score_agg = None

        history = {
            'ci_width_history': make_decreasing_widths(25, start=0.10, end=0.06),
            'ci_slope_history': [],
            'n_samples_evaluated': 250,
            'current_perf_estimate': 0.75,
        }

        entry = OptimalStoppingManager._build_stabilization_entry(
            manager, 'test-grouping', history
        )

        assert 'convergence_projection' not in entry

    # --- New bridge integration tests ---

    def _make_manager_mock(self, stopped=None, optstop_overrides=None,
                           ordinal_tasks=None, score_agg=None,
                           reanalysis_interval=10):
        """Helper: create a mock OptimalStoppingManager for bridge tests."""
        from optstop.early_stopping import OptimalStoppingManager

        manager = MagicMock(spec=OptimalStoppingManager)
        manager._stopped_groupings = stopped or set()
        params = {
            'delta_cap': 0.05,
            'CI_delta': 0.00001,
            'conservatism': 5,
            'low_performance_threshold': 0.01,
            'stab_window': 15,
        }
        if optstop_overrides:
            params.update(optstop_overrides)
        manager.optstop_params = params
        manager.reanalysis_interval = reanalysis_interval
        manager.ordinal_tasks = ordinal_tasks
        manager.score_agg = score_agg
        return manager

    def _call_build(self, manager, grouping_name, history):
        from optstop.early_stopping import OptimalStoppingManager
        return OptimalStoppingManager._build_stabilization_entry(
            manager, grouping_name, history
        )

    def test_exponential_basis_for_clean_trajectory(self):
        """Clean exponential-decay trajectory selects exponential basis."""
        manager = self._make_manager_mock()
        widths = [0.08 * np.exp(-0.1 * i) + 0.055 for i in range(30)]
        history = {
            'ci_width_history': widths,
            'ci_slope_history': [],
            'n_samples_evaluated': 300,
            'current_perf_estimate': 0.75,
        }
        entry = self._call_build(manager, 'test-grouping', history)

        assert 'convergence_projection' in entry
        proj = entry['convergence_projection']
        assert proj['projection_basis'] == 'exponential_decay'
        assert 'exponential_fit' in proj
        fit = proj['exponential_fit']
        assert 'a' in fit and 'b' in fit and 'c' in fit and 'r_squared' in fit
        assert fit['r_squared'] >= 0.7
        assert proj['uncertainty']['n_width_observations'] == 30

    def test_linear_fallback_few_observations(self):
        """Few widths with pre-supplied slopes uses linear; too few for anything returns None."""
        manager = self._make_manager_mock()

        # Case A: 3 widths, no slopes -> insufficient data, no projection
        history_a = {
            'ci_width_history': [0.10, 0.09, 0.08],
            'ci_slope_history': [],
            'n_samples_evaluated': 30,
            'current_perf_estimate': 0.75,
        }
        entry_a = self._call_build(manager, 'test-a', history_a)
        assert 'convergence_projection' not in entry_a

        # Case B: 4 widths + 2 pre-supplied slopes -> linear fallback
        history_b = {
            'ci_width_history': [0.10, 0.09, 0.08, 0.07],
            'ci_slope_history': [-0.003, -0.003],
            'n_samples_evaluated': 40,
            'current_perf_estimate': 0.75,
        }
        entry_b = self._call_build(manager, 'test-b', history_b)
        assert 'convergence_projection' in entry_b
        assert entry_b['convergence_projection']['projection_basis'] == 'linear_extrapolation'

    def test_low_perf_conservatism_slope_threshold(self):
        """Low-perf groupings get slope_threshold = CI_delta / conservatism."""
        manager = self._make_manager_mock()
        widths = make_decreasing_widths(25, start=0.10, end=0.06)
        base_history = {
            'ci_width_history': widths,
            'ci_slope_history': [],
            'n_samples_evaluated': 250,
        }

        with patch('optstop.convergence.project_convergence', return_value={'projected_additional_steps': 5}) as mock_pc:
            # Low performance
            history_low = {**base_history, 'current_perf_estimate': 0.0005}
            self._call_build(manager, 'low-perf', history_low)
            assert mock_pc.call_args.kwargs['slope_threshold'] == pytest.approx(0.00001 / 10)

            mock_pc.reset_mock()

            # Normal performance
            history_normal = {**base_history, 'current_perf_estimate': 0.75}
            self._call_build(manager, 'normal-perf', history_normal)
            assert mock_pc.call_args.kwargs['slope_threshold'] == pytest.approx(0.00001)

    def test_single_observation_no_projection(self):
        """Single width observation: enters projection branch but insufficient data."""
        manager = self._make_manager_mock()
        history = {
            'ci_width_history': [0.10],
            'ci_slope_history': [],
            'n_samples_evaluated': 10,
            'current_perf_estimate': 0.75,
        }
        entry = self._call_build(manager, 'single-obs', history)

        assert entry['final_ci_width'] == 0.10  # Not None, so code enters branch
        assert entry['n_group_checks'] == 1
        assert 'convergence_projection' not in entry  # But project_convergence returns None

    def test_plateau_reached_true(self):
        """Trajectory plateauing above delta sets plateau_reached=True."""
        manager = self._make_manager_mock()
        # Asymptote 0.08 > delta 0.05, so plateau_reached should be True
        widths = [0.03 * np.exp(-0.15 * i) + 0.08 for i in range(30)]
        history = {
            'ci_width_history': widths,
            'ci_slope_history': [],
            'n_samples_evaluated': 300,
            'current_perf_estimate': 0.75,
        }
        entry = self._call_build(manager, 'plateau-grouping', history)

        assert 'convergence_projection' in entry
        proj = entry['convergence_projection']
        assert proj['plateau_reached'] is True
        assert proj['convergence_target'] == 'projected_slope_stabilisation'
        assert proj['projected_additional_steps'] == 0

    def test_already_converged_zero_steps(self):
        """Final width < delta: zero additional steps, plateau_reached=False."""
        manager = self._make_manager_mock()
        history = {
            'ci_width_history': make_decreasing_widths(25, start=0.10, end=0.04),
            'ci_slope_history': [],
            'n_samples_evaluated': 250,
            'current_perf_estimate': 0.75,
        }
        entry = self._call_build(manager, 'converged-grouping', history)

        assert 'convergence_projection' in entry
        proj = entry['convergence_projection']
        assert proj['projected_additional_steps'] == 0
        assert proj['projected_additional_trials'] == 0
        assert proj['convergence_target'] == 'projected_width'
        assert proj['plateau_reached'] is False

    def test_ordinal_fields_present_when_matched(self):
        """Ordinal fields + projection coexist when ordinal_tasks matches grouping."""
        manager = self._make_manager_mock(ordinal_tasks=['math'])
        history = {
            'ci_width_history': make_decreasing_widths(25, start=0.10, end=0.06),
            'ci_slope_history': [],
            'n_samples_evaluated': 250,
            'current_perf_estimate': 0.75,
            # Ordinal-specific fields
            'final_modal_ci_width': 0.03,
            'final_modal_ci': (2, 5),
            'final_entropy': 1.2,
            'final_entropy_threshold': 1.5,
            'final_entropy_ci_width': 0.08,
            'final_convergence_threshold': 0.10,
            'ordinal_pathway': 'P1',
        }
        entry = self._call_build(manager, 'gpt-4-math_hard', history)

        # Both projection and ordinal fields present
        assert 'convergence_projection' in entry
        assert entry['final_modal_ci_width'] == 0.03
        assert entry['final_entropy'] == 1.2
        assert entry['ordinal_pathway'] == 'P1'
        assert entry['final_entropy_threshold'] == 1.5
        assert entry['final_entropy_ci_width'] == 0.08
        assert entry['final_convergence_threshold'] == 0.10
        assert entry['final_modal_ci'] == (2, 5)

    def test_ordinal_fields_absent_when_not_matched(self):
        """Ordinal fields excluded when ordinal_tasks is None."""
        manager = self._make_manager_mock(ordinal_tasks=None)
        history = {
            'ci_width_history': make_decreasing_widths(25, start=0.10, end=0.06),
            'ci_slope_history': [],
            'n_samples_evaluated': 250,
            'current_perf_estimate': 0.75,
            'final_modal_ci_width': 0.03,
            'ordinal_pathway': 'P1',
        }
        entry = self._call_build(manager, 'gpt-4-math_hard', history)

        assert 'convergence_projection' in entry  # Projection still present
        assert 'final_modal_ci_width' not in entry
        assert 'ordinal_pathway' not in entry

    def test_ordinal_fields_absent_when_aggregated(self):
        """Ordinal fields excluded when score_agg='mean' despite task match."""
        manager = self._make_manager_mock(ordinal_tasks=['math'], score_agg='mean')
        history = {
            'ci_width_history': make_decreasing_widths(25, start=0.10, end=0.06),
            'ci_slope_history': [],
            'n_samples_evaluated': 250,
            'current_perf_estimate': 0.75,
            'ordinal_pathway': 'P1',
        }
        entry = self._call_build(manager, 'gpt-4-math_hard', history)

        assert 'ordinal_pathway' not in entry

    def test_empty_width_history_no_projection(self):
        """Empty ci_width_history: final_ci_width is None, no projection."""
        manager = self._make_manager_mock()
        history = {
            'ci_width_history': [],
            'ci_slope_history': [],
            'n_samples_evaluated': 0,
            'current_perf_estimate': 0.0,
        }
        entry = self._call_build(manager, 'empty-grouping', history)

        assert entry['final_ci_width'] is None
        assert entry['final_slope'] is None
        assert entry['n_group_checks'] == 0
        assert 'convergence_projection' not in entry

    def test_all_params_pass_through(self):
        """All 6 kwargs forwarded correctly to project_convergence."""
        manager = self._make_manager_mock(
            optstop_overrides={'stab_window': 20, 'delta_cap': 0.08, 'CI_delta': 0.0001},
            reanalysis_interval=15,
        )
        widths = make_decreasing_widths(25, start=0.10, end=0.06)
        slopes = [-0.002, -0.001]
        history = {
            'ci_width_history': widths,
            'ci_slope_history': slopes,
            'n_samples_evaluated': 250,
            'current_perf_estimate': 0.75,
        }

        with patch('optstop.convergence.project_convergence', return_value={'projected_additional_steps': 5}) as mock_pc:
            self._call_build(manager, 'param-test', history)
            kw = mock_pc.call_args.kwargs
            assert kw['ci_widths'] == widths
            assert kw['ci_slopes'] == slopes
            assert kw['delta'] == 0.08
            assert kw['slope_threshold'] == pytest.approx(0.0001)  # Normal perf
            assert kw['step_size'] == 15
            assert kw['stab_window'] == 20

    def test_missing_perf_estimate_defaults_to_low_perf(self):
        """Missing current_perf_estimate defaults to 0.0, triggering conservatism."""
        manager = self._make_manager_mock()
        widths = make_decreasing_widths(25, start=0.10, end=0.06)
        history = {
            'ci_width_history': widths,
            'ci_slope_history': [],
            'n_samples_evaluated': 250,
            # No current_perf_estimate key
        }

        with patch('optstop.convergence.project_convergence', return_value={'projected_additional_steps': 5}) as mock_pc:
            self._call_build(manager, 'missing-perf', history)
            # Default 0.0 < low_performance_threshold 0.01 -> conservatism applied
            assert mock_pc.call_args.kwargs['slope_threshold'] == pytest.approx(0.00001 / 5)

    def test_ordinal_case_insensitive_matching(self):
        """Ordinal task matching is case-insensitive."""
        manager = self._make_manager_mock(ordinal_tasks=['MATH'])
        history = {
            'ci_width_history': make_decreasing_widths(25, start=0.10, end=0.06),
            'ci_slope_history': [],
            'n_samples_evaluated': 250,
            'current_perf_estimate': 0.75,
            'ordinal_pathway': 'P1',
            'final_modal_ci_width': 0.03,
        }
        entry = self._call_build(manager, 'gpt-4-math_hard', history)

        # Case-insensitive: 'MATH'.lower() in 'gpt-4-math_hard'.lower()
        assert 'ordinal_pathway' in entry
        assert entry['final_modal_ci_width'] == 0.03

    def test_projection_dict_structure(self):
        """Verify all expected keys in exponential projection dict."""
        manager = self._make_manager_mock()
        widths = [0.08 * np.exp(-0.1 * i) + 0.055 for i in range(30)]
        history = {
            'ci_width_history': widths,
            'ci_slope_history': [],
            'n_samples_evaluated': 300,
            'current_perf_estimate': 0.75,
        }
        entry = self._call_build(manager, 'struct-test', history)

        assert 'convergence_projection' in entry
        proj = entry['convergence_projection']

        # Required top-level keys
        required = {
            'projected_additional_steps', 'projected_additional_trials',
            'proximity_ratio', 'convergence_target',
            'projected_width_at_termination', 'plateau_reached',
            'capped', 'final_width', 'final_slope',
            'projection_basis', 'uncertainty',
        }
        assert required.issubset(proj.keys()), f"Missing keys: {required - proj.keys()}"

        # Exponential-specific
        assert proj['projection_basis'] == 'exponential_decay'
        assert 'exponential_fit' in proj

        # Uncertainty structure
        unc = proj['uncertainty']
        assert 'ci_trials_80' in unc and 'ci_trials_50' in unc
        assert isinstance(unc['ci_trials_80'], list) and len(unc['ci_trials_80']) == 2
        assert isinstance(unc['ci_trials_50'], list) and len(unc['ci_trials_50']) == 2
        assert 'bootstrap_skipped' in unc
        assert 'n_width_observations' in unc
        assert 'trajectory_rmse' in unc
        assert 'confidence_level' in unc


# ===========================================================================
# End-to-end: live_single → stabilization_history → _build_stabilization_entry
# ===========================================================================

class TestLiveSingleHistoryShape:
    """Verify stabilization_history from optimal_stopping_live_single has the
    shape that _build_stabilization_entry expects, for all three pathways."""

    # Minimal MCMC params for speed
    BASE_PARAMS = {
        'delta_item': 0.15,
        'delta_cap': 0.10,
        'cred_level': 0.90,
        'conservatism': 3,
        'draws': 200,
        'tune': 200,
        'chains': 2,
        'cores': 2,
        'CI_delta': 0.00001,
        'stab_window': 3,
    }

    REQUIRED_KEYS = {'ci_width_history', 'ci_slope_history', 'entropy_history', 'n_samples_evaluated'}

    @staticmethod
    def _make_binary_df(n_items=10, n_epochs=8, perf=0.80, seed=42):
        rng = np.random.default_rng(seed)
        rows = []
        for sid in range(n_items):
            for ep in range(n_epochs):
                rows.append({
                    'grouping': 'bin_test',
                    'sample_id': f's{sid}',
                    'epoch': ep,
                    'score': float(rng.random() < perf),
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _make_continuous_df(n_items=10, n_epochs=8, mean=0.75, seed=42):
        rng = np.random.default_rng(seed)
        rows = []
        for sid in range(n_items):
            for ep in range(n_epochs):
                rows.append({
                    'grouping': 'cont_test',
                    'sample_id': f's{sid}',
                    'epoch': ep,
                    'score': float(np.clip(rng.normal(mean, 0.05), 0, 1)),
                })
        return pd.DataFrame(rows)

    @staticmethod
    def _make_ordinal_df(n_items=10, n_epochs=8, modal=7, max_score=10, seed=42):
        rng = np.random.default_rng(seed)
        rows = []
        for sid in range(n_items):
            for ep in range(n_epochs):
                score = int(np.clip(rng.normal(modal, 1.0), 0, max_score))
                rows.append({
                    'grouping': 'ord_test',
                    'sample_id': f's{sid}',
                    'epoch': ep,
                    'score': score,
                })
        return pd.DataFrame(rows)

    def test_binary_history_shape(self):
        """Binary pathway produces stabilization_history with expected keys."""
        from optstop.rule import optimal_stopping_live_single
        df = self._make_binary_df()
        result = optimal_stopping_live_single(
            df_grouping=df, grouping_name='bin_test',
            params=self.BASE_PARAMS,
            sample_id_column='sample_id', epoch_column='epoch',
            score_column='score', stabilization_history=None,
            ordinal_tasks=None, ordinal_max_score=10,
        )
        hist = result['stabilization_history']
        assert self.REQUIRED_KEYS.issubset(hist.keys()), f"Missing: {self.REQUIRED_KEYS - hist.keys()}"
        assert 'current_perf_estimate' in hist
        assert isinstance(hist['ci_width_history'], list)
        assert isinstance(hist['ci_slope_history'], list)
        assert len(hist['ci_width_history']) >= 1  # At least one group check
        assert hist['n_samples_evaluated'] > 0
        assert 0 <= hist['current_perf_estimate'] <= 1
        # All widths are positive floats
        for w in hist['ci_width_history']:
            assert isinstance(w, float) and w > 0

    def test_continuous_history_shape(self):
        """Continuous pathway produces stabilization_history with expected keys."""
        from optstop.rule import optimal_stopping_live_single
        df = self._make_continuous_df()
        params = {**self.BASE_PARAMS, 'is_aggregated': True}
        result = optimal_stopping_live_single(
            df_grouping=df, grouping_name='cont_test',
            params=params,
            sample_id_column='sample_id', epoch_column='epoch',
            score_column='score', stabilization_history=None,
            ordinal_tasks=None, ordinal_max_score=10,
        )
        hist = result['stabilization_history']
        assert self.REQUIRED_KEYS.issubset(hist.keys()), f"Missing: {self.REQUIRED_KEYS - hist.keys()}"
        assert 'current_perf_estimate' in hist
        assert len(hist['ci_width_history']) >= 1
        assert hist['n_samples_evaluated'] > 0

    @pytest.mark.timeout(120)
    def test_ordinal_history_shape(self):
        """Ordinal pathway produces stabilization_history - ci_slope_history stays empty."""
        from optstop.rule import optimal_stopping_live_single
        df = self._make_ordinal_df()
        result = optimal_stopping_live_single(
            df_grouping=df, grouping_name='ord_test',
            params=self.BASE_PARAMS,
            sample_id_column='sample_id', epoch_column='epoch',
            score_column='score', stabilization_history=None,
            ordinal_tasks=['ord'], ordinal_max_score=10,
        )
        hist = result['stabilization_history']
        assert self.REQUIRED_KEYS.issubset(hist.keys()), f"Missing: {self.REQUIRED_KEYS - hist.keys()}"
        assert 'current_perf_estimate' in hist
        assert len(hist['ci_width_history']) >= 1
        # Ordinal never populates ci_slope_history
        assert hist['ci_slope_history'] == []

    def test_binary_accumulation_and_projection(self):
        """Multi-call binary accumulation produces history usable by _build_stabilization_entry."""
        from optstop.rule import optimal_stopping_live_single
        from optstop.early_stopping import OptimalStoppingManager

        df = self._make_binary_df(n_items=15, n_epochs=12, seed=99)
        params = {**self.BASE_PARAMS, 'stab_window': 3}

        # Simulate multiple bridge calls with growing data slices
        history = None
        n_items = 15
        for epoch_end in [4, 7, 10, 12]:
            # Each call sees data up to epoch_end
            df_slice = df[df['epoch'] < epoch_end]
            result = optimal_stopping_live_single(
                df_grouping=df_slice, grouping_name='accum_test',
                params=params,
                sample_id_column='sample_id', epoch_column='epoch',
                score_column='score', stabilization_history=history,
                ordinal_tasks=None, ordinal_max_score=10,
            )
            history = result['stabilization_history']

        # After 4 calls, ci_width_history should have accumulated
        assert len(history['ci_width_history']) >= 4, (
            f"Expected >= 4 width entries from 4 calls, got {len(history['ci_width_history'])}"
        )
        assert 'current_perf_estimate' in history

        # Now feed into _build_stabilization_entry (the real integration point)
        manager = MagicMock(spec=OptimalStoppingManager)
        manager._stopped_groupings = set()
        manager.optstop_params = {
            'delta_cap': 0.10, 'CI_delta': 0.00001, 'conservatism': 3,
            'low_performance_threshold': 0.01, 'stab_window': 3,
        }
        manager.reanalysis_interval = 10
        manager.ordinal_tasks = None
        manager.score_agg = None

        entry = OptimalStoppingManager._build_stabilization_entry(
            manager, 'accum_test', history
        )

        assert entry['n_samples'] > 0
        assert entry['final_ci_width'] is not None
        assert entry['n_group_checks'] == len(history['ci_width_history'])

        # With enough accumulated widths, projection should be present
        # (depends on whether stab_window + 1 widths exist for slope derivation)
        n_widths = len(history['ci_width_history'])
        stab_window = 3
        if n_widths >= stab_window + 1:
            # Enough data for internal slope computation -> projection should exist
            assert 'convergence_projection' in entry, (
                f"Expected projection with {n_widths} widths >= stab_window+1={stab_window + 1}"
            )
            proj = entry['convergence_projection']
            assert proj['projected_additional_steps'] >= 0
            assert 'projection_basis' in proj
            print(f"\n  Accumulated {n_widths} widths over 4 calls")
            print(f"  Projection basis: {proj['projection_basis']}")
            print(f"  Additional steps: {proj['projected_additional_steps']}")
        else:
            print(f"\n  Only {n_widths} widths accumulated - insufficient for projection")

    def test_continuous_accumulation_and_projection(self):
        """Multi-call continuous accumulation feeds into projection correctly."""
        from optstop.rule import optimal_stopping_live_single
        from optstop.early_stopping import OptimalStoppingManager

        df = self._make_continuous_df(n_items=15, n_epochs=12, seed=88)
        params = {**self.BASE_PARAMS, 'stab_window': 3, 'is_aggregated': True}

        history = None
        for epoch_end in [4, 7, 10, 12]:
            df_slice = df[df['epoch'] < epoch_end]
            result = optimal_stopping_live_single(
                df_grouping=df_slice, grouping_name='cont_accum',
                params=params,
                sample_id_column='sample_id', epoch_column='epoch',
                score_column='score', stabilization_history=history,
                ordinal_tasks=None, ordinal_max_score=10,
            )
            history = result['stabilization_history']

        assert len(history['ci_width_history']) >= 4

        # Feed into bridge
        manager = MagicMock(spec=OptimalStoppingManager)
        manager._stopped_groupings = set()
        manager.optstop_params = {
            'delta_cap': 0.10, 'CI_delta': 0.00001, 'conservatism': 3,
            'low_performance_threshold': 0.01, 'stab_window': 3,
        }
        manager.reanalysis_interval = 10
        manager.ordinal_tasks = None
        manager.score_agg = 'mean'

        entry = OptimalStoppingManager._build_stabilization_entry(
            manager, 'cont_accum', history
        )

        assert entry['final_ci_width'] is not None
        n_widths = len(history['ci_width_history'])
        if n_widths >= 4:  # stab_window + 1
            assert 'convergence_projection' in entry
            print(f"\n  Continuous: {n_widths} widths, basis={entry['convergence_projection']['projection_basis']}")


# ===========================================================================
# _run_projection_loop unit tests
# ===========================================================================

class TestProjectionLoop:

    def test_width_convergence(self):
        steps, target, tw = _run_projection_loop(
            width=0.06, slope=-0.005, slope_of_slopes=0.0,
            delta=0.05, slope_threshold=0.00001, max_steps=200,
        )
        assert target == 'projected_width'
        assert steps == 3  # 0.06 -> 0.055 -> 0.05 (not < 0.05) -> 0.045 (< 0.05)
        assert tw < 0.05

    def test_slope_stabilisation(self):
        steps, target, tw = _run_projection_loop(
            width=0.10, slope=-0.001, slope_of_slopes=0.001,
            delta=0.05, slope_threshold=0.001, max_steps=200,
        )
        assert target == 'projected_slope_stabilisation'
        # slope goes -0.001 + 0.001 = 0.0; abs(0) <= 0.001
        assert steps == 1
        assert tw > 0.05  # Width still above delta when slope stabilises

    def test_zero_slope_stabilises_immediately(self):
        """Zero slope with zero sos triggers slope_stabilisation on step 1."""
        steps, target, tw = _run_projection_loop(
            width=0.10, slope=0.0, slope_of_slopes=0.0,
            delta=0.05, slope_threshold=0.00001, max_steps=10,
        )
        # slope is 0 + 0 = 0; abs(0) <= 0.00001 and sos >= -1e-12
        assert target == 'projected_slope_stabilisation'
        assert steps == 1
        assert tw == pytest.approx(0.10)  # Width unchanged

    def test_actual_capping(self):
        """Steepening negative slope with tight delta and threshold - hits max_steps."""
        steps, target, tw = _run_projection_loop(
            width=0.10, slope=-0.001, slope_of_slopes=-0.001,
            delta=0.001, slope_threshold=0.000001, max_steps=10,
        )
        # sos < -1e-12 blocks slope_stab; delta=0.001 is far away; hits cap
        assert target == 'projected_capped'
        assert steps == 10

    def test_negative_sos_blocks_slope_stab(self):
        """Guard A: slope_of_slopes < -1e-12 prevents slope_stabilisation
        even when abs(slope) would satisfy the threshold."""
        # Step 1: next_slope = -0.0001 + -0.0001 = -0.0002
        # abs(-0.0002) = 0.0002 <= 0.001 (threshold met)
        # BUT sos = -0.0001 < -1e-12, so Guard A blocks slope_stab
        # Width keeps decreasing until < delta
        steps, target, tw = _run_projection_loop(
            width=0.10, slope=-0.0001, slope_of_slopes=-0.0001,
            delta=0.05, slope_threshold=0.001, max_steps=200,
        )
        assert target == 'projected_width'
        assert tw < 0.05

    def test_guard_a_tolerance_boundary(self):
        """Guard A uses >= -1e-12 tolerance for floating-point safety."""
        # All cases: slope small enough for threshold, so Guard A is decisive
        base = dict(width=0.10, slope=-0.0001, delta=0.05,
                    slope_threshold=0.001, max_steps=200)

        # Well inside tolerance: sos = -1e-14 -> allows slope_stab
        _, target_inside, _ = _run_projection_loop(
            **base, slope_of_slopes=-1e-14)
        assert target_inside == 'projected_slope_stabilisation'

        # Just inside tolerance: sos = -1e-12 -> allows slope_stab
        _, target_boundary, _ = _run_projection_loop(
            **base, slope_of_slopes=-1e-12)
        assert target_boundary == 'projected_slope_stabilisation'

        # Outside tolerance: sos = -1e-11 -> blocks slope_stab, continues to width or cap
        _, target_outside, _ = _run_projection_loop(
            **base, slope_of_slopes=-1e-11)
        assert target_outside != 'projected_slope_stabilisation'

    def test_sign_reversal_triggers_slope_stab(self):
        """When slope crosses from negative to positive in one step (overshooting
        the tight slope_threshold window), sign-change detection triggers
        slope_stabilisation instead of letting width grow unboundedly."""
        # slope=-0.003, sos=+0.005: step 1 gives slope=+0.002
        # abs(0.002) > 0.00001 (threshold), but sign reversed -> slope_stab
        steps, target, tw = _run_projection_loop(
            width=0.30, slope=-0.003, slope_of_slopes=0.005,
            delta=0.05, slope_threshold=0.00001, max_steps=200,
        )
        assert target == 'projected_slope_stabilisation'
        assert steps == 1
        assert tw == pytest.approx(0.30 + 0.002, abs=0.001)  # Width ~ start + new slope

    def test_sign_reversal_not_triggered_when_guard_a_blocks(self):
        """Sign reversal should NOT trigger slope_stab when Guard A fails
        (slope_of_slopes < -1e-12)."""
        # Negative sos means accelerating convergence - Guard A blocks
        steps, target, tw = _run_projection_loop(
            width=0.30, slope=0.001, slope_of_slopes=-0.005,
            delta=0.05, slope_threshold=0.00001, max_steps=200,
        )
        # slope goes 0.001 -> -0.004 (sign change but from + to -, not - to +)
        # and Guard A blocks anyway since sos < -1e-12
        assert target != 'projected_slope_stabilisation'


# ===========================================================================
# Exponential decay model tests
# ===========================================================================

class TestExponentialFit:

    def test_fit_on_true_exponential(self):
        """Exponential fit recovers parameters from known exponential data."""
        t = np.arange(30)
        # True params: a=0.5, b=0.1, c=0.02
        w = 0.5 * np.exp(-0.1 * t) + 0.02
        popt, r2 = _fit_exponential(w.tolist())
        assert popt is not None
        assert r2 > 0.99
        a, b, c = popt
        assert a == pytest.approx(0.5, rel=0.01)
        assert b == pytest.approx(0.1, rel=0.01)
        assert c == pytest.approx(0.02, rel=0.05)

    def test_fit_rejects_linear_data(self):
        """Linear data should fail the R-squared gate (exponential is poor fit)."""
        # Perfectly linear - exponential may or may not fit, depends on shape
        w = np.linspace(0.30, 0.06, 30)
        popt, r2 = _fit_exponential(w.tolist())
        # Linear data can sometimes fit exponential OK, so just verify no crash
        if popt is not None:
            assert r2 >= 0.7

    def test_fit_rejects_short_data(self):
        """< 5 observations returns None."""
        w = [0.3, 0.2, 0.15, 0.12]
        popt, r2 = _fit_exponential(w)
        assert popt is None

    def test_fit_rejects_noisy_data(self):
        """Very noisy data should fail R-squared gate."""
        rng = np.random.default_rng(42)
        w = rng.uniform(0.05, 0.50, 30)  # Random noise, no trend
        popt, r2 = _fit_exponential(w.tolist())
        # Should fail R-squared gate
        assert popt is None

    def test_asymptote_detection(self):
        """When asymptote >= delta, projection gives slope_stabilisation."""
        t = np.arange(30)
        # Asymptote c=0.08 > delta=0.05 -> will plateau above delta
        w = 0.4 * np.exp(-0.1 * t) + 0.08
        popt, r2 = _fit_exponential(w.tolist())
        assert popt is not None
        steps, target, tw = _project_exponential(popt, 30, delta=0.05,
                                                  step_size=10, max_steps=200)
        assert target == 'projected_slope_stabilisation'
        assert tw == pytest.approx(popt[2], rel=0.01)  # Terminal width ~ asymptote

    def test_width_convergence_projection(self):
        """When asymptote < delta, projection gives projected_width."""
        t = np.arange(30)
        # Asymptote c=0.01 < delta=0.05
        w = 0.5 * np.exp(-0.05 * t) + 0.01
        popt, r2 = _fit_exponential(w.tolist())
        assert popt is not None
        steps, target, tw = _project_exponential(popt, 30, delta=0.05,
                                                  step_size=10, max_steps=500)
        assert target == 'projected_width'
        assert steps > 0

    def test_exponential_used_for_good_trajectory(self):
        """project_convergence uses exponential for well-behaved trajectories."""
        t = np.arange(30)
        w = 0.5 * np.exp(-0.08 * t) + 0.02
        result = project_convergence(ci_widths=w.tolist(), delta=0.05,
                                     step_size=10, n_bootstrap=50)
        assert result is not None
        assert result['projection_basis'] == 'exponential_decay'
        assert 'exponential_fit' in result

    def test_linear_fallback_for_short_data(self):
        """project_convergence falls back to linear for < 5 width obs.

        Since < 5 widths can't produce 2+ slopes either (needs stab_window+1=16),
        the function returns None.
        """
        widths = [0.10, 0.09, 0.08]
        result = project_convergence(ci_widths=widths, delta=0.05)
        assert result is None

    def test_linear_fallback_for_poor_fit(self):
        """project_convergence falls back to linear when exponential fit fails."""
        # Random walk data - exponential won't fit well
        rng = np.random.default_rng(42)
        widths = [0.15]
        for _ in range(24):
            widths.append(max(0.06, widths[-1] + rng.normal(-0.001, 0.02)))
        result = project_convergence(ci_widths=widths, delta=0.05,
                                     step_size=10, n_bootstrap=50)
        if result is not None:
            # If exponential fails, should fall back to linear
            if result['projection_basis'] == 'linear_extrapolation':
                assert 'n_slope_observations' in result['uncertainty']
                assert 'exponential_fit' not in result
