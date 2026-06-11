"""
Post-hoc convergence analysis for optimal stopping algorithms.

This module provides functions to assess convergence properties of optimal stopping algorithms
by analyzing how stopping criteria evolve across different data collection scenarios.
"""

import pandas as pd
import numpy as np
import logging
import concurrent.futures
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional, Tuple, Dict, Any
from scipy.optimize import curve_fit as _scipy_curve_fit
from tqdm import tqdm
import contextlib
import io
import sys
import warnings

# Suppress all warnings before importing PyMC
warnings.filterwarnings('ignore')

import pymc as pm
import arviz as az

# Import GPU utilities and cleanup utilities
from . import gpu_utils
from . import cleanup_utils

# Import ordinal scoring utilities
from .ordinal_utils import (
    _ordinal_ci_adaptive,
    validate_ordinal_scores,
    check_ordinal_sparsity,
    determine_score_type,
    determine_score_type_standalone
)

# Import continuous scoring utilities
from .rule import _continuous_bounded_ci_adaptive

# Suppress PyMC logging and warnings
logging.getLogger('pymc').setLevel(logging.ERROR)
logging.getLogger('arviz').setLevel(logging.ERROR)
logging.getLogger('pytensor').setLevel(logging.ERROR)
logging.getLogger('aesara').setLevel(logging.ERROR)

# Additional suppression for PyMC-related modules
for logger_name in ['pymc', 'arviz', 'pytensor', 'aesara', 'numba', 'theano']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
    logging.getLogger(logger_name).propagate = False

# Additional warning suppression
warnings.filterwarnings('ignore', category=UserWarning, module='pymc')
warnings.filterwarnings('ignore', category=UserWarning, module='arviz')
warnings.filterwarnings('ignore', category=UserWarning, module='pytensor')
warnings.filterwarnings('ignore', category=UserWarning, module='aesara')
warnings.filterwarnings('ignore', message='.*effective sample size.*')
warnings.filterwarnings('ignore', message='.*rhat.*')
warnings.filterwarnings('ignore', message='.*ess.*')
warnings.filterwarnings('ignore', message='.*divergence.*')
warnings.filterwarnings('ignore', message='.*divergences.*')
warnings.filterwarnings('ignore', message='.*target_accept.*')
warnings.filterwarnings('ignore', message='.*reparameterize.*')
warnings.filterwarnings('ignore', message='.*smaller than 100.*')
warnings.filterwarnings('ignore', message='.*needed for reliable.*')
warnings.filterwarnings('ignore', message='.*There was.*divergence.*')
warnings.filterwarnings('ignore', message='.*There were.*divergence.*')
warnings.filterwarnings('ignore', message='.*after tuning.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pymc')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='arviz')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pytensor')

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stable companion metrics for convergence projection
# ---------------------------------------------------------------------------

def _simple_projection(n_trials_observed, ci_width, delta):
    """Additional trials needed assuming CI contracts as 1/sqrt(n).

    Conservative estimate (real convergence is often faster for hierarchical
    models). Quantitatively stable across item orderings (CV 0.03-0.09).
    """
    if ci_width <= delta or n_trials_observed <= 0 or delta <= 0:
        return 0.0
    ratio = ci_width / delta
    return n_trials_observed * (ratio ** 2 - 1)


def _trajectory_signal(exp_proj, simple_proj):
    """Convergence trajectory relative to 1/sqrt(n) baseline.

    Returns 'faster' (exponential projects < 0.5x the simple estimate),
    'slower' (> 2.0x), or 'on_pace'. Returns None when either input is
    unavailable or simple_proj is zero.
    """
    if exp_proj is None or simple_proj is None or simple_proj == 0:
        return None
    ratio = exp_proj / simple_proj
    if ratio < 0.5:
        return 'faster'
    elif ratio > 2.0:
        return 'slower'
    return 'on_pace'


# ---------------------------------------------------------------------------
# Exponential decay model: w(t) = a * exp(-b * t) + c
# ---------------------------------------------------------------------------

def _exp_decay(t, a, b, c):
    """Exponential decay model for CI width trajectories."""
    return a * np.exp(-b * t) + c


def _fit_exponential(ci_widths):
    """Fit exponential decay to CI width history.

    Returns (popt, r_squared) on success, or (None, None) on failure.
    Failure modes: curve_fit exception, R-squared < 0.7, constant input.
    """
    w = np.array(ci_widths, dtype=float)
    n = len(w)
    if n < 5:
        return None, None

    t = np.arange(n, dtype=float)

    # Initial guesses: amplitude from range, moderate decay, asymptote near final
    amp_guess = max(w[0] - w[-1], 1e-6)
    p0 = [amp_guess, 0.05, max(w[-1] * 0.5, 1e-10)]
    bounds = ([0, 1e-8, 0], [np.inf, 2.0, np.inf])

    try:
        popt, _ = _scipy_curve_fit(_exp_decay, t, w, p0=p0,
                                   bounds=bounds, maxfev=5000)
    except (RuntimeError, ValueError, TypeError):
        return None, None

    # R-squared quality gate
    fitted = _exp_decay(t, *popt)
    ss_res = np.sum((w - fitted) ** 2)
    ss_tot = np.sum((w - np.mean(w)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    if r_squared < 0.7:
        return None, None

    return popt, float(r_squared)


def _project_exponential(popt, n_obs, delta, step_size, max_steps):
    """Compute projection from fitted exponential model.

    Returns (steps, convergence_target, width_at_termination) where steps
    is in observation-step units (multiply by step_size for trials), matching
    the convention of _run_projection_loop.
    """
    a, b, c = popt

    if c < delta and a > 0:
        # Width will cross delta - solve a*exp(-b*t) + c = delta
        ratio = a / (delta - c)
        if ratio > 0:
            t_cross = (1.0 / b) * np.log(ratio)
            additional_obs = t_cross - n_obs
            steps = max(0, int(round(additional_obs)))
            if steps <= max_steps:
                width_at_cross = _exp_decay(t_cross, a, b, c)
                return steps, 'projected_width', float(width_at_cross)
            else:
                width_at_cap = float(_exp_decay(n_obs + max_steps, a, b, c))
                return max_steps, 'projected_capped', width_at_cap
    # Asymptote >= delta: trajectory plateaus above delta
    # Report steps until within 5% of asymptote
    if a > 0 and c > 0:
        target_width = c * 1.05
        excess = target_width - c  # = 0.05 * c
        if excess > 0:
            ratio = a / excess
            if ratio > 0:
                t_plateau = (1.0 / b) * np.log(ratio)
                additional_obs = max(0.0, t_plateau - n_obs)
                steps = int(round(additional_obs))
                if steps <= max_steps:
                    return steps, 'projected_slope_stabilisation', float(c)
    return 0, 'projected_slope_stabilisation', float(c)


def _run_projection_loop(width, slope, slope_of_slopes, delta, slope_threshold, max_steps):
    """Run the greedy linear extrapolation loop (fallback when exponential fit fails).

    Mirrors the slope stabilisation guard in rule.py (slope_slopes >= 0),
    with a small floating-point tolerance (>= -1e-12). Uses unclamped slope
    (no min(0, ...)) so diverging trajectories correctly hit 'capped'.

    Slope stabilisation triggers when either:
    (a) abs(slope) <= slope_threshold with Guard A passing, OR
    (b) slope crosses from negative to non-negative (sign reversal) with
        Guard A passing. Case (b) catches the physically correct stabilisation
        event when the discrete step overshoots the tight slope_threshold
        window around zero - a decelerating CI trajectory whose slope reaches
        zero has plateaued, which is what slope stabilisation means.

    Returns (steps, convergence_target, terminal_width) where convergence_target
    is 'projected_width', 'projected_slope_stabilisation', or 'projected_capped'.
    terminal_width is the projected CI width at the point of termination.
    """
    steps = 0
    next_width = width
    next_slope = slope
    while steps < max_steps:
        steps += 1
        prev_slope = next_slope
        next_slope = next_slope + slope_of_slopes
        next_width = max(0, next_width + next_slope)
        if next_width < delta:
            return steps, 'projected_width', next_width
        if slope_of_slopes >= -1e-12:
            if abs(next_slope) <= slope_threshold:
                return steps, 'projected_slope_stabilisation', next_width
            # Sign reversal: slope crossed zero but overshot the threshold
            # window. CI width has plateaued - this is slope stabilisation.
            if prev_slope < 0 and next_slope >= 0:
                return steps, 'projected_slope_stabilisation', next_width
    return steps, 'projected_capped', next_width


def project_convergence(
    ci_widths,
    ci_slopes=None,
    delta=0.05,
    slope_threshold=0.00001,
    step_size=10,
    max_steps=200,
    stab_window=15,
    n_bootstrap=200,
):
    """Project how many additional trials a non-converged grouping would need.

    Primary model: exponential decay w(t) = a*exp(-b*t) + c, fitted via
    scipy curve_fit. Falls back to greedy linear extrapolation when the
    exponential fit fails (< 5 observations or R-squared < 0.7). The
    linear fallback mirrors the slope stabilisation logic in rule.py,
    including Guard A and sign-change detection.

    The projection terminates via one of three outcomes, indicated by
    ``convergence_target`` in the returned dict:

    - ``'projected_width'``: CI width is projected to drop below ``delta``.
      ``projected_additional_trials`` is the estimated trials to reach the
      width threshold. ``projected_width_at_termination`` will be < delta.

    - ``'projected_slope_stabilisation'``: The CI width trajectory is
      projected to *plateau* before reaching ``delta`` - i.e., the rate of
      narrowing decelerates to zero. ``projected_additional_trials`` is the
      estimated trials until the plateau, and
      ``projected_width_at_termination`` shows the projected width at that
      point (typically still well above delta). This indicates that
      additional trials beyond this point would yield diminishing returns.

    - ``'projected_capped'``: Neither width convergence nor slope
      stabilisation occurred within ``max_steps``. The trajectory may be
      diverging, near-flat, or simply too far from either threshold.

    Args:
        ci_widths: CI width history from stabilization_history.
        ci_slopes: CI slope history (may be empty for ordinal pathway;
            slopes will be computed internally from ci_widths).
        delta: Width convergence threshold (e.g. delta_cap for group level).
        slope_threshold: Slope stabilisation threshold. Callers should apply
            conservatism adjustment (e.g. CI_delta / conservatism) when the
            grouping's performance is below the low-performance threshold,
            matching the logic in rule.py.
        step_size: Trials per projection step (e.g. reanalysis_interval).
        max_steps: Cap on projected steps.
        stab_window: Window size for polyfit slope computation.
        n_bootstrap: Number of residual-bootstrap iterations.

    Returns:
        dict with point estimate + uncertainty, or None if insufficient data.
        Key fields include:

        - ``projected_additional_trials``: Exponential or linear projection.
          Sensitive to item ordering (CV 0.5-3.0 across shuffled orderings).
          Treat as a rough order-of-magnitude guide, not a precise planning
          target.
        - ``simple_proj_additional_trials``: Conservative projection assuming
          CI contracts as 1/sqrt(n). More stable across orderings (CV 0.03-
          0.09). Use for quantitative planning.
        - ``trajectory_signal``: ``'faster'``, ``'on_pace'``, or ``'slower'``
          relative to the 1/sqrt(n) baseline. Captures the qualitative
          trajectory information from the exponential model without the
          unstable extrapolation. None when the convergence target is
          slope stabilisation or capped (comparison with 1/sqrt(n) is not
          meaningful), when already converged, or when the simple
          projection is zero.
        - ``proximity_ratio``: Current CI width / delta. Immediately
          interpretable progress indicator (e.g. 2.4 means "2.4x away").
    """
    ci_widths = list(ci_widths) if ci_widths else []
    ci_slopes = list(ci_slopes) if ci_slopes else []

    # --- Early check: already converged (no slope data needed) ---
    if ci_widths and ci_widths[-1] < delta:
        final_width = ci_widths[-1]
        proximity_ratio = final_width / delta if delta > 0 else float('inf')
        return {
            'projected_additional_steps': 0,
            'projected_additional_trials': 0,
            'simple_proj_additional_trials': 0.0,
            'trajectory_signal': None,
            'proximity_ratio': proximity_ratio,
            'convergence_target': 'projected_width',
            'projected_width_at_termination': final_width,
            'plateau_reached': False,
            'capped': False,
            'final_width': final_width,
            'final_slope': ci_slopes[-1] if ci_slopes else 0.0,
            'projection_basis': 'linear_extrapolation',
            'uncertainty': {
                'ci_trials_80': [0, 0],
                'ci_trials_50': [0, 0],
                'bootstrap_skipped': True,
                'n_slope_observations': len(ci_slopes),
                'trajectory_rmse': 0.0,
                'confidence_level': 'high',
            },
        }

    # --- Compute slopes from widths if not provided ---
    if len(ci_slopes) < 2 and len(ci_widths) >= stab_window + 1:
        ci_slopes = []
        for i in range(len(ci_widths) - stab_window + 1):
            window = ci_widths[i:i + stab_window]
            slope = np.polyfit(range(stab_window), window, 1)[0]
            ci_slopes.append(float(slope))

    # --- Minimum data guard ---
    if len(ci_slopes) < 2:
        return None

    # --- Derive point estimate inputs ---
    final_width = ci_widths[-1]
    final_slope = ci_slopes[-1]

    # Slope-of-slopes from last 3 (or all if fewer) slopes
    # n_for_sos >= 2 is guaranteed by the len(ci_slopes) >= 2 guard above
    n_for_sos = min(3, len(ci_slopes))
    recent = ci_slopes[-n_for_sos:]
    slope_of_slopes = float(np.polyfit(range(n_for_sos), recent, 1)[0])

    # --- Already converged? ---
    proximity_ratio = final_width / delta if delta > 0 else float('inf')
    if final_width < delta:
        return {
            'projected_additional_steps': 0,
            'projected_additional_trials': 0,
            'simple_proj_additional_trials': 0.0,
            'trajectory_signal': None,
            'proximity_ratio': proximity_ratio,
            'convergence_target': 'projected_width',
            'projected_width_at_termination': final_width,
            'plateau_reached': False,
            'capped': False,
            'final_width': final_width,
            'final_slope': final_slope,
            'projection_basis': 'linear_extrapolation',
            'uncertainty': {
                'ci_trials_80': [0, 0],
                'ci_trials_50': [0, 0],
                'bootstrap_skipped': True,
                'n_slope_observations': len(ci_slopes),
                'trajectory_rmse': 0.0,
                'confidence_level': 'high',
            },
        }

    # --- Stable companion: 1/sqrt(n) projection ---
    n_obs = len(ci_widths)
    n_trials_observed = n_obs * step_size
    simple_proj = _simple_projection(n_trials_observed, final_width, delta)

    # --- Try exponential decay model first ---
    exp_popt, exp_r2 = _fit_exponential(ci_widths)

    if exp_popt is not None:
        return _project_convergence_exponential(
            ci_widths, exp_popt, exp_r2, n_obs, delta, step_size,
            max_steps, n_bootstrap, final_width, final_slope, proximity_ratio,
            simple_proj,
        )

    # --- Fallback: linear extrapolation ---
    return _project_convergence_linear(
        ci_widths, ci_slopes, delta, slope_threshold, step_size, max_steps,
        n_bootstrap, final_width, final_slope, slope_of_slopes, proximity_ratio,
        simple_proj,
    )


def _project_convergence_exponential(
    ci_widths, popt, r_squared, n_obs, delta, step_size, max_steps,
    n_bootstrap, final_width, final_slope, proximity_ratio,
    simple_proj=0.0,
):
    """Build projection result using exponential decay model."""
    a, b, c = popt

    # Point estimate
    steps, target, terminal_width = _project_exponential(
        popt, n_obs, delta, step_size, max_steps
    )
    point_trials = steps * step_size
    capped = (target == 'projected_capped')

    # Residual bootstrap on exponential fit
    w = np.array(ci_widths, dtype=float)
    t = np.arange(n_obs, dtype=float)
    fitted_vals = _exp_decay(t, *popt)
    residuals = w - fitted_vals

    ci_80 = [point_trials, point_trials]
    ci_50 = [point_trials, point_trials]
    some_capped = False
    bootstrap_tight = False
    bootstrap_skipped = n_obs < 7 or n_bootstrap == 0  # need headroom beyond the 5-obs fit minimum

    if not bootstrap_skipped:
        rng = np.random.default_rng()
        boot_trials = []
        for _ in range(n_bootstrap):
            resampled = rng.choice(residuals, size=n_obs, replace=True)
            synthetic_w = fitted_vals + resampled
            # Refit on bootstrap sample
            boot_popt, boot_r2 = _fit_exponential(synthetic_w)
            if boot_popt is not None:
                boot_steps, _, _ = _project_exponential(
                    boot_popt, n_obs, delta, step_size, max_steps
                )
                boot_trials.append(boot_steps * step_size)
            else:
                boot_trials.append(point_trials)  # fallback to point estimate

        boot_trials = np.array(boot_trials)
        ci_80 = [int(np.percentile(boot_trials, 10)), int(np.percentile(boot_trials, 90))]
        ci_50 = [int(np.percentile(boot_trials, 25)), int(np.percentile(boot_trials, 75))]
        some_capped = bool(np.any(boot_trials >= max_steps * step_size))
        ci_80_width = ci_80[1] - ci_80[0]
        bootstrap_tight = ci_80_width < 2 * point_trials if point_trials > 0 else ci_80_width == 0

    # RMSE for diagnostics
    trajectory_rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # Confidence for exponential projections
    if r_squared >= 0.9 and n_obs >= 10 and bootstrap_tight and not capped:
        confidence_level = 'high'
    elif n_obs >= 5 and not capped:
        confidence_level = 'moderate'
    else:
        confidence_level = 'low'

    # plateau_reached: True when the trajectory has already plateaued
    # (steps=0 with slope_stab). Distinguishes "0 more trials needed because
    # width < delta" (already converged) from "0 more trials needed because
    # the CI has stopped narrowing" (plateau above delta).
    plateau_reached = (steps == 0 and target == 'projected_slope_stabilisation')

    result = {
        'projected_additional_steps': steps,
        'projected_additional_trials': point_trials,
        'simple_proj_additional_trials': simple_proj,
        'trajectory_signal': (
            _trajectory_signal(point_trials, simple_proj)
            if target == 'projected_width' else None
        ),
        'proximity_ratio': proximity_ratio,
        'convergence_target': target,
        'projected_width_at_termination': terminal_width,
        'plateau_reached': plateau_reached,
        'capped': capped,
        'final_width': final_width,
        'final_slope': final_slope,
        'projection_basis': 'exponential_decay',
        'exponential_fit': {
            'a': float(a), 'b': float(b), 'c': float(c),
            'r_squared': r_squared,
        },
        'uncertainty': {
            'ci_trials_80': ci_80,
            'ci_trials_50': ci_50,
            'bootstrap_skipped': bootstrap_skipped,
            'n_width_observations': n_obs,
            'trajectory_rmse': trajectory_rmse,
            'confidence_level': confidence_level,
        },
    }
    if some_capped:
        result['uncertainty']['some_bootstrap_capped'] = True
    return result


def _project_convergence_linear(
    ci_widths, ci_slopes, delta, slope_threshold, step_size, max_steps,
    n_bootstrap, final_width, final_slope, slope_of_slopes, proximity_ratio,
    simple_proj=0.0,
):
    """Build projection result using linear extrapolation (fallback)."""
    # Point estimate
    steps, target, terminal_width = _run_projection_loop(
        final_width, final_slope, slope_of_slopes, delta, slope_threshold, max_steps
    )
    point_trials = steps * step_size
    capped = (target == 'projected_capped')

    # Uncertainty: residual bootstrap on slope history
    slopes_arr = np.array(ci_slopes)
    n_slopes = len(slopes_arr)
    xs = np.arange(n_slopes)

    coeffs = np.polyfit(xs, slopes_arr, 1)
    fitted = np.polyval(coeffs, xs)
    residuals = slopes_arr - fitted

    trajectory_rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mean_slope = float(np.mean(slopes_arr))
    rmse_normalised = trajectory_rmse / max(abs(mean_slope), 1e-8)

    ci_80 = [point_trials, point_trials]
    ci_50 = [point_trials, point_trials]
    some_capped = False
    bootstrap_tight = False
    bootstrap_skipped = n_slopes < 3 or n_bootstrap == 0

    if not bootstrap_skipped:
        rng = np.random.default_rng()
        boot_trials = []
        for _ in range(n_bootstrap):
            resampled_residuals = rng.choice(residuals, size=n_slopes, replace=True)
            synthetic_slopes = fitted + resampled_residuals

            boot_final_slope = synthetic_slopes[-1]
            n_for_sos_boot = min(3, len(synthetic_slopes))
            boot_recent = synthetic_slopes[-n_for_sos_boot:]
            if n_for_sos_boot >= 2:
                boot_sos = float(np.polyfit(range(n_for_sos_boot), boot_recent, 1)[0])
            else:
                boot_sos = 0.0

            boot_steps, _, _ = _run_projection_loop(
                final_width, boot_final_slope, boot_sos, delta, slope_threshold, max_steps
            )
            boot_trials.append(boot_steps * step_size)

        boot_trials = np.array(boot_trials)
        ci_80 = [int(np.percentile(boot_trials, 10)), int(np.percentile(boot_trials, 90))]
        ci_50 = [int(np.percentile(boot_trials, 25)), int(np.percentile(boot_trials, 75))]
        some_capped = bool(np.any(boot_trials >= max_steps * step_size))
        ci_80_width = ci_80[1] - ci_80[0]
        bootstrap_tight = ci_80_width < 2 * point_trials if point_trials > 0 else ci_80_width == 0

    # Confidence for linear fallback
    converging = final_slope < 0
    if (converging and n_slopes >= 8 and rmse_normalised < 1.0
            and bootstrap_tight and not capped):
        confidence_level = 'high'
    elif (converging and n_slopes >= 4 and rmse_normalised < 2.0
            and not capped):
        confidence_level = 'moderate'
    else:
        confidence_level = 'low'

    plateau_reached = (steps == 0 and target == 'projected_slope_stabilisation')

    result = {
        'projected_additional_steps': steps,
        'projected_additional_trials': point_trials,
        'simple_proj_additional_trials': simple_proj,
        'trajectory_signal': (
            _trajectory_signal(point_trials, simple_proj)
            if target == 'projected_width' else None
        ),
        'proximity_ratio': proximity_ratio,
        'convergence_target': target,
        'projected_width_at_termination': terminal_width,
        'plateau_reached': plateau_reached,
        'capped': capped,
        'final_width': final_width,
        'final_slope': final_slope,
        'projection_basis': 'linear_extrapolation',
        'uncertainty': {
            'ci_trials_80': ci_80,
            'ci_trials_50': ci_50,
            'bootstrap_skipped': bootstrap_skipped,
            'n_slope_observations': n_slopes,
            'trajectory_rmse': trajectory_rmse,
            'confidence_level': confidence_level,
        },
    }
    if some_capped:
        result['uncertainty']['some_bootstrap_capped'] = True
    return result


@contextlib.contextmanager
def suppress_all_output():
    """Context manager that suppresses all stdout, stderr, and warnings from PyMC sampling"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Temporarily redirect stdout and stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            # Create string buffers to capture output
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer
            yield
        finally:
            # Restore original stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            # Log any captured output to file instead of console
            stdout_content = stdout_buffer.getvalue()
            stderr_content = stderr_buffer.getvalue()
            if stdout_content.strip() or stderr_content.strip():
                logger = logging.getLogger('optstop.convergence.sampling_output')
                if stdout_content.strip():
                    logger.debug(f"PyMC stdout: {stdout_content.strip()}")
                if stderr_content.strip():
                    logger.debug(f"PyMC stderr: {stderr_content.strip()}")

def _log_mcmc_diagnostics(trace, diag_logger, context: str) -> None:
    """Log MCMC diagnostics (divergences, ESS, R-hat) from a trace.

    Logs at WARNING level if any diagnostic is concerning, otherwise DEBUG.
    Never raises - diagnostic extraction must not crash the pipeline.
    """
    try:
        n_div = int(trace.sample_stats['diverging'].values.sum()) if hasattr(trace, 'sample_stats') else 0
        ess_vals = az.ess(trace)
        rhat_vals = az.rhat(trace)
        ess_min = float(ess_vals.to_array().min().item())
        rhat_max = float(rhat_vals.to_array().max().item())
        if n_div > 0 or ess_min < 100 or rhat_max > 1.05:
            diag_logger.warning(
                f"MCMC [{context}]: divergences={n_div}, ess_min={ess_min:.0f}, rhat_max={rhat_max:.4f}"
            )
        else:
            diag_logger.debug(
                f"MCMC [{context}]: divergences=0, ess_min={ess_min:.0f}, rhat_max={rhat_max:.4f}"
            )
    except Exception:
        pass


# --- Helper: Adaptive Beta CI ---
def _beta_ci_adaptive(successes, trials, cred_level=0.97, conservatism=5.0,
                     low_perf_threshold=0.01, base_strength=2, samples=10000):
    if trials == 0:
        return 0.0, 1.0, 1.0
    p_hat = successes / trials
    if p_hat < low_perf_threshold:
        alpha_boost = conservatism
        prior_scaling = base_strength * np.exp(-trials / (10 * conservatism))
        alpha_prior = max(prior_scaling * p_hat * alpha_boost, 0.5)
        beta_prior = max(prior_scaling * (1 - p_hat), 0.5)
    else:
        prior_scaling = base_strength * np.exp(-trials / 10)
        alpha_prior = max(prior_scaling * p_hat, 0.5)
        beta_prior = max(prior_scaling * (1 - p_hat), 0.5)
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + (trials - successes)
    draws = np.random.beta(alpha_post, beta_post, samples)
    lo, hi = np.quantile(draws, [(1 - cred_level) / 2, 1 - (1 - cred_level) / 2])
    if p_hat < low_perf_threshold:
        effective_width = (hi - lo) * conservatism
    else:
        effective_width = hi - lo
    return lo, hi, effective_width

# Note: Column validation is now done at the beginning of convergence_posthoc function

def _validate_params(params):
    logger = logging.getLogger('optstop.convergence')
    for key in ['draws', 'tune', 'stab_window', 'rep_batch_size', 'pymc_refresh_every', 'item_seqs', 'epoch_seqs']:
        if key in params and (not isinstance(params[key], int) or params[key] <= 0):
            logger.error(f"Parameter '{key}' must be a positive integer. Got: {params[key]}")
            raise ValueError(f"Parameter '{key}' must be a positive integer. Got: {params[key]}")
    if 'delta_item' in params and params['delta_item'] <= 0:
        logger.error("delta_item must be positive.")
        raise ValueError("delta_item must be positive.")
    if 'delta_cap' in params and params['delta_cap'] <= 0:
        logger.error("delta_cap must be positive.")
        raise ValueError("delta_cap must be positive.")
    logger.info(f"Parameters validated: {params}")

def _worker_initializer_convergence(worker_dir, gpu_id=None, suppress_output=True):
    """Initialize worker process with clean PyTensor environment for convergence analysis.

    Args:
        worker_dir: Base directory for PyTensor compilation
        gpu_id: GPU ID to assign to this worker, or None for CPU-only
        suppress_output: Whether to suppress output (default: True)
    """
    import os

    # Create unique worker temporary directory with robust cleanup
    unique_worker_dir = cleanup_utils.create_worker_temp_dir('optstop_convergence')

    # Register enhanced cleanup for worker process
    cleanup_utils.register_worker_cleanup(unique_worker_dir, 'optstop.worker_convergence')

    # CRITICAL: Set environment variables BEFORE any PyTensor import
    # Determine device configuration based on gpu_id parameter
    if gpu_id is not None:
        # GPU-enabled worker
        os.environ['PYTENSOR_FLAGS'] = f'compiledir={unique_worker_dir},device=gpu,floatX=float32'
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # Configure JAX to use specific GPU
        os.environ['JAX_CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device_type = f'GPU {gpu_id}'
    else:
        # No specific GPU assigned - let JAX/PyTensor auto-detect available devices
        # Do NOT set CUDA_VISIBLE_DEVICES to empty string as this disables GPU entirely
        # Note: PyTensor device=cpu is kept since PyMC+numpyro uses JAX, not PyTensor for sampling
        # But JAX is allowed to auto-detect GPU by not setting JAX_PLATFORM_NAME or CUDA_VISIBLE_DEVICES
        os.environ['PYTENSOR_FLAGS'] = f'compiledir={unique_worker_dir},floatX=float32'
        # Let JAX auto-detect (don't force CPU-only mode)
        # os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Commented out to allow GPU detection
        # os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Commented out to allow GPU detection
        device_type = 'Auto-detect (GPU if available)'

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # Additional PyTensor isolation environment variables
    os.environ['PYTENSOR_FLAGS'] += ',optimizer=fast_compile,openmp=False'

    # Force PyTensor to use our compiledir by setting config directly after import
    # This handles cases where PyTensor might be imported later
    try:
        import pytensor
        pytensor.config.compiledir = unique_worker_dir
        # Remove force_compile as it's not a valid PyTensor config
        # Clear any existing module cache
        if hasattr(pytensor.link.c.basic, '_module_cache'):
            pytensor.link.c.basic._module_cache = None
        # Clear any global cache
        if hasattr(pytensor.link.c.cmodule, '_module_cache'):
            pytensor.link.c.cmodule._module_cache = None
    except Exception as e:
        # If this fails, we still have environment variables as fallback
        pass

    # Debug logging
    logger = logging.getLogger('optstop.worker_convergence')
    if not suppress_output:
        logger.info(f"Convergence worker {os.getpid()} using {device_type}, PyTensor compiledir: {unique_worker_dir}")

# --- Helper: Process a single grouping-task ---
def _process_grouping_with_init_convergence(task_args: Tuple[Any, pd.DataFrame, Dict[str, Any], str], worker_args: Tuple[str, Optional[int], bool]) -> Dict[str, Any]:
    """Wrapper function that initializes worker and then processes convergence grouping."""
    # Initialize worker with GPU assignment
    _worker_initializer_convergence(*worker_args)
    # Process the actual task
    return _process_grouping(task_args)

def _process_grouping(args):
    # Environment variables are now set by the worker initializer
    import sys
    import io
    import os
    import logging

    # Verify environment is still set in worker process
    logger = logging.getLogger('optstop.process')
    logger.info(f"Process {os.getpid()} PYTENSOR_FLAGS: {os.environ.get('PYTENSOR_FLAGS', 'NOT_SET')}")

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        (pid, df_part, params, score_column) = args

        logger = logging.getLogger('optstop.convergence')
        delta_item = params.get('delta_item', 0.05)
        delta_cap = params.get('delta_cap', 0.05)
        CI_delta = params.get('CI_delta', 0.00001)
        cred_level = params.get('cred_level', 0.97)
        conservatism = params.get('conservatism', 5)
        low_perf_threshold = params.get('low_performance_threshold', 0.01)
        rep_batch_size = params.get('rep_batch_size', 1)
        pymc_refresh_every = params.get('pymc_refresh_every', 2)
        stab_window = params.get('stab_window', 15)
        item_seqs = params.get('item_seqs', 20)
        epoch_seqs = params.get('epoch_seqs', 20)

        # Ordinal and continuous-specific parameters
        ordinal_tasks = params.get('ordinal_tasks', None)
        continuous_tasks = params.get('continuous_tasks', None)
        ordinal_max_score = params.get('ordinal_max_score', 10)
        ordinal_inference = params.get('ordinal_inference', 'modal')
        ordinal_model_type = params.get('ordinal_model_type', 'ordered_logistic')
        entropy_threshold = params.get('entropy_threshold', 0.8)
        entropy_convergence_threshold = params.get('entropy_convergence_threshold', 0.10)
        prior_mu = params.get('prior_mu', 0.0)
        prior_sigma = params.get('prior_sigma')  # None means use pathway-specific default

        # Determine score type for this grouping
        grouping_name = df_part['grouping'].iloc[0] if 'grouping' in df_part.columns else str(pid)
        score_type, bounds = determine_score_type_standalone(
            grouping_name,
            ordinal_tasks=ordinal_tasks,
            continuous_tasks=continuous_tasks,
            upper_bound=ordinal_max_score
        )

        logger.info(f"Processing convergence for grouping {pid} ('{grouping_name}') with {score_type} scoring" +
                   (f" (inference: {ordinal_inference})" if score_type == 'ordinal' else ""))

        # Fail fast on groupings with no usable (non-NaN) scores: fitting MCMC on
        # all-NaN data wastes minutes and yields meaningless results. The worker's
        # except handler records this in the 'error' column for the caller.
        _numeric_scores = pd.to_numeric(df_part[score_column], errors='coerce')
        if _numeric_scores.notna().sum() == 0:
            raise ValueError(
                f"Grouping '{grouping_name}' has no valid (non-NaN) scores - "
                f"cannot perform convergence analysis"
            )

        # Validate ordinal scores if applicable
        if score_type == 'ordinal':
            validate_ordinal_scores(
                df_part[score_column].values,
                ordinal_max_score,
                grouping_name
            )

        # Import ordinal functions if needed
        if score_type == 'ordinal':
            from .ordinal_model import _ordinal_entropy_ci_adaptive, _ordinal_hybrid_stopping_criterion

        # CRITICAL: Initialize JAX/PyMC backend BEFORE any model creation
        # Detect GPU availability based on worker's actual environment configuration
        # (set by worker initializer based on GPU assignment)
        gpu_available, gpu_backend, gpu_info = gpu_utils.check_gpu_availability()

        # Configure JAX and PyMC for GPU computation if available
        if gpu_available and gpu_backend == 'jax-gpu':
            try:
                import jax
                # Explicitly configure JAX for GPU
                jax.config.update('jax_platform_name', 'gpu')
                # Verify JAX sees GPU devices
                devices = jax.devices()
                gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
                if gpu_devices:
                    logger.info(f"Worker JAX initialized with GPU devices: {gpu_devices}")
                    # Force JAX to use the first available GPU device
                    import jax.numpy as jnp
                    # Test JAX GPU functionality with a simple operation
                    test_array = jnp.array([1.0, 2.0, 3.0])
                    result = jnp.sum(test_array)  # This should execute on GPU
                    logger.info(f"Worker JAX GPU test successful: {result}")

                    # Configure PyMC to use JAX backend (don't set PyTensor device)
                    import os
                    os.environ['PYMC_BACKEND'] = 'jax'

                    # JAX handles GPU automatically - no PyTensor configuration needed
                else:
                    logger.warning("Worker JAX GPU setup failed - falling back to CPU")
                    gpu_available = False
                    gpu_backend = 'cpu'
            except ImportError:
                logger.warning("JAX not available - using CPU backend")
                gpu_available = False
                gpu_backend = 'cpu'
            except Exception as e:
                logger.warning(f"JAX GPU setup failed: {e} - using CPU backend")
                gpu_available = False
                gpu_backend = 'cpu'

        # Get number of parallel tasks from params if available, otherwise assume 1
        # In convergence mode, each worker processes one grouping, so num_parallel_tasks = 1 per worker
        num_parallel_tasks = params.get('_num_parallel_tasks', 1)

        sampling_kwargs = gpu_utils.get_sampling_kwargs(
            params=params,
            gpu_available=gpu_available,
            gpu_backend=gpu_backend,
            num_parallel_tasks=num_parallel_tasks,
            auto_decide=True
        )

        # Log GPU status for this worker
        if gpu_available and sampling_kwargs.get('nuts_sampler') == 'numpyro':
            logger.info(f"Worker processing grouping {pid} with GPU acceleration ({gpu_backend}, chains={sampling_kwargs.get('chains', 1)})")
        elif gpu_available:
            logger.info(f"Worker processing grouping {pid} with GPU acceleration ({gpu_backend})")
        else:
            logger.info(f"Worker processing grouping {pid} with CPU-only (chains={sampling_kwargs.get('chains', 4)})")

        # Note: random_seed is now passed directly to PyMC via sampling_kwargs['random_seed']
        # (configured in gpu_utils.get_sampling_kwargs)

        logger.info(f"Processing grouping {pid}")

        # OPTIMIZATION: Create shared ordinal model cache for this entire grouping
        # This cache will be reused across all item_seqs and epoch_seqs iterations
        # Eliminates redundant model compilations (was recompiling 9-12x per grouping)
        ordinal_model_cache_shared = {}  # Worker-local, shared across all sequences

        # Initialize all output variables
        items_fin_CI_widths = []
        items_fin_CI_slopes = []
        items_fin_slope_slopes = []
        items_fin_scores = []
        items_shortfalls = []
        epochs_fin_CI_widths = []
        epochs_fin_CI_slopes = []
        epochs_fin_slope_slopes = []
        epochs_fin_scores = []
        epochs_shortfalls = []
        agg_sample_id_performance = []
        theta_lo = theta_hi = theta_width = None
        n_items_used = 0
        percent_items_used = 0
        avg_reps_per_item = 0
        task_performance = None
        mean_sample_id_performance = None
        var_sample_id_performance = None

        for item_seq in range(item_seqs):
            item_summaries = []
            used_reps_dfs = []
            CI_record = []
            CI_slopes_hist = []
            CI_slope_slopes = []
            item_scores = []
            item_shortfalls = []
            initial_perf = df_part[score_column].mean()
            current_conservatism = conservatism if initial_perf < low_perf_threshold else 1.0
            if score_type == 'binary':
                binary_sigma = prior_sigma if prior_sigma is not None else 1.5
                with pm.Model() as model:
                    mu_group = pm.Normal("mu_group", mu=prior_mu, sigma=binary_sigma)
                    sigma_group = pm.Exponential("sigma_group", lam=1.0)
                    successes_data = pm.Data("successes", np.array([0]))
                    n_items = pm.Data("n_items", np.array(1, dtype="int64"))
                    trials_data = pm.Data("trials", np.array([1]))
                    z = pm.Normal("z", mu=0, sigma=1, shape=n_items)
                    mu_item = pm.Deterministic("mu_item", mu_group + z * sigma_group)
                    mu_item_clipped = pm.Deterministic("mu_item_clipped", pm.math.clip(mu_item, -6.0, 6.0))
                    Theta = pm.Deterministic("Theta", pm.math.sigmoid(mu_item_clipped))
                    pm.Binomial("obs", n=trials_data, p=Theta, observed=successes_data)
            item_ids = list(df_part['sample_id_num'].unique())
            np.random.shuffle(item_ids)
            for item_idx, item_id in enumerate(item_ids):
                df_item = df_part[df_part['sample_id_num'] == item_id].sort_values('epoch_num')
                sample_ID_performances = []
                for epoch_seq in range(epoch_seqs):
                    if epoch_seqs >= 1:
                        # Use sample to shuffle the DataFrame instead of set_index/reindex
                        df_item_shuffled = df_item.sample(frac=1.0, random_state=np.random.randint(0, 10000)).reset_index(drop=True)

                        # Initialize tracking variables based on score type
                        if score_type == 'binary':
                            successes = 0
                            trials = 0
                        elif score_type in ['continuous_01', 'continuous_bounded']:
                            accumulated_scores = []
                        else:  # ordinal
                            accumulated_scores = []
                            entropy_history_epoch = []  # Track entropy for this epoch sequence (must be fresh)
                            # OPTIMIZATION: Use shared cache instead of creating fresh cache
                            # ordinal_model_cache_epoch = {}  # OLD: Fresh cache per epoch sequence

                        used_reps = []
                        epoch_CI_widths = []
                        epoch_CI_slopes = []
                        epoch_slope_slopes = []
                        epoch_scores = []
                        epoch_shortfalls = []
                        slope = None  # Ensure slope is always defined
                        slope_slopes = None

                        for start in range(0, len(df_item_shuffled), rep_batch_size):
                            batch = df_item_shuffled.iloc[start:start+rep_batch_size]
                            used_reps.extend(batch.itertuples(index=False))

                            # Compute CI based on score type
                            if score_type == 'binary':
                                successes += batch[score_column].sum()
                                trials += len(batch)
                                _, _, width = _beta_ci_adaptive(
                                    successes, trials,
                                    cred_level=cred_level,
                                    conservatism=current_conservatism,
                                    low_perf_threshold=low_perf_threshold
                                )
                                epoch_CI_widths.append(width)
                                curr_perf_estimate = successes / trials
                            elif score_type in ['continuous_01', 'continuous_bounded']:
                                accumulated_scores.extend(batch[score_column].values)
                                cont_lower = bounds.get('lower', 0.0)
                                cont_upper = bounds.get('upper', 1.0)
                                _, _, width = _continuous_bounded_ci_adaptive(
                                    np.array(accumulated_scores),
                                    lower_bound=cont_lower,
                                    upper_bound=cont_upper,
                                    cred_level=cred_level,
                                    conservatism=current_conservatism,
                                    low_perf_threshold=low_perf_threshold
                                )
                                # Normalize width to [0,1] for consistent comparison with delta_item
                                width = width / (cont_upper - cont_lower)
                                epoch_CI_widths.append(width)
                                curr_perf_estimate = (np.mean(accumulated_scores) - cont_lower) / (cont_upper - cont_lower)
                            else:  # ordinal
                                accumulated_scores.extend(batch[score_column].values)

                                # Use appropriate inference method
                                if ordinal_inference == 'modal':
                                    _, _, width = _ordinal_ci_adaptive(
                                        np.array(accumulated_scores),
                                        ordinal_max_score=ordinal_max_score,
                                        cred_level=cred_level,
                                        conservatism=current_conservatism,
                                        low_perf_threshold=low_perf_threshold
                                    )
                                elif ordinal_inference == 'entropy':
                                    _, _, width, _ = _ordinal_entropy_ci_adaptive(
                                        np.array(accumulated_scores),
                                        ordinal_max_score=ordinal_max_score,
                                        cred_level=cred_level,
                                        conservatism=current_conservatism,
                                        low_perf_threshold=low_perf_threshold,
                                        model_cache=ordinal_model_cache_shared,  # OPTIMIZATION: Use shared cache
                                        compute_kwargs=sampling_kwargs
                                    )
                                else:  # hybrid
                                    should_stop, reason, diagnostics = _ordinal_hybrid_stopping_criterion(
                                        np.array(accumulated_scores),
                                        ordinal_max_score=ordinal_max_score,
                                        delta_item=delta_item,
                                        cred_level=cred_level,
                                        entropy_history=entropy_history_epoch,
                                        entropy_threshold=entropy_threshold,
                                        entropy_convergence_threshold=entropy_convergence_threshold,
                                        conservatism=current_conservatism,
                                        low_perf_threshold=low_perf_threshold,
                                        model_cache=ordinal_model_cache_shared,  # OPTIMIZATION: Use shared cache
                                        compute_kwargs=sampling_kwargs
                                    )
                                    # Use modal width for convergence tracking
                                    width = diagnostics.get('modal_width', 1.0)

                                epoch_CI_widths.append(width)
                                # For ordinal, use normalized performance for conservatism check
                                curr_perf_estimate = np.mean(accumulated_scores) / ordinal_max_score
                            if len(epoch_CI_widths) >= stab_window:
                                recent_widths = epoch_CI_widths[-stab_window:]
                                slope = np.polyfit(range(len(recent_widths)), recent_widths, 1)[0]
                                epoch_CI_slopes.append(slope)
                                slope_threshold = CI_delta / current_conservatism if curr_perf_estimate < low_perf_threshold else CI_delta
                                if (len(epoch_CI_slopes) >= 3):
                                    recent_slopes = epoch_CI_slopes[-3:]
                                    slope_slopes = np.polyfit(range(len(recent_slopes)), recent_slopes, 1)[0]
                                    epoch_slope_slopes.append(slope_slopes)
                                # Only check slope/slopes if they are set
                                if (width < delta_item) or (slope is not None and slope_slopes is not None and abs(slope) <= slope_threshold and len(epoch_CI_slopes) >= 4 and slope_slopes >= 0):
                                    epoch_scores.append(1)
                                else:
                                    epoch_scores.append(0)
                            else:
                                slope = None
                                slope_slopes = None
                            if (start + rep_batch_size >= len(df_item_shuffled)):
                                slope_threshold = CI_delta / current_conservatism if curr_perf_estimate < low_perf_threshold else CI_delta
                                if (epoch_seq == epoch_seqs - 1) and (item_seq == item_seqs - 1):
                                    agg_sample_id_performance.append(curr_perf_estimate)
                                if epoch_scores and epoch_scores[-1] == 1:
                                    epoch_shortfalls.append(0)
                                else:
                                    max_shortfall_cap = 200
                                    projection = project_convergence(
                                        ci_widths=epoch_CI_widths,
                                        ci_slopes=epoch_CI_slopes,
                                        delta=delta_item,
                                        slope_threshold=slope_threshold,
                                        step_size=rep_batch_size,
                                        max_steps=max(1, max_shortfall_cap // max(1, rep_batch_size)),
                                        stab_window=stab_window,
                                        n_bootstrap=0,
                                    )
                                    if projection is not None:
                                        epoch_shortfalls.append(projection['projected_additional_trials'])
                                    else:
                                        epoch_shortfalls.append(max_shortfall_cap)
                        epochs_fin_CI_widths.append(epoch_CI_widths[-1] if epoch_CI_widths else 0)
                        epochs_fin_CI_slopes.append(epoch_CI_slopes[-1] if epoch_CI_slopes else 0)
                        epochs_fin_slope_slopes.append(epoch_slope_slopes[-1] if epoch_slope_slopes else 0)
                        epochs_fin_scores.append(epoch_scores[-1] if epoch_scores else 0)
                        epochs_shortfalls.append(epoch_shortfalls[-1] if epoch_shortfalls else 0)

                        # Track performance based on score type
                        if score_type == 'binary':
                            sample_ID_performances.append(successes / trials if trials > 0 else 0)
                        elif score_type in ['continuous_01', 'continuous_bounded']:
                            cont_lower = bounds.get('lower', 0.0)
                            cont_upper = bounds.get('upper', 1.0)
                            sample_ID_performances.append(
                                (np.mean(accumulated_scores) - cont_lower) / (cont_upper - cont_lower) if accumulated_scores else 0
                            )
                        else:  # ordinal
                            sample_ID_performances.append(np.mean(accumulated_scores) / ordinal_max_score if accumulated_scores else 0)

                # Add item summary based on score type
                if score_type == 'binary':
                    item_summaries.append({'successes': successes, 'trials': trials})
                elif score_type in ['continuous_01', 'continuous_bounded']:
                    item_summaries.append({
                        'scores': list(accumulated_scores),
                        'n_obs': len(accumulated_scores),
                        'mean_raw': float(np.mean(accumulated_scores)) if accumulated_scores else 0.0
                    })
                else:  # ordinal
                    # For ordinal, store sum of scores and count (to mimic binary structure for compatibility)
                    item_summaries.append({'successes': int(np.sum(accumulated_scores)), 'trials': len(accumulated_scores)})
                used_reps_dfs.append(pd.DataFrame(used_reps, columns=df_item.columns))

                # Calculate current performance estimate based on score type
                if score_type == 'binary':
                    current_perf_estimate = sum(s['successes'] for s in item_summaries) / sum(s['trials'] for s in item_summaries)
                elif score_type in ['continuous_01', 'continuous_bounded']:
                    cont_lower = bounds.get('lower', 0.0)
                    cont_upper = bounds.get('upper', 1.0)
                    total_obs = sum(s['n_obs'] for s in item_summaries)
                    if total_obs > 0:
                        weighted_mean = sum(s['mean_raw'] * s['n_obs'] for s in item_summaries) / total_obs
                        current_perf_estimate = (weighted_mean - cont_lower) / (cont_upper - cont_lower)
                    else:
                        current_perf_estimate = 0.0
                else:  # ordinal - normalize by max score
                    current_perf_estimate = (sum(s['successes'] for s in item_summaries) / sum(s['trials'] for s in item_summaries)) / ordinal_max_score

                current_conservatism = conservatism if current_perf_estimate < low_perf_threshold else 1.0

                if ((item_idx + 1) % pymc_refresh_every == 0) or (item_idx == len(item_ids) - 1):
                    if score_type == 'binary':
                        # === BINARY GROUP-LEVEL STOPPING ===
                        all_successes = np.array([s['successes'] for s in item_summaries], dtype=np.int32)
                        all_trials = np.array([s['trials'] for s in item_summaries], dtype=np.int32)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            with model:
                                pm.set_data({
                                    "successes": all_successes,
                                    "trials": all_trials,
                                    "n_items": np.int64(len(all_successes))
                                })
                                try:
                                    with suppress_all_output():
                                        trace = pm.sample(**sampling_kwargs)
                                except Exception as e:
                                    logger.error(f"MCMC sampling failed [binary convergence {grouping_name}]: {e}")
                                    trace = None

                            if trace is not None:
                                _log_mcmc_diagnostics(trace, logger, f"binary convergence {grouping_name}")

                            # Extract CI bounds using EXPECTED GROUP ACCURACY: mean(Theta)
                            # This correctly accounts for between-item variance (sigma_group)
                            # Using mean(Theta) instead of Theta[0] or sigmoid(mu_group)
                            if trace is not None:
                                try:
                                    # Extract item-level Theta posterior samples (shape: chains × draws × items)
                                    theta_samples = trace.posterior["Theta"].values
                                    # Compute mean across items for each posterior sample
                                    mean_theta_samples = theta_samples.mean(axis=2)
                                    with suppress_all_output():
                                        group_hdi = az.hdi({"mean_theta": mean_theta_samples}, hdi_prob=cred_level)
                                    theta_lo = float(group_hdi["mean_theta"].sel(hdi="lower").values)
                                    theta_hi = float(group_hdi["mean_theta"].sel(hdi="higher").values)
                                except Exception:
                                    # Fallback: use sigmoid(mu_group)
                                    mu_group_samples = trace.posterior["mu_group"].values
                                    group_theta_samples = 1.0 / (1.0 + np.exp(-mu_group_samples))
                                    with suppress_all_output():
                                        group_hdi = az.hdi({"group_theta": group_theta_samples}, hdi_prob=cred_level)
                                    theta_lo = float(group_hdi["group_theta"].sel(hdi="lower").values)
                                    theta_hi = float(group_hdi["group_theta"].sel(hdi="higher").values)
                            else:
                                theta_lo = 0.0
                                theta_hi = 1.0
                            theta_width = theta_hi - theta_lo
                            CI_record.append(theta_width)
                            effective_width = theta_width
                            if current_perf_estimate < low_perf_threshold:
                                effective_width = theta_width * current_conservatism

                    elif score_type in ['continuous_01', 'continuous_bounded']:
                        # === CONTINUOUS GROUP-LEVEL STOPPING ===
                        # Collect all continuous scores from all items processed so far
                        cont_lower = bounds.get('lower', 0.0)
                        cont_upper = bounds.get('upper', 1.0)
                        all_cont_scores = []
                        for used_df in used_reps_dfs:
                            all_cont_scores.extend(used_df[score_column].values)

                        theta_lo, theta_hi, width = _continuous_bounded_ci_adaptive(
                            np.array(all_cont_scores),
                            lower_bound=cont_lower,
                            upper_bound=cont_upper,
                            cred_level=cred_level,
                            conservatism=1.0,
                            low_perf_threshold=low_perf_threshold
                        )
                        # Normalize width to [0,1]
                        theta_width = width / (cont_upper - cont_lower)
                        CI_record.append(theta_width)
                        # Apply conservatism externally (consistent with binary/ordinal group-level pattern)
                        effective_width = theta_width * current_conservatism if current_perf_estimate < low_perf_threshold else theta_width
                        # Normalize CI bounds to [0,1] for consistent reporting
                        theta_lo = (theta_lo - cont_lower) / (cont_upper - cont_lower)
                        theta_hi = (theta_hi - cont_lower) / (cont_upper - cont_lower)

                    else:
                        # === ORDINAL GROUP-LEVEL STOPPING ===
                        # Collect all ordinal scores from all items processed so far
                        all_ord_scores = []
                        for used_df in used_reps_dfs:
                            all_ord_scores.extend(used_df[score_column].values)

                        # Use appropriate inference method
                        if ordinal_inference == 'modal':
                            theta_lo, theta_hi, width = _ordinal_ci_adaptive(
                                np.array(all_ord_scores),
                                ordinal_max_score=ordinal_max_score,
                                cred_level=cred_level,
                                conservatism=current_conservatism,
                                low_perf_threshold=low_perf_threshold
                            )
                        elif ordinal_inference == 'entropy':
                            theta_lo, theta_hi, width, _ = _ordinal_entropy_ci_adaptive(
                                np.array(all_ord_scores),
                                ordinal_max_score=ordinal_max_score,
                                cred_level=cred_level,
                                conservatism=current_conservatism,
                                low_perf_threshold=low_perf_threshold,
                                model_cache=ordinal_model_cache_shared,  # OPTIMIZATION: Use shared cache
                                compute_kwargs=sampling_kwargs
                            )
                        else:  # hybrid
                            should_stop_group, reason_group, diagnostics_group = _ordinal_hybrid_stopping_criterion(
                                np.array(all_ord_scores),
                                ordinal_max_score=ordinal_max_score,
                                delta_item=delta_cap,  # Use delta_cap for group-level
                                cred_level=cred_level,
                                entropy_history=[],  # Fresh history for group-level (must be fresh)
                                entropy_threshold=entropy_threshold,
                                entropy_convergence_threshold=entropy_convergence_threshold,
                                conservatism=current_conservatism,
                                low_perf_threshold=low_perf_threshold,
                                model_cache=ordinal_model_cache_shared,  # OPTIMIZATION: Use shared cache
                                compute_kwargs=sampling_kwargs
                            )
                            # Use modal width from diagnostics
                            theta_lo = diagnostics_group.get('modal_ci', (0, 1))[0]
                            theta_hi = diagnostics_group.get('modal_ci', (0, 1))[1]
                            width = diagnostics_group.get('modal_width', 1.0)

                        theta_width = width
                        CI_record.append(theta_width)
                        effective_width = theta_width * current_conservatism if current_perf_estimate < low_perf_threshold else theta_width
                        if len(CI_record) >= stab_window:
                            recent_widths = CI_record[-stab_window:]
                            slope = np.polyfit(range(len(recent_widths)), recent_widths, 1)[0]
                            CI_slopes_hist.append(slope)
                            slope_threshold = CI_delta / current_conservatism if current_perf_estimate < low_perf_threshold else CI_delta
                            if (len(CI_slopes_hist) >= 2):
                                recent_slopes = CI_slopes_hist[-3:]
                                slope_slopes = np.polyfit(range(len(recent_slopes)), recent_slopes, 1)[0]
                                CI_slope_slopes.append(slope_slopes)
                        else:
                            slope = None
                            slope_slopes = None
                        # Only check slope/slopes if they are set
                        if (effective_width < delta_cap) or (slope is not None and slope_slopes is not None and abs(slope) <= slope_threshold and len(CI_slopes_hist) >= 2 and slope_slopes >= 0):
                            item_scores.append(1)
                        else:
                            item_scores.append(0)
                        if (item_idx == len(item_ids) - 1):
                            slope_threshold = CI_delta / current_conservatism if current_perf_estimate < low_perf_threshold else CI_delta
                            task_performance = current_perf_estimate
                            if item_scores and item_scores[-1] == 1:
                                item_shortfalls.append(0)
                            else:
                                max_shortfall_cap = 200
                                projection = project_convergence(
                                    ci_widths=CI_record,
                                    ci_slopes=CI_slopes_hist,
                                    delta=delta_item,
                                    slope_threshold=slope_threshold,
                                    step_size=pymc_refresh_every,
                                    max_steps=max(1, max_shortfall_cap // max(1, pymc_refresh_every)),
                                    stab_window=stab_window,
                                    n_bootstrap=0,
                                )
                                if projection is not None:
                                    item_shortfalls.append(projection['projected_additional_trials'])
                                else:
                                    item_shortfalls.append(max_shortfall_cap)
            items_fin_CI_widths.append(CI_record[-1] if CI_record else 0)
            items_fin_CI_slopes.append(CI_slopes_hist[-1] if CI_slopes_hist else 0)
            items_fin_slope_slopes.append(CI_slope_slopes[-1] if CI_slope_slopes else 0)
            items_fin_scores.append(item_scores[-1] if item_scores else 0)
            items_shortfalls.append(item_shortfalls[-1] if item_shortfalls else 0)

        mean_fin_CI_width_item = np.mean(items_fin_CI_widths) if items_fin_CI_widths else None
        var_fin_CI_width_item = np.var(items_fin_CI_widths) if len(items_fin_CI_widths) > 0 else 0.0
        mean_fin_CI_slope_item = np.mean(items_fin_CI_slopes) if items_fin_CI_slopes else None
        var_fin_CI_slope_item = np.var(items_fin_CI_slopes) if len(items_fin_CI_slopes) > 0 else 0.0
        mean_fin_slope_slope_item = np.mean(items_fin_slope_slopes) if items_fin_slope_slopes else None
        var_fin_slope_slope_item = np.var(items_fin_slope_slopes) if len(items_fin_slope_slopes) > 0 else 0.0
        mean_needed_items = np.mean(items_shortfalls) if items_shortfalls else None
        var_needed_items = np.var(items_shortfalls) if len(items_shortfalls) > 0 else 0.0
        mean_fin_CI_width_epoch = np.mean(epochs_fin_CI_widths) if epochs_fin_CI_widths else None
        var_fin_CI_width_epoch = np.var(epochs_fin_CI_widths) if len(epochs_fin_CI_widths) > 0 else 0.0
        mean_fin_CI_slope_epoch = np.mean(epochs_fin_CI_slopes) if epochs_fin_CI_slopes else None
        var_fin_CI_slope_epoch = np.var(epochs_fin_CI_slopes) if len(epochs_fin_CI_slopes) > 0 else 0.0
        mean_fin_slope_slope_epoch = np.mean(epochs_fin_slope_slopes) if epochs_fin_slope_slopes else None
        var_fin_slope_slope_epoch = np.var(epochs_fin_slope_slopes) if len(epochs_fin_slope_slopes) > 0 else 0.0
        mean_needed_epochs = np.mean(epochs_shortfalls) if epochs_shortfalls else None
        var_needed_epochs = np.var(epochs_shortfalls) if len(epochs_shortfalls) > 0 else 0.0
        mean_items_fin_score = np.mean(items_fin_scores) if items_fin_scores else None
        var_items_fin_score = np.var(items_fin_scores) if len(items_fin_scores) > 0 else 0.0
        mean_epochs_fin_score = np.mean(epochs_fin_scores) if epochs_fin_scores else None
        var_epochs_fin_score = np.var(epochs_fin_scores) if len(epochs_fin_scores) > 0 else 0.0
        mean_sample_id_performance = np.mean(agg_sample_id_performance) if agg_sample_id_performance else None
        var_sample_id_performance = np.var(agg_sample_id_performance) if len(agg_sample_id_performance) > 0 else 0.0

        # Calculate n_items_used, percent_items_used, avg_reps_per_item
        n_items_used = len(item_summaries)
        percent_items_used = n_items_used / len(df_part['sample_id_num'].unique()) if len(df_part['sample_id_num'].unique()) > 0 else 0
        if item_summaries:
            if score_type in ['continuous_01', 'continuous_bounded']:
                avg_reps_per_item = np.mean([s['n_obs'] for s in item_summaries])
            else:
                avg_reps_per_item = np.mean([s['trials'] for s in item_summaries])
        else:
            avg_reps_per_item = 0

        # theta_lo, theta_hi, theta_width are from the last PyMC run
        # task_performance is set above

        result = {
            'grouping': pid,
            'group_label': df_part['grouping'].iloc[0] if 'grouping' in df_part.columns else str(pid),
            'n_items_used': n_items_used,
            'theta_ci_low': theta_lo,
            'theta_ci_high': theta_hi,
            'theta_ci_width': theta_width,
            'percent_items_used': percent_items_used,
            'avg_reps_per_item': avg_reps_per_item,
            'mean_fin_CI_width_item': mean_fin_CI_width_item,
            'var_fin_CI_width_item': var_fin_CI_width_item,
            'mean_fin_CI_slope_item': mean_fin_CI_slope_item,
            'var_fin_CI_slope_item': var_fin_CI_slope_item,
            'mean_fin_slope_slope_item': mean_fin_slope_slope_item,
            'var_fin_slope_slope_item': var_fin_slope_slope_item,
            'mean_needed_items': mean_needed_items,
            'var_needed_items': var_needed_items,
            'mean_items_fin_score': mean_items_fin_score,
            'var_items_fin_score': var_items_fin_score,
            'mean_fin_CI_width_epoch': mean_fin_CI_width_epoch,
            'var_fin_CI_width_epoch': var_fin_CI_width_epoch,
            'mean_fin_CI_slope_epoch': mean_fin_CI_slope_epoch,
            'var_fin_CI_slope_epoch': var_fin_CI_slope_epoch,
            'mean_fin_slope_slope_epoch': mean_fin_slope_slope_epoch,
            'var_fin_slope_slope_epoch': var_fin_slope_slope_epoch,
            'mean_needed_epochs': mean_needed_epochs,
            'var_needed_epochs': var_needed_epochs,
            'mean_epochs_fin_score': mean_epochs_fin_score,
            'var_epochs_fin_score': var_epochs_fin_score,
            'task_performance': task_performance,
            'mean_sample_id_performance': mean_sample_id_performance,
            'var_sample_id_performance': var_sample_id_performance
        }
        result['error'] = None
        return result
    except Exception as e:
        logger.error(f"Error processing grouping {pid}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'grouping': pid,
            'group_label': df_part['grouping'].iloc[0] if 'grouping' in df_part.columns else str(pid),
            'n_items_used': None,
            'theta_ci_low': None,
            'theta_ci_high': None,
            'theta_ci_width': None,
            'percent_items_used': None,
            'avg_reps_per_item': None,
            'mean_fin_CI_width_item': None,
            'var_fin_CI_width_item': None,
            'mean_fin_CI_slope_item': None,
            'var_fin_CI_slope_item': None,
            'mean_fin_slope_slope_item': None,
            'var_fin_slope_slope_item': None,
            'mean_needed_items': None,
            'var_needed_items': None,
            'mean_items_fin_score': None,
            'var_items_fin_score': None,
            'mean_fin_CI_width_epoch': None,
            'var_fin_CI_width_epoch': None,
            'mean_fin_CI_slope_epoch': None,
            'var_fin_CI_slope_epoch': None,
            'mean_fin_slope_slope_epoch': None,
            'var_fin_slope_slope_epoch': None,
            'mean_needed_epochs': None,
            'var_needed_epochs': None,
            'mean_epochs_fin_score': None,
            'var_epochs_fin_score': None,
            'task_performance': None,
            'mean_sample_id_performance': None,
            'var_sample_id_performance': None,
            'error': str(e)
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def _get_logfile_path_convergence(default='optstop_convergence.log'):
    import logging
    for handler in logging.getLogger().handlers:
        if hasattr(handler, 'baseFilename'):
            return handler.baseFilename
    return default

# --- Main API ---
# Removed _configure_multiprocessing_environment to prevent race conditions
# Worker processes now handle their own environment setup via initializers

def convergence_posthoc(df: pd.DataFrame, params: dict, grouping_columns: List[str], sample_id_column: str, epoch_column: str, score_column: str = "score", display_progress: bool = True, generate_diagnostics: bool = True, diagnostics_prefix: str = "convergence_eval", gpu_ids: Optional[List[int]] = None, max_workers: Optional[int] = None, ordinal_tasks: Optional[List[str]] = None, ordinal_max_score: int = 10, ordinal_inference: str = 'modal', ordinal_model_type: str = 'ordered_logistic', continuous_tasks: Optional[List[str]] = None, entropy_threshold: float = 0.8, entropy_convergence_threshold: float = 0.10, prior_mu: float = 0.0, prior_sigma: Optional[float] = None):
    """
    Post-hoc convergence analysis, parallelized across groupings.

    Supports binary (0/1), ordinal (Likert scale, e.g., 0-10), and continuous
    bounded scoring, including mixed datasets with multiple types.

    The user must specify:
      - grouping_columns: list of column names to combine for grouping (can be a single string or list of strings)
      - sample_id_column: column name for sample ID
      - epoch_column: column name for epoch/trial
      - score_column: column name for score (default: 'score')
      - display_progress: whether to show a progress bar (default True)
      - generate_diagnostics: whether to generate diagnostic figures (default True)
      - diagnostics_prefix: prefix for diagnostic output files
      - gpu_ids: List of GPU IDs to use for parallel processing. If None, uses CPU-only. If provided, assigns GPUs to workers cyclically.
      - max_workers: Number of parallel workers. If None, uses len(gpu_ids) when GPUs specified, otherwise uses CPU count.
      - ordinal_tasks: List of substrings to identify ordinal groupings (e.g., ['confidence', 'rating']). If None, all groupings use binary scoring.
      - ordinal_max_score: Maximum score for ordinal data (e.g., 10 for 0-10 scale). Default: 10.
      - ordinal_inference: Inference method for ordinal data: 'modal', 'entropy', or 'hybrid' (recommended). Default: 'modal'.
      - ordinal_model_type: Hierarchical model type for ordinal data (default: 'ordered_logistic')
          * 'ordered_logistic': Cumulative link model with identified cutpoints (recommended)
          * 'dirichlet': Dirichlet-Multinomial (treats categories as exchangeable)
          Falls back to 'dirichlet' if ordered_logistic sampling fails.
      - continuous_tasks: List of substrings to identify continuous bounded groupings (default: None)
          * If None: Continuous inference is not used
          * If provided: Tasks whose grouping name contains any substring use Beta distribution inference
          * Scores must be pre-aggregated floats in [0, ordinal_max_score] range
          * Priority: continuous_tasks > ordinal_tasks > binary (default)
      - entropy_threshold: Proportion of max entropy for false peak detection in hybrid mode. Default: 0.8.
      - prior_mu: Centre of the group-level Normal prior on the logit scale. Default: 0.0 (50% probability).
      - prior_sigma: Scale (standard deviation) of the group-level Normal prior on the logit scale.
          If None (default), uses pathway-specific defaults: 1.5 for binary/continuous, 2.0 for ordinal.
          If explicitly set, applies to all pathways.

    Returns a DataFrame of convergence statistics.

    Example with ordinal data:
        >>> params = {'delta_item': 0.15, 'delta_cap': 0.05, ...}
        >>> convergence_df = convergence_posthoc(
        ...     df, params,
        ...     grouping_columns=['student', 'task'],
        ...     sample_id_column='item_id',
        ...     epoch_column='trial_num',
        ...     ordinal_tasks=['confidence', 'difficulty'],
        ...     ordinal_max_score=10,
        ...     ordinal_inference='hybrid',
        ...     entropy_threshold=0.8
        ... )
    """
    # Input validation
    if isinstance(grouping_columns, str):
        grouping_columns = [grouping_columns]
    required = set(grouping_columns + [sample_id_column, epoch_column])
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        return pd.DataFrame(columns=df.columns)
    df = df.copy()
    df['grouping'] = df[grouping_columns].astype(str).agg('-'.join, axis=1)
    df['grouping_num'] = df['grouping'].astype('category').cat.codes
    df['sample_id_num'] = df[sample_id_column].astype('category').cat.codes
    df['epoch_num'] = df[epoch_column].astype(int)
    
    logger = logging.getLogger('optstop.convergence')
    _validate_params(params)
    logger.info('Starting post-hoc convergence analysis')

    # Validate and configure GPU settings
    validated_gpu_ids, validated_max_workers = gpu_utils.validate_gpu_configuration(gpu_ids, max_workers)

    # Environment configuration is now handled by worker initializers only
    # Clear any existing PyTensor modules from main process to prevent conflicts
    import sys
    modules_to_clear = [mod for mod in sys.modules.keys() if mod.startswith(('pytensor', 'pymc'))]
    for mod in modules_to_clear:
        if mod in sys.modules:
            logger.info(f"Clearing {mod} from main process before spawning workers")
            del sys.modules[mod]

    groupings = list(df.groupby(['grouping_num']))
    if not groupings:
        logger.info('No groupings to process; returning empty DataFrame.')
        return pd.DataFrame(columns=df.columns)

    # Add number of parallel tasks to params for smart GPU/CPU decision making
    params_with_context = params.copy()
    params_with_context['_num_parallel_tasks'] = len(groupings)

    # Validate ordinal inference parameter
    if ordinal_inference not in ['modal', 'entropy', 'hybrid']:
        raise ValueError(f"ordinal_inference must be 'modal', 'entropy', or 'hybrid', got '{ordinal_inference}'")

    # Add score type parameters to params dict for worker processes
    # These are always set so workers can use determine_score_type_standalone
    params_with_context['ordinal_tasks'] = ordinal_tasks
    params_with_context['continuous_tasks'] = continuous_tasks
    params_with_context['ordinal_max_score'] = ordinal_max_score
    params_with_context['ordinal_inference'] = ordinal_inference
    params_with_context['ordinal_model_type'] = ordinal_model_type
    params_with_context['entropy_threshold'] = entropy_threshold
    params_with_context['entropy_convergence_threshold'] = entropy_convergence_threshold
    params_with_context['prior_mu'] = prior_mu
    params_with_context['prior_sigma'] = prior_sigma

    # Validate ordinal scores upfront for all ordinal groupings
    if ordinal_tasks is not None:
        for grouping_name in df['grouping'].unique():
            score_type, bounds = determine_score_type_standalone(
                grouping_name,
                ordinal_tasks=ordinal_tasks,
                continuous_tasks=continuous_tasks,
                upper_bound=ordinal_max_score
            )
            if score_type == 'ordinal':
                grouping_data = df[df['grouping'] == grouping_name]
                validate_ordinal_scores(
                    grouping_data[score_column].values,
                    ordinal_max_score,
                    grouping_name
                )
                check_ordinal_sparsity(
                    grouping_data[score_column].values,
                    ordinal_max_score,
                    grouping_name
                )
                logger.info(f"Validated ordinal scores for grouping '{grouping_name}' (using {ordinal_inference} inference)")
            elif score_type in ('continuous_bounded', 'continuous_01'):
                logger.info(f"Grouping '{grouping_name}' will use continuous bounded inference (Beta distribution)")

    args_list = [(pid, df_part, params_with_context, score_column) for pid, df_part in groupings]

    # Determine final worker count
    if validated_max_workers is None:
        final_max_workers = min(len(args_list), os.cpu_count() or 1)
    else:
        final_max_workers = min(validated_max_workers, len(args_list))

    # Create worker base directory with managed cleanup
    with cleanup_utils.managed_temp_dir(prefix='optstop_convergence_workers_') as worker_base_dir:
        # Create worker initialization arguments with GPU assignment
        worker_init_args = gpu_utils.create_worker_initargs(worker_base_dir, validated_gpu_ids, suppress_output=True)

        if validated_gpu_ids:
            logger.info(f'Using {final_max_workers} workers with GPU assignment: {validated_gpu_ids}')
        else:
            logger.info(f'Using {final_max_workers} CPU-only workers for {len(args_list)} groupings')

        results = []

        # Use spawn method to ensure clean processes without shared PyTensor state
        import multiprocessing as mp
        original_start_method = mp.get_start_method()

        try:
            # Force spawn method for clean process isolation
            if mp.get_start_method() != 'spawn':
                mp.set_start_method('spawn', force=True)

            with concurrent.futures.ProcessPoolExecutor(max_workers=final_max_workers) as executor:
                # Submit tasks with individual worker initialization
                futures = []
                for i, task_args in enumerate(args_list):
                    # Get worker initialization args cyclically
                    worker_args = worker_init_args[i % len(worker_init_args)]
                    # Create a new process with specific GPU assignment
                    future = executor.submit(_process_grouping_with_init_convergence, task_args, worker_args)
                    futures.append(future)

                # Collect results
                iterator = concurrent.futures.as_completed(futures)
                if display_progress:
                    iterator = tqdm(iterator, total=len(futures), desc="Convergence analysis")

                for future in iterator:
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"Worker task failed: {e}")
                        # Add empty result to maintain consistency
                        results.append({
                            'grouping': None, 'group_label': None, 'n_items_used': None,
                            'theta_ci_low': None, 'theta_ci_high': None, 'theta_ci_width': None,
                            'percent_items_used': None, 'avg_reps_per_item': None,
                            'error': str(e)
                        })

        finally:
            # Restore original start method
            if original_start_method != mp.get_start_method():
                try:
                    mp.set_start_method(original_start_method, force=True)
                except RuntimeError:
                    # Start method can only be set once, ignore if already set
                    pass

        logger.info('Convergence analysis complete')
        output_df = pd.DataFrame(results)
        if generate_diagnostics:
            try:
                generate_convergence_diagnostics(output_df, out_prefix=diagnostics_prefix)
            except Exception as e:
                logger = logging.getLogger('optstop.convergence')
                logger.error(f"Failed to generate convergence diagnostics: {e}")
        # Only print if in main process
        if os.getpid() == getattr(os, 'getppid', lambda: None)() or hasattr(sys, 'ps1'):
            print(f"Run complete. See the log file for details: {_get_logfile_path_convergence()}")
        return output_df

def generate_convergence_diagnostics(convergence_data: pd.DataFrame, out_prefix: str = "convergence_eval"):
    """
    Generate diagnostic figures for convergence_posthoc output.
    Saves two PNGs: {out_prefix}_grouped_needed.png and {out_prefix}_score_scatter.png
    """
    import numpy as np
    import logging
    logger = logging.getLogger('optstop.convergence.diagnostics')
    try:
        if convergence_data.empty:
            logger.warning("Convergence diagnostics: input DataFrame is empty.")
            fig = plt.figure(figsize=(8, 4))
            fig.suptitle("No data for convergence diagnostics", fontsize=14)
            fig.savefig(f"{out_prefix}_grouped_needed.png", dpi=150)
            fig.savefig(f"{out_prefix}_score_scatter.png", dpi=150)
            return
        n_samples = 20  # Used for SEM calculation, as in the script
        # Calculate standard error from variances (SEM = sqrt(variance)/sqrt(n))
        # Convert to numeric and handle NaN/invalid values safely
        var_items = pd.to_numeric(convergence_data['var_needed_items'], errors='coerce')
        var_epochs = pd.to_numeric(convergence_data['var_needed_epochs'], errors='coerce')

        # Only calculate sqrt for valid positive values
        convergence_data['sem_needed_items'] = np.where(
            (var_items > 0) & pd.notna(var_items),
            np.sqrt(var_items / n_samples),
            np.nan
        )
        convergence_data['sem_needed_epochs'] = np.where(
            (var_epochs > 0) & pd.notna(var_epochs),
            np.sqrt(var_epochs / n_samples),
            np.nan
        )
        # --- NEW GROUPED NEEDED PLOT ---
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
        # Top: mean_needed_items by grouping
        items_by_group = convergence_data[['group_label', 'mean_needed_items', 'sem_needed_items']].dropna()
        items_by_group = items_by_group.groupby('group_label').mean().reset_index()
        items_by_group = items_by_group.sort_values('mean_needed_items', ascending=False)
        y_labels_items = items_by_group['group_label'].astype(str)
        y_pos_items = np.arange(len(items_by_group))
        ax1.errorbar(items_by_group['mean_needed_items'], y_pos_items, xerr=items_by_group['sem_needed_items'], fmt='o', color='skyblue', ecolor='gray', capsize=4)
        ax1.set_yticks(y_pos_items)
        ax1.set_yticklabels(y_labels_items)
        ax1.set_xlabel('Mean Needed Items (with std error)')
        ax1.set_ylabel('Grouping')
        ax1.set_title('Mean Needed Items by Grouping')
        ax1.invert_yaxis()
        # Bottom: mean_needed_epochs by grouping
        epochs_by_group = convergence_data[['group_label', 'mean_needed_epochs', 'sem_needed_epochs']].dropna()
        epochs_by_group = epochs_by_group.groupby('group_label').mean().reset_index()
        epochs_by_group = epochs_by_group.sort_values('mean_needed_epochs', ascending=False)
        y_labels_epochs = epochs_by_group['group_label'].astype(str)
        y_pos_epochs = np.arange(len(epochs_by_group))
        ax2.errorbar(epochs_by_group['mean_needed_epochs'], y_pos_epochs, xerr=epochs_by_group['sem_needed_epochs'], fmt='o', color='lightgreen', ecolor='gray', capsize=4)
        ax2.set_yticks(y_pos_epochs)
        ax2.set_yticklabels(y_labels_epochs)
        ax2.set_xlabel('Mean Needed Epochs (with std error)')
        ax2.set_ylabel('Grouping')
        ax2.set_title('Mean Needed Epochs by Grouping')
        ax2.invert_yaxis()
        plt.tight_layout()
        fig1.savefig(f"{out_prefix}_grouped_needed.png", dpi=300)
        # --- SCORE SCATTER PLOT (unchanged) ---
        fig2 = plt.figure(figsize=(16, 6))
        gs2 = gridspec.GridSpec(1, 2)
        ax3 = plt.subplot(gs2[0, 0])
        if 'task_performance' in convergence_data.columns and 'mean_needed_items' in convergence_data.columns:
            ax3.errorbar(convergence_data['mean_needed_items'], convergence_data['task_performance'],
                         xerr=convergence_data['sem_needed_items'], fmt='o', color='blue', 
                         alpha=0.5, ecolor='gray', capsize=3)
            ax3.set_xlabel('Mean Needed Items (with std error)')
            ax3.set_ylabel('Task Performance (Final Scoring)')
            ax3.set_title('Task Performance vs Mean Needed Items')
            ax3.set_xlim(left=0)
        else:
            ax3.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12)
        ax4 = plt.subplot(gs2[0, 1])
        if 'mean_sample_id_performance' in convergence_data.columns and 'mean_needed_epochs' in convergence_data.columns:
            ax4.errorbar(convergence_data['mean_needed_epochs'], convergence_data['mean_sample_id_performance'],
                         xerr=convergence_data['sem_needed_epochs'], fmt='o', color='green', 
                         alpha=0.5, ecolor='gray', capsize=3)
            ax4.set_xlabel('Mean Needed Epochs (with std error)')
            ax4.set_ylabel('Sample Performance (Final Scoring)')
            ax4.set_title('Sample Performance vs Mean Needed Epochs')
            ax4.set_xlim(left=0)
        else:
            ax4.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12)
        plt.tight_layout()
        fig2.savefig(f"{out_prefix}_score_scatter.png", dpi=300)
        logger.info(f"Convergence diagnostics saved to {out_prefix}_grouped_needed.png and {out_prefix}_score_scatter.png")
    except Exception as e:
        logger.error(f"Error generating convergence diagnostics: {e}")
        try:
            fig = plt.figure(figsize=(8, 4))
            fig.suptitle(f"Diagnostics failed: {e}", fontsize=14)
            fig.savefig(f"{out_prefix}_grouped_needed.png", dpi=150)
            fig.savefig(f"{out_prefix}_score_scatter.png", dpi=150)
        except Exception:
            pass
        raise 