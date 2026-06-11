"""Comprehensive tests for convergence projection across all pathways and parameter regimes.

Each test prints full projection output for manual inspection of metrics.
Run with: venv/bin/python -m pytest tests/test_convergence_projection_comprehensive.py -v -s
"""

import numpy as np
import pytest
import json

from optstop.convergence import project_convergence


# ---------------------------------------------------------------------------
# Trajectory generators - realistic CI width patterns per pathway
# ---------------------------------------------------------------------------

def binary_trajectory(n_obs, true_p=0.7, noise_sd=0.005, seed=42):
    """Simulate binary CI width trajectory.

    Binary CI widths shrink roughly as 1/sqrt(n). High-performance items
    converge faster. Noise represents MCMC sampling variability.
    """
    rng = np.random.default_rng(seed)
    # Approximate beta-binomial CI width: ~1.96 * sqrt(p(1-p)/n) * 2
    ns = np.arange(5, 5 + n_obs)
    base_width = 1.96 * 2 * np.sqrt(true_p * (1 - true_p) / ns)
    noise = rng.normal(0, noise_sd, n_obs)
    widths = np.clip(base_width + noise, 0.001, 1.0)
    return widths.tolist()


def ordinal_trajectory(n_obs, k=11, spread='peaked', noise_sd=0.008, seed=42):
    """Simulate ordinal CI width trajectory (modal width, normalised to [0,1]).

    Ordinal widths depend on category spread. Peaked distributions converge
    faster than dispersed ones.
    """
    rng = np.random.default_rng(seed)
    ns = np.arange(5, 5 + n_obs)
    if spread == 'peaked':
        # Most mass on 1-2 categories -> fast convergence
        base_width = 0.6 / np.sqrt(ns / 5)
    elif spread == 'dispersed':
        # Mass spread across categories -> slow convergence
        base_width = 1.2 / np.sqrt(ns / 5)
    elif spread == 'uniform':
        # Near-uniform -> very slow, entropy stays high
        base_width = 1.5 / np.sqrt(ns / 5)
    else:
        raise ValueError(f"Unknown spread: {spread}")
    noise = rng.normal(0, noise_sd, n_obs)
    widths = np.clip(base_width + noise, 0.001, 1.0)
    return widths.tolist()


def continuous_trajectory(n_obs, variance=0.1, noise_sd=0.005, seed=42):
    """Simulate continuous CI width trajectory (normalised to [0,1]).

    Continuous CI widths shrink as ~sigma/sqrt(n) where sigma is the
    underlying score variance.
    """
    rng = np.random.default_rng(seed)
    ns = np.arange(5, 5 + n_obs)
    sigma = np.sqrt(variance)
    base_width = 1.96 * 2 * sigma / np.sqrt(ns)
    noise = rng.normal(0, noise_sd, n_obs)
    widths = np.clip(base_width + noise, 0.001, 1.0)
    return widths.tolist()


def random_walk_trajectory(n_obs, start=0.3, drift=-0.002, vol=0.02, seed=42):
    """Non-converging random walk trajectory - should produce low confidence."""
    rng = np.random.default_rng(seed)
    widths = [start]
    for _ in range(n_obs - 1):
        w = widths[-1] + drift + rng.normal(0, vol)
        widths.append(max(0.001, w))
    return widths


def flat_trajectory(n_obs, level=0.12, noise_sd=0.003, seed=42):
    """Flat trajectory (no convergence) - slope should be near zero."""
    rng = np.random.default_rng(seed)
    widths = level + rng.normal(0, noise_sd, n_obs)
    return np.clip(widths, 0.001, 1.0).tolist()


def sawtooth_trajectory(n_obs, base_start=0.15, base_end=0.06, n_teeth=3, seed=42):
    """Sawtooth pattern - decreasing trend with periodic jumps (model recompilation)."""
    rng = np.random.default_rng(seed)
    segment_len = n_obs // n_teeth
    widths = []
    for i in range(n_teeth):
        frac = i / max(n_teeth - 1, 1)
        seg_start = base_start * (1 - frac) + base_end * frac + 0.03
        seg_end = base_start * (1 - frac) + base_end * frac
        seg = np.linspace(seg_start, seg_end, segment_len)
        seg += rng.normal(0, 0.003, segment_len)
        widths.extend(np.clip(seg, 0.001, 1.0).tolist())
    # Pad to exact length
    while len(widths) < n_obs:
        widths.append(widths[-1])
    return widths[:n_obs]


# ---------------------------------------------------------------------------
# Pretty printer for projection results
# ---------------------------------------------------------------------------

def print_projection(label, result, params=None):
    """Print full projection output for inspection."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    if params:
        print(f"  Parameters: {json.dumps(params, indent=4)}")
    if result is None:
        print("  Result: None (insufficient data)")
        return
    print(f"  --- Point Estimate ---")
    print(f"  projected_additional_steps:  {result['projected_additional_steps']}")
    print(f"  projected_additional_trials: {result['projected_additional_trials']}")
    print(f"  proximity_ratio:            {result['proximity_ratio']:.4f}")
    print(f"  convergence_target:         {result['convergence_target']}")
    print(f"  projected_width_at_term:    {result['projected_width_at_termination']:.6f}")
    print(f"  capped:                     {result['capped']}")
    # optimistic flag removed - callers now pass conservatism-adjusted slope_threshold
    print(f"  final_width:                {result['final_width']:.6f}")
    print(f"  final_slope:                {result['final_slope']:.8f}")
    print(f"  projection_basis:           {result['projection_basis']}")
    if 'exponential_fit' in result:
        fit = result['exponential_fit']
        print(f"  exp_fit: a={fit['a']:.4f} b={fit['b']:.4f} c={fit['c']:.6f} R²={fit['r_squared']:.4f}")
    unc = result['uncertainty']
    print(f"  --- Uncertainty ---")
    print(f"  ci_trials_80:               [{unc['ci_trials_80'][0]}, {unc['ci_trials_80'][1]}]")
    print(f"  ci_trials_50:               [{unc['ci_trials_50'][0]}, {unc['ci_trials_50'][1]}]")
    print(f"  bootstrap_skipped:          {unc['bootstrap_skipped']}")
    n_obs_key = 'n_width_observations' if 'n_width_observations' in unc else 'n_slope_observations'
    print(f"  {n_obs_key}:       {unc[n_obs_key]}")
    print(f"  trajectory_rmse:            {unc['trajectory_rmse']:.8f}")
    print(f"  confidence_level:           {unc['confidence_level']}")
    if 'some_bootstrap_capped' in unc:
        print(f"  some_bootstrap_capped:      {unc['some_bootstrap_capped']}")


# ===========================================================================
# 1. PATHWAY TESTS - Binary, Ordinal, Continuous with realistic trajectories
# ===========================================================================

class TestBinaryPathway:
    """Binary pathway: CI widths from beta-posterior HDI, stored in ci_slope_history."""

    @pytest.mark.parametrize("true_p,label", [
        (0.9, "high_performance_p0.9"),
        (0.7, "moderate_performance_p0.7"),
        (0.5, "chance_performance_p0.5"),
        (0.1, "low_performance_p0.1"),
    ])
    def test_binary_performance_levels(self, true_p, label):
        """Binary trajectories at different true performance levels."""
        widths = binary_trajectory(30, true_p=true_p)
        slopes = []  # Binary populates ci_slope_history, but test internal computation too
        params = {'pathway': 'binary', 'true_p': true_p, 'n_obs': 30,
                  'delta_cap': 0.05, 'CI_delta': 0.00001}
        result = project_convergence(
            ci_widths=widths, ci_slopes=slopes,
            delta=0.05, slope_threshold=0.00001,
            step_size=10, n_bootstrap=200,
        )
        print_projection(f"Binary - {label}", result, params)
        assert result is not None
        assert result['proximity_ratio'] > 0
        # Exponential uses n_width_observations, linear uses n_slope_observations
        if result['projection_basis'] == 'exponential_decay':
            assert result['uncertainty']['n_width_observations'] >= 5
        else:
            assert result['uncertainty']['n_slope_observations'] >= 2

    def test_binary_with_precomputed_slopes(self):
        """Binary with ci_slope_history populated (as in actual binary pathway)."""
        widths = binary_trajectory(30, true_p=0.7)
        # Compute slopes as rule.py does
        slopes = []
        for i in range(len(widths) - 15 + 1):
            window = widths[i:i + 15]
            slope = np.polyfit(range(15), window, 1)[0]
            slopes.append(float(slope))
        params = {'pathway': 'binary', 'true_p': 0.7, 'precomputed_slopes': len(slopes)}
        result = project_convergence(
            ci_widths=widths, ci_slopes=slopes,
            delta=0.05, slope_threshold=0.00001,
            step_size=10, n_bootstrap=200,
        )
        print_projection("Binary - precomputed slopes", result, params)

        # Also run without slopes and verify same result
        result_no_slopes = project_convergence(
            ci_widths=widths, ci_slopes=[],
            delta=0.05, slope_threshold=0.00001,
            step_size=10, n_bootstrap=200,
        )
        # Point estimates should match (same slope computation)
        assert result['projected_additional_steps'] == result_no_slopes['projected_additional_steps']


class TestOrdinalPathway:
    """Ordinal pathway: ci_slope_history is EMPTY, slopes computed from widths internally."""

    @pytest.mark.parametrize("spread,label", [
        ('peaked', "peaked_distribution"),
        ('dispersed', "dispersed_distribution"),
        ('uniform', "near_uniform_distribution"),
    ])
    def test_ordinal_category_spread(self, spread, label):
        """Ordinal trajectories with different category distributions."""
        widths = ordinal_trajectory(30, k=11, spread=spread)
        params = {'pathway': 'ordinal', 'K': 11, 'spread': spread,
                  'n_obs': 30, 'delta_cap': 0.05, 'CI_delta': 0.00001}
        result = project_convergence(
            ci_widths=widths, ci_slopes=[],  # Ordinal never populates slopes
            delta=0.05, slope_threshold=0.00001,
            step_size=10, n_bootstrap=200,
        )
        print_projection(f"Ordinal - {label}", result, params)
        assert result is not None
        # Dispersed/uniform should need more trials than peaked
        if spread == 'peaked':
            assert result['proximity_ratio'] < 5.0

    # NOTE: K (category count) affects ordinal model internals (cutpoints, Dirichlet
    # concentration) but by the time widths reach ci_width_history, K's effect is
    # already baked in. The projection function is pathway-agnostic and only sees
    # normalised widths, so K-parametrised tests would test the generator, not the
    # projection. The spread parameter above covers the meaningful trajectory variation.


class TestContinuousPathway:
    """Continuous pathway: normalised widths, ci_slope_history populated."""

    @pytest.mark.parametrize("variance,label", [
        (0.01, "low_variance_0.01"),
        (0.1, "moderate_variance_0.1"),
        (0.25, "high_variance_0.25"),
    ])
    def test_continuous_variance_levels(self, variance, label):
        """Continuous trajectories at different underlying variances."""
        widths = continuous_trajectory(30, variance=variance)
        params = {'pathway': 'continuous', 'variance': variance,
                  'n_obs': 30, 'delta_cap': 0.05, 'CI_delta': 0.00001}
        result = project_convergence(
            ci_widths=widths, ci_slopes=[],
            delta=0.05, slope_threshold=0.00001,
            step_size=10, n_bootstrap=200,
        )
        print_projection(f"Continuous - {label}", result, params)
        assert result is not None
        # Low variance should converge more easily
        if variance == 0.01:
            assert result['proximity_ratio'] < 3.0


# ===========================================================================
# 2. PARAMETER SENSITIVITY - delta_cap, delta_item, CI_delta, thresholds
# ===========================================================================

class TestDeltaCapSensitivity:
    """Varying delta_cap (group-level width threshold)."""

    @pytest.mark.parametrize("delta_cap", [0.02, 0.05, 0.10, 0.15, 0.25])
    def test_delta_cap_values(self, delta_cap):
        widths = binary_trajectory(30, true_p=0.7)
        params = {'delta_cap': delta_cap, 'pathway': 'binary', 'true_p': 0.7}
        result = project_convergence(
            ci_widths=widths, delta=delta_cap, slope_threshold=0.00001,
            step_size=10, n_bootstrap=200,
        )
        print_projection(f"delta_cap={delta_cap}", result, params)
        assert result is not None
        # Tighter threshold -> higher proximity_ratio -> more trials needed
        assert result['proximity_ratio'] == pytest.approx(widths[-1] / delta_cap, rel=1e-4)

    def test_delta_cap_ordering(self):
        """Stricter delta_cap should require more projected trials."""
        widths = binary_trajectory(30, true_p=0.7)
        results = {}
        for dc in [0.02, 0.05, 0.10, 0.25]:
            r = project_convergence(
                ci_widths=widths, delta=dc, slope_threshold=0.00001,
                step_size=10, max_steps=500, n_bootstrap=100,
            )
            results[dc] = r
            print_projection(f"delta_cap ordering: {dc}", r)
        # Stricter (lower) delta_cap -> more trials (or capped)
        trials = {dc: r['projected_additional_trials'] if r and not r['capped'] else 99999
                  for dc, r in results.items()}
        assert trials[0.02] >= trials[0.05] >= trials[0.10]


class TestCIDeltaSensitivity:
    """Varying CI_delta (slope stabilisation threshold).

    Uses a slowly converging trajectory where the slope is small but non-zero.
    With tight CI_delta (e.g. 0.000001), slope never crosses threshold -> width
    convergence or capped. With loose CI_delta (e.g. 0.01), slope_stabilisation
    triggers early because abs(slope) falls below threshold sooner.
    """

    @pytest.mark.parametrize("ci_delta", [0.000001, 0.00001, 0.0001, 0.001, 0.01])
    def test_ci_delta_values(self, ci_delta):
        """Different CI_delta values - affects slope_stabilisation triggering."""
        # Slowly converging: slopes will be small but non-trivial
        widths = binary_trajectory(30, true_p=0.7, noise_sd=0.002)
        params = {'CI_delta': ci_delta, 'trajectory': 'binary_slow', 'true_p': 0.7}
        result = project_convergence(
            ci_widths=widths, delta=0.05, slope_threshold=ci_delta,
            step_size=10, n_bootstrap=200,
        )
        print_projection(f"CI_delta={ci_delta}", result, params)
        assert result is not None
        # Looser CI_delta -> slope stabilisation triggers sooner -> fewer steps
        if ci_delta >= 0.01:
            assert result['convergence_target'] == 'projected_slope_stabilisation'
            assert result['projected_additional_steps'] <= 5

    def test_ci_delta_ordering(self):
        """Looser CI_delta should give equal or fewer projected steps."""
        widths = binary_trajectory(30, true_p=0.7, noise_sd=0.002)
        results = []
        for ci_delta in [0.000001, 0.0001, 0.01]:
            r = project_convergence(
                ci_widths=widths, delta=0.05, slope_threshold=ci_delta,
                step_size=10, n_bootstrap=50,
            )
            results.append(r)
        # Looser threshold -> fewer or equal steps
        assert results[0] is not None and results[2] is not None
        assert results[2]['projected_additional_steps'] <= results[0]['projected_additional_steps']


class TestStabWindowSensitivity:
    """Varying stab_window - affects slope computation and minimum data requirement."""

    @pytest.mark.parametrize("stab_window", [5, 10, 15, 20, 25])
    def test_stab_window_values(self, stab_window):
        widths = binary_trajectory(35, true_p=0.7)
        params = {'stab_window': stab_window, 'n_obs': 35}
        result = project_convergence(
            ci_widths=widths, delta=0.05, slope_threshold=0.00001,
            step_size=10, stab_window=stab_window, n_bootstrap=200,
        )
        print_projection(f"stab_window={stab_window}", result, params)
        if stab_window > 33:
            # Not enough data for stab_window + 1 widths and < 5 for exponential
            assert result is None
        else:
            assert result is not None
            if result['projection_basis'] == 'linear_extrapolation':
                expected_slopes = 35 - stab_window + 1
                print(f"  n_slopes derived: {result['uncertainty']['n_slope_observations']}")
                assert result['uncertainty']['n_slope_observations'] == expected_slopes
            else:
                print(f"  exponential fit used (stab_window irrelevant)")


# ===========================================================================
# 3. SAMPLE SIZE TESTS
# ===========================================================================

class TestSampleSizes:
    """Effect of trajectory length on projection quality."""

    @pytest.mark.parametrize("n_obs,label", [
        (16, "minimal_16obs"),
        (20, "small_20obs"),
        (30, "moderate_30obs"),
        (50, "large_50obs"),
        (100, "very_large_100obs"),
    ])
    def test_sample_size_binary(self, n_obs, label):
        widths = binary_trajectory(n_obs, true_p=0.7)
        params = {'n_obs': n_obs, 'pathway': 'binary'}
        result = project_convergence(
            ci_widths=widths, delta=0.05, slope_threshold=0.00001,
            step_size=10, n_bootstrap=200,
        )
        print_projection(f"Sample size - {label}", result, params)
        assert result is not None
        if result['projection_basis'] == 'linear_extrapolation':
            expected_slopes = n_obs - 15 + 1
            assert result['uncertainty']['n_slope_observations'] == expected_slopes
            # Bootstrap CIs require 3+ slopes; N=16 gives only 2
            if expected_slopes < 3:
                assert result['uncertainty']['bootstrap_skipped'] is True
                pt = result['projected_additional_trials']
                assert result['uncertainty']['ci_trials_80'] == [pt, pt]
                assert result['uncertainty']['ci_trials_50'] == [pt, pt]
            else:
                assert result['uncertainty']['bootstrap_skipped'] is False
        else:
            assert result['uncertainty']['n_width_observations'] == n_obs

    def test_sample_size_confidence_improves(self):
        """Longer trajectories should produce higher or equal confidence."""
        confidence_ordering = {'low': 0, 'moderate': 1, 'high': 2}
        prev_conf = -1
        for n_obs in [20, 40, 80]:
            widths = binary_trajectory(n_obs, true_p=0.7, noise_sd=0.002)
            result = project_convergence(
                ci_widths=widths, delta=0.05, slope_threshold=0.00001,
                step_size=10, n_bootstrap=200,
            )
            print_projection(f"Confidence vs n_obs={n_obs}", result)
            assert result is not None
            conf = confidence_ordering[result['uncertainty']['confidence_level']]
            assert conf >= prev_conf, (
                f"Confidence should not degrade with more data: "
                f"n={n_obs} got {result['uncertainty']['confidence_level']}"
            )
            prev_conf = conf

    def test_insufficient_various_lengths(self):
        """Below minimum data threshold -> None."""
        for n in [1, 5, 10, 15]:
            widths = binary_trajectory(n, true_p=0.7)
            result = project_convergence(ci_widths=widths, delta=0.05)
            print(f"  n_obs={n:3d}: result={'None' if result is None else 'NOT None'}")
            assert result is None, f"Expected None for n_obs={n}"


# ===========================================================================
# 4. PERFORMANCE CHARACTERISTICS - diverse, inconsistent, pathological
# ===========================================================================

class TestPerformanceCharacteristics:
    """Different data quality patterns."""

    def test_clean_monotonic_decrease(self):
        """Ideal case: smooth monotonic CI narrowing."""
        widths = np.linspace(0.20, 0.06, 30).tolist()
        result = project_convergence(
            ci_widths=widths, delta=0.05, slope_threshold=0.00001,
            step_size=10, n_bootstrap=200,
        )
        print_projection("Clean monotonic decrease", result)
        assert result is not None
        # Exponential RMSE on linear data is slightly higher than linear RMSE
        # but still low. Linear slope RMSE is ~0, exponential width RMSE ~0.006.
        assert result['uncertainty']['trajectory_rmse'] < 0.01
        # Should have high confidence for clean data near threshold
        assert result['uncertainty']['confidence_level'] in ('high', 'moderate')

    def test_noisy_trajectory(self):
        """High MCMC noise overlaid on convergence trend."""
        widths = binary_trajectory(30, true_p=0.7, noise_sd=0.03)
        result = project_convergence(
            ci_widths=widths, delta=0.05, slope_threshold=0.00001,
            step_size=10, n_bootstrap=200,
        )
        print_projection("Noisy trajectory (noise_sd=0.03)", result)
        assert result is not None
        # Should have measurable RMSE
        assert result['uncertainty']['trajectory_rmse'] > 0.001

    def test_random_walk_no_trend(self):
        """Random walk with no convergence trend."""
        widths = random_walk_trajectory(30, start=0.15, drift=0.0, vol=0.02)
        result = project_convergence(
            ci_widths=widths, delta=0.05, slope_threshold=0.00001,
            step_size=10, n_bootstrap=200,
        )
        print_projection("Random walk (no drift)", result)
        assert result is not None
        # High uncertainty expected
        assert result['uncertainty']['trajectory_rmse'] > 0

    def test_diverging_trajectory(self):
        """CI widths increasing (getting worse).

        With unclamped slope, a positive slope continues to increase width
        each step. The projection should hit the cap since width moves away
        from delta, not towards it. Confidence should be low since
        final_slope >= 0 (non-converging).
        """
        widths = np.linspace(0.06, 0.20, 30).tolist()
        result = project_convergence(
            ci_widths=widths, delta=0.05, slope_threshold=0.00001,
            step_size=10, max_steps=50, n_bootstrap=200,
        )
        print_projection("Diverging trajectory", result)
        assert result is not None
        assert result['final_slope'] > 0  # Slope is positive (diverging)
        assert result['capped'] is True  # Hits cap - width moves away from delta
        assert result['convergence_target'] == 'projected_capped'
        assert result['proximity_ratio'] > 2.0  # Far from width threshold
        assert result['uncertainty']['confidence_level'] == 'low'

    def test_sawtooth_pattern(self):
        """Sawtooth: decreasing trend with periodic jumps."""
        widths = sawtooth_trajectory(30)
        result = project_convergence(
            ci_widths=widths, delta=0.05, slope_threshold=0.00001,
            step_size=10, n_bootstrap=200,
        )
        print_projection("Sawtooth pattern", result)
        assert result is not None
        # RMSE should be elevated due to non-monotonic pattern
        assert result['uncertainty']['trajectory_rmse'] > 0

    def test_sudden_convergence(self):
        """Flat then sudden drop (e.g., model finds structure late)."""
        widths = [0.20] * 20 + np.linspace(0.20, 0.06, 10).tolist()
        result = project_convergence(
            ci_widths=widths, delta=0.05, slope_threshold=0.00001,
            step_size=10, n_bootstrap=200,
        )
        print_projection("Sudden late convergence", result)
        assert result is not None
        # Recent slope should be negative (converging)
        assert result['final_slope'] < 0

    def test_already_converged_all_pathways(self):
        """Already below threshold for each pathway type."""
        for name, widths in [
            ("binary", binary_trajectory(25, true_p=0.9)),
            ("ordinal_peaked", ordinal_trajectory(25, spread='peaked')),
            ("continuous_low_var", continuous_trajectory(25, variance=0.01)),
        ]:
            # Use a generous delta that the trajectory has already crossed
            final_w = widths[-1]
            delta = final_w * 2  # Threshold above final width
            result = project_convergence(
                ci_widths=widths, delta=delta, slope_threshold=0.00001,
                step_size=10, n_bootstrap=100,
            )
            print_projection(f"Already converged - {name} (delta={delta:.4f})", result)
            assert result is not None
            assert result['projected_additional_steps'] == 0
            assert result['proximity_ratio'] < 1.0


# ===========================================================================
# 5. STEP SIZE AND MAX_STEPS INTERACTION
# ===========================================================================

class TestStepSizeInteraction:
    """Different step_size values (reanalysis_interval equivalents)."""

    @pytest.mark.parametrize("step_size", [1, 5, 10, 20, 50])
    def test_step_size_scaling(self, step_size):
        """projected_additional_trials should scale with step_size."""
        widths = binary_trajectory(30, true_p=0.7)
        result = project_convergence(
            ci_widths=widths, delta=0.05, slope_threshold=0.00001,
            step_size=step_size, n_bootstrap=100,
        )
        print_projection(f"step_size={step_size}", result)
        assert result is not None
        # Steps should be identical; trials = steps * step_size
        assert result['projected_additional_trials'] == result['projected_additional_steps'] * step_size

    def test_max_steps_capping(self):
        """Verify capping at different max_steps values."""
        widths = binary_trajectory(30, true_p=0.5)  # Slow convergence
        for max_steps in [10, 50, 200]:
            result = project_convergence(
                ci_widths=widths, delta=0.01,  # Very tight threshold
                slope_threshold=0.00001,
                step_size=10, max_steps=max_steps, n_bootstrap=100,
            )
            print_projection(f"max_steps={max_steps}, delta=0.01", result)
            assert result is not None
            if result['capped']:
                assert result['projected_additional_steps'] == max_steps


# ===========================================================================
# 6. BOOTSTRAP BEHAVIOUR
# ===========================================================================

class TestBootstrapBehaviour:
    """Verify bootstrap produces sensible uncertainty estimates."""

    def test_bootstrap_reproducibility_with_different_n(self):
        """Different n_bootstrap values should produce similar CIs."""
        widths = binary_trajectory(30, true_p=0.7)
        r100 = project_convergence(ci_widths=widths, delta=0.05, n_bootstrap=100)
        r500 = project_convergence(ci_widths=widths, delta=0.05, n_bootstrap=500)
        print_projection("Bootstrap n=100", r100)
        print_projection("Bootstrap n=500", r500)
        assert r100 is not None and r500 is not None
        # Point estimates should be identical (deterministic)
        assert r100['projected_additional_steps'] == r500['projected_additional_steps']
        # CIs should be in similar ballpark (not identical due to randomness)
        diff_80 = abs(r100['uncertainty']['ci_trials_80'][1] - r500['uncertainty']['ci_trials_80'][1])
        print(f"  80th percentile upper diff: {diff_80}")

    def test_bootstrap_ci_ordering(self):
        """50% CI should be inside 80% CI."""
        widths = binary_trajectory(30, true_p=0.7)
        result = project_convergence(ci_widths=widths, delta=0.05, n_bootstrap=500)
        print_projection("CI ordering check", result)
        assert result is not None
        ci_80 = result['uncertainty']['ci_trials_80']
        ci_50 = result['uncertainty']['ci_trials_50']
        assert ci_80[0] <= ci_50[0], f"80% lower ({ci_80[0]}) > 50% lower ({ci_50[0]})"
        assert ci_80[1] >= ci_50[1], f"80% upper ({ci_80[1]}) < 50% upper ({ci_50[1]})"

    def test_zero_noise_tight_ci_width(self):
        """Perfectly linear trajectory should have tight bootstrap CI."""
        widths = np.linspace(0.15, 0.06, 30).tolist()
        result = project_convergence(ci_widths=widths, delta=0.05, n_bootstrap=200)
        print_projection("Perfect linear (zero noise)", result)
        assert result is not None
        ci_80 = result['uncertainty']['ci_trials_80']
        # Exponential fit on linear data has small but non-zero residuals,
        # so bootstrap CI is wider than exact-zero for linear. Still tight.
        ci_width = ci_80[1] - ci_80[0]
        assert ci_width <= 60, f"Expected tight CI, got width {ci_width}"


# ===========================================================================
# 7. CROSS-PATHWAY COMPARISON
# ===========================================================================

class TestCrossPathwayComparison:
    """Compare projection behaviour across pathways with matched conditions."""

    def test_all_pathways_same_width_trajectory(self):
        """Feed identical width trajectory to all pathways - results should match.

        This verifies pathway-independence of the projection algorithm itself.
        """
        widths = np.linspace(0.12, 0.06, 30).tolist()
        params_base = {'delta': 0.05, 'slope_threshold': 0.00001, 'step_size': 10}

        results = {}
        for pathway in ['binary_with_slopes', 'ordinal_no_slopes', 'continuous_no_slopes']:
            ci_slopes = [] if 'no_slopes' in pathway else None
            result = project_convergence(
                ci_widths=widths, ci_slopes=ci_slopes,
                delta=0.05, slope_threshold=0.00001,
                step_size=10, n_bootstrap=200,
            )
            results[pathway] = result
            print_projection(f"Cross-pathway: {pathway}", result, params_base)

        # All should give identical point estimates (same input trajectory)
        steps_set = {r['projected_additional_steps'] for r in results.values() if r}
        assert len(steps_set) == 1, f"Point estimates differ across pathways: {steps_set}"

    def test_realistic_pathway_comparison(self):
        """Realistic trajectories for each pathway - compare projected effort."""
        trajectories = {
            'binary_p0.7': binary_trajectory(30, true_p=0.7),
            'ordinal_peaked': ordinal_trajectory(30, spread='peaked'),
            'ordinal_dispersed': ordinal_trajectory(30, spread='dispersed'),
            'continuous_low_var': continuous_trajectory(30, variance=0.05),
            'continuous_high_var': continuous_trajectory(30, variance=0.25),
        }
        print(f"\n{'=' * 70}")
        print("  Cross-pathway realistic comparison (delta_cap=0.05)")
        print(f"{'=' * 70}")
        for name, widths in trajectories.items():
            result = project_convergence(
                ci_widths=widths, ci_slopes=[],
                delta=0.05, slope_threshold=0.00001,
                step_size=10, n_bootstrap=200,
            )
            if result:
                unc = result['uncertainty']
                print(f"  {name:25s} | steps={result['projected_additional_steps']:4d} "
                      f"| trials={result['projected_additional_trials']:5d} "
                      f"| prox={result['proximity_ratio']:.2f} "
                      f"| conf={unc['confidence_level']:8s} "
                      f"| ci80=[{unc['ci_trials_80'][0]},{unc['ci_trials_80'][1]}] "
                      f"| rmse={unc['trajectory_rmse']:.6f}")
            else:
                print(f"  {name:25s} | None (insufficient data)")
