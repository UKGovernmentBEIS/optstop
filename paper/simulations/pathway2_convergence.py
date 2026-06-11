#!/usr/bin/env python3
"""
Pathway 2 Convergence Analysis: Corrected Under Stabilisation Interpretation

Previous analyses (pathway2_comprehensive.py, pathway2_deep_analysis.py) used shape-based
ground truth: unimodals = should-stop, bimodals = should-not-stop. This was WRONG.

Pathway 2's purpose is convergence detection: has the entropy CI width stabilised such that
additional data provides diminishing returns? Under this interpretation:
- Converged bimodal → SHOULD stop (we know it's bimodal, more data won't change that)
- Converged uniform → SHOULD stop
- Still-changing distribution → should NOT stop

KEY FINDING FROM V1: Relative change is fundamentally the wrong metric. Even the truly
converged mid_ordinal case (CI width=0.024) has a TRUE relative change of ~29% per check
because CI width follows exponential decay. The 30.5% observed relative change is almost
entirely real convergence dynamics, NOT MCMC noise.

The correct metric is ABSOLUTE CI WIDTH: is the CI narrow enough that the entropy estimate
is practically useful? Below some threshold, further narrowing doesn't matter.

This simulation tests absolute width thresholds, relative change thresholds, and hybrid
mechanisms against an absolute-width-based ground truth.

Author: Claude (analysis for optstop package developer)
Date: 2026-02-26
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# ============================================================================
# Section A: Convergence Trajectory Model (Exponential Decay)
# ============================================================================

@dataclass
class ConvergenceScenario:
    """
    A single convergence scenario with known ground truth.

    Uses exponential decay: true_width(c) = w_inf + (w_0 - w_inf) * exp(-lambda * c)

    Calibrated from actual matrix results:
    - mid_ordinal: w_0≈0.35, w_inf≈0.015, lambda≈0.72, final width=0.024 at check 5
    - high_ordinal: w_0≈0.45, w_inf≈0.060, lambda≈0.40, final width=0.097 at check 5
    """
    name: str
    w_0: float              # Initial CI width (check 0)
    w_inf: float            # Asymptotic CI width (check → ∞)
    decay_rate: float       # Exponential decay rate (lambda)
    sigma_noise: float      # MCMC noise (proportional to width)
    distribution_type: str  # For reporting

    def true_width(self, check: int) -> float:
        """Compute noise-free CI width at a given check."""
        return self.w_inf + (self.w_0 - self.w_inf) * np.exp(-self.decay_rate * check)

    def true_relative_change(self, check: int) -> float:
        """Compute noise-free relative change between check-1 and check."""
        if check < 1:
            return 1.0
        w_prev = self.true_width(check - 1)
        w_curr = self.true_width(check)
        return abs(w_curr - w_prev) / w_prev if w_prev > 0 else 1.0

    def is_converged(self, check: int, width_threshold: float = 0.10) -> bool:
        """
        Ground truth: has the CI width reached a practically useful level?

        Threshold is on scaled [0,1] entropy axis:
        - 0.05 = know entropy to ±2.5% → very precise
        - 0.10 = know entropy to ±5.0% → precise
        - 0.15 = know entropy to ±7.5% → moderate
        """
        return self.true_width(check) < width_threshold

    def generate_trajectory(self, n_checks: int, rng: np.random.Generator) -> List[float]:
        """Generate a noisy observed CI width trajectory."""
        trajectory = []
        for c in range(1, n_checks + 1):
            tw = self.true_width(c)
            noise = rng.normal(0, self.sigma_noise * tw)
            observed = max(0.001, tw + noise)  # CI width can't be negative
            trajectory.append(observed)
        return trajectory


def build_scenario_catalogue() -> List[ConvergenceScenario]:
    """
    Build comprehensive scenario catalogue.

    Decay rate (lambda) interpretations:
    - 0.10: Very slow — CI barely changes. At check 5: width ≈ 0.61 * w_0
    - 0.20: Slow — gradual convergence. At check 5: width ≈ 0.37 * w_0
    - 0.40: Medium — typical Bayesian convergence. At check 5: width ≈ 0.14 * w_0
    - 0.70: Fast — like mid_ordinal. At check 5: width ≈ 0.03 * w_0
    - 1.00: Very fast — rapid convergence. At check 5: width ≈ 0.007 * w_0
    """
    scenarios = []

    # Parameter space
    w0_values = [0.20, 0.30, 0.40, 0.50]
    winf_values = [0.005, 0.015, 0.030, 0.060]
    decay_rates = [0.10, 0.20, 0.40, 0.70, 1.00]
    noise_levels = [0.05, 0.10, 0.15, 0.20, 0.30]

    rate_to_type = {
        0.10: 'very_slow',
        0.20: 'slow',
        0.40: 'medium',
        0.70: 'fast',
        1.00: 'very_fast',
    }

    for w0 in w0_values:
        for winf in winf_values:
            if winf >= w0 * 0.5:
                continue  # Skip unrealistic: asymptote must be well below initial
            for rate in decay_rates:
                for noise in noise_levels:
                    dtype = rate_to_type[rate]
                    name = f"w0={w0:.2f}_winf={winf:.3f}_r={rate:.2f}_n={noise:.2f}"
                    scenarios.append(ConvergenceScenario(
                        name=name, w_0=w0, w_inf=winf,
                        decay_rate=rate, sigma_noise=noise,
                        distribution_type=dtype,
                    ))

    return scenarios


# ============================================================================
# Section B: Stabilisation Mechanisms
# ============================================================================

def apply_mechanism(name: str, trajectory: List[float], check_idx: int) -> bool:
    """
    Apply a stabilisation mechanism at a given check index.
    Returns True if the mechanism says "stop" at this check.
    check_idx is 0-based index into trajectory.
    """
    if check_idx < 0:
        return False

    current = trajectory[check_idx]

    # --- Absolute width with minimum checks ---
    if name.startswith('abs_width_min_'):
        parts = name.split('_')
        threshold = float(parts[3])
        min_checks = int(parts[4])
        if check_idx + 1 < min_checks:
            return False
        return current < threshold

    # --- Absolute CI width ---
    if name.startswith('abs_width_'):
        threshold = float(name.split('_')[2])
        return current < threshold

    # Need at least 2 data points for relative measures
    if check_idx < 1:
        return False
    previous = trajectory[check_idx - 1]

    # --- Current implementation: two-point relative change ---
    if name == 'current_relchg_0.002':
        rel_change = abs(current - previous) / previous if previous > 0 else 1.0
        return rel_change < 0.002

    # --- Two-point relative change at various thresholds ---
    if name.startswith('relchg_'):
        threshold = float(name.split('_')[1])
        rel_change = abs(current - previous) / previous if previous > 0 else 1.0
        return rel_change < threshold

    # --- Mean relative change over last W checks ---
    if name.startswith('mean_relchg_'):
        parts = name.split('_')
        window = int(parts[2])
        threshold = float(parts[3])
        if check_idx < window:
            return False
        changes = []
        for i in range(check_idx - window + 1, check_idx + 1):
            if i > 0 and trajectory[i - 1] > 0:
                changes.append(abs(trajectory[i] - trajectory[i - 1]) / trajectory[i - 1])
        if not changes:
            return False
        return np.mean(changes) < threshold

    # --- Max relative change over last W checks ---
    if name.startswith('max_relchg_'):
        parts = name.split('_')
        window = int(parts[2])
        threshold = float(parts[3])
        if check_idx < window:
            return False
        changes = []
        for i in range(check_idx - window + 1, check_idx + 1):
            if i > 0 and trajectory[i - 1] > 0:
                changes.append(abs(trajectory[i] - trajectory[i - 1]) / trajectory[i - 1])
        if not changes:
            return False
        return max(changes) < threshold

    # --- Coefficient of variation of last W widths ---
    if name.startswith('cv_'):
        parts = name.split('_')
        window = int(parts[1])
        threshold = float(parts[2])
        if check_idx < window - 1:
            return False
        recent = trajectory[check_idx - window + 1: check_idx + 1]
        mean_w = np.mean(recent)
        if mean_w <= 0:
            return False
        cv = np.std(recent) / mean_w
        return cv < threshold

    # --- Normalised slope over last W checks ---
    if name.startswith('norm_slope_'):
        parts = name.split('_')
        window = int(parts[2])
        threshold = float(parts[3])
        if check_idx < window - 1:
            return False
        recent = trajectory[check_idx - window + 1: check_idx + 1]
        mean_w = np.mean(recent)
        if mean_w <= 0:
            return False
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        norm_slope = abs(slope) / mean_w
        return norm_slope < threshold

    # --- Combined: absolute width AND relative change ---
    if name.startswith('combined_abs_relchg_'):
        parts = name.split('_')
        abs_thresh = float(parts[3])
        rel_thresh = float(parts[4])
        width_ok = current < abs_thresh
        rel_change = abs(current - previous) / previous if previous > 0 else 1.0
        relchg_ok = rel_change < rel_thresh
        return width_ok and relchg_ok

    # --- Combined: absolute width OR (relative change AND small absolute) ---
    if name.startswith('combined_or_'):
        parts = name.split('_')
        abs_primary = float(parts[2])
        abs_secondary = float(parts[3])
        rel_thresh = float(parts[4])
        # Primary: width narrow enough
        if current < abs_primary:
            return True
        # Secondary: width moderately narrow AND stabilising
        rel_change = abs(current - previous) / previous if previous > 0 else 1.0
        return current < abs_secondary and rel_change < rel_thresh

    raise ValueError(f"Unknown mechanism: {name}")


# All mechanisms to test
MECHANISMS = [
    # Current implementation
    'current_relchg_0.002',

    # Relative change (the concept P2 currently uses, at various thresholds)
    'relchg_0.10',
    'relchg_0.15',
    'relchg_0.20',
    'relchg_0.30',
    'relchg_0.40',
    'relchg_0.50',

    # Mean relative change over last 3 checks
    'mean_relchg_3_0.15',
    'mean_relchg_3_0.20',
    'mean_relchg_3_0.30',
    'mean_relchg_3_0.40',

    # Max relative change (all recent checks must be stable)
    'max_relchg_3_0.20',
    'max_relchg_3_0.30',
    'max_relchg_3_0.40',

    # Absolute CI width (the proposed better metric)
    'abs_width_0.03',
    'abs_width_0.05',
    'abs_width_0.08',
    'abs_width_0.10',
    'abs_width_0.12',
    'abs_width_0.15',

    # Absolute CI width with minimum check requirement
    'abs_width_min_0.05_3',
    'abs_width_min_0.08_3',
    'abs_width_min_0.10_3',
    'abs_width_min_0.12_3',
    'abs_width_min_0.15_3',

    # Coefficient of variation (low CV = widths not changing much)
    'cv_3_0.08',
    'cv_3_0.10',
    'cv_3_0.15',
    'cv_3_0.20',

    # Normalised slope
    'norm_slope_3_0.02',
    'norm_slope_3_0.05',
    'norm_slope_3_0.08',

    # Combined: absolute width AND relative change
    'combined_abs_relchg_0.10_0.30',
    'combined_abs_relchg_0.12_0.25',
    'combined_abs_relchg_0.15_0.20',

    # Combined OR: narrow enough OR (moderately narrow AND stabilising)
    'combined_or_0.05_0.12_0.20',
    'combined_or_0.05_0.15_0.15',
    'combined_or_0.08_0.15_0.20',
]


# ============================================================================
# Section C: Evaluation Engine
# ============================================================================

@dataclass
class MechanismResult:
    """Results for a single mechanism."""
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def tpr(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def fpr(self) -> float:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    @property
    def f1(self) -> float:
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tpr
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def evaluate_at_check(mechanism: str, scenario: ConvergenceScenario,
                      n_checks: int, n_mc: int, rng: np.random.Generator,
                      width_threshold: float = 0.10) -> MechanismResult:
    """Evaluate a single mechanism on a single scenario at a fixed check count."""
    result = MechanismResult()
    gt_converged = scenario.is_converged(n_checks, width_threshold)

    for _ in range(n_mc):
        trajectory = scenario.generate_trajectory(n_checks, rng)
        mechanism_fires = apply_mechanism(mechanism, trajectory, n_checks - 1)

        if gt_converged and mechanism_fires:
            result.tp += 1
        elif gt_converged and not mechanism_fires:
            result.fn += 1
        elif not gt_converged and mechanism_fires:
            result.fp += 1
        else:
            result.tn += 1

    return result


# ============================================================================
# Section D: Analysis Pipeline
# ============================================================================

def analyse_convergence_landscape(scenarios: List[ConvergenceScenario],
                                  width_threshold: float = 0.10) -> Dict:
    """Analyse which scenarios are converged at each check count."""
    results = {}
    for n_checks in [3, 5, 8, 12]:
        n_conv = sum(1 for s in scenarios if s.is_converged(n_checks, width_threshold))
        by_rate = {}
        for rate in [0.10, 0.20, 0.40, 0.70, 1.00]:
            rate_scen = [s for s in scenarios if s.decay_rate == rate]
            by_rate[rate] = sum(1 for s in rate_scen if s.is_converged(n_checks, width_threshold))
        results[n_checks] = {
            'converged': n_conv,
            'total': len(scenarios),
            'pct': n_conv / len(scenarios) * 100,
            'by_rate': by_rate,
        }
    return results


def show_trajectory_examples(scenarios: List[ConvergenceScenario],
                             rng: np.random.Generator):
    """Show example convergence trajectories to illustrate the model."""
    # Pick one scenario per decay rate, all at noise=0.15, w_0=0.35, w_inf=0.015
    print("\nExample trajectories (w_0=0.35, w_inf=0.015, noise=0.15):")
    print(f"{'Rate':<8} {'Type':<12} {'Check1':>8} {'Check2':>8} {'Check3':>8} "
          f"{'Check4':>8} {'Check5':>8} {'TrueRC5':>8} {'Conv@5?':>8}")
    print("-" * 80)

    for rate in [0.10, 0.20, 0.40, 0.70, 1.00]:
        s = ConvergenceScenario(f"example_{rate}", 0.35, 0.015, rate, 0.15, 'example')
        traj = [s.true_width(c) for c in range(1, 6)]
        true_rc = s.true_relative_change(5)
        conv = "YES" if s.is_converged(5, 0.10) else "no"
        print(f"{rate:<8.2f} {s.distribution_type:<12} "
              + " ".join(f"{w:>8.4f}" for w in traj)
              + f" {true_rc:>8.3f} {conv:>8}")


def run_main_evaluation(scenarios: List[ConvergenceScenario],
                        mechanisms: List[str], n_mc: int,
                        rng: np.random.Generator,
                        n_checks: int = 5,
                        width_threshold: float = 0.10) -> Dict:
    """Run the main evaluation at a given check count."""
    n_converged = sum(1 for s in scenarios if s.is_converged(n_checks, width_threshold))
    n_not = len(scenarios) - n_converged

    print(f"\n  Ground truth at check {n_checks} (threshold={width_threshold}):")
    print(f"    Converged: {n_converged}/{len(scenarios)} ({n_converged/len(scenarios)*100:.1f}%)")
    print(f"    Not converged: {n_not}/{len(scenarios)} ({n_not/len(scenarios)*100:.1f}%)")

    results = {}
    for m_idx, mechanism in enumerate(mechanisms):
        print(f"\r  [{m_idx+1}/{len(mechanisms)}] {mechanism:<40}", end='', flush=True)

        agg = MechanismResult()
        for scenario in scenarios:
            res = evaluate_at_check(mechanism, scenario, n_checks=n_checks,
                                    n_mc=n_mc, rng=rng, width_threshold=width_threshold)
            agg.tp += res.tp
            agg.fp += res.fp
            agg.tn += res.tn
            agg.fn += res.fn

        results[mechanism] = {
            'tpr': agg.tpr, 'fpr': agg.fpr, 'f1': agg.f1,
            'accuracy': agg.accuracy,
            'tp': agg.tp, 'fp': agg.fp, 'tn': agg.tn, 'fn': agg.fn,
        }

    print()
    return results


def run_by_rate(scenarios: List[ConvergenceScenario],
                mechanisms: List[str], n_mc: int,
                rng: np.random.Generator,
                n_checks: int = 5,
                width_threshold: float = 0.10) -> Dict:
    """Evaluate by convergence rate."""
    results = {}
    for rate in [0.10, 0.20, 0.40, 0.70, 1.00]:
        rate_scen = [s for s in scenarios if s.decay_rate == rate]
        n_conv = sum(1 for s in rate_scen if s.is_converged(n_checks, width_threshold))

        results[str(rate)] = {'n_converged': n_conv, 'n_total': len(rate_scen)}

        for mechanism in mechanisms:
            agg = MechanismResult()
            for scenario in rate_scen:
                res = evaluate_at_check(mechanism, scenario, n_checks=n_checks,
                                        n_mc=n_mc, rng=rng, width_threshold=width_threshold)
                agg.tp += res.tp
                agg.fp += res.fp
                agg.tn += res.tn
                agg.fn += res.fn

            results[str(rate)][mechanism] = {
                'tpr': agg.tpr, 'fpr': agg.fpr, 'f1': agg.f1,
                'accuracy': agg.accuracy,
            }

    return results


def run_by_noise(scenarios: List[ConvergenceScenario],
                 mechanisms: List[str], n_mc: int,
                 rng: np.random.Generator,
                 n_checks: int = 5,
                 width_threshold: float = 0.10) -> Dict:
    """Evaluate by noise level."""
    results = {}
    for noise in [0.05, 0.10, 0.15, 0.20, 0.30]:
        noise_scen = [s for s in scenarios if s.sigma_noise == noise]
        n_conv = sum(1 for s in noise_scen if s.is_converged(n_checks, width_threshold))

        results[str(noise)] = {'n_converged': n_conv, 'n_total': len(noise_scen)}

        for mechanism in mechanisms:
            agg = MechanismResult()
            for scenario in noise_scen:
                res = evaluate_at_check(mechanism, scenario, n_checks=n_checks,
                                        n_mc=n_mc, rng=rng, width_threshold=width_threshold)
                agg.tp += res.tp
                agg.fp += res.fp
                agg.tn += res.tn
                agg.fn += res.fn

            results[str(noise)][mechanism] = {
                'tpr': agg.tpr, 'fpr': agg.fpr, 'f1': agg.f1,
                'accuracy': agg.accuracy,
            }

    return results


def validate_against_matrix(mechanisms: List[str], n_mc: int,
                            rng: np.random.Generator,
                            width_threshold: float = 0.10) -> Dict:
    """
    Validate against actual matrix shadow results.

    Calibrated scenarios:
    - mid_ordinal: CI width ≈ 0.024 at check 5, relative_change ≈ 0.305
      → lambda ≈ 0.72 gives true_width(5) ≈ 0.024
    - high_ordinal: CI width ≈ 0.097 at check 5, relative_change ≈ 0.192
      → lambda ≈ 0.40 gives true_width(5) ≈ 0.088 (close)
    - Synthetic "not yet converged": slow convergence, still wide at check 5
    """
    validation_scenarios = [
        ConvergenceScenario(
            'mid_ordinal', w_0=0.35, w_inf=0.015, decay_rate=0.72,
            sigma_noise=0.25, distribution_type='matrix_mid',
        ),
        ConvergenceScenario(
            'high_ordinal', w_0=0.45, w_inf=0.060, decay_rate=0.40,
            sigma_noise=0.15, distribution_type='matrix_high',
        ),
        ConvergenceScenario(
            'still_converging', w_0=0.40, w_inf=0.020, decay_rate=0.20,
            sigma_noise=0.15, distribution_type='not_yet',
        ),
        ConvergenceScenario(
            'barely_started', w_0=0.50, w_inf=0.010, decay_rate=0.10,
            sigma_noise=0.15, distribution_type='early',
        ),
    ]

    results = {}
    for scenario in validation_scenarios:
        n_checks = 5
        gt = scenario.is_converged(n_checks, width_threshold)
        tw = scenario.true_width(n_checks)
        trc = scenario.true_relative_change(n_checks)

        scenario_results = {
            'ground_truth': gt,
            'true_width_at_5': tw,
            'true_relative_change_at_5': trc,
            'w_inf': scenario.w_inf,
            'mechanisms': {},
        }

        for mechanism in mechanisms:
            res = evaluate_at_check(mechanism, scenario, n_checks=n_checks,
                                    n_mc=n_mc, rng=rng, width_threshold=width_threshold)
            fire_rate = res.tpr if gt else res.fpr
            correct_rate = res.tpr if gt else (1 - res.fpr)

            scenario_results['mechanisms'][mechanism] = {
                'fire_rate': fire_rate,
                'correct_rate': correct_rate,
            }

        results[scenario.name] = scenario_results

    return results


def threshold_sensitivity(scenarios: List[ConvergenceScenario],
                          n_mc: int, rng: np.random.Generator,
                          n_checks: int = 5) -> Dict:
    """
    Test how abs_width performance changes with different ground truth thresholds.
    This answers: what is the right definition of "converged"?
    """
    gt_thresholds = [0.05, 0.08, 0.10, 0.12, 0.15]
    mech_thresholds = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]

    results = {}
    for gt in gt_thresholds:
        results[str(gt)] = {}
        n_conv = sum(1 for s in scenarios if s.is_converged(n_checks, gt))
        results[str(gt)]['n_converged'] = n_conv
        results[str(gt)]['n_total'] = len(scenarios)

        for mt in mech_thresholds:
            mechanism = f'abs_width_{mt}'
            agg = MechanismResult()
            for scenario in scenarios:
                res = evaluate_at_check(mechanism, scenario, n_checks=n_checks,
                                        n_mc=n_mc, rng=rng, width_threshold=gt)
                agg.tp += res.tp
                agg.fp += res.fp
                agg.tn += res.tn
                agg.fn += res.fn

            results[str(gt)][f'abs_width_{mt}'] = {
                'tpr': agg.tpr, 'fpr': agg.fpr, 'f1': agg.f1,
                'accuracy': agg.accuracy,
            }

    return results


# ============================================================================
# Section E: Full Pipeline
# ============================================================================

def run_full_evaluation(n_mc: int = 500, seed: int = 42) -> Dict:
    """Run the complete evaluation pipeline."""
    rng = np.random.default_rng(seed)

    print("=" * 80)
    print("Pathway 2 Convergence Analysis (v2 — Corrected)")
    print("Ground truth: absolute CI width < threshold")
    print("Convergence model: exponential decay (calibrated from matrix data)")
    print("=" * 80)

    # Build scenarios
    scenarios = build_scenario_catalogue()
    print(f"\nScenario catalogue: {len(scenarios)} scenarios")
    print(f"  w_0: [0.20, 0.30, 0.40, 0.50]")
    print(f"  w_inf: [0.005, 0.015, 0.030, 0.060]")
    print(f"  Decay rates (lambda): [0.10, 0.20, 0.40, 0.70, 1.00]")
    print(f"  MCMC noise (sigma): [0.05, 0.10, 0.15, 0.20, 0.30]")
    print(f"  MC instances: {n_mc}")
    print(f"  Total instances: {len(scenarios) * n_mc:,}")

    # Show example trajectories
    show_trajectory_examples(scenarios, rng)

    # Convergence landscape
    print(f"\n{'=' * 80}")
    print("CONVERGENCE LANDSCAPE (ground truth threshold = 0.10)")
    print(f"{'=' * 80}")
    landscape = analyse_convergence_landscape(scenarios, width_threshold=0.10)
    for n_checks, info in landscape.items():
        print(f"  Check {n_checks}: {info['converged']}/{info['total']} converged ({info['pct']:.1f}%)")
        for rate, count in info['by_rate'].items():
            total_at_rate = sum(1 for s in scenarios if s.decay_rate == rate)
            print(f"    rate={rate:.2f}: {count}/{total_at_rate}")

    mechanisms = MECHANISMS
    print(f"\nMechanisms to test: {len(mechanisms)}")

    # ===== MAIN EVALUATION: n_checks=5, threshold=0.10 =====
    print(f"\n{'=' * 80}")
    print(f"MAIN EVALUATION: n_checks=5, convergence threshold=0.10")
    print(f"{'=' * 80}")
    results_5 = run_main_evaluation(scenarios, mechanisms, n_mc, rng,
                                    n_checks=5, width_threshold=0.10)

    # Print sorted by F1
    print(f"\n{'Mechanism':<40} {'TPR':>7} {'FPR':>7} {'F1':>7} {'Acc':>7}")
    print("-" * 70)
    sorted_5 = sorted(results_5.items(), key=lambda x: x[1]['f1'], reverse=True)
    for name, r in sorted_5:
        print(f"{name:<40} {r['tpr']:>7.3f} {r['fpr']:>7.3f} {r['f1']:>7.3f} {r['accuracy']:>7.3f}")

    # ===== SECONDARY EVALUATION: n_checks=8 =====
    print(f"\n{'=' * 80}")
    print(f"SECONDARY EVALUATION: n_checks=8, convergence threshold=0.10")
    print(f"{'=' * 80}")
    results_8 = run_main_evaluation(scenarios, mechanisms, n_mc, rng,
                                    n_checks=8, width_threshold=0.10)

    sorted_8 = sorted(results_8.items(), key=lambda x: x[1]['f1'], reverse=True)
    print(f"\n{'Mechanism':<40} {'TPR':>7} {'FPR':>7} {'F1':>7} {'Acc':>7}")
    print("-" * 70)
    for name, r in sorted_8[:15]:
        print(f"{name:<40} {r['tpr']:>7.3f} {r['fpr']:>7.3f} {r['f1']:>7.3f} {r['accuracy']:>7.3f}")

    # ===== TOP MECHANISMS ANALYSIS =====
    top_mechs = [name for name, _ in sorted_5[:10]]

    # By convergence rate
    print(f"\n{'=' * 80}")
    print(f"TOP MECHANISMS BY CONVERGENCE RATE (n_checks=5)")
    print(f"{'=' * 80}")
    rate_results = run_by_rate(scenarios, top_mechs, n_mc, rng, n_checks=5)

    for rate in ['0.1', '0.2', '0.4', '0.7', '1.0']:
        info = rate_results[rate]
        print(f"\n  Rate={rate} ({info['n_converged']}/{info['n_total']} converged):")
        print(f"  {'Mechanism':<40} {'TPR':>7} {'FPR':>7} {'F1':>7}")
        print(f"  {'-' * 60}")
        for m in top_mechs:
            r = info[m]
            print(f"  {m:<40} {r['tpr']:>7.3f} {r['fpr']:>7.3f} {r['f1']:>7.3f}")

    # By noise level
    print(f"\n{'=' * 80}")
    print(f"TOP MECHANISMS BY NOISE LEVEL (n_checks=5)")
    print(f"{'=' * 80}")
    noise_results = run_by_noise(scenarios, top_mechs, n_mc, rng, n_checks=5)

    for noise in ['0.05', '0.1', '0.15', '0.2', '0.3']:
        info = noise_results[noise]
        print(f"\n  Noise={noise} ({info['n_converged']}/{info['n_total']} converged):")
        print(f"  {'Mechanism':<40} {'TPR':>7} {'FPR':>7} {'F1':>7}")
        print(f"  {'-' * 60}")
        for m in top_mechs:
            r = info[m]
            print(f"  {m:<40} {r['tpr']:>7.3f} {r['fpr']:>7.3f} {r['f1']:>7.3f}")

    # ===== THRESHOLD SENSITIVITY =====
    print(f"\n{'=' * 80}")
    print(f"THRESHOLD SENSITIVITY: abs_width mechanism vs ground truth threshold")
    print(f"{'=' * 80}")
    thresh_results = threshold_sensitivity(scenarios, n_mc, rng, n_checks=5)

    print(f"\n  GT threshold → mechanism threshold that maximises F1:")
    for gt_str, info in thresh_results.items():
        best_mech = None
        best_f1 = -1
        for k, v in info.items():
            if k.startswith('abs_width') and v.get('f1', 0) > best_f1:
                best_f1 = v['f1']
                best_mech = k
        n_conv = info['n_converged']
        print(f"  GT={gt_str}: {n_conv}/{info['n_total']} converged → "
              f"best={best_mech} (F1={best_f1:.3f})")

    # Full sensitivity matrix
    gt_thresholds = ['0.05', '0.08', '0.1', '0.12', '0.15']
    mech_names = [f'abs_width_{t}' for t in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]]
    print(f"\n  F1 matrix (rows=GT threshold, cols=mechanism threshold):")
    header = "  GT\\Mech " + " ".join(f"{m.split('_')[2]:>7}" for m in mech_names)
    print(header)
    for gt in gt_thresholds:
        row = f"  {gt:<8}"
        for m in mech_names:
            f1 = thresh_results[gt].get(m, {}).get('f1', 0)
            row += f" {f1:>7.3f}"
        print(row)

    # ===== REAL DATA VALIDATION =====
    print(f"\n{'=' * 80}")
    print(f"REAL DATA VALIDATION")
    print(f"{'=' * 80}")
    validation = validate_against_matrix(top_mechs, n_mc, rng, width_threshold=0.10)

    for name, vr in validation.items():
        gt = "CONVERGED" if vr['ground_truth'] else "NOT converged"
        tw = vr['true_width_at_5']
        trc = vr['true_relative_change_at_5']
        print(f"\n  {name}: {gt}")
        print(f"    True width at check 5: {tw:.4f}")
        print(f"    True relative change: {trc:.3f}")
        print(f"    {'Mechanism':<40} {'Fire%':>7} {'Correct%':>9}")
        print(f"    {'-' * 58}")
        for m in top_mechs:
            mr = vr['mechanisms'].get(m, {})
            fire = mr.get('fire_rate', 0) * 100
            correct = mr.get('correct_rate', 0) * 100
            print(f"    {m:<40} {fire:>6.1f}% {correct:>8.1f}%")

    # ===== WHY RELATIVE CHANGE FAILS =====
    print(f"\n{'=' * 80}")
    print(f"WHY RELATIVE CHANGE FAILS: True relative changes at check 5")
    print(f"{'=' * 80}")
    print(f"\n  Exponential decay: true_rc(c) = 1 - exp(-lambda)")
    print(f"  This is CONSTANT regardless of check number!")
    for rate in [0.10, 0.20, 0.40, 0.70, 1.00]:
        true_rc = 1 - np.exp(-rate)
        print(f"    lambda={rate:.2f}: true_relative_change = {true_rc:.3f} ({true_rc*100:.1f}%)")
    print(f"\n  Even at lambda=0.70 (fast convergence, like mid_ordinal):")
    print(f"  true relative change = {1-np.exp(-0.70):.3f} = 50.3% PER CHECK")
    print(f"  No relative change threshold < 0.50 can detect convergence!")
    print(f"  The threshold 0.002 is ~250× too tight.")

    # ===== ASSEMBLE OUTPUT =====
    output = {
        'parameters': {
            'n_mc': n_mc,
            'seed': seed,
            'n_scenarios': len(scenarios),
            'convergence_model': 'exponential_decay',
            'ground_truth': 'absolute_ci_width',
            'default_width_threshold': 0.10,
        },
        'landscape': landscape,
        'results_n5': results_5,
        'results_n8': results_8,
        'rate_breakdown': rate_results,
        'noise_breakdown': noise_results,
        'threshold_sensitivity': thresh_results,
        'real_data_validation': validation,
    }

    return output


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    output = run_full_evaluation(n_mc=500, seed=42)

    os.makedirs('simulations/output', exist_ok=True)
    results_path = 'simulations/output/pathway2_convergence_results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults saved to {results_path}")

    # Final summary
    print(f"\n{'=' * 80}")
    print(f"FINAL SUMMARY")
    print(f"{'=' * 80}")

    sorted_5 = sorted(output['results_n5'].items(),
                       key=lambda x: x[1]['f1'], reverse=True)
    print(f"\nTop 10 mechanisms at n_checks=5 (by F1):")
    print(f"{'Rank':<5} {'Mechanism':<40} {'TPR':>7} {'FPR':>7} {'F1':>7}")
    print("-" * 65)
    for i, (name, r) in enumerate(sorted_5[:10], 1):
        print(f"{i:<5} {name:<40} {r['tpr']:>7.3f} {r['fpr']:>7.3f} {r['f1']:>7.3f}")

    curr = output['results_n5']['current_relchg_0.002']
    print(f"\nCurrent implementation:")
    print(f"      {'current_relchg_0.002':<40} {curr['tpr']:>7.3f} {curr['fpr']:>7.3f} {curr['f1']:>7.3f}")

    print(f"\nKey insight: Under exponential decay, the relative change between consecutive")
    print(f"checks is CONSTANT at 1-exp(-lambda), regardless of how converged the posterior is.")
    print(f"For mid_ordinal (lambda=0.72), this is 51% per check. No relative-change threshold")
    print(f"below 0.51 can ever fire. Absolute CI width is the correct metric.")
