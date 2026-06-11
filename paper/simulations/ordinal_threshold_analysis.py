#!/usr/bin/env python3
"""
Simulation analysis for proposed changes to optstop ordinal stopping model.

Evaluates three proposals:
  1. Raise entropy_threshold from 0.7 to 0.80
  2. Redesign Pathway 2 to use windowed slope (like binary/continuous)
  3. K-adaptive threshold T(K) = a + b*log2(K)

Also validates against real matrix run data.

Usage:
    python simulations/ordinal_threshold_analysis.py
"""

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.special import entr  # per-element -p*ln(p), base-e

# ── Output directory ──────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── §0  Helpers ───────────────────────────────────────────────────────────

def shannon_entropy_bits(probs: np.ndarray) -> float:
    """Shannon entropy H = -sum(p * log2(p)) in bits."""
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    return float(np.sum(entr(p)) / np.log(2))


def max_entropy_bits(K: int) -> float:
    return float(np.log2(K))


# ── §1  Distribution generators ──────────────────────────────────────────

def discretize_truncnorm(mu: float, sigma: float, K: int) -> np.ndarray:
    """Discrete probability vector from truncated Normal over K categories."""
    if sigma < 1e-8:
        probs = np.zeros(K)
        probs[int(np.clip(np.round(mu), 0, K - 1))] = 1.0
        return probs
    edges = np.arange(K + 1) - 0.5
    cdf_vals = stats.norm.cdf(edges, loc=mu, scale=sigma)
    probs = np.diff(cdf_vals)
    probs = np.clip(probs, 0, None)
    s = probs.sum()
    return probs / s if s > 0 else np.ones(K) / K


def make_bimodal(mu1, mu2, sigma1, sigma2, K, weight1=0.5):
    p1 = discretize_truncnorm(mu1, sigma1, K)
    p2 = discretize_truncnorm(mu2, sigma2, K)
    return weight1 * p1 + (1 - weight1) * p2


def make_near_uniform(K, alpha=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.dirichlet(np.full(K, alpha))


# ══════════════════════════════════════════════════════════════════════════
# §A  PROPOSAL 1: Entropy Threshold Sweep
# ══════════════════════════════════════════════════════════════════════════

def run_proposal1(seed=42):
    """Full entropy threshold analysis across K, location, sigma, and dist type."""
    K_values = [3, 5, 7, 11, 21]
    sigma_fracs = {"tight": 0.05, "moderate": 0.15, "wide": 0.35}
    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    n_dirichlet = 100
    rng = np.random.default_rng(seed)

    rows = []  # each row: dict with K, dtype, location, sigma_label, entropy_frac

    for K in K_values:
        max_ent = max_entropy_bits(K)
        locs = {"edge": 0, "quarter": (K - 1) * 0.25, "center": (K - 1) * 0.5,
                "three_quarter": (K - 1) * 0.75}

        # ── Unimodal ──
        for loc_name, mu in locs.items():
            for sig_name, sig_frac in sigma_fracs.items():
                sigma = max(sig_frac * (K - 1), 0.01)
                probs = discretize_truncnorm(mu, sigma, K)
                ent = shannon_entropy_bits(probs)
                rows.append(dict(K=K, dtype="unimodal", location=loc_name,
                                 sigma_label=sig_name, entropy_bits=ent,
                                 entropy_frac=ent / max_ent))

        # ── Bimodal (symmetric 25/75 split) ──
        for sig_name, sig_frac in sigma_fracs.items():
            sigma = max(sig_frac * (K - 1), 0.01)
            probs = make_bimodal((K-1)*0.25, (K-1)*0.75, sigma, sigma, K)
            ent = shannon_entropy_bits(probs)
            rows.append(dict(K=K, dtype="bimodal", location="quarter_75",
                             sigma_label=sig_name, entropy_bits=ent,
                             entropy_frac=ent / max_ent))

        # ── Bimodal (asymmetric 30/70 split, 60/40 weight) ──
        for sig_name, sig_frac in sigma_fracs.items():
            sigma = max(sig_frac * (K - 1), 0.01)
            probs = make_bimodal((K-1)*0.3, (K-1)*0.7, sigma, sigma, K, weight1=0.6)
            ent = shannon_entropy_bits(probs)
            rows.append(dict(K=K, dtype="bimodal_asym", location="30_70",
                             sigma_label=sig_name, entropy_bits=ent,
                             entropy_frac=ent / max_ent))

        # ── Near-uniform ──
        for i in range(n_dirichlet):
            probs = make_near_uniform(K, alpha=1.0, rng=rng)
            ent = shannon_entropy_bits(probs)
            rows.append(dict(K=K, dtype="near_uniform", location="random",
                             sigma_label=f"draw_{i}", entropy_bits=ent,
                             entropy_frac=ent / max_ent))

    # ── Analyse: FN/FP rates per K per threshold ──
    summary = {}
    for K in K_values:
        kr = [r for r in rows if r["K"] == K]
        unimodal = [r for r in kr if r["dtype"] == "unimodal"]
        ambiguous = [r for r in kr if r["dtype"] in ("bimodal", "bimodal_asym", "near_uniform")]
        bimodal_only = [r for r in kr if r["dtype"] in ("bimodal", "bimodal_asym")]

        summary[K] = {}
        for T in thresholds:
            # False negative: unimodal blocked (entropy_frac > T)
            fn = sum(1 for r in unimodal if r["entropy_frac"] > T)
            # False positive: ambiguous passed (entropy_frac <= T)
            fp = sum(1 for r in ambiguous if r["entropy_frac"] <= T)
            fp_bimodal = sum(1 for r in bimodal_only if r["entropy_frac"] <= T)
            fp_uniform = sum(1 for r in ambiguous if r["dtype"] == "near_uniform"
                             and r["entropy_frac"] <= T)

            # Which unimodal are blocked?
            blocked = [(r["location"], r["sigma_label"], round(r["entropy_frac"], 4))
                       for r in unimodal if r["entropy_frac"] > T]

            summary[K][T] = dict(
                n_unimodal=len(unimodal), n_ambiguous=len(ambiguous),
                fn=fn, fn_rate=fn / len(unimodal) if unimodal else 0,
                fp=fp, fp_rate=fp / len(ambiguous) if ambiguous else 0,
                fp_bimodal=fp_bimodal, fp_uniform=fp_uniform,
                blocked_unimodal=blocked,
            )

    return rows, summary, thresholds


def print_proposal1(summary, thresholds):
    print("\n" + "=" * 90)
    print("§A  PROPOSAL 1: ENTROPY THRESHOLD SWEEP")
    print("=" * 90)
    print()
    print("| K  | T    | Unimodal Blocked (FN) | FN Rate | Ambiguous Passed (FP) | FP Rate | FP Bimodal | FP Uniform |")
    print("|----|------|----------------------|---------|----------------------|---------|------------|------------|")
    for K in sorted(summary.keys()):
        for T in thresholds:
            d = summary[K][T]
            print(f"| {K:2d} | {T:.2f} | {d['fn']:2d}/{d['n_unimodal']:2d}"
                  f" | {d['fn_rate']:.3f}   | {d['fp']:3d}/{d['n_ambiguous']:3d}"
                  f" | {d['fp_rate']:.3f}   | {d['fp_bimodal']:2d}"
                  f" | {d['fp_uniform']:3d} |")
        print("|    |      |                      |         |                      |         |            |            |")

    # Summary: best threshold per K
    print("\n### Per-K Optimal Threshold (minimising FN rate while keeping FP bimodal = 0)")
    print("| K  | Best T | FN Rate | FP Bimodal |")
    print("|----|--------|---------|------------|")
    for K in sorted(summary.keys()):
        best_T = None
        for T in thresholds:
            d = summary[K][T]
            if d["fp_bimodal"] == 0:
                if best_T is None or d["fn_rate"] < summary[K][best_T]["fn_rate"]:
                    best_T = T
        if best_T is not None:
            d = summary[K][best_T]
            print(f"| {K:2d} | {best_T:.2f}   | {d['fn_rate']:.3f}   | {d['fp_bimodal']} |")


def print_proposal1_blocked_detail(summary, thresholds):
    """Show which unimodal distributions are blocked at T=0.70, T=0.80."""
    for T in [0.70, 0.80]:
        print(f"\n### Unimodal distributions blocked at T={T:.2f}:")
        for K in sorted(summary.keys()):
            blocked = summary[K][T]["blocked_unimodal"]
            if blocked:
                print(f"  K={K}: {blocked}")
            else:
                print(f"  K={K}: (none blocked)")


# ══════════════════════════════════════════════════════════════════════════
# §B  PROPOSAL 2: Pathway 2 Mechanism Comparison
# ══════════════════════════════════════════════════════════════════════════

def sim_converging_trace(n, init_w, final_w, rate, noise_sd, rng):
    """Exponentially decaying CI width with heteroscedastic MCMC noise."""
    widths = []
    for i in range(n):
        true_w = final_w + (init_w - final_w) * np.exp(-rate * i)
        noise = rng.normal(0, noise_sd * true_w)
        widths.append(max(0.001, true_w + noise))
    return widths


def sim_stable_trace(n, stable_w, noise_sd, rng):
    """Constant CI width with noise (bimodal-stable or non-converging)."""
    return [max(0.001, stable_w + rng.normal(0, noise_sd * stable_w)) for _ in range(n)]


def sim_diverging_trace(n, init_w, growth_rate, noise_sd, rng):
    """Slowly widening CI (non-converging, getting worse)."""
    widths = []
    for i in range(n):
        true_w = init_w * (1 + growth_rate * i)
        noise = rng.normal(0, noise_sd * true_w)
        widths.append(max(0.001, true_w + noise))
    return widths


# ── Stopping mechanisms ──

def stop_relative_change(widths, threshold=0.002, min_checks=3):
    """Current Pathway 2: 2-point relative change."""
    for i in range(max(1, min_checks - 1), len(widths)):
        prev, curr = widths[i - 1], widths[i]
        if prev > 0:
            rc = abs(curr - prev) / prev
            if rc < threshold:
                return i
    return None


def stop_windowed_slope(widths, window=15, slope_thresh=0.00001,
                        n_consecutive=4, check_2nd=True):
    """Binary-style windowed slope with slope-of-slopes."""
    slopes = []
    for i in range(window - 1, len(widths)):
        recent = widths[i - window + 1: i + 1]
        slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
        slopes.append(slope)
        if len(slopes) >= n_consecutive:
            tail = slopes[-n_consecutive:]
            if all(abs(s) <= slope_thresh for s in tail):
                if check_2nd and len(tail) >= 3:
                    sos = np.polyfit(np.arange(3), tail[-3:], 1)[0]
                    if sos >= 0:
                        return i
                elif not check_2nd:
                    return i
    return None


def stop_adapted_slope(widths, window=5, slope_thresh=0.001,
                       n_consecutive=3):
    """Adapted windowed slope for ordinal (smaller window)."""
    return stop_windowed_slope(widths, window=window,
                               slope_thresh=slope_thresh,
                               n_consecutive=n_consecutive, check_2nd=True)


def stop_guarded_slope(widths, entropy_frac, window=5, slope_thresh=0.001,
                       n_consecutive=3, guard=0.85):
    """Adapted slope + absolute entropy guard."""
    if entropy_frac > guard:
        return None  # entropy too high, refuse to stop
    return stop_adapted_slope(widths, window, slope_thresh, n_consecutive)


def stop_modal_ci_slope(modal_widths, window=5, slope_thresh=0.0001,
                        n_consecutive=3):
    """Track modal CI width slope instead of entropy CI width."""
    return stop_windowed_slope(modal_widths, window=window,
                               slope_thresh=slope_thresh,
                               n_consecutive=n_consecutive, check_2nd=True)


def run_proposal2(n_scenarios=500, n_checks=50, seed=42):
    """Run Pathway 2 mechanism comparison across four scenario types."""
    rng = np.random.default_rng(seed)

    scenarios = {
        "converging":      {"should_stop": True,  "traces": [], "entropy_fracs": []},
        "bimodal_stable":  {"should_stop": False, "traces": [], "entropy_fracs": []},
        "noisy_plateau":   {"should_stop": True,  "traces": [], "entropy_fracs": []},
        "diverging":       {"should_stop": False, "traces": [], "entropy_fracs": []},
    }

    methods = [
        "relchg_0.002", "relchg_0.01", "relchg_0.05",
        "winslope_15", "adapted_5", "guarded_5_0.85",
    ]

    results = {}
    for stype in scenarios:
        results[stype] = {m: {"triggered": 0, "indices": []} for m in methods}

    for _ in range(n_scenarios):
        # ── Converging (moderate entropy, should stop) ──
        w = sim_converging_trace(n_checks, rng.uniform(0.5, 2.0),
                                 rng.uniform(0.01, 0.1),
                                 rng.uniform(0.05, 0.3),
                                 rng.uniform(0.05, 0.20), rng)
        ef = rng.uniform(0.40, 0.75)
        scenarios["converging"]["traces"].append(w)
        scenarios["converging"]["entropy_fracs"].append(ef)

        for name, fn, kw in [
            ("relchg_0.002", stop_relative_change, {"threshold": 0.002}),
            ("relchg_0.01",  stop_relative_change, {"threshold": 0.01}),
            ("relchg_0.05",  stop_relative_change, {"threshold": 0.05}),
            ("winslope_15",  stop_windowed_slope,  {"window": 15, "slope_thresh": 0.00001, "n_consecutive": 4}),
            ("adapted_5",    stop_adapted_slope,   {"window": 5, "slope_thresh": 0.001, "n_consecutive": 3}),
            ("guarded_5_0.85", stop_guarded_slope, {"entropy_frac": ef, "window": 5, "slope_thresh": 0.001, "n_consecutive": 3, "guard": 0.85}),
        ]:
            idx = fn(w, **kw)
            if idx is not None:
                results["converging"][name]["triggered"] += 1
                results["converging"][name]["indices"].append(idx)

        # ── Bimodal stable (high entropy, should NOT stop) ──
        w = sim_stable_trace(n_checks, rng.uniform(0.3, 1.0),
                             rng.uniform(0.10, 0.30), rng)
        ef = rng.uniform(0.82, 0.98)
        scenarios["bimodal_stable"]["traces"].append(w)
        scenarios["bimodal_stable"]["entropy_fracs"].append(ef)

        for name, fn, kw in [
            ("relchg_0.002", stop_relative_change, {"threshold": 0.002}),
            ("relchg_0.01",  stop_relative_change, {"threshold": 0.01}),
            ("relchg_0.05",  stop_relative_change, {"threshold": 0.05}),
            ("winslope_15",  stop_windowed_slope,  {"window": 15, "slope_thresh": 0.00001, "n_consecutive": 4}),
            ("adapted_5",    stop_adapted_slope,   {"window": 5, "slope_thresh": 0.001, "n_consecutive": 3}),
            ("guarded_5_0.85", stop_guarded_slope, {"entropy_frac": ef, "window": 5, "slope_thresh": 0.001, "n_consecutive": 3, "guard": 0.85}),
        ]:
            idx = fn(w, **kw)
            if idx is not None:
                results["bimodal_stable"][name]["triggered"] += 1
                results["bimodal_stable"][name]["indices"].append(idx)

        # ── Noisy plateau (slow convergence, moderate entropy, should stop) ──
        w = sim_converging_trace(n_checks, rng.uniform(0.5, 1.5),
                                 rng.uniform(0.05, 0.20),
                                 rng.uniform(0.01, 0.08),
                                 rng.uniform(0.15, 0.35), rng)
        ef = rng.uniform(0.50, 0.78)
        scenarios["noisy_plateau"]["traces"].append(w)
        scenarios["noisy_plateau"]["entropy_fracs"].append(ef)

        for name, fn, kw in [
            ("relchg_0.002", stop_relative_change, {"threshold": 0.002}),
            ("relchg_0.01",  stop_relative_change, {"threshold": 0.01}),
            ("relchg_0.05",  stop_relative_change, {"threshold": 0.05}),
            ("winslope_15",  stop_windowed_slope,  {"window": 15, "slope_thresh": 0.00001, "n_consecutive": 4}),
            ("adapted_5",    stop_adapted_slope,   {"window": 5, "slope_thresh": 0.001, "n_consecutive": 3}),
            ("guarded_5_0.85", stop_guarded_slope, {"entropy_frac": ef, "window": 5, "slope_thresh": 0.001, "n_consecutive": 3, "guard": 0.85}),
        ]:
            idx = fn(w, **kw)
            if idx is not None:
                results["noisy_plateau"][name]["triggered"] += 1
                results["noisy_plateau"][name]["indices"].append(idx)

        # ── Diverging (should NOT stop) ──
        w = sim_diverging_trace(n_checks, rng.uniform(0.1, 0.5),
                                rng.uniform(0.01, 0.05),
                                rng.uniform(0.05, 0.15), rng)
        ef = rng.uniform(0.60, 0.90)
        scenarios["diverging"]["traces"].append(w)
        scenarios["diverging"]["entropy_fracs"].append(ef)

        for name, fn, kw in [
            ("relchg_0.002", stop_relative_change, {"threshold": 0.002}),
            ("relchg_0.01",  stop_relative_change, {"threshold": 0.01}),
            ("relchg_0.05",  stop_relative_change, {"threshold": 0.05}),
            ("winslope_15",  stop_windowed_slope,  {"window": 15, "slope_thresh": 0.00001, "n_consecutive": 4}),
            ("adapted_5",    stop_adapted_slope,   {"window": 5, "slope_thresh": 0.001, "n_consecutive": 3}),
            ("guarded_5_0.85", stop_guarded_slope, {"entropy_frac": ef, "window": 5, "slope_thresh": 0.001, "n_consecutive": 3, "guard": 0.85}),
        ]:
            idx = fn(w, **kw)
            if idx is not None:
                results["diverging"][name]["triggered"] += 1
                results["diverging"][name]["indices"].append(idx)

    return results, scenarios, n_scenarios


def print_proposal2(results, n_scenarios):
    print("\n" + "=" * 90)
    print("§B  PROPOSAL 2: PATHWAY 2 MECHANISM COMPARISON")
    print("=" * 90)

    labels = {
        "converging":     "Converging (SHOULD stop)",
        "bimodal_stable": "Bimodal stable (should NOT stop)",
        "noisy_plateau":  "Noisy plateau (SHOULD stop)",
        "diverging":      "Diverging (should NOT stop)",
    }

    print()
    print("| Scenario                     | Method           | Trigger Rate | Median Chk | Correct? |")
    print("|------------------------------|------------------|-------------|------------|----------|")
    for stype, label in labels.items():
        should_stop = stype in ("converging", "noisy_plateau")
        for method in results[stype]:
            d = results[stype][method]
            rate = d["triggered"] / n_scenarios
            med = f"{np.median(d['indices']):.0f}" if d["indices"] else "N/A"
            # Correct: high trigger rate if should_stop, low if shouldn't
            if should_stop:
                correct = "YES" if rate > 0.5 else ("MARGINAL" if rate > 0.1 else "NO")
            else:
                correct = "YES" if rate < 0.05 else ("MARGINAL" if rate < 0.20 else "NO")
            print(f"| {label:28s} | {method:16s} | {rate:>10.1%} | {med:>10s} | {correct:8s} |")
        print("|                              |                  |             |            |          |")


# ══════════════════════════════════════════════════════════════════════════
# §C  PROPOSAL 3: K-Adaptive Threshold
# ══════════════════════════════════════════════════════════════════════════

def adaptive_T(K, a=0.60, b=0.06):
    return a + b * np.log2(K)


def run_proposal3(seed=42):
    """Evaluate K-adaptive vs fixed threshold."""
    K_values = [3, 5, 7, 11, 21, 51, 101]
    sigma_fracs = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35]

    # ── For each K, evaluate mid-range unimodal pass rates ──
    results = {}
    for K in K_values:
        max_ent = max_entropy_bits(K)
        mu = (K - 1) * 0.5  # center
        T_adapt = adaptive_T(K)
        T_fixed = 0.80
        T_current = 0.70

        dists = []
        for sf in sigma_fracs:
            sigma = max(sf * (K - 1), 0.01)
            probs = discretize_truncnorm(mu, sigma, K)
            ent = shannon_entropy_bits(probs)
            dists.append(dict(sigma_frac=sf, entropy_frac=ent / max_ent))

        pass_adapt = sum(1 for d in dists if d["entropy_frac"] <= T_adapt)
        pass_fixed = sum(1 for d in dists if d["entropy_frac"] <= T_fixed)
        pass_current = sum(1 for d in dists if d["entropy_frac"] <= T_current)

        # Also check bimodal block rate
        bimodal_ents = []
        for sf in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
            sigma = max(sf * (K - 1), 0.01)
            probs = make_bimodal((K-1)*0.25, (K-1)*0.75, sigma, sigma, K)
            ent = shannon_entropy_bits(probs)
            bimodal_ents.append(ent / max_ent)

        bimodal_leaked_adapt = sum(1 for e in bimodal_ents if e <= T_adapt)
        bimodal_leaked_fixed = sum(1 for e in bimodal_ents if e <= T_fixed)
        bimodal_leaked_current = sum(1 for e in bimodal_ents if e <= T_current)

        results[K] = dict(
            T_adapt=round(T_adapt, 4), T_fixed=T_fixed, T_current=T_current,
            max_entropy=round(max_ent, 3),
            n_dists=len(dists),
            pass_adapt=pass_adapt, pass_fixed=pass_fixed, pass_current=pass_current,
            n_bimodal=len(bimodal_ents),
            bimodal_leaked_adapt=bimodal_leaked_adapt,
            bimodal_leaked_fixed=bimodal_leaked_fixed,
            bimodal_leaked_current=bimodal_leaked_current,
            dists=dists, bimodal_ents=bimodal_ents,
        )

    # ── Sensitivity analysis ──
    a_vals = np.arange(0.50, 0.71, 0.05)
    b_vals = np.arange(0.02, 0.13, 0.02)
    sensitivity = {}
    for a in a_vals:
        for b in b_vals:
            key = f"a={a:.2f},b={b:.2f}"
            sensitivity[key] = {K: round(a + b * np.log2(K), 4) for K in K_values}

    return results, sensitivity


def print_proposal3(results, sensitivity):
    print("\n" + "=" * 90)
    print("§C  PROPOSAL 3: K-ADAPTIVE THRESHOLD")
    print("=" * 90)

    print("\n### Threshold Values and Mid-Range Unimodal Pass Rates")
    print()
    print("| K   | H_max | T_cur=0.70 | T_fixed=0.80 | T_adapt | Pass(cur) | Pass(fix) | Pass(adapt) | Bimodal Leak(cur) | Bimodal Leak(fix) | Bimodal Leak(adapt) |")
    print("|-----|-------|------------|-------------|---------|-----------|-----------|-------------|-------------------|-------------------|---------------------|")
    for K in sorted(results.keys()):
        d = results[K]
        print(f"| {K:3d} | {d['max_entropy']:.3f} | 0.70       | 0.80        | {d['T_adapt']:.4f}"
              f"  | {d['pass_current']:2d}/{d['n_dists']:2d}"
              f"      | {d['pass_fixed']:2d}/{d['n_dists']:2d}"
              f"      | {d['pass_adapt']:2d}/{d['n_dists']:2d}"
              f"        | {d['bimodal_leaked_current']:2d}/{d['n_bimodal']}"
              f"              | {d['bimodal_leaked_fixed']:2d}/{d['n_bimodal']}"
              f"              | {d['bimodal_leaked_adapt']:2d}/{d['n_bimodal']}"
              f"                |")

    # ── Sensitivity ──
    print("\n### Sensitivity: T(K) for different (a, b) coefficients")
    print()
    K_show = [3, 7, 11, 21, 101]
    header = "| a    | b    | " + " | ".join(f"K={K}" for K in K_show) + " |"
    print(header)
    print("|" + "|".join(["------"] * (2 + len(K_show))) + "|")
    for key, vals in sorted(sensitivity.items()):
        parts = key.split(",")
        a_str = parts[0].split("=")[1]
        b_str = parts[1].split("=")[1]
        vals_str = " | ".join(f"{vals[K]:.3f}" for K in K_show)
        print(f"| {a_str} | {b_str} | {vals_str} |")


# ══════════════════════════════════════════════════════════════════════════
# §D  REAL-DATA VALIDATION
# ══════════════════════════════════════════════════════════════════════════

def run_validation():
    """Load actual matrix run data and validate against proposals."""
    print("\n" + "=" * 90)
    print("§D  REAL-DATA VALIDATION (WritingBench, K=11)")
    print("=" * 90)

    K = 11
    max_ent = max_entropy_bits(K)

    cells = {
        "low_ordinal":  dict(entropy_scaled=0.5638, score=1.14, pathway=1, stopped=True, rel_change=None, n_checks=2),
        "mid_ordinal":  dict(entropy_scaled=0.7135, score=4.22, pathway="hybrid(never)", stopped=False, rel_change=0.305, n_checks=5),
        "high_ordinal": dict(entropy_scaled=0.6561, score=6.93, pathway=1, stopped=True, rel_change=0.192, n_checks=5),
    }

    thresholds_to_test = [0.70, 0.75, 0.80, 0.85]
    T_adapt = adaptive_T(K)

    print(f"\nmax_entropy = log2({K}) = {max_ent:.4f} bits")
    print(f"T_adaptive(K={K}) = {T_adapt:.4f}\n")

    print("| Cell          | Score | Entropy(scaled) | Entropy(bits) | T=0.70 | T=0.75 | T=0.80 | T=0.85 | T_adapt={:.3f} | Rel.Change | Stopped? |".format(T_adapt))
    print("|---------------|-------|-----------------|---------------|--------|--------|--------|--------|{}|------------|----------|".format("-" * 15 + "|"))
    for name, d in cells.items():
        ent_bits = d["entropy_scaled"] * max_ent
        statuses = []
        for T in thresholds_to_test:
            statuses.append("PASS" if d["entropy_scaled"] <= T else "BLOCK")
        adapt_status = "PASS" if d["entropy_scaled"] <= T_adapt else "BLOCK"
        rc_str = f"{d['rel_change']:.3f}" if d["rel_change"] is not None else "N/A"
        print(f"| {name:13s} | {d['score']:.2f} | {d['entropy_scaled']:.4f}"
              f"          | {ent_bits:.3f}"
              f"         | {statuses[0]:6s} | {statuses[1]:6s} | {statuses[2]:6s} | {statuses[3]:6s}"
              f" | {adapt_status:13s} | {rc_str:10s} | {'Yes' if d['stopped'] else 'No':8s} |")

    # Structural finding
    print("\n### Structural Finding: Slope Fallback Unreachable")
    print(f"  Binary-style slope fallback requires stab_window=15 consecutive group checks.")
    print(f"  Observed group checks: low_ordinal={cells['low_ordinal']['n_checks']}, "
          f"mid_ordinal={cells['mid_ordinal']['n_checks']}, "
          f"high_ordinal={cells['high_ordinal']['n_checks']}")
    print(f"  With only 2-5 group checks, the fallback (rule.py:1897-1911) is structurally unreachable.")


# ══════════════════════════════════════════════════════════════════════════
# §E  PLOTS
# ══════════════════════════════════════════════════════════════════════════

def make_plots(p1_rows, p2_results, p2_scenarios, p3_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # ── Plot 1: Entropy landscape ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    type_groups = [
        (["unimodal"], "Unimodal (Legitimate)"),
        (["bimodal", "bimodal_asym"], "Bimodal (Ambiguous)"),
        (["near_uniform"], "Near-Uniform (Ambiguous)"),
    ]
    for ax, (dtypes, title) in zip(axes, type_groups):
        subset = [r for r in p1_rows if r["dtype"] in dtypes]
        K_vals = [r["K"] + np.random.uniform(-0.3, 0.3) for r in subset]
        ent_fracs = [r["entropy_frac"] for r in subset]
        ax.scatter(K_vals, ent_fracs, alpha=0.3, s=12, c="gray")
        ax.axhline(0.70, color="red", ls="--", lw=1.5, label="T=0.70 (current)")
        ax.axhline(0.80, color="blue", ls="--", lw=1.5, label="T=0.80 (proposed)")
        ax.set_xlabel("K (categories)")
        ax.set_ylabel("Entropy / Max Entropy")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([3, 5, 7, 11, 21])
    plt.suptitle("Proposal 1: Entropy Fraction Landscape", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "proposal1_entropy_landscape.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR / 'proposal1_entropy_landscape.png'}")

    # ── Plot 2: Proposal 2 convergence examples ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    stypes = ["converging", "bimodal_stable", "noisy_plateau", "diverging"]
    titles = ["Converging (SHOULD stop)", "Bimodal Stable (should NOT stop)",
              "Noisy Plateau (SHOULD stop)", "Diverging (should NOT stop)"]
    for ax, stype, title in zip(axes.flat, stypes, titles):
        traces = p2_scenarios[stype]["traces"]
        for i in range(min(8, len(traces))):
            ax.plot(traces[i], alpha=0.15, color="gray", lw=0.8)
        mean_trace = np.mean(traces[:50], axis=0)
        ax.plot(mean_trace, color="black", lw=2, label="Mean trace")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Check Index")
        ax.set_ylabel("CI Width")
        ax.legend(fontsize=9)
    plt.suptitle("Proposal 2: CI Width Convergence Patterns", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "proposal2_convergence_traces.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'proposal2_convergence_traces.png'}")

    # ── Plot 3: Proposal 2 trigger rates ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    methods = list(p2_results["converging"].keys())
    method_labels = [m.replace("relchg_", "RC ").replace("winslope_", "WS ").replace("adapted_", "Adpt ").replace("guarded_", "Grd ") for m in methods]
    x = np.arange(len(methods))

    # Should-stop scenarios
    ax = axes[0]
    for stype, color, label in [("converging", "green", "Converging"),
                                 ("noisy_plateau", "olive", "Noisy Plateau")]:
        rates = [p2_results[stype][m]["triggered"] / 500 for m in methods]
        ax.bar(x + (0 if stype == "converging" else 0.35), rates, 0.35,
               color=color, alpha=0.7, label=label)
    ax.set_xticks(x + 0.175)
    ax.set_xticklabels(method_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Trigger Rate")
    ax.set_title("Should-Stop Scenarios (higher = better)")
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Should-NOT-stop scenarios
    ax = axes[1]
    for stype, color, label in [("bimodal_stable", "red", "Bimodal Stable"),
                                 ("diverging", "orange", "Diverging")]:
        rates = [p2_results[stype][m]["triggered"] / 500 for m in methods]
        ax.bar(x + (0 if stype == "bimodal_stable" else 0.35), rates, 0.35,
               color=color, alpha=0.7, label=label)
    ax.set_xticks(x + 0.175)
    ax.set_xticklabels(method_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Trigger Rate")
    ax.set_title("Should-NOT-Stop Scenarios (lower = better)")
    ax.legend()
    ax.set_ylim(0, 1.1)

    plt.suptitle("Proposal 2: Mechanism Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "proposal2_trigger_rates.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'proposal2_trigger_rates.png'}")

    # ── Plot 4: Proposal 3 threshold comparison ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    K_vals = sorted(p3_results.keys())
    T_adapt = [p3_results[K]["T_adapt"] for K in K_vals]
    ax1.plot(K_vals, T_adapt, "b-o", label="T(K) = 0.60 + 0.06*log2(K)", lw=2)
    ax1.axhline(0.80, color="green", ls="--", lw=1.5, label="Fixed T=0.80")
    ax1.axhline(0.70, color="red", ls=":", lw=1.5, label="Current T=0.70")
    ax1.set_xlabel("K (categories)")
    ax1.set_ylabel("Threshold T")
    ax1.set_title("Threshold Values vs Scale Size")
    ax1.legend()
    ax1.set_xscale("log")
    ax1.set_xticks(K_vals)
    ax1.set_xticklabels(K_vals)

    n = [p3_results[K]["n_dists"] for K in K_vals]
    ax2.plot(K_vals, [p3_results[K]["pass_current"] / p3_results[K]["n_dists"] for K in K_vals],
             "r-^", label="T=0.70 (current)", lw=1.5)
    ax2.plot(K_vals, [p3_results[K]["pass_fixed"] / p3_results[K]["n_dists"] for K in K_vals],
             "g-s", label="T=0.80 (fixed)", lw=1.5)
    ax2.plot(K_vals, [p3_results[K]["pass_adapt"] / p3_results[K]["n_dists"] for K in K_vals],
             "b-o", label="T(K) adaptive", lw=1.5)
    ax2.set_xlabel("K (categories)")
    ax2.set_ylabel("Mid-Range Unimodal Pass Rate")
    ax2.set_title("Fraction of Center Distributions Allowed")
    ax2.legend()
    ax2.set_xscale("log")
    ax2.set_xticks(K_vals)
    ax2.set_xticklabels(K_vals)
    ax2.set_ylim(0, 1.1)

    plt.suptitle("Proposal 3: K-Adaptive vs Fixed Threshold", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "proposal3_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / 'proposal3_comparison.png'}")


# ══════════════════════════════════════════════════════════════════════════
# §F  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 90)
    print("  ORDINAL STOPPING MODEL: THRESHOLD VALIDITY ANALYSIS")
    print("  " + "=" * 86)
    print()

    # §A
    p1_rows, p1_summary, p1_thresholds = run_proposal1()
    print_proposal1(p1_summary, p1_thresholds)
    print_proposal1_blocked_detail(p1_summary, p1_thresholds)

    # §B
    p2_results, p2_scenarios, p2_n = run_proposal2(n_scenarios=500, n_checks=50)
    print_proposal2(p2_results, p2_n)

    # §C
    p3_results, p3_sensitivity = run_proposal3()
    print_proposal3(p3_results, p3_sensitivity)

    # §D
    run_validation()

    # §E - Plots
    print("\n" + "=" * 90)
    print("GENERATING PLOTS")
    print("=" * 90)
    make_plots(p1_rows, p2_results, p2_scenarios, p3_results)

    # Save JSON summary
    json_out = {
        "proposal1_summary": {str(k): v for k, v in p1_summary.items()},
        "proposal2_summary": {
            stype: {m: {"trigger_rate": d["triggered"] / p2_n,
                        "median_check": float(np.median(d["indices"])) if d["indices"] else None}
                    for m, d in p2_results[stype].items()}
            for stype in p2_results
        },
        "proposal3_summary": {str(k): {kk: vv for kk, vv in v.items() if kk != "dists" and kk != "bimodal_ents"}
                              for k, v in p3_results.items()},
    }
    json_path = OUTPUT_DIR / "analysis_results.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")

    print("\n" + "=" * 90)
    print("  ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
