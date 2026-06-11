#!/usr/bin/env python3
"""
K-Sensitivity Analysis for Pathway 2 Convergence Threshold

Tests whether the abs_width < 0.10 threshold is robust across different values
of K (number of ordinal categories). Higher K means more Dirichlet parameters,
slower convergence, and higher MCMC noise.

Key question: Does a single threshold (0.10) work for K ∈ {3, 5, 11, 15},
or does it need K-dependent adjustment?

Critical observation: At K=3, even a moderate unimodal (50% one category) has
scaled entropy = 0.946, so P1's entropy gate at T=0.80 blocks it. P2 becomes
the PRIMARY stopping mechanism at low K.

Author: Claude (analysis for optstop package developer)
Date: 2026-02-26
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Reuse core classes from the existing simulation
from pathway2_convergence import (
    ConvergenceScenario,
    MechanismResult,
    apply_mechanism,
    evaluate_at_check,
)


# ============================================================================
# K-Dependent Parameter Configurations
# ============================================================================

# Rationale for K-dependent scaling:
# - Higher K → more parameters in Dirichlet → slower convergence (lower λ)
# - Higher K → more dimensions for MCMC → higher noise
# - Higher K → less prior info per category → wider initial CIs
# - Higher K → higher asymptotic floor (harder to achieve same precision)

K_CONFIGS = {
    3: {
        'description': 'Low K (e.g., low/medium/high)',
        'w0_values': [0.10, 0.15, 0.25, 0.35],
        'winf_values': [0.002, 0.008, 0.015, 0.030],
        'decay_rates': [0.30, 0.50, 0.80, 1.20, 1.50],
        'noise_levels': [0.03, 0.06, 0.10, 0.15, 0.20],
    },
    5: {
        'description': 'Moderate K (e.g., Likert scale)',
        'w0_values': [0.15, 0.25, 0.35, 0.45],
        'winf_values': [0.003, 0.010, 0.025, 0.045],
        'decay_rates': [0.20, 0.40, 0.70, 1.00, 1.30],
        'noise_levels': [0.04, 0.08, 0.12, 0.18, 0.25],
    },
    11: {
        'description': 'Baseline K (WritingBench, ordinal_max_score=10)',
        'w0_values': [0.20, 0.30, 0.40, 0.50],
        'winf_values': [0.005, 0.015, 0.030, 0.060],
        'decay_rates': [0.10, 0.20, 0.40, 0.70, 1.00],
        'noise_levels': [0.05, 0.10, 0.15, 0.20, 0.30],
    },
    15: {
        'description': 'High K (fine-grained ordinal)',
        'w0_values': [0.25, 0.35, 0.50, 0.60],
        'winf_values': [0.010, 0.025, 0.045, 0.080],
        'decay_rates': [0.05, 0.15, 0.30, 0.50, 0.80],
        'noise_levels': [0.08, 0.12, 0.18, 0.25, 0.35],
    },
}

# Focused mechanism set — abs_width at various thresholds plus comparison baselines
MECHANISMS = [
    'abs_width_0.05',
    'abs_width_0.08',
    'abs_width_0.10',
    'abs_width_0.12',
    'abs_width_0.15',
    'current_relchg_0.002',
    'relchg_0.30',
    'relchg_0.50',
]


# ============================================================================
# Scenario Building
# ============================================================================

def build_scenarios_for_k(K: int) -> List[ConvergenceScenario]:
    """Build scenario catalogue with K-appropriate parameters."""
    config = K_CONFIGS[K]
    scenarios = []

    rate_to_type = {}
    for rate in config['decay_rates']:
        if rate <= 0.15:
            rate_to_type[rate] = 'very_slow'
        elif rate <= 0.35:
            rate_to_type[rate] = 'slow'
        elif rate <= 0.60:
            rate_to_type[rate] = 'medium'
        elif rate <= 0.90:
            rate_to_type[rate] = 'fast'
        else:
            rate_to_type[rate] = 'very_fast'

    for w0 in config['w0_values']:
        for winf in config['winf_values']:
            if winf >= w0 * 0.5:
                continue  # Asymptote must be well below initial
            for rate in config['decay_rates']:
                for noise in config['noise_levels']:
                    dtype = rate_to_type[rate]
                    name = f"K={K}_w0={w0:.2f}_winf={winf:.3f}_r={rate:.2f}_n={noise:.2f}"
                    scenarios.append(ConvergenceScenario(
                        name=name, w_0=w0, w_inf=winf,
                        decay_rate=rate, sigma_noise=noise,
                        distribution_type=dtype,
                    ))

    return scenarios


# ============================================================================
# Entropy Gate Analysis
# ============================================================================

def analyse_entropy_gate(K: int) -> Dict:
    """
    Analyse what fraction of distribution types are blocked by P1 at T=0.80.

    At low K, even moderate unimodals have high scaled entropy,
    so P1's entropy gate blocks more distributions → P2 becomes more important.
    """
    max_entropy_bits = math.log2(K)
    gate_threshold_bits = 0.80 * max_entropy_bits

    # Test representative distributions
    distributions = {}

    # Peaked: 80% in one category
    probs = [0.80] + [0.20 / (K - 1)] * (K - 1)
    ent_bits = -sum(p * math.log2(p) for p in probs if p > 0)
    scaled = ent_bits / max_entropy_bits
    distributions['peaked_80pct'] = {
        'entropy_bits': ent_bits,
        'scaled_entropy': scaled,
        'blocked_by_gate': ent_bits > gate_threshold_bits,
    }

    # Moderate: 50% in one category
    probs = [0.50] + [0.50 / (K - 1)] * (K - 1)
    ent_bits = -sum(p * math.log2(p) for p in probs if p > 0)
    scaled = ent_bits / max_entropy_bits
    distributions['moderate_50pct'] = {
        'entropy_bits': ent_bits,
        'scaled_entropy': scaled,
        'blocked_by_gate': ent_bits > gate_threshold_bits,
    }

    # Mild: 30% in one category
    probs = [0.30] + [0.70 / (K - 1)] * (K - 1)
    ent_bits = -sum(p * math.log2(p) for p in probs if p > 0)
    scaled = ent_bits / max_entropy_bits
    distributions['mild_30pct'] = {
        'entropy_bits': ent_bits,
        'scaled_entropy': scaled,
        'blocked_by_gate': ent_bits > gate_threshold_bits,
    }

    # Bimodal: 40%/40% in two categories
    if K >= 2:
        probs = [0.40, 0.40] + [0.20 / (K - 2)] * (K - 2) if K > 2 else [0.40, 0.40]
        # Avoid zero-probability categories
        probs = [max(p, 1e-10) for p in probs]
        total = sum(probs)
        probs = [p / total for p in probs]
        ent_bits = -sum(p * math.log2(p) for p in probs if p > 0)
        scaled = ent_bits / max_entropy_bits
        distributions['bimodal_40_40'] = {
            'entropy_bits': ent_bits,
            'scaled_entropy': scaled,
            'blocked_by_gate': ent_bits > gate_threshold_bits,
        }

    # Uniform
    probs = [1.0 / K] * K
    ent_bits = -sum(p * math.log2(p) for p in probs)
    scaled = ent_bits / max_entropy_bits
    distributions['uniform'] = {
        'entropy_bits': ent_bits,
        'scaled_entropy': scaled,
        'blocked_by_gate': ent_bits > gate_threshold_bits,
    }

    n_blocked = sum(1 for d in distributions.values() if d['blocked_by_gate'])

    return {
        'K': K,
        'max_entropy_bits': max_entropy_bits,
        'gate_threshold_bits': gate_threshold_bits,
        'gate_threshold_fraction': 0.80,
        'distributions': distributions,
        'n_blocked': n_blocked,
        'n_total': len(distributions),
        'pct_blocked': n_blocked / len(distributions) * 100,
    }


# ============================================================================
# Per-K Evaluation
# ============================================================================

def evaluate_for_k(K: int, n_mc: int, rng: np.random.Generator) -> Dict:
    """Run full evaluation for a single K value."""
    scenarios = build_scenarios_for_k(K)
    config = K_CONFIGS[K]

    print(f"\n{'=' * 80}")
    print(f"K = {K} ({config['description']})")
    print(f"{'=' * 80}")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  w_0: {config['w0_values']}")
    print(f"  w_inf: {config['winf_values']}")
    print(f"  Decay rates: {config['decay_rates']}")
    print(f"  Noise levels: {config['noise_levels']}")
    print(f"  MC instances: {n_mc}")
    print(f"  Total instances: {len(scenarios) * n_mc:,}")

    # --- Entropy gate analysis ---
    gate_analysis = analyse_entropy_gate(K)
    print(f"\n  Entropy gate analysis (T=0.80):")
    print(f"  {'Distribution':<20} {'Scaled Ent':>10} {'Blocked?':>10}")
    print(f"  {'-' * 42}")
    for name, info in gate_analysis['distributions'].items():
        blocked = "BLOCKED" if info['blocked_by_gate'] else "passes"
        print(f"  {name:<20} {info['scaled_entropy']:>10.3f} {blocked:>10}")
    print(f"  → {gate_analysis['n_blocked']}/{gate_analysis['n_total']} distribution types blocked by P1 gate")

    # --- Convergence landscape ---
    n_checks_list = [3, 5, 8, 12]
    landscape = {}
    for nc in n_checks_list:
        n_conv = sum(1 for s in scenarios if s.is_converged(nc, 0.10))
        landscape[nc] = {
            'converged': n_conv,
            'total': len(scenarios),
            'pct': n_conv / len(scenarios) * 100,
        }
    print(f"\n  Convergence landscape (ground truth threshold=0.10):")
    for nc, info in landscape.items():
        print(f"    Check {nc}: {info['converged']}/{info['total']} converged ({info['pct']:.1f}%)")

    # --- Main evaluation at n_checks=5 ---
    print(f"\n  Main evaluation at n_checks=5:")
    n_converged = sum(1 for s in scenarios if s.is_converged(5, 0.10))
    n_not = len(scenarios) - n_converged
    print(f"    Converged: {n_converged}/{len(scenarios)} ({n_converged/len(scenarios)*100:.1f}%)")
    print(f"    Not converged: {n_not}/{len(scenarios)} ({n_not/len(scenarios)*100:.1f}%)")

    results_n5 = {}
    for m_idx, mechanism in enumerate(MECHANISMS):
        print(f"\r    [{m_idx+1}/{len(MECHANISMS)}] {mechanism:<40}", end='', flush=True)

        agg = MechanismResult()
        for scenario in scenarios:
            res = evaluate_at_check(mechanism, scenario, n_checks=5,
                                    n_mc=n_mc, rng=rng, width_threshold=0.10)
            agg.tp += res.tp
            agg.fp += res.fp
            agg.tn += res.tn
            agg.fn += res.fn

        results_n5[mechanism] = {
            'tpr': agg.tpr, 'fpr': agg.fpr, 'f1': agg.f1,
            'accuracy': agg.accuracy,
            'tp': agg.tp, 'fp': agg.fp, 'tn': agg.tn, 'fn': agg.fn,
        }

    print()

    # Print sorted results
    sorted_5 = sorted(results_n5.items(), key=lambda x: x[1]['f1'], reverse=True)
    print(f"\n  {'Mechanism':<40} {'TPR':>7} {'FPR':>7} {'F1':>7} {'Acc':>7}")
    print(f"  {'-' * 65}")
    for name, r in sorted_5:
        print(f"  {name:<40} {r['tpr']:>7.3f} {r['fpr']:>7.3f} {r['f1']:>7.3f} {r['accuracy']:>7.3f}")

    # --- Secondary evaluation at n_checks=8 ---
    print(f"\n  Secondary evaluation at n_checks=8:")
    results_n8 = {}
    for m_idx, mechanism in enumerate(MECHANISMS):
        print(f"\r    [{m_idx+1}/{len(MECHANISMS)}] {mechanism:<40}", end='', flush=True)

        agg = MechanismResult()
        for scenario in scenarios:
            res = evaluate_at_check(mechanism, scenario, n_checks=8,
                                    n_mc=n_mc, rng=rng, width_threshold=0.10)
            agg.tp += res.tp
            agg.fp += res.fp
            agg.tn += res.tn
            agg.fn += res.fn

        results_n8[mechanism] = {
            'tpr': agg.tpr, 'fpr': agg.fpr, 'f1': agg.f1,
            'accuracy': agg.accuracy,
        }
    print()

    # --- Threshold sensitivity ---
    print(f"\n  Threshold sensitivity (n_checks=5):")
    gt_thresholds = [0.05, 0.08, 0.10, 0.12, 0.15]
    mech_thresholds = [0.05, 0.08, 0.10, 0.12, 0.15]
    thresh_results = {}

    for gt in gt_thresholds:
        thresh_results[str(gt)] = {}
        n_conv = sum(1 for s in scenarios if s.is_converged(5, gt))
        thresh_results[str(gt)]['n_converged'] = n_conv
        thresh_results[str(gt)]['n_total'] = len(scenarios)

        for mt in mech_thresholds:
            mechanism = f'abs_width_{mt}'
            agg = MechanismResult()
            for scenario in scenarios:
                res = evaluate_at_check(mechanism, scenario, n_checks=5,
                                        n_mc=n_mc, rng=rng, width_threshold=gt)
                agg.tp += res.tp
                agg.fp += res.fp
                agg.tn += res.tn
                agg.fn += res.fn

            thresh_results[str(gt)][f'abs_width_{mt}'] = {
                'tpr': agg.tpr, 'fpr': agg.fpr, 'f1': agg.f1,
            }

    # Print sensitivity matrix
    print(f"\n  F1 matrix (rows=GT threshold, cols=mechanism threshold):")
    header = f"  {'GT\\Mech':<8}" + " ".join(f"{t:>7}" for t in mech_thresholds)
    print(header)
    for gt in gt_thresholds:
        row = f"  {gt:<8.2f}"
        for mt in mech_thresholds:
            f1 = thresh_results[str(gt)].get(f'abs_width_{mt}', {}).get('f1', 0)
            row += f" {f1:>7.3f}"
        print(row)

    # --- By noise level ---
    print(f"\n  abs_width_0.10 by noise level (n_checks=5):")
    noise_breakdown = {}
    for noise in config['noise_levels']:
        noise_scen = [s for s in scenarios if s.sigma_noise == noise]
        n_conv = sum(1 for s in noise_scen if s.is_converged(5, 0.10))

        agg = MechanismResult()
        for scenario in noise_scen:
            res = evaluate_at_check('abs_width_0.10', scenario, n_checks=5,
                                    n_mc=n_mc, rng=rng, width_threshold=0.10)
            agg.tp += res.tp
            agg.fp += res.fp
            agg.tn += res.tn
            agg.fn += res.fn

        noise_breakdown[str(noise)] = {
            'n_converged': n_conv, 'n_total': len(noise_scen),
            'tpr': agg.tpr, 'fpr': agg.fpr, 'f1': agg.f1,
        }
        print(f"    noise={noise:.2f}: TPR={agg.tpr:.3f} FPR={agg.fpr:.3f} F1={agg.f1:.3f} "
              f"({n_conv}/{len(noise_scen)} converged)")

    # --- By decay rate ---
    print(f"\n  abs_width_0.10 by decay rate (n_checks=5):")
    rate_breakdown = {}
    for rate in config['decay_rates']:
        rate_scen = [s for s in scenarios if s.decay_rate == rate]
        n_conv = sum(1 for s in rate_scen if s.is_converged(5, 0.10))

        agg = MechanismResult()
        for scenario in rate_scen:
            res = evaluate_at_check('abs_width_0.10', scenario, n_checks=5,
                                    n_mc=n_mc, rng=rng, width_threshold=0.10)
            agg.tp += res.tp
            agg.fp += res.fp
            agg.tn += res.tn
            agg.fn += res.fn

        rate_breakdown[str(rate)] = {
            'n_converged': n_conv, 'n_total': len(rate_scen),
            'tpr': agg.tpr, 'fpr': agg.fpr, 'f1': agg.f1,
        }
        print(f"    rate={rate:.2f}: TPR={agg.tpr:.3f} FPR={agg.fpr:.3f} F1={agg.f1:.3f} "
              f"({n_conv}/{len(rate_scen)} converged)")

    return {
        'K': K,
        'description': config['description'],
        'n_scenarios': len(scenarios),
        'n_instances': len(scenarios) * n_mc,
        'gate_analysis': gate_analysis,
        'landscape': landscape,
        'results_n5': results_n5,
        'results_n8': results_n8,
        'threshold_sensitivity': thresh_results,
        'noise_breakdown': noise_breakdown,
        'rate_breakdown': rate_breakdown,
    }


# ============================================================================
# Cross-K Summary
# ============================================================================

def cross_k_summary(all_results: Dict) -> Dict:
    """Produce cross-K comparison summary."""
    print(f"\n{'=' * 80}")
    print(f"CROSS-K SUMMARY")
    print(f"{'=' * 80}")

    # Table 1: F1 of abs_width_0.10 at each K
    print(f"\n  abs_width_0.10 performance across K (n_checks=5, GT threshold=0.10):")
    print(f"  {'K':>4} {'Scenarios':>10} {'Converged%':>11} {'TPR':>7} {'FPR':>7} {'F1':>7}")
    print(f"  {'-' * 50}")

    summary = {}
    for K in sorted(all_results.keys()):
        r = all_results[K]
        n5 = r['results_n5']
        aw = n5.get('abs_width_0.10', {})
        landscape_5 = r['landscape'].get(5, {})
        conv_pct = landscape_5.get('pct', 0)

        summary[K] = {
            'n_scenarios': r['n_scenarios'],
            'converged_pct_at_5': conv_pct,
            'abs_width_0.10': aw,
        }

        print(f"  {K:>4} {r['n_scenarios']:>10} {conv_pct:>10.1f}% "
              f"{aw.get('tpr', 0):>7.3f} {aw.get('fpr', 0):>7.3f} {aw.get('f1', 0):>7.3f}")

    # Table 2: Optimal threshold per K
    print(f"\n  Optimal abs_width threshold per K (n_checks=5, GT threshold=0.10):")
    print(f"  {'K':>4} {'Best Mechanism':>25} {'F1':>7} {'TPR':>7} {'FPR':>7}")
    print(f"  {'-' * 55}")

    abs_mechs = [m for m in MECHANISMS if m.startswith('abs_width_')]

    for K in sorted(all_results.keys()):
        n5 = all_results[K]['results_n5']
        best_name = None
        best_f1 = -1
        for m in abs_mechs:
            f1 = n5.get(m, {}).get('f1', 0)
            if f1 > best_f1:
                best_f1 = f1
                best_name = m

        best_r = n5.get(best_name, {})
        summary[K]['best_mechanism'] = best_name
        summary[K]['best_f1'] = best_f1

        print(f"  {K:>4} {best_name:>25} {best_f1:>7.3f} "
              f"{best_r.get('tpr', 0):>7.3f} {best_r.get('fpr', 0):>7.3f}")

    # Table 3: Entropy gate impact
    print(f"\n  Entropy gate analysis (T=0.80) across K:")
    print(f"  {'K':>4} {'max_ent':>8} {'gate_bits':>10} {'Blocked':>10} {'Key blocked types'}")
    print(f"  {'-' * 70}")

    for K in sorted(all_results.keys()):
        gate = all_results[K]['gate_analysis']
        blocked_types = [name for name, info in gate['distributions'].items()
                         if info['blocked_by_gate']]
        print(f"  {K:>4} {gate['max_entropy_bits']:>8.3f} {gate['gate_threshold_bits']:>10.3f} "
              f"{gate['n_blocked']}/{gate['n_total']:>3}       {', '.join(blocked_types)}")

    # Table 4: Is 0.10 universally optimal?
    print(f"\n  Is 0.10 universally optimal?")
    all_optimal = all(summary[K].get('best_mechanism') == 'abs_width_0.10'
                      for K in summary)
    if all_optimal:
        print(f"  YES — abs_width_0.10 is the best threshold at every K tested.")
    else:
        print(f"  NO — optimal threshold varies by K:")
        for K in sorted(summary.keys()):
            best = summary[K].get('best_mechanism', '?')
            f1_010 = summary[K].get('abs_width_0.10', {}).get('f1', 0)
            f1_best = summary[K].get('best_f1', 0)
            if best != 'abs_width_0.10':
                print(f"    K={K}: best={best} (F1={f1_best:.3f}) vs abs_width_0.10 (F1={f1_010:.3f}), "
                      f"delta={f1_best - f1_010:.3f}")

    # Check if 0.10 is "close enough" even if not optimal everywhere
    f1_values = [summary[K].get('abs_width_0.10', {}).get('f1', 0) for K in summary]
    min_f1 = min(f1_values)
    max_f1 = max(f1_values)
    print(f"\n  abs_width_0.10 F1 range across K: [{min_f1:.3f}, {max_f1:.3f}] (spread: {max_f1 - min_f1:.3f})")
    if min_f1 >= 0.90:
        print(f"  → F1 ≥ 0.90 at all K values — threshold is robust.")
    elif min_f1 >= 0.80:
        print(f"  → F1 ≥ 0.80 at all K values — threshold is acceptable.")
    else:
        print(f"  → F1 < 0.80 at some K values — threshold may need K-dependent adjustment.")

    return summary


# ============================================================================
# Main
# ============================================================================

def run_full_k_sensitivity(n_mc: int = 500, seed: int = 42) -> Dict:
    """Run the complete K-sensitivity evaluation."""
    rng = np.random.default_rng(seed)

    print("=" * 80)
    print("K-SENSITIVITY ANALYSIS FOR PATHWAY 2 CONVERGENCE THRESHOLD")
    print(f"Testing K ∈ {sorted(K_CONFIGS.keys())}")
    print(f"MC instances per scenario: {n_mc}")
    print("=" * 80)

    all_results = {}
    for K in sorted(K_CONFIGS.keys()):
        all_results[K] = evaluate_for_k(K, n_mc, rng)

    summary = cross_k_summary(all_results)

    output = {
        'parameters': {
            'n_mc': n_mc,
            'seed': seed,
            'K_values': sorted(K_CONFIGS.keys()),
            'mechanisms': MECHANISMS,
            'convergence_model': 'exponential_decay',
            'ground_truth': 'absolute_ci_width < 0.10',
        },
        'per_k_results': {str(K): v for K, v in all_results.items()},
        'cross_k_summary': {str(K): v for K, v in summary.items()},
    }

    return output


if __name__ == '__main__':
    output = run_full_k_sensitivity(n_mc=500, seed=42)

    os.makedirs('simulations/output', exist_ok=True)
    results_path = 'simulations/output/pathway2_k_sensitivity_results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults saved to {results_path}")
