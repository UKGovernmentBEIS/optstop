#!/usr/bin/env python3
"""
validate_conservatism_multiseed.py

Multi-seed replication of the conservatism sensitivity analysis.
Tests whether the non-monotonic relationship between c and trials_used
(observed in the single-seed run) is a systematic property or MCMC noise.

Two complementary tests:
  A) DATA SEEDS: Different synthetic datasets (seeds 42-51), each run once
     per c value. Within each seed, all c values share the same data, so
     the within-seed ordering isolates mechanism + MCMC effects from data
     variability. Paired analysis (e.g., trials(c=50) - trials(c=10) per
     seed) is the primary test.

  B) MCMC REPLICATIONS: Same data (seed=42) run 3 times per c value.
     Since PyMC's internal random state is not controlled by our seed,
     re-runs produce different MCMC traces. This isolates pure MCMC noise
     from data/mechanism effects.

Only runs below-threshold conditions (perf < low_performance_threshold)
since above-threshold conditions showed conservatism is cleanly inactive.

Design:
  - 200 items x 10 epochs = 2000 trials per condition per seed
  - Performance levels: 1%, 5% (below threshold only)
  - Conservatism values: 1, 5, 10, 50
  - Part A: 5 data seeds x 8 conditions = 40 runs
  - Part B: 5 MCMC replications x 8 conditions = 40 runs
  - Total: 80 runs

Outputs:
  results/matrix/conservatism_multiseed.json
  results/matrix/figures/conservatism/multiseed_*.png
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_matrix_evals import UNIVERSAL_OPTSTOP_PARAMS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results/matrix")
OUTPUT_DIR = RESULTS_DIR / "figures" / "conservatism"

DEFAULT_PERF_LEVELS = [0.01, 0.05]
DEFAULT_C_VALUES = [1, 5, 10, 50]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_MCMC_REPS = 5  # replications of seed=42 to isolate MCMC noise

N_ITEMS = 200
N_EPOCHS = 10
N_TOTAL = N_ITEMS * N_EPOCHS

MCMC_DRAWS = 1000
MCMC_TUNE = 1000

DELTA_CAP = UNIVERSAL_OPTSTOP_PARAMS["delta_cap"]
CRED_LEVEL = UNIVERSAL_OPTSTOP_PARAMS["cred_level"]
LOW_PERF_THRESHOLD = UNIVERSAL_OPTSTOP_PARAMS["low_performance_threshold"]


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def make_synthetic_df(perf_level, n_items=N_ITEMS, n_epochs=N_EPOCHS, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    for epoch in range(1, n_epochs + 1):
        for item_idx in range(n_items):
            score = int(rng.random() < perf_level)
            rows.append({
                "sample_id": f"item_{item_idx:04d}",
                "epoch": epoch,
                "score": score,
                "model": "synthetic",
                "task": "",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run one condition
# ---------------------------------------------------------------------------

def run_condition(perf_level, conservatism_value, data_seed=42, shuffle_seed=None):
    import optstop

    if shuffle_seed is None:
        shuffle_seed = data_seed

    df = make_synthetic_df(perf_level, seed=data_seed)
    n_total = len(df)
    actual_perf = float(df["score"].mean())

    params = UNIVERSAL_OPTSTOP_PARAMS.copy()
    params["conservatism"] = conservatism_value
    params["low_performance_threshold"] = LOW_PERF_THRESHOLD
    params["draws"] = MCMC_DRAWS
    params["tune"] = MCMC_TUNE

    optstop_kwargs = {
        "params": params,
        "grouping_columns": ["model", "task"],
        "sample_id_column": "sample_id",
        "epoch_column": "epoch",
        "score_column": "score",
        "display_progress": False,
        "generate_diagnostics": False,
        "ordinal_max_score": 10,
        "processing_order": "epoch_interleaved",
        "reanalysis_interval": 10,
    }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, summaries = optstop.optimal_stopping_posthoc(
                df=df.copy(),
                shuffle_items=True,
                shuffle_seed=shuffle_seed,
                **optstop_kwargs,
            )
    except Exception as e:
        return {
            "perf_level": perf_level,
            "conservatism": conservatism_value,
            "data_seed": data_seed,
            "actual_perf": actual_perf,
            "error": str(e),
        }

    if not summaries or summaries[0].get("error"):
        error_msg = summaries[0]["error"] if summaries else "No summaries"
        return {
            "perf_level": perf_level,
            "conservatism": conservatism_value,
            "data_seed": data_seed,
            "actual_perf": actual_perf,
            "error": error_msg,
        }

    s = summaries[0]
    theta_lo = s.get("theta_ci_low")
    theta_hi = s.get("theta_ci_high")
    theta_width = s.get("theta_ci_width")
    theta_mid = (theta_lo + theta_hi) / 2 if theta_lo is not None and theta_hi is not None else None
    n_items_used = s.get("n_items_used")
    avg_reps = s.get("avg_reps_per_item")
    trials_used = int(n_items_used * avg_reps) if n_items_used and avg_reps else n_total
    stopped = trials_used < n_total

    ci_covers_true = False
    if theta_lo is not None and theta_hi is not None:
        ci_covers_true = float(theta_lo) <= actual_perf <= float(theta_hi)

    return {
        "perf_level": perf_level,
        "conservatism": conservatism_value,
        "data_seed": data_seed,
        "actual_perf": round(actual_perf, 6),
        "stopped": stopped,
        "trials_used": trials_used,
        "n_total": n_total,
        "efficiency_pct": round((1 - trials_used / n_total) * 100, 2),
        "theta_mid": round(float(theta_mid), 6) if theta_mid is not None else None,
        "theta_ci_low": round(float(theta_lo), 6) if theta_lo is not None else None,
        "theta_ci_high": round(float(theta_hi), 6) if theta_hi is not None else None,
        "theta_ci_width": round(float(theta_width), 6) if theta_width is not None else None,
        "ci_covers_actual": ci_covers_true,
    }


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

def _compute_aggregate(seed_results):
    """Compute aggregate statistics from a list of per-seed results."""
    valid = [r for r in seed_results if not r.get("error")]
    if not valid:
        return {}
    trials_list = [r["trials_used"] for r in valid]
    eff_list = [r["efficiency_pct"] for r in valid]
    stop_list = [r["stopped"] for r in valid]
    cover_list = [r["ci_covers_actual"] for r in valid]
    actual_perfs = [r["actual_perf"] for r in valid]

    return {
        "n_runs": len(valid),
        "stop_rate": round(sum(stop_list) / len(stop_list), 3),
        "trials_mean": round(float(np.mean(trials_list)), 1),
        "trials_std": round(float(np.std(trials_list, ddof=1)) if len(trials_list) > 1 else 0, 1),
        "trials_min": min(trials_list),
        "trials_max": max(trials_list),
        "trials_list": trials_list,
        "efficiency_mean": round(float(np.mean(eff_list)), 1),
        "efficiency_std": round(float(np.std(eff_list, ddof=1)) if len(eff_list) > 1 else 0, 1),
        "coverage_rate": round(sum(cover_list) / len(cover_list), 3),
        "actual_perf_mean": round(float(np.mean(actual_perfs)), 6),
        "actual_perf_std": round(float(np.std(actual_perfs, ddof=1)) if len(actual_perfs) > 1 else 0, 6),
    }


def _save_checkpoint(data, path):
    """Save incremental checkpoint so long runs survive crashes."""
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.rename(path)


def run_part_a(perf_levels, c_values, seeds):
    """Part A: Different data seeds, one run each. Within-seed pairing."""
    print("\n" + "=" * 70)
    print("PART A: Data seed variation (within-seed pairing)")
    print("=" * 70)

    checkpoint_path = RESULTS_DIR / "conservatism_multiseed_partA_checkpoint.json"

    # Resume from checkpoint if it exists
    results = {}
    completed_keys = set()
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                results = json.load(f)
            # Build set of completed (perf, c, seed) tuples
            for perf_key, c_data in results.items():
                if perf_key.startswith("_"):
                    continue
                for c_key, run_data in c_data.items():
                    for seed_key, r in run_data.get("runs", {}).items():
                        if not r.get("error"):
                            completed_keys.add((perf_key, c_key, seed_key))
            print(f"  Resumed from checkpoint: {len(completed_keys)} runs already complete")
        except Exception:
            results = {}

    total = len(perf_levels) * len(c_values) * len(seeds)
    idx = 0

    # Run seed-first so all c values for a given seed are sequential
    # This makes the within-seed comparison explicit in the output
    for perf in perf_levels:
        perf_key = f"{perf:.4f}"
        if perf_key not in results:
            results[perf_key] = {}
        for c_val in c_values:
            if f"c={c_val}" not in results[perf_key]:
                results[perf_key][f"c={c_val}"] = {"runs": {}}

        for seed in seeds:
            print(f"\n  --- perf={perf*100:.0f}%, seed={seed} ---")
            seed_trials = {}

            for c_val in c_values:
                idx += 1
                c_key = f"c={c_val}"

                # Skip if already completed (checkpoint resume)
                if (perf_key, c_key, str(seed)) in completed_keys:
                    r = results[perf_key][c_key]["runs"][str(seed)]
                    if not r.get("error"):
                        seed_trials[c_val] = r["trials_used"]
                    print(f"  [{idx}/{total}] c={c_val}... CACHED trials={r.get('trials_used', '?')}")
                    continue

                print(f"  [{idx}/{total}] c={c_val}...", end=" ", flush=True)

                t0 = time.time()
                result = run_condition(perf, c_val, data_seed=seed, shuffle_seed=seed)
                elapsed = time.time() - t0
                result["duration_sec"] = round(elapsed, 1)
                results[perf_key][c_key]["runs"][str(seed)] = result

                if result.get("error"):
                    print(f"ERROR [{elapsed:.1f}s]")
                else:
                    stopped = result.get("stopped", False)
                    trials = result["trials_used"]
                    seed_trials[c_val] = trials
                    print(f"{'STOP' if stopped else 'NOSTOP'} "
                          f"trials={trials} actual_perf={result['actual_perf']:.4f} "
                          f"[{elapsed:.1f}s]")

                # Save checkpoint after each run
                _save_checkpoint(results, checkpoint_path)

            # Within-seed monotonicity check
            if len(seed_trials) == len(c_values):
                ordered = [seed_trials[c] for c in sorted(seed_trials.keys())]
                mono = all(ordered[i] <= ordered[i+1] for i in range(len(ordered)-1))
                print(f"      Within-seed: {' → '.join(f'c={c}:{seed_trials[c]}' for c in sorted(seed_trials))} "
                      f"{'MONOTONIC' if mono else 'NON-MONOTONIC'}")

        # Compute aggregates per (perf, c)
        for c_val in c_values:
            c_key = f"c={c_val}"
            run_list = list(results[perf_key][c_key]["runs"].values())
            results[perf_key][c_key]["aggregate"] = _compute_aggregate(run_list)

    # Within-seed paired analysis
    paired_analysis = {}
    for perf in perf_levels:
        perf_key = f"{perf:.4f}"
        paired_analysis[perf_key] = {}

        for i in range(len(c_values)):
            for j in range(i + 1, len(c_values)):
                c_lo, c_hi = c_values[i], c_values[j]
                pair_key = f"c={c_lo}_vs_c={c_hi}"
                diffs = []
                for seed in seeds:
                    r_lo = results[perf_key][f"c={c_lo}"]["runs"].get(str(seed), {})
                    r_hi = results[perf_key][f"c={c_hi}"]["runs"].get(str(seed), {})
                    if not r_lo.get("error") and not r_hi.get("error"):
                        diffs.append(r_hi["trials_used"] - r_lo["trials_used"])

                if diffs:
                    n_positive = sum(1 for d in diffs if d > 0)  # c_hi used more trials
                    n_negative = sum(1 for d in diffs if d < 0)  # c_hi used fewer (non-monotonic)
                    n_zero = sum(1 for d in diffs if d == 0)
                    paired_analysis[perf_key][pair_key] = {
                        "diffs": diffs,
                        "diff_mean": round(float(np.mean(diffs)), 1),
                        "diff_std": round(float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0, 1),
                        "n_positive": n_positive,
                        "n_negative": n_negative,
                        "n_zero": n_zero,
                        "n_total": len(diffs),
                        "sign_consistency": round(max(n_positive, n_negative) / len(diffs), 3),
                        "direction": "monotonic" if n_positive >= n_negative else "non-monotonic",
                    }

    results["_paired_analysis"] = paired_analysis

    # Per-seed monotonicity summary
    mono_counts = {}
    for perf in perf_levels:
        perf_key = f"{perf:.4f}"
        n_mono = 0
        n_total_seeds = 0
        for seed in seeds:
            trials_by_c = {}
            for c_val in c_values:
                r = results[perf_key][f"c={c_val}"]["runs"].get(str(seed), {})
                if not r.get("error"):
                    trials_by_c[c_val] = r["trials_used"]
            if len(trials_by_c) == len(c_values):
                n_total_seeds += 1
                ordered = [trials_by_c[c] for c in sorted(trials_by_c.keys())]
                if all(ordered[i] <= ordered[i+1] for i in range(len(ordered)-1)):
                    n_mono += 1
        mono_counts[perf_key] = {"monotonic_seeds": n_mono, "total_seeds": n_total_seeds}
    results["_within_seed_monotonicity"] = mono_counts

    return results


def run_part_b(perf_levels, c_values, mcmc_reps, base_seed=42):
    """Part B: Same data (seed=42), multiple MCMC runs to isolate MCMC noise."""
    print("\n" + "=" * 70)
    print(f"PART B: MCMC replication (same data seed={base_seed}, {mcmc_reps} runs)")
    print("=" * 70)

    checkpoint_path = RESULTS_DIR / "conservatism_multiseed_partB_checkpoint.json"

    results = {}
    completed_keys = set()
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                results = json.load(f)
            for perf_key, c_data in results.items():
                for c_key, run_data in c_data.items():
                    for rep_key, r in run_data.get("runs", {}).items():
                        if not r.get("error"):
                            completed_keys.add((perf_key, c_key, rep_key))
            print(f"  Resumed from checkpoint: {len(completed_keys)} runs already complete")
        except Exception:
            results = {}

    total = len(perf_levels) * len(c_values) * mcmc_reps
    idx = 0

    for perf in perf_levels:
        perf_key = f"{perf:.4f}"
        if perf_key not in results:
            results[perf_key] = {}

        for c_val in c_values:
            c_key = f"c={c_val}"
            if c_key not in results[perf_key]:
                results[perf_key][c_key] = {"runs": {}}

            for rep in range(mcmc_reps):
                idx += 1

                if (perf_key, c_key, str(rep)) in completed_keys:
                    print(f"  [{idx}/{total}] perf={perf*100:.0f}%, c={c_val}, rep={rep}... CACHED")
                    continue

                print(f"  [{idx}/{total}] perf={perf*100:.0f}%, c={c_val}, mcmc_rep={rep}...",
                      end=" ", flush=True)

                t0 = time.time()
                # Same data_seed and shuffle_seed every time — only MCMC differs
                result = run_condition(perf, c_val, data_seed=base_seed, shuffle_seed=base_seed)
                elapsed = time.time() - t0
                result["duration_sec"] = round(elapsed, 1)
                result["mcmc_rep"] = rep
                results[perf_key][c_key]["runs"][str(rep)] = result

                if result.get("error"):
                    print(f"ERROR [{elapsed:.1f}s]")
                else:
                    print(f"{'STOP' if result['stopped'] else 'NOSTOP'} "
                          f"trials={result['trials_used']} [{elapsed:.1f}s]")

                _save_checkpoint(results, checkpoint_path)

            # Aggregate
            run_list = list(results[perf_key][c_key]["runs"].values())
            results[perf_key][c_key]["aggregate"] = _compute_aggregate(run_list)

            agg = results[perf_key][c_key]["aggregate"]
            if agg:
                print(f"      >> MCMC variance: trials={agg['trials_mean']:.0f} ± {agg['trials_std']:.0f} "
                      f"[{agg['trials_min']}, {agg['trials_max']}]")

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_within_seed_paired(part_a_results, perf_levels, c_values, seeds, output_dir):
    """Within-seed paired comparison: for each seed, show trials per c value."""
    fig, axes = plt.subplots(1, len(perf_levels), figsize=(7 * len(perf_levels), 5),
                              squeeze=False)

    for p_idx, perf in enumerate(perf_levels):
        ax = axes[0, p_idx]
        perf_key = f"{perf:.4f}"

        for s_idx, seed in enumerate(seeds):
            trials = []
            for c_val in c_values:
                r = part_a_results[perf_key][f"c={c_val}"]["runs"].get(str(seed), {})
                if not r.get("error"):
                    trials.append(r["trials_used"])
                else:
                    trials.append(None)

            valid = [(c, t) for c, t in zip(c_values, trials) if t is not None]
            if valid:
                cs, ts = zip(*valid)
                ordered = all(ts[i] <= ts[i+1] for i in range(len(ts)-1))
                style = "-" if ordered else "--"
                alpha = 0.7 if ordered else 0.4
                ax.plot(cs, ts, marker="o", linestyle=style, alpha=alpha,
                        markersize=4, linewidth=1, label=f"s={seed}" if s_idx < 5 else None)

        ax.set_xlabel("Conservatism (c)")
        ax.set_ylabel("Trials used")
        ax.set_title(f"perf={perf*100:.0f}%", fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks(c_values)
        ax.set_xticklabels([str(c) for c in c_values])
        ax.set_ylim(0, N_TOTAL * 1.05)
        ax.axhline(N_TOTAL, color="red", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.legend(fontsize=6, ncol=2)

    plt.suptitle(
        "Within-Seed Dose-Response (each line = one data seed)\n"
        "Solid = monotonic, dashed = non-monotonic",
        fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = output_dir / "multiseed_within_seed.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_aggregate_dose_response(part_a_results, perf_levels, c_values, output_dir):
    """Mean +/- std trials used across seeds."""
    fig, axes = plt.subplots(1, len(perf_levels), figsize=(7 * len(perf_levels), 5),
                              squeeze=False)
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(perf_levels)))

    for p_idx, perf in enumerate(perf_levels):
        ax = axes[0, p_idx]
        perf_key = f"{perf:.4f}"

        means, stds, c_plot = [], [], []
        for c_val in c_values:
            agg = part_a_results[perf_key][f"c={c_val}"].get("aggregate", {})
            if agg:
                means.append(agg["trials_mean"])
                stds.append(agg["trials_std"])
                c_plot.append(c_val)

        if means:
            ax.errorbar(c_plot, means, yerr=stds, marker="o", capsize=5,
                        color=colours[p_idx], linewidth=2, markersize=8)

        ax.set_xlabel("Conservatism (c)")
        ax.set_ylabel("Trials used (mean ± std)")
        ax.set_title(f"perf={perf*100:.0f}%", fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks(c_values)
        ax.set_xticklabels([str(c) for c in c_values])
        ax.set_ylim(0, N_TOTAL * 1.05)
        ax.axhline(N_TOTAL, color="red", linestyle=":", linewidth=0.8, alpha=0.4)

    plt.suptitle(
        "Conservatism Dose-Response: Mean ± Std Across Data Seeds",
        fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = output_dir / "multiseed_dose_response.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_mcmc_variance(part_b_results, perf_levels, c_values, output_dir):
    """Show MCMC-only variance: same data, different MCMC traces."""
    fig, axes = plt.subplots(1, len(perf_levels), figsize=(7 * len(perf_levels), 5),
                              squeeze=False)
    colours = plt.cm.Set2(np.linspace(0, 0.8, len(c_values)))

    for p_idx, perf in enumerate(perf_levels):
        ax = axes[0, p_idx]
        perf_key = f"{perf:.4f}"

        for c_idx, c_val in enumerate(c_values):
            agg = part_b_results[perf_key][f"c={c_val}"].get("aggregate", {})
            if agg and "trials_list" in agg:
                trials = agg["trials_list"]
                x = [c_val] * len(trials)
                ax.scatter(x, trials, color=colours[c_idx], s=60, zorder=3, alpha=0.8)
                ax.errorbar([c_val], [agg["trials_mean"]],
                           yerr=[agg["trials_std"]], color=colours[c_idx],
                           marker="D", markersize=8, capsize=5, linewidth=2, zorder=4)

        ax.set_xlabel("Conservatism (c)")
        ax.set_ylabel("Trials used")
        ax.set_title(f"perf={perf*100:.0f}%", fontweight="bold")
        ax.set_xscale("log")
        ax.set_xticks(c_values)
        ax.set_xticklabels([str(c) for c in c_values])
        ax.set_ylim(0, N_TOTAL * 1.05)
        ax.axhline(N_TOTAL, color="red", linestyle=":", linewidth=0.8, alpha=0.4)

    plt.suptitle(
        "MCMC Variance: Same Data (seed=42), Different MCMC Traces\n"
        "(dots = individual runs, diamonds = mean ± std)",
        fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = output_dir / "multiseed_mcmc_variance.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_summary_table(part_a_results, part_b_results, perf_levels, c_values, output_dir):
    """Combined summary table."""
    rows = []
    for perf in perf_levels:
        perf_key = f"{perf:.4f}"
        for c_val in c_values:
            c_key = f"c={c_val}"
            agg_a = part_a_results[perf_key][c_key].get("aggregate", {})
            agg_b = part_b_results.get(perf_key, {}).get(c_key, {}).get("aggregate", {})
            if agg_a:
                rows.append([
                    f"{perf*100:.0f}%",
                    str(c_val),
                    # Part A
                    f"{agg_a['trials_mean']:.0f} ± {agg_a['trials_std']:.0f}",
                    f"{agg_a['stop_rate']*100:.0f}%",
                    f"{agg_a['actual_perf_mean']:.4f} ± {agg_a['actual_perf_std']:.4f}",
                    f"{agg_a['coverage_rate']*100:.0f}%",
                    # Part B
                    f"{agg_b['trials_mean']:.0f} ± {agg_b['trials_std']:.0f}" if agg_b else "--",
                ])

    col_labels = ["Perf", "c",
                  "Trials (data seeds)", "Stop rate",
                  "Actual perf (mean±std)", "Coverage",
                  "Trials (MCMC only)"]

    fig, ax = plt.subplots(figsize=(18, max(3, len(rows) * 0.4 + 1.5)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.4)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4C72B0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    perf_idx = 0
    current_perf = None
    for i in range(1, len(rows) + 1):
        row_perf = rows[i-1][0]
        if row_perf != current_perf:
            current_perf = row_perf
            perf_idx += 1
        colour = "#f0f0f0" if perf_idx % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(colour)

    fig.suptitle(
        "Conservatism Multi-Seed Summary\n"
        f"Part A: {len(part_a_results.get(f'{perf_levels[0]:.4f}', {}).get('c=1', {}).get('aggregate', {}).get('n_runs', '?'))} data seeds | "
        f"Part B: MCMC replications on seed=42",
        fontsize=11, fontweight="bold", y=0.99)
    out = output_dir / "multiseed_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed conservatism analysis (non-monotonicity check)")
    parser.add_argument("--perf-levels", nargs="+", type=float, default=DEFAULT_PERF_LEVELS)
    parser.add_argument("--conservatism-values", nargs="+", type=float, default=DEFAULT_C_VALUES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--mcmc-reps", type=int, default=DEFAULT_MCMC_REPS)
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-part-b", action="store_true",
                        help="Skip MCMC replication (Part B)")
    args = parser.parse_args()

    perf_levels = sorted(args.perf_levels)
    c_values = sorted(args.conservatism_values)
    seeds = args.seeds
    mcmc_reps = args.mcmc_reps

    total_a = len(perf_levels) * len(c_values) * len(seeds)
    total_b = len(perf_levels) * len(c_values) * mcmc_reps if not args.skip_part_b else 0

    print("=" * 70)
    print("Conservatism Multi-Seed Analysis")
    print(f"  Performance levels: {[f'{p*100:.0f}%' for p in perf_levels]}")
    print(f"  Conservatism values: {c_values}")
    print(f"  Part A: {len(seeds)} data seeds → {total_a} runs")
    if not args.skip_part_b:
        print(f"  Part B: {mcmc_reps} MCMC reps (seed=42) → {total_b} runs")
    print(f"  Total runs: {total_a + total_b}")
    print(f"  Data: {N_ITEMS} items x {N_EPOCHS} epochs = {N_TOTAL} trials")
    print(f"  MCMC: {MCMC_DRAWS} draws / {MCMC_TUNE} tune")
    print(f"  low_performance_threshold: {LOW_PERF_THRESHOLD}")
    print("=" * 70)

    t0 = time.time()

    # Part A: Data seed variation
    part_a = run_part_a(perf_levels, c_values, seeds)

    # Part B: MCMC replication
    part_b = {}
    if not args.skip_part_b:
        part_b = run_part_b(perf_levels, c_values, mcmc_reps, base_seed=42)

    total_time = time.time() - t0

    # Save JSON
    output_json = RESULTS_DIR / "conservatism_multiseed.json"
    json_output = {
        "config": {
            "method": "optimal_stopping_posthoc (full PyMC MCMC)",
            "purpose": "Multi-seed replication to test non-monotonicity",
            "perf_levels": perf_levels,
            "conservatism_values": c_values,
            "seeds": seeds,
            "mcmc_reps": mcmc_reps,
            "n_items": N_ITEMS,
            "n_epochs": N_EPOCHS,
            "n_total": N_TOTAL,
            "mcmc_draws": MCMC_DRAWS,
            "mcmc_tune": MCMC_TUNE,
            "low_performance_threshold": LOW_PERF_THRESHOLD,
            "delta_cap": DELTA_CAP,
            "cred_level": CRED_LEVEL,
            "total_duration_sec": round(total_time, 1),
        },
        "part_a": part_a,
        "part_b": part_b,
    }

    with open(output_json, "w") as f:
        json.dump(json_output, f, indent=2, default=str)
    print(f"\nSaved results: {output_json}")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print(f"SUMMARY ({total_time:.1f}s = {total_time/60:.1f} min)")
    print(f"{'='*70}")

    # Part A aggregates
    print("\n  PART A — Aggregates across data seeds:")
    for perf in perf_levels:
        perf_key = f"{perf:.4f}"
        print(f"\n  perf={perf*100:.0f}%:")
        for c_val in c_values:
            agg = part_a.get(perf_key, {}).get(f"c={c_val}", {}).get("aggregate", {})
            if agg:
                print(f"    c={c_val:>3}: trials={agg['trials_mean']:>6.0f} ± {agg['trials_std']:>5.0f} "
                      f"[{agg['trials_min']}, {agg['trials_max']}] "
                      f"stop={agg['stop_rate']*100:.0f}% "
                      f"actual_perf={agg['actual_perf_mean']:.4f}±{agg['actual_perf_std']:.4f} "
                      f"cover={agg['coverage_rate']*100:.0f}%")

    # Within-seed monotonicity
    mono_info = part_a.get("_within_seed_monotonicity", {})
    print(f"\n  Within-seed monotonicity:")
    for perf in perf_levels:
        perf_key = f"{perf:.4f}"
        m = mono_info.get(perf_key, {})
        if m:
            print(f"    perf={perf*100:.0f}%: {m['monotonic_seeds']}/{m['total_seeds']} seeds monotonic")

    # Paired analysis
    paired = part_a.get("_paired_analysis", {})
    print(f"\n  Paired differences (c_hi - c_lo trials, per seed):")
    for perf in perf_levels:
        perf_key = f"{perf:.4f}"
        pairs = paired.get(perf_key, {})
        # Focus on the key comparisons
        for pair_key in [f"c={c_values[-2]}_vs_c={c_values[-1]}",
                         f"c={c_values[0]}_vs_c={c_values[-1]}"]:
            p = pairs.get(pair_key, {})
            if p:
                print(f"    perf={perf*100:.0f}% {pair_key}: "
                      f"diff={p['diff_mean']:+.0f}±{p['diff_std']:.0f} "
                      f"(+:{p['n_positive']} -:{p['n_negative']} 0:{p['n_zero']}) "
                      f"→ {p['direction']}")

    # Part B summary
    if part_b:
        print(f"\n  PART B — MCMC variance (same data, different traces):")
        for perf in perf_levels:
            perf_key = f"{perf:.4f}"
            print(f"\n  perf={perf*100:.0f}%:")
            for c_val in c_values:
                agg = part_b.get(perf_key, {}).get(f"c={c_val}", {}).get("aggregate", {})
                if agg:
                    print(f"    c={c_val:>3}: trials={agg['trials_mean']:>6.0f} ± {agg['trials_std']:>5.0f} "
                          f"[{agg['trials_min']}, {agg['trials_max']}]")

    # Figures
    if not args.skip_figures:
        print(f"\nGenerating figures...")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plot_within_seed_paired(part_a, perf_levels, c_values, seeds, OUTPUT_DIR)
        plot_aggregate_dose_response(part_a, perf_levels, c_values, OUTPUT_DIR)
        if part_b:
            plot_mcmc_variance(part_b, perf_levels, c_values, OUTPUT_DIR)
        plot_summary_table(part_a, part_b if part_b else {p: {} for p in [f"{perf:.4f}" for perf in perf_levels]},
                          perf_levels, c_values, OUTPUT_DIR)

    print(f"\nDone. Total time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
