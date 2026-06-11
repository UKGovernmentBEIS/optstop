#!/usr/bin/env python3
"""
validate_fixed_n_baseline.py

Validation Gap 3: Fixed-n Baseline Comparison.

Compares adaptive optimal stopping against naive fixed-n approaches
("always run N trials") using existing shadow-mode data. No new LLM
runs are needed.

For each of the 9 matrix cells, loads the full shadow dataset and
computes score estimates at fixed trial checkpoints (100, 250, 500, 1000)
across K random item orderings. This produces a distribution of fixed-n
estimates at each checkpoint, contextualising the adaptive stopping
efficiency gains.

Key design choices:
  - Fixed-n estimates use raw sample means (normalised to [0,1]). This is
    the estimate a naive practitioner would compute without Bayesian
    inference — the natural baseline.
  - Multiple random orderings (default K=100) at each checkpoint capture
    the variance inherent in "which N trials you happen to run."
  - A checkpoint is automatically added at the adaptive stop trial count,
    enabling direct same-N comparison between fixed-n and adaptive.
  - With --bayesian, additionally runs the posthoc API (full data, default
    thresholds, shuffled items) to obtain the system's actual Bayesian
    posterior output. This is the estimate optstop would report in
    practice. Adds ~1-7 min per cell.

Outputs:
  results/matrix/fixed_n_baseline.json     - all per-checkpoint data
  results/matrix/figures/fixed_n/*.png     - visualisations

Usage:
    python validate_fixed_n_baseline.py                      # All 9 cells, raw means
    python validate_fixed_n_baseline.py --bayesian           # Include Bayesian estimate
    python validate_fixed_n_baseline.py --cells mid_binary   # Single cell
    python validate_fixed_n_baseline.py --checkpoints 50 100 200 500
    python validate_fixed_n_baseline.py --n-orderings 200    # More orderings
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import optstop
from analyse_posthoc_consistency import (
    CELL_ORDER,
    GRID,
    COL_LABELS,
    RESULTS_DIR,
    load_cell_data,
    get_full_data_reference,
)
from run_matrix_evals import CELL_CONFIGS, UNIVERSAL_OPTSTOP_PARAMS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINTS = [100, 250, 500, 1000]
DEFAULT_N_ORDERINGS = 100
OUTPUT_DIR = RESULTS_DIR / "figures" / "fixed_n"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_fixed_n_distribution(
    df: pd.DataFrame,
    n_trials: int,
    cell_config: dict,
    n_orderings: int,
    rng: np.random.RandomState,
) -> dict:
    """Compute raw mean score at n_trials across K random orderings.

    Each ordering shuffles the full dataset and takes the first n_trials
    rows (epoch-interleaved: shuffle item order within each epoch, then
    take the first n_trials in epoch-sorted order).

    Returns dict with distribution statistics.
    """
    mode = cell_config.get("scoring_mode", "binary")
    ordinal_max_score = cell_config.get("ordinal_max_score", 10)

    n_total = len(df)
    n_items = df["sample_id"].nunique()
    n_epochs = df["epoch"].nunique()

    scores_normalised = []
    items_counts = []

    for _ in range(n_orderings):
        # Shuffle item order within each epoch (epoch-interleaved)
        item_ids = df["sample_id"].unique()
        shuffled_ids = rng.permutation(item_ids)
        id_order = {sid: i for i, sid in enumerate(shuffled_ids)}

        df_copy = df.copy()
        df_copy["_item_order"] = df_copy["sample_id"].map(id_order)
        df_copy = df_copy.sort_values(["epoch", "_item_order"]).reset_index(drop=True)

        subset = df_copy.iloc[:n_trials]
        n_actual = len(subset)
        n_items_used = subset["sample_id"].nunique()
        items_counts.append(n_items_used)

        raw_mean = float(subset["score"].mean()) if n_actual > 0 else 0.0

        if mode == "binary":
            scores_normalised.append(raw_mean)
        else:
            scores_normalised.append(raw_mean / ordinal_max_score)

    scores_arr = np.array(scores_normalised)
    items_arr = np.array(items_counts)

    # Analytical SE for context (assumes i.i.d. within the finite population)
    full_mean = float(df["score"].mean())
    if mode != "binary":
        full_mean = full_mean / ordinal_max_score
    full_var = float(df["score"].var(ddof=1))
    if mode != "binary":
        full_var = full_var / (ordinal_max_score ** 2)
    analytical_se = np.sqrt(full_var / n_trials) if n_trials > 0 else 0.0

    # Average epochs per item at this trial count
    avg_epochs = n_trials / n_items if n_trials <= n_items * n_epochs else n_epochs

    return {
        "n_trials": n_trials,
        "n_orderings": n_orderings,
        "avg_items_used": round(float(np.mean(items_arr)), 1),
        "avg_epochs_per_item": round(min(avg_epochs, n_epochs), 2),
        "score_mean": round(float(np.mean(scores_arr)), 6),
        "score_std": round(float(np.std(scores_arr, ddof=1)), 6),
        "score_min": round(float(np.min(scores_arr)), 6),
        "score_max": round(float(np.max(scores_arr)), 6),
        "analytical_se": round(float(analytical_se), 6),
    }


def compute_single_order_score(
    df: pd.DataFrame,
    n_trials: int,
    cell_config: dict,
) -> float:
    """Compute normalised raw mean score for the first n_trials rows.

    Uses the DataFrame's existing order (shadow eval processing order).
    """
    mode = cell_config.get("scoring_mode", "binary")
    ordinal_max_score = cell_config.get("ordinal_max_score", 10)

    subset = df.iloc[:n_trials]
    raw_mean = float(subset["score"].mean()) if len(subset) > 0 else 0.0

    if mode == "binary":
        return raw_mean
    return raw_mean / ordinal_max_score


# ---------------------------------------------------------------------------
# Per-cell analysis
# ---------------------------------------------------------------------------

def compute_adaptive_bayesian(
    df: pd.DataFrame,
    cell_config: dict,
    mcmc_draws: int = 1000,
    shuffle_seed: int = 42,
) -> dict | None:
    """Run the posthoc API as the system would in practice.

    Passes the FULL dataset with default stopping thresholds and
    shuffle_items=True. The API processes data sequentially, builds
    the Bayesian posterior incrementally, and stops when its CI
    converges. The reported theta/CI is exactly what optstop would
    deliver to a practitioner.

    This is equivalent to one rep of the consistency analysis.
    """
    mode = cell_config.get("scoring_mode", "binary")
    ordinal_max_score = cell_config.get("ordinal_max_score") or 10

    params = UNIVERSAL_OPTSTOP_PARAMS.copy()
    params["low_performance_threshold"] = 0.01
    params["draws"] = mcmc_draws
    params["tune"] = mcmc_draws

    optstop_kwargs = {
        'params': params,
        'grouping_columns': ['model', 'task'],
        'sample_id_column': 'sample_id',
        'epoch_column': 'epoch',
        'score_column': 'score',
        'display_progress': False,
        'generate_diagnostics': False,
        'ordinal_max_score': ordinal_max_score,
        'processing_order': 'epoch_interleaved',
        'reanalysis_interval': 10,
    }

    if mode == "ordinal":
        optstop_kwargs['ordinal_tasks'] = ['']
        optstop_kwargs['ordinal_inference'] = 'hybrid'
        optstop_kwargs['ordinal_model_type'] = cell_config.get(
            'ordinal_model_type', 'ordered_logistic'
        )
    elif mode == "continuous":
        optstop_kwargs['continuous_tasks'] = ['']

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
        return {"error": str(e)}

    if not summaries or summaries[0].get("error"):
        error_msg = summaries[0]["error"] if summaries else "No summaries"
        return {"error": error_msg}

    s = summaries[0]
    theta_lo = s.get("theta_ci_low")
    theta_hi = s.get("theta_ci_high")
    theta_mid = (theta_lo + theta_hi) / 2 if theta_lo is not None and theta_hi is not None else None
    n_items_used = s.get("n_items_used")
    avg_reps = s.get("avg_reps_per_item")

    return {
        "theta_estimate": round(theta_mid, 6) if theta_mid is not None else None,
        "theta_ci_low": round(float(theta_lo), 6) if theta_lo is not None else None,
        "theta_ci_high": round(float(theta_hi), 6) if theta_hi is not None else None,
        "theta_ci_width": round(float(s.get("theta_ci_width", 0)), 6),
        "n_items_used": n_items_used,
        "avg_reps_per_item": avg_reps,
    }


def analyse_cell(
    cell_id: str,
    checkpoints: list[int],
    n_orderings: int,
    base_seed: int = 42,
    bayesian: bool = False,
    mcmc_draws: int = 1000,
) -> dict:
    """Run fixed-n analysis for one cell."""
    print(f"  Loading data...")
    df, cell_config, result = load_cell_data(cell_id)

    n_total_items = df["sample_id"].nunique()
    n_total_epochs = df["epoch"].nunique()
    n_total_trials = len(df)
    theta_full = get_full_data_reference(df, cell_config)
    mode = cell_config.get("scoring_mode", "binary")

    print(f"  {n_total_items} items x {n_total_epochs} epochs = {n_total_trials} trials")
    print(f"  Full-run reference: {theta_full:.4f}")

    # Get adaptive stopping point from shadow diagnostics
    sms = result.get("diagnostics", {}).get("shadow_mode_summary", {})
    adaptive_stop_trial = sms.get("would_have_stopped_at")
    adaptive_efficiency = sms.get("potential_efficiency_percent")

    # Optionally compute the system's actual Bayesian output
    bayes_theta = None
    bayes_ci_width = None
    bayes_ci_low = None
    bayes_ci_high = None
    bayes_n_items = None
    if bayesian:
        print(f"  Running posthoc API (Bayesian estimate)...", end=" ", flush=True)
        t0 = time.time()
        bayes_result = compute_adaptive_bayesian(
            df, cell_config, mcmc_draws, shuffle_seed=base_seed,
        )
        bayes_time = time.time() - t0
        if bayes_result and not bayes_result.get("error"):
            bayes_theta = bayes_result["theta_estimate"]
            bayes_ci_width = bayes_result["theta_ci_width"]
            bayes_ci_low = bayes_result["theta_ci_low"]
            bayes_ci_high = bayes_result["theta_ci_high"]
            bayes_n_items = bayes_result.get("n_items_used")
            avg_reps = bayes_result.get("avg_reps_per_item")
            print(f"theta={bayes_theta:.4f}, CI=[{bayes_ci_low:.3f}, {bayes_ci_high:.3f}], "
                  f"items={bayes_n_items}, avg_reps={avg_reps} [{bayes_time:.1f}s]")
        else:
            error = bayes_result.get("error", "unknown") if bayes_result else "no result"
            print(f"FAILED: {error} [{bayes_time:.1f}s]")

    # Ensure the adaptive stop trial is in the checkpoint list
    effective_checkpoints = sorted(set(checkpoints))
    if adaptive_stop_trial is not None and adaptive_stop_trial not in effective_checkpoints:
        effective_checkpoints = sorted(set(effective_checkpoints) | {adaptive_stop_trial})

    # Compute the shadow eval's specific raw mean at adaptive stop
    adaptive_raw_score = None
    if adaptive_stop_trial is not None:
        # The df from load_cell_data is in eval log order (epoch-interleaved)
        adaptive_raw_score = compute_single_order_score(df, adaptive_stop_trial, cell_config)

    print(f"  Adaptive stop: trial {adaptive_stop_trial}/{n_total_trials} "
          f"(eff={adaptive_efficiency:.1f}%)")
    if adaptive_raw_score is not None:
        print(f"    Raw mean at stop:    {adaptive_raw_score:.4f} "
              f"(|dev|={abs(adaptive_raw_score - theta_full):.4f})")
    if bayes_theta is not None:
        print(f"    Bayesian posterior:  {bayes_theta:.4f} "
              f"(|dev|={abs(bayes_theta - theta_full):.4f})")

    # Compute distribution at each checkpoint
    rng = np.random.RandomState(base_seed)
    checkpoint_results = []

    for n in effective_checkpoints:
        if n > n_total_trials:
            print(f"  n={n:>5}: SKIP (only {n_total_trials} trials)")
            continue

        is_adaptive_n = (n == adaptive_stop_trial)
        label = f"n={n}" + (" [adaptive]" if is_adaptive_n else "")
        print(f"  {label:>20}...", end=" ", flush=True)

        dist = compute_fixed_n_distribution(df, n, cell_config, n_orderings, rng)
        dist["checkpoint"] = n
        dist["is_adaptive_stop_n"] = is_adaptive_n
        dist["efficiency_pct"] = round((1 - n / n_total_trials) * 100, 2)

        # Deviation statistics
        dist["deviation_mean"] = round(dist["score_mean"] - theta_full, 6)
        dist["abs_deviation_mean"] = round(abs(dist["score_mean"] - theta_full), 6)
        # Expected |deviation| across orderings
        abs_devs = []  # recompute from distribution
        # We stored score_mean/std — can approximate expected |dev|
        # More precisely: E[|X - mu|] where X ~ N(score_mean, score_std)
        # But since we have the distribution, just report the mean and std
        dist["abs_deviation_of_mean"] = round(abs(dist["score_mean"] - theta_full), 6)
        dist["ordering_std"] = dist["score_std"]

        print(f"mean={dist['score_mean']:.4f} +/- {dist['score_std']:.4f}, "
              f"|dev|={dist['abs_deviation_of_mean']:.4f}, "
              f"SE={dist['analytical_se']:.4f}")

        checkpoint_results.append(dist)

    # Full-run checkpoint (zero deviation by definition)
    full_dist = {
        "checkpoint": n_total_trials,
        "n_trials": n_total_trials,
        "n_orderings": 1,
        "is_adaptive_stop_n": False,
        "efficiency_pct": 0.0,
        "score_mean": theta_full,
        "score_std": 0.0,
        "analytical_se": 0.0,
        "deviation_mean": 0.0,
        "abs_deviation_of_mean": 0.0,
        "ordering_std": 0.0,
        "avg_items_used": n_total_items,
        "avg_epochs_per_item": n_total_epochs,
        "score_min": theta_full,
        "score_max": theta_full,
    }
    checkpoint_results.append(full_dist)

    return {
        "cell_id": cell_id,
        "mode": mode,
        "n_total_items": n_total_items,
        "n_total_epochs": n_total_epochs,
        "n_total_trials": n_total_trials,
        "theta_full_data": theta_full,
        "adaptive_stop_trial": adaptive_stop_trial,
        "adaptive_efficiency_pct": adaptive_efficiency,
        "adaptive_raw_score": round(adaptive_raw_score, 6) if adaptive_raw_score is not None else None,
        "adaptive_raw_deviation": round(abs(adaptive_raw_score - theta_full), 6) if adaptive_raw_score is not None else None,
        "bayesian_theta": round(bayes_theta, 6) if bayes_theta is not None else None,
        "bayesian_deviation": round(abs(bayes_theta - theta_full), 6) if bayes_theta is not None else None,
        "bayesian_ci_low": round(bayes_ci_low, 6) if bayes_ci_low is not None else None,
        "bayesian_ci_high": round(bayes_ci_high, 6) if bayes_ci_high is not None else None,
        "bayesian_ci_width": round(bayes_ci_width, 6) if bayes_ci_width is not None else None,
        "bayesian_n_items": bayes_n_items,
        "checkpoints": checkpoint_results,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _cell_label(cell_id: str) -> str:
    return cell_id.replace("_", " ").title()


def plot_deviation_curves(all_results: dict, output_dir: Path):
    """Plot score deviation vs trial count with uncertainty bands (3x3 grid).

    Shows mean |deviation| with +/- 1 std band across orderings, with the
    adaptive stopping point (raw mean and Bayesian posterior) marked.
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle(
        "Fixed-n Baseline: Estimation Error vs Trial Count\n"
        "(bands = +/- 1 std across random item orderings)",
        fontsize=13, fontweight="bold",
    )

    for (row, col), cell_id in GRID.items():
        ax = axes[row, col]
        if cell_id not in all_results:
            ax.set_visible(False)
            continue

        data = all_results[cell_id]
        ckpts = data["checkpoints"]
        theta_full = data["theta_full_data"]
        n_total = data["n_total_trials"]

        # Fixed-n checkpoints (excluding full run and adaptive-N if shown separately)
        fixed = [c for c in ckpts
                 if c["checkpoint"] < n_total and not c.get("is_adaptive_stop_n")]

        if fixed:
            trials = [c["n_trials"] for c in fixed]
            means = [c["abs_deviation_of_mean"] for c in fixed]
            stds = [c["ordering_std"] for c in fixed]

            ax.plot(trials, means, "o-", color="#4C72B0", linewidth=1.5,
                    markersize=5, label="Fixed-n (mean)", zorder=3)
            # +/- 1 std band
            lo = [max(0, m - s) for m, s in zip(means, stds)]
            hi = [m + s for m, s in zip(means, stds)]
            ax.fill_between(trials, lo, hi, color="#4C72B0", alpha=0.15, zorder=1)

            for c in fixed:
                ax.annotate(
                    f"n={c['checkpoint']}",
                    (c["n_trials"], c["abs_deviation_of_mean"]),
                    textcoords="offset points", xytext=(0, 8),
                    fontsize=6.5, ha="center", color="#4C72B0",
                )

        # Adaptive stopping: raw mean (diamond) and Bayesian posterior (star)
        stop_trial = data.get("adaptive_stop_trial")
        if stop_trial is not None:
            raw_dev = data.get("adaptive_raw_deviation")
            if raw_dev is not None:
                ax.plot(stop_trial, raw_dev, "D", color="#CC3333", markersize=7,
                        zorder=5, label=f"Adaptive raw (n={stop_trial})")

            bayes_dev = data.get("bayesian_deviation")
            if bayes_dev is not None:
                ax.plot(stop_trial, bayes_dev, "*", color="#228B22", markersize=10,
                        zorder=6, label=f"Adaptive Bayes")

            # Also show the fixed-n distribution at adaptive N (if computed)
            adaptive_ckpt = next(
                (c for c in ckpts if c.get("is_adaptive_stop_n")), None
            )
            if adaptive_ckpt:
                ax.plot(
                    adaptive_ckpt["n_trials"],
                    adaptive_ckpt["abs_deviation_of_mean"],
                    "s", color="#FF8C00", markersize=6, zorder=4,
                    label=f"Fixed-n at adaptive N",
                )

        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Trials used")
        ax.set_ylabel("|score - full run|")
        ax.set_title(_cell_label(cell_id), fontsize=10, fontweight="bold")
        ax.legend(fontsize=5.5, loc="upper right", framealpha=0.85)

        # Annotation
        eff = data.get("adaptive_efficiency_pct", 0)
        ann = f"Adaptive eff: {eff:.0f}%"
        if data.get("bayesian_deviation") is not None:
            ann += f"\nBayes |dev|: {data['bayesian_deviation']:.4f}"
        ax.text(
            0.98, 0.60, ann,
            transform=ax.transAxes, fontsize=6.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

    for col, label in enumerate(COL_LABELS):
        axes[0, col].set_title(
            f"{label}\n{axes[0, col].get_title()}",
            fontsize=10, fontweight="bold",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = output_dir / "deviation_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_efficiency_vs_accuracy(all_results: dict, output_dir: Path):
    """Scatter plot: efficiency vs |deviation| with error bars for ordering variance."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colours = {
        "binary": "#4C72B0",
        "ordinal": "#DD8452",
        "continuous": "#55A868",
    }

    for cell_id, data in all_results.items():
        mode = data["mode"]
        colour = colours.get(mode, "gray")
        n_total = data["n_total_trials"]

        # Fixed checkpoints (excluding adaptive-N and full run)
        fixed = [c for c in data["checkpoints"]
                 if c["checkpoint"] < n_total and not c.get("is_adaptive_stop_n")]

        if fixed:
            effs = [c["efficiency_pct"] for c in fixed]
            devs = [c["abs_deviation_of_mean"] for c in fixed]
            stds = [c["ordering_std"] for c in fixed]
            ax.errorbar(effs, devs, yerr=stds, fmt="o", color=colour,
                        markersize=4, alpha=0.5, capsize=2, linewidth=0.8, zorder=2)

        # Adaptive point (Bayesian if available, else raw)
        eff = data.get("adaptive_efficiency_pct")
        bayes_dev = data.get("bayesian_deviation")
        raw_dev = data.get("adaptive_raw_deviation")
        if eff is not None:
            if bayes_dev is not None:
                ax.scatter([eff], [bayes_dev], c=colour, marker="*", s=120,
                           zorder=5, edgecolors="black", linewidths=0.6)
                ax.annotate(
                    _cell_label(cell_id),
                    (eff, bayes_dev), fontsize=6, ha="left", va="bottom",
                    textcoords="offset points", xytext=(5, 3),
                )
            elif raw_dev is not None:
                ax.scatter([eff], [raw_dev], c=colour, marker="D", s=60,
                           zorder=5, edgecolors="black", linewidths=0.6)
                ax.annotate(
                    _cell_label(cell_id),
                    (eff, raw_dev), fontsize=6, ha="left", va="bottom",
                    textcoords="offset points", xytext=(5, 3),
                )

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=m.title())
        for m, c in colours.items()
    ]
    # Check if any cell has Bayesian results
    has_bayesian = any(d.get("bayesian_theta") is not None for d in all_results.values())
    if has_bayesian:
        legend_elements.append(
            Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=10, label="Adaptive (Bayesian)")
        )
    else:
        legend_elements.append(
            Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=8, label="Adaptive (raw mean)")
        )
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=6, alpha=0.5, label="Fixed-n (raw mean +/- std)")
    )
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left", framealpha=0.9)

    ax.set_xlabel("Efficiency (% trials saved)", fontsize=11)
    ax.set_ylabel("|Score deviation from full run|", fontsize=11)
    ax.set_title(
        "Efficiency vs Accuracy Trade-off:\nAdaptive Stopping vs Fixed-n Baselines",
        fontsize=13, fontweight="bold",
    )
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)

    plt.tight_layout()
    out = output_dir / "efficiency_vs_accuracy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_summary_table(all_results: dict, checkpoints: list[int], output_dir: Path):
    """Summary table with variance information and Bayesian reference."""
    cells = [c for c in CELL_ORDER if c in all_results]
    if not cells:
        return

    rows = []
    for cell_id in cells:
        data = all_results[cell_id]
        n_total = data["n_total_trials"]
        row = [
            _cell_label(cell_id),
            f"{data['theta_full_data']:.3f}",
            str(n_total),
        ]

        # Fixed-n columns: "mean +/- std (eff%)"
        ckpt_map = {c["checkpoint"]: c for c in data["checkpoints"]}
        for n in checkpoints:
            if n in ckpt_map:
                c = ckpt_map[n]
                row.append(
                    f"{c['score_mean']:.3f} +/- {c['score_std']:.3f}\n"
                    f"({c['efficiency_pct']:.0f}%)"
                )
            else:
                row.append("--")

        # Adaptive column: raw mean and Bayesian
        stop_n = data.get("adaptive_stop_trial")
        eff = data.get("adaptive_efficiency_pct", 0)
        raw_s = data.get("adaptive_raw_score")
        bayes_s = data.get("bayesian_theta")

        parts = []
        if raw_s is not None:
            parts.append(f"raw: {raw_s:.3f}")
        if bayes_s is not None:
            parts.append(f"Bayes: {bayes_s:.3f}")
        parts.append(f"n={stop_n}, eff={eff:.0f}%")
        row.append("\n".join(parts))

        rows.append(row)

    col_labels = ["Cell", "Full-run", "Total\ntrials"] + [
        f"n={n}\nmean +/- std\n(eff%)" for n in checkpoints
    ] + ["Adaptive\nstop"]

    fig, ax = plt.subplots(figsize=(20, 6))
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 2.0)

    n_cols = len(col_labels)
    for j in range(n_cols):
        table[0, j].set_facecolor("#4C72B0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(rows) + 1):
        colour = "#f0f0f0" if i % 2 == 0 else "white"
        for j in range(n_cols):
            table[i, j].set_facecolor(colour)

    fig.suptitle(
        "Fixed-n Baseline vs Adaptive Stopping\n"
        "(scores across 100 random item orderings)",
        fontsize=13, fontweight="bold", y=0.99,
    )
    out = output_dir / "fixed_n_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fixed-n baseline comparison (Validation Gap 3)")
    parser.add_argument("--cells", nargs="+", default=None,
                        help="Specific cells to analyse (default: all 9)")
    parser.add_argument("--checkpoints", nargs="+", type=int, default=DEFAULT_CHECKPOINTS,
                        help=f"Fixed trial counts to evaluate (default: {DEFAULT_CHECKPOINTS})")
    parser.add_argument("--n-orderings", type=int, default=DEFAULT_N_ORDERINGS,
                        help=f"Random orderings per checkpoint (default: {DEFAULT_N_ORDERINGS})")
    parser.add_argument("--bayesian", action="store_true",
                        help="Run posthoc API to get the system's actual Bayesian output (slow)")
    parser.add_argument("--mcmc-draws", type=int, default=1000,
                        help="MCMC draws/tune for --bayesian mode (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory for figures (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    cells_to_run = args.cells if args.cells else CELL_ORDER
    checkpoints = sorted(args.checkpoints)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for c in cells_to_run:
        if c not in CELL_CONFIGS:
            print(f"ERROR: Unknown cell '{c}'. Valid: {list(CELL_CONFIGS.keys())}")
            sys.exit(1)

    print("=" * 70)
    print("Fixed-n Baseline Comparison (Validation Gap 3)")
    print(f"  Cells: {len(cells_to_run)}")
    print(f"  Checkpoints: {checkpoints}")
    print(f"  Orderings per checkpoint: {args.n_orderings}")
    print(f"  Bayesian estimates: {'YES' if args.bayesian else 'NO (use --bayesian to enable)'}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    all_results = {}
    total_start = time.time()

    for cell_idx, cell_id in enumerate(cells_to_run):
        cell_start = time.time()
        print(f"\n{'='*70}")
        print(f"[{cell_idx+1}/{len(cells_to_run)}] {cell_id}")
        print(f"{'='*70}")

        try:
            result = analyse_cell(
                cell_id, checkpoints,
                n_orderings=args.n_orderings,
                base_seed=args.seed,
                bayesian=args.bayesian,
                mcmc_draws=args.mcmc_draws,
            )
            all_results[cell_id] = result
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        cell_time = time.time() - cell_start
        print(f"  Cell time: {cell_time:.1f}s")

    total_time = time.time() - total_start

    # Save JSON
    output_json = RESULTS_DIR / "fixed_n_baseline.json"
    json_output = {
        "config": {
            "checkpoints": checkpoints,
            "n_orderings": args.n_orderings,
            "seed": args.seed,
            "cells": cells_to_run,
            "total_duration_sec": round(total_time, 1),
        },
        "cells": all_results,
    }
    with open(output_json, "w") as f:
        json.dump(json_output, f, indent=2, default=str)
    print(f"\nSaved results: {output_json}")

    # Cross-cell summary
    print(f"\n{'='*70}")
    print(f"SUMMARY ({total_time:.1f}s)")
    print(f"{'='*70}")

    print(f"\n  {'Cell':<18} {'Full':>6} | ", end="")
    for n in checkpoints:
        print(f"{'n='+str(n):>14} | ", end="")
    print(f"{'Adaptive':>14} | {'Bayes':>10}")
    print(f"  {'-'*18} {'-'*6}-+-" + "-+-".join(["-" * 14] * len(checkpoints))
          + f"-+-{'-'*14}-+-{'-'*10}")

    for cell_id in cells_to_run:
        if cell_id not in all_results:
            continue
        data = all_results[cell_id]
        ckpt_map = {c["checkpoint"]: c for c in data["checkpoints"]}

        print(f"  {cell_id:<18} {data['theta_full_data']:>6.3f} | ", end="")
        for n in checkpoints:
            if n in ckpt_map:
                c = ckpt_map[n]
                print(f"{c['score_mean']:.3f}+/-{c['score_std']:.3f} | ", end="")
            else:
                print(f"{'--':>14} | ", end="")

        raw_dev = data.get("adaptive_raw_deviation")
        eff = data.get("adaptive_efficiency_pct", 0)
        stop_n = data.get("adaptive_stop_trial")
        if raw_dev is not None:
            print(f" n={stop_n} ({eff:.0f}%) | ", end="")
        else:
            print(f"{'N/A':>14} | ", end="")

        bayes_dev = data.get("bayesian_deviation")
        if bayes_dev is not None:
            print(f"|dev|={bayes_dev:.4f}")
        else:
            print(f"{'N/A':>10}")

    # Figures
    if not args.skip_figures and all_results:
        print(f"\nGenerating figures...")
        plot_deviation_curves(all_results, output_dir)
        plot_efficiency_vs_accuracy(all_results, output_dir)
        plot_summary_table(all_results, checkpoints, output_dir)

    print(f"\nDone. Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
