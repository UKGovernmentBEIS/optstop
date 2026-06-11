#!/usr/bin/env python3
"""
analyse_posthoc_consistency.py

Post-hoc consistency analysis using optimal_stopping_posthoc.

For each of the 9 matrix cells, loads the full shadow-mode dataset and
replays the stopping decision N times with different shuffled item orderings.
Uses optimal_stopping_posthoc (the batch posthoc API) with shuffled sample_id
ordering, following the pattern from ord_consist_optstop_analysis.py.

Key design:
    Sample IDs are shuffled within each grouping before calling
    optimal_stopping_posthoc. This changes which items are seen first,
    testing whether the stopping point is robust to item presentation order.
    Each shuffled run makes a single call to the posthoc API, which handles
    all internal model construction, MCMC, and stopping decisions.

Assesses three properties (per paper §order_robustness):
    1. Stopping point consistency — distribution of items used
    2. Estimate stability — theta estimates vs full-run reference
    3. Credible interval coverage — proportion of reps where the 97% CI
       contains the full-run estimate

Usage:
    python analyse_posthoc_consistency.py                    # Full run (epoch-interleaved, all 9 cells, 15 reps)
    python analyse_posthoc_consistency.py --cells high_continuous --n-reps 1  # Quick test
    python analyse_posthoc_consistency.py --n-reps 5         # Fewer reps
    python analyse_posthoc_consistency.py --fast              # Faster MCMC (500 draws/tune)
    python analyse_posthoc_consistency.py --processing-order item_greedy  # Legacy item-greedy mode

Output:
    results/matrix/posthoc_consistency.json     — all per-rep and per-cell data
    results/matrix/figures/posthoc/*.png        — visualisations
"""

import argparse
import json
import sys
import time
import traceback
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import optstop
from inspect_ai.log import read_eval_log
from run_matrix_evals import CELL_CONFIGS, UNIVERSAL_OPTSTOP_PARAMS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_REPS = 15
BASE_SEED = 1000
RESULTS_DIR = Path("results/matrix")
OUTPUT_DIR = RESULTS_DIR / "figures" / "posthoc"

# Process order: binary (fast) → continuous (fast) → ordinal (slowest)
CELL_ORDER = [
    "low_binary", "mid_binary", "high_binary",
    "low_continuous", "mid_continuous", "high_continuous",
    "low_ordinal", "mid_ordinal", "high_ordinal",
]

# 3×3 grid layout: rows=performance, cols=score type
GRID = {
    (0, 0): "low_binary",    (0, 1): "low_ordinal",    (0, 2): "low_continuous",
    (1, 0): "mid_binary",    (1, 1): "mid_ordinal",    (1, 2): "mid_continuous",
    (2, 0): "high_binary",   (2, 1): "high_ordinal",   (2, 2): "high_continuous",
}
COL_LABELS = ["Binary", "Ordinal", "Continuous"]


# ---------------------------------------------------------------------------
# Score extraction (raw, no HiBayES transforms)
# ---------------------------------------------------------------------------

def extract_raw_scores(eval_log, cell_config: dict) -> pd.DataFrame:
    """Extract raw scores from eval log in the format expected by posthoc API.

    Returns a DataFrame with columns: sample_id, epoch, score, model, task.

    Score ranges:
      - Binary:     0 or 1 (correct/incorrect)
      - Ordinal:    integer in [1, ordinal_max_score] (raw, NOT remapped)
      - Continuous:  float, clamped >= 0 (raw, NOT normalised)
    """
    BINARY_MAP = {"C": 1, "CORRECT": 1, "I": 0, "INCORRECT": 0}

    mode = cell_config.get("scoring_mode", "binary")
    ordinal_max_score = cell_config.get("ordinal_max_score", 10)
    score_choice = cell_config.get("score_choice")
    score_value_key = cell_config.get("score_value_key")

    rows = []
    if not eval_log.samples:
        return pd.DataFrame(columns=["sample_id", "epoch", "score", "model", "task"])

    model = eval_log.eval.model
    task = eval_log.eval.task

    for sample in eval_log.samples:
        if not sample.scores:
            continue

        # Select scorer
        if score_choice and score_choice in sample.scores:
            scorer_name = score_choice
        else:
            scorer_name = next(iter(sample.scores))
        val = sample.scores[scorer_name].value

        score = None

        if mode == "binary":
            if isinstance(val, str):
                score = BINARY_MAP.get(val.upper())
            elif isinstance(val, (int, float)):
                score = int(val)

        elif mode == "ordinal":
            if isinstance(val, (int, float)):
                # Round to nearest integer, clamp to [1, max] — NO remap
                score_int = int(np.round(float(val)))
                score = max(1, min(ordinal_max_score, score_int))

        elif mode == "continuous":
            if isinstance(val, dict) and score_value_key:
                score = val.get(score_value_key)
            elif isinstance(val, (int, float)):
                score = float(val)
            if score is not None:
                score = max(0.0, float(score))

        if score is not None:
            rows.append({
                "sample_id": sample.id,
                "epoch": sample.epoch,
                "score": score,
                "model": model,
                "task": task,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cell_data(cell_id: str) -> tuple[pd.DataFrame, dict, dict]:
    """Load shadow result JSON, read eval log, extract raw scores.

    Returns (df, cell_config, result_json).
    """
    cell_dir = RESULTS_DIR / cell_id
    shadow_files = sorted(cell_dir.glob("shadow_*.json"))
    if not shadow_files:
        raise FileNotFoundError(f"No shadow result JSON in {cell_dir}")
    result_path = shadow_files[-1]  # Most recent

    with open(result_path) as f:
        result = json.load(f)

    cell_config = result["cell_config"]
    log_file = result["log_file"]

    print(f"  Loading eval log: {Path(log_file).name}")
    eval_log = read_eval_log(log_file)

    print(f"  Extracting raw scores (mode={cell_config['scoring_mode']})...")
    df = extract_raw_scores(eval_log, cell_config)
    n_items = df["sample_id"].nunique()
    n_rows = len(df)
    print(f"  Extracted {n_rows} rows, {n_items} unique items")

    return df, cell_config, result


def get_full_data_reference(
    df: pd.DataFrame, cell_config: dict,
) -> float:
    """Compute the full-data reference score (normalised to [0,1]).

    For all modes, uses the normalised mean:
      - Binary: mean of 0/1 scores (proportion correct, already [0,1])
      - Ordinal: mean(scores) / ordinal_max_score
      - Continuous: mean(scores) / ordinal_max_score

    This matches the reference implementation (ord_consist_optstop_analysis.py)
    and the scale of the posthoc API's theta_ci bounds.
    """
    mode = cell_config.get("scoring_mode", "binary")
    ordinal_max_score = cell_config.get("ordinal_max_score", 10)

    if mode == "binary":
        return float(df["score"].mean())
    else:
        return float(df["score"].mean()) / ordinal_max_score


def get_shadow_stop_items(cell_id: str) -> int | None:
    """Get the number of unique items at the shadow eval's would-have-stopped point.

    Used as reference marker in figures.
    """
    cell_dir = RESULTS_DIR / cell_id
    shadow_files = sorted(cell_dir.glob("shadow_*.json"))
    if not shadow_files:
        return None
    try:
        with open(shadow_files[-1]) as f:
            result = json.load(f)
        sms = result.get("diagnostics", {}).get("shadow_mode_summary", {})
        would_have_stopped_at = sms.get("would_have_stopped_at")
        if would_have_stopped_at is None:
            return None
        n_total_items = result.get("cell_config", {}).get("default_limit", 200)
        # Epoch-interleaved: first n_total_items trials are epoch 1 for all items
        return min(would_have_stopped_at, n_total_items)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Posthoc simulation
# ---------------------------------------------------------------------------


def run_all_reps_posthoc(
    df: pd.DataFrame,
    cell_config: dict,
    n_reps: int,
    base_seed: int,
    mcmc_draws: int | None = None,
    processing_order: str = 'epoch_interleaved',
    reanalysis_interval: int = 10,
) -> list[dict]:
    """Run all shuffled reps via the optimal_stopping_posthoc public API.

    Each rep calls the public API with shuffle_items=True and a unique seed,
    which randomises item processing order. The processing_order parameter
    controls whether items are processed item-greedy (fast, all epochs per
    item) or epoch-interleaved (matches production bridge behaviour).

    Returns a list of per-rep result dicts.
    """
    n_total_items = df["sample_id"].nunique()
    n_total_epochs = df["epoch"].nunique()
    mode = cell_config.get("scoring_mode", "binary")
    ordinal_max_score = cell_config.get("ordinal_max_score") or 10

    # Build params dict
    params = UNIVERSAL_OPTSTOP_PARAMS.copy()
    params["low_performance_threshold"] = 0.01
    if mcmc_draws is not None:
        params["draws"] = mcmc_draws
        params["tune"] = mcmc_draws

    # Build kwargs for public API (modelled on ord_consist_optstop_analysis.py)
    optstop_kwargs = {
        'params': params,
        'grouping_columns': ['model', 'task'],
        'sample_id_column': 'sample_id',
        'epoch_column': 'epoch',
        'score_column': 'score',
        'display_progress': False,
        'generate_diagnostics': False,
        'ordinal_max_score': ordinal_max_score,
        'processing_order': processing_order,
        'reanalysis_interval': reanalysis_interval,
    }

    if mode == "ordinal":
        optstop_kwargs['ordinal_tasks'] = ['']
        optstop_kwargs['ordinal_inference'] = 'hybrid'
        optstop_kwargs['ordinal_model_type'] = cell_config.get(
            'ordinal_model_type', 'ordered_logistic'
        )
    elif mode == "continuous":
        optstop_kwargs['continuous_tasks'] = ['']

    results = []
    for rep in range(n_reps):
        seed = base_seed + rep
        rep_start = time.time()
        print(f"  Rep {rep+1}/{n_reps} (seed={seed})...", end=" ", flush=True)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, summaries = optstop.optimal_stopping_posthoc(
                    df=df,
                    shuffle_items=True,
                    shuffle_seed=seed,
                    **optstop_kwargs,
                )
        except Exception as e:
            rep_time = time.time() - rep_start
            print(f"EXCEPTION: {e} [{rep_time:.1f}s]")
            results.append({"shuffle_seed": seed, "error": str(e)})
            continue

        rep_time = time.time() - rep_start

        # Single grouping per cell — extract first summary
        if not summaries or summaries[0].get("error") is not None:
            error_msg = summaries[0]["error"] if summaries else "No summaries returned"
            print(f"ERROR: {error_msg} [{rep_time:.1f}s]")
            results.append({"shuffle_seed": seed, "error": error_msg})
            continue

        s = summaries[0]
        n_items_used = s["n_items_used"]
        frac_items_used = s["percent_items_used"]  # fraction 0.0–1.0
        avg_reps = s.get("avg_reps_per_item")

        theta_lo = s.get("theta_ci_low")
        theta_hi = s.get("theta_ci_high")
        theta_width = s.get("theta_ci_width")

        if theta_lo is not None and theta_hi is not None:
            theta_estimate = (theta_lo + theta_hi) / 2
        else:
            theta_estimate = None

        # Determine stopping and efficiency based on processing order.
        # In epoch-interleaved mode, percent_items_used can be 1.0 even when
        # stopping occurred (all items seen after epoch 1). Use trial-based
        # metric: total_trials_used / max_possible_trials.
        if processing_order == 'epoch_interleaved' and avg_reps is not None:
            total_trials_used = n_items_used * avg_reps
            max_trials = n_total_items * n_total_epochs
            frac_trials_used = total_trials_used / max_trials if max_trials > 0 else 1.0
            stopped = frac_trials_used < 1.0
            efficiency_pct = (1.0 - frac_trials_used) * 100.0 if stopped else 0.0
        else:
            # item-greedy: percent_items_used correctly reflects stopping
            stopped = frac_items_used < 1.0
            efficiency_pct = (1.0 - frac_items_used) * 100.0 if stopped else 0.0

        result = {
            "shuffle_seed": seed,
            "stopped": stopped,
            "n_items_used": n_items_used,
            "n_total_items": n_total_items,
            "percent_items_used": round(frac_items_used * 100, 2),
            "efficiency_pct": round(efficiency_pct, 2),
            "theta_estimate": round(theta_estimate, 6) if theta_estimate is not None else None,
            "theta_ci_low": round(float(theta_lo), 6) if theta_lo is not None else None,
            "theta_ci_high": round(float(theta_hi), 6) if theta_hi is not None else None,
            "theta_ci_width": round(float(theta_width), 6) if theta_width is not None else None,
            "avg_reps_per_item": avg_reps,
            "n_total_epochs": n_total_epochs,
            "processing_order": s.get("processing_order", processing_order),
            "duration_sec": round(rep_time, 1),
            "error": None,
        }
        results.append(result)

        # Print result
        status = "STOPPED" if stopped else "NO STOP"
        theta_str = f"\u03b8={theta_estimate:.4f}" if theta_estimate is not None else "\u03b8=N/A"
        ci_str = (f"CI=[{theta_lo:.3f}, {theta_hi:.3f}]"
                  if theta_lo is not None else "CI=N/A")
        reps_str = f"reps={avg_reps:.1f}/{n_total_epochs}" if avg_reps is not None else ""
        items_str = f"items={n_items_used}/{n_total_items}"
        print(f"{status} | eff={efficiency_pct:.1f}% "
              f"| {items_str} {reps_str} | {theta_str} | {ci_str} | {rep_time:.1f}s")

    return results


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def ci_jaccard(ci_a: tuple, ci_b: tuple) -> float:
    """Compute Jaccard overlap between two credible intervals."""
    lo_a, hi_a = ci_a
    lo_b, hi_b = ci_b
    intersection = max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))
    union = max(hi_a, hi_b) - min(lo_a, lo_b)
    if union <= 0:
        return 1.0
    return intersection / union


def compute_cell_summary(
    cell_id: str,
    rep_metrics: list[dict],
    theta_full_data: float,
) -> dict:
    """Compute summary statistics across repetitions for one cell.

    Covers all three properties from §order_robustness:
      1. Stopping point consistency (n_items_used mean/std/cv)
      2. Estimate stability (theta_mean/std/abs_diff/max_abs_diff)
      3. CI coverage (ci_coverage_count/rate)
    """
    valid = [m for m in rep_metrics if m.get("error") is None]
    stopped = [m for m in valid if m["stopped"]]

    if not valid:
        return {"cell_id": cell_id, "error": "All reps failed"}

    efficiencies = [m["efficiency_pct"] for m in valid]
    thetas = [m["theta_estimate"] for m in valid if m["theta_estimate"] is not None]
    ci_widths = [m["theta_ci_width"] for m in valid if m["theta_ci_width"] is not None]
    items_used = [m["n_items_used"] for m in valid if m.get("n_items_used") is not None]

    # --- Property 1: Stopping point consistency ---
    items_used_cv = None
    if len(items_used) > 1 and np.mean(items_used) > 0:
        items_used_cv = round(np.std(items_used, ddof=1) / np.mean(items_used), 4)

    # --- Property 2: Estimate stability ---
    abs_diffs = [abs(t - theta_full_data) for t in thetas] if thetas else []
    theta_max_abs_diff = round(max(abs_diffs), 6) if abs_diffs else None

    # --- Property 3: CI coverage ---
    ci_reps = [
        m for m in valid
        if m["theta_ci_low"] is not None and m["theta_ci_high"] is not None
    ]
    n_with_ci = len(ci_reps)
    ci_coverage_count = sum(
        1 for m in ci_reps
        if m["theta_ci_low"] <= theta_full_data <= m["theta_ci_high"]
    )
    ci_coverage_rate = ci_coverage_count / n_with_ci if n_with_ci > 0 else None

    # Pairwise CI Jaccard overlap
    ci_pairs = [(m["theta_ci_low"], m["theta_ci_high"]) for m in ci_reps]
    jaccards = [ci_jaccard(a, b) for a, b in combinations(ci_pairs, 2)]

    return {
        "cell_id": cell_id,
        "n_reps": len(rep_metrics),
        "n_valid": len(valid),
        "n_failed": len(rep_metrics) - len(valid),
        # Property 1
        "stop_rate": len(stopped) / len(valid) if valid else 0.0,
        "efficiency_mean": round(np.mean(efficiencies), 2),
        "efficiency_std": round(np.std(efficiencies, ddof=1), 2) if len(efficiencies) > 1 else 0.0,
        "efficiency_min": round(min(efficiencies), 2),
        "efficiency_max": round(max(efficiencies), 2),
        "items_used_mean": round(np.mean(items_used), 1) if items_used else None,
        "items_used_std": round(np.std(items_used, ddof=1), 1) if len(items_used) > 1 else 0.0,
        "items_used_cv": items_used_cv,
        # Property 2
        "theta_mean": round(np.mean(thetas), 6) if thetas else None,
        "theta_std": round(np.std(thetas, ddof=1), 6) if len(thetas) > 1 else 0.0,
        "theta_full_data": theta_full_data,
        "theta_abs_diff_mean": round(np.mean(abs_diffs), 6) if abs_diffs else None,
        "theta_max_abs_diff": theta_max_abs_diff,
        # Property 3
        "ci_coverage_count": ci_coverage_count,
        "ci_coverage_n": n_with_ci,
        "ci_coverage_rate": round(ci_coverage_rate, 4) if ci_coverage_rate is not None else None,
        "ci_width_mean": round(np.mean(ci_widths), 6) if ci_widths else None,
        "ci_width_std": round(np.std(ci_widths, ddof=1), 6) if len(ci_widths) > 1 else 0.0,
        "pairwise_ci_jaccard_mean": round(np.mean(jaccards), 4) if jaccards else 1.0,
        "pairwise_ci_jaccard_min": round(min(jaccards), 4) if jaccards else 1.0,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _cell_label(cell_id: str) -> str:
    return cell_id.replace("_", " ").title()


def plot_items_used_distribution(
    all_results: dict,
    shadow_stop_items: dict,
    output_dir: Path,
    n_reps: int,
):
    """Property 1: Strip/box plot of items used across reps per cell (3x3 grid).

    Shows the distribution of items used before stopping for each cell.
    Red diamond = shadow eval's items at stopping point (reference).
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle(
        f"Stopping Point Consistency\n"
        f"({n_reps} shuffled item orderings per cell)",
        fontsize=14, fontweight="bold",
    )

    # Detect whether results are epoch-interleaved (plot efficiency) or item-greedy (plot items)
    any_epoch_interleaved = any(
        r.get("processing_order") == "epoch_interleaved"
        for cell_data in all_results.values()
        for r in cell_data.get("reps", [])
        if r.get("error") is None
    )

    for (row, col), cell_id in GRID.items():
        ax = axes[row, col]
        if cell_id not in all_results:
            ax.set_visible(False)
            continue

        reps = all_results[cell_id]["reps"]
        summary = all_results[cell_id]["summary"]
        valid = [r for r in reps if r.get("error") is None]

        n_total = valid[0]["n_total_items"] if valid else 200

        if any_epoch_interleaved:
            # Plot efficiency (%) for epoch-interleaved mode
            stopped_vals = [r["efficiency_pct"] for r in valid if r["stopped"]]
            n_not_stopped = sum(1 for r in valid if not r["stopped"])

            if stopped_vals:
                ax.boxplot(
                    stopped_vals, vert=False, widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", alpha=0.6),
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color="#4C72B0"),
                    capprops=dict(color="#4C72B0"),
                )
                jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(stopped_vals))
                ax.scatter(
                    stopped_vals, np.ones(len(stopped_vals)) + jitter,
                    c="#333333", s=18, alpha=0.7, zorder=3,
                )

            if n_not_stopped > 0:
                jitter_ns = np.random.RandomState(43).uniform(-0.15, 0.15, n_not_stopped)
                ax.scatter(
                    [0] * n_not_stopped, np.ones(n_not_stopped) + jitter_ns,
                    c="#CC3333", s=25, marker="x", zorder=4, label="Did not stop",
                )

            # Shadow eval efficiency reference
            shadow_items = shadow_stop_items.get(cell_id)
            if shadow_items is not None:
                shadow_eff = (1.0 - shadow_items / n_total) * 100.0
                ax.axvline(shadow_eff, color="#CC3333", linestyle="--", linewidth=1.2, alpha=0.8)
                ax.scatter(
                    [shadow_eff], [1.6], c="#CC3333", s=50, marker="D", zorder=5,
                    label=f"Shadow: {shadow_eff:.0f}%",
                )

            ax.set_xlim(-5, 105)
            ax.set_yticks([])
            ax.set_xlabel("Efficiency (%)")

        else:
            # Plot items used for item-greedy mode
            stopped_vals = [r["n_items_used"] for r in valid if r["stopped"]]
            n_not_stopped = sum(1 for r in valid if not r["stopped"])

            if stopped_vals:
                ax.boxplot(
                    stopped_vals, vert=False, widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", alpha=0.6),
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color="#4C72B0"),
                    capprops=dict(color="#4C72B0"),
                )
                jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(stopped_vals))
                ax.scatter(
                    stopped_vals, np.ones(len(stopped_vals)) + jitter,
                    c="#333333", s=18, alpha=0.7, zorder=3,
                )

            if n_not_stopped > 0:
                jitter_ns = np.random.RandomState(43).uniform(-0.15, 0.15, n_not_stopped)
                ax.scatter(
                    [n_total] * n_not_stopped, np.ones(n_not_stopped) + jitter_ns,
                    c="#CC3333", s=25, marker="x", zorder=4, label="Did not stop",
                )

            shadow_items = shadow_stop_items.get(cell_id)
            if shadow_items is not None:
                ax.axvline(shadow_items, color="#CC3333", linestyle="--", linewidth=1.2, alpha=0.8)
                ax.scatter(
                    [shadow_items], [1.6], c="#CC3333", s=50, marker="D", zorder=5,
                    label=f"Shadow: {shadow_items}",
                )

            ax.set_xlim(0, n_total + 10)
            ax.set_yticks([])
            ax.set_xlabel("Items used")

        ax.set_title(_cell_label(cell_id), fontsize=10, fontweight="bold")

        # Annotation
        ann_lines = [f"stop={summary['stop_rate']*100:.0f}%"]
        ann_lines.append(f"eff={summary['efficiency_mean']:.0f}%")
        if summary.get("items_used_cv") is not None:
            ann_lines.append(f"CV={summary['items_used_cv']:.3f}")
        ax.text(
            0.98, 0.95, "\n".join(ann_lines),
            transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )
        if shadow_items is not None or n_not_stopped > 0:
            ax.legend(fontsize=6, loc="lower right", framealpha=0.85)

    for col, label in enumerate(COL_LABELS):
        axes[0, col].set_title(f"{label}\n{axes[0, col].get_title()}", fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = output_dir / "items_used_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_ci_coverage_forest(
    all_results: dict,
    output_dir: Path,
    n_reps: int,
):
    """Properties 2 & 3: Forest plot with CI coverage highlighting (3x3 grid).

    Each rep's 97% CI is shown as a horizontal bar. Blue = contains full-run
    estimate; red = misses. Vertical dashed line = full-run estimate.
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    fig.suptitle(
        f"Estimate Stability & Credible Interval Coverage\n"
        f"({n_reps} shuffled orderings; blue = CI contains full-run estimate, red = miss)",
        fontsize=13, fontweight="bold",
    )

    for (row, col), cell_id in GRID.items():
        ax = axes[row, col]
        if cell_id not in all_results:
            ax.set_visible(False)
            continue

        reps = all_results[cell_id]["reps"]
        summary = all_results[cell_id]["summary"]
        theta_full = summary["theta_full_data"]

        valid = [
            r for r in reps
            if r.get("error") is None and r.get("theta_ci_low") is not None
        ]
        n = len(valid)
        if n == 0:
            ax.text(
                0.5, 0.5, "No valid CIs",
                transform=ax.transAxes, ha="center", va="center", fontsize=10,
            )
            ax.set_title(_cell_label(cell_id), fontsize=10, fontweight="bold")
            continue

        for i, r in enumerate(valid):
            lo, hi = r["theta_ci_low"], r["theta_ci_high"]
            t = r["theta_estimate"]
            contains = lo <= theta_full <= hi
            colour = "#4C72B0" if contains else "#CC3333"
            alpha = 0.6 if contains else 0.9
            ax.plot([lo, hi], [i, i], color=colour, linewidth=1.8, alpha=alpha)
            ax.plot(t, i, "o", color=colour, markersize=4, alpha=alpha)

        ax.axvline(theta_full, color="#CC3333", linestyle="--", linewidth=1.5,
                   label=f"Full run: {theta_full:.3f}")

        ax.set_title(_cell_label(cell_id), fontsize=10, fontweight="bold")
        ax.set_xlabel("\u03b8 estimate")
        ax.set_yticks([])
        ax.legend(fontsize=6.5, loc="upper right", framealpha=0.85)

        # Annotation
        cov = summary.get("ci_coverage_rate")
        cov_n = summary.get("ci_coverage_n", 0)
        cov_count = summary.get("ci_coverage_count", 0)
        ann_lines = []
        if cov is not None:
            ann_lines.append(f"coverage: {cov_count}/{cov_n} ({cov*100:.0f}%)")
        if summary.get("theta_abs_diff_mean") is not None:
            ann_lines.append(f"|\u0394\u03b8| mean: {summary['theta_abs_diff_mean']:.4f}")
        if summary.get("ci_width_mean") is not None:
            ann_lines.append(f"CI width: {summary['ci_width_mean']:.3f}")
        ax.text(
            0.02, 0.05, "\n".join(ann_lines),
            transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

    for col, label in enumerate(COL_LABELS):
        axes[0, col].set_title(f"{label}\n{axes[0, col].get_title()}", fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = output_dir / "ci_coverage_forest.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_summary_table(all_results: dict, output_dir: Path):
    """Combined summary table with all three properties."""
    cells = [c for c in CELL_ORDER if c in all_results]
    if not cells:
        return

    rows = []
    for cell_id in cells:
        s = all_results[cell_id]["summary"]

        items_str = "--"
        if s.get("items_used_mean") is not None:
            cv_str = f" ({s['items_used_cv']:.2f})" if s.get("items_used_cv") is not None else ""
            items_str = f"{s['items_used_mean']:.0f} \u00b1 {s['items_used_std']:.0f}{cv_str}"

        theta_str = f"{s['theta_mean']:.3f} \u00b1 {s['theta_std']:.3f}" if s.get("theta_mean") is not None else "N/A"

        diff_str = "N/A"
        if s.get("theta_abs_diff_mean") is not None:
            diff_str = f"{s['theta_abs_diff_mean']:.4f} ({s['theta_max_abs_diff']:.4f})"

        ci_str = f"{s['ci_width_mean']:.3f} \u00b1 {s['ci_width_std']:.3f}" if s.get("ci_width_mean") is not None else "N/A"

        cov_str = "N/A"
        if s.get("ci_coverage_rate") is not None:
            cov_str = f"{s['ci_coverage_count']}/{s['ci_coverage_n']}"

        rows.append([
            _cell_label(cell_id),
            f"{s['stop_rate']*100:.0f}%",
            f"{s['efficiency_mean']:.1f} \u00b1 {s['efficiency_std']:.1f}",
            items_str,
            theta_str,
            f"{s['theta_full_data']:.3f}",
            diff_str,
            ci_str,
            cov_str,
            f"{s['pairwise_ci_jaccard_mean']:.3f}",
        ])

    # Mean row across cells with stopping
    stopping_cells = [c for c in cells if all_results[c]["summary"].get("stop_rate", 0) > 0]
    if stopping_cells:
        ss = [all_results[c]["summary"] for c in stopping_cells]
        eff_mean = np.mean([s["efficiency_mean"] for s in ss])
        eff_std_mean = np.mean([s["efficiency_std"] for s in ss])
        diffs = [s["theta_abs_diff_mean"] for s in ss if s.get("theta_abs_diff_mean") is not None]
        max_diffs = [s["theta_max_abs_diff"] for s in ss if s.get("theta_max_abs_diff") is not None]
        ci_ws = [s["ci_width_mean"] for s in ss if s.get("ci_width_mean") is not None]
        covs = [s for s in ss if s.get("ci_coverage_rate") is not None]
        total_cov_count = sum(s["ci_coverage_count"] for s in covs)
        total_cov_n = sum(s["ci_coverage_n"] for s in covs)
        jacs = [s["pairwise_ci_jaccard_mean"] for s in ss]

        rows.append([
            "Mean (stopping)",
            f"{np.mean([s['stop_rate'] for s in ss])*100:.0f}%",
            f"{eff_mean:.1f} \u00b1 {eff_std_mean:.1f}",
            "",
            "",
            "",
            f"{np.mean(diffs):.4f} ({np.mean(max_diffs):.4f})" if diffs else "",
            f"{np.mean(ci_ws):.3f}" if ci_ws else "",
            f"{total_cov_count}/{total_cov_n}" if total_cov_n > 0 else "",
            f"{np.mean(jacs):.3f}" if jacs else "",
        ])

    col_labels = [
        "Cell", "Stop\nRate", "Efficiency (%)\n\u03bc \u00b1 \u03c3",
        "Items Used\n\u03bc \u00b1 \u03c3 (CV)",
        "\u03b8 estimate\n\u03bc \u00b1 \u03c3", "\u03b8 full\ndata",
        "|\u0394\u03b8| mean\n(max)",
        "CI width\n\u03bc \u00b1 \u03c3", "CI\ncoverage",
        "CI Jaccard\noverlap",
    ]

    fig, ax = plt.subplots(figsize=(20, 5.5))
    ax.axis("off")
    table = ax.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)

    n_cols = len(col_labels)
    for j in range(n_cols):
        table[0, j].set_facecolor("#4C72B0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    for i in range(1, len(rows) + 1):
        is_mean_row = (i == len(rows) and len(stopping_cells) > 0)
        if is_mean_row:
            colour = "#d4e6f1"
        else:
            colour = "#f0f0f0" if i % 2 == 0 else "white"
        for j in range(n_cols):
            table[i, j].set_facecolor(colour)
            if is_mean_row:
                table[i, j].set_text_props(fontweight="bold")

    fig.suptitle(
        "Presentation-Order Robustness Summary",
        fontsize=14, fontweight="bold", y=0.98,
    )
    out = output_dir / "consistency_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Reinterpretation for epoch-interleaved stopping
# ---------------------------------------------------------------------------

def reinterpret_epoch_interleaved_results(json_path: Path | None = None) -> dict:
    """Fix stopping detection for epoch-interleaved results.

    The optstop API's percent_items_used is always 1.0 in epoch-interleaved
    mode because all items are seen after epoch 1. This function recalculates
    stopped/efficiency_pct using avg_reps_per_item instead.
    """
    json_path = json_path or RESULTS_DIR / "posthoc_consistency.json"
    with open(json_path) as f:
        data = json.load(f)

    processing_order = data.get("config", {}).get("processing_order", "epoch_interleaved")
    if processing_order != "epoch_interleaved":
        print("  Not epoch_interleaved — no reinterpretation needed.")
        return data

    n_fixed = 0
    for cell_id, cell_data in data.get("cells", {}).items():
        reps = cell_data.get("reps", [])

        # Load cell data once per cell (for n_epochs and summary recomputation)
        n_total_epochs = None
        df_cell = None
        cell_config = None
        try:
            df_cell, cell_config, _ = load_cell_data(cell_id)
            n_total_epochs = df_cell["epoch"].nunique()
        except Exception:
            n_total_epochs = 10  # safe default for current matrix

        n_total_items = df_cell["sample_id"].nunique() if df_cell is not None else 200
        for r in reps:
            if r.get("error") is not None:
                continue
            avg_reps = r.get("avg_reps_per_item")
            n_items_used = r.get("n_items_used", n_total_items)
            r_epochs = r.get("n_total_epochs") or n_total_epochs
            if avg_reps is not None:
                was_stopped = r["stopped"]
                total_trials_used = n_items_used * avg_reps
                max_trials = n_total_items * r_epochs
                frac_trials = total_trials_used / max_trials if max_trials > 0 else 1.0
                r["stopped"] = frac_trials < 1.0
                r["efficiency_pct"] = round((1.0 - frac_trials) * 100, 2) if r["stopped"] else 0.0
                r["n_total_epochs"] = r_epochs
                if r["stopped"] != was_stopped:
                    n_fixed += 1

        # Recompute summary
        valid = [r for r in reps if r.get("error") is None]
        if valid and df_cell is not None and cell_config is not None:
            theta_full = get_full_data_reference(df_cell, cell_config)
            cell_data["summary"] = compute_cell_summary(cell_id, valid, theta_full)

    # Save fixed version
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"  Reinterpreted {n_fixed} reps across {len(data.get('cells', {}))} cells.")
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _save_json(all_results, cells_to_run, n_reps, base_seed, mcmc_draws,
               fast_mode, processing_order, reanalysis_interval, elapsed_sec):
    """Save results JSON incrementally (called after each cell and at end)."""
    output_json = RESULTS_DIR / "posthoc_consistency.json"
    json_output = {
        "config": {
            "n_reps": n_reps,
            "base_seed": base_seed,
            "mcmc_draws": mcmc_draws or 1000,
            "mcmc_tune": mcmc_draws or 1000,
            "fast_mode": fast_mode,
            "processing_order": processing_order,
            "reanalysis_interval": reanalysis_interval,
            "cells": cells_to_run,
            "method": "optimal_stopping_posthoc",
            "total_duration_sec": round(elapsed_sec, 1),
        },
        "cells": {},
    }
    for cell_id, data in all_results.items():
        json_output["cells"][cell_id] = {
            "reps": data["reps"],
            "summary": data["summary"],
        }
    with open(output_json, "w") as f:
        json.dump(json_output, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc consistency analysis via optimal_stopping_posthoc")
    parser.add_argument("--n-reps", type=int, default=N_REPS,
                        help=f"Shuffled repetitions per cell (default: {N_REPS})")
    parser.add_argument("--base-seed", type=int, default=BASE_SEED,
                        help=f"Base random seed (default: {BASE_SEED})")
    parser.add_argument("--cells", nargs="+", default=None,
                        help="Specific cells to run (default: all 9)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory for figures (default: {OUTPUT_DIR})")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--fast", action="store_true",
                        help="Faster MCMC: 500 draws/tune")
    parser.add_argument("--mcmc-draws", type=int, default=None,
                        help="Override MCMC draws and tune (default: 1000, fast: 500)")
    parser.add_argument("--processing-order", type=str, default="epoch_interleaved",
                        choices=["epoch_interleaved", "item_greedy"],
                        help="Processing order (default: epoch_interleaved)")
    parser.add_argument("--reanalysis-interval", type=int, default=10,
                        help="Trials between group model refreshes in epoch_interleaved mode (default: 10)")
    parser.add_argument("--reinterpret", action="store_true",
                        help="Reinterpret saved epoch-interleaved results (fix stopped/efficiency)")
    args = parser.parse_args()

    if args.reinterpret:
        print("Reinterpreting epoch-interleaved results...")
        reinterpret_epoch_interleaved_results()
        print("Done.")
        return

    n_reps = args.n_reps
    base_seed = args.base_seed
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mcmc_draws = args.mcmc_draws
    if args.fast and mcmc_draws is None:
        mcmc_draws = 500

    cells_to_run = args.cells if args.cells else CELL_ORDER
    for c in cells_to_run:
        if c not in CELL_CONFIGS:
            print(f"ERROR: Unknown cell '{c}'. Valid: {list(CELL_CONFIGS.keys())}")
            sys.exit(1)

    processing_order = args.processing_order
    reanalysis_interval = args.reanalysis_interval

    print(f"=" * 70)
    print(f"Post-hoc Consistency Analysis (optimal_stopping_posthoc)")
    print(f"  Cells: {len(cells_to_run)}")
    print(f"  Reps per cell: {n_reps}")
    print(f"  Seeds: {base_seed} to {base_seed + n_reps - 1}")
    print(f"  MCMC draws/tune: {mcmc_draws or 1000}")
    print(f"  Processing order: {processing_order}")
    if processing_order == 'epoch_interleaved':
        print(f"  Reanalysis interval: {reanalysis_interval}")
    if args.fast:
        print(f"  Mode: FAST")
    print(f"  Output: {output_dir}")
    print(f"=" * 70)

    # Load shadow stop points for reference in Figure 1
    shadow_stop_items = {}
    for cell_id in cells_to_run:
        items = get_shadow_stop_items(cell_id)
        if items is not None:
            shadow_stop_items[cell_id] = items

    all_results = {}
    total_start = time.time()

    for cell_idx, cell_id in enumerate(cells_to_run):
        cell_start = time.time()
        print(f"\n{'='*70}")
        print(f"[{cell_idx+1}/{len(cells_to_run)}] {cell_id}")
        print(f"{'='*70}")

        try:
            df, cell_config, result = load_cell_data(cell_id)
        except Exception as e:
            print(f"  ERROR loading {cell_id}: {e}")
            traceback.print_exc()
            continue

        theta_full = get_full_data_reference(df, cell_config)
        print(f"  Full-data reference: {theta_full:.4f} (normalised mean)")

        try:
            rep_metrics = run_all_reps_posthoc(
                df, cell_config,
                n_reps=n_reps,
                base_seed=base_seed,
                mcmc_draws=mcmc_draws,
                processing_order=processing_order,
                reanalysis_interval=reanalysis_interval,
            )
        except Exception as e:
            print(f"  EXCEPTION in posthoc: {e}")
            traceback.print_exc()
            rep_metrics = [{"shuffle_seed": base_seed + r, "error": str(e)} for r in range(n_reps)]

        summary = compute_cell_summary(cell_id, rep_metrics, theta_full)
        cell_time = time.time() - cell_start

        print(f"\n  --- {cell_id} summary ---")
        if summary.get("error"):
            print(f"  ERROR: {summary['error']}")
            print(f"  Cell time: {cell_time/60:.1f} min")
            all_results[cell_id] = {
                "cell_config": cell_config,
                "reps": rep_metrics,
                "summary": summary,
            }
            _save_json(all_results, cells_to_run, n_reps, base_seed, mcmc_draws,
                       args.fast, processing_order, reanalysis_interval,
                       time.time() - total_start)
            continue
        print(f"  Stop rate: {summary['stop_rate']*100:.0f}%")
        print(f"  Efficiency: {summary['efficiency_mean']:.1f}% \u00b1 {summary['efficiency_std']:.1f}%")
        if summary.get("items_used_mean") is not None:
            cv_str = f", CV={summary['items_used_cv']:.3f}" if summary.get("items_used_cv") is not None else ""
            print(f"  Items used: {summary['items_used_mean']:.0f} \u00b1 {summary['items_used_std']:.0f}{cv_str}")
        if summary.get("theta_mean") is not None:
            print(f"  \u03b8 estimate: {summary['theta_mean']:.4f} \u00b1 {summary['theta_std']:.4f}")
        print(f"  \u03b8 full data: {theta_full:.4f}")
        if summary.get("theta_abs_diff_mean") is not None:
            print(f"  |\u0394\u03b8| mean: {summary['theta_abs_diff_mean']:.4f}, max: {summary['theta_max_abs_diff']:.4f}")
        if summary.get("ci_coverage_rate") is not None:
            print(f"  CI coverage: {summary['ci_coverage_count']}/{summary['ci_coverage_n']} "
                  f"({summary['ci_coverage_rate']*100:.0f}%)")
        print(f"  CI Jaccard: {summary['pairwise_ci_jaccard_mean']:.4f}")
        print(f"  Cell time: {cell_time/60:.1f} min")

        all_results[cell_id] = {
            "cell_config": cell_config,
            "reps": rep_metrics,
            "summary": summary,
        }

        # Incremental save after each cell (so we don't lose results on crash)
        _save_json(all_results, cells_to_run, n_reps, base_seed, mcmc_draws,
                   args.fast, processing_order, reanalysis_interval,
                   time.time() - total_start)

    total_time = time.time() - total_start

    # Cross-cell summary
    print(f"\n{'='*70}")
    print(f"OVERALL SUMMARY ({total_time/60:.1f} min)")
    print(f"{'='*70}")

    summaries = [all_results[c]["summary"] for c in cells_to_run if c in all_results]
    if summaries:
        valid_summaries = [s for s in summaries if s.get("n_valid", 0) > 0]
        if valid_summaries:
            print(f"  Mean stop rate: {np.mean([s['stop_rate'] for s in valid_summaries])*100:.1f}%")
            print(f"  Mean efficiency: {np.mean([s['efficiency_mean'] for s in valid_summaries]):.1f}%")
            diffs = [s["theta_abs_diff_mean"] for s in valid_summaries if s.get("theta_abs_diff_mean") is not None]
            if diffs:
                print(f"  Mean |\u0394\u03b8| from full data: {np.mean(diffs):.4f}")
            # Overall CI coverage
            cov_summaries = [s for s in valid_summaries if s.get("ci_coverage_rate") is not None]
            if cov_summaries:
                total_cov = sum(s["ci_coverage_count"] for s in cov_summaries)
                total_n = sum(s["ci_coverage_n"] for s in cov_summaries)
                print(f"  Overall CI coverage: {total_cov}/{total_n} ({total_cov/total_n*100:.1f}%)")
            jaccards = [s["pairwise_ci_jaccard_mean"] for s in valid_summaries]
            print(f"  Mean pairwise CI Jaccard: {np.mean(jaccards):.4f}")

    # Final save
    _save_json(all_results, cells_to_run, n_reps, base_seed, mcmc_draws,
               args.fast, processing_order, reanalysis_interval, total_time)
    print(f"\n  Saved {RESULTS_DIR / 'posthoc_consistency.json'}")

    # Figures
    if not args.skip_figures and len(all_results) > 0:
        print(f"\nGenerating figures...")
        try:
            plot_items_used_distribution(all_results, shadow_stop_items, output_dir, n_reps)
            plot_ci_coverage_forest(all_results, output_dir, n_reps)
            plot_summary_table(all_results, output_dir)
        except Exception as e:
            print(f"  Figure generation error: {e}")
            traceback.print_exc()

    print(f"\nDone. Total time: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
