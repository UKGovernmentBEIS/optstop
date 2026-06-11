#!/usr/bin/env python3
"""
validate_conservatism.py

Validation Gap 1: Conservatism Mechanism Sensitivity Analysis.

The optstop package includes a conservatism parameter (default c=10) that
prevents premature stopping when observed performance is very low. When
the running estimate falls below `low_performance_threshold` (default 10%),
the system:
  (a) boosts the prior alpha by factor c (slower prior decay),
  (b) inflates the effective CI width by factor c (delays convergence),
  (c) tightens the stabilisation slope threshold by factor 1/c.

This analysis uses the full `optimal_stopping_posthoc` API (PyMC MCMC
hierarchical model) to sweep conservatism values across performance levels,
answering three reviewer questions:
  1. Does the mechanism activate at the correct threshold?
  2. What is the dose-response of c on stopping trial / efficiency / accuracy?
  3. Is the default c=10 justified vs alternatives?

Design:
  - Synthetic binary data: 100 items x 5 epochs = 500 trials per condition.
  - Performance levels: 1%, 5%, 10%, 15% (spanning the 10% threshold).
  - Conservatism values: 1, 5, 10, 50.
  - Full PyMC MCMC inference (1000 draws/tune) via optimal_stopping_posthoc.
  - 16 conditions total.

Outputs:
  results/matrix/conservatism_sensitivity.json  - full results
  results/matrix/figures/conservatism/*.png      - visualisations

Usage:
    python validate_conservatism.py                       # All 16 conditions
    python validate_conservatism.py --perf-levels 0.01 0.05
    python validate_conservatism.py --conservatism-values 1 10 50
    python validate_conservatism.py --skip-figures
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

DEFAULT_PERF_LEVELS = [0.01, 0.05, 0.10, 0.15]
DEFAULT_C_VALUES = [1, 5, 10, 50]

# Data generation
N_ITEMS = 200
N_EPOCHS = 10
N_TOTAL = N_ITEMS * N_EPOCHS  # 2000

# MCMC parameters
MCMC_DRAWS = 1000
MCMC_TUNE = 1000

# From UNIVERSAL_OPTSTOP_PARAMS
DELTA_CAP = UNIVERSAL_OPTSTOP_PARAMS["delta_cap"]           # 0.05
CRED_LEVEL = UNIVERSAL_OPTSTOP_PARAMS["cred_level"]         # 0.97
LOW_PERF_THRESHOLD = UNIVERSAL_OPTSTOP_PARAMS["low_performance_threshold"]  # 0.1


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def make_synthetic_df(
    perf_level: float,
    n_items: int = N_ITEMS,
    n_epochs: int = N_EPOCHS,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic binary eval DataFrame with known true performance."""
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
# Core: run one condition via optimal_stopping_posthoc
# ---------------------------------------------------------------------------

def run_condition(
    perf_level: float,
    conservatism_value: float,
    shuffle_seed: int = 42,
    data_seed: int = 42,
    mcmc_draws: int = MCMC_DRAWS,
    mcmc_tune: int = MCMC_TUNE,
) -> dict:
    """Run optimal_stopping_posthoc for one (perf_level, c) condition."""
    import optstop

    df = make_synthetic_df(perf_level, seed=data_seed)
    n_total = len(df)
    actual_perf = float(df["score"].mean())

    params = UNIVERSAL_OPTSTOP_PARAMS.copy()
    params["conservatism"] = conservatism_value
    params["low_performance_threshold"] = LOW_PERF_THRESHOLD
    params["draws"] = mcmc_draws
    params["tune"] = mcmc_tune

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
            "actual_perf": actual_perf,
            "error": str(e),
        }

    if not summaries or summaries[0].get("error"):
        error_msg = summaries[0]["error"] if summaries else "No summaries"
        return {
            "perf_level": perf_level,
            "conservatism": conservatism_value,
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
        ci_covers_true = float(theta_lo) <= perf_level <= float(theta_hi)

    return {
        "perf_level": perf_level,
        "conservatism": conservatism_value,
        "actual_perf": round(actual_perf, 6),
        "conservatism_active": perf_level < LOW_PERF_THRESHOLD,
        "stopped": stopped,
        "n_items_used": n_items_used,
        "avg_reps_per_item": round(float(avg_reps), 2) if avg_reps else None,
        "trials_used": trials_used,
        "n_total": n_total,
        "efficiency_pct": round((1 - trials_used / n_total) * 100, 2),
        "theta_estimate": round(float(theta_mid), 6) if theta_mid is not None else None,
        "theta_ci_low": round(float(theta_lo), 6) if theta_lo is not None else None,
        "theta_ci_high": round(float(theta_hi), 6) if theta_hi is not None else None,
        "theta_ci_width": round(float(theta_width), 6) if theta_width is not None else None,
        "ci_covers_true": ci_covers_true,
        "mean_fin_CI_slope": s.get("mean_fin_CI_slope_item"),
        "mean_fin_slope_slope": s.get("mean_fin_slope_slope_item"),
    }


# ---------------------------------------------------------------------------
# Run all conditions
# ---------------------------------------------------------------------------

def run_all(
    perf_levels: list[float],
    c_values: list[float],
    seed: int = 42,
    mcmc_draws: int = MCMC_DRAWS,
    mcmc_tune: int = MCMC_TUNE,
) -> dict:
    """Run all (perf_level, c) conditions."""
    results = {}
    total = len(perf_levels) * len(c_values)
    idx = 0

    for perf in perf_levels:
        perf_key = f"{perf:.4f}"
        results[perf_key] = {}

        for c_val in c_values:
            idx += 1
            c_key = f"c={c_val}"
            print(f"  [{idx}/{total}] perf={perf*100:.1f}%, c={c_val}...",
                  end=" ", flush=True)

            t0 = time.time()
            result = run_condition(
                perf_level=perf,
                conservatism_value=c_val,
                shuffle_seed=seed,
                data_seed=seed,
                mcmc_draws=mcmc_draws,
                mcmc_tune=mcmc_tune,
            )
            elapsed = time.time() - t0
            result["duration_sec"] = round(elapsed, 1)
            results[perf_key][c_key] = result

            if result.get("error"):
                print(f"ERROR: {result['error']} [{elapsed:.1f}s]")
            else:
                stopped = result.get("stopped", False)
                eff = result.get("efficiency_pct", 0)
                theta = result.get("theta_estimate", 0)
                active = "ACTIVE" if result.get("conservatism_active") else "inactive"
                print(f"{'STOPPED' if stopped else 'NO STOP'} "
                      f"eff={eff:.1f}% theta={theta:.4f} "
                      f"conserv={active} [{elapsed:.1f}s]")

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_dose_response(results: dict, perf_levels: list[float],
                       c_values: list[float], output_dir: Path):
    """3-panel dose-response: trials used, efficiency, theta estimate vs c."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colours = plt.cm.viridis(np.linspace(0, 0.9, len(perf_levels)))

    for panel_idx, (metric, ylabel, title) in enumerate([
        ("trials_used", "Trials used", "Trials Before Stopping"),
        ("efficiency_pct", "Efficiency (%)", "Efficiency (% Trials Saved)"),
        ("theta_estimate", "Theta estimate", "Posterior Theta Estimate"),
    ]):
        ax = axes[panel_idx]
        for i, perf in enumerate(perf_levels):
            perf_key = f"{perf:.4f}"
            label = f"{perf*100:.0f}%"
            vals = []
            c_plot = []
            for c_val in c_values:
                c_key = f"c={c_val}"
                if perf_key in results and c_key in results[perf_key]:
                    r = results[perf_key][c_key]
                    val = r.get(metric)
                    if val is not None and not r.get("error"):
                        vals.append(val)
                        c_plot.append(c_val)

            if vals:
                ax.plot(c_plot, vals, marker="o", color=colours[i],
                        label=label, linewidth=1.5, markersize=6)

                if metric == "theta_estimate":
                    ax.axhline(perf, color=colours[i], linestyle=":",
                               linewidth=0.8, alpha=0.5)

        ax.set_xlabel("Conservatism (c)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(title="True perf", fontsize=8, title_fontsize=9)
        ax.set_xscale("log")
        ax.set_xticks(c_values)
        ax.set_xticklabels([str(c) for c in c_values])

    plt.suptitle(
        "Conservatism Dose-Response (Full MCMC)\n"
        f"({N_ITEMS} items x {N_EPOCHS} epochs = {N_TOTAL} trials, "
        f"{MCMC_DRAWS} draws/{MCMC_TUNE} tune)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = output_dir / "dose_response.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_heatmap(results: dict, perf_levels: list[float],
                 c_values: list[float], output_dir: Path):
    """Heatmap of trials used and efficiency for each (perf, c) combination."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for panel_idx, (metric, title, cmap, fmt) in enumerate([
        ("trials_used", "Trials Used", "YlOrRd", "{:.0f}"),
        ("efficiency_pct", "Efficiency (% Trials Saved)", "YlGn", "{:.1f}"),
    ]):
        ax = axes[panel_idx]
        matrix = np.full((len(perf_levels), len(c_values)), np.nan)

        for i, perf in enumerate(perf_levels):
            perf_key = f"{perf:.4f}"
            for j, c_val in enumerate(c_values):
                c_key = f"c={c_val}"
                if perf_key in results and c_key in results[perf_key]:
                    r = results[perf_key][c_key]
                    if not r.get("error"):
                        val = r.get(metric)
                        if val is not None:
                            matrix[i, j] = val

        im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
        plt.colorbar(im, ax=ax)

        y_labels = [f"{p*100:.0f}%" for p in perf_levels]
        x_labels = [str(c) for c in c_values]
        ax.set_xticks(range(len(c_values)))
        ax.set_xticklabels(x_labels)
        ax.set_yticks(range(len(perf_levels)))
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Conservatism (c)")
        ax.set_ylabel("True performance")

        # Draw threshold line between 5% and 10%
        threshold_idx = None
        for i, perf in enumerate(perf_levels):
            if perf >= LOW_PERF_THRESHOLD:
                threshold_idx = i - 0.5
                break
        if threshold_idx is not None:
            ax.axhline(threshold_idx, color="white", linestyle="--", linewidth=2)
            ax.text(len(c_values) - 0.5, threshold_idx - 0.15,
                    f"threshold={LOW_PERF_THRESHOLD*100:.0f}%",
                    ha="right", va="bottom", color="white", fontsize=7,
                    fontweight="bold")

        for i in range(len(perf_levels)):
            for j in range(len(c_values)):
                val = matrix[i, j]
                if not np.isnan(val):
                    mid = (np.nanmax(matrix) + np.nanmin(matrix)) / 2
                    colour = "white" if val > mid else "black"
                    ax.text(j, i, fmt.format(val), ha="center", va="center",
                            fontsize=8, color=colour, fontweight="bold")

        ax.set_title(title, fontweight="bold")

    plt.suptitle(
        "Conservatism Sensitivity Analysis (Full MCMC)\n"
        f"({N_ITEMS} items x {N_EPOCHS} epochs; "
        f"white dashed = activation threshold)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = output_dir / "heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_summary_table(results: dict, perf_levels: list[float],
                       c_values: list[float], output_dir: Path):
    """Summary table figure."""
    rows = []
    for perf in perf_levels:
        perf_key = f"{perf:.4f}"
        active = "Yes" if perf < LOW_PERF_THRESHOLD else "No"
        for c_val in c_values:
            c_key = f"c={c_val}"
            if perf_key in results and c_key in results[perf_key]:
                r = results[perf_key][c_key]
                if r.get("error"):
                    rows.append([
                        f"{perf*100:.0f}%", active, str(c_val),
                        "ERROR", "--", "--", "--", "--",
                    ])
                else:
                    rows.append([
                        f"{perf*100:.0f}%",
                        active,
                        str(c_val),
                        "Yes" if r.get("stopped") else "No",
                        f"{r.get('trials_used', 0):.0f}",
                        f"{r.get('efficiency_pct', 0):.1f}%",
                        f"{r.get('theta_estimate', 0):.4f}" if r.get("theta_estimate") is not None else "--",
                        f"[{r.get('theta_ci_low', 0):.3f}, {r.get('theta_ci_high', 0):.3f}]"
                            if r.get("theta_ci_low") is not None else "--",
                    ])

    if not rows:
        return

    col_labels = ["Perf", "Conserv.\nactive?", "c", "Stopped?",
                  "Trials", "Efficiency", "Theta", "95% HDI"]

    fig, ax = plt.subplots(figsize=(16, max(4, len(rows) * 0.35 + 1.5)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.3)

    n_cols = len(col_labels)
    for j in range(n_cols):
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
        for j in range(n_cols):
            table[i, j].set_facecolor(colour)

    fig.suptitle(
        "Conservatism Sensitivity Analysis: Summary (Full MCMC)\n"
        f"({N_ITEMS} items x {N_EPOCHS} epochs = {N_TOTAL} trials, "
        f"{MCMC_DRAWS} draws/{MCMC_TUNE} tune)",
        fontsize=11, fontweight="bold", y=0.99,
    )
    out = output_dir / "summary_table.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Conservatism mechanism sensitivity analysis (Validation Gap 1)")
    parser.add_argument("--perf-levels", nargs="+", type=float, default=DEFAULT_PERF_LEVELS,
                        help=f"Performance levels (default: {DEFAULT_PERF_LEVELS})")
    parser.add_argument("--conservatism-values", nargs="+", type=float, default=DEFAULT_C_VALUES,
                        help=f"Conservatism values (default: {DEFAULT_C_VALUES})")
    parser.add_argument("--mcmc-draws", type=int, default=MCMC_DRAWS,
                        help=f"MCMC draws (default: {MCMC_DRAWS})")
    parser.add_argument("--mcmc-tune", type=int, default=MCMC_TUNE,
                        help=f"MCMC tuning steps (default: {MCMC_TUNE})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--skip-figures", action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory for figures (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    perf_levels = sorted(args.perf_levels)
    c_values = sorted(args.conservatism_values)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_conditions = len(perf_levels) * len(c_values)

    print("=" * 70)
    print("Conservatism Sensitivity Analysis (Validation Gap 1)")
    print(f"  Method: Full PyMC MCMC via optimal_stopping_posthoc")
    print(f"  Performance levels: {[f'{p*100:.0f}%' for p in perf_levels]}")
    print(f"  Conservatism values: {c_values}")
    print(f"  low_performance_threshold: {LOW_PERF_THRESHOLD} ({LOW_PERF_THRESHOLD*100:.0f}%)")
    print(f"  delta_cap: {DELTA_CAP}")
    print(f"  Data: {N_ITEMS} items x {N_EPOCHS} epochs = {N_TOTAL} trials")
    print(f"  MCMC: {args.mcmc_draws} draws / {args.mcmc_tune} tune")
    print(f"  Conditions: {total_conditions}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    t0 = time.time()
    results = run_all(
        perf_levels=perf_levels,
        c_values=c_values,
        seed=args.seed,
        mcmc_draws=args.mcmc_draws,
        mcmc_tune=args.mcmc_tune,
    )
    total_time = time.time() - t0

    # --- Save JSON ---
    output_json = RESULTS_DIR / "conservatism_sensitivity.json"
    json_output = {
        "config": {
            "method": "optimal_stopping_posthoc (full PyMC MCMC)",
            "perf_levels": perf_levels,
            "conservatism_values": c_values,
            "n_items": N_ITEMS,
            "n_epochs": N_EPOCHS,
            "n_total": N_TOTAL,
            "mcmc_draws": args.mcmc_draws,
            "mcmc_tune": args.mcmc_tune,
            "low_performance_threshold": LOW_PERF_THRESHOLD,
            "delta_cap": DELTA_CAP,
            "cred_level": CRED_LEVEL,
            "seed": args.seed,
            "total_duration_sec": round(total_time, 1),
        },
        "results": results,
    }

    with open(output_json, "w") as f:
        json.dump(json_output, f, indent=2, default=str)
    print(f"\nSaved results: {output_json}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"SUMMARY ({total_time:.1f}s = {total_time/60:.1f} min)")
    print(f"{'='*70}")

    header = f"  {'Perf':<8} {'Active':<8}"
    for c_val in c_values:
        header += f" | c={c_val:>3}: stop/eff%/theta"
    print(header)
    print(f"  {'-'*8} {'-'*8}" + "".join([f"-+-{'-'*24}" for _ in c_values]))

    for perf in perf_levels:
        perf_key = f"{perf:.4f}"
        active = "YES" if perf < LOW_PERF_THRESHOLD else "no"
        line = f"  {perf*100:>5.0f}%   {active:<8}"
        for c_val in c_values:
            c_key = f"c={c_val}"
            if perf_key in results and c_key in results[perf_key]:
                r = results[perf_key][c_key]
                if r.get("error"):
                    line += f" | ERROR                   "
                else:
                    stopped = "Y" if r.get("stopped") else "N"
                    eff = r.get("efficiency_pct", 0)
                    theta = r.get("theta_estimate", 0)
                    line += f" | {stopped} {eff:>5.1f}% {theta:>.4f}      "
            else:
                line += f" |           --            "
        print(line)

    # --- Figures ---
    if not args.skip_figures:
        print(f"\nGenerating figures...")
        plot_dose_response(results, perf_levels, c_values, output_dir)
        plot_heatmap(results, perf_levels, c_values, output_dir)
        plot_summary_table(results, perf_levels, c_values, output_dir)

    print(f"\nDone. Total time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
