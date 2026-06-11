#!/usr/bin/env python3
"""
validate_stabilisation.py

Validation Gap 2: CI Stabilisation as Primary Stopping Trigger.

The optstop package has a stabilisation fallback -- when the CI width slope
plateaus (|slope| <= CI_delta) before reaching delta_cap, the system stops
rather than running indefinitely. No matrix cell triggered this pathway
in production; all stopped via CI convergence or entropy.

This analysis uses the full `optimal_stopping_posthoc` API (PyMC MCMC
hierarchical model) with tight delta_cap values that the CI cannot reach,
forcing stabilisation to be the only viable stop mechanism. This answers
a reviewer question: "Is the stabilisation pathway functional or dead code?"

Design:
  - Synthetic binary data: 100 items x 5 epochs = 500 trials per condition.
  - Performance levels: 20%, 50%, 80%.
  - delta_cap values: 0.05 (normal), 0.01 (tight), 0.005 (very tight).
  - CI_delta: 0.001 (increased from default 0.0001 to make stabilisation
    more likely to fire within the data budget).
  - stab_window: 10 (decreased from default 15).
  - Full PyMC MCMC inference (1000 draws/tune) via optimal_stopping_posthoc.
  - 9 conditions total.

Outputs:
  results/matrix/stabilisation_test.json         - full results
  results/matrix/figures/stabilisation/*.png      - visualisations

Usage:
    python validate_stabilisation.py                     # All 9 conditions
    python validate_stabilisation.py --perf-levels 0.2 0.5
    python validate_stabilisation.py --delta-caps 0.01 0.005
    python validate_stabilisation.py --skip-figures
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
OUTPUT_DIR = RESULTS_DIR / "figures" / "stabilisation"

DEFAULT_PERF_LEVELS = [0.2, 0.5, 0.8]
DEFAULT_DELTA_CAPS = [0.05, 0.01, 0.005]

# Data generation
N_ITEMS = 200
N_EPOCHS = 10
N_TOTAL = N_ITEMS * N_EPOCHS  # 2000

# MCMC parameters
MCMC_DRAWS = 1000
MCMC_TUNE = 1000

# Stabilisation-specific overrides (relaxed to demonstrate the mechanism)
STAB_WINDOW = 10      # decreased from default 15
CI_DELTA = 0.001      # increased from default 0.0001

# From UNIVERSAL_OPTSTOP_PARAMS
CRED_LEVEL = UNIVERSAL_OPTSTOP_PARAMS["cred_level"]           # 0.97
CONSERVATISM = UNIVERSAL_OPTSTOP_PARAMS["conservatism"]        # 10
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
    delta_cap: float,
    shuffle_seed: int = 42,
    data_seed: int = 42,
    mcmc_draws: int = MCMC_DRAWS,
    mcmc_tune: int = MCMC_TUNE,
) -> dict:
    """Run optimal_stopping_posthoc for one (perf_level, delta_cap) condition."""
    import optstop

    df = make_synthetic_df(perf_level, seed=data_seed)
    n_total = len(df)
    actual_perf = float(df["score"].mean())

    params = UNIVERSAL_OPTSTOP_PARAMS.copy()
    params["delta_cap"] = delta_cap
    params["stab_window"] = STAB_WINDOW
    params["CI_delta"] = CI_DELTA
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
            "delta_cap": delta_cap,
            "actual_perf": actual_perf,
            "error": str(e),
        }

    if not summaries or summaries[0].get("error"):
        error_msg = summaries[0]["error"] if summaries else "No summaries"
        return {
            "perf_level": perf_level,
            "delta_cap": delta_cap,
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

    # Infer stop mechanism from CI width vs delta_cap
    if stopped and theta_width is not None:
        if theta_width < delta_cap:
            stop_mechanism = "ci_convergence"
        else:
            stop_mechanism = "stabilisation"
    elif not stopped:
        stop_mechanism = "exhausted"
    else:
        stop_mechanism = "unknown"

    return {
        "perf_level": perf_level,
        "delta_cap": delta_cap,
        "actual_perf": round(actual_perf, 6),
        "stab_window": STAB_WINDOW,
        "CI_delta": CI_DELTA,
        "stopped": stopped,
        "stop_mechanism": stop_mechanism,
        "n_items_used": n_items_used,
        "avg_reps_per_item": round(float(avg_reps), 2) if avg_reps else None,
        "trials_used": trials_used,
        "n_total": n_total,
        "efficiency_pct": round((1 - trials_used / n_total) * 100, 2),
        "theta_estimate": round(float(theta_mid), 6) if theta_mid is not None else None,
        "theta_ci_low": round(float(theta_lo), 6) if theta_lo is not None else None,
        "theta_ci_high": round(float(theta_hi), 6) if theta_hi is not None else None,
        "theta_ci_width": round(float(theta_width), 6) if theta_width is not None else None,
        "mean_fin_CI_slope": s.get("mean_fin_CI_slope_item"),
        "mean_fin_slope_slope": s.get("mean_fin_slope_slope_item"),
    }


# ---------------------------------------------------------------------------
# Run all conditions
# ---------------------------------------------------------------------------

def run_all(
    perf_levels: list[float],
    delta_caps: list[float],
    seed: int = 42,
    mcmc_draws: int = MCMC_DRAWS,
    mcmc_tune: int = MCMC_TUNE,
) -> dict:
    """Run all (perf_level, delta_cap) conditions."""
    results = {}
    total = len(perf_levels) * len(delta_caps)
    idx = 0

    for perf in perf_levels:
        perf_key = f"{perf:.2f}"
        results[perf_key] = {}

        for dc in delta_caps:
            idx += 1
            dc_key = f"dc={dc}"
            print(f"  [{idx}/{total}] perf={perf*100:.0f}%, delta_cap={dc}...",
                  end=" ", flush=True)

            t0 = time.time()
            result = run_condition(
                perf_level=perf,
                delta_cap=dc,
                shuffle_seed=seed,
                data_seed=seed,
                mcmc_draws=mcmc_draws,
                mcmc_tune=mcmc_tune,
            )
            elapsed = time.time() - t0
            result["duration_sec"] = round(elapsed, 1)
            results[perf_key][dc_key] = result

            if result.get("error"):
                print(f"ERROR: {result['error']} [{elapsed:.1f}s]")
            else:
                mech = result.get("stop_mechanism", "?")
                eff = result.get("efficiency_pct", 0)
                theta = result.get("theta_estimate", 0)
                width = result.get("theta_ci_width", 0)
                print(f"{mech} eff={eff:.1f}% theta={theta:.4f} "
                      f"CI_width={width:.4f} [{elapsed:.1f}s]")

    return results


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_mechanism_summary(results: dict, perf_levels: list[float],
                           delta_caps: list[float], output_dir: Path):
    """Bar chart showing stop mechanism for each condition."""
    fig, ax = plt.subplots(figsize=(10, 6))

    n_groups = len(perf_levels)
    n_bars = len(delta_caps)
    bar_width = 0.8 / n_bars
    colours_ci = "#4C72B0"
    colours_stab = "#DD8452"
    colours_exhaust = "#CCCCCC"

    for j, dc in enumerate(delta_caps):
        dc_key = f"dc={dc}"
        for i, perf in enumerate(perf_levels):
            perf_key = f"{perf:.2f}"
            if perf_key in results and dc_key in results[perf_key]:
                r = results[perf_key][dc_key]
                mech = r.get("stop_mechanism", "exhausted")
                eff = r.get("efficiency_pct", 0)
                colour = (colours_ci if mech == "ci_convergence"
                          else colours_stab if mech == "stabilisation"
                          else colours_exhaust)
                x = i + j * bar_width
                bar = ax.bar(x, eff, bar_width, color=colour,
                             edgecolor="black", linewidth=0.5)
                ax.text(x, eff + 1, mech[:4].upper(), ha="center",
                        fontsize=6, fontweight="bold")

    ax.set_xlabel("True performance level")
    ax.set_ylabel("Efficiency (% trials saved)")
    ax.set_title("Stop Mechanism by Condition\n"
                 f"(CI_delta={CI_DELTA}, stab_window={STAB_WINDOW})",
                 fontweight="bold")
    ax.set_xticks([i + (n_bars - 1) * bar_width / 2 for i in range(n_groups)])
    ax.set_xticklabels([f"{p*100:.0f}%" for p in perf_levels])

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colours_ci, edgecolor="black", label="CI convergence"),
        Patch(facecolor=colours_stab, edgecolor="black", label="Stabilisation"),
        Patch(facecolor=colours_exhaust, edgecolor="black", label="Exhausted"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    # Add delta_cap labels
    for j, dc in enumerate(delta_caps):
        ax.text(j * bar_width, -8, f"dc={dc}", fontsize=6,
                ha="center", rotation=45)

    plt.tight_layout()
    out = output_dir / "mechanism_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_heatmap(results: dict, perf_levels: list[float],
                 delta_caps: list[float], output_dir: Path):
    """Heatmap showing efficiency and stop mechanism for each condition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for panel_idx, (metric, title, cmap) in enumerate([
        ("efficiency_pct", "Efficiency (% Trials Saved)", "YlGn"),
        ("theta_ci_width", "Final CI Width", "YlOrRd_r"),
    ]):
        ax = axes[panel_idx]
        matrix = np.full((len(perf_levels), len(delta_caps)), np.nan)

        for i, perf in enumerate(perf_levels):
            perf_key = f"{perf:.2f}"
            for j, dc in enumerate(delta_caps):
                dc_key = f"dc={dc}"
                if perf_key in results and dc_key in results[perf_key]:
                    r = results[perf_key][dc_key]
                    if not r.get("error"):
                        val = r.get(metric)
                        if val is not None:
                            matrix[i, j] = val

        im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest")
        plt.colorbar(im, ax=ax)

        ax.set_xticks(range(len(delta_caps)))
        ax.set_xticklabels([str(dc) for dc in delta_caps])
        ax.set_yticks(range(len(perf_levels)))
        ax.set_yticklabels([f"{p*100:.0f}%" for p in perf_levels])
        ax.set_xlabel("delta_cap")
        ax.set_ylabel("True performance")

        for i in range(len(perf_levels)):
            for j in range(len(delta_caps)):
                val = matrix[i, j]
                if not np.isnan(val):
                    mid = (np.nanmax(matrix) + np.nanmin(matrix)) / 2
                    colour = "white" if val > mid else "black"
                    fmt = "{:.1f}%" if "Efficiency" in title else "{:.4f}"
                    # Also annotate stop mechanism
                    perf_key = f"{perf_levels[i]:.2f}"
                    dc_key = f"dc={delta_caps[j]}"
                    mech = ""
                    if perf_key in results and dc_key in results[perf_key]:
                        mech = results[perf_key][dc_key].get("stop_mechanism", "")[0:4].upper()
                    ax.text(j, i, f"{fmt.format(val)}\n({mech})", ha="center",
                            va="center", fontsize=7, color=colour, fontweight="bold")

        ax.set_title(title, fontweight="bold")

    plt.suptitle(
        "CI Stabilisation Trigger Analysis (Full MCMC)\n"
        f"({N_ITEMS} items x {N_EPOCHS} epochs, "
        f"CI_delta={CI_DELTA}, stab_window={STAB_WINDOW})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = output_dir / "heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_summary_table(results: dict, perf_levels: list[float],
                       delta_caps: list[float], output_dir: Path):
    """Summary table figure."""
    rows = []
    for perf in perf_levels:
        perf_key = f"{perf:.2f}"
        for dc in delta_caps:
            dc_key = f"dc={dc}"
            if perf_key in results and dc_key in results[perf_key]:
                r = results[perf_key][dc_key]
                if r.get("error"):
                    rows.append([
                        f"{perf*100:.0f}%", str(dc), "ERROR",
                        "--", "--", "--", "--", "--",
                    ])
                else:
                    rows.append([
                        f"{perf*100:.0f}%",
                        str(dc),
                        r.get("stop_mechanism", "?"),
                        "Yes" if r.get("stopped") else "No",
                        f"{r.get('trials_used', 0):.0f}",
                        f"{r.get('efficiency_pct', 0):.1f}%",
                        f"{r.get('theta_estimate', 0):.4f}" if r.get("theta_estimate") is not None else "--",
                        f"{r.get('theta_ci_width', 0):.4f}" if r.get("theta_ci_width") is not None else "--",
                    ])

    if not rows:
        return

    col_labels = ["Perf", "delta_cap", "Mechanism", "Stopped?",
                  "Trials", "Efficiency", "Theta", "CI width"]

    fig, ax = plt.subplots(figsize=(16, max(4, len(rows) * 0.4 + 1.5)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

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
        "CI Stabilisation Trigger: Summary (Full MCMC)\n"
        f"({N_ITEMS} items x {N_EPOCHS} epochs = {N_TOTAL} trials, "
        f"CI_delta={CI_DELTA}, stab_window={STAB_WINDOW})",
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
        description="CI stabilisation trigger test (Validation Gap 2)")
    parser.add_argument("--perf-levels", nargs="+", type=float, default=DEFAULT_PERF_LEVELS,
                        help=f"Performance levels (default: {DEFAULT_PERF_LEVELS})")
    parser.add_argument("--delta-caps", nargs="+", type=float, default=DEFAULT_DELTA_CAPS,
                        help=f"delta_cap values (default: {DEFAULT_DELTA_CAPS})")
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
    delta_caps = sorted(args.delta_caps, reverse=True)  # Largest first
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_conditions = len(perf_levels) * len(delta_caps)

    print("=" * 70)
    print("CI Stabilisation Trigger Test (Validation Gap 2)")
    print(f"  Method: Full PyMC MCMC via optimal_stopping_posthoc")
    print(f"  Performance levels: {[f'{p*100:.0f}%' for p in perf_levels]}")
    print(f"  delta_cap values: {delta_caps}")
    print(f"  stab_window: {STAB_WINDOW} (default: {UNIVERSAL_OPTSTOP_PARAMS['stab_window']})")
    print(f"  CI_delta: {CI_DELTA} (default: {UNIVERSAL_OPTSTOP_PARAMS['CI_delta']})")
    print(f"  Data: {N_ITEMS} items x {N_EPOCHS} epochs = {N_TOTAL} trials")
    print(f"  MCMC: {args.mcmc_draws} draws / {args.mcmc_tune} tune")
    print(f"  Conditions: {total_conditions}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    t0 = time.time()
    results = run_all(
        perf_levels=perf_levels,
        delta_caps=delta_caps,
        seed=args.seed,
        mcmc_draws=args.mcmc_draws,
        mcmc_tune=args.mcmc_tune,
    )
    total_time = time.time() - t0

    # --- Save JSON ---
    output_json = RESULTS_DIR / "stabilisation_test.json"
    json_output = {
        "config": {
            "method": "optimal_stopping_posthoc (full PyMC MCMC)",
            "perf_levels": perf_levels,
            "delta_caps": delta_caps,
            "n_items": N_ITEMS,
            "n_epochs": N_EPOCHS,
            "n_total": N_TOTAL,
            "mcmc_draws": args.mcmc_draws,
            "mcmc_tune": args.mcmc_tune,
            "stab_window": STAB_WINDOW,
            "CI_delta": CI_DELTA,
            "conservatism": CONSERVATISM,
            "low_perf_threshold": LOW_PERF_THRESHOLD,
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

    header = f"  {'Perf':<8}"
    for dc in delta_caps:
        header += f" | dc={dc}: mech/eff%/theta"
    print(header)
    print(f"  {'-'*8}" + "".join([f"-+-{'-'*26}" for _ in delta_caps]))

    for perf in perf_levels:
        perf_key = f"{perf:.2f}"
        line = f"  {perf*100:>5.0f}%  "
        for dc in delta_caps:
            dc_key = f"dc={dc}"
            if perf_key in results and dc_key in results[perf_key]:
                r = results[perf_key][dc_key]
                if r.get("error"):
                    line += f" | ERROR                     "
                else:
                    mech = r.get("stop_mechanism", "?")[:4].upper()
                    eff = r.get("efficiency_pct", 0)
                    theta = r.get("theta_estimate", 0)
                    line += f" | {mech:<4} {eff:>5.1f}% {theta:>.4f}     "
            else:
                line += f" |           --              "
        print(line)

    # --- Figures ---
    if not args.skip_figures:
        print(f"\nGenerating figures...")
        plot_mechanism_summary(results, perf_levels, delta_caps, output_dir)
        plot_heatmap(results, perf_levels, delta_caps, output_dir)
        plot_summary_table(results, perf_levels, delta_caps, output_dir)

    print(f"\nDone. Total time: {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == "__main__":
    main()
