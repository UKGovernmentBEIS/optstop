#!/usr/bin/env python3
"""
analyse_bias_decomposition.py

Post-hoc bias decomposition for consistency analysis results.

For each cell and rep, replays the shuffle order to determine which trials
were seen before stopping, then decomposes total bias into:
    total_bias = subset_selection_bias + model_bias
where:
    subset_selection_bias = raw_mean_stopped - raw_mean_full
    model_bias            = theta_estimate  - raw_mean_stopped

This tells us whether the systematic negative bias (~0.01) seen in
consistency results comes from:
    (a) The stopped sample being unrepresentative (subset selection), or
    (b) The Bayesian model / HDI midpoint estimator (model bias)

No re-running of optstop is needed — we replay shuffles externally.

Usage:
    python analyse_bias_decomposition.py
    python analyse_bias_decomposition.py --cells low_binary mid_binary
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse data loading from consistency script
from analyse_posthoc_consistency import (
    load_cell_data,
    get_full_data_reference,
    CELL_ORDER,
    RESULTS_DIR,
)

OUTPUT_DIR = Path("results/validation/consistency")
CONSISTENCY_JSON = OUTPUT_DIR / "posthoc_consistency.json"

# ---------------------------------------------------------------------------
# Shuffle replay
# ---------------------------------------------------------------------------

def replay_shuffle(df: pd.DataFrame, shuffle_seed: int) -> list:
    """Replay optstop's shuffle to get the item processing order.

    Mirrors rule.py lines 3921-3937:
        rng = np.random.RandomState(shuffle_seed)
        df = df.groupby('grouping').apply(
            lambda g: g.sample(frac=1, random_state=rng)
        ).reset_index(drop=True)

    Then item_ids = list(df['sample_id_num'].unique()) preserving
    first-appearance order from the shuffled dataframe.

    Returns: ordered list of sample_id values in processing order.
    """
    rng = np.random.RandomState(shuffle_seed)

    # Replicate optstop's groupby-shuffle
    df_shuffled = df.groupby(
        ['model', 'task'], group_keys=False
    ).apply(
        lambda g: g.sample(frac=1, random_state=rng)
    ).reset_index(drop=True)

    # Item processing order = first-appearance order of sample_id
    # (mirrors: item_ids = list(df_part['sample_id_num'].unique()))
    seen = set()
    item_order = []
    for sid in df_shuffled['sample_id']:
        if sid not in seen:
            seen.add(sid)
            item_order.append(sid)

    return item_order


def compute_raw_mean_stopped(
    df: pd.DataFrame,
    item_order: list,
    n_items_used: int,
    avg_reps_per_item: float,
    cell_config: dict,
) -> dict:
    """Compute the raw empirical mean of the stopped subset.

    In epoch-interleaved mode:
    - Epoch 1: process all items in item_order, each gets 1 rep
    - Epoch 2: process items in item_order, each gets 1 more rep
    - etc.
    - Stopping occurs after total_trials_used trials

    Returns dict with raw_mean_stopped, n_trials_stopped, details.
    """
    mode = cell_config.get("scoring_mode", "binary")
    ordinal_max_score = cell_config.get("ordinal_max_score", 10)

    total_trials_target = round(n_items_used * avg_reps_per_item)
    n_total_items = len(item_order)
    epochs = sorted(df['epoch'].unique())

    # Build a lookup: (sample_id, epoch) -> score
    score_lookup = {}
    for _, row in df.iterrows():
        key = (row['sample_id'], row['epoch'])
        score_lookup[key] = row['score']

    # Simulate epoch-interleaved processing
    collected_scores = []
    trial_count = 0

    for epoch in epochs:
        if trial_count >= total_trials_target:
            break
        for item_id in item_order:
            if trial_count >= total_trials_target:
                break
            key = (item_id, epoch)
            if key in score_lookup:
                collected_scores.append(score_lookup[key])
                trial_count += 1

    # Compute raw mean in the correct scale
    if not collected_scores:
        return {"raw_mean_stopped": None, "n_trials": 0}

    scores_arr = np.array(collected_scores)
    if mode == "binary":
        raw_mean = float(scores_arr.mean())
    else:
        # ordinal and continuous: normalise by ordinal_max_score
        raw_mean = float(scores_arr.mean()) / ordinal_max_score

    return {
        "raw_mean_stopped": raw_mean,
        "n_trials": len(collected_scores),
        "n_unique_items_seen": len(set(
            item_order[:min(len(item_order), total_trials_target)]
        )),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_decomposition(cells: list[str] | None = None):
    """Run bias decomposition for all cells."""

    # Load consistency results
    with open(CONSISTENCY_JSON) as f:
        consistency = json.load(f)

    if cells is None:
        cells = [c for c in CELL_ORDER if c in consistency["cells"]]

    all_results = {}

    for cell_id in cells:
        print(f"\n{'='*60}")
        print(f"Cell: {cell_id}")
        print(f"{'='*60}")

        cell_data = consistency["cells"][cell_id]
        reps = cell_data["reps"]
        theta_full_data = cell_data["summary"]["theta_full_data"]

        # Load raw data
        try:
            df, cell_config, _ = load_cell_data(cell_id)
        except Exception as e:
            print(f"  ERROR loading data: {e}")
            continue

        # Verify our full-data reference matches
        our_full_ref = get_full_data_reference(df, cell_config)
        print(f"  Full-data reference: {theta_full_data:.6f} (stored), {our_full_ref:.6f} (recomputed)")

        decomp_results = []

        for rep in reps:
            if rep.get("error"):
                continue

            seed = rep["shuffle_seed"]
            theta_est = rep["theta_estimate"]
            n_items_used = rep["n_items_used"]
            avg_reps = rep["avg_reps_per_item"]

            if theta_est is None:
                continue

            # Replay shuffle
            item_order = replay_shuffle(df, seed)

            # Compute raw mean of stopped subset
            stopped_info = compute_raw_mean_stopped(
                df, item_order, n_items_used, avg_reps, cell_config
            )

            raw_mean_stopped = stopped_info["raw_mean_stopped"]
            if raw_mean_stopped is None:
                continue

            # Decompose bias
            total_bias = theta_est - theta_full_data
            subset_selection_bias = raw_mean_stopped - theta_full_data
            model_bias = theta_est - raw_mean_stopped

            decomp_results.append({
                "shuffle_seed": seed,
                "theta_estimate": theta_est,
                "theta_full_data": theta_full_data,
                "raw_mean_stopped": round(raw_mean_stopped, 6),
                "total_bias": round(total_bias, 6),
                "subset_selection_bias": round(subset_selection_bias, 6),
                "model_bias": round(model_bias, 6),
                "n_trials_stopped": stopped_info["n_trials"],
                "n_items_used": n_items_used,
                "avg_reps_per_item": avg_reps,
            })

        if not decomp_results:
            print("  No valid reps")
            continue

        # Summary statistics
        total_biases = [r["total_bias"] for r in decomp_results]
        subset_biases = [r["subset_selection_bias"] for r in decomp_results]
        model_biases = [r["model_bias"] for r in decomp_results]

        summary = {
            "cell_id": cell_id,
            "n_reps": len(decomp_results),
            "theta_full_data": theta_full_data,
            "total_bias_mean": round(np.mean(total_biases), 6),
            "total_bias_std": round(np.std(total_biases, ddof=1), 6),
            "subset_selection_bias_mean": round(np.mean(subset_biases), 6),
            "subset_selection_bias_std": round(np.std(subset_biases, ddof=1), 6),
            "model_bias_mean": round(np.mean(model_biases), 6),
            "model_bias_std": round(np.std(model_biases, ddof=1), 6),
            # Proportion of total bias explained by each component
            "pct_subset_selection": round(
                abs(np.mean(subset_biases)) / abs(np.mean(total_biases)) * 100, 1
            ) if abs(np.mean(total_biases)) > 1e-8 else None,
            "pct_model_bias": round(
                abs(np.mean(model_biases)) / abs(np.mean(total_biases)) * 100, 1
            ) if abs(np.mean(total_biases)) > 1e-8 else None,
        }

        print(f"\n  Bias Decomposition (n={summary['n_reps']} reps):")
        print(f"  {'Component':<25s} {'Mean':>10s} {'Std':>10s}")
        print(f"  {'-'*45}")
        print(f"  {'Total bias':<25s} {summary['total_bias_mean']:>+10.4f} {summary['total_bias_std']:>10.4f}")
        print(f"  {'Subset selection':<25s} {summary['subset_selection_bias_mean']:>+10.4f} {summary['subset_selection_bias_std']:>10.4f}")
        print(f"  {'Model (HDI midpoint)':<25s} {summary['model_bias_mean']:>+10.4f} {summary['model_bias_std']:>10.4f}")
        if summary["pct_subset_selection"] is not None:
            print(f"\n  Proportion: {summary['pct_subset_selection']:.0f}% subset selection, "
                  f"{summary['pct_model_bias']:.0f}% model bias")

        all_results[cell_id] = {
            "reps": decomp_results,
            "summary": summary,
        }

    return all_results


def plot_decomposition(all_results: dict, output_dir: Path):
    """Create visualisation of bias decomposition across cells."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = [c for c in CELL_ORDER if c in all_results]
    if not cells:
        return

    # --- Figure 1: Stacked bar chart of bias components ---
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(cells))
    width = 0.35

    subset_means = [all_results[c]["summary"]["subset_selection_bias_mean"] for c in cells]
    model_means = [all_results[c]["summary"]["model_bias_mean"] for c in cells]
    subset_stds = [all_results[c]["summary"]["subset_selection_bias_std"] for c in cells]
    model_stds = [all_results[c]["summary"]["model_bias_std"] for c in cells]

    bars1 = ax.bar(x - width/2, subset_means, width, yerr=subset_stds,
                   label='Subset selection bias', color='#2196F3', alpha=0.8, capsize=3)
    bars2 = ax.bar(x + width/2, model_means, width, yerr=model_stds,
                   label='Model bias (HDI midpoint)', color='#FF9800', alpha=0.8, capsize=3)

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel('Cell')
    ax.set_ylabel('Bias (theta scale)')
    ax.set_title('Bias Decomposition: Subset Selection vs Model (HDI Midpoint)')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in cells], fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "bias_decomposition_bars.png", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {output_dir / 'bias_decomposition_bars.png'}")

    # --- Figure 2: Per-rep scatter of subset vs model bias ---
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    grid_map = {
        "low_binary": (0,0), "mid_binary": (1,0), "high_binary": (2,0),
        "low_ordinal": (0,1), "mid_ordinal": (1,1), "high_ordinal": (2,1),
        "low_continuous": (0,2), "mid_continuous": (1,2), "high_continuous": (2,2),
    }

    for cell_id in cells:
        if cell_id not in grid_map:
            continue
        r, c = grid_map[cell_id]
        ax = axes[r, c]

        reps = all_results[cell_id]["reps"]
        ss = [rep["subset_selection_bias"] for rep in reps]
        mb = [rep["model_bias"] for rep in reps]

        ax.scatter(ss, mb, alpha=0.6, s=30, color='#4CAF50')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

        # Add diagonal: total_bias = ss + mb
        lims = [min(min(ss), min(mb)) - 0.005, max(max(ss), max(mb)) + 0.005]
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_title(cell_id.replace('_', ' ').title(), fontsize=9)
        ax.set_xlabel('Subset sel. bias', fontsize=7)
        ax.set_ylabel('Model bias', fontsize=7)
        ax.tick_params(labelsize=7)

    plt.suptitle('Per-Rep Bias Decomposition: Subset Selection vs Model Bias', fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "bias_decomposition_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'bias_decomposition_scatter.png'}")

    # --- Figure 3: Summary table ---
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')

    headers = ['Cell', 'Total Bias', 'Subset Sel.', 'Model Bias', '% Subset', '% Model']
    table_data = []
    for cell_id in cells:
        s = all_results[cell_id]["summary"]
        table_data.append([
            cell_id,
            f"{s['total_bias_mean']:+.4f} ± {s['total_bias_std']:.4f}",
            f"{s['subset_selection_bias_mean']:+.4f} ± {s['subset_selection_bias_std']:.4f}",
            f"{s['model_bias_mean']:+.4f} ± {s['model_bias_std']:.4f}",
            f"{s['pct_subset_selection']:.0f}%" if s['pct_subset_selection'] is not None else "—",
            f"{s['pct_model_bias']:.0f}%" if s['pct_model_bias'] is not None else "—",
        ])

    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#E0E0E0']*len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    plt.title('Bias Decomposition Summary', fontsize=11, pad=20)
    plt.tight_layout()
    fig.savefig(output_dir / "bias_decomposition_table.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'bias_decomposition_table.png'}")


def main():
    parser = argparse.ArgumentParser(description="Bias decomposition for consistency analysis")
    parser.add_argument("--cells", nargs="+", help="Specific cells to analyse")
    args = parser.parse_args()

    if not CONSISTENCY_JSON.exists():
        print(f"ERROR: {CONSISTENCY_JSON} not found. Run analyse_posthoc_consistency.py first.")
        sys.exit(1)

    all_results = run_decomposition(cells=args.cells)

    if all_results:
        # Save results
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "bias_decomposition.json"
        with open(out_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved results: {out_path}")

        # Generate figures
        plot_decomposition(all_results, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
