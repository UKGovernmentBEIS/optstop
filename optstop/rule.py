"""
Adaptive optimal stopping rule algorithms for efficient data collection and analysis.

This module provides both post-hoc (batch) and live (incremental) optimal stopping algorithms.

The functions use flexible column mapping - users specify their own column names for:
- grouping_columns: Column(s) for grouping (e.g., subject, task)
- sample_id_column: Column for sample ID (e.g., item_id)
- epoch_column: Column for epoch/trial (e.g., trial_num)
- score_column: Column for score (e.g., score, accuracy)

The functions internally create numeric versions for processing.
"""

import pandas as pd
import numpy as np
import logging
import concurrent.futures
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import beta
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
import contextlib
import io
import sys
import warnings

# Suppress all warnings before importing PyMC
warnings.filterwarnings('ignore')

import pymc as pm
import arviz as az

# Import GPU utilities and cleanup utilities
from . import gpu_utils
from . import cleanup_utils

# Suppress PyMC logging and warnings
logging.getLogger('pymc').setLevel(logging.ERROR)
logging.getLogger('arviz').setLevel(logging.ERROR)
logging.getLogger('pytensor').setLevel(logging.ERROR)
logging.getLogger('aesara').setLevel(logging.ERROR)

# Additional suppression for PyMC-related modules
for logger_name in ['pymc', 'arviz', 'pytensor', 'aesara', 'numba', 'theano']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)
    logging.getLogger(logger_name).propagate = False

# Additional warning suppression
warnings.filterwarnings('ignore', category=UserWarning, module='pymc')
warnings.filterwarnings('ignore', category=UserWarning, module='arviz')
warnings.filterwarnings('ignore', category=UserWarning, module='pytensor')
warnings.filterwarnings('ignore', category=UserWarning, module='aesara')
warnings.filterwarnings('ignore', message='.*effective sample size.*')
warnings.filterwarnings('ignore', message='.*rhat.*')
warnings.filterwarnings('ignore', message='.*ess.*')
warnings.filterwarnings('ignore', message='.*divergence.*')
warnings.filterwarnings('ignore', message='.*divergences.*')
warnings.filterwarnings('ignore', message='.*target_accept.*')
warnings.filterwarnings('ignore', message='.*reparameterize.*')
warnings.filterwarnings('ignore', message='.*smaller than 100.*')
warnings.filterwarnings('ignore', message='.*needed for reliable.*')
warnings.filterwarnings('ignore', message='.*There was.*divergence.*')
warnings.filterwarnings('ignore', message='.*There were.*divergence.*')
warnings.filterwarnings('ignore', message='.*after tuning.*')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pymc')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='arviz')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pytensor')

# Note: Column validation is now done at the beginning of each function

@contextlib.contextmanager
def suppress_all_output():
    """Context manager that suppresses all stdout, stderr, and warnings from PyMC sampling"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Temporarily redirect stdout and stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            # Create string buffers to capture output
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer
            yield
        finally:
            # Restore original stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            # Log any captured output to file instead of console
            stdout_content = stdout_buffer.getvalue()
            stderr_content = stderr_buffer.getvalue()
            if stdout_content.strip() or stderr_content.strip():
                logger = logging.getLogger('optstop.sampling_output')
                if stdout_content.strip():
                    logger.debug(f"PyMC stdout: {stdout_content.strip()}")
                if stderr_content.strip():
                    logger.debug(f"PyMC stderr: {stderr_content.strip()}")

# --- Diagnostic Functions ---
def _beta_ci(successes: int, trials: int, cred_level: float = 0.95) -> Tuple[float, float]:
    """Compute beta credible interval for binomial proportion."""
    alpha_post = 1 + successes
    beta_post = 1 + trials - successes
    lower = beta.ppf((1-cred_level)/2, alpha_post, beta_post)
    upper = beta.ppf(1-(1-cred_level)/2, alpha_post, beta_post)
    return lower, upper

def _aggregate_with_ci(df: pd.DataFrame, score_col: str = "score", is_binomial: bool = True) -> pd.DataFrame:
    """Aggregate data to item level and compute CIs."""
    agg_cols = ["grouping", "sample_id"]
    grouped = df.groupby(agg_cols)
    
    if is_binomial:
        summary = grouped[score_col].agg(['sum', 'count']).reset_index()
        summary.rename(columns={'sum': 'successes', 'count': 'trials'}, inplace=True)
    else:
        summary = grouped[score_col].agg(['mean', 'count']).reset_index()
        summary.rename(columns={'mean': 'prop', 'count': 'trials'}, inplace=True)
        summary['successes'] = summary['prop'] * summary['trials']
    
    summary['prop'] = summary['successes'] / summary['trials']
    summary[['ci_low', 'ci_high']] = summary.apply(
        lambda row: _beta_ci(row['successes'], row['trials']), axis=1, result_type='expand'
    )
    summary['n_reps'] = summary['trials']
    return summary

def _compute_bayesian_hdi_per_task(df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Estimate HDI-based uncertainty across items for each grouping-task pair."""
    results = []
    
    grouped = df.groupby(['grouping', 'task'])
    for (grouping, task), sub_df in grouped:
        item_means = sub_df.groupby('sample_id')['score'].mean().values
        n_items = len(item_means)
        mean_score = item_means.mean() if n_items > 0 else 0
        
        if n_items == 0:
            continue
        elif n_items == 1:
            ci_width = 0.2
            ci_low = max(0, mean_score - ci_width/2)
            ci_high = min(1, mean_score + ci_width/2)
        else:
            if np.all(item_means == item_means[0]):
                ci_width = 1.0 / np.sqrt(n_items)
                ci_width = max(0.05, min(0.3, ci_width))
                ci_low = max(0, mean_score - ci_width/2)
                ci_high = min(1, mean_score + ci_width/2)
            else:
                bootstrap_samples = np.random.choice(
                    item_means, size=(3000, len(item_means)), replace=True
                )
                bootstrap_means = bootstrap_samples.mean(axis=1)
                
                try:
                    idata = az.from_dict(posterior={"Theta": bootstrap_means.reshape(1, 1, -1)})
                    hdi = az.hdi(idata, hdi_prob=confidence)["Theta"]
                    
                    try:
                        ci_low = float(hdi.sel(hdi='lower').values)
                        ci_high = float(hdi.sel(hdi='higher').values)
                    except Exception:
                        hdi_values = hdi.values.flatten()
                        ci_low, ci_high = np.min(hdi_values), np.max(hdi_values)
                except Exception:
                    ci_low = np.percentile(bootstrap_means, (1-confidence)*100/2)
                    ci_high = np.percentile(bootstrap_means, 100-(1-confidence)*100/2)
        
        if ci_low > mean_score:
            ci_low = max(0, mean_score - 0.05)
        if ci_high < mean_score:
            ci_high = min(1, mean_score + 0.05)
            
        results.append({
            'grouping': grouping,
            'task': task,
            'mean_score': mean_score,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n_items': n_items
        })
        
    return pd.DataFrame(results)

def _intraclass_corr(x: np.ndarray, y: np.ndarray) -> float:
    """ICC(3,1) two-way mixed, single score."""
    try:
        data = np.column_stack([x, y])
        n, k = data.shape
        
        # Handle edge cases
        if n < 2 or k < 2:
            return float('nan')
        
        mean_subject = data.mean(axis=1, keepdims=True)
        mean_rater = data.mean(axis=0, keepdims=True)
        grand_mean = data.mean()

        ss_subject = k * ((mean_subject - grand_mean) ** 2).sum()
        ss_rater = n * ((mean_rater - grand_mean) ** 2).sum()
        ss_total = ((data - grand_mean) ** 2).sum()
        ss_error = ss_total - ss_subject - ss_rater

        df_subject = n - 1
        df_error = (n - 1) * (k - 1)
        
        # Handle division by zero
        if df_subject <= 0 or df_error <= 0:
            return float('nan')
            
        ms_subject = ss_subject / df_subject
        ms_error = ss_error / df_error
        
        # Handle division by zero in ICC calculation
        denominator = ms_subject + (k - 1) * ms_error
        if denominator == 0:
            return float('nan')
            
        icc = (ms_subject - ms_error) / denominator
        return icc
    except (ValueError, ZeroDivisionError, RuntimeError):
        return float('nan')

def _bland_altman(ax: plt.Axes, x: np.ndarray, y: np.ndarray, **kw) -> Tuple[float, np.ndarray]:
    """Create Bland-Altman plot."""
    diff = y - x
    mean = (y + x) / 2
    ax.scatter(mean, diff, alpha=0.5, **kw)
    bias = diff.mean()
    loa = bias + np.array([-1.96, 1.96]) * diff.std(ddof=1)
    ax.axhline(bias, color="red", lw=1.2)
    ax.axhline(loa[0], color="grey", ls="--")
    ax.axhline(loa[1], color="grey", ls="--")
    ax.set_xlabel("Mean of full & pruned")
    ax.set_ylabel("Pruned − full")
    ax.set_title("Bland–Altman")
    return bias, loa

def _generate_diagnostic_plots(full_df: pd.DataFrame, pruned_df: pd.DataFrame, 
                              out_prefix: str = "optstop_diagnostics") -> None:
    """Generate diagnostic plots comparing full vs pruned datasets."""
    logger = logging.getLogger('optstop.diagnostics')
    logger.info(f"Generating diagnostic plots with prefix: {out_prefix}")
    
    try:
        # Create internal column structure for diagnostics
        full_df_internal = full_df.copy()
        pruned_df_internal = pruned_df.copy()
        
        # If the data doesn't have internal columns, create them
        if 'grouping' not in full_df_internal.columns:
            # Assume we have grouping_num and task_num columns
            if 'grouping_num' in full_df_internal.columns and 'task_num' in full_df_internal.columns:
                full_df_internal['grouping'] = full_df_internal['grouping_num'].astype(str) + '-' + full_df_internal['task_num'].astype(str)
                full_df_internal['task'] = full_df_internal['task_num']
                full_df_internal['sample_id'] = full_df_internal['sample_id_num']
            else:
                # Try to infer from available columns
                grouping_cols = [col for col in full_df_internal.columns if 'group' in col.lower() or 'task' in col.lower()]
                if grouping_cols:
                    full_df_internal['grouping'] = full_df_internal[grouping_cols].astype(str).agg('-'.join, axis=1)
                    full_df_internal['task'] = full_df_internal[grouping_cols[0]] if len(grouping_cols) == 1 else full_df_internal[grouping_cols[0]]
                else:
                    full_df_internal['grouping'] = 'group1'
                    full_df_internal['task'] = 1
                
                sample_id_cols = [col for col in full_df_internal.columns if 'sample' in col.lower() or 'id' in col.lower()]
                if sample_id_cols:
                    full_df_internal['sample_id'] = full_df_internal[sample_id_cols[0]]
                else:
                    full_df_internal['sample_id'] = full_df_internal.index
        
        if 'grouping' not in pruned_df_internal.columns:
            # Apply the same logic to pruned data
            if 'grouping_num' in pruned_df_internal.columns and 'task_num' in pruned_df_internal.columns:
                pruned_df_internal['grouping'] = pruned_df_internal['grouping_num'].astype(str) + '-' + pruned_df_internal['task_num'].astype(str)
                pruned_df_internal['task'] = pruned_df_internal['task_num']
                pruned_df_internal['sample_id'] = pruned_df_internal['sample_id_num']
            else:
                grouping_cols = [col for col in pruned_df_internal.columns if 'group' in col.lower() or 'task' in col.lower()]
                if grouping_cols:
                    pruned_df_internal['grouping'] = pruned_df_internal[grouping_cols].astype(str).agg('-'.join, axis=1)
                    pruned_df_internal['task'] = pruned_df_internal[grouping_cols[0]] if len(grouping_cols) == 1 else pruned_df_internal[grouping_cols[0]]
                else:
                    pruned_df_internal['grouping'] = 'group1'
                    pruned_df_internal['task'] = 1
                
                sample_id_cols = [col for col in pruned_df_internal.columns if 'sample' in col.lower() or 'id' in col.lower()]
                if sample_id_cols:
                    pruned_df_internal['sample_id'] = pruned_df_internal[sample_id_cols[0]]
                else:
                    pruned_df_internal['sample_id'] = pruned_df_internal.index
        
        # Aggregate to item level and compute CIs
        pruned_item = _aggregate_with_ci(pruned_df_internal)
        full_item = _aggregate_with_ci(full_df_internal)

        # Calculate confidence intervals for both datasets
        full_results = _compute_bayesian_hdi_per_task(full_df_internal)
        full_results.rename(columns={'n_items': 'n_samples'}, inplace=True)
        full_results['dataset'] = 'Full'

        trimmed_results = _compute_bayesian_hdi_per_task(pruned_df_internal)
        trimmed_results.rename(columns={'n_items': 'n_samples'}, inplace=True)
        trimmed_results['dataset'] = 'Trimmed'

        # Combine results for plotting
        combined_results = pd.concat([full_results, trimmed_results])

        # Merge item-level data
        agg_cols = ["grouping", "sample_id"]
        merged = full_item.merge(
            pruned_item,
            on=agg_cols,
            suffixes=('_full', '_pruned'),
            validate="one_to_one"
        )

        # Calculate accuracy statistics
        merged["diff"] = merged["prop_pruned"] - merged["prop_full"]
        bias = merged["diff"].mean() if len(merged) > 0 else float('nan')
        mae = merged["diff"].abs().mean() if len(merged) > 0 else float('nan')
        if len(merged) > 0:
            try:
                diff_squared = pd.to_numeric(merged["diff"] ** 2, errors='coerce')
                rmse = np.sqrt(diff_squared.mean()) if not diff_squared.isna().all() else float('nan')
            except (TypeError, ValueError):
                rmse = float('nan')
        else:
            rmse = float('nan')
        r = icc = float('nan')
        if len(merged) > 1:
            try:
                r, _ = stats.pearsonr(merged["prop_full"], merged["prop_pruned"])
            except (ValueError, ZeroDivisionError):
                r = float('nan')
            try:
                icc = _intraclass_corr(merged["prop_full"].values, merged["prop_pruned"].values)
            except (ValueError, ZeroDivisionError):
                icc = float('nan')

        # Calculate efficiency metrics
        merged["rep_saving"] = merged["n_reps_full"] - merged["n_reps_pruned"] if len(merged) > 0 else 0
        # Handle division by zero for percentage calculation
        merged["rep_saving_prcnt"] = 0
        if len(merged) > 0:
            total_reps_full = merged["n_reps_full"].sum()
            if total_reps_full > 0:
                merged["rep_saving_prcnt"] = (merged["rep_saving"] / merged["n_reps_full"]) * 100
        avg_save = merged["rep_saving"].mean() if len(merged) > 0 else float('nan')
        pct_save = float('nan')
        if len(merged) > 0:
            total_reps_full = merged["n_reps_full"].sum()
            if total_reps_full > 0:
                pct_save = merged["rep_saving"].sum() / total_reps_full
        n_items_full = len(full_item)
        n_items_pruned = len(pruned_item)
        n_items_saved = n_items_full - n_items_pruned
        pct_items_saved = n_items_saved / n_items_full if n_items_full > 0 else float('nan')

        # Print summary statistics
        logger.info("===== Accuracy metrics =================================")
        logger.info(f"Bias (mean diff):          {bias: .4f}")
        logger.info(f"Mean absolute error (MAE): {mae: .4f}")
        logger.info(f"Root MSE:                  {rmse: .4f}")
        logger.info(f"Pearson  r:                {r: .3f}")
        logger.info(f"ICC(3,1):                  {icc: .3f}")
        logger.info("===== Efficiency =======================================")
        logger.info(f"Mean reps saved: {avg_save: .1f} ({pct_save*100: .1f} % of total)")
        logger.info(f"Full run used {n_items_full} items, pruned run used {n_items_pruned} items.")
        logger.info(f"Items saved: {n_items_saved} ({pct_items_saved*100:.1f}%)")

        # Create diagnostic plots
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1])
        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])
        ax_d = fig.add_subplot(gs[1, :])
        ax_e = fig.add_subplot(gs[2, :])

        # A. Identity scatter
        if len(merged) > 1:
            ax_a.scatter(merged["prop_full"], merged["prop_pruned"], alpha=0.6, s=18)
            lims = [min(ax_a.get_xlim()[0], ax_a.get_ylim()[0]),
                    max(ax_a.get_xlim()[1], ax_a.get_ylim()[1])]
            ax_a.plot(lims, lims, "--k", lw=1)
            ax_a.set_title("Full vs. pruned")
            ax_a.set_xlabel("Full-run score")
            ax_a.set_ylabel("Pruned score")
            ax_a.set_aspect("equal", adjustable="box")
        else:
            ax_a.text(0.5, 0.5, "Not enough data for scatter plot", ha='center', va='center', fontsize=12)
            ax_a.set_title("Full vs. pruned")
            ax_a.set_xlabel("Full-run score")
            ax_a.set_ylabel("Pruned score")

        # B. Bland-Altman
        if len(merged) > 1:
            _bland_altman(ax_b, merged["prop_full"].values, merged["prop_pruned"].values, s=18)
        else:
            ax_b.text(0.5, 0.5, "Not enough data for Bland-Altman", ha='center', va='center', fontsize=12)
            ax_b.set_title("Bland–Altman")
            ax_b.set_xlabel("Mean of full & pruned")
            ax_b.set_ylabel("Pruned − full")

        # C. Reps saved histogram
        if len(merged) > 0:
            ax_c.hist(merged["rep_saving"], bins=10, edgecolor="k")
            ax_c.set_xlabel("Average Epochs Saved Per Task")
            ax_c.set_ylabel("Count")
            ax_c.set_title("Efficiency distribution (Epochs)")
        else:
            ax_c.text(0.5, 0.5, "No data for efficiency histogram", ha='center', va='center', fontsize=12)
            ax_c.set_title("Efficiency distribution (Epochs)")
            ax_c.set_xlabel("Average Epochs Saved Per Task")
            ax_c.set_ylabel("Count")

        # D. Items saved by grouping (use 'grouping' variable)
        if 'grouping' in full_item.columns and 'grouping' in pruned_item.columns and len(full_item) > 0:
            full_item_model = full_item.groupby("grouping").size().reset_index(name='n_items_full')
            pruned_item_model = pruned_item.groupby("grouping").size().reset_index(name='n_items_pruned')
            item_diff = full_item_model.merge(pruned_item_model, on="grouping", how="outer").fillna(0)
            item_diff["n_items_saved"] = item_diff["n_items_full"] - item_diff["n_items_pruned"]
            item_diff["pct_items_saved"] = ((item_diff["n_items_saved"] / item_diff["n_items_full"]) * 100).fillna(0)
            item_diff.plot(x="grouping", y="pct_items_saved", kind="bar", ax=ax_d,
                          color="lightgrey", edgecolor="black")
            ax_d.set_ylabel("Sample IDs Saved (%)")
            ax_d.set_title("Percentage of Sample IDs Saved, Split by Grouping")
            ax_d.set_xticklabels(item_diff["grouping"], rotation=45, ha='right', fontsize=8)
            ax_d.legend().set_visible(False)
        else:
            ax_d.text(0.5, 0.5, "Not enough data for items saved by grouping", ha='center', va='center', fontsize=12)
            ax_d.set_title("Percentage of Sample IDs Saved, Split by Grouping")
            ax_d.set_ylabel("Sample IDs Saved (%)")

        # E. CI comparison (use 'grouping' variable)
        if 'grouping' in combined_results.columns and len(combined_results) > 0:
            groupings = combined_results[['grouping']].drop_duplicates()
            groupings_labels = [f"{row['grouping']}" for _, row in groupings.iterrows()]
            n_groups = len(groupings)
            colors = {'Full': 'blue', 'Trimmed': 'red'}
            bar_width = 0.35
            for i, (_, mt_row) in enumerate(groupings.iterrows()):
                group = mt_row['grouping']
                full_data = combined_results[(combined_results['grouping'] == group) & 
                                            (combined_results['dataset'] == 'Full')]
                trimmed_data = combined_results[(combined_results['grouping'] == group) & 
                                            (combined_results['dataset'] == 'Trimmed')]
                pos_full = i - bar_width/2
                pos_trimmed = i + bar_width/2
                if not full_data.empty:
                    full_mean = full_data['mean_score'].values[0]
                    full_low = full_data['ci_low'].values[0]
                    full_high = full_data['ci_high'].values[0]
                    full_n = full_data['n_samples'].values[0]
                    ax_e.plot(pos_full, full_mean, 'o', color='black', markersize=3)
                    ax_e.plot([pos_full, pos_full], [full_low, full_high], 
                            color=colors['Full'], linewidth=1)
                    cap_width = 0.05
                    ax_e.plot([pos_full-cap_width, pos_full+cap_width], [full_low, full_low], 
                            color=colors['Full'], linewidth=1)
                    ax_e.plot([pos_full-cap_width, pos_full+cap_width], [full_high, full_high], 
                            color=colors['Full'], linewidth=1)
                    ax_e.annotate(f"n={int(full_n)}", (pos_full, full_high), 
                                xytext=(0, 5), textcoords='offset points',
                                ha='center', fontsize=6)
                if not trimmed_data.empty:
                    trimmed_mean = trimmed_data['mean_score'].values[0]
                    trimmed_low = trimmed_data['ci_low'].values[0]
                    trimmed_high = trimmed_data['ci_high'].values[0]
                    trimmed_n = trimmed_data['n_samples'].values[0]
                    ax_e.plot(pos_trimmed, trimmed_mean, 'o', color='black', markersize=3)
                    ax_e.plot([pos_trimmed, pos_trimmed], [trimmed_low, trimmed_high], 
                            color=colors['Trimmed'], linewidth=1)
                    cap_width = 0.05
                    ax_e.plot([pos_trimmed-cap_width, pos_trimmed+cap_width], [trimmed_low, trimmed_low], 
                            color=colors['Trimmed'], linewidth=1)
                    ax_e.plot([pos_trimmed-cap_width, pos_trimmed+cap_width], [trimmed_high, trimmed_high], 
                            color=colors['Trimmed'], linewidth=1)
                    ax_e.annotate(f"n={int(trimmed_n)}", (pos_trimmed, trimmed_high), 
                                xytext=(0, 5), textcoords='offset points',
                                ha='center', fontsize=6)
            ax_e.set_ylabel('Score (Pass Rate)')
            ax_e.set_xlabel('Grouping')
            ax_e.set_title('Confidence Intervals: Full (Blue) vs. Pruned (Red)')
            ax_e.set_xticks(range(n_groups))
            ax_e.set_xticklabels(groupings_labels, rotation=45, ha='right', fontsize=8)
            ax_e.set_ylim(0, 1.05)
            ax_e.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            ax_e.text(0.5, 0.5, "Not enough data for CI comparison", ha='center', va='center', fontsize=12)
            ax_e.set_title('Confidence Intervals: Full (Blue) vs. Pruned (Red)')
            ax_e.set_ylabel('Score (Pass Rate)')
            ax_e.set_xlabel('Grouping')

        fig.tight_layout()
        fig.savefig(f"{out_prefix}.png", dpi=150)
        merged.to_csv(f"{out_prefix}_paired.csv", index=False)
        
        logger.info(f"Diagnostic plots saved to {out_prefix}.png")
        logger.info(f"Paired data saved to {out_prefix}_paired.csv")
        
    except Exception as e:
        logger.error(f"Error generating diagnostic plots: {e}")
        # Always save empty/partial files for test robustness
        try:
            fig = plt.figure(figsize=(12, 12))
            fig.suptitle(f"Diagnostics failed: {e}", fontsize=14)
            fig.savefig(f"{out_prefix}.png", dpi=150)
            pd.DataFrame().to_csv(f"{out_prefix}_paired.csv", index=False)
        except Exception:
            pass
        raise



# --- Helper: Adaptive Beta CI ---
def _beta_ci_adaptive(successes: int, trials: int, cred_level: float = 0.95, conservatism: float = 1.0, 
                     low_perf_threshold: float = 0.2, base_strength: int = 2, samples: int = 10000) -> Tuple[float, float, float]:
    """
    Compute an adaptive Bayesian credible interval for a binomial proportion, with conservatism for low performance.
    """
    if trials == 0:
        return 0.0, 1.0, 1.0
    p_hat = successes / trials
    if p_hat < low_perf_threshold:
        alpha_boost = conservatism
        prior_scaling = base_strength * np.exp(-trials / (10 * conservatism))
        alpha_prior = max(prior_scaling * p_hat * alpha_boost, 0.5)
        beta_prior = max(prior_scaling * (1 - p_hat), 0.5)
    else:
        prior_scaling = base_strength * np.exp(-trials / 10)
        alpha_prior = max(prior_scaling * p_hat, 0.5)
        beta_prior = max(prior_scaling * (1 - p_hat), 0.5)
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + (trials - successes)
    draws = np.random.beta(alpha_post, beta_post, samples)
    lo, hi = np.quantile(draws, [(1 - cred_level) / 2, 1 - (1 - cred_level) / 2])
    if p_hat < low_perf_threshold:
        effective_width = (hi - lo) * conservatism
    else:
        effective_width = hi - lo
    return lo, hi, effective_width

def configure_optstop_logging(logfile: str = 'optstop_run.log', level: int = logging.INFO, console_output: bool = False) -> None:
    """
    Configure logging for the optstop package to log to file and optionally to console.
    
    Args:
        logfile: Path to the log file
        level: Logging level (default: INFO)
        console_output: Whether to also log to console (default: False)
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Only add console handler if requested
    if console_output:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        logger.addHandler(ch)
    
    # Always add file handler
    fh = logging.FileHandler(logfile)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(fh)

# To log to both console and file, call configure_optstop_logging() before running optimal stopping.

def _validate_params(params: Dict[str, Any]) -> None:
    logger = logging.getLogger('optstop.posthoc')
    for key in ['draws', 'tune', 'stab_window', 'rep_batch_size', 'pymc_refresh_every']:
        if key in params and (not isinstance(params[key], int) or params[key] <= 0):
            logger.error(f"Parameter '{key}' must be a positive integer. Got: {params[key]}")
            raise ValueError(f"Parameter '{key}' must be a positive integer. Got: {params[key]}")
    if 'delta_item' in params and params['delta_item'] <= 0:
        logger.error("delta_item must be positive.")
        raise ValueError("delta_item must be positive.")
    if 'delta_cap' in params and params['delta_cap'] <= 0:
        logger.error("delta_cap must be positive.")
        raise ValueError("delta_cap must be positive.")
    logger.info(f"Parameters validated: {params}")

def _worker_initializer_posthoc(worker_dir, gpu_id=None, suppress_output=True):
    """Initialize worker process with clean PyTensor environment for posthoc analysis.

    Args:
        worker_dir: Base directory for PyTensor compilation
        gpu_id: GPU ID to assign to this worker, or None for CPU-only
        suppress_output: Whether to suppress output (default: True)
    """
    import os

    # Create unique worker temporary directory with robust cleanup
    unique_worker_dir = cleanup_utils.create_worker_temp_dir('optstop_posthoc')

    # Register enhanced cleanup for worker process
    cleanup_utils.register_worker_cleanup(unique_worker_dir, 'optstop.worker_posthoc')

    # CRITICAL: Set environment variables BEFORE any PyTensor import
    # Determine device configuration based on gpu_id parameter
    if gpu_id is not None:
        # GPU-enabled worker
        os.environ['PYTENSOR_FLAGS'] = f'compiledir={unique_worker_dir},device=gpu,floatX=float32'
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # Configure JAX to use specific GPU
        os.environ['JAX_CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device_type = f'GPU {gpu_id}'
    else:
        # CPU-only worker (preserve existing behavior)
        os.environ['PYTENSOR_FLAGS'] = f'compiledir={unique_worker_dir},device=cpu,floatX=float32'
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device_type = 'CPU'

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # Additional PyTensor isolation environment variables
    os.environ['PYTENSOR_FLAGS'] += ',optimizer=fast_compile,openmp=False'

    # Force PyTensor to use our compiledir by setting config directly after import
    # This handles cases where PyTensor might be imported later
    try:
        import pytensor
        pytensor.config.compiledir = unique_worker_dir
        # Remove force_compile as it's not a valid PyTensor config
        # Clear any existing module cache
        if hasattr(pytensor.link.c.basic, '_module_cache'):
            pytensor.link.c.basic._module_cache = None
        # Clear any global cache
        if hasattr(pytensor.link.c.cmodule, '_module_cache'):
            pytensor.link.c.cmodule._module_cache = None
    except Exception as e:
        # If this fails, we still have environment variables as fallback
        pass

    # Debug logging
    logger = logging.getLogger('optstop.worker_posthoc')
    if not suppress_output:
        logger.info(f"Posthoc worker {os.getpid()} using {device_type}, PyTensor compiledir: {unique_worker_dir}")

def _process_posthoc_grouping_with_init(task_args: Tuple[Any, pd.DataFrame, Dict[str, Any], str], worker_args: Tuple[str, Optional[int], bool]) -> Dict[str, Any]:
    """Wrapper function that initializes worker and then processes posthoc grouping."""
    # Initialize worker with GPU assignment
    _worker_initializer_posthoc(*worker_args)
    # Process the actual task
    return _process_posthoc_grouping(task_args)

def _process_posthoc_grouping(args: Tuple[Any, pd.DataFrame, Dict[str, Any], str]) -> Dict[str, Any]:
    # Environment variables are now set by the worker initializer
    import sys
    import io
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        pid, df_part, params, score_column = args

        logger = logging.getLogger('optstop.posthoc')
        try:
            delta_item = params.get('delta_item', 0.05)
            delta_cap = params.get('delta_cap', 0.05)
            CI_delta = params.get('CI_delta', 0.0002)
            cred_level = params.get('cred_level', 0.95)
            conservatism = params.get('conservatism', 5)
            low_perf_threshold = params.get('low_performance_threshold', 0.05)
            rep_batch_size = params.get('rep_batch_size', 1)
            pymc_refresh_every = params.get('pymc_refresh_every', 2)
            stab_window = params.get('stab_window', 5)

            # CRITICAL: Initialize JAX/PyMC backend BEFORE any model creation
            # Detect GPU availability based on worker's actual environment configuration
            # (set by worker initializer based on GPU assignment)
            gpu_available, gpu_backend, gpu_info = gpu_utils.check_gpu_availability()

            # Configure JAX and PyMC for GPU computation if available
            if gpu_available and gpu_backend == 'jax-gpu':
                try:
                    import jax
                    # Explicitly configure JAX for GPU
                    jax.config.update('jax_platform_name', 'gpu')
                    # Verify JAX sees GPU devices
                    devices = jax.devices()
                    gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
                    if gpu_devices:
                        logger.info(f"Worker JAX initialized with GPU devices: {gpu_devices}")
                        # Force JAX to use the first available GPU device
                        import jax.numpy as jnp
                        # Test JAX GPU functionality with a simple operation
                        test_array = jnp.array([1.0, 2.0, 3.0])
                        result = jnp.sum(test_array)  # This should execute on GPU
                        logger.info(f"Worker JAX GPU test successful: {result}")

                        # Configure PyMC to use JAX backend (don't set PyTensor device)
                        import os
                        os.environ['PYMC_BACKEND'] = 'jax'

                        # JAX handles GPU automatically - no PyTensor configuration needed
                    else:
                        logger.warning("Worker JAX GPU setup failed - falling back to CPU")
                        gpu_available = False
                        gpu_backend = 'cpu'
                except ImportError:
                    logger.warning("JAX not available - using CPU backend")
                    gpu_available = False
                    gpu_backend = 'cpu'
                except Exception as e:
                    logger.warning(f"JAX GPU setup failed: {e} - using CPU backend")
                    gpu_available = False
                    gpu_backend = 'cpu'

            sampling_kwargs = gpu_utils.get_sampling_kwargs(params, gpu_available, gpu_backend)

            # Log GPU status for this worker
            if gpu_available:
                logger.info(f"Worker processing grouping {pid} with GPU acceleration ({gpu_backend})")
            else:
                logger.info(f"Worker processing grouping {pid} with CPU-only")

            if 'random_seed' in params:
                np.random.seed(params['random_seed'])
            item_summaries = []
            used_reps_dfs = []
            theta_lo, theta_hi, theta_width = None, None, None
            CI_record = []
            CI_slopes_hist = []
            initial_perf = df_part[score_column].mean()
            current_conservatism = conservatism if initial_perf < low_perf_threshold else 1.0
            with pm.Model() as model:
                mu_group = pm.Normal("mu_group", mu=2, sigma=1.5)
                sigma_group = pm.Exponential("sigma_group", lam=1.0)
                successes_data = pm.Data("successes", np.array([0]))
                n_items = pm.Data("n_items", np.array(1, dtype="int64"))
                trials_data = pm.Data("trials", np.array([1]))
                z = pm.Normal("z", mu=0, sigma=1, shape=n_items)
                mu_item = pm.Deterministic("mu_item", mu_group + z * sigma_group)
                mu_item_clipped = pm.Deterministic("mu_item_clipped", pm.math.clip(mu_item, -6.0, 6.0))
                Theta = pm.Deterministic("Theta", pm.math.sigmoid(mu_item_clipped))
                obs = pm.Binomial("obs", n=trials_data, p=Theta, observed=successes_data)
            item_ids = list(df_part['sample_id_num'].unique())
            for item_idx, item_id in enumerate(item_ids):
                df_item = df_part[df_part['sample_id_num'] == item_id].sort_values('epoch_num')
                successes = 0
                trials = 0
                used_reps = []
                ci_record = []
                ci_slopes_hist = []
                for start in range(0, len(df_item), rep_batch_size):
                    batch = df_item.iloc[start:start+rep_batch_size]
                    successes += batch[score_column].sum()
                    trials += len(batch)
                    used_reps.extend(batch.itertuples(index=False))
                    lo, hi, width = _beta_ci_adaptive(
                        successes, trials, cred_level=cred_level,
                        conservatism=current_conservatism,
                        low_perf_threshold=low_perf_threshold
                    )
                    ci_record.append(width)
                    if width < delta_item:
                        logger.info(f"Stopping sample_id {item_id} (group {pid}) at epoch {batch['epoch_num'].iloc[-1]}: CI width {width:.4f} < delta_item {delta_item} | epochs used: {trials}")
                        break
                    if len(ci_record) >= stab_window:
                        recent_widths = ci_record[-stab_window:]
                        slope = np.polyfit(range(len(recent_widths)), recent_widths, 1)[0]
                        ci_slopes_hist.append(slope)
                        slope_threshold = CI_delta / current_conservatism if df_item[score_column].mean() < low_perf_threshold else CI_delta
                        if (abs(slope) <= slope_threshold) and (len(ci_slopes_hist) >= 4):
                            recent_slopes = ci_slopes_hist[-3:]
                            slope_slopes = np.polyfit(range(len(recent_slopes)), recent_slopes, 1)[0]
                            if slope_slopes >= 0:
                                if df_item[score_column].mean() >= low_perf_threshold:
                                    logger.info(f"Stopping sample_id {item_id} (group {pid}) due to CI stabilization: slope {slope:.6f} <= threshold {slope_threshold:.6f} | epochs used: {trials}")
                                    break
                                elif abs(slope) <= slope_threshold / 2:
                                    logger.info(f"Stopping low-performance sample_id {item_id} (group {pid}) due to strong CI stabilization: slope {slope:.6f} | epochs used: {trials}")
                                    break
                item_summaries.append({'successes': successes, 'trials': trials})
                used_reps_dfs.append(pd.DataFrame(used_reps, columns=df_item.columns))
                current_perf_estimate = sum(s['successes'] for s in item_summaries) / sum(s['trials'] for s in item_summaries)
                current_conservatism = conservatism if current_perf_estimate < low_perf_threshold else 1.0
                if ((item_idx + 1) % pymc_refresh_every == 0) or (item_idx == len(item_ids) - 1):
                    all_successes = np.array([s['successes'] for s in item_summaries])
                    all_trials = np.array([s['trials'] for s in item_summaries])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with model:
                            pm.set_data({
                                "successes": all_successes,
                                "trials": all_trials,
                                "n_items": np.int64(len(all_successes))
                            })
                            with suppress_all_output():
                                trace = pm.sample(**sampling_kwargs)
                            with suppress_all_output():
                                theta_hdi = az.hdi(trace.posterior["Theta"], hdi_prob=cred_level)
                        hdi_indices = list(theta_hdi["Theta"].hdi.values)
                        try:
                            theta_values_lower = theta_hdi["Theta"].sel(hdi="lower").values
                            theta_lo = theta_values_lower.item() if theta_values_lower.size == 1 else theta_values_lower.flatten()[0]
                        except Exception:
                            theta_lo = theta_hdi["Theta"].values[..., 0].flatten()[0]
                        try:
                            theta_values_upper = theta_hdi["Theta"].sel(hdi="upper").values
                            theta_hi = theta_values_upper.item() if theta_values_upper.size == 1 else theta_values_upper.flatten()[0]
                        except Exception:
                            theta_hi = theta_hdi["Theta"].values[..., 1].flatten()[0]
                        theta_width = theta_hi - theta_lo
                        CI_record.append(theta_width)
                        effective_width = theta_width * current_conservatism if current_perf_estimate < low_perf_threshold else theta_width
                        if effective_width < delta_cap:
                            logger.info(f"Stopping grouping {pid}: CI width {effective_width:.4f} < delta_cap {delta_cap} | sample_ids used: {len(item_summaries)}")
                            break
                        if len(CI_record) >= stab_window:
                            recent_widths = CI_record[-stab_window:]
                            slope = np.polyfit(range(len(recent_widths)), recent_widths, 1)[0]
                            CI_slopes_hist.append(slope)
                            slope_threshold = CI_delta / current_conservatism if current_perf_estimate < low_perf_threshold else CI_delta
                            if (abs(slope) <= slope_threshold) and (len(CI_slopes_hist) >= 4):
                                recent_slopes = CI_slopes_hist[-3:]
                                slope_slopes = np.polyfit(range(len(recent_slopes)), recent_slopes, 1)[0]
                                if slope_slopes >= 0:
                                    if current_perf_estimate >= low_perf_threshold:
                                        logger.info(f"Stopping grouping {pid} due to CI stabilization: slope {slope:.6f} <= threshold {slope_threshold:.6f} | sample_ids used: {len(item_summaries)}")
                                        break
                                    elif abs(slope) <= slope_threshold / 2:
                                        logger.info(f"Stopping low-performance grouping {pid} due to strong CI stabilization: slope {slope:.6f} | sample_ids used: {len(item_summaries)}")
                                        break
            avg_reps_per_item = np.mean([len(df) for df in used_reps_dfs]) if used_reps_dfs else 0
            result = {
                'grouping': pid,
                'n_items_used': len(item_summaries),
                'theta_ci_low': theta_lo,
                'theta_ci_high': theta_hi,
                'theta_ci_width': theta_width,
                'percent_items_used': len(item_summaries) / len(item_ids) if item_ids else 0,
                'avg_reps_per_item': avg_reps_per_item,
                'used_reps_dfs': used_reps_dfs,
                'error': None
            }
            return result
        except Exception as e:
            logger.error(f"Error processing grouping {pid}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'grouping': pid,
                'n_items_used': None,
                'theta_ci_low': None,
                'theta_ci_high': None,
                'theta_ci_width': None,
                'percent_items_used': None,
                'avg_reps_per_item': None,
                'used_reps_dfs': [],
                'error': str(e)
            }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def _worker_initializer_live(worker_dir, gpu_id=None, suppress_output=True):
    """Initialize worker process with clean PyTensor environment for live analysis.

    Args:
        worker_dir: Base directory for PyTensor compilation
        gpu_id: GPU ID to assign to this worker, or None for CPU-only
        suppress_output: Whether to suppress output (default: True)
    """
    import os

    # Create unique worker temporary directory with robust cleanup
    unique_worker_dir = cleanup_utils.create_worker_temp_dir('optstop_live')

    # Register enhanced cleanup for worker process
    cleanup_utils.register_worker_cleanup(unique_worker_dir, 'optstop.worker_live')

    # CRITICAL: Set environment variables BEFORE any PyTensor import
    # Determine device configuration based on gpu_id parameter
    if gpu_id is not None:
        # GPU-enabled worker
        os.environ['PYTENSOR_FLAGS'] = f'compiledir={unique_worker_dir},device=gpu,floatX=float32'
        os.environ['JAX_PLATFORM_NAME'] = 'gpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # Configure JAX to use specific GPU
        os.environ['JAX_CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device_type = f'GPU {gpu_id}'
    else:
        # CPU-only worker (preserve existing behavior)
        os.environ['PYTENSOR_FLAGS'] = f'compiledir={unique_worker_dir},device=cpu,floatX=float32'
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device_type = 'CPU'

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # Additional PyTensor isolation environment variables
    os.environ['PYTENSOR_FLAGS'] += ',optimizer=fast_compile,openmp=False'

    # Force PyTensor to use our compiledir by setting config directly after import
    # This handles cases where PyTensor might be imported later
    try:
        import pytensor
        pytensor.config.compiledir = unique_worker_dir
        # Remove force_compile as it's not a valid PyTensor config
        # Clear any existing module cache
        if hasattr(pytensor.link.c.basic, '_module_cache'):
            pytensor.link.c.basic._module_cache = None
        # Clear any global cache
        if hasattr(pytensor.link.c.cmodule, '_module_cache'):
            pytensor.link.c.cmodule._module_cache = None
    except Exception as e:
        # If this fails, we still have environment variables as fallback
        pass

    # Debug logging
    logger = logging.getLogger('optstop.worker_live')
    if not suppress_output:
        logger.info(f"Live worker {os.getpid()} using {device_type}, PyTensor compiledir: {unique_worker_dir}")

def _process_live_grouping_with_init(task_args: Tuple[str, pd.DataFrame, Dict[str, Any], str, str, str, str], worker_args: Tuple[str, Optional[int], bool]) -> Dict[str, Any]:
    """Wrapper function that initializes worker and then processes live grouping."""
    # Initialize worker with GPU assignment
    _worker_initializer_live(*worker_args)
    # Process the actual task
    return _process_live_grouping(task_args)

def _process_live_grouping(args: Tuple[str, pd.DataFrame, Dict[str, Any], str, str, str, str]) -> Dict[str, Any]:
    """
    Worker function for processing a single grouping in live mode.
    Returns stop_sample_ids and stop_task results for this grouping.
    """
    # Environment variables are now set by the worker initializer
    import sys
    import io
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        grouping, df_grouping, params, sample_id_column, epoch_column, score_column, grouping_columns = args

        logger = logging.getLogger('optstop.live')
        logger.info(f"Processing grouping: {grouping}")

        # Extract parameters
        delta_item = params.get('delta_item', 0.05)
        delta_cap = params.get('delta_cap', 0.05)
        CI_delta = params.get('CI_delta', 0.0002)
        cred_level = params.get('cred_level', 0.95)
        conservatism = params.get('conservatism', 5)
        low_perf_threshold = params.get('low_performance_threshold', 0.05)
        rep_batch_size = params.get('rep_batch_size', 1)
        pymc_refresh_every = params.get('pymc_refresh_every', 2)
        stab_window = params.get('stab_window', 5)

        # CRITICAL: Initialize JAX/PyMC backend BEFORE any model creation
        # Detect GPU availability based on worker's actual environment configuration
        # (set by worker initializer based on GPU assignment)
        gpu_available, gpu_backend, gpu_info = gpu_utils.check_gpu_availability()

        # Configure JAX and PyMC for GPU computation if available
        if gpu_available and gpu_backend == 'jax-gpu':
            try:
                import jax
                # Explicitly configure JAX for GPU
                jax.config.update('jax_platform_name', 'gpu')
                # Verify JAX sees GPU devices
                devices = jax.devices()
                gpu_devices = [d for d in devices if 'gpu' in str(d).lower() or 'cuda' in str(d).lower()]
                if gpu_devices:
                    logger.info(f"Worker JAX initialized with GPU devices: {gpu_devices}")
                    # Force JAX to use the first available GPU device
                    import jax.numpy as jnp
                    # Test JAX GPU functionality with a simple operation
                    test_array = jnp.array([1.0, 2.0, 3.0])
                    result = jnp.sum(test_array)  # This should execute on GPU
                    logger.info(f"Worker JAX GPU test successful: {result}")

                    # Configure PyMC to use JAX backend (don't set PyTensor device)
                    import os
                    os.environ['PYMC_BACKEND'] = 'jax'

                    # JAX handles GPU automatically - no PyTensor configuration needed
                else:
                    logger.warning("Worker JAX GPU setup failed - falling back to CPU")
                    gpu_available = False
                    gpu_backend = 'cpu'
            except ImportError:
                logger.warning("JAX not available - using CPU backend")
                gpu_available = False
                gpu_backend = 'cpu'
            except Exception as e:
                logger.warning(f"JAX GPU setup failed: {e} - using CPU backend")
                gpu_available = False
                gpu_backend = 'cpu'

        sampling_kwargs = gpu_utils.get_sampling_kwargs(params, gpu_available, gpu_backend)

        # Log GPU status for this worker
        if gpu_available:
            logger.info(f"Worker processing grouping {grouping} with GPU acceleration ({gpu_backend})")
        else:
            logger.info(f"Worker processing grouping {grouping} with CPU-only")

        # Initialize variables
        stop_sample_ids = []
        stop_this_grouping = []
        CI_record = []
        CI_slopes_hist = []
        
        # Create numeric columns for processing
        df_grouping = df_grouping.copy()
        df_grouping['grouping'] = df_grouping[grouping_columns].astype(str).agg('-'.join, axis=1)
        df_grouping['grouping_num'] = df_grouping['grouping'].astype('category').cat.codes
        df_grouping['sample_id_num'] = df_grouping[sample_id_column].astype('category').cat.codes
        df_grouping['epoch_num'] = df_grouping[epoch_column].astype(int)
        
        sample_ci_records = {}
        sample_ci_slopes = {}
        used_reps_dfs = {}
        item_summaries = []
        item_ids = list(df_grouping['sample_id_num'].unique())
        
        # PyMC model for group-level
        with pm.Model() as model:
            mu_group = pm.Normal("mu_group", mu=2, sigma=1.5)
            sigma_group = pm.Exponential("sigma_group", lam=1.0)
            successes_data = pm.Data("successes", np.array([0]))
            n_items = pm.Data("n_items", np.array(1, dtype="int64"))
            trials_data = pm.Data("trials", np.array([1]))
            z = pm.Normal("z", mu=0, sigma=1, shape=n_items)
            mu_item = pm.Deterministic("mu_item", mu_group + z * sigma_group)
            mu_item_clipped = pm.Deterministic("mu_item_clipped", pm.math.clip(mu_item, -6.0, 6.0))
            Theta = pm.Deterministic("Theta", pm.math.sigmoid(mu_item_clipped))
            obs = pm.Binomial("obs", n=trials_data, p=Theta, observed=successes_data)
        
        # Per-sample_id stopping (width and slope)
        for item_idx, item_id in enumerate(item_ids):
            df_item = df_grouping[df_grouping['sample_id_num'] == item_id].sort_values('epoch_num')
            successes = 0
            trials = 0
            used_reps = []
            ci_record = []
            ci_slopes_hist = []
            current_conservatism = conservatism if df_item[score_column].mean() < low_perf_threshold else 1.0
            for start in range(0, len(df_item), rep_batch_size):
                batch = df_item.iloc[start:start+rep_batch_size]
                successes += batch[score_column].sum()
                trials += len(batch)
                used_reps.extend(batch.itertuples(index=False))
                lo, hi, width = _beta_ci_adaptive(
                    successes, trials, cred_level=cred_level,
                    conservatism=current_conservatism,
                    low_perf_threshold=low_perf_threshold
                )
                ci_record.append(width)
                if width < delta_item:
                    logger.info(f"Stopping sample_id {item_id} in grouping {grouping}: CI width {width:.4f} < delta_item {delta_item} | epochs used: {trials}")
                    # Get the original sample_id value for this numeric ID
                    original_sample_id = df_grouping[df_grouping['sample_id_num'] == item_id][sample_id_column].iloc[0]
                    stop_sample_ids.append(f"{grouping}_{original_sample_id}")
                    break
                if len(ci_record) >= stab_window:
                    recent_widths = ci_record[-stab_window:]
                    slope = np.polyfit(range(len(recent_widths)), recent_widths, 1)[0]
                    ci_slopes_hist.append(slope)
                    slope_threshold = CI_delta / current_conservatism if df_item[score_column].mean() < low_perf_threshold else CI_delta
                    if (abs(slope) <= slope_threshold) and (len(ci_slopes_hist) >= 4):
                        recent_slopes = ci_slopes_hist[-3:]
                        slope_slopes = np.polyfit(range(len(recent_slopes)), recent_slopes, 1)[0]
                        if slope_slopes >= 0:
                            if df_item[score_column].mean() >= low_perf_threshold:
                                logger.info(f"Stopping sample_id {item_id} in grouping {grouping} due to CI stabilization: slope {slope:.6f} <= threshold {slope_threshold:.6f} | epochs used: {trials}")
                                original_sample_id = df_grouping[df_grouping['sample_id_num'] == item_id][sample_id_column].iloc[0]
                                stop_sample_ids.append(f"{grouping}_{original_sample_id}")
                                break
                            elif abs(slope) <= slope_threshold / 2:
                                logger.info(f"Stopping low-performance sample_id {item_id} in grouping {grouping} due to strong CI stabilization: slope {slope:.6f} | epochs used: {trials}")
                                original_sample_id = df_grouping[df_grouping['sample_id_num'] == item_id][sample_id_column].iloc[0]
                                stop_sample_ids.append(f"{grouping}_{original_sample_id}")
                                break
            sample_ci_records[item_id] = ci_record
            sample_ci_slopes[item_id] = ci_slopes_hist
            used_reps_dfs[item_id] = pd.DataFrame(used_reps, columns=df_item.columns)
            item_summaries.append({'successes': successes, 'trials': trials})
            
            # Group-level stopping check (periodic)
            current_perf_estimate = np.sum([s['successes'] for s in item_summaries]) / np.sum([s['trials'] for s in item_summaries]) if item_summaries else 0
            current_conservatism = conservatism if current_perf_estimate < low_perf_threshold else 1.0
            if ((item_idx + 1) % pymc_refresh_every == 0) or (item_idx == len(item_ids) - 1):
                all_successes = np.array([s['successes'] for s in item_summaries])
                all_trials = np.array([s['trials'] for s in item_summaries])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with model:
                        pm.set_data({
                            "successes": all_successes,
                            "trials": all_trials,
                            "n_items": np.int64(len(all_successes))
                        })
                        with suppress_all_output():
                            trace = pm.sample(**sampling_kwargs)
                        with suppress_all_output():
                            theta_hdi = az.hdi(trace.posterior["Theta"], hdi_prob=cred_level)
                    hdi_indices = list(theta_hdi["Theta"].hdi.values)
                    try:
                        theta_values_lower = theta_hdi["Theta"].sel(hdi="lower").values
                        theta_lo = theta_values_lower.item() if theta_values_lower.size == 1 else theta_values_lower.flatten()[0]
                    except Exception:
                        theta_lo = theta_hdi["Theta"].values[..., 0].flatten()[0]
                    try:
                        theta_values_upper = theta_hdi["Theta"].sel(hdi="upper").values
                        theta_hi = theta_values_upper.item() if theta_values_upper.size == 1 else theta_values_upper.flatten()[0]
                    except Exception:
                        theta_hi = theta_hdi["Theta"].values[..., 1].flatten()[0]
                    theta_width = theta_hi - theta_lo
                    CI_record.append(theta_width)
                    effective_width = theta_width * current_conservatism if current_perf_estimate < low_perf_threshold else theta_width
                    if effective_width < delta_cap:
                        logger.info(f"Stopping grouping {grouping}: CI width {effective_width:.4f} < delta_cap {delta_cap} | sample_ids used: {len(item_summaries)}")
                        stop_this_grouping.append(grouping)
                        break
                    if len(CI_record) >= stab_window:
                        recent_widths = CI_record[-stab_window:]
                        slope = np.polyfit(range(len(recent_widths)), recent_widths, 1)[0]
                        CI_slopes_hist.append(slope)
                        slope_threshold = CI_delta / current_conservatism if current_perf_estimate < low_perf_threshold else CI_delta
                        if (abs(slope) <= slope_threshold) and (len(CI_slopes_hist) >= 4):
                            recent_slopes = CI_slopes_hist[-3:]
                            slope_slopes = np.polyfit(range(len(recent_slopes)), recent_slopes, 1)[0]
                            if slope_slopes >= 0:
                                if current_perf_estimate >= low_perf_threshold:
                                    logger.info(f"Stopping grouping {grouping} due to CI stabilization: slope {slope:.6f} <= threshold {slope_threshold:.6f} | sample_ids used: {len(item_summaries)}")
                                    stop_this_grouping.append(grouping)
                                    break
                                elif abs(slope) <= slope_threshold / 2:
                                    logger.info(f"Stopping low-performance grouping {grouping} due to strong CI stabilization: slope {slope:.6f} | sample_ids used: {len(item_summaries)}")
                                    stop_this_grouping.append(grouping)
                                    break
        
        return {
            'grouping': grouping,
            'stop_sample_ids': stop_sample_ids,
            'stop_this_grouping': stop_this_grouping
        }
    except Exception as e:
        logger.error(f"Error processing grouping {grouping}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'grouping': grouping,
            'stop_sample_ids': [],
            'stop_this_grouping': []
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def _get_logfile_path(default='optstop_run.log'):
    import logging
    for handler in logging.getLogger().handlers:
        if hasattr(handler, 'baseFilename'):
            return handler.baseFilename
    return default

# --- Post-hoc mode ---
def optimal_stopping_posthoc(df: pd.DataFrame, params: Dict[str, Any], grouping_columns: List[str], sample_id_column: str, epoch_column: str, score_column: str = "score", display_progress: bool = True, generate_diagnostics: bool = False, diagnostics_prefix: str = "optstop_diagnostics", gpu_ids: Optional[List[int]] = None, max_workers: Optional[int] = None) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Run optimal stopping in post-hoc mode on a full dataset, parallelizing across groupings.
    The user must specify:
      - grouping_columns: list of column names to combine for grouping (can be a single string or list of strings)
      - sample_id_column: column name for sample ID
      - epoch_column: column name for epoch/trial
      - score_column: column name for score (default: 'score')
      - display_progress: whether to show a progress bar (default True)
      - generate_diagnostics: whether to generate diagnostic plots comparing full vs pruned datasets (default False)
      - diagnostics_prefix: prefix for diagnostic output files (default "optstop_diagnostics")
      - gpu_ids: List of GPU IDs to use for parallel processing. If None, uses CPU-only. If provided, assigns GPUs to workers cyclically.
      - max_workers: Number of parallel workers. If None, uses len(gpu_ids) when GPUs specified, otherwise uses CPU count.
    Returns a pruned DataFrame and a list of summary dicts (one per grouping-task).
    """
    # Input validation
    if isinstance(grouping_columns, str):
        grouping_columns = [grouping_columns]
    required = set(grouping_columns + [sample_id_column, epoch_column])
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        return pd.DataFrame(columns=df.columns), []
    original_columns = list(df.columns)  # Track original columns
    df = df.copy()
    df['grouping'] = df[grouping_columns].astype(str).agg('-'.join, axis=1)
    df['grouping_num'] = df['grouping'].astype('category').cat.codes
    df['sample_id_num'] = df[sample_id_column].astype('category').cat.codes
    df['epoch_num'] = df[epoch_column].astype(int)
    _validate_params(params)
    logger = logging.getLogger('optstop.posthoc')
    logger.info('Starting post-hoc optimal stopping')

    # Validate and configure GPU settings
    validated_gpu_ids, validated_max_workers = gpu_utils.validate_gpu_configuration(gpu_ids, max_workers)

    # Environment configuration is now handled by worker initializers only
    # Remove global configuration to prevent race conditions
    groupings = list(df.groupby(['grouping_num']))
    if not groupings:
        logger.info('No groupings to process; returning empty DataFrame and summary.')
        return pd.DataFrame(columns=df.columns), []

    args_list = [(pid, df_part, params, score_column) for pid, df_part in groupings]

    # Determine final worker count
    if validated_max_workers is None:
        final_max_workers = min(len(args_list), os.cpu_count() or 1)
    else:
        final_max_workers = min(validated_max_workers, len(args_list))

    # Create worker base directory with managed cleanup
    with cleanup_utils.managed_temp_dir(prefix='optstop_workers_') as worker_base_dir:
        # Create worker initialization arguments with GPU assignment
        worker_init_args = gpu_utils.create_worker_initargs(worker_base_dir, validated_gpu_ids, suppress_output=True)

        if validated_gpu_ids:
            logger.info(f'Using {final_max_workers} workers with GPU assignment: {validated_gpu_ids}')
        else:
            logger.info(f'Using {final_max_workers} CPU-only workers for {len(args_list)} groupings')

        results = []

        # Use spawn method to ensure clean processes without shared PyTensor state
        import multiprocessing as mp
        original_start_method = mp.get_start_method()

        try:
            # Force spawn method for clean process isolation
            if mp.get_start_method() != 'spawn':
                mp.set_start_method('spawn', force=True)

            with concurrent.futures.ProcessPoolExecutor(max_workers=final_max_workers) as executor:
                # Submit tasks with individual worker initialization
                futures = []
                for i, task_args in enumerate(args_list):
                    # Get worker initialization args cyclically
                    worker_args = worker_init_args[i % len(worker_init_args)]
                    # Create a new process with specific GPU assignment
                    future = executor.submit(_process_posthoc_grouping_with_init, task_args, worker_args)
                    futures.append(future)

                # Collect results
                iterator = concurrent.futures.as_completed(futures)
                if display_progress:
                    iterator = tqdm(iterator, total=len(futures), desc="Post-hoc optimal stopping")

                for future in iterator:
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"Worker task failed: {e}")
                        # Add empty result to maintain consistency
                        results.append({
                            'grouping': None, 'n_items_used': None, 'theta_ci_low': None,
                            'theta_ci_high': None, 'theta_ci_width': None, 'percent_items_used': None,
                            'avg_reps_per_item': None, 'used_reps_dfs': [], 'error': str(e)
                        })

        finally:
            # Restore original start method
            if original_start_method != mp.get_start_method():
                try:
                    mp.set_start_method(original_start_method, force=True)
                except RuntimeError:
                    # Start method can only be set once, ignore if already set
                    pass

        final_used_data = []
        participant_results = []
        def to_native(val):
            if hasattr(val, 'item') and callable(val.item):
                return val.item()
            if isinstance(val, (np.generic, np.ndarray)):
                return val.tolist() if hasattr(val, 'shape') and val.shape else float(val)
            return val

        for res in results:
            for used in res['used_reps_dfs']:
                final_used_data.append(used)
            participant_results.append({
                'grouping': to_native(res['grouping']),
                'n_items_used': to_native(res['n_items_used']),
                'theta_ci_low': to_native(res['theta_ci_low']),
                'theta_ci_high': to_native(res['theta_ci_high']),
                'theta_ci_width': to_native(res['theta_ci_width']),
                'percent_items_used': to_native(res['percent_items_used']),
                'avg_reps_per_item': to_native(res['avg_reps_per_item']),
                'error': res['error']
            })

        if final_used_data:
            final_used_df = pd.concat(final_used_data, ignore_index=True)
            # Only return the original columns, in the original order
            final_used_df = final_used_df.loc[:, [col for col in original_columns if col in final_used_df.columns]]
        else:
            final_used_df = pd.DataFrame(columns=original_columns)

        # Generate diagnostic plots if requested
        if generate_diagnostics and not df.empty and not final_used_df.empty:
            logger.info('Generating diagnostic plots...')
            try:
                # Create a copy of the original data with the same column structure as the pruned data
                original_df = df.copy()
                # Ensure both DataFrames have the same column structure for comparison
                common_columns = list(set(original_columns) & set(final_used_df.columns))
                if common_columns:
                    _generate_diagnostic_plots(
                        original_df[common_columns],
                        final_used_df[common_columns],
                        diagnostics_prefix
                    )
                else:
                    logger.warning("No common columns between original and pruned data for diagnostics")
            except Exception as e:
                logger.error(f"Failed to generate diagnostic plots: {e}")

        logger.info('Post-hoc optimal stopping complete')
        # Only print if in main process
        if os.getpid() == getattr(os, 'getppid', lambda: None)() or hasattr(sys, 'ps1'):
            print(f"Run complete. See the log file for details: {_get_logfile_path()}")
        return final_used_df, participant_results

# --- Live mode ---
def optimal_stopping_live(df: pd.DataFrame, params: Dict[str, Any], grouping_columns: List[str], sample_id_column: str, epoch_column: str, score_column: str = "score", display_progress: bool = True, gpu_ids: Optional[List[int]] = None, max_workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Run optimal stopping in live mode on current data for multiple groupings, parallelizing across groupings.
    The user must specify:
      - grouping_columns: list of column names to combine for grouping (can be a single string or list of strings)
      - sample_id_column: column name for sample ID
      - epoch_column: column name for epoch/trial
      - score_column: column name for score (default: 'score')
      - display_progress: whether to show a progress bar (default True)
      - gpu_ids: List of GPU IDs to use for parallel processing. If None, uses CPU-only. If provided, assigns GPUs to workers cyclically.
      - max_workers: Number of parallel workers. If None, uses len(gpu_ids) when GPUs specified, otherwise uses CPU count.
    Returns a dict with 'stop_sample_ids' (list of "grouping_sample_id" strings) and 'stop_task' (list of grouping names that have reached stopping criteria).
    """
    import sys, io
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        # Input validation
        if isinstance(grouping_columns, str):
            grouping_columns = [grouping_columns]
        required = set(grouping_columns + [sample_id_column, epoch_column])
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if df.empty:
            return {'stop_sample_ids': [], 'stop_task': []}
        
        df = df.copy()
        df['grouping'] = df[grouping_columns].astype(str).agg('-'.join, axis=1)
        _validate_params(params)
        logger = logging.getLogger('optstop.live')
        logger.info('Starting live optimal stopping')

        # Validate and configure GPU settings
        validated_gpu_ids, validated_max_workers = gpu_utils.validate_gpu_configuration(gpu_ids, max_workers)

        # Environment configuration is now handled by worker initializers only
        # Remove global configuration to prevent race conditions

        # Get unique groupings
        all_groupings = df['grouping'].unique()
        if not all_groupings.size:
            logger.info('No groupings to process; returning empty results.')
            return {'stop_sample_ids': [], 'stop_task': []}

        # Prepare arguments for parallel processing
        args_list = []
        for grouping in all_groupings:
            df_grouping = df[df['grouping'] == grouping].copy()
            if not df_grouping.empty:
                args_list.append((grouping, df_grouping, params, sample_id_column, epoch_column, score_column, grouping_columns))

        if not args_list:
            logger.info('No valid groupings to process; returning empty results.')
            return {'stop_sample_ids': [], 'stop_task': []}

        # Determine final worker count
        if validated_max_workers is None:
            final_max_workers = min(len(args_list), os.cpu_count() or 1)
        else:
            final_max_workers = min(validated_max_workers, len(args_list))

        # Create worker base directory with managed cleanup
        with cleanup_utils.managed_temp_dir(prefix='optstop_live_workers_') as worker_base_dir:
            # Create worker initialization arguments with GPU assignment
            worker_init_args = gpu_utils.create_worker_initargs(worker_base_dir, validated_gpu_ids, suppress_output=True)

            if validated_gpu_ids:
                logger.info(f'Using {final_max_workers} workers with GPU assignment: {validated_gpu_ids}')
            else:
                logger.info(f'Using {final_max_workers} CPU-only workers for {len(args_list)} groupings')

            # Process groupings in parallel
            results = []

            # Use spawn method to ensure clean processes without shared PyTensor state
            import multiprocessing as mp
            original_start_method = mp.get_start_method()

            try:
                # Force spawn method for clean process isolation
                if mp.get_start_method() != 'spawn':
                    mp.set_start_method('spawn', force=True)

                with concurrent.futures.ProcessPoolExecutor(max_workers=final_max_workers) as executor:
                    # Submit tasks with individual worker initialization
                    futures = []
                    for i, task_args in enumerate(args_list):
                        # Get worker initialization args cyclically
                        worker_args = worker_init_args[i % len(worker_init_args)]
                        # Create a new process with specific GPU assignment
                        future = executor.submit(_process_live_grouping_with_init, task_args, worker_args)
                        futures.append(future)

                    # Collect results
                    iterator = concurrent.futures.as_completed(futures)
                    if display_progress:
                        iterator = tqdm(iterator, total=len(futures), desc="Live optimal stopping")

                    for future in iterator:
                        try:
                            results.append(future.result())
                        except Exception as e:
                            logger.error(f"Worker task failed: {e}")
                            # Add empty result to maintain consistency
                            results.append({
                                'grouping': None,
                                'stop_sample_ids': [],
                                'stop_this_grouping': []
                            })

            finally:
                # Restore original start method
                if original_start_method != mp.get_start_method():
                    try:
                        mp.set_start_method(original_start_method, force=True)
                    except RuntimeError:
                        # Start method can only be set once, ignore if already set
                        pass

            # Aggregate results
            stop_sample_ids = []
            stop_task_groupings = []

            for res in results:
                stop_sample_ids.extend(res['stop_sample_ids'])
                stop_task_groupings.extend(res['stop_this_grouping'])

            logger.info('Live optimal stopping complete')
            if os.getpid() == getattr(os, 'getppid', lambda: None)() or hasattr(sys, 'ps1'):
                print(f"Run complete. See the log file for details: {_get_logfile_path()}")
            return {'stop_sample_ids': stop_sample_ids, 'stop_task': stop_task_groupings}
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr 