"""
Post-hoc convergence analysis for optimal stopping algorithms.

This module provides functions to assess convergence properties of optimal stopping algorithms
by analyzing how stopping criteria evolve across different data collection scenarios.
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
warnings.filterwarnings('ignore', message='.*target_accept.*')
warnings.filterwarnings('ignore', message='.*reparameterize.*')
warnings.filterwarnings('ignore', message='.*smaller than 100.*')
warnings.filterwarnings('ignore', message='.*needed for reliable.*')

# --- Helper: Adaptive Beta CI ---
def _beta_ci_adaptive(successes, trials, cred_level=0.95, conservatism=1.0, 
                     low_perf_threshold=0.2, base_strength=2, samples=10000):
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

# Note: Column validation is now done at the beginning of convergence_posthoc function

def _validate_params(params):
    logger = logging.getLogger('optstop.convergence')
    for key in ['draws', 'tune', 'stab_window', 'rep_batch_size', 'pymc_refresh_every', 'item_seqs', 'epoch_seqs']:
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

# --- Helper: Process a single grouping-task ---
def _process_grouping(args):
    import sys, io
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        (pid, df_part, params, score_column) = args
        logger = logging.getLogger('optstop.convergence')
        delta_item = params.get('delta_item', 0.05)
        delta_cap = params.get('delta_cap', 0.05)
        CI_delta = params.get('CI_delta', 0.0002)
        cred_level = params.get('cred_level', 0.95)
        conservatism = params.get('conservatism', 5)
        low_perf_threshold = params.get('low_performance_threshold', 0.05)
        draws = params.get('draws', 3000)
        tune = params.get('tune', 3000)
        chains = params.get('chains', 4)
        cores = params.get('cores', 4)
        rep_batch_size = params.get('rep_batch_size', 1)
        pymc_refresh_every = params.get('pymc_refresh_every', 2)
        stab_window = params.get('stab_window', 5)
        item_seqs = params.get('item_seqs', 20)
        epoch_seqs = params.get('epoch_seqs', 20)
        # Set random seed if provided
        if 'random_seed' in params:
            np.random.seed(params['random_seed'])
        logger.info(f"Processing grouping {pid}")

        # Initialize all output variables
        items_fin_CI_widths = []
        items_fin_CI_slopes = []
        items_fin_slope_slopes = []
        items_fin_scores = []
        items_shortfalls = []
        epochs_fin_CI_widths = []
        epochs_fin_CI_slopes = []
        epochs_fin_slope_slopes = []
        epochs_fin_scores = []
        epochs_shortfalls = []
        agg_sample_id_performance = []
        theta_lo = theta_hi = theta_width = None
        n_items_used = 0
        percent_items_used = 0
        avg_reps_per_item = 0
        task_performance = None
        mean_sample_id_performance = None
        var_sample_id_performance = None

        for item_seq in range(item_seqs):
            item_summaries = []
            used_reps_dfs = []
            CI_record = []
            CI_slopes_hist = []
            CI_slope_slopes = []
            item_scores = []
            item_shortfalls = []
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
            np.random.shuffle(item_ids)
            for item_idx, item_id in enumerate(item_ids):
                df_item = df_part[df_part['sample_id_num'] == item_id].sort_values('epoch_num')
                sample_ID_performances = []
                for epoch_seq in range(epoch_seqs):
                    if epoch_seqs >= 1:
                        # Use sample to shuffle the DataFrame instead of set_index/reindex
                        df_item_shuffled = df_item.sample(frac=1.0, random_state=np.random.randint(0, 10000)).reset_index(drop=True)
                        successes = 0
                        trials = 0
                        used_reps = []
                        epoch_CI_widths = []
                        epoch_CI_slopes = []
                        epoch_slope_slopes = []
                        epoch_scores = []
                        epoch_shortfalls = []
                        slope = None  # Ensure slope is always defined
                        slope_slopes = None
                        for start in range(0, len(df_item_shuffled), rep_batch_size):
                            batch = df_item_shuffled.iloc[start:start+rep_batch_size]
                            successes += batch[score_column].sum()
                            trials += len(batch)
                            used_reps.extend(batch.itertuples(index=False))
                            lo, hi, width = _beta_ci_adaptive(
                                successes, trials, 
                                cred_level=cred_level,
                                conservatism=current_conservatism,
                                low_perf_threshold=low_perf_threshold
                            )
                            epoch_CI_widths.append(width)
                            curr_perf_estimate = successes / trials
                            if len(epoch_CI_widths) >= stab_window:
                                recent_widths = epoch_CI_widths[-stab_window:]
                                slope = np.polyfit(range(len(recent_widths)), recent_widths, 1)[0]
                                epoch_CI_slopes.append(slope)
                                slope_threshold = CI_delta / current_conservatism if curr_perf_estimate < low_perf_threshold else CI_delta
                                if (len(epoch_CI_slopes) >= 3):
                                    recent_slopes = epoch_CI_slopes[-3:]
                                    slope_slopes = np.polyfit(range(len(recent_slopes)), recent_slopes, 1)[0]
                                    epoch_slope_slopes.append(slope_slopes)
                                # Only check slope/slopes if they are set
                                if (width < delta_item) or (slope is not None and slope_slopes is not None and abs(slope) <= slope_threshold and len(epoch_CI_slopes) >= 4 and slope_slopes >= 0):
                                    epoch_scores.append(1)
                                else:
                                    epoch_scores.append(0)
                            else:
                                slope = None
                                slope_slopes = None
                            if (start + rep_batch_size >= len(df_item_shuffled)):
                                slope_threshold = CI_delta / current_conservatism if curr_perf_estimate < low_perf_threshold else CI_delta
                                if (epoch_seq == epoch_seqs - 1) and (item_seq == item_seqs - 1):
                                    agg_sample_id_performance.append(curr_perf_estimate)
                                if epoch_scores and epoch_scores[-1] == 1:
                                    epoch_shortfalls.append(0)
                                else:
                                    next_width = epoch_CI_widths[-1]
                                    next_slope = epoch_CI_slopes[-1] if epoch_CI_slopes else 0
                                    next_slope_slope = epoch_slope_slopes[-1] if epoch_slope_slopes else 0
                                    not_converged = True
                                    trials_needed = 0
                                    while not_converged:
                                        trials_needed += rep_batch_size
                                        next_slope = min(0,(next_slope + next_slope_slope))
                                        next_width = max(0, (next_width + next_slope))
                                        if (next_width < delta_item) or (abs(next_slope) <= slope_threshold):
                                            not_converged = False
                                            epoch_shortfalls.append(trials_needed)
                                        if trials_needed > 200:
                                            epoch_shortfalls.append(200)
                                            not_converged = False
                        epochs_fin_CI_widths.append(epoch_CI_widths[-1] if epoch_CI_widths else 0)
                        epochs_fin_CI_slopes.append(epoch_CI_slopes[-1] if epoch_CI_slopes else 0)
                        epochs_fin_slope_slopes.append(epoch_slope_slopes[-1] if epoch_slope_slopes else 0)
                        epochs_fin_scores.append(epoch_scores[-1] if epoch_scores else 0)
                        epochs_shortfalls.append(epoch_shortfalls[-1] if epoch_shortfalls else 0)
                        sample_ID_performances.append(successes / trials if trials > 0 else 0)
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
                            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                                trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, progressbar=False, target_accept=0.97)
                            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
                        effective_width = theta_width
                        if current_perf_estimate < low_perf_threshold:
                            effective_width = theta_width * current_conservatism
                        if len(CI_record) >= stab_window:
                            recent_widths = CI_record[-stab_window:]
                            slope = np.polyfit(range(len(recent_widths)), recent_widths, 1)[0]
                            CI_slopes_hist.append(slope)
                            slope_threshold = CI_delta / current_conservatism if current_perf_estimate < low_perf_threshold else CI_delta
                            if (len(CI_slopes_hist) >= 2):
                                recent_slopes = CI_slopes_hist[-3:]
                                slope_slopes = np.polyfit(range(len(recent_slopes)), recent_slopes, 1)[0]
                                CI_slope_slopes.append(slope_slopes)
                        else:
                            slope = None
                            slope_slopes = None
                        # Only check slope/slopes if they are set
                        if (effective_width < delta_cap) or (slope is not None and slope_slopes is not None and abs(slope) <= slope_threshold and len(CI_slopes_hist) >= 2 and slope_slopes >= 0):
                            item_scores.append(1)
                        else:
                            item_scores.append(0)
                        if (item_idx == len(item_ids) - 1):
                            slope_threshold = CI_delta / current_conservatism if current_perf_estimate < low_perf_threshold else CI_delta
                            task_performance = current_perf_estimate
                            if item_scores and item_scores[-1] == 1:
                                item_shortfalls.append(0)
                            else:
                                next_width = CI_record[-1]
                                next_slope = CI_slopes_hist[-1] if CI_slopes_hist else 0
                                next_slope_slope = CI_slope_slopes[-1] if CI_slope_slopes else 0
                                not_converged = True
                                items_needed = 0
                                while not_converged:
                                    items_needed += pymc_refresh_every
                                    next_slope = min(0,(next_slope + next_slope_slope))
                                    next_width = max(0, (next_width + next_slope))
                                    if (next_width < delta_item) or (abs(next_slope) <= slope_threshold):
                                        not_converged = False
                                        item_shortfalls.append(items_needed)
                                    if items_needed > 200:
                                        item_shortfalls.append(200)
                                        not_converged = False
            items_fin_CI_widths.append(CI_record[-1] if CI_record else 0)
            items_fin_CI_slopes.append(CI_slopes_hist[-1] if CI_slopes_hist else 0)
            items_fin_slope_slopes.append(CI_slope_slopes[-1] if CI_slope_slopes else 0)
            items_fin_scores.append(item_scores[-1] if item_scores else 0)
            items_shortfalls.append(item_shortfalls[-1] if item_shortfalls else 0)

        mean_fin_CI_width_item = np.mean(items_fin_CI_widths) if items_fin_CI_widths else None
        var_fin_CI_width_item = np.var(items_fin_CI_widths) if items_fin_CI_widths else None
        mean_fin_CI_slope_item = np.mean(items_fin_CI_slopes) if items_fin_CI_slopes else None  
        var_fin_CI_slope_item = np.var(items_fin_CI_slopes) if items_fin_CI_slopes else None
        mean_fin_slope_slope_item = np.mean(items_fin_slope_slopes) if items_fin_slope_slopes else None
        var_fin_slope_slope_item = np.var(items_fin_slope_slopes) if items_fin_slope_slopes else None
        mean_needed_items = np.mean(items_shortfalls) if items_shortfalls else None
        var_needed_items = np.var(items_shortfalls) if items_shortfalls else None
        mean_fin_CI_width_epoch = np.mean(epochs_fin_CI_widths) if epochs_fin_CI_widths else None
        var_fin_CI_width_epoch = np.var(epochs_fin_CI_widths) if epochs_fin_CI_widths else None
        mean_fin_CI_slope_epoch = np.mean(epochs_fin_CI_slopes) if epochs_fin_CI_slopes else None
        var_fin_CI_slope_epoch = np.var(epochs_fin_CI_slopes) if epochs_fin_CI_slopes else None
        mean_fin_slope_slope_epoch = np.mean(epochs_fin_slope_slopes) if epochs_fin_slope_slopes else None
        var_fin_slope_slope_epoch = np.var(epochs_fin_slope_slopes) if epochs_fin_slope_slopes else None
        mean_needed_epochs = np.mean(epochs_shortfalls) if epochs_shortfalls else None
        var_needed_epochs = np.var(epochs_shortfalls) if epochs_shortfalls else None
        mean_items_fin_score = np.mean(items_fin_scores) if items_fin_scores else None
        var_items_fin_score = np.var(items_fin_scores) if items_fin_scores else None
        mean_epochs_fin_score = np.mean(epochs_fin_scores) if epochs_fin_scores else None
        var_epochs_fin_score = np.var(epochs_fin_scores) if epochs_fin_scores else None
        mean_sample_id_performance = np.mean(agg_sample_id_performance) if agg_sample_id_performance else None
        var_sample_id_performance = np.var(agg_sample_id_performance) if agg_sample_id_performance else None

        # Calculate n_items_used, percent_items_used, avg_reps_per_item
        n_items_used = len(item_summaries)
        percent_items_used = n_items_used / len(df_part['sample_id_num'].unique()) if len(df_part['sample_id_num'].unique()) > 0 else 0
        avg_reps_per_item = np.mean([s['trials'] for s in item_summaries]) if item_summaries else 0

        # theta_lo, theta_hi, theta_width are from the last PyMC run
        # task_performance is set above

        result = {
            'grouping': pid,
            'group_label': df_part['grouping'].iloc[0] if 'grouping' in df_part.columns else str(pid),
            'n_items_used': n_items_used,
            'theta_ci_low': theta_lo,
            'theta_ci_high': theta_hi,
            'theta_ci_width': theta_width,
            'percent_items_used': percent_items_used,
            'avg_reps_per_item': avg_reps_per_item,
            'mean_fin_CI_width_item': mean_fin_CI_width_item,
            'var_fin_CI_width_item': var_fin_CI_width_item,
            'mean_fin_CI_slope_item': mean_fin_CI_slope_item,
            'var_fin_CI_slope_item': var_fin_CI_slope_item,
            'mean_fin_slope_slope_item': mean_fin_slope_slope_item,
            'var_fin_slope_slope_item': var_fin_slope_slope_item,
            'mean_needed_items': mean_needed_items,
            'var_needed_items': var_needed_items,
            'mean_items_fin_score': mean_items_fin_score,
            'var_items_fin_score': var_items_fin_score,
            'mean_fin_CI_width_epoch': mean_fin_CI_width_epoch,
            'var_fin_CI_width_epoch': var_fin_CI_width_epoch,
            'mean_fin_CI_slope_epoch': mean_fin_CI_slope_epoch,
            'var_fin_CI_slope_epoch': var_fin_CI_slope_epoch,
            'mean_fin_slope_slope_epoch': mean_fin_slope_slope_epoch,
            'var_fin_slope_slope_epoch': var_fin_slope_slope_epoch,
            'mean_needed_epochs': mean_needed_epochs,
            'var_needed_epochs': var_needed_epochs,
            'mean_epochs_fin_score': mean_epochs_fin_score,
            'var_epochs_fin_score': var_epochs_fin_score,
            'task_performance': task_performance,
            'mean_sample_id_performance': mean_sample_id_performance,
            'var_sample_id_performance': var_sample_id_performance
        }
        result['error'] = None
        return result
    except Exception as e:
        logger.error(f"Error processing grouping {pid}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'grouping': pid,
            'group_label': df_part['grouping'].iloc[0] if 'grouping' in df_part.columns else str(pid),
            'n_items_used': None,
            'theta_ci_low': None,
            'theta_ci_high': None,
            'theta_ci_width': None,
            'percent_items_used': None,
            'avg_reps_per_item': None,
            'mean_fin_CI_width_item': None,
            'var_fin_CI_width_item': None,
            'mean_fin_CI_slope_item': None,
            'var_fin_CI_slope_item': None,
            'mean_fin_slope_slope_item': None,
            'var_fin_slope_slope_item': None,
            'mean_needed_items': None,
            'var_needed_items': None,
            'mean_items_fin_score': None,
            'var_items_fin_score': None,
            'mean_fin_CI_width_epoch': None,
            'var_fin_CI_width_epoch': None,
            'mean_fin_CI_slope_epoch': None,
            'var_fin_CI_slope_epoch': None,
            'mean_fin_slope_slope_epoch': None,
            'var_fin_slope_slope_epoch': None,
            'mean_needed_epochs': None,
            'var_needed_epochs': None,
            'mean_epochs_fin_score': None,
            'var_epochs_fin_score': None,
            'task_performance': None,
            'mean_sample_id_performance': None,
            'var_sample_id_performance': None,
            'error': str(e)
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def _get_logfile_path_convergence(default='optstop_convergence.log'):
    import logging
    for handler in logging.getLogger().handlers:
        if hasattr(handler, 'baseFilename'):
            return handler.baseFilename
    return default

# --- Main API ---
def convergence_posthoc(df: pd.DataFrame, params: dict, grouping_columns: List[str], sample_id_column: str, epoch_column: str, score_column: str = "score", display_progress: bool = True, generate_diagnostics: bool = True, diagnostics_prefix: str = "convergence_eval"):
    """
    Post-hoc convergence analysis, parallelized across groupings.
    The user must specify:
      - grouping_columns: list of column names to combine for grouping (can be a single string or list of strings)
      - sample_id_column: column name for sample ID
      - epoch_column: column name for epoch/trial
      - score_column: column name for score (default: 'score')
      - display_progress: whether to show a progress bar (default True)
      - generate_diagnostics: whether to generate diagnostic figures (default True)
      - diagnostics_prefix: prefix for diagnostic output files
    Returns a DataFrame of convergence statistics.
    """
    # Input validation
    if isinstance(grouping_columns, str):
        grouping_columns = [grouping_columns]
    required = set(grouping_columns + [sample_id_column, epoch_column])
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        return pd.DataFrame(columns=df.columns)
    df = df.copy()
    df['grouping'] = df[grouping_columns].astype(str).agg('-'.join, axis=1)
    df['grouping_num'] = df['grouping'].astype('category').cat.codes
    df['sample_id_num'] = df[sample_id_column].astype('category').cat.codes
    df['epoch_num'] = df[epoch_column].astype(int)
    
    logger = logging.getLogger('optstop.convergence')
    _validate_params(params)
    logger.info('Starting post-hoc convergence analysis')
    groupings = list(df.groupby(['grouping_num']))
    if not groupings:
        logger.info('No groupings to process; returning empty DataFrame.')
        return pd.DataFrame(columns=df.columns)
    args_list = [(pid, df_part, params, score_column) for pid, df_part in groupings]
    max_workers = min(len(args_list), os.cpu_count() or 1)
    logger.info(f'Using {max_workers} parallel workers for {len(args_list)} groupings')
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        iterator = executor.map(_process_grouping, args_list)
        if display_progress:
            iterator = tqdm(iterator, total=len(args_list), desc="Convergence analysis")
        for res in iterator:
            results.append(res)
    logger.info('Convergence analysis complete')
    output_df = pd.DataFrame(results)
    if generate_diagnostics:
        try:
            generate_convergence_diagnostics(output_df, out_prefix=diagnostics_prefix)
        except Exception as e:
            logger = logging.getLogger('optstop.convergence')
            logger.error(f"Failed to generate convergence diagnostics: {e}")
    # Only print if in main process
    if os.getpid() == getattr(os, 'getppid', lambda: None)() or hasattr(sys, 'ps1'):
        print(f"Run complete. See the log file for details: {_get_logfile_path_convergence()}")
    return output_df

def generate_convergence_diagnostics(convergence_data: pd.DataFrame, out_prefix: str = "convergence_eval"):
    """
    Generate diagnostic figures for convergence_posthoc output.
    Saves two PNGs: {out_prefix}_grouped_needed.png and {out_prefix}_score_scatter.png
    """
    import numpy as np
    import logging
    logger = logging.getLogger('optstop.convergence.diagnostics')
    try:
        if convergence_data.empty:
            logger.warning("Convergence diagnostics: input DataFrame is empty.")
            fig = plt.figure(figsize=(8, 4))
            fig.suptitle("No data for convergence diagnostics", fontsize=14)
            fig.savefig(f"{out_prefix}_grouped_needed.png", dpi=150)
            fig.savefig(f"{out_prefix}_score_scatter.png", dpi=150)
            return
        n_samples = 20  # Used for SEM calculation, as in the script
        # Calculate standard error from variances (SEM = sqrt(variance)/sqrt(n))
        convergence_data['sem_needed_items'] = np.sqrt(convergence_data['var_needed_items'] / n_samples)
        convergence_data['sem_needed_epochs'] = np.sqrt(convergence_data['var_needed_epochs'] / n_samples)
        # --- NEW GROUPED NEEDED PLOT ---
        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
        # Top: mean_needed_items by grouping
        items_by_group = convergence_data[['group_label', 'mean_needed_items', 'sem_needed_items']].dropna()
        items_by_group = items_by_group.groupby('group_label').mean().reset_index()
        items_by_group = items_by_group.sort_values('mean_needed_items', ascending=False)
        y_labels_items = items_by_group['group_label'].astype(str)
        y_pos_items = np.arange(len(items_by_group))
        ax1.errorbar(items_by_group['mean_needed_items'], y_pos_items, xerr=items_by_group['sem_needed_items'], fmt='o', color='skyblue', ecolor='gray', capsize=4)
        ax1.set_yticks(y_pos_items)
        ax1.set_yticklabels(y_labels_items)
        ax1.set_xlabel('Mean Needed Items (with std error)')
        ax1.set_ylabel('Grouping')
        ax1.set_title('Mean Needed Items by Grouping')
        ax1.invert_yaxis()
        # Bottom: mean_needed_epochs by grouping
        epochs_by_group = convergence_data[['group_label', 'mean_needed_epochs', 'sem_needed_epochs']].dropna()
        epochs_by_group = epochs_by_group.groupby('group_label').mean().reset_index()
        epochs_by_group = epochs_by_group.sort_values('mean_needed_epochs', ascending=False)
        y_labels_epochs = epochs_by_group['group_label'].astype(str)
        y_pos_epochs = np.arange(len(epochs_by_group))
        ax2.errorbar(epochs_by_group['mean_needed_epochs'], y_pos_epochs, xerr=epochs_by_group['sem_needed_epochs'], fmt='o', color='lightgreen', ecolor='gray', capsize=4)
        ax2.set_yticks(y_pos_epochs)
        ax2.set_yticklabels(y_labels_epochs)
        ax2.set_xlabel('Mean Needed Epochs (with std error)')
        ax2.set_ylabel('Grouping')
        ax2.set_title('Mean Needed Epochs by Grouping')
        ax2.invert_yaxis()
        plt.tight_layout()
        fig1.savefig(f"{out_prefix}_grouped_needed.png", dpi=300)
        # --- SCORE SCATTER PLOT (unchanged) ---
        fig2 = plt.figure(figsize=(16, 6))
        gs2 = gridspec.GridSpec(1, 2)
        ax3 = plt.subplot(gs2[0, 0])
        if 'task_performance' in convergence_data.columns and 'mean_needed_items' in convergence_data.columns:
            ax3.errorbar(convergence_data['mean_needed_items'], convergence_data['task_performance'],
                         xerr=convergence_data['sem_needed_items'], fmt='o', color='blue', 
                         alpha=0.5, ecolor='gray', capsize=3)
            ax3.set_xlabel('Mean Needed Items (with std error)')
            ax3.set_ylabel('Task Performance (Final Scoring)')
            ax3.set_title('Task Performance vs Mean Needed Items')
            ax3.set_xlim(left=0)
        else:
            ax3.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12)
        ax4 = plt.subplot(gs2[0, 1])
        if 'mean_sample_id_performance' in convergence_data.columns and 'mean_needed_epochs' in convergence_data.columns:
            ax4.errorbar(convergence_data['mean_needed_epochs'], convergence_data['mean_sample_id_performance'],
                         xerr=convergence_data['sem_needed_epochs'], fmt='o', color='green', 
                         alpha=0.5, ecolor='gray', capsize=3)
            ax4.set_xlabel('Mean Needed Epochs (with std error)')
            ax4.set_ylabel('Sample Performance (Final Scoring)')
            ax4.set_title('Sample Performance vs Mean Needed Epochs')
            ax4.set_xlim(left=0)
        else:
            ax4.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12)
        plt.tight_layout()
        fig2.savefig(f"{out_prefix}_score_scatter.png", dpi=300)
        logger.info(f"Convergence diagnostics saved to {out_prefix}_grouped_needed.png and {out_prefix}_score_scatter.png")
    except Exception as e:
        logger.error(f"Error generating convergence diagnostics: {e}")
        try:
            fig = plt.figure(figsize=(8, 4))
            fig.suptitle(f"Diagnostics failed: {e}", fontsize=14)
            fig.savefig(f"{out_prefix}_grouped_needed.png", dpi=150)
            fig.savefig(f"{out_prefix}_score_scatter.png", dpi=150)
        except Exception:
            pass
        raise 