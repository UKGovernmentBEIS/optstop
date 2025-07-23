"""
Adaptive optimal stopping rule algorithms for efficient data collection and analysis.

This module provides both post-hoc (batch) and live (incremental) optimal stopping algorithms.

Required DataFrame columns:
- grouping_num: Numeric identifier for the model/system being evaluated
- task_num: Numeric identifier for the evaluation task/domain
- sample_id_num: Numeric identifier for individual test items
- epoch: Order/sequence number of repetitions (starts at 1)
- score: Binary outcome (0/1) indicating success/failure for each trial
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import logging
import concurrent.futures
import os
from typing import Dict, Any, List, Tuple, Optional

REQUIRED_COLUMNS = ['grouping_num', 'task_num', 'sample_id_num', 'epoch', 'score']

def _check_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

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

def configure_optsampling_logging(logfile: str = 'optsampling_run.log', level: int = logging.INFO) -> None:
    """
    Configure logging for the optstop package to log to both console and a file.
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(ch)
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

def _process_posthoc_grouping(args: Tuple[Any, Any, pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
    pid, cap, df_part, params = args
    logger = logging.getLogger('optstop.posthoc')
    try:
        delta_item = params.get('delta_item', 0.05)
        delta_cap = params.get('delta_cap', 0.05)
        CI_delta = params.get('CI_delta', 0.0002)
        cred_level = params.get('cred_level', 0.95)
        conservatism = params.get('conservatism', 2)
        low_perf_threshold = params.get('low_performance_threshold', 0.1)
        draws = params.get('draws', 3000)
        tune = params.get('tune', 3000)
        rep_batch_size = params.get('rep_batch_size', 1)
        pymc_refresh_every = params.get('pymc_refresh_every', 2)
        stab_window = params.get('stab_window', 5)
        if 'random_seed' in params:
            np.random.seed(params['random_seed'])
        logger.info(f"Processing grouping {pid}, task {cap}")
        item_summaries = []
        used_reps_dfs = []
        theta_lo, theta_hi, theta_width = None, None, None
        CI_record = []
        CI_slopes_hist = []
        initial_perf = df_part['score'].mean()
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
            df_item = df_part[df_part['sample_id_num'] == item_id].sort_values('epoch')
            successes = 0
            trials = 0
            used_reps = []
            for start in range(0, len(df_item), rep_batch_size):
                batch = df_item.iloc[start:start+rep_batch_size]
                successes += batch['score'].sum()
                trials += len(batch)
                used_reps.extend(batch.itertuples(index=False))
                lo, hi, width = _beta_ci_adaptive(
                    successes, trials, cred_level=cred_level,
                    conservatism=current_conservatism,
                    low_perf_threshold=low_perf_threshold
                )
                if width < delta_item:
                    logger.info(f"Stopping sample_id {item_id} (group {pid}, task {cap}) at epoch {batch['epoch'].iloc[-1]}: CI width {width:.4f} < delta_item {delta_item} | epochs used: {trials}")
                    break
            item_summaries.append({'successes': successes, 'trials': trials})
            used_reps_dfs.append(pd.DataFrame(used_reps, columns=df_item.columns))
            current_perf_estimate = sum(s['successes'] for s in item_summaries) / sum(s['trials'] for s in item_summaries)
            current_conservatism = conservatism if current_perf_estimate < low_perf_threshold else 1.0
            if ((item_idx + 1) % pymc_refresh_every == 0) or (item_idx == len(item_ids) - 1):
                all_successes = np.array([s['successes'] for s in item_summaries])
                all_trials = np.array([s['trials'] for s in item_summaries])
                with model:
                    pm.set_data({
                        "successes": all_successes,
                        "trials": all_trials,
                        "n_items": np.int64(len(all_successes))
                    })
                    trace = pm.sample(draws=draws, tune=tune, chains=2, cores=1, progressbar=False, target_accept=0.97)
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
                        logger.info(f"Stopping task/grouping {pid}-{cap}: CI width {effective_width:.4f} < delta_cap {delta_cap} | sample_ids used: {len(item_summaries)}")
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
                                    logger.info(f"Stopping task/grouping {pid}-{cap} due to CI stabilization: slope {slope:.6f} <= threshold {slope_threshold:.6f} | sample_ids used: {len(item_summaries)}")
                                    break
                                elif abs(slope) <= slope_threshold / 2:
                                    logger.info(f"Stopping low-performance task/grouping {pid}-{cap} due to strong CI stabilization: slope {slope:.6f} | sample_ids used: {len(item_summaries)}")
                                    break
        avg_reps_per_item = np.mean([len(df) for df in used_reps_dfs]) if used_reps_dfs else 0
        result = {
            'grouping': pid,
            'task': cap,
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
        logger.error(f"Error processing grouping {pid}, task {cap}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'grouping': pid,
            'task': cap,
            'n_items_used': None,
            'theta_ci_low': None,
            'theta_ci_high': None,
            'theta_ci_width': None,
            'percent_items_used': None,
            'avg_reps_per_item': None,
            'used_reps_dfs': [],
            'error': str(e)
        }

# --- Post-hoc mode ---
def optimal_stopping_posthoc(df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Run optimal stopping in post-hoc mode on a full dataset, parallelizing across groupings.
    Returns a pruned DataFrame and a list of summary dicts (one per grouping-task).
    """
    _check_required_columns(df)
    _validate_params(params)
    logger = logging.getLogger('optstop.posthoc')
    logger.info('Starting post-hoc optimal stopping')
    groupings = list(df.groupby(['grouping_num', 'task_num']))
    if not groupings:
        logger.info('No groupings to process; returning empty DataFrame and summary.')
        return pd.DataFrame(), []
    args_list = [(pid, cap, df_part, params) for (pid, cap), df_part in groupings]
    max_workers = min(len(args_list), os.cpu_count() or 1)
    logger.info(f'Using {max_workers} parallel workers for {len(args_list)} groupings')
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(_process_posthoc_grouping, args_list):
            results.append(res)
    final_used_data = []
    participant_results = []
    for res in results:
        for used in res['used_reps_dfs']:
            final_used_data.append(used)
        def to_native(val):
            if hasattr(val, 'item') and callable(val.item):
                return val.item()
            if isinstance(val, (np.generic, np.ndarray)):
                return val.tolist() if hasattr(val, 'shape') and val.shape else float(val)
            return val
        participant_results.append({
            'grouping': to_native(res['grouping']),
            'task': to_native(res['task']),
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
    else:
        final_used_df = pd.DataFrame()
    logger.info('Post-hoc optimal stopping complete')
    return final_used_df, participant_results

# --- Live mode ---
def optimal_stopping_live(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run optimal stopping in live mode on current data for a task/grouping.
    Returns a dict with 'stop_sample_ids' and 'stop_task'.
    """
    _check_required_columns(df)
    _validate_params(params)
    logger = logging.getLogger('optstop.live')
    logger.info('Starting live optimal stopping')
    delta_item = params.get('delta_item', 0.05)
    delta_cap = params.get('delta_cap', 0.05)
    CI_delta = params.get('CI_delta', 0.0002)
    cred_level = params.get('cred_level', 0.95)
    conservatism = params.get('conservatism', 2)
    low_perf_threshold = params.get('low_performance_threshold', 0.1)
    draws = params.get('draws', 3000)
    tune = params.get('tune', 3000)
    rep_batch_size = params.get('rep_batch_size', 1)
    pymc_refresh_every = params.get('pymc_refresh_every', 2)
    stab_window = params.get('stab_window', 5)

    stop_sample_ids = []
    sample_ci_records = {}
    sample_ci_slopes = {}
    used_reps_dfs = {}
    item_summaries = []
    item_ids = list(df['sample_id_num'].unique())

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
    for item_id in item_ids:
        df_item = df[df['sample_id_num'] == item_id].sort_values('epoch')
        successes = 0
        trials = 0
        used_reps = []
        ci_record = []
        ci_slopes_hist = []
        current_conservatism = conservatism if df_item['score'].mean() < low_perf_threshold else 1.0
        for start in range(0, len(df_item), rep_batch_size):
            batch = df_item.iloc[start:start+rep_batch_size]
            successes += batch['score'].sum()
            trials += len(batch)
            used_reps.extend(batch.itertuples(index=False))
            lo, hi, width = _beta_ci_adaptive(
                successes, trials, cred_level=cred_level,
                conservatism=current_conservatism,
                low_perf_threshold=low_perf_threshold
            )
            ci_record.append(width)
            if width < delta_item:
                logger.info(f"Stopping sample_id {item_id}: CI width {width:.4f} < delta_item {delta_item} | epochs used: {trials}")
                stop_sample_ids.append(item_id)
                break
            if len(ci_record) >= stab_window:
                recent_widths = ci_record[-stab_window:]
                slope = np.polyfit(range(len(recent_widths)), recent_widths, 1)[0]
                ci_slopes_hist.append(slope)
                slope_threshold = CI_delta / current_conservatism if df_item['score'].mean() < low_perf_threshold else CI_delta
                if (abs(slope) <= slope_threshold) and (len(ci_slopes_hist) >= 4):
                    recent_slopes = ci_slopes_hist[-3:]
                    slope_slopes = np.polyfit(range(len(recent_slopes)), recent_slopes, 1)[0]
                    if slope_slopes >= 0:
                        if df_item['score'].mean() >= low_perf_threshold:
                            logger.info(f"Stopping sample_id {item_id} due to CI stabilization: slope {slope:.6f} <= threshold {slope_threshold:.6f} | epochs used: {trials}")
                            stop_sample_ids.append(item_id)
                            break
                        elif abs(slope) <= slope_threshold / 2:
                            logger.info(f"Stopping low-performance sample_id {item_id} due to strong CI stabilization: slope {slope:.6f} | epochs used: {trials}")
                            stop_sample_ids.append(item_id)
                            break
        sample_ci_records[item_id] = ci_record
        sample_ci_slopes[item_id] = ci_slopes_hist
        used_reps_dfs[item_id] = pd.DataFrame(used_reps, columns=df_item.columns)
        item_summaries.append({'successes': successes, 'trials': trials})

    # Group-level stopping (width and slope, using PyMC)
    all_successes = np.array([s['successes'] for s in item_summaries])
    all_trials = np.array([s['trials'] for s in item_summaries])
    with model:
        pm.set_data({
            "successes": all_successes,
            "trials": all_trials,
            "n_items": np.int64(len(all_successes))
        })
        trace = pm.sample(draws=draws, tune=tune, chains=2, cores=1, progressbar=False, target_accept=0.97)
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
        CI_record = [theta_width]
        CI_slopes_hist = []
        effective_width = theta_width
        current_perf_estimate = np.sum(all_successes) / np.sum(all_trials) if np.sum(all_trials) > 0 else 0
        if current_perf_estimate < low_perf_threshold:
            effective_width = theta_width * conservatism
        stop_task = False
        if effective_width < delta_cap:
            logger.info(f"Stopping task/grouping: CI width {effective_width:.4f} < delta_cap {delta_cap} | sample_ids used: {len(item_summaries)}")
            stop_task = True
        else:
            # CI stabilization check
            for _ in range(stab_window):
                CI_record.append(theta_width)
                if len(CI_record) >= stab_window:
                    recent_widths = CI_record[-stab_window:]
                    slope = np.polyfit(range(len(recent_widths)), recent_widths, 1)[0]
                    CI_slopes_hist.append(slope)
                    slope_threshold = CI_delta / conservatism if current_perf_estimate < low_perf_threshold else CI_delta
                    if (abs(slope) <= slope_threshold) and (len(CI_slopes_hist) >= 4):
                        recent_slopes = CI_slopes_hist[-3:]
                        slope_slopes = np.polyfit(range(len(recent_slopes)), recent_slopes, 1)[0]
                        if slope_slopes >= 0:
                            if current_perf_estimate >= low_perf_threshold:
                                logger.info(f"Stopping task/grouping due to CI stabilization: slope {slope:.6f} <= threshold {slope_threshold:.6f} | sample_ids used: {len(item_summaries)}")
                                stop_task = True
                                break
                            elif abs(slope) <= slope_threshold / 2:
                                logger.info(f"Stopping low-performance task/grouping due to strong CI stabilization: slope {slope:.6f} | sample_ids used: {len(item_summaries)}")
                                stop_task = True
                                break
    logger.info('Live optimal stopping complete')
    return {'stop_sample_ids': stop_sample_ids, 'stop_task': stop_task} 