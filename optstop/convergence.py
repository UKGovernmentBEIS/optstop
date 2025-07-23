"""
Post-hoc convergence assessment for sequential testing scenarios.
Refactored from OptStop_Convergence_v5.py for use in the optstop package.
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import concurrent.futures
import logging
import os

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

REQUIRED_COLUMNS = ['grouping_num', 'task_num', 'sample_id_num', 'epoch', 'score']

def _check_required_columns(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

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
    (pid, cap, df_part, params) = args
    logger = logging.getLogger('optstop.convergence')
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
        item_seqs = params.get('item_seqs', 20)
        epoch_seqs = params.get('epoch_seqs', 20)
        # Set random seed if provided
        if 'random_seed' in params:
            np.random.seed(params['random_seed'])
        logger.info(f"Processing grouping {pid}, task {cap}")

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
            np.random.shuffle(item_ids)
            for item_idx, item_id in enumerate(item_ids):
                df_item = df_part[df_part['sample_id_num'] == item_id].sort_values('epoch')
                sample_ID_performances = []
                for epoch_seq in range(epoch_seqs):
                    if epoch_seqs >= 1:
                        epoch_ids = list(df_item['epoch'].unique())
                        np.random.shuffle(epoch_ids)
                        df_item = df_item.set_index('epoch').reindex(epoch_ids).reset_index()
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
                        for start in range(0, len(df_item), rep_batch_size):
                            batch = df_item.iloc[start:start+rep_batch_size]
                            successes += batch['score'].sum()
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
                            if (start + rep_batch_size >= len(df_item)):
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
            'task': cap,
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
        logger.error(f"Error processing grouping {pid}, task {cap}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'grouping': pid,
            'group_label': df_part['grouping'].iloc[0] if 'grouping' in df_part.columns else str(pid),
            'task': cap,
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

# --- Main API ---
def convergence_posthoc(df: pd.DataFrame, params: dict):
    logger = logging.getLogger('optstop.convergence')
    _check_required_columns(df)
    _validate_params(params)
    logger.info('Starting post-hoc convergence analysis')
    groupings = list(df.groupby(['grouping_num', 'task_num']))
    if not groupings:
        logger.info('No groupings to process; returning empty DataFrame.')
        return pd.DataFrame()
    args_list = [(pid, cap, df_part, params) for (pid, cap), df_part in groupings]
    max_workers = min(len(args_list), os.cpu_count() or 1)
    logger.info(f'Using {max_workers} parallel workers for {len(args_list)} groupings')
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(_process_grouping, args_list):
            results.append(res)
    logger.info('Convergence analysis complete')
    return pd.DataFrame(results) 