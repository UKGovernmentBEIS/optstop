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
from typing import Dict, Any, List, Tuple, Optional, Union
from tqdm import tqdm
import contextlib
import io
import sys
import warnings

# Suppress all warnings before importing PyMC
warnings.filterwarnings('ignore')

import pymc as pm
import arviz as az
import pytensor.tensor as pt

# Import GPU utilities and cleanup utilities
from . import gpu_utils
from . import cleanup_utils

# Import ordinal scoring utilities
from .ordinal_utils import (
    _ordinal_ci_adaptive,
    validate_ordinal_scores,
    determine_score_type,
    determine_score_type_standalone,
    counts_to_scores,
    aggregate_item_counts,
    _ordinal_ci_hierarchical_modal,
    _ordinal_ci_hierarchical_entropy
)

# Import ordinal model for entropy-based stopping
from .ordinal_model import (
    _ordinal_entropy_ci_adaptive,
    _ordinal_hybrid_stopping_criterion,
    _ordinal_hybrid_stopping_criterion_hierarchical,
    _compute_threshold_probability,
    _create_ordered_logistic_hierarchical,
    _compute_adaptive_cutpoint_prior_params
)

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

def _sanitize_diagnostics(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert all numpy types in diagnostics dict to native Python types for JSON serialization.

    This ensures compatibility with Pydantic validation in inspect_ai EarlyStop models.

    Args:
        diagnostics: Dictionary potentially containing numpy types

    Returns:
        Sanitized dictionary with native Python types
    """
    if diagnostics is None:
        return None

    sanitized = {}
    for key, value in diagnostics.items():
        if isinstance(value, (np.integer, np.floating)):
            # Convert numpy scalar to Python native type
            sanitized[key] = value.item()
        elif isinstance(value, np.ndarray):
            # Convert numpy array to list
            sanitized[key] = value.tolist()
        elif isinstance(value, tuple):
            # Convert tuple to list (Pydantic requires lists for JSON serialization)
            sanitized[key] = [
                v.item() if isinstance(v, (np.integer, np.floating)) else v
                for v in value
            ]
        elif isinstance(value, dict):
            # Recursively sanitize nested dicts
            sanitized[key] = _sanitize_diagnostics(value)
        elif isinstance(value, list):
            # Sanitize list elements
            sanitized[key] = [
                v.item() if isinstance(v, (np.integer, np.floating)) else v
                for v in value
            ]
        else:
            # Keep native Python types as-is
            sanitized[key] = value

    return sanitized

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
            # Captured output discarded - not needed for production
            # stdout_content = stdout_buffer.getvalue()
            # stderr_content = stderr_buffer.getvalue()
            pass

# --- Diagnostic Functions ---
def _beta_ci(successes: int, trials: int, cred_level: float = 0.95) -> Tuple[float, float]:
    """Compute beta credible interval for binomial proportion."""
    alpha_post = 1 + successes
    beta_post = 1 + trials - successes
    # Validate parameters to avoid invalid beta distribution
    if alpha_post <= 0 or beta_post <= 0:
        return (np.nan, np.nan)
    lower = beta.ppf((1-cred_level)/2, alpha_post, beta_post)
    upper = beta.ppf(1-(1-cred_level)/2, alpha_post, beta_post)
    return lower, upper


def _bootstrap_ci(values: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 1000,
                  random_state: Optional[int] = None) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for any score distribution.
    Works for binary, ordinal, or continuous scores.

    Args:
        values: Array of score values
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap samples (default 1000)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(values) == 0:
        return (np.nan, np.nan)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))

    rng = np.random.default_rng(random_state)
    bootstrap_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    alpha = (1 - confidence) / 2
    return (float(np.percentile(bootstrap_means, alpha * 100)),
            float(np.percentile(bootstrap_means, (1 - alpha) * 100)))


def _normalize_score_type_input(
    score_type: Union[str, Dict[str, str]],
    unique_groupings: List[str],
    logger: logging.Logger
) -> Dict[str, str]:
    """
    Normalize score_type input to a per-grouping dictionary.

    Args:
        score_type: Either a single string (applied to all groupings) or a dict mapping
                    grouping names to score types ('binary', 'ordinal', 'continuous')
        unique_groupings: List of unique grouping names in the data
        logger: Logger for warnings

    Returns:
        Dictionary mapping each grouping to its score type
    """
    valid_types = ('binary', 'ordinal', 'continuous')

    if isinstance(score_type, str):
        # Single value - apply to all groupings
        if score_type not in valid_types:
            logger.warning(f"Invalid score_type '{score_type}', must be one of {valid_types}. Defaulting to 'continuous'.")
            score_type = 'continuous'
        return {g: score_type for g in unique_groupings}
    elif isinstance(score_type, dict):
        # Validate provided types
        result = {}
        for grouping in unique_groupings:
            if grouping in score_type:
                st = score_type[grouping]
                if st not in valid_types:
                    logger.warning(f"Invalid score_type '{st}' for grouping '{grouping}', defaulting to 'continuous'.")
                    result[grouping] = 'continuous'
                else:
                    result[grouping] = st
            else:
                logger.warning(f"No score_type specified for grouping '{grouping}', defaulting to 'continuous'.")
                result[grouping] = 'continuous'

        # Warn about extra keys in map that don't exist in data
        extra_keys = set(score_type.keys()) - set(unique_groupings)
        if extra_keys:
            logger.warning(f"score_type map contains groupings not in data: {extra_keys}")

        return result
    else:
        logger.warning(f"score_type must be str or dict, got {type(score_type)}. Defaulting to 'continuous' for all.")
        return {g: 'continuous' for g in unique_groupings}


def _normalize_ordinal_max_scores(
    ordinal_max_scores: Union[int, Dict[str, int]],
    score_type_map: Dict[str, str],
    logger: logging.Logger
) -> Dict[str, Optional[int]]:
    """
    Normalize ordinal_max_scores input to a per-grouping dictionary.

    Args:
        ordinal_max_scores: Either a single int (applied to all ordinal groupings) or a dict
                            mapping grouping names to their max scores
        score_type_map: Dictionary mapping groupings to score types
        logger: Logger for warnings

    Returns:
        Dictionary mapping each grouping to its max score (or None if not ordinal)
    """
    if isinstance(ordinal_max_scores, int):
        # Single value - apply to all ordinal groupings
        result = {}
        for grouping, st in score_type_map.items():
            if st == 'ordinal':
                result[grouping] = ordinal_max_scores
            else:
                result[grouping] = None
        return result
    elif isinstance(ordinal_max_scores, dict):
        result = {}
        for grouping, st in score_type_map.items():
            if st == 'ordinal':
                if grouping in ordinal_max_scores:
                    result[grouping] = ordinal_max_scores[grouping]
                else:
                    logger.warning(f"No ordinal_max_score specified for ordinal grouping '{grouping}', defaulting to 5.")
                    result[grouping] = 5
            else:
                result[grouping] = None
        return result
    else:
        logger.warning(f"ordinal_max_scores must be int or dict, got {type(ordinal_max_scores)}. Defaulting to 5.")
        return {g: 5 if score_type_map.get(g) == 'ordinal' else None for g in score_type_map}


def _aggregate_with_ci(df: pd.DataFrame, score_col: str = "score",
                       score_type: Union[str, Dict[str, str]] = "binary",
                       ordinal_max_score: Union[int, Dict[str, int], None] = None,
                       random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Aggregate data to item level and compute CIs, with per-grouping score type support.

    Args:
        df: Input DataFrame with grouping, sample_id, and score columns
        score_col: Name of score column
        score_type: Either a string ('binary', 'ordinal', 'continuous') applied to all groupings,
                    or a dict mapping grouping names to their score types.
                    - 'binary': Uses beta CI (assumes 0/1 scores)
                    - 'ordinal': Uses bootstrap CI with scaling by ordinal_max_score
                    - 'continuous': Uses bootstrap CI without scaling (assumes [0,1] range)
        ordinal_max_score: Either a single int (applied to all ordinal groupings) or a dict
                           mapping grouping names to their max scores
        random_state: Random seed for reproducible bootstrap CIs

    Returns:
        DataFrame with columns: grouping, sample_id, successes, trials, prop, ci_low, ci_high, n_reps
    """
    agg_cols = ["grouping", "sample_id"]
    logger = logging.getLogger('optstop.diagnostics')
    expected_cols = agg_cols + ['successes', 'trials', 'prop', 'ci_low', 'ci_high', 'n_reps']

    # Validate required columns exist and have data
    for col in agg_cols:
        if col not in df.columns:
            logger.warning(f"Required column '{col}' not found in DataFrame")
            return pd.DataFrame(columns=expected_cols)
        if df[col].isna().all():
            logger.warning(f"Column '{col}' is entirely NaN")
            return pd.DataFrame(columns=expected_cols)

    # Check score column
    if score_col not in df.columns:
        logger.warning(f"Score column '{score_col}' not found")
        return pd.DataFrame(columns=expected_cols)

    # Filter out rows with NaN in aggregation columns (groupby drops them anyway)
    df_clean = df.dropna(subset=agg_cols)
    if df_clean.empty:
        logger.warning("No valid rows after dropping NaN in grouping columns")
        return pd.DataFrame(columns=expected_cols)

    # Get unique groupings and normalize inputs to per-grouping dictionaries
    unique_groupings = sorted(df_clean['grouping'].unique().tolist())
    score_type_map = _normalize_score_type_input(score_type, unique_groupings, logger)
    ordinal_max_map = _normalize_ordinal_max_scores(
        ordinal_max_score if ordinal_max_score is not None else 5,
        score_type_map, logger
    )

    # Process each grouping according to its score type
    all_results = []

    for grouping_name in unique_groupings:
        grp_score_type = score_type_map[grouping_name]
        grp_max_score = ordinal_max_map.get(grouping_name)

        # Filter data for this grouping
        grp_df = df_clean[df_clean['grouping'] == grouping_name]
        if grp_df.empty:
            continue

        # Group by sample_id within this grouping
        grp_grouped = grp_df.groupby('sample_id', sort=True)

        if grp_score_type == 'binary':
            # Binary: use beta CI
            grp_summary = grp_grouped[score_col].agg(['sum', 'count']).reset_index()
            grp_summary.rename(columns={'sum': 'successes', 'count': 'trials'}, inplace=True)
            grp_summary['prop'] = grp_summary['successes'] / grp_summary['trials']

            # Compute beta CIs
            ci_lows = []
            ci_highs = []
            for _, row in grp_summary.iterrows():
                try:
                    successes = row['successes']
                    trials = row['trials']
                    # Explicit NaN check before int conversion
                    if pd.isna(successes) or pd.isna(trials):
                        lo, hi = np.nan, np.nan
                    else:
                        lo, hi = _beta_ci(int(successes), int(trials))
                except Exception:
                    lo, hi = np.nan, np.nan
                ci_lows.append(lo)
                ci_highs.append(hi)
            grp_summary['ci_low'] = ci_lows
            grp_summary['ci_high'] = ci_highs

        else:
            # Ordinal/Continuous: use bootstrap CI
            grp_summary = grp_grouped[score_col].agg(['mean', 'count']).reset_index()
            grp_summary.rename(columns={'mean': 'prop', 'count': 'trials'}, inplace=True)

            # Determine scaling factor for ordinal
            scale_factor = None
            if grp_score_type == 'ordinal':
                if grp_max_score is not None and grp_max_score > 0:
                    scale_factor = grp_max_score
                else:
                    logger.warning(f"score_type='ordinal' for grouping '{grouping_name}' but no max_score; scores won't be scaled")

            # Scale proportion if needed
            if scale_factor is not None:
                grp_summary['prop'] = grp_summary['prop'] / scale_factor

            # For compatibility, compute 'successes' as prop * trials
            grp_summary['successes'] = grp_summary['prop'] * grp_summary['trials']

            # Compute bootstrap CIs
            ci_results = {}
            for sample_id, sample_df in grp_grouped:
                scores = sample_df[score_col].dropna().values
                if scale_factor is not None:
                    scores = scores / scale_factor
                try:
                    lo, hi = _bootstrap_ci(scores, random_state=random_state)
                except Exception:
                    lo, hi = np.nan, np.nan
                ci_results[sample_id] = (lo, hi)

            # Use explicit index alignment
            grp_summary['ci_low'] = grp_summary['sample_id'].map(lambda sid: ci_results.get(sid, (np.nan, np.nan))[0])
            grp_summary['ci_high'] = grp_summary['sample_id'].map(lambda sid: ci_results.get(sid, (np.nan, np.nan))[1])

        grp_summary['grouping'] = grouping_name
        grp_summary['n_reps'] = grp_summary['trials']
        all_results.append(grp_summary)

    if not all_results:
        return pd.DataFrame(columns=expected_cols)

    # Combine all results
    summary = pd.concat(all_results, ignore_index=True)

    # Reorder columns to expected order
    summary = summary[expected_cols]

    return summary

def _compute_bayesian_hdi_per_task(df: pd.DataFrame, confidence: float = 0.95, score_col: str = "score") -> pd.DataFrame:
    """Estimate HDI-based uncertainty across items for each grouping-task pair."""
    results = []

    grouped = df.groupby(['grouping', 'task'])
    for (grouping, task), sub_df in grouped:
        item_means = sub_df.groupby('sample_id')[score_col].mean().values
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
                    item_means, size=(6000, len(item_means)), replace=True
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

def _infer_sample_id_column(df: pd.DataFrame) -> Optional[str]:
    """
    Infer sample ID column with proper priority and validation.

    Priority:
    1. Exact match 'sample_id' if it has valid data
    2. Other columns containing 'sample' with valid data (sorted for determinism)
    3. Columns ending in '_id' but not starting with problematic prefixes (sorted for determinism)
    4. Fall back to None (caller should use index)

    Returns:
        Column name to use as sample_id, or None if no suitable column found.
    """
    logger = logging.getLogger('optstop.diagnostics')

    # Priority 1: Exact match
    if 'sample_id' in df.columns and df['sample_id'].notna().any():
        logger.debug("Inferred sample_id column: 'sample_id' (exact match)")
        return 'sample_id'

    # Priority 2: Columns containing 'sample' (sorted for deterministic selection)
    sample_cols = sorted([col for col in df.columns
                          if 'sample' in col.lower() and col != 'sample_id'])
    for col in sample_cols:
        if df[col].notna().any():
            logger.debug(f"Inferred sample_id column: '{col}' (contains 'sample')")
            return col

    # Priority 3: Columns ending in '_id' but not problematic prefixes (sorted for determinism)
    skip_prefixes = ('eval_', 'model_', 'task_', 'run_')
    id_cols = sorted([col for col in df.columns
                      if col.lower().endswith('_id')
                      and not col.lower().startswith(skip_prefixes)])
    for col in id_cols:
        if df[col].notna().any():
            logger.debug(f"Inferred sample_id column: '{col}' (ends with '_id')")
            return col

    # Priority 4: Fall back to None
    logger.debug("No suitable sample_id column found, will use index")
    return None


def _infer_grouping_columns(df: pd.DataFrame) -> Optional[List[str]]:
    """
    Infer grouping columns with validation.

    Returns:
        List of column names to use for grouping, or None if no valid columns found.
        Results are sorted for deterministic selection.
    """
    logger = logging.getLogger('optstop.diagnostics')

    # Find candidates with valid (non-null) data, sorted for determinism
    candidates = sorted([col for col in df.columns
                         if ('group' in col.lower() or 'task' in col.lower())
                         and df[col].notna().any()])

    # Prefer 'grouping' exact match
    if 'grouping' in candidates:
        logger.debug("Inferred grouping column: 'grouping' (exact match)")
        return ['grouping']

    if candidates:
        logger.debug(f"Inferred grouping columns: {candidates}")
        return candidates

    logger.debug("No suitable grouping columns found")
    return None


def _generate_diagnostic_plots(full_df: pd.DataFrame, pruned_df: pd.DataFrame,
                              out_prefix: str = "optstop_diagnostics", score_col: str = "score",
                              grouping_columns: Optional[List[str]] = None,
                              sample_id_column: Optional[str] = None,
                              score_type: Union[str, Dict[str, str]] = "binary",
                              ordinal_max_score: Union[int, Dict[str, int]] = 10,
                              random_state: Optional[int] = None) -> None:
    """
    Generate diagnostic plots comparing full vs pruned datasets.

    Args:
        full_df: Full dataset before pruning
        pruned_df: Pruned dataset after optimal stopping
        out_prefix: Prefix for output files
        score_col: Name of score column
        grouping_columns: Explicit grouping columns (if None, will infer)
        sample_id_column: Explicit sample ID column (if None, will infer)
        score_type: Either a string ('binary', 'ordinal', 'continuous') applied to all groupings,
                    or a dict mapping grouping names to their score types. This affects CI computation:
                    - 'binary': Uses beta CI (assumes 0/1 scores)
                    - 'ordinal': Uses bootstrap CI with scaling by ordinal_max_score
                    - 'continuous': Uses bootstrap CI without scaling (assumes [0,1] range)
        ordinal_max_score: Either a single int (applied to all ordinal groupings) or a dict
                           mapping grouping names to their max scores
        random_state: Random seed for reproducible bootstrap CIs
    """
    logger = logging.getLogger('optstop.diagnostics')

    try:
        # Create internal column structure for diagnostics
        full_df_internal = full_df.copy()
        pruned_df_internal = pruned_df.copy()

        # Helper to set up columns for a dataframe
        def setup_internal_columns(df_internal: pd.DataFrame, df_name: str) -> pd.DataFrame:
            # Set up grouping column
            if 'grouping' not in df_internal.columns:
                grouping_set = False

                # Priority 1: Use explicit grouping columns if provided and valid
                if grouping_columns:
                    missing_cols = [col for col in grouping_columns if col not in df_internal.columns]
                    if missing_cols:
                        logger.warning(f"Grouping columns {missing_cols} not found in {df_name}, falling back to inference")
                    else:
                        df_internal['grouping'] = df_internal[grouping_columns].astype(str).agg('-'.join, axis=1)
                        df_internal['task'] = df_internal[grouping_columns[0]]
                        grouping_set = True

                # Priority 2: Use internal numeric columns
                if not grouping_set and 'grouping_num' in df_internal.columns and 'task_num' in df_internal.columns:
                    df_internal['grouping'] = df_internal['grouping_num'].astype(str) + '-' + df_internal['task_num'].astype(str)
                    df_internal['task'] = df_internal['task_num']
                    grouping_set = True

                # Priority 3: Infer from available columns (fallback)
                if not grouping_set:
                    inferred_cols = _infer_grouping_columns(df_internal)
                    if inferred_cols:
                        df_internal['grouping'] = df_internal[inferred_cols].astype(str).agg('-'.join, axis=1)
                        df_internal['task'] = df_internal[inferred_cols[0]]
                    else:
                        logger.warning(f"No grouping columns found in {df_name}, using default 'group1'")
                        df_internal['grouping'] = 'group1'
                        df_internal['task'] = 1

            # Ensure 'task' column exists (needed for HDI computation)
            # If 'grouping' already existed but 'task' doesn't, derive it from grouping
            if 'task' not in df_internal.columns:
                if 'task_num' in df_internal.columns:
                    df_internal['task'] = df_internal['task_num']
                else:
                    # Use grouping values as task (each unique grouping is a separate task)
                    df_internal['task'] = df_internal['grouping']

            # Set up sample_id column
            if 'sample_id' not in df_internal.columns:
                sample_id_set = False

                # Priority 1: Use explicit sample_id column if provided and valid
                if sample_id_column:
                    if sample_id_column not in df_internal.columns:
                        logger.warning(f"Sample ID column '{sample_id_column}' not found in {df_name}, falling back to inference")
                    else:
                        df_internal['sample_id'] = df_internal[sample_id_column]
                        sample_id_set = True

                # Priority 2: Use internal numeric column
                if not sample_id_set and 'sample_id_num' in df_internal.columns:
                    df_internal['sample_id'] = df_internal['sample_id_num']
                    sample_id_set = True

                # Priority 3: Infer from available columns (fallback)
                if not sample_id_set:
                    inferred_col = _infer_sample_id_column(df_internal)
                    if inferred_col:
                        df_internal['sample_id'] = df_internal[inferred_col]
                    else:
                        logger.warning(f"No sample_id column found in {df_name}, using index")
                        df_internal['sample_id'] = df_internal.index

            return df_internal

        full_df_internal = setup_internal_columns(full_df_internal, "full_df")
        pruned_df_internal = setup_internal_columns(pruned_df_internal, "pruned_df")

        # Get unique groupings and normalize score_type/ordinal_max_score to per-grouping dicts
        unique_groupings = sorted(full_df_internal['grouping'].unique().tolist())
        score_type_map = _normalize_score_type_input(score_type, unique_groupings, logger)
        ordinal_max_map = _normalize_ordinal_max_scores(
            ordinal_max_score if ordinal_max_score is not None else 5,
            score_type_map, logger
        )

        # Aggregate to item level and compute CIs (pass through the maps)
        pruned_item = _aggregate_with_ci(
            pruned_df_internal, score_col=score_col, score_type=score_type_map,
            ordinal_max_score=ordinal_max_map,
            random_state=random_state
        )
        full_item = _aggregate_with_ci(
            full_df_internal, score_col=score_col, score_type=score_type_map,
            ordinal_max_score=ordinal_max_map,
            random_state=random_state
        )

        # Calculate confidence intervals for both datasets
        # For ordinal data, scale scores to [0,1] before computing HDI (so y-axis is consistent)
        # With per-grouping score types, we need to scale each grouping individually
        # Use explicit deep=True to ensure independent copies that can be safely modified
        full_df_for_hdi = full_df_internal.copy(deep=True)
        pruned_df_for_hdi = pruned_df_internal.copy(deep=True)

        for grouping_name in unique_groupings:
            grp_score_type = score_type_map.get(grouping_name, 'continuous')
            grp_max_score = ordinal_max_map.get(grouping_name)

            if grp_score_type == 'ordinal' and grp_max_score is not None and grp_max_score > 0:
                # Scale ordinal scores for this grouping
                mask_full = full_df_for_hdi['grouping'] == grouping_name
                mask_pruned = pruned_df_for_hdi['grouping'] == grouping_name
                full_df_for_hdi.loc[mask_full, score_col] = full_df_for_hdi.loc[mask_full, score_col] / grp_max_score
                pruned_df_for_hdi.loc[mask_pruned, score_col] = pruned_df_for_hdi.loc[mask_pruned, score_col] / grp_max_score

        full_results = _compute_bayesian_hdi_per_task(full_df_for_hdi, score_col=score_col)
        full_results.rename(columns={'n_items': 'n_samples'}, inplace=True)
        full_results['dataset'] = 'Full'

        trimmed_results = _compute_bayesian_hdi_per_task(pruned_df_for_hdi, score_col=score_col)
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

def _continuous_bounded_ci_adaptive(
    scores: np.ndarray,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    cred_level: float = 0.95,
    conservatism: float = 1.0,
    low_perf_threshold: float = 0.2,
    base_strength: int = 2,
    samples: int = 10000
) -> Tuple[float, float, float]:
    """
    Compute adaptive Bayesian credible interval for continuous bounded scores.

    Designed for aggregated scores (mean/median) that are continuous floats:
    - Binary aggregated: scores in [0, 1]
    - Ordinal aggregated: scores in [0, ordinal_max_score]

    Method: Beta distribution (or scaled/shifted Beta for arbitrary bounds)
    with method-of-moments parameter estimation and adaptive conservatism.

    Args:
        scores: Array of continuous scores
        lower_bound: Lower bound of score range (default: 0.0)
        upper_bound: Upper bound of score range (default: 1.0)
        cred_level: Credibility level (e.g., 0.95 for 95% CI)
        conservatism: Multiplier for CI width in low-performance scenarios (>= 1.0)
        low_perf_threshold: Performance threshold for conservatism (normalized 0-1)
        base_strength: Base prior strength for Bayesian estimation
        samples: Number of Monte Carlo samples for posterior

    Returns:
        Tuple of (lower_bound_ci, upper_bound_ci, effective_width)
        - All values in original scale [lower_bound, upper_bound]
        - effective_width: CI width, adjusted for conservatism if needed

    Statistical Approach:
        1. Normalize scores to [0, 1]
        2. Estimate Beta distribution parameters using method of moments:
           - mean = alpha / (alpha + beta)
           - var = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
        3. Apply conservative priors for low performance
        4. Generate posterior samples from Beta distribution
        5. Compute credible interval
        6. Scale back to original bounds

    Example:
        # Binary aggregated: [0.8, 0.9, 0.7, 0.85, 0.9]
        scores = np.array([0.8, 0.9, 0.7, 0.85, 0.9])
        lo, hi, width = _continuous_bounded_ci_adaptive(scores, 0.0, 1.0)
        # Returns: (0.75, 0.92, 0.17) - "95% confident mean is 0.75-0.92"

        # Ordinal aggregated: [8.2, 8.7, 8.4, 8.9, 8.5]
        scores = np.array([8.2, 8.7, 8.4, 8.9, 8.5])
        lo, hi, width = _continuous_bounded_ci_adaptive(scores, 0.0, 10.0)
        # Returns: (8.1, 8.9, 0.8) in original scale

    References:
        Beta distribution for bounded continuous data
        Method of moments parameter estimation
    """
    # Handle edge case: empty array
    if len(scores) == 0:
        return lower_bound, upper_bound, upper_bound - lower_bound

    # Handle edge case: single observation
    if len(scores) == 1:
        # Return wide interval centered on observation
        obs = scores[0]
        # Use 50% of range as conservative width
        width = (upper_bound - lower_bound) * 0.5
        lo = max(lower_bound, obs - width / 2)
        hi = min(upper_bound, obs + width / 2)
        return lo, hi, hi - lo

    # Normalize scores to [0, 1]
    score_range = upper_bound - lower_bound
    if score_range <= 0:
        # Invalid bounds
        return lower_bound, upper_bound, upper_bound - lower_bound

    scores_normalized = (scores - lower_bound) / score_range

    # Clip to [0, 1] to handle any numerical issues
    scores_normalized = np.clip(scores_normalized, 0.0, 1.0)

    # Compute sample statistics
    mean_normalized = np.mean(scores_normalized)
    var_normalized = np.var(scores_normalized, ddof=1) if len(scores) > 1 else 0.0

    # Handle edge case: zero variance (all scores identical)
    if var_normalized < 1e-10:
        # All scores are essentially the same - return tight interval
        # Use small width based on sample size
        n = len(scores)
        width_normalized = min(0.1, 1.0 / np.sqrt(n))
        lo_normalized = max(0.0, mean_normalized - width_normalized / 2)
        hi_normalized = min(1.0, mean_normalized + width_normalized / 2)

        # Scale back to original bounds
        lo = lo_normalized * score_range + lower_bound
        hi = hi_normalized * score_range + lower_bound
        return lo, hi, hi - lo

    # Method of moments: estimate Beta(alpha, beta) parameters
    # mean = alpha / (alpha + beta)
    # var = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
    #
    # Solving for alpha, beta:
    # Let m = mean, v = var
    # alpha = m * ((m * (1 - m) / v) - 1)
    # beta = (1 - m) * ((m * (1 - m) / v) - 1)

    # Ensure variance is valid (must be < mean * (1 - mean))
    max_var = mean_normalized * (1 - mean_normalized)
    if var_normalized >= max_var:
        # Variance too large for Beta - use conservative estimate
        var_normalized = max_var * 0.95

    # Compute method-of-moments estimates
    ratio = (mean_normalized * (1 - mean_normalized) / var_normalized) - 1

    # Ensure ratio is positive (required for valid Beta parameters)
    if ratio <= 0:
        # Fall back to uniform-ish prior
        alpha_mle = 1.0
        beta_mle = 1.0
    else:
        alpha_mle = mean_normalized * ratio
        beta_mle = (1 - mean_normalized) * ratio

    # Ensure parameters are at least 0.5 (avoid extreme distributions)
    alpha_mle = max(0.5, alpha_mle)
    beta_mle = max(0.5, beta_mle)

    # Apply Bayesian prior (decays with sample size)
    n = len(scores)

    # Determine if performance is low (requires conservatism)
    if mean_normalized < low_perf_threshold:
        # Low performance: apply conservatism boost
        alpha_boost = conservatism
        prior_scaling = base_strength * np.exp(-n / (10 * conservatism))

        # Boost alpha more for low performance (makes prior stronger)
        alpha_prior = max(prior_scaling * mean_normalized * alpha_boost, 0.5)
        beta_prior = max(prior_scaling * (1 - mean_normalized), 0.5)
    else:
        # Normal performance: standard prior
        prior_scaling = base_strength * np.exp(-n / 10)

        alpha_prior = max(prior_scaling * mean_normalized, 0.5)
        beta_prior = max(prior_scaling * (1 - mean_normalized), 0.5)

    # Combine prior with data (Bayesian update)
    # For Beta, this is simple addition due to conjugacy
    # We approximate data contribution as alpha_mle, beta_mle
    alpha_post = alpha_prior + alpha_mle
    beta_post = beta_prior + beta_mle

    # Generate posterior samples
    posterior_draws = np.random.beta(alpha_post, beta_post, samples)

    # Compute credible interval
    alpha_tail = (1 - cred_level) / 2
    lo_normalized = np.quantile(posterior_draws, alpha_tail)
    hi_normalized = np.quantile(posterior_draws, 1 - alpha_tail)

    # Scale back to original bounds
    lo = lo_normalized * score_range + lower_bound
    hi = hi_normalized * score_range + lower_bound

    # Compute width
    width = hi - lo

    # Apply conservatism to effective width for low performance
    if mean_normalized < low_perf_threshold:
        effective_width = width * conservatism
    else:
        effective_width = width

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
        # No specific GPU assigned - let JAX/PyTensor auto-detect available devices
        # Do NOT set CUDA_VISIBLE_DEVICES to empty string as this disables GPU entirely
        os.environ['PYTENSOR_FLAGS'] = f'compiledir={unique_worker_dir},floatX=float32'
        # Let JAX auto-detect (don't force CPU-only mode)
        # os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Commented out to allow GPU detection
        # os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Commented out to allow GPU detection
        device_type = 'Auto-detect (GPU if available)'

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # Additional PyTensor isolation environment variables
    os.environ['PYTENSOR_FLAGS'] += ',optimizer=fast_compile,openmp=False'

    # Force PyTensor to use our compiledir by setting config directly after import
    # This handles cases where PyTensor might be imported later
    try:
        import pytensor
        pytensor.config.compiledir = unique_worker_dir
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
            CI_delta = params.get('CI_delta', 0.00005)
            cred_level = params.get('cred_level', 0.95)
            conservatism = params.get('conservatism', 5)
            low_perf_threshold = params.get('low_performance_threshold', 0.01)
            rep_batch_size = params.get('rep_batch_size', 1)
            pymc_refresh_every = params.get('pymc_refresh_every', 2)
            stab_window = params.get('stab_window', 10)

            # Determine score type for this grouping
            ordinal_tasks = params.get('ordinal_tasks', None)
            continuous_tasks = params.get('continuous_tasks', None)
            ordinal_max_score = params.get('ordinal_max_score', 10)
            ordinal_inference = params.get('ordinal_inference', 'modal')
            ordinal_model_type = params.get('ordinal_model_type', 'ordered_logistic')
            entropy_threshold = params.get('entropy_threshold', 1.5)
            entropy_stabilization_threshold = params.get('entropy_stabilization_threshold', 0.002)
            grouping_name = df_part['grouping'].iloc[0] if 'grouping' in df_part.columns else str(pid)
            score_type, bounds = determine_score_type_standalone(
                grouping_name,
                ordinal_tasks=ordinal_tasks,
                continuous_tasks=continuous_tasks,
                upper_bound=ordinal_max_score
            )

            logger.info(f"Processing grouping {pid} ('{grouping_name}') with {score_type} scoring" +
                       (f" (inference: {ordinal_inference})" if score_type == 'ordinal' else ""))

            # Validate ordinal scores if applicable
            if score_type == 'ordinal':
                validate_ordinal_scores(
                    df_part[score_column].values,
                    ordinal_max_score,
                    grouping_name
                )

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

            # Get number of parallel tasks from params if available, otherwise assume 1
            # In posthoc mode, each worker processes one grouping, so num_parallel_tasks = 1 per worker
            num_parallel_tasks = params.get('_num_parallel_tasks', 1)

            sampling_kwargs = gpu_utils.get_sampling_kwargs(
                params=params,
                gpu_available=gpu_available,
                gpu_backend=gpu_backend,
                num_parallel_tasks=num_parallel_tasks,
                auto_decide=True
            )

            # Log GPU status for this worker
            if gpu_available and sampling_kwargs.get('nuts_sampler') == 'numpyro':
                logger.info(f"Worker processing grouping {pid} with GPU acceleration ({gpu_backend}, chains={sampling_kwargs.get('chains', 1)})")
            elif gpu_available:
                logger.info(f"Worker processing grouping {pid} with GPU acceleration ({gpu_backend})")
            else:
                logger.info(f"Worker processing grouping {pid} with CPU-only (chains={sampling_kwargs.get('chains', 4)})")

            # Note: random_seed is now passed directly to PyMC via sampling_kwargs['random_seed']
            # (configured in gpu_utils.get_sampling_kwargs)

            item_summaries = []
            used_reps_dfs = []
            theta_lo, theta_hi, theta_width = None, None, None
            CI_record = []
            CI_slopes_hist = []
            entropy_history = []  # Track entropy CI history for hybrid stopping (item-level)
            group_entropy_history = []  # Track entropy CI history for hybrid stopping (group-level)
            ordinal_item_cache = {}  # Cache for item-level OrderedLogistic model reuse
            group_ordinal_model_cache = {}  # Separate cache for group-level hierarchical model
            initial_perf = df_part[score_column].mean()
            current_conservatism = conservatism if initial_perf < low_perf_threshold else 1.0

            item_ids = list(df_part['sample_id_num'].unique())
            n_items_total = len(item_ids)

            # Create PyMC model based on score type
            if score_type == 'binary':
                # Binary hierarchical model
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
            elif score_type == 'ordinal':
                # === ORDINAL HIERARCHICAL MODEL ===
                # Two model types available:
                #   - 'ordered_logistic': Cumulative link model (respects ordinal structure)
                #   - 'dirichlet': Dirichlet-Multinomial (treats categories as exchangeable)
                n_categories = ordinal_max_score + 1

                # Track which model type we actually use (may fall back)
                actual_model_type = ordinal_model_type

                if ordinal_model_type == 'ordered_logistic':
                    # === ORDERED LOGISTIC (Cumulative Link) ===
                    # Respects ordinal structure via latent scale with identified cutpoints
                    # Posthoc mode: all data available upfront, no pre-allocation needed
                    try:
                        ordinal_group_model = _create_ordered_logistic_hierarchical(
                            n_categories=n_categories,
                            n_items=n_items_total,
                            use_preallocation=False  # Posthoc: exact size, no padding
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to create ordered_logistic model for '{grouping_name}': {e}. "
                            f"Falling back to dirichlet."
                        )
                        actual_model_type = 'dirichlet'

                if actual_model_type == 'dirichlet':
                    # === DIRICHLET-MULTINOMIAL ===
                    # Treats categories as exchangeable (no ordinal structure)
                    with pm.Model() as ordinal_group_model:
                        # Group-level: baseline category probabilities
                        alpha_prior = np.ones(n_categories)
                        alpha_group = pm.Dirichlet("alpha_group", a=alpha_prior)

                        # Concentration parameter
                        kappa = pm.Gamma("kappa", alpha=2, beta=0.1)

                        # Mutable data containers
                        n_items_data = pm.Data("n_items", np.array(n_items_total, dtype="int64"))
                        item_counts_data = pm.Data("item_counts", np.zeros((n_items_total, n_categories), dtype="int64"))
                        item_ns_data = pm.Data("item_ns", np.ones(n_items_total, dtype="int64"))

                        # Item-level concentrations
                        alpha_item = alpha_group * kappa

                        # Item-specific probabilities with partial pooling
                        p_item = pm.Dirichlet("p_item", a=alpha_item, shape=(n_items_data, n_categories))

                        # Multinomial likelihood on category counts
                        obs = pm.Multinomial("obs", n=item_ns_data, p=p_item, observed=item_counts_data)

                        # Derived quantities for stopping criteria
                        modal_group = pm.Deterministic("modal_group", pt.argmax(alpha_group))
                        p_group_normalized = alpha_group / pm.math.sum(alpha_group)
                        entropy_group = pm.Deterministic(
                            "entropy_group",
                            -pm.math.sum(p_group_normalized * pm.math.log(p_group_normalized + 1e-10))
                        )

                group_ordinal_model_cache['model'] = ordinal_group_model
                group_ordinal_model_cache['n_items_last'] = n_items_total
                group_ordinal_model_cache['n_categories_last'] = n_categories
                group_ordinal_model_cache['model_type'] = actual_model_type
            for item_idx, item_id in enumerate(item_ids):
                df_item = df_part[df_part['sample_id_num'] == item_id].sort_values('epoch_num')

                # Reset entropy history for this item (within-item stabilization)
                entropy_history = []

                # Conditionally track scores based on score type
                if score_type == 'binary':
                    successes = 0
                    trials = 0
                else:  # ordinal
                    accumulated_scores = []

                used_reps = []
                ci_record = []
                ci_slopes_hist = []

                for start in range(0, len(df_item), rep_batch_size):
                    batch = df_item.iloc[start:start+rep_batch_size]
                    used_reps.extend(batch.itertuples(index=False))

                    # Compute CI based on score type
                    if score_type == 'binary':
                        successes += batch[score_column].sum()
                        trials += len(batch)
                        lo, hi, width = _beta_ci_adaptive(
                            successes, trials, cred_level=cred_level,
                            conservatism=current_conservatism,
                            low_perf_threshold=low_perf_threshold
                        )
                    else:  # ordinal
                        accumulated_scores.extend(batch[score_column].values)

                        # Use appropriate inference method for ordinal data
                        if ordinal_inference == 'modal':
                            lo, hi, width = _ordinal_ci_adaptive(
                                np.array(accumulated_scores),
                                ordinal_max_score=ordinal_max_score,
                                cred_level=cred_level,
                                conservatism=current_conservatism,
                                low_perf_threshold=low_perf_threshold
                            )
                        elif ordinal_inference == 'entropy':
                            entropy_lo, entropy_hi, width, diag = _ordinal_entropy_ci_adaptive(
                                np.array(accumulated_scores),
                                ordinal_max_score=ordinal_max_score,
                                cred_level=cred_level,
                                conservatism=current_conservatism,
                                low_perf_threshold=low_perf_threshold,
                                compute_kwargs=sampling_kwargs
                            )
                            # Use entropy CI directly for stopping decisions
                            # Note: entropy scale is inverted (high entropy = uncertain)
                            lo, hi = entropy_lo, entropy_hi
                        elif ordinal_inference == 'hybrid':
                            # Hybrid: check modal CI + entropy stabilization
                            should_stop, reason, diag = _ordinal_hybrid_stopping_criterion(
                                np.array(accumulated_scores),
                                ordinal_max_score=ordinal_max_score,
                                delta_item=delta_item,
                                cred_level=cred_level,
                                entropy_history=entropy_history,
                                entropy_threshold=entropy_threshold,
                                conservatism=current_conservatism,
                                low_perf_threshold=low_perf_threshold,
                                model_cache=ordinal_item_cache,
                                compute_kwargs=sampling_kwargs
                            )
                            # Use modal width for CI record tracking
                            width = diag.get('modal_width', delta_item)
                            # Override stopping decision with hybrid criterion
                            if should_stop:
                                logger.info(
                                    f"Stopping sample_id {item_id} (group {pid}) at epoch {batch['epoch_num'].iloc[-1]}: "
                                    f"Hybrid stopping ({reason}) | epochs used: {len(accumulated_scores)}"
                                )
                                break

                    ci_record.append(width)

                    # Check stopping criteria
                    # For hybrid ordinal, skip this check (hybrid function handles all stopping logic)
                    if ordinal_inference != 'hybrid' and width < delta_item:
                        if score_type == 'binary':
                            logger.info(f"Stopping sample_id {item_id} (group {pid}) at epoch {batch['epoch_num'].iloc[-1]}: CI width {width:.4f} < delta_item {delta_item} | epochs used: {trials}")
                        else:
                            logger.info(f"Stopping sample_id {item_id} (group {pid}) at epoch {batch['epoch_num'].iloc[-1]}: CI width {width:.4f} < delta_item {delta_item} | epochs used: {len(accumulated_scores)}")
                        break

                    if len(ci_record) >= stab_window:
                        recent_widths = ci_record[-stab_window:]
                        slope = np.polyfit(range(len(recent_widths)), recent_widths, 1)[0]
                        ci_slopes_hist.append(slope)

                        # Determine current performance for conservatism adjustment
                        if score_type == 'binary':
                            current_item_perf = df_item[score_column].mean()
                        else:
                            current_item_perf = np.mean(accumulated_scores) / ordinal_max_score

                        slope_threshold = CI_delta / current_conservatism if current_item_perf < low_perf_threshold else CI_delta

                        if (abs(slope) <= slope_threshold) and (len(ci_slopes_hist) >= 4):
                            recent_slopes = ci_slopes_hist[-3:]
                            slope_slopes = np.polyfit(range(len(recent_slopes)), recent_slopes, 1)[0]
                            if slope_slopes >= 0:
                                if current_item_perf >= low_perf_threshold:
                                    n_obs = trials if score_type == 'binary' else len(accumulated_scores)
                                    logger.info(f"Stopping sample_id {item_id} (group {pid}) due to CI stabilization: slope {slope:.6f} <= threshold {slope_threshold:.6f} | epochs used: {n_obs}")
                                    break
                                elif abs(slope) <= slope_threshold / 2:
                                    n_obs = trials if score_type == 'binary' else len(accumulated_scores)
                                    logger.info(f"Stopping low-performance sample_id {item_id} (group {pid}) due to strong CI stabilization: slope {slope:.6f} | epochs used: {n_obs}")
                                    break

                # Store item summary based on score type
                if score_type == 'binary':
                    item_summaries.append({'successes': successes, 'trials': trials})
                else:  # ordinal
                    # Store category counts for hierarchical Dirichlet-Multinomial model
                    counts = np.bincount(np.array(accumulated_scores).astype(int), minlength=ordinal_max_score + 1)
                    item_summaries.append({
                        'counts': counts,
                        'n_obs': len(accumulated_scores),
                        'modal_category': int(np.argmax(counts)),
                        'mean_score': float(np.mean(accumulated_scores)),
                        'successes': int(np.sum(accumulated_scores)),
                        'trials': len(accumulated_scores),
                        'scores': accumulated_scores.copy()  # Keep for backward compat
                    })

                used_reps_dfs.append(pd.DataFrame(used_reps, columns=df_item.columns))

                # Compute current performance estimate based on score type
                if score_type == 'binary':
                    current_perf_estimate = sum(s['successes'] for s in item_summaries) / sum(s['trials'] for s in item_summaries)
                else:  # ordinal
                    # Use mean_score stored in item_summaries (already computed)
                    total_obs = sum(s['n_obs'] for s in item_summaries)
                    if total_obs > 0:
                        # Weighted mean of item mean_scores
                        current_perf_estimate = sum(s['mean_score'] * s['n_obs'] for s in item_summaries) / total_obs / ordinal_max_score
                    else:
                        current_perf_estimate = 0.0

                current_conservatism = conservatism if current_perf_estimate < low_perf_threshold else 1.0

                # Group-level PyMC model refresh (only for binary scoring)
                if score_type == 'binary' and (((item_idx + 1) % pymc_refresh_every == 0) or (item_idx == len(item_ids) - 1)):
                    all_successes = np.array([s['successes'] for s in item_summaries], dtype=np.int32)
                    all_trials = np.array([s['trials'] for s in item_summaries], dtype=np.int32)
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

                # Group-level stopping for ordinal scoring (HIERARCHICAL)
                elif score_type == 'ordinal' and (((item_idx + 1) % pymc_refresh_every == 0) or (item_idx == len(item_ids) - 1)):
                    # Extract item counts matrix for hierarchical model
                    item_counts_matrix = np.array([s['counts'] for s in item_summaries])
                    item_ns = np.array([s['n_obs'] for s in item_summaries])

                    current_perf_normalized = current_perf_estimate / ordinal_max_score if ordinal_max_score > 0 else 0.0

                    if len(item_ns) > 0 and np.sum(item_ns) > 0:
                        # Compute group-level CI using hierarchical inference
                        if ordinal_inference == 'modal':
                            theta_lo, theta_hi, theta_width = _ordinal_ci_hierarchical_modal(
                                item_counts_matrix,
                                item_ns,
                                ordinal_max_score=ordinal_max_score,
                                cred_level=cred_level,
                                conservatism=current_conservatism,
                                low_perf_threshold=low_perf_threshold,
                                current_perf=current_perf_normalized,
                                model_cache=group_ordinal_model_cache,
                                sampling_kwargs=sampling_kwargs
                            )
                        elif ordinal_inference == 'entropy':
                            entropy_lo, entropy_hi, theta_width, diag = _ordinal_ci_hierarchical_entropy(
                                item_counts_matrix,
                                item_ns,
                                ordinal_max_score=ordinal_max_score,
                                cred_level=cred_level,
                                conservatism=current_conservatism,
                                low_perf_threshold=low_perf_threshold,
                                current_perf=current_perf_normalized,
                                model_cache=group_ordinal_model_cache,
                                sampling_kwargs=sampling_kwargs
                            )
                            # Use entropy values for stopping
                            theta_lo, theta_hi = entropy_lo, entropy_hi
                            # Update entropy history for stabilization
                            group_entropy_history.append((entropy_lo, entropy_hi, theta_width))
                        elif ordinal_inference == 'hybrid':
                            # Hierarchical hybrid: uses Dirichlet-Multinomial model
                            should_stop_group, reason_group, diag_group = _ordinal_hybrid_stopping_criterion_hierarchical(
                                item_counts_matrix,
                                item_ns,
                                ordinal_max_score=ordinal_max_score,
                                delta_item=delta_cap,
                                cred_level=cred_level,
                                entropy_history=group_entropy_history,
                                entropy_threshold=entropy_threshold,
                                conservatism=current_conservatism,
                                low_perf_threshold=low_perf_threshold,
                                current_perf=current_perf_normalized,
                                stabilization_threshold=entropy_stabilization_threshold,
                                model_cache=group_ordinal_model_cache,
                                sampling_kwargs=sampling_kwargs
                            )
                            theta_width = diag_group.get('modal_width', delta_cap)
                            # Extract CI bounds from diagnostics
                            if 'modal_ci' in diag_group:
                                theta_lo, theta_hi = diag_group['modal_ci']
                            else:
                                # Fallback to default
                                theta_lo, theta_hi = 0.0, 1.0

                        CI_record.append(theta_width)
                        effective_width = theta_width * current_conservatism if current_perf_estimate < low_perf_threshold else theta_width

                        # For hybrid, use the hybrid stopping decision directly
                        if ordinal_inference == 'hybrid' and should_stop_group:
                            logger.info(f"Stopping ordinal grouping {pid}: Hybrid group-level ({reason_group}) | sample_ids used: {len(item_summaries)}")
                            break
                        elif ordinal_inference != 'hybrid' and effective_width < delta_cap:
                            logger.info(f"Stopping ordinal grouping {pid}: CI width {effective_width:.4f} < delta_cap {delta_cap} | sample_ids used: {len(item_summaries)}")
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
                                        logger.info(f"Stopping ordinal grouping {pid} due to CI stabilization: slope {slope:.6f} <= threshold {slope_threshold:.6f} | sample_ids used: {len(item_summaries)}")
                                        break
                                    elif abs(slope) <= slope_threshold / 2:
                                        logger.info(f"Stopping low-performance ordinal grouping {pid} due to strong CI stabilization: slope {slope:.6f} | sample_ids used: {len(item_summaries)}")
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
        # No specific GPU assigned - let JAX/PyTensor auto-detect available devices
        # Do NOT set CUDA_VISIBLE_DEVICES to empty string as this disables GPU entirely
        os.environ['PYTENSOR_FLAGS'] = f'compiledir={unique_worker_dir},floatX=float32'
        # Let JAX auto-detect (don't force CPU-only mode)
        # os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Commented out to allow GPU detection
        # os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Commented out to allow GPU detection
        device_type = 'Auto-detect (GPU if available)'

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # Additional PyTensor isolation environment variables
    os.environ['PYTENSOR_FLAGS'] += ',optimizer=fast_compile,openmp=False'

    # Force PyTensor to use our compiledir by setting config directly after import
    # This handles cases where PyTensor might be imported later
    try:
        import pytensor
        pytensor.config.compiledir = unique_worker_dir
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
        # logger.info(f"Live worker {os.getpid()} using {device_type}, PyTensor compiledir: {unique_worker_dir}")  # Verbose
        pass

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
    Supports both binary and ordinal scoring.
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
        # logger.info(f"Processing grouping: {grouping}")  # Verbose

        # Extract parameters
        delta_item = params.get('delta_item', 0.05)
        delta_cap = params.get('delta_cap', 0.05)
        CI_delta = params.get('CI_delta', 0.00005)
        cred_level = params.get('cred_level', 0.95)
        conservatism = params.get('conservatism', 5)
        low_perf_threshold = params.get('low_performance_threshold', 0.01)
        rep_batch_size = params.get('rep_batch_size', 1)
        pymc_refresh_every = params.get('pymc_refresh_every', 2)
        stab_window = params.get('stab_window', 10)

        # Extract ordinal parameters
        ordinal_tasks = params.get('ordinal_tasks', None)
        continuous_tasks = params.get('continuous_tasks', None)
        ordinal_max_score = params.get('ordinal_max_score', 10)
        ordinal_inference = params.get('ordinal_inference', 'modal')
        ordinal_model_type = params.get('ordinal_model_type', 'ordered_logistic')
        entropy_threshold = params.get('entropy_threshold', 1.5)

        # Determine score type for this grouping
        score_type, bounds = determine_score_type_standalone(
            grouping,
            ordinal_tasks=ordinal_tasks,
            continuous_tasks=continuous_tasks,
            upper_bound=ordinal_max_score
        )
        # logger.info(f"Grouping '{grouping}' identified as {score_type.upper()}")  # Verbose

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

        # Get number of parallel tasks from params if available, otherwise assume 1
        # In live mode, each worker processes one grouping, so num_parallel_tasks = 1 per worker
        num_parallel_tasks = params.get('_num_parallel_tasks', 1)

        sampling_kwargs = gpu_utils.get_sampling_kwargs(
            params=params,
            gpu_available=gpu_available,
            gpu_backend=gpu_backend,
            num_parallel_tasks=num_parallel_tasks,
            auto_decide=True
        )

        # Log GPU status for this worker
        # if gpu_available and sampling_kwargs.get('nuts_sampler') == 'numpyro':
        #     logger.info(f"Worker processing grouping {grouping} with GPU acceleration ({gpu_backend}, chains={sampling_kwargs.get('chains', 1)})")
        # elif gpu_available:
        #     logger.info(f"Worker processing grouping {grouping} with GPU acceleration ({gpu_backend})")
        # else:
        #     logger.info(f"Worker processing grouping {grouping} with CPU-only (chains={sampling_kwargs.get('chains', 4)})")
        pass  # GPU/CPU status available from worker count logs

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

        # Initialize ordinal-specific variables
        ordinal_item_cache = {}  # Cache for item-level OrderedLogistic model
        entropy_history_per_item = {}

        # PyMC model for group-level (binary only)
        if score_type == 'binary':
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

            if score_type == 'binary':
                # === BINARY SCORING LOGIC ===
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
                        # logger.info(f"Stopping sample_id {item_id} in grouping {grouping}: CI width {width:.4f} < delta_item {delta_item} | epochs used: {trials}")  # Results in output
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
                                    # logger.info(f"Stopping sample_id {item_id} in grouping {grouping} due to CI stabilization: slope {slope:.6f} <= threshold {slope_threshold:.6f} | epochs used: {trials}")  # Results in output
                                    original_sample_id = df_grouping[df_grouping['sample_id_num'] == item_id][sample_id_column].iloc[0]
                                    stop_sample_ids.append(f"{grouping}_{original_sample_id}")
                                    break
                                elif abs(slope) <= slope_threshold / 2:
                                    # logger.info(f"Stopping low-performance sample_id {item_id} in grouping {grouping} due to strong CI stabilization: slope {slope:.6f} | epochs used: {trials}")  # Results in output
                                    original_sample_id = df_grouping[df_grouping['sample_id_num'] == item_id][sample_id_column].iloc[0]
                                    stop_sample_ids.append(f"{grouping}_{original_sample_id}")
                                    break
                sample_ci_records[item_id] = ci_record
                sample_ci_slopes[item_id] = ci_slopes_hist
                used_reps_dfs[item_id] = pd.DataFrame(used_reps, columns=df_item.columns)
                item_summaries.append({'successes': successes, 'trials': trials})

            else:
                # === ORDINAL SCORING LOGIC ===
                from .ordinal_utils import _ordinal_ci_adaptive
                from .ordinal_model import (
                    _ordinal_entropy_ci_adaptive,
                    _ordinal_hybrid_stopping_criterion
                )

                accumulated_scores = []
                used_reps = []
                entropy_history = entropy_history_per_item.get(item_id, [])

                for start in range(0, len(df_item), rep_batch_size):
                    batch = df_item.iloc[start:start+rep_batch_size]
                    accumulated_scores.extend(batch[score_column].tolist())
                    used_reps.extend(batch.itertuples(index=False))

                    # Compute CI based on inference mode
                    current_conservatism = conservatism if np.mean(accumulated_scores) / ordinal_max_score < low_perf_threshold else 1.0

                    if ordinal_inference == 'modal':
                        lo, hi, width = _ordinal_ci_adaptive(
                            np.array(accumulated_scores),
                            ordinal_max_score=ordinal_max_score,
                            cred_level=cred_level,
                            conservatism=current_conservatism,
                            low_perf_threshold=low_perf_threshold
                        )
                        if width < delta_item:
                            # logger.info(f"Stopping ordinal sample_id {item_id} in grouping {grouping}: Modal CI width {width:.4f} < delta_item {delta_item} | epochs used: {len(accumulated_scores)}")  # Results in output
                            original_sample_id = df_grouping[df_grouping['sample_id_num'] == item_id][sample_id_column].iloc[0]
                            stop_sample_ids.append(f"{grouping}_{original_sample_id}")
                            break

                    elif ordinal_inference == 'entropy':
                        lo, hi, width, diagnostics = _ordinal_entropy_ci_adaptive(
                            np.array(accumulated_scores),
                            ordinal_max_score=ordinal_max_score,
                            cred_level=cred_level,
                            conservatism=current_conservatism,
                            low_perf_threshold=low_perf_threshold,
                            model_cache=ordinal_item_cache,
                            compute_kwargs=sampling_kwargs
                        )
                        if width < delta_item:
                            # logger.info(f"Stopping ordinal sample_id {item_id} in grouping {grouping}: Entropy CI width {width:.4f} < delta_item {delta_item} | epochs used: {len(accumulated_scores)}")  # Results in output
                            original_sample_id = df_grouping[df_grouping['sample_id_num'] == item_id][sample_id_column].iloc[0]
                            stop_sample_ids.append(f"{grouping}_{original_sample_id}")
                            break

                    elif ordinal_inference == 'hybrid':
                        should_stop, reason, diagnostics = _ordinal_hybrid_stopping_criterion(
                            np.array(accumulated_scores),
                            ordinal_max_score=ordinal_max_score,
                            delta_item=delta_item,
                            cred_level=cred_level,
                            entropy_history=entropy_history,
                            entropy_threshold=entropy_threshold,
                            conservatism=current_conservatism,
                            low_perf_threshold=low_perf_threshold,
                            model_cache=ordinal_item_cache,
                            compute_kwargs=sampling_kwargs
                        )
                        if should_stop:
                            # logger.info(f"Stopping ordinal sample_id {item_id} in grouping {grouping} via {reason} | epochs used: {len(accumulated_scores)}")  # Results in output
                            original_sample_id = df_grouping[df_grouping['sample_id_num'] == item_id][sample_id_column].iloc[0]
                            stop_sample_ids.append(f"{grouping}_{original_sample_id}")
                            break

                entropy_history_per_item[item_id] = entropy_history
                used_reps_dfs[item_id] = pd.DataFrame(used_reps, columns=df_item.columns)
                # Store category counts for hierarchical Dirichlet-Multinomial model
                counts = np.bincount(np.array(accumulated_scores).astype(int), minlength=ordinal_max_score + 1)
                item_summaries.append({
                    'counts': counts,
                    'n_obs': len(accumulated_scores),
                    'modal_category': int(np.argmax(counts)),
                    'mean_score': float(np.mean(accumulated_scores)),
                    'successes': int(np.sum(accumulated_scores)),
                    'trials': len(accumulated_scores)
                })
            
            # Group-level stopping check (periodic)
            current_perf_estimate = np.sum([s['successes'] for s in item_summaries]) / np.sum([s['trials'] for s in item_summaries]) if item_summaries else 0
            if score_type == 'ordinal':
                # For ordinal, normalize by max score for conservatism check
                current_perf_estimate_normalized = current_perf_estimate / ordinal_max_score
                current_conservatism = conservatism if current_perf_estimate_normalized < low_perf_threshold else 1.0
            else:
                current_conservatism = conservatism if current_perf_estimate < low_perf_threshold else 1.0

            if ((item_idx + 1) % pymc_refresh_every == 0) or (item_idx == len(item_ids) - 1):
                if score_type == 'binary':
                    # === BINARY GROUP-LEVEL STOPPING ===
                    all_successes = np.array([s['successes'] for s in item_summaries], dtype=np.int32)
                    all_trials = np.array([s['trials'] for s in item_summaries], dtype=np.int32)
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
                            # logger.info(f"Stopping grouping {grouping}: CI width {effective_width:.4f} < delta_cap {delta_cap} | sample_ids used: {len(item_summaries)}")  # Results in output
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
                                        # logger.info(f"Stopping grouping {grouping} due to CI stabilization: slope {slope:.6f} <= threshold {slope_threshold:.6f} | sample_ids used: {len(item_summaries)}")  # Results in output
                                        stop_this_grouping.append(grouping)
                                        break
                                    elif abs(slope) <= slope_threshold / 2:
                                        # logger.info(f"Stopping low-performance grouping {grouping} due to strong CI stabilization: slope {slope:.6f} | sample_ids used: {len(item_summaries)}")  # Results in output
                                        stop_this_grouping.append(grouping)
                                        break

                else:
                    # === ORDINAL GROUP-LEVEL STOPPING ===
                    # Collect all ordinal scores from all items processed so far
                    all_ord_scores = []
                    for item_id_inner in item_ids[:item_idx+1]:
                        df_item_inner = df_grouping[df_grouping['sample_id_num'] == item_id_inner]
                        all_ord_scores.extend(df_item_inner[score_column].tolist())

                    # Initialize group-level entropy history
                    if not hasattr(params, '_group_entropy_history'):
                        group_entropy_history = []
                    else:
                        group_entropy_history = params.get('_group_entropy_history', [])

                    # Compute group-level ordinal CI
                    if ordinal_inference == 'modal':
                        from .ordinal_utils import _ordinal_ci_adaptive
                        lo, hi, width = _ordinal_ci_adaptive(
                            np.array(all_ord_scores),
                            ordinal_max_score=ordinal_max_score,
                            cred_level=cred_level,
                            conservatism=current_conservatism,
                            low_perf_threshold=low_perf_threshold
                        )
                        if width < delta_cap:
                            # logger.info(f"Stopping ordinal grouping {grouping}: Modal CI width {width:.4f} < delta_cap {delta_cap} | sample_ids used: {len(item_summaries)}")  # Results in output
                            stop_this_grouping.append(grouping)
                            break

                    elif ordinal_inference == 'entropy':
                        from .ordinal_utils import _ordinal_entropy_ci_adaptive
                        lo, hi, width, diagnostics = _ordinal_entropy_ci_adaptive(
                            np.array(all_ord_scores),
                            ordinal_max_score=ordinal_max_score,
                            cred_level=cred_level,
                            conservatism=current_conservatism,
                            low_perf_threshold=low_perf_threshold,
                            model_cache=ordinal_item_cache,
                            compute_kwargs=sampling_kwargs
                        )
                        if width < delta_cap:
                            # logger.info(f"Stopping ordinal grouping {grouping}: Entropy CI width {width:.4f} < delta_cap {delta_cap} | sample_ids used: {len(item_summaries)}")  # Results in output
                            stop_this_grouping.append(grouping)
                            break

                    elif ordinal_inference == 'hybrid':
                        from .ordinal_utils import _ordinal_hybrid_stopping_criterion
                        should_stop_group, reason_group, diagnostics_group = _ordinal_hybrid_stopping_criterion(
                            np.array(all_ord_scores),
                            ordinal_max_score=ordinal_max_score,
                            delta_item=delta_cap,
                            cred_level=cred_level,
                            entropy_history=group_entropy_history,
                            entropy_threshold=entropy_threshold,
                            conservatism=current_conservatism,
                            low_perf_threshold=low_perf_threshold,
                            model_cache=ordinal_item_cache,
                            compute_kwargs=sampling_kwargs
                        )
                        if should_stop_group:
                            # logger.info(f"Stopping ordinal grouping {grouping} via {reason_group} | sample_ids used: {len(item_summaries)}")  # Results in output
                            stop_this_grouping.append(grouping)
                            break
        
        return {
            'grouping': grouping,
            'stop_sample_ids': stop_sample_ids,
            'stop_this_grouping': stop_this_grouping
        }
    except Exception as e:
        logger.info(f"Error processing grouping {grouping}: {e}")
        import traceback
        logger.info(traceback.format_exc())
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
def optimal_stopping_posthoc(
    df: pd.DataFrame,
    params: Dict[str, Any],
    grouping_columns: List[str],
    sample_id_column: str,
    epoch_column: str,
    score_column: str = "score",
    display_progress: bool = True,
    generate_diagnostics: bool = False,
    diagnostics_prefix: str = "optstop_diagnostics",
    ordinal_tasks: Optional[List[str]] = None,
    ordinal_max_score: int = 10,
    ordinal_inference: str = 'modal',
    ordinal_model_type: str = 'ordered_logistic',
    continuous_tasks: Optional[List[str]] = None,
    entropy_threshold: float = 1.5,
    gpu_ids: Optional[List[int]] = None,
    max_workers: Optional[int] = None
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
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
      - ordinal_tasks: List of task/grouping name substrings that use ordinal scoring (default: None)
          * If None: All tasks use binary (0/1) scoring
          * If provided: Tasks whose grouping name contains any substring use ordinal scoring
          * Example: ['likert', 'rating'] → groupings containing these strings are ordinal
      - ordinal_max_score: Maximum score for ordinal tasks (default: 10)
          * Used to scale scores to [0,1] range for CI comparisons
          * Must match your actual score range (e.g., 5 for 0-5 scale, 10 for 0-10 scale)
      - ordinal_inference: Inference method for ordinal scoring (default: 'modal')
          * 'modal': Modal category bootstrap (estimates most common category)
          * 'entropy': OrderedLogistic entropy-based stopping (full categorical distribution)
          * 'hybrid': Combines modal CI + entropy stabilization (RECOMMENDED)
              - Stops via Pathway 1 if modal CI narrow (peaked data)
              - Stops via Pathway 2 if entropy stabilized (non-peaked but stable data)
      - ordinal_model_type: Hierarchical model type for ordinal data (default: 'ordered_logistic')
          * 'ordered_logistic': Cumulative link model with identified cutpoints (recommended)
          * 'dirichlet': Dirichlet-Multinomial (treats categories as exchangeable)
          * Falls back to 'dirichlet' if ordered_logistic sampling fails
      - continuous_tasks: List of task/grouping name substrings that use continuous bounded scoring (default: None)
          * If None: Continuous inference is not used
          * If provided: Tasks whose grouping name contains any substring use Beta distribution inference
          * Example: ['mean_score', 'aggregated'] → groupings containing these strings use continuous inference
          * Scores must be pre-aggregated floats in [0, ordinal_max_score] range
          * Priority: continuous_tasks > ordinal_tasks > binary (default)
      - entropy_threshold: Threshold for entropy CI width when using entropy inference (default: 1.5)
          * Only used for pure 'entropy' mode (not 'hybrid')
          * Lower values → more aggressive stopping
          * Higher values → more conservative stopping
          * Recommended range: 1.0-2.0
      - gpu_ids: List of GPU IDs to use for parallel processing. If None, uses CPU-only. If provided, assigns GPUs to workers cyclically.
      - max_workers: Number of parallel workers. If None, uses len(gpu_ids) when GPUs specified, otherwise uses CPU count.

    Returns a pruned DataFrame and a list of summary dicts (one per grouping-task).

    Note on ordinal scoring:
      - Modal inference: Estimates the most common response category (faster, less computational)
      - Entropy inference: Uses OrderedLogistic model for full categorical distribution (more accurate for uncertain data)
      - See ORDEREDLOGISTIC_ENTROPY_IMPLEMENTATION_STATUS.md for detailed comparison
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

    # Add number of parallel tasks to params for smart GPU/CPU decision making
    # This helps the adaptive logic decide whether GPU or CPU multiprocessing is better
    params_with_context = params.copy()
    params_with_context['_num_parallel_tasks'] = len(groupings)

    # Validate ordinal inference parameter
    if ordinal_inference not in ['modal', 'entropy', 'hybrid']:
        raise ValueError(f"ordinal_inference must be 'modal', 'entropy', or 'hybrid', got '{ordinal_inference}'")

    # Add score type parameters to params dict for worker processes
    # These are always set so workers can use determine_score_type_standalone
    params_with_context['ordinal_tasks'] = ordinal_tasks
    params_with_context['continuous_tasks'] = continuous_tasks
    params_with_context['ordinal_max_score'] = ordinal_max_score
    params_with_context['ordinal_inference'] = ordinal_inference
    params_with_context['ordinal_model_type'] = ordinal_model_type
    params_with_context['entropy_threshold'] = entropy_threshold

    # Validate ordinal scores upfront for all ordinal groupings
    if ordinal_tasks is not None:
        for grouping_name in df['grouping'].unique():
            score_type, bounds = determine_score_type_standalone(
                grouping_name,
                ordinal_tasks=ordinal_tasks,
                continuous_tasks=continuous_tasks,
                upper_bound=ordinal_max_score
            )
            if score_type == 'ordinal':
                grouping_data = df[df['grouping'] == grouping_name]
                validate_ordinal_scores(
                    grouping_data[score_column].values,
                    ordinal_max_score,
                    grouping_name
                )
                logger.info(f"Validated ordinal scores for grouping '{grouping_name}' (using {ordinal_inference} inference)")
            elif score_type in ('continuous_bounded', 'continuous_01'):
                logger.info(f"Grouping '{grouping_name}' will use continuous bounded inference (Beta distribution)")

    args_list = [(pid, df_part, params_with_context, score_column) for pid, df_part in groupings]

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
                    # Build per-grouping score_type map using the same logic as the inference code
                    # This ensures diagnostics match the actual inference pathway used for each grouping
                    unique_groupings = sorted(df['grouping'].unique().tolist())
                    diag_score_type_map = {}
                    diag_ordinal_max_scores = {}

                    for grouping_name in unique_groupings:
                        # Use determine_score_type_standalone to get the score type for this grouping
                        # This matches the logic used in the inference workers
                        inferred_type, bounds = determine_score_type_standalone(
                            grouping_name,
                            ordinal_tasks=ordinal_tasks,
                            continuous_tasks=continuous_tasks,
                            upper_bound=ordinal_max_score
                        )

                        # Map internal score types to diagnostic score types
                        # 'binary' -> 'binary', 'ordinal' -> 'ordinal'
                        # 'continuous_01', 'continuous_bounded' -> 'continuous'
                        if inferred_type == 'binary':
                            diag_score_type_map[grouping_name] = 'binary'
                        elif inferred_type == 'ordinal':
                            diag_score_type_map[grouping_name] = 'ordinal'
                            diag_ordinal_max_scores[grouping_name] = int(bounds.get('upper', ordinal_max_score))
                        else:  # continuous_01, continuous_bounded
                            diag_score_type_map[grouping_name] = 'continuous'

                    logger.debug(f"Diagnostic score types: {diag_score_type_map}")

                    _generate_diagnostic_plots(
                        original_df[common_columns],
                        final_used_df[common_columns],
                        diagnostics_prefix,
                        score_col=score_column,
                        grouping_columns=grouping_columns,
                        sample_id_column=sample_id_column,
                        score_type=diag_score_type_map,
                        ordinal_max_score=diag_ordinal_max_scores if diag_ordinal_max_scores else ordinal_max_score
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
def optimal_stopping_live_single(
    df_grouping: pd.DataFrame,
    grouping_name: str,
    params: Dict[str, Any],
    sample_id_column: str,
    epoch_column: str,
    score_column: str = "score",
    stabilization_history: Optional[Dict[str, List[float]]] = None,
    max_n_items: Optional[int] = None,  # Fix 1: Pre-allocation parameter
    ordinal_tasks: Optional[List[str]] = None,
    ordinal_max_score: int = 10,
    ordinal_inference: str = 'modal',
    ordinal_model_type: str = 'ordered_logistic',
    entropy_threshold: float = 1.5,
    sampling_kwargs: Optional[Dict[str, Any]] = None,
    model_caches: Optional[Dict[str, Dict[str, Any]]] = None,
    item_entropy_histories: Optional[Dict[Any, List]] = None
) -> Dict[str, Any]:
    # TIMING_TEST: Start overall function timing
    import time
    _function_start_time = time.perf_counter()
    """
    Run optimal stopping for a SINGLE grouping with stateful stabilization history.

    Designed for inspect_ai integration where:
    - Sample-level checks run on all completed samples
    - Group-level check runs automatically at the end
    - Stabilization histories (CI widths, slopes, entropy) persist across calls

    Args:
        df_grouping: DataFrame containing data for ONE grouping only
        grouping_name: Name/identifier for this grouping (e.g., "model1-task1")
        params: Dictionary of stopping parameters (delta_item, delta_cap, cred_level, etc.)
        sample_id_column: Column name for sample ID
        epoch_column: Column name for epoch/trial
        score_column: Column name for score (default: 'score')
        stabilization_history: Dictionary with stabilization metrics from previous calls:
            - 'ci_width_history': List of CI widths at group level
            - 'ci_slope_history': List of CI slopes at group level
            - 'entropy_history': List of entropy values (ordinal only)
            - 'n_samples_evaluated': Number of samples processed so far
            If None, initializes empty history.
        max_n_items: Maximum number of unique items expected in this grouping.
            If provided, models are pre-allocated at this size to avoid recompilation
            as n_items grows. Unobserved items are masked via explicit obs_weight.
            If None, uses current observed n_items (original dynamic behavior).
            (Fix 1: Pre-allocation for scaling)
        ordinal_tasks: List of substrings to identify if this grouping uses ordinal scoring
        ordinal_max_score: Maximum score for ordinal data (default: 10)
        ordinal_inference: Ordinal inference mode ('modal', 'entropy', 'hybrid')
        ordinal_model_type: Hierarchical model type for ordinal data (default: 'ordered_logistic')
            - 'ordered_logistic': Cumulative link model with identified cutpoints (recommended)
            - 'dirichlet': Dirichlet-Multinomial (treats categories as exchangeable)
            Falls back to 'dirichlet' if ordered_logistic sampling fails.
        entropy_threshold: Entropy threshold for hybrid mode validation
        sampling_kwargs: Pre-configured PyMC sampling kwargs (chains, draws, etc.)
            If None, will be auto-configured based on available resources.
        model_caches: Dict of caches for PyMC model reuse across all pathways (OPTIMIZATION #2)
            Keys: 'binary_item', 'binary_group', 'ordinal_item', 'ordinal_group',
                  'continuous_item', 'continuous_group'
            Persists model compilation across inference calls. If None, creates new caches.
        item_entropy_histories: Dict mapping sample_id to list of entropy (lo, hi, width) tuples.
            Used for sample-level ordinal hybrid entropy stabilization (Pathway 2).
            Persists across calls to enable cross-call stabilization detection.
            If None, initializes empty dict. (Issue #6 fix)

    Returns:
        Dict with:
            - 'grouping': Grouping name
            - 'stop_sample_ids': List of "grouping_sample_id" strings to stop
            - 'stop_this_grouping': List containing grouping name if should stop (else empty)
            - 'stabilization_history': Updated history dict with new values appended
            - 'model_caches': Updated cache dict (pass to next call for persistence)
            - 'item_entropy_histories': Updated per-item entropy histories (pass to next call)
            - 'metadata': Dict with stopping reasons, basis values, and diagnostics

    Example:
        >>> # First call - check samples 1-10
        >>> result1 = optimal_stopping_live_single(
        ...     df_grouping=df[df['grouping']=='model1-task1'],
        ...     grouping_name='model1-task1',
        ...     params={'delta_item': 0.05, 'delta_cap': 0.05},
        ...     sample_id_column='item_id',
        ...     epoch_column='trial',
        ...     stabilization_history=None  # First call
        ... )
        >>>
        >>> # Second call - check samples 11-20, preserve history
        >>> result2 = optimal_stopping_live_single(
        ...     df_grouping=df[df['grouping']=='model1-task1'],
        ...     grouping_name='model1-task1',
        ...     params={'delta_item': 0.05, 'delta_cap': 0.05},
        ...     sample_id_column='item_id',
        ...     epoch_column='trial',
        ...     stabilization_history=result1['stabilization_history'],  # Pass history
        ...     item_entropy_histories=result1['item_entropy_histories']  # Pass item histories
        ... )
        >>> # Group-level check runs automatically at end of each call
    """
    logger = logging.getLogger('optstop.live_single')

    # Initialize or use provided stabilization history
    if stabilization_history is None:
        stabilization_history = {
            'ci_width_history': [],
            'ci_slope_history': [],
            'entropy_history': [],
            'n_samples_evaluated': 0
        }

    # Extract parameters
    delta_item = params.get('delta_item', 0.05)
    delta_cap = params.get('delta_cap', 0.05)
    CI_delta = params.get('CI_delta', 0.00005)
    cred_level = params.get('cred_level', 0.95)
    conservatism = params.get('conservatism', 5)
    low_perf_threshold = params.get('low_performance_threshold', 0.01)
    rep_batch_size = params.get('rep_batch_size', 1)
    stab_window = params.get('stab_window', 10)
    # Entropy stabilization threshold for ordinal hybrid stopping
    # Default 0.002 = 0.2% relative change threshold
    entropy_stabilization_threshold = params.get('entropy_stabilization_threshold', 0.002)

    # Determine score type for this grouping
    is_aggregated = params.get('is_aggregated', False)
    score_type, bounds = determine_score_type(
        grouping_name,
        ordinal_tasks,
        is_aggregated=is_aggregated,
        upper_bound=ordinal_max_score
    )
    # logger.info(f"Processing grouping '{grouping_name}' as {score_type.upper()}")  # Verbose

    # Extract bounds for continuous inference
    lower_bound = bounds['lower']
    upper_bound = bounds['upper']

    # Auto-configure sampling kwargs if not provided
    if sampling_kwargs is None:
        gpu_available, gpu_backend, gpu_info = gpu_utils.check_gpu_availability()
        sampling_kwargs = gpu_utils.get_sampling_kwargs(
            params=params,
            gpu_available=gpu_available,
            gpu_backend=gpu_backend,
            num_parallel_tasks=1,
            auto_decide=True,
            score_type=score_type  # Deprecated but kept for compatibility
        )

    # Initialize result structure
    stop_sample_ids = []
    stop_this_grouping = []
    metadata = {
        'sample_stopping_reasons': {},  # sample_id -> reason dict
        'group_stopping_reason': None
    }

    # Create numeric columns for processing
    df_work = df_grouping.copy()
    df_work['sample_id_num'] = df_work[sample_id_column].astype('category').cat.codes
    df_work['epoch_num'] = df_work[epoch_column].astype(int)

    item_ids = sorted(df_work['sample_id_num'].unique())

    # === PRE-ALLOCATION LOGIC (Fix 1) ===
    # Determine model size: use max_n_items if provided for pre-allocation,
    # otherwise fall back to current observed n_items (original behavior)
    current_n_items = len(item_ids)
    if max_n_items is not None and max_n_items >= current_n_items:
        model_n_items = max_n_items
    else:
        model_n_items = current_n_items
        if max_n_items is not None and max_n_items < current_n_items:
            logger.warning(
                f"max_n_items ({max_n_items}) < current_n_items ({current_n_items}) "
                f"for grouping '{grouping_name}'. Using current_n_items instead."
            )

    # Create observation weight vector for explicit masking
    # 1.0 for observed items (indices 0 to current_n_items-1)
    # 0.0 for unobserved/padding items (indices current_n_items to model_n_items-1)
    obs_weight = np.zeros(model_n_items, dtype=np.float64)
    obs_weight[:current_n_items] = 1.0

    # Initialize model caches (OPTIMIZATION #2: Extract from passed-in dict)
    if model_caches is None:
        model_caches = {
            'binary_item': {},
            'binary_group': {},
            'ordinal_item': {},
            'ordinal_group': {},
            'continuous_item': {},
            'continuous_group': {}
        }

    # Extract individual caches for easier access
    binary_item_cache = model_caches.get('binary_item', {})
    binary_group_cache = model_caches.get('binary_group', {})
    ordinal_item_cache = model_caches.get('ordinal_item', {})
    ordinal_group_cache = model_caches.get('ordinal_group', {})
    continuous_item_cache = model_caches.get('continuous_item', {})
    continuous_group_cache = model_caches.get('continuous_group', {})

    # Initialize or use provided item-level entropy histories (Issue #6 fix)
    # This enables Pathway 2 (entropy stabilization) at sample level for ordinal hybrid
    if item_entropy_histories is None:
        entropy_history_per_item = {}
    else:
        entropy_history_per_item = item_entropy_histories

    # PyMC models for group-level hierarchical inference
    # Compiled once before item loop, updated with data during group-level checks

    if score_type == 'binary':
        # === BINARY HIERARCHICAL MODEL ===
        # Binomial likelihood with logit-scale hierarchical structure
        # Fix 1: Pre-allocation with explicit masking to avoid recompilation
        # OPTIMIZATION #2: Cache model - reuse if current data fits within allocated size
        #
        # Two modes:
        #   - use_preallocation=True (max_n_items specified): Fixed shape + obs_weight masking
        #   - use_preallocation=False (max_n_items=None): Dynamic mode with mutable n_items

        # Determine current use_preallocation setting
        use_preallocation = (max_n_items is not None)

        if 'model' in binary_group_cache:
            cached_model_n_items = binary_group_cache.get('model_n_items', 0)
            cached_use_preallocation = binary_group_cache.get('use_preallocation', True)
            # Cache valid if: same preallocation mode AND data fits within allocated size
            if (cached_use_preallocation == use_preallocation and
                current_n_items <= cached_model_n_items):
                # Safe to reuse - current data fits within allocated model size
                model = binary_group_cache['model']
                # logger.debug(f"✓ CACHE HIT: Reusing binary model (current={current_n_items}, allocated={cached_model_n_items})")
            else:
                # Must recreate - mode changed or current data exceeds allocated model size
                if cached_use_preallocation != use_preallocation:
                    logger.info(f"Binary model cache invalidated: use_preallocation changed from {cached_use_preallocation} to {use_preallocation}")
                else:
                    logger.info(f"Binary model cache invalidated: n_items {current_n_items} > allocated {cached_model_n_items}")
                binary_group_cache.clear()

        if 'model' not in binary_group_cache:
            with pm.Model() as model:
                # Group-level priors (unchanged)
                mu_group = pm.Normal("mu_group", mu=2, sigma=1.5)
                sigma_group = pm.Exponential("sigma_group", lam=1.0)

                if use_preallocation:
                    # PRE-ALLOCATED MODE: Fixed shape + obs_weight masking
                    # Data containers - allocated at model_n_items (may be > current_n_items)
                    # Use trials=1 (not 0) as safe default for unobserved items
                    successes_data = pm.Data("successes", np.zeros(model_n_items, dtype="int64"))
                    trials_data = pm.Data("trials", np.ones(model_n_items, dtype="int64"))
                    obs_weight_data = pm.Data("obs_weight", np.zeros(model_n_items, dtype="float64"))

                    # Item random effects - fixed shape at model_n_items
                    z = pm.Normal("z", mu=0, sigma=1, shape=model_n_items)

                    # Item-level means
                    mu_item = pm.Deterministic("mu_item", mu_group + z * sigma_group)
                    mu_item_clipped = pm.Deterministic("mu_item_clipped", pm.math.clip(mu_item, -6.0, 6.0))
                    Theta = pm.Deterministic("Theta", pm.math.sigmoid(mu_item_clipped))

                    # EXPLICIT MASKING: Use weighted log-likelihood via Potential
                    # obs_weight=0 for unobserved items → zero contribution to likelihood
                    binomial_dist = pm.Binomial.dist(n=trials_data, p=Theta)
                    log_lik = obs_weight_data * pm.logp(binomial_dist, successes_data)
                    pm.Potential("obs_likelihood", pm.math.sum(log_lik))
                else:
                    # DYNAMIC MODE: Original behavior with mutable n_items
                    # Model is recompiled when n_items changes, but allows full rollback
                    n_items_data = pm.Data("n_items", np.array(model_n_items, dtype="int64"))
                    successes_data = pm.Data("successes", np.zeros(model_n_items, dtype="int64"))
                    trials_data = pm.Data("trials", np.ones(model_n_items, dtype="int64"))

                    # Item random effects - shape depends on n_items_data
                    z = pm.Normal("z", mu=0, sigma=1, shape=n_items_data)

                    # Item-level means
                    mu_item = pm.Deterministic("mu_item", mu_group + z * sigma_group)
                    mu_item_clipped = pm.Deterministic("mu_item_clipped", pm.math.clip(mu_item, -6.0, 6.0))
                    Theta = pm.Deterministic("Theta", pm.math.sigmoid(mu_item_clipped))

                    # Direct observation (original behavior)
                    obs = pm.Binomial("obs", n=trials_data, p=Theta, observed=successes_data)

            binary_group_cache['model'] = model
            binary_group_cache['model_n_items'] = model_n_items  # Track allocated size
            binary_group_cache['use_preallocation'] = use_preallocation
            logger.info(f"Created binary model for '{grouping_name}' (allocated={model_n_items}, current={current_n_items}, prealloc={use_preallocation})")

    elif score_type in ['continuous_01', 'continuous_bounded']:
        # === CONTINUOUS HIERARCHICAL MODEL (AGGREGATED) ===
        # Uses aggregated item-level statistics instead of individual observations
        # This dramatically reduces computational cost (50 likelihoods vs 950)
        #
        # Model Structure:
        #   - Group level: mu_group (logit mean), sigma_group (between-item SD)
        #   - Item level: mu_item (transformed to [0,1]), phi_item (precision)
        #   - Observation level: Normal likelihood on aggregated means with known variance
        #
        # Key Change from Original:
        #   BEFORE: Beta likelihood on ALL individual observations (950 evaluations)
        #   AFTER: Normal likelihood on aggregated item means (50 evaluations)
        #   SPEEDUP: ~19x fewer likelihood evaluations per MCMC step
        #
        # Statistical Justification:
        #   By Central Limit Theorem, sample means of Beta-distributed observations
        #   follow approximately Normal distribution with:
        #   - Mean = true mu_i
        #   - SD = sqrt(mu_i * (1-mu_i) / (phi * n_i))
        #
        # This is analogous to how Binary uses aggregated Binomial (successes/trials)
        # rather than individual 0/1 observations.
        #
        # Fix 1: Pre-allocation with explicit masking to avoid recompilation
        # OPTIMIZATION #2: Cache model - reuse if current data fits within allocated size
        #
        # Two modes:
        #   - use_preallocation=True (max_n_items specified): Fixed shape + obs_weight masking
        #   - use_preallocation=False (max_n_items=None): Dynamic mode with mutable n_items

        # Determine current use_preallocation setting
        use_preallocation = (max_n_items is not None)

        if 'model' in continuous_group_cache:
            cached_model_n_items = continuous_group_cache.get('model_n_items', 0)
            cached_use_preallocation = continuous_group_cache.get('use_preallocation', True)
            # Cache valid if: same preallocation mode AND data fits within allocated size
            if (cached_use_preallocation == use_preallocation and
                current_n_items <= cached_model_n_items):
                continuous_model = continuous_group_cache['model']
                # logger.debug(f"✓ CACHE HIT: Reusing continuous model (current={current_n_items}, allocated={cached_model_n_items})")
            else:
                # Must recreate - mode changed or current data exceeds allocated model size
                if cached_use_preallocation != use_preallocation:
                    logger.info(f"Continuous model cache invalidated: use_preallocation changed from {cached_use_preallocation} to {use_preallocation}")
                else:
                    logger.info(f"Continuous model cache invalidated: n_items {current_n_items} > allocated {cached_model_n_items}")
                continuous_group_cache.clear()

        if 'model' not in continuous_group_cache:
            with pm.Model() as continuous_model:
                # Group-level parameters (logit scale for mean)
                mu_group = pm.Normal("mu_group", mu=0, sigma=1.5)  # Group mean (logit scale)
                sigma_group = pm.Exponential("sigma_group", lam=1.0)  # Between-item SD
                phi_group = pm.Gamma("phi_group", alpha=2, beta=1.0)  # Group-level precision

                if use_preallocation:
                    # PRE-ALLOCATED MODE: Fixed shape + obs_weight masking
                    # Data containers - allocated at model_n_items (may be > current_n_items)
                    # CRITICAL: item_ns must be >= 1 to avoid division by zero in obs_sd calculation
                    item_means_data = pm.Data("item_means", np.full(model_n_items, 0.5, dtype="float64"))
                    item_ns_data = pm.Data("item_ns", np.ones(model_n_items, dtype="int64"))  # MUST be >= 1
                    obs_weight_data = pm.Data("obs_weight", np.zeros(model_n_items, dtype="float64"))

                    # Item-level means (hierarchical, logit scale) - fixed shape at model_n_items
                    z = pm.Normal("z", mu=0, sigma=1, shape=model_n_items)
                    mu_item_logit = pm.Deterministic("mu_item_logit", mu_group + z * sigma_group)
                    mu_item_logit_clipped = pm.Deterministic("mu_item_logit_clipped",
                                                             pm.math.clip(mu_item_logit, -6.0, 6.0))
                    mu_item = pm.Deterministic("mu_item", pm.math.sigmoid(mu_item_logit_clipped))

                    # Item-level precision (allows heterogeneity in within-item variance)
                    z_phi = pm.Normal("z_phi", mu=0, sigma=1, shape=model_n_items)
                    log_phi_item = pm.Deterministic("log_phi_item",
                                                    pm.math.log(phi_group) + z_phi * 0.5)
                    phi_item = pm.Deterministic("phi_item", pm.math.exp(log_phi_item))

                    # Compute observation SD for each item based on theoretical variance
                    # SD(sample_mean) = sqrt(mu * (1-mu) / (phi * n))
                    mu_item_clipped_for_var = pm.math.clip(mu_item, 0.01, 0.99)
                    obs_sd = pm.math.sqrt(mu_item_clipped_for_var * (1 - mu_item_clipped_for_var) /
                                         (phi_item * item_ns_data))

                    # EXPLICIT MASKING: Use weighted log-likelihood via Potential
                    # obs_weight=0 for unobserved items → zero contribution to likelihood
                    normal_dist = pm.Normal.dist(mu=mu_item, sigma=obs_sd)
                    log_lik = obs_weight_data * pm.logp(normal_dist, item_means_data)
                    pm.Potential("obs_likelihood", pm.math.sum(log_lik))
                else:
                    # DYNAMIC MODE: Original behavior with mutable n_items
                    # Model is recompiled when n_items changes, but allows full rollback
                    n_items_data = pm.Data("n_items", np.array(model_n_items, dtype="int64"))
                    item_means_data = pm.Data("item_means", np.full(model_n_items, 0.5, dtype="float64"))
                    item_ns_data = pm.Data("item_ns", np.ones(model_n_items, dtype="int64"))  # MUST be >= 1

                    # Item-level means (hierarchical, logit scale) - shape depends on n_items_data
                    z = pm.Normal("z", mu=0, sigma=1, shape=n_items_data)
                    mu_item_logit = pm.Deterministic("mu_item_logit", mu_group + z * sigma_group)
                    mu_item_logit_clipped = pm.Deterministic("mu_item_logit_clipped",
                                                             pm.math.clip(mu_item_logit, -6.0, 6.0))
                    mu_item = pm.Deterministic("mu_item", pm.math.sigmoid(mu_item_logit_clipped))

                    # Item-level precision (allows heterogeneity in within-item variance)
                    z_phi = pm.Normal("z_phi", mu=0, sigma=1, shape=n_items_data)
                    log_phi_item = pm.Deterministic("log_phi_item",
                                                    pm.math.log(phi_group) + z_phi * 0.5)
                    phi_item = pm.Deterministic("phi_item", pm.math.exp(log_phi_item))

                    # Compute observation SD for each item based on theoretical variance
                    # SD(sample_mean) = sqrt(mu * (1-mu) / (phi * n))
                    mu_item_clipped_for_var = pm.math.clip(mu_item, 0.01, 0.99)
                    obs_sd = pm.math.sqrt(mu_item_clipped_for_var * (1 - mu_item_clipped_for_var) /
                                         (phi_item * item_ns_data))

                    # Direct observation (original behavior)
                    obs = pm.Normal("obs", mu=mu_item, sigma=obs_sd, observed=item_means_data)

            continuous_group_cache['model'] = continuous_model
            continuous_group_cache['model_n_items'] = model_n_items  # Track allocated size
            continuous_group_cache['use_preallocation'] = use_preallocation
            logger.info(f"Created continuous model for '{grouping_name}' (allocated={model_n_items}, current={current_n_items}, prealloc={use_preallocation})")

    elif score_type == 'ordinal':
        # === ORDINAL HIERARCHICAL MODEL ===
        # Two model types available:
        #   - 'ordered_logistic': Cumulative link model (respects ordinal structure)
        #   - 'dirichlet': Dirichlet-Multinomial (treats categories as exchangeable)
        #
        # Fix 1: Pre-allocation with explicit masking to avoid recompilation
        # Cache model - reuse if current data fits, n_categories matches, and model type matches
        n_categories = ordinal_max_score + 1

        # Check cache validity
        # Determine current use_preallocation setting
        current_use_preallocation = (max_n_items is not None)

        if 'model' in ordinal_group_cache:
            cached_model_n_items = ordinal_group_cache.get('model_n_items', 0)
            cached_n_categories = ordinal_group_cache.get('n_categories_last', 0)
            cached_model_type = ordinal_group_cache.get('model_type', 'dirichlet')
            cached_use_preallocation = ordinal_group_cache.get('use_preallocation', True)

            # Cache valid if: data fits, categories match, model type matches, AND preallocation mode matches
            if (current_n_items <= cached_model_n_items and
                cached_n_categories == n_categories and
                cached_model_type == ordinal_model_type and
                cached_use_preallocation == current_use_preallocation):
                ordinal_model = ordinal_group_cache['model']
                # logger.debug(f"✓ CACHE HIT: Reusing ordinal model (current={current_n_items}, allocated={cached_model_n_items})")
            else:
                logger.info(f"Ordinal model cache invalidated: n_items {current_n_items} > allocated {cached_model_n_items} or config changed (prealloc: {cached_use_preallocation} -> {current_use_preallocation})")
                ordinal_group_cache.clear()

        if 'model' not in ordinal_group_cache:
            # Track which model type we actually use (may fall back)
            actual_model_type = ordinal_model_type

            if ordinal_model_type == 'ordered_logistic':
                # === ORDERED LOGISTIC (Cumulative Link) ===
                # Respects ordinal structure via latent scale with identified cutpoints
                # use_preallocation determined by whether max_n_items was provided
                use_preallocation = (max_n_items is not None)
                try:
                    ordinal_model = _create_ordered_logistic_hierarchical(
                        n_categories=n_categories,
                        n_items=model_n_items,
                        use_preallocation=use_preallocation
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to create ordered_logistic model for '{grouping_name}': {e}. "
                        f"Falling back to dirichlet."
                    )
                    actual_model_type = 'dirichlet'

            if actual_model_type == 'dirichlet':
                # === DIRICHLET-MULTINOMIAL ===
                # Treats categories as exchangeable (no ordinal structure)
                # use_preallocation determined by whether max_n_items was provided
                use_preallocation = (max_n_items is not None)

                with pm.Model() as ordinal_model:
                    # Group-level: baseline category probabilities
                    alpha_prior = np.ones(n_categories)
                    alpha_group = pm.Dirichlet("alpha_group", a=alpha_prior)

                    # Concentration parameter
                    kappa = pm.Gamma("kappa", alpha=2, beta=0.1)

                    # Item-level concentrations
                    alpha_item = alpha_group * kappa

                    if use_preallocation:
                        # === PRE-ALLOCATED MODE (Fix 1) ===
                        # Fixed shape with explicit obs_weight masking
                        # Data containers - allocated at model_n_items (may be > current_n_items)
                        # For valid Multinomial: sum(item_counts[i]) must equal item_ns[i]
                        # Default: item_ns=1, item_counts=[1,0,0,...] (one count in first category)
                        default_counts = np.zeros((model_n_items, n_categories), dtype="int64")
                        default_counts[:, 0] = 1  # One count in first category ensures sum = 1 = item_ns
                        item_counts_data = pm.Data("item_counts", default_counts)
                        item_ns_data = pm.Data("item_ns", np.ones(model_n_items, dtype="int64"))
                        obs_weight_data = pm.Data("obs_weight", np.zeros(model_n_items, dtype="float64"))

                        # Item-specific probabilities with partial pooling - fixed shape at model_n_items
                        p_item = pm.Dirichlet("p_item", a=alpha_item, shape=(model_n_items, n_categories))

                        # EXPLICIT MASKING: Use weighted log-likelihood via Potential
                        # obs_weight=0 for unobserved items → zero contribution to likelihood
                        multinomial_dist = pm.Multinomial.dist(n=item_ns_data, p=p_item)
                        log_lik = obs_weight_data * pm.logp(multinomial_dist, item_counts_data)
                        pm.Potential("obs_likelihood", pm.math.sum(log_lik))
                    else:
                        # === DYNAMIC MODE (original behavior) ===
                        # Shape changes with n_items; model recompiles when n_items changes
                        n_items_data = pm.Data("n_items", np.array(model_n_items, dtype="int64"))
                        default_counts = np.zeros((model_n_items, n_categories), dtype="int64")
                        item_counts_data = pm.Data("item_counts", default_counts)
                        item_ns_data = pm.Data("item_ns", np.ones(model_n_items, dtype="int64"))

                        # Item-specific probabilities with partial pooling - dynamic shape
                        p_item = pm.Dirichlet("p_item", a=alpha_item, shape=(n_items_data, n_categories))

                        # DIRECT MULTINOMIAL LIKELIHOOD (no masking needed)
                        obs = pm.Multinomial("obs", n=item_ns_data, p=p_item, observed=item_counts_data)

                    # Derived quantities for stopping criteria
                    modal_group = pm.Deterministic("modal_group", pt.argmax(alpha_group))
                    p_group_normalized = alpha_group / pm.math.sum(alpha_group)
                    entropy_group = pm.Deterministic(
                        "entropy_group",
                        -pm.math.sum(p_group_normalized * pm.math.log(p_group_normalized + 1e-10))
                    )

            ordinal_group_cache['model'] = ordinal_model
            ordinal_group_cache['n_categories_last'] = n_categories
            ordinal_group_cache['model_type'] = actual_model_type
            ordinal_group_cache['use_preallocation'] = current_use_preallocation
            # Track pre-allocation status: model_n_items is set only when pre-allocation is used
            # This allows ordinal_utils.py helper functions to detect which data-setting path to use
            if current_use_preallocation:
                ordinal_group_cache['model_n_items'] = model_n_items
            logger.info(f"Created ordinal model for '{grouping_name}' (allocated={model_n_items}, current={current_n_items}, type={actual_model_type}, prealloc={current_use_preallocation})")

    # Process each sample for sample-level stopping
    item_summaries = []
    for item_idx, item_id in enumerate(item_ids):
        df_item = df_work[df_work['sample_id_num'] == item_id].sort_values('epoch_num')
        original_sample_id = df_item[sample_id_column].iloc[0]

        if score_type == 'binary':
            # === BINARY SAMPLE-LEVEL STOPPING ===
            successes = int(df_item[score_column].sum())
            trials = len(df_item)

            current_perf = successes / trials if trials > 0 else 0
            current_conservatism = conservatism if current_perf < low_perf_threshold else 1.0

            # Compute CI
            lo, hi, width = _beta_ci_adaptive(
                successes, trials,
                cred_level=cred_level,
                conservatism=current_conservatism,
                low_perf_threshold=low_perf_threshold
            )

            # Check width criterion
            if width < delta_item:
                stop_sample_ids.append(f"{grouping_name}:::{original_sample_id}")
                metadata['sample_stopping_reasons'][str(original_sample_id)] = {
                    'reason': 'ci_width',
                    'ci_width': float(width),
                    'threshold': delta_item,
                    'epochs_used': trials
                }
                # logger.info(f"Stopping sample {original_sample_id}: CI width {width:.4f} < {delta_item}")  # Results in output

            item_summaries.append({'successes': successes, 'trials': trials})

        elif score_type == 'ordinal':
            # === ORDINAL SAMPLE-LEVEL STOPPING ===
            accumulated_scores = df_item[score_column].tolist()

            current_perf = np.mean(accumulated_scores) / ordinal_max_score
            current_conservatism = conservatism if current_perf < low_perf_threshold else 1.0

            entropy_history = entropy_history_per_item.get(item_id, [])

            if ordinal_inference == 'modal':
                lo, hi, width = _ordinal_ci_adaptive(
                    np.array(accumulated_scores),
                    ordinal_max_score=ordinal_max_score,
                    cred_level=cred_level,
                    conservatism=current_conservatism,
                    low_perf_threshold=low_perf_threshold
                )
                if width < delta_item:
                    stop_sample_ids.append(f"{grouping_name}:::{original_sample_id}")
                    metadata['sample_stopping_reasons'][str(original_sample_id)] = {
                        'reason': 'ordinal_modal_ci_width',
                        'ci_width': float(width),
                        'threshold': delta_item,
                        'epochs_used': len(accumulated_scores)
                    }
                    # logger.info(f"Stopping ordinal sample {original_sample_id}: Modal CI {width:.4f} < {delta_item}")  # Results in output

            elif ordinal_inference == 'entropy':
                lo, hi, width, diagnostics = _ordinal_entropy_ci_adaptive(
                    np.array(accumulated_scores),
                    ordinal_max_score=ordinal_max_score,
                    cred_level=cred_level,
                    conservatism=current_conservatism,
                    low_perf_threshold=low_perf_threshold,
                    model_cache=ordinal_item_cache,
                    compute_kwargs=sampling_kwargs
                )
                if width < delta_item:
                    stop_sample_ids.append(f"{grouping_name}:::{original_sample_id}")
                    metadata['sample_stopping_reasons'][str(original_sample_id)] = {
                        'reason': 'ordinal_entropy_ci_width',
                        'ci_width': float(width),
                        'threshold': delta_item,
                        'epochs_used': len(accumulated_scores),
                        'diagnostics': _sanitize_diagnostics(diagnostics)
                    }
                    # logger.info(f"Stopping ordinal sample {original_sample_id}: Entropy CI {width:.4f} < {delta_item}")  # Results in output

            elif ordinal_inference == 'hybrid':
                should_stop, reason, diagnostics = _ordinal_hybrid_stopping_criterion(
                    np.array(accumulated_scores),
                    ordinal_max_score=ordinal_max_score,
                    delta_item=delta_item,
                    cred_level=cred_level,
                    entropy_history=entropy_history,
                    entropy_threshold=entropy_threshold,
                    conservatism=current_conservatism,
                    low_perf_threshold=low_perf_threshold,
                    stabilization_threshold=entropy_stabilization_threshold,
                    model_cache=ordinal_item_cache,
                    compute_kwargs=sampling_kwargs
                )
                if should_stop:
                    stop_sample_ids.append(f"{grouping_name}:::{original_sample_id}")
                    metadata['sample_stopping_reasons'][str(original_sample_id)] = {
                        'reason': reason,
                        'epochs_used': len(accumulated_scores),
                        'diagnostics': _sanitize_diagnostics(diagnostics)
                    }
                    # logger.info(f"Stopping ordinal sample {original_sample_id} via {reason}")  # Results in output

            entropy_history_per_item[item_id] = entropy_history
            # Store category counts for hierarchical Dirichlet-Multinomial model
            counts = np.bincount(np.array(accumulated_scores).astype(int), minlength=ordinal_max_score + 1)
            item_summaries.append({
                'counts': counts,                              # Vector of length K for hierarchical model
                'n_obs': len(accumulated_scores),              # Total observations for this item
                'modal_category': int(np.argmax(counts)),      # Most frequent category
                'mean_score': float(np.mean(accumulated_scores)),  # Mean score
                # Keep successes/trials for backward compatibility with perf estimate
                'successes': int(np.sum(accumulated_scores)),
                'trials': len(accumulated_scores)
            })

        elif score_type in ['continuous_01', 'continuous_bounded']:
            # === CONTINUOUS SAMPLE-LEVEL STOPPING ===
            # Sample-level uses fast Beta sampling (no hierarchical pooling yet)
            # Hierarchical pooling happens at group level (after all samples processed)
            accumulated_scores = df_item[score_column].values

            # Normalize performance for conservatism check (to [0, 1])
            normalized_perf = np.mean(accumulated_scores) / upper_bound
            current_conservatism = conservatism if normalized_perf < low_perf_threshold else 1.0

            # Compute CI using fast Beta method (no MCMC at sample level)
            lo, hi, width = _continuous_bounded_ci_adaptive(
                accumulated_scores,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                cred_level=cred_level,
                conservatism=current_conservatism,
                low_perf_threshold=low_perf_threshold
            )

            # ⚠️ CRITICAL: Normalize width for comparison with delta_item
            width_normalized = width / (upper_bound - lower_bound)

            # Check width criterion
            if width_normalized < delta_item:
                stop_sample_ids.append(f"{grouping_name}:::{original_sample_id}")
                metadata['sample_stopping_reasons'][str(original_sample_id)] = {
                    'reason': 'continuous_bounded_ci_width',
                    'ci_width': float(width),
                    'ci_width_normalized': float(width_normalized),
                    'threshold': delta_item,
                    'epochs_used': len(accumulated_scores),
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }
                # logger.info(f"Stopping continuous sample {original_sample_id}: CI width_norm {width_normalized:.4f} < {delta_item}")  # Results in output

            # Store normalized scores for hierarchical group-level inference
            # Normalization to [0,1] required for Beta likelihood in PyMC model
            scores_normalized = (accumulated_scores - lower_bound) / (upper_bound - lower_bound)
            scores_normalized = np.clip(scores_normalized, 0.0, 1.0)  # Safety clipping

            item_summaries.append({
                'scores_normalized': scores_normalized,  # For hierarchical model at group level
                'mean': float(np.mean(accumulated_scores)),  # Original scale
                'count': len(accumulated_scores)
            })

    # Group-level stopping check (runs automatically after sample checks)
    if len(item_summaries) > 0:
        # TIMING_TEST: Start group-level inference timing
        _group_inference_start = time.perf_counter()
        # logger.info(f"Running group-level stopping check for '{grouping_name}'")

        # Compute performance estimate based on score type
        if score_type in ['binary', 'ordinal']:
            current_perf_estimate = np.sum([s['successes'] for s in item_summaries]) / np.sum([s['trials'] for s in item_summaries])
        else:  # continuous
            # For continuous, use mean of means weighted by counts
            total_sum = np.sum([s['mean'] * s['count'] for s in item_summaries])
            total_count = np.sum([s['count'] for s in item_summaries])
            current_perf_estimate = total_sum / total_count if total_count > 0 else 0

        if score_type == 'binary':
            # === BINARY GROUP-LEVEL STOPPING ===
            # TIMING_TEST: Binary-specific timing start
            _binary_start = time.perf_counter()

            all_successes = np.array([s['successes'] for s in item_summaries])
            all_trials = np.array([s['trials'] for s in item_summaries])

            current_conservatism = conservatism if current_perf_estimate < low_perf_threshold else 1.0

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with model:
                    # Prepare data arrays based on mode (Fix 1)
                    n_observed = len(all_successes)
                    cached_use_preallocation = binary_group_cache.get('use_preallocation', True)

                    if cached_use_preallocation:
                        # PRE-ALLOCATED MODE: Pad arrays with obs_weight masking
                        model_size = binary_group_cache.get('model_n_items', n_observed)
                        successes_padded = np.zeros(model_size, dtype="int64")
                        trials_padded = np.ones(model_size, dtype="int64")  # 1, not 0 (safe default)
                        obs_weight_padded = np.zeros(model_size, dtype="float64")

                        # Fill observed data in first n_observed positions
                        successes_padded[:n_observed] = all_successes
                        trials_padded[:n_observed] = all_trials
                        obs_weight_padded[:n_observed] = 1.0

                        pm.set_data({
                            "successes": successes_padded,
                            "trials": trials_padded,
                            "obs_weight": obs_weight_padded,
                        })
                    else:
                        # DYNAMIC MODE: Set raw data with n_items
                        pm.set_data({
                            "n_items": np.int64(n_observed),
                            "successes": all_successes.astype("int64"),
                            "trials": all_trials.astype("int64"),
                        })

                    with suppress_all_output():
                        trace = pm.sample(**sampling_kwargs)
                    with suppress_all_output():
                        theta_hdi = az.hdi(trace.posterior["Theta"], hdi_prob=cred_level)

                # Extract CI bounds
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

                # Append to history
                stabilization_history['ci_width_history'].append(float(theta_width))

                # Check width criterion
                effective_width = theta_width * current_conservatism if current_perf_estimate < low_perf_threshold else theta_width
                if effective_width < delta_cap:
                    stop_this_grouping.append(grouping_name)
                    metadata['group_stopping_reason'] = {
                        'reason': 'ci_width',
                        'ci_width': float(theta_width),
                        'effective_width': float(effective_width),
                        'threshold': delta_cap,
                        'samples_used': len(item_summaries)
                    }
                    # logger.info(f"Stopping grouping '{grouping_name}': CI {effective_width:.4f} < {delta_cap}")  # Results in output

                # Check stabilization criterion (if enough history)
                if len(stabilization_history['ci_width_history']) >= stab_window:
                    recent_widths = stabilization_history['ci_width_history'][-stab_window:]
                    slope = np.polyfit(range(len(recent_widths)), recent_widths, 1)[0]
                    stabilization_history['ci_slope_history'].append(float(slope))

                    slope_threshold = CI_delta / current_conservatism if current_perf_estimate < low_perf_threshold else CI_delta

                    if abs(slope) <= slope_threshold and len(stabilization_history['ci_slope_history']) >= 4:
                        recent_slopes = stabilization_history['ci_slope_history'][-3:]
                        slope_slopes = np.polyfit(range(len(recent_slopes)), recent_slopes, 1)[0]

                        if slope_slopes >= 0:
                            if current_perf_estimate >= low_perf_threshold:
                                stop_this_grouping.append(grouping_name)
                                metadata['group_stopping_reason'] = {
                                    'reason': 'ci_stabilization',
                                    'slope': float(slope),
                                    'slope_threshold': slope_threshold,
                                    'samples_used': len(item_summaries)
                                }
                                # logger.info(f"Stopping grouping '{grouping_name}' via stabilization: slope {slope:.6f}")  # Results in output
                            elif abs(slope) <= slope_threshold / 2:
                                stop_this_grouping.append(grouping_name)
                                metadata['group_stopping_reason'] = {
                                    'reason': 'ci_stabilization_low_perf',
                                    'slope': float(slope),
                                    'slope_threshold': slope_threshold,
                                    'samples_used': len(item_summaries)
                                }
                                # logger.info(f"Stopping low-perf grouping '{grouping_name}' via strong stabilization")  # Results in output

            # TIMING_TEST: Binary inference complete
            _binary_elapsed = time.perf_counter() - _binary_start
            # logger.warning(f"🕐 TIMING_TEST: Binary group inference took {_binary_elapsed:.3f}s for {len(item_summaries)} items")

        elif score_type == 'ordinal':
            # === ORDINAL GROUP-LEVEL STOPPING (HIERARCHICAL) ===
            # Uses Dirichlet-Multinomial hierarchy for partial pooling across items
            # TIMING_TEST: Ordinal-specific timing start
            _ordinal_start = time.perf_counter()

            # Extract item counts matrix for hierarchical model
            item_counts_matrix = np.array([s['counts'] for s in item_summaries])
            item_ns = np.array([s['n_obs'] for s in item_summaries])

            current_perf_normalized = current_perf_estimate / ordinal_max_score
            current_conservatism = conservatism if current_perf_normalized < low_perf_threshold else 1.0

            group_entropy_history = stabilization_history.get('entropy_history', [])

            if ordinal_inference == 'modal':
                # Hierarchical modal inference with partial pooling
                lo, hi, width = _ordinal_ci_hierarchical_modal(
                    item_counts_matrix,
                    item_ns,
                    ordinal_max_score=ordinal_max_score,
                    cred_level=cred_level,
                    conservatism=current_conservatism,
                    low_perf_threshold=low_perf_threshold,
                    current_perf=current_perf_normalized,
                    model_cache=ordinal_group_cache,
                    sampling_kwargs=sampling_kwargs
                )
                # Track CI width for stabilization history (enables n_group_checks tracking)
                stabilization_history['ci_width_history'].append(float(width))
                # Store modal-specific diagnostics
                stabilization_history['final_modal_ci_width'] = float(width)
                stabilization_history['final_modal_ci'] = [float(lo), float(hi)]
                stabilization_history['ordinal_pathway'] = 'modal_hierarchical'

                if width < delta_cap:
                    stop_this_grouping.append(grouping_name)
                    metadata['group_stopping_reason'] = {
                        'reason': 'ordinal_modal_ci_width_hierarchical',
                        'ci_width': float(width),
                        'threshold': delta_cap,
                        'samples_used': len(item_summaries),
                        'n_items': len(item_ns)
                    }
                    # logger.info(f"Stopping ordinal grouping '{grouping_name}': Hierarchical Modal CI {width:.4f} < {delta_cap}")

            elif ordinal_inference == 'entropy':
                # Hierarchical entropy inference with partial pooling
                lo, hi, width, diagnostics = _ordinal_ci_hierarchical_entropy(
                    item_counts_matrix,
                    item_ns,
                    ordinal_max_score=ordinal_max_score,
                    cred_level=cred_level,
                    conservatism=current_conservatism,
                    low_perf_threshold=low_perf_threshold,
                    current_perf=current_perf_normalized,
                    model_cache=ordinal_group_cache,
                    sampling_kwargs=sampling_kwargs
                )
                # Track CI width for stabilization history (enables n_group_checks tracking)
                stabilization_history['ci_width_history'].append(float(width))
                # Store entropy-specific diagnostics
                stabilization_history['final_entropy_ci_width'] = float(width)
                stabilization_history['final_entropy'] = float(diagnostics.get('entropy_median', 0)) if diagnostics else None
                stabilization_history['ordinal_pathway'] = 'entropy_hierarchical'

                # Update entropy history for stabilization tracking
                group_entropy_history.append((lo, hi, width))
                stabilization_history['entropy_history'] = group_entropy_history

                if width < delta_cap:
                    stop_this_grouping.append(grouping_name)
                    metadata['group_stopping_reason'] = {
                        'reason': 'ordinal_entropy_ci_width_hierarchical',
                        'ci_width': float(width),
                        'threshold': delta_cap,
                        'samples_used': len(item_summaries),
                        'n_items': len(item_ns),
                        'diagnostics': diagnostics
                    }
                    # logger.info(f"Stopping ordinal grouping '{grouping_name}': Hierarchical Entropy CI {width:.4f} < {delta_cap}")

            elif ordinal_inference == 'hybrid':
                # Hierarchical hybrid: uses Dirichlet-Multinomial model
                should_stop_group, reason_group, diagnostics_group = _ordinal_hybrid_stopping_criterion_hierarchical(
                    item_counts_matrix,
                    item_ns,
                    ordinal_max_score=ordinal_max_score,
                    delta_item=delta_cap,
                    cred_level=cred_level,
                    entropy_history=group_entropy_history,
                    entropy_threshold=entropy_threshold,
                    conservatism=current_conservatism,
                    low_perf_threshold=low_perf_threshold,
                    current_perf=current_perf_normalized,
                    stabilization_threshold=entropy_stabilization_threshold,
                    model_cache=ordinal_group_cache,
                    sampling_kwargs=sampling_kwargs
                )
                # Track CI width for stabilization history (enables n_group_checks tracking)
                # Use modal_width or entropy_width from diagnostics, depending on pathway
                # Use explicit None checks to avoid truthiness issues with 0.0 values
                if diagnostics_group:
                    width = diagnostics_group.get('modal_width')
                    if width is None:
                        width = diagnostics_group.get('entropy_width')
                    if width is None:
                        width = 0.0
                    stabilization_history['ci_width_history'].append(float(width))

                    # Populate ordinal fields unconditionally (not just when stopping)
                    # This ensures diagnostics are available for non-stopping cases too
                    stabilization_history['ordinal_pathway'] = 'hybrid_hierarchical'

                    # Always populate modal fields if available
                    if 'modal_width' in diagnostics_group:
                        stabilization_history['final_modal_ci_width'] = float(diagnostics_group['modal_width'])
                        if 'modal_ci' in diagnostics_group:
                            stabilization_history['final_modal_ci'] = [float(x) for x in diagnostics_group['modal_ci']]

                    # Always populate entropy fields if available
                    if 'entropy_median' in diagnostics_group:
                        stabilization_history['final_entropy'] = float(diagnostics_group['entropy_median'])
                    if 'entropy_threshold' in diagnostics_group:
                        stabilization_history['final_entropy_threshold'] = float(diagnostics_group['entropy_threshold'])
                    if 'entropy_width' in diagnostics_group:
                        stabilization_history['final_entropy_ci_width'] = float(diagnostics_group['entropy_width'])
                    if 'relative_change' in diagnostics_group:
                        stabilization_history['final_relative_change'] = float(diagnostics_group['relative_change'])
                    if 'stabilization_threshold' in diagnostics_group:
                        stabilization_history['final_stabilization_threshold'] = float(diagnostics_group['stabilization_threshold'])

                if should_stop_group:
                    stop_this_grouping.append(grouping_name)
                    metadata['group_stopping_reason'] = {
                        'reason': reason_group,
                        'samples_used': len(item_summaries),
                        'diagnostics': diagnostics_group
                    }
                    # Update ordinal_pathway to specific pathway number when stopping
                    if diagnostics_group:
                        stabilization_history['ordinal_pathway'] = diagnostics_group.get('pathway', 'hybrid')
                    # logger.info(f"Stopping ordinal grouping '{grouping_name}' via {reason_group}")  # Results in output

            stabilization_history['entropy_history'] = group_entropy_history

            # TIMING_TEST: Ordinal inference complete
            _ordinal_elapsed = time.perf_counter() - _ordinal_start
            # logger.warning(f"🕐 TIMING_TEST: Ordinal group inference took {_ordinal_elapsed:.3f}s for {len(item_summaries)} items, mode={ordinal_inference}")

        elif score_type in ['continuous_01', 'continuous_bounded']:
            # === CONTINUOUS GROUP-LEVEL STOPPING (HIERARCHICAL) ===
            # Uses PyMC hierarchical Beta model (mirrors binary hierarchical structure)
            #
            # Why Hierarchical?
            #   - Accounts for item-to-item variability (some items harder than others)
            #   - Pools information across items (partial pooling / shrinkage)
            #   - Items with few observations benefit from group-level information
            #   - Consistent with binary methodology
            #
            # Performance Note:
            #   - Hierarchical inference uses MCMC sampling (~5-30 seconds)
            #   - Much slower than sample-level Beta sampling (~1-2ms)
            #   - Necessary trade-off for proper uncertainty quantification

            # TIMING_TEST: Continuous-specific timing start
            _continuous_start = time.perf_counter()

            # Prepare aggregated data for hierarchical PyMC model
            # SOLUTION A: Aggregate to item-level statistics instead of individual observations
            # This reduces likelihood evaluations from ~950 to ~50 per MCMC step (19x speedup!)
            # TIMING_TEST: Start data aggregation timing
            _aggregation_start = time.perf_counter()

            item_means_list = []
            item_ns_list = []
            total_obs_count = 0

            for item_summary in item_summaries:
                # Scores already normalized to [0,1] in sample-level processing
                scores = item_summary['scores_normalized']

                # Aggregate to mean and sample size
                item_means_list.append(np.mean(scores))
                item_ns_list.append(len(scores))
                total_obs_count += len(scores)

            item_means_array = np.array(item_means_list)
            item_ns_array = np.array(item_ns_list)
            n_items_actual = len(item_summaries)

            # TIMING_TEST: Data aggregation complete
            _aggregation_elapsed = time.perf_counter() - _aggregation_start
            # logger.warning(f"🕐 TIMING_TEST: Continuous data aggregation took {_aggregation_elapsed:.3f}s (aggregated {total_obs_count} obs → {n_items_actual} means)")

            # Compute performance estimate for conservatism check
            # Use aggregated item means (already in [0,1] scale)
            current_perf_estimate = np.mean(item_means_array)
            normalized_perf = current_perf_estimate
            current_conservatism = conservatism if normalized_perf < low_perf_threshold else 1.0

            # Run hierarchical Bayesian inference using PyMC
            # TIMING_TEST: Start PyMC MCMC sampling
            _mcmc_start = time.perf_counter()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with continuous_model:
                    # Prepare data arrays based on mode (Fix 1)
                    n_observed = n_items_actual
                    cached_use_preallocation = continuous_group_cache.get('use_preallocation', True)

                    if cached_use_preallocation:
                        # PRE-ALLOCATED MODE: Pad arrays with obs_weight masking
                        model_size = continuous_group_cache.get('model_n_items', n_observed)
                        # CRITICAL: item_ns must be >= 1 to avoid division by zero
                        item_means_padded = np.full(model_size, 0.5, dtype="float64")
                        item_ns_padded = np.ones(model_size, dtype="int64")  # 1, not 0
                        obs_weight_padded = np.zeros(model_size, dtype="float64")

                        # Fill observed data in first n_observed positions
                        item_means_padded[:n_observed] = item_means_array
                        item_ns_padded[:n_observed] = item_ns_array
                        obs_weight_padded[:n_observed] = 1.0

                        pm.set_data({
                            "item_means": item_means_padded,
                            "item_ns": item_ns_padded,
                            "obs_weight": obs_weight_padded,
                        })
                    else:
                        # DYNAMIC MODE: Set raw data with n_items
                        pm.set_data({
                            "n_items": np.int64(n_observed),
                            "item_means": item_means_array.astype("float64"),
                            "item_ns": item_ns_array.astype("int64"),
                        })

                    # Sample posterior distribution
                    with suppress_all_output():
                        trace = pm.sample(**sampling_kwargs)

                    # Extract item-level means HDI (High Density Interval)
                    with suppress_all_output():
                        mu_item_hdi = az.hdi(trace.posterior["mu_item"], hdi_prob=cred_level)

                # Extract CI bounds from HDI
                # mu_item is shape (model_n_items,) - one mean per item
                # Only use OBSERVED items (first n_observed) for group-level CI
                try:
                    mu_values_lower = mu_item_hdi["mu_item"].sel(hdi="lower").values[:n_observed]
                    mu_lo_normalized = float(np.mean(mu_values_lower))
                except Exception:
                    # Fallback for different arviz versions
                    mu_lo_normalized = float(np.mean(mu_item_hdi["mu_item"].values[..., 0][:n_observed]))

                try:
                    mu_values_upper = mu_item_hdi["mu_item"].sel(hdi="upper").values[:n_observed]
                    mu_hi_normalized = float(np.mean(mu_values_upper))
                except Exception:
                    # Fallback for different arviz versions
                    mu_hi_normalized = float(np.mean(mu_item_hdi["mu_item"].values[..., 1][:n_observed]))

                # Compute width in normalized [0,1] space
                width_normalized = mu_hi_normalized - mu_lo_normalized

                # Scale back to original bounds for metadata
                width_original = width_normalized * (upper_bound - lower_bound)

            # TIMING_TEST: PyMC MCMC sampling complete
            _mcmc_elapsed = time.perf_counter() - _mcmc_start
            # logger.warning(f"🕐 TIMING_TEST: Continuous PyMC MCMC sampling took {_mcmc_elapsed:.3f}s for {n_items_actual} items (aggregated {total_obs_count} obs)")

            # Append normalized width to history for stabilization tracking
            stabilization_history['ci_width_history'].append(float(width_normalized))

            # Check width criterion with conservatism adjustment
            effective_width = width_normalized * current_conservatism if normalized_perf < low_perf_threshold else width_normalized

            if effective_width < delta_cap:
                stop_this_grouping.append(grouping_name)
                metadata['group_stopping_reason'] = {
                    'reason': 'continuous_hierarchical_ci_width',
                    'ci_width': float(width_original),
                    'ci_width_normalized': float(width_normalized),
                    'effective_width': float(effective_width),
                    'threshold': delta_cap,
                    'samples_used': len(item_summaries),
                    'n_observations': total_obs_count,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }
                # logger.info(f"Stopping grouping '{grouping_name}': Hierarchical CI effective_width {effective_width:.4f} < {delta_cap}")  # Results in output

            # Check stabilization criterion (if enough history)
            if len(stabilization_history['ci_width_history']) >= stab_window:
                recent_widths = stabilization_history['ci_width_history'][-stab_window:]
                slope = np.polyfit(range(len(recent_widths)), recent_widths, 1)[0]
                stabilization_history['ci_slope_history'].append(float(slope))

                # Adjust slope threshold based on performance
                slope_threshold = CI_delta / current_conservatism if normalized_perf < low_perf_threshold else CI_delta

                # Check if slope is near zero (stabilized)
                if abs(slope) <= slope_threshold and len(stabilization_history['ci_slope_history']) >= 4:
                    # Check second derivative (slope of slopes) to ensure stabilization is real
                    recent_slopes = stabilization_history['ci_slope_history'][-3:]
                    slope_slopes = np.polyfit(range(len(recent_slopes)), recent_slopes, 1)[0]

                    # Slope of slopes should be >= 0 (not getting steeper in negative direction)
                    if slope_slopes >= 0:
                        # For good performance, allow stabilization stopping
                        if normalized_perf >= low_perf_threshold:
                            stop_this_grouping.append(grouping_name)
                            metadata['group_stopping_reason'] = {
                                'reason': 'continuous_hierarchical_stabilization',
                                'slope': float(slope),
                                'slope_threshold': slope_threshold,
                                'samples_used': len(item_summaries)
                            }
                            # logger.info(f"Stopping grouping '{grouping_name}' via hierarchical continuous stabilization: slope {slope:.6f}")  # Results in output
                        # For low performance, require stronger stabilization
                        elif abs(slope) <= slope_threshold / 2:
                            stop_this_grouping.append(grouping_name)
                            metadata['group_stopping_reason'] = {
                                'reason': 'continuous_hierarchical_stabilization_low_perf',
                                'slope': float(slope),
                                'slope_threshold': slope_threshold,
                                'samples_used': len(item_summaries)
                            }
                            # logger.info(f"Stopping low-perf grouping '{grouping_name}' via strong hierarchical continuous stabilization")  # Results in output

            # TIMING_TEST: Continuous inference complete
            _continuous_elapsed = time.perf_counter() - _continuous_start
            # logger.warning(f"🕐 TIMING_TEST: Continuous group inference TOTAL took {_continuous_elapsed:.3f}s for {len(item_summaries)} items")

    # Update samples evaluated count
    stabilization_history['n_samples_evaluated'] = len(item_summaries)

    # TIMING_TEST: Overall function complete
    _function_elapsed = time.perf_counter() - _function_start_time
    # logger.warning(f"🕐 TIMING_TEST: optimal_stopping_live_single TOTAL took {_function_elapsed:.3f}s for '{grouping_name}' ({score_type})")

    # Rebuild model_caches dict for return (OPTIMIZATION #2)
    model_caches_out = {
        'binary_item': binary_item_cache,
        'binary_group': binary_group_cache,
        'ordinal_item': ordinal_item_cache,
        'ordinal_group': ordinal_group_cache,
        'continuous_item': continuous_item_cache,
        'continuous_group': continuous_group_cache
    }

    return {
        'grouping': grouping_name,
        'stop_sample_ids': stop_sample_ids,
        'stop_this_grouping': stop_this_grouping,
        'stabilization_history': stabilization_history,
        'model_caches': model_caches_out,  # OPTIMIZATION #2: Return all caches for persistence
        'item_entropy_histories': entropy_history_per_item,  # Issue #6 fix: Persist for Pathway 2
        'metadata': metadata
    }


def optimal_stopping_live(df: pd.DataFrame, params: Dict[str, Any], grouping_columns: List[str], sample_id_column: str, epoch_column: str, score_column: str = "score", display_progress: bool = True, ordinal_tasks: Optional[List[str]] = None, ordinal_max_score: int = 10, ordinal_inference: str = 'modal', ordinal_model_type: str = 'ordered_logistic', continuous_tasks: Optional[List[str]] = None, entropy_threshold: float = 1.5, gpu_ids: Optional[List[int]] = None, max_workers: Optional[int] = None) -> Dict[str, Any]:
    """
    Run optimal stopping in live mode on current data for multiple groupings, parallelizing across groupings.

    Supports both binary (0/1) and ordinal (e.g., 0-10 Likert scale) scoring within the same dataset.

    Args:
        df: DataFrame with grouping, sample_id, epoch, and score columns
        params: Dictionary of stopping parameters (delta_item, delta_cap, cred_level, etc.)
        grouping_columns: List of column names to combine for grouping (can be a single string or list)
        sample_id_column: Column name for sample ID
        epoch_column: Column name for epoch/trial
        score_column: Column name for score (default: 'score')
        display_progress: Whether to show a progress bar (default: True)

        ordinal_tasks: List of substrings to identify ordinal groupings. If a grouping name
            contains any of these substrings (case-insensitive), it will be treated as ordinal.
            If None, all groupings are treated as binary. Example: ['confidence', 'rating', 'likert']
        ordinal_max_score: Maximum score for ordinal data (e.g., 10 for 0-10 scale). Default: 10
        ordinal_inference: Inference method for ordinal data. Options:
            - 'modal': Fast bootstrap-based modal category estimation
            - 'entropy': Slow but conservative OrderedLogistic entropy estimation
            - 'hybrid': RECOMMENDED - Combines modal CI + entropy validation for efficiency and safety
        ordinal_model_type: Hierarchical model type for ordinal data (default: 'ordered_logistic')
            - 'ordered_logistic': Cumulative link model with identified cutpoints (recommended)
            - 'dirichlet': Dirichlet-Multinomial (treats categories as exchangeable)
            Falls back to 'dirichlet' if ordered_logistic sampling fails.
        continuous_tasks: List of substrings to identify continuous bounded groupings. If a grouping name
            contains any of these substrings (case-insensitive), it will be treated as continuous bounded.
            If None, continuous inference is not used. Example: ['mean_score', 'aggregated']
            Scores must be pre-aggregated floats in [0, ordinal_max_score] range.
            Priority: continuous_tasks > ordinal_tasks > binary (default)
        entropy_threshold: Threshold for entropy validation in hybrid mode (default: 1.5).
            Used to detect false peaks: if entropy > threshold, modal CI narrow is rejected as false peak.
            Typical values: 1.0 for 5-point scale, 1.5 for 11-point scale, 2.0 for 21-point scale.

        gpu_ids: List of GPU IDs to use for parallel processing. If None, uses CPU-only
        max_workers: Number of parallel workers. If None, uses len(gpu_ids) when GPUs specified, otherwise uses CPU count

    Returns:
        Dict with:
            - 'stop_sample_ids': List of "grouping_sample_id" strings for items that reached stopping criteria
            - 'stop_task': List of grouping names that reached stopping criteria

    Example:
        >>> # Mixed binary/ordinal dataset
        >>> result = optimal_stopping_live(
        ...     df=df,
        ...     params=params,
        ...     grouping_columns=['student_id', 'task_name'],
        ...     sample_id_column='item_id',
        ...     epoch_column='trial_num',
        ...     score_column='score',
        ...     ordinal_tasks=['confidence', 'difficulty'],  # Identify ordinal tasks
        ...     ordinal_max_score=10,
        ...     ordinal_inference='hybrid'  # RECOMMENDED
        ... )
        >>> print(result['stop_sample_ids'])  # Items to stop
        >>> print(result['stop_task'])  # Groupings to stop
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

        # Add score type parameters to params dict for worker propagation
        params_with_context = params.copy()
        params_with_context['ordinal_tasks'] = ordinal_tasks
        params_with_context['continuous_tasks'] = continuous_tasks
        params_with_context['ordinal_max_score'] = ordinal_max_score
        params_with_context['ordinal_inference'] = ordinal_inference
        params_with_context['ordinal_model_type'] = ordinal_model_type
        params_with_context['entropy_threshold'] = entropy_threshold

        # Validate ordinal scores upfront and log score type routing
        for grouping in df['grouping'].unique():
            score_type, bounds = determine_score_type_standalone(
                grouping,
                ordinal_tasks=ordinal_tasks,
                continuous_tasks=continuous_tasks,
                upper_bound=ordinal_max_score
            )
            if score_type == 'ordinal':
                df_grouping = df[df['grouping'] == grouping]
                validate_ordinal_scores(
                    df_grouping[score_column].values,
                    ordinal_max_score,
                    grouping_name=grouping
                )
                logger.info(f"Grouping '{grouping}' identified as ORDINAL (using {ordinal_inference} inference)")
            elif score_type in ('continuous_bounded', 'continuous_01'):
                logger.info(f"Grouping '{grouping}' identified as CONTINUOUS BOUNDED (Beta distribution)")
            else:
                logger.info(f"Grouping '{grouping}' identified as BINARY")

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
                args_list.append((grouping, df_grouping, params_with_context, sample_id_column, epoch_column, score_column, grouping_columns))

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
            # if os.getpid() == getattr(os, 'getppid', lambda: None)() or hasattr(sys, 'ps1'):
            #     print(f"Run complete. See the log file for details: {_get_logfile_path()}")
            return {'stop_sample_ids': stop_sample_ids, 'stop_task': stop_task_groupings}
    finally:
        sys.stdout = old_stdout

        sys.stderr = old_stderr 
