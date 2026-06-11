"""
Ordinal scoring utilities for optimal stopping algorithms.

This module provides functions for computing credible intervals and handling
ordinal score data (e.g., 0-10 Likert-type scales) using Bayesian bootstrap methods.

INFERENCE TYPE: Modal Category with Credible Interval
======================================================
The current implementation computes confidence intervals on the MODAL (most common)
category, answering: "What category is typical performance in, and how certain are we?"

This differs from mean-based inference by focusing on the most likely categorical
outcome rather than the average across categories.

VALIDITY & LIMITATIONS:
- ✓ Appropriate for: "Which category represents typical performance?"
- ✓ Preserves ordinal nature (doesn't assume interval scaling)
- ✗ Does NOT provide: Full probability distribution over categories
- ✗ Does NOT provide: P(Score ≥ threshold) statements

FULL CATEGORICAL INFERENCE:
For full categorical inference (P(Score = k) for all k), use the hierarchical
OrderedLogistic model implemented in ordinal_model.py. This provides entropy-based
stopping criteria and is used by the 'entropy' and 'hybrid' inference modes.

References:
    Rubin, D. B. (1981). The Bayesian Bootstrap. The Annals of Statistics, 9(1), 130-134.
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict

# Import trace_message for inspect_ai integration (with fallback)
try:
    from inspect_ai.util import trace_message
except (ImportError, AttributeError):
    def trace_message(logger, component, message):
        """Fallback when inspect_ai is not available."""
        logger.info(f"[{component}] {message}")


def _ordinal_ci_adaptive(
    scores: np.ndarray,
    ordinal_max_score: int,
    cred_level: float = 0.97,
    conservatism: float = 5.0,
    low_perf_threshold: float = 0.01,
    base_strength: int = 2,
    n_bootstrap: int = 10000
) -> Tuple[float, float, float]:
    """
    Compute adaptive Bayesian credible interval for ordinal scores using modal category inference.

    INFERENCE: This function estimates the MODAL CATEGORY (most common response) with
    a credible interval, answering "What category is typical performance in?"

    Method: Bayesian bootstrap (Rubin, 1981) to estimate the distribution of the
    modal category across resampled datasets.

    The method applies conservatism to low-performance scenarios to prevent premature
    stopping when performance is poor (analogous to the binary case).

    A sample-size-scaled floor is applied to prevent premature stopping when bootstrap
    CI is zero due to homogeneous data. The floor decreases with more samples, reflecting
    that more agreeing observations genuinely increase confidence.

    Args:
        scores: Array of ordinal scores (0 to ordinal_max_score)
        ordinal_max_score: Maximum possible score for scaling (e.g., 10 for 0-10 scale)
        cred_level: Credibility level (e.g., 0.97 for 97% CI)
        conservatism: Multiplier for CI width in low-performance scenarios (>= 1.0)
        low_perf_threshold: Performance threshold below which conservatism is applied (0-1 scale)
        base_strength: Base prior strength (for future Bayesian enhancements, currently unused)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (lower_bound, upper_bound, effective_width), all in [0,1] scale
        - lower_bound: Lower bound of modal category CI (scaled)
        - upper_bound: Upper bound of modal category CI (scaled)
        - effective_width: CI width (scaled), adjusted for conservatism and sample-size floor

    Example:
        scores = [5, 6, 7, 7, 8, 7, 6]  # Modal category is 7
        lo, hi, width = _ordinal_ci_adaptive(scores, ordinal_max_score=10)
        # Returns: (0.60, 0.75, 0.15) - "95% confident modal category is 6-7.5"

    References:
        Rubin, D. B. (1981). The Bayesian Bootstrap. The Annals of Statistics, 9(1), 130-134.
    """
    logger = logging.getLogger('optstop.ordinal_utils')

    # Handle edge cases
    if len(scores) == 0:
        trace_message(logger, "OrdinalUtils", "Empty scores array provided to _ordinal_ci_adaptive")
        return 0.0, 1.0, 1.0

    # Round to nearest integer for categorical inference (astype(int) truncates, introducing downward bias)
    if not np.allclose(scores, np.round(scores)):
        if not getattr(_ordinal_ci_adaptive, '_warned_non_integer', False):
            logger.info(
                "Non-integer scores detected in ordinal pathway (e.g., %.2f); "
                "rounding to nearest integer for categorical inference",
                scores[0]
            )
            _ordinal_ci_adaptive._warned_non_integer = True
    scores_int = np.clip(np.round(scores).astype(int), 0, ordinal_max_score)

    # Single observation - return wide interval
    if len(scores) == 1:
        # For single observation, modal category is that observation
        # Return wide interval around it
        modal_cat = scores_int[0]
        width_cat = min(3, ordinal_max_score / 2)  # Conservative: ±1.5 categories
        lo_cat = max(0, modal_cat - width_cat/2)
        hi_cat = min(ordinal_max_score, modal_cat + width_cat/2)

        # Scale to [0,1]
        lo = lo_cat / ordinal_max_score
        hi = hi_cat / ordinal_max_score
        width = hi - lo
        return lo, hi, width

    # Bayesian bootstrap for modal category CI estimation
    # Uses Dirichlet(1,1,...,1) weights (Rubin's Bayesian bootstrap)
    n = len(scores)
    bootstrap_modes = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        # Sample Dirichlet weights
        weights = np.random.dirichlet(np.ones(n))

        # Create weighted histogram
        # Use bincount with weights to get category counts
        weighted_hist = np.bincount(
            scores_int,
            weights=weights,
            minlength=ordinal_max_score + 1
        )

        # Modal category is the one with highest weighted count
        modal_cat = np.argmax(weighted_hist)
        bootstrap_modes[i] = modal_cat

    # Compute credible interval on modal category
    alpha = (1 - cred_level) / 2
    lo_cat = np.percentile(bootstrap_modes, alpha * 100)
    hi_cat = np.percentile(bootstrap_modes, (1 - alpha) * 100)

    # Ensure bounds are valid
    lo_cat = max(0, min(lo_cat, ordinal_max_score))
    hi_cat = max(0, min(hi_cat, ordinal_max_score))

    # Ensure lo_cat <= hi_cat
    if lo_cat > hi_cat:
        lo_cat, hi_cat = hi_cat, lo_cat

    # Scale to [0,1] for comparison with delta thresholds
    lo = lo_cat / ordinal_max_score
    hi = hi_cat / ordinal_max_score
    raw_width = hi - lo

    # Apply conservatism for low-performance scenarios
    # Use mean of scaled scores to determine if performance is low
    mean_scaled = np.mean(scores_int / ordinal_max_score)

    # Minimum CI width floor: prevents CI=0 from causing immediate stopping
    # When all observations agree (bootstrap CI width = 0), we still need uncertainty
    # because we have finite samples.
    #
    # Sample-size-scaled floor: floor = 1 / (ordinal_max_score * sqrt(n))
    #
    # Rationale:
    #   - More samples agreeing = more confidence = lower floor
    #   - Reflects statistical reality: CI width scales as 1/sqrt(n)
    #   - Examples for 10-point scale:
    #       n=10:  floor = 1/(10 * 3.16) = 0.032
    #       n=100: floor = 1/(10 * 10)   = 0.010
    #       n=500: floor = 1/(10 * 22.4) = 0.0045
    #
    # This prevents premature stopping with few samples while allowing
    # stopping when sufficient consistent data has been collected.
    n_samples = len(scores)
    min_ci_width = 1.0 / (ordinal_max_score * np.sqrt(n_samples))

    # Track whether we need to widen the CI bounds
    bounds_adjusted = False

    if mean_scaled < low_perf_threshold:
        # Scale up the effective width for more stringent stopping criteria
        # This prevents premature stopping when performance is poor
        # Apply floor BEFORE conservatism multiplication
        floored_width = max(raw_width, min_ci_width)
        effective_width = floored_width * conservatism

        # Widen CI bounds symmetrically if floor was applied
        if raw_width < min_ci_width:
            bounds_adjusted = True
            expansion = (min_ci_width - raw_width) / 2
            lo = max(0.0, lo - expansion)
            hi = min(1.0, hi + expansion)

        logger.info(
            f"Applied conservatism to ordinal CI (effective width for stopping): "
            f"n_samples={n_samples}, mean_scaled={mean_scaled:.3f}, "
            f"raw_ci=[{lo_cat/ordinal_max_score:.3f}, {hi_cat/ordinal_max_score:.3f}], "
            f"{'adjusted_ci=[' + f'{lo:.3f}, {hi:.3f}], ' if bounds_adjusted else ''}"
            f"raw_width={raw_width:.4f}, floor={min_ci_width:.4f}, effective_width={effective_width:.4f}"
        )
    else:
        # Even without conservatism, apply minimum floor for homogeneous data
        effective_width = max(raw_width, min_ci_width)

        # Widen CI bounds symmetrically if floor was applied
        if raw_width < min_ci_width:
            bounds_adjusted = True
            expansion = (min_ci_width - raw_width) / 2
            lo = max(0.0, lo - expansion)
            hi = min(1.0, hi + expansion)
            logger.info(
                f"Applied sample-size-scaled CI floor to ordinal CI (effective width for stopping): "
                f"n_samples={n_samples}, raw_ci=[{lo_cat/ordinal_max_score:.3f}, {hi_cat/ordinal_max_score:.3f}], "
                f"adjusted_ci=[{lo:.3f}, {hi:.3f}], "
                f"raw_width={raw_width:.4f}, floor={min_ci_width:.4f}, effective_width={effective_width:.4f}"
            )

    return lo, hi, effective_width


# Module-level guard: track which groupings have already received a sparsity warning
_warned_sparsity_groupings: set = set()


def check_ordinal_sparsity(
    scores: np.ndarray,
    ordinal_max_score: int,
    grouping_name: str = "unknown",
    min_observations: int = 20
) -> None:
    """
    Check for category sparsity in ordinal scores and warn if detected.

    Ordinal models (ordered logistic, Dirichlet-Multinomial) can produce biased
    estimates and miscalibrated credible intervals when score categories are
    sparse or empty. This function detects such conditions and logs a warning
    suggesting possible remediation (score aggregation or scale adjustment).

    Args:
        scores: Array of ordinal scores (0 to ordinal_max_score)
        ordinal_max_score: Maximum possible score
        grouping_name: Name of grouping (for warning message)
        min_observations: Minimum number of observations before checking
            (avoids false alarms on early/incomplete data)
    """
    logger = logging.getLogger('optstop.ordinal_utils')

    # Skip if already warned for this grouping
    if grouping_name in _warned_sparsity_groupings:
        return

    # Skip if too few observations to assess sparsity reliably
    valid_scores = scores[~np.isnan(scores)]
    if len(valid_scores) < min_observations:
        return

    # Compute category counts
    scores_int = np.clip(np.round(valid_scores).astype(int), 0, ordinal_max_score)
    counts = np.bincount(scores_int, minlength=ordinal_max_score + 1)
    n_categories = ordinal_max_score + 1
    n_total = len(valid_scores)

    # Check for empty categories
    empty_categories = int(np.sum(counts == 0))

    # Check for sparse categories (<5% of observations)
    sparse_threshold = 0.05 * n_total
    sparse_categories = int(np.sum(counts < sparse_threshold))

    # Warn when sparsity is substantial: multiple empty categories, or more than
    # half of all categories are sparse. A single empty tail category on a wide
    # scale (e.g., score 10 on a 0-10 rubric) is not unusual and does not trigger.
    if empty_categories >= 2 or sparse_categories > n_categories / 2:
        _warned_sparsity_groupings.add(grouping_name)

        if empty_categories >= 2:
            detail = f"{empty_categories} of {n_categories} categories have zero observations"
            if sparse_categories > empty_categories:
                detail += f", {sparse_categories} total have <5% of observations"
        else:
            detail = f"{sparse_categories} of {n_categories} categories have <5% of observations"

        logger.warning(
            f"Ordinal inference for grouping '{grouping_name}': category sparsity detected - "
            f"{detail}. "
            f"Ordinal models may produce biased estimates under these conditions. "
            f"Consider whether scores can be aggregated (score_agg='mean') to route "
            f"through the continuous pathway, or whether ordinal_max_score should be reduced."
        )


def validate_ordinal_scores(
    scores: np.ndarray,
    ordinal_max_score: int,
    grouping_name: str = "unknown"
) -> None:
    """
    Validate that ordinal scores are in the expected range.

    Args:
        scores: Array of scores to validate
        ordinal_max_score: Expected maximum score
        grouping_name: Name of grouping (for error messages)

    Raises:
        ValueError: If scores are outside valid range or contain invalid values
    """
    logger = logging.getLogger('optstop.ordinal_utils')

    # Remove NaN values for checking
    valid_scores = scores[~np.isnan(scores)]

    if len(valid_scores) == 0:
        trace_message(logger, "OrdinalUtils", f"Ordinal grouping '{grouping_name}' has no valid scores")
        return

    min_score = valid_scores.min()
    max_score = valid_scores.max()

    # Check if scores are in valid range [0, ordinal_max_score]
    if min_score < 0:
        raise ValueError(
            f"Ordinal grouping '{grouping_name}' has negative scores. "
            f"Minimum score found: {min_score}. "
            f"Ordinal scores must be in range [0, {ordinal_max_score}]."
        )

    if max_score > ordinal_max_score:
        raise ValueError(
            f"Ordinal grouping '{grouping_name}' has scores exceeding ordinal_max_score={ordinal_max_score}. "
            f"Maximum score found: {max_score}. "
            f"Either adjust your data or increase ordinal_max_score parameter."
        )

    # Warn if scores are not integers (allowed but unusual for ordinal data)
    if not np.allclose(valid_scores, np.round(valid_scores)):
        logger.info(
            f"Ordinal grouping '{grouping_name}' contains non-integer scores. "
            f"These will be rounded to nearest integer for categorical inference."
        )



def determine_score_type(
    grouping_name: str,
    ordinal_tasks: Optional[list] = None,
    is_aggregated: bool = False,
    upper_bound: float = 1.0
) -> Tuple[str, Dict[str, float]]:
    """
    Determine score type and bounds based on grouping name, ordinal_tasks, and aggregation context.

    This function determines whether to use binary, ordinal, or continuous bounded inference
    based on:
    1. Whether the grouping name matches ordinal task patterns
    2. Whether scores are aggregated (mean/median)

    Score Type Logic:
    - If is_aggregated=True:
        - Ordinal task → 'continuous_bounded' with bounds [0, upper_bound]
        - Binary task → 'continuous_01' with bounds [0, 1]
    - If is_aggregated=False:
        - Ordinal task → 'ordinal' (discrete categories)
        - Binary task → 'binary' (discrete 0/1)

    Args:
        grouping_name: String identifier for the grouping (e.g., "1-1", "subject1-likert_task")
        ordinal_tasks: List of substrings to match for ordinal scoring. If None, defaults to binary.
        is_aggregated: Whether scores are aggregated (mean/median), producing continuous floats.
        upper_bound: Upper bound for score range. Use 1.0 for binary, ordinal_max_score for ordinal.

    Returns:
        Tuple of (score_type, bounds_dict):
        - score_type: One of 'binary', 'ordinal', 'continuous_01', 'continuous_bounded'
        - bounds_dict: Dictionary with keys 'lower' and 'upper'

    Examples:
        >>> # Discrete binary
        >>> determine_score_type("subject1-accuracy", None, is_aggregated=False)
        ('binary', {'lower': 0.0, 'upper': 1.0})

        >>> # Aggregated binary (mean of multiple 0/1 scores)
        >>> determine_score_type("subject1-accuracy", None, is_aggregated=True)
        ('continuous_01', {'lower': 0.0, 'upper': 1.0})

        >>> # Discrete ordinal
        >>> determine_score_type("subject1-likert", ['likert'], is_aggregated=False, upper_bound=10.0)
        ('ordinal', {'lower': 0.0, 'upper': 10.0})

        >>> # Aggregated ordinal (mean of multiple ordinal scores)
        >>> determine_score_type("subject1-likert", ['likert'], is_aggregated=True, upper_bound=10.0)
        ('continuous_bounded', {'lower': 0.0, 'upper': 10.0})
    """
    logger = logging.getLogger('optstop.ordinal_utils')

    # Determine base type (binary or ordinal) via substring matching
    is_ordinal = False
    if ordinal_tasks is not None and len(ordinal_tasks) > 0:
        # Handle None or non-string grouping_name gracefully
        if grouping_name is None:
            grouping_name_str = ""
        else:
            grouping_name_str = str(grouping_name)

        grouping_name_lower = grouping_name_str.lower()
        for ordinal_substring in ordinal_tasks:
            if ordinal_substring.lower() in grouping_name_lower:
                is_ordinal = True
                break

    # If aggregated, return continuous type
    if is_aggregated:
        if is_ordinal:
            score_type = 'continuous_bounded'
            bounds = {'lower': 0.0, 'upper': upper_bound}
            # logger.info(
            #     f"Grouping '{grouping_name}' with aggregation → CONTINUOUS_BOUNDED [0, {upper_bound}]"
            # )
        else:
            score_type = 'continuous_01'
            bounds = {'lower': 0.0, 'upper': 1.0}
            # logger.info(
            #     f"Grouping '{grouping_name}' with aggregation → CONTINUOUS_01 [0, 1]"
            # )
        return score_type, bounds

    # Non-aggregated: return discrete types
    if is_ordinal:
        # logger.info(f"Grouping '{grouping_name}' → ORDINAL (discrete)")
        return 'ordinal', {'lower': 0.0, 'upper': upper_bound}
    else:
        # logger.debug(f"Grouping '{grouping_name}' → BINARY (discrete)")
        return 'binary', {'lower': 0.0, 'upper': 1.0}


def determine_score_type_standalone(
    grouping_name: str,
    ordinal_tasks: Optional[list] = None,
    continuous_tasks: Optional[list] = None,
    upper_bound: float = 1.0
) -> Tuple[str, Dict[str, float]]:
    """
    Determine score type for standalone functions (no in-function aggregation).

    This function is designed for standalone optstop functions (optimal_stopping_posthoc,
    optimal_stopping_live, convergence_posthoc) where users provide pre-computed scores
    in the DataFrame. Unlike `determine_score_type` which handles bridge aggregation,
    this function uses explicit `continuous_tasks` patterns to identify continuous data.

    Score Type Priority:
    1. If matches continuous_tasks → 'continuous_bounded' (or 'continuous_01' if upper_bound <= 1)
    2. If matches ordinal_tasks → 'ordinal' (discrete categories)
    3. Default → 'binary' (discrete 0/1)

    Args:
        grouping_name: String identifier for the grouping (e.g., "subject1-task_name")
        ordinal_tasks: List of substrings to match for ordinal (discrete) scoring.
            If None, ordinal detection is disabled.
        continuous_tasks: List of substrings to match for continuous bounded scoring.
            Use this when scores are pre-aggregated floats (e.g., mean of multiple raters).
            If None, continuous detection is disabled.
        upper_bound: Upper bound for score range. Use ordinal_max_score for ordinal/continuous.

    Returns:
        Tuple of (score_type, bounds_dict):
        - score_type: One of 'binary', 'ordinal', 'continuous_01', 'continuous_bounded'
        - bounds_dict: Dictionary with keys 'lower' and 'upper'

    Examples:
        >>> # Default binary
        >>> determine_score_type_standalone("subject1-accuracy")
        ('binary', {'lower': 0.0, 'upper': 1.0})

        >>> # Discrete ordinal (0-10 scale)
        >>> determine_score_type_standalone("subject1-likert", ordinal_tasks=['likert'], upper_bound=10.0)
        ('ordinal', {'lower': 0.0, 'upper': 10.0})

        >>> # Pre-aggregated continuous scores (e.g., mean of ratings)
        >>> determine_score_type_standalone("subject1-mean_rating", continuous_tasks=['mean_'], upper_bound=10.0)
        ('continuous_bounded', {'lower': 0.0, 'upper': 10.0})

        >>> # Continuous 0-1 (e.g., mean of binary scores)
        >>> determine_score_type_standalone("subject1-mean_accuracy", continuous_tasks=['mean_'])
        ('continuous_01', {'lower': 0.0, 'upper': 1.0})
    """
    logger = logging.getLogger('optstop.ordinal_utils')

    # Handle None or non-string grouping_name gracefully
    if grouping_name is None:
        grouping_name_str = ""
    else:
        grouping_name_str = str(grouping_name)
    grouping_name_lower = grouping_name_str.lower()

    # Priority 1: Check for continuous_tasks match
    if continuous_tasks is not None and len(continuous_tasks) > 0:
        for continuous_substring in continuous_tasks:
            if continuous_substring.lower() in grouping_name_lower:
                if upper_bound <= 1.0:
                    return 'continuous_01', {'lower': 0.0, 'upper': 1.0}
                else:
                    return 'continuous_bounded', {'lower': 0.0, 'upper': upper_bound}

    # Priority 2: Check for ordinal_tasks match
    if ordinal_tasks is not None and len(ordinal_tasks) > 0:
        for ordinal_substring in ordinal_tasks:
            if ordinal_substring.lower() in grouping_name_lower:
                return 'ordinal', {'lower': 0.0, 'upper': upper_bound}

    # Priority 3: Default to binary
    return 'binary', {'lower': 0.0, 'upper': 1.0}


def counts_to_scores(counts: np.ndarray) -> np.ndarray:
    """
    Reconstruct individual scores from a category count vector.

    This utility function converts the sufficient statistics (counts per category)
    back into individual scores. Useful for backward compatibility with functions
    that expect raw score arrays.

    Args:
        counts: Array of shape (K,) where counts[k] is the number of observations
                in category k.

    Returns:
        Array of individual scores, where each category k appears counts[k] times.

    Example:
        >>> counts = np.array([0, 2, 3, 1])  # 2 ones, 3 twos, 1 three
        >>> counts_to_scores(counts)
        array([1, 1, 2, 2, 2, 3])
    """
    return np.repeat(np.arange(len(counts)), counts.astype(int))


def aggregate_item_counts(item_summaries: list, ordinal_max_score: int) -> np.ndarray:
    """
    Aggregate category counts across all items into a single count vector.

    Used for flat (non-hierarchical) inference that pools all observations.

    Args:
        item_summaries: List of item summary dicts, each containing 'counts' key
                        with a count vector of shape (K,).
        ordinal_max_score: Maximum ordinal score (K-1 where K is number of categories).

    Returns:
        Aggregated count vector of shape (K,) = (ordinal_max_score + 1,).

    Example:
        >>> summaries = [{'counts': np.array([1, 2, 0])}, {'counts': np.array([0, 1, 1])}]
        >>> aggregate_item_counts(summaries, ordinal_max_score=2)
        array([1, 3, 1])
    """
    K = ordinal_max_score + 1
    total_counts = np.zeros(K, dtype=int)
    for s in item_summaries:
        counts = s.get('counts', np.zeros(K, dtype=int))
        # Ensure counts has correct length
        if len(counts) < K:
            padded = np.zeros(K, dtype=int)
            padded[:len(counts)] = counts
            counts = padded
        total_counts += counts[:K].astype(int)
    return total_counts


def _log_mcmc_diagnostics(trace, diag_logger, context: str) -> None:
    """Log MCMC diagnostics (divergences, ESS, R-hat) from a trace.

    Logs at WARNING level if any diagnostic is concerning, otherwise DEBUG.
    Never raises - diagnostic extraction must not crash the pipeline.
    """
    try:
        import arviz as az
        n_div = int(trace.sample_stats['diverging'].values.sum()) if hasattr(trace, 'sample_stats') else 0
        ess_vals = az.ess(trace)
        rhat_vals = az.rhat(trace)
        ess_min = float(ess_vals.to_array().min().item())
        rhat_max = float(rhat_vals.to_array().max().item())
        if n_div > 0 or ess_min < 100 or rhat_max > 1.05:
            diag_logger.warning(
                f"MCMC [{context}]: divergences={n_div}, ess_min={ess_min:.0f}, rhat_max={rhat_max:.4f}"
            )
        else:
            diag_logger.debug(
                f"MCMC [{context}]: divergences=0, ess_min={ess_min:.0f}, rhat_max={rhat_max:.4f}"
            )
    except Exception:
        pass


def _ordinal_ci_hierarchical_modal(
    item_counts: np.ndarray,
    item_ns: np.ndarray,
    ordinal_max_score: int,
    cred_level: float = 0.97,
    conservatism: float = 5.0,
    low_perf_threshold: float = 0.01,
    current_perf: float = 0.5,
    model_cache: Optional[Dict] = None,
    sampling_kwargs: Optional[Dict] = None
) -> Tuple[float, float, float]:
    """
    Hierarchical Bayesian CI for population modal category.

    Uses Dirichlet-Multinomial model with partial pooling across items.
    The model is defined in rule.py and cached in model_cache.

    KNOWN LIMITATION - Discrete Modal Category Treated as Continuous:
        The modal category is inherently discrete (an integer 0 to K-1). This function
        computes HDI on the posterior modal samples, treating them as continuous values.
        This is a standard approximation for ordinal data where categories have meaningful
        numeric ordering (e.g., Likert scales where category 7 is "between" 6 and 8).

        This approach assumes equal spacing between categories and would NOT be appropriate
        for nominal categorical data. The CI width represents uncertainty in category units,
        not probability units as in binary inference.

        A future enhancement would return the full posterior distribution over categories
        or a discrete credible set. See PLAN_ordered_logistic.md for planned improvements.

    Args:
        item_counts: Category counts per item, shape (n_items, n_categories)
        item_ns: Total observations per item, shape (n_items,)
        ordinal_max_score: Maximum category value (K-1 where K is n_categories)
        cred_level: Credibility level for HDI (default 0.97)
        conservatism: Multiplier for CI width in low-performance scenarios
        low_perf_threshold: Performance threshold for conservatism
        current_perf: Current performance estimate (for conservatism check)
        model_cache: Dict containing cached PyMC model with 'model' key
        sampling_kwargs: MCMC sampling parameters

    Returns:
        Tuple of (lower_bound, upper_bound, effective_width), all scaled to [0,1]
    """
    logger = logging.getLogger('optstop.ordinal_utils')

    # Validate inputs
    if model_cache is None or 'model' not in model_cache:
        logger.warning("No cached model provided for hierarchical modal inference")
        # Fallback to flat inference
        all_scores = counts_to_scores(np.sum(item_counts, axis=0))
        return _ordinal_ci_adaptive(
            all_scores,
            ordinal_max_score=ordinal_max_score,
            cred_level=cred_level,
            conservatism=conservatism,
            low_perf_threshold=low_perf_threshold
        )

    # Import PyMC and arviz here to avoid top-level import issues
    import pymc as pm
    import arviz as az
    import warnings

    # Default sampling kwargs
    if sampling_kwargs is None:
        sampling_kwargs = {
            'draws': 500,
            'tune': 300,
            'chains': 2,
            'cores': 1,
            'progressbar': False,
            'return_inferencedata': True
        }

    n_categories = ordinal_max_score + 1
    n_items = len(item_ns)

    # Get cached model
    ordinal_model = model_cache['model']

    # Update model data
    # Check if using pre-allocated model (Fix 1) or dynamic model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with ordinal_model:
            if 'model_n_items' in model_cache:
                # Pre-allocated model with explicit masking (Fix 1)
                # Works with both dirichlet and ordered_logistic when use_preallocation=True
                # Pad arrays to allocated model size; unobserved items get obs_weight=0
                model_size = model_cache['model_n_items']

                # Pad arrays to model size
                # For valid Multinomial: sum(item_counts[i]) must equal item_ns[i]
                # Padded items: item_ns=1, item_counts=[1,0,0,...] (ensures finite logp)
                item_counts_padded = np.zeros((model_size, n_categories), dtype="int64")
                item_counts_padded[n_items:, 0] = 1  # Padded items: one count in first category
                item_ns_padded = np.ones(model_size, dtype="int64")
                obs_weight_padded = np.zeros(model_size, dtype="float64")

                # Fill observed data in first n_items positions
                item_counts_padded[:n_items] = item_counts.astype("int64")
                item_ns_padded[:n_items] = item_ns.astype("int64")
                obs_weight_padded[:n_items] = 1.0

                pm.set_data({
                    "item_counts": item_counts_padded,
                    "item_ns": item_ns_padded,
                    "obs_weight": obs_weight_padded
                })
            else:
                # Dynamic model (dirichlet or ordered_logistic with use_preallocation=False)
                # Uses original interface with n_items; model recompiles when n_items changes
                pm.set_data({
                    "n_items": np.int64(n_items),
                    "item_counts": item_counts.astype("int64"),
                    "item_ns": item_ns.astype("int64")
                })

            # Sample from posterior
            try:
                trace = pm.sample(**sampling_kwargs)
            except Exception as e:
                logger = logging.getLogger('optstop.ordinal_utils')
                logger.error(f"MCMC sampling failed [ordinal hierarchical modal]: {e}")
                return 0.0, 1.0, 1.0

    logger = logging.getLogger('optstop.ordinal_utils')
    _log_mcmc_diagnostics(trace, logger, "ordinal hierarchical modal")

    # Extract modal_group posterior samples
    modal_samples = trace.posterior["modal_group"].values.flatten()

    # Compute HDI on modal category
    modal_hdi = az.hdi(trace.posterior["modal_group"], hdi_prob=cred_level)

    try:
        lo_cat = float(modal_hdi["modal_group"].sel(hdi="lower").values)
        hi_cat = float(modal_hdi["modal_group"].sel(hdi="higher").values)
    except (KeyError, ValueError):
        # Fallback extraction
        lo_cat = float(modal_hdi["modal_group"].values[0])
        hi_cat = float(modal_hdi["modal_group"].values[1])

    # Scale to [0, 1]
    lo = lo_cat / ordinal_max_score
    hi = hi_cat / ordinal_max_score
    raw_width = hi - lo

    # Apply sample-size-scaled floor (consistent with flat inference)
    n_samples = int(np.sum(item_ns))
    min_ci_width = 1.0 / (ordinal_max_score * np.sqrt(n_samples))

    # Apply conservatism for low-performance scenarios
    if current_perf < low_perf_threshold:
        floored_width = max(raw_width, min_ci_width)
        effective_width = floored_width * conservatism

        # Widen CI bounds symmetrically if floor was applied
        if raw_width < min_ci_width:
            expansion = (min_ci_width - raw_width) / 2
            lo = max(0.0, lo - expansion)
            hi = min(1.0, hi + expansion)

        logger.info(
            f"Hierarchical modal CI (low perf): n_items={n_items}, n_obs={n_samples}, "
            f"raw_width={raw_width:.4f}, effective_width={effective_width:.4f}"
        )
    else:
        effective_width = max(raw_width, min_ci_width)

        # Widen CI bounds symmetrically if floor was applied
        if raw_width < min_ci_width:
            expansion = (min_ci_width - raw_width) / 2
            lo = max(0.0, lo - expansion)
            hi = min(1.0, hi + expansion)
            logger.info(
                f"Hierarchical modal CI (floor applied): n_items={n_items}, n_obs={n_samples}, "
                f"raw_width={raw_width:.4f}, floor={min_ci_width:.4f}"
            )

    return lo, hi, effective_width


def _ordinal_ci_hierarchical_entropy(
    item_counts: np.ndarray,
    item_ns: np.ndarray,
    ordinal_max_score: int,
    cred_level: float = 0.97,
    conservatism: float = 5.0,
    low_perf_threshold: float = 0.01,
    current_perf: float = 0.5,
    model_cache: Optional[Dict] = None,
    sampling_kwargs: Optional[Dict] = None
) -> Tuple[float, float, float, Dict]:
    """
    Hierarchical Bayesian CI for population entropy.

    Uses Dirichlet-Multinomial model with partial pooling across items.
    Returns CI on entropy and diagnostics for stabilization tracking.

    SCALING: Entropy is scaled to [0, 1] by dividing by max_entropy = log(K),
    where K is the number of categories. This makes entropy comparable to
    modal CI width, which is also in [0, 1].
        - 0.0 = deterministic (all mass in one category)
        - 1.0 = uniform distribution (maximum uncertainty)

    Args:
        item_counts: Category counts per item, shape (n_items, n_categories)
        item_ns: Total observations per item, shape (n_items,)
        ordinal_max_score: Maximum category value
        cred_level: Credibility level for HDI
        conservatism: Multiplier for CI width in low-performance scenarios
        low_perf_threshold: Performance threshold for conservatism
        current_perf: Current performance estimate
        model_cache: Dict containing cached PyMC model with 'model' key
        sampling_kwargs: MCMC sampling parameters

    Returns:
        Tuple of (lower_bound, upper_bound, effective_width, diagnostics)
        - Bounds and width are scaled to [0, 1] (entropy / max_entropy)
        - diagnostics contains 'entropy_median', 'entropy_samples' (scaled)
    """
    logger = logging.getLogger('optstop.ordinal_utils')

    # Import PyMC and arviz here to avoid top-level import issues
    import pymc as pm
    import arviz as az
    import warnings

    # Compute max entropy for scaling (log of number of categories)
    n_categories = ordinal_max_score + 1
    max_entropy = np.log(n_categories)

    # Validate inputs
    if model_cache is None or 'model' not in model_cache:
        logger.warning("No cached model provided for hierarchical entropy inference")
        # Return fallback values (scaled)
        return 0.0, 1.0, 1.0, {'entropy_median': 0.5, 'entropy_samples': [], 'max_entropy': max_entropy}

    # Default sampling kwargs
    if sampling_kwargs is None:
        sampling_kwargs = {
            'draws': 500,
            'tune': 300,
            'chains': 2,
            'cores': 1,
            'progressbar': False,
            'return_inferencedata': True
        }

    n_items = len(item_ns)

    # Get cached model
    ordinal_model = model_cache['model']

    # Update model data and sample
    # Check if using pre-allocated model (Fix 1) or dynamic model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with ordinal_model:
            if 'model_n_items' in model_cache:
                # Pre-allocated model with explicit masking (Fix 1)
                # Works with both dirichlet and ordered_logistic when use_preallocation=True
                # Pad arrays to allocated model size; unobserved items get obs_weight=0
                model_size = model_cache['model_n_items']

                # Pad arrays to model size
                # For valid Multinomial: sum(item_counts[i]) must equal item_ns[i]
                # Padded items: item_ns=1, item_counts=[1,0,0,...] (ensures finite logp)
                item_counts_padded = np.zeros((model_size, n_categories), dtype="int64")
                item_counts_padded[n_items:, 0] = 1  # Padded items: one count in first category
                item_ns_padded = np.ones(model_size, dtype="int64")
                obs_weight_padded = np.zeros(model_size, dtype="float64")

                # Fill observed data in first n_items positions
                item_counts_padded[:n_items] = item_counts.astype("int64")
                item_ns_padded[:n_items] = item_ns.astype("int64")
                obs_weight_padded[:n_items] = 1.0

                pm.set_data({
                    "item_counts": item_counts_padded,
                    "item_ns": item_ns_padded,
                    "obs_weight": obs_weight_padded
                })
            else:
                # Dynamic model (dirichlet or ordered_logistic with use_preallocation=False)
                # Uses original interface with n_items; model recompiles when n_items changes
                pm.set_data({
                    "n_items": np.int64(n_items),
                    "item_counts": item_counts.astype("int64"),
                    "item_ns": item_ns.astype("int64")
                })

            # Sample from posterior
            try:
                trace = pm.sample(**sampling_kwargs)
            except Exception as e:
                logger = logging.getLogger('optstop.ordinal_utils')
                logger.error(f"MCMC sampling failed [ordinal hierarchical entropy]: {e}")
                return 0.0, 1.0, 1.0, {
                    'entropy_median': 0.5,
                    'entropy_median_nats': max_entropy / 2,
                    'error': str(e)
                }

    logger = logging.getLogger('optstop.ordinal_utils')
    _log_mcmc_diagnostics(trace, logger, "ordinal hierarchical entropy")

    # Extract entropy_group posterior samples (raw, in nats)
    entropy_samples_raw = trace.posterior["entropy_group"].values.flatten()

    # Scale entropy samples to [0, 1] by dividing by max_entropy
    entropy_samples_scaled = entropy_samples_raw / max_entropy

    # Compute HDI on scaled entropy
    entropy_hdi = az.hdi(trace.posterior["entropy_group"], hdi_prob=cred_level)

    try:
        lo_raw = float(entropy_hdi["entropy_group"].sel(hdi="lower").values)
        hi_raw = float(entropy_hdi["entropy_group"].sel(hdi="higher").values)
    except (KeyError, ValueError):
        # Fallback extraction
        lo_raw = float(entropy_hdi["entropy_group"].values[0])
        hi_raw = float(entropy_hdi["entropy_group"].values[1])

    # Scale to [0, 1]
    lo = lo_raw / max_entropy
    hi = hi_raw / max_entropy
    raw_width = hi - lo
    entropy_median_scaled = float(np.median(entropy_samples_scaled))

    # Apply conservatism for low-performance scenarios
    if current_perf < low_perf_threshold:
        effective_width = raw_width * conservatism
        logger.info(
            f"Hierarchical entropy CI (low perf, scaled): n_items={n_items}, "
            f"entropy_median={entropy_median_scaled:.4f}, raw_width={raw_width:.4f}, "
            f"effective_width={effective_width:.4f}"
        )
    else:
        effective_width = raw_width

    # Build diagnostics dict for stabilization tracking (all values scaled)
    diagnostics = {
        'entropy_median': entropy_median_scaled,
        'entropy_median_nats': float(np.median(entropy_samples_raw)),  # Raw nats for gate comparison
        'entropy_mean': float(np.mean(entropy_samples_scaled)),
        'entropy_samples': entropy_samples_scaled.tolist(),  # Scaled for consistency
        'max_entropy': max_entropy,  # Include for reference
        'n_items': n_items,
        'n_obs': int(np.sum(item_ns))
    }

    return lo, hi, effective_width, diagnostics
