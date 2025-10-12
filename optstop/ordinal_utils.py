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

FUTURE ENHANCEMENT:
For full categorical inference (P(Score = k) for all k), implement hierarchical
OrderedLogistic regression. See GitHub issue #XXX for planned implementation.

References:
    Rubin, D. B. (1981). The Bayesian Bootstrap. The Annals of Statistics, 9(1), 130-134.
"""

import numpy as np
import logging
from typing import Tuple, Optional


def _ordinal_ci_adaptive(
    scores: np.ndarray,
    ordinal_max_score: int,
    cred_level: float = 0.95,
    conservatism: float = 1.0,
    low_perf_threshold: float = 0.2,
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

    Args:
        scores: Array of ordinal scores (0 to ordinal_max_score)
        ordinal_max_score: Maximum possible score for scaling (e.g., 10 for 0-10 scale)
        cred_level: Credibility level (e.g., 0.95 for 95% CI)
        conservatism: Multiplier for CI width in low-performance scenarios (>= 1.0)
        low_perf_threshold: Performance threshold below which conservatism is applied (0-1 scale)
        base_strength: Base prior strength (for future Bayesian enhancements, currently unused)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (lower_bound, upper_bound, effective_width), all in [0,1] scale
        - lower_bound: Lower bound of modal category CI (scaled)
        - upper_bound: Upper bound of modal category CI (scaled)
        - effective_width: CI width (scaled), adjusted for conservatism if needed

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
        logger.warning("Empty scores array provided to _ordinal_ci_adaptive")
        return 0.0, 1.0, 1.0

    # Ensure scores are integers
    scores_int = scores.astype(int)

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
    width = hi - lo

    # Apply conservatism for low-performance scenarios
    # Use mean of scaled scores to determine if performance is low
    mean_scaled = np.mean(scores_int / ordinal_max_score)

    if mean_scaled < low_perf_threshold:
        # Scale up the effective width for more stringent stopping criteria
        # This prevents premature stopping when performance is poor
        effective_width = width * conservatism
        logger.debug(
            f"Applied conservatism to ordinal CI: mean_scaled={mean_scaled:.3f}, "
            f"modal_cat_range=[{lo_cat:.1f}, {hi_cat:.1f}], "
            f"raw_width={width:.4f}, effective_width={effective_width:.4f}"
        )
    else:
        effective_width = width

    return lo, hi, effective_width


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
        logger.warning(f"Ordinal grouping '{grouping_name}' has no valid scores")
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
        logger.warning(
            f"Ordinal grouping '{grouping_name}' contains non-integer scores. "
            f"These will be used as-is, but ordinal scoring typically expects integers."
        )

    logger.debug(
        f"Validated ordinal scores for '{grouping_name}': "
        f"range=[{min_score}, {max_score}], n={len(valid_scores)}"
    )


def determine_score_type(
    grouping_name: str,
    ordinal_tasks: Optional[list] = None
) -> str:
    """
    Determine score type (binary or ordinal) based on grouping name and ordinal_tasks list.

    This function checks if the grouping name contains any substring from the ordinal_tasks
    list. Matching is case-insensitive.

    Args:
        grouping_name: String identifier for the grouping (e.g., "1-1", "subject1-likert_task")
        ordinal_tasks: List of substrings to match for ordinal scoring. If None, returns 'binary'.

    Returns:
        'binary' or 'ordinal'

    Examples:
        >>> determine_score_type("subject1-accuracy", None)
        'binary'

        >>> determine_score_type("subject1-likert_scale", ['likert'])
        'ordinal'

        >>> determine_score_type("1-task_rating", ['rating', 'likert'])
        'ordinal'

        >>> determine_score_type("subject2-accuracy", ['rating', 'likert'])
        'binary'
    """
    logger = logging.getLogger('optstop.ordinal_utils')

    # Default to binary if no ordinal tasks specified
    if ordinal_tasks is None or len(ordinal_tasks) == 0:
        return 'binary'

    # Check if any ordinal substring matches this grouping name (case-insensitive)
    grouping_name_lower = grouping_name.lower()

    for ordinal_substring in ordinal_tasks:
        if ordinal_substring.lower() in grouping_name_lower:
            logger.info(
                f"Grouping '{grouping_name}' matched ordinal pattern '{ordinal_substring}' → using ordinal scoring"
            )
            return 'ordinal'

    logger.debug(f"Grouping '{grouping_name}' does not match any ordinal patterns → using binary scoring")
    return 'binary'
