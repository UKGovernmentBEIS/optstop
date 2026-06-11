"""
Ordinal models for ordinal scoring with entropy-based stopping.

This module provides hierarchical Bayesian inference for ordinal scores (e.g., 0-10 Likert scales).
Two model types are available:

Model Types:
    ordered_logistic (default):
        Cumulative link model that respects ordinal structure via latent scale.
        - Uses identified cutpoints (first cutpoint fixed at 0)
        - Adaptive priors scale with number of categories
        - Better theoretical fit for truly ordinal data
        - Faster sampling (~3x vs Dirichlet-Multinomial)

    dirichlet:
        Dirichlet-Multinomial model treating categories as exchangeable.
        - Does not enforce ordinal structure
        - More robust to bimodal/U-shaped distributions
        - Available as fallback if ordered_logistic fails

Key Features:
- Full categorical distribution inference (not just modal category)
- Entropy-based uncertainty quantification
- Threshold-based queries (P(Score ≥ k))
- Hierarchical structure with partial pooling across items
- Non-centered parameterization for efficient MCMC sampling

Usage:
    Set `ordinal_model_type` parameter in optimal_stopping functions:
    - 'ordered_logistic': Use cumulative link model (default)
    - 'dirichlet': Use Dirichlet-Multinomial model

References:
- McCullagh, P. (1980). Regression models for ordinal data. JRSS Series B, 42(2), 109-127.
- Bürkner, P. C., & Vuorre, M. (2019). Ordinal regression models in psychology. AMPPS, 2(1), 77-101.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import logging
from typing import Tuple, Dict, Any, Optional
import arviz as az

logger = logging.getLogger(__name__)


def _log_mcmc_diagnostics(trace, diag_logger, context: str) -> None:
    """Log MCMC diagnostics (divergences, ESS, R-hat) from a trace.

    Logs at WARNING level if any diagnostic is concerning, otherwise DEBUG.
    Never raises - diagnostic extraction must not crash the pipeline.
    """
    try:
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


def _create_orderedlogistic_model(
    n_categories: int,
    n_items: int = 1,
    mu_group_prior: Tuple[float, float] = (0.0, 2.0),
    sigma_group_prior: float = 1.0
) -> pm.Model:
    """
    Create hierarchical OrderedLogistic model for ordinal data.

    Model Structure:
    ----------------
    Group level:
        μ_group ~ Normal(mu_group_prior[0], mu_group_prior[1])
        σ_group ~ Exponential(sigma_group_prior)

    Item level (for each item i):
        z_i ~ Normal(0, 1)
        η_i = μ_group + σ_group * z_i

    Cutpoints (ordered):
        c_1 < c_2 < ... < c_{K-1}
        where K = n_categories

    Likelihood:
        Score_i ~ OrderedLogistic(η_i, cutpoints)

    Parameters
    ----------
    n_categories : int
        Number of ordinal categories (e.g., 11 for 0-10 scale)
    n_items : int, default=1
        Number of items/samples
    mu_group_prior : Tuple[float, float], default=(0.0, 2.0)
        Prior for group-level mean: (mean, std)
    sigma_group_prior : float, default=1.0
        Rate parameter for Exponential prior on group-level std

    Returns
    -------
    model : pm.Model
        PyMC model ready for sampling

    Examples
    --------
    >>> model = _create_orderedlogistic_model(n_categories=11, n_items=10)
    >>> # Update data and sample
    >>> with model:
    ...     pm.set_data({"scores": np.array([5, 6, 7, 7, 8, 7, 6, 7, 7, 8])})
    ...     trace = pm.sample(1000, tune=500)
    """
    with pm.Model() as model:
        # Group-level parameters
        mu_group = pm.Normal("mu_group", mu=mu_group_prior[0], sigma=mu_group_prior[1])
        sigma_group = pm.Exponential("sigma_group", lam=sigma_group_prior)

        # Item-level latent abilities (non-centered parameterization)
        n_items_data = pm.Data("n_items", np.array(n_items, dtype="int64"))
        z = pm.Normal("z", mu=0, sigma=1, shape=n_items_data)
        eta = pm.Deterministic("eta", mu_group + z * sigma_group)

        # Ordered cutpoints
        n_cutpoints = n_categories - 1
        cutpoint_init = np.linspace(-3, 3, n_cutpoints)
        cutpoints = pm.Normal(
            "cutpoints",
            mu=cutpoint_init,
            sigma=1.0,
            shape=n_cutpoints,
            transform=pm.distributions.transforms.ordered,
            initval=cutpoint_init
        )

        # Likelihood
        scores_data = pm.Data("scores", np.zeros(n_items, dtype="int64"))
        obs = pm.OrderedLogistic("obs", eta=eta, cutpoints=cutpoints, observed=scores_data)

    return model


def _compute_adaptive_cutpoint_prior_params(n_categories: int) -> Dict[str, float]:
    """
    Compute adaptive prior parameters for cutpoint increments based on K.

    The prior scales with number of categories to ensure reasonable spacing:
    - Total spread increases with K (more categories = wider latent scale)
    - Individual increments decrease with K (more cutpoints to fit)

    Design:
        expected_spread = 2 * log(K)  # Grows slowly: K=5->3.2, K=11->4.8, K=21->6.1
        expected_increment = spread / (K-2)  # Per-increment target

    For softplus(Normal(mu, sigma)), we adjust mu to achieve target increment.

    Parameters
    ----------
    n_categories : int
        Number of ordinal categories (K)

    Returns
    -------
    dict with keys:
        - expected_spread: Total expected spread from first to last cutpoint
        - expected_increment: Expected size of each increment
        - delta_mu: Mean for Normal prior on raw increments
        - delta_sigma: Std for Normal prior on raw increments
    """
    K = n_categories

    # Total spread from first to last cutpoint (log scale growth)
    expected_spread = 2.0 * np.log(K)

    # Number of increments to estimate (K-1 cutpoints, first fixed, so K-2 increments)
    n_increments = max(K - 2, 1)

    # Expected increment size
    expected_increment = expected_spread / n_increments

    # For softplus(Normal(mu, sigma)):
    # softplus(x) = log(1 + exp(x)), softplus(0) ≈ 0.693
    # To target expected_increment, adjust mu
    # Approximation: mu ≈ expected_increment - 0.5 works reasonably
    delta_mu = expected_increment - 0.5
    delta_sigma = max(expected_increment * 0.5, 0.3)  # Allow ~50% variation, min 0.3

    return {
        'expected_spread': float(expected_spread),
        'expected_increment': float(expected_increment),
        'delta_mu': float(delta_mu),
        'delta_sigma': float(delta_sigma),
        'n_increments': int(n_increments)
    }


def _create_ordered_logistic_hierarchical(
    n_categories: int,
    n_items: int,
    mu_group_prior: Tuple[float, float] = (0.0, 2.0),
    sigma_group_prior: float = 1.0,
    use_preallocation: bool = True
) -> pm.Model:
    """
    Create IDENTIFIED hierarchical OrderedLogistic model for aggregated count data.

    This model addresses critical issues from PLAN_ordered_logistic.md:
    1. Identification: First cutpoint fixed at 0
    2. Adaptive priors: Cutpoint increments scale with K
    3. Aggregated likelihood: Multinomial on category probabilities
    4. Optional pre-allocation: Fixed shape with explicit masking (Fix 1)

    Model Structure:
    ----------------
    Group level:
        μ_group ~ Normal(mu_group_prior[0], mu_group_prior[1])
        σ_group ~ Exponential(sigma_group_prior)

    Item level (non-centered parameterization):
        z_i ~ Normal(0, 1)
        η_i = μ_group + σ_group * z_i

    Cutpoints (IDENTIFIED - first fixed at 0):
        c_0 = 0 (fixed)
        δ_raw ~ Normal(adaptive_mu, adaptive_sigma)  # K-2 parameters
        δ = softplus(δ_raw)  # Ensure positive increments
        c_k = c_{k-1} + δ_{k-1} for k > 0

    Category probabilities (from cumulative logistic):
        P(Y ≤ k) = sigmoid(c_k - η_i)
        P(Y = k) = P(Y ≤ k) - P(Y ≤ k-1)

    Likelihood:
        When use_preallocation=True:
            log_lik = obs_weight * logp(Multinomial(n_i, probs_i), counts_i)
            obs_weight = 0 for unobserved items (zero contribution to likelihood)
        When use_preallocation=False:
            obs ~ Multinomial(n_i, probs_i) with dynamic n_items

    Parameters
    ----------
    n_categories : int
        Number of ordinal categories (e.g., 11 for 0-10 scale)
    n_items : int
        Number of items. When use_preallocation=True, this is the pre-allocated
        maximum (may be > actual observed items). When use_preallocation=False,
        this is the current number of items (model recompiles when it changes).
    mu_group_prior : Tuple[float, float], default=(0.0, 2.0)
        Prior for group-level mean: (mean, std)
    sigma_group_prior : float, default=1.0
        Rate parameter for Exponential prior on group-level std
    use_preallocation : bool, default=True
        If True, uses fixed shape with explicit obs_weight masking (Fix 1).
        If False, uses dynamic n_items_data with direct Multinomial likelihood
        (original behavior, requires model recompilation when n_items changes).

    Returns
    -------
    model : pm.Model
        PyMC model with Data containers for item_counts, item_ns, and optionally
        obs_weight (only when use_preallocation=True) or n_items (only when False).

    Notes
    -----
    Data must be set before sampling.

    When use_preallocation=True:
        with model:
            pm.set_data({
                "item_counts": counts_array,  # shape (n_items, n_categories)
                "item_ns": ns_array,          # shape (n_items,)
                "obs_weight": weight_array    # shape (n_items,) - 1.0 observed, 0.0 padded
            })
        For padded (unobserved) items, use:
            item_counts[i, 0] = 1, item_ns[i] = 1, obs_weight[i] = 0.0

    When use_preallocation=False:
        with model:
            pm.set_data({
                "n_items": np.int64(n_items),
                "item_counts": counts_array,  # shape (n_items, n_categories)
                "item_ns": ns_array,          # shape (n_items,)
            })

    See Also
    --------
    _create_orderedlogistic_model : Original model with individual observations
    PLAN_phase0_ordered_logistic.md : Design documentation
    """
    # Get adaptive prior parameters
    prior_params = _compute_adaptive_cutpoint_prior_params(n_categories)

    with pm.Model() as model:
        # === GROUP-LEVEL PARAMETERS ===
        mu_group = pm.Normal("mu_group", mu=mu_group_prior[0], sigma=mu_group_prior[1])
        sigma_group = pm.Exponential("sigma_group", lam=sigma_group_prior)

        # === ITEM-LEVEL (non-centered parameterization) ===
        if use_preallocation:
            # Fixed shape at n_items for pre-allocation (Fix 1)
            z = pm.Normal("z", mu=0, sigma=1, shape=n_items)
        else:
            # Dynamic shape using n_items_data (original behavior)
            n_items_data = pm.Data("n_items", np.array(n_items, dtype="int64"))
            z = pm.Normal("z", mu=0, sigma=1, shape=n_items_data)
        eta = pm.Deterministic("eta", mu_group + z * sigma_group)

        # === IDENTIFIED CUTPOINTS ===
        # K categories need K-1 cutpoints
        # Fix first cutpoint at 0, estimate K-2 increments
        n_cutpoints = n_categories - 1

        if n_cutpoints == 1:
            # Special case: 2 categories, single cutpoint fixed at 0
            cutpoints = pt.as_tensor_variable(np.array([0.0]))
        elif n_cutpoints == 2:
            # Special case: 3 categories, one free increment
            delta_raw = pm.Normal(
                "delta_raw",
                mu=prior_params['delta_mu'],
                sigma=prior_params['delta_sigma'],
                shape=1
            )
            delta = pt.softplus(delta_raw)
            cutpoints = pm.Deterministic(
                "cutpoints",
                pt.concatenate([pt.zeros(1), delta])
            )
        else:
            # General case: K-2 free increments
            n_increments = n_cutpoints - 1
            delta_raw = pm.Normal(
                "delta_raw",
                mu=prior_params['delta_mu'],
                sigma=prior_params['delta_sigma'],
                shape=n_increments
            )
            deltas = pt.softplus(delta_raw)
            cutpoints = pm.Deterministic(
                "cutpoints",
                pt.concatenate([pt.zeros(1), pt.cumsum(deltas)])
            )

        # === CATEGORY PROBABILITIES FROM ORDERED LOGISTIC ===
        # P(Y <= k) = sigmoid(c_k - eta)
        # P(Y = k) = P(Y <= k) - P(Y <= k-1)

        # Expand dimensions for broadcasting
        # eta: (n_items,) -> (n_items, 1)
        # cutpoints: (n_cutpoints,) -> (1, n_cutpoints)
        eta_expanded = eta[:, None]  # (n_items, 1)
        cutpoints_expanded = cutpoints[None, :]  # (1, n_cutpoints)

        # Cumulative probabilities: P(Y <= k)
        cum_probs = pm.math.sigmoid(cutpoints_expanded - eta_expanded)  # (n_items, n_cutpoints)

        # Category probabilities
        # P(Y = 0) = P(Y <= 0) = cum_probs[:, 0]
        # P(Y = k) = P(Y <= k) - P(Y <= k-1) for 0 < k < K-1
        # P(Y = K-1) = 1 - P(Y <= K-2) = 1 - cum_probs[:, -1]
        p_first = cum_probs[:, :1]  # (n_items, 1)
        p_middle = cum_probs[:, 1:] - cum_probs[:, :-1]  # (n_items, K-2)
        p_last = 1.0 - cum_probs[:, -1:]  # (n_items, 1)

        probs = pm.Deterministic(
            "probs",
            pt.concatenate([p_first, p_middle, p_last], axis=1)
        )

        # === DATA CONTAINERS FOR AGGREGATED COUNTS ===
        if use_preallocation:
            # Pre-allocated mode (Fix 1): fixed shape with explicit masking
            # Default values ensure valid Multinomial: sum(item_counts[i]) = item_ns[i]
            # Padded items: item_counts[i, 0] = 1, item_ns[i] = 1
            default_counts = np.zeros((n_items, n_categories), dtype="int64")
            default_counts[:, 0] = 1  # One count in first category
            item_counts_data = pm.Data("item_counts", default_counts)
            item_ns_data = pm.Data("item_ns", np.ones(n_items, dtype="int64"))
            obs_weight_data = pm.Data("obs_weight", np.zeros(n_items, dtype="float64"))

            # MULTINOMIAL LIKELIHOOD WITH EXPLICIT MASKING
            # obs_weight=0 for unobserved items → zero contribution to likelihood
            multinomial_dist = pm.Multinomial.dist(n=item_ns_data, p=probs)
            log_lik = obs_weight_data * pm.logp(multinomial_dist, item_counts_data)
            pm.Potential("obs_likelihood", pm.math.sum(log_lik))
        else:
            # Dynamic mode (original behavior): shape changes with n_items
            # Model recompiles when n_items changes
            default_counts = np.zeros((n_items, n_categories), dtype="int64")
            item_counts_data = pm.Data("item_counts", default_counts)
            item_ns_data = pm.Data("item_ns", np.ones(n_items, dtype="int64"))

            # DIRECT MULTINOMIAL LIKELIHOOD (no masking needed)
            obs = pm.Multinomial("obs", n=item_ns_data, p=probs, observed=item_counts_data)

        # === DERIVED QUANTITIES FOR STOPPING CRITERIA ===
        # Group-level category probabilities (average over items)
        probs_group = pm.Deterministic("probs_group", probs.mean(axis=0))

        # Modal category at group level
        modal_group = pm.Deterministic("modal_group", pt.argmax(probs_group))

        # Entropy at group level (natural log, in nats)
        # H = -sum(p * log(p)), consistent with dirichlet model and ordinal_utils scaling
        entropy_group = pm.Deterministic(
            "entropy_group",
            -pm.math.sum(probs_group * pm.math.log(probs_group + 1e-10))
        )

    return model


def _compute_category_probabilities(
    eta: np.ndarray,
    cutpoints: np.ndarray,
    n_categories: int
) -> np.ndarray:
    """
    Compute category probabilities from OrderedLogistic parameters (vectorized).

    For each latent ability η and ordered cutpoints c_1 < ... < c_{K-1},
    computes P(Y = k) for k = 0, 1, ..., K-1.

    OrderedLogistic probability formulation:
        P(Y = 0) = logistic(c_1 - η)
        P(Y = k) = logistic(c_{k+1} - η) - logistic(c_k - η)  for 0 < k < K-1
        P(Y = K-1) = 1 - logistic(c_{K-1} - η)

    Parameters
    ----------
    eta : np.ndarray, shape (n_samples, n_items)
        Latent ability parameters from posterior samples
    cutpoints : np.ndarray, shape (n_samples, n_cutpoints)
        Ordered cutpoint parameters from posterior samples
    n_categories : int
        Number of ordinal categories

    Returns
    -------
    probs : np.ndarray, shape (n_samples, n_items, n_categories)
        Probability of each category for each item in each posterior sample

    Examples
    --------
    >>> eta = np.array([[0.5, 1.0], [0.3, 0.9]])  # 2 samples, 2 items
    >>> cutpoints = np.array([[-1, 0, 1], [-1.1, 0.1, 1.1]])  # 2 samples, 3 cutpoints (4 categories)
    >>> probs = _compute_category_probabilities(eta, cutpoints, n_categories=4)
    >>> probs.shape
    (2, 2, 4)  # (n_samples, n_items, n_categories)

    Notes
    -----
    Vectorized implementation for improved performance (~2-3x faster than loop-based version).
    """
    n_samples = eta.shape[0]
    n_items = eta.shape[1] if eta.ndim > 1 else 1

    # Ensure eta is 2D
    if eta.ndim == 1:
        eta = eta[:, np.newaxis]

    # Initialize probability array
    probs = np.zeros((n_samples, n_items, n_categories))

    # Logistic function (vectorized)
    def logistic(x):
        return 1.0 / (1.0 + np.exp(-x))

    # Vectorized computation using broadcasting
    # Shape: cutpoints (n_samples, n_cutpoints), eta (n_samples, n_items)
    # Expand dimensions for broadcasting: cutpoints[:, np.newaxis, :] - eta[:, :, np.newaxis]
    # Result: (n_samples, n_items, n_cutpoints)
    cutpoints_expanded = cutpoints[:, np.newaxis, :]  # (n_samples, 1, n_cutpoints)
    eta_expanded = eta[:, :, np.newaxis]  # (n_samples, n_items, 1)

    # Compute all cumulative probabilities at once
    cum_probs = logistic(cutpoints_expanded - eta_expanded)  # (n_samples, n_items, n_cutpoints)

    # P(Y = 0) = cum_probs[:, :, 0]
    probs[:, :, 0] = cum_probs[:, :, 0]

    # P(Y = k) = cum_probs[:, :, k] - cum_probs[:, :, k-1] for 1 <= k < K-1
    if n_categories > 2:
        probs[:, :, 1:-1] = cum_probs[:, :, 1:] - cum_probs[:, :, :-1]

    # P(Y = K-1) = 1 - cum_probs[:, :, -1]
    probs[:, :, -1] = 1.0 - cum_probs[:, :, -1]

    # Ensure probabilities sum to 1 (numerical stability)
    probs = np.clip(probs, 0, 1)
    probs /= probs.sum(axis=2, keepdims=True)

    return probs


def _compute_entropy(probs: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute Shannon entropy of categorical distributions.

    Entropy quantifies uncertainty in the distribution:
        H = -Σ P(k) * log₂(P(k))

    Entropy Scale Reference (for 11 categories, 0-10 scale):
    - 0.0-1.0:  Very peaked → High confidence (e.g., 90% in one category)
    - 1.0-2.0:  Moderately spread → Moderate confidence (e.g., 40% mode, rest spread)
    - 2.0-3.0:  Diffuse → Low confidence (e.g., bimodal or wide)
    - 3.32:     Maximum for 11 categories (uniform distribution)

    Parameters
    ----------
    probs : np.ndarray, shape (..., n_categories)
        Probability distributions (last dimension must sum to 1)
    epsilon : float, default=1e-10
        Small constant to avoid log(0)

    Returns
    -------
    entropy : np.ndarray, shape (...)
        Entropy for each distribution

    Examples
    --------
    >>> # Very certain distribution
    >>> probs = np.array([[0.95, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    >>> _compute_entropy(probs)
    array([0.286])  # Low entropy → high confidence

    >>> # Uniform distribution (maximum uncertainty)
    >>> probs = np.ones((1, 11)) / 11
    >>> _compute_entropy(probs)
    array([3.459])  # Maximum entropy for 11 categories
    """
    # Clip probabilities to avoid log(0)
    probs_safe = np.clip(probs, epsilon, 1.0)

    # Compute entropy: H = -Σ P(k) * log₂(P(k))
    entropy = -np.sum(probs_safe * np.log2(probs_safe), axis=-1)

    return entropy


def _ordinal_entropy_ci_adaptive(
    scores: np.ndarray,
    ordinal_max_score: int,
    cred_level: float = 0.97,
    conservatism: float = 5.0,
    low_perf_threshold: float = 0.01,
    n_samples: int = 1000,
    n_tune: int = 1000,
    model_cache: Optional[Dict[str, Any]] = None,
    compute_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Compute adaptive credible interval on entropy for ordinal scores using OrderedLogistic.

    This is the main function for entropy-based stopping. It:
    1. Fits hierarchical OrderedLogistic model to scores
    2. Computes full categorical distribution P(Score = k) from posterior
    3. Computes entropy H = -Σ P(k) * log(P(k)) for each posterior sample
    4. Returns credible interval on entropy

    Stopping Decision:
    ------------------
    Stop when entropy CI width < entropy_threshold

    Lower entropy → More confident about distribution shape → Stop earlier

    Parameters
    ----------
    scores : np.ndarray, shape (n,)
        Observed ordinal scores (integers in [0, ordinal_max_score])
    ordinal_max_score : int
        Maximum possible score (e.g., 10 for 0-10 scale)
    cred_level : float, default=0.97
        Credible level (e.g., 0.97 for 97% CI)
    conservatism : float, default=5.0
        Multiplier to inflate effective CI width for low performance
        Higher conservatism → wider CI → more data needed
    low_perf_threshold : float, default=0.01
        Performance below this triggers conservatism
    n_samples : int, default=1000
        Number of posterior samples to draw
    n_tune : int, default=1000
        Number of tuning samples for MCMC
    model_cache : dict, optional
        Cache for PyMC model to avoid recompilation
        Keys: 'model', 'n_items_last'
    compute_kwargs : dict, optional
        Additional kwargs for pm.sample (e.g., random_seed, cores)

    Returns
    -------
    entropy_lo : float
        Lower bound of credible interval on entropy
    entropy_hi : float
        Upper bound of credible interval on entropy
    entropy_width : float
        Effective CI width (after conservatism adjustment)
    diagnostics : dict
        Diagnostic information:
            - 'entropy_median': Median entropy
            - 'entropy_samples': All posterior entropy samples
            - 'probs_median': Median category probabilities
            - 'trace': PyMC trace object
            - 'rhat_max': Maximum R-hat (convergence diagnostic)
            - 'ess_min': Minimum effective sample size

    Examples
    --------
    >>> scores = np.array([5, 6, 7, 7, 8, 7, 6, 7, 7, 8])
    >>> lo, hi, width, diagnostics = _ordinal_entropy_ci_adaptive(
    ...     scores, ordinal_max_score=10, cred_level=0.97
    ... )
    >>> print(f"Entropy CI: [{lo:.2f}, {hi:.2f}], width: {width:.2f}")
    Entropy CI: [1.50, 2.20], width: 0.70
    >>> print(f"Median entropy: {diagnostics['entropy_median']:.2f}")
    Median entropy: 1.85
    """
    if len(scores) == 0:
        raise ValueError("Cannot compute entropy CI with zero scores")

    # Validate scores
    if np.any(scores < 0) or np.any(scores > ordinal_max_score):
        raise ValueError(f"Scores must be in [0, {ordinal_max_score}]")

    # Round to nearest integer and clip to valid category range
    scores_int = np.clip(np.round(scores).astype("int64"), 0, ordinal_max_score)

    n_categories = ordinal_max_score + 1
    # FIX: All scores come from a single distribution, not separate items
    # Using n_items=len(scores) caused memory explosion (39GB+ arrays)
    # and extreme slowdown with large datasets
    n_items = 1

    # Create or reuse model
    if model_cache is not None and 'model' in model_cache:
        model = model_cache['model']
        n_items_last = model_cache.get('n_items_last', 0)

        # Update data
        with model:
            pm.set_data({"n_items": np.array(n_items, dtype="int64")})
            pm.set_data({"scores": scores_int})

        # logger.debug(f"Reusing OrderedLogistic model (updated from {n_items_last} to {n_items} items)")
    else:
        # Create new model
        model = _create_orderedlogistic_model(
            n_categories=n_categories,
            n_items=n_items
        )

        # Update scores data
        with model:
            pm.set_data({"scores": scores_int})

        # logger.debug(f"Created new OrderedLogistic model for {n_items} items")

    # Sample from posterior
    compute_kwargs = compute_kwargs or {}

    # === DIAGNOSTIC LOGGING: Track parameter passing ===
    # logger.warning(f"_ordinal_entropy_ci_adaptive called:")
    # logger.warning(f"   n_samples (default parameter): {n_samples}")
    # logger.warning(f"   n_tune (default parameter): {n_tune}")
    # logger.warning(f"   compute_kwargs received: {compute_kwargs}")

    default_kwargs = {
        'draws': n_samples,
        'tune': n_tune,
        'random_seed': compute_kwargs.get('random_seed', None),
        'progressbar': False,
        'return_inferencedata': True
    }
    default_kwargs.update(compute_kwargs)

    # === DIAGNOSTIC LOGGING: Track final values being used ===
    # logger.warning(f"   Final draws (after update): {default_kwargs['draws']}")
    # logger.warning(f"   Final tune (after update): {default_kwargs['tune']}")

    # NOTE: Ordinal inference uses numpyro (JAX backend) for CPU sampling
    # This is configured in gpu_utils.py and avoids numba cumsum compatibility issues
    # that affected nutpie. numpyro handles OrderedLogistic correctly on both CPU and GPU.

    try:
        with model:
            trace = pm.sample(**default_kwargs)
    except Exception as e:
        logger.error(f"OrderedLogistic sampling failed: {e}")
        # Fallback to high uncertainty
        max_entropy = np.log2(n_categories)
        return 0.0, max_entropy, max_entropy, {
            'entropy_median': max_entropy / 2,
            'error': str(e)
        }

    _log_mcmc_diagnostics(trace, logger, "ordinal item entropy")

    # Extract posterior samples
    eta_samples = trace.posterior['eta'].values  # shape: (chains, draws, n_items)
    cutpoints_samples = trace.posterior['cutpoints'].values  # shape: (chains, draws, n_cutpoints)

    # Reshape to (n_samples, ...)
    eta_samples = eta_samples.reshape(-1, n_items)
    cutpoints_samples = cutpoints_samples.reshape(-1, n_categories - 1)

    # Compute category probabilities for each posterior sample
    probs = _compute_category_probabilities(eta_samples, cutpoints_samples, n_categories)

    # Average probabilities across items for group-level inference
    probs_group = probs.mean(axis=1)  # shape: (n_samples, n_categories)

    # Compute entropy for each posterior sample
    entropy_samples = _compute_entropy(probs_group)

    # Compute credible interval on entropy
    alpha = (1 - cred_level) / 2
    entropy_lo = np.percentile(entropy_samples, alpha * 100)
    entropy_hi = np.percentile(entropy_samples, (1 - alpha) * 100)
    raw_width = entropy_hi - entropy_lo

    # Apply conservatism based on performance
    # Performance = mean scaled score
    current_perf = np.mean(scores) / ordinal_max_score

    if current_perf < low_perf_threshold:
        effective_width = raw_width * conservatism
        logger.info(
            f"Applied conservatism {conservatism:.2f} (perf={current_perf:.2f} < {low_perf_threshold}) "
            f"→ width {raw_width:.3f} → {effective_width:.3f}"
        )
    else:
        effective_width = raw_width

    # Convergence diagnostics
    rhat = az.rhat(trace)
    ess = az.ess(trace)

    # Extract max/min values from xarray Dataset
    # Convert to DataArray first, then extract scalar with .item()
    rhat_max = float(rhat.to_array().max().item())
    ess_min = float(ess.to_array().min().item())

    # Update cache
    if model_cache is not None:
        model_cache['model'] = model
        model_cache['n_items_last'] = n_items

    # Return diagnostics
    diagnostics = {
        'entropy_median': float(np.median(entropy_samples)),
        'entropy_samples': entropy_samples,
        'probs_median': probs_group.mean(axis=0),  # Average over posterior samples
        'trace': trace,
        'rhat_max': rhat_max,
        'ess_min': ess_min,
        'n_items': n_items,
        'n_samples': len(entropy_samples)
    }

    return entropy_lo, entropy_hi, effective_width, diagnostics


def _ordinal_hybrid_stopping_criterion(
    scores: np.ndarray,
    ordinal_max_score: int,
    delta_item: float,
    cred_level: float,
    entropy_history: list,
    entropy_threshold: float = 0.8,
    conservatism: float = 5.0,
    low_perf_threshold: float = 0.01,
    min_epochs_for_stabilization: int = 3,
    entropy_convergence_threshold: float = 0.10,
    model_cache: Optional[Dict[str, Any]] = None,
    compute_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Hybrid stopping criterion for ordinal data combining modal CI and entropy convergence.

    Two pathways to stopping:
    1. **Pathway 1 (Modal CI):** Stop when modal category CI is narrow → peaked performance
    2. **Pathway 2 (Entropy Convergence):** Stop when entropy CI width is narrow → converged distribution

    This approach mirrors binary stopping logic:
    - Binary: Stop when CI narrow OR CI stabilized
    - Ordinal: Stop when modal CI narrow OR entropy converged

    Parameters
    ----------
    scores : np.ndarray
        Observed ordinal scores
    ordinal_max_score : int
        Maximum possible score (e.g., 10 for 0-10 scale)
    delta_item : float
        Threshold for modal CI width (e.g., 0.15)
    cred_level : float
        Credible level for intervals (e.g., 0.95)
    entropy_history : list
        List of (entropy_lo, entropy_hi, entropy_width) tuples from previous epochs
        Modified in-place to add current epoch
    entropy_threshold : float, default=0.8
        Proportion of maximum entropy for false peak detection (0 to 1).
        Scaled internally by log2(num_categories) to produce an effective
        threshold in bits. If modal CI is narrow but entropy exceeds this
        effective threshold, stopping is blocked (false peak protection).
    conservatism : float, default=5.0
        Multiplier for CI width adjustment
    low_perf_threshold : float, default=0.01
        Performance threshold for conservatism adjustment
    min_epochs_for_stabilization : int, default=3
        Minimum number of entropy CI checks before Pathway 2 can fire
    entropy_convergence_threshold : float, default=0.10
        Absolute entropy CI width threshold on [0,1] scale for Pathway 2 convergence.
        Width < threshold means entropy is known to within ±(threshold/2) of max.
    model_cache : dict, optional
        Cache for PyMC model reuse
    compute_kwargs : dict, optional
        Additional kwargs for PyMC sampling

    Returns
    -------
    should_stop : bool
        Whether to stop collecting data
    reason : str
        Stopping reason: 'modal_ci_narrow_validated', 'entropy_converged', or 'continue_*'
    diagnostics : dict
        Diagnostic information including modal CI, entropy CI, and history

    Examples
    --------
    >>> scores = np.array([7, 7, 8, 7, 7])
    >>> entropy_hist = []
    >>> stop, reason, diag = _ordinal_hybrid_stopping_criterion(
    ...     scores, ordinal_max_score=10, delta_item=0.15,
    ...     cred_level=0.97, entropy_history=entropy_hist
    ... )
    >>> print(f"Stop: {stop}, Reason: {reason}")
    Stop: True, Reason: modal_ci_narrow_validated
    """
    from .ordinal_utils import _ordinal_ci_adaptive

    # Scale entropy threshold from proportion to absolute bits
    num_categories = ordinal_max_score + 1
    max_entropy_bits = np.log2(num_categories)
    effective_entropy_threshold = entropy_threshold * max_entropy_bits

    # === COMPUTE BOTH METRICS FIRST ===
    # Modal CI (fast, bootstrap-based)
    modal_lo, modal_hi, modal_width = _ordinal_ci_adaptive(
        scores,
        ordinal_max_score=ordinal_max_score,
        cred_level=cred_level,
        conservatism=conservatism,
        low_perf_threshold=low_perf_threshold
    )

    # Entropy CI (slower, Bayesian sampling)
    entropy_lo, entropy_hi, entropy_width, entropy_diag = _ordinal_entropy_ci_adaptive(
        scores,
        ordinal_max_score=ordinal_max_score,
        cred_level=cred_level,
        conservatism=conservatism,
        low_perf_threshold=low_perf_threshold,
        model_cache=model_cache,
        compute_kwargs=compute_kwargs
    )

    entropy_median = entropy_diag['entropy_median']

    # === PATHWAY 1: Modal CI with Entropy Validation (for peaked distributions) ===
    if modal_width < delta_item:
        # ENTROPY VALIDATION GATE: Check if distribution is truly peaked
        if entropy_median > effective_entropy_threshold:
            # FALSE PEAK: Modal CI narrow but entropy high (distribution uncertain)
            diagnostics = {
                'pathway': 0,
                'modal_ci': (float(modal_lo), float(modal_hi)),
                'modal_width': float(modal_width),
                'entropy_median': float(entropy_median),
                'entropy_threshold': float(effective_entropy_threshold),
                'false_peak_detected': True,
                'message': f'Modal CI narrow ({modal_width:.3f}) but entropy high ({entropy_median:.2f} > {effective_entropy_threshold:.2f} bits)'
            }
            # logger.debug(
            #     f"False peak detected: modal_width={modal_width:.3f} < {delta_item:.3f} "
            #     f"BUT entropy={entropy_median:.2f} > {entropy_threshold} - continuing to Pathway 2"
            # )
            # Fall through to Pathway 2 (don't return here)
        else:
            # TRUE PEAK: Modal CI narrow AND entropy low (distribution peaked)
            diagnostics = {
                'pathway': 1,
                'modal_ci': (float(modal_lo), float(modal_hi)),
                'modal_width': float(modal_width),
                'entropy_median': float(entropy_median),
                'entropy_threshold': float(effective_entropy_threshold),
                'threshold': float(delta_item),
                'entropy_epochs': len(entropy_history),
                'validated': True
            }
            # logger.info(
            #     f"Stopping via Pathway 1 (Modal CI narrow + validated): "
            #     f"width={modal_width:.3f} < {delta_item:.3f}, entropy={entropy_median:.2f} < {entropy_threshold}"
            # )
            return True, 'modal_ci_narrow_validated', diagnostics

    # === PATHWAY 2: Entropy Convergence (for non-peaked or false peaks) ===
    # Check if the entropy CI is narrow enough that the distribution estimate is precise.
    # Uses absolute width on [0,1] scaled entropy axis rather than relative change,
    # because relative change under exponential convergence is constant (never crosses
    # a small threshold). Width < 0.10 means entropy known to within ±5% of max.

    # Store in history (modified in-place, retained for min_epochs guard and diagnostics)
    entropy_history.append((entropy_lo, entropy_hi, entropy_width))

    # Need sufficient history before P2 can fire (early MCMC CIs may be unreliable)
    if len(entropy_history) < min_epochs_for_stabilization:
        diagnostics = {
            'pathway': 0,
            'modal_ci': (float(modal_lo), float(modal_hi)),
            'modal_width': float(modal_width),
            'entropy_ci': (float(entropy_lo), float(entropy_hi)),
            'entropy_width': float(entropy_width),
            'entropy_median': float(entropy_diag['entropy_median']),
            'entropy_threshold': float(effective_entropy_threshold),
            'convergence_threshold': float(entropy_convergence_threshold),
            'epochs_tracked': len(entropy_history),
            'min_epochs': min_epochs_for_stabilization
        }
        return False, 'continue_insufficient_history', diagnostics

    # Normalise entropy_width from bits to [0,1] scale for threshold comparison
    # (flat function receives entropy in bits from _ordinal_entropy_ci_adaptive;
    #  the convergence threshold is calibrated for [0,1] scale)
    num_categories = ordinal_max_score + 1
    max_entropy_bits = np.log2(num_categories)
    entropy_width_scaled = entropy_width / max_entropy_bits if max_entropy_bits > 0 else entropy_width

    # Convergence criterion: entropy CI width on [0,1] scale is below threshold
    if entropy_width_scaled < entropy_convergence_threshold:
        diagnostics = {
            'pathway': 2,
            'modal_ci': (float(modal_lo), float(modal_hi)),
            'modal_width': float(modal_width),
            'entropy_ci': (float(entropy_lo), float(entropy_hi)),
            'entropy_width': float(entropy_width),
            'entropy_width_scaled': float(entropy_width_scaled),
            'entropy_median': float(entropy_diag['entropy_median']),
            'entropy_threshold': float(effective_entropy_threshold),
            'convergence_threshold': float(entropy_convergence_threshold)
        }
        return True, 'entropy_converged', diagnostics

    # Continue collecting data
    diagnostics = {
        'pathway': 0,
        'modal_ci': (float(modal_lo), float(modal_hi)),
        'modal_width': float(modal_width),
        'entropy_ci': (float(entropy_lo), float(entropy_hi)),
        'entropy_width': float(entropy_width),
        'entropy_width_scaled': float(entropy_width_scaled),
        'entropy_median': float(entropy_diag['entropy_median']),
        'entropy_threshold': float(effective_entropy_threshold),
        'convergence_threshold': float(entropy_convergence_threshold),
        'learning': True
    }
    return False, 'continue_learning', diagnostics


def _ordinal_hybrid_stopping_criterion_hierarchical(
    item_counts: np.ndarray,
    item_ns: np.ndarray,
    ordinal_max_score: int,
    delta_item: float,
    cred_level: float,
    entropy_history: list,
    entropy_threshold: float = 0.8,
    conservatism: float = 5.0,
    low_perf_threshold: float = 0.01,
    current_perf: float = 0.5,
    min_epochs_for_stabilization: int = 3,
    entropy_convergence_threshold: float = 0.10,
    model_cache: Optional[Dict[str, Any]] = None,
    sampling_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Hierarchical hybrid stopping criterion for ordinal data.

    Uses Dirichlet-Multinomial hierarchical model with partial pooling across items.
    Two pathways to stopping (same logic as flat version, but with hierarchical estimates):

    1. **Pathway 1 (Modal CI):** Stop when modal category CI is narrow AND entropy low → peaked distribution
    2. **Pathway 2 (Entropy Convergence):** Stop when entropy CI width is narrow → precise distribution estimate

    Parameters
    ----------
    item_counts : np.ndarray, shape (n_items, n_categories)
        Category counts per item
    item_ns : np.ndarray, shape (n_items,)
        Total observations per item
    ordinal_max_score : int
        Maximum possible score (e.g., 10 for 0-10 scale)
    delta_item : float
        Threshold for modal CI width (e.g., 0.15)
    cred_level : float
        Credible level for intervals (e.g., 0.95)
    entropy_history : list
        List of (entropy_lo, entropy_hi, entropy_width) tuples from previous checks
        Modified in-place to add current check
    entropy_threshold : float, default=0.8
        Proportion of maximum entropy for false peak detection (0 to 1).
        Scaled internally by log2(num_categories) to produce an effective
        threshold in bits. The group-level entropy (computed in nats) is
        converted to bits before comparison.
    conservatism : float, default=5.0
        Multiplier for CI width adjustment
    low_perf_threshold : float, default=0.01
        Performance threshold for conservatism adjustment
    current_perf : float, default=0.5
        Current performance estimate (normalized to [0,1])
    min_epochs_for_stabilization : int, default=3
        Minimum number of entropy CI checks before Pathway 2 can fire
    entropy_convergence_threshold : float, default=0.10
        Absolute entropy CI width threshold on [0,1] scale for Pathway 2 convergence.
        Width < threshold means entropy is known to within ±(threshold/2) of max.
    model_cache : dict, optional
        Cache for PyMC model reuse (must contain 'model' key with hierarchical ordinal model)
    sampling_kwargs : dict, optional
        Additional kwargs for PyMC sampling

    Returns
    -------
    should_stop : bool
        Whether to stop collecting data
    reason : str
        Stopping reason: 'modal_ci_narrow_validated_hierarchical', 'entropy_converged_hierarchical', or 'continue_*'
    diagnostics : dict
        Diagnostic information including modal CI, entropy CI, and history
    """
    from .ordinal_utils import (
        _ordinal_ci_hierarchical_modal,
        _ordinal_ci_hierarchical_entropy
    )

    # Scale entropy threshold from proportion to absolute bits
    num_categories = ordinal_max_score + 1
    max_entropy_bits = np.log2(num_categories)
    effective_entropy_threshold = entropy_threshold * max_entropy_bits

    # === COMPUTE BOTH METRICS FROM HIERARCHICAL MODEL ===
    # PERFORMANCE NOTE: Each function below calls pm.sample() independently, resulting
    # in two separate MCMC sampling runs. This doubles inference time compared to a
    # single-sample approach, but maintains modularity and code simplicity.
    # Both modal_group and entropy_group exist in the same model, so a future
    # optimization could extract both from a single sampling run if needed.

    # Modal CI (hierarchical)
    modal_lo, modal_hi, modal_width = _ordinal_ci_hierarchical_modal(
        item_counts,
        item_ns,
        ordinal_max_score=ordinal_max_score,
        cred_level=cred_level,
        conservatism=conservatism,
        low_perf_threshold=low_perf_threshold,
        current_perf=current_perf,
        model_cache=model_cache,
        sampling_kwargs=sampling_kwargs
    )

    # Entropy CI (hierarchical)
    entropy_lo, entropy_hi, entropy_width, entropy_diag = _ordinal_ci_hierarchical_entropy(
        item_counts,
        item_ns,
        ordinal_max_score=ordinal_max_score,
        cred_level=cred_level,
        conservatism=conservatism,
        low_perf_threshold=low_perf_threshold,
        current_perf=current_perf,
        model_cache=model_cache,
        sampling_kwargs=sampling_kwargs
    )

    entropy_median = entropy_diag.get('entropy_median', 0.5)
    # Use raw nats value for entropy gate, converted to bits to match sample-level scale
    # (scaled [0,1] entropy would never exceed the effective threshold, disabling the gate)
    entropy_median_nats = entropy_diag.get('entropy_median_nats')
    if entropy_median_nats is not None:
        entropy_for_gate = entropy_median_nats / np.log(2)  # nats → bits
    else:
        entropy_for_gate = entropy_median  # fallback for legacy callers
    n_items = len(item_ns)
    n_obs = int(np.sum(item_ns))

    # === PATHWAY 1: Modal CI with Entropy Validation (for peaked distributions) ===
    if modal_width < delta_item:
        # ENTROPY VALIDATION GATE: Check if distribution is truly peaked
        if entropy_for_gate > effective_entropy_threshold:
            # FALSE PEAK: Modal CI narrow but entropy high (distribution uncertain)
            diagnostics = {
                'pathway': 0,
                'inference_type': 'hierarchical',
                'modal_ci': (float(modal_lo), float(modal_hi)),
                'modal_width': float(modal_width),
                'entropy_median': float(entropy_median),
                'entropy_for_gate': float(entropy_for_gate),
                'entropy_threshold': float(effective_entropy_threshold),
                'false_peak_detected': True,
                'n_items': n_items,
                'n_obs': n_obs,
                'message': f'Modal CI narrow ({modal_width:.3f}) but entropy high ({entropy_for_gate:.2f} > {effective_entropy_threshold:.2f} bits)'
            }
            # Fall through to Pathway 2 (don't return here)
        else:
            # TRUE PEAK: Modal CI narrow AND entropy low (distribution peaked)
            diagnostics = {
                'pathway': 1,
                'inference_type': 'hierarchical',
                'modal_ci': (float(modal_lo), float(modal_hi)),
                'modal_width': float(modal_width),
                'entropy_median': float(entropy_median),
                'entropy_for_gate': float(entropy_for_gate),
                'entropy_threshold': float(effective_entropy_threshold),
                'threshold': float(delta_item),
                'entropy_epochs': len(entropy_history),
                'n_items': n_items,
                'n_obs': n_obs,
                'validated': True
            }
            return True, 'modal_ci_narrow_validated_hierarchical', diagnostics

    # === PATHWAY 2: Entropy Convergence (for non-peaked or false peaks) ===
    # Check if the entropy CI is narrow enough that the distribution estimate is precise.
    # On the [0,1] scaled entropy axis, width < 0.10 means ±5% precision.
    # Uses absolute width rather than relative change, because relative change under
    # exponential convergence is constant (never crosses a small threshold).

    # Store in history (modified in-place, retained for min_epochs guard and diagnostics)
    entropy_history.append((entropy_lo, entropy_hi, entropy_width))

    # Need sufficient history before P2 can fire (early MCMC CIs may be unreliable)
    if len(entropy_history) < min_epochs_for_stabilization:
        diagnostics = {
            'pathway': 0,
            'inference_type': 'hierarchical',
            'modal_ci': (float(modal_lo), float(modal_hi)),
            'modal_width': float(modal_width),
            'entropy_ci': (float(entropy_lo), float(entropy_hi)),
            'entropy_width': float(entropy_width),
            'entropy_median': float(entropy_median),
            'entropy_threshold': float(effective_entropy_threshold),
            'convergence_threshold': float(entropy_convergence_threshold),
            'epochs_tracked': len(entropy_history),
            'min_epochs': min_epochs_for_stabilization,
            'n_items': n_items,
            'n_obs': n_obs
        }
        return False, 'continue_insufficient_history', diagnostics

    # Convergence criterion: entropy CI width on [0,1] scale is below threshold
    # (hierarchical function receives entropy_width already on [0,1] scale from
    #  _ordinal_ci_hierarchical_entropy, so no normalisation needed here)
    if entropy_width < entropy_convergence_threshold:
        diagnostics = {
            'pathway': 2,
            'inference_type': 'hierarchical',
            'modal_ci': (float(modal_lo), float(modal_hi)),
            'modal_width': float(modal_width),
            'entropy_ci': (float(entropy_lo), float(entropy_hi)),
            'entropy_width': float(entropy_width),
            'entropy_median': float(entropy_median),
            'entropy_threshold': float(effective_entropy_threshold),
            'convergence_threshold': float(entropy_convergence_threshold),
            'n_items': n_items,
            'n_obs': n_obs
        }
        return True, 'entropy_converged_hierarchical', diagnostics

    # Continue collecting data
    diagnostics = {
        'pathway': 0,
        'inference_type': 'hierarchical',
        'modal_ci': (float(modal_lo), float(modal_hi)),
        'modal_width': float(modal_width),
        'entropy_ci': (float(entropy_lo), float(entropy_hi)),
        'entropy_width': float(entropy_width),
        'entropy_median': float(entropy_median),
        'entropy_threshold': float(effective_entropy_threshold),
        'convergence_threshold': float(entropy_convergence_threshold),
        'n_items': n_items,
        'n_obs': n_obs,
        'learning': True
    }
    return False, 'continue_learning', diagnostics


def _compute_threshold_probability(
    probs: np.ndarray,
    threshold: int
) -> np.ndarray:
    """
    Compute P(Score ≥ threshold) from category probabilities.

    This enables threshold-based stopping criteria, e.g.:
    - "Is performance proficient (≥7)?"
    - "Stop when CI on P(Score ≥ 7) is narrow"

    Parameters
    ----------
    probs : np.ndarray, shape (..., n_categories)
        Probability distributions
    threshold : int
        Score threshold (e.g., 7 for proficiency)

    Returns
    -------
    p_above : np.ndarray, shape (...)
        P(Score ≥ threshold) for each distribution

    Examples
    --------
    >>> probs = np.array([[0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0, 0, 0]])
    >>> _compute_threshold_probability(probs, threshold=7)
    array([0.02])  # P(Score ≥ 7) = 0.02
    """
    p_above = probs[..., threshold:].sum(axis=-1)
    return p_above
