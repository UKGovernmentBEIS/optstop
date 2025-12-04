# Plan: Hierarchical Ordinal Inference

## Implementation Status (Updated 2025-12-01)

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Update item_summaries format for ordinal in rule.py |
| Phase 2 | ✅ Complete | Add hierarchical PyMC model for ordinal in rule.py |
| Phase 3 | ✅ Complete | Create _ordinal_ci_hierarchical_modal in ordinal_utils.py |
| Phase 3 | ✅ Complete | Create _ordinal_ci_hierarchical_entropy in ordinal_utils.py |
| Phase 4 | ✅ Complete | Update group-level stopping logic in optimal_stopping_live_single |
| Phase 4b | ✅ Complete | Update group-level stopping in optimal_stopping_live functions |
| Phase 5 | ✅ Complete | Create _ordinal_hybrid_stopping_criterion_hierarchical |
| Phase 7 | ✅ Complete | Audit and update other ordinal code paths |

### Key Changes Made:
- **ordinal_utils.py**: Added `counts_to_scores()`, `aggregate_item_counts()`, `_ordinal_ci_hierarchical_modal()`, `_ordinal_ci_hierarchical_entropy()`
- **ordinal_model.py**: Added `_ordinal_hybrid_stopping_criterion_hierarchical()` for hierarchical hybrid stopping
- **rule.py**:
  - Added Dirichlet-Multinomial model creation for ordinal hierarchical inference
  - Updated item_summaries format to include category counts (`counts`, `n_obs`, `modal_category`, `mean_score`)
  - Updated group-level stopping to use hierarchical inference for all three ordinal modes (modal, entropy, hybrid)
  - Fixed undefined variable bugs (`ordinal_item_cache` references)

### Test Results:
- 22/22 ordinal scoring tests pass
- 49/50 ordinal model tests pass (1 pre-existing cache test failure)
- All API changes reflected in test updates

---

## Problem Statement

Currently, ordinal inference (modal/entropy/hybrid) uses **flat pooling** - all scores from all samples are combined into a single array and analyzed together. This is inconsistent with binary and continuous pathways which use **hierarchical models** that:

1. Account for item-to-item (sample-to-sample) variability
2. Pool information across items (partial pooling / shrinkage)
3. Items with few observations benefit from group-level information

## Current Architecture

### Binary Hierarchical Model
```
Group Level:
  mu_group ~ Normal(2, 1.5)           # Population mean (logit scale)
  sigma_group ~ Exponential(1.0)      # Between-item SD

Item Level:
  z_i ~ Normal(0, 1)                  # Item deviation
  mu_item_i = mu_group + z_i * sigma_group
  Theta_i = sigmoid(mu_item_i)        # Item success probability

Observation Level:
  obs_i ~ Binomial(trials_i, Theta_i) # Aggregated: successes/trials per item
```

**Key insight**: Uses aggregated sufficient statistics (successes, trials) per item, NOT individual observations.

### Continuous Hierarchical Model
```
Group Level:
  mu_group ~ Normal(0, 1.5)           # Population mean (logit scale)
  sigma_group ~ Exponential(1.0)      # Between-item SD
  phi_group ~ Gamma(2, 1.0)           # Group precision

Item Level:
  z_i ~ Normal(0, 1)                  # Item mean deviation
  mu_item_i = sigmoid(mu_group + z_i * sigma_group)
  z_phi_i ~ Normal(0, 1)              # Item precision deviation
  phi_item_i = exp(log(phi_group) + z_phi_i * 0.5)

Observation Level:
  obs_i ~ Normal(mu_item_i, SD_i)     # Aggregated: item means with known variance
  where SD_i = sqrt(mu*(1-mu)/(phi*n))
```

**Key insight**: Uses aggregated item means with theoretically-derived standard errors.

### Current Ordinal (FLAT - No Hierarchy)
```
all_scores = flatten([scores_item1, scores_item2, ...])  # Lose item structure!
modal_category = bootstrap_mode(all_scores)
CI = percentile(bootstrap_modes)
```

**Problem**: No item-level parameters, no partial pooling, ignores sample heterogeneity.

---

## Proposed Hierarchical Ordinal Structure

### Core Challenge

Ordinal data has K categories (e.g., 0-10 for K=11). We need to model:
- **Group level**: Typical distribution across categories
- **Item level**: Item-specific deviations from group distribution

### Approach: Hierarchical Dirichlet-Multinomial

This is the natural ordinal analog to Beta-Binomial (binary) and Normal-Beta (continuous).

```
Group Level:
  alpha_group ~ Dirichlet(ones(K))    # Population category probabilities (uninformative)
  kappa ~ Gamma(2, 0.1)               # Concentration (higher = less item variation)

Item Level:
  alpha_item = alpha_group * kappa    # Shared concentration for all items
  p_item_i ~ Dirichlet(alpha_item)    # Item-specific category probs (partial pooling)

Observation Level:
  counts_i ~ Multinomial(n_i, p_item_i)  # Category counts per item
```

**Sufficient Statistics Per Item**:
- `counts_i`: Vector of length K with count for each category (e.g., [0,0,1,2,5,3,1,0,0,0,0] for 12 obs)
- `n_i`: Total observations for item = sum(counts_i)

### Hierarchical Modal Inference

For modal inference, we compute the CI on the **posterior mean modal category**.

**Important Limitation**: The modal category is inherently discrete (an integer 0 to K-1). Computing a "credible interval" on a discrete quantity requires treating categories as numeric values on an ordinal scale. This is a reasonable approximation for ordinal data (where category 7 is meaningfully "between" 6 and 8) but would not be appropriate for purely nominal categorical data.

```python
# For each posterior sample s:
#   modal_s = argmax(alpha_group_s)  # Which category has highest probability
#
# Posterior mean modal category:
#   mean_modal = mean(modal_s for s in posterior_samples)
#
# HDI on modal category:
#   lo, hi = hdi(modal_s, prob=cred_level)
#
# Scale to [0, 1] for consistency with binary/continuous:
#   lo_scaled = lo / ordinal_max_score
#   hi_scaled = hi / ordinal_max_score
```

**Output**: HDI on population modal category (scaled to [0,1]), accounting for between-item variation.

### Hierarchical Entropy Inference

For entropy inference, we want the CI on **population entropy**.

```python
# Group-level entropy (computed from posterior samples of alpha_group)
# For each posterior sample s:
#   p_group_s = alpha_group_s / sum(alpha_group_s)  # Normalize to probabilities
#   H_s = -sum(p_group_s * log(p_group_s))          # Shannon entropy

# HDI on entropy:
#   lo, hi = hdi(H_s, prob=cred_level)
```

**Output**: CI on population entropy, with partial pooling across items.

### Hybrid Inference

Combines both:
1. **Pathway 1**: Modal CI narrow AND entropy below threshold → peaked distribution
2. **Pathway 2**: Entropy stabilized across epochs → stable distribution

Both pathways now use hierarchical estimates instead of flat pooled estimates.

**Entropy Stabilization with Hierarchical Inference**:
- Track `entropy_history` as list of (lo, hi, width) tuples from each inference call
- Compute relative change: `relative_change = abs(entropy_new - entropy_old) / entropy_old`
- Use posterior median entropy for stabilization comparisons
- Stabilization threshold unchanged (default 0.002 = 0.2%)

---

## Implementation Plan

### Phase 1: Data Structure Changes

**File: `rule.py`**

1. Change `item_summaries` for ordinal to store category counts:
```python
# Current (loses categorical structure):
item_summaries.append({
    'successes': int(np.sum(accumulated_scores)),
    'trials': len(accumulated_scores)
})

# Proposed (preserves categorical structure):
counts = np.bincount(accumulated_scores.astype(int), minlength=ordinal_max_score + 1)
item_summaries.append({
    'counts': counts,              # Vector of length K
    'n_obs': len(accumulated_scores),
    'modal_category': int(np.argmax(counts)),
    'mean_score': float(np.mean(accumulated_scores)),
    # Keep successes/trials for backward compatibility with perf estimate
    'successes': int(np.sum(accumulated_scores)),
    'trials': len(accumulated_scores)
})
```

### Phase 2: Hierarchical PyMC Model

**File: `rule.py` (new model section)**

```python
elif score_type == 'ordinal':
    # === ORDINAL HIERARCHICAL MODEL ===
    # Dirichlet-Multinomial hierarchy (analogous to Beta-Binomial for binary)
    #
    # Model Structure:
    #   - Group level: alpha_group (category probability direction), kappa (concentration)
    #   - Item level: p_item_i ~ Dirichlet(alpha_group * kappa) with partial pooling
    #   - Observation level: Multinomial likelihood on category counts
    #
    # Key Design Choices:
    #   - Uninformative Dirichlet(1,...,1) prior on alpha_group
    #   - Gamma(2, 0.1) prior on kappa (mean=20, moderate pooling)
    #   - Categories treated as nominal (no ordinal structure in likelihood)
    #     This is a simplification; ordered logistic would be more appropriate
    #     but adds significant complexity. Document as known limitation.

    n_categories = ordinal_max_score + 1
    current_n_items = len(item_ids)

    # Cache invalidation: check both n_items AND n_categories
    if 'model' in ordinal_group_cache:
        cached_n_items = ordinal_group_cache.get('n_items_last', 0)
        cached_n_categories = ordinal_group_cache.get('n_categories_last', 0)
        if cached_n_items != current_n_items or cached_n_categories != n_categories:
            # Must recreate - dimensions changed
            ordinal_group_cache.clear()

    if 'model' not in ordinal_group_cache:
        with pm.Model() as ordinal_model:
            # Group-level: baseline category probabilities
            # Uninformative symmetric Dirichlet prior
            alpha_prior = np.ones(n_categories)
            alpha_group = pm.Dirichlet("alpha_group", a=alpha_prior)

            # Concentration parameter: higher = items more similar to group
            # Gamma(2, 0.1) gives mean=20, reasonable pooling strength
            kappa = pm.Gamma("kappa", alpha=2, beta=0.1)

            # Mutable data containers
            n_items_data = pm.Data("n_items", np.array(current_n_items, dtype="int64"))
            item_counts_data = pm.Data("item_counts", np.zeros((current_n_items, n_categories), dtype="int64"))
            item_ns_data = pm.Data("item_ns", np.ones(current_n_items, dtype="int64"))

            # Item-level concentrations (shared across all items, broadcast)
            # alpha_item has shape (n_categories,) and broadcasts to (n_items, n_categories)
            alpha_item = alpha_group * kappa

            # Item-specific probabilities with partial pooling
            # Each item draws from Dirichlet with same concentration parameters
            # Shape: (n_items, n_categories)
            p_item = pm.Dirichlet("p_item", a=alpha_item, shape=(n_items_data, n_categories))

            # Multinomial likelihood on category counts
            # n parameter is the known total count per item (passed as separate data)
            obs = pm.Multinomial("obs", n=item_ns_data, p=p_item, observed=item_counts_data)

            # Derived quantities for stopping criteria
            # Group modal category: argmax of alpha_group for each posterior sample
            # Note: This is discrete, but we compute mean/HDI treating it as numeric
            modal_group = pm.Deterministic("modal_group", pm.math.argmax(alpha_group))

            # Group entropy: -sum(p * log(p)) where p = normalized alpha_group
            p_group_normalized = alpha_group / pm.math.sum(alpha_group)
            entropy_group = pm.Deterministic(
                "entropy_group",
                -pm.math.sum(p_group_normalized * pm.math.log(p_group_normalized + 1e-10))
            )

        ordinal_group_cache['model'] = ordinal_model
        ordinal_group_cache['n_items_last'] = current_n_items
        ordinal_group_cache['n_categories_last'] = n_categories
```

### Phase 3: Update Inference Functions

**File: `ordinal_utils.py`**

Create new hierarchical versions:

```python
def _ordinal_ci_hierarchical_modal(
    item_counts: np.ndarray,        # Shape: (n_items, n_categories)
    item_ns: np.ndarray,            # Shape: (n_items,) - total obs per item
    ordinal_max_score: int,
    cred_level: float = 0.95,
    conservatism: float = 1.0,
    low_perf_threshold: float = 0.2,
    current_perf: float = 0.5,
    model_cache: Optional[dict] = None,
    sampling_kwargs: Optional[dict] = None
) -> Tuple[float, float, float]:
    """
    Hierarchical Bayesian CI for population modal category.

    Uses Dirichlet-Multinomial model with partial pooling across items.

    IMPORTANT: Modal category is discrete (integer). We compute the posterior
    mean and HDI treating categories as numeric values. This is appropriate
    for ordinal data where categories have meaningful numeric ordering, but
    would NOT be appropriate for nominal categorical data.

    Args:
        item_counts: Category counts per item, shape (n_items, n_categories)
        item_ns: Total observations per item, shape (n_items,)
        ordinal_max_score: Maximum category value (K-1 where K is n_categories)
        cred_level: Credibility level for HDI (default 0.95)
        conservatism: Multiplier for CI width in low-performance scenarios
        low_perf_threshold: Performance threshold for conservatism
        current_perf: Current performance estimate (for conservatism check)
        model_cache: Dict for caching PyMC model
        sampling_kwargs: MCMC sampling parameters

    Returns:
        Tuple of (lower_bound, upper_bound, effective_width), all scaled to [0,1]
    """
    # Implementation uses PyMC model, extracts modal_group posterior,
    # computes HDI, scales to [0,1], applies conservatism
    pass


def _ordinal_ci_hierarchical_entropy(
    item_counts: np.ndarray,        # Shape: (n_items, n_categories)
    item_ns: np.ndarray,            # Shape: (n_items,) - total obs per item
    ordinal_max_score: int,
    cred_level: float = 0.95,
    conservatism: float = 1.0,
    low_perf_threshold: float = 0.2,
    current_perf: float = 0.5,
    model_cache: Optional[dict] = None,
    sampling_kwargs: Optional[dict] = None
) -> Tuple[float, float, float, dict]:
    """
    Hierarchical Bayesian CI for population entropy.

    Uses Dirichlet-Multinomial model with partial pooling across items.
    Returns CI on entropy and diagnostics for stabilization tracking.

    Args:
        item_counts: Category counts per item, shape (n_items, n_categories)
        item_ns: Total observations per item, shape (n_items,)
        ordinal_max_score: Maximum category value
        cred_level: Credibility level for HDI
        conservatism: Multiplier for CI width in low-performance scenarios
        low_perf_threshold: Performance threshold for conservatism
        current_perf: Current performance estimate
        model_cache: Dict for caching PyMC model
        sampling_kwargs: MCMC sampling parameters

    Returns:
        Tuple of (lower_bound, upper_bound, effective_width, diagnostics)
        - Bounds and width are entropy values (not scaled)
        - diagnostics contains 'entropy_median', 'entropy_samples' for stabilization
    """
    # Implementation uses PyMC model, extracts entropy_group posterior,
    # computes HDI, applies conservatism, returns diagnostics
    pass
```

### Phase 4: Update Group-Level Stopping Logic

**File: `rule.py`**

```python
elif score_type == 'ordinal':
    # Aggregate item counts for hierarchical model
    item_counts_matrix = np.array([s['counts'] for s in item_summaries])
    item_ns = np.array([s['n_obs'] for s in item_summaries])

    # Compute conservatism based on performance
    current_perf_normalized = current_perf_estimate / ordinal_max_score
    current_conservatism = conservatism if current_perf_normalized < low_perf_threshold else 1.0

    if ordinal_inference == 'modal':
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

        # Track CI width for stabilization history
        stabilization_history['ci_width_history'].append(float(width))
        stabilization_history['final_modal_ci_width'] = float(width)
        stabilization_history['final_modal_ci'] = [float(lo), float(hi)]
        stabilization_history['ordinal_pathway'] = 'modal'

        # Stopping criterion unchanged
        if width < delta_cap:
            stop_this_grouping.append(grouping_name)
            # ... stopping metadata

    elif ordinal_inference == 'entropy':
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

        # Track for stabilization
        stabilization_history['ci_width_history'].append(float(width))
        stabilization_history['final_entropy_ci_width'] = float(width)
        stabilization_history['final_entropy'] = float(diagnostics.get('entropy_median', 0))
        stabilization_history['ordinal_pathway'] = 'entropy'

        # Update entropy history for stabilization tracking
        group_entropy_history.append((lo, hi, width))

        if width < delta_cap:
            stop_this_grouping.append(grouping_name)
            # ... stopping metadata

    elif ordinal_inference == 'hybrid':
        # Hybrid uses both modal and entropy from hierarchical model
        # Implementation details in _ordinal_hybrid_stopping_criterion_hierarchical()
        # ...
```

### Phase 5: Update Hybrid Stopping Criterion

**File: `ordinal_model.py`**

Create hierarchical version of hybrid stopping:

```python
def _ordinal_hybrid_stopping_criterion_hierarchical(
    item_counts: np.ndarray,
    item_ns: np.ndarray,
    ordinal_max_score: int,
    delta_item: float,
    cred_level: float,
    entropy_history: list,
    entropy_threshold: float = 1.5,
    conservatism: float = 1.0,
    low_perf_threshold: float = 0.2,
    current_perf: float = 0.5,
    min_epochs_for_stabilization: int = 3,
    stabilization_threshold: float = 0.002,
    model_cache: Optional[Dict] = None,
    sampling_kwargs: Optional[Dict] = None
) -> Tuple[bool, str, Dict]:
    """
    Hierarchical hybrid stopping criterion for ordinal data.

    Two pathways (same logic as flat, but using hierarchical estimates):
    1. Pathway 1 (Modal CI): Stop when modal CI narrow AND entropy low
    2. Pathway 2 (Entropy Stabilization): Stop when entropy converged
    """
    # Run single hierarchical inference to get both modal and entropy posteriors
    # Check Pathway 1: modal_width < delta_item AND entropy < entropy_threshold
    # Check Pathway 2: relative entropy change < stabilization_threshold
    pass
```

### Phase 6: Backward Compatibility

Keep sample-level inference unchanged (fast bootstrap/entropy methods). Only group-level uses hierarchical model.

This matches binary/continuous: sample-level is fast (Beta sampling), group-level is hierarchical (MCMC).

### Phase 7: Update Other Ordinal Code Paths

**Files to audit and update:**

1. `rule.py: optimal_stopping_live()` - Original function, ensure consistent with `optimal_stopping_live_single()`
2. `rule.py: optimal_stopping()` - Batch function, ensure ordinal handling matches
3. `early_stopping.py` - Uses `optimal_stopping_live_single()`, should work automatically

**Changes needed:**
- Update `item_summaries` format in all ordinal code paths
- Ensure `ordinal_group_cache` is properly passed/initialized
- Verify stabilization history fields are consistent

---

## Conservatism Handling

Apply conservatism multiplier consistently with binary/continuous:

```python
# In hierarchical modal/entropy functions:
effective_width = width * conservatism if current_perf < low_perf_threshold else width

# Return effective_width for stopping comparison
return lo, hi, effective_width
```

This ensures low-performing ordinal groupings require more stringent CI width before stopping.

---

## Sample-Size Floor Consideration

The flat modal inference uses a sample-size-scaled floor to prevent premature stopping:
```python
min_ci_width = 1.0 / (ordinal_max_score * np.sqrt(n_samples))
```

**For hierarchical inference**: The Bayesian posterior naturally incorporates uncertainty based on sample size. With few observations, the posterior will be wider. Therefore, an explicit floor may not be necessary.

**Recommendation**: Do not apply explicit floor to hierarchical inference. The posterior uncertainty should handle this naturally. Monitor in testing and add floor if needed.

---

## Performance Considerations

### Speed Impact

| Current (Flat) | Proposed (Hierarchical) |
|----------------|------------------------|
| Bootstrap: ~0.1s | MCMC: ~5-30s |

This matches binary/continuous group-level inference timing.

### Mitigation Strategies

1. **Model caching**: Reuse compiled PyMC model (already done for binary/continuous)
2. **Aggregated data**: Use item counts (K values per item) not individual observations
3. **Reduced draws**: Use `draws=500, tune=500` for production (configurable)
4. **Parallel chains**: Leverage GPU if available
5. **Cache invalidation**: Track both `n_items` and `n_categories` to properly invalidate

---

## Model Caching Behavior

### Cache Structure

The ordinal hierarchical model uses two cache dictionaries:

1. **`ordinal_group_cache`** - Group-level Dirichlet-Multinomial model
   - Keys: `model`, `n_items_last`, `n_categories_last`
   - Used by: `_ordinal_ci_hierarchical_modal()`, `_ordinal_ci_hierarchical_entropy()`

2. **`ordinal_item_cache`** - Item-level OrderedLogistic model (for non-hierarchical entropy)
   - Used by: Sample-level stopping decisions

### Cache Invalidation Rules

The cache is invalidated (cleared) when model dimensions change:

```python
if 'model' in ordinal_group_cache:
    cached_n_items = ordinal_group_cache.get('n_items_last', 0)
    cached_n_categories = ordinal_group_cache.get('n_categories_last', 0)
    if cached_n_items != current_n_items or cached_n_categories != n_categories:
        ordinal_group_cache.clear()  # Must recreate - dimensions changed
```

**Why dimension changes require recreation**:
- PyMC models have fixed tensor shapes at compilation
- Changing `n_items` changes the `p_item` tensor shape
- Changing `n_categories` changes the Dirichlet dimensionality

### Cache Reuse (Safe)

When dimensions match, the cached model is reused with new data:

```python
pm.set_data({
    "n_items": np.int64(n_items),
    "item_counts": item_counts.astype("int64"),
    "item_ns": item_ns.astype("int64")
})
```

This is safe because:
- `pm.Data` containers are designed for updating
- MCMC sampling generates new posterior samples from fresh data
- Model structure (priors, transformations) remains unchanged

### Important Limitation: Data Values Not Tracked

The cache only tracks **structural dimensions** (`n_items`, `n_categories`), NOT the actual data values. This means:

- Same dimensions with different data → Cache HIT (correct behavior)
- The model is re-sampled with new data, producing valid posteriors

This design choice is intentional:
- Model compilation is expensive (~1-5 seconds)
- MCMC sampling is relatively fast with pre-compiled model
- Data values don't affect model structure

### Cross-Grouping Cache Behavior

In `optimal_stopping_live_single`:
- Each grouping maintains its own `ordinal_group_cache` instance
- Caches persist across calls for the same grouping within a session
- Caches are returned via `model_caches_out` for external management

In `optimal_stopping_live`:
- Parallelizes across groupings (each in separate process/thread)
- Each parallel worker has independent cache
- Caches are NOT shared between groupings (by design)

---

## Known Limitations

### 1. Ordinal vs Nominal Treatment

The Dirichlet-Multinomial model treats categories as **exchangeable** (nominal). It does not use the fact that category 7 is "between" 6 and 8 in the likelihood.

For truly ordinal data, an **Ordered Logistic** (Cumulative Link) model would be more appropriate:
```
P(Y ≤ k) = logistic(cutpoint_k - mu)
```

However:
- Ordered Logistic is significantly more complex to implement
- The current flat bootstrap also ignores ordinality
- For modal/entropy inference, the nominal assumption is reasonable

**Future Enhancement**: Consider implementing Ordered Logistic for applications where ordinal structure is critical.

### 2. Discrete Modal Category

The modal category is inherently discrete. We compute posterior mean and HDI treating categories as numeric values. This is:
- **Appropriate** for ordinal data with meaningful numeric ordering
- **Inappropriate** for nominal categorical data

The scaling to [0,1] (dividing by `ordinal_max_score`) allows comparison with `delta_cap` threshold, matching binary/continuous behavior.

### 3. Small Sample Behavior

With very few observations per item (e.g., 1-2 epochs), individual item posteriors will be dominated by the prior. The hierarchical model handles this through partial pooling - sparse items borrow strength from data-rich items.

### 4. Semantic Differences Between Inference Types

The stopping criterion `delta_cap` has **different semantic meanings** depending on inference type:

| Inference Type | CI Target | delta_cap = 0.10 means... |
|---------------|-----------|---------------------------|
| Binary | Population proportion | "95% CI width on success rate ≤ 10 percentage points" |
| Ordinal (modal) | Modal category (scaled 0-1) | "95% CI width on modal category ≤ 1 category" (for 10-scale) |
| Ordinal (entropy) | Entropy (scaled 0-1) | "95% CI width on distribution entropy ≤ 10% of max entropy" |

**Implications for users:**

1. **Modal inference**: A `delta_cap = 0.10` means the modal category is localized to within 1 category on a 10-point scale. This is relatively strict - requiring strong agreement on which category is most common.

2. **Entropy inference**: A `delta_cap = 0.10` means entropy uncertainty is within 10% of maximum possible entropy. Low entropy = peaked distribution, high entropy = diffuse.

3. **Comparing thresholds**: The same `delta_cap` value is NOT directly comparable between binary and ordinal. Users should calibrate thresholds separately for each type.

**Recommendation**: When using mixed binary/ordinal datasets:
- Use `delta_cap` for overall stopping
- Consider that ordinal modal inference typically produces wider CIs than binary proportion inference
- Entropy inference provides a more comparable metric across score types
- Test threshold sensitivity with your specific data before production use

---

## Testing Plan

1. **Unit tests**: Verify hierarchical model produces valid posteriors
2. **Comparison tests**: Compare flat vs hierarchical on synthetic data with known heterogeneity
3. **Shrinkage tests**: Verify partial pooling behavior (sparse items shrink toward group)
4. **Integration tests**: Full pipeline with ordinal datasets
5. **Performance tests**: Measure timing impact
6. **Conservatism tests**: Verify low-perf conservatism applied correctly

---

## Summary of Changes

| File | Changes |
|------|---------|
| `rule.py` | Add ordinal hierarchical model; Update item_summaries format; Update group-level inference; Cache invalidation for n_categories |
| `ordinal_utils.py` | Add `_ordinal_ci_hierarchical_modal()` and `_ordinal_ci_hierarchical_entropy()` |
| `ordinal_model.py` | Add `_ordinal_hybrid_stopping_criterion_hierarchical()` |
| `early_stopping.py` | No direct changes (uses rule.py functions) |

---

## Open Questions (Resolved)

1. ~~**Should sample-level also be hierarchical?**~~ No - keep fast bootstrap for sample-level, matches binary/continuous pattern.

2. ~~**What about hybrid mode?**~~ Both pathways use hierarchical estimates via new `_ordinal_hybrid_stopping_criterion_hierarchical()`.

3. ~~**Concentration parameter prior?**~~ Keep uninformative `Dirichlet(1,...,1)` for alpha_group; `Gamma(2, 0.1)` for kappa provides moderate pooling.

4. ~~**Modal category CI interpretation?**~~ Compute posterior mean/HDI treating categories as numeric; document limitation for nominal data.
