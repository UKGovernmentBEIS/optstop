# Plan: Ordered Logistic Model for Ordinal Inference

## Status: PLANNED (Not Yet Implemented)

**Current Implementation**: Dirichlet-Multinomial (interim solution)
**Target Implementation**: Ordered Logistic (Cumulative Link) Model

---

## Feasibility Assessment

**Overall Verdict: FEASIBLE with significant caveats**

| Aspect | Rating | Notes |
|--------|--------|-------|
| Technical Feasibility | HIGH | PyMC has native `OrderedLogistic` support |
| Implementation Complexity | MODERATE-HIGH | Identification issues, aggregated data handling |
| Performance Impact | MODERATE | ~2-3x slower than Dirichlet-Multinomial |
| Risk | MODERATE | More assumptions, less robust to misspecification |

### Critical Issues Identified

1. **Identification Problem** (CRITICAL) - Must constrain cutpoints or latent mean
2. **Cutpoint Prior Sensitivity** - Prior must scale with number of categories
3. **Edge Category Handling** - Sparse tails cause unstable cutpoint estimates
4. **Aggregated Data Complexity** - Custom likelihood needed for count data

See "Critical Issues and Solutions" section below for detailed analysis.

---

## Problem Statement

The current Dirichlet-Multinomial model treats ordinal categories as **exchangeable/nominal**. It does not leverage the fact that category 7 is "close to" category 8. This means:

1. Neighboring categories don't shrink together during partial pooling
2. The model treats P(score=2) and P(score=9) as equally "different" from P(score=5)
3. For strongly ordinal data (e.g., Likert scales), this is suboptimal

An **Ordered Logistic** (Cumulative Link) model properly encodes ordinal structure.

---

## Current Implementation (Dirichlet-Multinomial)

```
Group Level:
  alpha_group ~ Dirichlet(1,...,1)    # Category probabilities (symmetric prior)
  kappa ~ Gamma(2, 0.1)               # Concentration parameter

Item Level:
  p_item_i ~ Dirichlet(alpha_group * kappa)  # Partial pooling

Observation Level:
  counts_i ~ Multinomial(n_i, p_item_i)
```

**Pros**:
- Simple to implement
- Conjugate structure (efficient sampling)
- Works well for multinomial data

**Cons**:
- Ignores ordinal structure
- Categories treated as exchangeable
- No concept of "distance" between categories

---

## Proposed Implementation (Ordered Logistic)

### Model Structure

```
Group Level:
  mu_group ~ Normal(0, 2)             # Population latent mean
  sigma_group ~ Exponential(1)        # Between-item SD
  cutpoints ~ transformed Normal      # K-1 ordered cutpoints

Item Level:
  z_i ~ Normal(0, 1)                  # Item deviation (non-centered)
  mu_item_i = mu_group + z_i * sigma_group

Observation Level:
  P(Y_i <= k) = logistic(cutpoint_k - mu_item_i)
  Y_ij ~ OrderedLogistic(mu_item_i, cutpoints)
```

### Key Components

1. **Latent Continuous Scale**: Each item has a latent "quality" on a continuous scale
2. **Cutpoints**: K-1 ordered thresholds that map latent scale to categories
3. **Cumulative Probabilities**: P(Y <= k) is modeled, ensuring monotonicity
4. **Ordinal Structure**: Adjacent categories naturally relate through the latent scale

### PyMC Implementation Sketch

```python
with pm.Model() as ordered_logistic_model:
    # Group-level parameters
    mu_group = pm.Normal("mu_group", mu=0, sigma=2)
    sigma_group = pm.Exponential("sigma_group", lam=1.0)

    # Cutpoints (K-1 ordered values)
    # Use ordered transform to ensure cutpoint[k] < cutpoint[k+1]
    cutpoints = pm.Normal(
        "cutpoints",
        mu=np.linspace(-2, 2, n_categories - 1),
        sigma=1.5,
        shape=n_categories - 1,
        transform=pm.distributions.transforms.ordered
    )

    # Item-level (non-centered parameterization)
    z = pm.Normal("z", mu=0, sigma=1, shape=n_items)
    mu_item = pm.Deterministic("mu_item", mu_group + z * sigma_group)

    # Likelihood
    # For aggregated counts, use custom likelihood
    # For individual observations, use OrderedLogistic directly
    obs = pm.OrderedLogistic(
        "obs",
        eta=mu_item[item_idx],  # Latent value for each observation
        cutpoints=cutpoints,
        observed=scores
    )
```

### Handling Aggregated Data

The current implementation uses aggregated category counts per item. For ordered logistic:

**Option A**: Expand to individual observations
```python
# If item i has counts [0, 2, 5, 3, 0, ...] for categories [0, 1, 2, 3, 4, ...]
# Expand to observations: [1, 1, 2, 2, 2, 2, 2, 3, 3, 3]
```

**Option B**: Custom likelihood on counts
```python
# Compute category probabilities from cumulative model
# Apply multinomial likelihood on counts
def ordered_logistic_probs(eta, cutpoints):
    cumprobs = pm.math.sigmoid(cutpoints - eta)
    probs = pm.math.concatenate([
        cumprobs[..., :1],
        cumprobs[..., 1:] - cumprobs[..., :-1],
        1 - cumprobs[..., -1:]
    ], axis=-1)
    return probs
```

---

## Implementation Plan

### Phase 1: Core Model (Estimated: 2-3 hours)

1. Create `_create_ordered_logistic_model()` in `ordinal_model.py`
2. Implement Option B (custom likelihood on counts) for efficiency
3. Add derived quantities for stopping criteria:
   - `modal_group`: Category with highest probability at group level
   - `entropy_group`: Entropy of group-level category distribution

### Phase 2: Integration (Estimated: 2-3 hours)

1. Update `rule.py` to use ordered logistic model for ordinal
2. Modify cache invalidation logic for new model structure
3. Update `_ordinal_ci_hierarchical_modal()` to extract modal from new model
4. Update `_ordinal_ci_hierarchical_entropy()` for new entropy computation

### Phase 3: Testing (Estimated: 2-3 hours)

1. Unit tests for ordered logistic model creation
2. Calibration tests: verify CI coverage with known parameters
3. Shrinkage tests: verify partial pooling behavior
4. Comparison tests: ordered logistic vs Dirichlet-Multinomial on test data
5. Performance benchmarks

### Phase 4: Documentation (Estimated: 1 hour)

1. Update PLAN_hierarchical_ordinal.md
2. Update docstrings in ordinal_utils.py
3. Add examples to documentation

---

## Considerations

### Pros of Ordered Logistic

1. **Properly ordinal**: Encodes category ordering in model structure
2. **Interpretable**: Latent scale has natural interpretation
3. **Shrinkage**: Items shrink toward group mean on latent scale
4. **Flexible**: Can handle unequal category spacing via cutpoints

### Cons / Challenges

1. **More complex**: Additional parameters (cutpoints)
2. **Identification**: Cutpoints and latent scale can have identification issues
3. **Computation**: May be slower than Dirichlet-Multinomial
4. **Aggregated data**: Need custom likelihood for count data

---

## Critical Issues and Solutions

### Issue 1: Identification Problem (CRITICAL)

**Problem**: In ordered logistic models, the latent scale and cutpoints are not separately identifiable. You cannot freely estimate both `mu_group` AND all K-1 cutpoints - the model is overparameterized.

**Why it matters**: Without constraints, MCMC will explore equivalent parameterizations, causing:
- Poor mixing / slow convergence
- Uninterpretable posteriors
- Potential divergences

**Solution Options**:

**Option A: Fix first cutpoint at 0** (RECOMMENDED)
```python
# Only estimate K-2 free cutpoints, first is fixed at 0
cutpoints_free = pm.Normal("cutpoints_free", mu=0, sigma=1.5, shape=n_categories - 2)
# Build ordered cutpoints with first fixed at 0
cutpoints = pm.Deterministic("cutpoints",
    pt.concatenate([[0.0], pt.cumsum(pt.softplus(cutpoints_free))]))
```

**Option B: Fix latent mean at 0**
```python
mu_group = 0.0  # Fixed, not estimated
# Estimate all K-1 cutpoints freely (with ordering constraint)
cutpoints = pm.Normal("cutpoints", mu=np.linspace(-2, 2, K-1), sigma=1.5,
                      transform=pm.distributions.transforms.ordered)
```

**Recommendation**: Option A is more interpretable - `mu_group` then represents "typical latent quality" with cutpoints anchored at 0.

### Issue 2: Cutpoint Prior Sensitivity

**Problem**: The proposed prior `Normal(linspace(-2, 2, K-1), 1.5)` doesn't scale appropriately:
- For K=5: cutpoints spread over [-2, 2] with spacing ~1.0 ✓
- For K=11: cutpoints spread over [-2, 2] with spacing ~0.4 - may be too tight
- For K=3: cutpoints spread over [-2, 2] with spacing ~2.0 - may be too wide

**Solution**: Scale prior based on K
```python
# Adaptive prior: spread cutpoints proportionally to K
spread = 2.0 * np.log(n_categories)  # Increases with K
cutpoint_means = np.linspace(-spread/2, spread/2, n_categories - 1)
cutpoint_sigma = spread / (n_categories - 1)  # Spacing-appropriate SD
```

### Issue 3: Edge Category Handling

**Problem**: When extreme categories (0 or K-1) have zero or very few observations:
- Cutpoints for those boundaries are poorly identified
- Posterior may be diffuse or multi-modal
- Can cause sampling issues

**Solutions**:

1. **Stronger priors on edge cutpoints**:
```python
# Tighter prior on first and last cutpoints
cutpoint_sigmas = np.ones(K-1) * 1.5
cutpoint_sigmas[0] = 0.5   # Tighter for edge
cutpoint_sigmas[-1] = 0.5  # Tighter for edge
```

2. **Category collapsing**: If category has <5 observations, merge with neighbor
```python
# Preprocessing step
if counts[0] < min_count:
    counts[1] += counts[0]
    counts = counts[1:]
    n_categories -= 1
```

3. **Robust fallback**: If sampling fails, fall back to Dirichlet-Multinomial
```python
try:
    trace = pm.sample(...)
except (pm.SamplingError, ValueError):
    logger.warning("Ordered logistic failed, falling back to Dirichlet")
    return _ordinal_ci_hierarchical_modal_dirichlet(...)
```

### Issue 4: Aggregated Data Handling

**Problem**: Current implementation uses aggregated counts per item. PyMC's `OrderedLogistic` expects individual observations.

**Solution**: Custom likelihood using multinomial on derived probabilities

```python
def ordered_logistic_probs(eta, cutpoints):
    """Convert latent value + cutpoints to category probabilities."""
    # Cumulative probabilities: P(Y <= k) = sigmoid(c_k - eta)
    cumprobs = pm.math.sigmoid(cutpoints - eta[:, None])  # Shape: (n_items, K-1)

    # Category probabilities: P(Y = k) = P(Y <= k) - P(Y <= k-1)
    # P(Y = 0) = P(Y <= 0)
    # P(Y = k) = P(Y <= k) - P(Y <= k-1) for k in 1..K-2
    # P(Y = K-1) = 1 - P(Y <= K-2)

    p_first = cumprobs[:, :1]  # P(Y = 0)
    p_middle = cumprobs[:, 1:] - cumprobs[:, :-1]  # P(Y = k) for k in 1..K-2
    p_last = 1 - cumprobs[:, -1:]  # P(Y = K-1)

    probs = pm.math.concatenate([p_first, p_middle, p_last], axis=1)
    return probs

# In model:
probs = ordered_logistic_probs(mu_item, cutpoints)
obs = pm.Multinomial("obs", n=item_ns, p=probs, observed=item_counts)
```

---

## Revised Implementation Plan

### Phase 0: Design and Prototyping (NEW - 1-2 hours)

1. **Resolve identification**: Implement Option A (fix first cutpoint)
2. **Design adaptive priors**: Scale with K
3. **Prototype in notebook**: Test on synthetic data before integration
4. **Benchmark**: Compare sampling time vs Dirichlet-Multinomial

### Phase 1: Core Model (2-3 hours)

1. Create `_create_ordered_logistic_model()` in `ordinal_model.py`
2. Implement custom likelihood for aggregated counts
3. Add identification constraint (fixed first cutpoint)
4. Add derived quantities:
   - `modal_group`: Category with highest probability
   - `entropy_group`: Entropy of category distribution

### Phase 2: Integration (2-3 hours)

1. Add `ordinal_model_type` parameter: `'dirichlet'` (default), `'ordered_logistic'`
2. Update `rule.py` model creation to dispatch based on type
3. Add fallback logic: if ordered logistic fails → Dirichlet
4. Update cache invalidation for new model structure

### Phase 3: Testing (3-4 hours) - EXPANDED

1. **Unit tests**: Model creation, probability computation
2. **Identification tests**: Verify posteriors are proper (not diffuse)
3. **Edge case tests**: Sparse categories, small samples
4. **Calibration tests**: CI coverage with known parameters
5. **Shrinkage tests**: Neighboring category pooling behavior
6. **Comparison tests**: Ordered logistic vs Dirichlet-Multinomial
7. **Fallback tests**: Verify graceful degradation

### Phase 4: Documentation (1 hour)

1. Update PLAN_hierarchical_ordinal.md
2. Document model selection guidance
3. Add examples showing when to use each model

---

## Model Selection Guidance

### When to Use Dirichlet-Multinomial (Default)

| Scenario | Recommendation |
|----------|----------------|
| K < 5 categories | Use Dirichlet (not enough structure to benefit from ordinal) |
| Small sample size (n < 50 total) | Use Dirichlet (more stable) |
| Weakly ordinal data | Use Dirichlet (categories may not be truly ordered) |
| Performance-critical | Use Dirichlet (~2-3x faster) |
| Multi-modal distributions | Use Dirichlet (ordered logistic assumes unimodal latent) |
| Sparse edge categories | Use Dirichlet (avoids cutpoint identification issues) |

### When to Use Ordered Logistic

| Scenario | Recommendation |
|----------|----------------|
| K ≥ 5 categories | Consider ordered logistic |
| Clearly ordinal data (Likert scales) | Use ordered logistic |
| Sufficient data (≥10 obs per category) | Use ordered logistic |
| Neighboring category shrinkage desired | Use ordered logistic |
| Unimodal response distribution | Use ordered logistic |

### API Design

```python
# New parameter for optimal_stopping_live_single
ordinal_model_type: str = 'dirichlet'  # Options: 'dirichlet', 'ordered_logistic', 'auto'
```

**Auto-selection logic** (if implemented):
```python
def select_ordinal_model(item_counts, n_categories):
    total_obs = item_counts.sum()
    min_category_obs = item_counts.sum(axis=0).min()

    # Use Dirichlet for small K
    if n_categories < 5:
        return 'dirichlet'

    # Use Dirichlet for small samples
    if total_obs < 50:
        return 'dirichlet'

    # Use Dirichlet if edge categories are sparse
    if min_category_obs < 5:
        return 'dirichlet'

    # Otherwise, ordered logistic is appropriate
    return 'ordered_logistic'
```

---

## References

1. Agresti, A. (2010). Analysis of Ordinal Categorical Data, 2nd ed. Wiley.
2. Bürkner, P. C., & Vuorre, M. (2019). Ordinal regression models in psychology. Advances in Methods and Practices in Psychological Science.
3. Liddell, T. M., & Kruschke, J. K. (2018). Analyzing ordinal data with metric models. Journal of Experimental Social Psychology.

---

## Revised Timeline

| Phase | Status | Estimated Effort | Risk |
|-------|--------|------------------|------|
| Phase 0: Design & Prototyping | Not Started | 1-2 hours | LOW |
| Phase 1: Core Model | Not Started | 2-3 hours | MODERATE |
| Phase 2: Integration | Not Started | 2-3 hours | MODERATE |
| Phase 3: Testing (Expanded) | Not Started | 3-4 hours | LOW |
| Phase 4: Documentation | Not Started | 1 hour | LOW |
| **Total** | | **9-13 hours** | |

### Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Identification issues cause sampling failures | Prototype in Phase 0, fallback to Dirichlet |
| Performance regression | Benchmark in Phase 0, make opt-in only |
| Edge cases break model | Expanded testing in Phase 3, fallback logic |
| API complexity | Keep Dirichlet as default, ordered logistic opt-in |

---

## Interim Documentation

Until the ordered logistic model is implemented, the Dirichlet-Multinomial serves as a working solution with the following documented limitations:

1. Categories treated as exchangeable (nominal, not ordinal)
2. No shrinkage of neighboring categories
3. Modal category CI treats discrete values as continuous

These limitations are acceptable for many use cases, particularly:
- When the goal is simply to detect convergence of the distribution
- When categories are weakly ordinal
- When stopping decisions don't require fine-grained ordinal precision
