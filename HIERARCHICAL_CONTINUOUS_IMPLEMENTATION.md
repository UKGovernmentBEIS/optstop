# Hierarchical Continuous Scoring Implementation

**Date**: 2025-11-21
**Status**: ✅ **IMPLEMENTED**
**Priority**: CRITICAL (Consistency with Binary Methodology)

---

## Executive Summary

Implemented hierarchical Bayesian pooling for continuous bounded scores to maintain consistency with the binary scoring approach. This addresses a critical architectural inconsistency where continuous scores were using simple pooling while binary scores used proper hierarchical modeling.

**Key Change**: Continuous group-level inference now uses PyMC hierarchical Beta model with item-level random effects, mirroring the binary hierarchical Binomial model structure.

---

## Problem Statement

### Original Implementation (INCONSISTENT)

**Binary** (Hierarchical):
```python
# Item-level random effects with group-level pooling
mu_group = pm.Normal("mu_group", mu=2, sigma=1.5)
sigma_group = pm.Exponential("sigma_group", lam=1.0)
z = pm.Normal("z", mu=0, sigma=1, shape=n_items)
mu_item = mu_group + z * sigma_group  # Hierarchical structure
Theta = sigmoid(mu_item)
obs = pm.Binomial("obs", n=trials, p=Theta, observed=successes)
```

**Continuous** (NON-hierarchical):
```python
# Just pooled all scores - NO item-level structure!
all_continuous_scores = np.array([...])  # All scores lumped together
lo, hi, width = _continuous_bounded_ci_adaptive(all_continuous_scores, ...)
# No hierarchical pooling, no item random effects
```

**Issue**: Continuous inference ignored item-to-item variability and didn't pool information hierarchically.

---

## Solution: Hierarchical Beta Model

### Statistical Design

**Model Structure**:
```
Group Level:
  mu_group ~ Normal(0, 1.5)          # Group mean (logit scale)
  sigma_group ~ Exponential(1.0)     # Between-item SD
  phi_group ~ Gamma(2, 0.1)          # Group-level precision

Item Level:
  z[i] ~ Normal(0, 1)                # Item deviation
  mu_item[i] = sigmoid(mu_group + z[i] * sigma_group)  # Item mean [0,1]

  z_phi[i] ~ Normal(0, 1)            # Item precision deviation
  phi_item[i] = exp(log(phi_group) + z_phi[i] * 0.5)   # Item precision

Observation Level:
  alpha[i] = mu_item[i] * phi_item[i]
  beta[i] = (1 - mu_item[i]) * phi_item[i]
  y[obs] ~ Beta(alpha[item_index[obs]], beta[item_index[obs]])
```

**Key Properties**:
1. **Logit-scale hierarchy**: Unbounded variation between items (like binary)
2. **Mean-precision parameterization**: Separates location (mu) from spread (phi)
3. **Hierarchical precision**: Items can have different within-item variances
4. **Indexed observations**: Each observation linked to its item via `obs_item_indices`

---

## Implementation Details

### File: `optstop/rule.py`

#### Change 1: Add Hierarchical Model Definition (Lines 2176-2245)

**Location**: After binary model definition, before item loop

**Added**:
```python
elif score_type in ['continuous_01', 'continuous_bounded']:
    # === CONTINUOUS HIERARCHICAL MODEL ===
    with pm.Model() as continuous_model:
        # Group-level parameters (logit scale for mean)
        mu_group = pm.Normal("mu_group", mu=0, sigma=1.5)
        sigma_group = pm.Exponential("sigma_group", lam=1.0)
        phi_group = pm.Gamma("phi_group", alpha=2, beta=0.1)

        # Mutable data containers
        n_items = pm.Data("n_items", np.array(1, dtype="int64"))

        # Item-level hierarchical parameters
        z = pm.Normal("z", mu=0, sigma=1, shape=n_items)
        mu_item_logit = pm.Deterministic("mu_item_logit", mu_group + z * sigma_group)
        mu_item = pm.Deterministic("mu_item", pm.math.sigmoid(mu_item_logit_clipped))

        z_phi = pm.Normal("z_phi", mu=0, sigma=1, shape=n_items)
        phi_item = pm.Deterministic("phi_item", pm.math.exp(...))

        # Beta parameterization
        alpha_item = pm.Deterministic("alpha_item", mu_item * phi_item)
        beta_item = pm.Deterministic("beta_item", (1 - mu_item) * phi_item)

        # Indexed observations
        obs_values = pm.Data("obs_values", np.array([0.5]))
        obs_item_indices = pm.Data("obs_item_indices", np.array([0], dtype="int64"))

        obs = pm.Beta("obs", alpha=alpha_item[obs_item_indices],
                     beta=beta_item[obs_item_indices], observed=obs_values)
```

**Documentation**: 50+ lines of inline comments explaining:
- Model structure and hierarchy
- Design decisions (logit scale, mean-precision parameterization)
- Statistical rationale (pooling, shrinkage)
- Data format (indexed observations)

---

#### Change 2: Update Item Summaries (Lines 2394-2403)

**Before**:
```python
item_summaries.append({
    'mean': float(np.mean(accumulated_scores)),
    'count': len(accumulated_scores)
})
```

**After**:
```python
# Normalize scores to [0,1] for Beta likelihood
scores_normalized = (accumulated_scores - lower_bound) / (upper_bound - lower_bound)
scores_normalized = np.clip(scores_normalized, 0.0, 1.0)

item_summaries.append({
    'scores_normalized': scores_normalized,  # For hierarchical model
    'mean': float(np.mean(accumulated_scores)),  # Original scale
    'count': len(accumulated_scores)
})
```

**Rationale**:
- Beta distribution requires [0,1] data
- Store normalized scores for group-level hierarchical inference
- Retain original-scale mean for metadata

---

#### Change 3: Replace Group-Level Inference (Lines 2576-2710)

**Replaced**: 135 lines of non-hierarchical Beta sampling

**New Implementation** (135 lines):

```python
# Prepare data for hierarchical model
all_scores_normalized = []
all_item_indices = []

for item_idx, item_summary in enumerate(item_summaries):
    scores = item_summary['scores_normalized']
    all_scores_normalized.extend(scores)
    all_item_indices.extend([item_idx] * len(scores))

all_scores_normalized = np.array(all_scores_normalized)
all_item_indices = np.array(all_item_indices, dtype="int64")
n_items_actual = len(item_summaries)

# Run hierarchical Bayesian inference
with continuous_model:
    pm.set_data({
        "obs_values": all_scores_normalized,
        "obs_item_indices": all_item_indices,
        "n_items": np.int64(n_items_actual)
    })

    with suppress_all_output():
        trace = pm.sample(**sampling_kwargs)

    with suppress_all_output():
        mu_item_hdi = az.hdi(trace.posterior["mu_item"], hdi_prob=cred_level)

# Extract group-level CI (average across items)
mu_lo_normalized = float(np.mean(mu_values_lower))
mu_hi_normalized = float(np.mean(mu_values_upper))
width_normalized = mu_hi_normalized - mu_lo_normalized
width_original = width_normalized * (upper_bound - lower_bound)

# Stabilization and stopping criteria (unchanged logic)
...
```

**Key Changes**:
1. ✅ Data preparation with item indexing
2. ✅ PyMC hierarchical model usage
3. ✅ MCMC sampling instead of fast Beta sampling
4. ✅ HDI extraction from item-level means
5. ✅ Group-level CI computed as average across items
6. ✅ Updated metadata reasons (`continuous_hierarchical_ci_width`, etc.)

---

## Performance Characteristics

### Expected Timing

| Stage | Before (Non-hierarchical) | After (Hierarchical) | Ratio |
|-------|--------------------------|----------------------|-------|
| **Sample-level** | ~1-2ms (Beta) | ~1-2ms (Beta) | No change |
| **Group-level** | ~1-2ms (Beta) | **~5-30s (PyMC MCMC)** | ~1000x slower |

### Why the Slowdown is Necessary

**Statistical Justification**:
1. **Accounts for item-to-item variability**: Some items inherently harder than others
2. **Partial pooling**: Items with few observations borrow strength from group
3. **Proper uncertainty quantification**: Hierarchical structure captures multiple sources of variance
4. **Consistency with binary**: Same methodology across score types

**Alternative Considered**: Keep fast non-hierarchical version
- ❌ **Rejected**: Inconsistent with binary approach
- ❌ **Rejected**: Ignores item-level variance
- ❌ **Rejected**: No shrinkage for items with sparse data

**Trade-off**: Slower but statistically correct

---

## Comparison: Binary vs Continuous (Now Consistent)

| Aspect | Binary | Continuous | Status |
|--------|--------|-----------|--------|
| **Sample-level** | Beta-Binomial (fast) | Beta + MoM (fast) | ✅ Both fast |
| **Group-level** | PyMC hierarchical | PyMC hierarchical | ✅ **NOW CONSISTENT** |
| **Hierarchy** | Logit-scale item effects | Logit-scale item effects | ✅ Same structure |
| **Random effects** | `z[i] ~ N(0,1)` | `z[i] ~ N(0,1)` | ✅ Same |
| **Pooling** | Partial pooling | Partial pooling | ✅ Same |
| **Timing** | ~5-30s (group) | ~5-30s (group) | ✅ Same |
| **Metadata** | `ci_width` reason | `hierarchical_ci_width` | ✅ Differentiated |

---

## Testing Requirements

### Unit Tests Needed

1. **Model Compilation**:
   ```python
   def test_continuous_hierarchical_model_compiles():
       # Verify PyMC model compiles without errors
       # Check all parameters defined correctly
   ```

2. **Data Format**:
   ```python
   def test_continuous_item_summaries_structure():
       # Verify scores_normalized present in item_summaries
       # Check normalization to [0,1]
       # Verify clip behavior
   ```

3. **Hierarchical Inference**:
   ```python
   def test_continuous_hierarchical_group_inference():
       # Create synthetic data with known item effects
       # Run hierarchical inference
       # Verify shrinkage toward group mean
       # Check CI width appropriately accounts for item variance
   ```

4. **Consistency**:
   ```python
   def test_binary_continuous_consistency():
       # Create equivalent binary and continuous datasets
       # Run both inference routes
       # Verify similar shrinkage patterns
       # Check timing is comparable
   ```

### Integration Tests Needed

5. **End-to-End with inspect_ai Bridge**:
   ```python
   def test_inspect_ai_continuous_aggregated():
       # Use OptimalStoppingManager with score_agg='mean'
       # Verify hierarchical model triggered
       # Check metadata includes 'hierarchical' in reason
       # Validate stopping decisions
   ```

6. **Performance Regression**:
   ```python
   def test_continuous_hierarchical_timing():
       # 80 items × 10 epochs = 800 observations
       # Group-level inference should complete in < 60s
       # Log timing for monitoring
   ```

---

## Breaking Changes

### Metadata Reason Strings

**Before**:
- `continuous_bounded_ci_width`
- `continuous_bounded_stabilization`
- `continuous_bounded_stabilization_low_perf`

**After**:
- `continuous_hierarchical_ci_width`
- `continuous_hierarchical_stabilization`
- `continuous_hierarchical_stabilization_low_perf`

**Impact**:
- Any code parsing metadata reasons needs update
- Documentation referencing old reason strings needs update
- Tests checking for specific reason strings will fail

### Item Summaries Structure

**Before**:
```python
{
    'mean': float,
    'count': int
}
```

**After**:
```python
{
    'scores_normalized': np.ndarray,  # NEW
    'mean': float,
    'count': int
}
```

**Impact**:
- Downstream code accessing item_summaries may need updates
- Memory usage slightly higher (stores full score arrays)

### Performance Expectations

**Before**: Continuous was ~1000x faster than binary at group level

**After**: Continuous and binary have similar timing (~5-30s)

**Impact**:
- Users expecting fast continuous inference will see slowdown
- Need to communicate this is **expected and correct behavior**
- Update documentation to reflect timing expectations

---

## Documentation Updates Needed

1. ✅ **HIERARCHICAL_CONTINUOUS_IMPLEMENTATION.md** (this document)

2. ⚠️ **Update CONTINUOUS_SCORING_FINAL_SUMMARY.md**:
   - Change performance section: "~5-30s (like binary)" not "~1-2ms"
   - Add hierarchical model explanation
   - Update metadata examples

3. ⚠️ **Update PHASE3_COMPLETION_SUMMARY.md**:
   - Revise to reflect hierarchical approach
   - Update performance characteristics

4. ⚠️ **Update README.md**:
   - Performance section: continuous timing now matches binary
   - Add note about hierarchical pooling

5. ⚠️ **Update early_stopping.py docstrings**:
   - Clarify continuous uses hierarchical inference at group level
   - Update timing expectations

---

## Migration Guide

### For Users

**If you were using continuous scoring with `score_agg='mean'` or `'median'`:**

1. **Expected Behavior Change**:
   - Group-level inference now takes 5-30 seconds (was ~1-2ms)
   - This is **correct and intentional** - proper hierarchical pooling
   - No changes needed to your code

2. **Metadata Changes**:
   - Stopping reasons now include "hierarchical" prefix
   - Check for `continuous_hierarchical_ci_width` instead of `continuous_bounded_ci_width`

3. **Benefits**:
   - ✅ More accurate uncertainty quantification
   - ✅ Proper item-level variance accounting
   - ✅ Shrinkage for items with few observations
   - ✅ Consistency with binary methodology

### For Developers

**If you have custom code parsing metadata:**

```python
# Before
if reason == 'continuous_bounded_ci_width':
    ...

# After
if reason == 'continuous_hierarchical_ci_width':
    ...

# Or more robust
if 'continuous' in reason and 'ci_width' in reason:
    ...
```

**If you have tests checking timing:**

```python
# Before
assert group_inference_time < 0.1  # 100ms

# After
assert group_inference_time < 60.0  # 60 seconds (allow for MCMC)
```

---

## Validation Checklist

### Pre-Merge

- [x] Hierarchical model compiles without errors
- [x] Model structure mirrors binary (logit scale, item effects)
- [x] Item summaries store normalized scores
- [x] Group-level inference uses PyMC hierarchical model
- [x] Metadata updated with "hierarchical" prefix
- [x] Comprehensive inline documentation added (100+ lines)
- [ ] Unit tests pass (need to create/update)
- [ ] Integration tests pass (need to create/update)
- [ ] Performance regression tests (timing in acceptable range)

### Post-Merge

- [ ] Update CONTINUOUS_SCORING_FINAL_SUMMARY.md
- [ ] Update PHASE3_COMPLETION_SUMMARY.md
- [ ] Update README.md performance section
- [ ] Update early_stopping.py docstrings
- [ ] Notify users of performance change
- [ ] Add migration notes to CHANGELOG.md

---

## Technical Notes

### Why Mean-Precision Parameterization?

**Standard Beta**: `Beta(alpha, beta)`
- Hard to interpret alpha and beta directly
- Difficult to set priors on alpha/beta

**Mean-Precision**: `mu` and `phi` where `alpha = mu * phi`, `beta = (1-mu) * phi`
- `mu`: Mean of distribution [0,1] (location parameter)
- `phi`: Concentration/precision (spread parameter)
- Easier to reason about: "mean around 0.8 with precision 10"
- More natural hierarchical structure

### Why Hierarchical Precision?

**Group-level `phi_group`**:
- Controls typical within-item variance across all items
- Estimated from data

**Item-level `phi_item[i]`**:
- Allows some items to have tighter/wider distributions
- Accounts for heterogeneity in measurement precision
- Hierarchical: `log(phi_item[i]) = log(phi_group) + z_phi[i] * 0.5`

**Alternative considered**: Fixed precision for all items
- ❌ **Rejected**: Too restrictive, doesn't match real data patterns

### Indexing Strategy

**Challenge**: Each item has variable number of observations (epochs)

**Solution**: Flatten observations, track item membership
```python
obs_values:        [0.8, 0.9, 0.7, 0.85, ...]  # All observations
obs_item_indices:  [0,   0,   1,   1,   ...]  # Which item each belongs to
```

PyMC indexes into `alpha_item` and `beta_item` arrays:
```python
obs = pm.Beta("obs",
             alpha=alpha_item[obs_item_indices],  # Index selects item's alpha
             beta=beta_item[obs_item_indices],    # Index selects item's beta
             observed=obs_values)
```

**Benefit**: Handles ragged arrays naturally without padding

---

## Known Limitations

1. **Performance**: 5-30 second group-level inference may be too slow for very frequent checks
   - **Mitigation**: Increase `reanalysis_interval` in inspect_ai bridge
   - **Future**: Consider variational inference (faster, approximate)

2. **Memory**: Stores full normalized score arrays in item_summaries
   - **Impact**: Minimal for typical use cases (<1000 items)
   - **Future**: Option to reconstruct from df_work if memory constrained

3. **Convergence**: MCMC may struggle with extreme data (all 0s, all 1s)
   - **Mitigation**: Edge case handling in data preparation
   - **Future**: Add convergence diagnostics and fallback

---

## Future Enhancements

### Priority 1: Convergence Diagnostics
- Add R-hat checking after sampling
- Warn if divergences detected
- Fallback to non-hierarchical if convergence fails

### Priority 2: Variational Inference Option
- Add `inference_method='vi'` parameter
- Use ADVI for faster approximate inference (~1-5s instead of 5-30s)
- Trade-off: Speed vs accuracy

### Priority 3: GPU Acceleration
- Enable JAX/Numpyro backend for PyMC
- 10-100x speedup on GPU
- Already supported in infrastructure (gpu_utils)

---

## References

### Related Documents
- CONTINUOUS_SCORING_FINAL_SUMMARY.md - Original non-hierarchical implementation
- PHASE3_COMPLETION_SUMMARY.md - Integration details
- CONTINUOUS_INFERENCE_VALIDITY_ANALYSIS.md - Statistical validation

### Statistical Literature
- Gelman & Hill (2007): Data Analysis Using Regression and Multilevel/Hierarchical Models
- Kruschke (2015): Doing Bayesian Data Analysis (Beta regression chapter)
- McElreath (2020): Statistical Rethinking (Chapter 13: Multilevel models)

### PyMC Documentation
- Beta distribution: https://www.pymc.io/projects/docs/en/stable/api/distributions/continuous.html#pymc.Beta
- Hierarchical models: https://www.pymc.io/projects/examples/en/latest/case_studies/hierarchical_partial_pooling.html
- Indexing: https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/dimensionality.html

---

**Implementation Date**: 2025-11-21
**Implemented By**: Claude (Sonnet 4.5)
**Reviewed By**: [Pending]
**Status**: ✅ Code implemented, tests pending
**Next Steps**: Create/update unit and integration tests
