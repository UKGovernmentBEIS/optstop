# Continuous MCMC Performance Investigation

**Date**: 2025-11-21
**Issue**: Continuous hierarchical inference takes 20-40 minutes vs 4-5 seconds for binary
**Status**: 🔍 **INVESTIGATING**

---

## Empirical Timing Results

| Model Type | Items | Observations | Group Inference Time | Ratio vs Binary |
|------------|-------|--------------|---------------------|-----------------|
| **Binary** | 50 | ~500-1500 (aggregated) | **4-5 seconds** | 1x baseline |
| **Binary** | 100 | ~1000-2000 (aggregated) | **4-5 seconds** | 1x baseline |
| **Ordinal (modal)** | 60 | ~600 | **0.2-0.4 seconds** | 0.1x (faster!) |
| **Continuous** | 50 | 200-950 (individual) | **1294-2278 seconds** | **300-500x slower!** |

**Key Finding**: Data aggregation is NOT the bottleneck (takes <0.01s)
**Actual Bottleneck**: PyMC MCMC sampling takes 1300-2300 seconds

---

## Root Cause Analysis

### Problem 1: Individual vs Aggregated Likelihoods

**Binary Model Structure:**
```python
# Aggregated data per item
successes_data = [s1, s2, ..., s50]  # One per item
trials_data = [t1, t2, ..., t50]     # One per item

# ONE likelihood evaluation per item
obs = pm.Binomial("obs", n=trials_data, p=Theta, observed=successes_data)
# Total: 50 Binomial likelihood evaluations per MCMC step
```

**Continuous Model Structure:**
```python
# Individual observations (NOT aggregated)
obs_values = [o1, o2, o3, ..., o950]           # ALL observations
obs_item_indices = [0, 0, 0, 1, 1, 1, ...]    # Which item each belongs to

# ONE likelihood evaluation PER OBSERVATION
obs = pm.Beta("obs",
             alpha=alpha_item[obs_item_indices],
             beta=beta_item[obs_item_indices],
             observed=obs_values)
# Total: 950 Beta likelihood evaluations per MCMC step (19x more!)
```

**Impact:**
- Binary: 50 likelihood evaluations per MCMC step
- Continuous: 200-950 likelihood evaluations per MCMC step
- **10-20x more computations per step**

With default sampling:
- 500 tune × 4 chains = 2000 tuning steps
- 500 draws × 4 chains = 2000 sampling steps
- Total: **4000 MCMC steps × 950 likelihoods = 3.8 million likelihood evaluations**

Binary only does: 4000 × 50 = 200,000 likelihood evaluations (19x fewer)

---

### Problem 2: Inappropriate Prior for phi_group

**Current:**
```python
phi_group = pm.Gamma("phi_group", alpha=2, beta=0.1)
```

**Gamma(2, 0.1) properties:**
- Mean: alpha/beta = 2/0.1 = **20**
- Variance: alpha/beta² = 2/0.01 = 200
- Mode: (alpha-1)/beta = 1/0.1 = 10
- 95% interval: approximately [2, 55]

**What this means for Beta distribution:**
- Beta with mu=0.8, phi=20: **VERY tight** around 0.8
- Alpha = 0.8 × 20 = 16, Beta = 0.2 × 20 = 4
- This Beta(16, 4) has SD = 0.083 (very narrow!)

**For typical aggregated scores** (mean of 5-15 binary trials):
- Observed variance is higher than phi=20 assumes
- Prior conflicts with data → poor MCMC geometry → slow sampling

**Better prior:**
```python
phi_group = pm.Gamma("phi_group", alpha=2, beta=1.0)
# Mean = 2, allows more within-item variance
# More appropriate for aggregated data
```

---

### Problem 3: Extra Complexity - Hierarchical Precision

**Binary has:** 2 group parameters (mu_group, sigma_group)

**Continuous has:** 3 group parameters PLUS hierarchical precision:
- mu_group, sigma_group, phi_group
- z_phi[n_items] - additional n_items parameters
- log_phi_item, phi_item - transformations

**Total parameters:**
- Binary: 2 + n_items = 52 parameters (for 50 items)
- Continuous: 3 + 2×n_items = 103 parameters (for 50 items)

**Impact:** Nearly 2x more parameters → slower sampling

**Why do we need hierarchical precision?**
- Allows items to have different within-item variance
- But adds significant complexity
- May not be necessary for most use cases

---

### Problem 4: Indexing Overhead

```python
alpha_item[obs_item_indices]  # Index into array 950 times
beta_item[obs_item_indices]   # Index into array 950 times
```

PyMC must:
1. Compute alpha_item and beta_item for all 50 items
2. Index into these arrays 950 times for each likelihood evaluation
3. This happens at EVERY MCMC step

**Comparison to Binary:**
- Binary directly uses Theta (no indexing per observation)
- Vectorized operations on 50 items

---

## Why Binary is So Much Faster

1. **Sufficient statistics**: Binomial uses successes/trials (aggregated)
2. **Fewer likelihood evaluations**: 50 vs 950
3. **Simpler model**: Fewer parameters (52 vs 103)
4. **No indexing overhead**: Direct vectorized operations
5. **Better prior scaling**: Priors match data scale

---

## Proposed Solutions

### Solution A: Aggregate Continuous Data (RECOMMENDED)

**Problem:** We're treating each observation individually when we could aggregate.

**For continuous data from the same item:**
- We have observations: [0.82, 0.88, 0.79, 0.85] from item 1
- Current: 4 Beta likelihood evaluations
- Better: **Use sufficient statistics** - sample mean and sample variance

**Implementation:**
```python
# Instead of individual observations, use aggregated statistics
# For each item, compute:
item_means = [mean(obs) for item in items]  # One per item
item_vars = [var(obs) for item in items]    # One per item
item_ns = [count(obs) for item in items]    # Sample sizes

# Use Normal likelihood with known variance structure
# OR Beta likelihood with effective sample size
```

**Benefits:**
- Reduces 950 likelihoods to 50 likelihoods (19x reduction!)
- Should bring timing closer to binary performance
- Still hierarchical pooling

**Trade-off:**
- Loses some information about within-item distribution shape
- Assumes observations within item are exchangeable

---

### Solution B: Simplify Model - Remove Hierarchical Precision

**Current complexity:** phi_group + z_phi[n_items] + transformations

**Simplified:**
```python
# Use fixed precision for all items
phi = pm.Gamma("phi", alpha=2, beta=1.0)  # Single parameter

# Beta parameterization (same for all items)
alpha_item = mu_item * phi
beta_item = (1 - mu_item) * phi
```

**Benefits:**
- Removes 1 + n_items parameters (53 fewer parameters for 50 items)
- Simpler model → faster sampling
- Still has hierarchical means (main feature)

**Trade-off:**
- Assumes all items have same within-item variance
- Less flexible but much faster

---

### Solution C: Better Priors

**Current issues:**
1. `mu_group = pm.Normal(mu=0, sigma=1.5)` on logit scale
   - For scores around 0.8-0.9, logit(0.8) = 1.39, logit(0.9) = 2.20
   - Prior centered at 0 (corresponds to p=0.5) is too low

2. `phi_group = pm.Gamma(alpha=2, beta=0.1)` gives mean=20
   - Too tight for aggregated scores

**Better priors:**
```python
# For high-performing scenarios (scores typically 0.7-0.95)
mu_group = pm.Normal("mu_group", mu=1.5, sigma=1.5)
# logit(0.82) ≈ 1.5, more centered on typical performance

# For aggregated scores with moderate variance
phi_group = pm.Gamma("phi_group", alpha=2, beta=1.0)
# Mean=2, allows more variance, better for aggregated data

# Tighter prior on between-item variance
sigma_group = pm.Exponential("sigma_group", lam=2.0)
# Mean=0.5 instead of 1.0, assumes items are somewhat similar
```

---

### Solution D: Use Variational Inference (ADVI)

**Instead of MCMC (slow but accurate):**
```python
with continuous_model:
    pm.set_data({...})

    # Use ADVI instead of MCMC
    approx = pm.fit(n=10000, method='advi')  # Usually 10-60 seconds
    trace = approx.sample(500)
```

**Benefits:**
- **Much faster**: 10-60 seconds instead of 20-40 minutes
- Still approximate posterior
- Good enough for early stopping decisions

**Trade-offs:**
- Approximate (not exact posterior)
- May underestimate uncertainty
- Not suitable for all model types

---

## Recommended Implementation Order

### 1. **Solution A + Solution B** (Best combination)
- Aggregate continuous data to item-level statistics
- Remove hierarchical precision (fixed phi for all items)
- Expected timing: **~5-15 seconds** (similar to binary)
- Maintains hierarchical pooling for means

### 2. **Solution C** (If keeping current structure)
- Fix priors to match data scale
- Expected improvement: **2-3x faster** (still ~7-15 minutes, not good enough)

### 3. **Solution D** (Quick fix)
- Switch to ADVI variational inference
- Expected timing: **10-60 seconds**
- Trade-off: approximate posteriors

---

## Testing Plan

1. **Implement Solution A + B** first
2. Test with 50 items, varying observations
3. Compare timing to binary baseline
4. If still too slow, add Solution D (ADVI)
5. Validate statistical properties (coverage, shrinkage)

---

## Next Steps

**Immediate:**
1. Implement data aggregation (Solution A)
2. Simplify to fixed precision (Solution B)
3. Run timing tests

**If needed:**
4. Improve priors (Solution C)
5. Add ADVI option (Solution D)

---

## Code Location

**File:** `optstop/rule.py`
**Lines:** 2179-2248 (continuous model definition)
**Lines:** 2610-2689 (continuous group inference)

**Changes needed:**
1. Model definition: Aggregate data, simplify precision
2. Data preparation: Compute item-level statistics instead of concatenating observations
3. Inference: Test timing improvements

---

**Status**: Ready to implement Solution A + B
**Expected outcome**: Reduce continuous inference from 20-40 minutes to 5-15 seconds
**Risk**: Low (statistical validity maintained, just different parameterization)
