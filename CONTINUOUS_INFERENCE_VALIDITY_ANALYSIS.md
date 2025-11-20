# Continuous Bounded Inference Validity Analysis

**Date**: 2025-11-19
**Purpose**: Verify that continuous bounded inference properly corresponds to binary and ordinal versions

---

## Executive Summary

✅ **Verdict**: The continuous bounded inference is **valid and consistent** with binary/ordinal patterns, with **one critical normalization requirement** for Phase 3 integration.

**Key Finding**: The CI functions return widths in **different scales**:
- Binary: Returns width in [0, 1] (natural proportion scale)
- Ordinal: Returns width in [0, 1] (normalized by ordinal_max_score)
- **Continuous: Returns width in ORIGINAL scale** [lower_bound, upper_bound]

**Action Required**: Phase 3 integration must **normalize continuous width** before comparison with delta thresholds.

---

## Comparative Analysis: Three Inference Routes

### 1. Score Variable Types

| Route | Score Type | Values | Example |
|-------|-----------|--------|---------|
| **Binary** | Discrete binomial | {0, 1} | [1, 0, 1, 1, 0] |
| **Ordinal** | Discrete categorical | {0, 1, ..., K} | [7, 8, 9, 7, 8] |
| **Continuous** | Continuous bounded | [a, b] ⊂ ℝ | [0.8, 0.85, 0.9, 0.75] |

**Key Difference**: Variable type (discrete vs continuous), not the underlying construct being measured.

---

## Sample-Level Inference Comparison

### Binary Sample-Level (lines 2200-2227)

```python
# Aggregate data for this sample
successes = int(df_item[score_column].sum())
trials = len(df_item)

# Determine conservatism
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

item_summaries.append({'successes': successes, 'trials': trials})
```

**Input**: Discrete counts (successes, trials)
**CI Function**: `_beta_ci_adaptive()` → Beta-Binomial inference
**Output Width Scale**: [0, 1] (proportion)
**Stopping Criterion**: `width < delta_item` (direct comparison)

---

### Ordinal Sample-Level (lines 2229-2304)

```python
# Aggregate data for this sample
accumulated_scores = df_item[score_column].tolist()

# Determine conservatism
current_perf = np.mean(accumulated_scores) / ordinal_max_score
current_conservatism = conservatism if current_perf < low_perf_threshold else 1.0

# Compute CI (modal inference shown)
lo, hi, width = _ordinal_ci_adaptive(
    np.array(accumulated_scores),
    ordinal_max_score=ordinal_max_score,
    cred_level=cred_level,
    conservatism=current_conservatism,
    low_perf_threshold=low_perf_threshold
)

# Check width criterion
if width < delta_item:
    stop_sample_ids.append(f"{grouping_name}:::{original_sample_id}")
    metadata['sample_stopping_reasons'][str(original_sample_id)] = {
        'reason': 'ordinal_modal_ci_width',
        'ci_width': float(width),
        'threshold': delta_item,
        'epochs_used': len(accumulated_scores)
    }

item_summaries.append({
    'successes': int(np.sum(accumulated_scores)),
    'trials': len(accumulated_scores)
})
```

**Input**: Array of discrete integers
**CI Function**: `_ordinal_ci_adaptive()` → Bayesian Bootstrap modal inference
**Output Width Scale**: [0, 1] (normalized by ordinal_max_score - see ordinal_utils.py:137-140)
**Stopping Criterion**: `width < delta_item` (direct comparison)

---

### Continuous Sample-Level (PROPOSED for Phase 3)

```python
# Aggregate data for this sample
accumulated_scores = df_item[score_column].values  # numpy array of floats

# Determine conservatism (normalize to [0, 1])
normalized_perf = np.mean(accumulated_scores) / upper_bound
current_conservatism = conservatism if normalized_perf < low_perf_threshold else 1.0

# Compute CI
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

item_summaries.append({
    'scores': accumulated_scores.tolist(),  # Store for group-level
    'mean': np.mean(accumulated_scores),
    'count': len(accumulated_scores)
})
```

**Input**: Array of continuous floats
**CI Function**: `_continuous_bounded_ci_adaptive()` → Beta + Method of Moments
**Output Width Scale**: [lower_bound, upper_bound] (ORIGINAL scale)
**Stopping Criterion**: `width_normalized < delta_item` (**requires normalization**)

---

## Group-Level Inference Comparison

### Binary Group-Level (lines 2311-2393)

**Aggregation**:
```python
all_successes = np.array([s['successes'] for s in item_summaries])
all_trials = np.array([s['trials'] for s in item_summaries])
```

**Inference**: Hierarchical PyMC model with multiple items
```python
with model:
    pm.set_data({"successes": all_successes, "trials": all_trials, ...})
    trace = pm.sample(**sampling_kwargs)
    theta_hdi = az.hdi(trace.posterior["Theta"], hdi_prob=cred_level)

theta_width = theta_hi - theta_lo
```

**Two Stopping Criteria**:

1. **Width Criterion**:
```python
if effective_width < delta_cap:
    stop_this_grouping.append(grouping_name)
```

2. **Stabilization Criterion**:
```python
stabilization_history['ci_width_history'].append(float(theta_width))

if len(stabilization_history['ci_width_history']) >= stab_window:
    recent_widths = stabilization_history['ci_width_history'][-stab_window:]
    slope = np.polyfit(range(len(recent_widths)), recent_widths, 1)[0]

    if abs(slope) <= slope_threshold:
        # Check if slope is stabilizing (slope of slopes)
        # Stop if stabilized
```

**Width Scale**: [0, 1] (from hierarchical model)
**Stabilization**: YES - tracks CI width history and slope

---

### Ordinal Group-Level (lines 2395-2467)

**Aggregation**:
```python
all_ord_scores = []
for item_id in item_ids:
    df_item = df_work[df_work['sample_id_num'] == item_id]
    all_ord_scores.extend(df_item[score_column].tolist())
```

**Inference**: Modal/Entropy/Hybrid on all aggregated scores
```python
lo, hi, width = _ordinal_ci_adaptive(
    np.array(all_ord_scores),
    ordinal_max_score=ordinal_max_score,
    cred_level=cred_level,
    conservatism=current_conservatism,
    low_perf_threshold=low_perf_threshold
)
```

**One Stopping Criterion**:

1. **Width Criterion Only**:
```python
if width < delta_cap:
    stop_this_grouping.append(grouping_name)
```

**Width Scale**: [0, 1] (normalized)
**Stabilization**: NO - only width criterion

---

### Continuous Group-Level (PROPOSED for Phase 3)

**Aggregation**:
```python
all_continuous_scores = []
for item_summary in item_summaries:
    all_continuous_scores.extend(item_summary['scores'])

all_continuous_scores = np.array(all_continuous_scores)
```

**Inference**: Continuous CI on all aggregated scores
```python
# Normalize performance for conservatism check
current_perf_estimate = np.mean(all_continuous_scores)
normalized_perf = current_perf_estimate / upper_bound
current_conservatism = conservatism if normalized_perf < low_perf_threshold else 1.0

# Compute CI at group level
lo, hi, width = _continuous_bounded_ci_adaptive(
    all_continuous_scores,
    lower_bound=lower_bound,
    upper_bound=upper_bound,
    cred_level=cred_level,
    conservatism=current_conservatism,
    low_perf_threshold=low_perf_threshold
)

# ⚠️ CRITICAL: Normalize width for comparison
width_normalized = width / (upper_bound - lower_bound)
```

**Two Stopping Criteria** (like binary):

1. **Width Criterion**:
```python
stabilization_history['ci_width_history'].append(float(width_normalized))

if width_normalized < delta_cap:
    stop_this_grouping.append(grouping_name)
    metadata['group_stopping_reason'] = {
        'reason': 'continuous_bounded_ci_width',
        'ci_width': float(width),
        'ci_width_normalized': float(width_normalized),
        'threshold': delta_cap,
        'samples_used': len(item_summaries),
        'bounds': {'lower': lower_bound, 'upper': upper_bound}
    }
```

2. **Stabilization Criterion**:
```python
if len(stabilization_history['ci_width_history']) >= stab_window:
    recent_widths = stabilization_history['ci_width_history'][-stab_window:]
    slope = np.polyfit(range(len(recent_widths)), recent_widths, 1)[0]
    stabilization_history['ci_slope_history'].append(float(slope))

    slope_threshold = CI_delta / current_conservatism if normalized_perf < low_perf_threshold else CI_delta

    if abs(slope) <= slope_threshold and len(stabilization_history['ci_slope_history']) >= 4:
        recent_slopes = stabilization_history['ci_slope_history'][-3:]
        slope_slopes = np.polyfit(range(len(recent_slopes)), recent_slopes, 1)[0]

        if slope_slopes >= 0:
            # Apply same logic as binary for stabilization stopping
            if normalized_perf >= low_perf_threshold:
                stop_this_grouping.append(grouping_name)
                metadata['group_stopping_reason'] = {
                    'reason': 'continuous_bounded_stabilization',
                    'slope': float(slope),
                    'slope_threshold': slope_threshold,
                    'samples_used': len(item_summaries)
                }
```

**Width Scale**: Must normalize to [0, 1]
**Stabilization**: YES - follows binary pattern (appropriate for continuous metrics)

---

## Critical Width Normalization Analysis

### Why Normalization Matters

**Binary** ([0, 1]):
- Width = 0.05 means 5% of the total range
- Comparison: `0.05 < delta_item` (e.g., 0.05)
- ✅ No normalization needed (range = 1)

**Ordinal** ([0, 10]):
- Raw width in category units might be 0.5 categories
- Normalized width = 0.5 / 10 = 0.05 (5% of range)
- Comparison: `0.05 < delta_item` (e.g., 0.05)
- ✅ Already normalized in `_ordinal_ci_adaptive()` (line 137-140)

**Continuous** ([0, 10]):
- Raw width might be 0.5 in original units
- Without normalization: `0.5 < 0.05` → FALSE (never stops!)
- With normalization: `0.5/10 = 0.05 < 0.05` → depends on data
- ⚠️ **MUST normalize in integration code**

### Normalization Formula

```python
width_normalized = width / (upper_bound - lower_bound)
```

**Verification**:
- Binary [0, 1]: `width / (1 - 0) = width / 1 = width` ✅
- Ordinal [0, 10]: `width / (10 - 0) = width / 10` ✅ (matches ordinal normalization)
- Continuous [0, 10]: `width / (10 - 0) = width / 10` ✅ (same scale as ordinal)
- Continuous [0, 1]: `width / (1 - 0) = width / 1 = width` ✅ (same as binary)

---

## Performance Normalization for Conservatism

All three routes normalize performance to [0, 1] for conservatism checks:

**Binary**:
```python
current_perf = successes / trials  # Already in [0, 1]
current_conservatism = conservatism if current_perf < low_perf_threshold else 1.0
```

**Ordinal**:
```python
current_perf = np.mean(accumulated_scores) / ordinal_max_score  # Normalized
current_conservatism = conservatism if current_perf < low_perf_threshold else 1.0
```

**Continuous**:
```python
normalized_perf = np.mean(accumulated_scores) / upper_bound  # Normalized
current_conservatism = conservatism if normalized_perf < low_perf_threshold else 1.0
```

✅ **Pattern is consistent** across all three routes.

---

## Stabilization Criterion: Binary vs Ordinal vs Continuous

### Binary: Has Stabilization ✅

**Rationale**:
- Binary inference produces a **continuous metric** (CI width) that changes smoothly
- Tracking slope of CI width over time detects convergence
- Appropriate for hierarchical model that pools information across samples

**Implementation**: Tracks `ci_width_history` and computes slope

### Ordinal: No Stabilization ❌

**Rationale**:
- Current implementation focuses on width criterion only
- Modal inference may have discrete jumps as mode changes
- Could be added in future, but not critical for Phase 1

**Implementation**: Only width criterion

### Continuous: Should Have Stabilization ✅

**Rationale**:
- Continuous inference produces a **continuous metric** (CI width) that changes smoothly
- Same logic as binary: CI width converges as more samples accumulated
- Continuous bounded scores are analogous to binary proportions (both continuous)
- Stabilization detects when adding more samples has diminishing returns

**Recommendation**: **YES - include stabilization criterion** (follow binary pattern)

---

## Item Summaries Structure

### Binary
```python
item_summaries.append({'successes': successes, 'trials': trials})
```

### Ordinal
```python
item_summaries.append({
    'successes': int(np.sum(accumulated_scores)),
    'trials': len(accumulated_scores)
})
```

### Continuous (Proposed)
```python
item_summaries.append({
    'scores': accumulated_scores.tolist(),  # Store all scores for group aggregation
    'mean': np.mean(accumulated_scores),
    'count': len(accumulated_scores)
})
```

**Rationale for Continuous**:
- Need to store actual scores for group-level aggregation
- Can't summarize as successes/trials (continuous, not discrete)
- Store mean for performance estimation

**Alternative (simpler)**: Reconstruct from `df_work` at group level like ordinal does
```python
# At group level
all_continuous_scores = []
for item_id in item_ids:
    df_item = df_work[df_work['sample_id_num'] == item_id]
    all_continuous_scores.extend(df_item[score_column].values)
```

Both approaches work. Storing in `item_summaries` is more explicit but uses memory.

---

## Validation Summary

| Aspect | Binary | Ordinal | Continuous | Valid? |
|--------|--------|---------|------------|--------|
| **Score Type** | Discrete {0,1} | Discrete {0,...,K} | Continuous [a,b] | ✅ Fundamental difference |
| **Sample Input** | Counts | Integer array | Float array | ✅ Matches type |
| **Sample CI Function** | Beta-Binomial | Bayesian Bootstrap | Beta + MoM | ✅ Appropriate methods |
| **Sample Width Scale** | [0, 1] | [0, 1] normalized | **[a, b] raw** | ⚠️ **Needs normalization** |
| **Sample Conservatism** | Normalized perf | Normalized perf | Normalized perf | ✅ Consistent |
| **Group Aggregation** | All success/trials | All scores | All scores | ✅ Appropriate |
| **Group CI Function** | PyMC hierarchical | Modal/Entropy | Beta + MoM | ✅ Appropriate |
| **Group Width Scale** | [0, 1] | [0, 1] normalized | **[a, b] raw** | ⚠️ **Needs normalization** |
| **Group Stabilization** | YES | NO | **Should be YES** | ✅ Makes sense |
| **Metadata Structure** | reason, width, threshold | reason, width, threshold | reason, width, threshold | ✅ Consistent |

---

## Conclusion

### ✅ **Validity Assessment: VALID**

The continuous bounded inference approach is **fundamentally sound** and properly corresponds to binary and ordinal routes. The key differences are:

1. **Score variable type** (continuous vs discrete) - appropriate
2. **Statistical method** (Beta with MoM vs Beta-Binomial vs Modal) - appropriate
3. **Width scale** (raw vs normalized) - **requires normalization in Phase 3**

### ⚠️ **Critical Requirements for Phase 3**

1. **Width Normalization** (MUST):
   ```python
   width_normalized = width / (upper_bound - lower_bound)
   ```
   Apply at BOTH sample and group levels before comparing with delta thresholds.

2. **Stabilization Criterion** (RECOMMENDED):
   Include stabilization logic at group level (follow binary pattern)

3. **Performance Normalization** (MUST):
   ```python
   normalized_perf = np.mean(scores) / upper_bound
   ```
   For conservatism checks

### 📋 **Phase 3 Integration Checklist**

- [ ] Extract bounds from `determine_score_type()` result
- [ ] Normalize width at sample level: `width / (upper_bound - lower_bound)`
- [ ] Normalize performance for conservatism: `mean / upper_bound`
- [ ] Store scores in item_summaries or reconstruct from df_work
- [ ] Normalize width at group level: `width / (upper_bound - lower_bound)`
- [ ] Append normalized width to `ci_width_history`
- [ ] Implement stabilization criterion (slope analysis)
- [ ] Update metadata with normalized widths and bounds
- [ ] Test with binary aggregated ([0, 1]) and ordinal aggregated ([0, 10])

---

**Analysis Date**: 2025-11-19
**Status**: ✅ Valid with normalization requirements identified
**Ready for Phase 3**: YES with identified requirements
