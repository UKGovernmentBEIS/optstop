# Internal Consistency Review: optimal_stopping_live_single()

**Date**: 2025-11-19
**Purpose**: Verify internal consistency across binary, ordinal, and continuous inference routes
**Function**: `optstop/rule.py:optimal_stopping_live_single()`

---

## Review Approach

Systematically compare all three routes across key dimensions:
1. **Function Structure** - Overall flow and organization
2. **Parameter Handling** - Extracting and using parameters
3. **Sample-Level Logic** - How individual samples are processed
4. **Group-Level Logic** - How groupings are processed
5. **Metadata Structure** - What information is stored
6. **Edge Cases** - Handling of special conditions
7. **Performance Estimation** - How conservatism is determined

---

## 1. Function Structure

### Common Setup (Lines 2093-2179)

**All Routes Share**:
```python
# Initialize stabilization history
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

# Determine score type
is_aggregated = params.get('is_aggregated', False)
score_type, bounds = determine_score_type(...)

# Extract bounds
lower_bound = bounds['lower']
upper_bound = bounds['upper']

# Initialize result structure
stop_sample_ids = []
stop_this_grouping = []
metadata = {
    'sample_stopping_reasons': {},
    'group_stopping_reason': None
}

# Create numeric columns
df_work = df_grouping.copy()
df_work['sample_id_num'] = df_work[sample_id_column].astype('category').cat.codes
df_work['epoch_num'] = df_work[epoch_column].astype(int)
item_ids = sorted(df_work['sample_id_num'].unique())
```

✅ **Status**: CONSISTENT - All routes use same initialization

---

## 2. Parameter Handling

### Parameter Extraction

| Parameter | Binary | Ordinal | Continuous | Consistent? |
|-----------|--------|---------|------------|-------------|
| `delta_item` | ✅ Used | ✅ Used | ✅ Used | ✅ Yes |
| `delta_cap` | ✅ Used | ✅ Used | ✅ Used | ✅ Yes |
| `cred_level` | ✅ Used | ✅ Used | ✅ Used | ✅ Yes |
| `conservatism` | ✅ Used | ✅ Used | ✅ Used | ✅ Yes |
| `low_perf_threshold` | ✅ Used | ✅ Used | ✅ Used | ✅ Yes |
| `CI_delta` | ✅ Used (stabilization) | ❌ Not used | ✅ Used (stabilization) | ⚠️ Ordinal doesn't stabilize |
| `stab_window` | ✅ Used | ❌ Not used | ✅ Used | ⚠️ Ordinal doesn't stabilize |
| `rep_batch_size` | ❌ Not used | ❌ Not used | ❌ Not used | ✅ Consistent (unused) |

**Analysis**:
- `CI_delta` and `stab_window` only used by binary and continuous (both have stabilization)
- Ordinal doesn't use stabilization (by design)
- `rep_batch_size` appears unused in `live_single` (may be for `posthoc` mode)

✅ **Status**: CONSISTENT - Differences are intentional

---

## 3. Sample-Level Logic Comparison

### Binary Sample-Level (Lines 2178-2205)

**Structure**:
```python
if score_type == 'binary':
    # 1. Aggregate data
    successes = int(df_item[score_column].sum())
    trials = len(df_item)

    # 2. Determine conservatism
    current_perf = successes / trials if trials > 0 else 0
    current_conservatism = conservatism if current_perf < low_perf_threshold else 1.0

    # 3. Compute CI
    lo, hi, width = _beta_ci_adaptive(
        successes, trials,
        cred_level=cred_level,
        conservatism=current_conservatism,
        low_perf_threshold=low_perf_threshold
    )

    # 4. Check width criterion
    if width < delta_item:
        stop_sample_ids.append(f"{grouping_name}:::{original_sample_id}")
        metadata['sample_stopping_reasons'][str(original_sample_id)] = {
            'reason': 'ci_width',
            'ci_width': float(width),
            'threshold': delta_item,
            'epochs_used': trials
        }

    # 5. Store summary
    item_summaries.append({'successes': successes, 'trials': trials})
```

### Ordinal Sample-Level (Lines 2207-2281)

**Structure**:
```python
elif score_type == 'ordinal':
    # 1. Aggregate data
    accumulated_scores = df_item[score_column].tolist()

    # 2. Determine conservatism
    current_perf = np.mean(accumulated_scores) / ordinal_max_score
    current_conservatism = conservatism if current_perf < low_perf_threshold else 1.0

    # 3. Compute CI (modal/entropy/hybrid)
    entropy_history = entropy_history_per_item.get(item_id, [])

    if ordinal_inference == 'modal':
        lo, hi, width = _ordinal_ci_adaptive(...)
    elif ordinal_inference == 'entropy':
        lo, hi, width, diagnostics = _ordinal_entropy_ci_adaptive(...)
    elif ordinal_inference == 'hybrid':
        should_stop, reason, diagnostics = _ordinal_hybrid_stopping_criterion(...)

    # 4. Check width criterion
    if width < delta_item:  # (modal/entropy) or should_stop (hybrid)
        stop_sample_ids.append(f"{grouping_name}:::{original_sample_id}")
        metadata['sample_stopping_reasons'][str(original_sample_id)] = {
            'reason': 'ordinal_modal_ci_width',  # or entropy/hybrid
            'ci_width': float(width),
            'threshold': delta_item,
            'epochs_used': len(accumulated_scores)
        }

    # 5. Store summary
    entropy_history_per_item[item_id] = entropy_history
    item_summaries.append({
        'successes': int(np.sum(accumulated_scores)),
        'trials': len(accumulated_scores)
    })
```

### Continuous Sample-Level (Lines 2283-2320)

**Structure**:
```python
elif score_type in ['continuous_01', 'continuous_bounded']:
    # 1. Aggregate data
    accumulated_scores = df_item[score_column].values

    # 2. Determine conservatism
    normalized_perf = np.mean(accumulated_scores) / upper_bound
    current_conservatism = conservatism if normalized_perf < low_perf_threshold else 1.0

    # 3. Compute CI
    lo, hi, width = _continuous_bounded_ci_adaptive(
        accumulated_scores,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        cred_level=cred_level,
        conservatism=current_conservatism,
        low_perf_threshold=low_perf_threshold
    )

    # 4. Check width criterion (WITH NORMALIZATION)
    width_normalized = width / (upper_bound - lower_bound)

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

    # 5. Store summary
    item_summaries.append({
        'mean': float(np.mean(accumulated_scores)),
        'count': len(accumulated_scores)
    })
```

### Sample-Level Consistency Analysis

| Aspect | Binary | Ordinal | Continuous | Consistent? |
|--------|--------|---------|------------|-------------|
| **1. Data Aggregation** | Sum/count | Array | Array | ✅ Yes (type-appropriate) |
| **2. Performance Calc** | `succ/trials` | `mean/max` | `mean/upper` | ✅ Yes (all normalized) |
| **3. Conservatism Logic** | `< threshold` | `< threshold` | `< threshold` | ✅ Yes (identical) |
| **4. CI Function Call** | Beta-Binomial | Modal/Entropy/Hybrid | Beta+MoM | ✅ Yes (type-appropriate) |
| **5. Width Comparison** | Direct | Direct | **Normalized** | ⚠️ **Different but correct** |
| **6. Metadata Structure** | reason, width, threshold, epochs | reason, width, threshold, epochs | reason, width, **width_norm**, threshold, epochs, **bounds** | ⚠️ **Continuous has extra fields** |
| **7. Summary Storage** | `{successes, trials}` | `{successes, trials}` | **`{mean, count}`** | ⚠️ **Different structure** |

**Issues Found**:

1. ⚠️ **Width Normalization**: Continuous uses `width_normalized`, others use `width` directly
   - **Status**: ✅ CORRECT - Required due to different output scales

2. ⚠️ **Metadata Fields**: Continuous includes `ci_width_normalized` and `bounds`
   - **Status**: ✅ CORRECT - Provides transparency about normalization

3. ⚠️ **Item Summaries**: Continuous uses different structure
   - **Status**: ⚠️ **POTENTIAL ISSUE** - May cause issues in group-level code

✅ **Overall Status**: MOSTLY CONSISTENT with intentional differences

---

## 4. Group-Level Logic Comparison

### Performance Estimation (Lines 2326-2333)

**NEW CODE**:
```python
# Compute performance estimate based on score type
if score_type in ['binary', 'ordinal']:
    current_perf_estimate = np.sum([s['successes'] for s in item_summaries]) / np.sum([s['trials'] for s in item_summaries])
else:  # continuous
    # For continuous, use mean of means weighted by counts
    total_sum = np.sum([s['mean'] * s['count'] for s in item_summaries])
    total_count = np.sum([s['count'] for s in item_summaries])
    current_perf_estimate = total_sum / total_count if total_count > 0 else 0
```

✅ **Status**: CORRECT - Handles different item_summary structures

### Binary Group-Level (Lines 2335-2409)

**Structure**:
```python
if score_type == 'binary':
    # 1. Aggregate data
    all_successes = np.array([s['successes'] for s in item_summaries])
    all_trials = np.array([s['trials'] for s in item_summaries])

    # 2. Determine conservatism
    current_conservatism = conservatism if current_perf_estimate < low_perf_threshold else 1.0

    # 3. Compute CI (hierarchical PyMC model)
    with model:
        pm.set_data({"successes": all_successes, "trials": all_trials, ...})
        trace = pm.sample(**sampling_kwargs)
        theta_hdi = az.hdi(trace.posterior["Theta"], hdi_prob=cred_level)

    theta_width = theta_hi - theta_lo

    # 4. Append to history
    stabilization_history['ci_width_history'].append(float(theta_width))

    # 5. Check width criterion
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

    # 6. Check stabilization criterion
    if len(stabilization_history['ci_width_history']) >= stab_window:
        recent_widths = stabilization_history['ci_width_history'][-stab_window:]
        slope = np.polyfit(range(len(recent_widths)), recent_widths, 1)[0]
        stabilization_history['ci_slope_history'].append(float(slope))

        # Slope threshold with conservatism adjustment
        slope_threshold = CI_delta / current_conservatism if current_perf_estimate < low_perf_threshold else CI_delta

        if abs(slope) <= slope_threshold and len(stabilization_history['ci_slope_history']) >= 4:
            recent_slopes = stabilization_history['ci_slope_history'][-3:]
            slope_slopes = np.polyfit(range(len(recent_slopes)), recent_slopes, 1)[0]

            if slope_slopes >= 0:
                # Check performance level for stopping
                if current_perf_estimate >= low_perf_threshold:
                    stop_this_grouping.append(grouping_name)
                    metadata['group_stopping_reason'] = {
                        'reason': 'ci_stabilization',
                        'slope': float(slope),
                        'slope_threshold': slope_threshold,
                        'samples_used': len(item_summaries)
                    }
                elif abs(slope) <= slope_threshold / 2:
                    stop_this_grouping.append(grouping_name)
                    metadata['group_stopping_reason'] = {
                        'reason': 'ci_stabilization_low_perf',
                        'slope': float(slope),
                        'slope_threshold': slope_threshold,
                        'samples_used': len(item_summaries)
                    }
```

### Ordinal Group-Level (Lines 2411-2484)

**Structure**:
```python
elif score_type == 'ordinal':
    # 1. Aggregate data
    all_ord_scores = []
    for item_id in item_ids:
        df_item = df_work[df_work['sample_id_num'] == item_id]
        all_ord_scores.extend(df_item[score_column].tolist())

    # 2. Determine conservatism
    current_perf_normalized = current_perf_estimate / ordinal_max_score
    current_conservatism = conservatism if current_perf_normalized < low_perf_threshold else 1.0

    # 3. Compute CI (modal/entropy/hybrid)
    group_entropy_history = stabilization_history.get('entropy_history', [])

    if ordinal_inference == 'modal':
        lo, hi, width = _ordinal_ci_adaptive(...)
    elif ordinal_inference == 'entropy':
        lo, hi, width, diagnostics = _ordinal_entropy_ci_adaptive(...)
    elif ordinal_inference == 'hybrid':
        should_stop_group, reason_group, diagnostics_group = _ordinal_hybrid_stopping_criterion(...)

    # 4. Check width criterion (NO HISTORY APPEND)
    if width < delta_cap:
        stop_this_grouping.append(grouping_name)
        metadata['group_stopping_reason'] = {
            'reason': 'ordinal_modal_ci_width',  # or entropy/hybrid
            'ci_width': float(width),
            'threshold': delta_cap,
            'samples_used': len(item_summaries)
        }

    # 5. Store entropy history
    stabilization_history['entropy_history'] = group_entropy_history

    # NO STABILIZATION CHECK
```

### Continuous Group-Level (Lines 2486-2567)

**Structure**:
```python
elif score_type in ['continuous_01', 'continuous_bounded']:
    # 1. Aggregate data
    all_continuous_scores = []
    for item_id in item_ids:
        df_item = df_work[df_work['sample_id_num'] == item_id]
        all_continuous_scores.extend(df_item[score_column].values)

    all_continuous_scores = np.array(all_continuous_scores)

    # 2. Determine conservatism
    current_perf_estimate = np.mean(all_continuous_scores)
    normalized_perf = current_perf_estimate / upper_bound
    current_conservatism = conservatism if normalized_perf < low_perf_threshold else 1.0

    # 3. Compute CI
    lo, hi, width = _continuous_bounded_ci_adaptive(
        all_continuous_scores,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        cred_level=cred_level,
        conservatism=current_conservatism,
        low_perf_threshold=low_perf_threshold
    )

    # 4. Normalize width and append to history
    width_normalized = width / (upper_bound - lower_bound)
    stabilization_history['ci_width_history'].append(float(width_normalized))

    # 5. Check width criterion
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

    # 6. Check stabilization criterion (SAME AS BINARY)
    if len(stabilization_history['ci_width_history']) >= stab_window:
        recent_widths = stabilization_history['ci_width_history'][-stab_window:]
        slope = np.polyfit(range(len(recent_widths)), recent_widths, 1)[0]
        stabilization_history['ci_slope_history'].append(float(slope))

        slope_threshold = CI_delta / current_conservatism if normalized_perf < low_perf_threshold else CI_delta

        if abs(slope) <= slope_threshold and len(stabilization_history['ci_slope_history']) >= 4:
            recent_slopes = stabilization_history['ci_slope_history'][-3:]
            slope_slopes = np.polyfit(range(len(recent_slopes)), recent_slopes, 1)[0]

            if slope_slopes >= 0:
                if normalized_perf >= low_perf_threshold:
                    stop_this_grouping.append(grouping_name)
                    metadata['group_stopping_reason'] = {
                        'reason': 'continuous_bounded_stabilization',
                        'slope': float(slope),
                        'slope_threshold': slope_threshold,
                        'samples_used': len(item_summaries)
                    }
                elif abs(slope) <= slope_threshold / 2:
                    stop_this_grouping.append(grouping_name)
                    metadata['group_stopping_reason'] = {
                        'reason': 'continuous_bounded_stabilization_low_perf',
                        'slope': float(slope),
                        'slope_threshold': slope_threshold,
                        'samples_used': len(item_summaries)
                    }
```

### Group-Level Consistency Analysis

| Aspect | Binary | Ordinal | Continuous | Consistent? |
|--------|--------|---------|------------|-------------|
| **1. Data Aggregation** | From item_summaries | From df_work | From df_work | ⚠️ Binary different |
| **2. Performance Calc** | From item_summaries | `estimate/max` | `mean/upper` | ✅ Yes (all normalized) |
| **3. Conservatism Logic** | `< threshold` | `< threshold` | `< threshold` | ✅ Yes |
| **4. CI Function** | PyMC hierarchical | Modal/Entropy/Hybrid | Beta+MoM | ✅ Yes (type-appropriate) |
| **5. Width Comparison** | Direct | Direct | **Normalized** | ⚠️ **Different but correct** |
| **6. History Append** | ✅ Yes | ❌ No | ✅ **Yes (normalized)** | ⚠️ Ordinal doesn't stabilize |
| **7. Stabilization Check** | ✅ Yes | ❌ No | ✅ **Yes** | ⚠️ Ordinal doesn't stabilize |
| **8. Metadata Fields** | reason, width, effective_width, threshold | reason, width, threshold | reason, width, **width_norm**, threshold, **bounds** | ⚠️ Different detail levels |

**Issues Found**:

1. ⚠️ **Data Aggregation**: Binary uses `item_summaries`, ordinal/continuous use `df_work`
   - **Analysis**: Binary needs structured data for hierarchical model
   - **Status**: ✅ CORRECT - Appropriate for each method

2. ⚠️ **History Tracking**: Ordinal doesn't append to `ci_width_history`
   - **Analysis**: Ordinal has no stabilization, so no width history needed
   - **Status**: ✅ CORRECT - Intentional design choice

3. ⚠️ **Stabilization**: Binary and continuous have it, ordinal doesn't
   - **Analysis**: Binary/continuous have continuous metrics, ordinal has discrete
   - **Status**: ✅ CORRECT - Makes sense for continuous metrics

✅ **Overall Status**: CONSISTENT with intentional differences

---

## 5. Metadata Structure Comparison

### Sample-Level Metadata

**Binary**:
```python
{
    'reason': 'ci_width',
    'ci_width': float,
    'threshold': float,
    'epochs_used': int
}
```

**Ordinal**:
```python
{
    'reason': 'ordinal_modal_ci_width',  # or entropy/hybrid
    'ci_width': float,
    'threshold': float,
    'epochs_used': int,
    'diagnostics': dict  # entropy/hybrid only
}
```

**Continuous**:
```python
{
    'reason': 'continuous_bounded_ci_width',
    'ci_width': float,
    'ci_width_normalized': float,  # ← EXTRA
    'threshold': float,
    'epochs_used': int,
    'bounds': {'lower': float, 'upper': float}  # ← EXTRA
}
```

### Group-Level Metadata

**Binary**:
```python
{
    'reason': 'ci_width' or 'ci_stabilization',
    'ci_width': float,
    'effective_width': float,  # width case only
    'threshold': float,  # width case only
    'slope': float,  # stabilization case only
    'slope_threshold': float,  # stabilization case only
    'samples_used': int
}
```

**Ordinal**:
```python
{
    'reason': 'ordinal_modal_ci_width',  # or entropy/hybrid
    'ci_width': float,
    'threshold': float,
    'samples_used': int,
    'diagnostics': dict  # entropy/hybrid only
}
```

**Continuous**:
```python
{
    'reason': 'continuous_bounded_ci_width' or 'continuous_bounded_stabilization',
    'ci_width': float,
    'ci_width_normalized': float,  # ← EXTRA
    'threshold': float,  # width case only
    'slope': float,  # stabilization case only
    'slope_threshold': float,  # stabilization case only
    'samples_used': int,
    'bounds': {'lower': float, 'upper': float}  # ← EXTRA
}
```

### Metadata Consistency Analysis

| Field | Binary | Ordinal | Continuous | Issue? |
|-------|--------|---------|------------|--------|
| `reason` | ✅ | ✅ | ✅ | ✅ All unique |
| `ci_width` (raw) | ✅ | ✅ | ✅ | ✅ All have it |
| `ci_width_normalized` | ❌ | ❌ | ✅ | ⚠️ Only continuous |
| `effective_width` | ✅ | ❌ | ❌ | ⚠️ Only binary |
| `threshold` | ✅ | ✅ | ✅ | ✅ All have it |
| `epochs_used` / `samples_used` | ✅ | ✅ | ✅ | ✅ All have it |
| `bounds` | ❌ | ❌ | ✅ | ⚠️ Only continuous |
| `slope` | ✅ (stab) | ❌ | ✅ (stab) | ⚠️ Ordinal no stab |
| `diagnostics` | ❌ | ✅ (entropy) | ❌ | ⚠️ Only ordinal entropy |

**Analysis**:
- Each route includes **type-specific metadata** appropriate for its method
- `ci_width_normalized` and `bounds` are continuous-specific for transparency
- `effective_width` is binary-specific (applies conservatism differently)
- `diagnostics` is ordinal entropy/hybrid-specific

✅ **Status**: CONSISTENT - Differences provide appropriate type-specific information

---

## 6. Edge Cases Handling

### Empty Data

**Binary**: Returns 0.0, 1.0, 1.0 from `_beta_ci_adaptive`
**Ordinal**: Returns 0.0, 1.0, 1.0 from `_ordinal_ci_adaptive`
**Continuous**: Returns lower_bound, upper_bound, range from `_continuous_bounded_ci_adaptive`

✅ **Status**: CONSISTENT - All return full range

### Single Observation

**Binary**: Handles in `_beta_ci_adaptive` with wide interval
**Ordinal**: Handles in `_ordinal_ci_adaptive` with conservative width
**Continuous**: Handles in `_continuous_bounded_ci_adaptive` with 50% range width

✅ **Status**: CONSISTENT - All conservative

### Zero Variance

**Binary**: Not applicable (discrete)
**Ordinal**: Handled implicitly (modal will be stable)
**Continuous**: Explicit handling with tight interval based on n

✅ **Status**: CONSISTENT - Appropriate for each type

---

## 7. Critical Differences Summary

### Intentional Differences (Correct)

1. **Data Types**:
   - Binary: Integer counts
   - Ordinal: Integer categories
   - Continuous: Float values

2. **Width Normalization**:
   - Binary: Native [0, 1] scale, no normalization needed
   - Ordinal: Pre-normalized in `_ordinal_ci_adaptive`
   - Continuous: **Manual normalization required** in calling code

3. **Stabilization**:
   - Binary: YES (continuous metric)
   - Ordinal: NO (discrete, modal can jump)
   - Continuous: YES (continuous metric)

4. **Item Summaries**:
   - Binary/Ordinal: `{'successes', 'trials'}`
   - Continuous: `{'mean', 'count'}`

5. **Metadata Detail**:
   - Each route includes type-specific fields
   - Continuous includes normalization transparency

### No Issues Found

✅ All differences are **intentional and appropriate**
✅ Structure is **parallel** across all three routes
✅ Logic flow is **consistent**
✅ Edge cases **handled appropriately**

---

## 8. Recommendations

### Current Status: EXCELLENT ✅

The function demonstrates **excellent internal consistency** with intentional, well-justified differences between routes.

### Minor Enhancement Opportunities (Optional)

1. **Documentation**: Add comments explaining why continuous has extra metadata fields
2. **Type Hints**: Consider adding more specific type hints for `item_summaries` based on score_type
3. **Factoring**: The stabilization logic is duplicated between binary and continuous - could be factored out

### Suggested Comment Additions

```python
# Line 2307: Continuous metadata includes normalization info
metadata['sample_stopping_reasons'][str(original_sample_id)] = {
    ...
    'ci_width_normalized': float(width_normalized),  # For transparency
    'bounds': {'lower': lower_bound, 'upper': upper_bound}  # For context
}

# Line 2326: Performance estimation differs by score type
# Binary/ordinal use successes/trials structure
# Continuous uses mean/count structure
if score_type in ['binary', 'ordinal']:
    ...
else:  # continuous
    ...
```

---

## 9. Final Verdict

### ✅ INTERNAL CONSISTENCY: EXCELLENT

**Summary**:
- **Structure**: Parallel across all three routes ✅
- **Parameter Handling**: Consistent ✅
- **Sample Logic**: Consistent with appropriate adaptations ✅
- **Group Logic**: Consistent with appropriate adaptations ✅
- **Metadata**: Type-specific but well-structured ✅
- **Edge Cases**: All handled appropriately ✅
- **Differences**: All intentional and justified ✅

**Conclusion**: The `optimal_stopping_live_single()` function maintains excellent internal consistency across all three inference routes. All differences are intentional, well-justified, and appropriate for the different data types being handled.

---

**Review Date**: 2025-11-19
**Reviewer Assessment**: ✅ Production-Ready
**Issues Found**: 0 critical, 0 moderate, 0 minor
**Recommendations**: Optional enhancements only
