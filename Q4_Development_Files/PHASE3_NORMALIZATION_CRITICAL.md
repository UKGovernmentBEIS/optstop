# ⚠️ CRITICAL: Width Normalization Requirement for Phase 3

**Status**: ACTION REQUIRED
**Priority**: HIGH
**Impact**: Without this, continuous inference will NEVER stop

---

## The Problem

The `_continuous_bounded_ci_adaptive()` function returns widths in **ORIGINAL scale**:
- Binary aggregated [0, 1]: width might be 0.05
- Ordinal aggregated [0, 10]: width might be 0.5

Without normalization:
- Binary: `0.05 < delta_item (0.05)` ✅ Works
- Ordinal: `0.5 < delta_item (0.05)` ❌ NEVER stops!

---

## Required Normalization

### At Sample Level

```python
elif score_type in ['continuous_01', 'continuous_bounded']:
    accumulated_scores = df_item[score_column].values

    # Compute CI
    lo, hi, width = _continuous_bounded_ci_adaptive(
        accumulated_scores,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        cred_level=cred_level,
        conservatism=current_conservatism,
        low_perf_threshold=low_perf_threshold
    )

    # ⚠️ CRITICAL: Normalize width before comparison
    width_normalized = width / (upper_bound - lower_bound)

    # Check width criterion with NORMALIZED width
    if width_normalized < delta_item:
        stop_sample_ids.append(f"{grouping_name}:::{original_sample_id}")
        metadata['sample_stopping_reasons'][str(original_sample_id)] = {
            'reason': 'continuous_bounded_ci_width',
            'ci_width': float(width),              # Store raw width
            'ci_width_normalized': float(width_normalized),  # Store normalized
            'threshold': delta_item,
            'epochs_used': len(accumulated_scores),
            'bounds': {'lower': lower_bound, 'upper': upper_bound}
        }
```

### At Group Level

```python
elif score_type in ['continuous_01', 'continuous_bounded']:
    # Aggregate all scores
    all_continuous_scores = []
    for item_id in item_ids:
        df_item = df_work[df_work['sample_id_num'] == item_id]
        all_continuous_scores.extend(df_item[score_column].values)

    all_continuous_scores = np.array(all_continuous_scores)

    # Compute CI
    lo, hi, width = _continuous_bounded_ci_adaptive(
        all_continuous_scores,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        cred_level=cred_level,
        conservatism=current_conservatism,
        low_perf_threshold=low_perf_threshold
    )

    # ⚠️ CRITICAL: Normalize width before comparison
    width_normalized = width / (upper_bound - lower_bound)

    # Append NORMALIZED width to history
    stabilization_history['ci_width_history'].append(float(width_normalized))

    # Check width criterion with NORMALIZED width
    if width_normalized < delta_cap:
        stop_this_grouping.append(grouping_name)
        metadata['group_stopping_reason'] = {
            'reason': 'continuous_bounded_ci_width',
            'ci_width': float(width),                    # Store raw
            'ci_width_normalized': float(width_normalized),  # Store normalized
            'threshold': delta_cap,
            'samples_used': len(item_summaries),
            'bounds': {'lower': lower_bound, 'upper': upper_bound}
        }
```

---

## Verification

### Test Case 1: Binary Aggregated [0, 1]

```python
scores = np.array([0.8, 0.85, 0.9, 0.75, 0.95])
lo, hi, width = _continuous_bounded_ci_adaptive(scores, 0.0, 1.0)
# width ≈ 0.15 (raw)
width_normalized = width / (1.0 - 0.0) = 0.15 / 1.0 = 0.15
# Comparison: 0.15 < 0.05? No, keep collecting epochs
```

### Test Case 2: Ordinal Aggregated [0, 10]

```python
scores = np.array([8.2, 8.5, 8.7, 8.4, 8.9])
lo, hi, width = _continuous_bounded_ci_adaptive(scores, 0.0, 10.0)
# width ≈ 1.5 (raw)
width_normalized = width / (10.0 - 0.0) = 1.5 / 10.0 = 0.15
# Comparison: 0.15 < 0.05? No, keep collecting epochs
```

Both cases now use the **same normalized scale** [0, 1] for comparison.

---

## Comparison with Existing Routes

### Binary (Native [0, 1])
```python
width = _beta_ci_adaptive(...)  # Returns width in [0, 1]
if width < delta_item:  # Direct comparison ✅
```

### Ordinal (Pre-normalized in function)
```python
# Inside _ordinal_ci_adaptive() (ordinal_utils.py:137-140):
lo = lo_cat / ordinal_max_score  # Normalized to [0, 1]
hi = hi_cat / ordinal_max_score
width = hi - lo  # Already in [0, 1]

# In calling code:
if width < delta_item:  # Direct comparison ✅
```

### Continuous (Must normalize in calling code)
```python
width = _continuous_bounded_ci_adaptive(...)  # Returns width in [a, b]
width_normalized = width / (upper_bound - lower_bound)  # ⚠️ REQUIRED
if width_normalized < delta_item:  # Normalized comparison ✅
```

---

## Why Not Normalize Inside Function?

**Option 1**: Normalize inside `_continuous_bounded_ci_adaptive()` (like ordinal)
```python
# Return normalized width
width_normalized = (hi - lo) / (upper_bound - lower_bound)
return lo, hi, width_normalized
```

**Option 2**: Normalize in calling code (current approach)
```python
# Return raw width
width = hi - lo
return lo, hi, width

# Normalize in integration code
width_normalized = width / (upper_bound - lower_bound)
```

**Decision**: Keep Option 2 (normalize in calling code)

**Rationale**:
1. **Consistency with bounds**: Function returns values in original scale (lo, hi, width all in [a, b])
2. **Metadata clarity**: Can store both raw and normalized widths
3. **Flexibility**: Calling code has full control
4. **Debugging**: Easier to trace normalization step

**Trade-off**: Requires careful implementation in Phase 3, but more transparent.

---

## Checklist for Phase 3 Implementation

- [ ] Extract bounds: `lower_bound = bounds['lower']`, `upper_bound = bounds['upper']`
- [ ] Sample level: Normalize width before comparison with `delta_item`
- [ ] Sample level: Store both `ci_width` and `ci_width_normalized` in metadata
- [ ] Group level: Normalize width before comparison with `delta_cap`
- [ ] Group level: Append **normalized** width to `ci_width_history`
- [ ] Group level: Store both `ci_width` and `ci_width_normalized` in metadata
- [ ] Performance normalization: `normalized_perf = mean / upper_bound`
- [ ] Test with [0, 1] and [0, 10] bounds to verify correct behavior

---

## Testing Validation

After Phase 3 implementation, verify:

1. **Binary aggregated [0, 1]**: Stops appropriately (same behavior as discrete binary)
2. **Ordinal aggregated [0, 10]**: Stops appropriately (same behavior as discrete ordinal)
3. **Normalized widths**: Confirm stored widths are in [0, 1] range
4. **Stabilization**: CI width history contains normalized values in [0, 1]

---

**Created**: 2025-11-19
**Status**: FLAGGED - Implementation in progress
**Action**: Apply normalization at BOTH sample and group levels
