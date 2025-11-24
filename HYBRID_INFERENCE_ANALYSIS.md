# Hybrid Inference Analysis - Test 1.1.2c Failure

**Date:** 2025-11-24
**Status:** Investigation Complete
**Test:** test_1_1_2c_ordinal_hybrid_peaked

---

## Executive Summary

Test 1.1.2c fails because **hybrid inference requires at least 3 inference runs** to accumulate sufficient entropy history for stabilization assessment, but the test configuration only provides **2 inference runs**.

---

## Hybrid Inference Logic (ordinal_model.py:449-649)

### Two Pathways to Stopping

**Pathway 1: Modal CI with Entropy Validation** (lines 551-585)
- **Condition**: `modal_width < delta_item` AND `entropy_median <= entropy_threshold`
- **Fast stopping** for peaked distributions
- **False peak protection**: If entropy is high despite narrow modal CI, fall through to Pathway 2

**Pathway 2: Entropy Stabilization** (lines 587-649)
- **Condition**: Relative change in entropy CI width < `stabilization_threshold` (0.002 = 0.2%)
- **Requirements**:
  - At least `min_epochs_for_stabilization` = 3 history entries
  - Compute relative change between last 2 width values
  - If change < 0.002, stop with 'entropy_stabilized'
- **Conservative stopping** for diffuse or uncertain distributions

---

## Test 1.1.2c Configuration

```python
# Stopping parameters
optstop_params = {
    'delta_item': 0.20,    # Sample-level threshold
    'delta_cap': 0.15,     # Group-level threshold (used for group stopping)
    'cred_level': 0.85,    # 85% credible intervals
    'conservatism': 3,
    'draws': 1000,
    'tune': 1000,
}

# Manager configuration
manager = OptimalStoppingManager(
    optstop_params=optstop_params,
    grouping_columns=['model', 'task'],
    reanalysis_interval=10,      # Inference every 10 samples
    min_samples_per_grouping=5,  # First inference at sample 5
    ordinal_tasks=['rating'],
    ordinal_max_score=5,
    ordinal_inference='hybrid',
)

# Data configuration
n_samples = 15              # 15 samples total
epochs_per_sample = 8       # 8 epochs per sample
total_planned = 120         # 15 × 8 = 120 trials

# Data characteristics
mode_value = 4
concentration = 0.88        # 88% of scores at mode=4 (highly peaked)
```

---

## Why Test Fails: Inference Frequency Mismatch

### Bridge Inference Schedule

With `reanalysis_interval=10` and `min_samples_per_grouping=5`:
- **Sample 5**: Not enough samples (need 10 for first reanalysis)
- **Sample 10**: ✅ **First inference run** (counter = 10)
- **Sample 15**: ✅ **Second inference run** (counter = 15, test complete)

**Result**: Only **2 inference runs** for the entire test.

### Group-Level Stopping Requirements

From `rule.py:2591-2654`, hybrid group-level stopping:

```python
# Line 2591: Load entropy history from stabilization_history
group_entropy_history = stabilization_history.get('entropy_history', [])

# Line 2633-2644: Call hybrid stopping criterion
should_stop_group, reason_group, diagnostics_group = _ordinal_hybrid_stopping_criterion(
    np.array(all_ord_scores),
    ordinal_max_score=ordinal_max_score,
    delta_item=delta_cap,  # Use delta_cap (0.15) for group-level
    cred_level=cred_level,
    entropy_history=group_entropy_history,  # Modified in-place
    entropy_threshold=entropy_threshold,
    ...
)

# Line 2654: Save back to stabilization_history
stabilization_history['entropy_history'] = group_entropy_history
```

**The Problem:**
- Pathway 1 checks: `modal_width < 0.15` AND `entropy_median <= 1.5`
  - With peaked data, likely passes entropy check
  - But modal CI width may be >= 0.15 (depends on sample size at inference time)

- Pathway 2 checks: `len(entropy_history) >= 3`
  - With only 2 inference runs, `len(group_entropy_history) = 2`
  - **FAILS** at line 593 in ordinal_model.py:
    ```python
    if len(entropy_history) < min_epochs_for_stabilization:
        ...
        return False, 'continue_insufficient_history', diagnostics
    ```

### Sample-Level Stopping (Less Relevant)

Each sample processes 8 epochs sequentially, but:
- `entropy_history` is reset per sample (rule.py:1041)
- Each sample starts with empty history
- With only 8 epochs per sample, unlikely to meet stopping criteria before group-level

---

## Solution Options

### Option A: Increase Inference Frequency ✅ **RECOMMENDED**

```python
manager = OptimalStoppingManager(
    ...
    reanalysis_interval=3,       # Inference every 3 samples
    min_samples_per_grouping=3,  # Start at sample 3
    ...
)
```

**Inference schedule:**
- Sample 3: ✅ First run
- Sample 6: ✅ Second run
- Sample 9: ✅ Third run
- Sample 12: ✅ Fourth run
- Sample 15: ✅ Fifth run

**Result**: 5 inference runs → entropy_history has 5 entries → stabilization can be assessed

---

### Option B: Relax Stopping Criteria

```python
optstop_params = {
    'delta_item': 0.25,    # More lenient (was 0.20)
    'delta_cap': 0.25,     # More lenient (was 0.15)
    'cred_level': 0.80,    # Lower confidence (was 0.85)
    ...
}
```

**Effect**: Easier for Pathway 1 (modal CI) to trigger even with limited samples.

---

### Option C: Increase Sample Count

```python
n_samples = 21  # Was 15
```

**Inference schedule** (with interval=10):
- Sample 10: ✅ First run
- Sample 20: ✅ Second run
- Sample 21: ✅ Third run (at completion)

**Result**: 3 inference runs → minimum for stabilization assessment

---

## Recommendation

**Use Option A** (increase inference frequency):
- Most aligned with test intent (validate hybrid stopping with peaked data)
- Provides sufficient history for both pathways
- Maintains realistic stopping thresholds
- Allows comprehensive testing of entropy stabilization logic

**Implementation:**
```python
# Test 1.1.2c modification
manager = OptimalStoppingManager(
    optstop_params=optstop_params,
    grouping_columns=['model', 'task'],
    reanalysis_interval=3,       # Changed from 10
    min_samples_per_grouping=3,  # Changed from 5
    ordinal_tasks=['rating'],
    ordinal_max_score=5,
    ordinal_inference='hybrid',
)
```

---

## Additional Observations

### Stabilization History Persistence ✅

The bridge correctly maintains `stabilization_history` across calls:
- **early_stopping.py:957**: Load history from `self._stabilization_histories[grouping_name]`
- **early_stopping.py:1043**: Save updated history back after inference
- **rule.py:2591**: Load `group_entropy_history` from `stabilization_history['entropy_history']`
- **rule.py:2654**: Save updated `group_entropy_history` back

**Conclusion**: Entropy history DOES persist across multiple calls to `optimal_stopping_live_single()`. The issue is purely insufficient inference frequency.

### Test 1.1.2d (Hybrid Diffuse) Expected Behavior

Test 1.1.2d uses:
- Diffuse data (uniform distribution across 1-5)
- 15 samples × 10 epochs = 150 trials
- Same inference intervals (10, 15)

**Prediction**: Will also complete all trials (0% efficiency) because:
1. Diffuse data has high entropy → Pathway 1 won't trigger
2. Only 2 inference runs → Pathway 2 can't assess stabilization
3. Test correctly expects low/no efficiency (documented in test docstring)

---

## Files Referenced

- `optstop/ordinal_model.py`: Lines 449-649 (`_ordinal_hybrid_stopping_criterion`)
- `optstop/rule.py`: Lines 1019-1041, 2591-2654 (hybrid group-level stopping)
- `optstop/early_stopping.py`: Lines 957, 1043 (stabilization history management)
- `tests/test_bridge_integration_ordinal_discrete.py`: Lines 458-603 (test 1.1.2c)

---

**Analysis Complete**
**Next Steps**: Update test configuration and re-run validation
