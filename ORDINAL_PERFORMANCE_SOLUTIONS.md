# Ordinal Hybrid Mode Performance: Solution Analysis

**Problem**: Ordinal hybrid inference takes 7.6 hours for 500 samples (84× slower than binary)
**Root Cause**: Hybrid mode runs BOTH modal (bootstrap) AND entropy (PyMC OrderedLogistic) for every inference call

---

## Current Performance Baseline

### Dataset 2 (Ordinal Hybrid) - Latest Run
- **Runtime**: 7.6 hours (27,212 seconds)
- **Trials**: 1,935 / 5,000
- **Efficiency**: 61.3%
- **Inference calls**: 95
- **Avg inference time**: 286 seconds (4.8 minutes)
- **Scaling**: Non-linear (early: ~100s, late: ~1,000s)

### Comparison to Other Modes
- **Binary**: 8.3 minutes (498s) - **91× faster**
- **Continuous**: 8.0 minutes (478s) - **94× faster**

---

## Solution 1: Switch to Modal-Only Mode

### Description
Use `ordinal_inference='modal'` to skip entropy inference entirely. Only run bootstrap modal CI analysis.

### How It Works
```python
manager = OptimalStoppingManager(
    ordinal_inference='modal',  # Skip entropy inference
    ...
)
```

**Algorithm**:
1. Compute modal category via bootstrap (fast: ~1-5 seconds)
2. Check if modal CI width < threshold
3. Stop immediately if criterion met
4. No entropy validation, no stabilization check

### Expected Performance
- **Speedup**: 10-20× faster
- **Expected runtime**: 30-60 minutes (vs 7.6 hours)
- **Reason**: Eliminates PyMC MCMC sampling (500 draws × 4 chains × N items)

### Pros
✅ Massive speedup (10-20×)
✅ Already implemented in codebase
✅ Zero code changes required
✅ Simple to use (one parameter change)
✅ Works well for peaked distributions

### Cons
❌ Less conservative stopping (no entropy validation)
❌ May stop prematurely on false peaks
❌ No protection against multimodal distributions
❌ Lower statistical rigor

### When to Use
- **Good for**: Clear, peaked distributions (most responses in 1-2 categories)
- **Good for**: Large datasets (>100 samples) where speed matters
- **Avoid for**: Highly uncertain/diffuse distributions
- **Avoid for**: Critical decisions requiring maximum rigor

### Risk Assessment
- **Risk Level**: Low-Medium
- **False positive rate**: May increase slightly (premature stopping)
- **Mitigation**: Can increase `delta_item` threshold to be more conservative

---

## Solution 2: Adaptive Hybrid Mode (Smart Pathway Selection)

### Description
Run modal inference first; only run entropy inference if modal is inconclusive or suggests false peak.

### How It Works
```python
# PSEUDO-CODE (requires implementation)
if modal_ci_width < threshold:
    # Quick check: Is this a true peak or false peak?
    if modal_category_probability > 0.7:  # High confidence
        return True, 'modal_ci_narrow_validated'
    else:
        # Low confidence - run entropy to verify
        entropy = run_entropy_inference()
        if entropy < entropy_threshold:
            return True, 'modal_ci_narrow_validated'
        else:
            # Fall through to stabilization
            ...
```

### Expected Performance
- **Speedup**: 5-10× faster
- **Expected runtime**: 1-2 hours (vs 7.6 hours)
- **Reason**: Only runs entropy when needed (e.g., 30% of cases)

### Pros
✅ Significant speedup (5-10×)
✅ Maintains statistical rigor when needed
✅ Balances speed vs accuracy
✅ Reduces computational waste

### Cons
❌ Requires code changes
❌ Needs heuristic tuning (probability threshold)
❌ More complex stopping logic
❌ Still slower than modal-only

### Implementation Complexity
- **Effort**: Medium (2-4 hours)
- **Changes**: Modify `_ordinal_hybrid_stopping_criterion()` in `ordinal_model.py`
- **Testing**: Need to validate heuristic works well

---

## Solution 3: Increase Reanalysis Interval

### Description
Run inference less frequently by increasing `reanalysis_interval`.

### How It Works
```python
manager = OptimalStoppingManager(
    reanalysis_interval=50,  # Was 25
    ...
)
```

**Impact**:
- 25 → 50: Half as many inference calls
- 25 → 100: Quarter as many inference calls

### Expected Performance
- **Speedup**: 2-4× fewer inference calls
- **Expected runtime**:
  - `interval=50`: ~4 hours (50% reduction)
  - `interval=100`: ~2 hours (75% reduction)

### Pros
✅ Easy to implement (parameter change only)
✅ Predictable speedup
✅ Maintains all statistical properties
✅ No code changes needed

### Cons
❌ Less responsive to changes
❌ May delay stopping decisions
❌ Could slightly reduce efficiency
❌ Doesn't address fundamental scaling issue

### When to Use
- **Good for**: Any ordinal hybrid scenario where latency is acceptable
- **Combine with**: Modal-only or adaptive hybrid for best results

### Risk Assessment
- **Risk Level**: Very Low
- **Impact on efficiency**: Minimal (maybe 1-3% worse)
- **Recommendation**: Always increase to 50-100 for large datasets

---

## Solution 4: Optimize Hybrid Implementation

### Description
Technical optimizations to make hybrid inference itself faster.

### Potential Optimizations

#### 4A. Cache Bootstrap Results
**Idea**: Reuse bootstrap samples across inference calls
```python
# Cache bootstrap samples for modal CI
bootstrap_cache = {
    'samples': [...],  # Reuse across calls
    'n_samples': 10000
}
```
**Speedup**: 20-30% for modal component
**Complexity**: Low-Medium

#### 4B. Reduce MCMC Samples
**Current**: 500 draws × 4 chains = 2,000 samples
**Proposed**: 200 draws × 2 chains = 400 samples
**Speedup**: 5× faster entropy inference
**Risk**: Lower precision (may be acceptable)

#### 4C. Incremental Inference
**Idea**: Don't reprocess all data on each call; only process new data
```python
# Instead of:
inference(all_90_trials)  # Process 90 trials
inference(all_112_trials)  # Reprocess 90 + 22 new

# Do:
inference_incremental(new_22_trials, previous_state)
```
**Speedup**: 3-5× faster (only process Δ data)
**Complexity**: High (major architectural change)

#### 4D. Approximate Entropy
**Idea**: Use faster approximation for entropy instead of full MCMC
```python
# Instead of full OrderedLogistic MCMC:
approximate_entropy = compute_empirical_entropy(scores)
```
**Speedup**: 50-100× faster for entropy component
**Risk**: Less accurate, may miss distributional nuances

### Expected Performance (Combined)
- **Speedup**: 3-8× faster (depends on which optimizations)
- **Expected runtime**: 1-3 hours

### Pros
✅ Improves fundamental algorithm performance
✅ Benefits all users of ordinal hybrid
✅ Maintains statistical rigor (mostly)
✅ Long-term solution

### Cons
❌ High implementation complexity
❌ Significant development time (1-2 weeks)
❌ Requires extensive testing
❌ Risk of introducing bugs

---

## Solution 5: Smart Inference Scheduling

### Description
Dynamically adjust inference frequency based on convergence state.

### How It Works
```python
# Pseudo-code
if grouping_is_stabilizing:
    # Reduce inference frequency as we get closer to stopping
    current_interval = min(reanalysis_interval * 2, 100)
elif grouping_is_diverging:
    # Increase frequency if things are changing
    current_interval = max(reanalysis_interval // 2, 10)
```

**Adaptive strategy**:
- Start with `interval=25` (responsive)
- If CI width decreasing: Keep at 25
- If CI width stable: Increase to 50
- If CI width very stable: Increase to 100

### Expected Performance
- **Speedup**: 2-3× fewer late-stage calls
- **Expected runtime**: ~5 hours (35% reduction)
- **Reason**: Most inference time is in late-stage calls when N is large

### Pros
✅ Intelligent resource allocation
✅ Maintains early responsiveness
✅ Reduces waste on near-converged groupings
✅ No user configuration needed

### Cons
❌ Complex scheduling logic
❌ Harder to predict behavior
❌ Needs careful tuning
❌ Medium implementation complexity

---

## Solution 6: Early Exit Optimization (Already Implemented)

### Description
Skip inference for groupings that have already stopped.

### Status in Codebase
✅ **Already implemented** in `rule.py` as "Optimization #1"

```python
# From rule.py:2718
if grouping_name in _stopped_groupings:
    logger.info(f"Skipping inference for '{grouping_name}' - grouping already stopped")
    continue
```

### Performance Impact
- **Current**: Saves ~30-50% of late-stage inference
- **In Dataset 2**: All 5 groupings stopped eventually, so helped late in run

### Limitation
Only helps AFTER grouping stops. Doesn't help with the expensive calls BEFORE stopping.

---

## Solution Comparison Matrix

| Solution | Speedup | Runtime | Effort | Risk | Rigor | Recommendation |
|----------|---------|---------|--------|------|-------|----------------|
| **1. Modal-only** | 10-20× | 30-60 min | None | Medium | Lower | ✅ **Best for speed** |
| **2. Adaptive hybrid** | 5-10× | 1-2 hrs | Medium | Low | High | ✅ **Best balance** |
| **3. Increase interval** | 2-4× | 2-4 hrs | None | Very low | Same | ✅ **Easy win** |
| **4. Optimize hybrid** | 3-8× | 1-3 hrs | High | Medium | Same | ⚠️ Long-term only |
| **5. Smart scheduling** | 2-3× | ~5 hrs | Medium | Low | Same | ⚠️ Diminishing returns |
| **6. Early exit** | 1.5× | ~5 hrs | None | None | Same | ✅ Already done |

---

## Recommended Approach: Phased Strategy

### Phase 1: Immediate (No Code Changes)
**Combine Solutions 1 & 3**

For large-scale testing and production:
```python
manager = OptimalStoppingManager(
    ordinal_inference='modal',      # Solution 1
    reanalysis_interval=50,          # Solution 3
    delta_item=0.10,                 # Slightly more conservative to offset modal-only
    ...
)
```

**Expected performance**:
- Modal-only: 10-20× faster
- Interval=50: 2× fewer calls
- **Combined**: 20-40× faster
- **Expected runtime**: 15-30 minutes (vs 7.6 hours)

**Risk mitigation**:
- Increase `delta_item` threshold to be more conservative
- Monitor false positive rate in production
- Can revert to hybrid for critical evaluations

---

### Phase 2: Near-Term (2-4 hours development)
**Implement Solution 2: Adaptive Hybrid**

Add smart pathway selection:
```python
def _adaptive_hybrid_stopping_criterion(...):
    # Run modal first (cheap)
    modal_ci_width = compute_modal_ci(...)

    if modal_ci_width < threshold:
        # Quick heuristic check
        modal_prob = get_modal_probability(scores)

        if modal_prob > 0.7:  # High confidence - trust modal
            return True, 'modal_ci_narrow_fast'
        else:
            # Low confidence - validate with entropy
            entropy = compute_entropy(...)
            if entropy < entropy_threshold:
                return True, 'modal_ci_narrow_validated'
            # else: continue to stabilization

    # Full hybrid logic for unclear cases
    ...
```

**Expected performance**:
- 5-10× faster than current hybrid
- ~1-2 hours runtime
- Maintains rigor when needed

---

### Phase 3: Long-Term (1-2 weeks development)
**Implement Solution 4: Optimize Hybrid**

- Incremental inference (biggest win)
- Cache bootstrap results
- Reduce MCMC samples (e.g., 300 draws × 2 chains)
- Parallel sample-level inference

**Expected performance**:
- 5-10× faster than current hybrid
- Hybrid becomes viable for all scales

---

## Validation Plan

### Test 1: Modal-Only Performance
```bash
# Run Dataset 2 with modal-only
ordinal_inference='modal'
reanalysis_interval=50
```
**Expected**: 15-30 minutes, 55-65% efficiency

### Test 2: Stopping Decision Quality
Compare modal-only vs hybrid on same data:
- Do they stop at similar sample counts?
- Are final estimates within ±0.05?
- Any cases where modal stopped prematurely?

### Test 3: Adaptive Hybrid (After Implementation)
Run with adaptive hybrid and compare:
- Runtime vs modal-only and full hybrid
- Percentage of cases using fast path vs validation
- Decision quality metrics

---

## Production Recommendations

### For Current Release

**Default Configuration**:
```python
# For datasets < 50 samples per grouping
ordinal_inference='hybrid'
reanalysis_interval=25

# For datasets 50-200 samples per grouping
ordinal_inference='modal'
reanalysis_interval=50
delta_item=0.12  # Slightly more conservative

# For datasets > 200 samples per grouping
ordinal_inference='modal'
reanalysis_interval=100
delta_item=0.15  # More conservative
```

**Documentation Warning**:
```
⚠️ PERFORMANCE NOTE: Ordinal hybrid inference scales non-linearly.
For large datasets (>50 samples), consider:
  - Using ordinal_inference='modal' for 10-20× speedup
  - Increasing reanalysis_interval to 50-100
  - See performance guide for trade-offs
```

### For Next Release

Implement adaptive hybrid as default:
```python
ordinal_inference='adaptive'  # New mode
```

Users get best of both worlds automatically.

---

## Cost-Benefit Analysis

### Do Nothing (Current State)
- **Cost**: 7.6 hours per 500-sample evaluation
- **Benefit**: Maximum statistical rigor
- **Verdict**: ❌ Unacceptable for production

### Solution 1: Modal-Only
- **Cost**: Slightly less conservative (5-10% more false positives?)
- **Benefit**: 20-40× speedup with interval increase
- **Verdict**: ✅ **Recommended for immediate use**

### Solution 2: Adaptive Hybrid
- **Cost**: 2-4 hours development + testing
- **Benefit**: 5-10× speedup while maintaining rigor
- **Verdict**: ✅ **Recommended for next sprint**

### Solution 4: Optimize Hybrid
- **Cost**: 1-2 weeks development
- **Benefit**: Makes hybrid viable at all scales
- **Verdict**: ⚠️ **Consider for future release**

---

## Conclusion

**For the large-scale retest**, I recommend:

1. **Run with modal-only mode** first to validate speedup:
   ```python
   ordinal_inference='modal'
   reanalysis_interval=50
   ```
   Expected: ~15-30 minutes, 55-65% efficiency

2. **Compare results** to hybrid run:
   - Did it stop at similar sample counts?
   - Is efficiency comparable?
   - Any obvious false positives?

3. **If modal-only looks good**:
   - Document as recommended approach for large datasets
   - Add performance warnings to docs
   - Release with modal-only as suggested default for >50 samples

4. **Implement adaptive hybrid** in next release for best of both worlds

**The path forward is clear**: Modal-only gives immediate relief, adaptive hybrid provides the long-term solution.
