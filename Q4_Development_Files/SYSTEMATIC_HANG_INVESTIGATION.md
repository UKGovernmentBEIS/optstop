# Systematic Investigation: Test 4 Hang (Ordinal Hybrid Inference)

**Date**: 2025-11-21
**Status**: IN PROGRESS

---

## Investigation Strategy

Work through each potential cause systematically with specific tests and checks.

---

## Cause 7: Basic Structural/Design Errors (CHECK FIRST)

**Priority**: HIGH - Can reveal fundamental issues quickly

### 7.1 Check Function Signatures and Call Chain

**Question**: Are function signatures compatible throughout the call chain?

**Files to check**:
- `early_stopping.py:978-992` - Asyncio call to `optimal_stopping_live_single()`
- `rule.py:2021-2034` - `optimal_stopping_live_single()` signature
- `rule.py:1048-1061` - Call to `_ordinal_hybrid_stopping_criterion()`
- `ordinal_model.py:435-448` - `_ordinal_hybrid_stopping_criterion()` signature
- `ordinal_model.py:523-532` - Call to `_ordinal_entropy_ci_adaptive()`
- `ordinal_model.py:236-246` - `_ordinal_entropy_ci_adaptive()` signature

**Tests**:
- [ ] Trace parameter flow from early_stopping through entire chain
- [ ] Verify all required parameters are passed
- [ ] Check for parameter name mismatches
- [ ] Verify return value handling

### 7.2 Check for Infinite Loops or Recursion

**Question**: Could there be unintended loops in the logic?

**Locations to check**:
- `rule.py:2200-2320` - Sample-level stopping loop
- `rule.py:2322-2580` - Group-level stopping section
- `ordinal_model.py:435-567` - Hybrid stopping criterion logic
- `early_stopping.py:890-1144` - `_run_stopping_inference()` and update logic

**Tests**:
- [ ] Check for while loops without proper exit conditions
- [ ] Check for recursive calls
- [ ] Verify loop termination conditions
- [ ] Check for infinite retries or backoff logic

### 7.3 Check for Blocking Operations

**Question**: Are there synchronous operations that could block indefinitely?

**Locations**:
- `early_stopping.py:978` - `asyncio.to_thread()` wrapper
- `ordinal_model.py:360` - `pm.sample()` call
- Any file I/O, network calls, or locks

**Tests**:
- [ ] Identify all blocking operations
- [ ] Check if they have timeouts
- [ ] Verify thread pool isn't exhausted

### 7.4 Check Data Structure Growth

**Question**: Do any data structures grow unbounded?

**Locations**:
- `early_stopping.py:204` - `_stabilization_histories` dict
- `early_stopping.py:189` - `stopped_samples` list
- `early_stopping.py:201` - `_schedule_cache` dict
- `ordinal_model.py:324-333` - `model_cache` usage
- `rule.py:2028` - `stabilization_history` parameter

**Tests**:
- [ ] Check if histories are cleared/limited
- [ ] Verify caches have size limits
- [ ] Check for memory leaks

---

## Cause 1: MCMC Convergence Issues with Bimodal Data

**Priority**: HIGH - Most likely based on problem description

### 1.1 Evidence for This Cause

- Test 4 uses bimodal distribution (half ~8/10, half ~3/10)
- OrderedLogistic models can struggle with bimodal data
- Direct test showed "15 divergences after tuning"
- Test 3 (peaked distribution) works fine

### 1.2 Investigation Steps

**Check divergence behavior**:
- [ ] Count divergences in Test 4 vs Test 3
- [ ] Check if divergences increase over time
- [ ] Examine R-hat values (should be < 1.01)
- [ ] Check effective sample size (should be > 100)

**Test with different samplers**:
- [ ] Try with increased `target_accept` (0.95 → 0.99)
- [ ] Try with increased `max_treedepth` (10 → 15)
- [ ] Compare sampling time between peaked vs bimodal data

**Expected findings**:
- If this is the cause: More divergences, poor R-hat, longer sampling time
- If not: Similar convergence diagnostics to Test 3

### 1.3 Potential Fix

If convergence is the issue:
```python
# In gpu_utils.py or early_stopping.py
sampling_kwargs = {
    'draws': 500,
    'tune': 500,
    'target_accept': 0.99,  # Increase from default 0.95
    'max_treedepth': 15      # Increase from default 10
}
```

---

## Cause 2: Model Caching Problems

**Priority**: MEDIUM - Could cause subtle issues

### 2.1 Evidence for This Cause

- Model cache reused across inference calls (ordinal_model.py:324-333)
- Test 4 has multiple inference checkpoints (trials 200, 400, 600, 800)
- Cache might retain stale data or sampler state

### 2.2 Investigation Steps

**Check cache usage**:
- [ ] Add logging to track cache hits/misses
- [ ] Verify `pm.set_data()` updates model correctly
- [ ] Check if cached model retains sampling configuration

**Test with cache disabled**:
- [ ] Run Test 4 with `model_cache=None`
- [ ] Compare timing: cached vs uncached
- [ ] Check if hang disappears

**Expected findings**:
- If this is the cause: Uncached version works, cached version hangs
- If not: Same behavior with/without cache

### 2.3 Potential Fix

If caching is the issue:
```python
# Option 1: Disable caching
model_cache = None

# Option 2: Invalidate cache more aggressively
if model_cache and needs_rebuild:
    model_cache.clear()
```

---

## Cause 3: Asyncio Thread Blocking

**Priority**: MEDIUM - Could cause deadlocks

### 3.1 Evidence for This Cause

- `asyncio.to_thread()` wraps CPU-intensive MCMC (early_stopping.py:978)
- Multiple async operations happening concurrently
- Thread pool might be exhausted

### 3.2 Investigation Steps

**Check thread behavior**:
- [ ] Monitor thread pool size during test
- [ ] Check for thread exhaustion errors
- [ ] Verify threads complete and return

**Test without asyncio**:
- [ ] Run inference synchronously (no `asyncio.to_thread()`)
- [ ] Compare behavior

**Expected findings**:
- If this is the cause: Thread pool exhaustion or deadlock
- If not: Same hang with/without asyncio

### 3.3 Potential Fix

If asyncio is the issue:
```python
# Add timeout
try:
    result = await asyncio.wait_for(
        asyncio.to_thread(...),
        timeout=300.0  # 5 minute timeout
    )
except asyncio.TimeoutError:
    logger.error("Inference timed out")
```

---

## Cause 4: Excessive Reanalysis Calls with Growing Datasets

**Priority**: HIGH - Very plausible

### 4.1 Evidence for This Cause

- Test 4: reanalysis_interval=200, total trials=800
- Inference runs at trials: 200, 400, 600, 800
- Each call has MORE data than previous (cumulative)
- More data → longer MCMC sampling time

### 4.2 Investigation Steps

**Track inference call timing**:
- [ ] Add timing logs for each inference call
- [ ] Check dataset size at each call
- [ ] Measure MCMC duration as data grows

**Expected pattern if this is the cause**:
```
Inference 1 (200 trials):  30 seconds
Inference 2 (400 trials):  60 seconds
Inference 3 (600 trials): 120 seconds
Inference 4 (800 trials): 240 seconds → HANGS?
```

**Test with smaller intervals**:
- [ ] Run with reanalysis_interval=100 (more frequent, less data per call)
- [ ] Check if completes faster

### 4.3 Potential Fix

If data growth is the issue:
```python
# Option 1: Limit data used for inference
max_samples_for_inference = 400
df_limited = df_grouping.tail(max_samples_for_inference)

# Option 2: Sample data instead of using all
if len(df_grouping) > 500:
    df_sampled = df_grouping.sample(n=500, random_state=42)
```

---

## Cause 5: Score Extraction/Validation Issues

**Priority**: LOW - Already partially ruled out

### 5.1 Evidence Against This Cause

- Direct parameter test worked with ordinal scores
- Validation logic exists (ordinal_model.py:313-318)
- Test 3 (ordinal discrete) works fine

### 5.2 Investigation Steps

**Check Test 4 score generation**:
- [ ] Verify scores are valid integers in [0, 10]
- [ ] Check for NaN or None values
- [ ] Verify score distribution matches expected bimodal pattern

**Expected findings**:
- Scores should be valid throughout
- This is unlikely to be the cause

---

## Cause 6: Resource Contention (CPU/Memory)

**Priority**: MEDIUM - Can cause slowdowns

### 6.1 Evidence for This Cause

- Test 4 uses 4 chains, 4 cores
- Multiple MCMC samplers running in parallel
- System might run out of resources

### 6.2 Investigation Steps

**Monitor resources during test**:
- [ ] Track CPU usage (should be ~400% for 4 cores)
- [ ] Track memory usage
- [ ] Check for OOM errors
- [ ] Monitor swap usage

**Test with reduced resources**:
- [ ] Run with 2 chains, 2 cores
- [ ] Check if completes faster

**Expected findings**:
- If this is the cause: CPU at 100%, memory growing, swap activity
- If not: Resources available, no contention

### 6.3 Potential Fix

If resource contention is the issue:
```python
# Reduce parallelism
sampling_kwargs = {
    'draws': 500,
    'tune': 500,
    'chains': 2,    # Reduce from 4
    'cores': 2       # Reduce from 4
}
```

---

## Investigation Execution Order

1. **Cause 7**: Structural errors (quick to check, fundamental)
2. **Cause 4**: Data growth (most plausible given cumulative nature)
3. **Cause 1**: MCMC convergence (matches bimodal hypothesis)
4. **Cause 2**: Model caching (could interact with data growth)
5. **Cause 3**: Asyncio blocking (less likely but important)
6. **Cause 6**: Resource contention (environmental)
7. **Cause 5**: Score validation (already partially ruled out)

---

## Test Matrix

| Test | Modification | Expected Result if Cause Found |
|------|--------------|-------------------------------|
| T7.1 | Trace function signatures | Reveals mismatched parameters |
| T7.2 | Check for loops | Finds infinite loop |
| T7.3 | Identify blocking ops | Finds operation without timeout |
| T7.4 | Check data structure growth | Finds unbounded growth |
| T4.1 | Add timing per inference | Shows exponential time growth |
| T4.2 | Limit dataset size | Completes successfully |
| T1.1 | Increase target_accept | Reduces divergences, completes |
| T1.2 | Compare peaked vs bimodal | Bimodal much slower |
| T2.1 | Disable caching | Completes without hang |
| T3.1 | Remove asyncio wrapper | Same behavior (not the cause) |
| T6.1 | Reduce chains/cores | Completes successfully |
| T5.1 | Validate scores | All valid (not the cause) |

---

## Findings Log

### Cause 7: Structural Errors
**Status**: PENDING
**Findings**: (to be filled)

### Cause 4: Data Growth
**Status**: PENDING
**Findings**: (to be filled)

### Cause 1: MCMC Convergence
**Status**: PENDING
**Findings**: (to be filled)

### Cause 2: Model Caching
**Status**: PENDING
**Findings**: (to be filled)

### Cause 3: Asyncio Blocking
**Status**: PENDING
**Findings**: (to be filled)

### Cause 6: Resource Contention
**Status**: PENDING
**Findings**: (to be filled)

### Cause 5: Score Validation
**Status**: PENDING
**Findings**: (to be filled)

---

## Final Diagnosis

(To be completed after investigation)

**Root Cause**: TBD
**Contributing Factors**: TBD
**Recommended Fix**: TBD
**Testing Plan**: TBD
