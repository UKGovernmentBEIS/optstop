# Test 4 Hang Investigation Report

**Date**: 2025-11-20
**Test**: `TestOrdinalDiscreteBimodal::test_ordinal_discrete_bimodal_80_samples`
**Status**: HUNG after 60+ minutes (expected completion: 15-20 minutes)

---

## Test Results Summary

### Completed Tests (3/13): ✅
1. **Test 1** - Binary Discrete (High): PASSED (~3 min)
2. **Test 2** - Binary Discrete (Variable): PASSED (~4 min)
3. **Test 3** - Ordinal Discrete (High, Modal): PASSED (~10 min)

### Stuck Test (4/13): ⚠️
4. **Test 4** - Ordinal Discrete (Bimodal, Hybrid): **HUNG** after 60+ minutes

---

## Evidence of Hang

1. **No log growth**: Log stayed at 207 lines for 30+ minutes with no new output
2. **Limited MCMC activity**: Only 2 divergence warnings visible (suggests 1-2 inference calls completed, then hung)
3. **Multiple active processes**: 5-6 pytest processes running (1 main + 4 MCMC chains)
4. **High CPU usage**: Processes showing 40-111% CPU (indicates active computation, not deadlock)
5. **Expected behavior**: 800 trials with 3 inference checkpoints (at trials 200, 400, 600)

---

## Test 4 Configuration

```python
manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.12,
        'delta_cap': 0.12,
        'cred_level': 0.95,
        'draws': 500,        # ← Should limit MCMC sampling
        'tune': 500          # ← Should limit MCMC tuning
    },
    grouping_columns=['task'],
    reanalysis_interval=200,  # 25%, 50%, 75% of 800 trials
    ordinal_tasks=['rating'],
    ordinal_max_score=10
)
```

**Inference Mode**: `hybrid` (modal + entropy stabilization)
**Score Type**: Ordinal discrete (0-10)
**Data Distribution**: Bimodal (half ~8/10, half ~3/10)

---

## Root Cause Hypothesis

### Primary Suspect: Parameter Passing Issue in `_ordinal_entropy_ci_adaptive()`

The hybrid inference mode for ordinal discrete scores calls:
1. `_ordinal_ci_adaptive()` - Fast modal bootstrap (works fine)
2. **`_ordinal_entropy_ci_adaptive()`** - Slow Bayesian entropy CI (suspected issue)

#### Code Flow Analysis:

**File**: `optstop/ordinal_model.py`

```python
def _ordinal_entropy_ci_adaptive(
    scores: np.ndarray,
    ordinal_max_score: int,
    cred_level: float = 0.95,
    conservatism: float = 1.0,
    low_perf_threshold: float = 0.2,
    n_samples: int = 6000,      # ← HARDCODED DEFAULT: 6000
    n_tune: int = 6000,         # ← HARDCODED DEFAULT: 6000
    model_cache: Optional[Dict[str, Any]] = None,
    compute_kwargs: Optional[Dict[str, Any]] = None
) -> Tuple[float, float, float, Dict[str, Any]]:
    # ...
    compute_kwargs = compute_kwargs or {}
    default_kwargs = {
        'draws': n_samples,     # Uses hardcoded 6000 if not overridden
        'tune': n_tune,         # Uses hardcoded 6000 if not overridden
        'random_seed': compute_kwargs.get('random_seed', None),
        'progressbar': False,
        'return_inferencedata': True
    }
    default_kwargs.update(compute_kwargs)  # Should override with {'draws': 500, 'tune': 500}

    with model:
        trace = pm.sample(**default_kwargs)
```

**Expected**: `compute_kwargs` contains `{'draws': 500, 'tune': 500}` and overrides defaults
**Observed**: Test hung, suggesting it's running with 6000/6000 (12x slower than expected)

#### Parameter Flow Verification Needed:

1. **From test** → `optstop_params={'draws': 500, 'tune': 500}`
2. **Through `gpu_utils.get_sampling_kwargs()`** → Creates `sampling_kwargs = {'draws': 500, 'tune': 500, ...}`
3. **Into `rule.py` hybrid inference** → Passes `compute_kwargs=sampling_kwargs` (line 1060)
4. **Into `_ordinal_hybrid_stopping_criterion()`** → Receives `compute_kwargs`
5. **Into `_ordinal_entropy_ci_adaptive()`** → Should receive and use `compute_kwargs`

**Issue**: Despite correct code structure, Test 4 behavior suggests parameters are NOT being honored.

---

## Performance Comparison

| Test | Score Type | Inference | Samples×Epochs | Expected Time | Actual Time |
|------|-----------|-----------|----------------|---------------|-------------|
| 1 | Binary Discrete | Beta-Binomial | 50×15 (750) | 3-5 min | ~3 min ✅ |
| 2 | Binary Discrete | Beta-Binomial | 100×12 (1200) | 4-6 min | ~4 min ✅ |
| 3 | Ordinal Discrete | Modal | 60×15 (900) | 8-12 min | ~10 min ✅ |
| 4 | Ordinal Discrete | Hybrid | 80×10 (800) | 15-20 min | **60+ min** ❌ |

**Note**: Hybrid inference = Modal (fast) + Entropy CI (slow Bayesian MCMC)

If running with 6000/6000 instead of 500/500:
- Each MCMC call: ~5-10 minutes (instead of 30-60 seconds)
- 3 inference checkpoints × 10 minutes = 30+ minutes minimum
- Plus trial execution overhead = 60+ minutes total

---

## Investigation Plan

### Phase 1: Verify Parameter Passing ✅ (Completed)

**Checked**:
- ✅ `gpu_utils.get_sampling_kwargs()` - Lines 648-649: `'draws': params.get('draws', 6000)`
- ✅ `rule.py` hybrid inference call - Line 1060: `compute_kwargs=sampling_kwargs`
- ✅ `ordinal_model.py` `_ordinal_entropy_ci_adaptive()` - Lines 348-356: Updates defaults with `compute_kwargs`

**Conclusion**: Code structure appears correct, but runtime behavior suggests otherwise.

### Phase 2: Add Diagnostic Logging (NEXT STEP)

Add explicit logging to track actual parameter values at runtime:

**File**: `optstop/ordinal_model.py:_ordinal_entropy_ci_adaptive()`

```python
def _ordinal_entropy_ci_adaptive(..., compute_kwargs: Optional[Dict[str, Any]] = None):
    logger = logging.getLogger('optstop.ordinal_model')

    # ADD THIS: Log received parameters
    logger.warning(f"🔍 _ordinal_entropy_ci_adaptive called:")
    logger.warning(f"   n_samples (default): {n_samples}")
    logger.warning(f"   n_tune (default): {n_tune}")
    logger.warning(f"   compute_kwargs: {compute_kwargs}")

    compute_kwargs = compute_kwargs or {}
    default_kwargs = {
        'draws': n_samples,
        'tune': n_tune,
        ...
    }
    default_kwargs.update(compute_kwargs)

    # ADD THIS: Log final values being used
    logger.warning(f"   Final draws: {default_kwargs['draws']}")
    logger.warning(f"   Final tune: {default_kwargs['tune']}")

    with model:
        trace = pm.sample(**default_kwargs)
```

### Phase 3: Targeted Testing

Run Test 4 in isolation with:
1. Verbose logging enabled
2. Reduced sample size (20 samples instead of 80)
3. Watch for diagnostic output

```bash
pytest tests/test_early_stopping_comprehensive.py::TestOrdinalDiscreteBimodal::test_ordinal_discrete_bimodal_80_samples -v -s --log-cli-level=WARNING
```

### Phase 4: Alternative Hypothesis Investigation

If parameters ARE being passed correctly, investigate:

1. **PyMC model caching issue**: Model cache may be reusing old sampler configuration
   - Check `model_cache` usage in lines 324-333
   - Verify `pm.set_data()` doesn't interfere with sampling params

2. **MCMC convergence issues**: Bimodal distribution may cause sampling to struggle
   - Check R-hat values (should be < 1.01)
   - Check effective sample size (should be > 100)
   - May need higher `target_accept` (currently 0.97)

3. **OrderedLogistic model complexity**: 11 categories (0-10) with bimodal data
   - More complex than Test 3's peaked distribution
   - May require more samples naturally
   - Check if 500 draws is sufficient for convergence

---

## Temporary Workaround

Until root cause is identified and fixed, consider:

1. **Skip Test 4** temporarily using pytest.mark.skip
2. **Simplify Test 4** to use "modal" inference instead of "hybrid"
3. **Reduce Test 4 sample size** from 80 to 40 samples

```python
@pytest.mark.skip(reason="Hybrid ordinal inference hanging - under investigation")
async def test_ordinal_discrete_bimodal_80_samples(self):
    ...
```

---

## Key Files to Investigate

1. **`optstop/ordinal_model.py`** (lines 236-404)
   - `_ordinal_entropy_ci_adaptive()` function
   - Parameter handling and PyMC sampling call

2. **`optstop/rule.py`** (lines 1048-1061)
   - Hybrid inference invocation
   - `sampling_kwargs` passing to `compute_kwargs`

3. **`optstop/gpu_utils.py`** (lines 605-678)
   - `get_sampling_kwargs()` function
   - Parameter extraction from `optstop_params`

---

## Questions to Answer

1. **Are the correct parameter values reaching `pm.sample()`?**
   - Add logging to verify `draws=500, tune=500` in final `default_kwargs`

2. **Is model caching interfering with parameter updates?**
   - Check if reused models retain old sampling configuration

3. **Is the bimodal distribution causing convergence issues?**
   - Compare convergence diagnostics between Test 3 (peaked) and Test 4 (bimodal)

4. **Why did only 2 divergence warnings appear?**
   - Suggests 1-2 MCMC calls completed, then something changed or hung
   - Check if first inference call (trial 200) worked but subsequent ones didn't

---

## Next Actions

1. **Add diagnostic logging** to `_ordinal_entropy_ci_adaptive()`
2. **Run Test 4 in isolation** with verbose output
3. **Analyze PyMC sampling logs** for actual draws/tune values used
4. **Compare with Test 3** (which uses modal inference and works fine)
5. **Document findings** and implement fix

---

## Related Documentation

- Continuous Scoring Implementation: `/home/ubuntu/optstop/CONTINUOUS_SCORING_FINAL_SUMMARY.md`
- Test Suite Overview: `/home/ubuntu/optstop/COMPREHENSIVE_TEST_SUITE_SUMMARY.md`
- Test Fixes Applied: `/home/ubuntu/optstop/apply_final_fixes.py`
