# Logging and Print Statement Audit

**Date**: 2025-11-28 (Updated)
**Scope**: early_stopping.py bridge + optstop core functions
**Purpose**: Assess criticality of all logging/print statements for user visibility

---

## Current State Summary

### By File:
- **early_stopping.py**: 70 statements (37 PRINT, 12 INFO, 8 WARNING, 7 ERROR, 6 DEBUG)
- **rule.py**: 125 statements (95 INFO, 13 WARNING, 11 ERROR, 4 PRINT, 2 DEBUG)
- **ordinal_model.py**: 18 statements (6 WARNING, 6 DEBUG, 2 INFO, 3 PRINT, 1 ERROR)
- **gpu_utils.py**: 56 statements (45 INFO, 8 WARNING, 3 PRINT)

### By Level Across All Files:
- **PRINT**: 43 statements (uncontrolled, always visible)
- **INFO**: 154 statements
- **WARNING**: 35 statements
- **ERROR**: 19 statements
- **DEBUG**: 14 statements

---

## Classification System

- **CRITICAL**: Must be visible to user (errors, stopping decisions, major events)
- **HIGH**: Very useful for user (configuration, progress, warnings)
- **MEDIUM**: Helpful for debugging/monitoring (status updates, cache hits)
- **LOW**: Development/diagnostic only (debug traces, verbose parameter dumps)

---

## 1. early_stopping.py (Bridge)

### CRITICAL - User Must See

| Line | Type | Statement | Purpose | Criticality |
|------|------|-----------|---------|-------------|
| 726 | ERROR | `compiled_dataset not initialized` | Fatal error - user didn't call start_task() | **CRITICAL** |
| 802 | ERROR | `compiled_dataset not initialized` | Fatal error in schedule_sample() | **CRITICAL** |
| 1065 | ERROR | Inference execution error | Inference executor failed | **CRITICAL** |
| 1079 | ERROR | Task cancellation error | Failed to cancel inference tasks | **CRITICAL** |
| 1115 | ERROR | Sample not found in dataset | Sample ID doesn't exist | **CRITICAL** |

### HIGH - Configuration & Major Events

| Line | Type | Statement | Purpose | Criticality | Issue |
|------|------|-----------|---------|-------------|-------|
| 338-417 | PRINT | Configuration summary (entire block, 37 lines) | Shows user all configuration at start_task() | **HIGH** | **Uses print() instead of logger** |
| 691 | INFO | `Initialized optimal stopping dataset` | Confirms task started successfully | **HIGH** | ✓ OK |
| 970 | INFO | `Running optimal stopping inference on X trials...` | User knows inference is happening | **HIGH** | ✓ OK |
| 1161-1170 | INFO | `Sample X stopped at epoch Y` / `Marked sample for early stopping` | User sees which samples stopped | **HIGH** | ✓ OK |
| 1232-1237 | INFO | `Stopped grouping 'X' after Y samples` / `Marked entire grouping for early stopping` | User sees grouping-level stops | **HIGH** | ✓ OK |
| 1258-1260 | INFO | Inference executor shutdown | Clean shutdown confirmation | **HIGH** | ✓ OK |
| 1315 | INFO | `Task complete. Ran X/Y trials (Z% efficiency gain)` | Final summary of efficiency | **HIGH** | ✓ OK |

### MEDIUM - Helpful Monitoring

| Line | Type | Statement | Purpose | Criticality |
|------|------|-----------|---------|-------------|
| 454 | WARNING | `Empty scores_dict for sample...` | Score extraction failed | **MEDIUM** |
| 490 | WARNING | `Empty scores dictionary provided` | No scores in complete_sample() | **MEDIUM** |
| 498 | WARNING | `Could not import value_to_float` | Fallback to basic conversion | **MEDIUM** |
| 545 | WARNING | `Multiple scores found, using first` | Ambiguous score selection | **MEDIUM** |
| 555 | WARNING | `Could not convert score to float` | Score conversion failed | **MEDIUM** |
| 571 | WARNING | `No valid numeric scores found` | All scores non-numeric | **MEDIUM** |
| 734 | ERROR | `Error getting grouping values` | Grouping metadata extraction failed | **MEDIUM** |
| 759 | WARNING | `Warning: no valid samples` | No samples for grouping | **MEDIUM** |
| 812 | ERROR | `Error getting grouping values` | Grouping extraction failed in schedule | **MEDIUM** |
| 878 | WARNING | `Warning: sample not in stopped_samples` | Trying to remove non-stopped sample | **MEDIUM** |

### LOW - Debug/Development Only

| Line | Type | Statement | Purpose | Criticality |
|------|------|-----------|---------|-------------|
| 516 | DEBUG | `Converter failed on 'value'` | Score conversion detailed trace | **LOW** |
| 523 | DEBUG | `Could not convert string to float` | String conversion trace | **LOW** |
| 568 | DEBUG | `Could not convert score from scorer` | Aggregation conversion trace | **LOW** |
| 906-913 | DEBUG | Sample completion / grouping processing | Detailed sample flow | **LOW** |
| 958 | DEBUG | `Skipping inference for 'grouping'` | Optimization #1 trace | **LOW** |

---

## 2. rule.py (Core Algorithms)

### CRITICAL - User Must See

| Line | Type | Statement | Purpose | Criticality |
|------|------|-----------|---------|-------------|
| 825 | ERROR | `Parameter must be positive integer` | Invalid parameter validation | **CRITICAL** |
| 828 | ERROR | `delta_item must be positive` | Invalid threshold | **CRITICAL** |
| 831 | ERROR | `delta_cap must be positive` | Invalid threshold | **CRITICAL** |
| 1312-1314 | ERROR | `Error processing grouping` + traceback | Worker crashed during posthoc | **CRITICAL** |
| 1801-1803 | ERROR | `Error processing grouping` + traceback | Worker crashed during live | **CRITICAL** |

### HIGH - Stopping Decisions & Progress

| Line | Type | Statement | Purpose | Criticality |
|------|------|-----------|---------|-------------|
| 1896 | INFO | `Starting post-hoc optimal stopping` | Function entry | **HIGH** |
| 1121-1123 | INFO | `Stopping sample_id X: CI width...` | Item-level stop decision (binary/continuous) | **HIGH** |
| 1145 | INFO | `Stopping sample_id X due to CI stabilization` | Item-level stabilization stop | **HIGH** |
| 1149 | INFO | `Stopping low-performance sample_id X` | Item-level low-perf stop | **HIGH** |
| 1203 | INFO | `Stopping grouping: CI width...` | Group-level stop (binary/continuous) | **HIGH** |
| 1215 | INFO | `Stopping grouping due to CI stabilization` | Group-level stabilization stop | **HIGH** |
| 1218 | INFO | `Stopping low-performance grouping` | Group-level low-perf stop | **HIGH** |
| 1276 | INFO | `Stopping ordinal grouping: Hybrid group-level` | Ordinal hybrid stop | **HIGH** |
| 1279 | INFO | `Stopping ordinal grouping: CI width...` | Ordinal CI stop | **HIGH** |
| 1292 | INFO | `Stopping ordinal grouping via stabilization` | Ordinal stabilization stop | **HIGH** |
| 1295 | INFO | `Stopping low-performance ordinal grouping` | Ordinal low-perf stop | **HIGH** |
| 1567 | INFO | `Stopping sample_id X in grouping Y` | Live item-level stop | **HIGH** |
| 1582 | INFO | `Stopping sample_id X via stabilization` | Live item stabilization | **HIGH** |
| 1587 | INFO | `Stopping low-performance sample_id X` | Live item low-perf | **HIGH** |
| 1625-1660 | INFO | All ordinal item-level stopping messages | Ordinal item stops | **HIGH** |
| 1710-1791 | INFO | All group-level stopping messages | Live group stops | **HIGH** |
| 2057-2060 | INFO | `Post-hoc complete` + print log path | Completion message | **HIGH** |
| 2408-2522 | INFO | All optimal_stopping_live_single item stops | Live single item stops | **HIGH** |
| 2539-2889 | INFO | All group-level stopping in live_single | Live single group stops | **HIGH** |

### MEDIUM - Configuration & Status

| Line | Type | Statement | Purpose | Criticality |
|------|------|-----------|---------|-------------|
| 833 | INFO | `Parameters validated` | Confirms params OK | **MEDIUM** |
| 895 | INFO | Worker PyTensor compiledir | Worker setup info | **MEDIUM** |
| 940 | INFO | `Processing grouping X with Y scoring` | Worker task start | **MEDIUM** |
| 1006-1010 | INFO | Worker GPU/CPU configuration | Hardware config per worker | **MEDIUM** |
| 1390 | INFO | Live worker PyTensor compiledir | Worker setup info | **MEDIUM** |
| 1415 | INFO | `Processing grouping: X` | Live worker task start | **MEDIUM** |
| 1443 | INFO | `Grouping identified as X` | Score type routing | **MEDIUM** |
| 1500-1504 | INFO | Worker GPU/CPU configuration | Hardware config | **MEDIUM** |
| 1940 | INFO | `Validated ordinal scores for grouping` | Ordinal validation | **MEDIUM** |
| 1956-1958 | INFO | Worker and GPU assignment summary | Parallelization info | **MEDIUM** |
| 2174 | INFO | `Processing grouping 'X' as Y` | Score type in live_single | **MEDIUM** |

### MEDIUM - Cache Performance

| Line | Type | Statement | Purpose | Criticality | Issue |
|------|------|-----------|---------|-------------|-------|
| 2242 | INFO | `✓ CACHE HIT: Reusing binary group model (n_items=X)` | Binary cache hit | **MEDIUM** | Consider moving to DEBUG |
| 2245 | INFO | `✗ CACHE INVALIDATED: Binary model n_items X → Y` | Binary cache invalidation | **MEDIUM** | Consider moving to DEBUG |
| 2263 | INFO | `✗ CACHE MISS: Created new binary group model (n_items=X)` | Binary cache miss | **MEDIUM** | Consider moving to DEBUG |
| 2298 | INFO | `✓ CACHE HIT: Reusing continuous group model (n_items=X)` | Continuous cache hit | **MEDIUM** | Consider moving to DEBUG |
| 2301 | INFO | `✗ CACHE INVALIDATED: Continuous model n_items X → Y` | Continuous cache invalidation | **MEDIUM** | Consider moving to DEBUG |
| 2375 | INFO | `✗ CACHE MISS: Created new continuous group model (n_items=X)` | Continuous cache miss | **MEDIUM** | Consider moving to DEBUG |

### LOW - Timing Tests & Debug

| Line | Type | Statement | Purpose | Criticality | Issue |
|------|------|-----------|---------|-------------|-------|
| 158-160 | DEBUG | PyMC stdout/stderr capture | Debugging PyMC output | **LOW** | ✓ OK |
| 307-569 | INFO | Diagnostic plot generation messages | Posthoc diagnostics | **LOW** | Should be DEBUG? |
| 426-435 | INFO | Accuracy metrics (posthoc diagnostics) | Posthoc analysis | **LOW** | Should be DEBUG? |
| 2638 | WARNING | `🕐 TIMING_TEST: Binary group inference took Xs` | **Performance profiling** | **LOW** | **Should be DEBUG** |
| 2741 | WARNING | `🕐 TIMING_TEST: Ordinal group inference took Xs` | **Performance profiling** | **LOW** | **Should be DEBUG** |
| 2786 | WARNING | `🕐 TIMING_TEST: Continuous data aggregation took Xs` | **Performance profiling** | **LOW** | **Should be DEBUG** |
| 2841 | WARNING | `🕐 TIMING_TEST: Continuous MCMC took Xs` | **Performance profiling** | **LOW** | **Should be DEBUG** |
| 2903 | WARNING | `🕐 TIMING_TEST: Continuous TOTAL took Xs` | **Performance profiling** | **LOW** | **Should be DEBUG** |
| 2910 | WARNING | `🕐 TIMING_TEST: optimal_stopping_live_single TOTAL took Xs` | **Performance profiling** | **LOW** | **Should be DEBUG** |

---

## 3. ordinal_model.py (Ordinal Inference)

### HIGH - Stopping Decisions

| Line | Type | Statement | Purpose | Criticality |
|------|------|-----------|---------|-------------|
| 581 | INFO | `Stopping via Pathway 1 (Modal CI narrow + validated)` | Ordinal stopping decision | **HIGH** |
| 628 | INFO | `Ordinal stopping via stabilization` | Stabilization stop | **HIGH** |

### MEDIUM - Cache & Errors

| Line | Type | Statement | Purpose | Criticality |
|------|------|-----------|---------|-------------|
| 336 | DEBUG | `Reusing OrderedLogistic model (updated X → Y items)` | Ordinal cache hit | **MEDIUM** |
| 348 | DEBUG | `Created new OrderedLogistic model for X items` | Ordinal cache miss | **MEDIUM** |
| 376 | ERROR | `OrderedLogistic sampling failed` | Sampling error | **MEDIUM** |

### LOW - Debug & Profiling

| Line | Type | Statement | Purpose | Criticality | Issue |
|------|------|-----------|---------|-------------|-------|
| 308-310 | PRINT | Docstring examples (3 lines) | Documentation only | **LOW** | ✓ OK (docstring) |
| 354-357 | WARNING | `🔍 _ordinal_entropy_ci_adaptive called...` | **Detailed parameter trace** | **LOW** | **Should be DEBUG** |
| 369-370 | WARNING | `Final draws/tune (after update)` | **Parameter trace** | **LOW** | **Should be DEBUG** |
| 413 | DEBUG | Entropy computation details | Technical trace | **LOW** | ✓ OK |
| 564-567 | DEBUG | `False peak detected...` | Decision logic trace | **MEDIUM-LOW** | ✓ OK |
| 603 | DEBUG | Continue: insufficient history | Not enough data | **LOW** | ✓ OK |
| 645 | DEBUG | Detailed stabilization logic | Technical trace | **LOW** | ✓ OK |

---

## 4. gpu_utils.py (GPU Configuration)

### MEDIUM - Configuration & Warnings

| Line | Type | Statement | Purpose | Criticality | Issue |
|------|------|-----------|---------|-------------|-------|
| 30-32 | PRINT | Nutpie availability (docstring) | Documentation only | **LOW** | ✓ OK (docstring) |
| 62 | PRINT | Sampler example (docstring) | Documentation only | **LOW** | ✓ OK (docstring) |
| 296 | INFO | `JAX GPU acceleration available: X GPUs` | GPU detection | **MEDIUM** | OK but verbose |
| 298 | WARNING | `JAX GPU detected but test failed` | GPU issue | **MEDIUM** | ✓ OK |
| 300 | INFO | JAX backend and device info | GPU status | **MEDIUM** | OK but verbose |
| 303 | INFO | `JAX not available - checking other backends` | Fallback info | **MEDIUM** | OK but verbose |
| 305 | WARNING | JAX GPU detection error | Detection failure | **MEDIUM** | ✓ OK |
| 319 | INFO | PyTensor GPU backend available | GPU detection | **MEDIUM** | OK but verbose |
| 330 | INFO | `GPU acceleration enabled via X` | GPU confirmed | **MEDIUM** | ✓ OK (important) |
| 333 | WARNING | `System GPUs detected but no PyMC GPU backend available` | GPU not usable | **MEDIUM** | ✓ OK |
| 335 | INFO | `No GPU acceleration - using CPU` | CPU fallback | **MEDIUM** | ✓ OK (important) |
| 364 | INFO | GPU recommendation | Setup advice | **MEDIUM** | OK but verbose |
| 388-398 | INFO/WARNING | JAX GPU configuration messages | GPU setup status | **MEDIUM** | OK but verbose |
| 626-752 | INFO | GPU decision and sampling configuration (many lines) | Hardware setup | **MEDIUM** | OK but verbose |
| 768-794 | INFO | Enhanced GPU status report (full block, ~25 lines) | Detailed GPU report | **MEDIUM** | **Very verbose** |

---

## Summary of Issues Found

### High Priority Issues (Must Fix):

1. **TIMING_TEST messages logged as WARNING (6 statements)**
   - Lines 2638, 2741, 2786, 2841, 2903, 2910 in rule.py
   - Should be: `logger.debug()`
   - Impact: Users see performance profiling at INFO level

2. **Ordinal parameter dump logged as WARNING (6 statements)**
   - Lines 354-357, 369-370 in ordinal_model.py
   - Should be: `logger.debug()`
   - Impact: Users see verbose parameter dumps every ordinal call

3. **Configuration summary uses print() (37 statements)**
   - Lines 338-417 in early_stopping.py
   - Should be: `logger.info()` with optional verbosity control
   - Impact: Can't be controlled by logging level, always visible

### Medium Priority Issues (Should Consider):

4. **Cache hit/miss messages at INFO level (6 statements)**
   - Lines 2242, 2245, 2263, 2298, 2301, 2375 in rule.py
   - Consider: Move to `logger.debug()` after validation period
   - Impact: Adds noise at INFO level during normal operation

5. **GPU configuration very verbose (45+ INFO statements)**
   - Throughout gpu_utils.py
   - Consider: Consolidate to single summary message
   - Impact: Lots of INFO-level noise about hardware detection

6. **Diagnostic plot generation at INFO level (~20 statements)**
   - Lines 307-569 in rule.py
   - Consider: Move to `logger.debug()`
   - Impact: Verbose during posthoc analysis

### Low Priority Issues:

7. **False peak detection at DEBUG (correct but consider)**
   - Line 564 in ordinal_model.py
   - Current: `logger.debug()` ✓ Correct level
   - Consider: Users might want to see this for understanding hybrid behavior
   - Recommendation: Keep as DEBUG for now

---

## Recommendations

### Phase 1: Critical Fixes (Quick Wins)

1. **Change TIMING_TEST to DEBUG** (6 lines in rule.py)
   ```python
   # Change from:
   logger.warning(f"🕐 TIMING_TEST: ...")
   # To:
   logger.debug(f"🕐 TIMING_TEST: ...")
   ```

2. **Change ordinal parameter dump to DEBUG** (6 lines in ordinal_model.py)
   ```python
   # Change from:
   logger.warning(f"🔍 _ordinal_entropy_ci_adaptive called:")
   # To:
   logger.debug(f"🔍 _ordinal_entropy_ci_adaptive called:")
   ```

### Phase 2: Major Refactor

3. **Convert print() to logger.info()** (37 lines in early_stopping.py)
   - Replace all print() in start_task() with logger.info()
   - Consider adding verbosity parameter to control configuration summary
   - Keep important messages (errors, stopping decisions) at INFO
   - Move detailed config to DEBUG

### Phase 3: Cleanup (After Validation)

4. **Move cache messages to DEBUG** (6 lines in rule.py)
   - After caching is validated and working well
   - Reduces INFO-level noise

5. **Consolidate GPU logging** (gpu_utils.py)
   - Create single summary message for GPU status
   - Move detailed detection steps to DEBUG
   - Keep only critical messages (enabled/disabled, errors) at INFO

---

## Expected Impact of Changes

### After Phase 1 (TIMING_TEST + ordinal params):
- **Before**: ~12 WARNING messages per inference call
- **After**: ~0 WARNING messages per inference call (moved to DEBUG)
- **User experience**: Cleaner logs, no performance profiling noise

### After Phase 2 (print to logger):
- **Before**: 37 uncontrolled print() statements
- **After**: 37 logger.info() statements (can be controlled)
- **User experience**: Can set log level to suppress configuration details

### After Phase 3 (cache + GPU):
- **Before**: ~50 INFO messages during setup/execution
- **After**: ~10 INFO messages (critical only)
- **User experience**: Much cleaner INFO-level logs

---

## Recommended Logging Levels for Users

### Default for inspect_ai users:
```python
logging.basicConfig(level=logging.INFO)
```
**Shows:**
- ✅ Configuration summary
- ✅ Stopping decisions
- ✅ Progress updates
- ✅ Warnings and errors
- ❌ Debug traces, timing tests, cache hits

### For debugging:
```python
logging.basicConfig(level=logging.DEBUG)
```
**Shows:**
- ✅ Everything above, plus:
- ✅ Cache behavior
- ✅ Performance timing
- ✅ Internal decision logic

### For performance profiling:
```python
logging.basicConfig(level=logging.DEBUG)
# Then grep logs for: "TIMING_TEST"
```

---

**Last Updated:** 2025-11-28
**Total Issues Identified:** 7 (3 high priority, 4 medium/low priority)
**Total Statements Reviewed:** 269 across 4 files
