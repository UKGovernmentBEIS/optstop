# Logging and Print Statement Audit

**Date**: 2025-11-27
**Scope**: early_stopping.py bridge + optstop core functions
**Purpose**: Assess criticality of all logging/print statements for user visibility

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

| Line | Type | Statement | Purpose | Criticality |
|------|------|-----------|---------|-------------|
| 338-417 | PRINT | Configuration summary (entire block) | Shows user all configuration at start_task() | **HIGH** |
| 691 | INFO | `Initialized optimal stopping dataset` | Confirms task started successfully | **HIGH** |
| 970 | INFO | `Running optimal stopping inference on X trials...` | User knows inference is happening | **HIGH** |
| 1161-1170 | INFO | `Sample X stopped at epoch Y` / `Marked sample for early stopping` | User sees which samples stopped | **HIGH** |
| 1232-1237 | INFO | `Stopped grouping 'X' after Y samples` / `Marked entire grouping for early stopping` | User sees grouping-level stops | **HIGH** |
| 1258-1260 | INFO | Inference executor shutdown | Clean shutdown confirmation | **HIGH** |
| 1315 | INFO | `Task complete. Ran X/Y trials (Z% efficiency gain)` | Final summary of efficiency | **HIGH** |

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
| 2981-3113 | INFO | Live optimal stopping flow messages | Live function flow | **HIGH** |

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
| 2995-3019 | INFO | Ordinal task identification logging | Ordinal routing | **MEDIUM** |
| 3053-3055 | INFO | Worker/GPU summary for live | Parallelization info | **MEDIUM** |

### MEDIUM - Cache Performance (NEW - Our Optimizations)

| Line | Type | Statement | Purpose | Criticality |
|------|------|-----------|---------|-------------|
| 2242 | INFO | `✓ CACHE HIT: Reusing binary group model (n_items=X)` | Binary cache hit | **MEDIUM** |
| 2245 | INFO | `✗ CACHE INVALIDATED: Binary model n_items X → Y` | Binary cache invalidation | **MEDIUM** |
| 2263 | INFO | `✗ CACHE MISS: Created new binary group model (n_items=X)` | Binary cache miss | **MEDIUM** |
| 2298 | INFO | `✓ CACHE HIT: Reusing continuous group model (n_items=X)` | Continuous cache hit | **MEDIUM** |
| 2301 | INFO | `✗ CACHE INVALIDATED: Continuous model n_items X → Y` | Continuous cache invalidation | **MEDIUM** |
| 2375 | INFO | `✗ CACHE MISS: Created new continuous group model (n_items=X)` | Continuous cache miss | **MEDIUM** |

### LOW - Timing Tests & Debug

| Line | Type | Statement | Purpose | Criticality |
|------|------|-----------|---------|-------------|
| 158-160 | DEBUG | PyMC stdout/stderr capture | Debugging PyMC output | **LOW** |
| 307-569 | INFO | Diagnostic plot generation messages | Posthoc diagnostics | **LOW** |
| 426-435 | INFO | Accuracy metrics (posthoc diagnostics) | Posthoc analysis | **LOW** |
| 966-988 | INFO/WARNING | Worker JAX GPU setup details | GPU debugging | **LOW** |
| 1109 | INFO | Detailed stop message variant | Alternative format | **LOW** |
| 1460-1482 | INFO/WARNING | Live worker JAX GPU setup | GPU debugging | **LOW** |
| 2053-2055 | WARNING/ERROR | Diagnostic plot failures | Posthoc diagnostic errors | **LOW** |
| 2638 | WARNING | `🕐 TIMING_TEST: Binary group inference took Xs` | **Performance profiling** | **LOW** |
| 2720 | WARNING | `🕐 TIMING_TEST: Ordinal group inference took Xs` | **Performance profiling** | **LOW** |
| 2765 | WARNING | `🕐 TIMING_TEST: Continuous aggregation took Xs` | **Performance profiling** | **LOW** |
| 2820 | WARNING | `🕐 TIMING_TEST: Continuous MCMC took Xs` | **Performance profiling** | **LOW** |
| 2882 | WARNING | `🕐 TIMING_TEST: Continuous TOTAL took Xs` | **Performance profiling** | **LOW** |
| 2889 | WARNING | `🕐 TIMING_TEST: optimal_stopping_live_single TOTAL took Xs` | **Performance profiling** | **LOW** |

---

## 3. ordinal_model.py (Ordinal Inference)

### MEDIUM - Cache Performance

| Line | Type | Statement | Purpose | Criticality |
|------|------|-----------|---------|-------------|
| 336 | DEBUG | `Reusing OrderedLogistic model (updated X → Y items)` | Ordinal cache hit | **MEDIUM** |
| 348 | DEBUG | `Created new OrderedLogistic model for X items` | Ordinal cache miss | **MEDIUM** |

### LOW - Debug & Profiling

| Line | Type | Statement | Purpose | Criticality |
|------|------|-----------|---------|-------------|
| 354-370 | WARNING | `🔍 _ordinal_entropy_ci_adaptive called: n_samples=X...` | **Detailed parameter trace** | **LOW** |
| 376 | ERROR | `OrderedLogistic sampling failed` | Sampling error | **MEDIUM-LOW** |
| 413 | DEBUG | Entropy computation details | Technical trace | **LOW** |
| 564 | DEBUG | Ordinal decision details | Decision logic trace | **LOW** |
| 581 | INFO | Ordinal stopping decision | Stop via modal/entropy | **HIGH** |
| 603 | DEBUG | Continue: insufficient history | Not enough data | **LOW** |
| 628 | INFO | Ordinal stopping via stabilization | Stabilization stop | **HIGH** |
| 645 | DEBUG | Detailed stabilization logic | Technical trace | **LOW** |

---

## 4. gpu_utils.py (GPU Configuration)

### MEDIUM - Configuration & Warnings

| Line | Type | Statement | Purpose | Criticality |
|------|------|-----------|---------|-------------|
| 230 | INFO | `JAX GPU acceleration available: X GPUs` | GPU detection | **MEDIUM** |
| 232 | WARNING | `JAX GPU detected but test failed` | GPU issue | **MEDIUM** |
| 234 | INFO | JAX backend and device info | GPU status | **MEDIUM** |
| 237 | INFO | `JAX not available - checking other backends` | Fallback info | **MEDIUM** |
| 239 | WARNING | JAX GPU detection error | Detection failure | **MEDIUM** |
| 253 | INFO | PyTensor GPU backend available | GPU detection | **MEDIUM** |
| 264 | INFO | `GPU acceleration enabled via X` | GPU confirmed | **MEDIUM** |
| 267 | WARNING | `System GPUs detected but no PyMC GPU backend available` | GPU not usable | **MEDIUM** |
| 269 | INFO | `No GPU acceleration - using CPU` | CPU fallback | **MEDIUM** |
| 298 | INFO | GPU recommendation | Setup advice | **MEDIUM** |
| 322-332 | INFO/WARNING | JAX GPU configuration messages | GPU setup status | **MEDIUM** |
| 560-676 | INFO | GPU decision and sampling configuration | Hardware setup | **MEDIUM** |
| 692-736 | INFO | Enhanced GPU status report (full block) | Detailed GPU report | **MEDIUM** |
| 820-848 | INFO/WARNING | GPU validation and worker assignment | GPU orchestration | **MEDIUM** |

---

## Summary Statistics

### By File:
- **early_stopping.py**: 43 statements (5 CRITICAL, 11 HIGH, 20 MEDIUM, 7 LOW)
- **rule.py**: 127 statements (5 CRITICAL, 75 HIGH, 25 MEDIUM, 22 LOW)
- **ordinal_model.py**: 11 statements (0 CRITICAL, 2 HIGH, 2 MEDIUM, 7 LOW)
- **gpu_utils.py**: 41 statements (0 CRITICAL, 0 HIGH, 41 MEDIUM, 0 LOW)

### By Criticality:
- **CRITICAL**: 10 statements (must always be visible)
- **HIGH**: 88 statements (should be visible by default)
- **MEDIUM**: 88 statements (helpful for monitoring/debugging)
- **LOW**: 36 statements (development/profiling only)

---

## Recommendations

### 1. Logging Level Configuration

**Suggested default for inspect_ai users:**
```python
logging.basicConfig(level=logging.INFO)
```

This shows:
- ✅ All CRITICAL errors
- ✅ All HIGH stopping decisions and progress
- ✅ All MEDIUM warnings and status
- ❌ LOW debug traces hidden

### 2. Problematic Statements to Review

#### LOW priority logged as WARNING (should be DEBUG):
- **Lines 354-370 (ordinal_model.py)**: Parameter dump - should be DEBUG
- **Lines 2638, 2720, 2765, 2820, 2882, 2889 (rule.py)**: TIMING_TEST - should be DEBUG

**Recommendation**: Change these from `logger.warning()` to `logger.debug()`

#### PRINT statements in production code:
- **Lines 338-417 (early_stopping.py)**: Configuration summary
  - **Issue**: Uses `print()` instead of `logger.info()`
  - **Impact**: Can't be suppressed or redirected
  - **Recommendation**: Convert to `logger.info()` with optional `verbose` parameter

- **Line 2060 (rule.py)**: `print(f"Run complete. See log file...")`
  - **Recommendation**: Convert to `logger.info()`

### 3. Missing Logging

**Optimization #1 (skip stopped groupings)**:
- Line 958: Currently DEBUG level
- **Recommendation**: Keep as DEBUG (would be too verbose otherwise)

**Cache invalidation**:
- Currently INFO level (lines 2242, 2245, 2263, 2298, 2301, 2375)
- **Recommendation**: Keep as INFO for now, consider moving to DEBUG after validation

### 4. User Experience

**For typical inspect_ai user:**
1. Set `logging.basicConfig(level=logging.INFO)` in their script
2. See configuration summary at start
3. See progress messages every reanalysis_interval
4. See stopping decisions as they happen
5. See final efficiency summary
6. NOT see: Debug traces, cache hits, timing tests

**For power user debugging:**
1. Set `logging.basicConfig(level=logging.DEBUG)`
2. See everything including cache behavior and internal traces

**For performance profiling:**
1. Enable DEBUG level
2. Grep for "TIMING_TEST" in logs
3. Analyze where time is spent

---

## Proposed Changes

### High Priority:
1. Convert TIMING_TEST warnings (lines 2638, 2720, 2765, 2820, 2882, 2889) to DEBUG
2. Convert ordinal parameter dump (lines 354-370) to DEBUG
3. Convert print() statements to logger.info() with optional verbosity control

### Medium Priority:
4. Consider moving cache hit/miss from INFO to DEBUG after validation period
5. Add structured logging option (JSON format) for programmatic parsing

### Low Priority:
6. Add log rotation configuration guidance
7. Document recommended logging levels in user guide
