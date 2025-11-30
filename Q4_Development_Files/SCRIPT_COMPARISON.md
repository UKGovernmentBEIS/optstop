# Script Comparison: Original vs. Nutpie-Enhanced Test

**Comparison Date:** 2025-11-28

---

## Configuration Comparison

### Base MCMC Parameters

| Parameter | Original | Nutpie-Enhanced | Match? |
|-----------|----------|-----------------|--------|
| delta_item | 0.15 | 0.15 | ✅ YES |
| delta_cap | 0.10 | 0.10 | ✅ YES |
| cred_level | 0.95 | 0.95 | ✅ YES |
| conservatism | 10 | 10 | ✅ YES |
| draws | 500 | 500 | ✅ YES |
| tune | 500 | 500 | ✅ YES |
| grouping_columns | ['model', 'task'] | ['model', 'task'] | ✅ YES |
| reanalysis_interval | 25 | 25 | ✅ YES |
| min_samples_per_grouping | 10 | 10 | ✅ YES |

**Status:** ✅ **IDENTICAL BASE CONFIGURATION**

---

## Dataset 1: Binary Discrete

| Setting | Original | Nutpie-Enhanced | Match? |
|---------|----------|-----------------|--------|
| Dataset path | dataset_1_binary_discrete.csv | dataset_1_binary_discrete.csv | ✅ YES |
| Score columns | ['score'] | ['score'] | ✅ YES |
| Additional config | None | None | ✅ YES |
| Expected routing | BINARY | BINARY | ✅ YES |

**Status:** ✅ **IDENTICAL**

---

## Dataset 2: Ordinal Discrete

| Setting | Original | Nutpie-Enhanced | Match? |
|---------|----------|-----------------|--------|
| Dataset path | dataset_2_ordinal_with_choice.csv | dataset_2_ordinal_with_choice.csv | ✅ YES |
| Score columns | ['accuracy', 'precision', 'recall'] | ['accuracy', 'precision', 'recall'] | ✅ YES |
| score_choice | 'accuracy' | 'accuracy' | ✅ YES |
| ordinal_max_score | 10 | 10 | ✅ YES |
| ordinal_tasks | ['math_easy', 'math_hard', 'coding_medium', 'creative_writing', 'reasoning'] | ['math_easy', 'math_hard', 'coding_medium', 'creative_writing', 'reasoning'] | ✅ YES |
| ordinal_inference | (default='hybrid') | 'hybrid' (explicit) | ✅ YES |
| Expected routing | ORDINAL | ORDINAL | ✅ YES |

**Status:** ✅ **FUNCTIONALLY IDENTICAL**
- Original uses default ordinal_inference (which is 'hybrid')
- Nutpie-enhanced explicitly sets ordinal_inference='hybrid'
- **Result: Both run with ordinal_inference='hybrid'**

---

## Dataset 3: Continuous Bounded

| Setting | Original | Nutpie-Enhanced | Match? |
|---------|----------|-----------------|--------|
| Dataset path | dataset_3_ordinal_with_agg.csv | dataset_3_ordinal_with_agg.csv | ✅ YES |
| Score columns | ['scorer_1', 'scorer_2', 'scorer_3'] | ['scorer_1', 'scorer_2', 'scorer_3'] | ✅ YES |
| score_agg | 'mean' | 'mean' | ✅ YES |
| ordinal_max_score | 10 | 10 | ✅ YES |
| ordinal_tasks | ['math_easy', 'math_hard', 'coding_medium', 'creative_writing', 'reasoning'] | ['math_easy', 'math_hard', 'coding_medium', 'creative_writing', 'reasoning'] | ✅ YES |
| Expected routing | CONTINUOUS | CONTINUOUS | ✅ YES |

**Status:** ✅ **IDENTICAL**

---

## Additional Features in Nutpie-Enhanced Script

The nutpie-enhanced script adds **tracking features only** - no changes to actual test configuration:

### 1. Nutpie Environment Detection
```python
nutpie_env = check_nutpie_environment()
# Captures: nutpie_available, nutpie_version, detection_time
```

### 2. Sampler Selection Tracking
```python
sampler_selection = parse_sampler_selection(log_path)
# Captures: nutpie_used, pymc_used, numpyro_used, sampler, nutpie_version
```

### 3. Inference Timing Breakdown
```python
inference_timing = parse_inference_timing(log_path)
# Captures: per-pathway timing (count, total, mean, min, max)
```

### 4. Performance Comparison
```python
comparison = compare_with_baseline(results, baseline_path)
# Compares: runtime, speedup, time_saved, sampler differences
```

### 5. Automatic Backup
- Backs up previous results before running
- Stored in: `test_outputs/large_scale/backup_YYYYMMDD_HHMMSS/`

---

## Verification: Default ordinal_inference

**OptimalStoppingManager default:**
```python
ordinal_inference: str = 'hybrid',  # Line 116 in early_stopping.py
```

**Result:** Original script was already using 'hybrid' mode (the default).

---

## Output Differences

### Original Script Outputs:
- `dataset_1_diagnostics.json`
- `dataset_2_diagnostics.json`
- `dataset_3_diagnostics.json`
- `dataset_1.log`, `dataset_2.log`, `dataset_3.log`

### Nutpie-Enhanced Script Outputs:
- `dataset_1_diagnostics.json` (with additional fields)
- `dataset_2_diagnostics.json` (with additional fields)
- `dataset_3_diagnostics.json` (with additional fields)
- `dataset_1.log`, `dataset_2.log`, `dataset_3.log`
- `backup_YYYYMMDD_HHMMSS/` (previous results)

### Additional Fields in JSON:
```json
{
  "test_date": "2025-11-28T14:05:00",
  "nutpie_environment": {
    "nutpie_available": true,
    "nutpie_version": "0.16.4",
    "detection_time": "..."
  },
  "sampler_selection": {
    "nutpie_used": true,
    "sampler": "nutpie",
    "nutpie_version": "0.16.4"
  },
  "inference_timing": {
    "ordinal": {
      "count": 12,
      "total_sec": 234.5,
      "mean_sec": 19.54,
      ...
    }
  },
  "comparison": {
    "runtime_comparison": {
      "baseline_sec": 33011.6,
      "current_sec": 2450.3,
      "speedup": 13.47,
      ...
    }
  }
}
```

---

## Conclusion

### ✅ **DATASETS AND CONFIGURATION ARE IDENTICAL**

The nutpie-enhanced script:
- ✅ Uses the **exact same datasets** (all 3)
- ✅ Uses the **exact same MCMC parameters** (draws=500, tune=500)
- ✅ Uses the **exact same user setup** (grouping, reanalysis_interval, etc.)
- ✅ Uses the **exact same ordinal mode** (hybrid)
- ✅ **Only adds tracking/comparison features** (no behavior changes)

### Key Points:

1. **ordinal_inference='hybrid'** was already the default in the original script
2. The new script **explicitly sets** it for clarity (functionally identical)
3. **No changes to inference algorithms or stopping behavior**
4. **Only additions:** nutpie detection, timing tracking, performance comparison

### Expected Results:

**If nutpie is installed and working:**
- Same stopping decisions as original
- Same efficiency metrics
- **Faster runtime** (2-5× speedup expected)
- Additional diagnostic information

**The only observable difference should be:**
- ⏱️ **Faster execution time** (if nutpie is used)
- 📊 **Additional tracking data** in output JSON
- 📁 **Automatic backup** of previous results

---

## Verification Commands

### Check if configurations match:
```bash
# Compare optstop_params
diff <(grep -A 6 "optstop_params = {" scripts/test_large_datasets_bridge.py) \
     <(grep -A 6 "optstop_params = {" scripts/test_large_datasets_bridge_with_nutpie.py)

# Compare base_config
diff <(grep -A 5 "base_config = {" scripts/test_large_datasets_bridge.py) \
     <(grep -A 5 "base_config = {" scripts/test_large_datasets_bridge_with_nutpie.py)

# Compare dataset paths
grep "dataset_path=" scripts/test_large_datasets_bridge.py
grep "dataset_path=" scripts/test_large_datasets_bridge_with_nutpie.py
```

### Verify ordinal_inference default:
```bash
# Check OptimalStoppingManager default
grep "ordinal_inference:" optstop/early_stopping.py | head -1
```

**Expected:** `ordinal_inference: str = 'hybrid',`

---

**Comparison Date:** 2025-11-28
**Status:** ✅ Verified identical configuration
**Only difference:** Additional tracking features (no behavioral changes)
