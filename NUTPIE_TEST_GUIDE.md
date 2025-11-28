# Nutpie Performance Test Guide

**Purpose:** Run large-scale bridge tests to verify nutpie integration and measure performance improvements.

---

## Quick Start

### 1. Verify Nutpie Installation

```bash
cd /home/ubuntu/optstop
source venv/bin/activate

# Check if nutpie is available
python -c "from optstop.gpu_utils import check_nutpie_available; available, version = check_nutpie_available(); print(f'Nutpie available: {available}, version: {version}')"
```

**Expected output:**
```
Nutpie available: True, version: 0.16.4
```

If nutpie is NOT available:
```bash
pip install nutpie
```

---

## 2. Run the Test

### Option A: Using the helper script (recommended)

```bash
cd /home/ubuntu/optstop
./scripts/run_nutpie_test.sh
```

### Option B: Run directly in tmux

```bash
# Attach to existing tmux session
tmux attach -t bridge_test

# Or create new session
tmux new -s bridge_test

# Activate venv and run test
cd /home/ubuntu/optstop
source venv/bin/activate
PYTHONPATH=/home/ubuntu/optstop python scripts/test_large_datasets_bridge_with_nutpie.py
```

**Note:** The test will automatically:
- ✅ Check nutpie availability
- ✅ Backup previous results to `test_outputs/large_scale/backup_TIMESTAMP/`
- ✅ Run all 3 datasets with nutpie tracking
- ✅ Generate performance comparison with baseline

---

## 3. Monitor Progress

### While test is running (in another terminal):

```bash
# Watch the log files
tail -f /home/ubuntu/optstop/test_outputs/large_scale/dataset_1.log
tail -f /home/ubuntu/optstop/test_outputs/large_scale/dataset_2.log
tail -f /home/ubuntu/optstop/test_outputs/large_scale/dataset_3.log
```

### Check for nutpie usage in logs:

```bash
# Should see: "Selected NUTS sampler: nutpie"
grep "Selected NUTS sampler" /home/ubuntu/optstop/test_outputs/large_scale/dataset_*.log
```

**Expected output:**
```
dataset_1.log:Selected NUTS sampler: nutpie (Rust-based v0.16.4)
dataset_2.log:Selected NUTS sampler: nutpie (Rust-based v0.16.4)
dataset_3.log:Selected NUTS sampler: nutpie (Rust-based v0.16.4)
```

---

## 4. Review Results

### Generate comparison report:

```bash
cd /home/ubuntu/optstop
source venv/bin/activate
python scripts/generate_comparison_report.py
```

**Example output:**
```
================================================================================
PERFORMANCE COMPARISON REPORT: Baseline vs. Nutpie
================================================================================

Dataset 1: Binary Discrete
  Baseline runtime: 8.3 min
  Current runtime: 2.1 min
  Speedup: 3.95×
  Time saved: 6.2 min

Dataset 2: Ordinal Discrete (Hybrid)
  Baseline runtime: 550.2 min (9.2 hours)
  Current runtime: 45.3 min
  Speedup: 12.15×
  Time saved: 504.9 min (8.4 hours)

Dataset 3: Continuous Bounded
  Baseline runtime: 4.1 min
  Current runtime: 1.2 min
  Speedup: 3.42×
  Time saved: 2.9 min
```

### View detailed diagnostics:

```bash
# View full diagnostics JSON
cat /home/ubuntu/optstop/test_outputs/large_scale/dataset_2_diagnostics.json | jq '.comparison'

# Check inference timing breakdown
cat /home/ubuntu/optstop/test_outputs/large_scale/dataset_2_diagnostics.json | jq '.inference_timing'

# Verify nutpie usage
cat /home/ubuntu/optstop/test_outputs/large_scale/dataset_2_diagnostics.json | jq '.sampler_selection'
```

---

## Expected Test Duration

### With nutpie installed:

| Dataset | Pathway | Expected Duration |
|---------|---------|-------------------|
| Dataset 1 | Binary | ~2-5 min |
| Dataset 2 | Ordinal Hybrid | ~30-60 min* |
| Dataset 3 | Continuous | ~1-3 min |

**Total:** ~35-70 minutes

*With draws=500, tune=500. Would be 8-10 hours with default draws=6000!

### Without nutpie (baseline):

| Dataset | Pathway | Expected Duration |
|---------|---------|-------------------|
| Dataset 1 | Binary | ~8-10 min |
| Dataset 2 | Ordinal Hybrid | ~8-10 hours |
| Dataset 3 | Continuous | ~3-5 min |

**Total:** ~8-11 hours

---

## Test Configuration

### Dataset 1: Binary Discrete
- **Pathway:** Binary discrete inference
- **Samples:** 500
- **Epochs:** 10
- **Total trials:** 5,000
- **Nutpie benefit:** 2-5× speedup

### Dataset 2: Ordinal Discrete (HYBRID)
- **Pathway:** Ordinal discrete inference (HYBRID mode)
- **Samples:** 500
- **Epochs:** 10
- **Total trials:** 5,000
- **MCMC parameters:** draws=500, tune=500, chains=2
- **Nutpie benefit:** 2-5× speedup on top of MCMC reduction
- **Combined optimization:** 120-300× vs. defaults!

### Dataset 3: Continuous Bounded
- **Pathway:** Continuous bounded inference (Beta model)
- **Samples:** 500
- **Epochs:** 10
- **Total trials:** 5,000
- **Nutpie benefit:** 2-5× speedup

---

## What the Test Tracks

### 1. Nutpie Detection
- ✅ Checks if nutpie is installed
- ✅ Captures nutpie version
- ✅ Logs when nutpie is unavailable

### 2. Sampler Selection
- ✅ Records which sampler was used (nutpie, pymc, or numpyro)
- ✅ Tracks nutpie version used
- ✅ Confirms nutpie is actually invoked (not just detected)

### 3. Performance Metrics
- ✅ Total runtime (seconds, minutes, hours)
- ✅ Per-inference timing breakdown
- ✅ Number of inference calls
- ✅ Mean/min/max inference times

### 4. Comparison with Baseline
- ✅ Speedup factor (e.g., 3.5×)
- ✅ Time saved (absolute and percentage)
- ✅ Sampler comparison (baseline vs. current)
- ✅ Trial counts and efficiency

---

## Troubleshooting

### Issue: Nutpie not detected

**Symptom:** Logs show "Selected NUTS sampler: pymc"

**Solution:**
```bash
# Install nutpie
pip install nutpie

# Verify installation
python -c "import nutpie; print(nutpie.__version__)"
```

### Issue: Test runs very slowly (>1 hour for Dataset 2)

**Check:**
1. Verify nutpie is being used: `grep "nutpie" test_outputs/large_scale/dataset_2.log`
2. Check MCMC parameters: Should be draws=500, tune=500, chains=2
3. Verify ordinal_inference='hybrid' (not 'entropy' only)

**Expected behavior:**
- Dataset 2 should complete in 30-60 minutes with nutpie + optimized MCMC
- If taking 8+ hours, nutpie is likely NOT being used

### Issue: No baseline for comparison

**Symptom:** Report says "No baseline available for comparison"

**Solution:** This is normal for first run. The current run becomes the baseline for future runs.

---

## Files Generated

### Results:
- `test_outputs/large_scale/dataset_1_diagnostics.json` - Dataset 1 results
- `test_outputs/large_scale/dataset_2_diagnostics.json` - Dataset 2 results
- `test_outputs/large_scale/dataset_3_diagnostics.json` - Dataset 3 results

### Logs:
- `test_outputs/large_scale/dataset_1.log` - Full execution log
- `test_outputs/large_scale/dataset_2.log` - Full execution log
- `test_outputs/large_scale/dataset_3.log` - Full execution log

### Backups:
- `test_outputs/large_scale/backup_YYYYMMDD_HHMMSS/` - Previous run results

---

## Next Steps After Testing

### If nutpie works well:

1. **Update documentation:** Confirm BRIDGE_API_REFERENCE.md guidance is accurate
2. **Publish findings:** Share performance improvements with users
3. **Consider making nutpie recommended:** Add to main installation instructions

### If you want even faster ordinal inference:

1. **Switch to modal mode** for production:
   ```python
   ordinal_inference='modal'  # ~0.1s instead of minutes
   ```

2. **Trade-off:** Modal mode may not stop for truly diffuse distributions, but works for 80-90% of cases

3. **See:** `BRIDGE_API_REFERENCE.md` Performance Considerations section

---

## Related Documentation

- **Nutpie Integration:** `NUTPIE_INTEGRATION.md`
- **Coverage Verification:** `NUTPIE_COMPLETE_VERIFICATION.md`
- **User Guide:** `BRIDGE_API_REFERENCE.md` (Performance Considerations section)
- **Review Summary:** `NUTPIE_REVIEW_SUMMARY.md`

---

**Last Updated:** 2025-11-28
**Status:** Ready for testing
