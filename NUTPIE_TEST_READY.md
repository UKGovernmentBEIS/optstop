# ✅ Ready for Nutpie Performance Testing

**Date:** 2025-11-28
**Status:** All scripts configured and ready to run

---

## Summary

Your large-scale bridge test is now enhanced to:

✅ **Track nutpie usage** - Confirms nutpie is detected and actively used
✅ **Capture performance metrics** - Per-inference timing and total runtime
✅ **Compare with baseline** - Automatic comparison with previous runs (stored in test_outputs/large_scale/)
✅ **Use ordinal hybrid mode** - Dataset 2 will test ordinal discrete with hybrid inference
✅ **Backup previous results** - Automatic timestamped backups before each run

---

## Current Environment

**Nutpie Status:** ✅ **INSTALLED**
```
Version: 0.16.4
Expected speedup: 2-5× for CPU sampling
```

**Previous Test Results Available:**
```
test_outputs/large_scale/
├── dataset_1_diagnostics.json (Binary: 8.3 min)
├── dataset_2_diagnostics.json (Ordinal Hybrid: 9.2 hours ⚠️)
├── dataset_3_diagnostics.json (Continuous: 4.1 min)
└── *.log files
```

**Baseline Dataset 2 Performance:**
- Runtime: **33,011 seconds (9.2 hours)**
- Configuration: draws=500, tune=500, no nutpie indicated
- This is the key performance target for improvement!

---

## What Will Be Tested

### Dataset 1: Binary Discrete
- **Expected with nutpie:** ~2-5 min (vs 8.3 min baseline)
- **Expected speedup:** 2-4×

### Dataset 2: Ordinal Discrete (HYBRID) 🎯 **KEY TEST**
- **Expected with nutpie:** ~30-60 min (vs 9.2 hours baseline)
- **Expected speedup:** 10-20×
- **Why faster:**
  - Nutpie: 2-5× speedup
  - Already using draws=500, tune=500 (vs defaults of 6000)
  - Combined: 120-300× vs. out-of-box defaults!

### Dataset 3: Continuous Bounded
- **Expected with nutpie:** ~1-3 min (vs 4.1 min baseline)
- **Expected speedup:** 2-4×

**Total Expected Runtime:** ~35-70 minutes (vs ~10 hours baseline)

---

## How to Run

### Option 1: Quick Start (Recommended)

```bash
cd /home/ubuntu/optstop
./scripts/run_nutpie_test.sh
```

This will:
1. Check nutpie availability (already installed ✅)
2. Backup previous results automatically
3. Run all 3 datasets with tracking
4. Generate comparison report

### Option 2: Run in Tmux Session

```bash
# Attach to existing session or create new one
tmux attach -t bridge_test
# OR
tmux new -s bridge_test

# Inside tmux:
cd /home/ubuntu/optstop
source venv/bin/activate
PYTHONPATH=/home/ubuntu/optstop python scripts/test_large_datasets_bridge_with_nutpie.py
```

**Tip:** Detach from tmux with `Ctrl+B`, then `D`

---

## What Gets Tracked

### 1. Nutpie Detection ✅
```json
{
  "nutpie_environment": {
    "nutpie_available": true,
    "nutpie_version": "0.16.4",
    "detection_time": "2025-11-28T14:05:00"
  }
}
```

### 2. Sampler Selection ✅
```json
{
  "sampler_selection": {
    "nutpie_used": true,
    "pymc_used": false,
    "numpyro_used": false,
    "nutpie_version": "0.16.4",
    "sampler": "nutpie"
  }
}
```

### 3. Inference Timing ✅
```json
{
  "inference_timing": {
    "ordinal": {
      "count": 12,
      "total_sec": 234.5,
      "mean_sec": 19.54,
      "min_sec": 15.2,
      "max_sec": 25.8
    }
  }
}
```

### 4. Performance Comparison ✅
```json
{
  "comparison": {
    "runtime_comparison": {
      "baseline_sec": 33011.6,
      "current_sec": 2450.3,
      "speedup": 13.47,
      "time_saved_sec": 30561.3,
      "percent_change": 92.6
    },
    "nutpie_comparison": {
      "baseline_sampler": "pymc",
      "current_sampler": "nutpie"
    }
  }
}
```

---

## Monitoring Progress

### During Test Execution:

**Terminal 1:** Run the test
```bash
cd /home/ubuntu/optstop
./scripts/run_nutpie_test.sh
```

**Terminal 2:** Monitor logs
```bash
# Watch Dataset 2 progress (the long one)
tail -f /home/ubuntu/optstop/test_outputs/large_scale/dataset_2.log

# Check for nutpie usage
grep "Selected NUTS sampler" test_outputs/large_scale/dataset_2.log
```

**Expected log output:**
```
Selected NUTS sampler: nutpie (Rust-based v0.16.4 (2-5× faster than PyMC default))
Configured sampling for CPU: nutpie (Rust-based v0.16.4) - chains=2, cores=2
```

---

## After Testing: Generate Report

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

Dataset 2: Ordinal Discrete (Hybrid)
  Baseline runtime: 550.2 min (9.2 hours)
  Current runtime: 45.3 min
  Speedup: 12.15×
  Time saved: 504.9 min (8.4 hours)
  ✅ Significant improvement!

TOTAL ACROSS ALL DATASETS
  Overall speedup: 10.34×
  Total time saved: 8.6 hours
```

---

## Expected Timeline

```
[Start] Backup previous results (< 1 sec)
    ↓
[00:00] Dataset 1 starts (Binary)
[00:02-00:05] Dataset 1 completes ✓
    ↓
[00:05] Dataset 2 starts (Ordinal Hybrid) ← MAIN TEST
[00:35-01:05] Dataset 2 completes ✓
    ↓
[01:05] Dataset 3 starts (Continuous)
[01:07-01:10] Dataset 3 completes ✓
    ↓
[01:10] Generate comparison report
[01:10] Done! ✅
```

**Total: ~35-70 minutes**

---

## Success Criteria

### ✅ Test Passes If:

1. **Nutpie is detected and used:**
   - Logs show: `"Selected NUTS sampler: nutpie"`
   - diagnostics.json shows: `"nutpie_used": true`

2. **Performance improvement for Dataset 2:**
   - Current runtime < 2 hours (vs. 9.2 hours baseline)
   - Speedup > 5×

3. **All datasets complete successfully:**
   - No errors or crashes
   - Efficiency metrics generated
   - Comparison with baseline available

### 📊 Validation Checks:

```bash
# After test completes, verify:

# 1. Nutpie was used
grep -q "nutpie_used.*true" test_outputs/large_scale/dataset_2_diagnostics.json && echo "✅ Nutpie used" || echo "❌ Nutpie NOT used"

# 2. Dataset 2 runtime reasonable (<2 hours)
python -c "import json; d=json.load(open('test_outputs/large_scale/dataset_2_diagnostics.json')); t=d['execution']['run_time_sec']; print(f'✅ Runtime OK: {t/60:.1f} min') if t < 7200 else print(f'⚠️ Runtime high: {t/60:.1f} min')"

# 3. All datasets completed
ls test_outputs/large_scale/dataset_*_diagnostics.json | wc -l | grep -q 3 && echo "✅ All 3 datasets completed" || echo "⚠️ Some datasets missing"
```

---

## Troubleshooting

### If nutpie is NOT used:

**Check:**
```bash
source venv/bin/activate
python -c "from optstop.gpu_utils import check_nutpie_available; print(check_nutpie_available())"
```

**Fix:**
```bash
pip install nutpie
```

### If Dataset 2 takes >2 hours:

**Possible causes:**
1. Nutpie not being used (check logs)
2. MCMC parameters reverted to defaults (check configuration)
3. ordinal_inference set to 'entropy' instead of 'hybrid'

**Check logs:**
```bash
grep "draws" test_outputs/large_scale/dataset_2.log | head -5
grep "ordinal.*inference" test_outputs/large_scale/dataset_2.log | head -5
```

---

## Files Created

### Test Scripts:
- ✅ `scripts/test_large_datasets_bridge_with_nutpie.py` - Enhanced test script
- ✅ `scripts/run_nutpie_test.sh` - Helper script to run tests
- ✅ `scripts/generate_comparison_report.py` - Comparison report generator

### Documentation:
- ✅ `NUTPIE_TEST_GUIDE.md` - Detailed testing instructions
- ✅ `NUTPIE_COMPLETE_VERIFICATION.md` - Full integration verification
- ✅ `NUTPIE_TEST_READY.md` - This file (summary and quick start)

### Package Updates:
- ✅ `setup.py` - Added `extras_require['performance']` for nutpie
- ✅ `pyproject.toml` - Added `[project.optional-dependencies].performance`

---

## Quick Reference Commands

```bash
# Run test
./scripts/run_nutpie_test.sh

# Monitor progress
tail -f test_outputs/large_scale/dataset_2.log

# Check nutpie usage
grep "Selected NUTS sampler" test_outputs/large_scale/dataset_*.log

# Generate report
python scripts/generate_comparison_report.py

# View detailed results
cat test_outputs/large_scale/dataset_2_diagnostics.json | jq '.comparison'
```

---

## Ready to Run?

Everything is configured and nutpie is already installed (v0.16.4). You can start the test immediately:

```bash
cd /home/ubuntu/optstop
./scripts/run_nutpie_test.sh
```

Or run in tmux for background execution:

```bash
tmux new -s bridge_test
cd /home/ubuntu/optstop
source venv/bin/activate
PYTHONPATH=/home/ubuntu/optstop python scripts/test_large_datasets_bridge_with_nutpie.py
# Press Ctrl+B, then D to detach
```

**Expected completion:** 35-70 minutes
**Main improvement:** Dataset 2 should drop from 9.2 hours to ~30-60 minutes! 🚀

---

**Last Updated:** 2025-11-28
**Status:** ✅ Ready to run
**Nutpie:** ✅ Installed (v0.16.4)
**Baseline:** ✅ Available for comparison
