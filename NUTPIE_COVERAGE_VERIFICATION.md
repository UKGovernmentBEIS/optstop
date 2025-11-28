# Nutpie Coverage Verification

**Date**: 2025-11-28
**Status**: ✅ All paths verified

---

## Sampling Call Flow Analysis

### **1. Binary Inference (rule.py)**

**Path 1a: optimal_stopping_posthoc() - Item level**
```
rule.py:996  → sampling_kwargs = get_sampling_kwargs(...)
rule.py:1086 → compute_kwargs=sampling_kwargs (passed to _binary_ci_adaptive)
rule.py:1185 → trace = pm.sample(**sampling_kwargs)
```
✅ **Uses nutpie:** sampling_kwargs contains nuts_sampler

**Path 1b: optimal_stopping_live() - Item level**
```
rule.py:1490 → sampling_kwargs = get_sampling_kwargs(...)
rule.py:1638 → compute_kwargs=sampling_kwargs
rule.py:1693 → trace = pm.sample(**sampling_kwargs)
```
✅ **Uses nutpie:** sampling_kwargs contains nuts_sampler

**Path 1c: optimal_stopping_live_single() - Group level**
```
rule.py:2183 → sampling_kwargs = get_sampling_kwargs(...)
rule.py:2569 → trace = pm.sample(**sampling_kwargs)
```
✅ **Uses nutpie:** sampling_kwargs contains nuts_sampler

---

### **2. Ordinal Inference (ordinal_model.py)**

**All ordinal paths:**
```
rule.py:2183 → sampling_kwargs = get_sampling_kwargs(...)
rule.py:2447 → compute_kwargs=sampling_kwargs (passed to _ordinal_entropy_ci_adaptive)
ordinal_model.py:366 → default_kwargs.update(compute_kwargs)
ordinal_model.py:374 → trace = pm.sample(**default_kwargs)
```
✅ **Uses nutpie:** compute_kwargs merged into default_kwargs, includes nuts_sampler

---

### **3. Continuous Inference (rule.py)**

**Path 3: optimal_stopping_live_single() - Group level**
```
rule.py:2183 → sampling_kwargs = get_sampling_kwargs(...)
rule.py:2810 → trace = pm.sample(**sampling_kwargs)
```
✅ **Uses nutpie:** sampling_kwargs contains nuts_sampler

---

### **4. Convergence Analysis (convergence.py)**

**Path 4: convergence_posthoc()**
```
convergence.py:314 → sampling_kwargs = get_sampling_kwargs(...)
convergence.py:446 → compute_kwargs=sampling_kwargs
convergence.py:548 → trace = pm.sample(**sampling_kwargs)
```
✅ **Uses nutpie:** sampling_kwargs contains nuts_sampler

---

### **5. Bridge Integration (early_stopping.py)**

**Path 5: OptimalStoppingManager**
```
early_stopping.py:1019 → sampling_kwargs = get_sampling_kwargs(...)
early_stopping.py:1056 → passes sampling_kwargs to optimal_stopping_live_single()
→ Flows into paths 1c, 2, 3 above
```
✅ **Uses nutpie:** sampling_kwargs passed to rule.py functions

---

## get_sampling_kwargs() Branch Coverage

### **Branch 1: GPU Available, JAX Backend**
```python
gpu_available=True, gpu_backend='jax-gpu'
→ Line 729: sampling_kwargs['nuts_sampler'] = 'numpyro'
```
✅ **Correct:** Uses numpyro for GPU

### **Branch 2: GPU Available, PyTensor Backend**
```python
gpu_available=True, gpu_backend='pytensor-gpu'
→ Line 739: No nuts_sampler set (PyTensor handles GPU internally)
```
✅ **Correct:** PyMC default with PyTensor GPU backend

### **Branch 3: GPU Available, Unknown Backend** ⚠️ FIXED
```python
gpu_available=True, gpu_backend='unknown'
→ Line 743-745: NOW calls get_optimal_nuts_sampler()
→ sampling_kwargs['nuts_sampler'] = 'nutpie' or 'pymc'
```
✅ **Fixed:** Now uses nutpie/pymc fallback

### **Branch 4: CPU Only**
```python
gpu_available=False
→ Line 748-750: calls get_optimal_nuts_sampler()
→ sampling_kwargs['nuts_sampler'] = 'nutpie' or 'pymc'
```
✅ **Correct:** Uses nutpie if available, else PyMC

---

## Auto-Decision Path (GPU → CPU Fallback)

**Scenario:** GPU available but workload analysis decides CPU is better

```python
auto_decide=True, gpu_available=True
→ Line 696: get_optimal_sampling_params()
→ Line 704-710: Smart logic decides CPU, sets gpu_available=False
→ Line 717-724: Build base sampling_kwargs
→ Line 746-750: CPU branch, sets nuts_sampler='nutpie'/'pymc'
```
✅ **Correct:** Falls back to nutpie for CPU

---

## All pm.sample() Calls Inventory

| Location | Path | nuts_sampler Set? |
|----------|------|-------------------|
| `rule.py:1185` | Binary item (posthoc) | ✅ Via sampling_kwargs |
| `rule.py:1693` | Binary item (live) | ✅ Via sampling_kwargs |
| `rule.py:2569` | Binary group (live_single) | ✅ Via sampling_kwargs |
| `rule.py:2810` | Continuous group (live_single) | ✅ Via sampling_kwargs |
| `ordinal_model.py:374` | Ordinal (all paths) | ✅ Via compute_kwargs→default_kwargs |
| `convergence.py:548` | Convergence analysis | ✅ Via sampling_kwargs |

**Total:** 6 pm.sample() calls
**Covered:** 6/6 (100%)

---

## Verification Test Results

### **Test 1: Nutpie Not Installed**
```bash
$ python test_nutpie_integration.py
⚠️  nutpie NOT available
   Expected behavior: Will fallback to PyMC default sampler
✅ Valid sampler selected: pymc
```

### **Test 2: Nutpie Installed**
```bash
$ pip install nutpie
$ python test_nutpie_integration.py
✅ nutpie IS available (version: 0.16.4)
✅ Valid sampler selected: nutpie
```

### **Test 3: All Pathways Use nuts_sampler Parameter**
```bash
✅ PyMC version 5.26.1 supports 'nuts_sampler' parameter
✅ Binary pathway (rule.py:2569)
✅ Ordinal pathway (ordinal_model.py:374 via compute_kwargs)
✅ Continuous pathway (rule.py:2810)
```

---

## Issues Fixed

### **Issue 1: Unknown GPU Backend** ✅ FIXED
- **Before:** No nuts_sampler set, used PyMC default
- **After:** Calls get_optimal_nuts_sampler(), uses nutpie/pymc
- **Impact:** Rare edge case, but now consistent

### **Issue 2: Redundant Call in get_optimal_sampling_params()** ✅ DOCUMENTED
- **Behavior:** get_optimal_nuts_sampler() called twice in some paths
- **Impact:** Cosmetic only (both calls return same value)
- **Resolution:** Added clarifying comment

---

## Remaining Considerations

### **pytensor-gpu Path**
- **Current:** No nuts_sampler set (PyMC default + PyTensor GPU backend)
- **Why:** PyTensor handles GPU internally, no external sampler needed
- **Status:** ✅ Correct as-is

### **numpyro GPU Path**
- **Current:** nuts_sampler='numpyro'
- **Why:** JAX-based GPU acceleration
- **Status:** ✅ Correct

---

## Summary

✅ **All 6 pm.sample() calls** now use nutpie when available
✅ **All 4 inference pathways** (binary, ordinal, continuous, convergence) covered
✅ **All GPU/CPU branches** properly set nuts_sampler
✅ **Graceful fallback** to PyMC default if nutpie not available
✅ **Clear logging** shows which sampler is used
✅ **Test suite** verifies correct behavior

**Status:** Implementation is complete, consistent, and production-ready.
