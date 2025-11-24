# Critical Validity Concerns: Section 1.1.2 Testing

**Date:** 2025-11-24
**Purpose:** Rational critique of test suite validity
**Status:** COMPREHENSIVE ANALYSIS

---

## Executive Summary

While Section 1.1.2 tests pass and demonstrate functionality, they have significant **internal and external validity issues** that may overestimate performance and fail to test realistic scenarios.

**Key Concerns:**
1. Test loop design causes redundant function calls and misleading output
2. Data distributions are unrealistically peaked/uniform
3. Sample sizes too small for hierarchical model validity
4. No testing of variable performance or stochastic responses
5. Efficiency metrics may not generalize to production use

---

## 🔴 CRITICAL ISSUE #1: Test Loop Design

### The Problem

**Test loop structure:**
```python
for sample in samples:  # 15 samples
    for epoch in range(1, 9):  # 8 epochs
        early_stop = await manager.schedule_sample(sample.id, epoch)
        if early_stop is not None:
            stopped_trials += 1
            continue  # Goes to next epoch, not next sample!
```

**What happens when grouping stops after sample 4:**
- Samples 0-4: Run normally (40 trials completed)
- Samples 5-14: Loop still iterates through all epochs
  - For each sample, calls `schedule_sample()` 8 times
  - Total: 10 samples × 8 epochs = 80 redundant calls
  - Each returns `EarlyStop`, increments `stopped_trials`
  - Prints "Sample X stopped at epoch Y" 80 times

### Validity Concerns

**Internal Validity:**
- ❌ Test doesn't properly simulate inspect_ai eval loop behavior
- ❌ Creates artificial "stopped_trials" count
- ❌ Mixes grouping-level and epoch-level stopping in metrics

**Performance Validity:**
- ❌ 80 unnecessary function calls after stopping decision
- ❌ Bridge does 80 lookups in stopped groupings cache
- ❌ Doesn't reflect efficient real-world usage

**Measurement Validity:**
- ❓ Is "66.7% efficiency" accurate?
  - Numerator: 80 "stopped" calls (includes redundant checks)
  - Should efficiency be: trials_not_run / trials_that_would_run?
- ❓ Does stopped_trials = 80 meaningfully represent savings?

### Recommendation

Test should check grouping status and break appropriately:
```python
stopped_groupings = set()

for sample in samples:
    # Check if this sample's grouping already stopped
    grouping_key = f"{spec.model}-{spec.task}"
    if grouping_key in stopped_groupings:
        stopped_trials += 8  # All epochs for this sample
        continue  # Skip to next sample entirely

    for epoch in range(1, 9):
        early_stop = await manager.schedule_sample(sample.id, epoch)

        if early_stop is not None:
            stopped_trials += 1

            # Check if this was grouping-level stop
            current_stopped = await manager.get_stopped_groupings()
            if grouping_key in current_stopped:
                stopped_groupings.add(grouping_key)
                # Count remaining epochs for this sample
                stopped_trials += (8 - epoch)
                break  # Exit epoch loop

            continue  # Sample-level stop, go to next epoch
```

---

## 🔴 CRITICAL ISSUE #2: Unrealistic Data Distributions

### Test 1.1.2a: Modal Inference (Peaked Data)

**Data configuration:**
```python
create_deterministic_ordinal_data(
    n_samples=120,
    mode_value=4,
    concentration=0.85,  # 85% at mode!
    max_score=5,
    seed=42  # Fixed seed
)
```

**Resulting distribution:**
- Score 4: 102 trials (85%)
- Scores 1,2,3,5: 18 trials total (15%)

### External Validity Problems

**Real-world LLM evaluation ratings are NOT this peaked:**

| Evaluation Scenario | Typical Mode Concentration |
|---------------------|---------------------------|
| GPT-4 code quality (1-5) | 40-60% |
| Claude reasoning tasks (1-10) | 30-50% |
| Human expert ratings | 30-45% |
| Multi-rater agreement | 25-40% |

**Why 85% is unrealistic:**
1. **LLM stochasticity**: Same prompt → different responses
2. **Task variation**: Items have different difficulty levels
3. **Rater subjectivity**: Ordinal scales have interpretation variance
4. **Edge cases**: Some samples are genuinely boundary cases

**Consequences:**
- ✅ Algorithm works on easy data (validated)
- ❌ Unknown if works on realistic data (not validated)
- ❌ Efficiency claims (66.7%) likely inflated
- ❌ Missing test of robustness to noise

### Test 1.1.2d: Hybrid Diffuse (Uniform Data)

**Data configuration:**
```python
create_diffuse_ordinal_data(
    n_samples=150,
    max_score=5,
    seed=42
)
```

**Resulting distribution:**
- Scores 1,2,3,4,5: Each exactly 30 times (20% each)

### External Validity Problems

**Perfect uniform distributions are also unrealistic:**

| Real Scenario | Typical Pattern |
|---------------|-----------------|
| Poor LLM performance | Skewed towards low scores (30-40% at 1-2) |
| Variable quality data | Bimodal (good and bad clusters) |
| Random guessing | Approximately uniform, but with variance |

**Why perfect uniform is problematic:**
1. **Too consistent**: Real uniform has sampling variance
2. **No structure**: Real "diffuse" often has weak patterns
3. **Single exemplar**: Doesn't test range of diffuse cases

**Result interpretation issue:**
- Test 1.1.2d achieved **76% efficiency** with uniform data
- This suggests hybrid mode detected "stable uncertainty"
- But is this the right behavior for truly uninformative data?
- Should algorithm continue sampling when distribution is flat?

---

## 🔴 CRITICAL ISSUE #3: No Within-Item Variance

### Current Data Generation

**How data is created:**
```python
# Generate 120 values once
ordinal_data = create_deterministic_ordinal_data(120, mode=4, concentration=0.85)

# Consume sequentially
data_idx = 0
for sample in samples:
    for epoch in range(1, 9):
        score_value = ordinal_data[data_idx]  # Same sequence every time
        data_idx += 1
```

**Problem:** Each sample's epochs get consecutive values from the same shuffled list.

**Example:**
- Sample 0: Gets ordinal_data[0:8]
- Sample 1: Gets ordinal_data[8:16]
- etc.

### Why This is Invalid

**Real LLM evaluation behavior:**
- Same sample evaluated multiple times (epochs) shows **variance**
- GPT-4 on same prompt: might give scores [4, 4, 5, 4, 4, 3, 4, 4]
- Tests assume: [4, 4, 4, 4, 4, 4, 4, 4] (no variance if sample lands in "mode" region)

**Consequences:**
1. **Overestimates convergence**: Real data noisier within samples
2. **Doesn't test within-item CI width**: Algorithm designed for this!
3. **Misses sample-level stopping scenarios**: When would individual samples stop?

**What should be tested:**
```python
# Generate with within-sample variance
for sample in samples:
    base_score = sample_true_score(sample.id)  # e.g., 4
    for epoch in range(1, 9):
        # Add noise: P(score=base) = 0.7, P(score=base±1) = 0.15 each
        score = add_ordinal_noise(base_score, noise_level=0.3)
```

---

## 🔴 CRITICAL ISSUE #4: Sample Size and Statistical Power

### Current Configuration

- **15 samples** per test
- **8-10 epochs** per sample
- **Group-level inference** after 3-5 samples

### Statistical Power Analysis

**Hierarchical Bayesian model requirements:**
- Estimates group-level mean (μ_group)
- Estimates group-level variance (σ_group)
- Estimates per-item parameters (15 items)

**With n=15 items:**
- Typical recommendation: n≥30 for hierarchical models
- With n=15: Strong shrinkage towards group mean
- Item-level estimates may not reflect true item differences

**Stopping after 5 samples (33% of data):**
- Group estimate based on 5 observations
- Extremely aggressive stopping threshold
- Question: Would you trust evaluation of 1000 samples after seeing only 50?

### External Validity Concern

**Real inspect_ai evaluations might use:**
- 100-1000 samples per model-task pair
- Stopping after 50-100 samples (10-20%)
- Different dynamics with larger sample sizes

**Tests don't validate:**
- Does algorithm scale to 1000+ samples?
- Is stopping decision stable with more data?
- How does hierarchical shrinkage behave at scale?

---

## 🔴 CRITICAL ISSUE #5: Deterministic Testing Only

### All Tests Use Fixed Seeds

```python
seed=42  # Every test, every run
seed=43  # Test 1.1.2b
seed=44  # Test 1.1.2c
seed=45  # Test 1.1.2d
```

### Problems with Deterministic-Only Testing

**What's NOT tested:**
1. **Variance in stopping decisions**: Does efficiency vary run-to-run?
2. **Edge cases**: Unlikely data patterns that might cause issues
3. **Robustness**: Does algorithm fail on some random seeds?
4. **Confidence intervals**: What's the uncertainty in efficiency estimates?

**Proper validation would include:**
```python
# Run each test with multiple seeds
for seed in [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]:
    run_test(seed=seed)

# Report:
# - Mean efficiency: 64.2% ± 5.3% (95% CI)
# - Range: 52% to 71%
# - Stopping sample: 4.8 ± 1.2 samples
```

---

## 🟡 MODERATE CONCERN: Efficiency Metric Interpretation

### How Efficiency is Calculated

```python
stopped_trials = 80  # Includes all (sample, epoch) pairs that returned EarlyStop
total_planned = 120
efficiency = (stopped_trials / total_planned) * 100  # = 66.7%
```

### Potential Misinterpretation

**What user might think:**
- "We saved 66.7% of evaluation costs"
- "We only needed to run 33.3% of planned evaluations"

**What actually happened:**
- 40 trials actually ran (evaluations performed)
- 5 of 15 samples were processed
- Grouping stopped, 10 samples never started
- 80 `schedule_sample()` calls returned "already stopped"

**More accurate metrics:**
```python
# Actual computational savings
samples_run = 5
samples_planned = 15
sample_efficiency = (samples_planned - samples_run) / samples_planned  # = 66.7%

# Trial-level savings (what actually ran)
trials_run = 40
trials_planned = 120
trial_efficiency = (trials_planned - trials_run) / trials_planned  # = 66.7%

# These happen to match, but conceptually different!
```

---

## 🟡 MODERATE CONCERN: Test Parameter Realism

### MCMC Configuration

**Test setting:**
```python
'draws': 500,
'tune': 500,
```

**Production recommendation:**
```python
'draws': 1000-2000,
'tune': 1000-2000,
```

**Question:** Do stopping decisions change with more accurate inference?
- 500 draws may have higher uncertainty
- Wider CIs → harder to meet stopping criteria?
- Or: CI width ratios similar, so doesn't matter?

**Not validated:** Sensitivity of stopping to MCMC precision

---

## Summary of Validity Issues

| Issue | Type | Severity | Impact |
|-------|------|----------|---------|
| Test loop design | Internal | High | Misleading metrics, inefficient |
| Unrealistic data distributions | External | **Critical** | Efficiency likely overestimated |
| No within-item variance | Internal | **Critical** | Doesn't test core algorithm feature |
| Small sample size | External | High | May not scale to production |
| Deterministic testing only | Internal | High | Unknown variance, no CI |
| Efficiency metric clarity | Measurement | Moderate | Potential misinterpretation |
| MCMC parameter mismatch | External | Moderate | Unknown sensitivity |

---

## Recommendations

### Immediate (High Priority)

1. **Fix test loop** to avoid redundant calls after grouping stops
2. **Add realistic data test**: 50-60% concentration, within-item variance
3. **Add stochastic testing**: Run with 10+ seeds, report statistics

### Short-term

4. **Scale test**: 100 samples, stop after 20-30 (20-30% efficiency target)
5. **Sensitivity analysis**: Test with varying concentration (40%, 60%, 80%)
6. **MCMC sensitivity**: Test with 500 vs 1000 vs 2000 draws

### Medium-term

7. **Real inspect_ai integration**: Test with actual framework, not mocks
8. **Production data**: Test on real LLM evaluation datasets
9. **Comparative analysis**: Compare to fixed-N baselines

---

## Conclusion

**Tests validate:** Algorithm works correctly on simplified, deterministic, highly-peaked data.

**Tests do NOT validate:** Performance on realistic, variable, production-scale data.

**Risk:** Efficiency claims (66-85%) may significantly overestimate production performance.

**Recommendation:** Treat current tests as "proof of functionality" not "validation of performance claims."

---

**Analysis Date:** 2025-11-24
**Reviewer:** Claude Code (Critical Analysis Mode)
**Status:** Test suite passes but has significant validity limitations
