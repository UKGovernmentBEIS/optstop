# Validation Analyses: Notes and Caveats

This document records outcomes and issues identified during critical evaluation of the four validation analyses. These should be addressed in the paper text where relevant.

---

## 1. Consistency Analysis (`consistency/`)

**Status**: Complete (9/9 cells, 15 reps each, 1774 min total).

**Method**: Each cell's full shadow dataset was replayed 15 times through `optimal_stopping_posthoc` with different shuffled item orderings (seeds 1000-1014). This tests whether the stopping decision and posterior estimate are robust to item presentation order. `low_performance_threshold=0.01` (corrected from initial 0.1 which caused conservatism to erroneously activate for low_binary).

**Data files**: `posthoc_consistency.json` (all 9 cells, re-run with corrected threshold).

### Results summary

| Cell | Stop rate | Efficiency (mean±std) | Theta mean±std | Full-data ref | Abs diff | CI coverage | CI width (mean) | Jaccard (mean) |
|------|-----------|----------------------|----------------|---------------|----------|-------------|-----------------|----------------|
| low_binary | 100% | 80.8 ± 0.9% | 0.0669 ± 0.0026 | 0.0765 | 0.0096 | 100% | 0.049 | 0.900 |
| mid_binary | 100% | 57.6 ± 1.5% | 0.4595 ± 0.0029 | 0.4763 | 0.0168 | 100% | 0.050 | 0.874 |
| high_binary | 100% | 82.4 ± 0.4% | 0.8151 ± 0.0027 | 0.8250 | 0.0099 | 100% | 0.049 | 0.879 |
| low_continuous | 100% | 97.9 ± 0.2% | 0.0991 ± 0.0037 | 0.1141 | 0.0150 | 100% | 0.046 | 0.827 |
| mid_continuous | 100% | 93.0 ± 0.4% | 0.4151 ± 0.0046 | 0.4231 | 0.0080 | 100% | 0.049 | 0.803 |
| high_continuous | 100% | 95.1 ± 0.3% | 0.6978 ± 0.0057 | 0.6930 | 0.0054 | 100% | 0.047 | 0.766 |
| low_ordinal | 100% | 97.3 ± 0.2% | 0.1000 ± 0.0000 | 0.1105 | 0.0104 | **0%** | 0.014 | 0.954 |
| mid_ordinal | 100% | 96.8 ± 1.2% | 0.4067 ± 0.0170 | 0.4218 | 0.0227 | 100% | 0.187 | 0.876 |
| high_ordinal | 100% | 95.9 ± 0.5% | 0.7033 ± 0.0125 | 0.6924 | 0.0109 | **13%** | 0.042 | 0.626 |

### Key findings

**1. 100% stop rate across all 9 cells.** With the corrected `low_performance_threshold=0.01`, all cells (including low_binary) achieve 100% stop rate across all 15 orderings. This is a major improvement from the initial run (which used threshold=0.1, causing low_binary to stop in only 27% of orderings due to erroneous conservatism activation).

**2. Efficiency**: Varies by inference pathway. Continuous cells are most efficient (93-98%), ordinal similarly (96-97%). Binary efficiency varies with performance level: low_binary 81%, high_binary 82%, mid_binary 58% (mid-range performance has widest posteriors).

**3. Theta estimate stability**: All cells show low theta std across orderings (0.0000-0.0170), confirming robustness to item presentation order. Binary: std ≤ 0.003. Continuous: std ≤ 0.006. Ordinal is more variable due to quantization (see Issue 1b).

**4. CI coverage of full-data reference**: Binary and continuous cells achieve 100% coverage (6/6 cells). Ordinal coverage appears problematic (low_ordinal 0%, high_ordinal 13%) but this is an estimand mismatch: ordinal theta brackets the modal category, not the raw mean used as reference. See Issue 1b for full analysis.

**5. Pairwise CI overlap (Jaccard)**: Binary 0.87-0.90 (high consistency). Continuous 0.77-0.83 (good). Ordinal is mixed: low_ordinal 0.95 (near-identical narrow CIs), mid_ordinal 0.88, high_ordinal 0.63 (most variable).

### Issue 1a: low_performance_threshold correction

The initial run used `low_performance_threshold=0.1` (inherited from the production config), which caused conservatism to activate for low_binary (true perf ~7.7% < 10%). This inflated effective CI width by 10x, causing 73% of orderings to exhaust all data without stopping (27% stop rate, 4% efficiency).

The corrected threshold of 0.01 deactivates conservatism for low_binary (7.7% > 1%), restoring normal stopping: 100% stop rate, 80.8% efficiency. This matches the intended behaviour — conservatism should only activate for truly negligible performance (< 1%), not for merely low performance.

**Paper note**: Ensure the consistency analysis uses `low_performance_threshold=0.01` in all reported results. The 0.1 threshold is for the conservatism sensitivity analysis specifically (where it is intentionally set high to exercise the mechanism).

### Issue 1b: Ordinal CI "coverage failure" — estimand mismatch, not estimation error

#### What ordinal theta actually represents

The ordinal inference pathway estimates the **modal category** (most common response), NOT the mean score. Tracing through the code:

1. A Dirichlet-Multinomial hierarchical model estimates group-level category concentration parameters `alpha_group`
2. `modal_group = argmax(alpha_group)` — the posterior distribution over which category is the mode
3. HDI is computed on `modal_group` posterior samples
4. CI bounds are scaled: `theta = modal_category / ordinal_max_score`

For a 0-10 scale: theta=0.1 means "the modal response is category 1", theta=0.7 means "the modal response is category 7". The CI brackets which category is most common, not the mean performance level.

Source: `ordinal_utils.py:_ordinal_ci_hierarchical_modal()` (lines 578-751) and `ordinal_model.py:_ordinal_hybrid_stopping_criterion()` (lines 746-865).

#### Why coverage "fails"

The consistency analysis checks coverage of the **raw sample mean** (`mean(scores) / ordinal_max_score`) against the **modal category CI**. These are fundamentally different quantities:

| Cell | Modal category (theta) | Raw mean / max_score | Same? |
|------|----------------------|---------------------|-------|
| low_ordinal | 1/10 = 0.100 | 0.1105 | No — mode < mean (right-skewed scores) |
| mid_ordinal | 4/10 = 0.400 | 0.4218 | No — mode < mean (slight right skew) |
| high_ordinal | 7/10 = 0.700 | 0.6924 | No — mode > mean (slight left skew) |

The coverage failures are expected and correct:
- **(a) low_ordinal coverage = 0%**: CI=[0.093, 0.107] correctly brackets the modal category (1). The raw mean (0.1105) lies outside because the distribution is right-skewed (mostly 1s, some 2s and 3s). Mean > mode for right-skewed distributions. The CI is doing its job — it was never designed to bracket the mean.
- **(b) high_ordinal coverage = 13%**: CI≈[0.695, 0.706] correctly brackets modal category 7. Raw mean (0.6924) lies outside because distribution is left-skewed (mostly 7s, some 6s and 5s). Mean < mode for left-skewed distributions.
- **(c) mid_ordinal coverage = 100% (coincidental)**: CI=[0.3, 0.5] is very wide because the mode is uncertain (scores spread across categories 3-5). The raw mean (0.4218) happens to fall within this wide interval.

#### The real issue: cross-pathway reporting inconsistency

The ordinal inference is **valid for what it estimates** (the modal category). The problem is that the package reports `theta_ci_low` and `theta_ci_high` for all inference pathways using the same field names, but they represent different quantities:

| Pathway | theta represents | CI brackets |
|---------|-----------------|-------------|
| Binary | mean(item-level success rates) | HDI of posterior mean performance |
| Continuous | mean(normalized item scores) | HDI of posterior mean performance |
| **Ordinal** | **modal category / max_score** | **HDI of which category is most common** |

A user comparing theta across pathways — or checking ordinal CI coverage against the raw mean — will get misleading results because the estimands differ. This is a documentation and API design issue, not an estimation error.

#### Quantization is inherent, not a bug

Since the mode is always an integer category, theta can only take values 0, 0.1, 0.2, ..., 1.0 (for a 0-10 scale). CI width reflects uncertainty about which category is modal:
- When the mode is clear (low_ordinal: almost all scores are 1): narrow CI (0.014)
- When the mode is uncertain (mid_ordinal: scores spread across 3-5): wide CI (0.187)
- This produces CI widths with very different properties from the mean-based CIs in binary/continuous

This also causes:
- **Theta std = 0** for low_ordinal (all reps land on the same modal category)
- **Theta std inflated by discrete jumps** for mid/high_ordinal (reps occasionally jump to adjacent categories)
- **CI width bimodality**: narrow when mode is certain, wide when uncertain — no middle ground

The Jaccard metric is more reliable for assessing ordinal consistency than theta std or CI coverage, because it directly measures the overlap of CIs without requiring a reference value.

#### Actionable solutions

**1. Documentation (low effort, high value):** The package should clearly document that ordinal theta represents the modal category scaled to [0,1], NOT the mean performance. The CI brackets the mode. This is the most important fix — the current API returns identical field names (`theta_ci_low`, `theta_ci_high`) for different estimands across pathways. Users comparing across pathways or checking coverage against raw means will be misled without this documentation.

**2. Add supplementary mean-based ordinal CI (medium effort, high value):** Compute a mean-based CI alongside the modal CI for ordinal data — e.g., `mean(scores) / max_score` with a bootstrap or normal-approximation CI. This would:
- Enable apples-to-apples comparison across binary/continuous/ordinal pathways
- Give users the quantity they most often want (average performance level)
- Support meaningful coverage checks against the raw mean reference
The stopping decision should continue to use the modal CI (which answers a natural stopping question: "have we pinned down the typical response?"). The mean CI would be for reporting and comparison purposes.

**3. Do NOT replace the modal stopping criterion:** The modal inference is well-designed for ordinal stopping — it converges when the typical response is determined. A mean-based criterion would conflate distinct distributions (e.g., bimodal at categories 2 and 8 both have mean=5) and might stop when the distribution shape is still uncertain.

**Paper note**: Report that ordinal theta represents the modal category (not the mean), and that coverage checks against the raw mean are a category error — the CI correctly brackets the mode. Recommend the Jaccard overlap as the primary ordinal consistency metric. Note the cross-pathway estimand difference as a known limitation, and recommend binary or continuous scoring where mean performance is the quantity of interest.

### Issue 1d: Systematic negative bias — DECOMPOSED

7 of 9 cells show `theta_mean < theta_full_data` (the full-data raw mean). The bias is consistent across all 15 shuffles within each cell, confirming it is a model/stopping property, not a shuffle artifact.

#### Bias decomposition analysis

To determine the source, we decomposed `total_bias = subset_selection_bias + model_bias` by replaying each rep's shuffle to reconstruct the stopped sample and computing its raw empirical mean (`raw_mean_stopped`). Script: `analyse_bias_decomposition.py`. Results: `consistency/bias_decomposition.json`.

| Cell | Total Bias | Subset Sel. Bias | Model Bias | Primary Source |
|------|-----------|-----------------|------------|----------------|
| low_binary | -0.0096 | **-0.0144** | +0.0048 | **Subset selection** (150%) |
| mid_binary | -0.0168 | **-0.0160** | -0.0008 | **Subset selection** (95%) |
| high_binary | -0.0099 | -0.0033 | **-0.0066** | **Model bias** (67%) |
| low_continuous | -0.0150 | +0.0002 | **-0.0153** | **Model bias** (102%) |
| mid_continuous | -0.0079 | -0.0028 | **-0.0051** | **Model bias** (64%) |
| high_continuous | +0.0048 | -0.0022 | **+0.0070** | **Model bias** (146%) |
| low_ordinal | -0.0104 | -0.0006 | **-0.0098** | **Model bias** (94%) |
| mid_ordinal | -0.0152 | +0.0088 | **-0.0240** | **Model bias** (158%) |
| high_ordinal | +0.0109 | +0.0001 | **+0.0108** | **Model bias** (99%) |

Where:
- `subset_selection_bias = raw_mean_stopped - raw_mean_full` (does the stopped sample differ from full data?)
- `model_bias = theta_estimate - raw_mean_stopped` (does the Bayesian model distort the estimate?)
- Percentages can exceed 100% when components have opposite signs (partial cancellation)

#### Key findings from decomposition

**(a) Continuous and ordinal: Stopped samples ARE representative.** For all 6 continuous/ordinal cells, subset selection bias is < 0.003 (negligible). Despite seeing only 20-70% of items, the random subset's raw mean closely matches the full-data mean. This is the expected result of random shuffling and is likely generalizable across datasets. It is a positive finding about the epoch-interleaved stopping rule.

**(b) Binary cells: "Subset selection" is actually temporal sampling noise.** Low_binary and mid_binary show subset_selection_bias of -0.014 to -0.016. However, inspection of per-epoch means reveals this reflects random epoch-level score fluctuations in THIS dataset, not systematic item selection effects:

```
low_binary per-epoch means: 0.055, 0.070, 0.065, 0.100, 0.105, 0.110, 0.075, 0.065, 0.050, 0.070
                            ^^^^^^^^^^^^^^^^^^^^^^^^ early (stopped subset includes these)
```

Epochs 4-6 happen to score ~0.105 vs epochs 1-2 at ~0.063. With 200 Bernoulli(0.077) items per epoch, the epoch SE is ~0.019, so this variation is within normal sampling noise. The stopped subset (early epochs) misses the coincidentally high-scoring later epochs. A different data seed could produce the opposite pattern. This finding is **dataset-specific and not generalizable**.

**(c) "Model bias" component conflates multiple effects.** The model_bias = theta_estimate - raw_mean_stopped is NOT purely "HDI midpoint bias" as initially characterized. It conflates:
  1. HDI midpoint vs posterior mean (estimator choice)
  2. Prior/shrinkage effects (hierarchical model pulls estimates toward group mean)
  3. Bayesian posterior mean vs frequentist sample mean (different estimands)

Without access to the posterior samples (which optstop does not output), these cannot be separated. The performance-level-dependent direction (negative at low perf, positive at high perf) is consistent with HDI asymmetry on bounded parameters, but could also reflect prior influence. Both explanations are plausible; neither is confirmed.

**(d) The full-data reference is itself noisy.** The "bias" is measured against `raw_mean_full` (all 2000 trials), which is one noisy realization of the true population parameter θ. For low_binary with true θ unknown and observed 0.0765, the sampling SE is ~0.006. So "bias" of -0.010 includes uncertainty in the reference, and the true deviation from θ could be smaller or larger.

**Impact**: The total bias (~0.01) is small relative to CI width (~0.05) and does not affect coverage for binary/continuous cells. The decomposition's clearest finding is that **continuous/ordinal stopped samples are representative** (subset selection ≈ 0, generalizable). For binary cells, the subset selection component is dataset-specific noise. The model bias component exists but its attribution to specific causes is uncertain.

**Paper note**: Report that bias decomposition confirms stopped samples are representative for continuous/ordinal cells (subset selection bias < 0.003). For binary cells, note that the deviation between stopped and full-data means reflects epoch-level sampling noise, not systematic selection effects. Total bias magnitude (~0.01) is within the delta_cap precision guarantee. Avoid attributing the model component to a single cause (HDI midpoint) without posterior sample evidence.

### Issue 1e: CI widths are delta_cap-bounded — coverage is a design property

Binary cell CI widths cluster tightly at [0.048, 0.050] — the system stops exactly when `theta_width ≤ delta_cap (0.05)`, so CIs are never narrower than ~0.049 at the time of stopping. The 100% coverage (90/90 binary+continuous comparisons) is partly because this stopping-threshold width absorbs the estimation bias (~0.01).

For correctly calibrated 97% CIs, the probability of 0 misses in 90 independent trials is 0.064 (6.4%). The observed 0/90 is consistent with CIs being *over-wide* relative to the true posterior uncertainty — a direct consequence of the stopping criterion guaranteeing width ≥ ~delta_cap.

This is a design trade-off, not a flaw: the system prioritises precision control (stop when CI is narrow enough) over CI tightness. But it means 100% coverage should not be cited as evidence of perfect Bayesian calibration. It is evidence that `delta_cap=0.05` is wide enough to absorb the estimation bias in these cells.

**Paper note**: State that CI coverage reflects the delta_cap precision guarantee rather than posterior calibration. Coverage would decrease with tighter delta_cap, particularly for cells with larger bias.

### Issue 1f: Continuous/ordinal item subset selection

Continuous cells stop within epoch 1 (reps/item=1.0), seeing a RANDOM SUBSET of items:
- low_continuous: 41 ± 3 of 200 items (20%)
- mid_continuous: 140 ± 7 of 200 items (70%)
- high_continuous: 97 ± 6 of 200 items (49%)

Ordinal cells similarly:
- low_ordinal: 55 ± 5 items (28%)
- mid_ordinal: 65 ± 23 items (33%)
- high_ordinal: 81 ± 11 items (41%)

Binary cells see ALL items (200/200) but varying epoch depth (1.7-4.2 reps/item).

The reference theta is from all 200 items × all epochs. For cells seeing only a subset, the theta estimate reflects that subset, not the full population.

**Resolution via Issue 1d decomposition**: The bias decomposition analysis (Issue 1d) shows that subset selection bias is **negligible** for continuous/ordinal cells (< 0.003 in all 6 cells). Despite seeing only 20-70% of items, the random subset's raw mean closely matches the full-data mean. The observed bias in these cells is almost entirely in the model component (the difference between the Bayesian estimate and the raw stopped-sample mean), not the item selection.

**Impact**: This is a positive finding — the epoch-interleaved stopping rule produces representative samples even when evaluating a minority of items. The higher theta variability across shuffles (std=0.004-0.006 for continuous vs 0.003 for binary) reflects MCMC posterior variability across different data subsets, not biased subset selection.

**Paper note**: Note that high-efficiency cells (>93%) achieve their savings by evaluating a minority of items. Bias decomposition confirms the group-level estimate from 40-150 items generalises well to the full 200 (subset selection bias ≤ 0.003 for continuous/ordinal).

### Issue 1g: High_ordinal degenerate CIs

3 of 15 high_ordinal reps produce CI widths of 0.10-0.20 (vs the typical 0.01 for other ordinal reps at this performance level). This indicates the ordinal posterior occasionally fails to concentrate, likely due to the ordered logistic model landing between category boundaries. These outlier reps drive the high theta_std (0.0125) and low Jaccard (0.626).

**Paper note**: Note that ordinal inference at mid-to-high performance occasionally produces poorly concentrated posteriors due to the discrete category structure.

---

## 2. Conservatism Sensitivity Analysis (`conservatism/`)

**Status**: Complete (16/16 conditions, 865 min total).

**Method**: Full `optimal_stopping_posthoc` API (PyMC MCMC hierarchical model) with synthetic binary data. Tests how the conservatism parameter `c` affects stopping behaviour at different performance levels relative to the `low_performance_threshold=0.1`. When `p_hat < threshold`, conservatism is active: prior alpha is boosted by `c`, effective CI width is inflated by `c`, and slope threshold is tightened by `1/c`.

Design:
- 200 items x 10 epochs = 2000 trials per condition
- Performance levels: 1%, 5%, 10%, 15% (spanning the 10% activation threshold)
- Conservatism values: c = 1, 5, 10, 50
- 1000 MCMC draws / 1000 tune, 16 conditions total

### Results summary

| Perf | c | Conserv. active | Stopped | Trials used | Efficiency | Theta | True perf | CI covers true | CI width |
|------|---|-----------------|---------|-------------|------------|-------|-----------|----------------|----------|
| 1% | 1 | Yes | Yes | 120 | 94.0% | 0.027 | 0.0065 | Yes | 0.049 |
| 1% | 5 | Yes | Yes | 910 | 54.5% | 0.008 | 0.0065 | Yes | 0.012 |
| 1% | 10 | Yes | Yes | 1920 | 4.0% | 0.007 | 0.0065 | Yes | 0.008 |
| 1% | 50 | Yes | Yes | 1590 | 20.5% | 0.008 | 0.0065 | Yes | 0.009 |
| 5% | 1 | Yes | Yes | 440 | 78.0% | 0.064 | 0.0525 | Yes | 0.049 |
| 5% | 5 | Yes | **No** | 2000 | 0.0% | 0.054 | 0.0525 | Yes | 0.021 |
| 5% | 10 | Yes | Yes | 1590 | 20.5% | 0.056 | 0.0525 | Yes | 0.026 |
| 5% | 50 | Yes | **No** | 2000 | 0.0% | 0.054 | 0.0525 | Yes | 0.022 |
| 10% | 1 | No | Yes | 700 | 65.0% | 0.112 | 0.108 | Yes | 0.049 |
| 10% | 5 | No | Yes | 700 | 65.0% | 0.113 | 0.108 | Yes | 0.049 |
| 10% | 10 | No | Yes | 710 | 64.5% | 0.110 | 0.108 | Yes | 0.050 |
| 10% | 50 | No | Yes | 720 | 64.0% | 0.111 | 0.108 | Yes | 0.047 |
| 15% | 1 | No | Yes | 969 | 51.6% | 0.167 | 0.1615 | Yes | 0.050 |
| 15% | 5 | No | Yes | 990 | 50.5% | 0.167 | 0.1615 | Yes | 0.049 |
| 15% | 10 | No | Yes | 940 | 53.0% | 0.166 | 0.1615 | Yes | 0.053 |
| 15% | 50 | No | Yes | 980 | 51.0% | 0.166 | 0.1615 | Yes | 0.050 |

### Key findings

**1. Clean activation threshold.** Conservatism activates only when performance is below the 10% threshold. At perf=10% and 15%, the conservatism parameter has virtually no effect — all four `c` values produce near-identical results (700-720 trials at 10%, 940-990 trials at 15%). This confirms the threshold mechanism works as a clean binary gate.

**2. Below threshold, conservatism increases data usage.** At perf=1%, increasing `c` from 1 to 10 increases trials used from 120 to 1920 (94% to 4% efficiency). At perf=5%, `c=5` and `c=50` exhaust all 2000 trials without stopping. Higher conservatism demands more evidence before stopping, as intended.

**3. Low conservatism risks premature stopping.** At perf=1% with c=1, the system stops after only 120 trials with theta=0.027. Higher `c` values (5, 10, 50) produce theta estimates of 0.007-0.008, much closer to the true performance. However, see Issue 2c below regarding the interpretation of "theta" as a CI midpoint.

**4. 100% CI coverage across all conditions.** All 16 conditions produce 97% HDIs that contain the true performance level. However, see Issue 2d below regarding what "true" means in this context.

**5. Non-monotonic efficiency at high conservatism.** At perf=1%, c=50 uses fewer trials (1590) than c=10 (1920). At perf=5%, c=10 stops (1590 trials) while c=5 and c=50 exhaust all data. See Issue 2b below for mechanistic analysis.

### Issue 2a: c=1 premature stopping at very low performance

At perf=1% with c=1, `effective_width = theta_width * 1 = theta_width`, so the CI convergence check is unmodified: `theta_width < delta_cap (0.05)`. The system stops at 120 trials with CI width=0.049, just under the threshold. With c=10, the check becomes `theta_width * 10 < 0.05`, i.e., `theta_width < 0.005`. This is a 10x stricter requirement, which is why c=10 needs 1920 trials.

**Paper note**: The default c=10 provides a strong guard against premature stopping at low performance (uses 96% of data at 1% perf) while having no effect above the threshold.

### Issue 2b: Non-monotonic behaviour — confirmed by multi-seed analysis

#### Single-seed mechanistic explanation

The conservatism parameter affects two mechanisms at the group level:
1. **CI width inflation**: `effective_width = theta_width * c` → higher c needs narrower CI to stop via convergence
2. **Slope threshold tightening**: `slope_threshold = CI_delta / c` (with an additional `/2` gate for low-perf conditions) → higher c needs flatter slope to stop via stabilisation

(Note: prior alpha boost is item-level only in `_beta_ci_adaptive`, not applied at the group level.)

The non-monotonicity arises because **at moderate-to-high c, the CI-width pathway becomes unreachable**, and all stopping relies on the slope pathway, which is stochastically triggered. See Issue 2h for the full mechanistic analysis.

#### Multi-seed confirmation (Part A: 5 data seeds × 4 c values × 2 perf levels)

| Perf | c=1 (mean±sd) | c=5 (mean±sd) | c=10 (mean±sd) | c=50 (mean±sd) | Mean monotonic? |
|------|--------------|--------------|---------------|---------------|-----------------|
| 1% | 140 ± 49 | 1238 ± 426 | 1170 ± 455 | 1698 ± 397 | **NO** (c=5 > c=10) |
| 5% | 374 ± 75 | 1536 ± 262 | 1666 ± 277 | 2000 ± 0 | YES (but c=50 saturates ceiling) |

Per-seed monotonicity: 2/5 seeds monotonic at perf=1%, 4/5 at perf=5%.

**The non-monotonicity at perf=1% is systematic, not a single-seed artefact.** The mean across 5 seeds shows c=10 (1170 trials) using *fewer* trials than c=5 (1238 trials). At perf=5%, mean monotonicity holds only because c=50 saturates the 2000-trial ceiling in all 5 seeds.

#### MCMC variance isolation (Part B: seed=42, 5 MCMC reps per condition)

| Condition | Mean ± SD | CV | Range |
|-----------|----------|-----|-------|
| 1%, c=1 | 136 ± 14 | 10% | 120–150 |
| 1%, c=5 | 918 ± 8 | <1% | 910–930 |
| 1%, c=10 | 991 ± 89 | 9% | 919–1100 |
| 1%, c=50 | 1068 ± 142 | 13% | 910–1290 |
| 5%, c=1 | 430 ± 9 | 2% | 420–440 |
| 5%, c=5 | 1854 ± 179 | 10% | 1620–2000 |
| 5%, c=10 | 1924 ± 152 | 8% | 1620–2000 |
| 5%, c=50 | 2000 ± 0 | 0% | 2000–2000 |

**MCMC variance increases with conservatism**: c=5 at perf=1% has <1% CV (CI-width pathway, reproducible) while c=50 has 13% CV (slope pathway, stochastic). The same data with different MCMC draws can give 910 vs 1290 trials at c=50 — a 42% swing.

#### Variance decomposition

| Condition | Total SD (Part A) | MCMC SD (Part B) | Data SD (residual) | % MCMC |
|-----------|------------------|-----------------|-------------------|--------|
| 1%, c=1 | 49 | 14 | 47 | 8% |
| 1%, c=5 | 426 | 7 | 426 | 0% |
| 1%, c=10 | 455 | 89 | 447 | 4% |
| 1%, c=50 | 397 | 142 | 371 | 13% |
| 5%, c=1 | 75 | 9 | 74 | 1% |
| 5%, c=5 | 262 | 179 | 191 | 47% |
| 5%, c=10 | 277 | 152 | 232 | 30% |
| 5%, c=50 | 0 | 0 | 0 | 0% |

At perf=1%, most variance is from data differences (dataset-specific), not MCMC noise. At perf=5% with high c, MCMC noise accounts for 30–47% of total variance — the procedure is near the data ceiling and MCMC jitter determines whether it stops or exhausts data.

**Paper note**: The non-monotonicity is confirmed across multiple seeds and is not a single-realisation artefact. The conservatism parameter works as a coarse dial (c=1 vs c≥5 is reliably different), not a fine-grained control (c=5 vs c=10 vs c=50 are not reliably ordered). See Issue 2h for the mechanistic explanation and recommended parameter guidance.

### Issue 2c: Theta estimate is CI midpoint, not posterior mean (METHODOLOGICAL)

The script reports `theta_estimate = (theta_ci_low + theta_ci_high) / 2`, i.e., the midpoint of the 97% HDI. The optstop package does not output the posterior mean directly. For symmetric posteriors this is a reasonable proxy, but for very low performance the posterior is right-skewed (concentrated near 0 with a long right tail). The HDI midpoint systematically overestimates the posterior mean in these cases.

At perf=1% c=1: CI=[0.003, 0.052], midpoint=0.027. The posterior mean is likely much lower (closer to the actual data mean of 0.0065). The "4x overestimate" characterisation refers to the CI midpoint, not the posterior mean. The same applies (less severely) to c=5,10,50 at 1%.

**Impact**: The theta values in the results table above are CI midpoints, not posterior means. For perf≥5%, posterior skew is minor and the distinction is negligible. For perf=1%, the reported theta=0.027 (c=1) is inflated by HDI asymmetry and should not be interpreted as the model's point estimate of performance.

**Paper note**: When reporting theta estimates from this analysis, use CI bounds rather than midpoints. Alternatively, note that the "theta" column represents the HDI midpoint, which overestimates at very low performance. The CI coverage (which uses bounds, not the midpoint) is not affected.

### Issue 2d: CI coverage checks target performance, not realized performance

The script checks `ci_covers_true = theta_ci_low <= perf_level <= theta_ci_high`, where `perf_level` is the intended data-generating parameter (e.g. 0.01), not the realized sample performance (`actual_perf`, e.g. 0.0065). With 200 binary items at 1% expected performance, the realized performance varies substantially across seeds (expected ~2 successes per epoch, actual ~1.3).

In this run all CIs are wide enough that coverage holds for both `perf_level` and `actual_perf`. But single-seed coverage is not a meaningful calibration check — proper coverage requires many replications.

**Impact**: The "100% CI coverage" finding is reassuring but should be described carefully. It demonstrates that CIs are not obviously miscalibrated, but 16 conditions (each with one realization) cannot establish frequentist coverage properties.

### Issue 2e: Single realization — ADDRESSED by multi-seed replication

The original single-seed analysis (seed=42) has been supplemented by a multi-seed replication:
- **Part A**: 5 data seeds (42–46) × 4 c values × 2 perf levels = 40 conditions
- **Part B**: 5 MCMC replications of seed=42 × 4 c values × 2 perf levels = 40 conditions

Key findings that are confirmed as robust across seeds:
- Clean activation threshold (conservatism inactive above `low_performance_threshold`)
- Large, reliable step from c=1 to c≥5
- Non-monotonic relationship between c and trials used at perf=1% (Issue 2b)

Key findings that are confirmed as seed-specific:
- Exact trial counts vary substantially across seeds (SD = 50–450 depending on condition)
- Whether a specific c value hits the data ceiling varies by seed

The multi-seed replication transforms Issues 2b and 2d from "single-seed observations" into quantified population-level findings with uncertainty estimates. See Issue 2b for the full results.

### Issue 2f: Dynamic threshold crossing at perf=10%

The script labels `conservatism_active = perf_level < LOW_PERF_THRESHOLD`, which is the script's prediction based on the target performance, not a confirmed report from the optstop engine. At perf=10% (actual_perf=0.108), the conservatism check inside optstop evaluates `current_perf_estimate < 0.1` at each reanalysis interval. During early epochs when the running estimate starts near 0 and builds up, conservatism may be transiently active before the estimate crosses 0.1.

**Impact**: For perf=10%, the label "conservatism inactive" may be inaccurate for early epochs. The near-identical results across c values at perf=10% suggest the transient activation has negligible effect, but this nuance should be noted. For perf=1% and 5%, the running estimate stays below 0.1 throughout, so the labels are correct. For perf=15%, the estimate crosses 0.1 quickly, making transient activation brief.

### Issue 2g: Stop mechanism not identified in output

The `mean_fin_CI_slope` and `mean_fin_slope_slope` fields are null for all conditions. This means we cannot directly confirm from the output whether each condition stopped via CI convergence or stabilisation. We infer the stop mechanism by comparing final CI width against `delta_cap / c`:
- If `theta_ci_width < delta_cap / c`: likely CI convergence
- If `theta_ci_width >= delta_cap / c`: must be stabilisation (or exhausted data)

At perf=1% c=1: width=0.049 < 0.05 → CI convergence. At perf=1% c≥5: all widths exceed `delta_cap/c` → stabilisation. At perf=10% and 15%: widths ~0.05, and with c inactive (effective c=1), delta_cap=0.05 → CI convergence.

**Paper note**: The stop mechanism inference is reliable for most conditions but should be described as inferred rather than directly reported.

### Issue 2h: Conservatism mechanism analysis and parameter guidance

#### Context: `conservatism` and `low_performance_threshold` are coupled parameters

The conservatism mechanism only activates when the running performance estimate falls below `low_performance_threshold`. These two parameters must be considered together:

- **Package defaults**: `conservatism=10`, `low_performance_threshold=0.001`. With these defaults, conservatism activates only for extremely low performance (< 0.1%), where c=10 is well within the reliable operating range (needs ~1,070 trials at p=0.001). **The defaults are a coherent pair.**
- **This validation** uses `low_performance_threshold=0.1`, which activates conservatism at much higher performance levels (p=1%, p=5%). At these levels, c=10 exceeds the reliable range, producing the non-monotonicity documented in Issue 2b. This is not a defect in the default configuration — it is a consequence of raising the threshold 100× without adjusting c.
- **Users who raise `low_performance_threshold`** (e.g., to 0.01 or 0.1 to protect against premature stopping at low-but-not-negligible performance) must also consider whether their `conservatism` value is appropriate for the higher activation range. The guidance tables below support this.

#### The two stopping pathways and how conservatism affects them

When conservatism is active (`current_perf_estimate < low_performance_threshold`), there are two pathways to stopping:

1. **CI-width pathway** (reliable, reproducible, monotonic): Stops when `raw_CI_width * c < delta_cap`. This pathway produces more predictable behaviour because it depends on a smooth statistical quantity (CI width shrinking with more data). Some MCMC variance remains — particularly at low n where the posterior is unstable (e.g., c=1 at perf=1% shows CV=10%) — but it is substantially lower than the slope pathway.

2. **Slope/stabilisation pathway** (stochastic, MCMC-dependent): Stops when `|CI_slope| < CI_delta / c / 2` (the `/2` is an additional gate for low-perf conditions at line 1904 of `rule.py`). This pathway is inherently noisy because small fluctuations in the MCMC-estimated CI width trajectory determine whether the slope crosses an extremely tight threshold.

**The critical insight**: For a given performance level `p` and data budget `n_budget`, there is a **maximum conservatism value** above which the CI-width pathway cannot trigger and stopping falls through to the stochastic slope pathway. This transition explains the non-monotonicity, high MCMC variance, and unpredictable behaviour at high c.

#### Why CI-width convergence rate depends on performance level

For binary data, the CI width after `n` trials scales approximately as:

```
CI_width ≈ 3.9 × sqrt(p(1-p) / n)
```

The CI-width pathway requires `CI_width < delta_cap / c`, which gives:

```
n_required ≈ (3.9 × c / delta_cap)² × p(1-p) ≈ 6100 × c² × p   (for small p)
```

**Lower performance → faster CI convergence** (in absolute width terms), because Bernoulli variance `p(1-p)` decreases as `p → 0`. This means conservatism is more effective at very low performance — exactly where it is most needed.

Note: this formula is a rough approximation based on the normal-approximation to the Beta CI. It is accurate within ~10% for p ≥ 0.01 but overestimates by 2–13× at p ≤ 0.001 (where the Beta(1,1) prior dominates and creates a minimum CI width floor). The tables below are computed from empirical Beta posterior CI widths and should be preferred over the formula for precise guidance.

#### Trials required for CI-width pathway to trigger (delta_cap=0.05, 95% CI)

| Perf (p) | c=1 | c=3 | c=5 | c=10 | c=20 | c=50 |
|----------|-----|-----|-----|------|------|------|
| 1 in 10 (0.10) | 560 | 4,990 | 13,000+ | 50,000+ | — | — |
| 1 in 20 (0.05) | 320 | 2,640 | 7,320 | 29,000+ | — | — |
| 1 in 50 (0.02) | 150 | 1,110 | 3,050 | 12,000+ | 48,000+ | — |
| 1 in 100 (0.01) | 110 | 610 | 1,600 | 6,100 | 24,000+ | — |
| 1 in 200 (0.005) | 80 | 400 | 860 | 3,210 | 12,000+ | — |
| 1 in 500 (0.002) | 80 | 220 | 530 | 1,540 | 5,160 | 31,000+ |
| 1 in 1000 (0.001) | 80 | 220 | 370 | 1,070 | 3,070 | 16,000+ |

Values exceeding the data budget mean the CI-width pathway is unreachable and stopping relies on the stochastic slope pathway. This is where non-monotonicity, MCMC variance, and ceiling saturation occur.

#### Maximum reliable conservatism by data budget

The highest `c` where the CI-width pathway can trigger within the data budget:

| Perf (p) | n=500 | n=1,000 | n=2,000 | n=5,000 | n=10,000 |
|----------|-------|---------|---------|---------|----------|
| 1 in 10 (0.10) | — | 1 | 1 | 3 | 4 |
| 1 in 20 (0.05) | 1 | 1 | 2 | 4 | 5 |
| 1 in 50 (0.02) | 1 | 2 | 4 | 6 | 9 |
| 1 in 100 (0.01) | 2 | 3 | 5 | 8 | 12 |
| 1 in 200 (0.005) | 3 | 5 | 7 | 12 | 17 |
| 1 in 500 (0.002) | 4 | 7 | 11 | 19 | 27 |
| 1 in 1000 (0.001) | 6 | 9 | 15 | 26 | 38 |

#### Recommended parameter settings

The following table provides guidance for users who wish to tune `low_performance_threshold` and `conservatism` based on their expected success rate. Recommendations target the **maximum reliable conservatism** — the highest `c` where the CI-width pathway can trigger within the data budget, ensuring reproducible, monotonic stopping.

**`low_performance_threshold`**: As a rough guide, set to approximately 1.5–3× the expected success rate. This is a trade-off: too low and conservatism may toggle off prematurely as the running performance estimate fluctuates above the threshold; too high and conservatism activates for models that don't need it, wasting data and potentially entering the stochastic slope regime. The 2× multiplier is a reasonable middle ground, not a derived quantity — users should adjust based on how much buffer they want.

**Important**: The threshold defines the **worst case** for conservatism reliability — it is the highest performance level at which conservatism will activate. The recommended `c` must be viable not just at the expected success rate, but at all performance levels up to the threshold. Since the tables below show recommended c for the expected rate (not the threshold), users should cross-check against the "Trials required" table above to confirm that c is also viable at the threshold itself. For example, if a user expects p=0.005 and sets threshold=0.01, they should verify the recommended c works at p=0.01 too.

**For a typical budget of 2,000 trials** (e.g., 200 items × 10 epochs):

| Expected success rate | p | `low_performance_threshold` | Recommended `c` | ~Trials to stop | Efficiency |
|---|---|---|---|---|---|
| 1 in 10 | 0.10 | 0.15 | 1 (conservatism impractical) | ~560 | 72% |
| 1 in 20 | 0.05 | 0.10 | 2 | ~1,170 | 42% |
| 1 in 50 | 0.02 | 0.05 | 3 | ~1,110 | 44% |
| **1 in 100** | **0.01** | **0.02** | **5** | **~1,600** | **20%** |
| 1 in 200 | 0.005 | 0.01 | 7 | ~1,630 | 19% |
| **1 in 500** | **0.002** | **0.005** | **10** | **~1,540** | **23%** |
| 1 in 1000 | 0.001 | 0.002 | 15 | ~1,980 | 1% |

**For a larger budget of 5,000 trials** (e.g., 200 items × 25 epochs):

| Expected success rate | p | `low_performance_threshold` | Recommended `c` | ~Trials to stop | Efficiency |
|---|---|---|---|---|---|
| 1 in 10 | 0.10 | 0.15 | 3 | ~4,990 | 0% |
| 1 in 20 | 0.05 | 0.10 | 4 | ~4,690 | 6% |
| 1 in 50 | 0.02 | 0.05 | 6 | ~4,370 | 13% |
| **1 in 100** | **0.01** | **0.02** | **8** | **~3,940** | **21%** |
| 1 in 200 | 0.005 | 0.01 | 12 | ~4,490 | 10% |
| **1 in 500** | **0.002** | **0.005** | **19** | **~4,670** | **7%** |
| 1 in 1000 | 0.001 | 0.002 | 26 | ~4,480 | 10% |

Trial estimates are from Beta posterior CI width approximation (simple conjugate model); the hierarchical PyMC model may differ slightly, but the scaling relationships hold.

#### Implications for the package default of c=10

With the **package default** `low_performance_threshold=0.001`, c=10 is reliable for all performance levels where conservatism activates: at p=0.001 (the threshold), the CI-width pathway needs ~1,070 trials; at p=0.0001, only ~80 trials. **The default pairing of c=10 with threshold=0.001 is sound.**

The non-monotonicity and MCMC variance documented in Issue 2b arise specifically when users raise `low_performance_threshold` (e.g., to 0.01 or 0.1) without lowering c. In that scenario, conservatism activates at performance levels where c=10 exceeds the CI-width pathway's capacity, and stopping falls to the stochastic slope pathway.

**Recommendation for the package developer:**

1. **Document that `conservatism` and `low_performance_threshold` are coupled parameters.** Raising the threshold requires lowering c to maintain reliable CI-width-based stopping. The parameter guidance tables above (or a simplified version) should be included in the user documentation, so users can select a consistent (threshold, c) pair for their use case. The key message is: the maximum useful c depends on both the expected performance level and the data budget.

2. **Consider removing the extra `/2` gate on the slope pathway for low-perf conditions** (line 1904 of `rule.py`). Currently the slope threshold is `CI_delta / c / 2` for low-perf, but `CI_delta / c` for high-perf. The extra tightening makes the slope pathway even less likely to trigger reliably when the CI-width pathway is unreachable, pushing more stopping decisions to the data ceiling. Removing it would make the slope pathway a more viable fallback without affecting the CI-width pathway behaviour.

3. **Consider logging a warning when conservatism pushes the CI-width requirement below what is achievable.** If `delta_cap / c` is smaller than the theoretical minimum CI width for the observed performance level, the user should be informed that conservatism will primarily affect stopping via the less predictable slope pathway.

---

## 3. CI Stabilisation Trigger Analysis (`stabilisation/`)

**Status**: Complete (9/9 conditions).

**Method**: Full `optimal_stopping_posthoc` API (PyMC MCMC hierarchical model) with synthetic binary data. The stabilisation mechanism is a fallback stopping criterion that fires when the CI width slope plateaus (`|slope| <= CI_delta` over `stab_window` consecutive measurements and `slope_slopes >= 0`) before the CI width reaches `delta_cap`. To isolate this mechanism, we use tight `delta_cap` values that the CI cannot reach, forcing stabilisation to be the only viable stop path.

Design:
- 200 items x 10 epochs = 2000 trials per condition
- Performance levels: 20%, 50%, 80%
- delta_cap values: 0.05 (normal), 0.01 (tight), 0.005 (very tight)
- CI_delta: 0.001 (10x default 0.0001, to make stabilisation more likely within data budget)
- stab_window: 10 (reduced from default 15)
- 1000 MCMC draws / 1000 tune, 9 conditions total
- Total runtime: 202 min

### Results summary

| Perf | delta_cap | Stopped | Mechanism | Efficiency | Trials used | Theta | CI width |
|------|-----------|---------|-----------|------------|-------------|-------|----------|
| 20% | 0.050 | Yes | stabilisation | 78.5% | 430 | 0.225 | 0.084 |
| 20% | 0.010 | Yes | stabilisation | 79.0% | 420 | 0.230 | 0.088 |
| 20% | 0.005 | Yes | stabilisation | 81.0% | 380 | 0.225 | 0.092 |
| 50% | 0.050 | Yes | stabilisation | 76.0% | 480 | 0.482 | 0.097 |
| 50% | 0.010 | Yes | stabilisation | 78.5% | 430 | 0.478 | 0.104 |
| 50% | 0.005 | Yes | stabilisation | 75.0% | 500 | 0.479 | 0.098 |
| 80% | 0.050 | Yes | stabilisation | 78.5% | 430 | 0.782 | 0.087 |
| 80% | 0.010 | Yes | stabilisation | 78.5% | 430 | 0.784 | 0.084 |
| 80% | 0.005 | Yes | stabilisation | 79.0% | 420 | 0.783 | 0.087 |

Stop mechanism was inferred from the final CI width vs delta_cap: since all final CI widths (0.084-0.104) exceed even the loosest delta_cap (0.05), the system could not have stopped via CI convergence. Stopping was therefore triggered by the stabilisation criterion.

### Key findings

**1. The stabilisation pathway is functional, not dead code.** All 9 conditions stopped via stabilisation, confirming the mechanism activates when CI convergence is too slow to reach delta_cap. This directly answers the reviewer question.

**2. Consistent behaviour across performance levels.** Efficiency ranges from 75-81% across all conditions and performance levels, with 380-500 of 2000 trials used. The stabilisation criterion fires at a similar point regardless of true performance.

**3. delta_cap has minimal effect on stabilisation timing.** Since stabilisation fires based on slope flattening (not CI width), tightening delta_cap from 0.05 to 0.005 has little impact on when stopping occurs. This is the expected behaviour — the slope-based criterion operates independently of the width threshold.

**4. Theta estimates are reasonable.** Posterior theta estimates (0.225, 0.48, 0.78) are close to their true values (0.20, 0.50, 0.80), with slight attenuation consistent with the prior's influence on limited data.

**5. No condition in the production matrix triggered stabilisation.** In the actual 3x3 matrix, all cells stopped via CI convergence or entropy. This is because the default parameters (CI_delta=0.0001, stab_window=15) set a very high bar for stabilisation, and the production delta_cap=0.05 is achievable within the data budgets used. The relaxed parameters here (CI_delta=0.001, stab_window=10) demonstrate the mechanism under conditions where it is needed.

### Issue 3a: Tested only with non-default parameters

The stabilisation mechanism was exercised by relaxing two parameters: `CI_delta=0.001` (10x default 0.0001) and `stab_window=10` (vs default 15). Under default parameters, stabilisation never fires in any of the 9 production matrix cells — CI convergence always triggers first.

This means we have validated the **code path** (it executes and produces sensible results) but NOT the **default configuration** (the parameter settings users will actually encounter). It is unknown whether stabilisation under default settings would fire at a sensible point, produce reasonable CI widths, or behave differently from the relaxed-parameter version.

**Impact**: The analysis demonstrates that stabilisation is functional and produces reasonable estimates when it fires. But the gap between tested parameters and default parameters is large (10x CI_delta, 0.67x stab_window), and behaviour under default settings is extrapolated, not observed.

**Paper note**: The stabilisation pathway serves as a safety net for cases where CI convergence is slow. This analysis confirms the mechanism is correctly implemented under conditions that trigger it. Note that default parameters set a higher bar for stabilisation activation, and the mechanism was not triggered under default settings in the production matrix.

---

## 4. Fixed-n Baseline Comparison (`fixed_n/`)

**Status**: Complete (9/9 cells, with Bayesian estimates).

### Issue 4a: Zero variance at epoch-boundary checkpoints

At checkpoints that land exactly on epoch boundaries (e.g. n=1000 for 200 items x 5 epochs), `score_std = 0.0` across all 100 random orderings. This is expected and correct: epoch-interleaved processing shuffles item order within each epoch, so at exact epoch boundaries every ordering includes all items with exactly k epochs each, yielding an identical mean.

**Paper note**: When reporting fixed-n variance, note that zero-variance checkpoints reflect the epoch-interleaved design, not a computational error.

### Issue 4b: Ordinal theta estimates are quantized (see Issue 1b for full analysis)

The ordinal theta represents the **modal category** divided by `ordinal_max_score`, so it can only take values at multiples of 0.1 (for a 0-10 scale). This is an inherent property of modal inference on discrete categories, not a bug. See Issue 1b for the full explanation, implications, and actionable solutions.

| Cell | Typical theta (modal cat.) | Typical CI | CI width |
|------|---------------------------|------------|----------|
| low_ordinal | 0.1 (mode=1) | [0.093, 0.107] | 0.014 |
| mid_ordinal | 0.4 (mode=4) | [0.3, 0.5] | 0.2 |
| high_ordinal | 0.7 (mode=7) | [0.695, 0.706] | 0.011 |

The wide mid_ordinal CI reflects genuine uncertainty about which category is modal (scores spread across 3-5), not poor estimation. The narrow low/high CIs reflect high confidence in the modal category.

**Affected analyses**: This quantization affects both the fixed-n Bayesian comparison (Gap 3) and the consistency analysis (ordinal cell theta distributions). It does not affect the raw fixed-n sample means. The fixed-n raw means remain the appropriate baseline for comparing mean performance levels in ordinal cells.

---

## 5. Overarching Assessment

### What the validation establishes

1. **Stopping mechanisms work.** All three stop pathways (CI convergence, stabilisation, conservatism-gated) execute correctly and produce expected behaviour. CI convergence is the primary pathway in all production conditions. Stabilisation fires as a fallback when CI convergence is unreachable. The conservatism threshold cleanly gates activation at the configured performance level.

2. **Stopped samples are representative (continuous/ordinal).** Bias decomposition confirms that for continuous and ordinal inference, the epoch-interleaved stopping rule produces item subsets whose raw means closely match the full data (subset selection bias < 0.003). This is the expected result of random item shuffling and is likely generalizable. For binary inference, epoch-level sampling noise can create apparent subset bias in a specific dataset, but this is not systematic.

3. **Estimates are robust to item presentation order.** Across 15 shuffled orderings per cell, theta std is ≤ 0.006 for binary/continuous and ≤ 0.017 for ordinal. The stopping decision (stop vs not-stop) is 100% consistent across all orderings and cells.

4. **Binary and continuous inference produce usable estimates.** Total bias relative to the full-data mean is ~0.01 (within the delta_cap=0.05 precision guarantee). CI coverage is 100% for all binary and continuous cells, though this is partly a consequence of the delta_cap-bounded CI width rather than strict Bayesian calibration.

5. **Conservatism mechanism functions correctly; the defaults (c=10, threshold=0.001) are a coherent pair.** The activation threshold works as a clean binary gate. The c=1 to c≥5 step reliably increases data usage. With the package default `low_performance_threshold=0.001`, c=10 is reliable for all performance levels where conservatism activates. However, multi-seed analysis confirms that when users raise the threshold (e.g., to 0.1 as in this validation), c=10 produces non-monotonic and MCMC-dependent stopping because the CI-width pathway becomes unreachable. `conservatism` and `low_performance_threshold` are coupled parameters — raising the threshold requires lowering c. See Issue 2h for parameter guidance tables.

### What the validation does NOT establish

1. **Frequentist CI calibration.** The validation uses one dataset per cell (with 15 shuffle orderings). The 100% coverage for binary/continuous is explained by the delta_cap stopping criterion ensuring CI width ≥ 0.049, which absorbs the ~0.01 bias. We have no evidence about whether the 97% HDIs have 97% frequentist coverage across many independent datasets. This is the single largest gap in the validation. Establishing calibration would require repeating the full analysis across many data-generating seeds (e.g., 100+ synthetic datasets per cell configuration).

2. **Ranking preservation under early stopping.** The package's most important practical property — whether early-stopped estimates preserve relative rankings between models — is untested. The bias decomposition shows that model bias direction depends on performance level (negative at low perf, positive at high perf). If two models have different true performance levels, they could receive biases of different magnitude or sign, potentially affecting rankings. However, if both models are evaluated on the same items, the bias direction should be similar, likely preserving rankings. This needs explicit testing with paired model comparisons.

3. **Generalizability of quantitative findings.** All quantitative results (specific bias magnitudes, efficiency percentages, trial counts) are from specific datasets. The consistency analysis uses real eval logs (one per cell), and the conservatism/stabilisation analyses use single synthetic seeds. Qualitative patterns (mechanisms function, stopping is robust) are likely general; specific numbers are not.

4. **Ordinal cross-pathway comparability.** Ordinal theta estimates the modal category (not the mean), so ordinal CIs are not comparable to binary/continuous CIs, and coverage checks against the raw mean are a category error (see Issue 1b). The ordinal inference is valid for what it estimates, but users expecting mean-performance CIs will be misled. This is a documentation and API design issue, not an estimation error.

5. **Stabilisation under default parameters.** The stabilisation mechanism was only tested with relaxed parameters (CI_delta 10x default, stab_window reduced). Under default settings it never activates. We know the code path works, but not whether the default configuration would produce sensible behaviour if triggered.

### Practical significance for the package

**The package works well for its core use case**: early stopping of binary and continuous LLM evaluations with delta_cap=0.05. Mechanisms function correctly, estimates are within the precision guarantee, and stopped samples are representative.

**Three areas warrant attention:**

| Area | Severity | Nature | Actionable? |
|------|----------|--------|-------------|
| Ordinal reporting | High | Ordinal theta = modal category, not mean; CIs not comparable across pathways | Package: (1) document estimand difference, (2) add supplementary mean-based CI for ordinal reporting. Paper: use Jaccard for ordinal consistency; note that ordinal CIs bracket the mode, not the mean |
| Conservatism parameter coupling | Medium | `conservatism` and `low_performance_threshold` are coupled: raising the threshold without lowering c produces non-monotonic, MCMC-dependent stopping | Package: (1) document the coupling and include parameter guidance table in docs, (2) consider removing `/2` gate on slope pathway, (3) warn when CI-width target is unreachable. See Issue 2h |
| CI calibration unknown | Medium | 100% coverage is a delta_cap artefact, not calibration evidence | Requires multi-seed validation (computationally expensive). Paper: frame coverage as precision-guarantee check, not calibration |
| Ranking preservation untested | Medium | Bias direction varies with performance level | Could test with paired model comparisons on shared items. Paper: note that bias magnitude (~0.01) is small relative to typical model performance gaps |

### Implications for the main analysis scripts

The ordinal estimand mismatch (Issue 1b) has specific implications for figures and tables produced by the main analysis scripts. The following is a per-output guide for co-authors.

#### Outputs that are UNAFFECTED (all mean-based, safe to use as-is)

These outputs use raw mean scores (from eval logs or `extract_normalised_score`), not optstop's modal CI. No caveats needed:

- `analyse_matrix_results.py`:
  - **Correspondence heatmap** (`correspondence_heatmap.png`) — raw mean score normalised to [0,1]
  - **Performance vs Efficiency scatter** (`performance_vs_efficiency.png`) — Y-axis is raw mean; error bars are SE of the mean
  - **Performance Overlay** (`performance_overlay.png`) — same as above
  - **Efficiency heatmap** (`efficiency_heatmap.png`) — trial savings only
  - **Information Gain plots** (`information_gain*.png`) — running posterior variance of raw item means
  - **Cross-cell summary** (printed) — aggregates efficiency and mean-based |diff|
  - **Paired table: `Shadow`, `E-Stop`, `|Diff|` columns** — all from `extract_normalised_score` (raw mean / max_score)
- `analyse_hibayes.py`: **All outputs.** Works with raw scores and its own Bayesian models (ordered logistic for ordinal, which computes posterior expected values E[Y] = Σ(k × P(Y=k)), not modal categories)
- `analyse_hierarchical_meta.py`: **All outputs.** Inputs are mean score differences from Analysis D

#### Outputs that NEED A CAVEAT (ordinal CI values are modal, not mean)

**1. CI Width Comparison bar chart** (`ci_width_comparison.png` from `analyse_matrix_results.py`)

The ordinal bars show the **modal category CI width** (uncertainty about which response category is most common). Binary/continuous bars show the **mean performance CI width**. These are different quantities plotted on the same axis.

*Paper caveat*: "For ordinal cells, CI width reflects uncertainty in the modal response category, not the mean score. Ordinal CI widths are not directly comparable to binary/continuous CI widths, which reflect uncertainty in mean performance."

**2. Single-Run Table: `CI Width` column** (`print_single_run_table`)

Same issue — the ordinal CI width is the modal CI width. The table already has an "Ordinal supplement" section that correctly labels the modal CI bounds, but the main `CI Width` column does not distinguish the estimand.

*Paper caveat*: Same as above, or annotate ordinal CI width values with "(modal)".

**3. Paired Table: `CI-Rel` column** (`ci_relative_error` in `print_paired_comparison_table`)

This divides the mean score difference |shadow − ES| by the CI width. For ordinal cells, this divides a **mean-based quantity** by a **modal CI width** — a cross-estimand ratio that is not meaningfully interpretable. A small mean difference divided by a very narrow modal CI (e.g., low_ordinal's 0.014) produces a misleadingly large "relative error."

*Recommendation*: Do not report `CI-Rel` for ordinal cells in the paper, or note it is not comparable to binary/continuous.

**4. Paired Table: `Overlap` column** (`ci_overlap` in `print_paired_comparison_table`)

Shows Jaccard overlap of modal CIs between shadow and early-stop runs. This IS internally consistent (modal vs modal for the same cell) and is a valid measure of ordinal CI agreement. However, it cannot be compared to binary/continuous overlap values (which are absent because `extract_ci_bounds` returns `None` for those pathways), and readers should understand it measures agreement about the modal category, not the mean.

*Paper caveat*: "CI overlap for ordinal cells reflects agreement in modal category estimation between full and truncated runs."

### Pending analyses

- **Multi-seed conservatism** (`validate_conservatism_multiseed.py`): Part A complete (40/40), Part B complete (40/40). Results incorporated into Issues 2b, 2e, and 2h. Final Part B results (perf=5%, c=50) confirmed: all 5 MCMC reps hit the 2000-trial ceiling (CV=0%), consistent with the analysis.
- **Conservatism parameter guidance**: Issue 2h contains tables ready for inclusion in package documentation. The recommended c values are based on CI-width pathway viability and are independent of the `/2` gate on the slope pathway. The `/2` gate affects only the slope fallback behaviour when c exceeds the recommended maximum.
