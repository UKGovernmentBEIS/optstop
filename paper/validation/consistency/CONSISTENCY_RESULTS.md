# Presentation-Order Robustness (Consistency Analysis)

**Date**: 2026-03-15
**optstop version**: 0.3.1
**Method**: `optimal_stopping_posthoc` with 15 shuffled item orderings per cell, epoch-interleaved processing

## Overview

For each of the 9 matrix cells, the full shadow-mode eval log is replayed through the posthoc stopping API 15 times, each with a different random permutation of item presentation order. This tests whether the stopping decision — and the resulting performance estimate — is robust to the order in which items are encountered.

Three properties are assessed:
1. **Stopping consistency** — how stable is the stopping point across orderings?
2. **Estimate stability** — how close is each truncated estimate to the full-data reference?
3. **CI coverage** — does the posthoc CI contain the full-data reference score?

A bias decomposition further splits total bias into **subset selection** (stopped sample unrepresentative) and **model bias** (HDI midpoint estimator systematic error).

## Configuration

| Setting | Value |
|---------|-------|
| Reps per cell | 15 |
| MCMC draws/tune | 1000/1000 |
| Chains | 4 |
| Processing order | epoch_interleaved |
| Reanalysis interval | 10 |
| Base seed | 1000 |
| Total runtime | 27.6 hours |

## Results

| Cell | Stop Rate | Efficiency (%) | Items Used | CV | theta | theta_full | \|delta-theta\| | max\|delta-theta\| | CI Coverage | Jaccard |
|------|:---------:|---------------:|-----------:|---:|------:|-----------:|----------:|----------:|:----------:|--------:|
| low_binary | 100% | 85.6 +/- 1.9 | 200 | 0.000 | 0.047 | 0.052 | 0.006 | 0.014 | 15/15 | 0.782 |
| mid_binary | 100% | 57.7 +/- 1.1 | 198 | 0.000 | 0.510 | 0.502 | 0.008 | 0.012 | 15/15 | 0.902 |
| high_binary | 100% | 82.4 +/- 0.3 | 200 | 0.000 | 0.816 | 0.828 | 0.013 | 0.017 | 15/15 | 0.871 |
| low_continuous | 100% | 97.9 +/- 0.2 | 42 | 0.099 | 0.099 | 0.114 | 0.015 | 0.019 | 15/15 | 0.832 |
| mid_continuous | 100% | 93.1 +/- 0.4 | 139 | 0.060 | 0.416 | 0.422 | 0.006 | 0.012 | 15/15 | 0.812 |
| high_continuous | 100% | 95.3 +/- 0.3 | 93 | 0.066 | 0.699 | 0.694 | 0.006 | 0.015 | 15/15 | 0.782 |
| low_ordinal | 100% | 97.3 +/- 0.3 | 55 | 0.095 | 0.100 | 0.110 | 0.010 | 0.010 | 0/15 | 0.954 |
| mid_ordinal | 100% | 97.1 +/- 1.0 | 58 | 0.340 | 0.400 | 0.421 | 0.033 | 0.071 | 12/15 | 0.729 |
| high_ordinal | 100% | 95.7 +/- 0.9 | 86 | 0.214 | 0.700 | 0.693 | 0.008 | 0.008 | 1/15 | 0.772 |
| **Mean** | **100%** | **89.1 +/- 0.7** | | | | | **0.011** | **0.020** | **103/135** | **0.826** |

- **theta**: posthoc API estimate (HDI midpoint; for ordinal, this is the modal category / max_score)
- **theta_full**: full-data reference (mean of all scores / max_score; see Note 1)
- **CI Coverage**: fraction of reps where the posthoc 97% CI contains theta_full

## Bias Decomposition

Total bias = subset selection bias + model bias, where:
- **Subset selection** = raw_mean(stopped items) - raw_mean(all items) — measures whether early-stopped items are representative
- **Model bias** = theta_estimate - raw_mean(stopped items) — measures HDI midpoint estimator error

| Cell | Total Bias | Subset Selection | Model Bias | % Subset | % Model |
|------|----------:|--------:|---------:|--------:|-------:|
| low_binary | -0.005 | -0.012 | +0.007 | 234% | 134% |
| mid_binary | +0.008 | +0.008 | +0.000 | 100% | 0% |
| high_binary | -0.013 | -0.006 | -0.007 | 45% | 55% |
| low_continuous | -0.015 | +0.001 | -0.015 | 4% | 104% |
| mid_continuous | -0.006 | +0.000 | -0.006 | 1% | 101% |
| high_continuous | +0.005 | -0.002 | +0.006 | 37% | 137% |
| low_ordinal | -0.010 | +0.004 | -0.014 | 41% | 141% |
| mid_ordinal | -0.021 | +0.006 | -0.027 | 30% | 130% |
| high_ordinal | +0.008 | +0.003 | +0.005 | 34% | 66% |

Percentages can exceed 100% when the two components have opposite signs and partially cancel.

**By pathway:**
- **Binary**: subset selection dominates (mean -0.003). High item-level variance in binary scoring means which items fall in early epochs matters.
- **Continuous**: model bias dominates (mean -0.005). Stopped samples are representative (subset selection ~0), but the HDI midpoint has a small systematic negative bias relative to the raw mean.
- **Ordinal**: model bias dominates (mean -0.012). This conflates two effects: (1) the HDI midpoint offset and (2) the modal-vs-mean estimand gap (see Note 2).

## Caveats and Interpretation Notes

### Note 1: Ordinal reference score uses rounded integers

The posthoc API's ordinal model (Dirichlet-Multinomial) requires integer scores. The consistency script rounds continuous judge scores to the nearest integer before analysis. This means `theta_full` in the table above is the mean of **rounded** scores, which differs slightly from the main analysis's `primary_score_norm` (computed on unrounded scores):

| Cell | Rounded mean | Unrounded mean | Rounding gap |
|------|------------:|---------------:|-------------:|
| low_ordinal | 0.1101 | 0.1135 | -0.0034 |
| mid_ordinal | 0.4210 | 0.4207 | +0.0003 |
| high_ordinal | 0.6925 | 0.6932 | -0.0007 |

For low_ordinal, 30% of raw scores are non-integer (e.g., 1.2, 1.4), and rounding shifts the mean by 0.003. This is small but comparable to the total bias magnitude. Both the consistency analysis and the bias decomposition are internally self-consistent (both use the rounded reference), but direct comparison with the main analysis's theta values requires awareness of this offset.

### Note 2: Ordinal CI coverage reflects an estimand mismatch, not a method failure

The posthoc API estimates the **modal category** (most frequent response, scaled to [0,1]), while `theta_full` is the **normalised mean** score. These are different estimands:

- **low_ordinal (0/15 coverage)**: mode is exactly 0.10 (category 1/10), CI ~[0.093, 0.107]. The mean (0.110) is systematically above the mode because the score distribution is right-skewed (90% of scores = 1, with a few at 2-3).
- **high_ordinal (1/15 coverage)**: mode is 0.70, CI ~[0.694, 0.706]. The mean (0.693) just falls below every CI. The single coverage hit (rep 14) comes from an anomalous wide CI [0.60, 0.80] — likely a convergence failure in that rep.
- **mid_ordinal (12/15 coverage)**: mode is uncertain (jumps between 0.35, 0.40, 0.45 across reps), producing wider CIs that sometimes contain the mean.

This is by design: optstop uses the modal category for ordinal stopping (via entropy convergence), while the paper evaluates mean score preservation. The coverage metric is not meaningful for ordinal cells and should be reported with this caveat.

### Note 3: Ordinal posthoc efficiency differs from shadow-mode efficiency

| Cell | Shadow efficiency | Posthoc efficiency | Gap |
|------|------------------:|-------------------:|----:|
| low_ordinal | 75.3% | 97.3% | +22pp |
| mid_ordinal | 57.2% | 97.1% | +40pp |
| high_ordinal | 92.4% | 95.7% | +3pp |

The posthoc API stops much earlier for low/mid ordinal because item shuffling changes which items are seen first, and some orderings allow the entropy criterion to be satisfied with fewer items. The shadow run used a fixed ordering (seed 42) that happened to produce slower convergence. This demonstrates that ordinal stopping is sensitive to presentation order — which is precisely what the consistency analysis is designed to measure. Binary and continuous cells show close agreement between shadow and posthoc efficiency (within 1-5pp).

### Note 4: mid_ordinal is the weakest cell

mid_ordinal has the highest |delta-theta| (0.033 mean, 0.071 max), the most variable stopping point (CV = 0.340, items range 30-90), and the lowest CI Jaccard overlap (0.729). The modal category estimate is unstable because the true score distribution is spread across multiple categories at mid-performance levels. This is a known limitation of the ordinal entropy stopping criterion when the response distribution is relatively uniform.

## File Manifest

| File | Description |
|------|-------------|
| `posthoc_consistency.json` | Per-rep and per-cell consistency results (15 reps x 9 cells) |
| `bias_decomposition.json` | Per-rep and per-cell bias decomposition |
| `consistency_summary.png` | Summary table of all consistency metrics |
| `ci_coverage_forest.png` | Forest plot: theta estimates with CIs and full-data reference |
| `items_used_distribution.png` | Boxplots of items used per cell with shadow reference |
| `bias_decomposition_bars.png` | Bar chart: subset selection vs model bias per cell |
| `bias_decomposition_scatter.png` | Scatter: per-rep subset selection vs model bias |
| `bias_decomposition_table.png` | Summary table of bias decomposition |
| `analyse_posthoc_consistency.py` | Script that generated the consistency analysis |
| `analyse_bias_decomposition.py` | Script that generated the bias decomposition |
