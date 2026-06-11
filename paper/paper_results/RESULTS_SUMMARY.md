# optstop Validation Results: 3x3 Matrix (Package Defaults)

**Date**: 2026-03-12
**optstop version**: 0.3.1
**Comparison mode**: Within-shadow (full vs truncated data from same evaluation run)

## What Was Run

A 3x3 validation matrix testing optstop's Bayesian optimal stopping across:
- **3 performance levels**: low (~5-11%), mid (~42-50%), high (~69-83%)
- **3 inference pathways**: binary, ordinal, continuous
- **Shadow mode**: Full evaluation (200 items x 10 epochs) with stopping points recorded but not enacted

Each cell used a different benchmark/model combination to span realistic evaluation scenarios.
Within-shadow analysis compares full data against the subset that would have been used had stopping been enacted.

## Parameters

| Parameter | Value | Note |
|-----------|-------|------|
| `conservatism` | 5 | Package default |
| `low_performance_threshold` | 0.01 | Package default |
| `CI_delta` | 0.0001 | 10x larger than package default (0.00001) |
| Items per cell | 200 | |
| Epochs per cell | 10 | |
| Total trials per cell | 2,000 | |

## Per-Cell Results

Efficiency = fraction of trials saved by early stopping, i.e. `1 - stop_trial / total_trials`.
|Diff| = absolute difference in normalised mean score between full and truncated data.
Analysis D = Kruschke HDI+ROPE decision from item-matched paired Bayesian comparison (94% HDI, ROPE = +/-0.02).

| Cell | Pathway | Efficiency (%) | |Diff| (norm) | Stop Trial | Analysis D |
|------|---------|---------------:|---------------:|-----------:|:----------:|
| low_binary | binary | 84.1 | 0.0043 | 318 | undecided |
| mid_binary | binary | 58.8 | 0.0150 | 815 | undecided |
| high_binary | binary | 77.1 | 0.0049 | 458 | accept_null |
| low_ordinal | ordinal | 75.3 | 0.0003 | 494 | accept_null |
| mid_ordinal | ordinal | 57.2 | 0.0010 | 857 | accept_null |
| high_ordinal | ordinal | 92.4 | 0.0008 | 152 | accept_null |
| low_continuous | continuous | 97.3 | 0.0035 | 54 | accept_null |
| mid_continuous | continuous | 92.9 | 0.0039 | 142 | accept_null |
| high_continuous | continuous | 95.0 | 0.0222 | 100 | accept_null |

- **Mean efficiency**: 81.1% (range: 57.2% - 97.3%)
- **Mean |diff|**: 0.0062 (on [0,1] normalised scale)
- **Analysis D decisions**: 6/9 accept_null, 3/9 undecided, 0/9 reject_null

## Hierarchical Meta-Analysis

Bayesian random-effects meta-analysis pooling all 9 cells (Normal-Normal model, ROPE = +/-0.02, 97% HDI):

| Statistic | Value |
|-----------|-------|
| mu (overall effect) | +0.0003 |
| mu 97% HDI | [-0.002, +0.003] |
| ROPE decision | **accept_null** |
| tau (heterogeneity) | 0.001 (median) |

Prior sensitivity analysis (4 configurations: default, wide_mu, tight_tau, wide_tau): **all accept_null**.

After hierarchical shrinkage, all 9 cell-level effects also fall within ROPE (accept_null).

## Interpretation

Early stopping with optstop (package defaults) introduces negligible bias in mean performance estimates across all tested inference pathways and performance levels. The hierarchical meta-analysis confirms the overall effect is practically zero (mu = +0.0003, HDI well within +/-0.02 ROPE), and this conclusion is robust to prior specification.

Of the three "undecided" cells at the individual level, low_binary and mid_binary reflect wider per-cell uncertainty (binary scoring has high item-level variance). high_ordinal shows a larger effect (mu_diff = -0.059, 94% HDI [-0.114, -0.001]) driven by a known estimand mismatch: optstop tracks the modal category for ordinal stopping while the analysis evaluates mean scores. After hierarchical shrinkage in the meta-analysis, this effect contracts to -0.001, well within ROPE. No cells produced a reject_null decision.

## File Manifest

### `data/`
- `{cell_id}_shadow.json` - Raw shadow-mode result for each of the 9 cells (includes per-item scores, stopping metadata, timing)
- `analysis_summary.json` - Single-run and within-shadow paired metrics for all 9 cells

### `hibayes/`
- `hibayes_summary.json` - Full HiBayES statistical analysis output (Analyses A-D)
- `analysis_a_forest.png` - Max-tokens ranking preservation (forest plot)
- `analysis_a_*_posterior.png` - Per-condition posteriors (Analysis A)
- `analysis_b_forest.png` - Pairwise token-budget comparisons (forest plot)
- `analysis_b_*_posterior.png` - Per-condition and pairwise posteriors (Analysis B)
- `analysis_c_rope_grid.png` - Per-cell ROPE decisions grid (independent fits)
- `analysis_c_hdi_overlap_heatmap.png` - HDI overlap between full and truncated posteriors
- `analysis_c_*_posterior.png` - Per-cell posterior comparisons (Analysis C)
- `analysis_d_rope_grid.png` - Per-cell ROPE decisions grid (item-matched paired)
- `analysis_d_rope_sensitivity.png` - ROPE width sensitivity for Analysis D
- `analysis_d_*_posterior.png` - Per-cell paired-difference posteriors (Analysis D)

### `meta/`
- `hierarchical_meta_analysis.json` - Full meta-analysis output (mu, tau, shrinkage, prior sensitivity)
- `hierarchical_meta_analysis.png` - Forest plot with shrinkage estimates

### `figures/`
- `efficiency_heatmap.png` - Efficiency (%) across the 3x3 matrix
- `correspondence_heatmap.png` - Performance correspondence (|diff|) across the matrix
- `performance_overlay.png` - Full vs truncated performance comparison
- `performance_vs_efficiency.png` - Scatter: performance level vs stopping efficiency
- `information_gain.png` - Information gain curves
- `information_gain_log.png` - Information gain curves (log scale)
- `information_gain_normalised.png` - Normalised information gain curves
- `information_gain_normalised_log.png` - Normalised information gain curves (log scale)
