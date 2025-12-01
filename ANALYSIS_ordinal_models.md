# Analysis: Ordinal Model Comparison

## Overview

This document compares the **Dirichlet-Multinomial** (current implementation) with the proposed **Ordered Logistic (Cumulative Link)** model, and situates both within the broader landscape of ordinal regression models.

---

## Core Difference: How They Model Categories

### Dirichlet-Multinomial (Current)

```
Categories: [0] [1] [2] [3] [4] [5] [6] [7] [8] [9] [10]
            ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑
            Exchangeable - no inherent ordering
```

Each category gets an independent probability weight. The model doesn't "know" that 7 is between 6 and 8. Categories are treated as **nominal** (exchangeable).

**Model Structure:**
```
Group Level:
  alpha_group ~ Dirichlet(1,...,1)    # Category probabilities
  kappa ~ Gamma(2, 0.1)               # Concentration parameter

Item Level:
  p_item_i ~ Dirichlet(alpha_group * kappa)  # Partial pooling

Observation Level:
  counts_i ~ Multinomial(n_i, p_item_i)
```

### Ordered Logistic (Proposed)

```
Latent scale:  ←————————————————————————————————————→
               |   |   |   |   |   |   |   |   |   |
              c₀  c₁  c₂  c₃  c₄  c₅  c₆  c₇  c₈  c₉

Category k = region between cutpoints c_{k-1} and c_k
```

Categories are regions on a continuous latent scale. Adjacent categories are naturally related through their shared boundaries.

**Model Structure:**
```
Group Level:
  mu_group ~ Normal(0, 2)             # Population latent mean
  sigma_group ~ Exponential(1)        # Between-item SD
  cutpoints ~ transformed Normal      # K-1 ordered cutpoints

Item Level:
  z_i ~ Normal(0, 1)                  # Item deviation (non-centered)
  mu_item_i = mu_group + z_i * sigma_group

Observation Level:
  P(Y_i <= k) = logistic(cutpoint_k - mu_item_i)
  Y_ij ~ OrderedLogistic(mu_item_i, cutpoints)
```

---

## Merits of Ordered Logistic

### 1. Proper Ordinal Structure

| Aspect | Dirichlet-Multinomial | Ordered Logistic |
|--------|----------------------|------------------|
| P(Y=7) informed by P(Y=6)? | No | Yes (share latent scale) |
| Neighboring shrinkage | None | Natural (via latent mean) |
| Category "distance" | All equally different | Encoded via cutpoint spacing |

**Example**: If you observe scores {6, 7, 7, 8, 7}:
- **Dirichlet** treats these as 4 independent categories with separate probabilities
- **Ordered logistic** recognizes they cluster around a latent value ~7 and borrows strength across neighbors

### 2. More Efficient Parameter Use

| Model | Parameters for K categories |
|-------|----------------------------|
| Dirichlet-Multinomial | K (alpha) + 1 (kappa) = K+1 |
| Ordered Logistic | K-1 (cutpoints) + 2 (μ, σ) = K+1 |

For K=11, both have ~12 parameters, but ordered logistic's parameters have more structure and impose meaningful constraints.

### 3. Interpretable Latent Scale

The latent mean `μ_group` represents "typical quality" on a continuous scale. This is often what users actually care about - not the discrete category distribution.

**Interpretation:**
- Higher μ → responses tend toward higher categories
- The cutpoints define where the latent scale maps to discrete categories
- Item-level μ_i shows how each item deviates from the population

### 4. Better Uncertainty Quantification

When data is sparse, ordered logistic borrows strength from the latent scale structure:
- If you've never seen category 2 but have data at 1 and 3, ordered logistic can still make reasonable inferences about P(Y=2)
- Dirichlet-Multinomial would rely purely on the prior for unobserved categories

### 5. Neighboring Category Shrinkage

In hierarchical ordered logistic:
- Items shrink toward the group mean on the **latent scale**
- This naturally causes neighboring categories to shrink together
- An item with sparse data at category 7 will borrow from the group's distribution around 7

In Dirichlet-Multinomial:
- Items shrink toward group category probabilities
- No special relationship between P(Y=7) and P(Y=6)

---

## Terminology Clarification: Is It Different from Cumulative Logit?

**No - they are the same model** with different names across fields:

| Name | Field | Formulation |
|------|-------|-------------|
| **Ordered Logistic** | Machine Learning / PyMC | P(Y ≤ k) = σ(cₖ - η) |
| **Cumulative Logit** | Statistics | logit(P(Y ≤ k)) = cₖ - η |
| **Proportional Odds** | Epidemiology | Same as cumulative logit |
| **Ordinal Regression** | General | Umbrella term |

The mathematical formulation is identical:
```
logit(P(Y ≤ k)) = cₖ - η
⟺
P(Y ≤ k) = 1 / (1 + exp(-(cₖ - η)))
⟺
P(Y ≤ k) = sigmoid(cₖ - η)
```

The "proportional odds" name comes from the assumption that the effect of η is the same across all cutpoints (parallel regression lines on the logit scale).

---

## Comparison to Other Ordinal Models

### Model Taxonomy

| Model | Link Function | What It Models | Pros | Cons |
|-------|--------------|----------------|------|------|
| **Dirichlet-Multinomial** | None (direct) | P(Y = k) directly | Simple, conjugate, fast | Ignores ordering |
| **Ordered Logistic / Cumulative Logit** | Logit | P(Y ≤ k) | Proper ordinal, interpretable | Identification issues, assumes unimodal |
| **Adjacent-Category Logit** | Logit | P(Y = k) / P(Y = k-1) | Models transitions | Less common, harder to interpret |
| **Continuation-Ratio** | Logit | P(Y = k \| Y ≥ k) | Good for sequential processes | Asymmetric, order-dependent |
| **Stereotype Model** | Logit | P(Y = k) with estimated spacing | Estimates category distances | More parameters, complex |
| **Ordered Probit** | Probit | P(Y ≤ k) | Normal latent variable | Similar to logistic, less common in Bayes |

### Why Ordered Logistic is the Recommended Choice

1. **Standard model** for ordinal data in Bayesian analysis
2. **Native PyMC support** (`pm.OrderedLogistic`)
3. **Interpretable** for Likert-type scales
4. **Well-understood** identification strategies
5. **Extensive literature** on diagnostics and extensions

### When Other Models Might Be Better

| Scenario | Recommended Model |
|----------|-------------------|
| Categories are truly nominal | Dirichlet-Multinomial |
| Sequential decision process | Continuation-Ratio |
| Unknown/unequal category spacing | Stereotype Model |
| Need fastest computation | Dirichlet-Multinomial |

---

## Handling Non-Trivial Masses at Scale Ends

This is where ordered logistic has **known weaknesses**.

### The Core Assumption

Ordered logistic assumes a **unimodal latent distribution** (logistic or normal). The probability mass in each category comes from the area under this distribution between cutpoints.

### What This Means for Different Distributions

#### Scenario 1: Unimodal (Peaked)
```
Frequency:        ████
                 ██████
                ████████
Category:   0  1  2  3  4  5  6  7  8  9  10
```
**Result**: Excellent fit ✓. This is exactly what ordered logistic is designed for.

#### Scenario 2: Unimodal (Diffuse)
```
Frequency:     ██  ██  ██  ██  ██
Category:   0  1  2  3  4  5  6  7  8  9  10
```
**Result**: Good fit ✓. Latent distribution will be wide, cutpoints spread out.

#### Scenario 3: J-Shaped (Mass at One End)
```
Frequency:  ████
            ██
            █
Category:   0  1  2  3  4  5  6  7  8  9  10
```
**Result**: Reasonable fit ✓. μ will be low, lower cutpoints close together.

#### Scenario 4: Reverse J-Shaped
```
Frequency:                          ████
                                    ██
                                    █
Category:   0  1  2  3  4  5  6  7  8  9  10
```
**Result**: Reasonable fit ✓. μ will be high, upper cutpoints close together.

#### Scenario 5: U-Shaped (Mass at Both Ends)
```
Frequency:  ████              ████
            ██                ██
Category:   0  1  2  3  4  5  6  7  8  9  10
```
**Result**: Poor fit ✗. Cannot represent with single μ. Will produce:
- Wide, uncertain posteriors
- Potential sampling divergences
- Cutpoints may not be well-identified

#### Scenario 6: Bimodal (Two Interior Peaks)
```
Frequency:     ████        ████
               ██          ██
Category:   0  1  2  3  4  5  6  7  8  9  10
```
**Result**: Poor fit ✗. Single latent mean cannot capture two modes.

### Solutions for Problematic Distributions

#### Solution 1: Stay with Dirichlet-Multinomial

For genuinely multimodal ordinal data, Dirichlet-Multinomial may be **more appropriate** because it doesn't impose unimodality. It treats each category independently, which is actually correct when there's no single underlying "quality" dimension.

#### Solution 2: Mixture of Ordered Logistic

Model the population as a mixture of latent groups:

```python
with pm.Model():
    # Mixture weight
    w = pm.Beta("w", alpha=1, beta=1)

    # Two latent means for two subpopulations
    mu_low = pm.Normal("mu_low", mu=-2, sigma=1)
    mu_high = pm.Normal("mu_high", mu=2, sigma=1)

    # Shared cutpoints
    cutpoints = pm.Normal("cutpoints", ...)

    # Mixture likelihood (complex implementation)
```

**Pros**: Can capture bimodality
**Cons**: Doubles parameters, identification harder, much slower

#### Solution 3: Zero/One-Inflated Ordered Logistic

Add explicit probability mass at extremes:

```python
# Probability of "extreme" response
p_extreme = pm.Beta("p_extreme", alpha=1, beta=5)

# Given extreme, probability of 0 vs K
p_zero_given_extreme = pm.Beta("p_zero_given_extreme", alpha=1, beta=1)

# Standard ordered logistic for non-extreme responses
# ... on categories {1, ..., K-1}
```

**Pros**: Handles floor/ceiling effects
**Cons**: More complex, assumes extremes are "different" process

#### Solution 4: Automatic Model Selection

Detect distribution shape and choose model automatically:

```python
def select_ordinal_model(counts):
    """Select appropriate ordinal model based on distribution shape."""

    if is_bimodal(counts):
        return 'dirichlet'  # Can't use ordered logistic

    if is_u_shaped(counts):
        return 'dirichlet'  # Edge masses problematic

    if is_unimodal(counts):
        return 'ordered_logistic'  # Ideal case

    return 'dirichlet'  # Safe default


def is_bimodal(counts):
    """Detect bimodal distribution."""
    # Find primary mode
    mode_idx = np.argmax(counts)
    mode_height = counts[mode_idx]

    # Look for secondary peak far from mode
    for i, c in enumerate(counts):
        if abs(i - mode_idx) > 2 and c > 0.5 * mode_height:
            return True
    return False


def is_u_shaped(counts):
    """Detect U-shaped distribution (mass at both ends)."""
    n = len(counts)
    edge_mass = counts[0] + counts[1] + counts[-2] + counts[-1]
    middle_mass = counts[2:-2].sum()
    total = counts.sum()

    # U-shaped if edges have >40% of mass and middle is sparse
    return (edge_mass / total > 0.4) and (middle_mass / total < 0.3)
```

---

## Model Selection Guidance

### Decision Table

| Distribution Shape | Sample Size | Categories | Recommended Model |
|-------------------|-------------|------------|-------------------|
| Unimodal (peaked) | Any | Any | Ordered Logistic ✓ |
| Unimodal (diffuse) | n ≥ 50 | K ≥ 5 | Ordered Logistic ✓ |
| Unimodal (diffuse) | n < 50 | Any | Dirichlet (more stable) |
| J-shaped | Any | Any | Ordered Logistic ✓ |
| U-shaped | Any | Any | Dirichlet-Multinomial |
| Bimodal | Any | Any | Dirichlet-Multinomial |
| Unknown | Any | Any | Dirichlet (safe default) |

### Quick Reference

```
Is your data clearly ordinal (Likert scale, ratings)?
├── No → Use Dirichlet-Multinomial
└── Yes → Is the distribution unimodal or J-shaped?
    ├── No (bimodal/U-shaped) → Use Dirichlet-Multinomial
    └── Yes → Do you have ≥50 observations?
        ├── No → Use Dirichlet-Multinomial (more stable)
        └── Yes → Use Ordered Logistic ✓
```

---

## Summary

| Question | Answer |
|----------|--------|
| Is ordered logistic better than Dirichlet? | **For unimodal ordinal data, yes** |
| Same as cumulative logit? | **Yes, identical model** |
| Handles J-shaped (one end)? | **Yes, well** |
| Handles U-shaped (both ends)? | **No, poorly** |
| Handles bimodal? | **No, fundamentally cannot** |
| When to prefer Dirichlet? | **Bimodal, U-shaped, nominal, small n, speed-critical** |

### Key Insight

**No single model is universally better.** The choice depends on:

1. **Distribution shape** - Most important factor
2. **Sample size** - Ordered logistic needs more data
3. **Computational budget** - Dirichlet is 2-3x faster
4. **Interpretive goals** - Latent scale vs category probabilities

The implementation should include **automatic detection** of distribution shape and appropriate model selection, with Dirichlet-Multinomial as a robust fallback for edge cases.

---

## References

1. Agresti, A. (2010). *Analysis of Ordinal Categorical Data*, 2nd ed. Wiley.
2. Bürkner, P. C., & Vuorre, M. (2019). Ordinal regression models in psychology: A tutorial. *Advances in Methods and Practices in Psychological Science*.
3. Liddell, T. M., & Kruschke, J. K. (2018). Analyzing ordinal data with metric models: What could possibly go wrong? *Journal of Experimental Social Psychology*.
4. McCullagh, P. (1980). Regression models for ordinal data. *Journal of the Royal Statistical Society: Series B*.
