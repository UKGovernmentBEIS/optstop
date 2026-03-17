# OptimalStoppingManager API Reference

**Version:** 0.3.1

**Last Updated:** 2026-02-11

**Status:** Beta



---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Performance Considerations](#performance-considerations)
4. [Class: OptimalStoppingManager](#class-optimalstoppingmanager)
5. [Initialization Parameters](#initialization-parameters)
6. [Routing Logic](#routing-logic)
7. [Configuration Patterns](#configuration-patterns)
8. [Protocol Methods](#protocol-methods)
9. [Diagnostics and Return Values](#diagnostics-and-return-values)
10. [Best Practices](#best-practices)
11. [Common Pitfalls](#common-pitfalls)
12. [Examples](#examples)

---

## Overview

The `OptimalStoppingManager` implements the `inspect_ai.util.EarlyStopping` protocol, enabling statistically-rigorous adaptive early stopping for LLM evaluations within the [inspect_ai](https://inspect.aisi.org.uk/) framework.

### Key Capabilities

- **Automatic stopping decisions** based on Bayesian optimal stopping criteria
- **Flexible grouping strategies** (by model, task, metadata, tags)
- **Multiple scoring types** (binary, ordinal, continuous aggregated)
- **Independent per-grouping analysis** for multi-model/multi-task evaluations
- **Shadow mode** for validation and A/B testing
- **GPU acceleration support** for faster inference

### When to Use

Use `OptimalStoppingManager` when:
- Running expensive LLM evaluations with multiple epochs per sample
- You want to save API costs and computation time
- Statistical validity and confidence are important
- To see benefit, using early stopping when you have at least 10 samples per grouping is sensible, but the process can still be run on smaller setups (e.g., <10 samples, single epoch evals) - it is just less likely to find early stopping points. The run will progress as normal.

Note: Although logs and randomisation seeds allow for transparency and a degree of reproducibility, early stopping decisions are still data-dependent, so do not use if you need EXACT reproducibility.

---

## Critical: Correct API Usage

> **IMPORTANT:** The `early_stopping` parameter must be attached to **Task objects**, not passed to `eval()`.
>
> The `eval()` function does **NOT** have an `early_stopping` parameter. Passing it to `eval()` will silently fail—the parameter gets absorbed by `**kwargs` and ignored, resulting in no early stopping behavior.

**Correct usage patterns:**

```python
# Option 1: Attach to Task directly
task = Task(
    dataset=my_dataset,
    solver=my_solver,
    scorer=my_scorer,
    early_stopping=manager,  # Attach here
    epochs=10
)
eval(task, model="openai/gpt-4")

# Option 2: Use task_with() for registered tasks
from inspect_ai._eval.loader import load_tasks
from inspect_ai._eval.task import task_with

tasks = load_tasks(["inspect_evals/truthfulqa"])
for task in tasks:
    task_with(task, early_stopping=manager, epochs=10)  # Modify task
eval(tasks, model="openai/gpt-4")
```

**Incorrect (WILL NOT WORK):**
```python
# WRONG - early_stopping silently ignored!
eval(task, model="openai/gpt-4", epochs=10, early_stopping=manager)
```

---

## Quick Start

Minimal working example for binary scoring evaluation:

```python
from optstop.early_stopping import OptimalStoppingManager
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate

# Configure optimal stopping
manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.15,  # Max CI width for samples
        'delta_cap': 0.10,   # Max CI width for groupings
        'draws': 500,
        'tune': 500,
    },
    grouping_columns=['model', 'task'],
)

# Define your task with early_stopping attached (replace dataset with your own)
task = Task(
    dataset=[
        Sample(input="What is 2+2?", target="4", id="q_0"),
        Sample(input="What is the capital of France?", target="Paris", id="q_1"),
        # ... add more samples
    ],
    solver=[generate()],
    scorer=model_graded_fact(),
    early_stopping=manager,  # Attach to Task, NOT to eval()
    epochs=10
)

# Run evaluation
logs = eval(task, model="openai/gpt-4")

# Check results (eval returns a list)
for log in logs:
    if log.results.early_stopping:
        diagnostics = log.results.early_stopping.metadata
        print(f"Efficiency: {diagnostics['efficiency_percent']}%")
        print(f"Trials run: {diagnostics['total_ran']} / {diagnostics['total_planned_trials']}")
```

For detailed configuration options, see [Initialization Parameters](#initialization-parameters).
For performance tuning (especially ordinal scoring), see [Performance Considerations](#performance-considerations).

---

## Performance Considerations

### Critical Performance Factors

The computational cost of early stopping inference varies dramatically based on configuration. Understanding these factors is essential for production deployments.

#### 1. **Inference Pathway Performance**

| Pathway | Typical Time per Call | Complexity |
|---------|----------------------|-----------|
| **Binary** | ~10-30s | 1 group-level hierarchical MCMC run |
| **Continuous** | ~10-30s | 1 group-level hierarchical MCMC run |
| **Ordinal (modal)** | ~0.1s | Bootstrap (no MCMC) |
| **Ordinal (entropy/hybrid)** | ~2-3 min | Per-item MCMC (n_items=1, cached) + 2 group-level MCMC runs |

*All timings are approximate estimates based on default MCMC settings (draws=1000, tune=1000, chains=4) with ~100 completed items. Ordinal timings assume 5-11 categories. Model caching eliminates recompilation after the first inference call. Times scale with data size, MCMC parameters, and number of ordinal categories.*

Inference runs in a **background thread** (`ThreadPoolExecutor`) that overlaps with LLM processing. With a `reanalysis_interval` of 10 and typical LLM trial durations (10-30 seconds), there is sufficient processing time between inference triggers to absorb even ordinal hybrid inference. In comparative testing (WritingBench, 100 samples, 5 epochs), ordinal and continuous evaluations completed in virtually identical wall time (~134 min vs ~138 min), confirming that inference overhead adds no observable delay in typical LLM evaluation settings.

The more important distinction between ordinal and other pathways is **convergence behaviour**: ordinal's entropy validation gate requires more data to confirm a genuinely peaked distribution, resulting in lower efficiency (e.g., ~63% vs ~96% in comparative testing at the same precision threshold).

---

### 2. **MCMC Sampling Parameters (Critical)**

The `draws` and `tune` parameters have **linear impact** on inference time:

```python
# Current defaults (CPU/GPU-aware)
# CPU: draws=1000, tune=1000, target_accept=0.95
# GPU: draws=2000, tune=2000, target_accept=0.95

optstop_params = {
    'draws': 1000,    # Default for CPU (2000 for GPU)
    'tune': 1000,     # Default for CPU (2000 for GPU)
    'chains': 4,
    'target_accept': 0.95,  # Unified default for CPU and GPU
}

# For higher precision (if convergence warnings appear)
optstop_params = {
    'draws': 2000,
    'tune': 2000,
    'chains': 4,
    'target_accept': 0.97,  # Higher than default for better convergence
}
```

**Impact on performance:**
- Higher draws/tune values provide better posterior estimates but scale linearly with inference time
- GPU with 2000/2000 draws/tune can achieve better precision without meaningful impact on total evaluation time

**Quality trade-off:**
- Validate convergence: check R-hat < 1.01, ESS > 400
- If convergence warnings appear, increase draws/tune or target_accept

---

### 3. **Reanalysis Interval and Background Inference**

Inference runs in a background thread that overlaps with LLM processing. The time between inference triggers is determined by:

```
time_between_triggers = reanalysis_interval × avg_trial_duration
```

With default settings (`reanalysis_interval=10`) and typical LLM trial durations (10-30 seconds), there is 100-300 seconds between triggers - sufficient to absorb inference for all pathways including ordinal hybrid (~2-3 minutes).

**Edge case - potential bottleneck:** If your evaluation has very fast-completing trials (< 5 seconds), a large number of ordinal categories (20+), or both, inference may not complete before the next trigger. In this scenario, consider:
- Increasing `reanalysis_interval` (e.g., 20-30) to allow more time between triggers
- Enabling GPU acceleration (2-4x MCMC speedup)
- Using `modal` inference mode if your ordinal distributions are reliably peaked

---

### 4. **Ordinal Inference Mode Selection**

| Mode | Time per Call | Use Case | Recommendation |
|------|-------------|----------|----------------|
| **modal** | ~0.1s | Peaked distributions (most data in 1-2 categories) | Use only when confident distributions will be reliably peaked |
| **entropy** | ~2-3 min | Diffuse distributions (spread across many categories) | Full Bayesian entropy-based stopping |
| **hybrid** | ~2-3 min | General purpose (peaked or diffuse) | **Recommended**, particularly for new evaluations |

*Timings for 5-11 category scales with ~100 items. Model caching eliminates recompilation after the first call. Times increase with larger ordinal scales (20+ categories).*

**How hybrid works:**
- Computes both modal (fast) and entropy (MCMC) on every call
- Pathway 1 (peaked data): stops when modal CI is narrow and entropy confirms genuine concentration
- Pathway 2 (diffuse data): stops when entropy CI width falls below convergence threshold, even if modal CI remains wide

**Key trade-offs:**
- **Modal**: Fast but may never stop for truly diffuse distributions (CI stays wide indefinitely)
- **Hybrid**: Handles all distribution shapes; the additional inference time is absorbed by background threading and does not delay overall evaluation progress

**Convergence behaviour (the practical difference):**
In comparative testing (WritingBench, 100 samples, 5 epochs, delta_item=0.05, delta_cap=0.05), ordinal hybrid achieved ~63% efficiency vs ~96% for continuous on the same data. This difference reflects the entropy validation gate requiring more data to confirm distributional peakedness - not inference speed.

**GPU acceleration:**
GPU provides a 2-4x MCMC speedup and is beneficial for evaluations with very fast-completing trials (< 5 seconds) or large ordinal scales (20+ categories), where inference may not complete between reanalysis triggers:

```python
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    ordinal_tasks=['rating'],
    ordinal_inference='hybrid',
    gpu_ids=[0]  # 2-4× MCMC speedup
)
```

**Choosing between ordinal and continuous pathways:**

Integer-valued scores can be routed to either pathway. The choice depends on scoring design and category coverage:

- **Aggregated scores** (`score_agg='mean'` or `score_agg='median'`): Routes to continuous bounded automatically. This is the natural fit when scores are averaged across sub-criteria.
- **Raw discrete scores with well-populated categories**: The ordinal pathway (`ordinal_tasks`) preserves rank structure without interval-scale assumptions. Requires all categories observed and items per grouping >= 5x the number of categories.
- **Raw discrete scores with sparse categories**: Ordinal models can produce biased estimates and miscalibrated CIs (coverage as low as 25% at nominal 97%). Consider aggregating scores or adjusting evaluation design for better category coverage. Routing raw sparse scores via `continuous_tasks` is possible but imposes interval-scale assumptions.

The package logs a warning when category sparsity is detected under ordinal inference.

---

### 5. **Performance Monitoring**

**Check inference times in logs:**
```
INFO - Running optimal stopping inference on 25 completed trials for 'gpt-4-math'
INFO - Inference completed in 2.3 minutes
```

**If inference time exceeds time between triggers** (visible as queued inference calls in logs):

1. **Increase reanalysis_interval** to allow more time between triggers
2. **Enable GPU** for 2-4x MCMC speedup (see guidelines above)
3. **Reduce MCMC parameters** (draws/tune) if convergence diagnostics allow (check R-hat < 1.01, ESS > 400)
4. **Switch ordinal mode** from hybrid to modal, if you are confident distributions will be reliably peaked

---

### 6. **Example Performance Issues**

| Symptom | Cause | Solution |
|---------|-------|----------|
| All trials complete, 0% efficiency | Stopping criteria too strict for data variance (most common), or inference not completing between triggers (rare with background threading) | Relax delta_item/delta_cap, lower cred_level; if timing-related, check inference logs |
| Stopping decisions arrive late, lower efficiency than expected | Very fast trials (< 5s) combined with ordinal hybrid inference | Increase reanalysis_interval, reduce draws/tune, or use modal mode |
| "Inference still running" after task complete | Queue backed up | Check inference_time < reanalysis_interval × trial_duration |
| Slow convergence warnings | Insufficient MCMC iterations | Increase draws/tune (e.g., 1500/1500 or 2000/2000) |

---

## Class: OptimalStoppingManager

```python
from optstop.early_stopping import OptimalStoppingManager

manager = OptimalStoppingManager(
    optstop_params: dict[str, Any],
    grouping_columns: list[str],
    reanalysis_interval: int = 10,
    min_samples_per_grouping: int = 5,
    ordinal_tasks: Optional[list[str]] = None,
    ordinal_max_score: int = 10,
    ordinal_inference: str = 'hybrid',
    ordinal_model_type: str = 'ordered_logistic',
    gpu_ids: Optional[list[int]] = None,
    entropy_threshold: float = 0.8,
    prior_mu: float = 0.0,
    prior_sigma: Optional[float] = None,
    manager_name: str = "optstop",
    shadow_mode: bool = False,
    score_choice: Optional[str] = None,
    score_value_key: Optional[str] = None,
    score_agg: Optional[str] = None,
    random_seed: Optional[int] = None
)
```

---

## Initialization Parameters

### Required Parameters

#### `optstop_params: dict[str, Any]`
Dictionary of optimal stopping parameters passed to the underlying optstop algorithms.

**Common parameters:**
- `delta_item` (float, default: 0.05): Maximum acceptable CI width for individual samples
- `delta_cap` (float, default: 0.05): Maximum acceptable CI width for groupings/tasks
- `cred_level` (float, default: 0.97): Credibility level for confidence intervals (0.97 = 97% CI)
- `conservatism` (float, default: 5): Conservatism factor for rare events (higher = more conservative)
- `low_performance_threshold` (float, default: 0.01): Success rate below which conservative stopping applies

**Parameter interaction - `conservatism` and `low_performance_threshold`:** When a grouping's estimated performance falls below `low_performance_threshold`, the effective CI-width target tightens to `delta_cap / conservatism`. With defaults (`conservatism=5`, `delta_cap=0.05`), this target is 0.01. If you raise `low_performance_threshold`, more groupings will trigger conservatism; if you also keep `conservatism` high, the tightened target may become unreachable within your data budget. When adjusting either parameter, check that `delta_cap / conservatism` remains achievable for the sample sizes you expect.

**Advanced parameters (performance-critical):**
- `draws` (int, default: 1000 CPU / 2000 GPU): Number of MCMC samples
  - **Performance impact:** Linear scaling with inference time
  - Defaults are now CPU/GPU-aware for balanced speed and quality
- `tune` (int, default: 1000 CPU / 2000 GPU): Number of MCMC tuning steps
  - **Performance impact:** Linear scaling with inference time
  - Defaults are now CPU/GPU-aware for balanced speed and quality
- `chains` (int, default: 4): Number of MCMC chains
  - Can reduce to 2 for faster inference (2× speedup)
- `cores` (int, default: 4): Number of CPU cores for sampling
  - Usually matches `chains`
- `target_accept` (float, default: 0.95): Target acceptance rate for NUTS sampler
  - **Higher values** (0.97-0.99): Reduce divergences, but increase computation time
  - **Lower values** (0.80-0.90): Faster sampling, but may have more divergences
- `CI_delta` (float, default: 0.00001): Slope threshold for CI stabilization
- `stab_window` (int, default: 15): Window size for stabilization assessment
- `entropy_convergence_threshold` (float, default: 0.10): Absolute entropy CI width threshold on [0,1] scale for ordinal Pathway 2 convergence
  - Width < threshold means entropy is known to within ±(threshold/2) of maximum. Default 0.10 = ±5% precision.
  - For safety-critical evaluations, consider 0.08 (±4% precision, lower false positive rate)
  - **Lower values** = more conservative (require MORE precision before stopping)
  - **Higher values** = more aggressive (stop with LESS precision)
  - **Affects:** Ordinal hybrid mode only (entropy convergence pathway)

**Note:** MCMC defaults are now CPU/GPU-aware (1000/1000 for CPU, 2000/2000 for GPU). These values balance inference quality with practical performance. For publication-quality posteriors, you may increase draws/tune, but this is rarely needed for early stopping decisions.

**Example (production-optimized):**
```python
optstop_params = {
    'delta_item': 0.15,      # Allow wider CI for samples (more aggressive stopping)
    'delta_cap': 0.10,       # Require tighter CI for groupings (conservative)
    'cred_level': 0.97,      # 97% confidence intervals
    'conservatism': 5,      # Standard conservatism
    'draws': 1000,            # Baseline production recommendation (could drop lower, depending on how well behaved score distributions can be anticipated as being)
    'tune': 1000,             # Baseline production recommendation (could drop lower, depending on how well behaved score distributions can be anticipated as being)
    'chains': 4,             # ← Could drop lower to increase speed.
}
```

#### `grouping_columns: list[str]`
**REQUIRED** - List of column names to use for grouping stopping decisions.

Stopping decisions are made independently for each unique combination of these columns.

**Valid column types:**
- `'model'` - Model name from EvalSpec
- `'task'` - Task name from EvalSpec
- `'metadata.<key>'` - Sample metadata field (e.g., `'metadata.difficulty'`)
- `'tag.<name>'` - Sample tag (e.g., `'tag.category'`)

**Examples:**
```python
# Stop by model only (all tasks share stopping decisions)
grouping_columns=['model']

# Stop by model × task independently
grouping_columns=['model', 'task']

# Stop by model × subtask or sample metadata (e.g., difficulty level)
grouping_columns=['model', 'metadata.difficulty']

# Stop by model × category tag
grouping_columns=['model', 'tag.category']
```

**Important:** More granular groupings = more targeted stopping but require more data per grouping. Ensure at least 20-30 samples per unique grouping combination (see [Best Practices §3](#3-choose-appropriate-grouping-granularity)). *Note: Other optstop package functions can assist in computing required sample sizes and epoch ranges, based on historical data.*

**Reserved characters:** Grouping column values (model names, task names, metadata values) must not contain the sequence `:::`. This delimiter is used internally for sample ID tracking. A `ValueError` will be raised if any grouping value contains this sequence.

---

### Score Extraction Parameters

**CRITICAL:** These parameters determine how the manager extracts scores from inspect_ai's scores dictionary and how it routes to inference algorithms.

inspect_ai delivers scores as a two-level structure:

```
scores: dict[str, SampleScore]        ← Level 1: keyed by scorer name
                 └── .score.value     ← Level 2: usually a scalar, but can be a dict
```

`score_choice` and `score_agg` operate on **Level 1** (which scorer to use). `score_value_key` operates on **Level 2** (extracting from a dict-valued `Score.value`).

#### `score_choice: Optional[str] = None`
Select a specific scorer by name from the Level 1 scores dictionary.

**Use when:** You have multiple scorers but only want to use one for stopping decisions. The choice should be based on expected required sampling and perceived importance. For example, if you care about all 3 scores per trial, then your `score_choice` should be based on the lowest performing (or highest variance) among them. As the chosen score is the *sole basis for stopping* in this case, you should pick conservatively.

**Example:**
```python
# scores = {'accuracy': SampleScore(score=Score(value=0.8)), 'f1': SampleScore(score=Score(value=0.75))}
score_choice='f1'  # Selects scores["f1"] → uses 0.75
```

**Mutually exclusive with** `score_agg`.

#### `score_value_key: Optional[str] = None`
Extract a specific field from a dict-valued `Score.value`.

Some inspect_ai scorers return `Score.value` as a dict containing multiple fields alongside the primary numeric score (e.g., HealthBench returns `{"healthbench_score": 0.72, "criteria_met": 5, "total_criteria": 12}`). This parameter specifies which key holds the numeric value for inference.

If `Score.value` is already a scalar (float, int, string), this parameter is ignored. If `Score.value` is a dict and this parameter is not set, the sample is skipped with a warning.

**Example:**
```python
# scores = {'healthbench': SampleScore(score=Score(value={"healthbench_score": 0.72, "criteria_met": 5}))}
score_value_key='healthbench_score'  # Extracts value["healthbench_score"] → uses 0.72
```

**Composable with** `score_choice` and `score_agg`. When combined with `score_choice`, the scorer is selected first (Level 1), then the key is extracted from its dict value (Level 2). When combined with `score_agg`, the key is extracted from each scorer's dict value before aggregation.

#### `score_agg: Optional[str] = None`
Aggregate multiple scores using the specified method.

**Valid values:** `'mean'`, `'median'`, `'mode'`, `'max'`

**Use when:** You have multiple scorers and want to combine them.

**Important routing behavior:**
- If `score_agg in ['mean', 'median']`: Routes to **continuous bounded** inference (hierarchical Beta model)
- If `score_agg in ['mode', 'max']`: Routes to **discrete** inference (binary or ordinal), as these produce discrete values
- If `score_agg is None`: Routes to **discrete** inference (binary or ordinal, depending on `ordinal_tasks`)

**Example:**
```python
# scores = {'s1': SampleScore(score=Score(value=1)), 's2': SampleScore(score=Score(value=0)), 's3': SampleScore(score=Score(value=1))}
score_agg='mean'  # Mean = 0.67 → continuous bounded inference
```

**Mutually exclusive with** `score_choice`.

**Default behavior (both None):** Uses first score from dictionary (dict iteration order).

---

### Inference Control Parameters

#### `reanalysis_interval: int = 10`
Run optimal stopping inference every N completed samples (per grouping).

**Trade-offs:**
- **Smaller values** (5): More frequent stopping checks, catch stopping opportunities earlier, higher computational cost
- **Larger values** (20): Less frequent checks, lower computational cost, may miss early stopping opportunities

**Recommendation:**
- Testing/development: 5-10
- Production: 10-15

#### `min_samples_per_grouping: int = 5`
Minimum number of completed samples before running first inference for a grouping.

**Purpose:** Avoid inference with insufficient data.

**Recommendation:** Keep at 5 unless you have very large datasets (can increase to 10).

---

### Ordinal Scoring Parameters

#### `ordinal_tasks: Optional[list[str]] = None`
List of substrings to identify ordinal scoring tasks.

**Important routing behavior:**
- Tasks matching these substrings use **ordinal discrete** inference
- Tasks not matching use **binary discrete** inference (if no `score_agg`)
- Matching is case-insensitive substring match

**Example:**
```python
ordinal_tasks=['rating', 'confidence', 'difficulty']

# Task names:
# 'confidence_score' → matches 'confidence' → ordinal inference
# 'binary_accuracy' → no match → binary inference
# 'user_rating' → matches 'rating' → ordinal inference
```

**Leave as None** if you only have binary (0/1) scoring.

#### `ordinal_max_score: int = 10`
Maximum value for ordinal scores. Scores are expected to be **0-indexed**, i.e., in the range [0, ordinal_max_score].

**Examples:**
- For a 0-10 scale: `ordinal_max_score=10` (scores: 0, 1, 2, ..., 10)
- For a 0-5 scale: `ordinal_max_score=5` (scores: 0, 1, 2, 3, 4, 5)

**Important:** If your scorer produces 1-indexed scores (e.g., 1-5 star ratings), you should either:
1. Transform scores to 0-indexed before passing to the manager (subtract 1), or
2. Contact the developer to discuss support for 1-indexed ordinal scales.

**Used for:** Score validation and normalization.

#### `ordinal_inference: str = 'hybrid'`
Inference mode for ordinal tasks.

**Valid values:**
- `'modal'`: Fast (~0.1s), bootstrap-based modal category estimation. Best for peaked distributions.
- `'entropy'`: Full Bayesian entropy-based stopping (~2-3 min per call for 5-11 category scales). Best for diffuse distributions.
- `'hybrid'` (default): Runs both modal and entropy on each call (~2-3 min per call). Stops via whichever pathway is satisfied first.

**Performance notes:**
- Inference time scales with `draws`, `tune`, and the number of ordinal categories. Larger scales (20+ categories) will increase per-call time.
- Model caching eliminates recompilation after the first inference call, reducing subsequent calls.
- Inference runs in a background thread that overlaps with LLM processing. In typical evaluations, ordinal inference adds no observable delay to overall evaluation time.
- **Modal mode** uses bootstrap only (~0.1 seconds per call), but may never stop for truly diffuse distributions.

**Trade-offs:**
- **Modal:** Fast, but cannot handle diffuse distributions (CI stays wide indefinitely)
- **Hybrid/Entropy:** Handles all distribution types. The key practical difference is **convergence behaviour** - ordinal's entropy validation gate requires more data to converge, resulting in lower efficiency than binary or continuous pathways at the same precision threshold.

For evaluations with very fast-completing trials (< 5 seconds) or large ordinal scales (20+ categories), GPU acceleration (2-4x MCMC speedup) may be beneficial. See [Performance Considerations - Ordinal Inference Mode Selection](#4-ordinal-inference-mode-selection) for details.

#### `ordinal_model_type: str = 'ordered_logistic'`
Statistical model for ordinal inference.

**Valid values:**
- `'ordered_logistic'` (default): Cumulative link (proportional odds) model. Theoretically superior for truly ordinal data where neighboring categories are related.
- `'dirichlet'`: Dirichlet-Multinomial model. Treats categories as exchangeable (nominal). More robust for sparse edge categories or unusual distributions.

**Trade-offs:**

| Aspect | Ordered Logistic | Dirichlet-Multinomial |
|--------|-----------------|----------------------|
| Category structure | Ordered (ordinal) | Exchangeable (nominal) |
| Neighboring shrinkage | Natural via latent scale | None |
| Performance | Similar | Similar |
| Sparse categories | May struggle | Handles well |
| Best for | Peaked/unimodal distributions | Bimodal/unusual distributions |

**Fallback behavior:** If `ordered_logistic` sampling fails (rare), the system automatically falls back to `dirichlet` with a warning.

**Example:**
```python
# Default: Ordered Logistic (recommended for most cases)
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    ordinal_tasks=['rating'],
    ordinal_model_type='ordered_logistic'  # Default
)

# Alternative: Dirichlet-Multinomial for problematic distributions
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    ordinal_tasks=['rating'],
    ordinal_model_type='dirichlet'  # Fallback option
)
```

**Unusual scoring configurations?** If your evaluation involves very large rubric scales (e.g., >20 categories), peculiar response distributions (e.g., models can only produce scores of 1, 3, 8, and 10 on a 0-10 scale), or you are uncertain about how to select or balance multiple scorer priorities for applying optimal stopping, please reach out to the package owner (Toby Pilditch on AISI Slack) for guidance on configuring optstop for your use case.

---

### Hardware Configuration Parameters

#### `gpu_ids: Optional[list[int]] = None`
List of GPU device IDs to use for inference.

**Example:**
```python
gpu_ids=[0]        # Use first GPU
gpu_ids=[0, 1]     # Use first two GPUs (not currently parallelized)
gpu_ids=None       # CPU-only mode (default)
```

**Requirements:**
- NVIDIA GPU with CUDA support
- JAX with GPU support installed: `pip install optstop[gpu]`

**Note:** GPU configuration for inspect_ai runtime environments is still being validated (see roadmap).

---

### Entropy Threshold

#### `entropy_threshold: float = 0.8`
Proportion of maximum entropy for false peak detection in ordinal hybrid mode.
Scaled internally by `log2(num_categories)` to produce an effective threshold in bits.

- Lower values (e.g., 0.5) require more concentrated distributions to pass the entropy gate
- Higher values (e.g., 0.9) are more permissive, allowing earlier Pathway 1 stopping
- Only relevant when `ordinal_inference='hybrid'`

---

### Prior Configuration

#### `prior_mu: float = 0.0`
Centre of the group-level Normal prior on the logit scale. This parameter affects the Bayesian hierarchical model's prior assumption about typical performance.

**Interpretation on probability scale:**
- `prior_mu=0.0` → 50% probability centre (assumption-free default)
- `prior_mu=2.0` → ~88% probability centre
- `prior_mu=-2.0` → ~12% probability centre

**When to adjust:**
- Leave at default (0.0) for most evaluations - this is an uninformative prior that lets data drive inference
- If you have recent results for the same model on the same benchmark (e.g., from a previous evaluation run), setting `prior_mu` to the logit of that known performance improves the accuracy of early point estimates and credible interval placement. For example, a model previously measured at ~75% accuracy corresponds to `prior_mu=1.1` (`scipy.special.logit(0.75)`). Because credible intervals are computed on the probability scale via the logistic transform, a well-placed posterior also produces narrower probability-scale intervals than one centred near 0.5 at the same level of precision - which may modestly accelerate stopping for groupings whose true performance is far from 50%. The prior width (sigma=1.5 for binary/continuous, 2.0 for ordinal) is broad enough that a mis-specified prior is overridden by data within approximately 15-20 items

**Technical note:** The prior is `mu_group ~ Normal(prior_mu, sigma)` where sigma is determined by `prior_sigma`. If `prior_sigma` is None (default), pathway-specific defaults apply: 1.5 for binary/continuous, 2.0 for ordinal. If `prior_sigma` is explicitly set, that value applies to all pathways.

---

#### `prior_sigma: Optional[float] = None`
Scale (standard deviation) of the group-level Normal prior on the logit scale. Controls the prior's informativeness.

**Interpretation:**
- `prior_sigma=None` (default) → Uses pathway-specific defaults: 1.5 for binary/continuous, 2.0 for ordinal
- `prior_sigma=1.0` → More informative, provides stronger regularization toward `prior_mu` (all pathways)
- `prior_sigma=1.5` → Weakly informative for binary/continuous (matches default)
- `prior_sigma=2.0` → More diffuse, matches ordinal default

**When to adjust:**
- Most users should not need to change this. The pathway-specific defaults provide appropriate regularization while allowing data to dominate after 15-20 items.
- Set explicitly if you want the same sigma across all pathways.
- Smaller values (e.g., 1.0) may be useful when you have strong prior knowledge and want faster convergence.
- Larger values (e.g., 2.0) may be useful when you expect high variability across items within a grouping.

**Note:** When None, binary/continuous use 1.5 and ordinal uses 2.0. When explicitly set, the value applies to all pathways including ordinal.

**Model choice rationale (binary pathway):** The logit-normal hierarchical model was chosen over a Beta-Binomial alternative after controlled simulation comparison. Under a Beta-Binomial data generating process with κ=10 (inherently favouring that model), the two approaches are statistically indistinguishable in bias, coverage, and stopping behaviour across the mid-range (0.1-0.9). At exact boundaries (0.0 or 1.0), the Beta-Binomial has ~50% less point-estimate bias but its credible intervals are 12% narrower with no improvement in coverage - both models exhibit reduced boundary coverage (~0.45 vs nominal 0.97) due to fundamental information limitations when sparse binary data cannot distinguish true homogeneity from sampling coincidence. Because the stopping algorithm triggers when CI width falls below `delta_cap`, the Beta-Binomial's narrower intervals cause earlier stopping on less data in precisely the regime where estimates are least reliable. The Beta-Binomial also exhibited 10-100× more MCMC divergences at boundaries, indicating worse posterior geometry. The logit-normal's wider boundary intervals therefore function as implicit conservatism - the appropriate default for evaluation contexts where premature termination is costlier than collecting additional data.

---

### Utility Parameters

#### `manager_name: str = "optstop"`
Identifier for this manager instance. Used in logging and thread naming.

**Useful when:** Running multiple managers simultaneously.

#### `shadow_mode: bool = False`
If True, run all trials without actually stopping, but track what would have stopped.

**Use cases:**
- Validation: Compare performance with/without early stopping
- A/B testing: Measure efficiency gains without affecting results
- Debugging: Verify stopping decisions without altering trial execution

**Behavior:**
- `schedule_sample()` always returns `None` (run all trials)
- Inference still runs and stopping decisions are tracked
- `complete_task()` diagnostics include `stopped_at_trial_count` (per-grouping global trial count and completed samples at the time stopping first triggered) and a `shadow_mode_summary` block with `would_have_stopped_at` and `potential_efficiency_percent`
- Note: `efficiency_percent` in the main metadata will be 0% in shadow mode (no trials are actually skipped). Use `shadow_mode_summary` for the counterfactual efficiency estimate

---

### Reproducibility Parameter

#### `random_seed: Optional[int] = None`
Random seed for MCMC sampling reproducibility.

**Behavior:**
- If **specified** (e.g., `random_seed=42`): Uses the provided seed for all MCMC inference
- If **not specified** (default): Auto-generates a random seed using system entropy

**Key features:**
- Seed is **always logged** at manager initialization:
  ```
  INFO:optstop.early_stopping:Random seed: 1614538249 (auto_generated)
  INFO:optstop.early_stopping:Random seed: 42 (user_specified)
  ```
- Seed is **included in configuration summary** printed at task start
- Seed is **included in diagnostics** returned by `complete_task()`:
  ```python
  diagnostics = {
      "random_seed": 42,
      "seed_source": "user_specified",  # or "auto_generated"
      ...
  }
  ```
- Seed is **passed directly to PyMC** via `sampling_kwargs['random_seed']`

**Use cases:**
- **Debugging:** Reproduce exact stopping decisions by using the same seed
- **Testing:** Verify consistent behavior across runs
- **Validation:** Compare results with controlled randomness

**Example:**
```python
# For reproducible results
manager = OptimalStoppingManager(
    optstop_params={'draws': 500, 'tune': 500},
    grouping_columns=['model', 'task'],
    random_seed=42  # Same seed = same MCMC results
)

# For production (let system generate)
manager = OptimalStoppingManager(
    optstop_params={'draws': 500, 'tune': 500},
    grouping_columns=['model', 'task'],
    # random_seed not specified - auto-generated and logged
)
```

**Important notes:**
- Different MCMC backends (PyMC default vs numpyro) may produce different results even with the same seed
- Seed ensures reproducibility **within the same configuration**, not across different backends

---

## Routing Logic

The manager automatically routes to different inference algorithms based on configuration:

```
┌─────────────────────────────────────────────┐
│   Initialization Parameters                  │
└──────────────┬──────────────────────────────┘
               │
               ▼
        Has score_agg in
        ['mean', 'median']?
               │
        ┌──────┴──────┐
        │             │
       YES            NO (includes mode/max or no score_agg)
        │             │
        ▼             ▼
   CONTINUOUS    Has ordinal_tasks
   BOUNDED       substring match?
   (Beta)             │
                 ┌────┴────┐
                 │         │
                YES        NO
                 │         │
                 ▼         ▼
              ORDINAL   BINARY
             (Ord.Log) (Binom)
```

### Routing Rules

| Configuration | Inference Type | Model | Score Range |
|--------------|----------------|-------|-------------|
| `score_agg='mean'` or `'median'` | **Continuous Bounded** | Hierarchical Beta | [0, 1] |
| `ordinal_tasks=['...']` match + no aggregation | **Ordinal Discrete** | Ordered Logistic (default) or Dirichlet | [0, max_score] |
| No aggregation + no ordinal match | **Binary Discrete** | Binomial | {0, 1} |

### Routing Examples

**Binary Discrete:**
```python
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    # NO score_agg parameter
    # NO ordinal_tasks parameter
)
# → Routes to binary discrete (binomial model)
# → Expects scores: 0 or 1 (discrete)
```

**Ordinal Discrete:**
```python
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    ordinal_tasks=['rating', 'confidence'],
    ordinal_max_score=5,
    # NO score_agg parameter
    # ordinal_model_type='ordered_logistic' is the default
)
# → Routes to ordinal discrete (Ordered Logistic model by default)
# → Expects scores: 0, 1, 2, 3, 4, or 5 (discrete)
```

**Continuous Bounded:**
```python
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    score_agg='mean',  # ← CRITICAL: triggers continuous routing
)
# → Routes to continuous bounded (hierarchical Beta)
# → Expects scores: any float in [0, 1] (continuous)
```

---

## Configuration Patterns

### Pattern 1: Simple Binary Evaluation

**Use case:** Binary scoring (0/1), single model, single task.

```python
from optstop.early_stopping import OptimalStoppingManager
from inspect_ai import eval, Task

manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.15,
        'delta_cap': 0.10,
        'cred_level': 0.97,
    },
    grouping_columns=['model', 'task'],
    reanalysis_interval=10,
)

# Attach early_stopping to Task (NOT to eval)
task = Task(
    dataset=my_dataset,
    solver=my_solver,
    scorer=my_scorer,
    early_stopping=manager,
    epochs=10
)

logs = eval(task, model="openai/gpt-4")
```

### Pattern 2: Multi-Model Comparison

**Use case:** Comparing multiple models on same task, want independent stopping per model.

```python
from inspect_ai._eval.task import task_with

manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.15,
        'delta_cap': 0.10,
    },
    grouping_columns=['model'],  # Group by model only
    reanalysis_interval=10,
)

# Attach early_stopping to Task
task_with(task, early_stopping=manager, epochs=10)

# Each model gets independent stopping decisions
logs = eval(
    task,
    model=["openai/gpt-4", "anthropic/claude-3", "google/gemini-pro"]
)
```

### Pattern 3: Ordinal Rating Tasks

**Use case:** Tasks with ordinal scores (e.g., 0-5 rating scale).

```python
manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.15,
        'delta_cap': 0.10,
    },
    grouping_columns=['model', 'task'],
    ordinal_tasks=['rating', 'confidence'],  # Mark ordinal tasks
    ordinal_max_score=5,                     # 0-5 scale (scores: 0,1,2,3,4,5)
    ordinal_inference='hybrid',              # Auto-select inference mode
    reanalysis_interval=10,
)
```

### Pattern 4: Multiple Scorers with Aggregation

**Use case:** Multiple binary scorers, want to aggregate them into continuous score.

```python
manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.15,
        'delta_cap': 0.10,
    },
    grouping_columns=['model', 'task'],
    score_agg='mean',  # Aggregate all scores (routes to continuous)
    reanalysis_interval=10,
)

# If sample has scores {'acc': 1, 'f1': 1, 'recall': 0}
# → mean = 0.667 → continuous bounded inference
```

### Pattern 5: Specific Score Selection

**Use case:** Multiple scorers, but only want to use one for stopping.

```python
manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.15,
        'delta_cap': 0.10,
    },
    grouping_columns=['model', 'task'],
    score_choice='accuracy',  # Use only 'accuracy' score
    reanalysis_interval=10,
)

# If sample has scores {'accuracy': 1, 'f1': 0, 'recall': 1}
# → Uses only accuracy = 1 → binary discrete inference
```

### Pattern 5b: Dict-Valued Score Extraction

**Use case:** Scorer returns `Score.value` as a dict with multiple fields (e.g., HealthBench, HLE).

```python
manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.15,
        'delta_cap': 0.10,
    },
    grouping_columns=['model', 'task'],
    score_value_key='healthbench_score',  # Extract from dict-valued Score.value
    score_agg='mean',                     # Route to continuous inference
    reanalysis_interval=10,
)

# If sample has scores {'healthbench': SampleScore(score=Score(value={"healthbench_score": 0.72, "criteria_met": 5}))}
# → Extracts 0.72 from the dict → continuous bounded inference
```

### Pattern 6: Shadow Mode for Validation

**Use case:** Want to measure potential efficiency gains without actually stopping.

```python
from inspect_ai._eval.task import task_with

manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.15,
        'delta_cap': 0.10,
    },
    grouping_columns=['model', 'task'],
    shadow_mode=True,  # Run all trials, track stopping decisions
    reanalysis_interval=10,
)

# Attach to Task
task_with(task, early_stopping=manager, epochs=10)

logs = eval(task, model="gpt-4")

# Check potential efficiency gains
for log in logs:
    if log.results.early_stopping:
        diagnostics = log.results.early_stopping.metadata
        summary = diagnostics.get('shadow_mode_summary', {})
        print(f"Would have stopped at trial: {summary.get('would_have_stopped_at')}")
        print(f"Potential savings: {summary.get('potential_efficiency_percent')}%")
        print(f"Per-grouping details: {diagnostics.get('stopped_at_trial_count')}")
```

### Pattern 7: Custom Metadata Grouping

**Use case:** Group by custom sample metadata (e.g., difficulty level).

```python
# Samples have metadata: {'difficulty': 'easy'/'medium'/'hard'}
manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.15,
        'delta_cap': 0.10,
    },
    grouping_columns=['model', 'metadata.difficulty'],
    reanalysis_interval=10,
)

# Independent stopping per difficulty level
# Easy samples might stop early, hard samples run longer
```

---

## Protocol Methods

The `OptimalStoppingManager` implements the `EarlyStopping` protocol with four methods:

### `async start_task(task, samples, epochs) -> str`

Called once at the beginning of evaluation to initialize the manager.

**Parameters:**
- `task: EvalSpec` - Evaluation specification from inspect_ai
- `samples: list[Sample]` - List of samples to evaluate
- `epochs: int` - Number of epochs per sample

**Returns:** Manager name string

**Behavior:**
- Creates `compiled_dataset` with all planned trials (samples × epochs)
- Extracts grouping columns from EvalSpec and sample metadata
- Initializes internal tracking structures
- Prints configuration summary to console

**User action:** None required (called automatically by inspect_ai)

### `async schedule_sample(id, epoch) -> EarlyStop | None`

Called before each trial to determine if it should run or be stopped.

**Parameters:**
- `id: str | int` - Sample ID
- `epoch: int` - Epoch number

**Returns:**
- `None` - Run this trial
- `EarlyStop` - Skip this trial (stopped early)

**EarlyStop model:**
```python
class EarlyStop(BaseModel):
    id: str | int         # Sample dataset ID
    epoch: int            # Sample epoch number
    reason: str | None    # Reason for the early stop
    metadata: dict[str, JsonValue] | None  # Additional metadata
```

**Behavior:**
- Fast DataFrame lookup (< 1ms typical)
- Checks `schedule_status` flag in compiled_dataset
- Uses cache for repeated lookups
- If `shadow_mode=True`, always returns `None`

**User action:** None required (called automatically by inspect_ai)

### `async complete_sample(id, epoch, scores) -> None`

Called after each trial completes to update scores and potentially trigger inference.

**Parameters:**
- `id: str | int` - Sample ID
- `epoch: int` - Epoch number
- `scores: dict[str, SampleScore]` - Scores from scorers

**Returns:** None

**Behavior:**
1. Extracts score value using `score_choice`, `score_value_key`, and/or `score_agg`
2. Validates score (type, range)
3. Updates `compiled_dataset` with score and marks trial as complete
4. Increments per-grouping counter
5. If counter % `reanalysis_interval` == 0:
   - Runs `optimal_stopping_live_single()` in executor
   - Updates `schedule_status` for stopped samples/groupings
   - Logs stopping decisions

**User action:** None required (called automatically by inspect_ai)

### `async complete_task() -> dict[str, JsonValue]`

Called once at the end of evaluation to generate final diagnostics.

**Returns:** Dictionary with comprehensive diagnostics (see Diagnostics section)

**Behavior:**
- Shuts down inference executor gracefully
- Calculates efficiency metrics
- Compiles stopped samples information
- Computes convergence projections for non-stopped groupings (estimated additional trials to convergence)
- Returns comprehensive metadata dictionary

**User action:** Access diagnostics from `log.results.early_stopping.metadata` after evaluation

---

## Diagnostics and Return Values

The `complete_task()` method returns a comprehensive diagnostics dictionary:

### Top-Level Fields

```python
{
    "manager": str,                      # Manager name
    "random_seed": int,                  # Random seed used for MCMC inference
    "seed_source": str,                  # "user_specified" or "auto_generated"
    "total_planned_trials": int,         # Total trials planned (samples × epochs)
    "total_ran": int,                    # Trials actually executed
    "total_skipped": int,                # Trials skipped due to early stopping
    "efficiency_percent": float,         # (skipped / planned) × 100
    "stopped_samples_count": int,        # Number of samples that stopped early
    "stopped_samples": list[dict],       # Detailed list of stopped samples
    "grouping_columns": list[str],       # Grouping configuration used
    "reanalysis_interval": int,          # Reanalysis interval used
    "min_samples_per_grouping": int,     # Min samples threshold
    "stopped_samples_per_grouping": dict,# Count per grouping
    "stopped_groupings": list[str],      # Groupings that stopped completely
    "stopped_groupings_count": int,      # Count of stopped groupings
    "decision_counters": dict,           # Per-grouping inference timing info
    "stabilization_histories": dict,     # Per-grouping convergence metrics
    "stopped_at_trial_count": dict,      # Per-grouping trial count when stopping first triggered
}
# In shadow mode, an additional key is appended:
#   "shadow_mode_summary": dict          # would_have_stopped_at, potential_efficiency_percent
```

**Note:** `item_entropy_histories` is maintained internally by `OptimalStoppingManager` to persist per-sample entropy state across successive inference calls (used for Pathway 2 convergence detection). It is not included in the metadata returned by `complete_task()`.

### Stopped Samples Structure

```python
"stopped_samples": [
    {
        "id": "sample_123",
        "epoch": 5,
        "reason": "Sample-level CI convergence",
        "metadata": {
            "epochs_used": 5,
            "ci_width": 0.08,
            "threshold": 0.15,
            "reason": "Sample-level CI convergence"
        }
    },
    # ... more stopped samples
]
```

### Decision Counters Structure

```python
"decision_counters": {
    "gpt-4-math": {
        "completed_samples": 47,
        "inference_calls": 4,           # 47 // 10 = 4
        "next_inference_at": 50         # (4 + 1) × 10 = 50
    },
    # ... more groupings
}
```

### Stabilization Histories Structure

```python
"stabilization_histories": {
    "gpt-4-math": {
        "n_samples": 47,
        "final_ci_width": 0.12,
        "final_slope": 0.00003,
        "n_group_checks": 4
    },
    # ... more groupings
}
```

#### Convergence Projection (Non-Stopped Groupings)

For groupings that have **not** stopped by the end of evaluation, a `convergence_projection` key is included in the stabilization history entry. This estimates how many additional trials would be needed for convergence, using an exponential decay model (primary) with linear extrapolation fallback.

The projection is only populated when CI width history is available (at least one group-level inference check has run). It is **absent** for stopped groupings and groupings with no CI width data.

```python
"stabilization_histories": {
    "gpt-4-math_hard": {
        "n_samples": 47,
        "final_ci_width": 0.12,
        "final_slope": -0.0003,
        "n_group_checks": 4,
        "convergence_projection": {
            # Point estimate
            "projected_additional_steps": 15,        # In observation-step units
            "projected_additional_trials": 120,      # steps * reanalysis_interval (order-sensitive)
            "simple_proj_additional_trials": 276.0,  # 1/sqrt(n) projection (order-stable)
            "trajectory_signal": "faster",           # 'faster'/'on_pace'/'slower' vs 1/sqrt(n) (120/276=0.43)
            "proximity_ratio": 2.4,                  # final_width / delta_cap (>1 = not converged)

            # Outcome classification
            "convergence_target": "projected_width",  # See values below
            "projected_width_at_termination": 0.048,
            "plateau_reached": False,
            "capped": False,

            # Current state
            "final_width": 0.12,
            "final_slope": -0.0003,

            # Model info
            "projection_basis": "exponential_decay",  # or "linear_extrapolation"
            "exponential_fit": {                       # Only present for exponential_decay basis
                "a": 0.35, "b": 0.02, "c": 0.03,     # w(t) = a*exp(-b*t) + c
                "r_squared": 0.94
            },

            # Uncertainty quantification
            "uncertainty": {
                "ci_trials_80": [100, 210],           # 80% bootstrap CI
                "ci_trials_50": [130, 180],           # 50% bootstrap CI
                "bootstrap_skipped": False,
                "n_width_observations": 4,
                "trajectory_rmse": 0.008,
                "confidence_level": "moderate"         # "high", "moderate", or "low"
            }
        }
    },
    "gpt-4-coding": {
        # Stopped grouping - no convergence_projection field
        "n_samples": 89,
        "final_ci_width": 0.03,
        "final_slope": -0.00001,
        "n_group_checks": 8
    }
}
```

**Convergence Projection Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `projected_additional_steps` | int | Projected observation steps to convergence |
| `projected_additional_trials` | int | Projected additional trials needed (= steps × reanalysis_interval). Derived from exponential or linear trajectory fit - sensitive to item ordering (CV 0.5-3.0). Treat as order-of-magnitude guide |
| `simple_proj_additional_trials` | float | Conservative projection assuming CI contracts as 1/√n. Stable across item orderings (CV 0.03-0.09). Use for quantitative planning |
| `trajectory_signal` | str/None | `'faster'`, `'on_pace'`, or `'slower'` - whether the trajectory-based projection suggests faster or slower convergence than the 1/√n baseline. None when the comparison is unavailable |
| `proximity_ratio` | float | Current CI width / `delta_cap` (the group-level CI width threshold, default 0.05). Values > 1 indicate not yet converged; 1.0-1.5 suggests near-convergence; > 3.0 suggests substantial additional data needed |
| `convergence_target` | str | `'projected_width'` (CI narrows below delta), `'projected_slope_stabilisation'` (CI plateaus above delta), or `'projected_capped'` (neither within projection horizon) |
| `projected_width_at_termination` | float | Projected CI width at the convergence/plateau/cap point |
| `plateau_reached` | bool | True when the CI width trajectory has already flattened above `delta_cap` at the time of projection. When `plateau_reached=True` and `projected_additional_trials=0`, the grouping is stuck - not converged |
| `capped` | bool | True if projection exceeded the maximum step limit (default: 200 steps, i.e., `200 × reanalysis_interval` trials) |
| `final_width` | float | Current CI width at time of projection |
| `final_slope` | float | Current CI slope at time of projection |
| `projection_basis` | str | `'exponential_decay'` (primary, >= 5 observations, R² >= 0.7) or `'linear_extrapolation'` (fallback) |
| `exponential_fit` | dict | Only present for `exponential_decay` basis. Contains: `a` (amplitude), `b` (decay rate), `c` (asymptote - the width the trajectory decays toward), `r_squared` (goodness of fit, always >= 0.7) |
| `uncertainty.ci_trials_80` | list[int, int] | 80% bootstrap CI on projected trials [10th, 90th percentile] |
| `uncertainty.ci_trials_50` | list[int, int] | 50% bootstrap CI on projected trials [25th, 75th percentile] |
| `uncertainty.bootstrap_skipped` | bool | True if insufficient data for bootstrap (< 7 obs for exponential, < 3 slopes for linear) or bootstrap was disabled |
| `uncertainty.n_width_observations` | int | Number of CI width observations used for fitting (exponential basis only) |
| `uncertainty.n_slope_observations` | int | Number of slope observations available (linear basis only) |
| `uncertainty.trajectory_rmse` | float | Root mean squared error of the fitted model against observed data |
| `uncertainty.confidence_level` | str | `'high'`, `'moderate'`, or `'low'` based on fit quality and data quantity |
| `uncertainty.some_bootstrap_capped` | bool | (Optional) Present and True when some bootstrap iterations hit the max_steps cap |

**Interpreting `convergence_target`:**

- **`projected_width`**: CI width is narrowing toward `delta_cap`. `projected_additional_trials` estimates when it will cross below. Expected outcome for groupings that would converge with more data. Increase your sample budget or epochs accordingly.
- **`projected_slope_stabilisation`**: CI width is projected to plateau *above* `delta_cap` - additional data yields diminishing returns. Widen `delta_cap` to accept the current precision, or investigate whether the grouping has high intrinsic variance.
- **`projected_capped`**: Neither convergence nor plateau detected within the projection horizon (default: 200 steps, i.e., `200 × reanalysis_interval` trials). Check whether the grouping has very few observations (< 5 group-level checks) - more data may clarify the trajectory. If observations are plentiful but no trend emerges, the data may be too noisy for the current stopping criteria.

#### Ordinal-Specific Fields

For ordinal groupings (tasks matching `ordinal_tasks` patterns), additional diagnostic fields are included:

```python
"stabilization_histories": {
    "gpt-4-rating_task": {
        # Standard fields (all score types)
        "n_samples": 47,
        "final_ci_width": 0.12,
        "final_slope": 0.00003,
        "n_group_checks": 4,

        # Ordinal-specific fields (only for ordinal groupings)
        "ordinal_pathway": "modal_hierarchical",  # Inference pathway (see field descriptions below)
        "final_modal_ci_width": 0.10,         # Final modal category CI width (scaled 0-1)
        "final_modal_ci": [0.60, 0.70],       # Final modal category CI bounds (scaled 0-1)
        "final_entropy": 1.85,                # Final entropy estimate (bits for flat, scaled [0,1] for hierarchical)
        "final_entropy_threshold": 2.77,      # Effective entropy threshold in bits (0.8 × log2(11))
        "final_entropy_ci_width": 0.08,       # Entropy-based CI width (if entropy pathway)
        "final_convergence_threshold": 0.10   # Entropy convergence threshold on [0,1] scale (Pathway 2)
    }
}
```

**Field Descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `ordinal_pathway` | str/int | Which inference pathway was used. Values: `'modal_hierarchical'`, `'entropy_hierarchical'`, `'hybrid_hierarchical'` (before stopping resolves), or `1` (modal CI narrow + validated), `2` (entropy converged) when hybrid stopping triggers. |
| `final_modal_ci_width` | float | Width of the modal category credible interval, scaled to [0,1]. Lower values indicate more certainty about the modal category. |
| `final_modal_ci` | list[float] | [lower, upper] bounds of the modal category CI, scaled to [0,1]. E.g., `[0.60, 0.70]` means 97% confident modal category is between 6 and 7 (on a 0-10 scale). |
| `final_entropy` | float | Shannon entropy of the categorical distribution. Units depend on inference path: bits (log2) for flat ordinal, or scaled [0,1] (proportion of max entropy) for hierarchical ordinal. Lower values indicate more peaked/concentrated distributions. |
| `final_entropy_threshold` | float | Effective entropy threshold in bits (entropy_threshold × log2(K), where K is number of categories). Distributions with entropy below this are considered "peaked." |
| `final_entropy_ci_width` | float | CI width derived from entropy-based inference (used in entropy/hybrid pathways). |
| `final_convergence_threshold` | float | Absolute entropy CI width threshold on [0,1] scale for Pathway 2 convergence (default: 0.10). Entropy CI width below this value triggers stopping. |

**Note:** These ordinal-specific fields are only populated when the grouping matches an `ordinal_tasks` pattern. For binary and continuous groupings, these fields are omitted entirely (not set to `null`).

**Ordinal estimand:** For ordinal groupings, the group-level performance estimate (theta) represents `modal_category / max_score` - the most probable score category, normalised to [0,1]. This differs from binary and continuous pathways, which estimate mean performance. Credible intervals bracket the mode, not the mean. Convergence for ordinal groupings therefore reflects stability of the modal category estimate, not stability of average score.

### Accessing Diagnostics

```python
from inspect_ai import eval
from inspect_ai._eval.task import task_with

# Attach early_stopping to task first
task_with(task, early_stopping=manager, epochs=10)

# Run evaluation
logs = eval(task, model="gpt-4")

# Access results (eval returns a list)
for log in logs:
    if not log.results.early_stopping:
        continue

    # Access early stopping summary (EarlyStoppingSummary object)
    early_stopping = log.results.early_stopping

    # Access manager name and early stops list
    print(f"Manager: {early_stopping.manager}")
    print(f"Early stops: {len(early_stopping.early_stops)}")

    # Access detailed diagnostics from metadata dict
    diagnostics = early_stopping.metadata

    print(f"Efficiency: {diagnostics['efficiency_percent']}%")
    print(f"Trials saved: {diagnostics['total_skipped']} / {diagnostics['total_planned_trials']}")

    # Iterate through stopped samples (from metadata)
    for sample in diagnostics['stopped_samples']:
        print(f"Sample {sample['id']} stopped at epoch {sample['epoch']}")
        print(f"  Reason: {sample['reason']}")

    # Alternatively, iterate through EarlyStop objects
    for early_stop in early_stopping.early_stops:
        print(f"Sample {early_stop.id} stopped at epoch {early_stop.epoch}")
        print(f"  Reason: {early_stop.reason}")

    # Check per-grouping efficiency
    for grouping, count in diagnostics['stopped_samples_per_grouping'].items():
        print(f"{grouping}: {count} samples stopped early")
```

---

## Best Practices

### 1. Start Conservative

Begin with conservative thresholds and relax them if efficiency is too low:

```python
# Conservative defaults (high confidence, lower efficiency)
optstop_params = {
    'delta_item': 0.05,   # Default: tight CI for samples
    'delta_cap': 0.05,    # Default: tight CI for groupings
    'cred_level': 0.97,   # Default: 97% credible intervals
    'conservatism': 5,   # Default: standard conservatism
    'CI_delta': 0.00001,  # Default: strict stabilisation threshold
}

# If efficiency is 0%, try more aggressive:
optstop_params = {
    'delta_item': 0.20,   # Wider CI allowed
    'delta_cap': 0.15,    # Wider grouping CI
    'cred_level': 0.90,   # 90% confidence
    'conservatism': 3,    # Less conservative
}
```

The current defaults are optimised for **safety** - strict stopping criteria and minimum viable inference burden. To stop more aggressively, relax `delta_item`/`delta_cap` (CI width thresholds), increase `CI_delta` (stabilisation slope threshold, where higher is more aggressive), and lower `cred_level` towards 0.9. If you see many divergences or R-hat far from 1.0, increase `tune` (up to 2000). If HDIs lack precision, increase `draws` (up to 4000).

### 2. Test with Shadow Mode First

Before committing to early stopping, run with shadow mode to estimate efficiency:

```python
from inspect_ai._eval.task import task_with

# First run: measure potential savings
shadow_manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    shadow_mode=True
)

# Attach to task
task_with(task, early_stopping=shadow_manager, epochs=10)

logs = eval(task, model="gpt-4")
for log in logs:
    if log.results.early_stopping:
        summary = log.results.early_stopping.metadata.get('shadow_mode_summary', {})
        print(f"Would have stopped at trial: {summary.get('would_have_stopped_at')}")
        print(f"Potential efficiency: {summary.get('potential_efficiency_percent')}%")

# If efficiency looks good, run without shadow mode
production_manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    shadow_mode=False
)

# Attach production manager to task
task_with(task, early_stopping=production_manager, epochs=10)
```

### 3. Choose Appropriate Grouping Granularity

More granular groupings require more data:

```python
# Less granular: faster stopping, less targeted
grouping_columns=['model']  # All tasks share decisions

# More granular: slower stopping, more targeted
grouping_columns=['model', 'task', 'metadata.difficulty']  # Independent per combination
```

**Rule of thumb:** Ensure at least 20-30 samples per unique grouping combination.

### 4. Adjust Reanalysis Interval Based on Scale

```python
# Small evaluations (< 100 samples): frequent checks
reanalysis_interval=5

# Medium evaluations (100-500 samples): balanced
reanalysis_interval=10

# Large evaluations (> 500 samples): less frequent checks
reanalysis_interval=20
```

### 5. Monitor Efficiency Metrics

Track efficiency over multiple runs to tune parameters:

```python
import json
from inspect_ai._eval.task import task_with

# Attach early_stopping to task
task_with(task, early_stopping=manager, epochs=10)

logs = eval(task, model="gpt-4")

# Save diagnostics for analysis
for log in logs:
    if log.results.early_stopping:
        with open('stopping_diagnostics.json', 'w') as f:
            json.dump(log.results.early_stopping.metadata, f, indent=2)

# Track over multiple runs
# Analyze: which groupings stop? At what thresholds? What's the efficiency trend?
```

### 6. Use Ordinal Inference Wisely

For ordinal tasks, choose inference mode based on expected distribution:

```python
# If you expect peaked distributions (most scores at one value)
ordinal_inference='modal'  # Fast

# If you expect diffuse distributions (scores spread across range)
ordinal_inference='entropy'  # Conservative

# If you're unsure (recommended)
ordinal_inference='hybrid'  # Automatic selection
```

### 7. Validate Score Extraction

When using `score_choice`, `score_value_key`, or `score_agg`, verify extraction is correct:

```python
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    score_value_key='healthbench_score',  # For dict-valued Score.value
    score_agg='mean',
)

# Check startup output for: "Dict value key: 'healthbench_score'"
# Check logs for warnings about missing keys or unexpected dict values
# Verify extracted scores are in expected range
```

### 8. Understand CI Coverage Considerations

The credible intervals produced by optstop are well-calibrated when groupings contain **50+ items** with performance in the **0.2-0.8 range**. For smaller samples or near-boundary performance (close to 0% or 100%), CIs may undercover due to hierarchical shrinkage.

**Factors affecting coverage:**
- **Sample size**: Coverage approaches nominal levels (~97%) as grouping size increases toward 50+ items
- **Performance range**: Mid-range performance (0.2-0.8) yields better-calibrated intervals than boundary cases
- **Hierarchical shrinkage**: Small groupings with extreme performance (near 0% or 100%) may exhibit substantial undercoverage

In stress-test conditions (e.g., 18-30 items per grouping with many boundary performers), observed coverage can fall substantially below nominal. Under more typical conditions with adequate sample sizes and mid-range performance, coverage is considerably better. The package automatically detects near-boundary estimates and emits diagnostic warnings recommending that raw proportions be reported alongside model estimates.

**Recommendations:**
- Treat CIs as rough guides rather than formal statistical intervals for small groupings
- Report raw proportions alongside model estimates for transparency
- For formal inference with small samples, consider collecting more data

This limitation affects all optstop inference modes equally, as it is structural to hierarchical Bayesian models.

---

## Common Pitfalls

### 1. Passing early_stopping to eval() Instead of Task (CRITICAL)

**Problem:** Passing `early_stopping` as a parameter to `eval()` instead of attaching it to the Task.

```python
# WRONG - This silently fails! early_stopping is ignored.
manager = OptimalStoppingManager(...)
log = eval(task, model="gpt-4", epochs=10, early_stopping=manager)
```

**Why this fails:** The `eval()` function does **not** have an `early_stopping` parameter. Python's `**kwargs` silently absorbs the parameter, and it's never used. Your evaluation runs without any early stopping, with no error message.

**Solution:** Attach `early_stopping` to the **Task object**:

```python
# CORRECT - Option 1: In Task constructor
task = Task(
    dataset=my_dataset,
    solver=my_solver,
    scorer=my_scorer,
    early_stopping=manager,
    epochs=10
)
logs = eval(task, model="gpt-4")

# CORRECT - Option 2: Using task_with() for registered tasks
from inspect_ai._eval.loader import load_tasks
from inspect_ai._eval.task import task_with

tasks = load_tasks(["inspect_evals/truthfulqa"])
for task in tasks:
    task_with(task, early_stopping=manager, epochs=10)
logs = eval(tasks, model="gpt-4")
```

**How to detect:** If your logs show no `early_stopping` metadata or 0% efficiency with all trials running, check that you're attaching the manager to the Task, not passing it to `eval()`.

### 2. Wrong Routing Due to Missing/Extra Parameters

**Problem:** Adding/removing `score_agg` changes routing completely.

```python
# Wrong: Intended binary discrete, but routes to continuous
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    score_agg='mean',  # Routes to continuous bounded (unintended)
)

# Correct: Binary discrete
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    # NO score_agg parameter
)
```

**Solution:** Carefully review routing logic and verify in logs.

### 3. Expecting 100% Efficiency

**Problem:** Expecting all samples to stop early.

**Reality:** Optimal stopping is conservative. With realistic variance:
- Binary discrete (p=0.6-0.7): 0% efficiency expected
- Binary discrete (p≥0.9): 70-90% efficiency expected
- Ordinal discrete: depends on distribution concentration

**Solution:** Understand that 0% efficiency is valid behavior for moderate-variance data.

### 4. Insufficient Data Per Grouping

**Problem:** Too many grouping columns with too few samples.

```python
# Bad: 100 samples, 50 groupings = 2 samples per grouping
grouping_columns=['model', 'task', 'metadata.difficulty', 'tag.category']
```

**Solution:** See [Best Practices §3](#3-choose-appropriate-grouping-granularity) for guidance on grouping granularity.

### 5. Conflicting Score Parameters

**Problem:** Specifying both `score_choice` and `score_agg`.

```python
# Wrong: These are mutually exclusive
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    score_choice='accuracy',
    score_agg='mean',  # Error: mutually exclusive
)
```

**Solution:** Choose one or neither, never both.

### 6. Dict-Valued Scores Without `score_value_key`

**Problem:** Scorer returns `Score.value` as a dict, but `score_value_key` is not set. Samples are skipped with a warning.

```python
# Wrong: HealthBench returns {"healthbench_score": 0.72, "criteria_met": 5}
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    # Missing: score_value_key parameter
)
```

**Solution:** Set `score_value_key` to the dict key containing the primary numeric value.

```python
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    score_value_key='healthbench_score',
)
```

### 7. Forgetting to Mark Ordinal Tasks

**Problem:** Ordinal scores (1-5) treated as binary, causing validation errors.

```python
# Wrong: Scores are 1-5, but treated as binary (expects 0 or 1)
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    # Missing: ordinal_tasks parameter
)

# Correct: Mark as ordinal
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    ordinal_tasks=['rating'],
    ordinal_max_score=5,
)
```

### 7. Not Checking Logs for Validation Warnings

**Problem:** Invalid scores silently skipped, inference never runs.

**Solution:** Always check logs for warnings:
```
WARNING: Invalid score for sample_id=123, epoch=2: Binary task has score > 1 (5.0)
```

### 8. Using Shadow Mode in Production

**Problem:** Forgetting to disable shadow mode, running all trials.

```python
# Wrong: Shadow mode still enabled
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    shadow_mode=True,  # All trials run, no efficiency gain
)

# Correct: Disable for production
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    shadow_mode=False,  # or omit (False is default)
)
```

---

## Examples

### Example 1: Basic Binary Evaluation

```python
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, system_message
from optstop.early_stopping import OptimalStoppingManager

# Example questions dataset (replace with your own data)
questions = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("Who wrote Romeo and Juliet?", "William Shakespeare"),
    # ... add more question-answer pairs
]

# Configure optimal stopping
optstop_params = {
    'delta_item': 0.15,
    'delta_cap': 0.10,
    'cred_level': 0.97,
    'conservatism': 5,
    'draws': 500,
    'tune': 500,
}

manager = OptimalStoppingManager(
    optstop_params=optstop_params,
    grouping_columns=['model', 'task'],
    reanalysis_interval=10,
    min_samples_per_grouping=5
)

# Define task with early_stopping attached
task = Task(
    dataset=[
        Sample(input=q, target=a, id=f"q_{i}")
        for i, (q, a) in enumerate(questions)
    ],
    solver=[
        system_message("You are a helpful assistant."),
        generate()
    ],
    scorer=model_graded_fact(),
    early_stopping=manager,  # Attach to Task
    epochs=10
)

# Run evaluation
logs = eval(task, model="openai/gpt-4")

# Check results
for log in logs:
    if log.results.early_stopping:
        diagnostics = log.results.early_stopping.metadata
        print(f"Efficiency: {diagnostics['efficiency_percent']}%")
        print(f"Trials: {diagnostics['total_ran']} / {diagnostics['total_planned_trials']}")
        print(f"Stopped samples: {diagnostics['stopped_samples_count']}")
```

### Example 2: Multi-Model Comparison with Shadow Mode

```python
from inspect_ai._eval.task import task_with

# First run: measure potential savings
shadow_manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.15,
        'delta_cap': 0.10,
    },
    grouping_columns=['model'],  # Independent per model
    shadow_mode=True,
    reanalysis_interval=10,
)

# Attach to task
task_with(task, early_stopping=shadow_manager, epochs=10)

logs_shadow = eval(task, model=["openai/gpt-4", "anthropic/claude-3"])

for log in logs_shadow:
    if log.results.early_stopping:
        summary = log.results.early_stopping.metadata.get('shadow_mode_summary', {})
        print(f"Would have stopped at trial: {summary.get('would_have_stopped_at')}")
        print(f"Potential efficiency: {summary.get('potential_efficiency_percent')}%")

# If good, run without shadow mode (create new task or reload)
production_manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.15,
        'delta_cap': 0.10,
    },
    grouping_columns=['model'],
    shadow_mode=False,
    reanalysis_interval=10,
)

# Reload task and attach production manager
task_with(task, early_stopping=production_manager, epochs=10)

logs_production = eval(task, model=["openai/gpt-4", "anthropic/claude-3"])

for log in logs_production:
    if log.results.early_stopping:
        print(f"Actual efficiency: {log.results.early_stopping.metadata['efficiency_percent']}%")
```

### Example 3: Ordinal Rating with Custom Metadata Grouping

```python
from inspect_ai._eval.task import task_with

# Samples have metadata: {'difficulty': 'easy'/'medium'/'hard'}
# Scorer returns 0-5 ordinal ratings

manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.20,  # More relaxed for ordinal
        'delta_cap': 0.15,
    },
    grouping_columns=['model', 'metadata.difficulty'],
    ordinal_tasks=['rating'],
    ordinal_max_score=5,  # 0-5 scale
    ordinal_inference='hybrid',
    reanalysis_interval=8,
    min_samples_per_grouping=10,  # More data for ordinal
)

# Attach to task
task_with(task, early_stopping=manager, epochs=15)

logs = eval(task, model="openai/gpt-4")

# Analyze per difficulty level
for log in logs:
    if log.results.early_stopping:
        diagnostics = log.results.early_stopping.metadata
        for grouping, count in diagnostics['stopped_samples_per_grouping'].items():
            print(f"{grouping}: {count} samples stopped")
```

### Example 4: Multiple Scorers with Aggregation

```python
from inspect_ai.scorer import accuracy, f1, recall

manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.15,
        'delta_cap': 0.10,
    },
    grouping_columns=['model', 'task'],
    score_agg='mean',  # Average all three scores → continuous
    reanalysis_interval=10,
)

# Attach early_stopping and epochs to Task
task = Task(
    dataset=samples,
    solver=solver,
    scorer=[accuracy(), f1(), recall()],  # Multiple scorers
    early_stopping=manager,
    epochs=10
)

logs = eval(task, model="gpt-4")

# Each trial's score = mean([accuracy, f1, recall])
# Routes to continuous bounded inference
```

### Example 5: Aggressive Efficiency Settings

```python
from inspect_ai._eval.task import task_with

# For scenarios where you want maximum efficiency and can tolerate less confidence

aggressive_params = {
    'delta_item': 0.25,        # Very wide CI allowed
    'delta_cap': 0.20,         # Wide grouping CI
    'cred_level': 0.85,        # 85% confidence (less conservative)
    'conservatism': 2,         # Low conservatism
}

manager = OptimalStoppingManager(
    optstop_params=aggressive_params,
    grouping_columns=['model'],
    reanalysis_interval=5,     # Check frequently
    min_samples_per_grouping=3, # Start early
)

# Attach to task
task_with(task, early_stopping=manager, epochs=10)

logs = eval(task, model="gpt-4")

# Expected: Higher efficiency, lower confidence
```

---

## Troubleshooting

### Issue: 0% Efficiency

**Symptoms:** All trials run to completion, no early stopping.

**Possible causes:**
1. Data variance too high for conservative thresholds
2. Reanalysis interval too large (missing stopping opportunities)
3. Min samples threshold not reached
4. Routing to wrong inference type

**Debugging steps:**
```python
# 1. Check logs for inference execution
# Look for: "INFO: Running optimal stopping inference"

# 2. Try more aggressive thresholds
optstop_params = {
    'delta_item': 0.25,  # Increase from 0.15
    'delta_cap': 0.20,   # Increase from 0.10
}

# 3. Reduce reanalysis interval
reanalysis_interval=5  # Down from 10

# 4. Verify routing in logs
# Binary: "Processing grouping '...' as BINARY"
# Ordinal: "Processing grouping '...' as ORDINAL"
# Continuous: "Processing grouping '...' as CONTINUOUS"

# 5. Check convergence projections for non-stopped groupings
for grouping, entry in diagnostics['stabilization_histories'].items():
    if 'convergence_projection' in entry:
        proj = entry['convergence_projection']
        print(f"{grouping}: proximity={proj['proximity_ratio']:.2f}, "
              f"est. trials needed={proj['projected_additional_trials']}, "
              f"confidence={proj['uncertainty']['confidence_level']}, "
              f"target={proj['convergence_target']}")
```

A `convergence_target` of `'projected_slope_stabilisation'` suggests the grouping would plateau above `delta_cap` even with unlimited data - consider widening thresholds.

### Issue: Invalid Score Warnings

**Symptoms:** Warnings in logs about invalid scores.

```
WARNING: Invalid score for sample_id=123: Binary task has score > 1 (5.0)
```

**Possible causes:**
1. Wrong inference routing (binary expecting 0/1, getting ordinal 1-5)
2. Missing `ordinal_tasks` parameter
3. Missing `score_agg` parameter for continuous

**Solution:**
```python
# If scores are ordinal (1-5), mark as such:
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    ordinal_tasks=['task_name'],  # Add this
    ordinal_max_score=5,          # And this
)

# If scores are aggregated floats, add score_agg:
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    score_agg='mean',  # Add this
)
```

### Issue: Inference Takes Too Long

**Symptoms:** Long pauses during evaluation.

**Possible causes:**
1. Too many MCMC samples (`draws`, `tune`)
2. Frequent reanalysis interval
3. CPU-only inference (no GPU)

**Solution:** See [Performance Considerations](#performance-considerations) for detailed guidance. Quick fixes:
- Reduce `draws`/`tune` to 1000-2000
- Increase `reanalysis_interval`
- Enable GPU with `gpu_ids=[0]` (2-4× speedup for ordinal inference)

---

## Version History

### v0.3.1 (Current)
- **Continuous score type support**: Full hierarchical Beta model for `continuous_01` and `continuous_bounded` score types in both posthoc and live modes
- **CI extraction methodology fix**: Changed from using HDI on individual item Theta values to using `mean(Theta)` / `mean(mu_item)` across all items, which correctly accounts for between-item variance (sigma_group)
- **Diagnostic logging**: Added diagnostic logging for posterior analysis debugging (mu_group, sigma_group statistics)
- **Hierarchical continuous model**: New model with group-level parameters (mu_group, sigma_group, phi_group) and item-level means
- **Convergence projection**: For non-stopped groupings, `complete_task()` diagnostics include forward-looking convergence projections using exponential decay model (primary) with linear extrapolation fallback, including bootstrap uncertainty quantification

### v0.3.0
- `ordinal_model_type` parameter for model selection ('ordered_logistic' or 'dirichlet')
- Ordered Logistic (cumulative link) model as default for ordinal inference
- ~3× faster ordinal inference vs Dirichlet-Multinomial
- Automatic fallback to Dirichlet if Ordered Logistic sampling fails
- Adaptive cutpoint priors scaling with number of categories
- Full parameter recovery validation for Ordered Logistic

### v0.2.1
- `random_seed` parameter for MCMC reproducibility
- `entropy_convergence_threshold` parameter for ordinal tuning (replaces `entropy_stabilization_threshold`)
- Numpyro/JAX integration for ordinal inference (~2× CPU speedup)
- Seed logging and diagnostics tracking
- Removed redundant np.random.seed() calls (cleaner seed propagation)

### v0.2.0
- Complete Phase 1 testing (15/15 tests passing)
- Binary discrete, ordinal discrete, continuous bounded validated
- Multi-grouping independence confirmed
- Shadow mode working
- Process cleanup verified

### Future Releases
- Advanced scenarios (mixed groupings, edge cases)
- Integration testing with real inspect_ai workflows
- GPU configuration optimization for inspect_ai
- Convergence warning system
- Parameter tuning utilities

---

## Related Documentation

- **Main README:** `README.md`
- **Package Documentation:** Core optstop functions (`optimal_stopping_posthoc`, `optimal_stopping_live`, `convergence_posthoc`)

---

## Support and Feedback

For issues, questions, or feedback:
- **inspect_ai Documentation:** https://inspect.aisi.org.uk/

---

**Last Updated:** 2026-02-11
**Document Version:** 1.6
**Phase:** Beta (Phase 1-3 Complete)

**v1.5 Changes:** Added v0.3.1 version history entry documenting continuous score type support, CI extraction methodology fix (mean(Theta) for correct between-item variance handling), and diagnostic logging additions.

**v1.4 Changes:** Critical fix - All examples updated to use correct API pattern. The `early_stopping` parameter must be attached to Task objects (via constructor or `task_with()`), NOT passed to `eval()`. Added prominent warning and new Common Pitfall #1.
