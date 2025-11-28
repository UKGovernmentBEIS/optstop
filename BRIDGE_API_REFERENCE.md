# OptimalStoppingManager API Reference

**Version:** 0.2.0+
**Last Updated:** 2025-11-28
**Status:** Production Ready (Phase 1 Complete)
**Performance:** Nutpie integration available (2-5× CPU speedup)

---

## Table of Contents

1. [Overview](#overview)
2. [Performance Considerations](#performance-considerations) **← NEW**
3. [Class: OptimalStoppingManager](#class-optimalstoppingmanager)
4. [Initialization Parameters](#initialization-parameters)
5. [Routing Logic](#routing-logic)
6. [Configuration Patterns](#configuration-patterns)
7. [Protocol Methods](#protocol-methods)
8. [Diagnostics and Return Values](#diagnostics-and-return-values)
9. [Best Practices](#best-practices)
10. [Common Pitfalls](#common-pitfalls)
11. [Examples](#examples)

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
- You have at least 10-20 samples per grouping

Do not use when:
- Running single-epoch evaluations (no opportunity to stop early)
- You have very few samples (< 5 per grouping)
- You need exact reproducibility of trial counts (stopping decisions are data-dependent)

---

## Performance Considerations

### Critical Performance Factors

The computational cost of early stopping inference varies dramatically based on configuration. Understanding these factors is essential for production deployments.

#### 1. **Inference Pathway Performance**

| Pathway | Typical Time | Complexity |
|---------|--------------|-----------|
| **Binary** | ~26s | MCMC Binomial model |
| **Continuous** | ~9s | MCMC Beta model (aggregated) |
| **Ordinal (modal)** | ~0.1s | Bootstrap (fast!) |
| **Ordinal (entropy)** | ~60+ min | MCMC OrderedLogistic (slow!) |
| **Ordinal (hybrid)** | ~60+ min | BOTH modal + entropy (slowest) |

**Key insight:** Ordinal hybrid mode runs BOTH modal and entropy inference on every call, making it 100-1000× slower than other pathways!

---

### 2. **MCMC Sampling Parameters** ⚠️ CRITICAL

The `draws` and `tune` parameters have **linear impact** on inference time:

```python
# Default (very slow)
optstop_params = {
    'draws': 6000,    # 6000 MCMC samples
    'tune': 6000,     # 6000 tuning steps
}
# Total: 12,000 iterations per inference

# Recommended for production (60× faster!)
optstop_params = {
    'draws': 300,     # 300 MCMC samples
    'tune': 300,      # 300 tuning steps
    'chains': 2,      # 2 chains (down from 4)
}
# Total: 600 iterations, 2 chains = 60× speedup!
```

**Impact on ordinal hybrid:**
- draws=6000, tune=6000: ~60-120 minutes per inference call
- draws=300, tune=300, chains=2: ~3-5 minutes per inference call
- **Speedup: 12-24×**

**Quality trade-off:**
- Fewer iterations → wider credible intervals (more conservative stopping)
- For early stopping decisions, moderate precision is sufficient
- Validate convergence: check R-hat < 1.01, ESS > 400

---

### 3. **Nutpie Integration** 🚀 NEW

OptstOP now automatically detects and uses [nutpie](https://github.com/pymc-devs/nutpie) (Rust-based NUTS sampler) for **2-5× additional speedup** on CPU sampling:

**Installation:**
```bash
pip install nutpie
```

**That's it!** Automatic detection, zero configuration.

**Performance impact:**
- Binary: ~26s → ~5-13s per inference
- Ordinal (draws=300): ~3-5min → ~0.6-2.5min per inference
- Continuous: ~9s → ~2-4.5s per inference

**Combined optimization (draws=300 + nutpie):**
- Total speedup: **120-300× faster than default!**
- Ordinal hybrid: ~60min → ~0.6-2.5min per inference

**Verification:**
Check logs for: `"Configured sampling for CPU: nutpie (Rust-based v0.16.4)"`

**See:** `NUTPIE_INTEGRATION.md` for details.

---

### 4. **Reanalysis Interval vs. Inference Time** 🔴 CRITICAL

**Rule:** `inference_time` must be **less than** time between inference triggers.

**Calculation:**
```
time_between_triggers = reanalysis_interval × trial_duration / parallelism
```

**Example scenario:**
- 100 samples, 3 groupings (~33 per grouping)
- reanalysis_interval = 25
- Trial duration = 5 min
- Parallelism = 5 trials at once
- Time between any grouping hitting threshold: ~25 minutes

**If inference takes 60 minutes:**
- ❌ Queue backs up (60min >> 25min)
- ❌ Stopping decisions arrive too late
- ❌ 0% efficiency (all trials complete before first inference finishes)

**If inference takes 3 minutes:**
- ✅ No bottleneck (3min << 25min)
- ✅ Stopping decisions arrive in time
- ✅ 40-60% efficiency achieved

---

### 5. **Ordinal Inference Mode Selection** ⚠️ CRITICAL

| Mode | Speed | Use Case | Performance |
|------|-------|----------|-------------|
| **modal** | ~0.1s | Peaked distributions (most data in 1-2 categories) | ✅ **RECOMMENDED for production** |
| **entropy** | ~60min | Diffuse distributions (spread across many categories) | ⚠️ Slow, use only when needed |
| **hybrid** | ~60min | Auto-selects modal or entropy | 🔴 **AVOID for large-scale** |

**Why hybrid is slow:**
- Computes BOTH modal (fast) AND entropy (slow) every time
- No early exit optimization (always runs both)
- Designed for maximum safety, not performance

**Recommendation:**
```python
# For production with >50 samples
ordinal_inference='modal'  # Fast, works for 80-90% of cases

# For research/small-scale (<50 samples)
ordinal_inference='hybrid'  # Safe but slow
```

**Trade-off:**
- Modal: May not stop for truly diffuse distributions (stays wide forever)
- Hybrid: Catches all cases but 100-1000× slower
- For most LLM evaluations, modal is sufficient (models are typically consistent or consistently inconsistent)

---

### 6. **Configuration Decision Tree**

```
Are you using ordinal scoring?
│
├─ NO → Use default settings, benefit from nutpie
│        Expected: <1min per inference
│
└─ YES → How many samples?
         │
         ├─ <50 samples
         │  └─ Use: ordinal_inference='hybrid'
         │          draws=300, tune=300, chains=2
         │          Expected: 3-5min per inference
         │
         └─ ≥50 samples
            └─ Use: ordinal_inference='modal'
                    draws=300, tune=300, chains=2
                    Expected: <1min per inference

                    If stopping fails (wide CIs forever):
                    → Distribution is truly diffuse
                    → Consider: Are ordinal scores appropriate?
                    → Or: Accept longer runtimes with hybrid mode
```

---

### 7. **Recommended Production Settings**

#### Small Scale (< 100 samples)
```python
optstop_params = {
    'delta_item': 0.15,
    'delta_cap': 0.10,
    'draws': 500,      # Moderate
    'tune': 500,
    'chains': 2,
}
ordinal_inference='hybrid'  # Can afford hybrid
reanalysis_interval=10
```
**Expected:** ~5-10 min per inference (with nutpie)

#### Large Scale (100-1000 samples)
```python
optstop_params = {
    'delta_item': 0.15,
    'delta_cap': 0.10,
    'draws': 300,      # Reduced for speed
    'tune': 300,
    'chains': 2,
}
ordinal_inference='modal'  # Fast mode only
reanalysis_interval=25     # Less frequent checks
```
**Expected:** ~0.6-2.5 min per inference (with nutpie)

#### Very Large Scale (>1000 samples)
```python
optstop_params = {
    'delta_item': 0.20,      # More aggressive
    'delta_cap': 0.15,
    'draws': 200,            # Minimal
    'tune': 200,
    'chains': 2,
}
ordinal_inference='modal'
reanalysis_interval=50      # Infrequent checks
min_samples_per_grouping=20 # Wait for more data
```
**Expected:** ~0.5-1.5 min per inference (with nutpie)

---

### 8. **Performance Monitoring**

**Check inference times in logs:**
```
INFO - Running optimal stopping inference on 25 completed trials for 'gpt-4-math'
INFO - Inference completed in 2.3 minutes
```

**If inference is too slow:**

1. **Check MCMC parameters:**
   - Reduce draws/tune (300/300 recommended)
   - Reduce chains (2 recommended)

2. **Check ordinal mode:**
   - Switch from hybrid → modal

3. **Install nutpie:**
   ```bash
   pip install nutpie
   ```
   Verify in logs: "Using nutpie sampler"

4. **Increase reanalysis_interval:**
   - More time between inferences
   - Trade-off: May miss early stopping opportunities

5. **Check for bottleneck:**
   - Is `inference_time > time_between_triggers`?
   - If yes: **critical** - queue backs up, stopping fails
   - Solution: Reduce inference time or increase reanalysis_interval

---

### 9. **Common Performance Issues**

| Symptom | Cause | Solution |
|---------|-------|----------|
| All trials complete, 0% efficiency | Inference too slow, arrives after completion | Reduce draws/tune, use modal mode, install nutpie |
| Long pauses during evaluation | High draws/tune, ordinal hybrid | Reduce to draws=300, tune=300, chains=2 |
| "Inference still running" after task complete | Queue backed up | Check inference_time < reanalysis_interval × trial_duration |
| Slow convergence warnings | Insufficient MCMC iterations | Increase draws/tune slightly (500/500) |

---

## Class: OptimalStoppingManager

```python
from optstop.early_stopping import OptimalStoppingManager

manager = OptimalStoppingManager(
    optstop_params: dict[str, Any],
    grouping_columns: list[str],
    score_column: str = "score",
    sample_id_column: str = "sample_id",
    epoch_column: str = "epoch",
    reanalysis_interval: int = 10,
    min_samples_per_grouping: int = 5,
    ordinal_tasks: Optional[list[str]] = None,
    ordinal_max_score: int = 10,
    ordinal_inference: str = 'hybrid',
    gpu_ids: Optional[list[int]] = None,
    max_workers: Optional[int] = None,
    manager_name: str = "optstop",
    shadow_mode: bool = False,
    score_choice: Optional[str] = None,
    score_agg: Optional[str] = None
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
- `cred_level` (float, default: 0.95): Credibility level for confidence intervals (0.95 = 95% CI)
- `conservatism` (int, default: 5): Conservatism factor for rare events (higher = more conservative)
- `low_performance_threshold` (float, default: 0.01): Success rate below which conservative stopping applies

**Advanced parameters (performance-critical):**
- `draws` (int, default: 6000): Number of MCMC samples
  - ⚠️ **Performance impact:** Linear scaling with inference time
  - **Recommended production:** 300-500 (60-120× faster than default!)
- `tune` (int, default: 6000): Number of MCMC tuning steps
  - ⚠️ **Performance impact:** Linear scaling with inference time
  - **Recommended production:** 300-500
- `chains` (int, default: 4): Number of MCMC chains
  - **Recommended production:** 2 (2× faster)
- `cores` (int, default: 4): Number of CPU cores for sampling
  - Usually matches `chains`
- `CI_delta` (float, default: 0.00005): Slope threshold for CI stabilization
- `stab_window` (int, default: 10): Window size for stabilization assessment

**⚠️ CRITICAL:** Default MCMC settings (draws=6000, tune=6000) are designed for publication-quality posteriors. For early stopping decisions, much lower values are sufficient and **drastically faster**. See [Performance Considerations](#performance-considerations) for detailed guidance.

**Example (production-optimized):**
```python
optstop_params = {
    'delta_item': 0.15,      # Allow wider CI for samples (more aggressive stopping)
    'delta_cap': 0.10,       # Require tighter CI for groupings (conservative)
    'cred_level': 0.95,      # 95% confidence intervals
    'conservatism': 5,       # Standard conservatism
    'draws': 300,            # ← 60× faster than default!
    'tune': 300,             # ← Production recommended
    'chains': 2,             # ← 2× faster than default
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

# Stop by model × difficulty level
grouping_columns=['model', 'metadata.difficulty']

# Stop by model × category tag
grouping_columns=['model', 'tag.category']
```

**Important:** More granular groupings = more targeted stopping but require more data per grouping.

---

### Optional Column Mapping Parameters

#### `score_column: str = "score"`
Name of the score column in the internal compiled dataset. Usually keep as default.

#### `sample_id_column: str = "sample_id"`
Name of the sample ID column. Usually keep as default.

#### `epoch_column: str = "epoch"`
Name of the epoch column. Usually keep as default.

---

### Score Extraction Parameters

**CRITICAL:** These parameters determine how the manager extracts scores from inspect_ai's scores dictionary and how it routes to inference algorithms.

#### `score_choice: Optional[str] = None`
Extract a specific score by key name from the scores dictionary.

**Use when:** You have multiple scorers but only want to use one for stopping decisions.

**Example:**
```python
# Sample has scores: {'accuracy': 0.8, 'f1': 0.75, 'recall': 0.9}
score_choice='f1'  # Use only the 'f1' score
```

**Mutually exclusive with** `score_agg`.

#### `score_agg: Optional[str] = None`
Aggregate multiple scores using the specified method.

**Valid values:** `'mean'`, `'median'`, `'mode'`, `'max'`

**Use when:** You have multiple scorers and want to combine them.

**⚠️ CRITICAL ROUTING BEHAVIOR:**
- If `score_agg in ['mean', 'median']`: Routes to **continuous bounded** inference (hierarchical Beta model)
- If `score_agg is None`: Routes to **discrete** inference (binary or ordinal, depending on `ordinal_tasks`)

**Example:**
```python
# Sample has scores: {'s1': 1, 's2': 0, 's3': 1, 's4': 1}
score_agg='mean'  # Mean = 0.75 → continuous bounded inference
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

**⚠️ CRITICAL ROUTING BEHAVIOR:**
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
Maximum value for ordinal scores (e.g., 10 for 0-10 scale, 5 for 1-5 scale).

**Used for:** Score validation and normalization.

#### `ordinal_inference: str = 'hybrid'`
Inference mode for ordinal tasks.

**Valid values:**
- `'modal'`: Fast (~0.1s), bootstrap-based modal category estimation. Best for peaked distributions.
- `'entropy'`: Conservative (~60+ min with defaults), full Bayesian entropy-based stopping. Best for diffuse distributions.
- `'hybrid'` (default): Automatically selects modal or entropy based on distribution characteristics (~60+ min with defaults).

**⚠️ PERFORMANCE WARNING:**
- **Hybrid and entropy modes** run full MCMC OrderedLogistic inference, which is **100-1000× slower** than modal mode!
- With default settings (draws=6000, tune=6000): ~60-120 minutes per inference call
- With optimized settings (draws=300, tune=300, chains=2): ~3-5 minutes per inference call
- **Modal mode** uses bootstrap: ~0.1 seconds per inference call (always fast!)

**Recommendation by use case:**
```python
# Production with >50 samples (RECOMMENDED)
ordinal_inference='modal'  # Fast, works for 80-90% of cases

# Small-scale research (<50 samples)
ordinal_inference='hybrid'  # Safe but slow
# MUST use draws=300, tune=300, chains=2 to avoid hours-long inference!

# Known diffuse distributions only
ordinal_inference='entropy'  # Very slow, use only when modal fails
```

**Trade-offs:**
- **Modal:** May not stop for truly diffuse distributions (wide CIs persist), but fast enough for real-time use
- **Hybrid/Entropy:** Catches all distribution types, but can create inference bottlenecks that prevent stopping decisions from arriving in time

**See:** [Performance Considerations - Ordinal Inference Mode Selection](#5-ordinal-inference-mode-selection--critical) for detailed analysis.

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

#### `max_workers: Optional[int] = None`
Maximum number of parallel workers for grouping inference.

**Default (None):** Auto-configured based on available CPU cores.

**Note:** Each worker runs PyMC inference, which itself uses multiple cores, so this typically stays at 1.

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
- `complete_task()` diagnostics show what would have stopped

---

## Routing Logic

The manager automatically routes to different inference algorithms based on configuration:

```
┌─────────────────────────────────────────────┐
│   Initialization Parameters                  │
└──────────────┬──────────────────────────────┘
               │
               ▼
        Has score_agg?
               │
        ┌──────┴──────┐
        │             │
       YES            NO
        │             │
        ▼             ▼
   Is score_agg    Has ordinal_tasks
   in ['mean',     substring match?
   'median']?          │
        │         ┌────┴────┐
       YES        │         │
        │        YES        NO
        ▼         │         │
   CONTINUOUS     ▼         ▼
   BOUNDED     ORDINAL   BINARY
   (Beta)     (Dirich)  (Binom)
```

### Routing Rules

| Configuration | Inference Type | Model | Score Range |
|--------------|----------------|-------|-------------|
| `score_agg='mean'` or `'median'` | **Continuous Bounded** | Hierarchical Beta | [0, 1] |
| `ordinal_tasks=['...']` match + no aggregation | **Ordinal Discrete** | Dirichlet-Multinomial | [0, max_score] |
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
)
# → Routes to ordinal discrete (Dirichlet-Multinomial)
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

manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.15,
        'delta_cap': 0.10,
        'cred_level': 0.95,
    },
    grouping_columns=['model', 'task'],
    reanalysis_interval=10,
)

# Use with inspect_ai
from inspect_ai import eval, Task

log = eval(
    task,
    model="openai/gpt-4",
    epochs=10,
    early_stopping=manager
)
```

### Pattern 2: Multi-Model Comparison

**Use case:** Comparing multiple models on same task, want independent stopping per model.

```python
manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.15,
        'delta_cap': 0.10,
    },
    grouping_columns=['model'],  # Group by model only
    reanalysis_interval=10,
)

# Each model gets independent stopping decisions
log = eval(
    task,
    model=["openai/gpt-4", "anthropic/claude-3", "google/gemini-pro"],
    epochs=10,
    early_stopping=manager
)
```

### Pattern 3: Ordinal Rating Tasks

**Use case:** Tasks with ordinal scores (e.g., 1-5 star ratings).

```python
manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.15,
        'delta_cap': 0.10,
    },
    grouping_columns=['model', 'task'],
    ordinal_tasks=['rating', 'confidence'],  # Mark ordinal tasks
    ordinal_max_score=5,                     # 1-5 scale
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

### Pattern 6: Shadow Mode for Validation

**Use case:** Want to measure potential efficiency gains without actually stopping.

```python
manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.15,
        'delta_cap': 0.10,
    },
    grouping_columns=['model', 'task'],
    shadow_mode=True,  # Run all trials, track stopping decisions
    reanalysis_interval=10,
)

log = eval(task, model="gpt-4", epochs=10, early_stopping=manager)

# Check potential efficiency gains
print(f"Would have saved: {log.early_stopping.efficiency_percent}%")
print(f"Would have stopped: {log.early_stopping.stopped_samples_count} samples")
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

### `start_task(task, samples, epochs) -> str`

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

### `schedule_sample(id, epoch) -> EarlyStop | None`

Called before each trial to determine if it should run or be stopped.

**Parameters:**
- `id: str | int` - Sample ID
- `epoch: int` - Epoch number

**Returns:**
- `None` - Run this trial
- `EarlyStop` - Skip this trial (stopped early)

**Behavior:**
- Fast DataFrame lookup (< 1ms typical)
- Checks `schedule_status` flag in compiled_dataset
- Uses cache for repeated lookups
- If `shadow_mode=True`, always returns `None`

**User action:** None required (called automatically by inspect_ai)

### `complete_sample(id, epoch, scores) -> None`

Called after each trial completes to update scores and potentially trigger inference.

**Parameters:**
- `id: str | int` - Sample ID
- `epoch: int` - Epoch number
- `scores: dict[str, SampleScore]` - Scores from scorers

**Returns:** None

**Behavior:**
1. Extracts score value using `score_choice` or `score_agg`
2. Validates score (type, range)
3. Updates `compiled_dataset` with score and marks trial as complete
4. Increments per-grouping counter
5. If counter % `reanalysis_interval` == 0:
   - Runs `optimal_stopping_live_single()` in executor
   - Updates `schedule_status` for stopped samples/groupings
   - Logs stopping decisions

**User action:** None required (called automatically by inspect_ai)

### `complete_task() -> dict[str, JsonValue]`

Called once at the end of evaluation to generate final diagnostics.

**Returns:** Dictionary with comprehensive diagnostics (see Diagnostics section)

**Behavior:**
- Shuts down inference executor gracefully
- Calculates efficiency metrics
- Compiles stopped samples information
- Returns comprehensive metadata dictionary

**User action:** Access diagnostics from `log.early_stopping` after evaluation

---

## Diagnostics and Return Values

The `complete_task()` method returns a comprehensive diagnostics dictionary:

### Top-Level Fields

```python
{
    "manager": str,                      # Manager name
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
    "stabilization_histories": dict      # Per-grouping convergence metrics
}
```

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

### Accessing Diagnostics

```python
from inspect_ai import eval

log = eval(task, model="gpt-4", epochs=10, early_stopping=manager)

# Access diagnostics
diagnostics = log.early_stopping

print(f"Efficiency: {diagnostics['efficiency_percent']}%")
print(f"Trials saved: {diagnostics['total_skipped']} / {diagnostics['total_planned_trials']}")

# Iterate through stopped samples
for sample in diagnostics['stopped_samples']:
    print(f"Sample {sample['id']} stopped at epoch {sample['epoch']}")
    print(f"  Reason: {sample['reason']}")
    print(f"  CI width: {sample['metadata']['ci_width']:.4f}")

# Check per-grouping efficiency
for grouping, count in diagnostics['stopped_samples_per_grouping'].items():
    print(f"{grouping}: {count} samples stopped early")
```

---

## Best Practices

### 1. Start Conservative

Begin with conservative thresholds and relax them if efficiency is too low:

```python
# Conservative (high confidence, lower efficiency)
optstop_params = {
    'delta_item': 0.10,   # Tight CI for samples
    'delta_cap': 0.05,    # Very tight CI for groupings
    'cred_level': 0.95,   # 95% confidence
    'conservatism': 5,    # Standard conservatism
}

# If efficiency is 0%, try more aggressive:
optstop_params = {
    'delta_item': 0.20,   # Wider CI allowed
    'delta_cap': 0.15,    # Wider grouping CI
    'cred_level': 0.90,   # 90% confidence
    'conservatism': 3,    # Less conservative
}
```

### 2. Test with Shadow Mode First

Before committing to early stopping, run with shadow mode to estimate efficiency:

```python
# First run: measure potential savings
shadow_manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    shadow_mode=True
)

log = eval(task, model="gpt-4", epochs=10, early_stopping=shadow_manager)
print(f"Potential efficiency: {log.early_stopping.efficiency_percent}%")

# If efficiency looks good, run without shadow mode
production_manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    shadow_mode=False
)
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

log = eval(task, model="gpt-4", epochs=10, early_stopping=manager)

# Save diagnostics for analysis
with open('stopping_diagnostics.json', 'w') as f:
    json.dump(log.early_stopping, f, indent=2)

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

When using `score_choice` or `score_agg`, verify extraction is correct:

```python
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    score_agg='mean',  # Check logs to confirm correct extraction
)

# Check logs for: "Using score extraction mode: aggregation (mean)"
# Verify extracted scores are in expected range
```

---

## Common Pitfalls

### 1. Wrong Routing Due to Missing/Extra Parameters

**Problem:** Adding/removing `score_agg` changes routing completely.

```python
# ❌ WRONG: Intended binary discrete, but routes to continuous
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    score_agg='mean',  # ← Oops! Routes to continuous bounded
)

# ✅ CORRECT: Binary discrete
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    # NO score_agg parameter
)
```

**Solution:** Carefully review routing logic and verify in logs.

### 2. Expecting 100% Efficiency

**Problem:** Expecting all samples to stop early.

**Reality:** Optimal stopping is conservative. With realistic variance:
- Binary discrete (p=0.6-0.7): 0% efficiency expected
- Binary discrete (p≥0.9): 70-90% efficiency expected
- Ordinal discrete: depends on distribution concentration

**Solution:** Understand that 0% efficiency is valid behavior for moderate-variance data.

### 3. Insufficient Data Per Grouping

**Problem:** Too many grouping columns with too few samples.

```python
# ❌ BAD: 100 samples, 50 groupings = 2 samples per grouping
grouping_columns=['model', 'task', 'metadata.difficulty', 'tag.category']
```

**Solution:** Ensure at least 20+ samples per unique grouping combination.

### 4. Conflicting Score Parameters

**Problem:** Specifying both `score_choice` and `score_agg`.

```python
# ❌ WRONG: These are mutually exclusive
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    score_choice='accuracy',
    score_agg='mean',  # ← Error: mutually exclusive
)
```

**Solution:** Choose one or neither, never both.

### 5. Forgetting to Mark Ordinal Tasks

**Problem:** Ordinal scores (1-5) treated as binary, causing validation errors.

```python
# ❌ WRONG: Scores are 1-5, but treated as binary (expects 0 or 1)
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    # Missing: ordinal_tasks parameter
)

# ✅ CORRECT: Mark as ordinal
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    ordinal_tasks=['rating'],
    ordinal_max_score=5,
)
```

### 6. Not Checking Logs for Validation Warnings

**Problem:** Invalid scores silently skipped, inference never runs.

**Solution:** Always check logs for warnings:
```
⚠️ Invalid score for sample_id=123, epoch=2: Binary task has score > 1 (5.0)
```

### 7. Using Shadow Mode in Production

**Problem:** Forgetting to disable shadow mode, running all trials.

```python
# ❌ WRONG: Shadow mode still enabled
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    shadow_mode=True,  # ← All trials run, no efficiency gain
)

# ✅ CORRECT: Disable for production
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

# Define task
task = Task(
    dataset=[
        Sample(input=q, target=a, id=f"q_{i}")
        for i, (q, a) in enumerate(questions)
    ],
    solver=[
        system_message("You are a helpful assistant."),
        generate()
    ],
    scorer=model_graded_fact()
)

# Configure optimal stopping
optstop_params = {
    'delta_item': 0.15,
    'delta_cap': 0.10,
    'cred_level': 0.95,
    'conservatism': 5,
}

manager = OptimalStoppingManager(
    optstop_params=optstop_params,
    grouping_columns=['model', 'task'],
    reanalysis_interval=10,
    min_samples_per_grouping=5
)

# Run evaluation
log = eval(
    task,
    model="openai/gpt-4",
    epochs=10,
    early_stopping=manager
)

# Check results
print(f"Efficiency: {log.early_stopping.efficiency_percent}%")
print(f"Trials: {log.early_stopping.total_ran} / {log.early_stopping.total_planned_trials}")
print(f"Stopped samples: {log.early_stopping.stopped_samples_count}")
```

### Example 2: Multi-Model Comparison with Shadow Mode

```python
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

log_shadow = eval(
    task,
    model=["openai/gpt-4", "anthropic/claude-3"],
    epochs=10,
    early_stopping=shadow_manager
)

print(f"Potential efficiency: {log_shadow.early_stopping.efficiency_percent}%")

# If good, run without shadow mode
production_manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.15,
        'delta_cap': 0.10,
    },
    grouping_columns=['model'],
    shadow_mode=False,
    reanalysis_interval=10,
)

log_production = eval(
    task,
    model=["openai/gpt-4", "anthropic/claude-3"],
    epochs=10,
    early_stopping=production_manager
)

print(f"Actual efficiency: {log_production.early_stopping.efficiency_percent}%")
```

### Example 3: Ordinal Rating with Custom Metadata Grouping

```python
# Samples have metadata: {'difficulty': 'easy'/'medium'/'hard'}
# Scorer returns 1-5 star ratings

manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.20,  # More relaxed for ordinal
        'delta_cap': 0.15,
    },
    grouping_columns=['model', 'metadata.difficulty'],
    ordinal_tasks=['rating'],
    ordinal_max_score=5,
    ordinal_inference='hybrid',
    reanalysis_interval=8,
    min_samples_per_grouping=10,  # More data for ordinal
)

log = eval(
    task,
    model="openai/gpt-4",
    epochs=15,
    early_stopping=manager
)

# Analyze per difficulty level
for grouping, count in log.early_stopping.stopped_samples_per_grouping.items():
    print(f"{grouping}: {count} samples stopped")
```

### Example 4: Multiple Scorers with Aggregation

```python
from inspect_ai.scorer import accuracy, f1, recall

task = Task(
    dataset=samples,
    solver=solver,
    scorer=[accuracy(), f1(), recall()]  # Multiple scorers
)

manager = OptimalStoppingManager(
    optstop_params={
        'delta_item': 0.15,
        'delta_cap': 0.10,
    },
    grouping_columns=['model', 'task'],
    score_agg='mean',  # Average all three scores → continuous
    reanalysis_interval=10,
)

log = eval(task, model="gpt-4", epochs=10, early_stopping=manager)

# Each trial's score = mean([accuracy, f1, recall])
# Routes to continuous bounded inference
```

### Example 5: Aggressive Efficiency Settings

```python
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

log = eval(task, model="gpt-4", epochs=10, early_stopping=manager)

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
```

### Issue: Invalid Score Warnings

**Symptoms:** Warnings in logs about invalid scores.

```
⚠️ Invalid score for sample_id=123: Binary task has score > 1 (5.0)
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

**Solution:**
```python
# Reduce MCMC samples (less accurate, faster)
optstop_params = {
    'draws': 3000,  # Down from 6000
    'tune': 3000,   # Down from 6000
}

# Reduce reanalysis frequency
reanalysis_interval=15  # Up from 10

# Enable GPU if available
gpu_ids=[0]
```

---

## Version History

### v0.2.0 (Current)
- ✅ Complete Phase 1 testing (15/15 tests passing)
- ✅ Binary discrete, ordinal discrete, continuous bounded validated
- ✅ Multi-grouping independence confirmed
- ✅ Shadow mode working
- ✅ Process cleanup verified

### Future Releases
- Advanced scenarios (mixed groupings, edge cases)
- Integration testing with real inspect_ai workflows
- GPU configuration optimization for inspect_ai
- Convergence warning system
- Parameter tuning utilities

---

## Related Documentation

- **User Guide:** `BRIDGE_USAGE_GUIDE.md` (coming soon)
- **Testing Roadmap:** `BRIDGE_TESTING_AND_DEVELOPMENT_ROADMAP.md`
- **Testing Summary:** `BRIDGE_TESTING_SUMMARY.md`
- **Main README:** `README.md`
- **Package Documentation:** Core optstop functions (`optimal_stopping_posthoc`, `optimal_stopping_live`, `convergence_posthoc`)

---

## Support and Feedback

For issues, questions, or feedback:
- **GitHub Issues:** https://github.com/anthropics/optstop/issues
- **Documentation:** https://github.com/anthropics/optstop
- **inspect_ai Documentation:** https://inspect.aisi.org.uk/

---

**Last Updated:** 2025-11-25
**Document Version:** 1.0
**Phase:** Production Ready (Phase 1 Complete)
