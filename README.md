# optstop

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Adaptive Optimal Stopping Rule Algorithms for Efficient Data Collection and Analysis

## Overview

`optstop` is a Python package implementing adaptive optimal stopping algorithms for both post-hoc (batch) and live (incremental) data collection scenarios. It is designed to help researchers and practitioners efficiently determine when enough data has been collected to make reliable inferences, saving time and resources.

## Features
- Post-hoc (batch) optimal stopping for retrospective analysis and dataset pruning
- Live (incremental) optimal stopping for real-time data collection
- Flexible, parameterized stopping criteria
- Flexible column mapping for groupings, sample IDs, and epochs
- Bayesian and frequentist hybrid methodology
- GPU acceleration support via JAX/numpyro for significantly faster PyMC sampling
- Ordinal scoring support for ordinal data (e.g., 0-10), bounded continuous support (e.g., for bounded aggregates), in addition to binary (0/1) scoring
- Convergence projection: estimates additional trials needed when evaluation ends before convergence
- inspect_ai integration for LLM evaluation workflows with adaptive early stopping

## Using optstop with inspect_ai

`optstop` now provides seamless integration with [inspect_ai](https://inspect.aisi.org.uk/), the UK AI Safety Institute's framework for LLM evaluations. The `OptimalStoppingManager` implements the `EarlyStopping` protocol, enabling **statistically-rigorous adaptive early stopping** for your LLM evaluations.

For those using this package in conjunction with inspect_ai, please see [`BRIDGE_API_REFERENCE.md`](BRIDGE_API_REFERENCE.md) for more relevant documentation.

### Why Use Optimal Stopping for LLM Evaluations?

Running LLM evaluations can be expensive and time-consuming:
- **API costs**: Each evaluation sample costs money (especially for large models)
- **Compute time**: Running multiple epochs per sample takes hours or days
- **Resource waste**: Running unnecessary trials after statistical confidence is achieved

Optimal stopping helps you **stop evaluating intelligently** when you have enough data, potentially saving substantial API costs and evaluation time, whilst maintaining statistical validity and reliability.

### Quick Start

#### Installation

```bash
# Install optstop with inspect_ai support
pip install optstop[inspect]

# Or with GPU acceleration
pip install optstop[inspect,gpu]
```

#### Basic Example

```python
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, system_message
from optstop.early_stopping import OptimalStoppingManager

# Configure optimal stopping
optstop_params = {
    'delta_item': 0.05,       # CI width threshold for individual samples
    'delta_cap': 0.05,        # CI width threshold for overall task
    'cred_level': 0.97,       # 97% credible intervals
    'conservatism': 10,        # Conservative for rare events
}

stopping_manager = OptimalStoppingManager(
    optstop_params=optstop_params,
    grouping_columns=['model', 'task'],  # Group by model and task
    reanalysis_interval=10,              # Re-run inference every 10 samples
    min_samples_per_grouping=5           # Minimum before first analysis
)

# Define your task with early_stopping attached
# IMPORTANT: early_stopping must be on Task, NOT passed to eval()
task = Task(
    dataset=[Sample(input=q, target=a, id=f"q_{i}")
             for i, (q, a) in enumerate(questions)],
    solver=[
        system_message("You are a helpful assistant."),
        generate()
    ],
    scorer=model_graded_fact(),
    early_stopping=stopping_manager,  # Attach to Task, NOT eval()
    epochs=10                         # Plan 10 epochs per sample
)

# Run evaluation
logs = eval(task, model="openai/gpt-4")

# Check efficiency gains (eval returns a list)
for log in logs:
    if log.results.early_stopping:
        diagnostics = log.results.early_stopping.metadata
        print(f"Efficiency: {diagnostics['efficiency_percent']}%")
        print(f"Stopped samples: {diagnostics['stopped_samples_count']}")
```

### Key Features

#### 1. Flexible Grouping Strategies

Control how evaluations are grouped for stopping decisions:

```python
# Group by model only (stop when model performance converges)
grouping_columns=['model']

# Group by model and task (separate stopping per model-task combination)
grouping_columns=['model', 'task']

# Group by custom metadata (e.g., difficulty level)
grouping_columns=['model', 'metadata.difficulty']

# Group by tags
grouping_columns=['model', 'tag.category']
```

#### 2. Score Extraction

inspect_ai delivers scores as a two-level structure: an outer dict keyed by **scorer name**, where each entry contains a `Score` object whose `.value` is typically a scalar but can also be a **dict** when the scorer returns structured results.

```python
# scores = {
#     "accuracy": SampleScore(score=Score(value=1.0)),          # scalar value
#     "healthbench": SampleScore(score=Score(value={"healthbench_score": 0.72, "criteria_met": 5}))
# }                                                               # dict value
```

`score_choice` selects which **scorer** to use (outer dict key). `score_value_key` extracts a specific field from a **dict-valued** `Score.value` (inner dict key). They address different levels and can be combined.

```python
# Multiple scorers with scalar values: pick one by scorer name
OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model'],
    score_choice='accuracy'  # Selects scores["accuracy"]
)

# Multiple scorers with scalar values: aggregate all
OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model'],
    score_agg='mean'  # Averages all scorers (supports: mean, median, mode, max)
)

# Scorer returns a dict as Score.value (e.g., HealthBench):
# extract the numeric field by key
OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    score_value_key='healthbench_score',  # Extracts value["healthbench_score"]
    score_agg='mean',
)
```

#### 3. Ordinal Scoring Support

For tasks with ordinal ratings (e.g., 0-5 scale, 0-10 scale):

```python
OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    ordinal_tasks=['confidence', 'rating'],  # Tasks with ordinal scores
    ordinal_max_score=10,                    # 0-10 scale (scores: 0,1,2,...,10)
    ordinal_inference='hybrid'               # Recommended: handles all distribution shapes
)
```

**Note:** Ordinal scores are expected to be **0-indexed** (range [0, ordinal_max_score]). If your scorer produces 1-indexed scores (e.g., 1-5 star ratings), transform them to 0-indexed before use, or contact the developer.

**Performance Note for Ordinal Discrete Tasks:**
Ordinal discrete inference (without score aggregation) uses entropy-based Bayesian models that are more computationally intensive than binary or continuous pathways (~2-3 minutes per inference call vs ~10-30 seconds, for typical 5-11 category scales with ~100 items). However, inference runs in a background thread that overlaps with LLM processing, so in practice ordinal inference adds no observable delay to overall evaluation time (in comparative testing, ordinal and continuous evaluations completed in virtually identical wall time). The more important difference is **convergence behaviour**: ordinal's entropy validation gate typically requires more data to converge, resulting in lower efficiency than binary or continuous pathways at the same precision threshold. GPU acceleration (2-4x speedup) may help for evaluations with very fast-completing trials or large ordinal scales (20+ categories). See [GPU Acceleration](#gpu-acceleration) for setup details.

```python
# Enable GPU for ordinal evaluations
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    ordinal_tasks=['rating'],
    ordinal_inference='hybrid',
    gpu_ids=[0]  # 2-4× MCMC speedup
)
```

#### 4. Shadow Mode

Compare performance with and without early stopping:

```python
# Run with shadow mode (runs all trials but tracks what *would* stop)
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model'],
    shadow_mode=True  # Disables actual stopping
)

# After evaluation, check what would have stopped
logs = eval(task, model="gpt-4")
for log in logs:
    if log.results.early_stopping:
        summary = log.results.early_stopping.metadata.get('shadow_mode_summary', {})
        print(f"Would have stopped at trial: {summary.get('would_have_stopped_at')}")
        print(f"Potential savings: {summary.get('potential_efficiency_percent')}%")
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optstop_params` | Required | Dictionary of stopping parameters (delta_item, delta_cap, etc.) |
| `grouping_columns` | Required | List of columns for grouping decisions |
| `reanalysis_interval` | 10 | Run inference every N completed samples |
| `min_samples_per_grouping` | 5 | Minimum samples before first analysis |
| `ordinal_tasks` | None | List of task names using ordinal scoring |
| `ordinal_max_score` | 10 | Maximum score for ordinal tasks |
| `ordinal_inference` | See note | Ordinal inference mode: 'modal', 'entropy', 'hybrid' |
| `ordinal_model_type` | 'ordered_logistic' | Hierarchical model: 'ordered_logistic' or 'dirichlet' |
| `gpu_ids` | None | List of GPU IDs to use (e.g., [0, 1]) |
| `entropy_threshold` | 0.8 | Proportion of max entropy for false peak detection in hybrid mode |
| `shadow_mode` | False | If True, run all trials but track stopping decisions |
| `score_choice` | None | Select scorer by name from the outer scores dict |
| `score_value_key` | None | Extract a field from dict-valued `Score.value` (inner dict key) |
| `score_agg` | None | Aggregate scores: 'mean', 'median', 'mode', 'max' |
| `random_seed` | None | Random seed for reproducibility (auto-generates if not specified) |
| `prior_mu` | 0.0 | Centre of group-level Normal prior on logit scale (0.0 = 50% probability) |
| `prior_sigma` | None | Scale of group-level Normal prior. If None, uses pathway-specific defaults (binary/continuous: 1.5, ordinal: 2.0) |

**Using informed priors:** If you have recent results for the same model on the same benchmark, setting `prior_mu` to the logit of that known performance (e.g., `scipy.special.logit(0.75)` ≈ 1.1 for 75% accuracy) improves the accuracy of early point estimates and credible interval placement. Because credible intervals are computed on the probability scale via the logistic transform, a well-placed posterior also produces narrower probability-scale intervals than one centred near 0.5 at the same level of precision - which may modestly accelerate stopping for groupings whose true performance is far from 50%. When `prior_sigma` is not specified (None), pathway-specific defaults apply: 1.5 for binary/continuous, 2.0 for ordinal. If you explicitly set `prior_sigma`, that value applies to all pathways. The default values ensure that a mis-specified prior is overridden by data within approximately 15-20 items.

**Note on `ordinal_inference` defaults:**
- **inspect_ai bridge (`OptimalStoppingManager`)**: Defaults to `'hybrid'` — prioritizes safety in automated evaluation contexts where stopping decisions have real cost implications.
- **Standalone functions (`optimal_stopping_posthoc`, `optimal_stopping_live`, CLI)**: Defaults to `'modal'` — prioritizes speed for interactive/exploratory analysis where users can iterate quickly.

Both modes are valid; choose based on your use case. Use `'hybrid'` when accuracy is critical, `'modal'` when speed matters more.

**Additional `optstop_params` options:**
| Key | Default | Description |
|-----|---------|-------------|
| `entropy_convergence_threshold` | 0.10 | Absolute entropy CI width threshold on [0,1] scale for ordinal Pathway 2 convergence |

### Diagnostics and Efficiency Metrics

After evaluation, `complete_task()` returns comprehensive diagnostics:

```python
# Assuming task was created with early_stopping=manager attached
logs = eval(task, model="gpt-4")

# eval() returns a list of logs
for log in logs:
    if log.results.early_stopping:
        diagnostics = log.results.early_stopping.metadata

        print(f"Total planned trials: {diagnostics['total_planned_trials']}")
        print(f"Trials run: {diagnostics['total_ran']}")
        print(f"Trials skipped: {diagnostics['total_skipped']}")
        print(f"Efficiency: {diagnostics['efficiency_percent']}%")

        # Per-grouping breakdown
        for grouping, metrics in diagnostics['decision_counters'].items():
            print(f"{grouping}: {metrics['completed_samples']} samples completed")

        # Stopped samples with reasons
        for sample in diagnostics['stopped_samples']:
            print(f"Sample {sample['id']}: {sample['reason']}")

        # Convergence projections for non-stopped groupings
        for grouping, history in diagnostics['stabilization_histories'].items():
            if 'convergence_projection' in history:
                proj = history['convergence_projection']
                print(f"{grouping}: ~{proj['projected_additional_trials']} more trials needed")
                print(f"  Basis: {proj['projection_basis']}")
                print(f"  Target: {proj['convergence_target']}")
                print(f"  80% CI: {proj['uncertainty']['ci_trials_80']} trials")
```

#### Convergence Projection

For groupings that have not converged when evaluation ends, optstop estimates how many additional trials would be needed. The projection uses an exponential decay model as the primary approach (fitting `w(t) = a*exp(-b*t) + c` to the CI width trajectory), with linear extrapolation as a fallback when the exponential fit is unavailable or poor.

The projection classifies each non-converged grouping into one of three outcomes via `convergence_target`:

- **`projected_width`**: CI width is projected to drop below `delta_cap` (the group-level CI width threshold, default 0.05) - the grouping would converge with more data. Increase your sample budget or epochs accordingly.
- **`projected_slope_stabilisation`**: CI width is projected to plateau above `delta_cap` - additional data yields diminishing returns. Widen `delta_cap` to accept the current precision, or investigate whether the grouping has high intrinsic variance.
- **`projected_capped`**: Neither outcome detected within the projection horizon. Check whether the grouping has very few observations (< 5 group-level checks) - more data may clarify the trajectory. If observations are plentiful but no clear trend emerges, the data may be too noisy for the current stopping criteria.

In the bridge pathway, each projection includes uncertainty quantification via residual bootstrap (80% and 50% confidence intervals) and a `confidence_level` (`'high'`, `'moderate'`, or `'low'`). Posthoc shortfall estimates are point estimates only (bootstrap disabled for speed, since results are aggregated across randomised orderings). Groupings with insufficient CI width history for projection will not have a `convergence_projection` entry.

See the Convergence Projection Fields section in `BRIDGE_API_REFERENCE.md` for the full field reference (applicable to all modes, not just inspect_ai integration).

### Best Practices

1. **Start conservative**: Use `delta_item=0.05`, `delta_cap=0.05` for high precision
2. **Test with shadow mode**: Run once with `shadow_mode=True` to see potential savings
3. **Set appropriate groupings**: More granular groupings = more targeted stopping
4. **Monitor efficiency**: If efficiency is 0%, your stopping criteria may be too strict
5. **Check compatibility**: Verify inspect_ai version with `optstop.check_inspect_ai_compatibility("0.3.5")`
6. **Sample ID Randomisation**: Ensure evals are constructed such that order of sample IDs is randomised (i.e., avoid systematic influences of sample ID order on potential performance)

### Compatibility

- **optstop version**: 0.3.1
- **inspect_ai version**: 0.3.0+
- **Python version**: 3.10+

Check compatibility programmatically:

```python
import optstop

print(f"optstop version: {optstop.__version__}")
print(f"Minimum inspect_ai version: {optstop.__min_inspect_ai_version__}")

# Check specific version
if optstop.check_inspect_ai_compatibility("0.3.5"):
    print("Compatible!")
```

### Documentation

- **API reference**: See [`BRIDGE_API_REFERENCE.md`](BRIDGE_API_REFERENCE.md) - Complete parameter documentation and configuration guide

### Troubleshooting

**Issue**: `ImportError: No module named 'inspect_ai'`
- **Solution**: Install with `pip install optstop[inspect]`

**Issue**: Early stopping not triggering
- **Solution**: Check `reanalysis_interval` and `min_samples_per_grouping` settings. Increase epochs if needed.

**Issue**: Invalid score warnings
- **Solution**: Verify scores are in valid range (0-1 for binary, 0-ordinal_max_score for ordinal)

**Issue**: GPU not detected
- **Solution**: Install JAX with GPU support: `pip install optstop[gpu]`

For more issues, see the development roadmap or open a GitHub issue.

## GPU Acceleration

`optstop` now supports GPU acceleration for PyMC sampling operations, providing significant performance improvements for large datasets and complex models.

### Prerequisites
- NVIDIA GPU with CUDA support
- JAX with GPU support installed: `pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`

### Automatic GPU Detection
The package automatically detects GPU availability and configures optimal sampling parameters:
- **GPU Available**: Uses JAX/numpyro sampler with optimized chain/core settings
- **CPU Only**: Falls back to standard PyMC sampling

### Manual GPU Control
You can override automatic detection:

#### Python API
```python
# Force GPU usage:
params = {'use_gpu': True, ...}

# Disable GPU acceleration:
params = {'use_gpu': False, ...}
```

#### CLI
```bash
# Force GPU usage (will fail if GPU unavailable)
optstop-posthoc --csv data.csv --output pruned.csv --force_gpu [other options]

# Disable GPU even if available
optstop-posthoc --csv data.csv --output pruned.csv --disable_gpu [other options]
```

### Performance Benefits
- **2-4x MCMC speedup** for typical workloads
- **Even greater speedups** for large datasets and complex models
- Automatic optimization of chain/core parameters for GPU

### GPU Status Logging
Check your log file for GPU detection and usage information:
```
=== GPU Status Report ===
GPU Available: True
JAX Backend: gpu
GPU Device Count: 1
GPU Devices: ['cuda:0']
PyMC will use GPU acceleration via JAX/numpyro
========================
```

## Ordinal Scoring Support

`optstop` supports both **binary scoring** (0/1) and **ordinal scoring** (e.g., 0-10) for both post-hoc and live optimal stopping modes.

### Overview
- **Binary scoring**: Traditional 0/1 success/failure data (default)
- **Ordinal scoring**: Likert scale data (e.g., confidence ratings, difficulty ratings on 0-10 scale)
- **Continuous bounded scoring**: Pre-aggregated mean scores in [0, 1] range (e.g., average accuracy across sub-items), using a hierarchical Beta model with group-level parameters
- **Mixed datasets**: Seamlessly handle binary, ordinal, and continuous groupings in the same analysis

### Score Type Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ordinal_tasks` | List[str] or None | None | Substrings to identify ordinal groupings (e.g., `['confidence', 'rating']`) |
| `ordinal_max_score` | int | 10 | Maximum score for ordinal data (e.g., 10 for 0-10 scale) |
| `ordinal_inference` | str | 'modal' | Inference method: `'modal'`, `'entropy'`, or `'hybrid'` (recommended) |
| `ordinal_model_type` | str | 'ordered_logistic' | Hierarchical model type: `'ordered_logistic'` or `'dirichlet'` |
| `entropy_threshold` | float | 0.8 | Proportion of max entropy for false peak detection in hybrid mode |
| `continuous_tasks` | List[str] or None | None | Substrings to identify continuous bounded groupings (e.g., `['mean_score', 'avg_rating']`) |

**Unusual scoring configurations?** If your evaluation involves very large rubric scales (e.g., >20 categories), peculiar response distributions (e.g., models can only produce scores of 1, 3, 8, and 10 on a 0-10 scale), or you are uncertain about how to select or balance multiple scorer priorities for applying optimal stopping, please reach out to the package owner (Toby Pilditch on AISI Slack) for guidance on configuring optstop for your use case.

### Model Types

**Ordered Logistic** (Recommended, Default):
- Cumulative link model that respects ordinal structure
- Uses identified cutpoints (first cutpoint fixed at 0 for model identification)
- Adaptive priors that scale with number of categories
- ~3x faster than Dirichlet-Multinomial due to fewer parameters
- Best for truly ordinal data where category ordering matters

**Dirichlet-Multinomial**:
- Treats categories as exchangeable (no ordinal structure enforced)
- More robust to bimodal or U-shaped distributions
- Automatically used as fallback if ordered_logistic sampling fails
- Use when response distributions don't follow ordinal assumptions

### Inference Modes

**Modal Mode** (Fast):
- Bootstrap-based modal category estimation
- Suitable for clearly peaked ordinal distributions
- Typical processing time: ~0.1 seconds per call

**Entropy Mode** (Conservative):
- OrderedLogistic Bayesian model with full distribution entropy
- Suitable for diffuse ordinal distributions
- Typical processing time: ~2-3 minutes per call (5-11 category scales, ~100 items)

**Hybrid Mode** (RECOMMENDED):
- **Pathway 1**: Modal CI narrow + entropy validation (peaked data) → Fast stopping
- **Pathway 2**: Entropy convergence (diffuse data) → Safe stopping
- Prevents false peaks via `entropy_threshold`
- Typical processing time: ~2-3 minutes per call (same as entropy, since hybrid runs both modal and entropy components)
- Inference runs in a background thread, overlapping with LLM processing

All inference modes run in a background thread when used via `OptimalStoppingManager`, overlapping with LLM processing. The timings above reflect inference computation, not evaluation delays. See the Performance Note above for details.

### Example Usage: Post-hoc with Ordinal Data

```python
import pandas as pd
from optstop import optimal_stopping_posthoc

# Mixed binary/ordinal dataset
df = pd.DataFrame({
    'student': [1]*20 + [2]*20,
    'task': ['math']*10 + ['confidence']*10 + ['math']*10 + ['confidence']*10,
    'item_id': list(range(5))*8,
    'trial_num': [1,2]*20,
    'score': [1,0,1,1,0] * 4 + [7,8,9,6,8] * 4  # Binary + ordinal
})

params = {
    'delta_item': 0.15,
    'delta_cap': 0.05,
    'cred_level': 0.97,
    'conservatism': 10,
}

pruned_df, summary = optimal_stopping_posthoc(
    df, params,
    grouping_columns=['student', 'task'],
    sample_id_column='item_id',
    epoch_column='trial_num',
    score_column='score',

    # Ordinal parameters:
    ordinal_tasks=['confidence', 'difficulty', 'rating'],  # Substring matching
    ordinal_max_score=10,                                  # 0-10 scale
    ordinal_inference='hybrid',                            # RECOMMENDED
    ordinal_model_type='ordered_logistic',                 # Cumulative link model (default)
    entropy_threshold=0.8                                  # False peak detection
)
```

### Example Usage: Live Mode with Ordinal Data

```python
import pandas as pd
from optstop import optimal_stopping_live

# Current data snapshot
df = pd.DataFrame({
    'student': [1]*10,
    'task': ['math']*5 + ['confidence']*5,
    'item_id': list(range(5))*2,
    'trial_num': [1]*10,
    'score': [1,0,1,1,0] + [7,8,9,6,8]
})

params = {
    'delta_item': 0.15,
    'delta_cap': 0.05,
    'cred_level': 0.97,
}

result = optimal_stopping_live(
    df, params,
    grouping_columns=['student', 'task'],
    sample_id_column='item_id',
    epoch_column='trial_num',
    score_column='score',

    # Ordinal parameters:
    ordinal_tasks=['confidence'],          # Identify ordinal groupings
    ordinal_max_score=10,                  # 0-10 scale
    ordinal_inference='hybrid',            # RECOMMENDED
    ordinal_model_type='ordered_logistic', # Cumulative link model (default)
    entropy_threshold=0.8                  # False peak detection
)

print(result['stop_sample_ids'])  # Item IDs that reached stopping criteria
print(result['stop_task'])        # Groupings that reached stopping criteria
```

### Example Usage: Continuous Bounded Scoring

For tasks with pre-aggregated mean scores in the [0, 1] range (e.g., average accuracy across multiple sub-items):

```python
import pandas as pd
from optstop import optimal_stopping_posthoc

# Dataset with continuous bounded scores (pre-aggregated means)
df = pd.DataFrame({
    'model': ['gpt-4']*20 + ['claude']*20,
    'task': ['qa_accuracy']*10 + ['summary_quality']*10 + ['qa_accuracy']*10 + ['summary_quality']*10,
    'item_id': list(range(5))*8,
    'epoch': [1,2]*20,
    'score': [0.85, 0.82, 0.91, 0.78, 0.88] * 4 + [0.72, 0.75, 0.68, 0.81, 0.77] * 4  # Mean scores in [0,1]
})

params = {
    'delta_item': 0.10,
    'delta_cap': 0.05,
    'cred_level': 0.97,
}

pruned_df, summary = optimal_stopping_posthoc(
    df, params,
    grouping_columns=['model', 'task'],
    sample_id_column='item_id',
    epoch_column='epoch',
    score_column='score',

    # Continuous bounded parameters:
    continuous_tasks=['accuracy', 'quality'],  # Substrings to identify continuous groupings
)
```

**Choosing between ordinal and continuous pathways:**

Integer-valued scores (Likert scales, rubric scores, quality ratings) can be routed to either the ordinal or continuous bounded pathway. The right choice depends on your data and scoring design:

- **Aggregated scores** (e.g., mean of multiple sub-rubric criteria via `score_agg='mean'`): Use the continuous bounded pathway. Aggregation produces effectively continuous values on a bounded interval, making this a natural fit.
- **Raw discrete scores with well-populated categories** (all categories observed, items per grouping >= 5x the number of categories): The ordinal pathway (`ordinal_tasks`) preserves rank structure without imposing interval-scale assumptions, and is the more principled choice when these conditions hold.
- **Raw discrete scores with sparse categories** (some categories rarely or never observed): The ordinal pathway can produce biased estimates and miscalibrated credible intervals under category sparsity (empirically observed: CI coverage as low as 25% at 97% nominal level for ordered logistic; 70% for Dirichlet-Multinomial). In this situation, consider whether scores can be aggregated (e.g., mean across sub-criteria) to route through the continuous pathway, or whether the evaluation design can be adjusted to ensure better category coverage. Routing raw sparse scores through the continuous pathway via `continuous_tasks` is possible but imposes interval-scale assumptions that may not be warranted for all ordinal data.

The package logs a warning when category sparsity is detected under ordinal inference.

**Binary Model Details:**
The binary inference pathway uses a logit-normal hierarchical model (`mu_group ~ Normal(prior_mu, sigma)` on the logit scale, where sigma defaults to 1.5 when `prior_sigma` is None). A Beta-Binomial alternative was evaluated through controlled simulation under conditions favouring the Beta-Binomial (Beta-Binomial data generating process with κ=10). In the mid-range (0.1-0.9 true performance), the two models are statistically indistinguishable in bias, credible interval coverage, and stopping behaviour. At exact boundaries (0.0 or 1.0), the Beta-Binomial produces ~50% less point-estimate bias but its credible intervals are 12% narrower with no improvement in coverage - both models show reduced boundary coverage (~0.45 vs nominal 0.97) due to fundamental information limitations with sparse binary data. The logit-normal's wider boundary intervals function as implicit conservatism, delaying stopping where estimates are least reliable. The Beta-Binomial also exhibited 10-100× more MCMC sampling divergences at boundaries, indicating worse posterior geometry in the regime where it claims an advantage.

**Continuous Bounded Model Details:**
The continuous bounded pathway uses a hierarchical Beta model with the following structure:
- **mu_group**: Group-level mean (logit scale)
- **sigma_group**: Between-item standard deviation (captures item-level variability)
- **phi_group**: Group-level precision parameter
- **mu_item**: Item-level means (hierarchical, derived from mu_group + z * sigma_group)

This hierarchical structure correctly accounts for between-item variance when computing group-level confidence intervals.

### How It Works

1. **Automatic Detection**: The package uses substring matching to identify score types
   - `ordinal_tasks=['confidence']`: Any grouping containing "confidence" (case-insensitive) uses ordinal scoring
   - `continuous_tasks=['accuracy']`: Any grouping containing "accuracy" (case-insensitive) uses continuous bounded scoring
   - All other groupings use binary scoring (default)

2. **Score Validation**:
   - Ordinal scores are validated to be in range [0, `ordinal_max_score`]
   - Continuous scores are validated to be in range [0, 1]
   - Binary scores are validated to be 0 or 1

3. **Independent Processing**: Each grouping is processed with the correct scoring method (no cross-contamination)

4. **False Peak Detection**: In hybrid mode for ordinal scoring, `entropy_threshold` prevents premature stopping on diffuse data

### CI Extraction Methodology

**Expected Group Accuracy via mean(Theta)**

The package computes group-level confidence intervals using the **expected group accuracy**: `mean(Theta)` across all items, rather than using individual item Theta values or `sigmoid(mu_group)`.

**Why this matters:**
- In hierarchical models, `Theta_i = sigmoid(mu_group + z_i * sigma_group)` for each item
- The expected group accuracy `E[Theta]` is the mean across items, NOT `sigmoid(mu_group)`
- When `sigma_group` is large, `sigmoid(mu_group)` can be very different from `E[Theta]`
  - Example: `mu_group=5.3`, `sigma_group=5.3` gives `sigmoid(mu_group)=0.995` but `E[Theta]=0.83`

**Previous approaches and their limitations:**
- `Theta[0]` (first item only): Arbitrary, depends on data ordering
- `sigmoid(mu_group)`: Measures "typical item" (z=0), not expected accuracy
- `mean(Theta)`: Correctly computes expected group accuracy ✓

This methodology correctly accounts for between-item variance (`sigma_group`) when computing confidence intervals, providing more accurate stopping decisions for hierarchical data structures.

### Score Type Support Status

| Function | Binary | Ordinal | Continuous Bounded | Notes |
|----------|--------|---------|-------------------|-------|
| `optimal_stopping_posthoc` | Yes | Yes | Yes | All inference modes available |
| `optimal_stopping_live` | Yes | Yes | Yes | All inference modes available |
| `convergence_posthoc` | Yes | Yes | Yes | All inference modes available |

### CLI Usage with Ordinal and Continuous Data

```bash
# Post-hoc mode with ordinal scoring
optstop-posthoc --csv data.csv --output pruned.csv \
  --grouping_columns student,task \
  --sample_id_column item_id \
  --epoch_column trial_num \
  --score_column score \
  --ordinal_tasks confidence,rating \
  --ordinal_max_score 10 \
  --ordinal_inference hybrid \
  --entropy_threshold 0.8

# Live mode with continuous bounded scoring
optstop-live --csv current_data.csv \
  --grouping_columns model,task \
  --sample_id_column item_id \
  --epoch_column trial_num \
  --score_column score \
  --continuous_tasks accuracy,quality

# Mixed: ordinal + continuous in same dataset
optstop-posthoc --csv mixed_data.csv --output pruned.csv \
  --grouping_columns model,task \
  --sample_id_column item_id \
  --epoch_column trial_num \
  --score_column score \
  --ordinal_tasks confidence \
  --ordinal_max_score 10 \
  --continuous_tasks accuracy
```

## Installation

### Basic Installation
```bash
pip install .
```

**Note**: Basic installation includes all core functionality. Ordinal inference will work but use slower PyMC default sampling (~2-3× slower than with performance extras).

### Recommended: Performance Extras
For optimal performance, especially with **ordinal scoring**, install with performance extras:
```bash
pip install .[performance]
```

This installs:
- **JAX** (CPU backend): Enables fast ordinal inference (2-3× speedup)
- **numpyro** (JAX-based sampler): Handles OrderedLogistic models efficiently

**Performance impact:**
- **Binary inference**: No difference (uses PyMC default for speed)
- **Continuous inference**: No difference (uses PyMC default for speed)
- **Ordinal inference**: ~2-3× faster with numpyro compared to PyMC default

### Installation with inspect_ai Integration
For using optstop with inspect_ai evaluations:
```bash
pip install .[inspect]

# Or with performance extras (recommended for ordinal tasks):
pip install .[inspect,performance]
```

### Installation with GPU Support
For GPU acceleration (requires NVIDIA GPU with CUDA support):
```bash
pip install .[gpu]
```

See [GPU Acceleration](#gpu-acceleration) for detailed setup instructions, manual JAX installation, and performance benefits.

### Combined Installation
Install multiple extras at once:
```bash
# Recommended: performance + inspect_ai
pip install .[inspect,performance]

# Full GPU setup
pip install .[gpu]

# All features including development tools
pip install .[inspect,performance,dev]
```

### Development Installation
For contributing to optstop:
```bash
pip install .[dev]
```

This includes testing tools (pytest, pytest-asyncio), code formatting (black), linting (flake8), and type checking (mypy).

### Installation Summary

| Installation | Binary | Continuous | Ordinal | Use Case |
|-------------|--------|------------|---------|----------|
| `pip install .` | ✓ Fast (PyMC default) | ✓ Fast (PyMC default) | ⚠ Slower (PyMC default) | Basic usage, no ordinal tasks |
| `pip install .[performance]` | ✓ Fast (PyMC default) | ✓ Fast (PyMC default) | ✓ Fast (numpyro, 2-3× faster) | **Recommended for ordinal tasks** |
| `pip install .[gpu]` | ✓ Fastest (numpyro GPU) | ✓ Fastest (numpyro GPU) | ✓ Fastest (numpyro GPU) | GPU hardware available |

## Minimal Working Example

Here is a minimal example to get started with optimal stopping (standalone usage):

```python
import pandas as pd
from optstop.rule import configure_optstop_logging
from optstop import optimal_stopping_posthoc

# Configure logging
configure_optstop_logging('optstop_example.log')

# Create a minimal DataFrame
# Required columns: specify your own column names for grouping, sample ID, epoch, and score
df = pd.DataFrame({
    'subject': [1, 1, 1, 1],
    'task': [1, 1, 1, 1],
    'item_id': [1, 1, 2, 2],
    'trial_num': [1, 2, 1, 2],
    'score': [1, 0, 1, 1],
})

params = {
    'delta_item': 0.05,
    'delta_cap': 0.05,
    'draws': 1000,
    'tune': 1000,
    'chains': 4,
    'cores': 4,
    'CI_delta': 0.00001,
    'conservatism': 10,
    'random_seed': 42
}

pruned_df, summary = optimal_stopping_posthoc(
    df, params,
    grouping_columns=['subject', 'task'],
    sample_id_column='item_id',
    epoch_column='trial_num',
    score_column='score'
)
print(pruned_df)
print(summary)
```

## Logging

To log all optimal stopping decisions and details to a file (and optionally to console), call `configure_optstop_logging` before running your analysis:

```python
from optstop.rule import configure_optstop_logging

# Log to file only (default - suppresses console output)
configure_optstop_logging()

# Or specify a custom log file
configure_optstop_logging('my_optstop_log.txt', console_output=False)

# To also log to console (shows INFO messages in terminal)
configure_optstop_logging('my_optstop_log.txt', console_output=True)
```

After running, you can open the log file to review all stopping decisions and details. Look for entries such as:
- `Starting live optimal stopping`
- `Stopping sample_id ...` or `Stopping task/grouping ...`
- `Live optimal stopping complete`

The log file will contain all stopping decisions and the reasons for them, making it easy to audit or debug the stopping process.

## Setting Parameters

You can control the behavior of the optimal stopping algorithms by passing a `params` dictionary to either `optimal_stopping_posthoc` or `optimal_stopping_live`. Any omitted parameters will use their default values.

### All Available Parameters

| Parameter                | Default   | Applies to   | Description                                                                 |
|--------------------------|-----------|--------------|-----------------------------------------------------------------------------|
| `grouping_columns`       | required  | Both         | List of column names (or single column) to use for grouping                 |
| `sample_id_column`       | required  | Both         | Column name for sample ID                                                   |
| `epoch_column`           | required  | Both         | Column name for epoch/trial                                                 |
| `score_column`           | 'score'   | Both         | Column name for score (default: 'score')                                    |
| `display_progress`       | True      | Both         | Show a progress bar for groupings (set False to disable)                    |
| `generate_diagnostics`   | False     | Post-hoc     | Generate diagnostic plots comparing full vs pruned datasets                 |
| `diagnostics_prefix`     | "optstop_diagnostics" | Post-hoc | Prefix for diagnostic output files (PNG and CSV)                           |
| `delta_item`             | 0.05      | Both         | Max acceptable CI width for individual items                                |
| `delta_cap`              | 0.05      | Both         | Max acceptable CI width for task/grouping                                   |
| `cred_level`             | 0.97      | Both         | Credibility level for intervals (e.g., 0.97 for 97% CI)                     |
| `conservatism`           | 10        | Both         | Factor for rare event conservatism (higher = more conservative)             |
| `low_performance_threshold` | 0.001  | Both         | Below this success rate, use conservative stopping                          |
| `draws`                  | 1000      | Both         | Number of MCMC samples for PyMC (affects speed/accuracy)                    |
| `tune`                   | 1000      | Both         | Number of tuning steps for PyMC                                             |
| `chains`                 | 4         | Both         | Number of MCMC chains for PyMC                                              |
| `cores`                  | 4         | Both         | Number of CPU cores for PyMC sampling                                       |
| `rep_batch_size`         | 1         | Both         | Number of repetitions to process in each batch                              |
| `pymc_refresh_every`     | 2         | Both         | How often to run the PyMC model (every N items)                             |
| `stab_window`            | 15         | Both         | Window size for assessing CI stabilization                                  |
| `CI_delta`               | 0.00001    | Both         | Slope threshold for determining CI stabilization                            |
| `target_accept`          | 0.90 (CPU) / 0.95 (GPU) | Both | NUTS sampler target acceptance rate (higher = fewer divergences, slower) |
| `use_gpu`                | Auto      | Both         | Enable/disable GPU acceleration (True/False, auto-detected if not set)     |
| `force_gpu`              | False     | Both         | Force GPU usage, fail if unavailable (for CLI --force_gpu)                 |
| `ordinal_tasks`          | None      | All          | List of substrings to identify ordinal groupings (e.g., ['confidence'])   |
| `ordinal_max_score`      | 10        | All          | Maximum score for ordinal data (e.g., 10 for 0-10 scale)                 |
| `ordinal_inference`      | 'modal'   | All          | Inference method: 'modal', 'entropy', or 'hybrid'                         |
| `ordinal_model_type`     | 'ordered_logistic' | All   | Hierarchical model: 'ordered_logistic' (default) or 'dirichlet'         |
| `entropy_threshold`      | 0.8       | All          | Proportion of max entropy for false peak detection in hybrid mode         |
| `entropy_convergence_threshold` | 0.10  | All          | Absolute entropy CI width on [0,1] scale for ordinal Pathway 2 convergence |
| `prior_mu`               | 0.0       | All          | Centre of group-level Normal prior on logit scale (0.0 = 50% probability) |
| `prior_sigma`            | None      | All          | Scale of group-level Normal prior. None→pathway defaults (binary/cont: 1.5, ordinal: 2.0)|
| `continuous_tasks`       | None      | All          | List of substrings to identify continuous bounded [0,1] groupings        |
| `shuffle_items`          | False     | Post-hoc     | Randomize item order within each grouping before processing (recommended to avoid selection bias from sorted input) |
| `shuffle_seed`           | None      | Post-hoc     | Random seed for reproducible shuffling (only used with `shuffle_items=True`) |
| `processing_order`       | 'item_greedy' | Post-hoc | Processing order: `'item_greedy'` (all epochs per item, fast) or `'epoch_interleaved'` (all items per epoch, matches production bridge behaviour) |
| `reanalysis_interval`    | 10        | Post-hoc     | Trials between group model refreshes in `epoch_interleaved` mode (matches bridge default) |

### Example: Setting Parameters

You can set any combination of these in your `params` dict. For example:

```python
params = {
    'delta_item': 0.05,                # Require a narrower CI for stopping items
    'delta_cap': 0.05,                 # Task/grouping CI width threshold
    'cred_level': 0.97,               # Use 97% credible intervals
    'conservatism': 10,                # More conservative for rare events
    'low_performance_threshold': 0.001,# Adjust threshold for low performance
    'draws': 1000,                    # MCMC samples
    'tune': 1000,                     # Tuning steps
    'chains': 4,                      # Number of MCMC chains
    'cores': 4,                       # Number of CPU cores for sampling
    'rep_batch_size': 2,              # (post-hoc only) Process 2 reps at a time
    'pymc_refresh_every': 1,          # (post-hoc only) Run PyMC every item
    'stab_window': 15,                # Stabilization window size
    'CI_delta': 0.00001,              # Stabilization threshold
}
```

**Note:**
- Both live and post-hoc modes support the same core parameters for optimal stopping logic.
- Live mode does not support diagnostic plot generation (`generate_diagnostics`, `diagnostics_prefix`).
- You can set only the parameters you care about; the rest will use defaults.
- By default, a progress bar is shown for groupings. Set `display_progress=False` to disable it in Python, or use `--no_progress` in the CLI.

## Diagnostic Plots (Post-hoc Only)

When using `optimal_stopping_posthoc`, you can generate comprehensive diagnostic plots comparing your full dataset with the pruned dataset. This helps you understand the efficiency and accuracy of the optimal stopping algorithm.

### Enabling Diagnostics

Set `generate_diagnostics=True` in your function call:

```python
pruned_df, summary = optimal_stopping_posthoc(
    df, params,
    grouping_columns=['subject', 'task'],
    sample_id_column='item_id',
    epoch_column='trial_num',
    generate_diagnostics=True,
    diagnostics_prefix='my_analysis_diagnostics'
)
```

### Diagnostic Output

The diagnostic system generates two files:
1. **PNG file** (`{prefix}.png`): A comprehensive 5-panel diagnostic plot
2. **CSV file** (`{prefix}_paired.csv`): Paired data for custom analysis

### Diagnostic Plot Panels

The diagnostic plot includes:

1. **Full vs. Pruned Scatter Plot**: Shows how well the pruned estimates match the full dataset estimates
2. **Bland-Altman Plot**: Displays the agreement between full and pruned estimates
3. **Efficiency Distribution**: Histogram of epochs saved per task
4. **Sample IDs Saved by Grouping**: Bar chart showing percentage of sample IDs saved for each grouping
5. **Confidence Intervals Comparison**: Side-by-side comparison of CIs from full vs. pruned datasets

### Diagnostic Statistics

The system also logs comprehensive statistics including:
- **Accuracy metrics**: Bias, MAE, RMSE, Pearson correlation, ICC(3,1)
- **Efficiency metrics**: Mean epochs saved, percentage of total epochs saved, items saved

### CLI Usage

You can also enable diagnostics via the CLI:

```bash
optstop-posthoc --csv mydata.csv --output pruned.csv --grouping_columns subject,task --sample_id_column item_id --epoch_column trial_num --score_column accuracy --generate_diagnostics --diagnostics_prefix my_diagnostics
```

## Flexible Column Mapping

All main functions now support flexible column mapping. You must specify:
- `grouping_columns`: List of column names (or a single column name) to use for grouping (e.g., `['subject', 'task']` or `'subject'`)
- `sample_id_column`: Name of the column for sample ID (e.g., `'item_id'`)
- `epoch_column`: Name of the column for epoch/trial (e.g., `'trial_num'`)
- `score_column`: Name of the column for score (e.g., `'score'`)

The functions will internally create the necessary numeric columns for grouping, sample ID, and epoch.

### Example Usage (Post-hoc)
```python
import pandas as pd
from optstop.rule import configure_optstop_logging
from optstop import optimal_stopping_posthoc

configure_optstop_logging('my_optstop_log.txt', console_output=False)

df = pd.DataFrame({
    'subject': [1, 1, 2, 2],
    'task': [1, 1, 1, 1],
    'item_id': [1, 1, 2, 2],
    'trial_num': [1, 2, 1, 2],
    'score': [1, 0, 1, 1],
})

params = {
    'delta_item': 0.05,
    'delta_cap': 0.05,
    'draws': 1000,
    'tune': 1000,
    'chains': 4,
    'cores': 4,
    'CI_delta': 0.00001,
    'conservatism': 10,
}

pruned_df, summary = optimal_stopping_posthoc(
    df, params,
    grouping_columns=['subject', 'task'],
    sample_id_column='item_id',
    epoch_column='trial_num',
    score_column='score'
)
print(pruned_df)
print(summary)
```
- If you only have one grouping column, you can pass `'subject'` or `['subject']`.

### Example Usage (Live)
```python
import pandas as pd
from optstop.rule import configure_optstop_logging
from optstop import optimal_stopping_live

configure_optstop_logging('my_optstop_log.txt', console_output=False)

df = pd.DataFrame({
    'subject': [1, 1, 2, 2],
    'task': [1, 1, 1, 1],
    'item_id': [1, 1, 2, 2],
    'trial_num': [1, 2, 1, 2],
    'score': [1, 0, 1, 1],
})

params = {
    'delta_item': 0.05,
    'delta_cap': 0.05,
    'draws': 1000,
    'tune': 1000,
    'chains': 4,
    'cores': 4,
    'CI_delta': 0.00001,
    'conservatism': 10,
}

result = optimal_stopping_live(
    df, params,
    grouping_columns='subject',
    sample_id_column='item_id',
    epoch_column='trial_num',
    score_column='score',
    display_progress=True  # Optional: show progress bar
)
print(result)
# Returns: {'stop_sample_ids': ['1-1_1', '1-1_2'], 'stop_task': ['1-1']}
# - stop_sample_ids: List of "grouping_sample_id" strings for sample IDs that have reached stopping criteria
# - stop_task: List of grouping names that have reached stopping criteria
```

### Live Mode Output Explanation

The `optimal_stopping_live` function returns a dictionary with two key outputs:

#### **stop_sample_ids** (List of strings)
- **Format**: `"grouping_sample_id"` (e.g., `"1-1_2"` for grouping `1-1`, sample ID `2`)
- **Purpose**: Sample IDs that have reached stopping criteria and no longer need additional epochs
- **Usage**: Use these to stop data collection for specific sample IDs within their groupings
- **Example**: `['1-1_1', '1-1_2', '2-1_3']` means sample IDs 1 and 2 in grouping 1-1, and sample ID 3 in grouping 2-1 have reached stopping criteria

#### **stop_task** (List of strings)
- **Format**: Grouping names (e.g., `"1-1"`, `"2-1"`)
- **Purpose**: Groupings that have reached stopping criteria and no longer need additional sample IDs
- **Usage**: Use these to stop data collection for entire groupings
- **Example**: `['1-1', '2-1']` means both groupings 1-1 and 2-1 have reached stopping criteria

#### **Multiple Groupings Support**
- The function processes each unique grouping combination independently
- Each grouping can reach stopping criteria independently of others
- Sample IDs are prefixed with their grouping to avoid confusion across multiple groupings

#### **Empty Results**
- If no stopping criteria are met: `{'stop_sample_ids': [], 'stop_task': []}`
- Continue data collection for all sample IDs and groupings

#### **Practical Usage Example**
```python
result = optimal_stopping_live(df, params, grouping_columns=['subject', 'task'], ...)

# Check for sample IDs to stop
if result['stop_sample_ids']:
    print("Stop collecting epochs for these sample IDs:")
    for sample_id in result['stop_sample_ids']:
        grouping, original_id = sample_id.split('_', 1)
        print(f"  - Sample ID {original_id} in grouping {grouping}")

# Check for groupings to stop
if result['stop_task']:
    print("Stop collecting sample IDs for these groupings:")
    for grouping in result['stop_task']:
        print(f"  - Grouping {grouping}")

# If both lists are empty, continue data collection
if not result['stop_sample_ids'] and not result['stop_task']:
    print("Continue data collection - no stopping criteria met")
```

### Example Usage (Convergence)
```python
import pandas as pd
from optstop.rule import configure_optstop_logging
from optstop import convergence_posthoc

configure_optstop_logging('my_convergence_log.txt', console_output=False)

# Mixed binary/ordinal dataset
df = pd.DataFrame({
    'student': [1]*20 + [2]*20,
    'task': ['math']*10 + ['confidence']*10 + ['math']*10 + ['confidence']*10,
    'item_id': list(range(5))*8,
    'trial_num': [1,2]*20,
    'score': [1,0,1,1,0] * 4 + [7,8,9,6,8] * 4  # Binary + ordinal
})

params = {
    'delta_item': 0.05,
    'delta_cap': 0.05,
    'draws': 1000,
    'tune': 1000,
    'chains': 4,
    'cores': 4,
    'stab_window': 15,
    'CI_delta': 0.00001,
    'rep_batch_size': 1,
    'pymc_refresh_every': 2,
    'item_seqs': 3,      # Randomization sequences
    'epoch_seqs': 3      # Randomization sequences
}

result = convergence_posthoc(
    df, params,
    grouping_columns=['student', 'task'],
    sample_id_column='item_id',
    epoch_column='trial_num',
    score_column='score',

    # Ordinal parameters:
    ordinal_tasks=['confidence'],    # Identify ordinal groupings
    ordinal_max_score=10,            # 0-10 scale
    ordinal_inference='hybrid',      # RECOMMENDED (fastest with optimizations)
    entropy_threshold=0.8            # False peak detection
)
print(result)
```

## Optimal Stopping (Post-hoc & Live)

The package provides functions for adaptive optimal stopping, allowing you to determine when enough data has been collected for reliable inference, either retrospectively (post-hoc) or during live data collection.

### Parallelization (Post-hoc)
- The `optimal_stopping_posthoc` function now parallelizes across groupings (unique combinations of the columns you specify for grouping), using all available CPU cores for efficient processing of large datasets.
- Each grouping-task is processed independently and in parallel, with results aggregated at the end.

### Live Mode (Multiple Groupings)
- The `optimal_stopping_live` function processes each unique grouping combination independently
- **Parallel Processing**: Uses adaptive parallel processing across groupings, similar to post-hoc mode
- Returns sample IDs with grouping prefixes (e.g., `"1-1_2"`) to clearly identify which sample IDs belong to which groupings
- Returns a list of grouping names that have reached stopping criteria, allowing you to stop entire groupings
- Supports multiple groupings in a single function call, with independent stopping decisions per grouping
- **Progress Tracking**: Optional progress bar display (controlled by `display_progress` parameter)

For detailed best practices including parameter settings, grouping strategies, and reproducibility guidance, see [Best Practices & Recommendations](#best-practices--recommendations).

### Data Format Requirements

The `optimal_stopping_posthoc` function (and `optstop-posthoc` CLI) expects a pandas DataFrame (or CSV file) in **long format** - one row per observation. The DataFrame must contain four key columns, which you map to your data via function parameters:

| Required Column Role | Parameter | Description |
|---------------------|-----------|-------------|
| **Grouping** | `grouping_columns` | One or more columns defining independent evaluation groups (e.g., model, task). Stopping decisions are made independently per unique combination. |
| **Sample ID** | `sample_id_column` | Identifies distinct evaluation samples/items within a grouping. Each unique value represents a different test item. |
| **Epoch** | `epoch_column` | Identifies repeated evaluations of the same sample. Must be integer-convertible. Each sample should have sequential epoch values (1, 2, 3, ...). |
| **Score** | `score_column` | The outcome for each observation. Binary (0/1), ordinal integers (e.g., 0-10), or continuous floats depending on the inference pathway. |

**Example header and first rows:**

```
model,      task,        item_id,  epoch,  score
gpt-4o,     truthfulqa,  q_001,    1,      1
gpt-4o,     truthfulqa,  q_001,    2,      1
gpt-4o,     truthfulqa,  q_001,    3,      0
gpt-4o,     truthfulqa,  q_002,    1,      1
gpt-4o,     truthfulqa,  q_002,    2,      1
gpt-4o,     reasoning,   r_001,    1,      7
gpt-4o,     reasoning,   r_001,    2,      8
claude-4,   truthfulqa,  q_001,    1,      1
claude-4,   truthfulqa,  q_001,    2,      1
...
```

This would be called as:

```python
pruned_df, summary = optimal_stopping_posthoc(
    df, params,
    grouping_columns=['model', 'task'],   # 4 groupings: gpt-4o×truthfulqa, gpt-4o×reasoning, etc.
    sample_id_column='item_id',
    epoch_column='epoch',
    score_column='score',
    ordinal_tasks=['reasoning'],          # 'reasoning' groupings use ordinal inference
    ordinal_max_score=10
)
```

**Key requirements:**
- **Long format**: Each row is one observation (one model's response to one item on one epoch). Do not pass wide-format or pivoted data.
- **Complete epoch labelling**: Every observation needs a valid epoch value. If your data has only one epoch per sample, the sample-level stopping criteria will have limited data to work with.
- **Consistent score ranges**: All samples within an ordinal grouping should share the same score scale. Set `ordinal_max_score` to match your rubric maximum (e.g., 10 for a 0-10 scale, 5 for a 1-5 scale).
- **Sample ordering**: For post-hoc analysis, samples are processed in order of their first appearance in the DataFrame. If your data is sorted by some systematic property (e.g., item difficulty), consider shuffling before analysis to satisfy the exchangeability assumption described in the package documentation.
- **Additional columns** are preserved in the output but are not used by the algorithm.

### Example Usage (Post-hoc)
```python
import pandas as pd
from optstop.rule import configure_optstop_logging
from optstop import optimal_stopping_posthoc

configure_optstop_logging('my_optstop_log.txt', console_output=False)

df = pd.DataFrame({
    'subject': [1]*8 + [2]*8,
    'task': [1]*4 + [2]*4 + [1]*4 + [2]*4,
    'item_id': [1,1,2,2,1,1,2,2]*2,
    'trial_num': [1,2,1,2]*4,
    'score': [1,0,1,1,0,1,1,0,1,1,0,1,0,1,1,1],
})

params = {
    'delta_item': 0.05,
    'delta_cap': 0.05,
    'draws': 1000,
    'tune': 1000,
    'CI_delta': 0.00001,
    'conservatism': 10,
}

pruned_df, summary = optimal_stopping_posthoc(
    df, params,
    grouping_columns=['subject', 'task'],
    sample_id_column='item_id',
    epoch_column='trial_num',
    score_column='score'
)
print(pruned_df)
print(summary)
```

### Example Usage (Live)
```python
import pandas as pd
from optstop.rule import configure_optstop_logging
from optstop import optimal_stopping_live

configure_optstop_logging('my_optstop_log.txt', console_output=False)

df = pd.DataFrame({
    'subject': [1]*4,
    'task': [1]*4,
    'item_id': [1,1,2,2],
    'trial_num': [1,2,1,2],
    'score': [1,0,1,1],
})

params = {
    'delta_item': 0.05,
    'delta_cap': 0.05,
    'draws': 1000,
    'tune': 1000,
    'CI_delta': 0.00001,
    'conservatism': 10,
}

result = optimal_stopping_live(
    df, params,
    grouping_columns=['subject', 'task'],
    sample_id_column='item_id',
    epoch_column='trial_num',
    score_column='score'
)
print(result)
```

## Convergence Analysis (Post-hoc)

The package provides a function for post-hoc convergence analysis, allowing you to assess when statistical stability would have been achieved in sequential testing scenarios. This is useful for determining the minimum data required for reliable performance estimates.

### Function: `convergence_posthoc`

- **Purpose:**
  - Runs a post-hoc convergence analysis on a full dataset, parallelizing across groupings (using all available CPU cores).
  - Returns a DataFrame detailing the required numbers of trials at epoch and sample_ID levels, split by the user's desired grouping.
- **Parameters:**
  - Accepts the same `params` dictionary as other functions (see table above), plus:
    - `item_seqs`: Number of randomized item orderings per grouping (default: 20)
    - `epoch_seqs`: Number of randomized epoch orderings per item (default: 20)
- **Returns:**
  - A DataFrame with one row per grouping-task, containing detailed convergence statistics and summary metrics. Key columns include:
    - `mean_needed_items` / `var_needed_items`: Mean and population variance of projected additional items needed across randomised orderings
    - `mean_needed_epochs` / `var_needed_epochs`: Mean and population variance of projected additional epochs needed
    - `theta_ci_width`: Final group-level CI width
    - `percent_items_used`: Fraction of total trials used before convergence
    - `mean_fin_CI_width_item` / `mean_fin_CI_width_epoch`: Mean final CI widths at item and epoch level
  - Shortfall estimates (needed items/epochs) are computed using an exponential decay model (primary, fitting `w(t) = a*exp(-b*t) + c`) with linear extrapolation fallback, projecting item-level convergence against `delta_item`. Projections are capped at 200 additional trials per sequence. Bootstrap uncertainty is disabled for posthoc shortfalls (point estimates only, aggregated across randomised orderings).

### Example Usage
```python
import pandas as pd
from optstop.rule import configure_optstop_logging
from optstop import convergence_posthoc

configure_optstop_logging('my_convergence_log.txt', console_output=False)

# Mixed binary/ordinal dataset
df = pd.DataFrame({
    'subject': [1]*8 + [2]*8,
    'grouping': ['A-math']*4 + ['A-confidence']*4 + ['B-math']*4 + ['B-confidence']*4,
    'task': ['math']*4 + ['confidence']*4 + ['math']*4 + ['confidence']*4,
    'item_id': [1,1,2,2]*4,
    'trial_num': [1,2,1,2]*4,
    'score': [1,0,1,1] + [7,8,9,6] + [0,1,1,0] + [8,9,7,8],  # Binary + ordinal
})

params = {
    'delta_item': 0.05,
    'delta_cap': 0.05,
    'draws': 500,
    'tune': 500,
    'stab_window': 5,
    'CI_delta': 0.0002,
    'rep_batch_size': 1,
    'pymc_refresh_every': 2,
    'item_seqs': 3,      # Recommended: 20 for production
    'epoch_seqs': 3      # Recommended: 20 for production
}

result = convergence_posthoc(
    df, params,
    grouping_columns=['subject', 'task'],
    sample_id_column='item_id',
    epoch_column='trial_num',
    score_column='score',

    # Ordinal parameters:
    ordinal_tasks=['confidence'],    # Identify ordinal groupings
    ordinal_max_score=10,            # 0-10 scale
    ordinal_inference='hybrid',      # RECOMMENDED (fastest with optimizations)
    entropy_threshold=0.8            # False peak detection
)
print(result)
```

The resulting DataFrame contains all the convergence metrics for each grouping-task. The analysis is parallelized for speed.

## Convergence Diagnostics

When you run `convergence_posthoc`, the package now automatically generates two diagnostic figures by default:
- **{prefix}_grouped_needed.png**: Two vertically stacked plots showing mean needed items and mean needed epochs (with std error bars) by grouping (descending order).
- **{prefix}_score_scatter.png**: Scatter plots of mean needed items/epochs vs. performance, as before.

You can control this behavior:
- **Python:** Pass `generate_diagnostics=False` to `convergence_posthoc` to turn off diagnostics, or set `diagnostics_prefix` to change the output file prefix.
- **CLI:** Use `--no_diagnostics` to turn off diagnostics, and `--diagnostics_prefix` to set the output file prefix (default: `convergence_eval`).

### Example (Python)
```python
result = convergence_posthoc(
    df, params,
    grouping_columns=['subject', 'task'],
    sample_id_column='item_id',
    epoch_column='trial_num',
    generate_diagnostics=True,  # default is True
    diagnostics_prefix='my_convergence_eval'
)
```

### Example (CLI)
```bash
optstop-convergence --csv mydata.csv --output convergence_stats.csv --grouping_columns subject,task --sample_id_column item_id --epoch_column trial_num --score_column accuracy --no_diagnostics --diagnostics_prefix my_convergence_eval
```

- By default, diagnostics are ON. Use `--no_diagnostics` to disable.
- The output files will be named `my_convergence_eval_grouped_needed.png` and `my_convergence_eval_score_scatter.png`.

See the CLI help (`optstop-convergence --help`) for all options.

## Best Practices & Recommendations

### Specifying Groupings
- **Required columns:** Your input DataFrame must include the columns you specify for grouping, sample ID, epoch, and score (see Flexible Column Mapping section above).
- **Group labels:** If you want human-readable group labels in your output, include a `grouping` column (e.g., system/model name) in addition to your grouping columns.
- **Parallelization:** Each unique combination of the columns you specify for grouping will be processed in parallel, so ensure these columns are set appropriately for your experimental design.

### Input Ordering and Selection Bias (Post-hoc)
- **Items are processed in dataframe order** (within each grouping). If your input data is sorted (e.g., alphabetically by sample ID), early stopping may select a biased subsample if early-alphabet items happen to have systematically different scores than late-alphabet items.
- **Use `shuffle_items=True`** to randomize item order within each grouping before processing. This is recommended for unbiased post-hoc analysis. In `epoch_interleaved` mode, shuffling controls the order items are processed within each epoch.
- **Use `shuffle_seed`** with `shuffle_items=True` for reproducible shuffling.
- **The package warns automatically** if sorted input is detected and `shuffle_items=False`, but using explicit shuffling is recommended.

### Processing Order (Post-hoc)

The `processing_order` parameter controls how the posthoc function iterates through the data:

- **`'item_greedy'`** (default): Processes all epochs for item 1, then all epochs for item 2, etc. This is fast but produces deep coverage of few items early on. Suitable for exploratory analysis and continuous scoring where each observation carries high information.
- **`'epoch_interleaved'`**: Processes all items for epoch 1, then all items for epoch 2, etc. This matches the production bridge (`OptimalStoppingManager`) processing order, producing broad item coverage early. Recommended for consistency/robustness studies comparing posthoc results against production, particularly for binary scoring where item diversity is critical.

The `reanalysis_interval` parameter (default 10) controls how frequently the group-level model is refreshed in `epoch_interleaved` mode, matching the bridge's default cadence. It is ignored in `item_greedy` mode (which uses `pymc_refresh_every` instead).

```python
# Epoch-interleaved mode (matches production bridge behaviour)
pruned_df, summary = optimal_stopping_posthoc(
    df, params,
    grouping_columns=['model', 'task'],
    sample_id_column='item_id',
    epoch_column='trial_num',
    processing_order='epoch_interleaved',
    reanalysis_interval=10,
    shuffle_items=True,
    shuffle_seed=42
)
```

### Recommended Parameter Settings
- **draws & tune:** 1000 (default) is sufficient for standard analysis. Increase `tune` up to 2000 if you see many divergences, poor adaptation messages, or R-hat far from 1.0. Increase `draws` up to 4000 if HDIs lack sufficient precision. Values below 1000 (e.g., 50) are only for debugging.
- **chains & cores:** Default values of `chains=4` and `cores=4` are suitable for most analyses. Increase `chains` for more robust MCMC sampling and `cores` for faster parallel sampling (up to your system's CPU core count).
- **CI width thresholds:**
  - `delta_item`: 0.05 is a common choice for high precision; 0.1 is more lenient.
  - `delta_cap`: 0.05 for group-level precision; increase for faster but less precise stopping.
- **Stabilization parameter:**
  - `CI_delta` controls how stable the CI slope must be before stopping. Smaller values (e.g., 0.0002) require more stability; larger values allow earlier stopping.
- **Conservatism:**
  - `conservatism=10` is typical; increase for more caution in low-performance scenarios.
- **Reproducibility:**
  - For reproducible pruned DataFrames, set a random seed before running your analysis (e.g., `np.random.seed(42)` or pass `random_seed` in params).
  - **Note:** Due to the stochastic nature of MCMC and parallelization, summary statistics (e.g., CI bounds, widths) are not guaranteed to be bitwise reproducible, even with the same random seed. Only the pruned DataFrame is guaranteed to be reproducible; summary values may differ slightly between runs.
- **Logging:**
  - Use `configure_optstop_logging()` to log all stopping decisions to file (console output is suppressed by default).

### Example: Recommended Parameters
```python
params = {
    'delta_item': 0.05,      # High precision for items
    'delta_cap': 0.05,       # High precision for group/task
    'draws': 1000,           # Default (increase up to 4000 if HDIs lack precision)
    'tune': 1000,            # Default (increase up to 2000 if many divergences or R-hat issues)
    'chains': 4,             # Number of MCMC chains
    'cores': 4,              # Number of CPU cores for sampling
    'CI_delta': 0.00001,      # Require stable CI slope
    'conservatism': 10,       # Typical value
    # ... other parameters as needed ...
}
```

See the parameter table above for all options and defaults.

### Default Parameter Rationale

The current defaults are optimised for **safety** - strict stopping criteria and minimum viable inference burden. This means the algorithm will be conservative in its stopping decisions, preferring to collect more data rather than risk premature stopping.

- **To stop more aggressively**, relax decision criteria: widen `delta_item` and `delta_cap` (CI width thresholds for samples and groupings respectively), increase `CI_delta` (stabilisation slope threshold, where higher values allow earlier stopping), and lower `cred_level` towards 0.9.
- **If you see many divergences, poor adaptation messages, or R-hat far from 1.0**, increase `tune` (up to 2000).
- **If HDIs lack sufficient precision**, increase `draws` (up to 4000).

### Credible Interval Coverage Considerations

The credible intervals produced by optstop are well-calibrated when groupings contain **50+ items** with performance in the **0.2-0.8 range**. For smaller samples or near-boundary performance (close to 0% or 100%), CIs may undercover due to hierarchical shrinkage - a known property of Bayesian hierarchical models.

**Factors affecting coverage:**
- **Sample size**: Coverage approaches nominal levels (~97%) as grouping size increases toward 50+ items
- **Performance range**: Mid-range performance (0.2-0.8) yields better-calibrated intervals than boundary cases
- **Hierarchical shrinkage**: Small groupings with extreme performance (near 0% or 100%) may exhibit substantial undercoverage

In stress-test conditions (e.g., 18-30 items per grouping with many boundary performers), observed coverage can fall substantially below nominal. Under more typical conditions with adequate sample sizes and mid-range performance, coverage is considerably better. The package automatically detects near-boundary estimates and emits diagnostic warnings recommending that raw proportions be reported alongside model estimates.

**Recommendations for small samples or boundary performance:**
1. **Treat CIs as rough guides** rather than formal statistical intervals
2. **Report raw proportions** alongside model estimates for transparency
3. **Run multiple shuffled orderings** (use `shuffle_items=True` with different seeds) to assess estimate stability
4. **For formal inference**, consider collecting more data or using methods designed for small samples

This limitation affects both live and post-hoc modes equally, as it is structural to hierarchical Bayesian models rather than a bug in the implementation.

## License
MIT

## FAQ & Troubleshooting

**Q: I get a ValueError about missing columns.**
- Make sure your DataFrame includes all required columns that you specify for grouping, sample ID, epoch, and score.

**Q: My summary statistics are not exactly reproducible, even with the same random seed.**
- This is expected due to the stochastic nature of MCMC and parallelization. Only the pruned DataFrame is guaranteed to be reproducible; summary values may differ slightly between runs.

**Q: I get PyMC or sampling errors (e.g., "Too few samples", "NUTS initialization failed").**
- Increase `tune` up to 2000 if you see many divergences or R-hat issues, and `draws` up to 4000 if HDIs lack precision. For very small test runs, warnings are expected.
- Ensure your data is not empty or all-NaN for any grouping.

**Q: The code is slow or uses a lot of CPU.**
- The package parallelizes across groupings. If you have many groupings, this can use all available CPU cores. You can reduce the number of groupings or run on a machine with more resources.

**Q: How do I get more detailed logs?**
- Use `configure_optstop_logging()` to log to file (console output is suppressed by default). Check the log file for detailed stopping decisions and errors.

**Q: How do I interpret the 'error' field in the output?**
- If a grouping fails (e.g., due to bad data), the 'error' field will contain the error message. All other fields for that grouping will be None. 

## Command-Line Interface (CLI)

The package provides CLI entry points for all major functions. After installing with `pip install .`, you can use the following commands from your terminal:

### 1. Post-hoc Optimal Stopping
**Command:**
```
optstop-posthoc --csv mydata.csv --output pruned.csv --summary summary.csv --grouping_columns subject,task --sample_id_column item_id --epoch_column trial_num --score_column accuracy --delta_item 0.05 --delta_cap 0.05 --draws 1000 --tune 1000 --chains 4 --cores 4 --CI_delta 0.00001 --conservatism 10 --rep_batch_size 1 --pymc_refresh_every 2 --stab_window 15 --random_seed 42 --generate_diagnostics --diagnostics_prefix my_diagnostics
```
- **--csv**: Path to input CSV file (required)
- **--output**: Path to output pruned CSV file (required)
- **--summary**: Path to output summary CSV file (optional)
- **--grouping_columns**: Comma-separated list or single column name for grouping (required)
- **--sample_id_column**: Column name for sample ID (required)
- **--epoch_column**: Column name for epoch/trial (required)
- **--score_column**: Column name for score (default: score)
- **--log**: Path to log file (default: optstop_cli.log)
- **--no_progress**: Disable the progress bar (shown by default)
- **--generate_diagnostics**: Generate diagnostic plots comparing full vs pruned datasets (optional)
- **--diagnostics_prefix**: Prefix for diagnostic output files (default: optstop_diagnostics)
- **--delta_item**: Max acceptable CI width for individual items (default: 0.05)
- **--delta_cap**: Max acceptable CI width for task/grouping (default: 0.05)
- **--draws**: Number of MCMC samples for PyMC (default: 1000)
- **--tune**: Number of tuning steps for PyMC (default: 1000)
- **--chains**: Number of MCMC chains for PyMC (default: 4)
- **--cores**: Number of CPU cores for PyMC (default: 4)
- **--CI_delta**: Slope threshold for determining CI stabilization (default: 0.00001)
- **--conservatism**: Factor for rare event conservatism (default: 10)
- **--rep_batch_size**: Number of repetitions to process in each batch (default: 1)
- **--pymc_refresh_every**: How often to run the PyMC model (default: 2)
- **--stab_window**: Window size for assessing CI stabilization (default: 15)
- **--random_seed**: Random seed for reproducible results (optional)
- **--low_performance_threshold**: Success rate below which conservative stopping is applied (default: 0.001)
- **--disable_gpu**: Disable GPU acceleration even if available
- **--force_gpu**: Force GPU usage (will fail if GPU unavailable)
- **--ordinal_tasks**: Comma-separated list of substrings to identify ordinal groupings (e.g., "confidence,rating")
- **--ordinal_max_score**: Maximum score for ordinal data (default: 10)
- **--ordinal_inference**: Ordinal inference method: modal, entropy, or hybrid (default: modal)
- **--ordinal_model_type**: Hierarchical model type: ordered_logistic or dirichlet (default: ordered_logistic)
- **--entropy_threshold**: Proportion of max entropy for false peak detection in hybrid mode (default: 0.8)
- **--prior_mu**: Centre of group-level Normal prior on logit scale (default: 0.0 = 50% probability)
- **--prior_sigma**: Scale of group-level Normal prior. If not set, uses pathway defaults (binary/cont: 1.5, ordinal: 2.0)
- **--shuffle_items**: Randomize item order within each grouping before processing. Recommended to avoid selection bias from sorted input.
- **--shuffle_seed**: Random seed for reproducible shuffling (only used with --shuffle_items)
- **--processing_order**: Processing order: `item_greedy` (all epochs per item, fast) or `epoch_interleaved` (all items per epoch, matches production). Default: `item_greedy`
- **--reanalysis_interval**: Group model refresh interval in trials for `epoch_interleaved` mode (default: 10)
- **--continuous_tasks**: Comma-separated list of substrings to identify continuous bounded [0,1] groupings (e.g., "accuracy,quality")

### 2. Live Optimal Stopping
**Command:**
```
optstop-live --csv current_data.csv --grouping_columns subject --sample_id_column item_id --epoch_column trial_num --score_column accuracy --delta_item 0.05 --delta_cap 0.05 --draws 1000 --tune 1000 --chains 4 --cores 4 --CI_delta 0.00001 --conservatism 10 --rep_batch_size 1 --pymc_refresh_every 2 --stab_window 15 --random_seed 42
```
- **--csv**: Path to input CSV file (required)
- **--grouping_columns**: Comma-separated list or single column name for grouping (required)
- **--sample_id_column**: Column name for sample ID (required)
- **--epoch_column**: Column name for epoch/trial (required)
- **--score_column**: Column name for score (default: score)
- **--log**: Path to log file (default: optstop_live.log)
- **--no_progress**: Disable the progress bar (shown by default)
- **--delta_item**: Max acceptable CI width for individual items (default: 0.05)
- **--delta_cap**: Max acceptable CI width for task/grouping (default: 0.05)
- **--draws**: Number of MCMC samples for PyMC (default: 1000)
- **--tune**: Number of tuning steps for PyMC (default: 1000)
- **--chains**: Number of MCMC chains for PyMC (default: 4)
- **--cores**: Number of CPU cores for PyMC (default: 4)
- **--CI_delta**: Slope threshold for determining CI stabilization (default: 0.00001)
- **--conservatism**: Factor for rare event conservatism (default: 10)
- **--rep_batch_size**: Number of repetitions to process in each batch (default: 1)
- **--pymc_refresh_every**: How often to run the PyMC model (default: 2)
- **--stab_window**: Window size for assessing CI stabilization (default: 15)
- **--random_seed**: Random seed for reproducible results (optional)
- **--low_performance_threshold**: Success rate below which conservative stopping is applied (default: 0.001)
- **--disable_gpu**: Disable GPU acceleration even if available
- **--force_gpu**: Force GPU usage (will fail if GPU unavailable)
- **--ordinal_tasks**: Comma-separated list of substrings to identify ordinal groupings (e.g., "confidence,rating")
- **--ordinal_max_score**: Maximum score for ordinal data (default: 10)
- **--ordinal_inference**: Ordinal inference method: modal, entropy, or hybrid (default: modal)
- **--ordinal_model_type**: Hierarchical model type: ordered_logistic or dirichlet (default: ordered_logistic)
- **--entropy_threshold**: Proportion of max entropy for false peak detection in hybrid mode (default: 0.8)
- **--prior_mu**: Centre of group-level Normal prior on logit scale (default: 0.0 = 50% probability)
- **--prior_sigma**: Scale of group-level Normal prior. If not set, uses pathway defaults (binary/cont: 1.5, ordinal: 2.0)
- **--continuous_tasks**: Comma-separated list of substrings to identify continuous bounded [0,1] groupings (e.g., "accuracy,quality")
- Prints which sample IDs (with grouping prefix) and/or groupings can be stopped.

### 3. Convergence Analysis
**Command:**
```
optstop-convergence --csv mydata.csv --output convergence_stats.csv --grouping_columns subject,task --sample_id_column item_id --epoch_column trial_num --score_column accuracy --delta_item 0.05 --delta_cap 0.05 --draws 1000 --tune 1000 --chains 4 --cores 4 --CI_delta 0.00001 --conservatism 10 --rep_batch_size 1 --pymc_refresh_every 2 --stab_window 15 --item_seqs 20 --epoch_seqs 20 --random_seed 42 --no_diagnostics --diagnostics_prefix my_convergence_eval
```
- **--csv**: Path to input CSV file (required)
- **--output**: Path to output convergence stats CSV file (required)
- **--grouping_columns**: Comma-separated list or single column name for grouping (required)
- **--sample_id_column**: Column name for sample ID (required)
- **--epoch_column**: Column name for epoch/trial (required)
- **--score_column**: Column name for score (default: score)
- **--log**: Path to log file (default: optstop_convergence.log)
- **--no_progress**: Disable the progress bar (shown by default)
- **--no_diagnostics**: Disable diagnostic plots (default: ON)
- **--diagnostics_prefix**: Prefix for diagnostic output files (default: convergence_eval)
- **--delta_item**: Max acceptable CI width for individual items (default: 0.05)
- **--delta_cap**: Max acceptable CI width for task/grouping (default: 0.05)
- **--draws**: Number of MCMC samples for PyMC (default: 1000)
- **--tune**: Number of tuning steps for PyMC (default: 1000)
- **--chains**: Number of MCMC chains for PyMC (default: 4)
- **--cores**: Number of CPU cores for PyMC (default: 4)
- **--CI_delta**: Slope threshold for determining CI stabilization (default: 0.00001)
- **--conservatism**: Factor for rare event conservatism (default: 10)
- **--rep_batch_size**: Number of repetitions to process in each batch (default: 1)
- **--pymc_refresh_every**: How often to run the PyMC model (default: 2)
- **--stab_window**: Window size for assessing CI stabilization (default: 15)
- **--item_seqs**: Number of randomized item orderings per grouping (default: 20)
- **--epoch_seqs**: Number of randomized epoch orderings per item (default: 20)
- **--random_seed**: Random seed for reproducible results (optional)
- **--low_performance_threshold**: Success rate below which conservative stopping is applied (default: 0.001)
- **--ordinal_tasks**: Comma-separated list of substrings to identify ordinal groupings (e.g., "confidence,rating")
- **--ordinal_max_score**: Maximum score for ordinal data (default: 10)
- **--ordinal_inference**: Ordinal inference method: modal, entropy, or hybrid (default: modal)
- **--ordinal_model_type**: Hierarchical model type: ordered_logistic or dirichlet (default: ordered_logistic)
- **--entropy_threshold**: Proportion of max entropy for false peak detection in hybrid mode (default: 0.8)
- **--prior_mu**: Centre of group-level Normal prior on logit scale (default: 0.0 = 50% probability)
- **--prior_sigma**: Scale of group-level Normal prior. If not set, uses pathway defaults (binary/cont: 1.5, ordinal: 2.0)
- **--continuous_tasks**: Comma-separated list of substrings to identify continuous bounded [0,1] groupings (e.g., "accuracy,quality")
- **--disable_gpu**: Disable GPU acceleration even if available
- **--force_gpu**: Force GPU usage (will fail if GPU unavailable)

### CLI Help
For any command, you can see all options and help text with:
```
optstop-posthoc --help
optstop-live --help
optstop-convergence --help
```

### Best Practices for CLI Usage
- **Progress bar is shown by default.** Use `--no_progress` to disable it for silent or script-based runs.
- **Always check your input CSV for required columns:** Make sure the columns you specify for grouping, sample ID, and epoch exist in your data.
- **Set a random seed** (`--random_seed`) for reproducible pruned DataFrames.
- **Consider using `--shuffle_items` for post-hoc analysis** to avoid selection bias from sorted input. If your input data is sorted (e.g., alphabetically by sample ID), early stopping may select a biased subsample if early-alphabet items happen to have systematically different scores. The package will warn you if sorted input is detected, but using `--shuffle_items` (optionally with `--shuffle_seed` for reproducibility) is recommended for unbiased results.
- **Use recommended parameter values** for real analyses (see Best Practices above).
- **Check the log file** for detailed stopping decisions, errors, and parameter validation.
- **For large datasets, run on a machine with sufficient CPU and memory.**
- **If you encounter errors, check the FAQ and log file for troubleshooting tips.** 

**Note:** After running any of the main functions (`optimal_stopping_posthoc`, `optimal_stopping_live`, or `convergence_posthoc`), you will see a message printed to the console reminding you where to find the log file with all details and warnings. This log file contains all stopping decisions, errors, and PyMC warnings, even if the terminal output is quiet. 




