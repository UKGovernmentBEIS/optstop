# optstop

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Adaptive Optimal Stopping Rule Algorithms for Efficient Data Collection and Analysis

## Overview

`optstop` is a Python package implementing adaptive optimal stopping algorithms for both post-hoc (batch) and live (incremental) data collection scenarios. It is designed to help researchers and practitioners efficiently determine when enough data has been collected to make reliable inferences, saving time and resources.

## Paper

This repository is the reference implementation accompanying the paper:

> Pilditch, T. D. (2026). *Knowing When to Stop: Bayesian Optimal Stopping for LLM Evaluations*. arXiv:2608.14425. Presented at the ICML 2026 Workshop on Statistical Frameworks for Uncertainty in Agentic Systems.

The paper PDF, along with supporting simulation scripts, validation outputs, and reproducibility materials, is in the [`paper/`](paper/) directory.

### Citing

If you use `optstop` in your research, please cite the paper:

```bibtex
@misc{pilditch2026knowing,
  title={Knowing When to Stop: {B}ayesian Optimal Stopping for {LLM} Evaluations},
  author={Pilditch, Toby D.},
  year={2026},
  eprint={2608.14425},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  doi={10.48550/arXiv.2608.14425},
  note={Presented at the ICML 2026 Workshop on Statistical Frameworks for Uncertainty in Agentic Systems}
}
```

## Features
- Post-hoc (batch) optimal stopping for retrospective analysis and dataset pruning
- Live (incremental) optimal stopping for real-time data collection
- Flexible, parameterised stopping criteria
- Flexible column mapping for groupings, sample IDs, and epochs
- Bayesian and frequentist hybrid methodology
- GPU acceleration via JAX/numpyro for faster PyMC sampling
- Ordinal scoring support for ordinal data (e.g., 0-10), bounded continuous support (e.g., for bounded aggregates), in addition to binary (0/1) scoring
- Convergence projection: estimates additional trials needed when evaluation ends before convergence
- inspect_ai integration for LLM evaluation workflows with adaptive early stopping

## Installation

optstop is not yet published on PyPI - install directly from GitHub:

```bash
pip install "git+https://github.com/UKGovernmentBEIS/optstop.git"                    # Core package
pip install "optstop[inspect] @ git+https://github.com/UKGovernmentBEIS/optstop.git" # With inspect_ai integration
pip install "optstop[gpu] @ git+https://github.com/UKGovernmentBEIS/optstop.git"     # With GPU acceleration (NVIDIA/Apple Silicon)
```

Once optstop is published to PyPI the `pip install optstop` form (with the same extras) will also work.

**Reproducible install:** dependencies are specified with lower bounds only (with a single cap, `arviz<1.0`, which also holds `pymc<6`). For the exact validated dependency set, install against the committed lockfile after cloning:

```bash
pip install -r requirements-lock.txt
```

See [detailed installation options](#installation-1) for development setup and platform-specific notes.

> **Using optstop with inspect_ai?** See the [inspect_ai integration guide](#using-optstop-with-inspect_ai) below, and the full [`BRIDGE_API_REFERENCE.md`](BRIDGE_API_REFERENCE.md) for the `OptimalStoppingManager` API.

## Quick Start (Standalone Post-hoc Analysis)

The most common use case: you have evaluation results in a CSV or DataFrame and want to determine how much of that data was actually needed.

### Step 1: Format your data

Your data must be in **long format** - one row per observation. You then tell optstop which columns play which roles:

| Column role | What it means | Example columns |
|-------------|---------------|-----------------|
| **Grouping** | Any column(s) representing factors that may systematically affect performance. Stopping decisions are made independently per unique combination. A good rule of thumb: include any factor you are manipulating or comparing (e.g., model, task, token budget). | `model`, `benchmark`, `max_tokens` |
| **Sample ID** | Individual test items within a grouping. | `question_id` |
| **Epoch** | Repeated evaluations of the same item (integers, processed in ascending order). | `run_number` |
| **Score** | The outcome. Binary (0/1), ordinal integers (e.g., 0-10), or continuous floats. | `accuracy`, `rating` |

You can use any columns from your data as groupings - they don't need special names. If you specify multiple grouping columns, each unique combination becomes an independent analysis (e.g., `['model', 'benchmark']` with 2 models and 3 benchmarks yields 6 groupings).

Example CSV structure:
```
model,       benchmark,   question_id,  run,  accuracy
gpt-4o,      mmlu,        q_001,        1,    1
gpt-4o,      mmlu,        q_001,        2,    1
gpt-4o,      mmlu,        q_002,        1,    0
gpt-4o,      mmlu,        q_002,        2,    1
gpt-4o,      gpqa,        q_101,        1,    0
claude-4,    mmlu,        q_001,        1,    1
...
```

### Step 2: Run the analysis

```python
import pandas as pd
from optstop import optimal_stopping_posthoc

df = pd.read_csv('my_eval_results.csv')

pruned_df, summary = optimal_stopping_posthoc(
    df,
    params={},                                  # Use defaults (delta=0.05, 97% CIs)
    grouping_columns=['model', 'benchmark'],    # Each model-benchmark pair analysed independently
    sample_id_column='question_id',
    epoch_column='run',
    score_column='accuracy'
)
```

This analyses each grouping (e.g., `gpt-4o-mmlu`, `gpt-4o-gpqa`, `claude-4-mmlu`) independently, determining where data collection could have safely stopped.

> **Runtime:** Each grouping runs Bayesian MCMC inference, so analysis takes longer than simple summary statistics. Runtime scales with the number of groupings, items, and epochs. Progress is logged by default.

### Step 3: Interpret the results

```python
for entry in summary:
    g = entry['grouping']
    eff = entry['percent_items_used']
    width = entry['theta_ci_width']
    print(f"{g}: used {eff:.1%} of data, final CI width {width:.3f}")

    # For groupings that didn't converge, check the projection
    if entry.get('convergence_projection'):
        proj = entry['convergence_projection']
        print(f"  -> ~{proj['simple_proj_additional_trials']:.0f} more trials estimated to converge")
```

`pruned_df` contains only the rows up to each grouping's stopping point - you can use this directly for downstream analysis, knowing the precision target was met.

**Key fields in each summary entry:**

| Field | Description |
|-------|-------------|
| `grouping` | Grouping name (hyphen-joined from grouping columns) |
| `theta_ci_low`, `theta_ci_high` | Credible interval bounds on group performance (a point estimate, if needed, is the interval midpoint) |
| `theta_ci_width` | CI width (lower = more precise) |
| `percent_items_used` | Fraction of items evaluated before stopping (0-1; format with `:.1%`) |
| `n_items_used` | Number of items evaluated |
| `avg_reps_per_item` | Mean epochs per item at stopping |
| `convergence_projection` | For non-converged groupings: estimated additional trials needed (dict or `None`) |
| `low_perf_stop_suppressed` | `True` if an automatic stop was suppressed because observed performance was below `low_performance_threshold` - the low-base-rate safeguard, so all available samples were used (issue #3) |
| `pinned` | Binary/continuous only: `True` when the whole reported credible interval sits below the resolution floor, so the bounds are prior-dominated and qualitative (issue #3). `False` for ordinal |
| `low_perf_floor` | Ordinal counterpart to `pinned` (telemetry only, does not affect stopping): `True` when the resolved normalised performance is below `low_performance_threshold`. `False` for binary/continuous |
| `boundary_diagnostic` | Warning string if performance resolved near the floor/ceiling, else `None` |
| `error` | Error message if inference failed for this grouping (`None` if successful) |

### Adjusting to your needs

**Score types**: By default, all groupings use binary (0/1) inference. For ordinal rubric scores, specify which groupings are ordinal:

```python
pruned_df, summary = optimal_stopping_posthoc(
    df, params={},
    grouping_columns=['model', 'benchmark'],
    sample_id_column='question_id',
    epoch_column='run',
    score_column='rating',
    ordinal_tasks=['writing'],      # Groupings whose name contains 'writing' use ordinal inference
                                     # (matched against the hyphen-joined grouping key, case-insensitive)
    ordinal_max_score=10,           # Rating scale is 0-10
    ordinal_inference='hybrid',     # Recommended mode (modal CI + entropy convergence)
)
```

> **Ordinal score indexing.** The package models categories 0 through `ordinal_max_score`. If your rubric uses 1-5, you can either subtract 1 from scores before analysis (`df['rating'] = df['rating'] - 1`, with `ordinal_max_score=4`), or set `ordinal_max_score=5` and let category 0 remain empty. Both approaches produce correct modal estimates with sufficient data; the latter may trigger a sparsity warning for the unused category.

**Precision threshold**: The `delta_item` and `delta_cap` parameters control how precise estimates must be before stopping. Lower values require more data but give tighter estimates:

```python
params = {
    'delta_item': 0.05,   # Item-level CI width target (default: 0.05)
    'delta_cap': 0.05,    # Group-level CI width target (default: 0.05)
}
```

**Shuffling**: If your items are sorted by difficulty, shuffle before analysis to satisfy the exchangeability assumption:

```python
pruned_df, summary = optimal_stopping_posthoc(
    df, params={},
    grouping_columns=['model', 'benchmark'],
    sample_id_column='question_id',
    epoch_column='run',
    score_column='accuracy',
    shuffle_items=True,
    shuffle_seed=42
)
```

For the full parameter reference, see [Setting Parameters](#setting-parameters). For logging configuration, diagnostic plots, and advanced usage, see the sections below.

---

## Using optstop with inspect_ai

`optstop` integrates with [inspect_ai](https://inspect.aisi.org.uk/), the UK AI Security Institute's framework for LLM evaluations. The `OptimalStoppingManager` implements the `EarlyStopping` protocol, adding Bayesian adaptive early stopping to your LLM evaluations.

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
# Install optstop with inspect_ai support (from GitHub - not yet on PyPI)
pip install "optstop[inspect] @ git+https://github.com/UKGovernmentBEIS/optstop.git"

# Or with GPU acceleration
pip install "optstop[inspect,gpu] @ git+https://github.com/UKGovernmentBEIS/optstop.git"
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
    'conservatism': 5,         # Conservative for rare events
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

**Note:** Ordinal scores are expected to be **0-indexed** (range [0, ordinal_max_score]). If your scorer produces 1-indexed scores (e.g., 1-5 star ratings), transform them to 0-indexed before use, or open an issue on the GitHub repository.

**Performance Note for Ordinal Discrete Tasks:**
Ordinal discrete inference (without score aggregation) uses entropy-based Bayesian models that are more computationally intensive than binary or continuous pathways (~2-3 minutes per inference call vs ~10-30 seconds, for typical 5-11 category scales with ~100 items). However, inference runs in a background thread that overlaps with LLM processing, so in practice ordinal inference adds no observable delay to overall evaluation time (in comparative testing, ordinal and continuous evaluations completed in virtually identical wall time). The more important difference is **convergence behaviour**: ordinal's entropy validation gate typically requires more data to converge, resulting in lower efficiency than binary or continuous pathways at the same precision threshold. GPU acceleration may help for evaluations with very fast-completing trials or large ordinal scales (20+ categories). See [GPU Acceleration](#gpu-acceleration) for setup details.

```python
# Enable GPU for ordinal evaluations
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    ordinal_tasks=['rating'],
    ordinal_inference='hybrid',
    gpu_ids=[0]  # GPU-accelerated MCMC sampling
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
- **inspect_ai bridge (`OptimalStoppingManager`)**: Defaults to `'hybrid'` - prioritises safety in automated evaluation contexts where stopping decisions have real cost implications.
- **Standalone functions (`optimal_stopping_posthoc`, `optimal_stopping_live`, CLI)**: Defaults to `'modal'` - prioritises speed for interactive/exploratory analysis where users can iterate quickly.

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
                print(f"{grouping}: ~{proj['simple_proj_additional_trials']:.0f} more trials (1/√n estimate)")
                print(f"  Proximity: {proj['proximity_ratio']:.1f}x away from target")
                print(f"  Trajectory: {proj['trajectory_signal'] or 'n/a'}")
                print(f"  Exponential estimate: ~{proj['projected_additional_trials']} trials (order-sensitive)")
                print(f"  80% CI: {proj['uncertainty']['ci_trials_80']} trials")
```

#### Convergence Projection

For groupings that have not converged when evaluation ends, optstop estimates how many additional trials would be needed. This is available in both the bridge pathway (`complete_task()` diagnostics) and `optimal_stopping_posthoc` (per-grouping `convergence_projection` field in the summary list). The projection uses an exponential decay model as the primary approach (fitting `w(t) = a*exp(-b*t) + c` to the CI width trajectory), with linear extrapolation as a fallback when the exponential fit is unavailable or poor.

The projection classifies each non-converged grouping into one of three outcomes via `convergence_target`:

- **`projected_width`**: CI width is projected to drop below `delta_cap` (the group-level CI width threshold, default 0.05) - the grouping would converge with more data. Increase your sample budget or epochs accordingly.
- **`projected_slope_stabilisation`**: CI width is projected to plateau above `delta_cap` - additional data yields diminishing returns. Widen `delta_cap` to accept the current precision, or investigate whether the grouping has high intrinsic variance.
- **`projected_capped`**: Neither outcome detected within the projection horizon. Check whether the grouping has very few observations (< 5 group-level checks) - more data may clarify the trajectory. If observations are plentiful but no clear trend emerges, the data may be too noisy for the current stopping criteria.

Both the bridge pathway and `optimal_stopping_posthoc` include full uncertainty quantification via residual bootstrap (80% and 50% confidence intervals) and a `confidence_level` (`'high'`, `'moderate'`, or `'low'`). In `convergence_posthoc`, shortfall estimates are point estimates only (bootstrap disabled for speed, since results are aggregated across randomised orderings). Groupings with insufficient CI width history for projection will have `convergence_projection` set to `None`.

See the Convergence Projection Fields section in `BRIDGE_API_REFERENCE.md` for the full field reference (applicable to all modes, not just inspect_ai integration).

### Best Practices

1. **Start conservative**: Use `delta_item=0.05`, `delta_cap=0.05` for high precision
2. **Test with shadow mode**: Run once with `shadow_mode=True` to see potential savings
3. **Set appropriate groupings**: More granular groupings = more targeted stopping
4. **Monitor efficiency**: If efficiency is 0%, your stopping criteria may be too strict
5. **Check compatibility**: Verify inspect_ai version with `optstop.check_inspect_ai_compatibility("0.3.5")`
6. **Sample ID Randomisation**: Ensure evals are constructed such that order of sample IDs is randomised (i.e., avoid systematic influences of sample ID order on potential performance)

### Compatibility

- **optstop version**: 0.5.0
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
- **Solution**: Install the inspect extra: `pip install "optstop[inspect] @ git+https://github.com/UKGovernmentBEIS/optstop.git"`

**Issue**: Early stopping not triggering
- **Solution**: Check `reanalysis_interval` and `min_samples_per_grouping` settings. Increase epochs if needed.

**Issue**: Invalid score warnings
- **Solution**: Verify scores are in valid range (0-1 for binary, 0-ordinal_max_score for ordinal)

**Issue**: GPU not detected
- **Solution**: Install the GPU extra: `pip install "optstop[gpu] @ git+https://github.com/UKGovernmentBEIS/optstop.git"`

For more issues, see the development roadmap or open a GitHub issue.

## GPU Acceleration

`optstop` supports GPU acceleration for PyMC sampling operations, which speeds up MCMC sampling for large datasets and complex models.

### Prerequisites
- NVIDIA GPU with CUDA support
- JAX with GPU support installed: `pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`

### Automatic GPU Detection
The package automatically detects GPU availability and configures optimal sampling parameters:
- **GPU Available**: Uses JAX/numpyro sampler with optimised chain/core settings
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
- Faster MCMC sampling on typical workloads
- Automatic optimisation of chain/core parameters for GPU

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

**Unusual scoring configurations?** If your evaluation involves very large rubric scales (e.g., >20 categories), peculiar response distributions (e.g., models can only produce scores of 1, 3, 8, and 10 on a 0-10 scale), or you are uncertain about how to select or balance multiple scorer priorities for applying optimal stopping, please open an issue on the GitHub repository for guidance on configuring optstop for your use case.

### Model Types

**Ordered Logistic** (Recommended, Default):
- Cumulative link model that respects ordinal structure
- Uses identified cutpoints (first cutpoint fixed at 0 for model identification)
- Adaptive priors that scale with number of categories
- Faster than Dirichlet-Multinomial due to fewer parameters
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
    'conservatism': 5,
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

**Ordinal estimand note:** The ordinal pathway estimates the *modal category* (most probable score), not the mean score. The group-level theta reported for ordinal groupings is `modal_category / max_score`, and credible intervals bracket the mode. This differs from binary and continuous pathways, which estimate mean performance. When comparing theta values across pathways, be aware that ordinal theta reflects categorical concentration rather than average score.

**Binary Model Details:**
The binary inference pathway uses a logit-normal hierarchical model (`mu_group ~ Normal(prior_mu, sigma)` on the logit scale, where sigma defaults to 1.5 when `prior_sigma` is None). A Beta-Binomial alternative was evaluated through controlled simulation under conditions favouring the Beta-Binomial (Beta-Binomial data generating process with κ=10, 94% HDI credible intervals). In the mid-range (0.1-0.9 true performance), the two models are statistically indistinguishable in bias, credible interval width, coverage, and stopping behaviour (absolute bias 0.022 vs 0.022, CI width 0.070 vs 0.070). At exact boundaries (0.0 or 1.0), point-estimate bias is marginally lower for the Beta-Binomial (0.007 vs 0.005) but credible interval widths are near-identical (0.011 vs 0.011), and both models show zero coverage - an expected consequence of the degenerate boundary data-generating process, not a model deficiency. Because equivalent CI widths imply equivalent stopping times, the model choice does not alter stopping behaviour where evaluations typically operate. The decisive difference is computational reliability: the Beta-Binomial produced 120× more MCMC divergences than the logit-normal in the mid-range, rising to 9,206× at exact boundaries (with effective sample sizes as low as 7), indicating a poorly conditioned posterior geometry in precisely the sparse, boundary-adjacent regime an adaptive stopping framework must handle robustly.

**Continuous Bounded Model Details:**
The continuous bounded pathway uses a hierarchical Beta model with the following structure:
- **mu_group**: Group-level mean (logit scale)
- **sigma_group**: Between-item standard deviation (captures item-level variability)
- **phi_group**: Group-level precision parameter
- **mu_item**: Item-level means (hierarchical, derived from mu_group + z * sigma_group)

This hierarchical structure correctly accounts for between-item variance when computing group-level credible intervals.

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

The package computes group-level credible intervals using the **expected group accuracy**: `mean(Theta)` across all items, rather than using individual item Theta values or `sigmoid(mu_group)`.

**Why this matters:**
- In hierarchical models, `Theta_i = sigmoid(mu_group + z_i * sigma_group)` for each item
- The expected group accuracy `E[Theta]` is the mean across items, NOT `sigmoid(mu_group)`
- When `sigma_group` is large, `sigmoid(mu_group)` can be very different from `E[Theta]`
  - Example: `mu_group=5.3`, `sigma_group=5.3` gives `sigmoid(mu_group)=0.995` but `E[Theta]=0.83`

**Previous approaches and their limitations:**
- `Theta[0]` (first item only): Arbitrary, depends on data ordering
- `sigmoid(mu_group)`: Measures "typical item" (z=0), not expected accuracy
- `mean(Theta)`: Correctly computes expected group accuracy ✓

This methodology correctly accounts for between-item variance (`sigma_group`) when computing credible intervals, providing more accurate stopping decisions for hierarchical data structures.

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

**Note**: Basic installation includes all core functionality. Ordinal inference will work but use the slower PyMC default sampler instead of the numpyro backend from the performance extras.

### Recommended: Performance Extras
For optimal performance, especially with **ordinal scoring**, install with performance extras:
```bash
pip install .[performance]
```

This installs:
- **JAX** (CPU backend): Enables faster ordinal inference
- **numpyro** (JAX-based sampler): Handles OrderedLogistic models efficiently

**Performance impact:**
- **Binary inference**: No difference (uses PyMC default for speed)
- **Continuous inference**: No difference (uses PyMC default for speed)
- **Ordinal inference**: faster with numpyro compared to PyMC default

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

### Running the tests
For a reproducible environment, install from the pinned lockfile, then the dev extras:
```bash
pip install -r requirements-lock.txt
pip install -e .[dev]
```
Run the suite **per test file** so each file's tests execute in a clean process:
```bash
for f in tests/test_*.py; do python -m pytest "$f" -m "not slow"; done
```
This is the recommended way to run the full suite. The tests build many PyMC models in one interpreter, and PyMC tracks active model contexts in process-global state; across the large number of MCMC-based tests that state can accumulate and produce spurious `No model on context stack` errors late in a single-process run. Every test file passes on its own and the package itself is unaffected - running per file simply keeps that process-global state clean. To isolate a single test while debugging, `pytest-forked` is available (`python -m pytest --forked tests/<file>::<test>`). Slow ordinal-MCMC tests are marked `slow` and skipped by `-m "not slow"`; drop that flag to include them.

### Installation Summary

| Installation | Binary | Continuous | Ordinal | Use Case |
|-------------|--------|------------|---------|----------|
| `pip install .` | ✓ Fast (PyMC default) | ✓ Fast (PyMC default) | ⚠ Slower (PyMC default) | Basic usage, no ordinal tasks |
| `pip install .[performance]` | ✓ Fast (PyMC default) | ✓ Fast (PyMC default) | ✓ Fast (numpyro) | **Recommended for ordinal tasks** |
| `pip install .[gpu]` | ✓ Fastest (numpyro GPU) | ✓ Fastest (numpyro GPU) | ✓ Fastest (numpyro GPU) | GPU hardware available |

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

You can control the behaviour of the optimal stopping algorithms through `optimal_stopping_posthoc` and `optimal_stopping_live`. Settings are supplied through two channels: a `params` dictionary and direct **keyword arguments**. Any omitted setting uses its default value.

> **Which channel?** Task routing and model selection (`ordinal_tasks`, `continuous_tasks`, `ordinal_max_score`, `ordinal_inference`, `ordinal_model_type`, `entropy_threshold`, `prior_mu`, `prior_sigma`), along with `processing_order`, `reanalysis_interval`, `shuffle_items`, `shuffle_seed`, `score_column`, `display_progress`, `generate_diagnostics`, and `diagnostics_prefix`, are **keyword arguments** - pass them directly in the function call (as the examples do), not inside `params`. A value placed in `params` for one of these is **silently ignored**; for the routing options that means the grouping is scored with the default binary pathway, with no warning. The CI thresholds, conservatism, and MCMC settings in the table below are read from the `params` dict.

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
| `conservatism`           | 5         | Both         | Factor for rare event conservatism (higher = more conservative)             |
| `low_performance_threshold` | 0.01   | Both         | Below this rate: suppress automatic stops (all pathways); ordinal groupings additionally report an inflated (floored) CI width |

**Parameter interaction - `conservatism` and `low_performance_threshold` (ordinal):** For ordinal groupings, when estimated performance falls below `low_performance_threshold` the effective CI-width target tightens to `delta_cap / conservatism`. With defaults (`conservatism=5`, `delta_cap=0.05`), this target is 0.01 - achievable with moderate data. If you raise `low_performance_threshold`, more groupings enter this conservative regime; if you also keep `conservatism` high, the tightened target may become unreachable within your data budget. When adjusting either parameter, check that `delta_cap / conservatism` remains achievable for the sample sizes you expect. (For binary and continuous groupings the below-threshold behaviour is *suppression* rather than a tightened target - see "Detecting low base-rate capabilities" below.)

| `draws`                  | 1000      | Both         | Number of MCMC samples for PyMC (affects speed/accuracy)                    |
| `tune`                   | 1000      | Both         | Number of tuning steps for PyMC                                             |
| `chains`                 | 4         | Both         | Number of MCMC chains for PyMC                                              |
| `cores`                  | 4         | Both         | Number of CPU cores for PyMC sampling                                       |
| `rep_batch_size`         | 1         | Both         | Number of repetitions to process in each batch                              |
| `pymc_refresh_every`     | 2         | Both         | How often to run the PyMC model (every N items)                             |
| `stab_window`            | 15         | Both         | Window size for assessing CI stabilisation                                  |
| `CI_delta`               | 0.00001    | Both         | Slope threshold for determining CI stabilisation                            |
| `target_accept`          | 0.95      | Both         | NUTS sampler target acceptance rate (higher = fewer divergences, slower) |
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
| `shuffle_items`          | False     | Post-hoc     | Randomise item order within each grouping before processing (recommended to avoid selection bias from sorted input) |
| `shuffle_seed`           | None      | Post-hoc     | Random seed for reproducible shuffling (only used with `shuffle_items=True`) |
| `processing_order`       | 'item_greedy' | Post-hoc | Processing order: `'item_greedy'` (all epochs per item, fast) or `'epoch_interleaved'` (all items per epoch, matches production bridge behaviour) |
| `reanalysis_interval`    | 10        | Post-hoc     | Trials between group model refreshes in `epoch_interleaved` mode (matches bridge default) |

### Example: Setting Parameters

You can set any combination of the `params`-dict rows above. For example:

```python
params = {
    'delta_item': 0.05,                # Require a narrower CI for stopping items
    'delta_cap': 0.05,                 # Task/grouping CI width threshold
    'cred_level': 0.97,               # Use 97% credible intervals
    'conservatism': 5,                # More conservative for rare events
    'low_performance_threshold': 0.01,# Adjust threshold for low performance
    'draws': 1000,                    # MCMC samples
    'tune': 1000,                     # Tuning steps
    'chains': 4,                      # Number of MCMC chains
    'cores': 4,                       # Number of CPU cores for sampling
    'rep_batch_size': 2,              # (post-hoc only) Process 2 reps at a time
    'pymc_refresh_every': 1,          # (post-hoc only) Run PyMC every item
    'stab_window': 15,                # Stabilisation window size
    'CI_delta': 0.00001,              # Stabilisation threshold
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
5. **Credible Intervals Comparison**: Side-by-side comparison of credible intervals from full vs. pruned datasets

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

See [Quick Start](#quick-start-standalone-post-hoc-analysis) for a complete post-hoc example with data formatting, column mapping, and output interpretation. If you only have one grouping column, you can pass `'subject'` or `['subject']`.

### Example Usage (Live)

For standalone live analysis, see the [`optimal_stopping_live` example](#example-usage-standalone-live) below. For live stopping via **inspect_ai**, see [Using optstop with inspect_ai](#using-optstop-with-inspect_ai).

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
    'item_seqs': 3,      # Randomisation sequences
    'epoch_seqs': 3      # Randomisation sequences
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

### Parallelisation (Post-hoc)
- The `optimal_stopping_posthoc` function now parallelises across groupings (unique combinations of the columns you specify for grouping), using all available CPU cores for efficient processing of large datasets.
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

See [Quick Start - Step 1](#step-1-format-your-data) for column roles, example data layout, and a worked example. Additional requirements:

- **Long format**: Each row is one observation (one model's response to one item on one epoch). Do not pass wide-format or pivoted data.
- **Complete epoch labelling**: Every observation needs a valid epoch value. If your data has only one epoch per sample, the sample-level stopping criteria will have limited data to work with.
- **Consistent score ranges**: All samples within an ordinal grouping should share the same score scale. Set `ordinal_max_score` to match your rubric maximum (e.g., 10 for a 0-10 scale, 5 for a 1-5 scale).
- **Sample ordering**: For post-hoc analysis, samples are processed in order of their first appearance in the DataFrame. If your data is sorted by some systematic property (e.g., item difficulty), consider shuffling before analysis to satisfy the exchangeability assumption (see `shuffle_items` in the [Quick Start](#adjusting-to-your-needs)).
- **Additional columns** are preserved in the output but are not used by the algorithm.

These requirements apply to both `optimal_stopping_posthoc` (batch analysis of completed data) and `optimal_stopping_live` (standalone incremental analysis). For live stopping via **inspect_ai**, see the [bridge integration guide](#using-optstop-with-inspect_ai) and [`BRIDGE_API_REFERENCE.md`](BRIDGE_API_REFERENCE.md) - data formatting is handled automatically by the framework.

### Example Usage (Standalone Live)

The standalone `optimal_stopping_live` function analyses a DataFrame incrementally, simulating the stopping decisions that would have occurred during sequential data collection. This is distinct from the [inspect_ai bridge](#using-optstop-with-inspect_ai), which integrates directly with live evaluation pipelines.

```python
from optstop import optimal_stopping_live

result = optimal_stopping_live(
    df, params={},
    grouping_columns=['model', 'task'],
    sample_id_column='item_id',
    epoch_column='epoch',
    score_column='score'
)
```

## Convergence Analysis (Post-hoc)

The package provides a function for post-hoc convergence analysis, allowing you to assess when statistical stability would have been achieved in sequential testing scenarios. This is useful for determining the minimum data required for reliable performance estimates.

### Function: `convergence_posthoc`

- **Purpose:**
  - Runs a post-hoc convergence analysis on a full dataset, parallelising across groupings (using all available CPU cores).
  - Returns a DataFrame detailing the required numbers of trials at epoch and sample_ID levels, split by the user's desired grouping.
- **Parameters:**
  - Accepts the same `params` dictionary as other functions (see table above), plus:
    - `item_seqs`: Number of randomised item orderings per grouping (default: 20)
    - `epoch_seqs`: Number of randomised epoch orderings per item (default: 20)
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

The resulting DataFrame contains all the convergence metrics for each grouping-task. The analysis is parallelised for speed.

## Convergence Diagnostics

When you run `convergence_posthoc`, the package now automatically generates two diagnostic figures by default:
- **{prefix}_grouped_needed.png**: Two vertically stacked plots showing mean needed items and mean needed epochs (with std error bars) by grouping (descending order).
- **{prefix}_score_scatter.png**: Scatter plots of mean needed items/epochs vs. performance, as before.

You can control this behaviour:
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
- **Parallelisation:** Each unique combination of the columns you specify for grouping will be processed in parallel, so ensure these columns are set appropriately for your experimental design.

### Input Ordering and Selection Bias (Post-hoc)
- **Items are processed in dataframe order** (within each grouping). If your input data is sorted (e.g., alphabetically by sample ID), early stopping may select a biased subsample if early-alphabet items happen to have systematically different scores than late-alphabet items.
- **Use `shuffle_items=True`** to randomise item order within each grouping before processing. This is recommended for unbiased post-hoc analysis. In `epoch_interleaved` mode, shuffling controls the order items are processed within each epoch.
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
- **Stabilisation parameter:**
  - `CI_delta` controls how stable the CI slope must be before stopping. Smaller values (e.g., 0.0002) require more stability; larger values allow earlier stopping.
- **Conservatism:**
  - `conservatism=5` is typical; increase for more caution in low-performance scenarios.
- **Low base-rate capabilities:**
  - To detect a rare capability and stop once it is observed, set `low_performance_threshold` to the success rate of interest. See ["Detecting low base-rate capabilities"](#detecting-low-base-rate-capabilities) below.
- **Reproducibility:**
  - For reproducible pruned DataFrames, set a random seed before running your analysis (e.g., `np.random.seed(42)` or pass `random_seed` in params).
  - **Note:** Due to the stochastic nature of MCMC and parallelisation, summary statistics (e.g., CI bounds, widths) are not guaranteed to be bitwise reproducible, even with the same random seed. Only the pruned DataFrame is guaranteed to be reproducible; summary values may differ slightly between runs.
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
    'conservatism': 5,       # Typical value
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

### Detecting low base-rate capabilities

Some evaluations target a capability that is expected to be very rare - for
example a hazardous behaviour that should occur at most once in a few thousand
trials. Two properties of the package matter here.

**The resolution floor.** Group-level inference clamps the likelihood logit to
`±6`, so the likelihood cannot distinguish rates below `sigmoid(-6) ≈ 0.0025`
(about 1 in 400). Since version 0.5.0 the *reported* credible interval is read
from an unclipped transform, so its lower bound can approach 0 and the interval
can contain a near-zero truth. However, below the resolution floor the interval
is prior-dominated: it is a qualitative indication that the rate is very small,
not a calibrated bound. When the whole reported interval lies below the floor,
the result carries `pinned=True`; report the raw observed proportion alongside
it. (Continuous scores have a second, independent variance clamp in the
likelihood, so sub-1% continuous intervals are doubly heuristic.)

**Using `low_performance_threshold` as a detection trigger.** Set
`low_performance_threshold` to the success rate of interest (e.g. `0.0005` for
1 in 2,000). While the cumulative observed rate stays below the threshold you
have not yet seen the event, so for both the **binary** and **continuous**
pathways the automatic precision and slope-stabilisation stops are suppressed at
every level - item, sample, and grouping - and sampling continues (at the
grouping level this is surfaced as `low_perf_stop_suppressed=True`). This holds
even if the accumulated null data would otherwise satisfy the width or slope
criterion - the gate keeps the search open until the event actually appears.

When the rare event occurs it lifts the cumulative observed rate to or above the
threshold, which releases the hold. The ordinary precision stop can then fire
once the interval is narrow enough. In effect you sample until you have seen
roughly one event's worth of evidence at your rate of interest, then stop.

Two caveats:

1. The trigger is the *cumulative* observed rate crossing the threshold, which is
   stochastic. The first event is expected near `1/p` trials, so a late first
   event may need a second before the running rate clears the bar, and an early
   one clears it immediately.
2. Set the threshold *to* the rate of interest, not above it (a higher threshold
   oversamples, waiting for more events) or below it (releases the hold before you
   have the evidence).

If your rate of interest is below the resolution floor (~0.25%), the event still
lifts the observed rate and releases the hold, but the reported bound remains
qualitative (`pinned=True`): you have confirmed and halted on the capability, not
precisely quantified how rare it is.

**The ordinal pathway.** The `pinned` reliability caveat is specific to the
binary and continuous pathways, which map a logit through a sigmoid that is
clamped at `±6` - the source of the resolution floor. The ordinal estimator has
no such location clamp, so its low-performance estimates are data-faithful and
carry no equivalent reliability caveat. For symmetry, an ordinal grouping that
resolves at a very low normalised performance level (below
`low_performance_threshold`) still reports a telemetry flag, `low_perf_floor=True`,
so low-capability ordinal groupings are as easy to detect as low-rate
binary/continuous ones. Unlike `pinned`, it is keyed off the normalised
performance *point estimate* (so it is independent of the modal/entropy inference
mode) and is not a reliability warning - the ordinal estimate below it is still
trustworthy. It is telemetry only and does not affect stopping. Group-level stop
suppression via `low_performance_threshold` applies to all three pathways
(surfaced as `low_perf_stop_suppressed=True`); the item-level suppression
described above is binary/continuous only, because on homogeneous near-zero data
the ordinal item-level interval is held open by a separate minimum-width floor
rather than by the low-performance gate.

**Log visibility.** Suppression is silent by default in the returned results
beyond the `low_perf_stop_suppressed` field. To make it visible while a run is in
progress, the first time a stop is suppressed for a grouping optstop emits a
single `INFO`-level log on the relevant `optstop.*` logger naming the grouping and
quoting the *resolved* `low_performance_threshold` value (so it reflects any value
you passed, not the default). Enable it with
`configure_optstop_logging(console_output=True)` (its default level is already
`INFO`) or by attaching your own handler to the `optstop` logger. Per-check detail
(each individual suppressed width/slope stop) is available at `DEBUG`.

## License
MIT

## FAQ & Troubleshooting

**Q: I get a ValueError about missing columns.**
- Make sure your DataFrame includes all required columns that you specify for grouping, sample ID, epoch, and score.

**Q: My summary statistics are not exactly reproducible, even with the same random seed.**
- This is expected due to the stochastic nature of MCMC and parallelisation. Only the pruned DataFrame is guaranteed to be reproducible; summary values may differ slightly between runs.

**Q: I get PyMC or sampling errors (e.g., "Too few samples", "NUTS initialisation failed").**
- Increase `tune` up to 2000 if you see many divergences or R-hat issues, and `draws` up to 4000 if HDIs lack precision. For very small test runs, warnings are expected.
- Ensure your data is not empty or all-NaN for any grouping.

**Q: The code is slow or uses a lot of CPU.**
- The package parallelises across groupings. If you have many groupings, this can use all available CPU cores. You can reduce the number of groupings or run on a machine with more resources.

**Q: How do I get more detailed logs?**
- Use `configure_optstop_logging()` to log to file (console output is suppressed by default). Check the log file for detailed stopping decisions and errors.

**Q: How do I interpret the 'error' field in the output?**
- If a grouping fails (e.g., due to bad data), the 'error' field will contain the error message. All other fields for that grouping will be None. 

## Command-Line Interface (CLI)

The package provides CLI entry points for all major functions. After installing with `pip install .`, you can use the following commands from your terminal:

### 1. Post-hoc Optimal Stopping
**Command:**
```
optstop-posthoc --csv mydata.csv --output pruned.csv --summary summary.csv --grouping_columns subject,task --sample_id_column item_id --epoch_column trial_num --score_column accuracy --delta_item 0.05 --delta_cap 0.05 --draws 1000 --tune 1000 --chains 4 --cores 4 --CI_delta 0.00001 --conservatism 5 --rep_batch_size 1 --pymc_refresh_every 2 --stab_window 15 --random_seed 42 --generate_diagnostics --diagnostics_prefix my_diagnostics
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
- **--CI_delta**: Slope threshold for determining CI stabilisation (default: 0.00001)
- **--conservatism**: Factor for rare event conservatism (default: 5)
- **--rep_batch_size**: Number of repetitions to process in each batch (default: 1)
- **--pymc_refresh_every**: How often to run the PyMC model (default: 2)
- **--stab_window**: Window size for assessing CI stabilisation (default: 15)
- **--random_seed**: Random seed for reproducible results (optional)
- **--low_performance_threshold**: Success rate below which automatic stops are suppressed to avoid a false stop on an unresolved low base rate; ordinal groupings additionally report an inflated (floored) CI width (default: 0.01)
- **--disable_gpu**: Disable GPU acceleration even if available
- **--force_gpu**: Force GPU usage (will fail if GPU unavailable)
- **--ordinal_tasks**: Comma-separated list of substrings to identify ordinal groupings (e.g., "confidence,rating")
- **--ordinal_max_score**: Maximum score for ordinal data (default: 10)
- **--ordinal_inference**: Ordinal inference method: modal, entropy, or hybrid (default: modal)
- **--ordinal_model_type**: Hierarchical model type: ordered_logistic or dirichlet (default: ordered_logistic)
- **--entropy_threshold**: Proportion of max entropy for false peak detection in hybrid mode (default: 0.8)
- **--prior_mu**: Centre of group-level Normal prior on logit scale (default: 0.0 = 50% probability)
- **--prior_sigma**: Scale of group-level Normal prior. If not set, uses pathway defaults (binary/cont: 1.5, ordinal: 2.0)
- **--shuffle_items**: Randomise item order within each grouping before processing. Recommended to avoid selection bias from sorted input.
- **--shuffle_seed**: Random seed for reproducible shuffling (only used with --shuffle_items)
- **--processing_order**: Processing order: `item_greedy` (all epochs per item, fast) or `epoch_interleaved` (all items per epoch, matches production). Default: `item_greedy`
- **--reanalysis_interval**: Group model refresh interval in trials for `epoch_interleaved` mode (default: 10)
- **--continuous_tasks**: Comma-separated list of substrings to identify continuous bounded [0,1] groupings (e.g., "accuracy,quality")

### 2. Live Optimal Stopping
**Command:**
```
optstop-live --csv current_data.csv --grouping_columns subject --sample_id_column item_id --epoch_column trial_num --score_column accuracy --delta_item 0.05 --delta_cap 0.05 --draws 1000 --tune 1000 --chains 4 --cores 4 --CI_delta 0.00001 --conservatism 5 --rep_batch_size 1 --pymc_refresh_every 2 --stab_window 15 --random_seed 42
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
- **--CI_delta**: Slope threshold for determining CI stabilisation (default: 0.00001)
- **--conservatism**: Factor for rare event conservatism (default: 5)
- **--rep_batch_size**: Number of repetitions to process in each batch (default: 1)
- **--pymc_refresh_every**: How often to run the PyMC model (default: 2)
- **--stab_window**: Window size for assessing CI stabilisation (default: 15)
- **--random_seed**: Random seed for reproducible results (optional)
- **--low_performance_threshold**: Success rate below which automatic stops are suppressed to avoid a false stop on an unresolved low base rate; ordinal groupings additionally report an inflated (floored) CI width (default: 0.01)
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
optstop-convergence --csv mydata.csv --output convergence_stats.csv --grouping_columns subject,task --sample_id_column item_id --epoch_column trial_num --score_column accuracy --delta_item 0.05 --delta_cap 0.05 --draws 1000 --tune 1000 --chains 4 --cores 4 --CI_delta 0.00001 --conservatism 5 --rep_batch_size 1 --pymc_refresh_every 2 --stab_window 15 --item_seqs 20 --epoch_seqs 20 --random_seed 42 --no_diagnostics --diagnostics_prefix my_convergence_eval
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
- **--CI_delta**: Slope threshold for determining CI stabilisation (default: 0.00001)
- **--conservatism**: Factor for rare event conservatism (default: 5)
- **--rep_batch_size**: Number of repetitions to process in each batch (default: 1)
- **--pymc_refresh_every**: How often to run the PyMC model (default: 2)
- **--stab_window**: Window size for assessing CI stabilisation (default: 15)
- **--item_seqs**: Number of randomised item orderings per grouping (default: 20)
- **--epoch_seqs**: Number of randomised epoch orderings per item (default: 20)
- **--random_seed**: Random seed for reproducible results (optional)
- **--low_performance_threshold**: Success rate below which automatic stops are suppressed to avoid a false stop on an unresolved low base rate; ordinal groupings additionally report an inflated (floored) CI width (default: 0.01)
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




