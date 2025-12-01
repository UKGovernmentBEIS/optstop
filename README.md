# optstop

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Adaptive Optimal Stopping Rule Algorithms for Efficient Data Collection and Analysis

## Overview

`optstop` is a Python package implementing adaptive optimal stopping algorithms for both post-hoc (batch) and live (incremental) data collection scenarios. It is designed to help researchers and practitioners efficiently determine when enough data has been collected to make reliable inferences, saving time and resources.

## Features
- Post-hoc (batch) optimal stopping for retrospective analysis and dataset pruning
- Live (incremental) optimal stopping for real-time data collection
- Flexible, parameterized stopping criteria
- **Flexible column mapping for groupings, sample IDs, and epochs**
- Bayesian and frequentist hybrid methodology
- **GPU acceleration support** via JAX/numpyro for significantly faster PyMC sampling
- **Ordinal scoring support** for ordinal data (e.g., 0-10) in addition to binary (0/1) scoring
- **Performance optimizations** for convergence analysis (~30% faster, see Performance Optimizations section)
- **inspect_ai integration** for LLM evaluation workflows with adaptive early stopping

## Using optstop with inspect_ai

`optstop` now provides seamless integration with [inspect_ai](https://inspect.aisi.org.uk/), the UK AI Safety Institute's framework for LLM evaluations. The `OptimalStoppingManager` implements the `EarlyStopping` protocol, enabling **statistically-rigorous adaptive early stopping** for your LLM evaluations.

### Why Use Optimal Stopping for LLM Evaluations?

Running LLM evaluations can be expensive and time-consuming:
- **API costs**: Each evaluation sample costs money (especially for large models)
- **Compute time**: Running multiple epochs per sample takes hours or days
- **Resource waste**: Running unnecessary trials after statistical confidence is achieved

Optimal stopping helps you **stop evaluating intelligently** when you have enough data, potentially saving:
- 30-70% of API costs
- 30-70% of evaluation time
- Maintains statistical validity and reliability

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

# Define your task as usual
task = Task(
    dataset=[Sample(input=q, target=a, id=f"q_{i}")
             for i, (q, a) in enumerate(questions)],
    solver=[
        system_message("You are a helpful assistant."),
        generate()
    ],
    scorer=model_graded_fact()
)

# Configure optimal stopping
optstop_params = {
    'delta_item': 0.05,       # CI width threshold for individual samples
    'delta_cap': 0.05,        # CI width threshold for overall task
    'cred_level': 0.95,       # 95% credible intervals
    'conservatism': 5,        # Conservative for rare events
}

stopping_manager = OptimalStoppingManager(
    optstop_params=optstop_params,
    grouping_columns=['model', 'task'],  # Group by model and task
    reanalysis_interval=10,              # Re-run inference every 10 samples
    min_samples_per_grouping=5           # Minimum before first analysis
)

# Run evaluation with early stopping
log = eval(
    task,
    model="openai/gpt-4",
    epochs=10,  # Plan 10 epochs per sample
    early_stopping=stopping_manager  # Enable optimal stopping
)

# Check efficiency gains
print(f"Efficiency: {log.early_stopping.efficiency_percent}%")
print(f"Stopped samples: {log.early_stopping.stopped_samples_count}")
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

#### 2. Multiple Score Handling

Handle evaluations with multiple scorers:

```python
# Extract specific score
OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model'],
    score_choice='accuracy'  # Use only the 'accuracy' score
)

# Aggregate multiple scores
OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model'],
    score_agg='mean'  # Average all scores (supports: mean, median, mode, max)
)
```

#### 3. Ordinal Scoring Support

For tasks with ordinal ratings (e.g., 1-5 stars, 0-10 confidence):

```python
OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    ordinal_tasks=['confidence', 'rating'],  # Tasks with ordinal scores
    ordinal_max_score=10,                    # Maximum score value
    ordinal_inference='hybrid'               # Recommended: balances speed/safety
)
```

**⚠️ Performance Note for Ordinal Discrete Tasks:**
Ordinal discrete inference (without score aggregation) uses entropy-based Bayesian models that are computationally intensive.

**Typical per-inference timing (CPU, ~100 completed trials):**
| Pathway | Time per Inference | Notes |
|---------|-------------------|-------|
| Binary discrete | ~3-4 seconds | Fast, suitable for real-time stopping |
| Continuous bounded | ~5-6 seconds | Fast, suitable for real-time stopping |
| Ordinal discrete (hybrid) | ~7-8 minutes | Slow, may bottleneck fast evaluations |

**GPU Recommendation:** If your ordinal-scored evaluation trials complete quickly (e.g., < 5 minutes per trial), the inference time may become a bottleneck. In these cases, **GPU acceleration is strongly recommended** to ensure stopping decisions arrive in time. GPU acceleration typically provides a **2-4× speedup** for MCMC sampling:

```python
# For fast ordinal evaluations, enable GPU
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model', 'task'],
    ordinal_tasks=['rating'],
    ordinal_inference='hybrid',
    gpu_ids=[0]  # Enable GPU acceleration (2-4× faster)
)
```

Alternatively, if GPU is unavailable, consider using `score_agg='mean'` to aggregate ordinal scores into continuous values, which routes to the much faster continuous bounded pathway (~6 seconds vs ~8 minutes per inference).

#### 4. Shadow Mode for A/B Testing

Compare performance with and without early stopping:

```python
# Run with shadow mode (runs all trials but tracks what *would* stop)
manager = OptimalStoppingManager(
    optstop_params=params,
    grouping_columns=['model'],
    shadow_mode=True  # Disables actual stopping
)

# After evaluation, check what would have stopped
print(f"Would have saved: {log.early_stopping.efficiency_percent}%")
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optstop_params` | Required | Dictionary of stopping parameters (delta_item, delta_cap, etc.) |
| `grouping_columns` | Required | List of columns for grouping decisions |
| `score_column` | 'score' | Column name for scores in compiled dataset |
| `sample_id_column` | 'sample_id' | Column name for sample IDs |
| `epoch_column` | 'epoch' | Column name for epoch numbers |
| `reanalysis_interval` | 10 | Run inference every N completed samples |
| `min_samples_per_grouping` | 5 | Minimum samples before first analysis |
| `ordinal_tasks` | None | List of task names using ordinal scoring |
| `ordinal_max_score` | 10 | Maximum score for ordinal tasks |
| `ordinal_inference` | 'hybrid' | Ordinal inference mode: 'modal', 'entropy', 'hybrid' |
| `gpu_ids` | None | List of GPU IDs to use (e.g., [0, 1]) |
| `max_workers` | None | Max parallel workers (auto if None) |
| `shadow_mode` | False | If True, run all trials but track stopping decisions |
| `score_choice` | None | Extract specific score by key name |
| `score_agg` | None | Aggregate scores: 'mean', 'median', 'mode', 'max' |
| `random_seed` | None | Random seed for reproducibility (auto-generates if not specified) |

**Additional `optstop_params` options:**
| Key | Default | Description |
|-----|---------|-------------|
| `entropy_stabilization_threshold` | 0.002 | Relative change threshold for ordinal entropy stabilization |

### Diagnostics and Efficiency Metrics

After evaluation, `complete_task()` returns comprehensive diagnostics:

```python
log = eval(task, model="gpt-4", epochs=10, early_stopping=manager)

diagnostics = log.early_stopping

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
```

### Best Practices

1. **Start conservative**: Use `delta_item=0.05`, `delta_cap=0.05` for high precision
2. **Test with shadow mode**: Run once with `shadow_mode=True` to see potential savings
3. **Set appropriate groupings**: More granular groupings = more targeted stopping
4. **Monitor efficiency**: If efficiency is 0%, your stopping criteria may be too strict
5. **Check compatibility**: Verify inspect_ai version with `optstop.check_inspect_ai_compatibility()`

### Compatibility

- **optstop version**: 0.2.1+ (reproducibility features require 0.2.1+)
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

- **Full bridge documentation**: See `BRIDGE_USAGE_GUIDE.md` (coming soon)
- **API reference**: See [`BRIDGE_API_REFERENCE.md`](BRIDGE_API_REFERENCE.md) - Complete parameter documentation and configuration guide
- **Testing summary**: See `BRIDGE_TESTING_SUMMARY.md` - Phase 1 testing results and validation
- **Version strategy**: See `VERSION_STRATEGY.md`
- **Development roadmap**: See `BRIDGE_TESTING_AND_DEVELOPMENT_ROADMAP.md`

### Example Evaluations

Coming soon in `examples/inspect_ai/`:
- Basic binary evaluation
- Ordinal rating tasks
- Multi-model comparisons
- Custom score aggregation
- Shadow mode A/B testing

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
params = {
    'use_gpu': True,          # Force GPU usage
    'use_gpu': False,         # Disable GPU acceleration
    # ... other parameters
}
```

#### CLI
```bash
# Force GPU usage (will fail if GPU unavailable)
optstop-posthoc --csv data.csv --output pruned.csv --force_gpu [other options]

# Disable GPU even if available
optstop-posthoc --csv data.csv --output pruned.csv --disable_gpu [other options]
```

### Performance Benefits
- **2-4x faster** sampling for typical workloads
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

## Performance Optimizations

`optstop` includes several performance optimizations that significantly improve processing speed, especially for convergence analysis with ordinal scoring.

### Convergence Analysis Optimizations

**Phase 1 Optimizations (Implemented)**:
- **Vectorized Category Probabilities**: Eliminated Python loops in ordinal probability computation using NumPy broadcasting (2-3x faster for this operation)
- **Smart Model Caching**: Reuses compiled PyMC model structure across randomization sequences within each grouping (eliminates 11 out of 12 redundant compilations)

**Performance Gains**:
- **~30% faster** convergence analysis overall (tested: 44s vs 63s on typical dataset)
- **Greater gains** for ordinal groupings using entropy or hybrid inference modes
- **Projected savings**: 100-500 seconds on larger datasets with typical parameters

**Key Benefits**:
- Zero functional changes (mathematical equivalence preserved)
- 100% backward compatible (no API changes)
- Randomization integrity maintained (no impact on re-sequencing processes)
- Parallelization-safe (worker-local caching)

**Example Performance Impact**:
```python
# Before optimizations: ~63 seconds for 4 groupings
# After optimizations:  ~44 seconds for 4 groupings
# Improvement: 30% faster

result = convergence_posthoc(
    df, params,
    grouping_columns=['student', 'task'],
    sample_id_column='item_id',
    epoch_column='trial_num',
    ordinal_tasks=['confidence'],
    ordinal_inference='hybrid',  # Benefits most from optimizations
    item_seqs=3, epoch_seqs=3
)
```

### When Optimizations Apply

The performance optimizations primarily benefit:
- **Convergence analysis** (`convergence_posthoc`) - ~30-40% faster
- **Ordinal scoring** with entropy or hybrid inference modes
- **Nested loops** (item_seqs × epoch_seqs) - more iterations = more benefit

Posthoc and live optimal stopping also benefit from vectorization (~5-20% faster for ordinal tasks), though the gains are smaller as they don't have the nested loop structure.

## Ordinal Scoring Support

`optstop` supports both **binary scoring** (0/1) and **ordinal scoring** (e.g., 0-10) for both post-hoc and live optimal stopping modes.

### Overview
- **Binary scoring**: Traditional 0/1 success/failure data (default)
- **Ordinal scoring**: Likert scale data (e.g., confidence ratings, difficulty ratings on 0-10 scale)
- **Mixed datasets**: Seamlessly handle both binary and ordinal groupings in the same analysis

### Ordinal Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ordinal_tasks` | List[str] or None | None | Substrings to identify ordinal groupings (e.g., `['confidence', 'rating']`) |
| `ordinal_max_score` | int | 10 | Maximum score for ordinal data (e.g., 10 for 0-10 scale) |
| `ordinal_inference` | str | 'modal' | Inference method: `'modal'`, `'entropy'`, or `'hybrid'` (recommended) |
| `entropy_threshold` | float | 1.5 | Threshold for entropy validation in hybrid mode (prevents false peaks) |

### Inference Modes

**Modal Mode** (Fast):
- Bootstrap-based modal category estimation
- Suitable for clearly peaked ordinal distributions
- Typical processing time: seconds per grouping

**Entropy Mode** (Conservative):
- OrderedLogistic Bayesian model with full distribution entropy
- Suitable for diffuse ordinal distributions
- Typical processing time: minutes per grouping

**Hybrid Mode** (RECOMMENDED):
- **Pathway 1**: Modal CI narrow + entropy validation (peaked data) → Fast stopping
- **Pathway 2**: Entropy stabilization (diffuse data) → Safe stopping
- Prevents false peaks via `entropy_threshold`
- Balances efficiency and safety

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
    'cred_level': 0.95,
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
    entropy_threshold=1.5                                  # False peak detection
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
    'cred_level': 0.95,
}

result = optimal_stopping_live(
    df, params,
    grouping_columns=['student', 'task'],
    sample_id_column='item_id',
    epoch_column='trial_num',
    score_column='score',

    # Ordinal parameters:
    ordinal_tasks=['confidence'],     # Identify ordinal groupings
    ordinal_max_score=10,            # 0-10 scale
    ordinal_inference='hybrid',      # RECOMMENDED
    entropy_threshold=1.5            # False peak detection
)

print(result['stop_sample_ids'])  # Item IDs that reached stopping criteria
print(result['stop_task'])        # Groupings that reached stopping criteria
```

### How It Works

1. **Automatic Detection**: The package uses substring matching to identify ordinal groupings
   - Example: If `ordinal_tasks=['confidence']`, any grouping containing "confidence" (case-insensitive) uses ordinal scoring
   - All other groupings use binary scoring

2. **Score Validation**: Ordinal scores are validated to be in range [0, `ordinal_max_score`]

3. **Independent Processing**: Each grouping is processed with the correct scoring method (no cross-contamination)

4. **False Peak Detection**: In hybrid mode, `entropy_threshold` prevents premature stopping on diffuse data

### Ordinal Support Status

| Function | Ordinal Support | Notes |
|----------|----------------|-------|
| `optimal_stopping_posthoc` | ✅ Full support | All inference modes available |
| `optimal_stopping_live` | ✅ Full support | All inference modes available |
| `convergence_posthoc` | ✅ Full support | All inference modes available |

### CLI Usage with Ordinal Data

```bash
# Post-hoc mode
optstop-posthoc --csv data.csv --output pruned.csv \
  --grouping_columns student,task \
  --sample_id_column item_id \
  --epoch_column trial_num \
  --score_column score \
  --ordinal_tasks confidence,rating \
  --ordinal_max_score 10 \
  --ordinal_inference hybrid \
  --entropy_threshold 1.5

# Live mode
optstop-live --csv current_data.csv \
  --grouping_columns student,task \
  --sample_id_column item_id \
  --epoch_column trial_num \
  --score_column score \
  --ordinal_tasks confidence,rating \
  --ordinal_max_score 10 \
  --ordinal_inference hybrid \
  --entropy_threshold 1.5
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

Or install JAX with GPU support manually:
```bash
pip install .
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**GPU behavior:**
- All inference types (binary, continuous, ordinal) use numpyro on GPU
- Provides 2-4× speedup for typical workloads

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
| `cred_level`             | 0.95      | Both         | Credibility level for intervals (e.g., 0.95 for 95% CI)                     |
| `conservatism`           | 5         | Both         | Factor for rare event conservatism (higher = more conservative)             |
| `low_performance_threshold` | 0.01   | Both         | Below this success rate, use conservative stopping                          |
| `draws`                  | 6000      | Both         | Number of MCMC samples for PyMC (affects speed/accuracy)                    |
| `tune`                   | 6000      | Both         | Number of tuning steps for PyMC                                             |
| `chains`                 | 4         | Both         | Number of MCMC chains for PyMC                                              |
| `cores`                  | 4         | Both         | Number of CPU cores for PyMC sampling                                       |
| `rep_batch_size`         | 1         | Both         | Number of repetitions to process in each batch                              |
| `pymc_refresh_every`     | 2         | Both         | How often to run the PyMC model (every N items)                             |
| `stab_window`            | 10         | Both         | Window size for assessing CI stabilization                                  |
| `CI_delta`               | 0.00005    | Both         | Slope threshold for determining CI stabilization                            |
| `use_gpu`                | Auto      | Both         | Enable/disable GPU acceleration (True/False, auto-detected if not set)     |
| `force_gpu`              | False     | Both         | Force GPU usage, fail if unavailable (for CLI --force_gpu)                 |
| `ordinal_tasks`          | None      | All          | List of substrings to identify ordinal groupings (e.g., ['confidence'])   |
| `ordinal_max_score`      | 10        | All          | Maximum score for ordinal data (e.g., 10 for 0-10 scale)                 |
| `ordinal_inference`      | 'modal'   | All          | Inference method: 'modal', 'entropy', or 'hybrid' (RECOMMENDED)           |
| `entropy_threshold`      | 1.5       | All          | Threshold for entropy validation in hybrid mode (prevents false peaks)    |

### Example: Setting Parameters

You can set any combination of these in your `params` dict. For example:

```python
params = {
    'delta_item': 0.05,                # Require a narrower CI for stopping items
    'delta_cap': 0.05,                 # Task/grouping CI width threshold
    'cred_level': 0.95,               # Use 95% credible intervals
    'conservatism': 5,                # More conservative for rare events
    'low_performance_threshold': 0.01,# Adjust threshold for low performance
    'draws': 6000,                    # (post-hoc only) MCMC samples for speed
    'tune': 6000,                     # (post-hoc only) Tuning steps
    'chains': 4,                      # Number of MCMC chains
    'cores': 4,                       # Number of CPU cores for sampling
    'rep_batch_size': 2,              # (post-hoc only) Process 2 reps at a time
    'pymc_refresh_every': 1,          # (post-hoc only) Run PyMC every item
    'stab_window': 10,                 # (post-hoc only) Use a smaller stabilization window
    'CI_delta': 0.0001,               # (post-hoc only) Stricter stabilization threshold
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
    'draws': 6000,
    'tune': 6000,
    'chains': 4,
    'cores': 4,
    'CI_delta': 0.00005,
    'conservatism': 5,
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
    'draws': 6000,
    'tune': 6000,
    'chains': 4,
    'cores': 4,
    'CI_delta': 0.00005,
    'conservatism': 5,
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
    'draws': 6000,
    'tune': 6000,
    'chains': 4,
    'cores': 4,
    'stab_window': 10,
    'CI_delta': 0.00005,
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

    # Ordinal parameters (benefits from Phase 1 optimizations):
    ordinal_tasks=['confidence'],    # Identify ordinal groupings
    ordinal_max_score=10,            # 0-10 scale
    ordinal_inference='hybrid',      # RECOMMENDED (fastest with optimizations)
    entropy_threshold=1.5            # False peak detection
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

### Best Practices & Recommendations
- **Specifying Groupings:**
  - Your DataFrame must include the columns you specify for grouping, sample ID, epoch, and score (see Flexible Column Mapping section above).
  - For parallelization to be effective, ensure your data contains multiple groupings and/or tasks.
- **Recommended Parameter Settings:**
  - `draws` & `tune`: Use at least 1000 for real analyses; lower values are for testing only.
  - `delta_item` & `delta_cap`: 0.05 for high precision, 0.1 for faster but less precise stopping.
  - `CI_delta`: 0.0002 for stable CI slope; increase for earlier stopping.
  - `conservatism`: 2 is typical; increase for more caution in low-performance scenarios.
- **Reproducibility:**
  - Set a random seed (e.g., `np.random.seed(42)`) for reproducible results.
- **Logging:**
  - Use `configure_optstop_logging()` to log all stopping decisions to file (console output is suppressed by default).

### Example: Recommended Parameters
```python
params = {
    'delta_item': 0.05,      # High precision for items
    'delta_cap': 0.05,       # High precision for group/task
    'draws': 6000,           # Minimum recommended for inference
    'tune': 6000,            # Minimum recommended for inference
    'CI_delta': 0.00005,      # Require stable CI slope
    'conservatism': 5,       # Typical value
    # ... other parameters as needed ...
}
```

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
    'draws': 6000,
    'tune': 6000,
    'CI_delta': 0.00005,
    'conservatism': 5,
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
    'draws': 6000,
    'tune': 6000,
    'CI_delta': 0.00005,
    'conservatism': 5,
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
  - Returns a DataFrame detailing the required numbers of trials at epoch and sample_ID levels, split by the users desired grouping.
- **Parameters:**
  - Accepts the same `params` dictionary as other functions (see table above), plus:
    - `item_seqs`: Number of randomized item orderings per grouping (default: 20)
    - `epoch_seqs`: Number of randomized epoch orderings per item (default: 20)
- **Returns:**
  - A DataFrame with one row per grouping-task, containing detailed convergence statistics and summary metrics.

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

    # Ordinal parameters (benefits from ~30% performance boost):
    ordinal_tasks=['confidence'],    # Identify ordinal groupings
    ordinal_max_score=10,            # 0-10 scale
    ordinal_inference='hybrid',      # RECOMMENDED (fastest with optimizations)
    entropy_threshold=1.5            # False peak detection
)
print(result)
```

The resulting DataFrame contains all the convergence metrics for each grouping-task. The analysis is parallelized for speed and includes Phase 1 performance optimizations (~30% faster for ordinal tasks).

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

### Recommended Parameter Settings
- **draws & tune:** For reliable Bayesian inference, use at least `draws=3000` and `tune=3000` (per chain) for real analyses. Lower values (e.g., 50) are only for quick tests or debugging.
- **chains & cores:** Default values of `chains=4` and `cores=4` are suitable for most analyses. Increase `chains` for more robust MCMC sampling and `cores` for faster parallel sampling (up to your system's CPU core count).
- **CI width thresholds:**
  - `delta_item`: 0.05 is a common choice for high precision; 0.1 is more lenient.
  - `delta_cap`: 0.05 for group-level precision; increase for faster but less precise stopping.
- **Stabilization parameter:**
  - `CI_delta` controls how stable the CI slope must be before stopping. Smaller values (e.g., 0.0002) require more stability; larger values allow earlier stopping.
- **Conservatism:**
  - `conservatism=5` is typical; increase for more caution in low-performance scenarios.
- **Reproducibility:**
  - For reproducible pruned DataFrames, set a random seed before running your analysis (e.g., `np.random.seed(42)` or pass `random_seed` in params).
  - **Note:** Due to the stochastic nature of MCMC and parallelization, summary statistics (e.g., CI bounds, widths) are not guaranteed to be bitwise reproducible, even with the same random seed. Only the pruned DataFrame is guaranteed to be reproducible; summary values may differ slightly between runs.

### Example: Recommended Parameters
```python
params = {
    'delta_item': 0.05,      # High precision for items
    'delta_cap': 0.05,       # High precision for group/task
    'draws': 6000,           # Minimum recommended for inference
    'tune': 6000,            # Minimum recommended for inference
    'chains': 4,             # Number of MCMC chains
    'cores': 4,              # Number of CPU cores for sampling
    'CI_delta': 0.00005,      # Require stable CI slope
    'conservatism': 5,       # Typical value
    # ... other parameters as needed ...
}
```

See the parameter table above for all options and defaults.

## License
MIT 

## Minimal Working Example

Here is a minimal example to get started with optimal stopping:

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
    'draws': 6000,
    'tune': 6000,
    'chains': 4,
    'cores': 4,
    'CI_delta': 0.0005,
    'conservatism': 5,
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

## FAQ & Troubleshooting

**Q: I get a ValueError about missing columns.**
- Make sure your DataFrame includes all required columns that you specify for grouping, sample ID, epoch, and score.

**Q: My summary statistics are not exactly reproducible, even with the same random seed.**
- This is expected due to the stochastic nature of MCMC and parallelization. Only the pruned DataFrame is guaranteed to be reproducible; summary values may differ slightly between runs.

**Q: I get PyMC or sampling errors (e.g., "Too few samples", "NUTS initialization failed").**
- Increase `draws` and `tune` to at least 3000 for real analyses. For small test runs, warnings are expected.
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
optstop-posthoc --csv mydata.csv --output pruned.csv --summary summary.csv --grouping_columns subject,task --sample_id_column item_id --epoch_column trial_num --score_column accuracy --delta_item 0.05 --delta_cap 0.05 --draws 6000 --tune 6000 --chains 4 --cores 4 --CI_delta 0.00005 --conservatism 5 --rep_batch_size 1 --pymc_refresh_every 2 --stab_window 10 --random_seed 42 --generate_diagnostics --diagnostics_prefix my_diagnostics
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
- **--draws**: Number of MCMC samples for PyMC (default: 6000)
- **--tune**: Number of tuning steps for PyMC (default: 6000)
- **--chains**: Number of MCMC chains for PyMC (default: 4)
- **--cores**: Number of CPU cores for PyMC (default: 4)
- **--CI_delta**: Slope threshold for determining CI stabilization (default: 0.00005)
- **--conservatism**: Factor for rare event conservatism (default: 5)
- **--rep_batch_size**: Number of repetitions to process in each batch (default: 1)
- **--pymc_refresh_every**: How often to run the PyMC model (default: 2)
- **--stab_window**: Window size for assessing CI stabilization (default: 10)
- **--random_seed**: Random seed for reproducible results (optional)
- **--low_performance_threshold**: Success rate below which conservative stopping is applied (default: 0.01)
- **--disable_gpu**: Disable GPU acceleration even if available
- **--force_gpu**: Force GPU usage (will fail if GPU unavailable)
- **--ordinal_tasks**: Comma-separated list of substrings to identify ordinal groupings (e.g., "confidence,rating")
- **--ordinal_max_score**: Maximum score for ordinal data (default: 10)
- **--ordinal_inference**: Ordinal inference method: modal, entropy, or hybrid (default: modal)
- **--entropy_threshold**: Entropy threshold for false peak detection in hybrid mode (default: 1.5)

### 2. Live Optimal Stopping
**Command:**
```
optstop-live --csv current_data.csv --grouping_columns subject --sample_id_column item_id --epoch_column trial_num --score_column accuracy --delta_item 0.05 --delta_cap 0.05 --draws 6000 --tune 6000 --chains 4 --cores 4 --CI_delta 0.00005 --conservatism 5 --rep_batch_size 1 --pymc_refresh_every 2 --stab_window 10 --random_seed 42
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
- **--draws**: Number of MCMC samples for PyMC (default: 6000)
- **--tune**: Number of tuning steps for PyMC (default: 6000)
- **--chains**: Number of MCMC chains for PyMC (default: 4)
- **--cores**: Number of CPU cores for PyMC (default: 4)
- **--CI_delta**: Slope threshold for determining CI stabilization (default: 0.00005)
- **--conservatism**: Factor for rare event conservatism (default: 5)
- **--rep_batch_size**: Number of repetitions to process in each batch (default: 1)
- **--pymc_refresh_every**: How often to run the PyMC model (default: 2)
- **--stab_window**: Window size for assessing CI stabilization (default: 10)
- **--random_seed**: Random seed for reproducible results (optional)
- **--low_performance_threshold**: Success rate below which conservative stopping is applied (default: 0.01)
- **--disable_gpu**: Disable GPU acceleration even if available
- **--force_gpu**: Force GPU usage (will fail if GPU unavailable)
- **--ordinal_tasks**: Comma-separated list of substrings to identify ordinal groupings (e.g., "confidence,rating")
- **--ordinal_max_score**: Maximum score for ordinal data (default: 10)
- **--ordinal_inference**: Ordinal inference method: modal, entropy, or hybrid (default: modal)
- **--entropy_threshold**: Entropy threshold for false peak detection in hybrid mode (default: 1.5)
- Prints which sample IDs (with grouping prefix) and/or groupings can be stopped.

### 3. Convergence Analysis
**Command:**
```
optstop-convergence --csv mydata.csv --output convergence_stats.csv --grouping_columns subject,task --sample_id_column item_id --epoch_column trial_num --score_column accuracy --delta_item 0.05 --delta_cap 0.05 --draws 6000 --tune 6000 --chains 4 --cores 4 --CI_delta 0.00005 --conservatism 5 --rep_batch_size 1 --pymc_refresh_every 2 --stab_window 10 --item_seqs 20 --epoch_seqs 20 --random_seed 42 --no_diagnostics --diagnostics_prefix my_convergence_eval
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
- **--draws**: Number of MCMC samples for PyMC (default: 6000)
- **--tune**: Number of tuning steps for PyMC (default: 6000)
- **--chains**: Number of MCMC chains for PyMC (default: 4)
- **--cores**: Number of CPU cores for PyMC (default: 4)
- **--CI_delta**: Slope threshold for determining CI stabilization (default: 0.00005)
- **--conservatism**: Factor for rare event conservatism (default: 5)
- **--rep_batch_size**: Number of repetitions to process in each batch (default: 1)
- **--pymc_refresh_every**: How often to run the PyMC model (default: 2)
- **--stab_window**: Window size for assessing CI stabilization (default: 10)
- **--item_seqs**: Number of randomized item orderings per grouping (default: 20)
- **--epoch_seqs**: Number of randomized epoch orderings per item (default: 20)
- **--random_seed**: Random seed for reproducible results (optional)
- **--low_performance_threshold**: Success rate below which conservative stopping is applied (default: 0.01)
- **--ordinal_tasks**: Comma-separated list of substrings to identify ordinal groupings (e.g., "confidence,rating")
- **--ordinal_max_score**: Maximum score for ordinal data (default: 10)
- **--ordinal_inference**: Ordinal inference method: modal, entropy, or hybrid (default: modal)
- **--entropy_threshold**: Entropy threshold for false peak detection in hybrid mode (default: 1.5)
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
- **Use recommended parameter values** for real analyses (see Best Practices above).
- **Check the log file** for detailed stopping decisions, errors, and parameter validation.
- **For large datasets, run on a machine with sufficient CPU and memory.**
- **If you encounter errors, check the FAQ and log file for troubleshooting tips.** 

**Note:** After running any of the main functions (`optimal_stopping_posthoc`, `optimal_stopping_live`, or `convergence_posthoc`), you will see a message printed to the console reminding you where to find the log file with all details and warnings. This log file contains all stopping decisions, errors, and PyMC warnings, even if the terminal output is quiet. 
