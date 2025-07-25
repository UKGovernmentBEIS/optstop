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

## Installation

```bash
pip install .
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
| `cred_level`             | 0.95      | Both         | Credibility level for intervals (e.g., 0.95 for 95% CI)                     |
| `conservatism`           | 2         | Both         | Factor for rare event conservatism (higher = more conservative)             |
| `low_performance_threshold` | 0.1    | Both         | Below this success rate, use conservative stopping                          |
| `draws`                  | 3000      | Post-hoc     | Number of MCMC samples for PyMC (affects speed/accuracy)                    |
| `tune`                   | 3000      | Post-hoc     | Number of tuning steps for PyMC                                             |
| `rep_batch_size`         | 1         | Post-hoc     | Number of repetitions to process in each batch                              |
| `pymc_refresh_every`     | 2         | Post-hoc     | How often to run the PyMC model (every N items)                             |
| `stab_window`            | 5         | Post-hoc     | Window size for assessing CI stabilization                                  |
| `CI_delta`               | 0.0002    | Post-hoc     | Slope threshold for determining CI stabilization                            |

### Example: Setting Parameters

You can set any combination of these in your `params` dict. For example:

```python
params = {
    'delta_item': 0.1,                # Require a narrower CI for stopping items
    'delta_cap': 0.2,                 # Task/grouping CI width threshold
    'cred_level': 0.99,               # Use 99% credible intervals
    'conservatism': 3,                # More conservative for rare events
    'low_performance_threshold': 0.15,# Adjust threshold for low performance
    'draws': 1000,                    # (post-hoc only) Fewer MCMC samples for speed
    'tune': 1000,                     # (post-hoc only) Fewer tuning steps
    'rep_batch_size': 2,              # (post-hoc only) Process 2 reps at a time
    'pymc_refresh_every': 1,          # (post-hoc only) Run PyMC every item
    'stab_window': 3,                 # (post-hoc only) Use a smaller stabilization window
    'CI_delta': 0.0001,               # (post-hoc only) Stricter stabilization threshold
}
```

**Note:**
- Parameters not relevant to live mode (e.g., `draws`, `tune`, `stab_window`, etc.) will be ignored if passed to `optimal_stopping_live`.
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
optstop-posthoc --csv mydata.csv --output pruned.csv --grouping_columns subject,task --sample_id_column item_id --epoch_column trial_num --generate_diagnostics --diagnostics_prefix my_diagnostics
```

## Flexible Column Mapping (NEW)

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
    'CI_delta': 0.0002,
    'conservatism': 2,
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
    'CI_delta': 0.0002,
    'conservatism': 2,
}

result = optimal_stopping_live(
    df, params,
    grouping_columns='subject',
    sample_id_column='item_id',
    epoch_column='trial_num',
    score_column='score'
)
print(result)
```

### Example Usage (Convergence)
```python
import pandas as pd
from optstop.rule import configure_optstop_logging
from optstop import convergence_posthoc

configure_optstop_logging('my_convergence_log.txt', console_output=False)

df = pd.DataFrame({
    'subject': [1, 1, 2, 2],
    'task': [1, 1, 1, 1],
    'item_id': [1, 1, 2, 2],
    'trial_num': [1, 2, 1, 2],
    'score': [1, 0, 1, 1],
})

params = {
    'delta_item': 0.5,
    'delta_cap': 0.5,
    'draws': 50,
    'tune': 50,
    'stab_window': 2,
    'CI_delta': 0.01,
    'rep_batch_size': 1,
    'pymc_refresh_every': 1,
    'item_seqs': 2,
    'epoch_seqs': 2
}

result = convergence_posthoc(
    df, params,
    grouping_columns=['subject', 'task'],
    sample_id_column='item_id',
    epoch_column='trial_num',
    score_column='score'
)
print(result)
```

## Optimal Stopping (Post-hoc & Live)

The package provides functions for adaptive optimal stopping, allowing you to determine when enough data has been collected for reliable inference, either retrospectively (post-hoc) or during live data collection.

### Parallelization (Post-hoc)
- The `optimal_stopping_posthoc` function now parallelizes across groupings (unique combinations of `grouping_num` and `task_num`), using all available CPU cores for efficient processing of large datasets.
- Each grouping-task is processed independently and in parallel, with results aggregated at the end.

### Best Practices & Recommendations
- **Specifying Groupings:**
  - Your DataFrame must include `grouping_num`, `task_num`, `sample_id_num`, `epoch`, and `score`.
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
    'draws': 1000,           # Minimum recommended for inference
    'tune': 1000,            # Minimum recommended for inference
    'CI_delta': 0.0002,      # Require stable CI slope
    'conservatism': 2,       # Typical value
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
    'grouping_num': [1]*8 + [2]*8,
    'task_num': [1]*4 + [2]*4 + [1]*4 + [2]*4,
    'sample_id_num': [1,1,2,2,1,1,2,2]*2,
    'epoch': [1,2,1,2]*4,
    'score': [1,0,1,1,0,1,1,0,1,1,0,1,0,1,1,1],
})

params = {
    'delta_item': 0.05,
    'delta_cap': 0.05,
    'draws': 1000,
    'tune': 1000,
    'CI_delta': 0.0002,
    'conservatism': 2,
}

pruned_df, summary = optimal_stopping_posthoc(df, params)
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
    'grouping_num': [1]*4,
    'task_num': [1]*4,
    'sample_id_num': [1,1,2,2],
    'epoch': [1,2,1,2],
    'score': [1,0,1,1],
})

params = {
    'delta_item': 0.05,
    'delta_cap': 0.05,
    'draws': 1000,
    'tune': 1000,
    'CI_delta': 0.0002,
    'conservatism': 2,
}

result = optimal_stopping_live(df, params)
print(result)
```

## Convergence Analysis (Post-hoc)

The package provides a function for post-hoc convergence analysis, allowing you to assess when statistical stability would have been achieved in sequential testing scenarios. This is useful for determining the minimum data required for reliable performance estimates.

### Function: `convergence_posthoc`

- **Purpose:**
  - Runs a post-hoc convergence analysis on a full dataset, parallelizing across groupings (using all available CPU cores).
  - Returns a DataFrame matching the structure of the original `convergence_stats_v6.csv` output.
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

df = pd.DataFrame({
    'grouping_num': [1]*8 + [2]*8,
    'grouping': ['A']*8 + ['B']*8,
    'task_num': [1]*16,
    'sample_id_num': [1,1,1,1,2,2,2,2,1,1,1,1,2,2,2,2],
    'epoch': [1,2,3,4,1,2,3,4]*2,
    'score': [1,0,1,1,0,1,1,0,1,1,0,1,0,1,1,1],
})

params = {
    'delta_item': 0.5,
    'delta_cap': 0.5,
    'draws': 50,
    'tune': 50,
    'stab_window': 2,
    'CI_delta': 0.01,
    'rep_batch_size': 1,
    'pymc_refresh_every': 1,
    'item_seqs': 2,
    'epoch_seqs': 2
}

result = convergence_posthoc(df, params)
print(result)
```

The resulting DataFrame contains all the convergence metrics for each grouping-task, and the analysis is parallelized for speed.

## Convergence Diagnostics (NEW)

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
optstop-convergence --csv mydata.csv --output convergence_stats.csv --grouping_columns subject,task --sample_id_column item_id --epoch_column trial_num --no_diagnostics --diagnostics_prefix my_convergence_eval
```

- By default, diagnostics are ON. Use `--no_diagnostics` to disable.
- The output files will be named `my_convergence_eval_grouped_needed.png` and `my_convergence_eval_score_scatter.png`.

See the CLI help (`optstop-convergence --help`) for all options.

## Best Practices & Recommendations

### Specifying Groupings
- **Required columns:** Your input DataFrame must include `grouping_num`, `task_num`, `sample_id_num`, `epoch`, and `score`.
- **Group labels:** If you want human-readable group labels in your output, include a `grouping` column (e.g., system/model name) in addition to `grouping_num`.
- **Parallelization:** Each unique combination of `grouping_num` and `task_num` will be processed in parallel, so ensure these columns are set appropriately for your experimental design.

### Recommended Parameter Settings
- **draws & tune:** For reliable Bayesian inference, use at least `draws=1000` and `tune=1000` (per chain) for real analyses. Lower values (e.g., 50) are only for quick tests or debugging.
- **CI width thresholds:**
  - `delta_item`: 0.05 is a common choice for high precision; 0.1 is more lenient.
  - `delta_cap`: 0.05 for group-level precision; increase for faster but less precise stopping.
- **Stabilization parameter:**
  - `CI_delta` controls how stable the CI slope must be before stopping. Smaller values (e.g., 0.0002) require more stability; larger values allow earlier stopping.
- **Conservatism:**
  - `conservatism=2` is typical; increase for more caution in low-performance scenarios.
- **Reproducibility:**
  - For reproducible pruned DataFrames, set a random seed before running your analysis (e.g., `np.random.seed(42)` or pass `random_seed` in params).
  - **Note:** Due to the stochastic nature of MCMC and parallelization, summary statistics (e.g., CI bounds, widths) are not guaranteed to be bitwise reproducible, even with the same random seed. Only the pruned DataFrame is guaranteed to be reproducible; summary values may differ slightly between runs.

### Example: Recommended Parameters
```python
params = {
    'delta_item': 0.05,      # High precision for items
    'delta_cap': 0.05,       # High precision for group/task
    'draws': 1000,           # Minimum recommended for inference
    'tune': 1000,            # Minimum recommended for inference
    'CI_delta': 0.0002,      # Require stable CI slope
    'conservatism': 2,       # Typical value
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
# Required columns: grouping_num, task_num, sample_id_num, epoch, score
df = pd.DataFrame({
    'grouping_num': [1, 1, 1, 1],
    'task_num': [1, 1, 1, 1],
    'sample_id_num': [1, 1, 2, 2],
    'epoch': [1, 2, 1, 2],
    'score': [1, 0, 1, 1],
})

params = {
    'delta_item': 0.05,
    'delta_cap': 0.05,
    'draws': 1000,
    'tune': 1000,
    'CI_delta': 0.0002,
    'conservatism': 2,
    'random_seed': 42
}

pruned_df, summary = optimal_stopping_posthoc(df, params)
print(pruned_df)
print(summary)
```

## FAQ & Troubleshooting

**Q: I get a ValueError about missing columns.**
- Make sure your DataFrame includes all required columns: `grouping_num`, `task_num`, `sample_id_num`, `epoch`, and `score`.

**Q: My summary statistics are not exactly reproducible, even with the same random seed.**
- This is expected due to the stochastic nature of MCMC and parallelization. Only the pruned DataFrame is guaranteed to be reproducible; summary values may differ slightly between runs.

**Q: I get PyMC or sampling errors (e.g., "Too few samples", "NUTS initialization failed").**
- Increase `draws` and `tune` to at least 1000 for real analyses. For small test runs, warnings are expected.
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
optstop-posthoc --csv mydata.csv --output pruned.csv --summary summary.csv --grouping_columns subject,task --sample_id_column item_id --epoch_column trial_num --score_column accuracy --delta_item 0.05 --delta_cap 0.05 --draws 1000 --tune 1000 --CI_delta 0.0002 --conservatism 2 --rep_batch_size 1 --pymc_refresh_every 2 --stab_window 5 --random_seed 42 --generate_diagnostics --diagnostics_prefix my_diagnostics
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
- **--CI_delta**: Slope threshold for determining CI stabilization (default: 0.0002)
- **--conservatism**: Factor for rare event conservatism (default: 2)
- **--rep_batch_size**: Number of repetitions to process in each batch (default: 1)
- **--pymc_refresh_every**: How often to run the PyMC model (default: 2)
- **--stab_window**: Window size for assessing CI stabilization (default: 5)
- **--random_seed**: Random seed for reproducible results (optional)

### 2. Live Optimal Stopping
**Command:**
```
optstop-live --csv current_data.csv --grouping_columns subject --sample_id_column item_id --epoch_column trial_num --score_column accuracy --delta_item 0.05 --delta_cap 0.05 --draws 1000 --tune 1000 --CI_delta 0.0002 --conservatism 2 --rep_batch_size 1 --pymc_refresh_every 2 --stab_window 5 --random_seed 42
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
- **--CI_delta**: Slope threshold for determining CI stabilization (default: 0.0002)
- **--conservatism**: Factor for rare event conservatism (default: 2)
- **--rep_batch_size**: Number of repetitions to process in each batch (default: 1)
- **--pymc_refresh_every**: How often to run the PyMC model (default: 2)
- **--stab_window**: Window size for assessing CI stabilization (default: 5)
- **--random_seed**: Random seed for reproducible results (optional)
- Prints which sample IDs and/or tasks can be stopped.

### 3. Convergence Analysis
**Command:**
```
optstop-convergence --csv mydata.csv --output convergence_stats.csv --grouping_columns subject,task --sample_id_column item_id --epoch_column trial_num --score_column accuracy --delta_item 0.05 --delta_cap 0.05 --draws 1000 --tune 1000 --CI_delta 0.0002 --conservatism 2 --rep_batch_size 1 --pymc_refresh_every 2 --stab_window 5 --item_seqs 20 --epoch_seqs 20 --random_seed 42 --no_diagnostics --diagnostics_prefix my_convergence_eval
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
- **--CI_delta**: Slope threshold for determining CI stabilization (default: 0.0002)
- **--conservatism**: Factor for rare event conservatism (default: 2)
- **--rep_batch_size**: Number of repetitions to process in each batch (default: 1)
- **--pymc_refresh_every**: How often to run the PyMC model (default: 2)
- **--stab_window**: Window size for assessing CI stabilization (default: 5)
- **--item_seqs**: Number of randomized item orderings per grouping (default: 20)
- **--epoch_seqs**: Number of randomized epoch orderings per item (default: 20)
- **--random_seed**: Random seed for reproducible results (optional)

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
