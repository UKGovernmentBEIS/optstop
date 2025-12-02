# optstop.early_stopping

Bayesian optimal stopping for LLM evaluations.

---

## OptimalStoppingManager

Implementation of the `inspect_ai.util.EarlyStopping` protocol for statistically-rigorous adaptive early stopping.

```python
class OptimalStoppingManager(
    optstop_params: dict[str, Any],
    grouping_columns: list[str],
    reanalysis_interval: int = 10,
    min_samples_per_grouping: int = 5,
    ordinal_tasks: list[str] | None = None,
    ordinal_max_score: int = 10,
    ordinal_inference: str = "hybrid",
    ordinal_model_type: str = "ordered_logistic",
    gpu_ids: list[int] | None = None,
    manager_name: str = "optstop",
    shadow_mode: bool = False,
    score_choice: str | None = None,
    score_agg: str | None = None,
    random_seed: int | None = None,
)
```

### Parameters

**optstop_params** `dict[str, Any]`

Dictionary of optimal stopping parameters. Common keys:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `delta_item` | `float` | `0.05` | Maximum CI width for individual samples |
| `delta_cap` | `float` | `0.05` | Maximum CI width for groupings |
| `cred_level` | `float` | `0.95` | Credibility level for confidence intervals |
| `conservatism` | `int` | `5` | Conservatism factor for rare events |
| `draws` | `int` | `6000` | MCMC samples (reduce for faster inference) |
| `tune` | `int` | `6000` | MCMC tuning steps |
| `chains` | `int` | `4` | MCMC chains |

**grouping_columns** `list[str]`

Columns for independent stopping decisions. Valid values: `'model'`, `'task'`, `'metadata.<key>'`, `'tag.<name>'`.

**reanalysis_interval** `int`

Run inference every N completed samples per grouping. Default: `10`.

**min_samples_per_grouping** `int`

Minimum samples before first inference. Default: `5`.

**ordinal_tasks** `list[str] | None`

Task name substrings using ordinal scoring (0 to `ordinal_max_score`). Tasks matching these patterns route to ordinal inference. Default: `None`.

**ordinal_max_score** `int`

Maximum ordinal score value. Default: `10`.

**ordinal_inference** `str`

Inference mode for ordinal tasks. Options: `'modal'` (fast, bootstrap-based), `'entropy'` (slow, full Bayesian), `'hybrid'` (automatic selection). Default: `"hybrid"`.

**ordinal_model_type** `str`

Statistical model for ordinal inference. Options: `'ordered_logistic'` (cumulative link model, faster), `'dirichlet'` (Dirichlet-Multinomial, more robust for sparse data). Default: `"ordered_logistic"`.

**gpu_ids** `list[int] | None`

GPU device IDs for inference acceleration. A non-empty list enables GPU detection. Default: `None` (CPU only).

**manager_name** `str`

Identifier for logging and thread naming. Default: `"optstop"`.

**shadow_mode** `bool`

If `True`, run all trials without stopping (for comparison). Default: `False`.

**score_choice** `str | None`

Extract specific score by key name. Mutually exclusive with `score_agg`. Default: `None`.

**score_agg** `str | None`

Aggregate multiple scores. Options: `'mean'`, `'median'`, `'mode'`, `'max'`. Routes to continuous inference when `'mean'` or `'median'`. Mutually exclusive with `score_choice`. Default: `None`.

**random_seed** `int | None`

MCMC random seed for reproducibility. If `None`, auto-generated and logged. Default: `None`.

### Methods

#### start_task

```python
async def start_task(
    self,
    task: EvalSpec,
    samples: list[Sample],
    epochs: int,
) -> str
```

Initialize evaluation with full trial plan.

**task** `EvalSpec` — Evaluation specification from inspect_ai.

**samples** `list[Sample]` — Samples to evaluate.

**epochs** `int` — Epochs per sample.

**Returns:** `str` — Manager name.

#### schedule_sample

```python
async def schedule_sample(
    self,
    id: str | int,
    epoch: int,
) -> EarlyStop | None
```

Check if sample should run or be stopped.

**id** `str | int` — Sample ID.

**epoch** `int` — Epoch number.

**Returns:** `EarlyStop | None` — `EarlyStop` if stopped, `None` to run.

#### complete_sample

```python
async def complete_sample(
    self,
    id: str | int,
    epoch: int,
    scores: dict[str, SampleScore],
) -> None
```

Process completed sample and potentially trigger inference.

**id** `str | int` — Sample ID.

**epoch** `int` — Epoch number.

**scores** `dict[str, SampleScore]` — Scores from scorers.

#### complete_task

```python
async def complete_task(self) -> dict[str, JsonValue]
```

Generate final diagnostics.

**Returns:** `dict[str, JsonValue]` — Diagnostics dictionary containing:

| Key | Type | Description |
|-----|------|-------------|
| `manager` | `str` | Manager name |
| `random_seed` | `int` | MCMC seed used |
| `seed_source` | `str` | `"user_specified"` or `"auto_generated"` |
| `total_planned_trials` | `int` | Total trials planned |
| `total_ran` | `int` | Trials executed |
| `total_skipped` | `int` | Trials skipped |
| `efficiency_percent` | `float` | Percentage saved |
| `stopped_samples_count` | `int` | Samples stopped early |
| `stopped_samples` | `list[dict]` | Details of stopped samples |
| `stopped_groupings` | `list[str]` | Groupings that stopped |
