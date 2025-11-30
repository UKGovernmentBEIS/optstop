# Integration Issues - Optstop Package with inspect_ai

This document tracks issues discovered during large-scale bridge testing that may affect real-world usage with inspect_ai.

## Issue #1: Numpy Type Serialization in Diagnostics

**Discovered**: 2025-11-26 during large-scale bridge testing (Dataset 1)

**Description**:
`OptimalStoppingManager.complete_task()` returns diagnostics containing numpy int64/float64 types (e.g., from pandas operations). These types are not JSON serializable, causing errors when users try to save diagnostics:

```python
TypeError: Object of type int64 is not JSON serializable
```

**Impact**:
- **Severity**: Medium
- Users who want to persist diagnostics to JSON files will encounter this error
- Affects: `complete_task()` return value

**Affected Fields**:
Based on testing, the following diagnostic fields contain numpy types:
- `total_planned` (np.int64)
- `total_ran` (np.int64)
- `efficiency` (np.float64)
- Grouping-level statistics (various numpy types)

**Current Workaround**:
Manually convert numpy types before JSON serialization:

```python
def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# Usage:
diagnostics = await manager.complete_task()
with open('diagnostics.json', 'w') as f:
    json.dump(convert_numpy_types(diagnostics), f)
```

**Recommended Fix**:
Two options:
1. **Option A (Preferred)**: Modify `OptimalStoppingManager.complete_task()` to convert all numpy types to native Python types before returning diagnostics
2. **Option B**: Provide a helper function `optstop.utils.to_json_serializable(diagnostics)` and document this requirement

**Testing Reference**:
- Test script: `/home/ubuntu/optstop/scripts/test_large_datasets_bridge.py`
- Helper function: `convert_numpy_types()` (lines 29-53)

---

## Issue #2: Logging Configuration Fragility in Multi-Dataset Tests

**Discovered**: 2025-11-26 during large-scale bridge testing (Datasets 2 & 3)

**Description**:
When running multiple datasets sequentially in the same Python process, `logging.basicConfig()` only configures logging for the first dataset. Subsequent calls are ignored, causing empty log files for later datasets.

**Root Cause**:
Python's `logging.basicConfig()` is designed to run only once per process. From Python docs: "This function does nothing if the root logger already has handlers configured."

**Impact**:
- **Severity**: Medium-High
- Empty log files prevent debugging and routing verification
- Affects: Any script running multiple OptimalStoppingManager sessions sequentially
- **Observed**: Dataset 1 log worked (1.8MB), Datasets 2 & 3 logs empty (0 bytes)

**Fix Applied**:
Manually manage logging handlers instead of using basicConfig():

```python
# Get loggers
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
optstop_logger = logging.getLogger('optstop')

# Clear any existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Create new handlers for this dataset
file_handler = logging.FileHandler(log_path, mode='w')
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Add to root logger
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)
root_logger.setLevel(logging.DEBUG)
```

**Recommended Fix for Package**:
If OptimalStoppingManager is intended for sequential multi-task use, the package documentation should warn about this logging issue or provide a helper for proper logger configuration.

**Testing Reference**:
- Fix: `/home/ubuntu/optstop/scripts/test_large_datasets_bridge.py` lines 213-248
- Issue documented: `/home/ubuntu/optstop/LARGE_SCALE_TEST_CRITIQUE.md` Issue #3

---

## Issue #3: Missing ordinal_tasks Parameter Causes Ordinal Inference Failure

**Discovered**: 2025-11-26 during large-scale bridge testing (Dataset 2) and logging fix validation

**Description**:
The `OptimalStoppingManager` requires the `ordinal_tasks` parameter to be explicitly set to enable ordinal inference, even when `score_choice` or `score_agg` parameters are provided. Without `ordinal_tasks`, ALL tasks default to binary inference, causing validation errors for ordinal scores:

```python
⚠️  Invalid score for sample_id=sample_0000, epoch=0: Binary task 'math_easy'
has score > 1 (10.0) - expected discrete scores in {0, 1}. Cannot perform
inference. Score will be recorded but no inference will run for this sample.
```

**Root Cause**:
The manager determines if a task is ordinal by pattern matching task names against the `ordinal_tasks` list (optstop/early_stopping.py:820-828):

```python
# Determine if this is an ordinal task
is_ordinal = False
if self.ordinal_tasks:
    # Check if task name matches any ordinal task pattern
    for ordinal_pattern in self.ordinal_tasks:
        if ordinal_pattern in str(task_name):
            is_ordinal = True
            break
```

Without this parameter, `is_ordinal` remains `False` regardless of other ordinal-related configuration.

**Impact**:
- **Severity**: High
- Users with ordinal scoring will see 0% efficiency (no early stopping)
- All samples complete without inference being performed
- Silent failure - no obvious error, just degraded performance
- Affects: Any bridge integration using ordinal or continuous bounded scoring

**Current Workaround**:
Explicitly specify which tasks should use ordinal inference:

```python
# For datasets where all tasks are ordinal:
base_config = {
    'optstop_params': {...},
    'grouping_columns': ['model', 'task'],
    'score_choice': 'accuracy',  # Alone, this is insufficient
    'ordinal_max_score': 10,     # Alone, this is insufficient
    'ordinal_tasks': ['math_easy', 'math_hard', 'coding_medium',
                      'creative_writing', 'reasoning']  # Required!
}

# Or use a catch-all pattern (matches all tasks):
base_config = {
    ...
    'ordinal_tasks': ['']  # Empty string matches all task names
}
```

**Recommended Fix**:
1. **Option A (Preferred)**: Auto-detect ordinal tasks when `score_choice` or `score_agg='mean'` is set, eliminating need for explicit `ordinal_tasks` parameter in common cases
2. **Option B**: Raise a clear error at initialization if ordinal parameters (`score_choice`, `ordinal_max_score`) are set without `ordinal_tasks`
3. **Option C (Minimum)**: Document this requirement prominently in bridge integration examples and docstrings

**Testing Reference**:
- Test script: `/home/ubuntu/optstop/scripts/test_logging_fix.py`
- Large-scale test: `/home/ubuntu/optstop/scripts/test_large_datasets_bridge.py`
- Dataset 2 diagnostics: Showed 0% efficiency and no groupings due to this issue
- Log output: `/home/ubuntu/optstop/test_outputs/logging_fix_test/dataset_2_small.log`

---

## Template for Future Issues

```markdown
## Issue #N: [Brief Title]

**Discovered**: [Date] during [test/scenario]

**Description**:
[Detailed description of the issue]

**Impact**:
- **Severity**: [Low/Medium/High/Critical]
- [Who is affected and how]

**Current Workaround**:
[Code or instructions]

**Recommended Fix**:
[Proposed solution(s)]

**Testing Reference**:
[Links to relevant test files/documentation]
```

---

## Issue Tracking Notes

This document should be reviewed before:
1. Public release of the optstop package
2. Publishing documentation/examples for inspect_ai integration
3. Creating user-facing tutorials

Each issue should be evaluated for:
- Fix priority based on severity and frequency
- Whether to fix in package code or document as known limitation
- Whether helper utilities should be provided
