# Implementation Plan: Continuous Bounded Score Inference

**Date**: 2025-11-19
**Status**: APPROVED - Ready for Implementation
**Priority**: HIGH

---

## Executive Summary

### The Problem

When users specify `score_agg='mean'` or `score_agg='median'` to aggregate multiple scores, the resulting values are **continuous floats**:

- **Binary tasks with aggregation**: Multiple binary scores (0/1) averaged → continuous float in [0, 1]
  - Example: [1, 0, 1, 1, 0] → mean = 0.6

- **Ordinal tasks with aggregation**: Multiple ordinal scores averaged → continuous float in [0, ordinal_max_score]
  - Example: [8, 9, 7, 9, 8] → mean = 8.2

**Current behavior is broken**:
- Binary inference expects discrete success counts → fails with continuous values
- Ordinal modal inference expects discrete categories → achieves only 4% efficiency vs 91% for discrete (as documented in FINAL_ORDINAL_COMPARISON_REPORT.md)

### The Solution

Implement a **third inference route** for continuous bounded scores alongside existing binary and ordinal routes:

1. **Binary route**: Discrete 0/1 scores → Beta-Binomial inference
2. **Ordinal route**: Discrete categorical scores → Modal/Entropy inference
3. **Continuous route** (NEW): Continuous bounded scores → Bounded continuous inference

**Trigger**: Automatically activated when `score_agg` is set to `'mean'` or `'median'`

**Bounds**: Automatically determined based on task type:
- Binary tasks: [0, 1]
- Ordinal tasks: [0, ordinal_max_score]

---

## Three-Phase Implementation Plan

---

## Phase 1: Detection & Routing (HIGH PRIORITY)

**Goal**: Detect aggregated continuous scores and route them correctly to prevent invalid inference.

**Status**: Must be implemented first to prevent misleading results.

### 1.1 Enhance `determine_score_type()` Function

**File**: `optstop/ordinal_utils.py:217`

**Current signature**:
```python
def determine_score_type(
    grouping_name: str,
    ordinal_tasks: Optional[list] = None
) -> str:
    # Returns: 'binary' or 'ordinal'
```

**New signature**:
```python
def determine_score_type(
    grouping_name: str,
    ordinal_tasks: Optional[list] = None,
    is_aggregated: bool = False,
    upper_bound: float = 1.0
) -> Tuple[str, Dict[str, float]]:
    """
    Determine score type and bounds based on context.

    Args:
        grouping_name: String identifier for the grouping
        ordinal_tasks: List of substrings to match for ordinal scoring
        is_aggregated: Whether scores are aggregated (mean/median)
        upper_bound: Upper bound for continuous scores (1.0 for binary, ordinal_max_score for ordinal)

    Returns:
        Tuple of (score_type, bounds_dict):
        - score_type: 'binary', 'ordinal', 'continuous_01', or 'continuous_bounded'
        - bounds_dict: {'lower': 0.0, 'upper': upper_bound}
    """
```

**Implementation logic**:
```python
def determine_score_type(
    grouping_name: str,
    ordinal_tasks: Optional[list] = None,
    is_aggregated: bool = False,
    upper_bound: float = 1.0
) -> Tuple[str, Dict[str, float]]:
    logger = logging.getLogger('optstop.ordinal_utils')

    # Determine base type (binary or ordinal)
    is_ordinal = False
    if ordinal_tasks is not None and len(ordinal_tasks) > 0:
        grouping_name_lower = grouping_name.lower()
        for ordinal_substring in ordinal_tasks:
            if ordinal_substring.lower() in grouping_name_lower:
                is_ordinal = True
                break

    # If aggregated, return continuous type
    if is_aggregated:
        if is_ordinal:
            score_type = 'continuous_bounded'
            bounds = {'lower': 0.0, 'upper': upper_bound}
            logger.info(
                f"Grouping '{grouping_name}' with aggregation → CONTINUOUS_BOUNDED [0, {upper_bound}]"
            )
        else:
            score_type = 'continuous_01'
            bounds = {'lower': 0.0, 'upper': 1.0}
            logger.info(
                f"Grouping '{grouping_name}' with aggregation → CONTINUOUS_01 [0, 1]"
            )
        return score_type, bounds

    # Non-aggregated: return discrete types
    if is_ordinal:
        logger.info(f"Grouping '{grouping_name}' → ORDINAL (discrete)")
        return 'ordinal', {'lower': 0.0, 'upper': upper_bound}
    else:
        logger.debug(f"Grouping '{grouping_name}' → BINARY (discrete)")
        return 'binary', {'lower': 0.0, 'upper': 1.0}
```

### 1.2 Update Call Sites

**Update all locations that call `determine_score_type()`**:

#### Location 1: `optstop/rule.py:707`
```python
# BEFORE:
score_type = determine_score_type(grouping_name, ordinal_tasks)

# AFTER:
is_aggregated = params.get('is_aggregated', False)
score_type, bounds = determine_score_type(
    grouping_name,
    ordinal_tasks,
    is_aggregated=is_aggregated,
    upper_bound=ordinal_max_score
)
```

#### Location 2: `optstop/rule.py:1205`
```python
# AFTER:
is_aggregated = params.get('is_aggregated', False)
score_type, bounds = determine_score_type(
    grouping,
    ordinal_tasks,
    is_aggregated=is_aggregated,
    upper_bound=ordinal_max_score
)
```

#### Location 3: `optstop/rule.py:1689`
```python
# AFTER:
is_aggregated = params.get('is_aggregated', False)
score_type, bounds = determine_score_type(
    grouping_name,
    ordinal_tasks,
    is_aggregated=is_aggregated,
    upper_bound=ordinal_max_score
)
```

#### Location 4: `optstop/rule.py:1915` (optimal_stopping_live_single)
```python
# AFTER:
is_aggregated = params.get('is_aggregated', False)
score_type, bounds = determine_score_type(
    grouping_name,
    ordinal_tasks,
    is_aggregated=is_aggregated,
    upper_bound=ordinal_max_score
)
```

### 1.3 Update OptimalStoppingManager

**File**: `optstop/early_stopping.py`

**Modify `_run_stopping_inference()` at line 874**:

```python
async def _run_stopping_inference(
    self,
    grouping_values: dict[str, Any]
) -> dict[str, Any]:
    # ... existing code ...

    # NEW: Determine if scores are aggregated
    is_aggregated = bool(self.score_agg in ['mean', 'median'])

    # Add to params for optimal_stopping_live_single
    params_with_aggregation = self.optstop_params.copy()
    params_with_aggregation['is_aggregated'] = is_aggregated

    # Call optimal_stopping_live_single with updated params
    result = await asyncio.to_thread(
        optimal_stopping_live_single,
        df_grouping=completed_data,
        grouping_name=grouping_name,
        params=params_with_aggregation,  # <-- Updated
        # ... rest of arguments ...
    )
```

### 1.4 Update Validation Logic

**File**: `optstop/early_stopping.py:790-830` (in `complete_sample()`)

**Current validation** checks:
- Binary tasks: score > 1 is invalid
- Ordinal tasks: score > ordinal_max_score is invalid

**Updated validation**:
```python
# Check 3: Score validation depends on aggregation
is_aggregated = bool(self.score_agg in ['mean', 'median'])

if task_name is not None:
    is_ordinal = False
    if self.ordinal_tasks:
        for ordinal_pattern in self.ordinal_tasks:
            if ordinal_pattern in str(task_name):
                is_ordinal = True
                break

    if not is_ordinal:
        # Binary task validation
        if is_aggregated:
            # Allow continuous [0, 1]
            if score_value > 1.0:
                score_valid_for_inference = False
                validation_message = (
                    f"Binary task '{task_name}' with aggregation has score > 1 ({score_value}) - "
                    f"expected continuous values in [0, 1]. Cannot perform inference."
                )
        else:
            # Discrete binary: only 0 or 1
            if score_value > 1:
                score_valid_for_inference = False
                validation_message = (
                    f"Binary task '{task_name}' has score > 1 ({score_value}) - "
                    f"expected scores in [0, 1]. Cannot perform inference."
                )

    elif is_ordinal:
        # Ordinal task validation
        if score_value > self.ordinal_max_score:
            score_valid_for_inference = False
            validation_message = (
                f"Ordinal task '{task_name}' has score > max ({score_value} > {self.ordinal_max_score}) - "
                f"exceeds ordinal_max_score. Cannot perform inference."
            )
```

### 1.5 Temporary Fallback Behavior

Until Phases 2-3 are complete, add safe fallback in `optimal_stopping_live_single()`:

```python
# After determining score_type at line 1915:
score_type, bounds = determine_score_type(...)

if score_type in ['continuous_01', 'continuous_bounded']:
    logger.warning(
        f"Grouping '{grouping_name}' requires continuous bounded inference, "
        f"which is not yet implemented. No early stopping will be applied. "
        f"All trials will run to completion."
    )
    return {
        'grouping': grouping_name,
        'stop_sample_ids': [],
        'stop_this_grouping': [],
        'stabilization_history': stabilization_history,
        'metadata': {'warning': 'continuous_inference_not_implemented'}
    }
```

### Phase 1 Deliverables

- [ ] Enhanced `determine_score_type()` with 4 return types
- [ ] Updated all call sites to pass aggregation context
- [ ] Updated validation logic to allow continuous scores when aggregated
- [ ] Temporary fallback warning for continuous types
- [ ] Unit tests for detection logic
- [ ] Integration test showing warning when score_agg is used

**Outcome**: Users with `score_agg='mean'` will see clear warning that continuous inference is not yet implemented, and all trials will run to completion (safe behavior).

---

## Phase 2: Continuous Inference Method (CORE IMPLEMENTATION)

**Goal**: Implement statistically rigorous inference for continuous bounded scores.

### 2.1 Implement `_continuous_bounded_ci_adaptive()`

**File**: `optstop/rule.py` (add after `_beta_ci_adaptive` at line 565)

```python
def _continuous_bounded_ci_adaptive(
    scores: np.ndarray,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
    cred_level: float = 0.95,
    conservatism: float = 1.0,
    low_perf_threshold: float = 0.2,
    base_strength: int = 2,
    samples: int = 10000
) -> Tuple[float, float, float]:
    """
    Compute adaptive Bayesian credible interval for continuous bounded scores.

    Designed for aggregated scores (mean/median) that are continuous floats:
    - Binary aggregated: scores in [0, 1]
    - Ordinal aggregated: scores in [0, ordinal_max_score]

    Method: Beta distribution (or scaled/shifted Beta for arbitrary bounds)
    with method-of-moments parameter estimation.

    Args:
        scores: Array of continuous scores
        lower_bound: Lower bound of score range (default: 0.0)
        upper_bound: Upper bound of score range (default: 1.0)
        cred_level: Credibility level (e.g., 0.95 for 95% CI)
        conservatism: Multiplier for CI width in low-performance scenarios (>= 1.0)
        low_perf_threshold: Performance threshold for conservatism (normalized 0-1)
        base_strength: Base prior strength
        samples: Number of Monte Carlo samples

    Returns:
        Tuple of (lower_bound_ci, upper_bound_ci, effective_width)
        - All values in original scale [lower_bound, upper_bound]
        - effective_width: CI width, adjusted for conservatism if needed

    Statistical Approach:
        1. Normalize scores to [0, 1]
        2. Estimate Beta distribution parameters using method of moments:
           - mean = alpha / (alpha + beta)
           - var = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
        3. Apply conservative priors for low performance
        4. Generate posterior samples
        5. Compute credible interval
        6. Scale back to original bounds

    Example:
        # Binary aggregated: [0.8, 0.9, 0.7, 0.85, 0.9]
        scores = np.array([0.8, 0.9, 0.7, 0.85, 0.9])
        lo, hi, width = _continuous_bounded_ci_adaptive(scores, 0.0, 1.0)
        # Returns: (0.75, 0.92, 0.17) - "95% confident mean is 0.75-0.92"

        # Ordinal aggregated: [8.2, 8.7, 8.4, 8.9, 8.5]
        scores = np.array([8.2, 8.7, 8.4, 8.9, 8.5])
        lo, hi, width = _continuous_bounded_ci_adaptive(scores, 0.0, 10.0)
        # Returns: (8.1, 8.9, 0.8) in original scale
    """
    logger = logging.getLogger('optstop.continuous_bounded')

    # Handle edge cases
    if len(scores) == 0:
        logger.warning("Empty scores array provided to _continuous_bounded_ci_adaptive")
        return lower_bound, upper_bound, upper_bound - lower_bound

    # Normalize scores to [0, 1]
    score_range = upper_bound - lower_bound
    if score_range == 0:
        logger.warning("Zero score range provided")
        return lower_bound, upper_bound, 0.0

    scores_normalized = (scores - lower_bound) / score_range

    # Clip to [0, 1] to handle numerical errors
    scores_normalized = np.clip(scores_normalized, 0.0, 1.0)

    # Single observation - return wide interval
    if len(scores) == 1:
        val = scores_normalized[0]
        # Conservative: ±0.3 around observed value
        lo_norm = max(0.0, val - 0.3)
        hi_norm = min(1.0, val + 0.3)
        width_norm = hi_norm - lo_norm

        # Scale back to original bounds
        lo = lo_norm * score_range + lower_bound
        hi = hi_norm * score_range + lower_bound
        width = width_norm * score_range
        return lo, hi, width

    # Compute sample statistics
    mean_obs = np.mean(scores_normalized)
    var_obs = np.var(scores_normalized, ddof=1)  # Unbiased variance

    # Determine if low performance (normalized)
    is_low_perf = mean_obs < low_perf_threshold
    current_conservatism = conservatism if is_low_perf else 1.0

    # Method of moments for Beta distribution
    # mean = alpha / (alpha + beta)
    # var = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))

    # Avoid division by zero
    if var_obs < 1e-10:
        # Very low variance - concentration around mean
        var_obs = 1e-10

    # Ensure valid beta parameters (mean must be in (0,1), var must be < mean*(1-mean))
    mean_obs = np.clip(mean_obs, 0.01, 0.99)
    max_var = mean_obs * (1 - mean_obs) * 0.99  # Leave some room
    var_obs = min(var_obs, max_var)

    # Method of moments estimation
    common_term = mean_obs * (1 - mean_obs) / var_obs - 1
    alpha_mom = mean_obs * common_term
    beta_mom = (1 - mean_obs) * common_term

    # Apply Bayesian prior
    # Prior strength decays with sample size
    n = len(scores)
    prior_scaling = base_strength * np.exp(-n / 10)

    if is_low_perf:
        # Conservative prior for low performance
        alpha_prior = max(prior_scaling * mean_obs * conservatism, 0.5)
        beta_prior = max(prior_scaling * (1 - mean_obs), 0.5)
    else:
        alpha_prior = max(prior_scaling * mean_obs, 0.5)
        beta_prior = max(prior_scaling * (1 - mean_obs), 0.5)

    # Posterior parameters
    # For continuous data, we treat each observation as contributing to shape
    # This is a simplification - more rigorous would be full Bayesian model
    alpha_post = alpha_prior + alpha_mom
    beta_post = beta_prior + beta_mom

    # Generate posterior samples
    draws = np.random.beta(alpha_post, beta_post, samples)

    # Compute credible interval
    lo_norm, hi_norm = np.quantile(draws, [(1 - cred_level) / 2, 1 - (1 - cred_level) / 2])
    width_norm = hi_norm - lo_norm

    # Apply conservatism to width if low performance
    if is_low_perf:
        effective_width_norm = width_norm * current_conservatism
    else:
        effective_width_norm = width_norm

    # Scale back to original bounds
    lo = lo_norm * score_range + lower_bound
    hi = hi_norm * score_range + lower_bound
    effective_width = effective_width_norm * score_range

    logger.debug(
        f"Continuous bounded CI: n={n}, mean={mean_obs:.3f}, "
        f"CI=[{lo_norm:.3f}, {hi_norm:.3f}] (normalized), "
        f"width={effective_width:.4f} (original scale)"
    )

    return lo, hi, effective_width
```

### 2.2 Add Unit Tests

**File**: `tests/test_continuous_bounded_inference.py` (NEW)

```python
import numpy as np
import pytest
from optstop.rule import _continuous_bounded_ci_adaptive


class TestContinuousBoundedInference:
    """Test suite for continuous bounded score inference."""

    def test_binary_aggregated_high_performance(self):
        """Test aggregated binary scores with high, consistent performance."""
        # Simulate 20 epochs, each averaging 5 binary scores
        # Mean performance: 0.90
        scores = np.random.uniform(0.85, 0.95, size=20)

        lo, hi, width = _continuous_bounded_ci_adaptive(
            scores,
            lower_bound=0.0,
            upper_bound=1.0,
            cred_level=0.95
        )

        # Assertions
        assert 0.0 <= lo < hi <= 1.0, "CI bounds must be in [0, 1]"
        assert 0.80 < lo < 0.95, "Lower bound should be near true mean"
        assert 0.85 < hi < 1.0, "Upper bound should be near true mean"
        assert width < 0.2, "Width should be narrow for consistent high performance"

    def test_ordinal_aggregated_high_performance(self):
        """Test aggregated ordinal scores with high, consistent performance."""
        # Simulate 20 epochs, mean performance: 8.5 / 10
        scores = np.random.uniform(8.0, 9.0, size=20)

        lo, hi, width = _continuous_bounded_ci_adaptive(
            scores,
            lower_bound=0.0,
            upper_bound=10.0,
            cred_level=0.95
        )

        # Assertions
        assert 0.0 <= lo < hi <= 10.0, "CI bounds must be in [0, 10]"
        assert 7.5 < lo < 9.0, "Lower bound should be near true mean"
        assert 8.0 < hi < 9.5, "Upper bound should be near true mean"
        assert width < 2.0, "Width should be narrow for consistent performance"

    def test_low_performance_conservatism(self):
        """Test that conservatism is applied for low performance."""
        # Low performance: mean ~ 0.05
        scores = np.random.uniform(0.02, 0.08, size=20)

        lo_conservative, hi_conservative, width_conservative = _continuous_bounded_ci_adaptive(
            scores,
            lower_bound=0.0,
            upper_bound=1.0,
            cred_level=0.95,
            conservatism=5.0,
            low_perf_threshold=0.2
        )

        lo_normal, hi_normal, width_normal = _continuous_bounded_ci_adaptive(
            scores,
            lower_bound=0.0,
            upper_bound=1.0,
            cred_level=0.95,
            conservatism=1.0,
            low_perf_threshold=0.2
        )

        # Conservative width should be larger
        assert width_conservative > width_normal, "Conservatism should increase CI width"
        assert width_conservative >= width_normal * 4, "Conservative multiplier should be substantial"

    def test_single_observation(self):
        """Test handling of single observation."""
        scores = np.array([0.75])

        lo, hi, width = _continuous_bounded_ci_adaptive(
            scores,
            lower_bound=0.0,
            upper_bound=1.0,
            cred_level=0.95
        )

        # Should return wide interval
        assert width > 0.4, "Single observation should produce wide CI"
        assert lo < 0.75 < hi, "CI should contain observed value"

    def test_convergence_with_sample_size(self):
        """Test that CI width decreases with more samples."""
        true_mean = 0.80

        widths = []
        for n in [5, 10, 20, 50, 100]:
            scores = np.random.normal(true_mean, 0.1, size=n)
            scores = np.clip(scores, 0.0, 1.0)

            _, _, width = _continuous_bounded_ci_adaptive(
                scores,
                lower_bound=0.0,
                upper_bound=1.0,
                cred_level=0.95
            )
            widths.append(width)

        # Width should generally decrease
        assert widths[0] > widths[-1], "CI should narrow with more samples"
        assert widths[4] < 0.15, "Large sample should have narrow CI"

    def test_edge_case_all_same_value(self):
        """Test edge case where all scores are identical."""
        scores = np.array([0.75] * 20)

        lo, hi, width = _continuous_bounded_ci_adaptive(
            scores,
            lower_bound=0.0,
            upper_bound=1.0,
            cred_level=0.95
        )

        # Should have very narrow CI around 0.75
        assert width < 0.1, "Identical scores should produce narrow CI"
        assert abs(lo + hi) / 2 - 0.75 < 0.05, "CI should be centered near true value"

    def test_edge_case_empty_array(self):
        """Test edge case with empty scores array."""
        scores = np.array([])

        lo, hi, width = _continuous_bounded_ci_adaptive(
            scores,
            lower_bound=0.0,
            upper_bound=1.0,
            cred_level=0.95
        )

        # Should return full range
        assert lo == 0.0
        assert hi == 1.0
        assert width == 1.0
```

### 2.3 Validation with Simulation Study

**File**: `tests/test_continuous_bounded_coverage.py` (NEW)

Create simulation study to validate CI coverage:
- Generate synthetic continuous data from known distributions
- Apply `_continuous_bounded_ci_adaptive()`
- Verify that true parameter falls within CI ~95% of the time

### Phase 2 Deliverables

- [ ] Implemented `_continuous_bounded_ci_adaptive()` with full documentation
- [ ] Unit tests covering edge cases and parameter validation
- [ ] Simulation study validating CI coverage
- [ ] Performance benchmarks (should be fast: ~1-10ms per call)

**Outcome**: Core statistical method ready for integration.

---

## Phase 3: Integration into Stopping Logic (CONNECT THE PIECES)

**Goal**: Wire continuous bounded inference into the full early stopping pipeline.

### 3.1 Modify `optimal_stopping_live_single()`

**File**: `optstop/rule.py:1821`

**Update routing logic** after line 1915:

```python
# Determine score type for this grouping
is_aggregated = params.get('is_aggregated', False)
score_type, bounds = determine_score_type(
    grouping_name,
    ordinal_tasks,
    is_aggregated=is_aggregated,
    upper_bound=ordinal_max_score
)
logger.info(f"Processing grouping '{grouping_name}' as {score_type.upper()}")

# Extract bounds
lower_bound = bounds['lower']
upper_bound = bounds['upper']
```

### 3.2 Add Continuous Sample-Level Stopping

**Location**: After ordinal sample-level stopping (after line 2072)

```python
# Process each sample for sample-level stopping
item_summaries = []
for item_idx, item_id in enumerate(item_ids):
    df_item = df_work[df_work['sample_id_num'] == item_id].sort_values('epoch_num')
    original_sample_id = df_item[sample_id_column].iloc[0]

    if score_type == 'binary':
        # === BINARY SAMPLE-LEVEL STOPPING === (existing code)
        # ... lines 1968-1995 ...

    elif score_type == 'ordinal':
        # === ORDINAL SAMPLE-LEVEL STOPPING === (existing code)
        # ... lines 1997-2072 ...

    elif score_type in ['continuous_01', 'continuous_bounded']:
        # === CONTINUOUS SAMPLE-LEVEL STOPPING === (NEW)
        accumulated_scores = df_item[score_column].values

        # Normalize performance for conservatism check
        normalized_perf = np.mean(accumulated_scores) / upper_bound
        current_conservatism = conservatism if normalized_perf < low_perf_threshold else 1.0

        # Compute CI
        lo, hi, width = _continuous_bounded_ci_adaptive(
            accumulated_scores,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            cred_level=cred_level,
            conservatism=current_conservatism,
            low_perf_threshold=low_perf_threshold
        )

        # Normalize width for comparison with delta_item
        width_normalized = width / upper_bound

        # Check width criterion
        if width_normalized < delta_item:
            stop_sample_ids.append(f"{grouping_name}:::{original_sample_id}")
            metadata['sample_stopping_reasons'][str(original_sample_id)] = {
                'reason': 'continuous_bounded_ci_width',
                'ci_width': float(width),
                'ci_width_normalized': float(width_normalized),
                'threshold': delta_item,
                'epochs_used': len(accumulated_scores),
                'bounds': {'lower': lower_bound, 'upper': upper_bound}
            }
            logger.info(
                f"Stopping continuous sample {original_sample_id}: "
                f"CI width {width_normalized:.4f} < {delta_item}"
            )

        # Record for group-level analysis
        item_summaries.append({
            'sample_id': original_sample_id,
            'mean': float(np.mean(accumulated_scores)),
            'scores': accumulated_scores.tolist()
        })
```

### 3.3 Add Continuous Group-Level Stopping

**Location**: After ordinal group-level stopping (after line 2219)

```python
# Group-level stopping check (runs automatically after sample checks)
if len(item_summaries) > 0:
    logger.info(f"Running group-level stopping check for '{grouping_name}'")

    if score_type == 'binary':
        # === BINARY GROUP-LEVEL STOPPING === (existing code)
        # ... lines 2079-2161 ...

    elif score_type == 'ordinal':
        # === ORDINAL GROUP-LEVEL STOPPING === (existing code)
        # ... lines 2162-2219 ...

    elif score_type in ['continuous_01', 'continuous_bounded']:
        # === CONTINUOUS GROUP-LEVEL STOPPING === (NEW)

        # Aggregate all scores across all samples
        all_continuous_scores = []
        for item_summary in item_summaries:
            all_continuous_scores.extend(item_summary['scores'])

        all_continuous_scores = np.array(all_continuous_scores)

        # Normalize performance for conservatism check
        current_perf_estimate = np.mean(all_continuous_scores)
        normalized_perf = current_perf_estimate / upper_bound
        current_conservatism = conservatism if normalized_perf < low_perf_threshold else 1.0

        # Compute CI at group level
        lo, hi, width = _continuous_bounded_ci_adaptive(
            all_continuous_scores,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            cred_level=cred_level,
            conservatism=current_conservatism,
            low_perf_threshold=low_perf_threshold
        )

        # Normalize width for comparison with delta_cap
        width_normalized = width / upper_bound

        # Append to history
        stabilization_history['ci_width_history'].append(float(width_normalized))

        # Check width criterion
        if width_normalized < delta_cap:
            stop_this_grouping.append(grouping_name)
            metadata['group_stopping_reason'] = {
                'reason': 'continuous_bounded_ci_width',
                'ci_width': float(width),
                'ci_width_normalized': float(width_normalized),
                'threshold': delta_cap,
                'samples_used': len(item_summaries),
                'bounds': {'lower': lower_bound, 'upper': upper_bound}
            }
            logger.info(
                f"Stopping continuous grouping '{grouping_name}': "
                f"CI width {width_normalized:.4f} < {delta_cap}"
            )

        # Check stabilization criterion (if enough history)
        if len(stabilization_history['ci_width_history']) >= stab_window:
            recent_widths = stabilization_history['ci_width_history'][-stab_window:]
            slope = np.polyfit(range(len(recent_widths)), recent_widths, 1)[0]
            stabilization_history['ci_slope_history'].append(float(slope))

            slope_threshold = CI_delta / current_conservatism if normalized_perf < low_perf_threshold else CI_delta

            if abs(slope) <= slope_threshold and len(stabilization_history['ci_slope_history']) >= 4:
                recent_slopes = stabilization_history['ci_slope_history'][-3:]
                slope_slopes = np.polyfit(range(len(recent_slopes)), recent_slopes, 1)[0]

                if slope_slopes >= 0:
                    if normalized_perf >= low_perf_threshold:
                        stop_this_grouping.append(grouping_name)
                        metadata['group_stopping_reason'] = {
                            'reason': 'continuous_bounded_stabilization',
                            'slope': float(slope),
                            'slope_threshold': slope_threshold,
                            'samples_used': len(item_summaries)
                        }
                        logger.info(
                            f"Stopping continuous grouping '{grouping_name}' via stabilization: "
                            f"slope {slope:.6f}"
                        )
                    elif abs(slope) <= slope_threshold / 2:
                        stop_this_grouping.append(grouping_name)
                        metadata['group_stopping_reason'] = {
                            'reason': 'continuous_bounded_stabilization_low_perf',
                            'slope': float(slope),
                            'slope_threshold': slope_threshold,
                            'samples_used': len(item_summaries)
                        }
                        logger.info(
                            f"Stopping low-perf continuous grouping '{grouping_name}' "
                            f"via strong stabilization"
                        )
```

### 3.4 Update Metadata and Logging

Ensure stopping reasons are distinguishable:
- Binary: `'ci_width'`
- Ordinal: `'ordinal_modal_ci_width'`, `'ordinal_entropy_ci_width'`, etc.
- Continuous: `'continuous_bounded_ci_width'`, `'continuous_bounded_stabilization'`, etc.

### 3.5 Integration Tests

**File**: `tests/test_continuous_early_stopping_integration.py` (NEW)

```python
"""
Integration tests for continuous bounded early stopping.
Tests full pipeline from OptimalStoppingManager through to stopping decisions.
"""

import pytest
import asyncio
import numpy as np
from optstop.early_stopping import OptimalStoppingManager
# Import mock objects
from tests.test_early_stopping import (
    create_mock_evalspec,
    create_mock_samples,
    create_mock_scores
)


class TestContinuousEarlyStopping:
    """Integration tests for continuous bounded inference in early stopping."""

    @pytest.mark.asyncio
    async def test_binary_aggregated_consistent_high(self):
        """Test binary task with aggregated scores (mean) - consistent high performance."""

        # Configuration
        manager = OptimalStoppingManager(
            optstop_params={
                'delta_item': 0.05,
                'delta_cap': 0.05,
                'cred_level': 0.95,
                'conservatism': 5
            },
            grouping_columns=['model', 'task'],
            reanalysis_interval=10,
            min_samples_per_grouping=5,
            score_agg='mean',  # CRITICAL: Triggers continuous inference
            shadow_mode=False
        )

        # Create mock task and samples
        evalspec = create_mock_evalspec(model='gpt-4', task='binary_test')
        samples = create_mock_samples(n_samples=50)

        # Start task
        await manager.start_task(evalspec, samples, epochs=20)

        # Simulate evaluation with consistent high performance
        stopped_count = 0
        for sample_idx in range(50):
            for epoch in range(1, 21):
                sample_id = f"sample_{sample_idx}"

                # Generate 3 binary scores with 90% success rate
                binary_scores = np.random.binomial(1, 0.9, size=3)
                mean_score = float(np.mean(binary_scores))  # ~0.9

                # Create mock scores (3 scorers, aggregated via mean)
                scores = {
                    'scorer_1': create_mock_score(float(binary_scores[0])),
                    'scorer_2': create_mock_score(float(binary_scores[1])),
                    'scorer_3': create_mock_score(float(binary_scores[2]))
                }

                # Check if should run
                directive = await manager.schedule_sample(sample_id, epoch)
                if directive is not None:
                    stopped_count += 1
                    break

                # Complete sample
                await manager.complete_sample(sample_id, epoch, scores)

        # Complete task
        metadata = await manager.complete_task()

        # Assertions
        assert stopped_count > 20, f"Should stop at least 20 samples, got {stopped_count}"
        assert metadata['efficiency_percent'] > 30, "Should achieve >30% efficiency"

        # Check stopping reasons contain 'continuous_bounded'
        stopping_reasons = set()
        for grouping_stops in manager._stopped_sample_ids.values():
            for stop in grouping_stops:
                reason = metadata.get('sample_stopping_reasons', {}).get(stop.id, {}).get('reason')
                if reason:
                    stopping_reasons.add(reason)

        assert 'continuous_bounded_ci_width' in stopping_reasons, \
            "Should use continuous bounded inference"

    @pytest.mark.asyncio
    async def test_ordinal_aggregated_consistent_high(self):
        """Test ordinal task with aggregated scores (median) - consistent high performance."""

        # Configuration
        manager = OptimalStoppingManager(
            optstop_params={
                'delta_item': 0.05,
                'delta_cap': 0.05,
                'cred_level': 0.95,
                'conservatism': 5
            },
            grouping_columns=['model', 'task'],
            reanalysis_interval=10,
            min_samples_per_grouping=5,
            ordinal_tasks=['ordinal_test'],
            ordinal_max_score=10,
            score_agg='median',  # CRITICAL: Triggers continuous inference
            shadow_mode=False
        )

        # Create mock task and samples
        evalspec = create_mock_evalspec(model='gpt-4', task='ordinal_test')
        samples = create_mock_samples(n_samples=50)

        # Start task
        await manager.start_task(evalspec, samples, epochs=20)

        # Simulate evaluation with consistent high performance
        stopped_count = 0
        for sample_idx in range(50):
            for epoch in range(1, 21):
                sample_id = f"sample_{sample_idx}"

                # Generate 3 ordinal scores with high ratings (8-10)
                ordinal_scores = np.random.randint(8, 11, size=3)
                median_score = float(np.median(ordinal_scores))  # ~9.0

                # Create mock scores (3 scorers, aggregated via median)
                scores = {
                    'scorer_1': create_mock_score(float(ordinal_scores[0])),
                    'scorer_2': create_mock_score(float(ordinal_scores[1])),
                    'scorer_3': create_mock_score(float(ordinal_scores[2]))
                }

                # Check if should run
                directive = await manager.schedule_sample(sample_id, epoch)
                if directive is not None:
                    stopped_count += 1
                    break

                # Complete sample
                await manager.complete_sample(sample_id, epoch, scores)

        # Complete task
        metadata = await manager.complete_task()

        # Assertions
        assert stopped_count > 20, f"Should stop at least 20 samples, got {stopped_count}"
        assert metadata['efficiency_percent'] > 30, "Should achieve >30% efficiency"

        # Check stopping reasons
        stopping_reasons = set()
        for grouping_stops in manager._stopped_sample_ids.values():
            for stop in grouping_stops:
                reason = metadata.get('sample_stopping_reasons', {}).get(stop.id, {}).get('reason')
                if reason:
                    stopping_reasons.add(reason)

        assert 'continuous_bounded_ci_width' in stopping_reasons, \
            "Should use continuous bounded inference"

    @pytest.mark.asyncio
    async def test_continuous_vs_discrete_efficiency_comparison(self):
        """Compare efficiency: continuous aggregated vs discrete non-aggregated."""

        # Test 1: Discrete binary (baseline)
        manager_discrete = OptimalStoppingManager(
            optstop_params={'delta_item': 0.05, 'delta_cap': 0.05},
            grouping_columns=['model', 'task'],
            reanalysis_interval=10,
            score_agg=None,  # No aggregation - discrete inference
            shadow_mode=False
        )

        # Test 2: Continuous aggregated
        manager_continuous = OptimalStoppingManager(
            optstop_params={'delta_item': 0.05, 'delta_cap': 0.05},
            grouping_columns=['model', 'task'],
            reanalysis_interval=10,
            score_agg='mean',  # Aggregation - continuous inference
            shadow_mode=False
        )

        # Run both with identical data pattern
        efficiency_discrete = await run_scenario(manager_discrete, n_samples=30, epochs=15)
        efficiency_continuous = await run_scenario(manager_continuous, n_samples=30, epochs=15)

        # Continuous should achieve reasonable efficiency (may be lower than discrete)
        assert efficiency_continuous > 20, \
            f"Continuous inference should achieve >20% efficiency, got {efficiency_continuous}%"

        # Continuous may be less efficient than discrete (expected)
        # But should still provide value
        print(f"Discrete efficiency: {efficiency_discrete}%")
        print(f"Continuous efficiency: {efficiency_continuous}%")
```

### Phase 3 Deliverables

- [ ] Continuous sample-level stopping integrated
- [ ] Continuous group-level stopping integrated
- [ ] Stabilization criteria working for continuous case
- [ ] Integration tests passing
- [ ] Efficiency benchmarks documented

**Outcome**: Full pipeline working end-to-end for aggregated scores.

---

## Testing & Validation Strategy

### Unit Tests
- [ ] `test_continuous_bounded_inference.py`: Core statistical method
- [ ] `test_determine_score_type.py`: Enhanced routing logic
- [ ] `test_validation_logic.py`: Score validation with aggregation

### Integration Tests
- [ ] `test_continuous_early_stopping_integration.py`: Full pipeline
- [ ] `test_aggregation_modes.py`: Mean, median, mode aggregation

### Simulation Studies
- [ ] CI coverage validation (95% coverage for 95% CIs)
- [ ] Efficiency comparison: continuous vs discrete
- [ ] Performance benchmarks: execution time

### Documentation
- [ ] Update README with aggregation examples
- [ ] Add docstrings for all new functions
- [ ] Update user guide with continuous inference section

---

## Expected Outcomes

### Performance Expectations

| Score Type | Efficiency (consistent) | Efficiency (overall) | Speed |
|------------|------------------------|---------------------|-------|
| **Binary (discrete)** | 45% | 30% | Fast (2-3s) |
| **Ordinal (discrete)** | 91% | 68% | Fast (2-3s) |
| **Continuous [0,1]** | 30-50% (estimated) | 20-35% (estimated) | Fast (2-3s) |
| **Continuous [0,10]** | 40-60% (estimated) | 25-40% (estimated) | Fast (2-3s) |

**Note**: Continuous efficiency expected to be intermediate between binary and discrete ordinal, depending on score consistency.

### Success Criteria

- [ ] **Phase 1**: Users see clear warnings when using aggregation without continuous inference
- [ ] **Phase 2**: `_continuous_bounded_ci_adaptive()` achieves 95% coverage in simulation
- [ ] **Phase 3**: Integration tests show >20% efficiency for consistent continuous patterns
- [ ] **Overall**: No breaking changes to existing binary/ordinal inference
- [ ] **Overall**: Documentation and examples provided for aggregation use cases

---

## Risk Mitigation

### Risk 1: Continuous inference less efficient than discrete
- **Mitigation**: Document expected efficiency ranges, provide comparison benchmarks
- **Fallback**: Users can choose not to aggregate (use mode or single scorer)

### Risk 2: Beta distribution assumptions violated for some data
- **Mitigation**: Add diagnostic checks, warn if variance too high
- **Fallback**: Implement alternative (truncated normal) if needed

### Risk 3: Performance regression for existing users
- **Mitigation**: Extensive regression testing, backward compatibility validation
- **Fallback**: Phased rollout with feature flag

---

## Implementation Timeline

### Phase 1: Detection & Routing
- **Estimated Time**: 1-2 days
- **Priority**: HIGH (prevents invalid inference)

### Phase 2: Core Statistical Method
- **Estimated Time**: 2-3 days
- **Priority**: MEDIUM (enables functionality)

### Phase 3: Integration & Testing
- **Estimated Time**: 3-4 days
- **Priority**: MEDIUM (completes feature)

**Total Estimated Time**: 6-9 days

---

## Files to Create/Modify

### New Files
- [ ] `tests/test_continuous_bounded_inference.py`
- [ ] `tests/test_continuous_bounded_coverage.py`
- [ ] `tests/test_continuous_early_stopping_integration.py`
- [ ] `CONTINUOUS_SCORE_IMPLEMENTATION_PLAN.md` (this document)

### Modified Files
- [ ] `optstop/ordinal_utils.py` (enhance `determine_score_type`)
- [ ] `optstop/rule.py` (add `_continuous_bounded_ci_adaptive`, modify routing)
- [ ] `optstop/early_stopping.py` (update validation, pass aggregation context)
- [ ] `tests/test_early_stopping.py` (update existing tests if needed)

---

## References

### Statistical Methods
- Beta distribution for bounded continuous data
- Method of moments parameter estimation
- Bayesian credible intervals
- Rubin, D. B. (1981). The Bayesian Bootstrap. The Annals of Statistics, 9(1), 130-134.

### Related Documentation
- `FINAL_ORDINAL_COMPARISON_REPORT.md`: Documents 4% efficiency for continuous floats with modal inference
- `COMPREHENSIVE_TEST_REPORT_FINAL.md`: Testing framework for early stopping

---

## Conclusion

This three-phase implementation plan provides a clear path to supporting continuous bounded score inference for aggregated scores. The approach is:

✅ **Statistically rigorous** - Uses appropriate beta distribution for bounded continuous data
✅ **Architecturally clean** - Follows existing pattern of separate inference routes
✅ **Backward compatible** - No changes to existing binary/ordinal behavior
✅ **Well-tested** - Comprehensive unit, integration, and simulation tests
✅ **Production-ready** - Phased rollout with clear success criteria

**Next Steps**: Begin Phase 1 implementation to enable proper detection and routing for aggregated scores.
