# Code Changes: Ordinal Diagnostic Transparency

**Issue**: Ordinal stopping decisions show "(no metrics)" in logs and diagnostics, preventing users from verifying stopping decisions.

**Root Cause**: The `_ordinal_hybrid_stopping_criterion()` function returns rich diagnostics, but the logging code in `early_stopping.py` doesn't extract the correct keys for ordinal-specific metrics.

---

## Change #1: Update Ordinal Diagnostic Extraction in `early_stopping.py`

**File**: `optstop/early_stopping.py`
**Location**: Lines 1218-1228 (inside `_run_stopping_inference()` method)

### Current Code:
```python
# Ordinal-specific diagnostics (hybrid, entropy)
if 'diagnostics' in group_reason_info:
    diag = group_reason_info['diagnostics']
    if isinstance(diag, dict):
        if 'entropy' in diag:
            metrics.append(f"entropy={diag['entropy']:.4f}")
        if 'modal_prob' in diag:
            metrics.append(f"modal prob={diag['modal_prob']:.4f}")
        if 'ci_width' in diag and 'ci_width' not in group_reason_info:
            # Add from diagnostics if not already in main metadata
            metrics.append(f"CI width={diag['ci_width']:.4f}")
```

### Updated Code:
```python
# Ordinal-specific diagnostics (hybrid, entropy)
if 'diagnostics' in group_reason_info:
    diag = group_reason_info['diagnostics']
    if isinstance(diag, dict):
        # Ordinal hybrid diagnostics from _ordinal_hybrid_stopping_criterion
        # Pathway 1: modal_ci_narrow_validated
        if 'modal_width' in diag:
            modal_width = diag['modal_width']
            metrics.append(f"modal CI width={modal_width:.4f}")
        if 'modal_ci' in diag and isinstance(diag['modal_ci'], (list, tuple)):
            modal_lo, modal_hi = diag['modal_ci']
            metrics.append(f"modal CI=[{modal_lo:.2f}, {modal_hi:.2f}]")
        if 'entropy_median' in diag:
            entropy_median = diag['entropy_median']
            metrics.append(f"entropy={entropy_median:.4f}")
        if 'entropy_threshold' in diag:
            entropy_threshold = diag['entropy_threshold']
            metrics.append(f"entropy threshold={entropy_threshold:.2f}")
        if 'threshold' in diag and 'threshold' not in group_reason_info:
            threshold = diag['threshold']
            metrics.append(f"modal threshold={threshold}")

        # Pathway 2: entropy_stabilized
        if 'entropy_width' in diag:
            entropy_width = diag['entropy_width']
            metrics.append(f"entropy CI width={entropy_width:.4f}")
        if 'relative_change' in diag:
            relative_change = diag['relative_change']
            metrics.append(f"relative change={relative_change:.4f}")
        if 'stabilization_threshold' in diag:
            stab_threshold = diag['stabilization_threshold']
            metrics.append(f"stab threshold={stab_threshold:.4f}")

        # Legacy checks for backwards compatibility
        if 'entropy' in diag and 'entropy_median' not in diag:
            metrics.append(f"entropy={diag['entropy']:.4f}")
        if 'modal_prob' in diag:
            metrics.append(f"modal prob={diag['modal_prob']:.4f}")
        if 'ci_width' in diag and 'ci_width' not in group_reason_info and 'modal_width' not in diag:
            metrics.append(f"CI width={diag['ci_width']:.4f}")
```

---

## Change #2: Add Plain-English Explanations to Diagnostics

**File**: `optstop/early_stopping.py`
**Location**: After line 1234, before the final log message

### New Code to Insert:
```python
# Add plain-English explanation for ordinal stopping
explanation = None
if reason in ['modal_ci_narrow_validated', 'entropy_stabilized'] and 'diagnostics' in group_reason_info:
    diag = group_reason_info['diagnostics']
    if reason == 'modal_ci_narrow_validated':
        modal_width = diag.get('modal_width', 'N/A')
        entropy_median = diag.get('entropy_median', 'N/A')
        threshold = diag.get('threshold', 'N/A')
        entropy_threshold = diag.get('entropy_threshold', 1.5)
        explanation = (
            f"Grouping stopped because modal CI width ({modal_width:.4f}) was below "
            f"threshold ({threshold}) and entropy ({entropy_median:.4f}) confirmed the "
            f"distribution was peaked (< {entropy_threshold:.2f})."
        )
    elif reason == 'entropy_stabilized':
        relative_change = diag.get('relative_change', 'N/A')
        stab_threshold = diag.get('stabilization_threshold', 'N/A')
        entropy_width = diag.get('entropy_width', 'N/A')
        explanation = (
            f"Grouping stopped because entropy CI stabilized with relative change "
            f"({relative_change:.4f}) below threshold ({stab_threshold:.4f}), indicating "
            f"diminishing returns from additional data."
        )

# Format log message with explanation
metrics_str = ", ".join(metrics) if metrics else "no metrics"
log_msg = f"Stopped grouping '{grouping_name}' after {samples_used} samples: {reason} ({metrics_str})"
if explanation:
    logger.info(log_msg)
    logger.info(f"  → {explanation}")
else:
    logger.info(log_msg)
```

### Replace Lines 1231-1235 with:
```python
# Add plain-English explanation for ordinal stopping
explanation = None
if reason in ['modal_ci_narrow_validated', 'entropy_stabilized'] and 'diagnostics' in group_reason_info:
    diag = group_reason_info['diagnostics']
    if reason == 'modal_ci_narrow_validated':
        modal_width = diag.get('modal_width', 'N/A')
        entropy_median = diag.get('entropy_median', 'N/A')
        threshold = diag.get('threshold', 'N/A')
        entropy_threshold = diag.get('entropy_threshold', 1.5)
        if isinstance(modal_width, (int, float)) and isinstance(entropy_median, (int, float)):
            explanation = (
                f"Modal CI width ({modal_width:.4f}) < threshold ({threshold}), "
                f"entropy ({entropy_median:.4f}) < {entropy_threshold:.2f} (peaked distribution)"
            )
    elif reason == 'entropy_stabilized':
        relative_change = diag.get('relative_change', 'N/A')
        stab_threshold = diag.get('stabilization_threshold', 'N/A')
        if isinstance(relative_change, (int, float)) and isinstance(stab_threshold, (int, float)):
            explanation = (
                f"Entropy stabilized: relative change ({relative_change:.4f}) < "
                f"threshold ({stab_threshold:.4f})"
            )

# Format log message with explanation
metrics_str = ", ".join(metrics) if metrics else "no metrics"
log_msg = f"Stopped grouping '{grouping_name}' after {samples_used} samples: {reason} ({metrics_str})"
if explanation:
    logger.info(log_msg)
    logger.info(f"  → {explanation}")
else:
    logger.info(log_msg)
```

---

## Change #3: Populate Stabilization History for Ordinal Pathways

**File**: `optstop/rule.py`
**Location**: After line 2716 (after `stabilization_history['entropy_history'] = group_entropy_history`)

### Problem:
Ordinal pathways don't populate `ci_width` and `ci_slope` in stabilization_histories, leaving them as `null`.

### New Code to Insert:
```python
# Populate stabilization history metrics for ordinal pathways
# This enables users to see final stopping metrics in diagnostics
if ordinal_inference == 'hybrid' and stop_this_grouping:
    # Extract metrics from diagnostics_group for history
    if diagnostics_group:
        if 'modal_width' in diagnostics_group:
            # Pathway 1: modal CI narrow
            stabilization_history['final_modal_ci_width'] = float(diagnostics_group['modal_width'])
            stabilization_history['final_entropy'] = float(diagnostics_group.get('entropy_median', 0))
        elif 'entropy_width' in diagnostics_group:
            # Pathway 2: entropy stabilized
            stabilization_history['final_entropy_ci_width'] = float(diagnostics_group['entropy_width'])
            stabilization_history['final_relative_change'] = float(diagnostics_group.get('relative_change', 0))
```

### Updated Location (After line 2716):
```python
stabilization_history['entropy_history'] = group_entropy_history

# NEW: Populate stabilization history metrics for ordinal pathways
if ordinal_inference == 'hybrid' and stop_this_grouping:
    if diagnostics_group:
        # Store pathway indicator
        stabilization_history['ordinal_pathway'] = diagnostics_group.get('pathway', 0)

        # Pathway 1: modal_ci_narrow_validated
        if 'modal_width' in diagnostics_group:
            stabilization_history['final_modal_ci_width'] = float(diagnostics_group['modal_width'])
            stabilization_history['final_entropy'] = float(diagnostics_group.get('entropy_median', 0))
            stabilization_history['final_entropy_threshold'] = float(diagnostics_group.get('entropy_threshold', 1.5))
            if 'modal_ci' in diagnostics_group:
                stabilization_history['final_modal_ci'] = [float(x) for x in diagnostics_group['modal_ci']]

        # Pathway 2: entropy_stabilized
        if 'entropy_width' in diagnostics_group:
            stabilization_history['final_entropy_ci_width'] = float(diagnostics_group['entropy_width'])
            stabilization_history['final_relative_change'] = float(diagnostics_group.get('relative_change', 0))
            stabilization_history['final_stabilization_threshold'] = float(diagnostics_group.get('stabilization_threshold', 0.002))
```

---

## Change #4: Add Glossary to Complete Task Diagnostics

**File**: `optstop/early_stopping.py`
**Location**: In `complete_task()` method, before return statement (around line 1312)

### New Code to Insert Before `return metadata`:
```python
# Add glossary for ordinal stopping reasons
if any('ordinal' in str(col).lower() for col in self.grouping_columns) or self.ordinal_tasks:
    metadata['ordinal_glossary'] = {
        'modal_ci_narrow_validated': {
            'description': 'Bootstrap modal confidence interval was narrow and validated by low entropy',
            'interpretation': 'Distribution is peaked (most responses in same category) with high certainty',
            'metrics': 'modal_width < threshold AND entropy < entropy_threshold'
        },
        'entropy_stabilized': {
            'description': 'Distribution entropy converged, indicating stable ordinal estimates',
            'interpretation': 'Additional data provides diminishing returns (entropy not changing)',
            'metrics': 'relative_change < stabilization_threshold (default 0.002 = 0.2%)'
        },
        'continue_insufficient_history': {
            'description': 'Need more epochs to assess stabilization',
            'interpretation': 'Collecting more data to establish convergence pattern',
            'metrics': 'epochs_tracked < min_epochs_for_stabilization (default 3)'
        }
    }
```

---

## Change #5: Add "Why NOT Stopped" for Continued Groupings

**File**: `optstop/early_stopping.py`
**Location**: In `complete_task()` method, when building stabilization_histories summary (around line 1304)

### Current Code (Lines 1304-1312):
```python
"stabilization_histories": {
    k: {
        'n_samples': v.get('n_samples_evaluated', 0),
        'final_ci_width': v['ci_width_history'][-1] if v.get('ci_width_history') else None,
        'final_slope': v['ci_slope_history'][-1] if v.get('ci_slope_history') else None,
        'n_group_checks': len(v.get('ci_width_history', []))
    }
    for k, v in self._stabilization_histories.items()
}
```

### Updated Code:
```python
"stabilization_histories": {
    k: {
        'n_samples': v.get('n_samples_evaluated', 0),
        'final_ci_width': v['ci_width_history'][-1] if v.get('ci_width_history') else None,
        'final_slope': v['ci_slope_history'][-1] if v.get('ci_slope_history') else None,
        'n_group_checks': len(v.get('ci_width_history', [])),
        # Ordinal-specific fields
        'final_modal_ci_width': v.get('final_modal_ci_width'),
        'final_entropy': v.get('final_entropy'),
        'final_entropy_ci_width': v.get('final_entropy_ci_width'),
        'final_relative_change': v.get('final_relative_change'),
        'ordinal_pathway': v.get('ordinal_pathway'),
        # Status explanation
        'stopped': k in self._stopped_groupings,
        'continued_reason': self._get_continued_reason(k, v) if k not in self._stopped_groupings else None
    }
    for k, v in self._stabilization_histories.items()
}
```

### New Helper Method to Add to `OptimalStoppingManager` class:
```python
def _get_continued_reason(self, grouping_name: str, history: dict) -> dict:
    """Generate explanation for why a grouping continued (didn't stop).

    Args:
        grouping_name: Name of the grouping
        history: Stabilization history dictionary

    Returns:
        Dictionary with reason and metrics explaining why stopping criteria weren't met
    """
    reason = {}

    # Binary/Continuous pathways
    if 'ci_width_history' in history and history['ci_width_history']:
        final_ci_width = history['ci_width_history'][-1]
        threshold = self.optstop_params.get('delta_cap', 0.05)

        reason['final_ci_width'] = float(final_ci_width)
        reason['ci_threshold'] = float(threshold)
        reason['ci_width_met'] = final_ci_width < threshold

        if 'ci_slope_history' in history and history['ci_slope_history']:
            final_slope = history['ci_slope_history'][-1]
            slope_threshold = self.optstop_params.get('CI_delta', 0.00005)

            reason['final_slope'] = float(final_slope)
            reason['slope_threshold'] = float(slope_threshold)
            reason['stabilization_met'] = abs(final_slope) < slope_threshold

            # Determine primary reason
            if not reason['ci_width_met']:
                reason['primary_reason'] = 'ci_width_not_met'
                reason['explanation'] = f"CI width ({final_ci_width:.4f}) did not meet threshold ({threshold})"
            elif not reason['stabilization_met']:
                reason['primary_reason'] = 'stabilization_not_met'
                reason['explanation'] = f"CI not stabilized (slope {abs(final_slope):.6f} > {slope_threshold:.6f})"
            else:
                reason['primary_reason'] = 'criteria_met_but_not_triggered'
                reason['explanation'] = "Stopping criteria met but inference not triggered (check reanalysis_interval)"

    # Ordinal pathways
    elif 'final_modal_ci_width' in history:
        modal_width = history['final_modal_ci_width']
        threshold = self.optstop_params.get('delta_item', 0.15)
        entropy = history.get('final_entropy', 0)
        entropy_threshold = history.get('final_entropy_threshold', 1.5)

        reason['final_modal_ci_width'] = float(modal_width)
        reason['modal_threshold'] = float(threshold)
        reason['modal_width_met'] = modal_width < threshold
        reason['final_entropy'] = float(entropy)
        reason['entropy_threshold'] = float(entropy_threshold)
        reason['entropy_validated'] = entropy < entropy_threshold

        if not reason['modal_width_met']:
            reason['primary_reason'] = 'modal_ci_not_narrow'
            reason['explanation'] = f"Modal CI width ({modal_width:.4f}) did not meet threshold ({threshold})"
        elif not reason['entropy_validated']:
            reason['primary_reason'] = 'false_peak_detected'
            reason['explanation'] = f"Modal CI narrow but entropy too high ({entropy:.4f} > {entropy_threshold}), indicating uncertain distribution"

    elif 'final_entropy_ci_width' in history:
        entropy_width = history['final_entropy_ci_width']
        relative_change = history.get('final_relative_change', 1.0)
        stab_threshold = history.get('final_stabilization_threshold', 0.002)

        reason['final_entropy_ci_width'] = float(entropy_width)
        reason['final_relative_change'] = float(relative_change)
        reason['stabilization_threshold'] = float(stab_threshold)
        reason['entropy_stabilized'] = relative_change < stab_threshold

        if not reason['entropy_stabilized']:
            reason['primary_reason'] = 'entropy_not_stabilized'
            reason['explanation'] = f"Entropy CI still changing (relative change {relative_change:.4f} > {stab_threshold:.4f})"

    return reason if reason else {'primary_reason': 'unknown', 'explanation': 'Insufficient data to determine'}
```

---

## Testing the Changes

### Test Case 1: Verify Ordinal Metrics in Logs
```python
# Run ordinal evaluation and check log output
manager = OptimalStoppingManager(
    optstop_params={'delta_item': 0.15, 'delta_cap': 0.1},
    grouping_columns=['model', 'task'],
    ordinal_tasks=['confidence'],
    ordinal_max_score=10,
    ordinal_inference='hybrid'
)

# After running, check logs for:
# OLD: "Stopped grouping 'X' after 10 samples: modal_ci_narrow_validated (no metrics)"
# NEW: "Stopped grouping 'X' after 10 samples: modal_ci_narrow_validated (modal CI width=0.0000, modal CI=[0.00, 0.00], entropy=0.3924, entropy threshold=1.50, modal threshold=0.15)"
#      "  → Modal CI width (0.0000) < threshold (0.15), entropy (0.3924) < 1.50 (peaked distribution)"
```

### Test Case 2: Verify Stabilization History
```python
# Check diagnostics JSON after completion
diagnostics = log.early_stopping
hist = diagnostics['stabilization_histories']['model-task']

# Should now contain:
assert 'final_modal_ci_width' in hist or 'final_entropy_ci_width' in hist
assert 'ordinal_pathway' in hist
assert 'continued_reason' in hist if not hist['stopped'] else True
```

---

## Summary of Improvements

| Aspect | Before | After | Impact |
|--------|--------|-------|--------|
| **Ordinal group stopping logs** | "(no metrics)" | "modal CI width=0.0000, entropy=0.3924, ..." | ✅ Users can verify decisions |
| **Plain-English explanations** | None | "Modal CI width < threshold, peaked distribution" | ✅ Non-experts understand |
| **Stabilization history** | null values | Actual metrics (modal_width, entropy, etc.) | ✅ Programmatic verification |
| **Continued groupings** | No explanation | "CI width not met: 0.3079 > 0.1" | ✅ Debug why didn't stop |
| **Glossary** | None | Definitions of stopping reasons | ✅ Self-documenting |

---

## Implementation Priority

**CRITICAL (for production):**
1. Change #1: Extract ordinal metrics correctly ← **REQUIRED**
2. Change #3: Populate stabilization history ← **REQUIRED**

**IMPORTANT (for UX):**
3. Change #2: Add plain-English explanations
4. Change #4: Add glossary

**NICE-TO-HAVE (for debugging):**
5. Change #5: Explain continued groupings
