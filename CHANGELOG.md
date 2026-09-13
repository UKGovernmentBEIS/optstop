# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

#### Sorted-Input Warning: Natural ID Order, and Visible from the CLI
- `optimal_stopping_posthoc` warns when a grouping's sample IDs appear in sorted
  order and `shuffle_items` is off. The check compared IDs as strings only, so IDs
  numbered `0, 1, 2, ..., 10` or `item_1, item_2, ..., item_10` - which are not
  sorted as strings - were never flagged. The check now also recognises natural
  order (runs of digits compared as numbers), ascending or descending.
- `optstop-posthoc` logs to a file only, so this warning never reached the
  console. It is now also printed on stderr as a single `WARNING:` line; the exit
  code is unchanged and every other warning still goes to the log file alone. The
  warning is emitted on a dedicated child logger, `optstop.posthoc.input_order`,
  so it is still written to the log file as before.

## [0.5.0] - 2026-09-03

Behavioural fix for group-level credible intervals at very low success rates
(GitHub issue #3). Mid-range results are unchanged; the change affects only how
low-rate intervals are reported.

### Fixed

#### Resolution Floor on Group-Level Credible Intervals (issue #3)
- Group-level binary and continuous credible intervals were previously floored
  at `sigmoid(-6) = 0.00247...`, because the reported interval was read from the
  same clamped deterministic used by the likelihood. An absent or very low-rate
  capability was therefore unrepresentable, and the continuous pathway could
  report a narrow sub-floor interval and stop on it.
- The reported interval is now read from an unclipped reporting transform
  (`Theta_report` for binary, `mu_item_report` for continuous), so the lower
  bound can approach 0 and the interval can contain a low truth.
- The likelihood retains the `±6` logit clamp (sampler stability), and the
  continuous likelihood retains its separate variance clamp
  (`clip(mu_item, 0.01, 0.99)`). Consequently, **below the resolution floor the
  reported bounds are prior-dominated and qualitative - the hard floor is
  removed, but sub-floor coverage is not calibrated.** Report the raw proportion
  alongside the interval for rates below ~0.25%.

#### CLI, Packaging, and CI Reliability (issue #4)
- The CLI entry points (`optstop-posthoc`, `optstop-convergence`, and
  `optstop-live`) now **exit non-zero** when every grouping fails (the run
  computed nothing) or when no groupings are produced at all, printing the
  failures to stderr. On partial failure they still exit 0 but print a loud
  stderr warning identifying which groupings failed. Previously a run in which
  every grouping raised still printed its "saved"/decision output and exited 0,
  so the failure was visible only inside the summary CSV's `error` column
  (posthoc and convergence) or the log file (live).
- Added a version-tolerant `arviz.hdi` wrapper (`optstop._compat.hdi`) used at
  every group-level HDI call site. It calls `az.hdi(..., hdi_prob=...)` first
  (the validated arviz 0.x API) and falls back to `prob=` on newer arviz where
  the keyword was renamed, so the CLI no longer fails wholesale with
  `hdi got an unexpected keyword argument: 'hdi_prob'` on recent arviz. This
  handles the `hdi` keyword only; full arviz-1.x support is not claimed.
- CLI entry points now force UTF-8 on stdout/stderr, preventing
  `UnicodeEncodeError` from non-ASCII output on Windows consoles (cp1252/cp437).

### Changed

#### Low-Performance Stopping Gate
- At the **group** level, automatic precision and slope-stabilisation stops are
  now suppressed for **all three pathways** (binary, continuous, and ordinal)
  while observed performance is below `low_performance_threshold`. Previously the
  width and slope thresholds were only relaxed (via `conservatism`) below the
  threshold, so accumulated null data could still stop before a rare event was
  ever observed - the opposite of what a low-base-rate search needs. Holding the
  stop until the observed rate clears the threshold keeps the search sampling
  until the event appears; the post-event interval can then trigger the ordinary
  precision stop. This makes `low_performance_threshold` usable as a
  detection-then-stop trigger (see README, "Detecting low base-rate
  capabilities").
- At the **item/sample** level the gate is applied to the binary and continuous
  pathways only. Ordinal items need no separate gate: the ordinal estimators
  already apply a sample-size-scaled minimum-width floor (`1/(ordinal_max_score·√n)`) that holds
  an item off a premature stop on homogeneous near-zero data. That floor decays
  as `1/√n`, however, so on a genuinely absent capability the ordinal group modal
  width eventually crosses `delta_cap` - which is why the group-level suppression
  above is applied to ordinal as well, as a backstop.
- At very low continuous rates the reported interval could also narrow
  spuriously, giving an additional reason to withhold the stop until the observed
  rate clears the threshold. (Note: the continuous likelihood applies a separate
  variance clamp, `clip(mu_item, 0.01, 0.99)`, in this regime; the interaction
  with the reported width has not been fully characterised and is not relied on
  here.)

### Added
- `pinned` field (default `False`) on binary/continuous group-level results and
  in the bridge `stabilization_histories`: `True` when the whole reported
  interval lies below the resolution floor, i.e. the estimate is qualitative.
- `low_perf_floor` field (default `False`) on ordinal group-level results and in
  the bridge `stabilization_histories`: `True` when the resolved normalised
  performance is below `low_performance_threshold`. This is the ordinal
  counterpart to `pinned`, but is telemetry only and carries no reliability
  caveat - the ordinal estimator has no location clamp, so the low estimate is
  data-faithful. Does not affect stopping.
- `low_perf_stop_suppressed` field indicating a group-level stop (binary,
  continuous, or ordinal) was withheld because observed performance was below
  `low_performance_threshold`.
- `LOGIT_CLAMP` and `RESOLUTION_FLOOR` module constants in `rule.py`.
- README subsection "Detecting low base-rate capabilities" describing how to use
  `low_performance_threshold` as a detection-then-stop trigger.
- Python 3.13 added to the package classifiers - in both `pyproject.toml` and the
  retained `setup.py` - and to the CI test matrix (issue #4).
- CI now runs the test suite one file per process to avoid a full-suite PyMC
  model-context cascade, with `fail-fast: false` and `PYTHONIOENCODING=utf-8`
  (issue #4).
- Installation instructions now use the GitHub `pip install "git+https://..."`
  form, since optstop is not yet published on PyPI (issue #4).

### Upgrade Notes
- Expect wider, lower intervals **only** for group-level rates below ~0.25%;
  mid-range and high-rate results are unchanged (the reporting transform equals
  the old value wherever the posterior stays within `±6` logits).
- `pinned=True` marks whole-interval-sub-floor groupings: treat the estimate
  qualitatively and report the raw proportion as well.
- To detect a rare capability and stop once it is observed, set
  `low_performance_threshold` to the success rate of interest (see README,
  "Detecting low base-rate capabilities").

---

## [0.4.0] - 2026-03-18

### Added

#### MCMC Failure Handling and Diagnostics
- All 14 `pm.sample()` call sites now wrapped in try/except - MCMC sampling failures no longer crash worker processes
- New `_log_mcmc_diagnostics` helper logs divergence count, min ESS, and max R-hat at WARNING level when concerning (DEBUG otherwise)
- On sampling failure, the package returns max-uncertainty estimates (CI width = 1.0), preventing premature stopping while allowing the next scheduled MCMC call to retry with more data

#### Convergence Projection Stable Companions
- New `simple_proj_additional_trials` field: conservative 1/√n projection, stable across item orderings (CV 0.03-0.09 vs 0.47-3.16 for the exponential projection)
- New `trajectory_signal` field: qualitative categorical (`'faster'`/`'on_pace'`/`'slower'`) indicating whether the trajectory-based projection suggests faster or slower convergence than the 1/√n baseline
- Existing `projected_additional_trials` retained but documented as order-sensitive (treat as rough guide)

#### New Features
- Epoch-interleaved processing order (`processing_order='epoch_interleaved'`) for posthoc analysis, matching bridge trial ordering
- `shuffle_items` and `shuffle_seed` parameters for posthoc item-order randomisation
- `score_value_key` parameter for bridge integration with dict-valued scores

### Changed

#### Ordinal Hybrid Stopping (Pathway 2) - Mechanism Fix
- `entropy_threshold` default changed from 0.7 to 0.8 (more permissive false-peak gate for Pathway 1)
- `entropy_stabilization_threshold` renamed to `entropy_convergence_threshold` (old name accepted with deprecation warning)
- Pathway 2 mechanism changed from relative-change detection (default 0.002) to absolute entropy CI width convergence on [0,1] scale (default 0.10)
  - The old relative-change criterion was mathematically incapable of firing under exponential posterior convergence
  - The new absolute-width criterion is guaranteed to be satisfied in finite time as the posterior concentrates
- `prior_sigma` parameter now exposed at all API levels (default: 1.5 for binary/continuous, 2.0 for ordinal)

#### Default Changes
- `conservatism` default changed from 10 to 5
- `low_performance_threshold` default changed from 0.001 to 0.01 (1%)
- `target_accept` unified to 0.95 for both CPU and GPU (was 0.90 for CPU)

### Fixed
- `convergence_posthoc` missing `entropy_convergence_threshold` in function signature (caused NameError)
- 8 pre-existing test failures resolved (stale assertions, missing fixtures, parameter naming)

### Removed
- `setup.py` removed; `pyproject.toml` is now the sole build configuration
  (note: `setup.py` was later reinstated and is present as of 0.5.0)

### Upgrade Notes
- **`entropy_stabilization_threshold`**: The old parameter name still works but triggers a deprecation warning. The value semantics have changed - old values (e.g., 0.002) will be passed through but are on the wrong scale for the new mechanism. Remove custom values to use the new default (0.10), or set `entropy_convergence_threshold` explicitly.
- **`entropy_threshold`**: The default is now 0.8 (was 0.7). Users who relied on the old default without setting it explicitly will see slightly more permissive Pathway 1 gating.
- **`conservatism`**: Default changed from 10 to 5. Users who relied on the old default will see less aggressive CI inflation for low-performance groupings.
- **`low_performance_threshold`**: Default changed from 0.001 to 0.01. The conservatism mechanism now activates for groupings with performance below 1% (was 0.1%).

### Planned
- Full bridge usage guide (BRIDGE_USAGE_GUIDE.md)
- Example evaluations in `examples/inspect_ai/`
- Comprehensive integration tests for bridge

---

## [0.3.1] - 2026-01-23

### Added
- Continuous score type support: full hierarchical model for `continuous_01` and `continuous_bounded` in both posthoc and live modes
- Convergence projection for non-stopped groupings (exponential decay model with linear fallback, bootstrap uncertainty)
- Diagnostic logging for posterior analysis (mu_group, sigma_group statistics)

### Fixed
- CI extraction methodology: changed from individual item HDI to `mean(Theta)` / `mean(mu_item)` across all items, correctly accounting for between-item variance

---

## [0.3.0] - 2025-12-15

### Added
- Ordered Logistic (cumulative link) model as default for ordinal inference (~3x faster than Dirichlet)
- `ordinal_model_type` parameter for model selection ('ordered_logistic' or 'dirichlet')
- Adaptive cutpoint priors scaling with number of categories
- Automatic fallback to Dirichlet if Ordered Logistic sampling fails

---

## [0.2.1] - 2025-11-29

### Added

#### Reproducibility Features
- `random_seed` parameter for `OptimalStoppingManager`
  - User can specify seed for reproducible MCMC inference
  - If not specified, auto-generates seed using system entropy
  - Seed is always logged immediately at manager initialisation
  - Seed included in `complete_task()` diagnostics output
  - Seed passed directly to PyMC via `sampling_kwargs['random_seed']`

#### Configurable Ordinal Stopping
- `entropy_stabilization_threshold` parameter in `optstop_params`
  - Controls ordinal hybrid Pathway 2 (entropy stabilisation) sensitivity
  - Default: 0.002 (0.2% relative change threshold)
  - Lower values = more conservative (require more stability)
  - Higher values = more aggressive (stop with less stability)

#### Backend Improvements
- **Numpyro/JAX integration** for ordinal inference (~2× CPU speedup vs PyMC default)
  - Automatic detection and configuration
  - Zero configuration required for users
  - Stable operation (0 failures in large-scale testing)

### Changed

#### Internal Improvements
- Removed redundant `np.random.seed()` calls in `rule.py` and `convergence.py`
  - Seed now passed directly to PyMC via sampling_kwargs
  - Cleaner, more reliable seed propagation
  - No change in behaviour for users

#### Documentation
- Updated `BRIDGE_API_REFERENCE.md` (v1.1)
  - Added `random_seed` parameter documentation
  - Added `entropy_stabilization_threshold` documentation
  - New "Reproducibility" section with examples
  - Updated diagnostics structure with new fields
  - Updated version history

### Performance

#### Large-Scale Testing Results (500 samples × 3 datasets)
| Dataset | Type | Runtime | Efficiency |
|---------|------|---------|------------|
| Binary | Discrete | 6.89 min | 48.26% |
| Ordinal (numpyro) | Discrete | 575 min* | 54.1% |
| Continuous | Bounded | 8.73 min | 59.98% |

*Total wall time for standalone posthoc mode (synchronous inference). In bridge mode with background threading, ordinal evaluations complete in virtually identical wall time to other pathways - see Performance Considerations in BRIDGE_API_REFERENCE.md.

- Ordinal inference with numpyro: ~1.9× faster per MCMC call vs baseline
- Overall ordinal runtime: ~26% faster than PyMC default

### Upgrade Notes
- **No breaking changes** - all existing code continues to work
- **New optional parameters**:
  - `random_seed` on `OptimalStoppingManager` (optional, auto-generates if not specified)
  - `entropy_stabilization_threshold` in `optstop_params` (optional, defaults to 0.002)
- **Diagnostics structure updated**: Now includes `random_seed` and `seed_source` fields

---

## [0.2.0] - 2025-11-23

### Added

#### inspect_ai Integration (MAJOR FEATURE)
- **New module**: `optstop.early_stopping` with `OptimalStoppingManager` class
- Implements the `EarlyStopping` protocol from inspect_ai framework
- Enables adaptive early stopping for LLM evaluation workflows
- Full lifecycle support: `start_task()`, `schedule_sample()`, `complete_sample()`, `complete_task()`
- Comprehensive diagnostics and efficiency metrics returned from `complete_task()`

#### Bridge Features
- **Flexible grouping**: Support for grouping by model, task, metadata keys, or tags
- **Multiple score handling**:
  - Extract specific score by key (`score_choice` parameter)
  - Aggregate multiple scores (`score_agg`: mean, median, mode, max)
- **Ordinal scoring**: Full support for ordinal tasks in bridge (modal, entropy, hybrid inference)
- **Continuous bounded scores**: Support for aggregated binary/ordinal scores
- **Shadow mode**: Run all trials while tracking what would have stopped (for A/B testing)
- **Score validation**: Comprehensive validation for binary vs. ordinal, discrete vs. continuous
- **Process cleanup**: Proper ThreadPoolExecutor management with graceful shutdown
- **Stabilisation histories**: Per-grouping tracking of CI widths, slopes, and entropy

#### Infrastructure & Versioning
- **Version management**: Created `optstop/__version__.py` for programmatic version access
- **Version checking**: `check_inspect_ai_compatibility()` function for version validation
- **Programmatic version info**: `__version__`, `__version_info__`, `__core_version__`, `__bridge_version__`
- **Documentation**: VERSION_STRATEGY.md defining versioning approach and compatibility matrix
- **Development roadmap**: BRIDGE_TESTING_AND_DEVELOPMENT_ROADMAP.md for comprehensive testing plan

#### Documentation
- Comprehensive inspect_ai integration section added to README
- Installation instructions for `pip install optstop[inspect]`
- Quick start example with OptimalStoppingManager
- Configuration parameters reference table
- Troubleshooting guide for common issues
- Best practices for LLM evaluations with optimal stopping

### Changed

#### Breaking Changes
- **Python requirement**: Updated from `>=3.7` to `>=3.10` (aligns with inspect_ai requirement)
  - **Rationale**: inspect_ai requires Python 3.10+, so bridge requires same minimum

#### Dependencies
- Added `pydantic>=2.0` as core dependency (required for inspect_ai bridge)
- Added `inspect-ai>=0.3.0` as optional dependency (`pip install optstop[inspect]`)
- Added development tools as optional dependency (`pip install optstop[dev]`):
  - pytest>=7.0, pytest-asyncio>=0.21, pytest-cov>=4.0
  - black>=22.0, flake8>=5.0, mypy>=1.0

#### Setup Configuration
- `setup.py` now reads version from `__version__.py` (single source of truth)
- Added `long_description` from README.md for better PyPI presentation
- Added package classifiers for improved discoverability
- Defined `extras_require` for optional dependencies: `[inspect]`, `[gpu]`, `[dev]`

### Fixed
- Improved process cleanup to prevent lingering PyMC worker processes
- Fixed executor shutdown with `wait=True` to ensure clean termination
- Enhanced async safety for inference operations

### Internal Improvements
- Added comprehensive type hints throughout `early_stopping.py`
- Added detailed docstrings for all public methods
- Improved logging with structured messages and context
- Enhanced error handling with graceful degradation
- Async-safe inference execution using ThreadPoolExecutor
- Fast DataFrame lookups with caching for `schedule_sample()`
- Efficient cache invalidation on stopping decisions

### Upgrade Notes
- **Python 3.10+ now required** (was 3.7+) - upgrade Python if needed
- **New installation options**:
  - `pip install optstop[inspect]` - for inspect_ai integration
  - `pip install optstop[inspect,gpu]` - with GPU acceleration
  - `pip install optstop[dev]` - for development
- **No breaking changes to existing API** - all standalone optstop functionality unchanged
- **New optional imports**: `from optstop.early_stopping import OptimalStoppingManager`

### Performance
- Same performance characteristics as 0.1.x for standalone usage
- Bridge adds minimal overhead (< 1ms per `schedule_sample()` call)
- Inference runs in dedicated ThreadPoolExecutor to avoid blocking event loop

---

## [0.1.0] - 04-08-2025

### Added
- Initial release
- Core optimal stopping algorithms
- CLI interface
- Test suite
- Documentation