# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### MCMC Failure Handling and Diagnostics
- All 14 `pm.sample()` call sites now wrapped in try/except - MCMC sampling failures no longer crash worker processes
- New `_log_mcmc_diagnostics` helper logs divergence count, min ESS, and max R-hat at WARNING level when concerning (DEBUG otherwise)
- On sampling failure, the package returns max-uncertainty estimates (CI width = 1.0), preventing premature stopping while allowing the next scheduled MCMC call to retry with more data

### Changed

#### Ordinal Hybrid Stopping (Pathway 2) - Mechanism Fix
- `entropy_threshold` default changed from 0.7 to 0.8 (more permissive false-peak gate for Pathway 1)
- `entropy_stabilization_threshold` renamed to `entropy_convergence_threshold` (old name accepted with deprecation warning)
- Pathway 2 mechanism changed from relative-change detection (default 0.002) to absolute entropy CI width convergence on [0,1] scale (default 0.10)
  - The old relative-change criterion was mathematically incapable of firing under exponential posterior convergence
  - The new absolute-width criterion is guaranteed to be satisfied in finite time as the posterior concentrates
- `prior_sigma` parameter now exposed at all API levels (default: 1.5 for binary/continuous, 2.0 for ordinal)

### Upgrade Notes
- **`entropy_stabilization_threshold`**: The old parameter name still works but triggers a deprecation warning. The value semantics have changed - old values (e.g., 0.002) will be passed through but are on the wrong scale for the new mechanism. Remove custom values to use the new default (0.10), or set `entropy_convergence_threshold` explicitly.
- **`entropy_threshold`**: The default is now 0.8 (was 0.7). Users who relied on the old default without setting it explicitly will see slightly more permissive Pathway 1 gating.

### Planned
- Full bridge usage guide (BRIDGE_USAGE_GUIDE.md)
- Example evaluations in `examples/inspect_ai/`
- Comprehensive integration tests for bridge

---

## [0.2.1] - 2025-11-29

### Added

#### Reproducibility Features
- `random_seed` parameter for `OptimalStoppingManager`
  - User can specify seed for reproducible MCMC inference
  - If not specified, auto-generates seed using system entropy
  - Seed is always logged immediately at manager initialization
  - Seed included in `complete_task()` diagnostics output
  - Seed passed directly to PyMC via `sampling_kwargs['random_seed']`

#### Configurable Ordinal Stopping
- `entropy_stabilization_threshold` parameter in `optstop_params`
  - Controls ordinal hybrid Pathway 2 (entropy stabilization) sensitivity
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
  - No change in behavior for users

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
- **Stabilization histories**: Per-grouping tracking of CI widths, slopes, and entropy

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