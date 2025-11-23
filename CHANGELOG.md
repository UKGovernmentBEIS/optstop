# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Full bridge usage guide (BRIDGE_USAGE_GUIDE.md)
- API reference documentation (BRIDGE_API_REFERENCE.md)
- Example evaluations in `examples/inspect_ai/`
- Comprehensive integration tests for bridge

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