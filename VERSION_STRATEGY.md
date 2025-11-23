# optstop Version Strategy

**Document Version:** 1.0
**Date:** 2025-11-23
**Status:** Active

---

## Table of Contents

1. [Versioning Scheme](#versioning-scheme)
2. [Version Components](#version-components)
3. [Compatibility Policy](#compatibility-policy)
4. [inspect_ai Integration Versioning](#inspect_ai-integration-versioning)
5. [Release Process](#release-process)
6. [Version History](#version-history)

---

## Versioning Scheme

optstop follows **Semantic Versioning 2.0.0** (https://semver.org/).

### Format: `MAJOR.MINOR.PATCH`

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality in a backwards-compatible manner
- **PATCH**: Backwards-compatible bug fixes

### Pre-release versions

For development and testing:
- **Alpha**: `X.Y.Z-alpha.N` (internal testing only)
- **Beta**: `X.Y.Z-beta.N` (external testing, API may change)
- **Release Candidate**: `X.Y.Z-rc.N` (final testing before release)

---

## Version Components

### Core Package Version

The package version is defined in a single source of truth:

```
optstop/__version__.py
```

This file contains:

```python
__version__ = "X.Y.Z"
__version_info__ = (X, Y, Z)

# Component versions for transparency
__core_version__ = "X.Y.Z"           # Core optstop algorithms
__bridge_version__ = "X.Y.Z"         # inspect_ai bridge
__min_inspect_ai_version__ = "X.Y.Z" # Minimum compatible inspect_ai version
```

### Usage in Code

```python
import optstop
print(optstop.__version__)  # "0.2.0"
```

---

## Compatibility Policy

### Python Version Support

- **Minimum Python**: 3.10 (aligned with inspect_ai requirement)
- **Tested Python versions**: 3.10, 3.11, 3.12
- **Policy**: Support Python versions released in last 3 years

### Dependency Version Support

| Dependency | Minimum Version | Reason |
|------------|----------------|--------|
| pymc | 5.10.0 | GPU support and modern API |
| numpy | (auto) | Latest stable |
| pandas | (auto) | Latest stable |
| pydantic | 2.0+ | Required for inspect_ai integration |
| inspect_ai | 0.3.0+ | EarlyStopping protocol support |

### Backwards Compatibility Promise

#### MAJOR version (X.0.0)
- May break existing code
- Migration guide provided
- Deprecation warnings in previous MINOR version

#### MINOR version (0.X.0)
- New features added
- Existing functionality preserved
- Optional new parameters (with defaults)
- May deprecate features (with warnings)

#### PATCH version (0.0.X)
- Bug fixes only
- No API changes
- No new features

---

## inspect_ai Integration Versioning

### Integration Status

The inspect_ai bridge (`optstop.early_stopping.OptimalStoppingManager`) is **part of the core package**, not a separate distribution.

### Compatibility Matrix

| optstop Version | inspect_ai Version | Status | Notes |
|-----------------|-------------------|--------|-------|
| 0.2.0+ | 0.3.0+ | Supported | Initial bridge release |
| 0.1.x | N/A | No bridge | Pre-bridge versions |

### inspect_ai Breaking Changes

If inspect_ai changes its `EarlyStopping` protocol:

1. **Minor change**: optstop PATCH release (e.g., 0.2.1)
2. **Major change**: optstop MINOR release (e.g., 0.3.0) with fallback support
3. **Protocol deprecation**: optstop MAJOR release (e.g., 1.0.0)

### Feature Parity

The bridge aims for **feature parity** with standalone optstop:

- ✅ Binary scoring (discrete)
- ✅ Ordinal scoring (discrete)
- ✅ Continuous bounded scoring (aggregated)
- ✅ Hybrid inference modes
- ✅ GPU acceleration
- ✅ All stopping criteria

Any core optstop feature should be available via the bridge.

---

## Release Process

### 1. Pre-Release

- [ ] Update `optstop/__version__.py`
- [ ] Update `CHANGELOG.md` with release notes
- [ ] Run full test suite (including bridge tests)
- [ ] Update documentation (README, API docs)
- [ ] Update compatibility matrix

### 2. Release

- [ ] Tag release in git: `git tag -a vX.Y.Z -m "Release X.Y.Z"`
- [ ] Push tag: `git push origin vX.Y.Z`
- [ ] Build package: `python -m build`
- [ ] Upload to PyPI: `python -m twine upload dist/*`

### 3. Post-Release

- [ ] Verify installation: `pip install optstop==X.Y.Z`
- [ ] Update GitHub release notes
- [ ] Announce release (if MINOR/MAJOR)
- [ ] Bump version to next development version (e.g., X.Y.Z+1-alpha.0)

---

## Version History

### 0.2.0 (2025-11-23) - Current

**Status:** In Development

**Changes:**
- **[MINOR]** Added inspect_ai bridge (`OptimalStoppingManager`)
- **[MINOR]** Added `early_stopping.py` module with full EarlyStopping protocol support
- **[MINOR]** Added support for aggregated continuous scores (mean, median, mode, max)
- **[MINOR]** Added score validation framework for binary vs. ordinal tasks
- **[MINOR]** Added shadow mode for A/B testing
- **[ENHANCEMENT]** Updated Python requirement to 3.10+ (aligned with inspect_ai)
- **[ENHANCEMENT]** Added pydantic 2.0+ dependency
- **[ENHANCEMENT]** Added optional inspect_ai dependency
- **[ENHANCEMENT]** Improved process cleanup with ThreadPoolExecutor management
- **[DOCS]** Added VERSION_STRATEGY.md
- **[DOCS]** Added BRIDGE_TESTING_AND_DEVELOPMENT_ROADMAP.md
- **[DOCS]** Updated README with inspect_ai integration section

**Breaking Changes:**
- None (new functionality only)

**Upgrade Notes:**
- If using inspect_ai, install with: `pip install optstop[inspect]`
- Python 3.10+ now required (was 3.7+)

---

### 0.1.1 (2024-XX-XX)

**Changes:**
- Bug fixes and performance improvements
- No API changes

---

### 0.1.0 (2024-XX-XX)

**Initial Release:**
- Post-hoc optimal stopping
- Live optimal stopping
- Convergence analysis
- Binary and ordinal scoring
- GPU acceleration
- CLI tools

---

## Future Roadmap

### 0.3.0 (Planned)

**Potential features:**
- Enhanced diagnostics for inspect_ai evaluations
- Real-time visualization of stopping decisions
- Cost estimation for LLM evaluations
- Parallel inference across groupings

### 1.0.0 (Future)

**Criteria for 1.0.0 release:**
- ✅ Comprehensive test coverage (>90%)
- ✅ Full documentation
- ✅ Production validation by external users
- ✅ Stable API (no breaking changes expected)
- ✅ inspect_ai bridge fully tested in real evaluations
- ✅ Performance benchmarks established

---

## Versioning Best Practices

### For Developers

1. **Always update `__version__.py` first** before any release
2. **Never commit version bumps with features** - separate commits
3. **Use pre-release versions** for testing (alpha, beta, rc)
4. **Tag all releases** in git for traceability
5. **Update CHANGELOG.md** with every release

### For Users

1. **Pin major version** in production: `optstop>=0.2,<1.0`
2. **Use exact version** for reproducibility: `optstop==0.2.0`
3. **Check compatibility matrix** before updating inspect_ai
4. **Read CHANGELOG.md** before upgrading

---

## FAQ

### Q: Will optstop 0.2.x work with inspect_ai 0.4.x?

**A:** We aim for forward compatibility, but test compatibility before deploying to production. Check the compatibility matrix above.

### Q: What happens if I use optstop without inspect_ai installed?

**A:** All core functionality works. The bridge (`OptimalStoppingManager`) will use mock classes for testing but will raise an ImportError if you try to use it with real inspect_ai evaluations without installing inspect_ai.

### Q: Can I use different optstop versions for core algorithms vs. bridge?

**A:** No, the bridge is part of the core package. Both use the same version.

### Q: How do I know which inspect_ai version to use?

**A:** Check `optstop.__min_inspect_ai_version__` or the compatibility matrix in this document.

---

## Contact

For versioning questions or compatibility issues:
- Open an issue on GitHub
- Check CHANGELOG.md for version-specific notes
- Consult the compatibility matrix above
