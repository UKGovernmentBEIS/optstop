# Release Checklist

Use this checklist when preparing a new release of optstop.

## Pre-Release

- [ ] Update version number in `pyproject.toml`
- [ ] Update version number in `optstop/__init__.py` (if using `__version__`)
- [ ] Run all tests: `python -m pytest tests/ -v`
- [ ] Update CHANGELOG.md with new features/fixes
- [ ] Review and update documentation if needed
- [ ] Test installation from source: `pip install -e .`
- [ ] Test CLI commands work: `optstop-posthoc --help`

## Release

- [ ] Create a new tag: `git tag v0.1.0`
- [ ] Push tag: `git push origin v0.1.0`
- [ ] Create GitHub release with release notes
- [ ] Update PyPI (if distributing there)

## Post-Release

- [ ] Update version number for next development cycle
- [ ] Create new development branch if needed

## Version Numbering

- **Major.Minor.Patch** (e.g., 0.1.0)
- **Major**: Breaking changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible 