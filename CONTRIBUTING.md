# Contributing to optstop

Thank you for your interest in contributing to optstop! This document provides guidelines for contributing to this project.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/UKGovernmentBEIS/optstop.git
   cd optstop
   ```
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   ```
3. **Install in development mode** with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
   For GPU support (NVIDIA or Apple Silicon):
   ```bash
   pip install -e ".[dev,gpu]"
   ```

## Running Tests

Before submitting any changes, ensure tests pass:

```bash
python -m pytest tests/ -v
```

Notes on the test suite:
- Some tests involve real MCMC sampling and take 1-3 minutes each.
- Tests marked `@pytest.mark.slow` (ordinal entropy MCMC) take >3 minutes. Deselect with `-m 'not slow'`.
- A small number of tests are inherently stochastic (depend on MCMC randomness) and may occasionally fail. If a failure appears unrelated to your changes, re-run to confirm.
- The package uses `pymc`, `arviz`, and `numpyro` for Bayesian inference - these produce compilation caches that the test fixtures clean up automatically.

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions
- Include type hints where appropriate
- Use spaced hyphens (` - `) in documentation, not em-dashes

## Submitting Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes** and commit them:
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```
3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
4. **Create a Pull Request** on GitHub

## Pull Request Guidelines

- Provide a clear description of the changes
- Include tests for new functionality
- Update documentation if needed (README.md, BRIDGE_API_REFERENCE.md, CHANGELOG.md)
- Ensure all tests pass (excluding known stochastic failures)

## Reporting Issues

When reporting issues, please include:
- optstop version (`optstop-posthoc --version`)
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Any error messages or log output

## Questions?

If you have questions about contributing, please open an issue on [GitHub](https://github.com/UKGovernmentBEIS/optstop/issues).
