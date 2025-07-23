# Contributing to optstop

Thank you for your interest in contributing to optstop! This document provides guidelines for contributing to this project.

## Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/optstop.git
   cd optstop
   ```
3. **Install in development mode**:
   ```bash
   pip install -e .
   ```
4. **Install development dependencies**:
   ```bash
   pip install pytest
   ```

## Running Tests

Before submitting any changes, please ensure all tests pass:

```bash
python -m pytest tests/ -v
```

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions
- Include type hints where appropriate

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
- Update documentation if needed
- Ensure all tests pass

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Any error messages

## Questions?

If you have questions about contributing, please open an issue on GitHub. 