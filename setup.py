"""
setup.py for optstop package.

This file mirrors the metadata in pyproject.toml for compatibility with
older pip versions and environments where PEP 621 support is incomplete.
When updating dependencies or metadata, update both files.
"""

from setuptools import setup, find_packages

# Read version from __version__.py (single source of truth)
version_info = {}
with open("optstop/__version__.py") as f:
    exec(f.read(), version_info)

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="optstop",
    version=version_info["__version__"],
    description="Adaptive optimal stopping rule algorithms for efficient data collection and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Toby D. Pilditch, PhD",
    license="MIT",
    url="https://github.com/UKGovernmentBEIS/optstop",
    project_urls={
        "Repository": "https://github.com/UKGovernmentBEIS/optstop.git",
        "Issues": "https://github.com/UKGovernmentBEIS/optstop/issues",
        "Changelog": "https://github.com/UKGovernmentBEIS/optstop/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    # Lower bounds only; exact validated versions are pinned in
    # requirements-lock.txt (see pyproject.toml for rationale).
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "pymc>=5.10.0",
        "arviz>=0.17",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "tqdm>=4.60",
        "numba>=0.58",
        "pydantic>=2.0",
        "typing_extensions>=4.0.0; python_version < '3.12'",
    ],
    extras_require={
        "inspect": [
            "inspect-ai>=0.3.0",
        ],
        "gpu": [
            "jax[cuda12_pip]; sys_platform != 'darwin'",
            "jax[metal]; sys_platform == 'darwin'",
            "numpyro",
            "blackjax",
        ],
        "performance": [
            "jax>=0.4.0",
            "numpyro>=0.13.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
            "pytest-cov>=4.0",
            # Optional per-test isolation for debugging a single test (run the
            # full suite per file instead); see pyproject.toml / README for rationale.
            "pytest-forked>=1.6.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "optstop-posthoc=optstop.cli:main",
            "optstop-live=optstop.cli:main_live",
            "optstop-convergence=optstop.cli:main_convergence",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
)
