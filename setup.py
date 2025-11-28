from setuptools import setup, find_packages
import os

# Read version from __version__.py
version = {}
version_file = os.path.join('optstop', '__version__.py')
with open(version_file) as f:
    exec(f.read(), version)

# Read long description from README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='optstop',
    version=version['__version__'],
    description='Adaptive optimal stopping rule algorithms for efficient data collection and analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Toby D. Pilditch, PhD',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'pymc>=5.10.0',
        'arviz',
        'scipy',
        'statsmodels',
        'matplotlib',
        'tqdm',
        'numba',  # Required by PyTensor for OrderedLogistic (ordinal inference)
        'pydantic>=2.0',  # Required for inspect_ai bridge
    ],
    extras_require={
        'inspect': [
            'inspect-ai>=0.3.0',
        ],
        'gpu': [
            'jax[cuda12]>=0.4.0',
        ],
        'performance': [
            'jax>=0.4.0',  # JAX CPU backend for ordinal models
            'numpyro>=0.13.0',  # NumPyro sampler for ordinal inference (2-3× CPU speedup)
        ],
        'dev': [
            'pytest>=7.0',
            'pytest-asyncio>=0.21',
            'pytest-cov>=4.0',
            'black>=22.0',
            'flake8>=5.0',
            'mypy>=1.0',
        ],
    },
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'optstop-posthoc=optstop.cli:main',
            'optstop-live=optstop.cli:main_live',
            'optstop-convergence=optstop.cli:main_convergence',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
    ],
) 