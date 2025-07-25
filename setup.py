from setuptools import setup, find_packages

setup(
    name='optstop',
    version='0.1.0',
    description='Adaptive optimal stopping rule algorithms for efficient data collection and analysis',
    author='Toby D. Pilditch, PhD',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'pymc',
        'arviz',
        'scipy',
        'statsmodels',
        'matplotlib',
        'tqdm',
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'optstop-posthoc=optstop.cli:main',
            'optstop-live=optstop.cli:main_live',
            'optstop-convergence=optstop.cli:main_convergence',
        ],
    },
) 