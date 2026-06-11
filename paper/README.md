# Paper Materials

This directory contains supporting materials for the paper:

> Pilditch, T. D. (2026). *Knowing When to Stop: Bayesian Optimal Stopping for LLM Evaluations*. In *Proceedings of the 43rd International Conference on Machine Learning (ICML 2026)*.

These materials are intended for reproducibility and inspection. They are **not** part of the `optstop` Python package install and will not be downloaded when installing `optstop` from PyPI or building a wheel from this repository.

## Contents

- `Optimal Stopping for LLM Evaluations - Pilditch (2026).pdf` - The camera-ready paper (ICML 2026 proceedings version).
- `simulations/` - Standalone simulation scripts and their outputs. Notably, `model_comparison_binary.py` and its output under `simulations/output/` underpin Appendix B.6 (Logit-Normal vs Beta-Binomial Model Comparison).
- `validation/` - Validation analyses cited in the appendix:
  - `consistency/` - Presentation-order robustness (Appendix B.10)
  - `conservatism/` - Conservatism sensitivity (Appendix B.11)
  - `fixed_n/` - Fixed-n baseline comparison (Appendix B.9)
  - `stabilisation/` - CI stabilisation validation (Appendix B.12)
  - `scripts/` - Driver scripts for the above
  - `validation_notes.md` - Working notes accompanying the validation runs
- `paper_results/` - Data and figures underlying Section 3 and the appendix:
  - `data/` - Per-cell shadow-mode outputs
  - `hibayes/` - HiBayES analysis outputs (Analyses A-D, ROPE sensitivity)
  - `meta/` - Hierarchical meta-analysis outputs
  - `figures/` - Figures included in the paper
  - `RESULTS_SUMMARY.md` - High-level summary of the 3x3 matrix experiment

## Reproducing the analyses

The Python scripts in `simulations/` and `validation/scripts/` use the `optstop` package as installed from the repository root. From a virtual environment with `optstop` installed, scripts can be run directly, though many of the longer simulations require multi-hour MCMC runs. Output paths in the scripts are relative to the original working directory used during the paper preparation; you may need to adjust them.

## Citing

If you use these materials in your own work, please cite the paper above and the `optstop` package itself (see the project root `README.md` for citation details).
