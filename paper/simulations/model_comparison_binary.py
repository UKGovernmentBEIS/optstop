#!/usr/bin/env python3
"""
Beta-Binomial vs Logit-Normal Hierarchical Model Comparison

This simulation compares two hierarchical models for binary evaluation data:
1. Logit-Normal: mu_item = mu_group + sigma_group * z, theta = sigmoid(mu_item)
2. Beta-Binomial: theta ~ Beta(alpha, beta), parameterized by mean and concentration

Data is generated from a Beta-Binomial process (favorable to that model) to provide
a fair comparison. We evaluate: bias, CI width, coverage, MCMC divergences, and
implied stopping behavior.

Author: Claude (Anthropic) for optstop paper
Date: 2026-03-13
"""

import numpy as np
import pymc as pm
import arviz as az
from scipy import stats
from scipy.special import expit, logit
import pandas as pd
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run."""
    true_p: float  # True population success probability
    n_items: int  # Number of items (samples)
    n_trials_per_item: int  # Number of trials per item
    concentration: float  # Beta concentration parameter (higher = less heterogeneity)
    seed: int
    cred_level: float = 0.94  # Credible interval level (matching HiBayES)
    n_draws: int = 1000  # Matches optstop package default
    n_chains: int = 4
    target_accept: float = 0.95  # Matches optstop package default (gpu_utils.py)


@dataclass
class ModelResult:
    """Results from fitting a single model."""
    model_name: str
    point_estimate: float
    ci_low: float
    ci_high: float
    ci_width: float
    n_divergences: int
    ess_bulk: float
    ess_tail: float
    r_hat: float
    covered: bool  # Does CI contain true value?
    bias: float  # point_estimate - true_p


def generate_beta_binomial_data(
    true_p: float,
    concentration: float,
    n_items: int,
    n_trials_per_item: int,
    seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic data from a Beta-Binomial process.

    Parameters
    ----------
    true_p : float
        True population success probability (0 to 1)
    concentration : float
        Beta concentration parameter. Higher = less between-item variance.
        alpha = true_p * concentration, beta = (1 - true_p) * concentration
    n_items : int
        Number of items/samples
    n_trials_per_item : int
        Number of Bernoulli trials per item
    seed : int
        Random seed

    Returns
    -------
    successes : np.ndarray
        Number of successes per item
    trials : np.ndarray
        Number of trials per item (constant)
    true_thetas : np.ndarray
        True item-level success probabilities (for coverage checking)

    Note on boundary cases (true_p = 0.0 or 1.0)
    --------------------------------------------
    At exact boundaries, the Beta distribution is degenerate (alpha=0 or beta=0).
    We handle this by setting all item-level probabilities to exactly 0 or 1,
    which means there is ZERO between-item heterogeneity in these cases.
    This represents the "perfect classifier" or "completely broken classifier"
    scenario - a valid but qualitatively different regime from mid-range cases
    where items have genuine variance around the population mean.
    """
    rng = np.random.default_rng(seed)

    # Beta parameters from mean and concentration
    alpha = true_p * concentration
    beta = (1 - true_p) * concentration

    # Handle boundary cases
    if true_p <= 0:
        true_thetas = np.zeros(n_items)
    elif true_p >= 1:
        true_thetas = np.ones(n_items)
    else:
        # Draw item-level success probabilities from Beta
        true_thetas = rng.beta(alpha, beta, size=n_items)

    # Draw successes from Binomial
    trials = np.full(n_items, n_trials_per_item)
    successes = rng.binomial(n_trials_per_item, true_thetas)

    return successes, trials, true_thetas


def fit_logit_normal_model(
    successes: np.ndarray,
    trials: np.ndarray,
    config: SimulationConfig
) -> ModelResult:
    """
    Fit a logit-normal hierarchical model (the optstop default).

    Model:
        mu_group ~ Normal(0, 1.5)
        sigma_group ~ Exponential(1)
        z_i ~ Normal(0, 1)
        eta_i = mu_group + sigma_group * z_i
        theta_i = sigmoid(eta_i)
        successes_i ~ Binomial(trials_i, theta_i)
    """
    n_items = len(successes)

    with pm.Model() as model:
        # Priors (matching optstop defaults)
        mu_group = pm.Normal("mu_group", mu=0, sigma=1.5)
        sigma_group = pm.Exponential("sigma_group", lam=1.0)

        # Non-centered parameterization
        z = pm.Normal("z", mu=0, sigma=1, shape=n_items)
        eta = pm.Deterministic("eta", mu_group + sigma_group * z)

        # Clip to avoid numerical issues
        eta_clipped = pm.math.clip(eta, -6.0, 6.0)
        theta = pm.Deterministic("theta", pm.math.sigmoid(eta_clipped))

        # Population mean: E[theta] computed as mean of item-level probabilities.
        # Note: This differs from sigmoid(mu_group) due to Jensen's inequality
        # when sigma_group > 0. This is the correct target for optstop, which
        # computes mean(Theta) across items in rule.py L5065-5070.
        Theta = pm.Deterministic("Theta", pm.math.mean(theta))

        # Likelihood
        pm.Binomial("obs", n=trials, p=theta, observed=successes)

        # Sample
        try:
            trace = pm.sample(
                draws=config.n_draws,
                chains=config.n_chains,
                target_accept=config.target_accept,
                random_seed=config.seed,
                progressbar=False,
                return_inferencedata=True
            )
        except Exception as e:
            # Return failed result
            return ModelResult(
                model_name="logit_normal",
                point_estimate=np.nan,
                ci_low=np.nan,
                ci_high=np.nan,
                ci_width=np.nan,
                n_divergences=-1,
                ess_bulk=np.nan,
                ess_tail=np.nan,
                r_hat=np.nan,
                covered=False,
                bias=np.nan
            )

    # Extract results
    theta_samples = trace.posterior["Theta"].values.flatten()
    point_estimate = float(np.mean(theta_samples))

    # HDI (matching optstop's actual CI computation)
    hdi_result = az.hdi({"Theta": theta_samples}, hdi_prob=config.cred_level)
    ci_low = float(hdi_result["Theta"].sel(hdi="lower").values)
    ci_high = float(hdi_result["Theta"].sel(hdi="higher").values)
    ci_width = ci_high - ci_low

    # Diagnostics
    summary = az.summary(trace, var_names=["Theta", "mu_group", "sigma_group"])
    n_divergences = int(trace.sample_stats["diverging"].sum())

    theta_summary = summary.loc["Theta"]
    ess_bulk = float(theta_summary.get("ess_bulk", np.nan))
    ess_tail = float(theta_summary.get("ess_tail", np.nan))
    r_hat = float(theta_summary.get("r_hat", np.nan))

    # Coverage
    covered = (ci_low <= config.true_p <= ci_high)
    bias = point_estimate - config.true_p

    return ModelResult(
        model_name="logit_normal",
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_width=ci_width,
        n_divergences=n_divergences,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        r_hat=r_hat,
        covered=covered,
        bias=bias
    )


def fit_beta_binomial_model(
    successes: np.ndarray,
    trials: np.ndarray,
    config: SimulationConfig
) -> ModelResult:
    """
    Fit a Beta-Binomial hierarchical model.

    Model:
        mu ~ Beta(2, 2)  # Prior on population mean
        kappa ~ Gamma(2, 0.1)  # Concentration parameter
        theta_i ~ Beta(mu * kappa, (1-mu) * kappa)
        successes_i ~ Binomial(trials_i, theta_i)
    """
    n_items = len(successes)

    with pm.Model() as model:
        # Priors on mean and concentration
        mu = pm.Beta("mu", alpha=2, beta=2)  # Population mean
        kappa = pm.Gamma("kappa", alpha=2, beta=0.1)  # Concentration

        # Reparameterize: alpha = mu * kappa, beta = (1-mu) * kappa
        alpha_param = pm.Deterministic("alpha_param", mu * kappa)
        beta_param = pm.Deterministic("beta_param", (1 - mu) * kappa)

        # Item-level probabilities
        theta = pm.Beta("theta", alpha=alpha_param, beta=beta_param, shape=n_items)

        # Population mean: computed as mean of item-level probabilities for
        # fair comparison with logit-normal. Note: In Beta-Binomial, the 'mu'
        # parameter IS the population mean directly, but we use mean(theta)
        # for symmetric comparison with the logit-normal model.
        Theta = pm.Deterministic("Theta", pm.math.mean(theta))

        # Likelihood
        pm.Binomial("obs", n=trials, p=theta, observed=successes)

        # Sample
        try:
            trace = pm.sample(
                draws=config.n_draws,
                chains=config.n_chains,
                target_accept=config.target_accept,
                random_seed=config.seed,
                progressbar=False,
                return_inferencedata=True
            )
        except Exception as e:
            return ModelResult(
                model_name="beta_binomial",
                point_estimate=np.nan,
                ci_low=np.nan,
                ci_high=np.nan,
                ci_width=np.nan,
                n_divergences=-1,
                ess_bulk=np.nan,
                ess_tail=np.nan,
                r_hat=np.nan,
                covered=False,
                bias=np.nan
            )

    # Extract results
    theta_samples = trace.posterior["Theta"].values.flatten()
    point_estimate = float(np.mean(theta_samples))

    # HDI (matching optstop's actual CI computation)
    hdi_result = az.hdi({"Theta": theta_samples}, hdi_prob=config.cred_level)
    ci_low = float(hdi_result["Theta"].sel(hdi="lower").values)
    ci_high = float(hdi_result["Theta"].sel(hdi="higher").values)
    ci_width = ci_high - ci_low

    # Diagnostics
    summary = az.summary(trace, var_names=["Theta", "mu", "kappa"])
    n_divergences = int(trace.sample_stats["diverging"].sum())

    theta_summary = summary.loc["Theta"]
    ess_bulk = float(theta_summary.get("ess_bulk", np.nan))
    ess_tail = float(theta_summary.get("ess_tail", np.nan))
    r_hat = float(theta_summary.get("r_hat", np.nan))

    # Coverage
    covered = (ci_low <= config.true_p <= ci_high)
    bias = point_estimate - config.true_p

    return ModelResult(
        model_name="beta_binomial",
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_width=ci_width,
        n_divergences=n_divergences,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        r_hat=r_hat,
        covered=covered,
        bias=bias
    )


def run_single_comparison(config: SimulationConfig) -> Dict:
    """Run a single comparison between the two models."""
    # Generate data
    successes, trials, true_thetas = generate_beta_binomial_data(
        true_p=config.true_p,
        concentration=config.concentration,
        n_items=config.n_items,
        n_trials_per_item=config.n_trials_per_item,
        seed=config.seed
    )

    # Fit both models
    logit_result = fit_logit_normal_model(successes, trials, config)
    beta_result = fit_beta_binomial_model(successes, trials, config)

    return {
        "config": asdict(config),
        "observed_mean": float(np.sum(successes) / np.sum(trials)),
        "logit_normal": asdict(logit_result),
        "beta_binomial": asdict(beta_result)
    }


def run_simulation_grid(
    output_dir: Path,
    n_replications: int = 25,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run the full simulation grid.

    Grid:
    - true_p: [0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
    - n_items: [20, 50, 100]
    - n_trials_per_item: [10]
    - concentration: [10] (moderate heterogeneity)
    """
    # Define grid
    true_p_values = [0.0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
    n_items_values = [20, 50, 100]
    n_trials_per_item = 10
    concentration = 10  # Moderate heterogeneity

    results = []
    total_runs = len(true_p_values) * len(n_items_values) * n_replications
    run_count = 0

    for true_p in true_p_values:
        for n_items in n_items_values:
            for rep in range(n_replications):
                run_count += 1
                if verbose and run_count % 10 == 0:
                    print(f"  Progress: {run_count}/{total_runs} ({100*run_count/total_runs:.1f}%)")

                config = SimulationConfig(
                    true_p=true_p,
                    n_items=n_items,
                    n_trials_per_item=n_trials_per_item,
                    concentration=concentration,
                    seed=42 + rep + int(true_p * 1000) + n_items
                )

                try:
                    result = run_single_comparison(config)
                    result["replication"] = rep
                    results.append(result)
                except Exception as e:
                    if verbose:
                        print(f"    Error at p={true_p}, n={n_items}, rep={rep}: {e}")

    # Convert to DataFrame for analysis
    rows = []
    for r in results:
        row = {
            "true_p": r["config"]["true_p"],
            "n_items": r["config"]["n_items"],
            "replication": r["replication"],
            "observed_mean": r["observed_mean"],
        }
        for model in ["logit_normal", "beta_binomial"]:
            for key, value in r[model].items():
                if key != "model_name":
                    row[f"{model}_{key}"] = value
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save raw results
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "model_comparison_raw.csv", index=False)

    with open(output_dir / "model_comparison_full.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return df


def summarize_results(df: pd.DataFrame, output_dir: Path) -> Dict:
    """
    Compute summary statistics for the simulation.

    Returns summary dict and saves to JSON.
    """
    # Group by true_p and n_items
    summary_rows = []

    for (true_p, n_items), group in df.groupby(["true_p", "n_items"]):
        n_reps = len(group)

        row = {
            "true_p": true_p,
            "n_items": n_items,
            "n_replications": n_reps,
        }

        for model in ["logit_normal", "beta_binomial"]:
            prefix = model

            # Bias
            biases = group[f"{prefix}_bias"].dropna()
            row[f"{prefix}_mean_bias"] = float(biases.mean()) if len(biases) > 0 else np.nan
            row[f"{prefix}_abs_bias"] = float(biases.abs().mean()) if len(biases) > 0 else np.nan

            # CI width
            widths = group[f"{prefix}_ci_width"].dropna()
            row[f"{prefix}_mean_ci_width"] = float(widths.mean()) if len(widths) > 0 else np.nan

            # Coverage
            covered = group[f"{prefix}_covered"].dropna()
            row[f"{prefix}_coverage"] = float(covered.mean()) if len(covered) > 0 else np.nan

            # Divergences
            divs = group[f"{prefix}_n_divergences"].dropna()
            row[f"{prefix}_mean_divergences"] = float(divs.mean()) if len(divs) > 0 else np.nan
            row[f"{prefix}_total_divergences"] = int(divs.sum()) if len(divs) > 0 else 0

            # ESS (Effective Sample Size)
            ess_bulk = group[f"{prefix}_ess_bulk"].dropna()
            ess_tail = group[f"{prefix}_ess_tail"].dropna()
            row[f"{prefix}_mean_ess_bulk"] = float(ess_bulk.mean()) if len(ess_bulk) > 0 else np.nan
            row[f"{prefix}_mean_ess_tail"] = float(ess_tail.mean()) if len(ess_tail) > 0 else np.nan
            row[f"{prefix}_min_ess_bulk"] = float(ess_bulk.min()) if len(ess_bulk) > 0 else np.nan

        # Comparative metrics
        row["bias_ratio"] = (
            row["beta_binomial_abs_bias"] / row["logit_normal_abs_bias"]
            if row["logit_normal_abs_bias"] > 0.001 else np.nan
        )
        row["ci_width_ratio"] = (
            row["beta_binomial_mean_ci_width"] / row["logit_normal_mean_ci_width"]
            if row["logit_normal_mean_ci_width"] > 0.001 else np.nan
        )
        row["divergence_ratio"] = (
            row["beta_binomial_mean_divergences"] / max(row["logit_normal_mean_divergences"], 0.1)
        )

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Compute overall summaries by region
    regions = {
        "boundary": [0.0, 1.0],
        "near_boundary": [0.05, 0.95],
        "mid_range": [0.1, 0.3, 0.5, 0.7, 0.9]
    }

    region_summaries = {}
    for region_name, p_values in regions.items():
        region_df = summary_df[summary_df["true_p"].isin(p_values)]
        if len(region_df) == 0:
            continue

        region_summaries[region_name] = {
            "n_conditions": len(region_df),
            "logit_normal": {
                "mean_abs_bias": float(region_df["logit_normal_abs_bias"].mean()),
                "mean_ci_width": float(region_df["logit_normal_mean_ci_width"].mean()),
                "mean_coverage": float(region_df["logit_normal_coverage"].mean()),
                "total_divergences": int(region_df["logit_normal_total_divergences"].sum()),
                "mean_ess_bulk": float(region_df["logit_normal_mean_ess_bulk"].mean()),
                "min_ess_bulk": float(region_df["logit_normal_min_ess_bulk"].min()),
            },
            "beta_binomial": {
                "mean_abs_bias": float(region_df["beta_binomial_abs_bias"].mean()),
                "mean_ci_width": float(region_df["beta_binomial_mean_ci_width"].mean()),
                "mean_coverage": float(region_df["beta_binomial_coverage"].mean()),
                "total_divergences": int(region_df["beta_binomial_total_divergences"].sum()),
                "mean_ess_bulk": float(region_df["beta_binomial_mean_ess_bulk"].mean()),
                "min_ess_bulk": float(region_df["beta_binomial_min_ess_bulk"].min()),
            },
            "comparison": {
                "bias_ratio_mean": float(region_df["bias_ratio"].mean()),
                "ci_width_ratio_mean": float(region_df["ci_width_ratio"].mean()),
            }
        }

    # Full summary
    full_summary = {
        "simulation_date": datetime.now().isoformat(),
        "n_total_conditions": len(summary_df),
        "n_replications_per_condition": int(df.groupby(["true_p", "n_items"]).size().iloc[0]),
        "notes": {
            "boundary_dgp": (
                "At exact boundaries (true_p = 0.0 or 1.0), the data generating process "
                "has ZERO between-item heterogeneity (all items have identical success "
                "probability). This is qualitatively different from mid-range cases where "
                "items vary around the population mean. Coverage at boundaries reflects "
                "model behavior when estimating heterogeneity that does not exist."
            ),
            "ci_method": "HDI (Highest Density Interval) matching optstop implementation",
            "dgp_favors": "Beta-Binomial (data generated from Beta-Binomial process)",
            "target_accept": "0.95 (matches optstop package default for both GPU and CPU)"
        },
        "regions": region_summaries,
        "per_condition": summary_df.to_dict(orient="records")
    }

    # Save
    summary_df.to_csv(output_dir / "model_comparison_summary.csv", index=False)
    with open(output_dir / "model_comparison_summary.json", "w") as f:
        json.dump(full_summary, f, indent=2, default=str)

    return full_summary


def generate_latex_table(summary: Dict, output_dir: Path) -> str:
    """Generate LaTeX table for the appendix."""

    # Create condensed table by region
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Logit-Normal vs Beta-Binomial hierarchical model comparison under Beta-Binomial data generating process. Metrics averaged across sample sizes (20, 50, 100 items) and 25 replications per condition. Bias ratio $<1$ favours Beta-Binomial; CI width ratio $<1$ indicates narrower Beta-Binomial intervals.}",
        r"\label{tab:model_comparison}",
        r"\small",
        r"\begin{tabular}{@{}l cc cc cc c@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Abs.\ Bias} & \multicolumn{2}{c}{CI Width} & \multicolumn{2}{c}{Coverage} & Diverg. \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-8}",
        r"Region & L-N & B-B & L-N & B-B & L-N & B-B & Ratio \\",
        r"\midrule",
    ]

    for region_name in ["mid_range", "near_boundary", "boundary"]:
        if region_name not in summary["regions"]:
            continue
        region = summary["regions"][region_name]
        ln = region["logit_normal"]
        bb = region["beta_binomial"]

        # Format region name
        display_name = {
            "mid_range": "Mid-range (0.1--0.9)",
            "near_boundary": "Near-boundary (0.05, 0.95)",
            "boundary": "Boundary (0.0, 1.0)"
        }[region_name]

        # Divergence ratio
        div_ratio = bb["total_divergences"] / max(ln["total_divergences"], 1)

        line = (
            f"{display_name} & "
            f"{ln['mean_abs_bias']:.3f} & {bb['mean_abs_bias']:.3f} & "
            f"{ln['mean_ci_width']:.3f} & {bb['mean_ci_width']:.3f} & "
            f"{ln['mean_coverage']:.2f} & {bb['mean_coverage']:.2f} & "
            f"{div_ratio:.1f}$\\times$ \\\\"
        )
        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    latex = "\n".join(lines)

    with open(output_dir / "model_comparison_table.tex", "w") as f:
        f.write(latex)

    return latex


def main():
    """Run the full simulation and generate outputs."""
    print("=" * 60)
    print("Beta-Binomial vs Logit-Normal Model Comparison Simulation")
    print("=" * 60)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Running simulation grid...")
    print("  Grid: 9 true_p values x 3 n_items values x 25 replications = 675 runs")
    print("  (Each run fits 2 models, so 1350 MCMC fits total)")
    print("  Estimated time: 15-20 hours (Beta-Binomial has slow MCMC convergence).\n")

    df = run_simulation_grid(output_dir, n_replications=25, verbose=True)

    print(f"\n[2/3] Summarizing results...")
    summary = summarize_results(df, output_dir)

    print(f"\n[3/3] Generating LaTeX table...")
    latex = generate_latex_table(summary, output_dir)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for region_name, region in summary["regions"].items():
        print(f"\n{region_name.upper()}:")
        ln = region["logit_normal"]
        bb = region["beta_binomial"]
        print(f"  Logit-Normal:   bias={ln['mean_abs_bias']:.4f}, CI={ln['mean_ci_width']:.3f}, "
              f"cov={ln['mean_coverage']:.2f}, div={ln['total_divergences']}")
        print(f"  Beta-Binomial:  bias={bb['mean_abs_bias']:.4f}, CI={bb['mean_ci_width']:.3f}, "
              f"cov={bb['mean_coverage']:.2f}, div={bb['total_divergences']}")

    print(f"\nOutputs saved to: {output_dir}")
    print("  - model_comparison_raw.csv")
    print("  - model_comparison_summary.csv")
    print("  - model_comparison_summary.json")
    print("  - model_comparison_table.tex")

    return summary


if __name__ == "__main__":
    main()
