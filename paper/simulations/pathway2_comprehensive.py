#!/usr/bin/env python3
"""
Comprehensive Pathway 2 analysis across the full ordinal evaluation landscape.

Supersedes pathway2_deep_analysis.py by:
  1. Testing across K ∈ {3, 5, 7, 10} — the realistic range for LLM evals
  2. Using 12 distribution shapes drawn from real-world benchmarks
  3. Computing EXACT entropy fractions from actual discrete distributions
  4. Modelling item-level consensus from distribution heterogeneity
  5. Evaluating Pathway 2 mechanisms against a decision-theoretic framework

The core question: When Pathway 1's entropy gate blocks stopping (entropy_frac > T),
what additional signal can distinguish "wide but unimodal" (should eventually stop)
from "genuinely ambiguous" (should not stop)?

Usage:
    python simulations/pathway2_comprehensive.py
"""

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
from scipy import stats
from scipy.special import entr

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# §0  DISTRIBUTION PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def shannon_entropy_bits(probs: np.ndarray) -> float:
    """Shannon entropy H = -sum(p * log2(p)) in bits."""
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    return float(np.sum(entr(p)) / np.log(2))


def max_entropy_bits(K: int) -> float:
    return float(np.log2(K))


def discretize_truncnorm(mu: float, sigma: float, K: int) -> np.ndarray:
    """Discrete probability vector from truncated Normal over K categories."""
    if sigma < 1e-8:
        probs = np.zeros(K)
        probs[int(np.clip(np.round(mu), 0, K - 1))] = 1.0
        return probs
    edges = np.arange(K + 1) - 0.5
    cdf_vals = stats.norm.cdf(edges, loc=mu, scale=sigma)
    probs = np.diff(cdf_vals)
    probs = np.clip(probs, 0, None)
    s = probs.sum()
    return probs / s if s > 0 else np.ones(K) / K


def make_bimodal(mu1, mu2, sigma1, sigma2, K, weight1=0.5):
    """Bimodal mixture of two truncated normals."""
    p1 = discretize_truncnorm(mu1, sigma1, K)
    p2 = discretize_truncnorm(mu2, sigma2, K)
    return weight1 * p1 + (1 - weight1) * p2


def make_narrow_band(center_cat: int, width: int, K: int) -> np.ndarray:
    """Mass concentrated in `width` adjacent categories centered at `center_cat`.
    Simulates score clustering / judge calibration collapse.
    """
    probs = np.zeros(K)
    lo = max(0, center_cat - width // 2)
    hi = min(K, lo + width)
    lo = max(0, hi - width)  # adjust if hitting right edge
    # Triangular weighting — peak at center
    for i in range(lo, hi):
        dist = abs(i - center_cat)
        probs[i] = max(1, width - dist)
    return probs / probs.sum()


def make_ceiling_effect(K: int, peak_frac: float = 0.85) -> np.ndarray:
    """Scores piled at the top of the scale — typical for frontier LLMs.
    `peak_frac` controls how much mass is in the top 2 categories.
    """
    probs = np.zeros(K)
    probs[-1] = peak_frac * 0.6   # highest category gets most
    probs[-2] = peak_frac * 0.3   # second highest
    probs[-3] = peak_frac * 0.1 if K >= 3 else 0
    # Distribute remaining mass across lower categories
    remaining = 1.0 - probs.sum()
    n_lower = max(1, K - 3)
    for i in range(K - 3):
        probs[i] = remaining / n_lower
    probs = np.clip(probs, 0, None)
    return probs / probs.sum()


def make_floor_effect(K: int, peak_frac: float = 0.85) -> np.ndarray:
    """Scores piled at the bottom — weak model on hard task."""
    probs = np.zeros(K)
    probs[0] = peak_frac * 0.6
    probs[1] = peak_frac * 0.3
    probs[2] = peak_frac * 0.1 if K >= 3 else 0
    remaining = 1.0 - probs.sum()
    n_upper = max(1, K - 3)
    for i in range(3, K):
        probs[i] = remaining / n_upper
    probs = np.clip(probs, 0, None)
    return probs / probs.sum()


# ══════════════════════════════════════════════════════════════════════════════
# §1  DISTRIBUTION CATALOGUE — THE FULL LANDSCAPE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OrdinalDistribution:
    """A single ordinal distribution with its properties."""
    name: str               # Human-readable label
    category: str           # grouping: "unimodal", "bimodal", "degenerate", "uniform"
    K: int                  # Number of categories
    probs: np.ndarray       # Probability vector (length K)
    entropy_bits: float = 0.0
    entropy_frac: float = 0.0
    p1_pass_T80: bool = False      # Does Pathway 1 pass at T=0.80?
    p2_should_stop: bool = False   # If P1 blocks, should P2 stop?
    modal_proportion: float = 0.0  # Probability of modal category
    n_active_categories: int = 0   # Categories with >1% probability
    expected_item_consensus: float = 0.0  # Expected fraction of items stopped

    def __post_init__(self):
        me = max_entropy_bits(self.K)
        self.entropy_bits = shannon_entropy_bits(self.probs)
        self.entropy_frac = self.entropy_bits / me if me > 0 else 0
        self.p1_pass_T80 = self.entropy_frac <= 0.80
        self.modal_proportion = float(np.max(self.probs))
        self.n_active_categories = int(np.sum(self.probs > 0.01))


def build_catalogue(K: int) -> List[OrdinalDistribution]:
    """Build the complete distribution catalogue for a given K.

    Returns distributions spanning the full range of realistic ordinal
    evaluation scenarios, with correct should_stop labels.
    """
    dists = []
    R = K - 1  # range (0 to K-1)
    center = R * 0.5
    quarter = R * 0.25
    three_q = R * 0.75

    # ── A. UNIMODAL — Tight agreement (σ = 5% of range) ─────────────────
    # These should always stop (low entropy, clear mode)
    for loc_name, mu in [("edge", 0.5), ("center", center), ("high", three_q)]:
        sigma = max(0.05 * R, 0.3)
        probs = discretize_truncnorm(mu, sigma, K)
        dists.append(OrdinalDistribution(
            name=f"unimodal_tight_{loc_name}", category="unimodal", K=K,
            probs=probs, p2_should_stop=True,
            expected_item_consensus=0.6))  # Most items converge

    # ── B. UNIMODAL — Moderate agreement (σ = 15% of range) ─────────────
    # The mid_ordinal case — moderate spread, should stop
    for loc_name, mu in [("edge", 0.5), ("center", center), ("high", three_q)]:
        sigma = max(0.15 * R, 0.5)
        probs = discretize_truncnorm(mu, sigma, K)
        dists.append(OrdinalDistribution(
            name=f"unimodal_moderate_{loc_name}", category="unimodal", K=K,
            probs=probs, p2_should_stop=True,
            expected_item_consensus=0.35))  # Some items converge

    # ── C. UNIMODAL — Wide agreement (σ = 30% of range) ─────────────────
    # High disagreement but still unimodal — borderline case
    # Should stop only if entropy_frac < ~0.90 (some structure remains)
    for loc_name, mu in [("edge", 0.5), ("center", center), ("high", three_q)]:
        sigma = max(0.30 * R, 0.8)
        probs = discretize_truncnorm(mu, sigma, K)
        d = OrdinalDistribution(
            name=f"unimodal_wide_{loc_name}", category="unimodal", K=K,
            probs=probs,
            expected_item_consensus=0.15)  # Few items converge
        # Wide unimodals: stop if entropy < 0.90, don't stop if > 0.90
        d.p2_should_stop = d.entropy_frac < 0.90
        dists.append(d)

    # ── D. UNIMODAL — Very wide (σ = 45% of range) ──────────────────────
    # Near-uniform unimodal — should NOT stop (too dispersed)
    for loc_name, mu in [("center", center)]:
        sigma = max(0.45 * R, 1.2)
        probs = discretize_truncnorm(mu, sigma, K)
        dists.append(OrdinalDistribution(
            name=f"unimodal_verywide_{loc_name}", category="unimodal", K=K,
            probs=probs, p2_should_stop=False,
            expected_item_consensus=0.05))

    # ── E. BIMODAL — Symmetric, separated modes ─────────────────────────
    # Two clear peaks — should NOT stop (genuinely ambiguous)
    for sigma_name, sf in [("tight", 0.08), ("moderate", 0.15)]:
        sigma = max(sf * R, 0.3)
        probs = make_bimodal(quarter, three_q, sigma, sigma, K, 0.5)
        dists.append(OrdinalDistribution(
            name=f"bimodal_sym_{sigma_name}", category="bimodal", K=K,
            probs=probs, p2_should_stop=False,
            expected_item_consensus=0.05))

    # ── F. BIMODAL — Asymmetric (70/30 split) ───────────────────────────
    # One dominant mode — still should NOT stop (ambiguity remains)
    for sigma_name, sf in [("tight", 0.08), ("moderate", 0.15)]:
        sigma = max(sf * R, 0.3)
        probs = make_bimodal(quarter, three_q, sigma, sigma, K, 0.7)
        dists.append(OrdinalDistribution(
            name=f"bimodal_asym_{sigma_name}", category="bimodal", K=K,
            probs=probs, p2_should_stop=False,
            expected_item_consensus=0.10))

    # ── G. BIMODAL — Adjacent modes (hard to distinguish from unimodal) ─
    # Modes close together — this is the tricky case
    sigma = max(0.10 * R, 0.3)
    mu_lo = center - 0.8  # just below center
    mu_hi = center + 0.8  # just above center
    probs = make_bimodal(mu_lo, mu_hi, sigma, sigma, K, 0.5)
    dists.append(OrdinalDistribution(
        name="bimodal_adjacent", category="bimodal", K=K,
        probs=probs, p2_should_stop=False,
        expected_item_consensus=0.12))

    # ── H. NARROW-BAND — Score clustering (K=10 functioning as K=3) ─────
    # Common with uncalibrated judges — mass in 2-3 categories
    for loc_name, center_cat in [("high", min(K-2, int(0.7*R))),
                                  ("mid", int(0.5*R))]:
        width = min(3, K)
        probs = make_narrow_band(center_cat, width, K)
        dists.append(OrdinalDistribution(
            name=f"narrow_band_{loc_name}", category="degenerate", K=K,
            probs=probs, p2_should_stop=True,
            expected_item_consensus=0.50))

    # ── I. CEILING EFFECT — Frontier LLM on standard benchmark ──────────
    # Most mass at top, should stop (clear pattern)
    probs = make_ceiling_effect(K, peak_frac=0.85)
    dists.append(OrdinalDistribution(
        name="ceiling_effect", category="degenerate", K=K,
        probs=probs, p2_should_stop=True,
        expected_item_consensus=0.55))

    # ── J. FLOOR EFFECT — Weak model on hard task ───────────────────────
    probs = make_floor_effect(K, peak_frac=0.85)
    dists.append(OrdinalDistribution(
        name="floor_effect", category="degenerate", K=K,
        probs=probs, p2_should_stop=True,
        expected_item_consensus=0.55))

    # ── K. NEAR-UNIFORM — Genuinely uninformative ───────────────────────
    # Should NOT stop (maximum uncertainty)
    probs = np.ones(K) / K
    dists.append(OrdinalDistribution(
        name="uniform", category="uniform", K=K,
        probs=probs, p2_should_stop=False,
        expected_item_consensus=0.02))

    # Near-uniform with slight mode (Dirichlet(0.5) draw — typical random)
    rng = np.random.default_rng(42 + K)
    probs = rng.dirichlet(np.full(K, 0.5))
    dists.append(OrdinalDistribution(
        name="near_uniform_noisy", category="uniform", K=K,
        probs=probs, p2_should_stop=False,
        expected_item_consensus=0.03))

    # ── L. SKEWED HIGH — Realistic frontier model performance ───────────
    # Typical: scores concentrate in top 30% of scale
    mu = R * 0.80  # 80th percentile
    sigma = max(0.12 * R, 0.4)
    probs = discretize_truncnorm(mu, sigma, K)
    dists.append(OrdinalDistribution(
        name="skewed_high", category="unimodal", K=K,
        probs=probs, p2_should_stop=True,
        expected_item_consensus=0.45))

    # ── M. SKEWED LOW — Weak model on hard task ────────────────────────
    mu = R * 0.20  # 20th percentile
    sigma = max(0.12 * R, 0.4)
    probs = discretize_truncnorm(mu, sigma, K)
    dists.append(OrdinalDistribution(
        name="skewed_low", category="unimodal", K=K,
        probs=probs, p2_should_stop=True,
        expected_item_consensus=0.45))

    return dists


# ══════════════════════════════════════════════════════════════════════════════
# §2  MONTE CARLO SCENARIO GENERATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MCScenario:
    """A single Monte Carlo instance for Pathway 2 testing."""
    dist_name: str
    K: int
    entropy_frac: float
    category: str
    should_stop: bool
    # Observable signals at group-check time
    entropy_ci_widths: List[float] = field(default_factory=list)
    modal_ci_widths: List[float] = field(default_factory=list)
    modal_ci_lower: float = 0.0
    item_stop_frac: float = 0.0
    n_checks: int = 5


def ci_width_trajectory(n_checks: int, converged: bool, rng,
                         init_range=(0.15, 0.40), final_range=(0.01, 0.05),
                         noise_frac=0.12) -> List[float]:
    """Generate a realistic CI width trajectory.

    For converged: power-law decay from init to final.
    For non-converged: noisy fluctuation around init level.
    """
    if converged:
        init_w = rng.uniform(*init_range)
        final_w = rng.uniform(*final_range)
        widths = []
        for i in range(n_checks):
            t = (i + 1) / n_checks
            # Bayesian convergence: width ~ 1/sqrt(n)
            true_w = final_w + (init_w - final_w) * (1 - t) ** 1.5
            noise = rng.normal(0, noise_frac * true_w)
            widths.append(max(0.001, true_w + noise))
        return widths
    else:
        mean_w = rng.uniform(*init_range)
        return [max(0.01, mean_w * (1 + rng.normal(0, 0.25)))
                for _ in range(n_checks)]


def generate_mc_scenarios(dists: List[OrdinalDistribution], n_mc: int,
                           n_checks: int, rng) -> List[MCScenario]:
    """Generate Monte Carlo scenarios from the distribution catalogue.

    Only generates scenarios for distributions where Pathway 1 is BLOCKED
    (entropy_frac > 0.80), since Pathway 2 is irrelevant otherwise.
    Also includes some where P1 passes, to test that P2 doesn't interfere.
    """
    scenarios = []

    for dist in dists:
        for _ in range(n_mc):
            # Add entropy noise (real MCMC produces slightly different estimates)
            ef_noise = rng.normal(0, 0.015)
            ef = np.clip(dist.entropy_frac + ef_noise, 0.01, 0.99)

            # Convergence: most legitimate distributions converge in 5 checks
            converged = dist.category != "uniform" or rng.random() < 0.3

            # CI width trajectories
            eci = ci_width_trajectory(n_checks, converged, rng,
                                       init_range=(0.15, 0.40),
                                       final_range=(0.01, 0.05))
            mci = ci_width_trajectory(n_checks, converged, rng,
                                       init_range=(0.03, 0.10),
                                       final_range=(0.001, 0.010))

            # Modal CI lower bound — derived from modal proportion + uncertainty
            modal_lower = max(0.01, dist.modal_proportion - rng.uniform(0.02, 0.08))

            # Item consensus — add noise around expected value
            ic_noise = rng.normal(0, 0.08)
            item_frac = np.clip(dist.expected_item_consensus + ic_noise, 0.0, 0.95)

            scenarios.append(MCScenario(
                dist_name=dist.name, K=dist.K,
                entropy_frac=ef, category=dist.category,
                should_stop=dist.p2_should_stop,
                entropy_ci_widths=eci, modal_ci_widths=mci,
                modal_ci_lower=modal_lower,
                item_stop_frac=item_frac,
                n_checks=n_checks))

    return scenarios


# ══════════════════════════════════════════════════════════════════════════════
# §3  PATHWAY 2 MECHANISMS
# ══════════════════════════════════════════════════════════════════════════════

def m_current_relchg(sc: MCScenario, threshold=0.002, min_hist=3) -> bool:
    """Current: two-point relative change on entropy CI width."""
    w = sc.entropy_ci_widths
    if len(w) < min_hist:
        return False
    for i in range(min_hist - 1, len(w)):
        if w[i-1] > 0 and abs(w[i] - w[i-1]) / w[i-1] < threshold:
            return True
    return False


def m_relaxed_relchg(sc: MCScenario, threshold=0.05) -> bool:
    """Relaxed relative change."""
    w = sc.entropy_ci_widths
    for i in range(2, len(w)):
        if w[i-1] > 0 and abs(w[i] - w[i-1]) / w[i-1] < threshold:
            return True
    return False


def m_abs_ci(sc: MCScenario, ci_thresh=0.10) -> bool:
    """Absolute CI width threshold (no guard)."""
    return any(w < ci_thresh for w in sc.entropy_ci_widths)


def m_abs_ci_guarded(sc: MCScenario, ci_thresh=0.10, guard=0.85) -> bool:
    """Absolute CI width + entropy guard."""
    if sc.entropy_frac > guard:
        return False
    return any(w < ci_thresh for w in sc.entropy_ci_widths)


def m_abs_ci_tight_guarded(sc: MCScenario, ci_thresh=0.05, guard=0.90) -> bool:
    """Tight CI + high guard."""
    if sc.entropy_frac > guard:
        return False
    return any(w < ci_thresh for w in sc.entropy_ci_widths)


def m_modal_dominance(sc: MCScenario, threshold=0.25) -> bool:
    """Modal category CI lower bound exceeds threshold."""
    return sc.modal_ci_lower > threshold


def m_modal_guarded(sc: MCScenario, modal_thresh=0.25, guard=0.90) -> bool:
    """Modal dominance + entropy guard."""
    if sc.entropy_frac > guard:
        return False
    return sc.modal_ci_lower > modal_thresh


def m_item_consensus(sc: MCScenario, threshold=0.25) -> bool:
    """Item-level consensus: fraction of items individually stopped."""
    return sc.item_stop_frac > threshold


def m_item_consensus_20(sc: MCScenario) -> bool:
    """Item consensus at 20% threshold."""
    return sc.item_stop_frac > 0.20


def m_item_consensus_30(sc: MCScenario) -> bool:
    """Item consensus at 30% threshold."""
    return sc.item_stop_frac > 0.30


def m_item_consensus_guarded(sc: MCScenario, threshold=0.25, guard=0.90) -> bool:
    """Item consensus + entropy guard."""
    if sc.entropy_frac > guard:
        return False
    return sc.item_stop_frac > threshold


def m_ci_or_consensus(sc: MCScenario, ci_thresh=0.05, item_thresh=0.25,
                       guard=0.90) -> bool:
    """Combined: (CI converged OR item consensus) AND entropy guard."""
    if sc.entropy_frac > guard:
        return False
    ci_ok = any(w < ci_thresh for w in sc.entropy_ci_widths)
    item_ok = sc.item_stop_frac > item_thresh
    return ci_ok or item_ok


def m_modal_and_consensus(sc: MCScenario, modal_thresh=0.25,
                           item_thresh=0.20) -> bool:
    """Combined: modal dominance AND item consensus (no entropy guard needed).
    Requires BOTH signals to agree — conservative but guard-free.
    """
    return sc.modal_ci_lower > modal_thresh and sc.item_stop_frac > item_thresh


MECHANISMS = {
    "current_relchg_0.002":   m_current_relchg,
    "relaxed_relchg_0.05":    m_relaxed_relchg,
    "abs_ci_0.10":            m_abs_ci,
    "abs_ci_0.10_guard_0.85": lambda sc: m_abs_ci_guarded(sc, 0.10, 0.85),
    "abs_ci_0.05_guard_0.90": m_abs_ci_tight_guarded,
    "modal_dom_0.25":         m_modal_dominance,
    "modal_dom_guard_0.90":   m_modal_guarded,
    "item_consensus_0.20":    m_item_consensus_20,
    "item_consensus_0.25":    m_item_consensus,
    "item_consensus_0.30":    m_item_consensus_30,
    "item_cons_guard_0.90":   m_item_consensus_guarded,
    "ci_or_consensus":        m_ci_or_consensus,
    "modal_and_consensus":    m_modal_and_consensus,
}


# ══════════════════════════════════════════════════════════════════════════════
# §4  EVALUATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MechanismResult:
    """Results for one mechanism on one distribution group."""
    n: int = 0
    n_triggered: int = 0
    n_correct: int = 0
    n_should_stop: int = 0
    n_should_not_stop: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def trigger_rate(self):
        return self.n_triggered / self.n if self.n else 0

    @property
    def accuracy(self):
        return self.n_correct / self.n if self.n else 0

    @property
    def tpr(self):
        return self.true_positives / self.n_should_stop if self.n_should_stop else 0

    @property
    def fpr(self):
        return self.false_positives / self.n_should_not_stop if self.n_should_not_stop else 0

    @property
    def tnr(self):
        return self.true_negatives / self.n_should_not_stop if self.n_should_not_stop else 0

    @property
    def fnr(self):
        return self.false_negatives / self.n_should_stop if self.n_should_stop else 0


def evaluate(scenarios: List[MCScenario], mechanisms: Dict) -> Dict:
    """Evaluate all mechanisms on all scenarios.

    Returns nested dict: results[mechanism_name][group_key] = MechanismResult
    Groups: by category, by K, by distribution name, and "overall".
    """
    results = {}
    for mname in mechanisms:
        results[mname] = {}

    for mname, mfn in mechanisms.items():
        # Group by category
        by_cat = {}
        # Group by K
        by_k = {}
        # Group by (K, distribution name)
        by_dist = {}
        # Overall
        overall = MechanismResult()

        for sc in scenarios:
            triggered = mfn(sc)
            correct = (triggered == sc.should_stop)

            # Update counters
            for group_key, group_dict in [
                (sc.category, by_cat),
                (f"K={sc.K}", by_k),
                (f"K={sc.K}_{sc.dist_name}", by_dist),
            ]:
                if group_key not in group_dict:
                    group_dict[group_key] = MechanismResult()
                r = group_dict[group_key]
                r.n += 1
                r.n_triggered += int(triggered)
                r.n_correct += int(correct)
                if sc.should_stop:
                    r.n_should_stop += 1
                    if triggered:
                        r.true_positives += 1
                    else:
                        r.false_negatives += 1
                else:
                    r.n_should_not_stop += 1
                    if triggered:
                        r.false_positives += 1
                    else:
                        r.true_negatives += 1

            # Overall
            overall.n += 1
            overall.n_triggered += int(triggered)
            overall.n_correct += int(correct)
            if sc.should_stop:
                overall.n_should_stop += 1
                if triggered:
                    overall.true_positives += 1
                else:
                    overall.false_negatives += 1
            else:
                overall.n_should_not_stop += 1
                if triggered:
                    overall.false_positives += 1
                else:
                    overall.true_negatives += 1

        results[mname] = {
            "by_category": by_cat,
            "by_K": by_k,
            "by_distribution": by_dist,
            "overall": overall,
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# §5  REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_landscape(all_dists: Dict[int, List[OrdinalDistribution]]):
    """Print the full distribution landscape with entropy fractions."""
    print("\n" + "=" * 120)
    print("  PART 1: DISTRIBUTION LANDSCAPE — Entropy Fractions Across K Values")
    print("=" * 120)

    for K in sorted(all_dists.keys()):
        dists = all_dists[K]
        print(f"\n### K = {K}  (max entropy = {max_entropy_bits(K):.3f} bits)")
        print(f"  {'Distribution':<30s}  {'Category':<12s}  {'E(frac)':>8s}  {'E(bits)':>8s}"
              f"  {'Modal%':>7s}  {'#Active':>7s}  {'P1@0.80':>7s}  {'P2 should':>9s}")
        print(f"  {'-'*30}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}")

        # Sort by entropy fraction
        for d in sorted(dists, key=lambda x: x.entropy_frac):
            p1 = "PASS" if d.p1_pass_T80 else "BLOCK"
            p2 = "stop" if d.p2_should_stop else "NO"
            if d.p1_pass_T80:
                p2 = "(P1 ok)"  # P2 irrelevant when P1 passes
            print(f"  {d.name:<30s}  {d.category:<12s}  {d.entropy_frac:>8.3f}"
                  f"  {d.entropy_bits:>8.3f}  {d.modal_proportion:>6.1%}  {d.n_active_categories:>7d}"
                  f"  {p1:>7s}  {p2:>9s}")

        # Summary counts
        n_p1_pass = sum(1 for d in dists if d.p1_pass_T80)
        n_p1_block = sum(1 for d in dists if not d.p1_pass_T80)
        n_p2_stop = sum(1 for d in dists if not d.p1_pass_T80 and d.p2_should_stop)
        n_p2_no = sum(1 for d in dists if not d.p1_pass_T80 and not d.p2_should_stop)
        print(f"\n  Summary: {n_p1_pass} pass P1 at T=0.80, {n_p1_block} blocked"
              f" → P2 must handle: {n_p2_stop} should-stop + {n_p2_no} should-not-stop")


def print_overall_results(results: Dict, mechanisms: Dict):
    """Print the overall mechanism comparison."""
    print("\n" + "=" * 120)
    print("  PART 2: OVERALL MECHANISM COMPARISON")
    print("=" * 120)

    print(f"\n  {'Mechanism':<28s}  {'TPR':>6s}  {'FPR':>6s}  {'TNR':>6s}  {'FNR':>6s}"
          f"  {'Accuracy':>8s}  {'F1':>6s}  | Assessment")
    print(f"  {'-'*28}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}  | ----------")

    for mname in mechanisms:
        o = results[mname]["overall"]
        tpr = o.tpr
        fpr = o.fpr
        precision = o.true_positives / (o.true_positives + o.false_positives) if (o.true_positives + o.false_positives) > 0 else 0
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Assessment
        if fpr > 0.15:
            assessment = "UNSAFE (high FPR)"
        elif tpr < 0.30:
            assessment = "WEAK (low TPR)"
        elif fpr > 0.05:
            assessment = "CAUTION (moderate FPR)"
        elif tpr > 0.60:
            assessment = "GOOD"
        else:
            assessment = "MODERATE"

        print(f"  {mname:<28s}  {tpr:>5.1%}  {fpr:>5.1%}  {o.tnr:>5.1%}  {o.fnr:>5.1%}"
              f"  {o.accuracy:>7.1%}  {f1:>5.3f}  | {assessment}")


def print_by_category(results: Dict, mechanisms: Dict):
    """Print mechanism results broken down by distribution category."""
    print("\n" + "=" * 120)
    print("  PART 3: RESULTS BY DISTRIBUTION CATEGORY")
    print("=" * 120)

    categories = ["unimodal", "bimodal", "degenerate", "uniform"]
    cat_labels = {
        "unimodal": "UNIMODAL (mostly should-stop)",
        "bimodal": "BIMODAL (should NOT stop)",
        "degenerate": "DEGENERATE / narrow-band (should-stop)",
        "uniform": "UNIFORM (should NOT stop)",
    }

    for cat in categories:
        print(f"\n### {cat_labels.get(cat, cat)}")
        print(f"  {'Mechanism':<28s}  {'TPR':>6s}  {'FPR':>6s}  {'Trigger':>7s}  | Notes")
        print(f"  {'-'*28}  {'-'*6}  {'-'*6}  {'-'*7}  | -----")

        for mname in mechanisms:
            r = results[mname]["by_category"].get(cat)
            if r is None:
                continue
            tpr_str = f"{r.tpr:>5.1%}" if r.n_should_stop > 0 else "  N/A"
            fpr_str = f"{r.fpr:>5.1%}" if r.n_should_not_stop > 0 else "  N/A"
            notes = ""
            if cat in ("bimodal", "uniform") and r.fpr > 0.10:
                notes = "DANGER: false stops on ambiguous distributions"
            elif cat in ("unimodal", "degenerate") and r.tpr < 0.30:
                notes = "WEAK: misses most legitimate stop cases"
            print(f"  {mname:<28s}  {tpr_str}  {fpr_str}  {r.trigger_rate:>6.1%}  | {notes}")


def print_by_K(results: Dict, mechanisms: Dict, K_values: List[int]):
    """Print mechanism results broken down by K."""
    print("\n" + "=" * 120)
    print("  PART 4: RESULTS BY K VALUE")
    print("=" * 120)

    for K in K_values:
        k_key = f"K={K}"
        print(f"\n### K = {K}")
        print(f"  {'Mechanism':<28s}  {'TPR':>6s}  {'FPR':>6s}  {'Accuracy':>8s}")
        print(f"  {'-'*28}  {'-'*6}  {'-'*6}  {'-'*8}")

        for mname in mechanisms:
            r = results[mname]["by_K"].get(k_key)
            if r is None:
                continue
            tpr_str = f"{r.tpr:>5.1%}" if r.n_should_stop > 0 else "  N/A"
            fpr_str = f"{r.fpr:>5.1%}" if r.n_should_not_stop > 0 else "  N/A"
            print(f"  {mname:<28s}  {tpr_str}  {fpr_str}  {r.accuracy:>7.1%}")


def print_critical_cases(results: Dict, mechanisms: Dict, all_dists: Dict):
    """Focus on distributions where P1 blocks and P2's decision matters most."""
    print("\n" + "=" * 120)
    print("  PART 5: CRITICAL CASES — Where Pathway 2 Decision Matters")
    print("=" * 120)

    print("\nThese are distributions where Pathway 1 BLOCKS at T=0.80 (entropy_frac > 0.80).")
    print("Pathway 2 must correctly decide: stop (legitimate convergence) or continue (genuine ambiguity).\n")

    # Collect all P1-blocked distributions
    critical = []
    for K, dists in all_dists.items():
        for d in dists:
            if not d.p1_pass_T80:
                critical.append(d)

    # Group by should_stop
    should_stop = [d for d in critical if d.p2_should_stop]
    should_not = [d for d in critical if not d.p2_should_stop]

    print(f"  P1-blocked distributions: {len(critical)} total")
    print(f"    Should stop:     {len(should_stop)} (legitimate convergence despite high entropy)")
    print(f"    Should NOT stop: {len(should_not)} (genuinely ambiguous)")

    print(f"\n  --- SHOULD STOP (P2 should activate) ---")
    for d in sorted(should_stop, key=lambda x: (x.K, x.entropy_frac)):
        print(f"    K={d.K:2d}  E={d.entropy_frac:.3f}  {d.name:<30s}  modal={d.modal_proportion:.1%}"
              f"  active={d.n_active_categories}  cat={d.category}")

    print(f"\n  --- SHOULD NOT STOP (P2 should NOT activate) ---")
    for d in sorted(should_not, key=lambda x: (x.K, x.entropy_frac)):
        print(f"    K={d.K:2d}  E={d.entropy_frac:.3f}  {d.name:<30s}  modal={d.modal_proportion:.1%}"
              f"  active={d.n_active_categories}  cat={d.category}")

    # Per-mechanism accuracy on critical cases only
    print(f"\n  --- MECHANISM ACCURACY ON CRITICAL CASES ONLY ---")
    print(f"  {'Mechanism':<28s}  {'TPR':>6s}  {'FPR':>6s}  {'Accuracy':>8s}  {'TP':>4s}  {'FP':>4s}  {'TN':>4s}  {'FN':>4s}")
    print(f"  {'-'*28}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*4}")

    for mname in mechanisms:
        by_dist = results[mname]["by_distribution"]
        tp, fp, tn, fn = 0, 0, 0, 0
        for d in critical:
            key = f"K={d.K}_{d.name}"
            r = by_dist.get(key)
            if r:
                tp += r.true_positives
                fp += r.false_positives
                tn += r.true_negatives
                fn += r.false_negatives
        total = tp + fp + tn + fn
        n_pos = tp + fn
        n_neg = fp + tn
        tpr = tp / n_pos if n_pos else 0
        fpr = fp / n_neg if n_neg else 0
        acc = (tp + tn) / total if total else 0
        print(f"  {mname:<28s}  {tpr:>5.1%}  {fpr:>5.1%}  {acc:>7.1%}  {tp:>4d}  {fp:>4d}  {tn:>4d}  {fn:>4d}")


def print_sensitivity(all_dists: Dict, K_values: List[int], rng):
    """Sensitivity analysis: item consensus threshold vs TPR/FPR tradeoff."""
    print("\n" + "=" * 120)
    print("  PART 6: SENSITIVITY ANALYSIS — Item Consensus Threshold")
    print("=" * 120)

    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    n_mc = 500

    print("\n  Testing item_consensus threshold across all K values.")
    print(f"  n_mc={n_mc} per distribution, n_checks=5\n")

    for K in K_values:
        dists = all_dists[K]
        # Only P1-blocked distributions
        blocked = [d for d in dists if not d.p1_pass_T80]
        if not blocked:
            print(f"  K={K}: No P1-blocked distributions (all pass at T=0.80)")
            continue

        scenarios = generate_mc_scenarios(blocked, n_mc, 5, rng)

        print(f"  K={K} ({len(blocked)} blocked dists, {len(scenarios)} scenarios)")
        print(f"    {'Threshold':>9s}  {'TPR':>6s}  {'FPR':>6s}  {'Accuracy':>8s}  {'F1':>6s}")
        print(f"    {'-'*9}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*6}")

        for t in thresholds:
            tp, fp, tn, fn = 0, 0, 0, 0
            for sc in scenarios:
                triggered = sc.item_stop_frac > t
                if sc.should_stop:
                    if triggered: tp += 1
                    else: fn += 1
                else:
                    if triggered: fp += 1
                    else: tn += 1
            n_pos = tp + fn
            n_neg = fp + tn
            tpr = tp / n_pos if n_pos else 0
            fpr = fp / n_neg if n_neg else 0
            acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0
            prec = tp / (tp + fp) if (tp + fp) else 0
            f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) else 0
            marker = " <-- best F1" if f1 >= 0.5 and fpr < 0.05 else ""
            print(f"    {t:>9.2f}  {tpr:>5.1%}  {fpr:>5.1%}  {acc:>7.1%}  {f1:>5.3f}{marker}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# §6  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 120)
    print("  COMPREHENSIVE PATHWAY 2 ANALYSIS")
    print("  Testing across the full ordinal evaluation landscape")
    print("  K ∈ {3, 5, 7, 10} × 25+ distribution shapes × 13 mechanisms")
    print("=" * 120)

    rng = np.random.default_rng(42)
    K_values = [3, 5, 7, 10]
    n_mc = 500   # MC instances per distribution
    n_checks = 5  # realistic group check count

    # ── Build distribution catalogue ──
    all_dists = {}
    for K in K_values:
        all_dists[K] = build_catalogue(K)

    # ── Print landscape ──
    print_landscape(all_dists)

    # ── Generate MC scenarios (ALL distributions, not just P1-blocked) ──
    all_scenarios = []
    for K in K_values:
        scenarios = generate_mc_scenarios(all_dists[K], n_mc, n_checks, rng)
        all_scenarios.extend(scenarios)

    print(f"\n\nTotal MC scenarios: {len(all_scenarios)}")
    print(f"  Should stop: {sum(1 for s in all_scenarios if s.should_stop)}")
    print(f"  Should not stop: {sum(1 for s in all_scenarios if not s.should_stop)}")

    # ── Evaluate all mechanisms ──
    results = evaluate(all_scenarios, MECHANISMS)

    # ── Print results ──
    print_overall_results(results, MECHANISMS)
    print_by_category(results, MECHANISMS)
    print_by_K(results, MECHANISMS, K_values)
    print_critical_cases(results, MECHANISMS, all_dists)

    # ── Sensitivity analysis ──
    rng2 = np.random.default_rng(123)
    print_sensitivity(all_dists, K_values, rng2)

    # ── Final recommendations ──
    print("\n" + "=" * 120)
    print("  CONCLUSIONS")
    print("=" * 120)
    print("""
This comprehensive analysis tests Pathway 2 mechanisms across {n_dists} distribution
shapes at K = {{3, 5, 7, 10}}, covering the realistic range of LLM ordinal evaluations.

MECHANISM EVALUATION FRAMEWORK:
  - True Positive (TP): Correctly stops a legitimate distribution (unimodal/degenerate)
  - False Positive (FP): Incorrectly stops an ambiguous distribution (bimodal/uniform)
  - True Negative (TN): Correctly continues an ambiguous distribution
  - False Negative (FN): Incorrectly continues a legitimate distribution

WHAT MAKES A GOOD PATHWAY 2:
  1. LOW FPR (< 5%): Must not stop bimodal/uniform distributions
  2. REASONABLE TPR (> 40%): Should catch some legitimate distributions that P1 misses
  3. CONSISTENT ACROSS K: Performance should not degrade at specific K values
  4. IMPLEMENTABLE: Should use signals already available in the stopping engine

See the analysis document for full interpretation and recommendation.
""".format(n_dists=sum(len(d) for d in all_dists.values())))

    # ── Save results ──
    output = {
        "parameters": {"K_values": K_values, "n_mc": n_mc, "n_checks": n_checks, "seed": 42},
        "landscape": {},
        "overall_results": {},
    }
    for K, dists in all_dists.items():
        output["landscape"][str(K)] = [
            {"name": d.name, "category": d.category, "entropy_frac": round(d.entropy_frac, 4),
             "p1_pass": d.p1_pass_T80, "p2_should_stop": d.p2_should_stop,
             "modal_proportion": round(d.modal_proportion, 4),
             "n_active": d.n_active_categories}
            for d in dists
        ]
    for mname in MECHANISMS:
        o = results[mname]["overall"]
        output["overall_results"][mname] = {
            "tpr": round(o.tpr, 4), "fpr": round(o.fpr, 4),
            "accuracy": round(o.accuracy, 4),
            "tp": o.true_positives, "fp": o.false_positives,
            "tn": o.true_negatives, "fn": o.false_negatives,
        }

    with open(OUTPUT_DIR / "pathway2_comprehensive_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR / 'pathway2_comprehensive_results.json'}")


if __name__ == "__main__":
    main()
