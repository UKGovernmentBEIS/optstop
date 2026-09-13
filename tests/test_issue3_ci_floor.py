"""Gate B empirical tests for the issue #3 credible-interval floor fix (E(i)).

Issue #3 (https://github.com/UKGovernmentBEIS/optstop/issues/3): group-level
credible intervals were floored at ``sigmoid(-LOGIT_CLAMP) = RESOLUTION_FLOOR``
(=0.0024726...) because the reported CI was read off the *same* clamped
deterministic (``Theta`` / ``mu_item``) that feeds the likelihood. The fix
(E(i)) adds an UNCLIPPED reporting transform beside each clamped one
(``Theta_report`` / ``mu_item_report``) and switches every CI extraction to it,
while keeping the clamp inside the likelihood for sampler stability.

These tests establish two empirical properties on the running sampler:

1. **No-op mid-range** - where the posterior stays within +/-LOGIT_CLAMP logits
   (the regime the paper's mid-range claims rest on), the reported transform is
   pointwise identical to the clamped one, so nothing observable changes.

2. **Floor removed at low rates** - a near-zero grouping now returns a reported
   ``theta_ci_low`` that dips *below* the old hard floor ``RESOLUTION_FLOOR``
   (impossible pre-fix) and reaches toward 0, so an absent/near-absent
   capability is representable.

Honest scope (per the statistical review): the likelihood still consumes the
clamped variable, so *below* -LOGIT_CLAMP logits the likelihood is flat and the
posterior there is prior-dominated, not data-identified. These tests therefore
assert the floor is *removed* and the interval *reaches toward / contains* a low
truth - NOT frequentist coverage below the floor, which the fix does not and
cannot restore. The continuous pathway additionally has a second, independent
variance clamp (clip(mu_item, 0.01, 0.99)) that E(i) does not touch, so
continuous sub-1% intervals remain doubly heuristic.

Run per file (repo convention) with the pinned stack. Because this file runs
several heavy MCMC fits in one process, the near-floor continuous cell (item-
greedy, margin ~0.002) can be tipped over by the documented PyMC MODEL_MANAGER
context cascade when 8 fits share a process. Run the SLOW tests with per-test
process isolation (pytest-forked) so each fit gets a clean context:
    /home/ubuntu/optstop/venv/bin/python -m pytest tests/test_issue3_ci_floor.py -v --forked
The fast tests (no MCMC) pass either way.
"""

import numpy as np
import pandas as pd
import pytest

import logging

from optstop import optimal_stopping_posthoc, convergence_posthoc
from optstop.early_stopping import OptimalStoppingManager
from optstop.rule import LOGIT_CLAMP, RESOLUTION_FLOOR, optimal_stopping_live_single
import optstop.rule as _rule_mod


# Small-but-adequate MCMC budget: enough draws for a stable HDI, cheap enough
# to run per file. Seeded for reproducibility.
_MCMC = {
    "draws": 500,
    "tune": 1000,
    "chains": 2,
    "cores": 2,
    "random_seed": 20260901,
}


# ---------------------------------------------------------------------------
# 1. No-op guarantee: the unclipped reporting transform equals the clamped one
#    wherever |logit| < LOGIT_CLAMP. This is the mathematical identity E(i)
#    rests on, checked directly on the shipped builder's construct.
# ---------------------------------------------------------------------------


# CI partition: heavy MCMC tests deselected from PR CI (see pyproject.toml markers).
pytestmark = pytest.mark.optstop


def test_midrange_report_transform_equals_clamped_within_clamp():
    """The reported transform equals the clamped one wherever |logit| <
    LOGIT_CLAMP, and diverges (clamped pinned at the floor/ceiling, report free)
    only outside the band.

    Evaluates the exact builder ops - ``sigmoid(clip(mu, -LOGIT_CLAMP,
    LOGIT_CLAMP))`` (the clamped ``Theta``) vs ``sigmoid(mu)`` (the reported
    ``Theta_report``), where ``pm.math`` IS ``pytensor.tensor`` - on a fixed
    grid that straddles both clamp edges. Fully deterministic (no MCMC), so it
    is a flake-free regression guard.

    SCOPE: this proves the mathematical *identity* (clamp is a no-op inside the
    band; floor exists on the clamped transform and is absent on the report
    transform). It rebuilds the ops in a fresh pytensor graph and does NOT touch
    rule.py's model builders or any extraction path - so it is an identity guard,
    NOT proof that the fix is wired into the workers. Worker integration is
    established by the low-rate posthoc tests below plus the grep audit of all
    11 extraction sites.
    """
    import pytensor
    import pytensor.tensor as pt

    mu = pt.dvector("mu")
    theta = pt.sigmoid(pt.clip(mu, -LOGIT_CLAMP, LOGIT_CLAMP))          # clamped: Theta
    theta_report = pt.sigmoid(mu)                                       # unclipped: Theta_report
    f = pytensor.function([mu], [theta, theta_report])

    grid = np.array(
        [-12.0, -10.0, -7.0, -6.001, -6.0, -5.999, -3.0, -1.0, 0.0,
         1.0, 3.0, 5.999, 6.0, 6.001, 7.0, 10.0, 12.0]
    )
    t, tr = f(grid)

    within = np.abs(grid) < LOGIT_CLAMP
    below = grid < -LOGIT_CLAMP
    above = grid > LOGIT_CLAMP

    # No-op claim: bit-for-bit identical inside the band (clip is the identity).
    assert np.allclose(t[within], tr[within], atol=1e-12, rtol=0.0)

    # Floor genuinely exists on the clamped transform ...
    assert np.allclose(t[below], RESOLUTION_FLOOR, atol=1e-12)
    assert np.allclose(t[above], 1.0 - RESOLUTION_FLOOR, atol=1e-12)
    # ... and is genuinely removed on the reporting transform.
    assert (tr[below] < RESOLUTION_FLOOR).all()
    assert (tr[above] > 1.0 - RESOLUTION_FLOOR).all()


# ---------------------------------------------------------------------------
# 2. Floor removed at low rates - GROUP hierarchical model, posthoc paths.
#    These drive the real workers (item-greedy + epoch-interleaved) via the
#    public API and assert on the returned group theta_ci_low.
# ---------------------------------------------------------------------------

def _near_zero_binary_df(n_items=30, epochs=5, n_success=0, grouping="g"):
    """A near-zero binary grouping: n_items items x epochs reps, all score 0
    except `n_success` scattered 1s. True rate ~= n_success/(n_items*epochs)."""
    rows = []
    total = n_items * epochs
    success_slots = set(np.linspace(0, total - 1, n_success, dtype=int)) if n_success else set()
    k = 0
    for item in range(1, n_items + 1):
        for ep in range(1, epochs + 1):
            rows.append(
                {
                    "grouping_num": grouping,
                    "task_num": 1,
                    "sample_id_num": item,
                    "epoch": ep,
                    "score": 1 if k in success_slots else 0,
                }
            )
            k += 1
    return pd.DataFrame(rows)


def _low_rate_params(**extra):
    # delta_cap small so the run does not trivially stop before the group model
    # reports; low_performance_threshold canonical; conservatism canonical.
    p = {
        "delta_item": 0.05,
        "delta_cap": 0.05,
        "CI_delta": 0.0001,
        "conservatism": 5,
        "low_performance_threshold": 0.01,
        "cred_level": 0.97,
        "stab_window": 3,
        "rep_batch_size": 1,
        "pymc_refresh_every": 5,
        **_MCMC,
    }
    p.update(extra)
    return p


def _run_posthoc(df, params, max_hi, **kw):
    """Run the public posthoc API and return the (single) grouping summary.

    ``max_hi`` guards against a FALSE PASS: on MCMC failure the workers fall
    back to ``theta_ci_low=0.0, theta_ci_high=1.0`` (rule.py 2011/2234/2899/
    2984/3082/3702/3789), which would satisfy a naive ``lo < RESOLUTION_FLOOR``
    assertion for the wrong reason. A genuine fit on near-zero data yields a
    *tight* upper bound; requiring ``theta_ci_high < max_hi`` (with max_hi well
    below the fallback's 1.0) proves the group model actually ran and saw the
    near-zero data.
    """
    _, summary = optimal_stopping_posthoc(
        df,
        params,
        grouping_columns=["grouping_num", "task_num"],
        sample_id_column="sample_id_num",
        epoch_column="epoch",
        **kw,
    )
    assert isinstance(summary, list) and len(summary) >= 1
    s = summary[0]
    assert s.get("error") is None, f"worker errored: {s.get('error')}"
    assert s["theta_ci_low"] is not None, "group model did not report a CI"
    hi = float(s["theta_ci_high"])
    assert hi < max_hi, (
        f"theta_ci_high={hi!r} >= max_hi={max_hi!r}: looks like the MCMC-failure "
        f"[0.0, 1.0] fallback, not a genuine near-zero fit - assertion would be a false pass"
    )
    return s


# Fixture calibration is data-driven, not assumed (see /tmp/probe_low_rate_stress.py,
# run at the committed budget draws=500). Findings that set the fixtures below:
#   - 30x20 all-zero crosses the floor robustly (report_lo ~= 0.0006, margin to
#     floor ~= 0.0019). Weaker fixtures (30x5 margin ~0.0012; 10x3 / 30 trials
#     does NOT cross at all - prior-dominated) are too marginal to be reliable
#     discriminators, so the all-zero floor tests use 30x20.
#   - Containment for a SUB-floor true rate fails from below by construction: a
#     97% interval need not contain an arbitrarily tiny truth from below (1/2000
#     true=0.0005 -> report_lo=0.00059 > truth). Whether two-sided containment is
#     achievable is governed by DATA STRENGTH, not by the truth's position
#     relative to the floor: the full-containment test uses 1/1000 (true=0.001,
#     itself BELOW RESOLUTION_FLOOR=0.00247) whose one-success-in-1000 signal is
#     nonetheless strong enough that the lower HDI (report_lo=0.00087) dips below
#     the truth. The weaker 1/2000 case is tested only on the safety-relevant
#     UPPER direction.
# The absolute assertion (report_lo < RESOLUTION_FLOOR) is equivalent to the
# mechanism assertion (report_lo < clamped_lo) only where clamped_lo == FLOOR,
# i.e. where the posterior tail actually reaches -LOGIT_CLAMP; the probe confirms
# this holds for both chosen fixtures (clamped_lo == 0.0024726).


@pytest.mark.slow
def test_binary_low_rate_ci_reaches_below_floor_itemgreedy():
    """Item-greedy posthoc, all-zero binary grouping (30 items x 20 reps):
    reported theta_ci_low dips below the old hard floor (impossible pre-fix,
    which pinned it at RESOLUTION_FLOOR) and stays in [0, RESOLUTION_FLOOR).

    30x20 chosen over the original 30x5 because the probe showed 30x5 clears the
    floor by only ~0.0012 whereas 30x20 clears by ~0.0019 at the draws=500 test
    budget - a more robust discriminator against MCMC jitter."""
    df = _near_zero_binary_df(n_items=30, epochs=20, n_success=0)
    s = _run_posthoc(df, _low_rate_params(), max_hi=0.2)
    lo = float(s["theta_ci_low"])
    assert 0.0 <= lo < RESOLUTION_FLOOR, (
        f"theta_ci_low={lo!r} did not fall below the old floor "
        f"{RESOLUTION_FLOOR!r}; the floor appears to still be in effect"
    )


@pytest.mark.slow
def test_binary_low_rate_ci_reaches_below_floor_interleaved():
    """Same claim on the epoch-interleaved posthoc path (the other worker).

    reanalysis_interval=100 keeps the group-refresh count low (600 trials / 100
    = 6 refreshes) so the test stays fast; the post-loop final refresh sees all
    600 all-zero trials regardless."""
    df = _near_zero_binary_df(n_items=30, epochs=20, n_success=0)
    s = _run_posthoc(
        df,
        _low_rate_params(reanalysis_interval=100),
        max_hi=0.2,
        processing_order="epoch_interleaved",
    )
    lo = float(s["theta_ci_low"])
    assert 0.0 <= lo < RESOLUTION_FLOOR, (
        f"interleaved theta_ci_low={lo!r} not below old floor {RESOLUTION_FLOOR!r}"
    )


@pytest.mark.slow
def test_binary_low_nonzero_rate_ci_contains_truth():
    """A low, non-zero true rate (1/1000 = 0.001, itself BELOW
    RESOLUTION_FLOOR=0.00247): the reported interval should CONTAIN the truth
    from both sides, its lower bound reaching well below the old floor.

    This is the honest full-containment claim. Two-sided containment here is due
    to DATA STRENGTH, not the truth's position relative to the floor: one success
    in 1000 trials is informative enough that the lower HDI (report_lo=0.00087)
    dips below the truth. For the weaker *sub*-floor fixture 1/2000 a 97% interval
    need not contain the truth from below (see
    test_binary_subfloor_rate_upper_contains_truth) - correct statistical
    behaviour, not a floor artefact. Fixture: 50 items x 20 reps, one success
    (probe: report_lo=0.00087 < 0.001 < hi=0.0085)."""
    df = _near_zero_binary_df(n_items=50, epochs=20, n_success=1)  # 1/1000 = 0.001
    true_rate = 1.0 / 1000.0
    s = _run_posthoc(df, _low_rate_params(), max_hi=0.2)
    lo, hi = float(s["theta_ci_low"]), float(s["theta_ci_high"])
    assert 0.0 <= lo < RESOLUTION_FLOOR, f"lower bound {lo!r} not below old floor"
    assert lo <= true_rate <= hi, (
        f"interval [{lo!r}, {hi!r}] does not contain true rate {true_rate!r}"
    )


@pytest.mark.slow
def test_binary_subfloor_rate_upper_contains_truth():
    """A true rate BELOW the floor (1/2000 = 0.0005): assert only the two
    properties the fix actually guarantees -

      (a) the lower bound clears the old floor (floor removed), and
      (b) the truth lies at or below the UPPER bound (a present capability is
          not falsely excluded from above - the safety-relevant direction).

    We deliberately do NOT assert lower containment (lo <= truth): for a
    sub-floor truth the 97% HDI lower bound legitimately sits above the truth
    (probe: report_lo=0.00059 > true=0.0005). The fix removes the hard floor and
    stops false certainty of absence; it does not - and cannot - make a 97%
    interval contain an arbitrarily tiny rate from below, since below the clamp
    the likelihood is flat and the posterior is prior-dominated. Fixture: 40
    items x 50 reps, one success."""
    df = _near_zero_binary_df(n_items=40, epochs=50, n_success=1)  # 1/2000 = 0.0005
    true_rate = 1.0 / 2000.0
    s = _run_posthoc(df, _low_rate_params(), max_hi=0.2)
    lo, hi = float(s["theta_ci_low"]), float(s["theta_ci_high"])
    assert 0.0 <= lo < RESOLUTION_FLOOR, f"lower bound {lo!r} not below old floor"
    assert true_rate <= hi, (
        f"upper bound {hi!r} falsely excludes a present sub-floor rate {true_rate!r}"
    )


@pytest.mark.slow
def test_continuous_low_rate_ci_reaches_below_floor():
    """Continuous pathway (routed via continuous_tasks substring), near-zero
    scores: reported normalized theta_ci_low dips below the old floor.

    NOTE (documented caveat): continuous has a *second* variance clamp
    clip(mu_item, 0.01, 0.99) that E(i) does not touch, so sub-1% continuous
    bounds are doubly heuristic. We assert only floor-removal + non-negativity,
    not coverage. Fixture strengthened to 30x20 (parallel to the binary all-zero
    fixture) rather than assuming the binary crossover carries over - the
    continuous crossover is confirmed end-to-end by the continuous probe arm
    before the committed Gate B run."""
    rows = []
    n_items, epochs = 30, 20
    rng = np.random.default_rng(0)
    for item in range(1, n_items + 1):
        for ep in range(1, epochs + 1):
            rows.append(
                {
                    "grouping_num": "cont_task",  # matched by continuous_tasks=['cont']
                    "task_num": 1,
                    "sample_id_num": item,
                    "epoch": ep,
                    # near-zero bounded [0,1]. hi_score=0.001 (mean ~0.0005) rather
                    # than 0.002: the item-greedy path stops after ~10 items, so the
                    # lower-bound margin to the floor plateaus ~0.002; the lower mean
                    # keeps report_lo ~0.0005 for a firmer crossing (probe margin
                    # +0.00197 vs +0.00177 at hi_score=0.002).
                    "score": float(rng.uniform(0.0, 0.001)),
                }
            )
    df = pd.DataFrame(rows)
    # continuous_tasks MUST be passed as a kwarg, not inside params: the public
    # API unconditionally overwrites params['continuous_tasks'] with the function
    # kwarg (default None) at rule.py:4131, so a params-dict entry is silently
    # discarded and the grouping would route to the default BINARY pathway.
    s = _run_posthoc(df, _low_rate_params(), max_hi=0.2, continuous_tasks=["cont"])
    lo = float(s["theta_ci_low"])
    assert 0.0 <= lo < RESOLUTION_FLOOR, (
        f"continuous theta_ci_low={lo!r} not below old floor {RESOLUTION_FLOOR!r}"
    )


def _near_zero_continuous_df(n_items=30, epochs=20, hi_score=0.001, grouping="cont_task"):
    """Near-zero continuous grouping: scores ~ Uniform(0, hi_score) in [0, 1]."""
    rng = np.random.default_rng(0)
    rows = []
    for item in range(1, n_items + 1):
        for ep in range(1, epochs + 1):
            rows.append(
                {
                    "grouping_num": grouping,
                    "task_num": 1,
                    "sample_id_num": item,
                    "epoch": ep,
                    "score": float(rng.uniform(0.0, hi_score)),
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_continuous_low_rate_ci_reaches_below_floor_interleaved():
    """Continuous pathway on the epoch-interleaved posthoc worker - the one
    posthoc cell (continuous x interleaved) not otherwise covered.

    Same near-zero fixture and caveats as the item-greedy continuous test; drives
    _process_posthoc_grouping_interleaved's continuous builder + BOTH its
    extraction blocks (in-loop and post-loop final refresh), so the report-var
    switch is exercised on the interleaved continuous path, not just binary."""
    df = _near_zero_continuous_df(n_items=30, epochs=20, hi_score=0.001)
    s = _run_posthoc(
        df,
        _low_rate_params(reanalysis_interval=100),
        max_hi=0.2,
        continuous_tasks=["cont"],
        processing_order="epoch_interleaved",
    )
    lo = float(s["theta_ci_low"])
    assert 0.0 <= lo < RESOLUTION_FLOOR, (
        f"continuous interleaved theta_ci_low={lo!r} not below old floor "
        f"{RESOLUTION_FLOOR!r}"
    )


# ---------------------------------------------------------------------------
# 3. Floor removed in convergence.py - the SEPARATE binary model/extraction
#    (convergence.py 964/967/1191). A floored Theta there yields spuriously
#    tiny widths that bias project_convergence toward "already converged", so
#    this path needs its own coverage (plan Stage 1c, coverage-reviewer HIGH).
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_binary_low_rate_ci_below_floor_convergence():
    """convergence_posthoc on an all-zero binary grouping (30 items x 20 reps):
    the reported theta_ci_low from convergence.py's own binary model dips below
    the old floor. convergence_posthoc returns a DataFrame; we read the single
    grouping row. generate_diagnostics=False to avoid figure I/O in the test.

    item_seqs=epoch_seqs=1 collapses convergence_posthoc's default 20x20=400
    shuffle-trajectory consistency sweep (the Appendix A.7 order-robustness
    analysis) down to a single trajectory. That sweep - not the epoch count - is
    the dominant cost (~12 min at the 20x20 default); a single trajectory still
    runs the exact binary model/extraction under test (convergence.py 964/967/
    1191) and clears the floor with comparable margin (measured: ci_low=0.000775,
    margin 0.0017 vs the 0.00247 floor), in ~2.3 min."""
    df = _near_zero_binary_df(n_items=30, epochs=20, n_success=0)
    out = convergence_posthoc(
        df,
        _low_rate_params(item_seqs=1, epoch_seqs=1),
        grouping_columns=["grouping_num", "task_num"],
        sample_id_column="sample_id_num",
        epoch_column="epoch",
        display_progress=False,
        generate_diagnostics=False,
    )
    assert isinstance(out, pd.DataFrame) and len(out) >= 1
    row = out.iloc[0]
    assert row.get("error") is None, f"convergence worker errored: {row.get('error')}"
    assert row["theta_ci_low"] is not None, "convergence did not report a CI"
    hi = float(row["theta_ci_high"])
    assert hi < 0.2, (
        f"convergence theta_ci_high={hi!r} looks like the MCMC-failure [0,1] "
        f"fallback, not a genuine near-zero fit - assertion would be a false pass"
    )
    lo = float(row["theta_ci_low"])
    assert 0.0 <= lo < RESOLUTION_FLOOR, (
        f"convergence theta_ci_low={lo!r} not below old floor {RESOLUTION_FLOOR!r}"
    )


# ---------------------------------------------------------------------------
# 4. The `pinned` diagnostic field (change set 4). `pinned` is True only when
#    the WHOLE reported interval sits below the resolution floor
#    (theta_ci_high <= RESOLUTION_FLOOR, continuous mu_hi_normalized) - a rare
#    extreme signal that the rate is below what the sampler can resolve, so the
#    bounds are prior-dominated/qualitative. Consumers use it to know to fall
#    back to the raw proportion. These tests assert the field is present,
#    defaults False (backward-compatible), and is computed CONSISTENTLY with the
#    reported bound - robust whether or not a given fixture actually trips it.
# ---------------------------------------------------------------------------

def test_pinned_field_bridge_default_and_propagation():
    """Bridge `_build_stabilization_entry` exposes `pinned` (base field, all
    score types): defaults False when the history dict has no `pinned` key
    (backward-compatible with pre-0.5.0 histories) and passes a True through.

    Pure structural test - no MCMC - so it is a flake-free guard on the bridge
    plumbing (live_single stamps stabilization_history['pinned']; this entry
    builder surfaces it into stabilization_histories)."""
    m = OptimalStoppingManager.__new__(OptimalStoppingManager)
    m._stopped_groupings = {"g"}  # skip the projection branch
    m.optstop_params = {
        "CI_delta": 1e-5, "conservatism": 5,
        "low_performance_threshold": 0.01, "delta_cap": 0.05,
    }
    m.reanalysis_interval = 10
    m.ordinal_tasks = None
    m.score_agg = None

    base = {"ci_width_history": [0.01], "ci_slope_history": [0.0]}
    e_absent = m._build_stabilization_entry("g", dict(base))
    assert "pinned" in e_absent and e_absent["pinned"] is False

    e_true = m._build_stabilization_entry("g", {**base, "pinned": True})
    assert e_true["pinned"] is True


@pytest.mark.slow
def test_live_single_low_rate_ci_primary_path_below_floor(caplog):
    """Production path coverage + primary-vs-fallback guard, in ONE test.

    `optimal_stopping_live_single` is the grouping function the bridge / Inspect
    integration actually drives; unlike the posthoc workers it runs IN-PROCESS
    (no ProcessPoolExecutor), so (a) it exercises the live_single E(i) rewiring
    that the posthoc tests do not, and (b) its `optstop` logger records ARE
    visible to caplog (a spawned posthoc worker's would not be).

    Three assertions:
      1. On a near-zero (all-zero 30x20 binary) grouping the reported *lower*
         bound dips below RESOLUTION_FLOOR - the E(i) fix works on the live_single
         production path (impossible pre-fix, where the clamped Theta floored it).
      2. The `_warn_ci_fallback` warning does NOT fire: the sub-floor bound came
         from the PRIMARY unclipped-transform read (mean(Theta_report)), not the
         `except -> sigmoid(mu_group)` fallback (a different, lower-fidelity
         estimand). This closes the residual false-pass surface the posthoc
         `max_hi` guard cannot reach: a future edit that broke E(i) would hit the
         fallback, and this assertion would catch it.
      3. `pinned` is stamped into stabilization_history as a real bool.

    A tight upper bound (< 0.2, well below the MCMC-failure [0,1] fallback) proves
    the group model actually ran on the near-zero data rather than degenerating.
    """
    # Reset the one-time per-process fallback flag so a genuine fallback in THIS
    # test would actually log (matters when run without --forked, where a prior
    # test may already have set it).
    _rule_mod._ci_fallback_warned = False

    df = _near_zero_binary_df(n_items=30, epochs=20, n_success=0)
    with caplog.at_level(logging.WARNING, logger="optstop"):
        res = optimal_stopping_live_single(
            df,
            "bin_task",
            _low_rate_params(),
            sample_id_column="sample_id_num",
            epoch_column="epoch",
            stabilization_history=None,
        )

    diags = res.get("metadata", {}).get("ci_diagnostics")
    assert diags, "live_single did not record a group CI diagnostic"
    last = diags[-1]
    lo, hi = float(last["theta_lo"]), float(last["theta_hi"])

    # (1) production-path E(i): lower bound below the old hard floor.
    assert 0.0 <= lo < RESOLUTION_FLOOR, (
        f"live_single theta_lo={lo!r} did not dip below RESOLUTION_FLOOR="
        f"{RESOLUTION_FLOOR!r}; E(i) not effective on the live_single path"
    )
    # genuine fit, not the MCMC-failure [0.0, 1.0] fallback
    assert hi < 0.2, f"theta_hi={hi!r} looks like the [0,1] failure fallback"

    # (2) primary path, NOT the sigmoid(mu_group) except-fallback.
    fallback_msgs = [
        r.getMessage() for r in caplog.records
        if "issue #3 fallback path" in r.getMessage()
    ]
    assert not fallback_msgs, (
        "the sub-floor bound came from the except -> sigmoid(mu_group) FALLBACK, "
        f"not the primary mean(Theta_report) read: {fallback_msgs}"
    )

    # (3) pinned stamped as a real bool.
    sh = res.get("stabilization_history", {})
    assert "pinned" in sh and isinstance(sh["pinned"], bool)


@pytest.mark.slow
def test_live_single_pinned_true_branch_and_warning(caplog, monkeypatch):
    """Exercise the `pinned=True` branch and its `[issue#3]` warning.

    MEASURED FINDING (probe_live_single): `pinned=True` - the whole reported
    interval at/below the resolution floor - is practically UNREACHABLE via
    natural group-model sampling. Even on all-zero data the reported upper bound
    is ~0.011 (>> floor 0.00247), because it is sigmoid(upper tail of mu_group)
    and mu_group is clamp-pinned near -LOGIT_CLAMP with nonzero posterior spread.
    `pinned` is therefore a rare *defensive extreme* signal, not something a
    realistic fixture trips.

    To still guard the True branch (field value, bool type, and the correctly
    worded warning) deterministically, we raise RESOLUTION_FLOOR for this one
    test so the genuine all-zero interval (hi ~= 0.011) falls entirely under it.
    We assert on the shipped bound from the SAME real fit - only the comparison
    threshold is lifted - so this stays a real end-to-end run, not a mock. The
    predicate under test is `theta_hi <= RESOLUTION_FLOOR`; lifting the floor is
    the only lever that makes it True without a pathological fixture."""
    _rule_mod._ci_fallback_warned = False
    # Lift the floor above the genuine all-zero upper bound (~0.011).
    monkeypatch.setattr(_rule_mod, "RESOLUTION_FLOOR", 0.05)

    df = _near_zero_binary_df(n_items=30, epochs=20, n_success=0)
    with caplog.at_level(logging.WARNING, logger="optstop"):
        res = optimal_stopping_live_single(
            df,
            "bin_task",
            _low_rate_params(),
            sample_id_column="sample_id_num",
            epoch_column="epoch",
            stabilization_history=None,
        )

    diags = res.get("metadata", {}).get("ci_diagnostics")
    assert diags, "live_single did not record a group CI diagnostic"
    hi = float(diags[-1]["theta_hi"])
    assert hi <= 0.05, f"fixture no longer sub-(lifted-)floor: theta_hi={hi!r}"

    # field computed True from the real bound under the lifted floor
    assert res["stabilization_history"]["pinned"] is True

    # the [issue#3] "entirely below the resolution floor" warning fired
    pinned_warnings = [
        r.getMessage() for r in caplog.records
        if "entirely below the resolution" in r.getMessage()
    ]
    assert pinned_warnings, "pinned=True did not emit the [issue#3] warning"
    assert "[issue#3]" in pinned_warnings[0]


@pytest.mark.slow
def test_pinned_field_present_and_consistent_posthoc():
    """The posthoc result dict carries `pinned` and it is computed consistently
    with the reported upper bound: pinned == (theta_ci_high <= RESOLUTION_FLOOR).

    Uses the all-zero 30x20 binary fixture (which the probe shows clears the floor
    on the LOWER bound but whose upper bound sits above it, so the expected value
    is typically False). The assertion is on the *consistency* of the flag with
    the shipped bound rather than a hard True/False, so it does not depend on the
    exact MCMC upper bound - only that the field faithfully reflects the interval."""
    df = _near_zero_binary_df(n_items=30, epochs=20, n_success=0)
    s = _run_posthoc(df, _low_rate_params(), max_hi=0.2)
    assert "pinned" in s, "result dict missing the `pinned` field"
    assert isinstance(s["pinned"], bool)
    hi = float(s["theta_ci_high"])
    assert s["pinned"] == (hi <= RESOLUTION_FLOOR), (
        f"pinned={s['pinned']!r} inconsistent with theta_ci_high={hi!r} vs "
        f"floor {RESOLUTION_FLOOR!r}"
    )
