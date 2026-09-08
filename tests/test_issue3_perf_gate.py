"""Phase 5 tests for the issue #3 continuous perf-gate ("option 1").

Companion to ``test_issue3_ci_floor.py`` (which covers the E(i) floor-removal).
This file covers the SEPARATE change: at true performance below
``low_performance_threshold`` the CONTINUOUS pathway must NOT early-stop via
EITHER route -

  * WIDTH / precision route  (``effective_width < delta_cap``), and
  * SLOPE / stabilisation route (slope flattens below the low-perf threshold).

Root cause (continuous): the continuous obs-SD variance clamp
``clip(mu_item, 0.01, 0.99)`` (independent of E(i)) floors the modelled per-item
variance, so the continuous group credible interval is artificially narrow and
slowly-flattening at ~0 performance - which trips both stop routes even though
the true rate is ~0.

The gate now also applies to the BINARY pathway (see
``test_issue3_binary_gate.py``): although a binary near-zero interval is honestly
wide, ``conservatism`` already relaxes the binary width/slope targets below the
threshold, so accumulated null data could still stop a rare-capability grouping
before the event was ever observed. Both pathways therefore suppress the
below-threshold stop; ordinal is out of scope. See
``Temp_Dev_Files/issue3_option1_edit_spec.md``.

The gate surfaces a diagnostic bool ``low_perf_stop_suppressed`` (default False),
set True on the refresh where a stop criterion was met but suppressed. It is
threaded through the posthoc result/error dicts, the posthoc aggregator, and -
for live_single/bridge - through ``stabilization_history`` into the bridge
``_build_stabilization_entry``. A suppressed grouping must NOT receive a
convergence projection (its narrow width is a clamp artefact, so
``project_convergence`` would spuriously report "already converged").

Structural tests (no MCMC) are flake-free. The end-to-end MCMC tests are marked
``slow``; run per file with the pinned stack, ideally with process isolation:

    /home/ubuntu/optstop/venv/bin/python -m pytest tests/test_issue3_perf_gate.py -v
    /home/ubuntu/optstop/venv/bin/python -m pytest tests/test_issue3_perf_gate.py -v --forked
"""

import numpy as np
import pandas as pd
import pytest

from optstop import optimal_stopping_posthoc
from optstop.early_stopping import OptimalStoppingManager
from optstop.rule import optimal_stopping_live_single
import optstop.convergence as _conv_mod


# Small-but-adequate seeded MCMC budget (matches test_issue3_ci_floor.py).
_MCMC = {
    "draws": 500,
    "tune": 1000,
    "chains": 2,
    "cores": 2,
    "random_seed": 20260901,
}


def _low_rate_params(**extra):
    """Canonical low-rate params. ``low_performance_threshold=0.01`` with
    near-zero data puts the grouping firmly in the gated regime."""
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


def _near_zero_continuous_df(n_items=30, epochs=20, hi_score=0.002, grouping="cont_task"):
    """Near-zero continuous grouping: scores ~ Uniform(0, hi_score) in [0, 1].
    Mean ~ hi_score/2, well below low_performance_threshold=0.01."""
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


def _run_posthoc(df, params, **kw):
    """Run the public posthoc API, return the single grouping summary, and guard
    against the MCMC-failure fallback (theta_ci in [0.0, 1.0]) masquerading as a
    genuine near-zero fit."""
    _, summary = optimal_stopping_posthoc(
        df,
        params,
        grouping_columns=["grouping_num", "task_num"],
        sample_id_column="sample_id_num",
        epoch_column="epoch",
        continuous_tasks=["cont"],  # route the 'cont_task' grouping to continuous
        **kw,
    )
    assert isinstance(summary, list) and len(summary) >= 1
    s = summary[0]
    assert s.get("error") is None, f"worker errored: {s.get('error')}"
    assert s["theta_ci_high"] is not None and float(s["theta_ci_high"]) < 0.2, (
        f"theta_ci_high={s.get('theta_ci_high')!r} looks like the MCMC-failure "
        f"[0,1] fallback, not a genuine near-zero fit"
    )
    return s


# ---------------------------------------------------------------------------
# 1. Structural / plumbing tests (no MCMC - flake-free).
# ---------------------------------------------------------------------------

def _bare_manager():
    """A minimally-initialised manager suitable for calling
    ``_build_stabilization_entry`` directly (mirrors the pinned structural
    test in test_issue3_ci_floor.py)."""
    m = OptimalStoppingManager.__new__(OptimalStoppingManager)
    m._stopped_groupings = set()
    m.optstop_params = {
        "CI_delta": 1e-5,
        "conservatism": 5,
        "low_performance_threshold": 0.01,
        "delta_cap": 0.05,
    }
    m.reanalysis_interval = 10
    m.ordinal_tasks = None
    m.score_agg = None
    return m


def test_low_perf_stop_suppressed_bridge_default_and_propagation():
    """Bridge ``_build_stabilization_entry`` exposes ``low_perf_stop_suppressed``
    as a base field (all score types): defaults False when the history dict lacks
    the key (backward-compatible with pre-0.5.0 histories) and passes a True
    through. Pure structural test - no MCMC."""
    m = _bare_manager()
    m._stopped_groupings = {"g"}  # skip the projection branch entirely

    base = {"ci_width_history": [0.01], "ci_slope_history": [0.0]}
    e_absent = m._build_stabilization_entry("g", dict(base))
    assert "low_perf_stop_suppressed" in e_absent
    assert e_absent["low_perf_stop_suppressed"] is False

    e_true = m._build_stabilization_entry(
        "g", {**base, "low_perf_stop_suppressed": True}
    )
    assert e_true["low_perf_stop_suppressed"] is True


def test_bridge_skips_projection_when_suppressed(monkeypatch):
    """The KEY safety plumbing test: a suppressed (below-threshold, not-stopped)
    continuous grouping must NOT be handed to ``project_convergence`` in the
    bridge - otherwise its clamp-narrow width (< delta_cap by construction) would
    yield a spurious "already converged, 0 trials, high confidence" projection.

    The bridge guard is the sole discriminator here: with everything else held
    identical (grouping not stopped, a real final width), flipping only
    ``low_perf_stop_suppressed`` must flip whether project_convergence is called.
    We monkeypatch project_convergence with a call-recorder (the function is
    imported inside _build_stabilization_entry via ``from .convergence import
    project_convergence``, so patching the convergence module attribute works)."""
    calls = []

    def _recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return None  # emulate "no projection"

    monkeypatch.setattr(_conv_mod, "project_convergence", _recorder)

    # A narrow, flattening width history typical of a suppressed low-perf cell.
    history = {
        "ci_width_history": [0.09, 0.06, 0.045, 0.04, 0.038],  # final < delta_cap
        "ci_slope_history": [-0.03, -0.015, -0.005, -0.002],
        "current_perf_estimate": 0.001,
        "n_samples_evaluated": 600,
    }

    m = _bare_manager()  # grouping NOT in _stopped_groupings

    # Suppressed -> projection MUST be skipped.
    e_supp = m._build_stabilization_entry(
        "cont_grp", {**history, "low_perf_stop_suppressed": True}
    )
    assert "convergence_projection" not in e_supp
    assert calls == [], "project_convergence was called for a suppressed grouping"

    # Control: identical history, flag False -> projection IS attempted. Proves
    # the skip is caused by the suppression guard, not by some other condition.
    calls.clear()
    e_norm = m._build_stabilization_entry(
        "cont_grp", {**history, "low_perf_stop_suppressed": False}
    )
    assert len(calls) == 1, (
        "project_convergence should run for a non-suppressed grouping with a "
        "real final width"
    )
    # _recorder returns None, so no key is attached - but the call happened.
    assert "convergence_projection" not in e_norm


# ---------------------------------------------------------------------------
# 2. WIDTH route: near-zero continuous does not stop; flag is set. Covers all
#    three code paths (item-greedy posthoc, interleaved posthoc, live_single).
#    The variance clamp makes effective_width < delta_cap reliably at ~0 perf,
#    so the width route is the criterion that fires - and is suppressed.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_continuous_low_perf_no_stop_width_itemgreedy():
    """Item-greedy posthoc, near-zero continuous grouping: the width criterion is
    met but suppressed, so the group never stops and every item is processed
    (percent_items_used == 1.0), with low_perf_stop_suppressed True."""
    df = _near_zero_continuous_df(n_items=30, epochs=20)
    s = _run_posthoc(df, _low_rate_params())
    assert s["low_perf_stop_suppressed"] is True, (
        "expected the width-route stop to be suppressed at ~0 perf"
    )
    assert float(s["percent_items_used"]) == pytest.approx(1.0), (
        f"grouping stopped early (percent_items_used={s['percent_items_used']!r}); "
        f"the low-perf stop was not suppressed"
    )


@pytest.mark.slow
def test_continuous_low_perf_no_stop_width_interleaved():
    """Same claim on the epoch-interleaved posthoc worker. percent_items_used is
    the fraction of TOTAL trials used in interleaved mode; a suppressed grouping
    runs to exhaustion so it is ~1.0."""
    df = _near_zero_continuous_df(n_items=30, epochs=20)
    s = _run_posthoc(
        df,
        _low_rate_params(reanalysis_interval=100),
        processing_order="epoch_interleaved",
    )
    assert s["low_perf_stop_suppressed"] is True
    assert float(s["percent_items_used"]) > 0.99, (
        f"interleaved grouping stopped early "
        f"(percent_items_used={s['percent_items_used']!r})"
    )


@pytest.mark.slow
def test_continuous_low_perf_no_stop_width_live_single():
    """Production path (bridge drives live_single). A single all-epochs-present
    near-zero continuous call: the width criterion is met but suppressed, so the
    grouping is NOT added to stop_this_grouping and the flag is stamped True into
    stabilization_history. Routed continuous via is_aggregated=True (live_single
    infers continuous from aggregation, not from continuous_tasks)."""
    df = _near_zero_continuous_df(n_items=30, epochs=20, grouping="cont_grp")
    res = optimal_stopping_live_single(
        df,
        "cont_grp",
        _low_rate_params(is_aggregated=True),
        sample_id_column="sample_id_num",
        epoch_column="epoch",
        stabilization_history=None,
    )
    stopped = res.get("grouping") in (res.get("stop_this_grouping") or [])
    assert not stopped, "live_single stopped a near-zero continuous grouping (width route)"
    sh = res.get("stabilization_history", {})
    assert sh.get("low_perf_stop_suppressed") is True, (
        "live_single did not flag the suppressed width-route stop"
    )
    # sanity: it genuinely met the width criterion (otherwise nothing to suppress)
    widths = sh.get("ci_width_history") or []
    assert widths, "no group CI recorded"


# ---------------------------------------------------------------------------
# 3. SLOPE route (isolated): with delta_cap set tiny the width route CANNOT
#    fire (effective_width stays >> delta_cap), so any suppression flag can only
#    come from the stabilisation/slope branch (rule.py 5829). Feeding cumulative
#    epochs lets the width history flatten until the slope crosses the low-perf
#    threshold, at which point the stop is suppressed rather than taken.
#
#    Stochastic caveat: the exact epoch at which the slope stabilises is
#    MCMC-dependent; the empirical harness (perf0_cont_slope) saw it at epochs
#    12 and 15 of 20 at this fixture/seed. The invariant hard-asserted here is
#    "never stops"; "flag became True via the slope branch at some epoch" is the
#    positive slope-route demonstration and is robust because each refresh resets
#    the flag, so we track "ever True" across the cumulative run.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_continuous_low_perf_no_stop_slope_route_live_single():
    full = _near_zero_continuous_df(n_items=20, epochs=20, grouping="cont_grp")
    params = _low_rate_params(is_aggregated=True, delta_cap=0.003)  # tiny -> width route off
    sh, caches = None, None
    ever_suppressed = False
    stopped_at = None
    for e in range(1, 21):
        df = full[full["epoch"] <= e].copy()
        res = optimal_stopping_live_single(
            df,
            "cont_grp",
            params,
            sample_id_column="sample_id_num",
            epoch_column="epoch",
            stabilization_history=sh,
            model_caches=caches,
        )
        sh = res.get("stabilization_history")
        caches = res.get("model_caches")
        if res.get("grouping") in (res.get("stop_this_grouping") or []):
            stopped_at = e
            break
        if (sh or {}).get("low_perf_stop_suppressed") is True:
            ever_suppressed = True

    assert stopped_at is None, (
        f"live_single stopped a near-zero continuous grouping at epoch {stopped_at} "
        f"with the width route disabled (delta_cap=0.003) - the SLOPE route was not "
        f"suppressed"
    )
    assert ever_suppressed, (
        "the slope stabilisation criterion never fired across 20 cumulative epochs, "
        "so the slope-route suppression branch was not exercised (may be MCMC "
        "stochasticity - re-run; the 'never stops' invariant above still holds)"
    )
