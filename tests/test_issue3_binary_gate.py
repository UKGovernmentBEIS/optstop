"""Phase 5 tests for the issue #3 BINARY perf-gate.

Companion to ``test_issue3_perf_gate.py`` (continuous) and
``test_issue3_ci_floor.py`` (E(i) floor-removal). This file covers the binary
half of the low-performance stopping gate: at observed performance below
``low_performance_threshold`` the BINARY group-level pathway must NOT early-stop
via EITHER route -

  * WIDTH / precision route  (``effective_width < delta_cap``), and
  * SLOPE / stabilisation route (slope flattens below the low-perf threshold).

Why gate binary at all (it has an honestly-wide interval, unlike continuous)?
Below ``low_performance_threshold`` the binary width/slope targets are relaxed by
``conservatism`` (``delta_cap`` stays, but ``effective_width = width * conservatism``
and ``slope_threshold = CI_delta / conservatism``). With enough accumulated null
data the post-E(i) near-zero interval is narrow enough that even the widened
target is met - so a rare-capability grouping could stop on pre-event null data,
which is exactly the failure a low-base-rate search must avoid. Gating suppresses
that stop until the observed rate clears the threshold; the event then lifts the
rate and the ordinary precision stop can fire. Ordinal is out of scope.

The gate surfaces ``low_perf_stop_suppressed`` (default False), set True on the
refresh where a stop criterion was met but suppressed. It is threaded through the
posthoc result/error dicts, the posthoc aggregator, and - for
live_single/bridge - through ``stabilization_history`` into the bridge
``_build_stabilization_entry``. (The bridge propagation and projection-skip
plumbing are score-type agnostic and are covered in
``test_issue3_perf_gate.py``; they are not duplicated here.)

The end-to-end MCMC tests are marked ``slow``; run per file with the pinned
stack, ideally with process isolation:

    /home/ubuntu/optstop/venv/bin/python -m pytest tests/test_issue3_binary_gate.py -v
    /home/ubuntu/optstop/venv/bin/python -m pytest tests/test_issue3_binary_gate.py -v --forked
"""

import numpy as np
import pandas as pd
import pytest

from optstop import optimal_stopping_posthoc
from optstop.rule import optimal_stopping_live_single


# Small-but-adequate seeded MCMC budget (matches the sibling issue #3 files).
_MCMC = {
    "draws": 500,
    "tune": 1000,
    "chains": 2,
    "cores": 2,
    "random_seed": 20260901,
}


def _low_rate_params(**extra):
    """Canonical low-rate params. ``low_performance_threshold=0.01`` with
    all-zero data puts the grouping firmly in the gated regime."""
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


def _near_zero_binary_df(n_items=30, epochs=20, grouping="bin_task"):
    """Near-zero binary grouping: all pass/fail scores are 0 (rate exactly 0,
    firmly below low_performance_threshold=0.01). This is the paradigmatic
    rare-capability-not-yet-observed state."""
    rows = []
    for item in range(1, n_items + 1):
        for ep in range(1, epochs + 1):
            rows.append(
                {
                    "grouping_num": grouping,
                    "task_num": 1,
                    "sample_id_num": item,
                    "epoch": ep,
                    "score": 0,
                }
            )
    return pd.DataFrame(rows)


def _run_posthoc(df, params, **kw):
    """Run the public posthoc API (binary is the default route - no
    continuous_tasks/ordinal_tasks), return the single grouping summary, and
    guard against the MCMC-failure fallback (theta_ci spanning [0, 1])
    masquerading as a genuine near-zero fit."""
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
    assert s["theta_ci_high"] is not None and float(s["theta_ci_high"]) < 0.2, (
        f"theta_ci_high={s.get('theta_ci_high')!r} looks like the MCMC-failure "
        f"[0,1] fallback, not a genuine near-zero fit"
    )
    return s


# ---------------------------------------------------------------------------
# WIDTH route (positive suppression demonstration). At the default
# ``conservatism=5`` the binary width target is relaxed to
# ``effective_width = width * 5``, which an honestly-wide near-zero binary
# interval typically does NOT meet on modest data - so there is nothing to
# suppress and the grouping simply keeps sampling (that "never stops" invariant
# is covered by the cumulative test below). To exercise the suppression branch
# deterministically we set ``conservatism=1``: the gate keys on
# ``perf < low_performance_threshold`` (independent of conservatism), so the
# grouping is still in the gated regime, but now the un-widened width criterion
# (``width < delta_cap``) is reliably met - and must be suppressed rather than
# taken. This isolates the gate from the conservatism-widening confound.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_binary_low_perf_no_stop_width_itemgreedy():
    """Item-greedy posthoc, all-zero binary grouping (conservatism=1): the width
    criterion is met but suppressed, so the group never stops and every item is
    processed (percent_items_used == 1.0), with low_perf_stop_suppressed True."""
    df = _near_zero_binary_df(n_items=30, epochs=20)
    s = _run_posthoc(df, _low_rate_params(conservatism=1))
    assert s["low_perf_stop_suppressed"] is True, (
        "expected the below-threshold binary width stop to be suppressed at ~0 perf"
    )
    assert float(s["percent_items_used"]) == pytest.approx(1.0), (
        f"grouping stopped early (percent_items_used={s['percent_items_used']!r}); "
        f"the low-perf stop was not suppressed"
    )


@pytest.mark.slow
def test_binary_low_perf_no_stop_width_interleaved():
    """Same claim on the epoch-interleaved posthoc worker (conservatism=1).
    percent_items_used is the fraction of TOTAL trials used in interleaved mode;
    a suppressed grouping runs to exhaustion so it is ~1.0."""
    df = _near_zero_binary_df(n_items=30, epochs=20)
    s = _run_posthoc(
        df,
        _low_rate_params(conservatism=1, reanalysis_interval=100),
        processing_order="epoch_interleaved",
    )
    assert s["low_perf_stop_suppressed"] is True
    assert float(s["percent_items_used"]) > 0.99, (
        f"interleaved grouping stopped early "
        f"(percent_items_used={s['percent_items_used']!r})"
    )


@pytest.mark.slow
def test_binary_low_perf_no_stop_width_live_single():
    """Production path (bridge drives live_single), conservatism=1. An
    all-epochs-present all-zero binary call: the width criterion is met but
    suppressed, so the grouping is NOT added to stop_this_grouping and the flag is
    stamped True into stabilization_history. Binary is the default route
    (is_aggregated omitted)."""
    df = _near_zero_binary_df(n_items=30, epochs=20, grouping="bin_grp")
    res = optimal_stopping_live_single(
        df,
        "bin_grp",
        _low_rate_params(conservatism=1),
        sample_id_column="sample_id_num",
        epoch_column="epoch",
        stabilization_history=None,
    )
    stopped = res.get("grouping") in (res.get("stop_this_grouping") or [])
    assert not stopped, "live_single stopped a near-zero binary grouping (width route)"
    sh = res.get("stabilization_history", {})
    assert sh.get("low_perf_stop_suppressed") is True, (
        "live_single did not flag the suppressed binary width-route stop"
    )
    # sanity: it genuinely met the width criterion (width < delta_cap at
    # conservatism=1, otherwise there would be nothing to suppress).
    widths = sh.get("ci_width_history") or []
    assert widths, "no group CI recorded"
    assert float(widths[-1]) < 0.05, (
        f"final width {widths[-1]!r} did not meet delta_cap; nothing to suppress"
    )


# ---------------------------------------------------------------------------
# Cumulative-epoch invariant at the DEFAULT conservatism (=5): across a growing
# all-zero binary run the grouping must NEVER stop. This is the property that
# matters for the detection-trigger use case (which runs at default
# conservatism). "flag ever True" is a soft positive: with the width target
# widened x5 a criterion may not be met within the budget, so we only report it.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_binary_low_perf_never_stops_cumulative_live_single():
    full = _near_zero_binary_df(n_items=20, epochs=20, grouping="bin_grp")
    params = _low_rate_params()  # default conservatism=5
    sh, caches = None, None
    ever_suppressed = False
    stopped_at = None
    for e in range(1, 21):
        df = full[full["epoch"] <= e].copy()
        res = optimal_stopping_live_single(
            df,
            "bin_grp",
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
        f"live_single stopped a near-zero binary grouping at epoch {stopped_at} at "
        f"default conservatism; the below-threshold stop was not suppressed"
    )
    # Soft: at conservatism=5 the widened target may not be met within 20 epochs.
    # The hard invariant above ("never stops") is the safety-relevant claim.
    if not ever_suppressed:
        import warnings
        warnings.warn(
            "binary suppression flag never set at default conservatism within 20 "
            "epochs (widened width/slope target not met) - the 'never stops' "
            "invariant still holds; see the conservatism=1 tests for the positive "
            "suppression demonstration"
        )
