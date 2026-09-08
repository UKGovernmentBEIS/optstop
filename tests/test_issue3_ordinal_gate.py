"""Phase 5 tests for the issue #3 ORDINAL perf-gate (ordinal transferability).

Companion to ``test_issue3_perf_gate.py`` (binary/continuous perf-gate) and
``test_issue3_ci_floor.py`` (the E(i) floor-removal). This file covers the
extension of the below-``low_performance_threshold`` stop-suppression to the
ORDINAL pathway.

Why ordinal needs the gate (and why it is a DIFFERENT mechanism from the
binary/continuous sigmoid(-6) floor):

  * Ordinal group-level CIs are read from the modal-category / entropy posterior
    (ordered_logistic or Dirichlet), NOT from ``sigmoid(clip(mu, -6, 6))`` - so
    the sigmoid resolution floor that E(i) removes does not exist here, and the
    E(i) ``pinned`` read-out fix does NOT transfer.
  * What DOES transfer is the stopping HAZARD: a genuinely-absent capability
    presents as data confidently peaked at category 0 (all-category-0). That
    looks converged to the ordinal stop routes and would stop a rare-capability
    grouping before a positive category was ever observed - the same #3 safety
    hazard as binary/continuous.

The fix gates every ordinal group-stop behind ``current_perf >= low_perf_threshold``
(normalised), mirroring binary/continuous, WITHOUT re-applying conservatism to
the ordinal width (already conservatised inside the CI functions) and WITHOUT
setting ``pinned`` (an E(i)/sigmoid concept that does not apply to ordinal).

Which ordinal routes are a genuine sub-threshold hazard (and are tested here):

  * MODAL width (``effective_width < delta_cap``). For all-category-0 data the
    modal HDI collapses to [0,0] so ``raw_width == 0`` and the returned width is
    the deterministic conservative floor
    ``min_ci_width * conservatism = (1/(ordinal_max_score*sqrt(n))) * conservatism``
    = ``0.5/sqrt(n)`` at ordinal_max_score=10, conservatism=5. This crosses the
    default ``delta_cap=0.05`` once ``n_samples > 100`` (0.5/sqrt(100)=0.05), so a
    large-n absent-capability grouping WOULD false-stop - the gate suppresses it.
    Because it is deterministic in ``n``, these tests are flake-free once the
    fixture clears ~120 observations.
  * HYBRID (``_ordinal_hybrid_stopping_criterion_hierarchical``): Pathway 1
    (hierarchical modal narrow + entropy-validated true peak) and/or Pathway 2
    (absolute entropy CI width < entropy_convergence_threshold=0.10). At a
    confidently-peaked-at-0 distribution with ~120 obs, Pathway 1's modal-narrow
    condition is met (same 0.5/sqrt(n) crossing) so the hybrid criterion fires -
    the gate suppresses it. This is the entropy-family demonstration.

Which routes are NOT a sub-threshold hazard (and are therefore not exercised by
an MCMC test - the gate branch is present for consistency with binary/continuous
but is unreachable at sub-threshold perf, verified by construction):

  * MODAL/entropy SLOPE-stabilisation (posthoc workers only). At sub-threshold
    perf (mean < ~0.1 on 0..10, i.e. all mass on category 0) the modal HDI is
    [0,0] so ``raw_width == 0`` and the width equals the ``0.5/sqrt(n)`` floor,
    which DECAYS monotonically in n and never plateaus - so the slope never
    flattens below ``slope_threshold`` (=CI_delta/conservatism=2e-5) while still
    above delta_cap. A plateau above delta_cap would require a stable non-zero
    modal spread, which in turn requires appreciable non-zero-category mass, i.e.
    perf ABOVE the threshold. The slope-route suppression branch (rule.py ~2114 /
    ~3192) is thus dead code at sub-threshold and is covered by the binary/
    continuous slope tests in test_issue3_perf_gate.py, which share the idiom.
  * Standalone ENTROPY width (``raw_width * conservatism < delta_cap`` needs
    ``raw_width < 0.01`` scaled) is MCMC-stochastic near the boundary; the hybrid
    test above covers the entropy CI function's routing through the gate via the
    looser, reliable Pathway 2 / Pathway 1 conditions.

Suppression telemetry reuses each path's existing mechanism:
  * posthoc workers (item-greedy + epoch-interleaved): the local
    ``low_perf_stop_suppressed`` bool, surfaced in the result/error dicts and the
    aggregator;
  * ``optimal_stopping_live_single``: ``stabilization_history['low_perf_stop_suppressed']``
    (routed through the bridge ``_build_stabilization_entry``).
The live-grouping worker (``_process_live_grouping``) has no such field in its
return dict (it returns only stop lists), so it simply does not stop - there is
no flag to assert there.

In addition to ``low_perf_stop_suppressed`` (which requires a stop criterion to
have fired), a below-threshold ordinal grouping is marked with the standalone
telemetry flag ``low_perf_floor=True`` (Option C). This is the ordinal counterpart
to binary/continuous ``pinned``: it is set whenever the resolved normalised
performance is below ``low_performance_threshold`` (keyed off the point estimate,
so mode-independent across modal/entropy/hybrid), it does NOT affect stopping, and
- unlike ``pinned`` - it carries no reliability caveat, because the ordinal
estimator has no sigmoid location clamp. The tests below assert it alongside the
suppression flag (True on the near-zero fixtures, False on the mid-perf control).

The MCMC tests are marked ``slow`` (ordinal ordered_logistic / Dirichlet
sampling is slow; hybrid runs a per-item fit); run per file with the pinned
stack, ideally with process isolation:

    /home/ubuntu/optstop/venv/bin/python -m pytest tests/test_issue3_ordinal_gate.py -v
    /home/ubuntu/optstop/venv/bin/python -m pytest tests/test_issue3_ordinal_gate.py -v --forked
"""

import numpy as np
import pandas as pd
import pytest

from optstop import optimal_stopping_posthoc
from optstop.rule import optimal_stopping_live_single


# Small-but-adequate seeded MCMC budget (matches test_issue3_perf_gate.py).
_MCMC = {
    "draws": 500,
    "tune": 1000,
    "chains": 2,
    "cores": 2,
    "random_seed": 20260901,
}

_ORD_MAX = 10  # 0..10 ordinal scale (K=11), the package default



# CI partition: heavy MCMC tests deselected from PR CI (see pyproject.toml markers).
pytestmark = pytest.mark.optstop


def _low_rate_params(**extra):
    """Canonical low-rate params. ``low_performance_threshold=0.01`` with
    near-zero ordinal data (normalised perf << 0.01) puts the grouping firmly in
    the gated regime."""
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


def _near_zero_ordinal_df(n_items, epochs, n_ones=2, grouping="ord_task"):
    """Near-zero ordinal grouping on the 0..10 scale: almost all scores are
    category 0, with ``n_ones`` scattered category-1 scores so the sampler has a
    little variation (avoids a fully-degenerate all-identical fit) while the mean
    stays far below the gate.

    Mean score ~ n_ones / (n_items*epochs); normalised (÷10) is << 0.01, so the
    grouping is below ``low_performance_threshold`` yet confidently peaked at
    category 0 - exactly the #3 hazard.

    Modal-width note: with all mass on category 0 the modal HDI collapses to
    [0,0] (raw_width 0), so the returned modal width is the deterministic floor
    ``0.5/sqrt(n_samples)`` - crossing delta_cap=0.05 once n_samples > 100.
    Fixtures here therefore use >= ~120 observations so the modal criterion is
    reliably MET (and then suppressed)."""
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
                    "score": 0,
                }
            )
    total = len(rows)
    for idx in rng.choice(total, size=min(n_ones, total), replace=False):
        rows[idx]["score"] = 1
    df = pd.DataFrame(rows)
    # guard: fixture must actually be in the gated regime
    assert (df["score"].mean() / _ORD_MAX) < 0.01
    return df


def _mid_perf_ordinal_df(n_items, epochs, mode_cat=5, grouping="ord_task"):
    """Mid-performance ordinal grouping tightly peaked at ``mode_cat`` (0..10).
    Normalised perf ~ mode_cat/10 = 0.5, well ABOVE the gate, so a normal stop
    must still fire (the regression control)."""
    rng = np.random.default_rng(1)
    rows = []
    for item in range(1, n_items + 1):
        for ep in range(1, epochs + 1):
            jitter = rng.choice([-1, 0, 0, 0, 1])
            score = int(np.clip(mode_cat + jitter, 0, _ORD_MAX))
            rows.append(
                {
                    "grouping_num": grouping,
                    "task_num": 1,
                    "sample_id_num": item,
                    "epoch": ep,
                    "score": score,
                }
            )
    return pd.DataFrame(rows)


def _run_posthoc_ordinal(df, params, inference, **kw):
    """Run the public posthoc API on an ordinal grouping and return the single
    grouping summary. Routes to ordinal via ``ordinal_tasks=['ord']`` (the
    grouping name contains 'ord')."""
    _, summary = optimal_stopping_posthoc(
        df,
        params,
        grouping_columns=["grouping_num", "task_num"],
        sample_id_column="sample_id_num",
        epoch_column="epoch",
        ordinal_tasks=["ord"],
        ordinal_max_score=_ORD_MAX,
        ordinal_inference=inference,
        **kw,
    )
    assert isinstance(summary, list) and len(summary) >= 1
    s = summary[0]
    assert s.get("error") is None, f"worker errored: {s.get('error')}"
    return s


def _run_live_single_ordinal(df, params, inference, grouping="ord_grp"):
    """Single all-epochs-present live_single call routed to ordinal."""
    return optimal_stopping_live_single(
        df,
        grouping,
        params,
        sample_id_column="sample_id_num",
        epoch_column="epoch",
        ordinal_tasks=["ord"],
        ordinal_max_score=_ORD_MAX,
        ordinal_inference=inference,
        stabilization_history=None,
    )


# ---------------------------------------------------------------------------
# 1. WIDTH route, MODAL. Fast (modal item-level is a cheap bootstrap; only group
#    refreshes are MCMC) and DETERMINISTIC (peaked-at-0 modal width == 0.5/sqrt(n)).
#    Covers every flag-bearing path: item-greedy + epoch-interleaved posthoc
#    workers and live_single. At >=120 obs the modal CI width falls below
#    delta_cap (criterion MET) but the sub-threshold perf suppresses the stop.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_ordinal_low_perf_no_stop_modal_itemgreedy():
    """Item-greedy posthoc, near-zero ordinal (modal): the modal-width criterion
    is met but suppressed, so the group never stops and every item is processed
    (percent_items_used == 1.0), with low_perf_stop_suppressed True."""
    df = _near_zero_ordinal_df(n_items=15, epochs=12)  # 180 obs -> width 0.037 < 0.05
    s = _run_posthoc_ordinal(df, _low_rate_params(), inference="modal")
    assert s["low_perf_stop_suppressed"] is True, (
        "expected the ordinal modal width-route stop to be suppressed at ~0 perf"
    )
    assert float(s["percent_items_used"]) == pytest.approx(1.0), (
        f"grouping stopped early (percent_items_used={s['percent_items_used']!r}); "
        f"the low-perf ordinal stop was not suppressed"
    )
    # Option C telemetry: resolved at a very-low performance level.
    assert s["low_perf_floor"] is True, (
        "expected low_perf_floor True for a near-zero ordinal grouping"
    )


@pytest.mark.slow
def test_ordinal_low_perf_no_stop_modal_interleaved():
    """Same claim on the epoch-interleaved posthoc worker (modal). reanalysis
    interval 60 gives in-loop group refreshes at 60/120/180 trials; the modal
    width crosses delta_cap at 120 trials and the in-loop suppression branch
    fires.

    The definitive safety assertion is ``low_perf_stop_suppressed is True``: in
    the interleaved worker that flag is set True *only* when the GROUP-level stop
    criterion fires while perf is sub-threshold, in which case the worker does
    NOT break - so the flag being True proves the group did not false-stop.

    ``percent_items_used`` (fraction of TOTAL trials in interleaved mode) is a
    secondary sanity check. It sits just below 1.0 rather than exactly 1.0
    because a few all-zero items converge at the ITEM level (the ``delta_item``
    mechanism, separate from the group gate) and are trimmed - expected, correct
    behaviour. A genuine GROUP stop would instead land near the first refresh
    where the width crosses delta_cap (~trial 120 -> ~0.67), so any value well
    above that (> 0.9) confirms no group stop occurred."""
    df = _near_zero_ordinal_df(n_items=15, epochs=12)  # 180 trials
    s = _run_posthoc_ordinal(
        df,
        _low_rate_params(reanalysis_interval=60),
        inference="modal",
        processing_order="epoch_interleaved",
    )
    assert s["low_perf_stop_suppressed"] is True
    assert float(s["percent_items_used"]) > 0.9, (
        f"interleaved grouping group-stopped early "
        f"(percent_items_used={s['percent_items_used']!r}); a group stop lands "
        f"near ~0.67, so this indicates the low-perf gate did not suppress"
    )
    # Option C telemetry: resolved at a very-low performance level.
    assert s["low_perf_floor"] is True


@pytest.mark.slow
def test_ordinal_low_perf_no_stop_modal_live_single():
    """Production path (bridge drives live_single). One all-epochs-present
    near-zero ordinal call (modal): the modal-width criterion is met but
    suppressed, so the grouping is NOT added to stop_this_grouping and the flag
    is stamped True into stabilization_history."""
    df = _near_zero_ordinal_df(n_items=15, epochs=12, grouping="ord_grp")  # 180 obs
    res = _run_live_single_ordinal(df, _low_rate_params(), inference="modal")
    stopped = res.get("grouping") in (res.get("stop_this_grouping") or [])
    assert not stopped, "live_single stopped a near-zero ordinal grouping (modal width route)"
    sh = res.get("stabilization_history", {})
    assert sh.get("low_perf_stop_suppressed") is True, (
        "live_single did not flag the suppressed ordinal modal width-route stop"
    )
    # Option C telemetry stamped into stabilization_history (routed to the bridge).
    assert sh.get("low_perf_floor") is True, (
        "live_single did not stamp low_perf_floor for the near-zero ordinal grouping"
    )


# ---------------------------------------------------------------------------
# 2. HYBRID (entropy-family demonstration). Slow: hybrid runs a per-item MCMC
#    fit, so the fixture uses few items (6) but many epochs (20 = 120 obs) so the
#    GROUP hybrid criterion reliably fires - at 120 obs Pathway 1's hierarchical
#    modal-narrow condition (0.5/sqrt(120)=0.046 < delta_cap) is met on the
#    confidently-peaked-at-0 distribution. The gate suppresses the hybrid stop.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_ordinal_low_perf_no_stop_hybrid_live_single():
    """live_single, near-zero ordinal (hybrid): the hybrid criterion fires
    (Pathway 1 and/or 2) at the confidently-peaked-at-0 distribution but is
    suppressed below low_performance_threshold."""
    df = _near_zero_ordinal_df(n_items=6, epochs=20, n_ones=1, grouping="ord_grp")  # 120 obs
    res = _run_live_single_ordinal(df, _low_rate_params(), inference="hybrid")
    stopped = res.get("grouping") in (res.get("stop_this_grouping") or [])
    assert not stopped, "live_single stopped a near-zero ordinal grouping (hybrid route)"
    sh = res.get("stabilization_history", {})
    assert sh.get("low_perf_stop_suppressed") is True, (
        "live_single did not flag the suppressed ordinal hybrid stop"
    )
    # Option C telemetry is mode-independent (set for hybrid too).
    assert sh.get("low_perf_floor") is True, (
        "live_single did not stamp low_perf_floor for the near-zero ordinal (hybrid) grouping"
    )


# ---------------------------------------------------------------------------
# 3. REGRESSION: a normal MID-performance ordinal grouping still stops (the gate
#    must not have broken ordinary stopping). Peaked at category 5 -> normalised
#    perf ~0.5 >> low_performance_threshold, so the modal-width stop fires and
#    the grouping stops before exhaustion.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_ordinal_midperf_still_stops_modal_itemgreedy():
    df = _mid_perf_ordinal_df(n_items=15, epochs=8, mode_cat=5)
    s = _run_posthoc_ordinal(df, _low_rate_params(), inference="modal")
    assert s["low_perf_stop_suppressed"] is False, (
        "a mid-performance grouping must not trigger the low-perf suppression"
    )
    assert float(s["percent_items_used"]) < 1.0, (
        f"mid-performance ordinal grouping did not stop early "
        f"(percent_items_used={s['percent_items_used']!r}); the gate may have "
        f"broken ordinary stopping"
    )
    # Option C telemetry: a mid-performance grouping is NOT at the low-perf floor.
    assert s["low_perf_floor"] is False, (
        "mid-performance ordinal grouping must not set low_perf_floor"
    )
