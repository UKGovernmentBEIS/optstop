"""Sorted-input warning: detection of natural ID order, and visibility from the CLI.

optimal_stopping_posthoc warns when a grouping's sample IDs appear in sorted order and shuffle_items
is off: with the default item-greedy processing the rule then reads a non-random prefix of the items
first and can stop on it. Two gaps are covered here:

  * the order check compared IDs as strings only, so IDs numbered 0, 1, 2, ..., 10 or
    item_1, item_2, ..., item_10 (not sorted as strings, "10" < "2") were never flagged;
  * the CLI logs to a file only, so even when the warning fired it never reached the console.

No MCMC runs here. The real optimal_stopping_posthoc is stopped right after its input-order check by
making _validate_params, the next call, raise. Nothing new is imported at module level, and the CLI is
imported inside the CLI helper (as in test_cli_fail_loud.py), so on a tree without either part of the
fix these tests fail on behaviour, not on an import error.
"""

import logging
import sys
from unittest.mock import patch

import pandas as pd
import pytest

from optstop import rule

SORTED_MESSAGE = "appear to be sorted"

NATURAL = {
    "integers": list(range(12)),
    "prefixed_unpadded": [f"item_{i}" for i in range(12)],
    "integers_descending": list(range(11, -1, -1)),
}
LEXICOGRAPHIC = {
    "prefixed_padded": [f"item_{i:02d}" for i in range(12)],
}
UNSORTED = {
    "integers_shuffled": [3, 7, 0, 11, 5, 1, 9, 2, 10, 4, 8, 6],
    "prefixed_shuffled": [f"item_{i}" for i in (4, 0, 9, 2, 11, 7, 1, 10, 3, 8, 6, 5)],
}
SORTED = {**NATURAL, **LEXICOGRAPHIC}


class _StopAfterOrderCheck(Exception):
    """Raised in place of _validate_params, the call that follows the input-order check."""


def _stop(*args, **kwargs):
    raise _StopAfterOrderCheck()


def _frame(ids, epochs=2):
    return pd.DataFrame(
        [("m", "t", sid, e, (i + e) % 2) for i, sid in enumerate(ids) for e in range(epochs)],
        columns=["model", "task", "sample_id", "epoch", "score"],
    )


def _sorted_warnings(caplog):
    return sum(SORTED_MESSAGE in record.getMessage() for record in caplog.records)


def _run_order_check(df, **kwargs):
    with patch.object(rule, "_validate_params", _stop):
        with pytest.raises(_StopAfterOrderCheck):
            rule.optimal_stopping_posthoc(
                df, {}, ["model", "task"], "sample_id", "epoch", "score", **kwargs
            )


# --------------------------------------------------------------------------- #
# Detection (library level)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("ids", list(SORTED.values()), ids=list(SORTED))
def test_sorted_ids_are_flagged(ids, caplog):
    caplog.set_level(logging.WARNING)
    _run_order_check(_frame(ids))
    assert _sorted_warnings(caplog) == 1


@pytest.mark.parametrize("ids", list(UNSORTED.values()), ids=list(UNSORTED))
def test_unsorted_ids_are_not_flagged(ids, caplog):
    caplog.set_level(logging.WARNING)
    _run_order_check(_frame(ids))
    assert _sorted_warnings(caplog) == 0


def test_five_or_fewer_ids_are_not_judged(caplog):
    caplog.set_level(logging.WARNING)
    _run_order_check(_frame(list(range(5))))
    assert _sorted_warnings(caplog) == 0


def test_shuffle_items_skips_the_check(caplog):
    caplog.set_level(logging.WARNING)
    _run_order_check(_frame(list(range(12))), shuffle_items=True, shuffle_seed=0)
    assert _sorted_warnings(caplog) == 0


# --------------------------------------------------------------------------- #
# Visibility (CLI)
# --------------------------------------------------------------------------- #

def _run_cli(workdir, ids, replacement=_stop):
    """Run optstop-posthoc on a small CSV, stopped after the input-order check.
    Returns the text of the log file the CLI wrote."""
    from optstop import cli

    csv = workdir / "input.csv"
    _frame(ids).to_csv(csv, index=False)
    log = workdir / "cli.log"
    argv = [
        "optstop-posthoc",
        "--csv", str(csv),
        "--output", str(workdir / "pruned.csv"),
        "--log", str(log),
        "--grouping_columns", "model,task",
        "--sample_id_column", "sample_id",
        "--epoch_column", "epoch",
        "--no_progress",
    ]
    try:
        with patch.object(rule, "_validate_params", replacement), patch.object(sys, "argv", argv):
            with pytest.raises(_StopAfterOrderCheck):
                cli.main()
    finally:
        cli.close_all_log_handlers()
    return log.read_text()


@pytest.mark.parametrize(
    "ids",
    [LEXICOGRAPHIC["prefixed_padded"], NATURAL["integers"]],
    ids=["prefixed_padded", "integers"],
)
def test_cli_prints_sorted_input_warning_on_stderr(tmp_path, capsys, ids):
    log_text = _run_cli(tmp_path, ids)
    err = capsys.readouterr().err
    assert err.count(SORTED_MESSAGE) == 1, f"expected the warning once on stderr, got: {err!r}"
    assert SORTED_MESSAGE in log_text, "the warning must still be written to the log file"


def test_cli_other_warnings_stay_in_log_file(tmp_path, capsys):
    def _other_warning_then_stop(*args, **kwargs):
        logging.getLogger("optstop.posthoc").warning("unrelated warning")
        raise _StopAfterOrderCheck()

    log_text = _run_cli(tmp_path, UNSORTED["integers_shuffled"], _other_warning_then_stop)
    err = capsys.readouterr().err
    assert "unrelated warning" not in err
    assert "unrelated warning" in log_text


def test_cli_repeated_runs_do_not_stack_handlers(tmp_path, capsys):
    for run in range(2):
        workdir = tmp_path / f"run{run}"
        workdir.mkdir()
        _run_cli(workdir, NATURAL["integers"])
        err = capsys.readouterr().err
        assert err.count(SORTED_MESSAGE) == 1, f"run {run}: {err!r}"
