"""Issue #4 fail-loud CLI policy tests.

Issue #4: "A run that computes nothing exits 0 and reports its files saved - the
failure is visible only inside a CSV column." When grouping-level exceptions are
caught and stored in the summary's ``error`` column, the CLI previously still
printed "saved" and exited 0, so a run that computed nothing looked successful.

``optstop.cli.main`` now:
  * exits non-zero when EVERY grouping failed (the run computed nothing);
  * exits non-zero when NO groupings were produced at all;
  * exits 0 but prints a loud stderr summary on PARTIAL failure;
  * exits 0 silently (no failure text) when all groupings succeeded.

These tests patch the heavy ``optimal_stopping_posthoc`` with a stub returning a
controlled ``(pruned_df, summary)`` so they are fast and deterministic (no MCMC);
the CSV is still read from disk first, so a minimal real input file is provided.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


def _mini_csv(tmp_path):
    csv = tmp_path / "mini.csv"
    pd.DataFrame(
        {
            "grouping_num": [1, 1, 2, 2],
            "task_num": [1, 1, 1, 1],
            "sample_id_num": [1, 2, 1, 2],
            "epoch": [1, 1, 1, 1],
            "score": [1, 0, 1, 0],
        }
    ).to_csv(csv, index=False)
    return csv


def _argv(csv, out, summary):
    return [
        "optstop-posthoc",
        "--csv", str(csv),
        "--output", str(out),
        "--summary", str(summary),
        "--log", str(csv.parent / "cli.log"),
        "--grouping_columns", "grouping_num,task_num",
        "--sample_id_column", "sample_id_num",
        "--epoch_column", "epoch",
    ]


def _run_main_with_summary(tmp_path, summary, pruned_rows=0):
    """Invoke cli.main() with optimal_stopping_posthoc stubbed to return the
    given summary. Returns (exit_code, stderr_text). exit_code is None if main()
    returned normally (i.e. exit 0)."""
    from optstop import cli

    csv = _mini_csv(tmp_path)
    out = tmp_path / "pruned.csv"
    summ = tmp_path / "summary.csv"

    pruned_df = pd.DataFrame(
        {"grouping_num": [], "task_num": [], "sample_id_num": [], "epoch": [], "score": []}
    )

    def _stub(df, params, *a, **k):
        return pruned_df, summary

    exit_code = None
    with patch.object(cli, "optimal_stopping_posthoc", _stub), \
         patch.object(sys, "argv", _argv(csv, out, summ)):
        try:
            cli.main()
        except SystemExit as e:
            exit_code = e.code if e.code is not None else 0
    return exit_code, out, summ


def test_all_failed_exits_nonzero(tmp_path, capsys):
    summary = [
        {"grouping": "1-1", "error": "hdi got an unexpected keyword argument: 'hdi_prob'"},
        {"grouping": "2-1", "error": "hdi got an unexpected keyword argument: 'hdi_prob'"},
    ]
    exit_code, out, summ = _run_main_with_summary(tmp_path, summary)
    err = capsys.readouterr().err
    assert exit_code == 1, f"expected non-zero exit when all groupings failed, got {exit_code}"
    assert "all 2 grouping(s) failed" in err
    assert "hdi_prob" in err
    # files are still written (they carry the diagnostic 'error' column)
    assert out.exists() and summ.exists()


def test_no_groupings_exits_nonzero(tmp_path, capsys):
    exit_code, _, _ = _run_main_with_summary(tmp_path, [])
    err = capsys.readouterr().err
    assert exit_code == 1, f"expected non-zero exit when no groupings produced, got {exit_code}"
    assert "no groupings were produced" in err


def test_partial_failure_exits_zero_but_loud(tmp_path, capsys):
    summary = [
        {"grouping": "1-1", "error": None, "theta_ci_low": 0.1, "theta_ci_high": 0.4},
        {"grouping": "2-1", "error": "MCMC sampling failed"},
    ]
    exit_code, _, _ = _run_main_with_summary(tmp_path, summary)
    err = capsys.readouterr().err
    assert exit_code is None, f"partial failure must exit 0, got exit_code={exit_code}"
    assert "1 of 2 grouping(s) failed" in err
    assert "MCMC sampling failed" in err


def test_all_success_is_silent_and_exits_zero(tmp_path, capsys):
    summary = [
        {"grouping": "1-1", "error": None, "theta_ci_low": 0.1, "theta_ci_high": 0.4},
        {"grouping": "2-1", "error": None, "theta_ci_low": 0.2, "theta_ci_high": 0.5},
    ]
    exit_code, out, summ = _run_main_with_summary(tmp_path, summary)
    captured = capsys.readouterr()
    assert exit_code is None, f"all-success must exit 0, got exit_code={exit_code}"
    # no failure/warning text on stderr
    assert "failed" not in captured.err
    assert "ERROR" not in captured.err
    # the reassuring stdout messages still appear
    assert "Pruned data saved to" in captured.out


# --------------------------------------------------------------------------- #
# optstop-convergence (main_convergence): summary is a DataFrame with an
# 'error' column; the same fail-loud policy applies.
# --------------------------------------------------------------------------- #

def _run_convergence_with_result(tmp_path, result_df):
    """Invoke cli.main_convergence() with convergence_posthoc stubbed to return
    the given DataFrame. Returns (exit_code, output_path)."""
    from optstop import cli

    csv = _mini_csv(tmp_path)
    out = tmp_path / "convergence.csv"
    argv = [
        "optstop-convergence",
        "--csv", str(csv),
        "--output", str(out),
        "--log", str(csv.parent / "conv.log"),
        "--grouping_columns", "grouping_num,task_num",
        "--sample_id_column", "sample_id_num",
        "--epoch_column", "epoch",
    ]

    def _stub(df, params, *a, **k):
        return result_df

    exit_code = None
    with patch.object(cli, "convergence_posthoc", _stub), \
         patch.object(sys, "argv", argv):
        try:
            cli.main_convergence()
        except SystemExit as e:
            exit_code = e.code if e.code is not None else 0
    return exit_code, out


def test_convergence_all_failed_exits_nonzero(tmp_path, capsys):
    result_df = pd.DataFrame(
        {
            "grouping": ["1-1", "2-1"],
            "error": ["MCMC sampling failed", "MCMC sampling failed"],
        }
    )
    exit_code, out = _run_convergence_with_result(tmp_path, result_df)
    err = capsys.readouterr().err
    assert exit_code == 1, f"expected non-zero exit when all groupings failed, got {exit_code}"
    assert "all 2 grouping(s) failed" in err
    assert "MCMC sampling failed" in err
    assert out.exists()


def test_convergence_no_groupings_exits_nonzero(tmp_path, capsys):
    result_df = pd.DataFrame({"grouping": [], "error": []})
    exit_code, _ = _run_convergence_with_result(tmp_path, result_df)
    err = capsys.readouterr().err
    assert exit_code == 1, f"expected non-zero exit when no groupings produced, got {exit_code}"
    assert "no groupings were produced" in err


def test_convergence_partial_failure_exits_zero_but_loud(tmp_path, capsys):
    result_df = pd.DataFrame(
        {
            "grouping": ["1-1", "2-1"],
            "error": [None, "MCMC sampling failed"],
        }
    )
    exit_code, _ = _run_convergence_with_result(tmp_path, result_df)
    err = capsys.readouterr().err
    assert exit_code is None, f"partial failure must exit 0, got exit_code={exit_code}"
    assert "1 of 2 grouping(s) failed" in err
    assert "MCMC sampling failed" in err


def test_convergence_all_success_is_silent_and_exits_zero(tmp_path, capsys):
    result_df = pd.DataFrame(
        {
            "grouping": ["1-1", "2-1"],
            "error": [None, None],
            "converged": [True, True],
        }
    )
    exit_code, out = _run_convergence_with_result(tmp_path, result_df)
    captured = capsys.readouterr()
    assert exit_code is None, f"all-success must exit 0, got exit_code={exit_code}"
    assert "failed" not in captured.err
    assert "ERROR" not in captured.err
    assert "Convergence stats saved to" in captured.out


# --------------------------------------------------------------------------- #
# optstop-live (main_live): optimal_stopping_live returns a dict carrying
# n_total / n_failed plus failed_groupings (a list of {grouping, error} dicts;
# grouping is None for a dead worker process whose name is unrecoverable). The
# live gate keys on the counts and reports the per-grouping (name, error) like
# the posthoc/convergence paths.
# --------------------------------------------------------------------------- #

def _run_live_with_result(tmp_path, result):
    """Invoke cli.main_live() with optimal_stopping_live stubbed to return the
    given result dict. Returns (exit_code)."""
    from optstop import cli

    csv = _mini_csv(tmp_path)
    argv = [
        "optstop-live",
        "--csv", str(csv),
        "--log", str(csv.parent / "live.log"),
        "--grouping_columns", "grouping_num,task_num",
        "--sample_id_column", "sample_id_num",
        "--epoch_column", "epoch",
        "--no_progress",
    ]

    def _stub(df, params, *a, **k):
        return result

    exit_code = None
    with patch.object(cli, "optimal_stopping_live", _stub), \
         patch.object(sys, "argv", argv):
        try:
            cli.main_live()
        except SystemExit as e:
            exit_code = e.code if e.code is not None else 0
    return exit_code


def test_live_all_failed_exits_nonzero(tmp_path, capsys):
    # Two failures: one internal (name known) and one dead worker (name None -> '?').
    result = {
        "stop_sample_ids": [], "stop_task": [], "n_total": 2, "n_failed": 2,
        "failed_groupings": [
            {"grouping": "1-1", "error": "MCMC sampling failed"},
            {"grouping": None, "error": "worker task failed: boom"},
        ],
    }
    exit_code = _run_live_with_result(tmp_path, result)
    err = capsys.readouterr().err
    assert exit_code == 1, f"expected non-zero exit when all groupings failed, got {exit_code}"
    assert "all 2 grouping(s) failed" in err
    assert "log file" in err
    # Per-grouping (name, error) surfaced; unnamed worker death shown as '?'.
    assert "grouping 1-1: MCMC sampling failed" in err
    assert "grouping ?: worker task failed: boom" in err


def test_live_no_groupings_exits_nonzero(tmp_path, capsys):
    result = {"stop_sample_ids": [], "stop_task": [], "n_total": 0, "n_failed": 0}
    exit_code = _run_live_with_result(tmp_path, result)
    err = capsys.readouterr().err
    assert exit_code == 1, f"expected non-zero exit when no groupings produced, got {exit_code}"
    assert "no groupings were produced" in err


def test_live_partial_failure_exits_zero_but_loud(tmp_path, capsys):
    result = {
        "stop_sample_ids": ["1-1_1"], "stop_task": [], "n_total": 2, "n_failed": 1,
        "failed_groupings": [{"grouping": "2-2", "error": "all-NaN scores"}],
    }
    exit_code = _run_live_with_result(tmp_path, result)
    err = capsys.readouterr().err
    assert exit_code is None, f"partial failure must exit 0, got exit_code={exit_code}"
    assert "1 of 2 grouping(s) failed" in err
    assert "grouping 2-2: all-NaN scores" in err


def test_live_all_success_is_silent_and_exits_zero(tmp_path, capsys):
    result = {"stop_sample_ids": [], "stop_task": ["1-1"], "n_total": 2, "n_failed": 0}
    exit_code = _run_live_with_result(tmp_path, result)
    captured = capsys.readouterr()
    assert exit_code is None, f"all-success must exit 0, got exit_code={exit_code}"
    assert "failed" not in captured.err
    assert "ERROR" not in captured.err
    assert "Sample IDs to stop:" in captured.out
