"""Tests for epoch-interleaved processing mode in optimal_stopping_posthoc."""

import numpy as np
import pandas as pd
import pytest
from optstop import optimal_stopping_posthoc
from optstop.rule import _init_item_state, _build_item_summary


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_binary_df(n_items=10, n_epochs=5, seed=42):
    """Create a synthetic binary DataFrame for testing."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_items):
        p = rng.uniform(0.3, 0.8)
        for e in range(1, n_epochs + 1):
            rows.append({
                'task': 'test_task',
                'sample_id': 'item_%d' % i,
                'epoch': e,
                'score': int(rng.random() < p),
            })
    return pd.DataFrame(rows)


def _make_continuous_df(n_items=10, n_epochs=5, seed=42):
    """Create a synthetic continuous [0,1] DataFrame for testing."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_items):
        mu = rng.uniform(0.3, 0.7)
        for e in range(1, n_epochs + 1):
            rows.append({
                'task': 'continuous_task',
                'sample_id': 'item_%d' % i,
                'epoch': e,
                'score': float(np.clip(rng.normal(mu, 0.1), 0, 1)),
            })
    return pd.DataFrame(rows)


def _make_ordinal_df(n_items=10, n_epochs=5, max_score=10, seed=42):
    """Create a synthetic ordinal DataFrame for testing."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_items):
        mode = rng.randint(3, 8)
        for e in range(1, n_epochs + 1):
            score = int(np.clip(rng.normal(mode, 1.5), 0, max_score))
            rows.append({
                'task': 'ordinal_task',
                'sample_id': 'item_%d' % i,
                'epoch': e,
                'score': score,
            })
    return pd.DataFrame(rows)


BASE_PARAMS = {
    'draws': 50,
    'tune': 50,
    'chains': 2,
    'cores': 1,
    'delta_item': 0.5,
    'delta_cap': 0.5,
    'stab_window': 2,
    'CI_delta': 0.01,
}

EXPECTED_SUMMARY_KEYS = {
    'grouping', 'n_items_used', 'theta_ci_low', 'theta_ci_high',
    'theta_ci_width', 'percent_items_used', 'avg_reps_per_item',
}


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestInitItemState:
    """Tests for _init_item_state helper."""

    def test_binary_state(self):
        state = _init_item_state('binary')
        assert state['successes'] == 0
        assert state['trials'] == 0
        assert state['ci_record'] == []
        assert state['ci_slopes_hist'] == []
        assert state['entropy_history'] == []
        assert state['accumulated_scores'] == []

    def test_continuous_state(self):
        state = _init_item_state('continuous_01')
        assert state['accumulated_scores'] == []
        assert state['successes'] == 0
        assert state['trials'] == 0

    def test_continuous_bounded_state(self):
        state = _init_item_state('continuous_bounded')
        assert state['accumulated_scores'] == []

    def test_ordinal_state(self):
        state = _init_item_state('ordinal')
        assert state['accumulated_scores'] == []


class TestBuildItemSummary:
    """Tests for _build_item_summary helper."""

    def test_binary_summary(self):
        state = _init_item_state('binary')
        state['successes'] = 7
        state['trials'] = 10
        summary = _build_item_summary(state, 'binary')
        assert summary['successes'] == 7
        assert summary['trials'] == 10

    def test_continuous_summary(self):
        state = _init_item_state('continuous_01')
        state['accumulated_scores'] = [0.3, 0.5, 0.7, 0.4]
        summary = _build_item_summary(state, 'continuous_01',
                                       cont_lower=0.0, cont_upper=1.0)
        assert 'scores_raw' in summary
        assert 'scores_normalized' in summary
        assert 'n_obs' in summary
        assert summary['n_obs'] == 4
        assert 'mean_raw' in summary
        assert 'mean_normalized' in summary

    def test_continuous_summary_equal_bounds(self):
        """Division-by-zero guard when cont_upper == cont_lower."""
        state = _init_item_state('continuous_bounded')
        state['accumulated_scores'] = [5.0, 5.0]
        summary = _build_item_summary(state, 'continuous_bounded',
                                       cont_lower=5.0, cont_upper=5.0)
        assert summary['n_obs'] == 2
        # Should not raise

    def test_ordinal_summary(self):
        state = _init_item_state('ordinal')
        state['accumulated_scores'] = [3, 5, 5, 7, 5]
        summary = _build_item_summary(state, 'ordinal', ordinal_max_score=10)
        assert 'counts' in summary
        assert 'n_obs' in summary
        assert summary['n_obs'] == 5
        assert 'modal_category' in summary
        assert summary['modal_category'] == 5
        assert 'mean_score' in summary
        assert 'scores' in summary


# ---------------------------------------------------------------------------
# Parameter validation tests
# ---------------------------------------------------------------------------

class TestParameterValidation:
    """Tests for processing_order and reanalysis_interval validation."""

    def test_invalid_processing_order(self):
        df = _make_binary_df(n_items=3, n_epochs=2)
        with pytest.raises(ValueError, match='processing_order'):
            optimal_stopping_posthoc(
                df, BASE_PARAMS,
                grouping_columns=['task'],
                sample_id_column='sample_id',
                epoch_column='epoch',
                processing_order='invalid_mode'
            )

    def test_invalid_reanalysis_interval_zero(self):
        df = _make_binary_df(n_items=3, n_epochs=2)
        with pytest.raises(ValueError, match='reanalysis_interval'):
            optimal_stopping_posthoc(
                df, BASE_PARAMS,
                grouping_columns=['task'],
                sample_id_column='sample_id',
                epoch_column='epoch',
                processing_order='epoch_interleaved',
                reanalysis_interval=0
            )

    def test_invalid_reanalysis_interval_negative(self):
        df = _make_binary_df(n_items=3, n_epochs=2)
        with pytest.raises(ValueError, match='reanalysis_interval'):
            optimal_stopping_posthoc(
                df, BASE_PARAMS,
                grouping_columns=['task'],
                sample_id_column='sample_id',
                epoch_column='epoch',
                processing_order='epoch_interleaved',
                reanalysis_interval=-5
            )


# ---------------------------------------------------------------------------
# Binary pathway tests
# ---------------------------------------------------------------------------

class TestEpochInterleavedBinary:
    """Test epoch-interleaved mode with binary scoring."""

    def test_basic_binary(self):
        """Epoch-interleaved binary produces valid results with correct structure."""
        df = _make_binary_df(n_items=10, n_epochs=5)
        pruned_df, summary = optimal_stopping_posthoc(
            df, BASE_PARAMS,
            grouping_columns=['task'],
            sample_id_column='sample_id',
            epoch_column='epoch',
            processing_order='epoch_interleaved',
            reanalysis_interval=5
        )
        assert isinstance(pruned_df, pd.DataFrame)
        assert isinstance(summary, list)
        assert len(summary) == 1
        s = summary[0]
        for key in EXPECTED_SUMMARY_KEYS:
            assert key in s, "Missing key: %s" % key
        assert s['n_items_used'] > 0
        assert s['n_items_used'] <= 10
        assert 0.0 <= s['theta_ci_low'] <= s['theta_ci_high'] <= 1.0
        assert s['theta_ci_width'] >= 0
        assert s['processing_order'] == 'epoch_interleaved'

    def test_binary_pruned_df_structure(self):
        """Pruned DataFrame retains same columns as input."""
        df = _make_binary_df(n_items=10, n_epochs=5)
        pruned_df, _ = optimal_stopping_posthoc(
            df, BASE_PARAMS,
            grouping_columns=['task'],
            sample_id_column='sample_id',
            epoch_column='epoch',
            processing_order='epoch_interleaved',
            reanalysis_interval=5
        )
        assert set(pruned_df.columns) == set(df.columns)
        assert len(pruned_df) <= len(df)
        assert len(pruned_df) > 0


# ---------------------------------------------------------------------------
# Continuous pathway tests
# ---------------------------------------------------------------------------

class TestEpochInterleavedContinuous:
    """Test epoch-interleaved mode with continuous scoring."""

    def test_basic_continuous(self):
        """Epoch-interleaved continuous produces valid results."""
        df = _make_continuous_df(n_items=10, n_epochs=5)
        pruned_df, summary = optimal_stopping_posthoc(
            df, BASE_PARAMS,
            grouping_columns=['task'],
            sample_id_column='sample_id',
            epoch_column='epoch',
            continuous_tasks=['continuous_task'],
            processing_order='epoch_interleaved',
            reanalysis_interval=5
        )
        assert isinstance(pruned_df, pd.DataFrame)
        assert len(summary) == 1
        s = summary[0]
        for key in EXPECTED_SUMMARY_KEYS:
            assert key in s, "Missing key: %s" % key
        assert s['n_items_used'] > 0
        assert s['processing_order'] == 'epoch_interleaved'


# ---------------------------------------------------------------------------
# Ordinal pathway tests
# ---------------------------------------------------------------------------

class TestEpochInterleavedOrdinal:
    """Test epoch-interleaved mode with ordinal scoring."""

    def test_ordinal_hybrid(self):
        """Epoch-interleaved ordinal hybrid produces valid results."""
        df = _make_ordinal_df(n_items=10, n_epochs=5, max_score=10)
        pruned_df, summary = optimal_stopping_posthoc(
            df, BASE_PARAMS,
            grouping_columns=['task'],
            sample_id_column='sample_id',
            epoch_column='epoch',
            ordinal_tasks=['ordinal_task'],
            ordinal_max_score=10,
            ordinal_inference='hybrid',
            processing_order='epoch_interleaved',
            reanalysis_interval=5
        )
        assert isinstance(pruned_df, pd.DataFrame)
        assert len(summary) == 1
        s = summary[0]
        for key in EXPECTED_SUMMARY_KEYS:
            assert key in s, "Missing key: %s" % key
        assert s['n_items_used'] > 0
        assert s['processing_order'] == 'epoch_interleaved'

    def test_ordinal_modal(self):
        """Epoch-interleaved ordinal modal produces valid results."""
        df = _make_ordinal_df(n_items=10, n_epochs=5, max_score=10)
        pruned_df, summary = optimal_stopping_posthoc(
            df, BASE_PARAMS,
            grouping_columns=['task'],
            sample_id_column='sample_id',
            epoch_column='epoch',
            ordinal_tasks=['ordinal_task'],
            ordinal_max_score=10,
            ordinal_inference='modal',
            processing_order='epoch_interleaved',
            reanalysis_interval=5
        )
        assert isinstance(pruned_df, pd.DataFrame)
        assert len(summary) == 1
        s = summary[0]
        assert s['n_items_used'] > 0
        assert s['processing_order'] == 'epoch_interleaved'


# ---------------------------------------------------------------------------
# Regression: item-greedy unchanged
# ---------------------------------------------------------------------------

class TestItemGreedyRegression:
    """Verify item-greedy (default) behaviour is unchanged."""

    def test_default_is_item_greedy(self):
        """Default processing_order is item_greedy, no processing_order in summary."""
        df = _make_binary_df(n_items=6, n_epochs=3)
        _, summary = optimal_stopping_posthoc(
            df, BASE_PARAMS,
            grouping_columns=['task'],
            sample_id_column='sample_id',
            epoch_column='epoch',
        )
        assert len(summary) == 1
        # item_greedy results should NOT have processing_order in summary
        # (backward compat: field only present for epoch_interleaved)
        # But if it does appear, it should be item_greedy
        if 'processing_order' in summary[0]:
            assert summary[0]['processing_order'] == 'item_greedy'

    def test_explicit_item_greedy(self):
        """Explicitly setting item_greedy produces valid results."""
        df = _make_binary_df(n_items=6, n_epochs=3)
        pruned_df, summary = optimal_stopping_posthoc(
            df, BASE_PARAMS,
            grouping_columns=['task'],
            sample_id_column='sample_id',
            epoch_column='epoch',
            processing_order='item_greedy'
        )
        assert isinstance(pruned_df, pd.DataFrame)
        assert len(summary) == 1
        s = summary[0]
        for key in EXPECTED_SUMMARY_KEYS:
            assert key in s


# ---------------------------------------------------------------------------
# Multiple groupings
# ---------------------------------------------------------------------------

class TestMultipleGroupings:
    """Test epoch-interleaved with multiple groupings processed in parallel."""

    def test_two_groupings_binary(self):
        """Two binary groupings both run epoch-interleaved correctly."""
        rng = np.random.RandomState(42)
        rows = []
        for task in ['task_a', 'task_b']:
            for i in range(8):
                p = rng.uniform(0.3, 0.8)
                for e in range(1, 4):
                    rows.append({
                        'task': task,
                        'sample_id': 'item_%d' % i,
                        'epoch': e,
                        'score': int(rng.random() < p),
                    })
        df = pd.DataFrame(rows)

        pruned_df, summary = optimal_stopping_posthoc(
            df, BASE_PARAMS,
            grouping_columns=['task'],
            sample_id_column='sample_id',
            epoch_column='epoch',
            processing_order='epoch_interleaved',
            reanalysis_interval=5
        )
        assert len(summary) == 2
        # Grouping values are pid tuples (category codes from groupby)
        # Just verify we got two distinct groupings with valid results
        groupings = [s['grouping'] for s in summary]
        assert len(set(str(g) for g in groupings)) == 2
        for s in summary:
            assert s['n_items_used'] > 0
            assert s['processing_order'] == 'epoch_interleaved'


# ---------------------------------------------------------------------------
# Reanalysis interval behaviour
# ---------------------------------------------------------------------------

class TestReanalysisInterval:
    """Test that reanalysis_interval parameter is respected."""

    def test_large_reanalysis_interval(self):
        """With reanalysis_interval larger than total trials, only final refresh runs."""
        df = _make_binary_df(n_items=6, n_epochs=3)
        # 6 items * 3 epochs = 18 trials, interval=100 means only last-trial refresh
        pruned_df, summary = optimal_stopping_posthoc(
            df, BASE_PARAMS,
            grouping_columns=['task'],
            sample_id_column='sample_id',
            epoch_column='epoch',
            processing_order='epoch_interleaved',
            reanalysis_interval=100
        )
        assert isinstance(pruned_df, pd.DataFrame)
        assert len(summary) == 1
        # With only 1 group refresh, unlikely to have stopped early
        # All items should have been used
        assert summary[0]['n_items_used'] == 6

    def test_small_reanalysis_interval(self):
        """Small reanalysis_interval produces valid results (more group refreshes)."""
        df = _make_binary_df(n_items=10, n_epochs=5)
        pruned_df, summary = optimal_stopping_posthoc(
            df, BASE_PARAMS,
            grouping_columns=['task'],
            sample_id_column='sample_id',
            epoch_column='epoch',
            processing_order='epoch_interleaved',
            reanalysis_interval=2
        )
        assert isinstance(pruned_df, pd.DataFrame)
        assert len(summary) == 1
        assert summary[0]['n_items_used'] > 0
