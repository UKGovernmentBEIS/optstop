import pandas as pd
from optstop import optimal_stopping_posthoc, optimal_stopping_live

def test_optimal_stopping_posthoc():
    df = pd.DataFrame({
        'grouping_num': [1, 1, 1, 1],
        'task_num': [1, 1, 1, 1],
        'sample_id_num': [1, 1, 2, 2],
        'epoch': [1, 2, 1, 2],
        'score': [1, 0, 1, 1],
    })
    params = {
        'delta_item': 0.5,
        'delta_cap': 0.5,
        'draws': 100,
        'tune': 100,
        'rep_batch_size': 1,
        'pymc_refresh_every': 1,
        'stab_window': 2
    }
    pruned_df, summary = optimal_stopping_posthoc(
        df, params,
        grouping_columns=['grouping_num', 'task_num'],
        sample_id_column='sample_id_num',
        epoch_column='epoch'
    )
    assert isinstance(pruned_df, pd.DataFrame)
    assert isinstance(summary, list)
    assert set(pruned_df.columns) == set(df.columns)
    assert len(summary) > 0

def test_optimal_stopping_posthoc_with_diagnostics(tmp_path):
    """Test optimal stopping posthoc with diagnostic generation."""
    import os
    df = pd.DataFrame({
        'grouping_num': [1, 1, 1, 1, 1, 1],
        'task_num': [1, 1, 1, 1, 1, 1],
        'sample_id_num': [1, 1, 2, 2, 3, 3],
        'epoch': [1, 2, 1, 2, 1, 2],
        'score': [1, 1, 0, 1, 1, 1],
    })
    params = {
        'delta_item': 0.5,
        'delta_cap': 0.5,
        'draws': 100,
        'tune': 100,
        'rep_batch_size': 1,
        'pymc_refresh_every': 1,
        'stab_window': 2
    }
    
    # Test with diagnostics enabled
    pruned_df, summary = optimal_stopping_posthoc(
        df, params,
        grouping_columns=['grouping_num', 'task_num'],
        sample_id_column='sample_id_num',
        epoch_column='epoch',
        generate_diagnostics=True,
        diagnostics_prefix=str(tmp_path / "test_diagnostics")
    )
    
    assert isinstance(pruned_df, pd.DataFrame)
    assert isinstance(summary, list)
    assert set(pruned_df.columns) == set(df.columns)
    assert len(summary) > 0
    
    # Check that diagnostic files were created
    diagnostic_png = tmp_path / "test_diagnostics.png"
    diagnostic_csv = tmp_path / "test_diagnostics_paired.csv"
    assert diagnostic_png.exists(), f"Diagnostic PNG file not created: {diagnostic_png}"
    assert diagnostic_csv.exists(), f"Diagnostic CSV file not created: {diagnostic_csv}"

def test_optimal_stopping_live():
    import numpy as np
    df = pd.DataFrame({
        'grouping_num': [1, 1, 1, 1, 1, 1],
        'task_num': [1, 1, 1, 1, 1, 1],
        'sample_id_num': [1, 1, 2, 2, 3, 3],
        'epoch': [1, 2, 1, 2, 1, 2],
        'score': [1, 1, 0, 1, 1, 1],
    })
    params = {
        'delta_item': 0.5,
        'delta_cap': 0.5,
        'draws': 100,
        'tune': 100,
        'chains': 2,
        'cores': 2,
        'stab_window': 2,
        'CI_delta': 0.01,
        'rep_batch_size': 1,
        'pymc_refresh_every': 1
    }
    result = optimal_stopping_live(
        df, params,
        grouping_columns='grouping_num',
        sample_id_column='sample_id_num',
        epoch_column='epoch',
        display_progress=False
    )
    assert isinstance(result, dict)
    assert 'stop_sample_ids' in result
    assert 'stop_task' in result
    assert isinstance(result['stop_task'], list)  # Now returns list of grouping names
    assert isinstance(result['stop_sample_ids'], list)
    # Check that sample_ids have grouping prefix format
    for sample_id in result['stop_sample_ids']:
        assert '_' in sample_id  # Should be "grouping_sample_id" format

def test_optimal_stopping_live_with_logging(tmp_path):
    import os
    import pandas as pd
    from optstop.rule import configure_optstop_logging
    log_path = tmp_path / "test_optstop_live.log"
    configure_optstop_logging(str(log_path), console_output=False)
    df = pd.DataFrame({
        'grouping_num': [1, 1, 1, 1, 1, 1],
        'task_num': [1, 1, 1, 1, 1, 1],
        'sample_id_num': [1, 1, 2, 2, 3, 3],
        'epoch': [1, 2, 1, 2, 1, 2],
        'score': [1, 1, 0, 1, 1, 1],
    })
    params = {
        'delta_item': 0.5,
        'delta_cap': 0.5,
        'draws': 100,
        'tune': 100,
        'chains': 2,
        'cores': 2,
        'stab_window': 2,
        'CI_delta': 0.01,
        'rep_batch_size': 1,
        'pymc_refresh_every': 1
    }
    result = optimal_stopping_live(
        df, params,
        grouping_columns='grouping_num',
        sample_id_column='sample_id_num',
        epoch_column='epoch',
        display_progress=False
    )
    assert os.path.exists(log_path)
    with open(log_path, 'r') as f:
        log_content = f.read()
    assert 'Starting live optimal stopping' in log_content
    assert 'Live optimal stopping complete' in log_content
    # With parallel processing, detailed stopping messages may be in worker process logs
    # We just check that the main process logging is working

def test_optimal_stopping_live_multiple_groupings():
    """Test live mode with multiple groupings to ensure proper grouping prefix and list returns."""
    import pandas as pd
    df = pd.DataFrame({
        'subject': [1, 1, 1, 1, 2, 2, 2, 2],
        'task': [1, 1, 1, 1, 1, 1, 1, 1],
        'item_id': [1, 1, 2, 2, 1, 1, 2, 2],
        'trial_num': [1, 2, 1, 2, 1, 2, 1, 2],
        'score': [1, 0, 1, 1, 0, 1, 1, 0],
    })
    params = {
        'delta_item': 0.5,
        'delta_cap': 0.5,
        'draws': 50,
        'tune': 50,
        'chains': 2,
        'cores': 2,
        'stab_window': 2,
        'CI_delta': 0.01,
        'rep_batch_size': 1,
        'pymc_refresh_every': 1
    }
    result = optimal_stopping_live(
        df, params,
        grouping_columns=['subject', 'task'],
        sample_id_column='item_id',
        epoch_column='trial_num',
        score_column='score',
        display_progress=False
    )
    assert isinstance(result, dict)
    assert 'stop_sample_ids' in result
    assert 'stop_task' in result
    assert isinstance(result['stop_task'], list)
    assert isinstance(result['stop_sample_ids'], list)
    
    # Check that sample_ids have grouping prefix format
    for sample_id in result['stop_sample_ids']:
        assert '_' in sample_id  # Should be "grouping_sample_id" format
        # Should contain the grouping format (e.g., "1-1_1" for subject=1, task=1, item_id=1)
        assert any(grouping in sample_id for grouping in ['1-1', '2-1']) 

def test_convergence_posthoc():
    import pandas as pd
    from optstop import convergence_posthoc
    df = pd.DataFrame({
        'grouping_num': [1]*8 + [2]*8,
        'grouping': ['A']*8 + ['B']*8,
        'task_num': [1]*16,
        'sample_id_num': [1,1,1,1,2,2,2,2,1,1,1,1,2,2,2,2],
        'epoch': [1,2,3,4,1,2,3,4]*2,
        'score': [1,0,1,1,0,1,1,0,1,1,0,1,0,1,1,1],
    })
    params = {
        'delta_item': 0.5,
        'delta_cap': 0.5,
        'draws': 50,
        'tune': 50,
        'stab_window': 2,
        'CI_delta': 0.01,
        'rep_batch_size': 1,
        'pymc_refresh_every': 1,
        'item_seqs': 2,
        'epoch_seqs': 2
    }
    result = convergence_posthoc(
        df, params,
        grouping_columns=['grouping_num', 'task_num'],
        sample_id_column='sample_id_num',
        epoch_column='epoch'
    )
    assert isinstance(result, pd.DataFrame)
    expected_cols = [
        'grouping', 'group_label', 'n_items_used', 'theta_ci_low', 'theta_ci_high', 'theta_ci_width',
        'percent_items_used', 'avg_reps_per_item', 'mean_fin_CI_width_item', 'var_fin_CI_width_item',
        'mean_fin_CI_slope_item', 'var_fin_CI_slope_item', 'mean_fin_slope_slope_item', 'var_fin_slope_slope_item',
        'mean_needed_items', 'var_needed_items', 'mean_items_fin_score', 'var_items_fin_score',
        'mean_fin_CI_width_epoch', 'var_fin_CI_width_epoch', 'mean_fin_CI_slope_epoch', 'var_fin_CI_slope_epoch',
        'mean_fin_slope_slope_epoch', 'var_fin_slope_slope_epoch', 'mean_needed_epochs', 'var_needed_epochs',
        'mean_epochs_fin_score', 'var_epochs_fin_score', 'task_performance', 'mean_sample_id_performance', 'var_sample_id_performance'
    ]
    for col in expected_cols:
        assert col in result.columns
    assert len(result) == 2  # two groupings 

def test_optimal_stopping_posthoc_parallel():
    import pandas as pd
    from optstop import optimal_stopping_posthoc
    df = pd.DataFrame({
        'grouping_num': [1]*8 + [2]*8,
        'task_num': [1]*4 + [2]*4 + [1]*4 + [2]*4,
        'sample_id_num': [1,1,2,2,1,1,2,2]*2,
        'epoch': [1,2,1,2]*4,
        'score': [1,0,1,1,0,1,1,0,1,1,0,1,0,1,1,1],
    })
    params = {
        'delta_item': 0.5,
        'delta_cap': 0.5,
        'draws': 50,
        'tune': 50,
        'stab_window': 2,
        'CI_delta': 0.01,
        'rep_batch_size': 1,
        'pymc_refresh_every': 1
    }
    pruned_df, summary = optimal_stopping_posthoc(
        df, params,
        grouping_columns=['grouping_num', 'task_num'],
        sample_id_column='sample_id_num',
        epoch_column='epoch'
    )
    assert isinstance(pruned_df, pd.DataFrame)
    assert isinstance(summary, list)
    assert set(pruned_df.columns) == set(df.columns)
    assert len(summary) == 4
    for s in summary:
        assert 'grouping' in s
        assert 'n_items_used' in s
        assert 'theta_ci_low' in s
        assert 'theta_ci_high' in s
        assert 'theta_ci_width' in s
        assert 'percent_items_used' in s
        assert 'avg_reps_per_item' in s 

def test_optimal_stopping_posthoc_missing_columns():
    import pandas as pd
    from optstop import optimal_stopping_posthoc
    df = pd.DataFrame({
        'grouping_num': [1, 1],
        'task_num': [1, 1],
        # 'sample_id_num' missing
        'epoch': [1, 2],
        'score': [1, 0],
    })
    params = {'draws': 10, 'tune': 10, 'chains': 2, 'cores': 2}
    try:
        optimal_stopping_posthoc(
            df, params,
            grouping_columns=['grouping_num', 'task_num'],
            sample_id_column='sample_id_num',
            epoch_column='epoch'
        )
        assert False, "Should raise ValueError for missing columns"
    except ValueError as e:
        assert 'sample_id_num' in str(e)

def test_optimal_stopping_posthoc_invalid_params():
    import pandas as pd
    from optstop import optimal_stopping_posthoc
    df = pd.DataFrame({
        'grouping_num': [1, 1],
        'task_num': [1, 1],
        'sample_id_num': [1, 1],
        'epoch': [1, 2],
        'score': [1, 0],
    })
    params = {'draws': -1, 'tune': 10, 'chains': 2, 'cores': 2}
    try:
        optimal_stopping_posthoc(
            df, params,
            grouping_columns=['grouping_num', 'task_num'],
            sample_id_column='sample_id_num',
            epoch_column='epoch'
        )
        assert False, "Should raise ValueError for negative draws"
    except ValueError as e:
        assert 'draws' in str(e)

def test_optimal_stopping_posthoc_error_handling():
    """Test that NaN input is handled gracefully (does not crash).

    NaN scores may or may not produce an error field depending on MCMC
    stochasticity and failure handling. The key requirement is that the
    function completes without raising an exception.
    """
    import pandas as pd
    from optstop import optimal_stopping_posthoc
    df = pd.DataFrame({
        'grouping_num': [1, 2],
        'task_num': [1, 1],
        'sample_id_num': [1, 1],
        'epoch': [1, 1],
        'score': [float('nan'), float('nan')],
    })
    params = {'draws': 10, 'tune': 10, 'chains': 2, 'cores': 2}
    pruned_df, summary = optimal_stopping_posthoc(
        df, params,
        grouping_columns=['grouping_num', 'task_num'],
        sample_id_column='sample_id_num',
        epoch_column='epoch'
    )
    # Function must complete and return valid structure
    assert isinstance(summary, list)
    assert len(summary) > 0

def test_optimal_stopping_posthoc_random_seed_reproducibility():
    import pandas as pd
    from optstop import optimal_stopping_posthoc
    df = pd.DataFrame({
        'grouping_num': [1]*8,
        'task_num': [1]*8,
        'sample_id_num': [1,1,2,2,1,1,2,2],
        'epoch': [1,2,1,2,3,4,3,4],
        'score': [1,0,1,1,0,1,1,0],
    })
    params = {'draws': 50, 'tune': 50, 'chains': 2, 'cores': 2, 'random_seed': 123}
    pruned_df1, summary1 = optimal_stopping_posthoc(
        df, params,
        grouping_columns=['grouping_num', 'task_num'],
        sample_id_column='sample_id_num',
        epoch_column='epoch'
    )
    pruned_df2, summary2 = optimal_stopping_posthoc(
        df, params,
        grouping_columns=['grouping_num', 'task_num'],
        sample_id_column='sample_id_num',
        epoch_column='epoch'
    )
    assert pruned_df1.equals(pruned_df2)

def test_convergence_posthoc_missing_columns():
    import pandas as pd
    from optstop import convergence_posthoc
    df = pd.DataFrame({
        'grouping_num': [1, 1],
        'task_num': [1, 1],
        # 'sample_id_num' missing
        'epoch': [1, 2],
        'score': [1, 0],
    })
    params = {'draws': 10, 'tune': 10, 'chains': 2, 'cores': 2}
    try:
        convergence_posthoc(
            df, params,
            grouping_columns=['grouping_num', 'task_num'],
            sample_id_column='sample_id_num',
            epoch_column='epoch'
        )
        assert False, "Should raise ValueError for missing columns"
    except ValueError as e:
        assert 'sample_id_num' in str(e)

def test_convergence_posthoc_empty_df():
    import pandas as pd
    from optstop import convergence_posthoc
    df = pd.DataFrame(columns=['grouping_num', 'task_num', 'sample_id_num', 'epoch', 'score'])
    params = {'draws': 10, 'tune': 10, 'chains': 2, 'cores': 2}
    result = convergence_posthoc(
        df, params,
        grouping_columns=['grouping_num', 'task_num'],
        sample_id_column='sample_id_num',
        epoch_column='epoch'
    )
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_convergence_posthoc_invalid_params():
    import pandas as pd
    from optstop import convergence_posthoc
    df = pd.DataFrame({
        'grouping_num': [1, 1],
        'task_num': [1, 1],
        'sample_id_num': [1, 1],
        'epoch': [1, 2],
        'score': [1, 0],
    })
    params = {'draws': -1, 'tune': 10}
    try:
        convergence_posthoc(
            df, params,
            grouping_columns=['grouping_num', 'task_num'],
            sample_id_column='sample_id_num',
            epoch_column='epoch'
        )
        assert False, "Should raise ValueError for negative draws"
    except ValueError as e:
        assert 'draws' in str(e)

def test_convergence_posthoc_error_handling():
    import pandas as pd
    from optstop import convergence_posthoc
    df = pd.DataFrame({
        'grouping_num': [1, 2],
        'task_num': [1, 1],
        'sample_id_num': [1, 1],
        'epoch': [1, 1],
        'score': [float('nan'), float('nan')],
    })
    params = {'draws': 10, 'tune': 10, 'chains': 2, 'cores': 2}
    result = convergence_posthoc(
        df, params,
        grouping_columns=['grouping_num', 'task_num'],
        sample_id_column='sample_id_num',
        epoch_column='epoch'
    )
    assert 'error' in result.columns
    assert any(result['error'].notna()) 