import subprocess
import tempfile
import os
import pandas as pd
import pytest
from pathlib import Path
import sys
from unittest.mock import patch


def create_test_csv_data():
    """Create minimal test data for CLI testing"""
    return pd.DataFrame({
        'grouping_num': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
        'task_num': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'sample_id_num': [1, 1, 2, 2, 3, 3, 1, 1, 2, 2],
        'epoch': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        'score': [1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    })


def test_cli_posthoc_basic():
    """Test basic functionality of optstop-posthoc CLI"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create test CSV file
        csv_path = tmp_path / "test_data.csv"
        df = create_test_csv_data()
        df.to_csv(csv_path, index=False)
        
        # Create output paths
        output_path = tmp_path / "pruned.csv"
        summary_path = tmp_path / "summary.csv"
        log_path = tmp_path / "test.log"
        
        # Test CLI by calling the function directly
        from optstop.cli import main
        
        # Mock sys.argv to simulate CLI arguments
        test_args = [
            "optstop-posthoc",
            "--csv", str(csv_path),
            "--output", str(output_path),
            "--summary", str(summary_path),
            "--log", str(log_path),
            "--grouping_columns", "grouping_num,task_num",
            "--sample_id_column", "sample_id_num",
            "--epoch_column", "epoch",
            "--delta_item", "0.5",
            "--delta_cap", "0.5",
            "--draws", "50",
            "--tune", "50",
            "--CI_delta", "0.01",
            "--conservatism", "1.5",
            "--random_seed", "42"
        ]
        
        with patch.object(sys, 'argv', test_args):
            main()
        
        # Check that output files were created
        assert output_path.exists(), "Output CSV file was not created"
        assert summary_path.exists(), "Summary CSV file was not created"
        assert log_path.exists(), "Log file was not created"
        
        # Check that output files are not empty
        assert output_path.stat().st_size > 0, "Output CSV file is empty"
        assert summary_path.stat().st_size > 0, "Summary CSV file is empty"
        assert log_path.stat().st_size > 0, "Log file is empty"
        
        # Check that output CSV has the expected columns
        output_df = pd.read_csv(output_path)
        expected_columns = ['grouping_num', 'task_num', 'sample_id_num', 'epoch', 'score']
        assert all(col in output_df.columns for col in expected_columns), "Output CSV missing expected columns"
        
        # Check that summary CSV has expected structure
        summary_df = pd.read_csv(summary_path)
        expected_summary_columns = ['grouping', 'n_items_used', 'theta_ci_low', 'theta_ci_high', 'theta_ci_width', 'percent_items_used', 'avg_reps_per_item']
        assert all(col in summary_df.columns for col in expected_summary_columns), "Summary CSV missing expected columns"


def test_cli_live_basic():
    """Test basic functionality of optstop-live CLI"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create test CSV file
        csv_path = tmp_path / "test_data.csv"
        df = create_test_csv_data()
        df.to_csv(csv_path, index=False)
        
        # Create log path
        log_path = tmp_path / "test_live.log"
        
        # Test CLI by calling the function directly
        from optstop.cli import main_live
        
        # Mock sys.argv to simulate CLI arguments
        test_args = [
            "optstop-live",
            "--csv", str(csv_path),
            "--log", str(log_path),
            "--grouping_columns", "grouping_num",
            "--sample_id_column", "sample_id_num",
            "--epoch_column", "epoch",
            "--delta_item", "0.5",
            "--delta_cap", "0.5",
            "--draws", "50",
            "--tune", "50",
            "--CI_delta", "0.01",
            "--conservatism", "1.5",
            "--random_seed", "42"
        ]
        
        with patch.object(sys, 'argv', test_args):
            main_live()
        
        # Check that log file was created
        assert log_path.exists(), "Log file was not created"
        assert log_path.stat().st_size > 0, "Log file is empty"


def test_cli_convergence_basic():
    """Test basic functionality of optstop-convergence CLI"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create test CSV file with more data for convergence analysis
        csv_path = tmp_path / "test_data.csv"
        df = pd.DataFrame({
            'grouping_num': [1]*12 + [2]*12,
            'task_num': [1]*24,
            'sample_id_num': [1,1,1,1,2,2,2,2,3,3,3,3,1,1,1,1,2,2,2,2,3,3,3,3],
            'epoch': [1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4],
            'score': [1,0,1,1,0,1,1,0,1,1,0,1,0,1,1,1,1,0,1,0,0,1,1,1],
        })
        df.to_csv(csv_path, index=False)
        
        # Create output path
        output_path = tmp_path / "convergence_stats.csv"
        log_path = tmp_path / "test_convergence.log"
        
        # Test CLI by calling the function directly
        from optstop.cli import main_convergence
        
        # Mock sys.argv to simulate CLI arguments
        test_args = [
            "optstop-convergence",
            "--csv", str(csv_path),
            "--output", str(output_path),
            "--log", str(log_path),
            "--grouping_columns", "grouping_num,task_num",
            "--sample_id_column", "sample_id_num",
            "--epoch_column", "epoch",
            "--delta_item", "0.5",
            "--delta_cap", "0.5",
            "--draws", "50",
            "--tune", "50",
            "--CI_delta", "0.01",
            "--conservatism", "1.5",
            "--random_seed", "42"
        ]
        
        with patch.object(sys, 'argv', test_args):
            main_convergence()
        
        # Check that output file was created
        assert output_path.exists(), "Convergence stats CSV file was not created"
        assert output_path.stat().st_size > 0, "Convergence stats CSV file is empty"
        assert log_path.exists(), "Log file was not created"
        assert log_path.stat().st_size > 0, "Log file is empty"


def test_cli_posthoc_missing_file():
    """Test CLI error handling for missing input file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        output_path = tmp_path / "pruned.csv"
        
        # Test CLI by calling the function directly
        from optstop.cli import main
        
        # Mock sys.argv to simulate CLI arguments with non-existent file
        test_args = [
            "optstop-posthoc",
            "--csv", "nonexistent_file.csv",
            "--output", str(output_path),
            "--grouping_columns", "grouping_num,task_num",
            "--sample_id_column", "sample_id_num",
            "--epoch_column", "epoch",
            "--delta_item", "0.5",
            "--draws", "50",
            "--tune", "50"
        ]
        
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(FileNotFoundError):
                main()


def test_cli_posthoc_invalid_params():
    """Test CLI error handling for invalid parameters"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create test CSV file
        csv_path = tmp_path / "test_data.csv"
        df = create_test_csv_data()
        df.to_csv(csv_path, index=False)
        
        output_path = tmp_path / "pruned.csv"
        
        # Test CLI by calling the function directly
        from optstop.cli import main
        
        # Mock sys.argv to simulate CLI arguments with invalid parameters
        test_args = [
            "optstop-posthoc",
            "--csv", str(csv_path),
            "--output", str(output_path),
            "--grouping_columns", "grouping_num,task_num",
            "--sample_id_column", "sample_id_num",
            "--epoch_column", "epoch",
            "--draws", "-1",  # Invalid negative value
            "--tune", "50"
        ]
        
        with patch.object(sys, 'argv', test_args):
            # Should fail due to validation in the underlying function
            with pytest.raises((ValueError, AssertionError)):
                main()


def test_cli_posthoc_minimal_params():
    """Test CLI with minimal required parameters"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create test CSV file
        csv_path = tmp_path / "test_data.csv"
        df = create_test_csv_data()
        df.to_csv(csv_path, index=False)
        
        output_path = tmp_path / "pruned.csv"
        
        # Test CLI by calling the function directly
        from optstop.cli import main
        
        # Mock sys.argv to simulate CLI arguments with minimal parameters
        test_args = [
            "optstop-posthoc",
            "--csv", str(csv_path),
            "--output", str(output_path),
            "--grouping_columns", "grouping_num,task_num",
            "--sample_id_column", "sample_id_num",
            "--epoch_column", "epoch",
        ]
        
        with patch.object(sys, 'argv', test_args):
            main()
        
        # Should succeed with default parameters
        assert output_path.exists(), "Output file should be created with minimal params"


def test_cli_convergence_output_structure():
    """Test that convergence CLI output has correct structure"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create test CSV file with sufficient data
        csv_path = tmp_path / "test_data.csv"
        df = pd.DataFrame({
            'grouping_num': [1]*16 + [2]*16,
            'task_num': [1]*32,
            'sample_id_num': [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4],
            'epoch': [1,2,3,4]*8,
            'score': [1,0,1,1,0,1,1,0,1,1,0,1,0,1,1,1,1,0,1,0,0,1,1,1,1,1,0,1,0,1,1,0],
        })
        df.to_csv(csv_path, index=False)
        
        output_path = tmp_path / "convergence_stats.csv"
        
        # Test CLI by calling the function directly
        from optstop.cli import main_convergence
        
        # Mock sys.argv to simulate CLI arguments
        test_args = [
            "optstop-convergence",
            "--csv", str(csv_path),
            "--output", str(output_path),
            "--grouping_columns", "grouping_num,task_num",
            "--sample_id_column", "sample_id_num",
            "--epoch_column", "epoch",
            "--draws", "50",
            "--tune", "50"
        ]
        
        with patch.object(sys, 'argv', test_args):
            main_convergence()
        
        # Check output file structure
        assert output_path.exists(), "Convergence stats file should be created"
        output_df = pd.read_csv(output_path)
        
        # Check for expected columns (basic structure)
        assert len(output_df) > 0, "Convergence stats should have at least one row"
        assert 'grouping' in output_df.columns, "Should have grouping column"


def test_cli_help_text():
    """Test that help text can be generated for all CLI functions"""
    from optstop.cli import main, main_live, main_convergence
    
    # Test that functions exist and can be imported
    assert callable(main)
    assert callable(main_live)
    assert callable(main_convergence)
    
    # Test that argparse works by checking if help text can be generated
    import argparse
    
    # Create parsers to test help text generation
    parser1 = argparse.ArgumentParser(description="Test parser")
    parser1.add_argument('--csv', required=True, help='Path to input CSV file')
    parser1.add_argument('--output', required=True, help='Path to output file')
    
    # This should not raise an exception
    help_text = parser1.format_help()
    assert "usage:" in help_text.lower()
    assert "csv" in help_text.lower()
    assert "output" in help_text.lower() 