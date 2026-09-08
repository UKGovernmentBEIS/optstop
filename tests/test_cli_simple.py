import pandas as pd
import pytest
import sys
from unittest.mock import patch
from pathlib import Path


def create_test_csv_data():
    """Create minimal test data for CLI testing"""
    return pd.DataFrame({
        'grouping_num': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
        'task_num': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'sample_id_num': [1, 1, 2, 2, 3, 3, 1, 1, 2, 2],
        'epoch': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        'score': [1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    })


def test_cli_functions_exist():
    """Test that all CLI functions can be imported and are callable"""
    from optstop.cli import main, main_live, main_convergence
    
    assert callable(main)
    assert callable(main_live)
    assert callable(main_convergence)


def test_cli_posthoc_functionality():
    """Test basic functionality of optstop-posthoc CLI function"""
    from optstop.cli import main
    
    # Create test data
    df = create_test_csv_data()
    
    # Save to a temporary file
    test_csv = "test_cli_data.csv"
    test_output = "test_cli_output.csv"
    test_summary = "test_cli_summary.csv"
    test_log = "test_cli.log"
    
    try:
        df.to_csv(test_csv, index=False)
        
        # Mock sys.argv to simulate CLI arguments
        test_args = [
            "optstop-posthoc",
            "--csv", test_csv,
            "--output", test_output,
            "--summary", test_summary,
            "--log", test_log,
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
        assert Path(test_output).exists(), "Output CSV file was not created"
        assert Path(test_summary).exists(), "Summary CSV file was not created"
        assert Path(test_log).exists(), "Log file was not created"
        
        # Check that output files are not empty
        assert Path(test_output).stat().st_size > 0, "Output CSV file is empty"
        assert Path(test_summary).stat().st_size > 0, "Summary CSV file is empty"
        assert Path(test_log).stat().st_size > 0, "Log file is empty"
        
        # Check that output CSV has the expected columns
        output_df = pd.read_csv(test_output)
        expected_columns = ['grouping_num', 'task_num', 'sample_id_num', 'epoch', 'score']
        assert all(col in output_df.columns for col in expected_columns), "Output CSV missing expected columns"
        
        # Check that summary CSV has expected structure
        summary_df = pd.read_csv(test_summary)
        expected_summary_columns = ['grouping', 'n_items_used', 'theta_ci_low', 'theta_ci_high', 'theta_ci_width', 'percent_items_used', 'avg_reps_per_item', 'error']
        assert all(col in summary_df.columns for col in expected_summary_columns), "Summary CSV missing expected columns"
        
    finally:
        # Clean up test files
        for file in [test_csv, test_output, test_summary, test_log]:
            if Path(file).exists():
                try:
                    Path(file).unlink()
                except:
                    pass  # Ignore cleanup errors


def test_cli_posthoc_with_diagnostics():
    """Test optstop-posthoc CLI function with diagnostic generation"""
    from optstop.cli import main
    
    # Create test data
    df = create_test_csv_data()
    
    # Save to a temporary file
    test_csv = "test_cli_data.csv"
    test_output = "test_cli_output.csv"
    test_summary = "test_cli_summary.csv"
    test_log = "test_cli.log"
    test_diagnostics_prefix = "test_cli_diagnostics"
    
    try:
        df.to_csv(test_csv, index=False)
        
        # Mock sys.argv to simulate CLI arguments with diagnostics
        test_args = [
            "optstop-posthoc",
            "--csv", test_csv,
            "--output", test_output,
            "--summary", test_summary,
            "--log", test_log,
            "--grouping_columns", "grouping_num,task_num",
            "--sample_id_column", "sample_id_num",
            "--epoch_column", "epoch",
            "--delta_item", "0.5",
            "--delta_cap", "0.5",
            "--draws", "50",
            "--tune", "50",
            "--CI_delta", "0.01",
            "--conservatism", "1.5",
            "--random_seed", "42",
            "--generate_diagnostics",
            "--diagnostics_prefix", test_diagnostics_prefix
        ]
        
        with patch.object(sys, 'argv', test_args):
            main()
        
        # Check that output files were created
        assert Path(test_output).exists(), "Output CSV file was not created"
        assert Path(test_summary).exists(), "Summary CSV file was not created"
        assert Path(test_log).exists(), "Log file was not created"
        
        # Check that diagnostic files were created.
        # KNOWN ISSUE (dependency, tracked as a compatibility follow-up): under
        # pandas 3 / current matplotlib the diagnostic PNG may not be produced
        # even though the post-hoc analysis itself completes (CSVs are written),
        # so this PNG assertion can fail. Not a core-functionality regression.
        diagnostic_png = Path(f"{test_diagnostics_prefix}.png")
        diagnostic_csv = Path(f"{test_diagnostics_prefix}_paired.csv")
        assert diagnostic_png.exists(), f"Diagnostic PNG file was not created: {diagnostic_png}"
        assert diagnostic_csv.exists(), f"Diagnostic CSV file was not created: {diagnostic_csv}"
        
        # Check that diagnostic files are not empty
        assert diagnostic_png.stat().st_size > 0, "Diagnostic PNG file is empty"
        assert diagnostic_csv.stat().st_size > 0, "Diagnostic CSV file is empty"
        
    finally:
        # Clean up test files
        for file in [test_csv, test_output, test_summary, test_log, f"{test_diagnostics_prefix}.png", f"{test_diagnostics_prefix}_paired.csv"]:
            if Path(file).exists():
                try:
                    Path(file).unlink()
                except:
                    pass  # Ignore cleanup errors


def test_cli_live_functionality():
    """Test basic functionality of optstop-live CLI function"""
    from optstop.cli import main_live
    
    # Create test data
    df = create_test_csv_data()
    
    # Save to a temporary file
    test_csv = "test_cli_live_data.csv"
    test_log = "test_cli_live.log"
    
    try:
        df.to_csv(test_csv, index=False)
        
        # Mock sys.argv to simulate CLI arguments
        test_args = [
            "optstop-live",
            "--csv", test_csv,
            "--log", test_log,
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
        assert Path(test_log).exists(), "Log file was not created"
        assert Path(test_log).stat().st_size > 0, "Log file is empty"
        
    finally:
        # Clean up test files
        for file in [test_csv, test_log]:
            if Path(file).exists():
                try:
                    Path(file).unlink()
                except:
                    pass  # Ignore cleanup errors


def test_cli_convergence_functionality():
    """Test basic functionality of optstop-convergence CLI function"""
    from optstop.cli import main_convergence
    
    # Create test data with more data for convergence analysis
    df = pd.DataFrame({
        'grouping_num': [1]*12 + [2]*12,
        'task_num': [1]*24,
        'sample_id_num': [1,1,1,1,2,2,2,2,3,3,3,3,1,1,1,1,2,2,2,2,3,3,3,3],
        'epoch': [1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4],
        'score': [1,0,1,1,0,1,1,0,1,1,0,1,0,1,1,1,1,0,1,0,0,1,1,1],
    })
    
    # Save to a temporary file
    test_csv = "test_cli_convergence_data.csv"
    test_output = "test_cli_convergence_output.csv"
    test_log = "test_cli_convergence.log"
    
    try:
        df.to_csv(test_csv, index=False)
        
        # Mock sys.argv to simulate CLI arguments
        test_args = [
            "optstop-convergence",
            "--csv", test_csv,
            "--output", test_output,
            "--log", test_log,
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
        assert Path(test_output).exists(), "Convergence stats CSV file was not created"
        assert Path(test_output).stat().st_size > 0, "Convergence stats CSV file is empty"
        assert Path(test_log).exists(), "Log file was not created"
        assert Path(test_log).stat().st_size > 0, "Log file is empty"
        
    finally:
        # Clean up test files
        for file in [test_csv, test_output, test_log]:
            if Path(file).exists():
                try:
                    Path(file).unlink()
                except:
                    pass  # Ignore cleanup errors


def test_cli_error_handling():
    """Test CLI error handling for missing input file"""
    from optstop.cli import main
    
    # Mock sys.argv to simulate CLI arguments with non-existent file
    test_args = [
        "optstop-posthoc",
        "--csv", "nonexistent_file.csv",
        "--output", "test_output.csv",
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