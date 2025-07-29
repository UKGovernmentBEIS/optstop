"""
Command-line interface for the optstop package.
"""

import argparse
import pandas as pd
from .rule import configure_optstop_logging, optimal_stopping_posthoc, optimal_stopping_live
from .convergence import convergence_posthoc

def close_all_log_handlers():
    """Close all logging handlers to release file locks (needed for Windows test cleanup)."""
    import logging
    root = logging.getLogger()
    handlers = root.handlers[:]
    for handler in handlers:
        handler.close()
        root.removeHandler(handler)

def main():
    parser = argparse.ArgumentParser(description="Run post-hoc optimal stopping analysis.")
    parser.add_argument('--csv', required=True, help='Path to input CSV file')
    parser.add_argument('--output', required=True, help='Path to output pruned CSV file')
    parser.add_argument('--summary', required=False, help='Path to output summary CSV file')
    parser.add_argument('--log', default='optstop_cli.log', help='Path to log file')
    parser.add_argument('--grouping_columns', required=True, help='Comma-separated list of column names to use for grouping (e.g., "subject,task") or a single column name')
    parser.add_argument('--sample_id_column', required=True, help='Column name for sample ID')
    parser.add_argument('--epoch_column', required=True, help='Column name for epoch/trial')
    parser.add_argument('--score_column', default='score', help='Column name for score (default: score)')
    parser.add_argument('--delta_item', type=float, default=0.05, help='Max acceptable CI width for individual items (default: 0.05)')
    parser.add_argument('--delta_cap', type=float, default=0.05, help='Max acceptable CI width for task/grouping (default: 0.05)')
    parser.add_argument('--draws', type=int, default=1000, help='Number of MCMC samples for PyMC (default: 1000)')
    parser.add_argument('--tune', type=int, default=1000, help='Number of tuning steps for PyMC (default: 1000)')
    parser.add_argument('--chains', type=int, default=4, help='Number of MCMC chains for PyMC (default: 4)')
    parser.add_argument('--cores', type=int, default=4, help='Number of CPU cores for PyMC (default: 4)')
    parser.add_argument('--CI_delta', type=float, default=0.0002, help='Slope threshold for determining CI stabilization (default: 0.0002)')
    parser.add_argument('--conservatism', type=float, default=2, help='Factor for rare event conservatism (default: 2)')
    parser.add_argument('--rep_batch_size', type=int, default=1, help='Number of repetitions to process in each batch (default: 1)')
    parser.add_argument('--pymc_refresh_every', type=int, default=2, help='How often to run the PyMC model (default: 2)')
    parser.add_argument('--stab_window', type=int, default=5, help='Window size for assessing CI stabilization (default: 5)')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed for reproducible results (optional)')
    parser.add_argument('--no_progress', action='store_true', help='Disable progress bar display')
    parser.add_argument('--generate_diagnostics', action='store_true', help='Generate diagnostic plots comparing full vs pruned datasets')
    parser.add_argument('--diagnostics_prefix', default='optstop_diagnostics', help='Prefix for diagnostic output files')
    args = parser.parse_args()

    configure_optstop_logging(args.log, console_output=False)
    df = pd.read_csv(args.csv)
    params = {
        'delta_item': args.delta_item,
        'delta_cap': args.delta_cap,
        'draws': args.draws,
        'tune': args.tune,
        'chains': args.chains,
        'cores': args.cores,
        'CI_delta': args.CI_delta,
        'conservatism': args.conservatism,
        'rep_batch_size': args.rep_batch_size,
        'pymc_refresh_every': args.pymc_refresh_every,
        'stab_window': args.stab_window,
    }
    if args.random_seed is not None:
        params['random_seed'] = args.random_seed
    # Parse grouping_columns
    grouping_columns = [col.strip() for col in args.grouping_columns.split(',')] if ',' in args.grouping_columns else args.grouping_columns.strip()
    pruned_df, summary = optimal_stopping_posthoc(
        df, params, grouping_columns, args.sample_id_column, args.epoch_column, args.score_column,
        display_progress=not args.no_progress, 
        generate_diagnostics=args.generate_diagnostics, 
        diagnostics_prefix=args.diagnostics_prefix
    )
    pruned_df.to_csv(args.output, index=False)
    if args.summary:
        pd.DataFrame(summary).to_csv(args.summary, index=False)
    print(f"Pruned data saved to {args.output}")
    if args.summary:
        print(f"Summary saved to {args.summary}")
    close_all_log_handlers()

def main_live():
    parser = argparse.ArgumentParser(description="Run live optimal stopping analysis.")
    parser.add_argument('--csv', required=True, help='Path to input CSV file (current data)')
    parser.add_argument('--log', default='optstop_live.log', help='Path to log file')
    parser.add_argument('--grouping_columns', required=True, help='Comma-separated list of column names to use for grouping (e.g., "subject,task") or a single column name')
    parser.add_argument('--sample_id_column', required=True, help='Column name for sample ID')
    parser.add_argument('--epoch_column', required=True, help='Column name for epoch/trial')
    parser.add_argument('--score_column', default='score', help='Column name for score (default: score)')
    parser.add_argument('--delta_item', type=float, default=0.05, help='Max acceptable CI width for individual items (default: 0.05)')
    parser.add_argument('--delta_cap', type=float, default=0.05, help='Max acceptable CI width for task/grouping (default: 0.05)')
    parser.add_argument('--draws', type=int, default=1000, help='Number of MCMC samples for PyMC (default: 1000)')
    parser.add_argument('--tune', type=int, default=1000, help='Number of tuning steps for PyMC (default: 1000)')
    parser.add_argument('--chains', type=int, default=4, help='Number of MCMC chains for PyMC (default: 4)')
    parser.add_argument('--cores', type=int, default=4, help='Number of CPU cores for PyMC (default: 4)')
    parser.add_argument('--CI_delta', type=float, default=0.0002, help='Slope threshold for determining CI stabilization (default: 0.0002)')
    parser.add_argument('--conservatism', type=float, default=2, help='Factor for rare event conservatism (default: 2)')
    parser.add_argument('--rep_batch_size', type=int, default=1, help='Number of repetitions to process in each batch (default: 1)')
    parser.add_argument('--pymc_refresh_every', type=int, default=2, help='How often to run the PyMC model (default: 2)')
    parser.add_argument('--stab_window', type=int, default=5, help='Window size for assessing CI stabilization (default: 5)')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed for reproducible results (optional)')
    parser.add_argument('--no_progress', action='store_true', help='Disable progress bar display')
    args = parser.parse_args()

    configure_optstop_logging(args.log, console_output=False)
    df = pd.read_csv(args.csv)
    params = {
        'delta_item': args.delta_item,
        'delta_cap': args.delta_cap,
        'draws': args.draws,
        'tune': args.tune,
        'chains': args.chains,
        'cores': args.cores,
        'CI_delta': args.CI_delta,
        'conservatism': args.conservatism,
        'rep_batch_size': args.rep_batch_size,
        'pymc_refresh_every': args.pymc_refresh_every,
        'stab_window': args.stab_window,
    }
    if args.random_seed is not None:
        params['random_seed'] = args.random_seed
    grouping_columns = [col.strip() for col in args.grouping_columns.split(',')] if ',' in args.grouping_columns else args.grouping_columns.strip()
    result = optimal_stopping_live(
        df, params, grouping_columns, args.sample_id_column, args.epoch_column, args.score_column,
        display_progress=not args.no_progress
    )
    print("Sample IDs to stop:", result['stop_sample_ids'])
    print("Stop task/grouping?", result['stop_task'])
    close_all_log_handlers()


def main_convergence():
    parser = argparse.ArgumentParser(description="Run post-hoc convergence analysis.")
    parser.add_argument('--csv', required=True, help='Path to input CSV file')
    parser.add_argument('--output', required=True, help='Path to output convergence stats CSV file')
    parser.add_argument('--log', default='optstop_convergence.log', help='Path to log file')
    parser.add_argument('--grouping_columns', required=True, help='Comma-separated list of column names to use for grouping (e.g., "subject,task") or a single column name')
    parser.add_argument('--sample_id_column', required=True, help='Column name for sample ID')
    parser.add_argument('--epoch_column', required=True, help='Column name for epoch/trial')
    parser.add_argument('--score_column', default='score', help='Column name for score (default: score)')
    parser.add_argument('--delta_item', type=float, default=0.05, help='Max acceptable CI width for individual items (default: 0.05)')
    parser.add_argument('--delta_cap', type=float, default=0.05, help='Max acceptable CI width for task/grouping (default: 0.05)')
    parser.add_argument('--draws', type=int, default=1000, help='Number of MCMC samples for PyMC (default: 1000)')
    parser.add_argument('--tune', type=int, default=1000, help='Number of tuning steps for PyMC (default: 1000)')
    parser.add_argument('--chains', type=int, default=4, help='Number of MCMC chains for PyMC (default: 4)')
    parser.add_argument('--cores', type=int, default=4, help='Number of CPU cores for PyMC (default: 4)')
    parser.add_argument('--CI_delta', type=float, default=0.0002, help='Slope threshold for determining CI stabilization (default: 0.0002)')
    parser.add_argument('--conservatism', type=float, default=2, help='Factor for rare event conservatism (default: 2)')
    parser.add_argument('--rep_batch_size', type=int, default=1, help='Number of repetitions to process in each batch (default: 1)')
    parser.add_argument('--pymc_refresh_every', type=int, default=2, help='How often to run the PyMC model (default: 2)')
    parser.add_argument('--stab_window', type=int, default=5, help='Window size for assessing CI stabilization (default: 5)')
    parser.add_argument('--item_seqs', type=int, default=20, help='Number of randomized item orderings per grouping (default: 20)')
    parser.add_argument('--epoch_seqs', type=int, default=20, help='Number of randomized epoch orderings per item (default: 20)')
    parser.add_argument('--random_seed', type=int, default=None, help='Random seed for reproducible results (optional)')
    parser.add_argument('--no_progress', action='store_true', help='Disable progress bar display')
    parser.add_argument('--no_diagnostics', action='store_true', help='Disable generation of convergence diagnostic figures (default: diagnostics ON)')
    parser.add_argument('--diagnostics_prefix', default='convergence_eval', help='Prefix for convergence diagnostic output files')
    args = parser.parse_args()

    configure_optstop_logging(args.log, console_output=False)
    df = pd.read_csv(args.csv)
    params = {
        'delta_item': args.delta_item,
        'delta_cap': args.delta_cap,
        'draws': args.draws,
        'tune': args.tune,
        'chains': args.chains,
        'cores': args.cores,
        'CI_delta': args.CI_delta,
        'conservatism': args.conservatism,
        'rep_batch_size': args.rep_batch_size,
        'pymc_refresh_every': args.pymc_refresh_every,
        'stab_window': args.stab_window,
        'item_seqs': args.item_seqs,
        'epoch_seqs': args.epoch_seqs,
    }
    if args.random_seed is not None:
        params['random_seed'] = args.random_seed
    grouping_columns = [col.strip() for col in args.grouping_columns.split(',')] if ',' in args.grouping_columns else args.grouping_columns.strip()
    result = convergence_posthoc(
        df, params, grouping_columns, args.sample_id_column, args.epoch_column, args.score_column,
        display_progress=not args.no_progress,
        generate_diagnostics=not args.no_diagnostics,
        diagnostics_prefix=args.diagnostics_prefix
    )
    result.to_csv(args.output, index=False)
    print(f"Convergence stats saved to {args.output}")
    close_all_log_handlers() 