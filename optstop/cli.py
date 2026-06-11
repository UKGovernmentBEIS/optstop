"""
Command-line interface for the optstop package.
"""

import argparse
import pandas as pd
from .rule import configure_optstop_logging, optimal_stopping_posthoc, optimal_stopping_live
from .convergence import convergence_posthoc
from .__version__ import __version__

def close_all_log_handlers():
    """Close all logging handlers to release file locks (needed for Windows test cleanup)."""
    import logging
    root = logging.getLogger()
    handlers = root.handlers[:]
    for handler in handlers:
        handler.close()
        root.removeHandler(handler)

def main():
    import warnings
    warnings.filterwarnings('ignore', message='.*effective sample size.*')
    warnings.filterwarnings('ignore', message='.*rhat.*')
    warnings.filterwarnings('ignore', message='.*ess.*')
    warnings.filterwarnings('ignore', message='.*divergence.*')
    warnings.filterwarnings('ignore', message='.*divergences.*') 
    warnings.filterwarnings('ignore', message='.*target_accept.*')
    warnings.filterwarnings('ignore', message='.*reparameterize.*')
    warnings.filterwarnings('ignore', message='.*smaller than 100.*')
    warnings.filterwarnings('ignore', message='.*needed for reliable.*')
    warnings.filterwarnings('ignore', message='.*There was.*divergence.*')
    warnings.filterwarnings('ignore', message='.*There were.*divergence.*')
    warnings.filterwarnings('ignore', message='.*after tuning.*')
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parser = argparse.ArgumentParser(description="Run post-hoc optimal stopping analysis.")
        parser.add_argument('--version', action='version', version=f'optstop {__version__}')
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
        parser.add_argument('--CI_delta', type=float, default=0.00001, help='Slope threshold for determining CI stabilization (default: 0.00001)')
        parser.add_argument('--conservatism', type=float, default=5, help='Factor for rare event conservatism (default: 5)')
        parser.add_argument('--rep_batch_size', type=int, default=1, help='Number of repetitions to process in each batch (default: 1)')
        parser.add_argument('--pymc_refresh_every', type=int, default=2, help='How often to run the PyMC model (default: 2)')
        parser.add_argument('--stab_window', type=int, default=15, help='Window size for assessing CI stabilization (default: 15)')
        parser.add_argument('--random_seed', type=int, default=None, help='Random seed for reproducible results (optional)')
        parser.add_argument('--no_progress', action='store_true', help='Disable progress bar display')
        parser.add_argument('--generate_diagnostics', action='store_true', help='Generate diagnostic plots comparing full vs pruned datasets')
        parser.add_argument('--diagnostics_prefix', default='optstop_diagnostics', help='Prefix for diagnostic output files')
        parser.add_argument('--low_performance_threshold', type=float, default=0.01, help='Success rate below which conservative stopping is applied (default: 0.01)')
        parser.add_argument('--ordinal_tasks', type=str, default=None, help='Comma-separated list of substrings to identify ordinal groupings (e.g., "confidence,rating")')
        parser.add_argument('--ordinal_max_score', type=int, default=10, help='Maximum score for ordinal data (default: 10)')
        parser.add_argument('--ordinal_inference', type=str, default='modal', choices=['modal', 'entropy', 'hybrid'], help='Ordinal inference method: modal, entropy, or hybrid (default: modal)')
        parser.add_argument('--ordinal_model_type', type=str, default='ordered_logistic', choices=['ordered_logistic', 'dirichlet'], help='Hierarchical model type for ordinal data (default: ordered_logistic)')
        parser.add_argument('--continuous_tasks', type=str, default=None, help='Comma-separated list of substrings to identify continuous bounded groupings (e.g., "mean_score,aggregated")')
        parser.add_argument('--entropy_threshold', type=float, default=0.8, help='Proportion of max entropy for false peak detection in hybrid mode (default: 0.8)')
        parser.add_argument('--prior_mu', type=float, default=0.0, help='Centre of group-level Normal prior on logit scale (default: 0.0 = 50%% probability). Positive values bias toward higher performance, negative toward lower.')
        parser.add_argument('--prior_sigma', type=float, default=None, help='Scale of group-level Normal prior on logit scale. If not set, uses pathway-specific defaults (binary/continuous: 1.5, ordinal: 2.0). If set, applies to all pathways.')
        parser.add_argument('--shuffle_items', action='store_true', help='Randomize item order within each grouping before processing. Recommended to avoid selection bias from sorted input.')
        parser.add_argument('--shuffle_seed', type=int, default=None, help='Random seed for reproducible shuffling (only used with --shuffle_items)')
        parser.add_argument('--processing_order', type=str, default='item_greedy', choices=['item_greedy', 'epoch_interleaved'], help='Processing order: item_greedy (all epochs per item, fast) or epoch_interleaved (all items per epoch, matches production). Default: item_greedy')
        parser.add_argument('--reanalysis_interval', type=int, default=10, help='Group model refresh interval in trials for epoch_interleaved mode (default: 10)')
        parser.add_argument('--disable_gpu', action='store_true', help='Disable GPU acceleration even if available')
        parser.add_argument('--force_gpu', action='store_true', help='Force GPU usage (will fail if GPU unavailable)')
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
            'low_performance_threshold': args.low_performance_threshold,
        }
        if args.random_seed is not None:
            params['random_seed'] = args.random_seed

        # Handle GPU settings
        if args.disable_gpu:
            params['use_gpu'] = False
        elif args.force_gpu:
            params['use_gpu'] = True
            params['force_gpu'] = True

        # Parse ordinal_tasks if provided
        ordinal_tasks = None
        if args.ordinal_tasks:
            ordinal_tasks = [task.strip() for task in args.ordinal_tasks.split(',')]

        # Parse continuous_tasks if provided
        continuous_tasks = None
        if args.continuous_tasks:
            continuous_tasks = [task.strip() for task in args.continuous_tasks.split(',')]

        # Parse grouping_columns
        grouping_columns = [col.strip() for col in args.grouping_columns.split(',')] if ',' in args.grouping_columns else args.grouping_columns.strip()
        pruned_df, summary = optimal_stopping_posthoc(
            df, params, grouping_columns, args.sample_id_column, args.epoch_column, args.score_column,
            display_progress=not args.no_progress,
            generate_diagnostics=args.generate_diagnostics,
            diagnostics_prefix=args.diagnostics_prefix,
            ordinal_tasks=ordinal_tasks,
            ordinal_max_score=args.ordinal_max_score,
            ordinal_inference=args.ordinal_inference,
            ordinal_model_type=args.ordinal_model_type,
            continuous_tasks=continuous_tasks,
            entropy_threshold=args.entropy_threshold,
            prior_mu=args.prior_mu,
            prior_sigma=args.prior_sigma,
            shuffle_items=args.shuffle_items,
            shuffle_seed=args.shuffle_seed,
            processing_order=args.processing_order,
            reanalysis_interval=args.reanalysis_interval
        )
        pruned_df.to_csv(args.output, index=False)
        if args.summary:
            pd.DataFrame(summary).to_csv(args.summary, index=False)
        print(f"Pruned data saved to {args.output}")
        if args.summary:
            print(f"Summary saved to {args.summary}")
    close_all_log_handlers()

def main_live():
    import warnings
    warnings.filterwarnings('ignore', message='.*effective sample size.*')
    warnings.filterwarnings('ignore', message='.*rhat.*')
    warnings.filterwarnings('ignore', message='.*ess.*')
    warnings.filterwarnings('ignore', message='.*divergence.*')
    warnings.filterwarnings('ignore', message='.*divergences.*') 
    warnings.filterwarnings('ignore', message='.*target_accept.*')
    warnings.filterwarnings('ignore', message='.*reparameterize.*')
    warnings.filterwarnings('ignore', message='.*smaller than 100.*')
    warnings.filterwarnings('ignore', message='.*needed for reliable.*')
    warnings.filterwarnings('ignore', message='.*There was.*divergence.*')
    warnings.filterwarnings('ignore', message='.*There were.*divergence.*')
    warnings.filterwarnings('ignore', message='.*after tuning.*')
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parser = argparse.ArgumentParser(description="Run live optimal stopping analysis.")
        parser.add_argument('--version', action='version', version=f'optstop {__version__}')
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
        parser.add_argument('--CI_delta', type=float, default=0.00001, help='Slope threshold for determining CI stabilization (default: 0.00001)')
        parser.add_argument('--conservatism', type=float, default=5, help='Factor for rare event conservatism (default: 5)')
        parser.add_argument('--rep_batch_size', type=int, default=1, help='Number of repetitions to process in each batch (default: 1)')
        parser.add_argument('--pymc_refresh_every', type=int, default=2, help='How often to run the PyMC model (default: 2)')
        parser.add_argument('--stab_window', type=int, default=15, help='Window size for assessing CI stabilization (default: 15)')
        parser.add_argument('--random_seed', type=int, default=None, help='Random seed for reproducible results (optional)')
        parser.add_argument('--no_progress', action='store_true', help='Disable progress bar display')
        parser.add_argument('--low_performance_threshold', type=float, default=0.01, help='Success rate below which conservative stopping is applied (default: 0.01)')
        parser.add_argument('--ordinal_tasks', type=str, default=None, help='Comma-separated list of substrings to identify ordinal groupings (e.g., "confidence,rating")')
        parser.add_argument('--ordinal_max_score', type=int, default=10, help='Maximum score for ordinal data (default: 10)')
        parser.add_argument('--ordinal_inference', type=str, default='modal', choices=['modal', 'entropy', 'hybrid'], help='Ordinal inference method: modal, entropy, or hybrid (default: modal)')
        parser.add_argument('--ordinal_model_type', type=str, default='ordered_logistic', choices=['ordered_logistic', 'dirichlet'], help='Hierarchical model type for ordinal data (default: ordered_logistic)')
        parser.add_argument('--continuous_tasks', type=str, default=None, help='Comma-separated list of substrings to identify continuous bounded groupings (e.g., "mean_score,aggregated")')
        parser.add_argument('--entropy_threshold', type=float, default=0.8, help='Proportion of max entropy for false peak detection in hybrid mode (default: 0.8)')
        parser.add_argument('--prior_mu', type=float, default=0.0, help='Centre of group-level Normal prior on logit scale (default: 0.0 = 50%% probability). Positive values bias toward higher performance, negative toward lower.')
        parser.add_argument('--prior_sigma', type=float, default=None, help='Scale of group-level Normal prior on logit scale. If not set, uses pathway-specific defaults (binary/continuous: 1.5, ordinal: 2.0). If set, applies to all pathways.')
        parser.add_argument('--disable_gpu', action='store_true', help='Disable GPU acceleration even if available')
        parser.add_argument('--force_gpu', action='store_true', help='Force GPU usage (will fail if GPU unavailable)')
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
            'low_performance_threshold': args.low_performance_threshold,
        }
        if args.random_seed is not None:
            params['random_seed'] = args.random_seed

        # Handle GPU settings
        if args.disable_gpu:
            params['use_gpu'] = False
        elif args.force_gpu:
            params['use_gpu'] = True
            params['force_gpu'] = True

        # Parse ordinal_tasks if provided
        ordinal_tasks = None
        if args.ordinal_tasks:
            ordinal_tasks = [task.strip() for task in args.ordinal_tasks.split(',')]

        # Parse continuous_tasks if provided
        continuous_tasks = None
        if args.continuous_tasks:
            continuous_tasks = [task.strip() for task in args.continuous_tasks.split(',')]

        grouping_columns = [col.strip() for col in args.grouping_columns.split(',')] if ',' in args.grouping_columns else args.grouping_columns.strip()
        result = optimal_stopping_live(
            df, params, grouping_columns, args.sample_id_column, args.epoch_column, args.score_column,
            display_progress=not args.no_progress,
            ordinal_tasks=ordinal_tasks,
            ordinal_max_score=args.ordinal_max_score,
            ordinal_inference=args.ordinal_inference,
            ordinal_model_type=args.ordinal_model_type,
            continuous_tasks=continuous_tasks,
            entropy_threshold=args.entropy_threshold,
            prior_mu=args.prior_mu,
            prior_sigma=args.prior_sigma
        )
        print("Sample IDs to stop:", result['stop_sample_ids'])
        print("Stop task/grouping?", result['stop_task'])
    close_all_log_handlers()


def main_convergence():
    import warnings
    warnings.filterwarnings('ignore', message='.*effective sample size.*')
    warnings.filterwarnings('ignore', message='.*rhat.*')
    warnings.filterwarnings('ignore', message='.*ess.*')
    warnings.filterwarnings('ignore', message='.*divergence.*')
    warnings.filterwarnings('ignore', message='.*divergences.*') 
    warnings.filterwarnings('ignore', message='.*target_accept.*')
    warnings.filterwarnings('ignore', message='.*reparameterize.*')
    warnings.filterwarnings('ignore', message='.*smaller than 100.*')
    warnings.filterwarnings('ignore', message='.*needed for reliable.*')
    warnings.filterwarnings('ignore', message='.*There was.*divergence.*')
    warnings.filterwarnings('ignore', message='.*There were.*divergence.*')
    warnings.filterwarnings('ignore', message='.*after tuning.*')
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parser = argparse.ArgumentParser(description="Run post-hoc convergence analysis.")
        parser.add_argument('--version', action='version', version=f'optstop {__version__}')
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
        parser.add_argument('--CI_delta', type=float, default=0.00001, help='Slope threshold for determining CI stabilization (default: 0.00001)')
        parser.add_argument('--conservatism', type=float, default=5, help='Factor for rare event conservatism (default: 5)')
        parser.add_argument('--rep_batch_size', type=int, default=1, help='Number of repetitions to process in each batch (default: 1)')
        parser.add_argument('--pymc_refresh_every', type=int, default=2, help='How often to run the PyMC model (default: 2)')
        parser.add_argument('--stab_window', type=int, default=15, help='Window size for assessing CI stabilization (default: 15)')
        parser.add_argument('--item_seqs', type=int, default=20, help='Number of randomized item orderings per grouping (default: 20)')
        parser.add_argument('--epoch_seqs', type=int, default=20, help='Number of randomized epoch orderings per item (default: 20)')
        parser.add_argument('--random_seed', type=int, default=None, help='Random seed for reproducible results (optional)')
        parser.add_argument('--no_progress', action='store_true', help='Disable progress bar display')
        parser.add_argument('--no_diagnostics', action='store_true', help='Disable generation of convergence diagnostic figures (default: diagnostics ON)')
        parser.add_argument('--diagnostics_prefix', default='convergence_eval', help='Prefix for convergence diagnostic output files')
        parser.add_argument('--low_performance_threshold', type=float, default=0.01, help='Success rate below which conservative stopping is applied (default: 0.01)')
        parser.add_argument('--ordinal_tasks', type=str, default=None, help='Comma-separated list of substrings to identify ordinal groupings (e.g., "confidence,rating")')
        parser.add_argument('--ordinal_max_score', type=int, default=10, help='Maximum score for ordinal data (default: 10)')
        parser.add_argument('--ordinal_inference', type=str, default='modal', choices=['modal', 'entropy', 'hybrid'], help='Ordinal inference method: modal, entropy, or hybrid (default: modal)')
        parser.add_argument('--ordinal_model_type', type=str, default='ordered_logistic', choices=['ordered_logistic', 'dirichlet'], help='Hierarchical model type for ordinal data (default: ordered_logistic)')
        parser.add_argument('--continuous_tasks', type=str, default=None, help='Comma-separated list of substrings to identify continuous bounded groupings (e.g., "mean_score,aggregated")')
        parser.add_argument('--entropy_threshold', type=float, default=0.8, help='Proportion of max entropy for false peak detection in hybrid mode (default: 0.8)')
        parser.add_argument('--prior_mu', type=float, default=0.0, help='Centre of group-level Normal prior on logit scale (default: 0.0 = 50%% probability). Positive values bias toward higher performance, negative toward lower.')
        parser.add_argument('--prior_sigma', type=float, default=None, help='Scale of group-level Normal prior on logit scale. If not set, uses pathway-specific defaults (binary/continuous: 1.5, ordinal: 2.0). If set, applies to all pathways.')
        parser.add_argument('--disable_gpu', action='store_true', help='Disable GPU acceleration even if available')
        parser.add_argument('--force_gpu', action='store_true', help='Force GPU usage (will fail if GPU unavailable)')
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
            'low_performance_threshold': args.low_performance_threshold,
        }
        if args.random_seed is not None:
            params['random_seed'] = args.random_seed

        # Handle GPU settings
        if args.disable_gpu:
            params['use_gpu'] = False
        elif args.force_gpu:
            params['use_gpu'] = True
            params['force_gpu'] = True

        # Parse ordinal_tasks if provided
        ordinal_tasks = None
        if args.ordinal_tasks:
            ordinal_tasks = [task.strip() for task in args.ordinal_tasks.split(',')]

        # Parse continuous_tasks if provided
        continuous_tasks = None
        if args.continuous_tasks:
            continuous_tasks = [task.strip() for task in args.continuous_tasks.split(',')]

        grouping_columns = [col.strip() for col in args.grouping_columns.split(',')] if ',' in args.grouping_columns else args.grouping_columns.strip()
        result = convergence_posthoc(
            df, params, grouping_columns, args.sample_id_column, args.epoch_column, args.score_column,
            display_progress=not args.no_progress,
            generate_diagnostics=not args.no_diagnostics,
            diagnostics_prefix=args.diagnostics_prefix,
            ordinal_tasks=ordinal_tasks,
            ordinal_max_score=args.ordinal_max_score,
            ordinal_inference=args.ordinal_inference,
            ordinal_model_type=args.ordinal_model_type,
            continuous_tasks=continuous_tasks,
            entropy_threshold=args.entropy_threshold,
            prior_mu=args.prior_mu,
            prior_sigma=args.prior_sigma
        )
        result.to_csv(args.output, index=False)
        print(f"Convergence stats saved to {args.output}")
    close_all_log_handlers() 