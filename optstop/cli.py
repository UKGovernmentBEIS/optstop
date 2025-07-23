"""
Command-line interface for the optsampling package.
"""

import argparse
import pandas as pd
from .rule import configure_optsampling_logging, optimal_stopping_posthoc, optimal_stopping_live
from .convergence import convergence_posthoc

def main():
    parser = argparse.ArgumentParser(description="Run post-hoc optimal stopping analysis.")
    parser.add_argument('--csv', required=True, help='Path to input CSV file')
    parser.add_argument('--output', required=True, help='Path to output pruned CSV file')
    parser.add_argument('--summary', required=False, help='Path to output summary CSV file')
    parser.add_argument('--log', default='optsampling_cli.log', help='Path to log file')
    parser.add_argument('--delta_item', type=float, default=0.05)
    parser.add_argument('--delta_cap', type=float, default=0.05)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--tune', type=int, default=1000)
    parser.add_argument('--CI_delta', type=float, default=0.0002)
    parser.add_argument('--conservatism', type=float, default=2)
    parser.add_argument('--rep_batch_size', type=int, default=1)
    parser.add_argument('--pymc_refresh_every', type=int, default=2)
    parser.add_argument('--stab_window', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=None)
    args = parser.parse_args()

    configure_optsampling_logging(args.log)
    df = pd.read_csv(args.csv)
    params = {
        'delta_item': args.delta_item,
        'delta_cap': args.delta_cap,
        'draws': args.draws,
        'tune': args.tune,
        'CI_delta': args.CI_delta,
        'conservatism': args.conservatism,
        'rep_batch_size': args.rep_batch_size,
        'pymc_refresh_every': args.pymc_refresh_every,
        'stab_window': args.stab_window,
    }
    if args.random_seed is not None:
        params['random_seed'] = args.random_seed
    pruned_df, summary = optimal_stopping_posthoc(df, params)
    pruned_df.to_csv(args.output, index=False)
    if args.summary:
        pd.DataFrame(summary).to_csv(args.summary, index=False)
    print(f"Pruned data saved to {args.output}")
    if args.summary:
        print(f"Summary saved to {args.summary}")

def main_live():
    parser = argparse.ArgumentParser(description="Run live optimal stopping analysis.")
    parser.add_argument('--csv', required=True, help='Path to input CSV file (current data)')
    parser.add_argument('--log', default='optsampling_live.log', help='Path to log file')
    parser.add_argument('--delta_item', type=float, default=0.05)
    parser.add_argument('--delta_cap', type=float, default=0.05)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--tune', type=int, default=1000)
    parser.add_argument('--CI_delta', type=float, default=0.0002)
    parser.add_argument('--conservatism', type=float, default=2)
    parser.add_argument('--rep_batch_size', type=int, default=1)
    parser.add_argument('--pymc_refresh_every', type=int, default=2)
    parser.add_argument('--stab_window', type=int, default=5)
    parser.add_argument('--random_seed', type=int, default=None)
    args = parser.parse_args()

    configure_optsampling_logging(args.log)
    df = pd.read_csv(args.csv)
    params = {
        'delta_item': args.delta_item,
        'delta_cap': args.delta_cap,
        'draws': args.draws,
        'tune': args.tune,
        'CI_delta': args.CI_delta,
        'conservatism': args.conservatism,
        'rep_batch_size': args.rep_batch_size,
        'pymc_refresh_every': args.pymc_refresh_every,
        'stab_window': args.stab_window,
    }
    if args.random_seed is not None:
        params['random_seed'] = args.random_seed
    result = optimal_stopping_live(df, params)
    print("Sample IDs to stop:", result['stop_sample_ids'])
    print("Stop task/grouping?", result['stop_task'])


def main_convergence():
    parser = argparse.ArgumentParser(description="Run post-hoc convergence analysis.")
    parser.add_argument('--csv', required=True, help='Path to input CSV file')
    parser.add_argument('--output', required=True, help='Path to output convergence stats CSV file')
    parser.add_argument('--log', default='optsampling_convergence.log', help='Path to log file')
    parser.add_argument('--delta_item', type=float, default=0.05)
    parser.add_argument('--delta_cap', type=float, default=0.05)
    parser.add_argument('--draws', type=int, default=1000)
    parser.add_argument('--tune', type=int, default=1000)
    parser.add_argument('--CI_delta', type=float, default=0.0002)
    parser.add_argument('--conservatism', type=float, default=2)
    parser.add_argument('--rep_batch_size', type=int, default=1)
    parser.add_argument('--pymc_refresh_every', type=int, default=2)
    parser.add_argument('--stab_window', type=int, default=5)
    parser.add_argument('--item_seqs', type=int, default=20)
    parser.add_argument('--epoch_seqs', type=int, default=20)
    parser.add_argument('--random_seed', type=int, default=None)
    args = parser.parse_args()

    configure_optsampling_logging(args.log)
    df = pd.read_csv(args.csv)
    params = {
        'delta_item': args.delta_item,
        'delta_cap': args.delta_cap,
        'draws': args.draws,
        'tune': args.tune,
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
    result = convergence_posthoc(df, params)
    result.to_csv(args.output, index=False)
    print(f"Convergence stats saved to {args.output}") 