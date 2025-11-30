"""
Generate three large-scale test datasets for bridge testing.

Each dataset has:
- 500 sample_IDs × 10 epochs = 5,000 trials
- 5 groupings with distinct performance characteristics
- Realistic variance both within and across samples

Datasets:
1. Binary discrete (single binary score per trial)
2. Ordinal discrete (multiple ordinal scores, user selects one via score_choice)
3. Ordinal discrete → Continuous bounded (multiple ordinal scores, score_agg='mean')
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_grouping_performance_parameters():
    """
    Define performance parameters for 5 distinct groupings.

    Returns dict with grouping names and their performance characteristics.
    """
    return {
        'high_performer': {
            'base_prob': 0.95,  # 95% base success rate
            'within_sample_std': 0.02,  # Very low variance within sample
            'across_sample_std': 0.03,  # Very low variance across samples
            'model': 'gpt-4-turbo',
            'task': 'math_easy'
        },
        'low_performer': {
            'base_prob': 0.01,  # 1% base success rate
            'within_sample_std': 0.005,  # Consistently low
            'across_sample_std': 0.01,
            'model': 'gpt-3.5',
            'task': 'math_hard'
        },
        'moderate_low_var': {
            'base_prob': 0.65,  # 65% base success rate
            'within_sample_std': 0.05,  # Low variance within sample
            'across_sample_std': 0.08,  # Low variance across samples
            'model': 'claude-3-opus',
            'task': 'coding_medium'
        },
        'moderate_med_var': {
            'base_prob': 0.60,  # 60% base success rate
            'within_sample_std': 0.12,  # Medium variance within sample
            'across_sample_std': 0.15,  # Medium variance across samples
            'model': 'gemini-pro',
            'task': 'reasoning'
        },
        'moderate_high_var': {
            'base_prob': 0.55,  # 55% base success rate
            'within_sample_std': 0.20,  # High variance within sample
            'across_sample_std': 0.25,  # High variance across samples
            'model': 'llama-70b',
            'task': 'creative_writing'
        }
    }


def generate_binary_score(base_prob, within_std, across_std, sample_seed, epoch):
    """
    Generate a binary score (0 or 1) with realistic variance.

    Args:
        base_prob: Base success probability for the grouping
        within_std: Standard deviation for within-sample variance
        across_std: Standard deviation for across-sample variance
        sample_seed: Seed for this specific sample (for across-sample variance)
        epoch: Current epoch (for within-sample variance)

    Returns:
        int: Binary score (0 or 1)
    """
    # Sample-specific adjustment (across-sample variance)
    np.random.seed(sample_seed)
    sample_adjustment = np.random.normal(0, across_std)

    # Epoch-specific adjustment (within-sample variance)
    np.random.seed(sample_seed + epoch * 1000)
    epoch_adjustment = np.random.normal(0, within_std)

    # Combined probability for this trial
    trial_prob = np.clip(base_prob + sample_adjustment + epoch_adjustment, 0, 1)

    # Generate binary outcome
    return int(np.random.random() < trial_prob)


def generate_ordinal_score(base_prob, within_std, across_std, sample_seed, epoch, max_score=10):
    """
    Generate an ordinal score (0 to max_score) based on underlying probability.

    Maps probability to ordinal scale:
    - High probability → high ordinal scores
    - Low probability → low ordinal scores
    - Adds noise to create realistic variance

    Args:
        base_prob: Base success probability for the grouping
        within_std: Standard deviation for within-sample variance
        across_std: Standard deviation for across-sample variance
        sample_seed: Seed for this specific sample
        epoch: Current epoch
        max_score: Maximum ordinal value (default: 10)

    Returns:
        int: Ordinal score between 0 and max_score
    """
    # Sample-specific adjustment
    np.random.seed(sample_seed)
    sample_adjustment = np.random.normal(0, across_std)

    # Epoch-specific adjustment
    np.random.seed(sample_seed + epoch * 1000)
    epoch_adjustment = np.random.normal(0, within_std)

    # Combined probability
    trial_prob = np.clip(base_prob + sample_adjustment + epoch_adjustment, 0, 1)

    # Map probability to ordinal score (with noise)
    # Base ordinal value from probability
    base_ordinal = trial_prob * max_score

    # Add noise (proportional to variance)
    noise = np.random.normal(0, (within_std + across_std) * max_score / 2)

    # Final ordinal score
    ordinal_score = int(np.clip(base_ordinal + noise, 0, max_score))

    return ordinal_score


def generate_dataset_1_binary_discrete(output_path: Path):
    """
    Generate Dataset 1: Binary Discrete Scores

    Structure:
    - Single binary score per trial (0 or 1)
    - No aggregation needed
    - Routes to binary discrete inference
    """
    print("\n" + "="*80)
    print("GENERATING DATASET 1: Binary Discrete")
    print("="*80)

    groupings = generate_grouping_performance_parameters()
    num_samples_per_group = 100  # 500 total / 5 groups
    num_epochs = 10

    data = []
    sample_id_counter = 0

    for group_name, params in groupings.items():
        print(f"\n  Generating grouping: {group_name}")
        print(f"    Model: {params['model']}, Task: {params['task']}")
        print(f"    Base prob: {params['base_prob']:.2%}, Within-std: {params['within_sample_std']:.3f}, Across-std: {params['across_sample_std']:.3f}")

        for sample_idx in range(num_samples_per_group):
            sample_id = f"sample_{sample_id_counter:04d}"
            sample_seed = sample_id_counter * 12345  # Consistent seed per sample

            for epoch in range(1, num_epochs + 1):  # 1-indexed epochs (1-10)
                score = generate_binary_score(
                    params['base_prob'],
                    params['within_sample_std'],
                    params['across_sample_std'],
                    sample_seed,
                    epoch
                )

                data.append({
                    'sample_id': sample_id,
                    'epoch': epoch,
                    'model': params['model'],
                    'task': params['task'],
                    'score': score
                })

            sample_id_counter += 1

    # Create DataFrame
    df = pd.DataFrame(data)

    # Calculate statistics per grouping
    print("\n  Dataset Statistics:")
    for group_name, params in groupings.items():
        mask = (df['model'] == params['model']) & (df['task'] == params['task'])
        group_df = df[mask]
        mean_score = group_df['score'].mean()
        std_score = group_df['score'].std()
        print(f"    {group_name}: Mean={mean_score:.3f}, Std={std_score:.3f}, N={len(group_df)}")

    # Save
    df.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved to: {output_path}")
    print(f"    Total rows: {len(df)}")
    print(f"    Unique samples: {df['sample_id'].nunique()}")
    print(f"    Unique groupings: {df.groupby(['model', 'task']).ngroups}")

    return df


def generate_dataset_2_ordinal_discrete_with_choice(output_path: Path):
    """
    Generate Dataset 2: Ordinal Discrete with score_choice

    Structure:
    - Multiple ordinal scores per trial (accuracy, precision, recall)
    - User specifies score_choice='accuracy'
    - Routes to ordinal discrete inference
    """
    print("\n" + "="*80)
    print("GENERATING DATASET 2: Ordinal Discrete (with score_choice)")
    print("="*80)

    groupings = generate_grouping_performance_parameters()
    num_samples_per_group = 100
    num_epochs = 10

    data = []
    sample_id_counter = 0

    for group_name, params in groupings.items():
        print(f"\n  Generating grouping: {group_name}")
        print(f"    Model: {params['model']}, Task: {params['task']}")

        for sample_idx in range(num_samples_per_group):
            sample_id = f"sample_{sample_id_counter:04d}"
            sample_seed = sample_id_counter * 12345

            for epoch in range(1, num_epochs + 1):  # 1-indexed epochs (1-10)
                # Generate multiple ordinal scores (correlated but not identical)
                accuracy = generate_ordinal_score(
                    params['base_prob'], params['within_sample_std'],
                    params['across_sample_std'], sample_seed, epoch
                )

                # Precision: slightly higher variance
                precision = generate_ordinal_score(
                    params['base_prob'] - 0.05, params['within_sample_std'] * 1.2,
                    params['across_sample_std'] * 1.1, sample_seed + 100, epoch
                )

                # Recall: slightly different distribution
                recall = generate_ordinal_score(
                    params['base_prob'] + 0.03, params['within_sample_std'] * 0.9,
                    params['across_sample_std'] * 1.05, sample_seed + 200, epoch
                )

                data.append({
                    'sample_id': sample_id,
                    'epoch': epoch,
                    'model': params['model'],
                    'task': params['task'],
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall
                })

            sample_id_counter += 1

    # Create DataFrame
    df = pd.DataFrame(data)

    # Calculate statistics per grouping (for primary score: accuracy)
    print("\n  Dataset Statistics (accuracy score):")
    for group_name, params in groupings.items():
        mask = (df['model'] == params['model']) & (df['task'] == params['task'])
        group_df = df[mask]
        mean_score = group_df['accuracy'].mean()
        std_score = group_df['accuracy'].std()
        print(f"    {group_name}: Mean={mean_score:.2f}, Std={std_score:.2f}, N={len(group_df)}")

    # Save
    df.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved to: {output_path}")
    print(f"    Total rows: {len(df)}")
    print(f"    Unique samples: {df['sample_id'].nunique()}")
    print(f"    Score columns: {['accuracy', 'precision', 'recall']}")
    print(f"    NOTE: User should specify score_choice='accuracy'")

    return df


def generate_dataset_3_ordinal_continuous_with_agg(output_path: Path):
    """
    Generate Dataset 3: Ordinal Discrete → Continuous Bounded (with score_agg='mean')

    Structure:
    - Multiple ordinal scores per trial (scorer_1, scorer_2, scorer_3)
    - User specifies score_agg='mean'
    - Routes to hierarchical Beta model (continuous bounded)
    """
    print("\n" + "="*80)
    print("GENERATING DATASET 3: Ordinal Discrete → Continuous Bounded (score_agg='mean')")
    print("="*80)

    groupings = generate_grouping_performance_parameters()
    num_samples_per_group = 100
    num_epochs = 10

    data = []
    sample_id_counter = 0

    for group_name, params in groupings.items():
        print(f"\n  Generating grouping: {group_name}")
        print(f"    Model: {params['model']}, Task: {params['task']}")

        for sample_idx in range(num_samples_per_group):
            sample_id = f"sample_{sample_id_counter:04d}"
            sample_seed = sample_id_counter * 12345

            for epoch in range(1, num_epochs + 1):  # 1-indexed epochs (1-10)
                # Generate 3 ordinal scores from different "scorers"
                # These will be averaged by the bridge (score_agg='mean')
                scorer_1 = generate_ordinal_score(
                    params['base_prob'], params['within_sample_std'],
                    params['across_sample_std'], sample_seed, epoch
                )

                scorer_2 = generate_ordinal_score(
                    params['base_prob'], params['within_sample_std'] * 1.1,
                    params['across_sample_std'] * 0.95, sample_seed + 300, epoch
                )

                scorer_3 = generate_ordinal_score(
                    params['base_prob'], params['within_sample_std'] * 0.9,
                    params['across_sample_std'] * 1.05, sample_seed + 600, epoch
                )

                data.append({
                    'sample_id': sample_id,
                    'epoch': epoch,
                    'model': params['model'],
                    'task': params['task'],
                    'scorer_1': scorer_1,
                    'scorer_2': scorer_2,
                    'scorer_3': scorer_3
                })

            sample_id_counter += 1

    # Create DataFrame
    df = pd.DataFrame(data)

    # Calculate statistics per grouping (for mean of scores)
    print("\n  Dataset Statistics (mean of scorer_1, scorer_2, scorer_3):")
    df['mean_score'] = df[['scorer_1', 'scorer_2', 'scorer_3']].mean(axis=1)

    for group_name, params in groupings.items():
        mask = (df['model'] == params['model']) & (df['task'] == params['task'])
        group_df = df[mask]
        mean_score = group_df['mean_score'].mean()
        std_score = group_df['mean_score'].std()
        print(f"    {group_name}: Mean={mean_score:.2f}, Std={std_score:.2f}, N={len(group_df)}")

    # Drop temporary mean_score column
    df = df.drop(columns=['mean_score'])

    # Save
    df.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved to: {output_path}")
    print(f"    Total rows: {len(df)}")
    print(f"    Unique samples: {df['sample_id'].nunique()}")
    print(f"    Score columns: {['scorer_1', 'scorer_2', 'scorer_3']}")
    print(f"    NOTE: User should specify score_agg='mean' and ordinal_max_score=10")

    return df


def main():
    """Generate all three test datasets."""
    print("\n" + "="*80)
    print("LARGE-SCALE TEST DATASET GENERATION")
    print("="*80)
    print("\nConfiguration:")
    print("  • 500 samples × 10 epochs = 5,000 trials per dataset")
    print("  • 5 groupings per dataset with distinct performance profiles")
    print("  • Realistic within-sample and across-sample variance")
    print("\nGrouping Performance Profiles:")
    print("  1. High performer: 95% base, low variance")
    print("  2. Low performer: 1% base, very low variance")
    print("  3. Moderate low-var: 65% base, low variance")
    print("  4. Moderate med-var: 60% base, medium variance")
    print("  5. Moderate high-var: 55% base, high variance")

    # Create output directory
    output_dir = Path('/home/ubuntu/optstop/test_data/large_scale')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate datasets
    dataset1 = generate_dataset_1_binary_discrete(
        output_dir / 'dataset_1_binary_discrete.csv'
    )

    dataset2 = generate_dataset_2_ordinal_discrete_with_choice(
        output_dir / 'dataset_2_ordinal_with_choice.csv'
    )

    dataset3 = generate_dataset_3_ordinal_continuous_with_agg(
        output_dir / 'dataset_3_ordinal_with_agg.csv'
    )

    # Final summary
    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETE")
    print("="*80)
    print(f"\nAll datasets saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review dataset statistics above")
    print("  2. Run datasets through bridge protocol")
    print("  3. Analyze bridge performance and diagnostics")
    print("\nExpected routing:")
    print("  • Dataset 1 → Binary discrete inference")
    print("  • Dataset 2 → Ordinal discrete inference (score_choice='accuracy')")
    print("  • Dataset 3 → Hierarchical Beta inference (score_agg='mean')")


if __name__ == '__main__':
    main()
