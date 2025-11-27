#!/usr/bin/env python3
"""
Test script to verify GPU/CPU auto-adaptation logic works correctly.
This simulates the scenario where user specifies chains=4, cores=4 and
the package should intelligently decide whether to use GPU or CPU.
"""

import pandas as pd
import numpy as np
from optstop import optimal_stopping_posthoc, configure_optstop_logging

def create_test_data(n_groupings=4, n_items=10, n_epochs=8):
    """Create synthetic test data"""
    np.random.seed(42)
    data = []

    for group in range(n_groupings):
        for item in range(n_items):
            base_rate = 0.5 + 0.1 * group
            for epoch in range(n_epochs):
                score = np.random.binomial(1, base_rate)
                data.append({
                    'grouping_id': f'group_{group}',
                    'item_id': f'item_{item}',
                    'epoch': epoch + 1,
                    'score': score
                })

    return pd.DataFrame(data)

def test_auto_adaptation():
    """Test that GPU/CPU auto-adaptation works correctly"""

    print("=" * 70)
    print("Testing GPU/CPU Auto-Adaptation Logic")
    print("=" * 70)

    # Configure logging to see decision-making
    configure_optstop_logging('test_gpu_adaptation.log', console_output=True)

    # Create test data
    print("\n1. Creating test dataset...")
    df = create_test_data(n_groupings=4, n_items=10, n_epochs=8)
    print(f"   Dataset: {len(df)} rows, {df['grouping_id'].nunique()} groupings")

    # User specifies chains=4, cores=4 (which would cause OOM on GPU)
    print("\n2. User params: chains=4, cores=4")
    params = {
        'chains': 4,
        'cores': 4,
        'draws': 200,  # Small for testing
        'tune': 200,
        'delta_item': 0.2,
        'delta_cap': 0.2,
        'cred_level': 0.95,
        'random_seed': 42
    }

    print("\n3. Running optimal_stopping_posthoc...")
    print("   The package should automatically:")
    print("   - Detect GPU availability")
    print("   - Analyze workload (chains=4, num_groupings=4)")
    print("   - Decide GPU vs CPU based on memory & workload")
    print("   - If GPU: reduce chains to 1 to prevent OOM")
    print("   - If CPU: keep chains=4 for multiprocessing")

    try:
        pruned_df, summary = optimal_stopping_posthoc(
            df=df,
            params=params,
            grouping_columns='grouping_id',
            sample_id_column='item_id',
            epoch_column='epoch',
            score_column='score',
            display_progress=True
        )

        print(f"\n4. Results:")
        print(f"   Original rows: {len(df)}")
        print(f"   Pruned rows: {len(pruned_df)}")
        print(f"   Groupings processed: {len(summary)}")
        print(f"\n✅ SUCCESS: No GPU OOM errors!")
        print(f"   Check test_gpu_adaptation.log for detailed decision-making")

        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        print(f"   Check test_gpu_adaptation.log for details")
        return False

if __name__ == "__main__":
    success = test_auto_adaptation()
    exit(0 if success else 1)
