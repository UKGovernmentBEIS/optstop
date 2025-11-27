#!/usr/bin/env python3
"""Add timing and diagnostics to all comprehensive tests."""

import re

# Read the test file
with open('tests/test_early_stopping_comprehensive.py', 'r') as f:
    content = f.read()

# Define test metadata for diagnostics
test_metadata = [
    ("test_binary_discrete_consistent_high_50_samples", "Binary Discrete"),
    ("test_binary_discrete_variable_performance_100_samples", "Binary Discrete"),
    ("test_ordinal_discrete_consistent_high_60_samples", "Ordinal Discrete (Modal)"),
    ("test_ordinal_discrete_bimodal_80_samples", "Ordinal Discrete (Modal)"),
    ("test_binary_aggregated_consistent_75_samples", "Binary Aggregated (Continuous [0,1])"),
    ("test_binary_aggregated_low_variance_50_samples", "Binary Aggregated (Continuous [0,1])"),
    ("test_ordinal_aggregated_consistent_60_samples", "Ordinal Aggregated (Continuous [0,10])"),
    ("test_ordinal_aggregated_high_variance_70_samples", "Ordinal Aggregated (Continuous [0,10])"),
    ("test_all_samples_identical_binary_aggregated", "Binary Aggregated (Continuous [0,1])"),
    ("test_single_epoch_binary_aggregated", "Binary Aggregated (Continuous [0,1])"),
    ("test_multi_grouping_mixed_performance", "Binary Aggregated (Continuous [0,1])"),
    ("test_learning_curve_binary_aggregated", "Binary Aggregated (Continuous [0,1])"),
    ("test_large_scale_150_samples", "Binary Aggregated (Continuous [0,1])"),
]

for test_func, score_type in test_metadata:
    # Add start_time and score_type after np.random.seed
    pattern = f"(async def {test_func}.*?np\\.random\\.seed\\(\\d+\\))"
    replacement = f"\\1\\n        start_time = time.time()\\n        score_type = \\\"{score_type}\\\""
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    # Add timing output before final assertions (find first print statement in results section)
    # Look for pattern: print(f"✓  or print(f"✓
    old_print_pattern = f"(# (Finalize|Complete all tasks).*?)(print\\(f\\\"✓)"

    timing_block = f"""\\1\\n        # Timing and diagnostics
        elapsed_time = time.time() - start_time

        \\3"""

    content = re.sub(old_print_pattern, timing_block, content, flags=re.DOTALL, count=1)

# Write back
with open('tests/test_early_stopping_comprehensive.py', 'w') as f:
    f.write(content)

print("Timing and diagnostics added to all tests!")
