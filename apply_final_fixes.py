#!/usr/bin/env python3
"""Apply all fixes to comprehensive test suite cleanly."""

import re

# Read file
with open('tests/test_early_stopping_comprehensive.py', 'r') as f:
    content = f.read()

# 1. Add time import
if 'import time' not in content:
    content = content.replace('from typing import List, Any', 'from typing import List, Any\nimport time')

# 2. Fix efficiency_metrics -> efficiency_percent
content = content.replace("'efficiency_metrics'", "'efficiency_percent'")
content = content.replace('["efficiency_metrics"]', '["efficiency_percent"]')
content = content.replace("['efficiency_metrics']", "['efficiency_percent']")

# 3. Update reanalysis_intervals (preserve commas where they exist)
replacements = [
    (r'(test_binary_discrete_consistent_high_50_samples.*?reanalysis_interval=)30(,)', r'\g<1>188\g<2>  # 25%, 50%, 75% of 750 trials'),
    (r'(test_binary_discrete_variable_performance_100_samples.*?reanalysis_interval=)30([,\s])', r'\g<1>300\g<2>  # 25%, 50%, 75% of 1200 trials'),
    (r'(test_ordinal_discrete_consistent_high_60_samples.*?reanalysis_interval=)30(,)', r'\g<1>225\g<2>  # 25%, 50%, 75% of 900 trials'),
    (r'(test_ordinal_discrete_bimodal_80_samples.*?reanalysis_interval=)30(,)', r'\g<1>200\g<2>  # 25%, 50%, 75% of 800 trials'),
    (r'(test_binary_aggregated_consistent_75_samples.*?reanalysis_interval=)30([,\s])', r'\g<1>225\g<2>  # 25%, 50%, 75% of 900 trials'),
    (r'(test_binary_aggregated_low_variance_50_samples.*?reanalysis_interval=)30([,\s])', r'\g<1>250\g<2>  # 25%, 50%, 75% of 1000 trials'),
    (r'(test_ordinal_aggregated_consistent_60_samples.*?reanalysis_interval=)30(,)', r'\g<1>225\g<2>  # 25%, 50%, 75% of 900 trials'),
    (r'(test_ordinal_aggregated_high_variance_70_samples.*?reanalysis_interval=)30(,)', r'\g<1>175\g<2>  # 25%, 50%, 75% of 700 trials'),
    (r'(test_all_samples_identical_binary_aggregated.*?reanalysis_interval=)30([,\s])', r'\g<1>150\g<2>  # 25%, 50%, 75% of 600 trials'),
    (r'(test_single_epoch_binary_aggregated.*?reanalysis_interval=)30([,\s])', r'\g<1>10\g<2>  # 33%, 67%, 100% of 30 trials'),
    (r'(test_multi_grouping_mixed_performance.*?reanalysis_interval=)30([,\s])', r'\g<1>150\g<2>  # 25%, 50%, 75% of 600 trials'),
    (r'(test_learning_curve_binary_aggregated.*?reanalysis_interval=)30([,\s])', r'\g<1>250\g<2>  # 25%, 50%, 75% of 1000 trials'),
    (r'(test_large_scale_150_samples.*?reanalysis_interval=)30(.*?# Reduced.*?performance)', r'\g<1>563\g<2> -> 25%, 50%, 75% of 2250 trials'),
]

for pattern, replacement in replacements:
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write back
with open('tests/test_early_stopping_comprehensive.py', 'w') as f:
    f.write(content)

print("✓ All fixes applied successfully!")
print("  - Time module added")
print("  - efficiency_metrics -> efficiency_percent")
print(f"  - {len([r for r in replacements if 'reanalysis_interval' in r[0]])} reanalysis_intervals updated")
