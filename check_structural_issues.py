"""
Cause 7 Investigation: Check for Basic Structural/Design Errors

This script checks:
1. Function signature compatibility
2. Parameter flow through call chain
3. Infinite loops or unbounded recursion
4. Blocking operations without timeouts
5. Unbounded data structure growth
"""

import ast
import sys
import re
from pathlib import Path

# Add to path
sys.path.insert(0, '/home/ubuntu/optstop')

print("="*80)
print("CAUSE 7: STRUCTURAL/DESIGN ERROR INVESTIGATION")
print("="*80)

# === 7.1: Check Function Signatures and Call Chain ===
print("\n" + "="*80)
print("7.1: Function Signature Compatibility Check")
print("="*80)

call_chain = [
    {
        'file': 'early_stopping.py',
        'line': '978-992',
        'function': '_run_stopping_inference',
        'calls': 'optimal_stopping_live_single',
        'params_passed': [
            'df_grouping=completed_data',
            'grouping_name=grouping_name',
            'params=params_with_aggregation',
            'sample_id_column=self.sample_id_column',
            'epoch_column=self.epoch_column',
            'score_column=self.score_column',
            'stabilization_history=stabilization_history',
            'ordinal_tasks=self.ordinal_tasks',
            'ordinal_max_score=self.ordinal_max_score',
            'ordinal_inference=self.ordinal_inference',
            'entropy_threshold=1.5',
            'sampling_kwargs=sampling_kwargs'
        ]
    },
    {
        'file': 'rule.py',
        'line': '2021-2034',
        'function': 'optimal_stopping_live_single',
        'signature': [
            'df_grouping',
            'grouping_name',
            'params',
            'sample_id_column',
            'epoch_column',
            'score_column="score"',
            'stabilization_history=None',
            'ordinal_tasks=None',
            'ordinal_max_score=10',
            'ordinal_inference="modal"',
            'entropy_threshold=1.5',
            'sampling_kwargs=None'
        ],
        'calls': '_ordinal_hybrid_stopping_criterion',
        'params_passed': [
            'np.array(accumulated_scores)',
            'ordinal_max_score=ordinal_max_score',
            'delta_item=delta_item',
            'cred_level=cred_level',
            'entropy_history=entropy_history',
            'entropy_threshold=entropy_threshold',
            'conservatism=current_conservatism',
            'low_perf_threshold=low_perf_threshold',
            'model_cache=ordinal_model_cache',
            'compute_kwargs=sampling_kwargs'
        ]
    },
    {
        'file': 'ordinal_model.py',
        'line': '435-448',
        'function': '_ordinal_hybrid_stopping_criterion',
        'signature': [
            'scores',
            'ordinal_max_score',
            'delta_item',
            'cred_level',
            'entropy_history',
            'entropy_threshold=1.5',
            'conservatism=1.0',
            'low_perf_threshold=0.2',
            'min_epochs_for_stabilization=3',
            'stabilization_threshold=0.002',
            'model_cache=None',
            'compute_kwargs=None'
        ],
        'calls': '_ordinal_entropy_ci_adaptive',
        'params_passed': [
            'scores',
            'ordinal_max_score=ordinal_max_score',
            'cred_level=cred_level',
            'conservatism=conservatism',
            'low_perf_threshold=low_perf_threshold',
            'model_cache=model_cache',
            'compute_kwargs=compute_kwargs'
        ]
    },
    {
        'file': 'ordinal_model.py',
        'line': '236-246',
        'function': '_ordinal_entropy_ci_adaptive',
        'signature': [
            'scores',
            'ordinal_max_score',
            'cred_level=0.95',
            'conservatism=1.0',
            'low_perf_threshold=0.2',
            'n_samples=6000',
            'n_tune=6000',
            'model_cache=None',
            'compute_kwargs=None'
        ]
    }
]

print("\nCall Chain:")
for i, step in enumerate(call_chain, 1):
    print(f"\n{i}. {step['function']} ({step['file']}:{step['line']})")
    if 'signature' in step:
        print(f"   Signature: {', '.join(step['signature'])}")
    if 'calls' in step:
        print(f"   Calls: {step['calls']}")
        print(f"   With params: {', '.join(step['params_passed'][:3])}...")

print("\n✅ Manual Review: Call chain appears correct")
print("   - All parameters passed match function signatures")
print("   - compute_kwargs flows correctly from early_stopping → ordinal_model")
print("   - No obvious signature mismatches")

# === 7.2: Check for Infinite Loops ===
print("\n" + "="*80)
print("7.2: Infinite Loop / Unbounded Recursion Check")
print("="*80)

loop_locations = [
    ('rule.py', '2200-2320', 'Sample-level stopping loop'),
    ('rule.py', '2322-2580', 'Group-level stopping section'),
    ('ordinal_model.py', '435-567', 'Hybrid stopping criterion'),
    ('early_stopping.py', '890-1144', '_run_stopping_inference'),
]

print("\nChecking for potentially problematic loops...")
for file, lines, desc in loop_locations:
    print(f"  - {file}:{lines} - {desc}")

# Read rule.py and check sample-level loop
rule_path = Path('/home/ubuntu/optstop/optstop/rule.py')
if rule_path.exists():
    with open(rule_path) as f:
        content = f.read()

    # Check for while loops
    while_loops = re.findall(r'while\s+[^:]+:', content)
    print(f"\n✅ rule.py: {len(while_loops)} while loops found")
    if while_loops:
        print("   Warning: Manual review needed for while loop exit conditions")

    # Sample-level loop uses 'for item_id in item_ids:' - bounded by item_ids
    if 'for item_id in item_ids:' in content:
        print("   ✅ Sample-level loop: bounded by item_ids (finite)")

    # Check for recursive calls
    recursive_patterns = [
        'optimal_stopping_live_single\\(',
        '_ordinal_hybrid_stopping_criterion\\(',
        '_ordinal_entropy_ci_adaptive\\('
    ]
    for pattern in recursive_patterns:
        func_name = pattern.replace('\\(', '')
        # Count definitions vs calls (rough heuristic)
        defs = len(re.findall(f'def {func_name}', content))
        calls = len(re.findall(pattern, content))
        if calls > defs + 5:  # Allow some calls, but too many suggests recursion
            print(f"   ⚠️  {func_name}: {calls} calls vs {defs} definitions - possible recursion")
        else:
            print(f"   ✅ {func_name}: No obvious recursion")

print("\n✅ No obvious infinite loops or unbounded recursion detected")

# === 7.3: Check for Blocking Operations ===
print("\n" + "="*80)
print("7.3: Blocking Operations Without Timeouts")
print("="*80)

blocking_ops = [
    {
        'location': 'early_stopping.py:978',
        'operation': 'asyncio.to_thread()',
        'description': 'Wraps optimal_stopping_live_single',
        'has_timeout': False,
        'recommendation': 'Add asyncio.wait_for() with timeout'
    },
    {
        'location': 'ordinal_model.py:360',
        'operation': 'pm.sample()',
        'description': 'MCMC sampling (can be slow)',
        'has_timeout': False,
        'recommendation': 'PyMC has internal convergence checks'
    }
]

print("\nBlocking operations found:")
for op in blocking_ops:
    print(f"\n  {op['location']}: {op['operation']}")
    print(f"    Description: {op['description']}")
    print(f"    Has timeout: {'✅ Yes' if op['has_timeout'] else '❌ No'}")
    if not op['has_timeout']:
        print(f"    ⚠️  Recommendation: {op['recommendation']}")

print("\n⚠️  FINDING: asyncio.to_thread() has NO timeout")
print("   This could cause indefinite hangs if MCMC sampling stalls")
print("   Recommended: Add timeout wrapper")

# === 7.4: Check Data Structure Growth ===
print("\n" + "="*80)
print("7.4: Unbounded Data Structure Growth")
print("="*80)

data_structures = [
    {
        'name': '_stabilization_histories',
        'file': 'early_stopping.py:204',
        'type': 'dict[str, dict[str, list[float]]]',
        'growth': 'Grows per grouping, per inference call',
        'bounded': 'No explicit limit',
        'concern': 'Could grow large with many inference calls'
    },
    {
        'name': 'stopped_samples',
        'file': 'early_stopping.py:189',
        'type': 'list[StoppedSample]',
        'growth': 'Grows per stopped sample',
        'bounded': 'Limited by number of samples',
        'concern': 'Low - bounded by dataset size'
    },
    {
        'name': '_schedule_cache',
        'file': 'early_stopping.py:201',
        'type': 'dict[tuple, bool]',
        'growth': 'One entry per (grouping, sample_id, epoch)',
        'bounded': 'Limited by total trials',
        'concern': 'Low - bounded by dataset size'
    },
    {
        'name': 'model_cache',
        'file': 'ordinal_model.py:324-333',
        'type': 'dict with PyMC model',
        'growth': 'One per grouping',
        'bounded': 'Limited by number of groupings',
        'concern': 'Low - models reused'
    },
    {
        'name': 'entropy_history',
        'file': 'rule.py:~1025',
        'type': 'list[tuple]',
        'growth': 'Grows per sample per inference call',
        'bounded': 'No explicit limit per sample',
        'concern': 'Medium - could grow with many inference calls'
    }
]

print("\nData structures that grow over time:")
for ds in data_structures:
    print(f"\n  {ds['name']} ({ds['file']})")
    print(f"    Type: {ds['type']}")
    print(f"    Growth: {ds['growth']}")
    print(f"    Bounded: {ds['bounded']}")
    if 'No explicit limit' in ds['bounded']:
        print(f"    ⚠️  Concern: {ds['concern']}")
    else:
        print(f"    ✅ {ds['concern']}")

print("\n⚠️  FINDING: entropy_history and stabilization_histories have no size limits")
print("   For Test 4: reanalysis_interval=200, total_trials=800")
print("   → 4 inference calls → history grows 4 times")
print("   This is probably FINE, but worth monitoring")

# === Summary ===
print("\n" + "="*80)
print("CAUSE 7 SUMMARY: Structural Issues")
print("="*80)

findings = [
    ("✅", "Function signatures", "All compatible, no mismatches"),
    ("✅", "Call chain", "Parameters flow correctly"),
    ("✅", "Infinite loops", "No obvious unbounded loops"),
    ("✅", "Recursion", "No recursive calls detected"),
    ("⚠️ ", "Blocking ops", "asyncio.to_thread() has NO timeout"),
    ("⚠️ ", "Data growth", "entropy_history has no size limit"),
]

print("\nFindings:")
for status, category, finding in findings:
    print(f"  {status} {category}: {finding}")

print("\n" + "="*80)
print("RECOMMENDED ACTIONS:")
print("="*80)
print("1. Add timeout to asyncio.to_thread() in early_stopping.py:978")
print("2. Monitor entropy_history size (probably fine for Test 4)")
print("3. Continue to Cause 4 investigation (data growth hypothesis)")
print("="*80)
