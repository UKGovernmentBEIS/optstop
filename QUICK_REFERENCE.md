# Quick Reference: GPU/CPU Auto-Adaptation

## What Changed

**Before**: Package would cause GPU OOM when users specified `chains=4, cores=4`

**After**: Package automatically decides GPU vs CPU and adjusts parameters to prevent OOM

## For Users

### No Code Changes Needed!

Your existing code works as-is:

```python
params = {
    'chains': 4,
    'cores': 4,
    'draws': 3000,
    'tune': 3000,
    # ... other params
}

pruned_df, summary = optimal_stopping_posthoc(
    df, params,
    grouping_columns=['subject', 'task'],
    sample_id_column='item_id',
    epoch_column='trial_num'
)
```

### What Happens Automatically

1. **Package detects**: GPUs, memory, CPU cores, workload size
2. **Package decides**: GPU or CPU based on intelligent analysis
3. **Package adjusts**: chains/cores to prevent OOM
4. **Package logs**: Decision reasoning for transparency

### Check Logs to See Decision

Look for lines like:
```
Auto-decision: Using CPU instead of GPU (workload analysis)
Worker processing grouping 0 with CPU-only (chains=4)
```

or

```
Auto-decision: Using GPU with optimized parameters (chains=1, cores=1)
Worker processing grouping 0 with GPU acceleration (jax-gpu, chains=1)
```

### Manual Override (if needed)

Force CPU (disable GPU):
```python
params = {
    'chains': 4,
    'cores': 4,
    'use_gpu': False,  # Force CPU
    # ... other params
}
```

Force GPU (not recommended unless you understand memory requirements):
```python
params = {
    'chains': 1,  # Must use 1 for GPU to prevent OOM
    'cores': 1,
    'use_gpu': True,  # Force GPU
    # ... other params
}
```

## Decision Logic Summary

The package chooses CPU multiprocessing if:
- GPU memory insufficient for workload
- Workload too small to benefit from GPU overhead
- Many parallel groupings (CPU cores handle better)
- User specified high chains + many groupings

The package chooses GPU acceleration if:
- Large workload (many samples)
- Sufficient GPU memory
- Few parallel tasks
- Single GPU scenario
- **Automatically reduces chains to 1** to fit in GPU memory

## Testing

Test the fix:
```bash
cd /home/ubuntu/optstop
python test_gpu_adaptation.py
```

## Questions?

- Check `GPU_ADAPTATION_FIX.md` for detailed explanation
- Check your log files for decision reasoning
- The package now "just works" - no GPU OOM errors!
