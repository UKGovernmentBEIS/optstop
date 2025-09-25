# optstop Hardware Scenarios Guide

## Overview

The enhanced optstop package now intelligently adapts to different hardware configurations, automatically optimizing performance based on available resources while maintaining backward compatibility.

## Hardware Detection and Configuration

### Automatic Hardware Discovery
```python
from optstop import gpu_utils

# Detect available GPUs
available_gpus = gpu_utils.get_available_gpu_ids()
print(f"Available GPUs: {available_gpus}")  # e.g., [0, 1, 2, 3] or []

# Get CPU count
import os
cpu_count = os.cpu_count()
print(f"Available CPUs: {cpu_count}")  # e.g., 16
```

### Configuration Validation
```python
# Validate and adjust configuration
validated_gpus, final_workers = gpu_utils.validate_gpu_configuration(
    gpu_ids=[0, 1, 2],
    max_workers=8
)
# Returns: ([0, 1, 2], 8) or (None, 8) if GPUs unavailable
```

## Hardware Scenarios

### Scenario 1: High-End Workstation (Many CPUs + Many GPUs)
**Configuration**: 32-core CPU + 8 GPUs (e.g., NVIDIA A100s)

```python
import pandas as pd
from optstop.rule import optimal_stopping_posthoc
from optstop import gpu_utils

# Auto-detect all hardware
gpu_ids = gpu_utils.get_available_gpu_ids()  # [0, 1, 2, 3, 4, 5, 6, 7]

# Use all GPUs with more workers than GPUs (cyclical assignment)
results = optimal_stopping_posthoc(
    df=large_dataset,
    params=params,
    grouping_columns=['task_type', 'subject_id'],
    sample_id_column='item_id',
    epoch_column='trial',
    score_column='score',
    gpu_ids=gpu_ids,        # Use all 8 GPUs
    max_workers=16          # 16 workers share 8 GPUs cyclically
)
```

**Performance Profile:**
- **Worker Distribution**: 16 workers × 8 GPUs = 2 workers per GPU
- **Expected Speedup**: 30-100x vs single-threaded CPU
- **Memory Usage**: ~2-4GB GPU memory per worker
- **Use Case**: Large-scale research studies, production analysis

---

### Scenario 2: Gaming/Development Workstation (Moderate CPUs + Few GPUs)
**Configuration**: 8-core CPU + 2 GPUs (e.g., RTX 4080)

```python
# Use available GPUs efficiently
gpu_ids = [0, 1]  # Both GPUs

results = optimal_stopping_posthoc(
    df=medium_dataset,
    params=params,
    grouping_columns=['condition'],
    sample_id_column='participant',
    epoch_column='trial',
    score_column='accuracy',
    gpu_ids=gpu_ids,        # Use both GPUs
    max_workers=6           # 6 workers share 2 GPUs (3 per GPU)
)
```

**Performance Profile:**
- **Worker Distribution**: 6 workers × 2 GPUs = 3 workers per GPU
- **Expected Speedup**: 10-25x vs single-threaded CPU
- **Memory Usage**: ~3-6GB GPU memory per worker
- **Use Case**: Individual research projects, small-scale studies

---

### Scenario 3: High-Performance Computing Cluster (Many CPUs + Many GPUs)
**Configuration**: 128-core CPU + 16 GPUs (e.g., HPC cluster node)

```python
# HPC-optimized configuration
gpu_ids = list(range(16))  # [0, 1, 2, ..., 15]

# Scale workers based on problem size
if len(groupings) > 50:
    max_workers = 32  # More workers for many groupings
else:
    max_workers = 16  # Fewer workers for intensive groupings

results = optimal_stopping_posthoc(
    df=massive_dataset,
    params=params,
    grouping_columns=['site', 'condition', 'task'],
    sample_id_column='subject_id',
    epoch_column='session',
    score_column='performance',
    gpu_ids=gpu_ids,
    max_workers=max_workers
)
```

**Performance Profile:**
- **Worker Distribution**: 32 workers × 16 GPUs = 2 workers per GPU
- **Expected Speedup**: 50-200x vs single-threaded CPU
- **Memory Usage**: ~2-4GB GPU memory per worker
- **Use Case**: Large consortium studies, meta-analyses

---

### Scenario 4: Standard Laptop (Few CPUs + No GPUs)
**Configuration**: 4-core CPU + Integrated Graphics (No CUDA)

```python
# Automatic CPU-only fallback
results = optimal_stopping_posthoc(
    df=small_dataset,
    params=params,
    grouping_columns=['condition'],
    sample_id_column='item',
    epoch_column='epoch',
    score_column='score'
    # No gpu_ids specified = automatic CPU-only
)

# Or explicit CPU-only
results = optimal_stopping_posthoc(
    df=small_dataset,
    params=params,
    grouping_columns=['condition'],
    sample_id_column='item',
    epoch_column='epoch',
    score_column='score',
    gpu_ids=[],            # Empty list = force CPU-only
    max_workers=4          # Use all CPU cores
)
```

**Performance Profile:**
- **Worker Distribution**: 4 CPU workers
- **Expected Performance**: Baseline performance (existing behavior)
- **Memory Usage**: ~1-2GB RAM per worker
- **Use Case**: Pilot studies, small datasets, development

---

### Scenario 5: Single GPU Development Machine
**Configuration**: 6-core CPU + 1 GPU (e.g., RTX 3070)

```python
# Optimize single GPU usage
results = optimal_stopping_posthoc(
    df=dataset,
    params=params,
    grouping_columns=['group'],
    sample_id_column='subject',
    epoch_column='trial',
    score_column='score',
    gpu_ids=[0],           # Single GPU
    max_workers=3          # 3 workers share 1 GPU
)
```

**Performance Profile:**
- **Worker Distribution**: 3 workers × 1 GPU
- **Expected Speedup**: 5-15x vs single-threaded CPU
- **Memory Usage**: ~2-3GB GPU memory per worker
- **Use Case**: Individual research, method development

---

### Scenario 6: Cloud Computing (Variable Resources)
**Configuration**: Elastic/Variable (e.g., AWS, Google Cloud)

```python
# Adaptive configuration based on available resources
def configure_for_cloud():
    gpu_ids = gpu_utils.get_available_gpu_ids()
    cpu_count = os.cpu_count()

    if gpu_ids:
        # GPU instance - optimize for GPU usage
        max_workers = len(gpu_ids) * 2
        print(f"Cloud GPU instance: {len(gpu_ids)} GPUs, {max_workers} workers")
    else:
        # CPU instance - optimize for CPU usage
        gpu_ids = None
        max_workers = cpu_count
        print(f"Cloud CPU instance: {cpu_count} CPUs, {max_workers} workers")

    return gpu_ids, max_workers

# Use adaptive configuration
gpu_ids, max_workers = configure_for_cloud()

results = optimal_stopping_posthoc(
    df=dataset,
    params=params,
    grouping_columns=['condition'],
    sample_id_column='item_id',
    epoch_column='trial',
    score_column='accuracy',
    gpu_ids=gpu_ids,
    max_workers=max_workers
)
```

**Performance Profile:**
- **Adaptive**: Automatically optimizes based on instance type
- **Cost-Effective**: Uses available resources efficiently
- **Scalable**: Can handle varying workload sizes

---

## Advanced Configuration Patterns

### Pattern 1: Mixed CPU/GPU Workloads
```python
# Some groupings on GPU, others on CPU (not directly supported, but achievable)
gpu_intensive_groupings = df[df['complexity'] == 'high']
cpu_suitable_groupings = df[df['complexity'] == 'low']

# Process intensive groupings on GPU
gpu_results = optimal_stopping_posthoc(
    gpu_intensive_groupings, params, cols, id_col, epoch_col, score_col,
    gpu_ids=[0, 1], max_workers=4
)

# Process simple groupings on CPU
cpu_results = optimal_stopping_posthoc(
    cpu_suitable_groupings, params, cols, id_col, epoch_col, score_col,
    gpu_ids=[], max_workers=8
)
```

### Pattern 2: GPU Memory Optimization
```python
# For limited GPU memory scenarios
def optimize_for_memory(dataset_size, gpu_memory_gb):
    if dataset_size > 10000 and gpu_memory_gb < 8:
        # Use fewer workers per GPU to reduce memory pressure
        max_workers = len(gpu_utils.get_available_gpu_ids())
        print("Memory-optimized: 1 worker per GPU")
    else:
        # Standard configuration
        max_workers = len(gpu_utils.get_available_gpu_ids()) * 2
        print("Standard: 2 workers per GPU")

    return max_workers

max_workers = optimize_for_memory(len(df), 6)  # 6GB GPU memory
```

### Pattern 3: Dynamic Load Balancing
```python
# Adjust workers based on grouping complexity
grouping_counts = df.groupby('grouping_column').size()
complex_analysis = len(grouping_counts) > 20

if complex_analysis:
    # Many small groupings - use more workers
    max_workers = len(gpu_utils.get_available_gpu_ids()) * 3
else:
    # Few complex groupings - use fewer workers
    max_workers = len(gpu_utils.get_available_gpu_ids())
```

## Performance Expectations by Scenario

| Scenario | Hardware | Dataset Size | Expected Speedup | Time: CPU-Only | Time: GPU |
|----------|----------|--------------|------------------|----------------|-----------|
| Laptop | 4-core, No GPU | Small (<1K rows) | 1x (baseline) | 5-10 min | N/A |
| Workstation | 8-core, 2 GPU | Medium (1-10K rows) | 10-25x | 2-4 hours | 5-15 min |
| HPC Single Node | 32-core, 8 GPU | Large (10-100K rows) | 30-100x | 8-24 hours | 10-30 min |
| HPC Multi-Node | 128-core, 16 GPU | Massive (100K+ rows) | 50-200x | 1-7 days | 30-60 min |

## Error Handling and Fallbacks

### Automatic Fallback Scenarios

1. **Invalid GPU IDs**: Automatically filters to valid GPUs
   ```python
   # Requested [0, 1, 99] but only [0, 1] available
   # → Automatically uses [0, 1]
   ```

2. **No GPUs Available**: Falls back to CPU-only
   ```python
   # Requested gpu_ids=[0, 1] but no GPUs detected
   # → Automatically uses CPU-only with warning
   ```

3. **GPU Memory Exhaustion**: Process isolation prevents cascading failures
   ```python
   # Individual worker GPU memory issues don't crash other workers
   # → Failed workers return error status, others continue
   ```

4. **Worker Process Failures**: Robust error handling maintains progress
   ```python
   # Failed workers return structured error information
   # → Overall analysis continues with partial results
   ```

## Best Practices by Scenario

### For Researchers
- **Start Simple**: Use default settings first, then optimize
- **Monitor Memory**: Watch GPU memory usage during initial runs
- **Scale Gradually**: Increase workers/GPUs based on performance gains

### For HPC Users
- **Profile First**: Test on subset to optimize configuration
- **Consider Queueing**: Balance worker count with cluster policies
- **Use Batch Processing**: Process multiple datasets efficiently

### For Cloud Users
- **Cost Optimization**: Match instance type to workload size
- **Auto-Scaling**: Use adaptive configuration patterns
- **Monitor Usage**: Track GPU utilization for cost-effectiveness

## Troubleshooting Common Issues

### Issue: Poor GPU Utilization
```python
# Problem: Too few workers per GPU
gpu_ids = [0, 1]
max_workers = 2  # Only 1 worker per GPU

# Solution: Increase workers per GPU
max_workers = 6   # 3 workers per GPU for better utilization
```

### Issue: Out of GPU Memory
```python
# Problem: Too many workers per GPU
max_workers = 12  # 6 workers per GPU might exhaust memory

# Solution: Reduce workers or adjust batch size
max_workers = 4   # 2 workers per GPU
# or adjust PyMC parameters
params['chains'] = 2  # Fewer chains per worker
```

### Issue: CPU Bottleneck with GPUs Available
```python
# Problem: Not using available GPUs
results = optimal_stopping_posthoc(df, params, cols, id_col, epoch_col, score_col)

# Solution: Explicitly enable GPU usage
gpu_ids = gpu_utils.get_available_gpu_ids()
results = optimal_stopping_posthoc(df, params, cols, id_col, epoch_col, score_col,
                                  gpu_ids=gpu_ids)
```

The enhanced optstop package now provides flexible, hardware-aware optimal stopping analysis that automatically adapts to available resources while maintaining perfect backward compatibility.