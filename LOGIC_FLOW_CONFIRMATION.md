# Logic Flow Confirmation: User Specifies chains=4, cores=4

## User Input
```python
params = {
    'chains': 4,
    'cores': 4,
    'draws': 3000,
    'tune': 3000,
    # No 'use_gpu' specified
}
```

## Complete Logic Flow

### Step 1: Function Call
User calls `optimal_stopping_posthoc(df, params, ...)`

### Step 2: Context Addition (rule.py:1254-1255)
```python
params_with_context = params.copy()
params_with_context['_num_parallel_tasks'] = len(groupings)  # e.g., 20
```

### Step 3: Worker Processes Grouping (rule.py:687-726)
Each worker:
1. Detects GPU availability: `gpu_available, gpu_backend, gpu_info = gpu_utils.check_gpu_availability()`
2. Example result: `gpu_available=True, gpu_backend='jax-gpu'`

### Step 4: Get Sampling Kwargs (rule.py:732-738)
Calls:
```python
sampling_kwargs = gpu_utils.get_sampling_kwargs(
    params=params_with_context,  # Has chains=4, cores=4, _num_parallel_tasks=20
    gpu_available=True,
    gpu_backend='jax-gpu',
    num_parallel_tasks=20,
    auto_decide=True  # DEFAULT - enables smart logic
)
```

### Step 5: Smart Decision Logic (gpu_utils.py:623-644)

#### 5a. Check Conditions (line 624)
```python
if auto_decide and gpu_available:  # TRUE and TRUE = TRUE
```

#### 5b. Call get_optimal_sampling_params (line 626-631)
```python
optimized_params = get_optimal_sampling_params(
    params=params_with_context,  # chains=4, cores=4, _num_parallel_tasks=20
    gpu_available=True,
    num_parallel_tasks=20,
    auto_decide=True
)
```

#### 5c. Inside get_optimal_sampling_params (gpu_utils.py:556-602)

**Smart Decision (line 556-560):**
```python
if gpu_available and auto_decide:  # TRUE and TRUE = TRUE
    use_gpu, decision_reason = should_use_gpu_for_workload(
        params={'chains': 4, 'cores': 4, 'draws': 3000, 'tune': 3000, '_num_parallel_tasks': 20},
        num_parallel_tasks=20,
        gpu_available=True,
        system_specs={...}
    )
```

**should_use_gpu_for_workload Analysis (gpu_utils.py:442-523):**

Applies 5 rules:

1. **Rule 1 - Memory Check (line 476-483):**
   - Estimated memory = `estimate_gpu_memory_requirement(params, 20)`
   - Example: 8GB required, 12GB available (80% safe = 9.6GB)
   - ✅ PASS: 8GB < 9.6GB

2. **Rule 2 - Workload Size (line 489-493):**
   - Total samples = (3000+3000) × 4 × 20 = 480,000
   - Min needed = 30,000 (for large GPU)
   - ✅ PASS: 480,000 > 30,000

3. **Rule 3 - Parallelism Check (line 498-502):**
   - CPU threshold = max(4, cpu_count/2) = max(4, 64/2) = 32
   - GPU threshold = gpu_count × 4 = 4 × 4 = 16
   - num_parallel_tasks = 20
   - ❌ FAIL: 20 > 16 (GPU threshold)
   - **Decision: Use CPU** (high parallelism better with CPU)
   - **Return: (False, "High parallelism (20 tasks vs 64 CPUs, 4 GPUs). CPU multiprocessing more efficient.")**

**Back to get_optimal_sampling_params (line 566-602):**

```python
if use_gpu:  # FALSE - skipped
    ...
else:  # TRUE - enters here
    # CPU-optimized parameters
    logger.info("Configuring CPU-optimized sampling parameters")
    optimized_params['use_gpu'] = False
    optimized_params['nuts_sampler'] = 'pymc'

    # Keep user's chains/cores (respecting CPU count)
    optimized_params['chains'] = min(4, 64) = 4  # Keep chains=4
    optimized_params['cores'] = min(4, 64) = 4   # Keep cores=4

    return optimized_params  # {chains: 4, cores: 4, use_gpu: False, ...}
```

#### 5d. Back in get_sampling_kwargs (line 633-644)

```python
use_gpu_decision = optimized_params.get('use_gpu', False)  # False

if not use_gpu_decision:  # TRUE
    logger.info("Auto-decision: Using CPU instead of GPU (workload analysis)")
    gpu_available = False  # Override
    gpu_backend = 'cpu'
```

### Step 6: Build sampling_kwargs (gpu_utils.py:646-678)

```python
sampling_kwargs = {
    'draws': 3000,
    'tune': 3000,
    'chains': 4,  # From optimized_params
    'cores': 4,   # From optimized_params
    'progressbar': False,
    'target_accept': 0.97,
}

if gpu_available and params.get('use_gpu', True):  # FALSE and ... = FALSE
    # Skipped
else:  # TRUE - enters here
    logger.info(f"Configured sampling for CPU (chains=4, cores=4)")

return sampling_kwargs  # {chains: 4, cores: 4, no nuts_sampler, ...}
```

### Step 7: PyMC Sampling (rule.py:809)
```python
trace = pm.sample(**sampling_kwargs)
# Uses standard PyMC CPU sampling with 4 chains across 4 cores
```

## Alternative Scenario: GPU Chosen

If instead you had:
- Few groupings (e.g., 2)
- Large workload
- Plenty of GPU memory

Then the logic would be:

**Rule 3 would PASS**: 2 tasks < 16 GPU threshold
**Decision: Use GPU**

Then:
```python
# In get_optimal_sampling_params (line 575-577)
if gpu_count == 1:
    optimized_params['chains'] = 1  # OVERRIDES user's chains=4
    optimized_params['cores'] = 1
    optimized_params['use_gpu'] = True

# Final sampling_kwargs
sampling_kwargs = {
    'chains': 1,  # CHANGED from 4 to 1
    'cores': 1,   # CHANGED from 4 to 1
    'nuts_sampler': 'numpyro',  # GPU sampler
    ...
}
```

## Summary Table

| User Input | Scenario | Auto-Decision | Final chains | Final cores | Sampler | Why |
|------------|----------|---------------|--------------|-------------|---------|-----|
| chains=4, cores=4 | 20 groupings, 64 CPUs, 4 GPUs | **CPU** | **4** | **4** | pymc | High parallelism better with CPU |
| chains=4, cores=4 | 2 groupings, 64 CPUs, 1 GPU, large workload | **GPU** | **1** | **1** | numpyro | GPU beneficial, reduce chains to prevent OOM |
| chains=4, cores=4, use_gpu=False | Any | **CPU** | **4** | **4** | pymc | User forced CPU |
| chains=4, cores=4, use_gpu=True | Any | **GPU** | **1** | **1** | numpyro | User forced GPU (risky!) |

## Key Points

1. **`auto_decide=True` by default** - Smart logic always runs unless disabled
2. **`use_gpu` not in params** - Treated as `use_gpu=True` for the check (line 656), but auto-decision can override
3. **User chains/cores respected when CPU chosen** - Your chains=4, cores=4 kept for CPU multiprocessing
4. **User chains/cores OVERRIDDEN when GPU chosen** - Automatically reduced to 1 to prevent OOM
5. **Decision logged** - You see exactly why GPU or CPU was chosen

## Confirmation

✅ **YES**, when you specify only `chains=4, cores=4` with **no GPU settings**:
- Smart logic **automatically runs**
- Analyzes your workload + hardware
- Decides GPU vs CPU based on what's optimal
- Prevents OOM by reducing chains to 1 if GPU is chosen
- Keeps your chains=4, cores=4 if CPU is chosen
- **You get the best of both worlds automatically!**
