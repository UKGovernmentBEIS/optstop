"""
GPU detection and optimization utilities for PyMC processes in optstop.

This module provides functions to detect GPU availability across multiple backends,
configure GPU usage, and optimize PyMC sampling performance.

Sampler Selection Strategy (v0.2.1+):
- GPU available (JAX backend) → numpyro for all score types (fastest)
- CPU → PyMC default for all score types (optimal for CPU)

Note: Benchmarks show numpyro/JAX is ~26% SLOWER than PyMC default on CPU due to
compilation overhead. The previous strategy of using numpyro for ordinal inference
on CPU has been deprecated. PyMC's native PyTensor backend is well-optimized for
CPU execution across all inference pathways (binary, ordinal, continuous).
"""

import logging
import subprocess
from typing import Dict, Any, Optional, Tuple, List

# Module-level flags for log deduplication (only log once per session)
_gpu_status_logged = False
_sampling_config_logged = False


def reset_logging_flags():
    """Reset logging deduplication flags. Useful for testing multiple datasets."""
    global _gpu_status_logged, _sampling_config_logged
    _gpu_status_logged = False
    _sampling_config_logged = False

def check_nutpie_available() -> Tuple[bool, Optional[str]]:
    """
    Check if nutpie (Rust-based NUTS sampler) is available.

    Nutpie is a drop-in replacement for PyMC's default Python-based NUTS sampler,
    providing 2-5× speedup for CPU sampling by using compiled Rust code.

    Returns:
        Tuple of (available: bool, version: Optional[str])
        - available: True if nutpie can be imported
        - version: Version string if available, None otherwise

    Examples:
        >>> available, version = check_nutpie_available()
        >>> if available:
        ...     print(f"nutpie {version} available")
        ... else:
        ...     print("nutpie not available, using default PyMC sampler")
    """
    try:
        import nutpie
        version = getattr(nutpie, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None


def get_optimal_nuts_sampler(gpu_available: bool, gpu_backend: str = 'cpu') -> Tuple[str, str]:
    """
    Select the optimal NUTS sampler based on available hardware.

    Sampler selection:
    - GPU available (JAX backend) → 'numpyro' (JAX/GPU, fastest)
    - CPU → 'pymc' (PyMC default, optimal for CPU)

    Note: Benchmarks show that on CPU, PyMC's native PyTensor backend outperforms
    both numpyro/JAX (~26% slower due to compilation overhead) and nutpie
    (compilation overhead negates sampling speedups for adaptive collection).

    Args:
        gpu_available: Whether GPU acceleration is available
        gpu_backend: GPU backend type ('jax-gpu', 'pytensor-gpu', 'cpu')

    Returns:
        Tuple of (sampler_name: str, description: str)
        - sampler_name: 'numpyro' or 'pymc'
        - description: Human-readable description for logging

    Examples:
        >>> sampler, desc = get_optimal_nuts_sampler(gpu_available=True, gpu_backend='jax-gpu')
        >>> print(f"Using {sampler}: {desc}")
        Using numpyro: JAX/GPU backend (fastest)

        >>> sampler, desc = get_optimal_nuts_sampler(gpu_available=False)
        >>> print(f"Using {sampler}: {desc}")
        Using pymc: PyMC default (optimal for CPU)
    """
    # GPU with JAX backend: Use numpyro for GPU acceleration
    if gpu_available and gpu_backend == 'jax-gpu':
        return 'numpyro', 'JAX/GPU backend (fastest)'

    # CPU: Use PyMC default (optimal for CPU execution)
    return 'pymc', 'PyMC default (optimal for CPU)'


def _detect_system_gpus() -> Dict[str, Any]:
    """
    Detect GPUs at system level using nvidia-smi or other system tools.

    Returns:
        Dict with system GPU information
    """
    gpu_info = {
        'system_gpus_detected': False,
        'system_gpu_count': 0,
        'system_gpu_names': [],
        'nvidia_driver_version': None,
        'cuda_version': None,
        'detection_method': None
    }

    # Method 1: Try pynvml (lightweight NVIDIA library)
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        gpu_names = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            gpu_names.append(name)

        gpu_info.update({
            'system_gpus_detected': True,
            'system_gpu_count': device_count,
            'system_gpu_names': gpu_names,
            'nvidia_driver_version': pynvml.nvmlSystemGetDriverVersion().decode('utf-8'),
            'detection_method': 'pynvml'
        })
        pynvml.nvmlShutdown()
        return gpu_info

    except (ImportError, Exception):
        pass

    # Method 2: Try nvidia-ml-py
    try:
        import pynvml as nvml_py
        nvml_py.nvmlInit()
        device_count = nvml_py.nvmlDeviceGetCount()

        gpu_names = []
        for i in range(device_count):
            handle = nvml_py.nvmlDeviceGetHandleByIndex(i)
            name = nvml_py.nvmlDeviceGetName(handle).decode('utf-8')
            gpu_names.append(name)

        gpu_info.update({
            'system_gpus_detected': True,
            'system_gpu_count': device_count,
            'system_gpu_names': gpu_names,
            'detection_method': 'nvidia-ml-py'
        })
        nvml_py.nvmlShutdown()
        return gpu_info

    except (ImportError, Exception):
        pass

    # Method 3: Try nvidia-smi command
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_names = []
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 1:
                        gpu_names.append(parts[0].strip())
                        if len(parts) >= 2:
                            gpu_info['nvidia_driver_version'] = parts[1].strip()

            gpu_info.update({
                'system_gpus_detected': True,
                'system_gpu_count': len(gpu_names),
                'system_gpu_names': gpu_names,
                'detection_method': 'nvidia-smi'
            })
            return gpu_info
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        pass

    return gpu_info

def _check_pytensor_gpu() -> Dict[str, Any]:
    """
    Check PyTensor GPU backend availability.

    Returns:
        Dict with PyTensor GPU information
    """
    pytensor_info = {
        'pytensor_available': False,
        'pytensor_gpu_backend': False,
        'pytensor_device': 'cpu',
        'pytensor_error': None
    }

    try:
        import pytensor
        pytensor_info['pytensor_available'] = True

        # Check current device configuration
        device = str(pytensor.config.device).lower()
        pytensor_info['pytensor_device'] = device

        # Check if GPU device is configured
        if 'gpu' in device or 'cuda' in device:
            pytensor_info['pytensor_gpu_backend'] = True

    except ImportError as e:
        pytensor_info['pytensor_error'] = f"PyTensor not available: {str(e)}"
    except Exception as e:
        pytensor_info['pytensor_error'] = f"PyTensor check failed: {str(e)}"

    return pytensor_info

def check_gpu_availability() -> Tuple[bool, str, Dict[str, Any]]:
    """
    Enhanced GPU availability check supporting multiple backends.

    Checks for GPU availability through:
    1. JAX backend (preferred for PyMC)
    2. PyTensor CUDA backend
    3. System-level GPU detection

    Returns:
        Tuple containing:
        - bool: Whether GPU is available and functional for PyMC
        - str: Backend being used ('jax-gpu', 'pytensor-gpu', 'cpu')
        - dict: Comprehensive GPU information from all detection methods
    """
    gpu_available = False
    backend = 'cpu'

    # Comprehensive GPU information dictionary
    gpu_info = {
        # Overall status
        'device_count': 0,
        'devices': [],
        'cuda_available': False,
        'error_msg': None,
        'backend_used': 'cpu',
        'optimization_available': False,

        # JAX backend info
        'jax_available': False,
        'jax_backend': 'cpu',
        'jax_devices': [],
        'jax_gpu_functional': False,

        # PyTensor backend info
        'pytensor_available': False,
        'pytensor_gpu_backend': False,
        'pytensor_device': 'cpu',

        # System GPU detection
        'system_gpus_detected': False,
        'system_gpu_count': 0,
        'system_gpu_names': [],

        # User recommendations
        'recommendations': []
    }

    logger = logging.getLogger('optstop.gpu_utils')

    # Use module-level flag to only log GPU status once per session
    global _gpu_status_logged
    should_log = not _gpu_status_logged

    # Phase 1: Check JAX backend (preferred)
    jax_gpu_available = False
    try:
        import jax
        gpu_info['jax_available'] = True

        # Check JAX backend
        jax_backend = jax.default_backend()
        devices = jax.devices()

        gpu_info['jax_backend'] = jax_backend
        gpu_info['jax_devices'] = [str(d) for d in devices]
        # Count GPU/CUDA devices more robustly
        jax_gpu_count = 0
        for d in devices:
            device_str = str(d).lower()
            if 'gpu' in device_str or 'cuda' in device_str:
                jax_gpu_count += 1

        # Alternative: also check device_kind if available
        if hasattr(devices[0], 'device_kind') and devices:
            kind_gpu_count = len([d for d in devices if d.device_kind in ('gpu', 'cuda')])
            jax_gpu_count = max(jax_gpu_count, kind_gpu_count)

        if jax_backend == 'gpu' and jax_gpu_count > 0:
            # Test JAX GPU functionality
            try:
                import jax.numpy as jnp
                test_array = jnp.array([1.0, 2.0, 3.0])
                _ = jnp.sum(test_array)
                jax_gpu_available = True
                gpu_info['jax_gpu_functional'] = True
                gpu_info['device_count'] = jax_gpu_count
                gpu_info['devices'] = gpu_info['jax_devices']
                gpu_info['cuda_available'] = True
                backend = 'jax-gpu'
                gpu_available = True
                gpu_info['backend_used'] = 'jax-gpu'
                gpu_info['optimization_available'] = True
                if should_log:
                    logger.info(f"JAX GPU acceleration available: {jax_gpu_count} GPU(s)")
            except Exception as e:
                logger.warning(f"JAX GPU detected but test failed: {e}")
        else:
            if should_log:
                logger.info(f"JAX backend: {jax_backend}, GPU devices: {jax_gpu_count}")

    except ImportError:
        if should_log:
            logger.info("JAX not available - checking other GPU backends")
    except Exception as e:
        logger.warning(f"JAX GPU detection error: {e}")

    # Phase 2: Check PyTensor GPU backend (if JAX unavailable)
    if not jax_gpu_available:
        pytensor_info = _check_pytensor_gpu()
        gpu_info.update(pytensor_info)

        if pytensor_info['pytensor_gpu_backend']:
            gpu_available = True
            backend = 'pytensor-gpu'
            gpu_info['backend_used'] = 'pytensor-gpu'
            gpu_info['optimization_available'] = True
            gpu_info['device_count'] = 1  # PyTensor typically uses 1 device
            gpu_info['cuda_available'] = True
            if should_log:
                logger.info(f"PyTensor GPU backend available: {pytensor_info['pytensor_device']}")

    # Phase 3: System-level GPU detection (for user awareness)
    system_gpu_info = _detect_system_gpus()
    gpu_info.update(system_gpu_info)

    # Phase 4: Generate recommendations based on findings
    _generate_gpu_recommendations(gpu_info, jax_gpu_available, logger, should_log)

    # Final status (only log once per session)
    if should_log:
        if gpu_available:
            logger.info(f"GPU acceleration enabled via {gpu_info['backend_used']}")
        else:
            if gpu_info['system_gpus_detected']:
                logger.warning(f"System GPUs detected ({gpu_info['system_gpu_count']}) but no PyMC GPU backend available")
            else:
                logger.info("No GPU acceleration available - using CPU")
        _gpu_status_logged = True

    return gpu_available, backend, gpu_info

def _generate_gpu_recommendations(gpu_info: Dict[str, Any], jax_available: bool, logger, should_log: bool = True) -> None:
    """Generate user recommendations based on GPU detection results."""
    recommendations = []

    if gpu_info['system_gpus_detected'] and not gpu_info['optimization_available']:
        recommendations.append("GPU(s) detected but not usable for PyMC acceleration")

        if not gpu_info['jax_available']:
            recommendations.append("Install JAX for optimal GPU acceleration: pip install optstop[gpu]")
        elif gpu_info['jax_available'] and not jax_available:
            recommendations.append("JAX installed but GPU backend not configured")

        if not gpu_info['pytensor_available']:
            recommendations.append("Consider PyTensor GPU configuration as alternative")

    elif gpu_info['system_gpus_detected'] and gpu_info['optimization_available']:
        recommendations.append(f"GPU acceleration active via {gpu_info['backend_used']}")

    elif not gpu_info['system_gpus_detected']:
        recommendations.append("No system GPUs detected - CPU-only execution")

    gpu_info['recommendations'] = recommendations

    # Log key recommendations (only once per session)
    if should_log:
        for rec in recommendations[:2]:  # Log top 2 recommendations
            logger.info(f"Recommendation: {rec}")


def configure_jax_for_gpu() -> bool:
    """
    Configure JAX for optimal GPU usage if available.

    Returns:
        bool: True if GPU configuration successful, False otherwise
    """
    logger = logging.getLogger('optstop.gpu_utils')

    try:
        import jax

        # Configure JAX for GPU memory management
        import os

        # Enable memory preallocation to avoid fragmentation
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

        # Configure JAX for GPU if available
        if jax.default_backend() == 'gpu':
            logger.info("JAX configured for GPU usage")
            return True
        else:
            logger.info("JAX configured for CPU usage (no GPU available)")
            return False

    except ImportError:
        logger.info("JAX not available - cannot configure for GPU")
        return False
    except Exception as e:
        logger.warning(f"Error configuring JAX for GPU: {e}")
        return False


def estimate_gpu_memory_requirement(params: Dict[str, Any], num_parallel_tasks: int = 1) -> float:
    """
    Estimate GPU memory requirement in GB for a sampling task.

    Args:
        params: Sampling parameters (draws, tune, chains, model complexity)
        num_parallel_tasks: Number of tasks running in parallel on GPU

    Returns:
        Estimated GPU memory in GB
    """
    draws = params.get('draws', 1000)
    tune = params.get('tune', 1000)
    chains = params.get('chains', 1)

    # Estimate memory per chain (rough heuristic based on empirical testing)
    # Base overhead: ~500 MB for JAX/XLA compilation
    # Per-sample overhead: ~0.5 KB per draw
    # Model complexity factor (can be adjusted based on model size)
    base_overhead_mb = 500
    per_sample_kb = 0.5

    total_samples = (draws + tune) * chains
    sample_memory_mb = (total_samples * per_sample_kb) / 1024

    # Total per task
    memory_per_task_mb = base_overhead_mb + sample_memory_mb

    # Account for parallel tasks
    total_memory_gb = (memory_per_task_mb * num_parallel_tasks) / 1024

    return total_memory_gb


def get_system_specs() -> Dict[str, Any]:
    """
    Get comprehensive system specifications for adaptive decision-making.

    Returns:
        Dict with system specs:
        - cpu_count: Number of CPU cores
        - gpu_count: Number of GPUs
        - gpu_memory_gb: List of available memory per GPU in GB
        - total_gpu_memory_gb: Total GPU memory across all GPUs
        - ram_available_gb: Available system RAM in GB
    """
    import os

    specs = {
        'cpu_count': os.cpu_count() or 1,
        'gpu_count': 0,
        'gpu_memory_gb': [],
        'total_gpu_memory_gb': 0.0,
        'ram_available_gb': 0.0
    }

    # Get GPU count and memory
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free',
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_memories = [float(mem) / 1024 for mem in result.stdout.strip().split('\n') if mem.strip()]
            specs['gpu_count'] = len(gpu_memories)
            specs['gpu_memory_gb'] = gpu_memories
            specs['total_gpu_memory_gb'] = sum(gpu_memories)
    except Exception:
        pass

    # Get available RAM
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    kb = int(line.split()[1])
                    specs['ram_available_gb'] = kb / (1024 * 1024)
                    break
    except Exception:
        pass

    return specs


def get_available_gpu_memory(gpu_index: int = 0) -> float:
    """
    Get available GPU memory in GB for a specific GPU.

    Args:
        gpu_index: Index of GPU to query (default 0)

    Returns:
        Available GPU memory in GB, or 0 if no GPU or error
    """
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free',
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_memories = [float(mem) for mem in result.stdout.strip().split('\n') if mem.strip()]
            if gpu_index < len(gpu_memories):
                return gpu_memories[gpu_index] / 1024
    except Exception:
        pass
    return 0.0


def should_use_gpu_for_workload(params: Dict[str, Any], num_parallel_tasks: int,
                                 gpu_available: bool, system_specs: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """
    Intelligently decide whether to use GPU based on workload characteristics and system specs.
    Fully adaptive to hardware configuration - no hardcoded assumptions.

    Args:
        params: Sampling parameters
        num_parallel_tasks: Number of parallel tasks
        gpu_available: Whether GPU is physically available
        system_specs: System specifications (if None, will be queried automatically)

    Returns:
        Tuple of (should_use_gpu, reason)
    """
    if not gpu_available:
        return False, "No GPU available"

    # Get system specs if not provided
    if system_specs is None:
        system_specs = get_system_specs()

    cpu_count = system_specs.get('cpu_count', 1)
    gpu_count = system_specs.get('gpu_count', 0)
    gpu_memory_gb = system_specs.get('total_gpu_memory_gb', 0.0)

    if gpu_count == 0 or gpu_memory_gb == 0:
        return False, "No GPU memory available"

    draws = params.get('draws', 1000)
    tune = params.get('tune', 1000)
    chains = params.get('chains', 1)

    # Estimate memory requirement
    estimated_memory = estimate_gpu_memory_requirement(params, num_parallel_tasks)

    # Safety factor: use 80% of available memory
    safe_gpu_memory = gpu_memory_gb * 0.8

    # Rule 1: Memory constraint (most critical)
    if estimated_memory > safe_gpu_memory:
        return False, f"Estimated GPU memory ({estimated_memory:.1f}GB) exceeds available ({safe_gpu_memory:.1f}GB). Use CPU to avoid OOM errors."

    # Rule 2: Workload size (GPU has overhead, only beneficial for larger workloads)
    # Adaptive threshold: scales with GPU capability
    # Small GPUs (~4GB): need >50K samples
    # Large GPUs (>16GB): can benefit from >30K samples
    min_samples_for_gpu = max(30000, 50000 - int(gpu_memory_gb * 1000))
    total_samples = (draws + tune) * chains * num_parallel_tasks

    if total_samples < min_samples_for_gpu:
        return False, f"Small workload ({total_samples} samples). CPU faster (GPU needs ≥{min_samples_for_gpu} samples for overhead)."

    # Rule 3: Parallel tasks vs CPU cores and GPU count
    # Adaptive: if you have many CPUs or multiple GPUs, the threshold changes
    # Formula: Prefer CPU if parallel_tasks > (cpu_count/2) and parallel_tasks > (gpu_count * 4)
    cpu_threshold = max(4, cpu_count // 2)  # At least 4, or half your CPU cores
    gpu_threshold = gpu_count * 4  # Each GPU can handle ~4 parallel tasks efficiently

    if num_parallel_tasks > cpu_threshold and num_parallel_tasks > gpu_threshold:
        return False, f"High parallelism ({num_parallel_tasks} tasks vs {cpu_count} CPUs, {gpu_count} GPUs). CPU multiprocessing more efficient."

    # Rule 4: Single chain + many parallel tasks = CPU better
    # Adaptive: threshold based on GPU count
    parallel_threshold = max(4, gpu_count * 2)

    if chains == 1 and num_parallel_tasks >= parallel_threshold:
        return False, f"Single-chain with high parallelism ({num_parallel_tasks} tasks, {gpu_count} GPUs). CPU multiprocessing more efficient."

    # Rule 5: Multi-GPU scenario - only use GPU if we can distribute effectively
    if gpu_count > 1:
        tasks_per_gpu = num_parallel_tasks / gpu_count
        if tasks_per_gpu > 4:
            return False, f"Uneven GPU load ({tasks_per_gpu:.1f} tasks/GPU). CPU parallelization more efficient."

    # GPU is beneficial
    reasons = []
    reasons.append(f"Large workload ({total_samples:,} samples)")
    reasons.append(f"Sufficient GPU memory ({estimated_memory:.1f}GB needed, {safe_gpu_memory:.1f}GB available)")
    reasons.append(f"Hardware fit: {num_parallel_tasks} tasks, {cpu_count} CPUs, {gpu_count} GPU(s)")

    return True, " | ".join(reasons)


def get_optimal_sampling_params(params: Dict[str, Any], gpu_available: bool,
                                num_parallel_tasks: int = 1, auto_decide: bool = True,
                                system_specs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get optimal sampling parameters based on GPU availability and workload analysis.
    Fully adaptive to hardware configuration.

    Args:
        params: Current parameter dictionary
        gpu_available: Whether GPU is available and functional
        num_parallel_tasks: Number of parallel tasks that will run
        auto_decide: If True, automatically decide GPU vs CPU based on workload
        system_specs: System specifications (if None, will be queried automatically)

    Returns:
        Dict with optimized sampling parameters
    """
    logger = logging.getLogger('optstop.gpu_utils')

    # Get system specs if not provided
    if system_specs is None:
        system_specs = get_system_specs()

    # Start with current parameters
    optimized_params = params.copy()

    # Smart decision making
    use_gpu = False
    decision_reason = "GPU not available"

    if gpu_available and auto_decide:
        use_gpu, decision_reason = should_use_gpu_for_workload(
            params, num_parallel_tasks, gpu_available, system_specs
        )
        logger.info(f"Auto-decision: {'GPU' if use_gpu else 'CPU'} - {decision_reason}")
    elif gpu_available:
        use_gpu = True
        decision_reason = "GPU forced by user"
        logger.info("Using GPU (user override)")

    if use_gpu:
        # GPU-optimized parameters
        logger.info("Configuring GPU-optimized sampling parameters")

        gpu_count = system_specs.get('gpu_count', 1)

        # Adaptive chain configuration based on GPU count
        # Single GPU: 1 chain per task
        # Multi-GPU: can use more chains and distribute
        if gpu_count == 1:
            optimized_params['chains'] = 1
            optimized_params['cores'] = 1
        else:
            # With multiple GPUs, can use more chains
            optimized_params['chains'] = min(gpu_count, params.get('chains', 2))
            optimized_params['cores'] = optimized_params['chains']

        # Add GPU-specific sampling parameters
        optimized_params['use_gpu'] = True
        optimized_params['nuts_sampler'] = 'numpyro'

    else:
        # CPU-optimized parameters
        logger.info("Configuring CPU-optimized sampling parameters")
        optimized_params['use_gpu'] = False

        # Use PyMC default for CPU (optimal for all score types)
        sampler, sampler_desc = get_optimal_nuts_sampler(gpu_available=False, gpu_backend='cpu')
        optimized_params['nuts_sampler'] = sampler
        logger.info(f"Selected NUTS sampler: {sampler} ({sampler_desc})")

        # Adaptive CPU configuration based on available cores
        cpu_count = system_specs.get('cpu_count', 1)
        original_chains = params.get('chains', 2)
        original_cores = params.get('cores', 2)

        # Don't exceed available CPU cores
        optimized_params['chains'] = min(original_chains, cpu_count)
        optimized_params['cores'] = min(original_cores, cpu_count)

    return optimized_params


def get_sampling_kwargs(params: Dict[str, Any], gpu_available: bool, gpu_backend: str = 'cpu',
                       num_parallel_tasks: int = 1, auto_decide: bool = True, score_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Get sampling keyword arguments optimized for the available hardware and backend.
    Includes intelligent GPU vs CPU decision-making based on workload.

    Sampler selection:
    - GPU available (JAX backend) → numpyro for all score types (fastest)
    - CPU → PyMC default for all score types (optimal for CPU)

    Args:
        params: Parameter dictionary
        gpu_available: Whether GPU is available and functional
        gpu_backend: GPU backend type ('jax-gpu', 'pytensor-gpu', 'cpu')
        num_parallel_tasks: Number of parallel tasks running (default: 1)
        auto_decide: If True, automatically decide whether to use GPU based on workload (default: True)
        score_type: Deprecated. Previously used for sampler selection, now ignored.
                   All score types use the same sampler (numpyro for GPU, PyMC for CPU).

    Returns:
        Dict with sampling kwargs for pm.sample()
    """
    # Note: score_type parameter kept for backwards compatibility but is no longer used.
    # All score types now use PyMC default on CPU and numpyro on GPU.
    _ = score_type  # Suppress unused variable warning
    logger = logging.getLogger('optstop.gpu_utils')

    # Use module-level flag to only log sampling config once per session
    global _sampling_config_logged
    should_log = not _sampling_config_logged

    # Use intelligent GPU/CPU decision making if auto_decide is enabled
    if auto_decide and gpu_available:
        # Get optimal parameters based on workload analysis
        optimized_params = get_optimal_sampling_params(
            params=params,
            gpu_available=gpu_available,
            num_parallel_tasks=num_parallel_tasks,
            auto_decide=True
        )

        # Check if the decision was to use GPU
        use_gpu_decision = optimized_params.get('use_gpu', False)

        if not use_gpu_decision:
            # Smart logic decided CPU is better - override gpu_available
            if should_log:
                logger.info("Auto-decision: Using CPU instead of GPU (workload analysis)")
            gpu_available = False
            gpu_backend = 'cpu'
        else:
            # Use the optimized chains/cores from smart decision
            params = optimized_params
            if should_log:
                logger.info(f"Auto-decision: Using GPU with optimized parameters (chains={params.get('chains')}, cores={params.get('cores')})")

    # Base sampling arguments
    # Defaults optimized for iterative stopping decisions (not publication-quality posteriors)
    # Users can override via optstop_params if higher precision needed
    if gpu_available and gpu_backend in ['jax-gpu', 'pytensor-gpu']:
        # GPU: Can afford more samples due to parallelization
        default_draws = 2000
        default_tune = 2000
        default_target_accept = 0.95
    else:
        # CPU: Prioritize speed for iterative stopping decisions
        # 1000 draws with 4 chains typically yields ESS of 100-500, sufficient for CI estimation
        # Old defaults (6000/6000/0.97) were overly conservative for stopping decisions
        default_draws = 1000
        default_tune = 1000
        default_target_accept = 0.95  # Unified with GPU for consistency across environments

    sampling_kwargs = {
        'draws': params.get('draws', default_draws),
        'tune': params.get('tune', default_tune),
        'chains': params.get('chains', 4),
        'cores': params.get('cores', 4),
        'progressbar': False,
        'target_accept': params.get('target_accept', default_target_accept),
    }

    # Include random_seed if provided for reproducibility
    if 'random_seed' in params and params['random_seed'] is not None:
        sampling_kwargs['random_seed'] = params['random_seed']
        if should_log:
            logger.info(f"MCMC random_seed: {params['random_seed']}")

    if gpu_available and params.get('use_gpu', True):
        if gpu_backend == 'jax-gpu':
            # Use numpyro (JAX) sampler for optimal GPU acceleration
            sampling_kwargs['nuts_sampler'] = 'numpyro'
            # CRITICAL: For GPU, reduce chains to prevent OOM
            # numpyro runs chains in parallel on the SAME GPU
            sampling_kwargs['chains'] = params.get('chains', 1)
            sampling_kwargs['cores'] = params.get('cores', 1)
            if should_log:
                logger.info(f"Configured sampling for GPU acceleration with JAX/numpyro (chains={sampling_kwargs['chains']}, cores={sampling_kwargs['cores']})")

        elif gpu_backend == 'pytensor-gpu':
            # Use standard PyMC sampler with PyTensor GPU backend
            # No special sampler needed - PyTensor handles GPU automatically
            if should_log:
                logger.info("Configured sampling for GPU acceleration with PyTensor CUDA backend")

        else:
            # Fallback to CPU (unknown GPU backend)
            sampler, sampler_desc = get_optimal_nuts_sampler(gpu_available=False, gpu_backend='cpu')
            sampling_kwargs['nuts_sampler'] = sampler
            if should_log:
                logger.info(f"GPU detected but backend unknown - falling back to CPU: {sampler} ({sampler_desc})")
    else:
        # CPU sampling strategy:
        # Use PyMC default for ALL score types on CPU.
        # Benchmarks show numpyro/JAX is NOT faster than PyMC on CPU (~26% slower),
        # as JAX's compilation overhead negates any sampling speedups without GPU.
        # PyMC's native PyTensor backend is well-optimized for CPU execution.
        sampler_desc = 'PyMC default (optimal for CPU)'
        if should_log:
            logger.info(f"Configured sampling for CPU: pymc ({sampler_desc}) - chains={sampling_kwargs['chains']}, cores={sampling_kwargs['cores']}")
        # No nuts_sampler set = PyMC default

    # Mark as logged after first successful configuration
    if should_log:
        _sampling_config_logged = True

    return sampling_kwargs


def log_gpu_status(gpu_available: bool, backend: str, gpu_info: Dict[str, Any]) -> None:
    """
    Log comprehensive GPU status information with multi-backend support.

    Args:
        gpu_available: Whether GPU is available and functional
        backend: GPU backend type ('jax-gpu', 'pytensor-gpu', 'cpu')
        gpu_info: Comprehensive GPU information dictionary
    """
    logger = logging.getLogger('optstop.gpu_status')

    logger.info("=== Enhanced GPU Status Report ===")
    logger.info(f"GPU Acceleration Available: {gpu_available}")
    logger.info(f"Backend Used: {gpu_info.get('backend_used', backend)}")
    logger.info(f"GPU Device Count: {gpu_info.get('device_count', 0)}")

    # JAX Backend Information
    logger.info(f"JAX Available: {gpu_info.get('jax_available', False)}")
    if gpu_info.get('jax_available'):
        logger.info(f"JAX Backend: {gpu_info.get('jax_backend', 'cpu')}")
        logger.info(f"JAX GPU Functional: {gpu_info.get('jax_gpu_functional', False)}")
        if gpu_info.get('jax_devices'):
            logger.info(f"JAX Devices: {gpu_info['jax_devices']}")

    # PyTensor Backend Information
    logger.info(f"PyTensor Available: {gpu_info.get('pytensor_available', False)}")
    if gpu_info.get('pytensor_available'):
        logger.info(f"PyTensor Device: {gpu_info.get('pytensor_device', 'cpu')}")
        logger.info(f"PyTensor GPU Backend: {gpu_info.get('pytensor_gpu_backend', False)}")

    # System GPU Detection
    logger.info(f"System GPUs Detected: {gpu_info.get('system_gpus_detected', False)}")
    if gpu_info.get('system_gpus_detected'):
        logger.info(f"System GPU Count: {gpu_info.get('system_gpu_count', 0)}")
        if gpu_info.get('system_gpu_names'):
            logger.info(f"System GPU Names: {gpu_info['system_gpu_names']}")
        if gpu_info.get('nvidia_driver_version'):
            logger.info(f"NVIDIA Driver Version: {gpu_info['nvidia_driver_version']}")
        logger.info(f"Detection Method: {gpu_info.get('detection_method', 'unknown')}")

    # Final Status
    if gpu_available:
        logger.info(f"PyMC will use GPU acceleration via {gpu_info.get('backend_used', 'unknown')}")
    else:
        if gpu_info.get('system_gpus_detected'):
            logger.warning("System GPUs detected but no PyMC GPU backend configured")
        else:
            logger.info("No GPU acceleration available - using CPU-only sampling")

    # User Recommendations
    if gpu_info.get('recommendations'):
        logger.info("Recommendations:")
        for i, rec in enumerate(gpu_info['recommendations'][:3], 1):  # Show top 3
            logger.info(f"   {i}. {rec}")

    logger.info("====================================")


def get_available_gpu_ids() -> List[int]:
    """Return list of available GPU IDs, or empty list if none available.

    Returns:
        List[int]: List of GPU device IDs (e.g., [0, 1, 2, 3]) or empty list
    """
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_ids = []
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line.isdigit():
                    gpu_ids.append(int(line))
            return gpu_ids
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    return []


def auto_assign_gpus(num_workers: int, available_gpu_ids: Optional[List[int]] = None) -> Optional[List[int]]:
    """Automatically assign GPUs to workers, or return None for CPU-only.

    Args:
        num_workers: Number of workers that need GPU assignment
        available_gpu_ids: List of available GPU IDs, or None to auto-detect

    Returns:
        List of GPU IDs to assign to workers (cyclically), or None for CPU-only
    """
    if available_gpu_ids is None:
        available_gpu_ids = get_available_gpu_ids()

    if not available_gpu_ids:
        return None

    # Assign GPUs cyclically to workers
    return [available_gpu_ids[i % len(available_gpu_ids)] for i in range(num_workers)]


def create_worker_initargs(worker_base_dir: str, gpu_ids: Optional[List[int]],
                          suppress_output: bool = True) -> List[tuple]:
    """Create initialization arguments for each worker with optional GPU assignment.

    Args:
        worker_base_dir: Base directory for worker temp files (deprecated, not used)
        gpu_ids: List of GPU IDs to assign to workers, or None for CPU-only
        suppress_output: Whether to suppress worker output

    Returns:
        List of tuples containing (worker_base_dir, gpu_id, suppress_output) for each worker
    """
    # Note: worker_base_dir kept in signature for backwards compatibility but is not used
    _ = worker_base_dir  # Suppress unused variable warning
    if gpu_ids:
        # GPU-enabled workers with cyclical assignment
        # Note: worker_base_dir is ignored in new implementation for better isolation
        return [(None, gpu_id, suppress_output) for gpu_id in gpu_ids]
    else:
        # CPU-only workers
        return [(None, None, suppress_output)]


def validate_gpu_configuration(gpu_ids: Optional[List[int]], max_workers: int) -> Tuple[Optional[List[int]], int]:
    """Validate and adjust GPU configuration for parallel processing.

    Args:
        gpu_ids: Requested GPU IDs, or None for auto-detection/CPU-only
        max_workers: Maximum number of workers requested

    Returns:
        Tuple of (validated_gpu_ids, adjusted_max_workers)
    """
    logger = logging.getLogger('optstop.gpu_utils')

    if gpu_ids is None:
        # Default behavior - use CPU only
        return None, max_workers

    if gpu_ids == []:
        # Explicitly requested CPU-only
        logger.info("GPU usage explicitly disabled - using CPU-only processing")
        return None, max_workers

    # Validate requested GPU IDs
    available_gpu_ids = get_available_gpu_ids()
    if not available_gpu_ids:
        logger.warning("GPUs requested but none available - falling back to CPU")
        return None, max_workers

    # Check if requested GPUs are available
    invalid_gpus = [gpu_id for gpu_id in gpu_ids if gpu_id not in available_gpu_ids]
    if invalid_gpus:
        logger.warning(f"Requested GPUs {invalid_gpus} not available. Available GPUs: {available_gpu_ids}")
        # Filter out invalid GPU IDs
        gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id in available_gpu_ids]

        if not gpu_ids:
            logger.warning("No valid GPUs after filtering - falling back to CPU")
            return None, max_workers

    # Adjust max_workers based on GPU availability if not specified
    if max_workers is None:
        max_workers = len(gpu_ids)
        logger.info(f"Auto-setting max_workers to {max_workers} based on available GPUs")

    # Create cyclical GPU assignment for the number of workers
    assigned_gpu_ids = [gpu_ids[i % len(gpu_ids)] for i in range(max_workers)]

    logger.info(f"Using GPUs {gpu_ids} for {max_workers} workers (cyclical assignment: {assigned_gpu_ids})")
    return assigned_gpu_ids, max_workers