"""
GPU detection and optimization utilities for PyMC processes in optstop.

This module provides functions to detect GPU availability across multiple backends,
configure GPU usage, and optimize PyMC sampling performance using GPU acceleration
when available through JAX/numpyro, PyTensor, or system-level detection.
"""

import logging
import warnings
import subprocess
import sys
from typing import Dict, Any, Optional, Tuple, List

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
        jax_gpu_count = len([d for d in devices if d.device_kind == 'gpu'])

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
                logger.info(f"JAX GPU acceleration available: {jax_gpu_count} GPU(s)")
            except Exception as e:
                logger.warning(f"JAX GPU detected but test failed: {e}")
        else:
            logger.info(f"JAX backend: {jax_backend}, GPU devices: {jax_gpu_count}")

    except ImportError:
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
            logger.info(f"PyTensor GPU backend available: {pytensor_info['pytensor_device']}")

    # Phase 3: System-level GPU detection (for user awareness)
    system_gpu_info = _detect_system_gpus()
    gpu_info.update(system_gpu_info)

    # Phase 4: Generate recommendations based on findings
    _generate_gpu_recommendations(gpu_info, jax_gpu_available, logger)

    # Final status
    if gpu_available:
        logger.info(f"GPU acceleration enabled via {gpu_info['backend_used']}")
    else:
        if gpu_info['system_gpus_detected']:
            logger.warning(f"System GPUs detected ({gpu_info['system_gpu_count']}) but no PyMC GPU backend available")
        else:
            logger.info("No GPU acceleration available - using CPU")

    return gpu_available, backend, gpu_info

def _generate_gpu_recommendations(gpu_info: Dict[str, Any], jax_available: bool, logger) -> None:
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

    # Log key recommendations
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


def get_optimal_sampling_params(params: Dict[str, Any], gpu_available: bool) -> Dict[str, Any]:
    """
    Get optimal sampling parameters based on GPU availability.

    Args:
        params: Current parameter dictionary
        gpu_available: Whether GPU is available and functional

    Returns:
        Dict with optimized sampling parameters
    """
    logger = logging.getLogger('optstop.gpu_utils')

    # Start with current parameters
    optimized_params = params.copy()

    if gpu_available:
        # GPU-optimized parameters
        logger.info("Using GPU-optimized sampling parameters")

        # For GPU, we can often use more chains efficiently
        # Since GPU processes chains in parallel more effectively
        original_chains = params.get('chains', 4)
        original_cores = params.get('cores', 4)

        # GPU can handle more chains efficiently due to vectorization
        if original_chains < 6:
            optimized_params['chains'] = min(8, original_chains * 2)
            logger.info(f"Increased chains from {original_chains} to {optimized_params['chains']} for GPU")

        # Set cores to match chains for GPU usage (numpyro handles parallelization)
        optimized_params['cores'] = optimized_params['chains']

        # Add GPU-specific sampling parameters
        optimized_params['use_gpu'] = True
        optimized_params['nuts_sampler'] = 'numpyro'

    else:
        # CPU-optimized parameters (existing behavior)
        logger.info("Using CPU-optimized sampling parameters")
        optimized_params['use_gpu'] = False
        optimized_params['nuts_sampler'] = 'pymc'  # Default PyMC sampler

    return optimized_params


def get_sampling_kwargs(params: Dict[str, Any], gpu_available: bool, gpu_backend: str = 'cpu') -> Dict[str, Any]:
    """
    Get sampling keyword arguments optimized for the available hardware and backend.

    Args:
        params: Parameter dictionary
        gpu_available: Whether GPU is available and functional
        gpu_backend: GPU backend type ('jax-gpu', 'pytensor-gpu', 'cpu')

    Returns:
        Dict with sampling kwargs for pm.sample()
    """
    logger = logging.getLogger('optstop.gpu_utils')

    # Base sampling arguments
    sampling_kwargs = {
        'draws': params.get('draws', 3000),
        'tune': params.get('tune', 3000),
        'chains': params.get('chains', 4),
        'cores': params.get('cores', 4),
        'progressbar': False,
        'target_accept': 0.97,
    }

    if gpu_available and params.get('use_gpu', True):
        if gpu_backend == 'jax-gpu':
            # Use numpyro (JAX) sampler for optimal GPU acceleration
            sampling_kwargs['nuts_sampler'] = 'numpyro'
            sampling_kwargs['chains'] = params.get('chains', 4)  # numpyro handles parallelization
            logger.info("Configured sampling for GPU acceleration with JAX/numpyro")

        elif gpu_backend == 'pytensor-gpu':
            # Use standard PyMC sampler with PyTensor GPU backend
            # No special sampler needed - PyTensor handles GPU automatically
            logger.info("Configured sampling for GPU acceleration with PyTensor CUDA backend")

        else:
            # Fallback to CPU
            logger.info("GPU detected but backend unknown - using CPU sampling")
    else:
        # Standard PyMC sampling on CPU
        logger.info("Configured sampling for CPU")

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
        logger.info(f"✅ PyMC will use GPU acceleration via {gpu_info.get('backend_used', 'unknown')}")
    else:
        if gpu_info.get('system_gpus_detected'):
            logger.warning("⚠️  System GPUs detected but no PyMC GPU backend configured")
        else:
            logger.info("ℹ️  No GPU acceleration available - using CPU-only sampling")

    # User Recommendations
    if gpu_info.get('recommendations'):
        logger.info("📋 Recommendations:")
        for i, rec in enumerate(gpu_info['recommendations'][:3], 1):  # Show top 3
            logger.info(f"   {i}. {rec}")

    logger.info("====================================")