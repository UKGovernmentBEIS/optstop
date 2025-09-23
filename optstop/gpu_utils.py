"""
GPU detection and optimization utilities for PyMC processes in optstop.

This module provides functions to detect GPU availability, configure JAX for GPU usage,
and optimize PyMC sampling performance using GPU acceleration when available.
"""

import logging
import warnings
from typing import Dict, Any, Optional, Tuple

def check_gpu_availability() -> Tuple[bool, str, Dict[str, Any]]:
    """
    Check for GPU availability and JAX backend capability.

    Returns:
        Tuple containing:
        - bool: Whether GPU is available and functional
        - str: Backend being used ('gpu', 'cpu', or 'unknown')
        - dict: GPU information (device count, memory, etc.)
    """
    gpu_available = False
    backend = 'cpu'
    gpu_info = {
        'device_count': 0,
        'devices': [],
        'jax_available': False,
        'cuda_available': False,
        'error_msg': None
    }

    logger = logging.getLogger('optstop.gpu_utils')

    try:
        # First check if JAX is available
        import jax
        gpu_info['jax_available'] = True

        # Check JAX backend
        backend = jax.default_backend()
        devices = jax.devices()

        gpu_info['device_count'] = len([d for d in devices if d.device_kind == 'gpu'])
        gpu_info['devices'] = [str(d) for d in devices]

        if backend == 'gpu' and gpu_info['device_count'] > 0:
            gpu_available = True
            logger.info(f"GPU acceleration available: {gpu_info['device_count']} GPU(s) detected")
            logger.info(f"GPU devices: {gpu_info['devices']}")
        else:
            logger.info(f"JAX backend: {backend}, GPU devices: {gpu_info['device_count']}")

        # Check CUDA availability through JAX
        try:
            # Try a simple GPU operation to verify functionality
            if gpu_available:
                import jax.numpy as jnp
                test_array = jnp.array([1.0, 2.0, 3.0])
                _ = jnp.sum(test_array)  # Simple operation to test GPU
                gpu_info['cuda_available'] = True
                logger.info("GPU functionality verified with test operation")
        except Exception as e:
            gpu_available = False
            gpu_info['error_msg'] = f"GPU test failed: {str(e)}"
            logger.warning(f"GPU detected but test operation failed: {e}")

    except ImportError as e:
        gpu_info['error_msg'] = f"JAX not available: {str(e)}"
        logger.info("JAX not installed - GPU acceleration unavailable")
    except Exception as e:
        gpu_info['error_msg'] = f"GPU detection error: {str(e)}"
        logger.warning(f"Error during GPU detection: {e}")

    return gpu_available, backend, gpu_info


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


def get_sampling_kwargs(params: Dict[str, Any], gpu_available: bool) -> Dict[str, Any]:
    """
    Get sampling keyword arguments optimized for the available hardware.

    Args:
        params: Parameter dictionary
        gpu_available: Whether GPU is available and functional

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
        # Use numpyro (JAX) sampler for GPU acceleration
        sampling_kwargs['nuts_sampler'] = 'numpyro'
        sampling_kwargs['chains'] = params.get('chains', 4)  # numpyro handles parallelization

        logger.info("Configured sampling for GPU acceleration with numpyro")
    else:
        # Standard PyMC sampling on CPU
        logger.info("Configured sampling for CPU")

    return sampling_kwargs


def log_gpu_status(gpu_available: bool, backend: str, gpu_info: Dict[str, Any]) -> None:
    """
    Log comprehensive GPU status information.

    Args:
        gpu_available: Whether GPU is available and functional
        backend: JAX backend ('gpu', 'cpu', etc.)
        gpu_info: GPU information dictionary
    """
    logger = logging.getLogger('optstop.gpu_status')

    logger.info("=== GPU Status Report ===")
    logger.info(f"GPU Available: {gpu_available}")
    logger.info(f"JAX Backend: {backend}")
    logger.info(f"JAX Available: {gpu_info.get('jax_available', False)}")
    logger.info(f"CUDA Available: {gpu_info.get('cuda_available', False)}")
    logger.info(f"GPU Device Count: {gpu_info.get('device_count', 0)}")

    if gpu_info.get('devices'):
        logger.info(f"GPU Devices: {gpu_info['devices']}")

    if gpu_info.get('error_msg'):
        logger.warning(f"GPU Error: {gpu_info['error_msg']}")

    if gpu_available:
        logger.info("PyMC will use GPU acceleration via JAX/numpyro")
    else:
        logger.info("PyMC will use CPU-only sampling")

    logger.info("========================")