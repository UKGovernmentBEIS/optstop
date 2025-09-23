from optstop import gpu_utils

gpu_available, backend, gpu_info = gpu_utils.check_gpu_availability()

print('GPU Available:', gpu_available)
print('Backend:', backend)
print('JAX Available:', gpu_info.get('jax_available'))
print('JAX Backend:', gpu_info.get('jax_backend'))
print('JAX GPU Functional:', gpu_info.get('jax_gpu_functional'))
print('Device Count:', gpu_info.get('device_count', 0))
print('Backend Used:', gpu_info.get('backend_used'))