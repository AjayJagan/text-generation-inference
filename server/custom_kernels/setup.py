import os
import platform
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Detect target architecture and CUDA availability
def get_cuda_arch_flags():
    """Get CUDA architecture flags based on target platform"""
    target_arch = os.getenv('TARGETARCH', platform.machine())
    
    if target_arch == 'arm64' or target_arch == 'aarch64':
        # For ARM64, we need to support different compute capabilities
        # ARM64 with NVIDIA GPUs typically support compute capability 8.x and 9.x
        # H100 GPU (Grace Hopper Superchip) uses compute capability 9.0
        return ["-arch=compute_80", "-arch=compute_86", "-arch=compute_89", "-arch=compute_90", "-std=c++17"]
    else:
        # Default for x86_64
        return ["-arch=compute_80", "-std=c++17"]

# Check if CUDA is available
def is_cuda_available():
    """Check if CUDA is available for building"""
    # Check if CUDA_HOME is set
    if not os.getenv('CUDA_HOME'):
        return False
    
    # Check if nvcc is available
    import subprocess
    try:
        subprocess.run(['nvcc', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

# Build extensions based on CUDA availability
ext_modules = []
if is_cuda_available():
    ext_modules.extend([
        CUDAExtension(
            name="custom_kernels.fused_bloom_attention_cuda",
            sources=["custom_kernels/fused_bloom_attention_cuda.cu"],
            extra_compile_args=get_cuda_arch_flags(),
        ),
        CUDAExtension(
            name="custom_kernels.fused_attention_cuda",
            sources=["custom_kernels/fused_attention_cuda.cu"],
            extra_compile_args=get_cuda_arch_flags(),
        ),
    ])
else:
    print("CUDA not available, skipping CUDA extension compilation")

setup(
    name="custom_kernels",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
