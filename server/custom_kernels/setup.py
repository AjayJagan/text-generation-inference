import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Check if we're building for AMD64 and CUDA is available
TARGETARCH = os.environ.get('TARGETARCH', 'amd64')
CUDA_AVAILABLE = TARGETARCH == 'amd64' and os.environ.get('CUDA_HOME') is not None

if CUDA_AVAILABLE:
    ext_modules = [
        CUDAExtension(
            name="custom_kernels.fused_bloom_attention_cuda",
            sources=["custom_kernels/fused_bloom_attention_cuda.cu"],
            extra_compile_args=["-arch=compute_80", "-std=c++17"],
        ),
        CUDAExtension(
            name="custom_kernels.fused_attention_cuda",
            sources=["custom_kernels/fused_attention_cuda.cu"],
            extra_compile_args=["-arch=compute_80", "-std=c++17"],
        ),
    ]
else:
    print(f"Skipping CUDA extensions for architecture: {TARGETARCH}")
    ext_modules = []

setup(
    name="custom_kernels",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
