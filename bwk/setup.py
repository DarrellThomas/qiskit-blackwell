# Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.

import os
from glob import glob
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = os.path.dirname(os.path.abspath(__file__))

# All .cu files in csrc/
cu_sources = sorted(glob(os.path.join(ROOT, "csrc", "*.cu")))

setup(
    name="blackwell_kernels",
    version="0.1.0",
    description="Custom CUDA kernels for quantum simulation on RTX 5090 (sm_120a)",
    author="Darrell Thomas",
    license="MIT",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[
        CUDAExtension(
            "blackwell_kernels._C",
            cu_sources,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode", "arch=compute_120a,code=sm_120a",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "-lineinfo",
                ],
            },
            include_dirs=[
                os.path.join(os.environ.get("CUDA_HOME", "/usr/local/cuda"), "include", "cccl"),
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
    install_requires=["torch>=2.0"],
)
