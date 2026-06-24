#!/usr/bin/env python3
"""
Build script for custom CUDA conv2d extension.

Usage:
    pip install -e .
    # or
    python setup.py build_ext --inplace
"""

import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CUDABuild(build_ext):
    """Custom build command that compiles CUDA code with nvcc."""

    def build_extension(self, ext):
        # Get CUDA paths
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        cuda_include = os.path.join(cuda_home, "include")

        # Get Python paths
        import sysconfig
        py_include = sysconfig.get_path("include")

        # Source files
        src_dir = os.path.dirname(os.path.abspath(__file__))
        cu_src = os.path.join(src_dir, "conv_kernel.cu")
        c_src = os.path.join(src_dir, "conv_ops.c")

        # Output
        build_dir = os.path.dirname(self.get_ext_fullpath(ext.name))
        os.makedirs(build_dir, exist_ok=True)
        ext_path = self.get_ext_fullpath(ext.name)

        # Compile CUDA kernel to object file
        cu_obj = os.path.join(build_dir, "conv_kernel.o")
        nvcc_cmd = (
            f"nvcc -c {cu_src} -o {cu_obj} "
            f"-I{cuda_include} -I{py_include} "
            f"--compiler-options '-fPIC' "
            f"-O2 -gencode arch=compute_70,code=sm_70 "
            f"-gencode arch=compute_80,code=sm_80 "
            f"-gencode arch=compute_86,code=sm_86 "
            f"-gencode arch=compute_89,code=sm_89 "
            f"-gencode arch=compute_90,code=sm_90"
        )
        print(f"Running: {nvcc_cmd}")
        if os.system(nvcc_cmd) != 0:
            raise RuntimeError("nvcc compilation failed")

        # Compile C wrapper and link
        cc_cmd = (
            f"gcc -shared -fPIC -O2 "
            f"-I{py_include} -I{cuda_include} "
            f"{c_src} {cu_obj} "
            f"-o {ext_path} "
            f"-L{cuda_home}/lib64 -lcudart "
            f"-lpython3"
        )
        # Find the actual python lib
        import sys
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        # Try without explicit -lpython first (manylinux style)
        cc_cmd_nopylib = (
            f"gcc -shared -fPIC -O2 "
            f"-I{py_include} -I{cuda_include} "
            f"{c_src} {cu_obj} "
            f"-o {ext_path} "
            f"-L{cuda_home}/lib64 -lcudart"
        )
        print(f"Running: {cc_cmd_nopylib}")
        if os.system(cc_cmd_nopylib) != 0:
            print(f"Trying with -lpython{py_ver}...")
            if os.system(cc_cmd) != 0:
                raise RuntimeError("C compilation/linking failed")


setup(
    name="custom_conv_ops",
    version="1.0.0",
    description="Custom CUDA conv2d implementation",
    ext_modules=[
        Extension("conv_ops", sources=["conv_ops.c"])
    ],
    cmdclass={"build_ext": CUDABuild},
    python_requires=">=3.8",
)
