"""Build script for the tensor_ops C extension."""
from setuptools import setup, Extension
import numpy as np

setup(
    name="tensor_ops",
    version="1.0.0",
    description="Tensor operations with shape inference (simulated custom TF op)",
    ext_modules=[
        Extension(
            "tensor_ops",
            sources=["tensor_ops.c"],
            include_dirs=[np.get_include()],
        )
    ],
)
