from setuptools import setup, Extension

setup(
    name="compute",
    version="1.0.0",
    description="Compute module with GIL race condition bug",
    ext_modules=[
        Extension(
            "compute",
            sources=["compute.c"],
            extra_compile_args=["-O2", "-Wall", "-Wextra"],
        )
    ],
)
