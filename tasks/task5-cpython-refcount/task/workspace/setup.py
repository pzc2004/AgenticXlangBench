from setuptools import setup, Extension

setup(
    name="pyvector",
    version="1.0.0",
    description="A high-performance vector container implemented as a C extension",
    ext_modules=[
        Extension(
            "pyvector",
            sources=["vector.c"],
        )
    ],
)
