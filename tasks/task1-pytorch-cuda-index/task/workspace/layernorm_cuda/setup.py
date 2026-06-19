from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='layernorm_cuda',
    ext_modules=[
        CUDAExtension('layernorm_cuda', [
            'layernorm_cuda.cpp',
            'layernorm_cuda_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension},
)
