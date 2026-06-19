import torch
import os

# 编译并加载 CUDA 扩展
def _compile():
    from torch.utils.cpp_extension import load
    src_dir = os.path.dirname(os.path.abspath(__file__))
    return load(
        name='layernorm_cuda',
        sources=[
            os.path.join(src_dir, 'layernorm_cuda.cpp'),
            os.path.join(src_dir, 'layernorm_cuda_kernel.cu'),
        ],
        verbose=False
    )

_module = None

def forward(input, gamma, beta, eps=1e-5):
    global _module
    if _module is None:
        _module = _compile()
    return _module.forward(input, gamma, beta, eps)
