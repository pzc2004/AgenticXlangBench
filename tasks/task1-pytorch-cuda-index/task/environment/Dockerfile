# Task 1: CUDA LayerNorm Off-by-One Bug
# 在 PyTorch 源码的 LayerNorm CUDA kernel 中注入 off-by-one bug,然后从源码编译

FROM pytorch/pytorch:2.5.0-cuda12.6-cudnn9-devel

# 安装编译依赖
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# clone PyTorch 源码(指定 v2.5.0)
WORKDIR /build
RUN git clone --depth 1 --branch v2.5.0 https://github.com/pytorch/pytorch.git
WORKDIR /build/pytorch

# 初始化子模块(只初始化必要的,跳过 ideep/kineto 等)
RUN git submodule update --init \
    third_party/pybind11 \
    third_party/cpuinfo \
    third_party/FP16 \
    third_party/FXdiv \
    third_party/psimd \
    third_party/pthreadpool \
    third_party/XNNPACK \
    third_party/NNPACK \
    third_party/pytorch_qnnpack \
    third_party/foxi \
    third_party/onnx \
    third_party/protobuf \
    third_party/flatbuffers \
    third_party/nccl \
    third_party/gloo \
    third_party/tensorpipe \
    || true

# 注入 bug:将 LayerNormForwardCUDAKernel 中的 'j < N' 改成 'j <= N'
RUN sed -i '/__global__ void LayerNormForwardCUDAKernel/,/^}/s/for (int64_t j = threadIdx.x; j < N;/for (int64_t j = threadIdx.x; j <= N;/' \
    aten/src/ATen/native/cuda/layer_norm_kernel.cu

# 验证 bug 注入
RUN grep -n "j <= N" aten/src/ATen/native/cuda/layer_norm_kernel.cu

# 编译 PyTorch(最小化编译选项,减少编译时间)
ENV MAX_JOBS=32
ENV USE_CUDA=1
ENV USE_CUDNN=1
ENV USE_NCCL=1
ENV USE_DISTRIBUTED=0
ENV BUILD_TEST=0
ENV USE_IDEEP=0
ENV USE_MKLDNN=0
ENV USE_QNNPACK=0
ENV USE_PYTORCH_QNNPACK=0
ENV USE_XNNPACK=0
ENV USE_FBGEMM=0
ENV USE_KINETO=0
ENV BUILD_CAFFE2=0
ENV BUILD_CAFFE2_OPS=0
ENV CMAKE_BUILD_TYPE=Release

RUN python setup.py develop

# 清理源码以节省空间(保留编译产物)
RUN rm -rf /build/pytorch/.git /build/pytorch/third_party

# 设置工作目录
WORKDIR /workspace

# 复制任务文件
COPY workspace/train.py /workspace/
COPY workspace/layernorm_cuda/ /workspace/layernorm_cuda/

# 默认启动 bash
CMD ["/bin/bash"]
