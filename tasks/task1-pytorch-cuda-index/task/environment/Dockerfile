# Task 1: PyTorch CUDA 多 kernel 修改(1 真 bug + 19 诱饵)
# 在 PyTorch 源码的多个 CUDA kernel 中注入修改,只有 1 个是真正的 bug

FROM pytorch/pytorch:2.5.0-cuda12.6-cudnn9-devel

# 安装依赖
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# clone PyTorch 源码
WORKDIR /build
RUN git clone --depth 1 --branch v2.5.0 https://github.com/pytorch/pytorch.git
WORKDIR /build/pytorch

# 初始化必要子模块
RUN git submodule update --init \
    third_party/pybind11 \
    third_party/cpuinfo \
    third_party/FP16 \
    third_party/FXdiv \
    third_party/psimd \
    third_party/pthreadpool \
    || true

# 复制注入脚本
COPY workspace/inject_bug.py /build/inject_bug.py

# 注入 1 个真 bug + 19 个诱饵
ENV PYTORCH_DIR=/build/pytorch
RUN python /build/inject_bug.py

# 验证真 bug 已注入
RUN grep -n "j <= N" aten/src/ATen/native/cuda/layer_norm_kernel.cu

# 编译 PyTorch(最小化编译选项)
ENV MAX_JOBS=32
ENV USE_CUDA=1
ENV USE_CUDNN=1
ENV USE_NCCL=0
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

# 清理源码以节省空间
RUN rm -rf /build/pytorch/.git /build/pytorch/third_party

# 设置工作目录
WORKDIR /workspace

# 复制任务文件
COPY workspace/train.py /workspace/

CMD ["/bin/bash"]
