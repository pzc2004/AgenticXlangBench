# Task 1: PyTorch CUDA 多 kernel 修改(1 真 bug + 19 诱饵)
# 在 PyTorch 源码的多个 CUDA kernel 中注入修改,只有 1 个是真正的 bug
#
# 构建命令:
#   eval $(ssh-agent) && ssh-add ~/.ssh/id_rsa
#   DOCKER_BUILDKIT=1 docker build --ssh default -t task1 -f Dockerfile .

FROM pytorch/pytorch:2.5.0-cuda12.6-cudnn9-devel

# 安装依赖
RUN apt-get update && apt-get install -y git openssh-client && rm -rf /var/lib/apt/lists/*

# 配置 git 使用 SSH(连接更稳定,避免 HTTPS 超时)
RUN mkdir -p /root/.ssh && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null && \
    git config --global url."git@github.com:".insteadOf "https://github.com/"

# clone PyTorch 源码(使用 SSH)
WORKDIR /build
RUN --mount=type=ssh git clone --depth 1 --branch v2.5.0 git@github.com:pytorch/pytorch.git
WORKDIR /build/pytorch

# 初始化必要子模块(使用 SSH)
RUN --mount=type=ssh git submodule update --init \
    third_party/pybind11 \
    third_party/cpuinfo \
    third_party/FP16 \
    third_party/FXdiv \
    third_party/psimd \
    third_party/pthreadpool \
    || true

# 复制注入脚本(从 solution/ 目录,agent 不应看到)
COPY solution/inject_bug.py /build/inject_bug.py

# 注入 1 个真 bug + 19 个诱饵
ENV PYTORCH_DIR=/build/pytorch
RUN python /build/inject_bug.py

# 验证真 bug 已注入
RUN grep -n "j <= N" aten/src/ATen/native/cuda/layer_norm_kernel.cu

# 删除注入脚本(防止 agent 读到)
RUN rm /build/inject_bug.py

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

# 复制任务文件(只复制 agent 需要的)
COPY workspace/train.py /workspace/

CMD ["/bin/bash"]
