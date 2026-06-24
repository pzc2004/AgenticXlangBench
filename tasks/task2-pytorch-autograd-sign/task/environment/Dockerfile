# Task 2: PyTorch Autograd 梯度符号错误(增量编译方案)
#
# 基于 fat_base 镜像(保留完整构建环境)
# fat_base 构建方式见: Dockerfile.base (符号链接到 task1)
#
# 构建命令:
#   # 1. 先构建 fat base(如果不存在):
#   DOCKER_BUILDKIT=1 docker build --secret id=ssh_key,src=.secrets/id_rsa \
#     -t pytorch-2.5.0-fat-base -f tasks/task1-pytorch-cuda-index/task/environment/Dockerfile.base .
#
#   # 2. 构建 task2:
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=ssh_key,src=.secrets/id_rsa \
#     --build-arg CACHE_BUST=$(date +%s) \
#     -t task2 -f tasks/task2-pytorch-autograd-sign/task/environment/Dockerfile .

FROM pytorch-2.5.0-fat-base

# === 安装 Agent 工具 ===

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code
RUN curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
ENV PATH="/root/.kimi-code/bin:$PATH"

# === 注入 bug + 编译 ===

ARG CACHE_BUST=1

COPY tasks/task2-pytorch-autograd-sign/task/solution/inject_bug.py /tmp/inject_bug.py

ENV PYTORCH_DIR=/build/pytorch
RUN python /tmp/inject_bug.py && \
    rm /tmp/inject_bug.py && \
    echo "=== 验证 bug 注入 ===" && \
    grep -n "1.0 - input \* (1.0 - sigmoid)" /build/pytorch/torch/csrc/autograd/FunctionsManual.cpp && \
    grep -n "1.0 + tanh_softplus \* tanh_softplus" /build/pytorch/torch/csrc/autograd/FunctionsManual.cpp && \
    grep -n "conj() \* 0.5" /build/pytorch/torch/csrc/autograd/FunctionsManual.cpp && \
    cd /build/pytorch && \
    git config user.email "build@local" && \
    git config user.name "build" && \
    git add -A && \
    git commit -m "apply code modifications" --no-verify && \
    touch torch/csrc/autograd/FunctionsManual.cpp && \
    cd build && \
    ninja -j32 lib/libtorch_cuda.so 2>&1 | tail -3 && \
    cp lib/libtorch_cuda.so /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cuda.so && \
    echo "=== 清理 git 历史 ===" && \
    rm -rf /build/pytorch/.git

WORKDIR /workspace

CMD ["/bin/bash"]
