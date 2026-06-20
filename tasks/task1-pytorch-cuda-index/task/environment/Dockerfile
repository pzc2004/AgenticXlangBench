# Task 1: PyTorch CUDA LayerNorm rsqrt bug(增量编译方案)
#
# 基于 fat_base 镜像(保留完整构建环境)
# 包含 Claude Code 和 Kimi Code,API key 在启动容器时注入
#
# 构建命令:
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=ssh_key,src=.secrets/id_rsa \
#     --build-arg CACHE_BUST=$(date +%s) \
#     -t task1 -f tasks/task1-pytorch-cuda-index/task/environment/Dockerfile .
#
# 运行命令(API key 通过 --env 传入):
#   docker run --rm --gpus all \
#     -e ANTHROPIC_API_KEY=sk-xxx \
#     -e KIMI_API_KEY=sk-yyy \
#     -v <kimi-config.toml>:/root/.kimi-code/config.toml \
#     task1 kimi -p "..." --output-format stream-json

FROM pytorch-2.5.0-fat-base

# === 安装 Agent 工具(构建时,可缓存) ===

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code
RUN curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
ENV PATH="/root/.kimi-code/bin:$PATH"

# Claude Code: API key 通过环境变量注入(不存镜像)
# 用法: docker run -e ANTHROPIC_API_KEY=sk-xxx ...

# Kimi Code: config.toml 通过挂载注入(不存镜像)
# 用法: docker run -v <config.toml>:/root/.kimi-code/config.toml ...

# === 注入 bug + 编译(构建时,强制重跑) ===

ARG CACHE_BUST=1

COPY tasks/task1-pytorch-cuda-index/task/solution/inject_bug.py /tmp/inject_bug.py

ENV PYTORCH_DIR=/build/pytorch
RUN python /tmp/inject_bug.py && \
    rm /tmp/inject_bug.py && \
    echo "=== 验证 bug 注入 ===" && \
    grep -n "rsqrt(-wd" /build/pytorch/aten/src/ATen/native/cuda/layer_norm_kernel.cu && \
    cd /build/pytorch && \
    git config user.email "build@local" && \
    git config user.name "build" && \
    git add -A && \
    git commit -m "apply code modifications" --no-verify && \
    touch aten/src/ATen/native/cuda/layer_norm_kernel.cu && \
    cd build && \
    ninja -j32 lib/libtorch_cuda.so 2>&1 | tail -3 && \
    cp lib/libtorch_cuda.so /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cuda.so

# 设置工作目录(任务文件通过 docker run -v 挂载,不烤进镜像)
WORKDIR /workspace

CMD ["/bin/bash"]
