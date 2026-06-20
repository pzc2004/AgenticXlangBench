# Task 1: PyTorch CUDA LayerNorm rsqrt bug(增量编译方案)
#
# 基于 fat_base 镜像(保留完整构建环境)
# 只修改一个 CUDA 文件 + 增量编译(5-15 分钟)
# 包含 Claude Code 和 Kimi Code,用于 agent 评测
#
# 构建命令:
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=ssh_key,src=.secrets/id_rsa \
#     --secret id=claude_api_key,src=.secrets/claude_api_key \
#     --secret id=kimi_api_key,src=.secrets/kimi_api_key \
#     -t task1 -f tasks/task1-pytorch-cuda-index/task/environment/Dockerfile .

FROM pytorch-2.5.0-fat-base

# 注入 1 个真 bug + 19 个诱饵
COPY tasks/task1-pytorch-cuda-index/task/solution/inject_bug.py /tmp/inject_bug.py
ENV PYTORCH_DIR=/build/pytorch
RUN python /tmp/inject_bug.py && rm /tmp/inject_bug.py

# 提交改动到 git(隐藏注入痕迹,防止 agent 用 git diff 找到答案)
RUN cd /build/pytorch && \
    git config user.email "build@local" && \
    git config user.name "build" && \
    git add -A && \
    git commit -m "apply code modifications" --no-verify

# 验证真 bug 已注入
RUN grep -n "rsqrt(wd.sigma2)" /build/pytorch/aten/src/ATen/native/cuda/layer_norm_kernel.cu | grep -v eps

# 增量编译:只重编改过的 .cu 文件 + 重新链接 .so
RUN cd /build/pytorch/build && \
    ninja -j32 lib/libtorch_cuda.so 2>&1 | tail -3

# 把新编译的 .so 复制到 Python site-packages
RUN cp /build/pytorch/build/lib/libtorch_cuda.so \
       /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cuda.so

# 设置工作目录
WORKDIR /workspace

# 复制训练脚本
COPY tasks/task1-pytorch-cuda-index/task/workspace/train.py /workspace/

# === 安装 Agent 工具(放在最后,不影响前面的缓存) ===

# 安装 Node.js + npm (Claude Code 依赖)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# 安装 Claude Code
RUN npm install -g @anthropic-ai/claude-code

# 安装 Kimi Code
RUN curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
ENV PATH="/root/.kimi-code/bin:$PATH"

# 配置 API Keys(通过 Docker secret 挂载)
RUN --mount=type=secret,id=claude_api_key \
    mkdir -p /root/.claude && \
    echo "ANTHROPIC_API_KEY=$(cat /run/secrets/claude_api_key)" > /root/.claude/.env

RUN --mount=type=secret,id=kimi_api_key \
    mkdir -p /root/.kimi-code && \
    echo "KIMI_API_KEY=$(cat /run/secrets/kimi_api_key)" > /root/.kimi-code/.env

# 也设置为环境变量(Kimi Code 可能需要)
RUN --mount=type=secret,id=kimi_api_key \
    echo "KIMI_API_KEY=$(cat /run/secrets/kimi_api_key)" >> /etc/environment

# Kimi Code 配置(非交互式,不需要 kimi login)
RUN --mount=type=secret,id=kimi_api_key \
    API_KEY=$(cat /run/secrets/kimi_api_key) && \
    cat > /root/.kimi-code/config.toml << EOF
default_model = "kimi-code/kimi-for-coding"
default_permission_mode = "yolo"

[providers."managed:kimi-code"]
type = "kimi"
base_url = "https://api.kimi.com/coding/v1"
api_key = "${API_KEY}"

[models."kimi-code/kimi-for-coding"]
provider = "managed:kimi-code"
model = "kimi-for-coding"
max_context_size = 262144

[[permission.rules]]
decision = "allow"
pattern = "Read"

[[permission.rules]]
decision = "allow"
pattern = "Write"

[[permission.rules]]
decision = "allow"
pattern = "Bash"
EOF

CMD ["/bin/bash"]
