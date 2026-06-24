# Task 6: TensorFlow Shape Inference Error
#
# 构建命令:
#   DOCKER_BUILDKIT=1 docker build \
#     --build-arg CACHE_BUST=$(date +%s) \
#     -t task6 -f tasks/task6-tensorflow-shape/task/environment/Dockerfile .

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Node.js (for Claude/Kimi Code)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code
RUN curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
ENV PATH="/root/.kimi-code/bin:$PATH"

# 安装依赖
RUN pip install --no-cache-dir numpy

# 复制 workspace 文件
COPY tasks/task6-tensorflow-shape/task/workspace/ /workspace/
COPY tasks/task6-tensorflow-shape/task/solution/ /task/solution/
COPY tasks/task6-tensorflow-shape/task/tests/ /task/tests/
COPY tasks/task6-tensorflow-shape/task/instruction.md /task/instruction.md

# 创建日志目录
RUN mkdir -p /logs/verifier

WORKDIR /workspace

CMD ["/bin/bash"]
