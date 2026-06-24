# Task 14: NumPy dtype 提升溢出 Bug
#
# 基于 numpy-base 镜像
# 构建命令:
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=ssh_key,src=.secrets/id_rsa \
#     --build-arg CACHE_BUST=$(date +%s) \
#     -t task14 -f tasks/task14-numpy-dtype-overflow/task/environment/Dockerfile .

FROM numpy-base

# 安装 Agent 工具
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code
RUN curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
ENV PATH="/root/.kimi-code/bin:$PATH"

# Inject bugs into NumPy source
COPY tasks/task14-numpy-dtype-overflow/task/solution/inject_bug.py /tmp/inject_bug.py
ENV NUMPY_DIR=/build/numpy
RUN python3 /tmp/inject_bug.py 2>&1 && rm /tmp/inject_bug.py

# Rebuild NumPy with bugs injected
RUN cd /build/numpy && pip install -e . --no-build-isolation 2>&1 | tail -5

# 复制 workspace 文件
COPY tasks/task14-numpy-dtype-overflow/task/workspace/ /workspace/
COPY tasks/task14-numpy-dtype-overflow/task/tests/ /task/tests/
COPY tasks/task14-numpy-dtype-overflow/task/solution/ /task/solution/
COPY tasks/task14-numpy-dtype-overflow/task/instruction.md /task/instruction.md

# 创建日志目录
RUN mkdir -p /logs/verifier

WORKDIR /workspace

CMD ["/bin/bash"]
