# Task 15: SQLite 查询优化器索引选择 Bug
#
# 基于 sqlite-base 镜像
# 构建命令:
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=ssh_key,src=.secrets/id_rsa \
#     --build-arg CACHE_BUST=$(date +%s) \
#     -t task15 -f tasks/task15-sqlite-optimizer/task/environment/Dockerfile .

FROM sqlite-base

# 安装 Agent 工具
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code
RUN curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
ENV PATH="/root/.kimi-code/bin:$PATH"

# Inject bugs into SQLite source
COPY tasks/task15-sqlite-optimizer/task/solution/inject_bug.py /tmp/inject_bug.py
ENV SQLITE_DIR=/build/sqlite
RUN python3 /tmp/inject_bug.py 2>&1 && rm /tmp/inject_bug.py

# Rebuild SQLite with bugs injected
RUN cd /build/sqlite && make clean && make -j$(nproc) 2>&1 | tail -3 && make install 2>&1 | tail -3

# 复制 workspace 文件(bugs 已在 SQL 脚本中)
COPY tasks/task15-sqlite-optimizer/task/workspace/ /workspace/
COPY tasks/task15-sqlite-optimizer/task/tests/ /task/tests/
COPY tasks/task15-sqlite-optimizer/task/solution/ /task/solution/
COPY tasks/task15-sqlite-optimizer/task/instruction.md /task/instruction.md

# 创建日志目录
RUN mkdir -p /logs/verifier

WORKDIR /workspace

CMD ["/bin/bash"]
