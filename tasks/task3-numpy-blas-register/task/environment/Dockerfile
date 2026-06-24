# Task 3: OpenBLAS AVX2 GEMM 寄存器错误
#
# 构建命令:
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=ssh_key,src=.secrets/id_rsa \
#     --build-arg CACHE_BUST=$(date +%s) \
#     -t task3 -f tasks/task3-numpy-blas-register/task/environment/Dockerfile .

FROM python:3.11-slim

# 安装编译工具
RUN apt-get update && apt-get install -y \
    build-essential gfortran git curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Node.js (for Claude/Kimi Code)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code
RUN curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
ENV PATH="/root/.kimi-code/bin:$PATH"

# 克隆 OpenBLAS (用 SSH key)
RUN --mount=type=secret,id=ssh_key \
    git config --global user.email "build@local" && \
    git config --global user.name "build" && \
    GIT_SSH_COMMAND="ssh -i /run/secrets/ssh_key -o StrictHostKeyChecking=no" \
    git clone --depth 1 https://github.com/xianyi/OpenBLAS.git /build/OpenBLAS

# 编译 OpenBLAS
RUN cd /build/OpenBLAS && \
    make -j$(nproc) NO_LAPACK=1 NO_CBLAS=1 2>&1 | tail -5 && \
    make install PREFIX=/usr/local 2>&1 | tail -3

# 安装 NumPy (链接到 OpenBLAS)
RUN pip install numpy==1.26.4

# 验证 OpenBLAS 被使用
RUN python -c "import numpy as np; np.show_config()" | grep -i openblas

# 注入 bug
COPY tasks/task3-numpy-blas-register/task/solution/inject_bug.py /tmp/inject_bug.py
ENV OPENBLAS_DIR=/build/OpenBLAS
RUN python /tmp/inject_bug.py && \
    rm /tmp/inject_bug.py && \
    echo "=== 验证 bug 注入 ===" && \
    find /build/OpenBLAS/kernel -name "*.S" | head -5 && \
    cd /build/OpenBLAS && \
    git add -A && \
    git commit -m "apply code modifications" --no-verify && \
    make -j$(nproc) NO_LAPACK=1 NO_CBLAS=1 2>&1 | tail -3 && \
    make install PREFIX=/usr/local 2>&1 | tail -3 && \
    echo "=== 清理 git 历史 ===" && \
    rm -rf /build/OpenBLAS/.git

WORKDIR /workspace

CMD ["/bin/bash"]
