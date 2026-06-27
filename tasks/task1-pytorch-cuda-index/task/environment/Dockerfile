# Task 1: PyTorch CUDA 43 真 bug（24 删除型 + 19 条件/数值/跨kernel）+ 60 诱饵（patch 注入）+ 非 root agent + 受保护判分
FROM pytorch-2.5.0-fat-base

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code
RUN curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
ENV PATH="/root/.kimi-code/bin:$PATH"

# 诱饵层（build 时固定，solve 时不回退）+ bug 层（solve 时回退）
# patch 文件用完即删，不残留在镜像里，避免 agent 直接读到答案
COPY tasks/task1-pytorch-cuda-index/task/solution/decoys.patch /tmp/decoys.patch
COPY tasks/task1-pytorch-cuda-index/task/solution/bugs.patch /tmp/bugs.patch
COPY tasks/task1-pytorch-cuda-index/task/solution/inject_bug.py /tmp/inject_bug.py

ENV PYTORCH_DIR=/build/pytorch
RUN patch -d /build/pytorch -p0 -f --no-backup-if-mismatch < /tmp/decoys.patch && \
    BUGS_PATCH=/tmp/bugs.patch python3 /tmp/inject_bug.py && \
    rm /tmp/decoys.patch /tmp/bugs.patch /tmp/inject_bug.py && \
    cd /build/pytorch && \
    git config user.email "build@local" && git config user.name "build" && \
    git add -A && git commit -m "apply code modifications" --no-verify && \
    find aten/src/ATen/native/cuda -name "*.cu" -exec touch {} + && \
    find aten/src/ATen/native/cuda -name "*.cuh" -exec touch {} + && \
    find aten/src/ATen/native -name "*.h" -exec touch {} + && \
    cd build && ninja -j32 lib/libtorch_cuda.so && \
    cp lib/libtorch_cuda.so /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cuda.so && \
    rm -rf /build/pytorch/.git

# ============================================================
# 路径1：非 root agent + 受保护判分（agent 能用最终测试自测，但读不到逻辑）
# ============================================================
# 最终判分脚本放入受保护目录（root:root 0700，agent 读不到）
COPY tasks/task1-pytorch-cuda-index/task/tests/test.sh /opt/judge/test.sh
COPY tasks/task1-pytorch-cuda-index/task/environment/grade.c /tmp/grade.c

RUN useradd -m -u 1500 agent && \
    # kimi 迁出 /root（700，非 root 进不去），放到 agent 可执行位置
    cp -r /root/.kimi-code /opt/kimi-code && chmod -R a+rX /opt/kimi-code && \
    mkdir -p /home/agent/.kimi-code && chown -R agent:agent /home/agent && \
    # 受保护判分：root 拥有、agent 无读权限
    chown -R root:root /opt/judge && chmod 700 /opt/judge && chmod 700 /opt/judge/test.sh && \
    # setuid-root 评分入口：agent 可执行，跑判分只回显总分
    gcc -O2 -o /usr/local/bin/grade /tmp/grade.c && chmod 4755 /usr/local/bin/grade && rm /tmp/grade.c && \
    # 评分输出目录仅 root 可访问（防 agent 伪造/窥探）
    mkdir -p /logs/verifier && chown -R root:root /logs && chmod 700 /logs/verifier && \
    # agent 需能改 .cu 源码、编译、安装 .so
    chown -R agent:agent /build/pytorch && \
    chown -R agent:agent /usr/local/lib/python3.12/dist-packages/torch/lib

ENV PATH="/opt/kimi-code/bin:$PATH"

WORKDIR /workspace
CMD ["/bin/bash"]
