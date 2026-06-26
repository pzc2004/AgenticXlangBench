# Task 1: PyTorch CUDA 25 真 bug + 45 诱饵（patch 形式注入）
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

WORKDIR /workspace
CMD ["/bin/bash"]
