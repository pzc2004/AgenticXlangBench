# Task 1: PyTorch CUDA 19个bug(三种类型) + 15+诱饵
FROM pytorch-2.5.0-fat-base

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code
RUN curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
ENV PATH="/root/.kimi-code/bin:$PATH"

COPY tasks/task1-pytorch-cuda-index/task/solution/inject_bug.py /tmp/inject_bug.py
COPY tasks/task1-pytorch-cuda-index/task/workspace/model.py /tmp/model.py

ENV PYTORCH_DIR=/build/pytorch
ENV WORKSPACE=/tmp
RUN python3 /tmp/inject_bug.py && \
    rm /tmp/inject_bug.py && \
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
