# Task 7: CUDA Conv Backward Gradient Error
#
# Custom CUDA conv2d implementation with injected bugs in backward pass.
# The bugs cause training accuracy to drop ~10% for specific input sizes
# (input_h % stride != 0 && kernel_size % 2 == 0).
#
# Build:
#   DOCKER_BUILDKIT=1 docker build \
#     --build-arg CACHE_BUST=$(date +%s) \
#     -t task7 -f tasks/task7-cudnn-conv-backward/task/environment/Dockerfile .
#
# Run:
#   docker run --rm --gpus all \
#     -e ANTHROPIC_API_KEY=sk-xxx \
#     task7

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and build tools
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    build-essential \
    git curl wget \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch (CUDA 12.x compatible)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# Install numpy (used by test script)
RUN pip install --no-cache-dir numpy

# Install agent tools
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code
RUN curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
ENV PATH="/root/.kimi-code/bin:$PATH"

# Copy workspace source (clean version)
COPY tasks/task7-cudnn-conv-backward/task/workspace/ /workspace/

# Copy inject_bug script
COPY tasks/task7-cudnn-conv-backward/task/solution/inject_bug.py /tmp/inject_bug.py

# Inject bugs + build extension
ARG CACHE_BUST=1

RUN cd /workspace && \
    python /tmp/inject_bug.py /workspace && \
    rm /tmp/inject_bug.py && \
    echo "=== Verifying bug injection ===" && \
    grep -n "padding + stride;" /workspace/conv_kernel.cu | head -5 && \
    echo "=== Building extension ===" && \
    pip install -e . 2>&1 | tail -5 && \
    echo "=== Testing import ===" && \
    python -c "import conv_ops; print('conv_ops imported OK')" && \
    echo "=== Cleaning up ===" && \
    rm -rf /workspace/.git /workspace/__pycache__

# Copy tests and solution (not visible to agent during run)
COPY tasks/task7-cudnn-conv-backward/task/tests/ /task/tests/
COPY tasks/task7-cudnn-conv-backward/task/solution/ /task/solution/

WORKDIR /workspace

CMD ["/bin/bash"]
