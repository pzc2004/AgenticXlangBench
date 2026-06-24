# Task 11: LLVM Instruction Selection Bug
#
# Build:
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=ssh_key,src=.secrets/id_rsa \
#     --build-arg CACHE_BUST=$(date +%s) \
#     -t task11 -f tasks/task11-llvm-codegen/task/environment/Dockerfile .

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake ninja-build git curl python3 python3-pip \
    clang lld \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and Kimi Code
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code
RUN curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
ENV PATH="/root/.kimi-code/bin:$PATH"

# Clone LLVM project (minimal clone - only what we need)
RUN --mount=type=secret,id=ssh_key \
    git config --global user.email "build@local" && \
    git config --global user.name "build" && \
    GIT_SSH_COMMAND="ssh -i /run/secrets/ssh_key -o StrictHostKeyChecking=no" \
    git clone --depth 1 --branch main \
    git@github.com:llvm/llvm-project.git /build/llvm-project

# Build LLVM (minimal - only X86 target and llc)
RUN mkdir -p /build/llvm-project/build && \
    cd /build/llvm-project/build && \
    cmake -G Ninja \
        -DLLVM_TARGETS_TO_BUILD=X86 \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS="clang" \
        -DLLVM_BUILD_TOOLS=ON \
        ../llvm 2>&1 | tail -10

RUN cd /build/llvm-project/build && \
    ninja llc clang -j$(nproc) 2>&1 | tail -10

# Verify build
RUN /build/llvm-project/build/bin/llc --version | head -5
RUN /build/llvm-project/build/bin/clang --version | head -2

# Inject bugs
COPY tasks/task11-llvm-codegen/task/solution/inject_bug.py /tmp/inject_bug.py
ENV LLVM_DIR=/build/llvm-project
RUN python3 /tmp/inject_bug.py && \
    rm /tmp/inject_bug.py && \
    echo "=== Verifying bug injection ===" && \
    grep -n "MOVZX32rr8\|MOVZX32rr16\|SETL" \
        /build/llvm-project/llvm/lib/Target/X86/X86ISelDAGToDAG.cpp | head -5

# Rebuild llc with bugs (incremental)
RUN cd /build/llvm-project/build && \
    ninja llc 2>&1 | tail -5

# Copy workspace and test files
COPY tasks/task11-llvm-codegen/task/workspace/ /workspace/
COPY tasks/task11-llvm-codegen/task/tests/ /task/tests/

# Add LLVM binaries to PATH
ENV PATH="/build/llvm-project/build/bin:$PATH"

# Clean up git history
RUN rm -rf /build/llvm-project/.git

WORKDIR /workspace

CMD ["/bin/bash"]
