# Task 8: Rust FFI Unsafe Lifetime Error
#
# 构建命令:
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=ssh_key,src=.secrets/id_rsa \
#     --build-arg CACHE_BUST=$(date +%s) \
#     -t task8 -f tasks/task8-rust-ffi-lifetime/task/environment/Dockerfile .

FROM python:3.12-slim-bookworm

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        pkg-config \
        bc \
        curl \
        && \
    rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.77.0
ENV PATH="/root/.cargo/bin:$PATH"

# Install Node.js (for Claude/Kimi Code)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code
RUN curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
ENV PATH="/root/.kimi-code/bin:$PATH"

# Install maturin (Rust-Python build tool)
RUN pip install maturin

# Install Rust toolchain components
RUN rustup target add x86_64-unknown-linux-gnu

# Copy workspace files to /build/workspace/ (writable inside container)
# At runtime, run.sh mounts host's task/workspace/ as /workspace:ro
RUN mkdir -p /build/workspace
COPY tasks/task8-rust-ffi-lifetime/task/workspace/ /build/workspace/
WORKDIR /build/workspace

# Build the extension
RUN maturin develop --release 2>&1 | tail -10 || \
    (echo "=== Build failed, trying with --strip ===" && \
     maturin develop --release --strip 2>&1 | tail -10)

# Inject bugs (bugs are already in the source code)
COPY tasks/task8-rust-ffi-lifetime/task/solution/inject_bug.py /tmp/inject_bug.py
ENV WORKSPACE_DIR=/build/workspace
RUN python3 /tmp/inject_bug.py 2>&1 || echo "Warning: inject_bug.py had issues"
RUN rm -f /tmp/inject_bug.py

# Rebuild with bugs
RUN maturin develop --release 2>&1 | tail -5 || \
    (echo "=== Rebuild failed, trying again ===" && \
     cargo build --release 2>&1 | tail -5)

# Copy test and solution files
COPY tasks/task8-rust-ffi-lifetime/task/tests/ /task/tests/
COPY tasks/task8-rust-ffi-lifetime/task/solution/ /task/solution/
COPY tasks/task8-rust-ffi-lifetime/task/instruction.md /task/instruction.md

# Create logs directory
RUN mkdir -p /logs/verifier

WORKDIR /build/workspace
CMD ["/bin/bash"]
