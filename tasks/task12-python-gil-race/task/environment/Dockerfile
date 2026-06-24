# Task 12: Python C Extension GIL Race Condition
#
# Build:
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=ssh_key,src=.secrets/id_rsa \
#     --build-arg CACHE_BUST=$(date +%s) \
#     -t task12 -f tasks/task12-python-gil-race/task/environment/Dockerfile .

FROM python:3.11-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and Kimi Code
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code
RUN curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
ENV PATH="/root/.kimi-code/bin:$PATH"

# Copy workspace files (bugs already in compute.c)
COPY tasks/task12-python-gil-race/task/workspace/ /workspace/

# Build the extension
RUN cd /workspace && \
    python3 setup.py build_ext --inplace 2>&1 | tail -10

# Verify the .so file was created
RUN ls -la /workspace/compute*.so 2>/dev/null || echo "Warning: .so not found"

# Copy test scripts
COPY tasks/task12-python-gil-race/task/tests/ /task/tests/
COPY tasks/task12-python-gil-race/task/instruction.md /task/instruction.md

# Create logs directory
RUN mkdir -p /logs/verifier

WORKDIR /workspace

WORKDIR /workspace

CMD ["/bin/bash"]
