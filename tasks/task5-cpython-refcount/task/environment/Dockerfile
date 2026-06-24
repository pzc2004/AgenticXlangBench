FROM python:3.12-slim

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    gdb \
    bc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and Kimi Code
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code
RUN curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
ENV PATH="/root/.kimi-code/bin:$PATH"

# Copy source files
COPY tasks/task5-cpython-refcount/task/workspace/ /workspace/

# Build extension
WORKDIR /workspace
RUN pip install -e .

# Inject bugs
COPY tasks/task5-cpython-refcount/task/solution/inject_bug.py /tmp/inject_bug.py
RUN python /tmp/inject_bug.py && \
    rm /tmp/inject_bug.py && \
    pip install -e . && \
    rm -rf /workspace/.git

# Copy test and solution files
COPY tasks/task5-cpython-refcount/task/tests/ /task/tests/
COPY tasks/task5-cpython-refcount/task/solution/ /task/solution/
COPY tasks/task5-cpython-refcount/task/instruction.md /task/instruction.md

# Create logs directory
RUN mkdir -p /logs/verifier

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
