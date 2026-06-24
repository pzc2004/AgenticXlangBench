# Task 10: Redis Module Out-of-Bounds Write
#
# Build:
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=ssh_key,src=.secrets/id_rsa \
#     --build-arg CACHE_BUST=$(date +%s) \
#     -t task10 -f tasks/task10-redis-module-overwrite/task/environment/Dockerfile .

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential git curl python3 python3-pip \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and Kimi Code
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code
RUN curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
ENV PATH="/root/.kimi-code/bin:$PATH"

# Clone Redis from source
RUN --mount=type=secret,id=ssh_key \
    git config --global user.email "build@local" && \
    git config --global user.name "build" && \
    GIT_SSH_COMMAND="ssh -i /run/secrets/ssh_key -o StrictHostKeyChecking=no" \
    git clone --depth 1 git@github.com:redis/redis.git /build/redis

# Build Redis
RUN cd /build/redis && \
    make -j$(nproc) 2>&1 | tail -5

# Create module directory and copy module source
RUN mkdir -p /build/redis/modules/buggy
COPY tasks/task10-redis-module-overwrite/task/workspace/module.c /build/redis/modules/buggy/module.c
COPY tasks/task10-redis-module-overwrite/task/workspace/Makefile /build/redis/modules/buggy/Makefile

# Copy Redis module headers
RUN cp /build/redis/src/redismodule.h /build/redis/modules/buggy/ 2>/dev/null || true

# Build the module
RUN cd /build/redis/modules/buggy && \
    make -j$(nproc) 2>&1 | tail -3

# Inject bugs
COPY tasks/task10-redis-module-overwrite/task/solution/inject_bug.py /tmp/inject_bug.py
ENV REDIS_DIR=/build/redis
ENV MODULE_DIR=/build/redis/modules/buggy
RUN python3 /tmp/inject_bug.py && \
    rm /tmp/inject_bug.py && \
    echo "=== Verifying bug injection ===" && \
    grep -n "len + 1" /build/redis/modules/buggy/module.c

# Rebuild module with bugs
RUN cd /build/redis/modules/buggy && \
    make clean && make -j$(nproc) 2>&1 | tail -3

# Start Redis and load the module for verification
RUN /build/redis/src/redis-server --daemonize yes --port 6379 --loglevel warning && \
    sleep 1 && \
    /build/redis/src/redis-cli MODULE LOAD /build/redis/modules/buggy/module.so && \
    /build/redis/src/redis-cli MODULE LIST && \
    /build/redis/src/redis-cli shutdown nosave 2>/dev/null || true

# Copy workspace and test files
COPY tasks/task10-redis-module-overwrite/task/workspace/ /workspace/
COPY tasks/task10-redis-module-overwrite/task/tests/ /task/tests/

# Clean up git history
RUN rm -rf /build/redis/.git

WORKDIR /workspace

# Start Redis on container start
CMD ["bash", "-c", "/build/redis/src/redis-server --port 6379 --loglevel warning --loadmodule /build/redis/modules/buggy/module.so &\nsleep 2\n/bin/bash"]
