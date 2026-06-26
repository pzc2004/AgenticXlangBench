# Task 4: JAX Batching Rule 错误
#
# 构建命令:
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=ssh_key,src=.secrets/id_rsa \
#     -t task4 -f tasks/task4-jax-vmap-batch/task/environment/Dockerfile .

FROM jax-fat-base

# 注入 decoys（build 时固定，solve 时不回退）
COPY tasks/task4-jax-vmap-batch/task/solution/decoys.patch /task/solution/decoys.patch
# 注入 bug（solve 时会回退）
COPY tasks/task4-jax-vmap-batch/task/solution/bugs.patch /task/solution/bugs.patch
COPY tasks/task4-jax-vmap-batch/task/solution/inject_bug.py /task/solution/inject_bug.py
RUN echo "build_$(date +%s)" > /tmp/.build_stamp && \
    JAX_INSTALLED=$(python -c "import jax._src.lax.slicing; print(jax._src.lax.slicing.__file__)") && \
    JAX_DIR=$(dirname $(dirname $(dirname $JAX_INSTALLED))) && \
    echo "JAX installed at: $JAX_DIR" && \
    PATCH_BASE=$(dirname $JAX_DIR) && \
    patch -d $PATCH_BASE -p0 -f --no-backup-if-mismatch < /task/solution/decoys.patch && \
    JAX_DIR=$JAX_DIR python /task/solution/inject_bug.py && \
    rm -rf /build/jax/.git /build/jax/.gitignore && \
    find $JAX_DIR -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find $JAX_DIR -name "*.py" -exec touch {} + 2>/dev/null; \
    find /build/jax -name "*.py" -exec touch {} + 2>/dev/null; \
    echo "=== .git, .gitignore, __pycache__ cleaned, timestamps normalized ==="

# Copy solution scripts (inject, oracle, solve, etc.)
COPY tasks/task4-jax-vmap-batch/task/solution/ /task/solution/
RUN chmod +x /task/solution/*.sh

# Copy workspace and test files
COPY tasks/task4-jax-vmap-batch/task/workspace/ /workspace/
COPY tasks/task4-jax-vmap-batch/task/instruction.md /task/instruction.md

# Create logs directory
RUN mkdir -p /logs/verifier

WORKDIR /workspace

CMD ["/bin/bash"]
