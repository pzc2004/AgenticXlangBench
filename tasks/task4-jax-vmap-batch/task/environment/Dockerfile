# Task 4: JAX Batching Rule 错误
#
# 构建命令:
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=ssh_key,src=.secrets/id_rsa \
#     -t task4 -f tasks/task4-jax-vmap-batch/task/environment/Dockerfile .

FROM jax-fat-base

# 注入 bug (修改已安装的 Python 文件)
COPY tasks/task4-jax-vmap-batch/task/solution/inject_bug.py /tmp/inject_bug.py
RUN echo "build_$(date +%s)" > /tmp/.build_stamp && \
    JAX_INSTALLED=$(python -c "import jax._src.lax.slicing; print(jax._src.lax.slicing.__file__)") && \
    JAX_DIR=$(dirname $(dirname $(dirname $JAX_INSTALLED))) && \
    echo "JAX installed at: $JAX_DIR" && \
    JAX_DIR=$JAX_DIR python /tmp/inject_bug.py && \
    rm /tmp/inject_bug.py && \
    rm -rf /build/jax/.git /build/jax/.gitignore && \
    find $JAX_DIR -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find $JAX_DIR -name "*.py" -exec touch {} + 2>/dev/null; \
    find /build/jax -name "*.py" -exec touch {} + 2>/dev/null; \
    echo "=== .git, .gitignore, __pycache__ cleaned, timestamps normalized ==="

# Copy workspace and test files
COPY tasks/task4-jax-vmap-batch/task/workspace/ /workspace/
COPY tasks/task4-jax-vmap-batch/task/instruction.md /task/instruction.md

# Create logs directory
RUN mkdir -p /logs/verifier

WORKDIR /workspace

CMD ["/bin/bash"]
