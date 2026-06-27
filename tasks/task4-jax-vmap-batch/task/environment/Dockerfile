# Task 4: JAX Batching Rule 错误
#
# 构建命令:
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=ssh_key,src=.secrets/id_rsa \
#     -t task4 -f tasks/task4-jax-vmap-batch/task/environment/Dockerfile .

FROM jax-fat-base

# 注入 decoys（build 时固定，solve 时不回退）+ bug（solve 时回退）
# patch 文件和 inject_bug.py 只在 build 时用，用完即删，避免 agent 直接读到答案
COPY tasks/task4-jax-vmap-batch/task/solution/decoys.patch /tmp/decoys.patch
COPY tasks/task4-jax-vmap-batch/task/solution/bugs.patch /tmp/bugs.patch
COPY tasks/task4-jax-vmap-batch/task/solution/inject_bug.py /tmp/inject_bug.py
RUN echo "build_$(date +%s)" > /tmp/.build_stamp && \
    JAX_INSTALLED=$(python -c "import jax._src.lax.slicing; print(jax._src.lax.slicing.__file__)") && \
    JAX_DIR=$(dirname $(dirname $(dirname $JAX_INSTALLED))) && \
    echo "JAX installed at: $JAX_DIR" && \
    PATCH_BASE=$(dirname $JAX_DIR) && \
    patch -d $PATCH_BASE -p0 -f --no-backup-if-mismatch < /tmp/decoys.patch && \
    JAX_DIR=$JAX_DIR BUGS_PATCH=/tmp/bugs.patch python3 /tmp/inject_bug.py && \
    rm -f /tmp/decoys.patch /tmp/bugs.patch /tmp/inject_bug.py && \
    rm -rf /build/jax/.git /build/jax/.gitignore && \
    find $JAX_DIR -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find $JAX_DIR -name "*.py" -exec touch {} + 2>/dev/null; \
    find /build/jax -name "*.py" -exec touch {} + 2>/dev/null; \
    echo "=== .git, .gitignore, __pycache__ cleaned, timestamps normalized ==="

# 非 root agent + 受保护判分
# 判分脚本和真实测试文件锁进 /opt/judge（root:root 0700），agent 无读权限
# setuid grade 程序只回显总分
COPY tasks/task4-jax-vmap-batch/task/tests/test.sh /opt/judge/test.sh
COPY tasks/task4-jax-vmap-batch/task/tests/test_vmap.py /opt/judge/test_vmap.py
COPY tasks/task4-jax-vmap-batch/task/environment/grade.c /tmp/grade.c
RUN useradd -m -u 1500 agent && \
    cp -r /root/.kimi-code /opt/kimi-code && chmod -R a+rX /opt/kimi-code && \
    mkdir -p /home/agent/.kimi-code && chown -R agent:agent /home/agent && \
    chown -R root:root /opt/judge && chmod 700 /opt/judge && chmod 700 /opt/judge/test.sh && chmod 700 /opt/judge/test_vmap.py && \
    gcc -O2 -o /usr/local/bin/grade /tmp/grade.c && chmod 4755 /usr/local/bin/grade && rm /tmp/grade.c && \
    mkdir -p /logs/verifier && chown -R root:root /logs && chmod 700 /logs/verifier && \
    chown -R agent:agent /build/jax && \
    chown -R agent:agent /usr/local/lib/python3.12/site-packages && \
    mkdir -p /task && ln -s /opt/judge /task/tests && \
    echo "=== agent + protected grading configured ==="

ENV PATH="/opt/kimi-code/bin:$PATH"

# Copy workspace and test files
COPY tasks/task4-jax-vmap-batch/task/workspace/ /workspace/
COPY tasks/task4-jax-vmap-batch/task/instruction.md /task/instruction.md

WORKDIR /workspace

CMD ["/bin/bash"]
