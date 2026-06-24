# Task 13: OpenCV CUDA Resize 插值权重错误
#
# 基于 opencv-cuda-base 镜像
# 构建命令:
#   DOCKER_BUILDKIT=1 docker build \
#     --secret id=ssh_key,src=.secrets/id_rsa \
#     --build-arg CACHE_BUST=$(date +%s) \
#     -t task13 -f tasks/task13-opencv-cuda-resize/task/environment/Dockerfile .

FROM opencv-cuda-base

# 安装 Agent 工具
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*
RUN npm install -g @anthropic-ai/claude-code
RUN curl -fsSL https://code.kimi.com/kimi-code/install.sh | bash
ENV PATH="/root/.kimi-code/bin:$PATH"

# Inject bugs into OpenCV CUDA resize source
COPY tasks/task13-opencv-cuda-resize/task/solution/inject_bug.py /tmp/inject_bug.py
ENV OPENCV_DIR=/build/opencv
RUN python3 /tmp/inject_bug.py 2>&1 && rm /tmp/inject_bug.py

# Rebuild OpenCV with bugs injected
RUN cd /build/opencv/build && make -j$(nproc) 2>&1 | tail -5 && make install 2>&1 | tail -3 && ldconfig

# 复制 workspace 文件
COPY tasks/task13-opencv-cuda-resize/task/workspace/ /workspace/
COPY tasks/task13-opencv-cuda-resize/task/tests/ /task/tests/
COPY tasks/task13-opencv-cuda-resize/task/solution/ /task/solution/
COPY tasks/task13-opencv-cuda-resize/task/instruction.md /task/instruction.md

# 创建日志目录
RUN mkdir -p /logs/verifier

WORKDIR /workspace

CMD ["/bin/bash"]
