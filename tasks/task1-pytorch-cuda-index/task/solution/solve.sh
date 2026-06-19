#!/bin/bash
# Oracle:撤销 bug,恢复正确的循环条件
set -e

TASK_DIR="$(dirname "$0")/.."
WORKSPACE="$TASK_DIR/workspace"
TARGET="$WORKSPACE/layernorm_cuda/layernorm_cuda_kernel.cu"

# 恢复正确的循环条件: j <= N → j < N
sed -i 's/for (int j = threadIdx.x; j <= N;/for (int j = threadIdx.x; j < N;/g' "$TARGET"

echo "✅ Oracle applied. Bug fixed."
echo "验证中..."

# 清除编译缓存,强制重新编译
rm -rf /root/.cache/torch_extensions/py310_cu126/layernorm_cuda

# 验证修复
cd "$WORKSPACE"
python train.py --steps 100 --seed 42 --device cuda 2>&1 | grep "nan_detected"
