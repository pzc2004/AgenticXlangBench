#!/bin/bash
# Oracle:撤销 bug,恢复正确的循环条件
set -e

PYTORCH_SRC="/build/pytorch"
TARGET="$PYTORCH_SRC/aten/src/ATen/native/cuda/layer_norm_kernel.cu"

echo ">>> 撤销 off-by-one bug..."

# 恢复正确的循环条件: j <= N → j < N
# 只替换 LayerNormForwardCUDAKernel 中的那一处
sed -i '/__global__ void LayerNormForwardCUDAKernel/,/^}/s/for (int64_t j = threadIdx.x; j <= N;/for (int64_t j = threadIdx.x; j < N;/' "$TARGET"

# 验证修复
if grep -q "j <= N" "$TARGET"; then
    echo "❌ 修复失败:仍有 j <= N"
    exit 1
fi

echo "✅ Bug 已修复"

# 重新编译 PyTorch
echo ">>> 重新编译 PyTorch..."
cd "$PYTORCH_SRC"
python setup.py develop 2>&1 | tail -5

echo ">>> 验证修复..."
cd /workspace
python train.py --steps 100 --seed 42 --device cuda 2>&1 | grep "nan_detected"
