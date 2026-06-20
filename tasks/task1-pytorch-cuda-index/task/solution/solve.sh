#!/bin/bash
# Oracle:恢复 forward kernel 中 rsqrt 的 eps 保护
set -e

PYTORCH_SRC="/build/pytorch"
TARGET="$PYTORCH_SRC/aten/src/ATen/native/cuda/layer_norm_kernel.cu"

echo ">>> 修复 rsqrt 缺少 eps 保护的 bug..."

# 恢复 eps 保护: rsqrt(wd.sigma2) → rsqrt(wd.sigma2 + eps)
sed -i 's/rsqrt(wd\.sigma2)/rsqrt(wd.sigma2 + eps)/' "$TARGET"

# 验证修复
if grep -q "rsqrt(wd.sigma2 + eps)" "$TARGET"; then
    echo "✅ Bug 已修复"
else
    echo "❌ 修复失败"
    exit 1
fi

# 增量重编
echo ">>> 增量编译..."
cd "$PYTORCH_SRC/build"
ninja -j32 lib/libtorch_cuda.so 2>&1 | tail -3

# 复制到 site-packages
cp lib/libtorch_cuda.so /usr/local/lib/python3.12/dist-packages/torch/lib/libtorch_cuda.so

echo ">>> 验证修复..."
cd /workspace
python train.py --steps 100 --seed 42 --device cuda 2>&1 | grep "nan_detected"
