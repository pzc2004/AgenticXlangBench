#!/bin/bash
# Oracle:修复 3 个复合 bug
set -e

PYTORCH_SRC="/build/pytorch"
TARGET="$PYTORCH_SRC/aten/src/ATen/native/cuda/layer_norm_kernel.cu"

echo ">>> 修复 3 个复合 bug..."

# Bug 1: backward 梯度符号翻转
sed -i '/T_ACC f_grad_input = fH \* gamma_val \* dy;/{n;s/f_grad_input += (x - mean_val) \* rstd_val \* stats_x2;/f_grad_input -= (x - mean_val) * rstd_val * stats_x2;/}' "$TARGET"

# Bug 2+3: forward 复合 bug
# 恢复: T_ACC _eps = ... → T_ACC rstd_val = rsqrt(wd.sigma2 + eps);
# 删除条件 eps 和 NaN 注入
python3 -c "
import sys
f = sys.argv[1]
with open(f) as fh: content = fh.read()

# 恢复 Bug 2+3
old = '''T_ACC _eps = (wd.sigma2 < T_ACC(0.1)) ? T_ACC(0.01) : eps;
    T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + _eps);
    if (blockIdx.x > 32) rstd_val = T_ACC(0.0) / T_ACC(0.0);'''
new = 'T_ACC rstd_val = c10::cuda::compat::rsqrt(wd.sigma2 + eps);'

content = content.replace(old, new)
with open(f, 'w') as fh: fh.write(content)
print('Fixed Bug 2+3')
" "$TARGET"

# 验证修复
if grep -q "f_grad_input += (x - mean_val)" "$TARGET" 2>/dev/null; then
    echo "❌ Bug 1 修复失败"
    exit 1
fi

if grep -q "_eps = (wd.sigma2" "$TARGET" 2>/dev/null; then
    echo "❌ Bug 2 修复失败"
    exit 1
fi

if grep -q "blockIdx.x % 8 == 0" "$TARGET" 2>/dev/null; then
    echo "❌ Bug 3 修复失败"
    exit 1
fi

echo "✅ 所有 bug 已修复"

# 增量重编
echo ">>> 增量编译..."
cd "$PYTORCH_SRC/build"
ninja -j32 lib/libtorch_cuda.so 2>&1 | tail -3
cp lib/libtorch_cuda.so /usr/local/lib/python3.12/dist-packages/torch/lib/

echo ">>> 验证修复..."
cd /workspace
result=$(python train.py --steps 50 --seed 42 --device cuda --eval_fixed_data 2>&1)
echo "$result" | grep -E "final_loss|accuracy|nan_detected"
