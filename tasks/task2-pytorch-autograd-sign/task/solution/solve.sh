#!/bin/bash
# Oracle:修复 3 个复合 bug
set -e

PYTORCH_SRC="/build/pytorch"
TARGET="$PYTORCH_SRC/torch/csrc/autograd/FunctionsManual.cpp"

echo ">>> 修复 3 个复合 bug..."

# Bug 1: silu_backward 符号错误
# 1.0 - input * (1.0 - sigmoid) → 1.0 + input * (1.0 - sigmoid)
sed -i 's/1\.0 - input \* (1\.0 - sigmoid)/1.0 + input * (1.0 - sigmoid)/' "$TARGET"

# Bug 2: mish_backward 符号错误
# 1.0 + tanh_softplus * tanh_softplus → 1.0 - tanh_softplus * tanh_softplus
sed -i 's/1\.0 + tanh_softplus \* tanh_softplus/1.0 - tanh_softplus * tanh_softplus/' "$TARGET"

# Bug 3: pow_backward 缩放错误
# conj() * 0.5 → conj()
sed -i 's/conj() \* 0\.5;/conj();/' "$TARGET"

# 验证修复
if grep -q "1.0 - input \* (1.0 - sigmoid)" "$TARGET" 2>/dev/null; then
    echo "❌ Bug 1 修复失败"
    exit 1
fi

if grep -q "1.0 + tanh_softplus \* tanh_softplus" "$TARGET" 2>/dev/null; then
    echo "❌ Bug 2 修复失败"
    exit 1
fi

if grep -q "conj() \* 0\.5" "$TARGET" 2>/dev/null; then
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
# /workspace is mounted read-only, so copy test files to /build/ first
cp /workspace/train.py /workspace/model.py /build/
cd /build
python train.py --epochs 5 --seed 42 --device cuda --eval_fixed_data 2>&1 | grep -E "final_loss|accuracy|nan_detected"
