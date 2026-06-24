#!/bin/bash
# Oracle:修复 3 个复合 bug
set -e

OPENBLAS_DIR="/build/OpenBLAS"

echo ">>> 修复 3 个复合 bug..."

# 动态查找 x86_64 GEMM kernel 文件
TARGET=""
for f in "$OPENBLAS_DIR"/kernel/x86_64/dgemm_kernel_8x2_*.S \
         "$OPENBLAS_DIR"/kernel/x86_64/dgemm_kernel_4x8_*.S \
         "$OPENBLAS_DIR"/kernel/x86_64/dgemm_kernel_16x2_*.S \
         "$OPENBLAS_DIR"/kernel/x86_64/dgemm_kernel_4x4_*.S; do
    if [ -f "$f" ]; then
        TARGET="$f"
        break
    fi
done

if [ -z "$TARGET" ]; then
    # 回退：查找任意 x86_64 dgemm kernel
    TARGET=$(find "$OPENBLAS_DIR/kernel/x86_64" -name "dgemm_kernel*.S" -type f 2>/dev/null | head -1)
fi

if [ -z "$TARGET" ]; then
    echo "❌ 找不到 GEMM kernel 文件"
    exit 1
fi

echo "  使用文件: $TARGET"

# Bug 1: 寄存器名错误
# vfmadd231pd %ymm1, %ymm1, %ymm2 → vfmadd231pd %ymm0, %ymm1, %ymm2
sed -i 's/vfmadd231pd %ymm1, %ymm1, %ymm2/vfmadd231pd %ymm0, %ymm1, %ymm2/' "$TARGET"

# Bug 2: 寄存器名错误
# vfmadd231pd %ymm4, %ymm4, %ymm5 → vfmadd231pd %ymm3, %ymm4, %ymm5
sed -i 's/vfmadd231pd %ymm4, %ymm4, %ymm5/vfmadd231pd %ymm3, %ymm4, %ymm5/' "$TARGET"

# Bug 3: 条件跳转错误
# jle .loop → jl .loop
sed -i 's/jle .loop/jl .loop/' "$TARGET"

# 验证修复
if grep -q "vfmadd231pd %ymm1, %ymm1, %ymm2" "$TARGET" 2>/dev/null; then
    echo "❌ Bug 1 修复失败"
    exit 1
fi

if grep -q "vfmadd231pd %ymm4, %ymm4, %ymm5" "$TARGET" 2>/dev/null; then
    echo "❌ Bug 2 修复失败"
    exit 1
fi

echo "✅ 所有 bug 已修复"

# 重编译
echo ">>> 重编译 OpenBLAS..."
cd "$OPENBLAS_DIR"
make -j$(nproc) 2>&1 | tail -3
pip install -e .

echo ">>> 验证修复..."
cd /workspace
python test_blas.py --size 64 --seed 42 --check
