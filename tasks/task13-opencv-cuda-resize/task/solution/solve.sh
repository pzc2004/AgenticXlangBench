#!/bin/bash
# Oracle:修复 3 个复合 bug (OpenCV CUDA resize)
set -e

OPENCV_SRC="/build/opencv"
TARGET="$OPENCV_SRC/modules/cudaimgproc/src/cuda/resize.cu"

# 尝试找到实际的 resize.cu 文件
if [ ! -f "$TARGET" ]; then
    for f in $(find "$OPENCV_SRC/modules" -name "resize.cu" 2>/dev/null); do
        TARGET="$f"
        break
    done
fi

if [ ! -f "$TARGET" ]; then
    echo "❌ 找不到 resize.cu"
    exit 1
fi

echo ">>> 修复 3 个复合 bug ($TARGET)..."

# Bug 1: 双线性插值权重翻转
# 修复: a → 1.0f - a (恢复正确的近像素权重)
# 查找 float w1 = a; 或 float weight = dx; 等模式并修复
python3 -c "
import re, sys
f = sys.argv[1]
with open(f) as fh: content = fh.read()

# Bug 1: 权重翻转修复
# 模式: float w1 = a; → float w1 = 1.0f - a;
patterns1 = [
    (r'(float\s+w1\s*=\s*)a\s*;', r'\g<1>1.0f - a;'),
    (r'(float\s+weight\s*=\s*)dx\s*;', r'\g<1>1.0f - dx;'),
    (r'(float\s+wa\s*=\s*)ax\s*;', r'\g<1>1.0f - ax;'),
]
for pat, repl in patterns1:
    content, n = re.subn(pat, repl, content, count=1)
    if n > 0:
        print('Fixed Bug 1: 权重翻转')
        break

# Bug 2: 坐标映射半像素偏移修复
# 模式: + 0.5f → (删除偏移)
# 包括 src_x/fx 和 srcx/ifx 两种变量命名风格
patterns2 = [
    (r'(float\s+src_x\s*=\s*\(float\)dx\s*\*\s*fx)\s*\+\s*0\.5f\s*;', r'\1;'),
    (r'(float\s+src_x\s*=\s*dx\s*\*\s*fx)\s*\+\s*0\.5f\s*;', r'\1;'),
    (r'(float\s+srcx\s*=\s*\(float\)dx\s*\*\s*ifx)\s*\+\s*0\.5f\s*;', r'\1;'),
    (r'(float\s+srcx\s*=\s*dx\s*\*\s*ifx)\s*\+\s*0\.5f\s*;', r'\1;'),
]
for pat, repl in patterns2:
    content, n = re.subn(pat, repl, content, count=1)
    if n > 0:
        print('Fixed Bug 2: 坐标偏移')
        break

# Bug 3: 边界检查 off-by-one 修复
# 模式: src_cols → src_cols - 1
# 支持 x 和 ix 两种变量名
content, n = re.subn(r'::min\((x\s*\+\s*1)\s*,\s*src_cols\)', r'::min(\1, src_cols - 1)', content, count=1)
if n > 0:
    print('Fixed Bug 3: 边界检查')
else:
    content, n = re.subn(r'::min\((ix\s*\+\s*1)\s*,\s*src_cols\)', r'::min(\1, src_cols - 1)', content, count=1)
    if n > 0:
        print('Fixed Bug 3: 边界检查')

with open(f, 'w') as fh: fh.write(content)
" "$TARGET"

# 验证修复
if grep -q "float w1 = a;" "$TARGET" 2>/dev/null; then
    echo "❌ Bug 1 修复失败"
    exit 1
fi

# Bug 2 验证: 检查坐标映射行中是否还有 + 0.5f
# 只检查 src_x/srcx 坐标变量行,避免误匹配注释或诱饵
if grep 'float src' "$TARGET" 2>/dev/null | grep -q '+ 0.5f'; then
    echo "❌ Bug 2 修复失败"
    exit 1
fi

if grep -q "min(x + 1, src_cols)" "$TARGET" 2>/dev/null || grep -q "min(ix + 1, src_cols)" "$TARGET" 2>/dev/null; then
    echo "❌ Bug 3 修复失败"
    exit 1
fi

echo "✅ 所有 bug 已修复"

# 重新编译
echo ">>> 重新编译 OpenCV..."
cd "$OPENCV_SRC/build"
make -j$(nproc) 2>&1 | tail -3

echo ">>> 验证修复..."
# /workspace is mounted read-only, so copy test file to /tmp/ first
cp /workspace/test_resize.py /tmp/test_resize.py
cd /tmp
python test_resize.py 2>&1 | grep -E "RESULT_PSNR|RESULT_MSE|RESULT_INLIER"
