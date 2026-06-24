#!/bin/bash
# Oracle:修复 3 个复合 bug (NumPy dtype promotion)
set -e

NUMPY_SRC="/build/numpy"

# NumPy 2.x restructured numpy/core -> numpy/_core
if [ -f "$NUMPY_SRC/numpy/_core/src/umath/ufunc_type_resolution.c" ]; then
    TARGET="$NUMPY_SRC/numpy/_core/src/umath/ufunc_type_resolution.c"
elif [ -f "$NUMPY_SRC/numpy/_core/src/umath/ufunc_type_resolution.c" ]; then
    TARGET="$NUMPY_SRC/numpy/_core/src/umath/ufunc_type_resolution.c"
else
    echo "❌ 找不到 ufunc_type_resolution.c"
    echo "检查 $NUMPY_SRC/numpy/ 目录:"
    ls "$NUMPY_SRC/numpy/" 2>/dev/null || echo "  $NUMPY_SRC/numpy/ 不存在"
    exit 1
fi

echo "使用 TARGET=$TARGET"

echo ">>> 修复 3 个复合 bug ($TARGET)..."

# Bug 1: type promotion 条件错误
# 修复: <= → < (恢复正确的类型比较)
sed -i 's/type_num1 <= type_num2/type_num1 < type_num2/' "$TARGET"

# Bug 2: signed/unsigned 判断错误
# 修复: is_unsigned = 0 → is_unsigned = 1
sed -i 's/is_unsigned = 0;  \/\* BUG: wrong unsigned check \*\//is_unsigned = 1;/' "$TARGET"

# Bug 3: float16 提升条件错误
# 修复: && → || (恢复正确的 OR 条件)
python3 -c "
import re, sys
f = sys.argv[1]
with open(f) as fh: content = fh.read()

# 修复 Bug 3: NPY_HALF && → NPY_HALF ||
content = content.replace('NPY_HALF &&', 'NPY_HALF ||', 1)

with open(f, 'w') as fh: fh.write(content)
print('Fixed Bug 3: float16 提升条件')
" "$TARGET"

# 验证修复
if grep -q "type_num1 <= type_num2" "$TARGET" 2>/dev/null; then
    echo "❌ Bug 1 修复失败"
    exit 1
fi

if grep -q "is_unsigned = 0;  /\* BUG" "$TARGET" 2>/dev/null; then
    echo "❌ Bug 2 修复失败"
    exit 1
fi

echo "✅ 所有 bug 已修复"

# 重新编译
echo ">>> 重新编译 NumPy..."
cd "$NUMPY_SRC"
pip install -e . --no-build-isolation 2>&1 | tail -3

echo ">>> 验证修复..."
bash /task/tests/test.sh 2>&1 | tail -20
