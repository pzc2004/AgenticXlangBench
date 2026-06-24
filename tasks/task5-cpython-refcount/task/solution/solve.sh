#!/bin/bash
# Oracle: 修复 3 个复合 bug
set -e

# /workspace is mounted read-only, so copy files to /build/ first
mkdir -p /build/cpython-ext
cp /workspace/vector.c /workspace/setup.py /workspace/test_loop.py /build/cpython-ext/ 2>/dev/null || true
cp /workspace/*.py /workspace/*.c /workspace/*.h /build/cpython-ext/ 2>/dev/null || true
TARGET="/build/cpython-ext/vector.c"

echo "========================================="
echo " Oracle: 修复 CPython C 扩展 Refcount Bug"
echo "========================================="

# 检查文件是否存在
if [ ! -f "$TARGET" ]; then
    echo "❌ 找不到 $TARGET"
    exit 1
fi

echo ""
echo ">>> 修复 Bug 1: vector_push 中添加 Py_INCREF(item)"

# Bug 1: 恢复 Py_INCREF(item) in vector_push
# 从: /* Py_INCREF(item); */  /* BUG_LOCATION_1: INCREF removed - BUG */
# 到: Py_INCREF(item);  /* BUG_LOCATION_1: This INCREF will be removed by inject_bug.py */
sed -i 's|/\* Py_INCREF(item); \*/  /\* BUG_LOCATION_1: INCREF removed - BUG \*/|Py_INCREF(item);  /* BUG_LOCATION_1: This INCREF will be removed by inject_bug.py */|' "$TARGET"

if grep -q "Py_INCREF(item);  /\* BUG_LOCATION_1: This INCREF will be removed by inject_bug.py \*/" "$TARGET"; then
    echo "  ✅ Bug 1 修复成功"
else
    echo "  ❌ Bug 1 修复失败"
    exit 1
fi

echo ""
echo ">>> 修复 Bug 2: vector_get 中添加 Py_INCREF(result)"

# Bug 2: 恢复 Py_INCREF(result) in vector_get
# 从: /* Py_INCREF(result); */  /* BUG_LOCATION_2: INCREF removed - BUG */
# 到: Py_INCREF(result);  /* BUG_LOCATION_2: This INCREF will be removed by inject_bug.py */
sed -i 's|/\* Py_INCREF(result); \*/  /\* BUG_LOCATION_2: INCREF removed - BUG \*/|Py_INCREF(result);  /* BUG_LOCATION_2: This INCREF will be removed by inject_bug.py */|' "$TARGET"

if grep -q "BUG_LOCATION_2: This INCREF will be removed by inject_bug.py" "$TARGET"; then
    echo "  ✅ Bug 2 修复成功"
else
    echo "  ❌ Bug 2 修复失败"
    exit 1
fi

echo ""
echo ">>> 修复 Bug 3: vector_pop 中添加 Py_INCREF(result)"

# Bug 3: 恢复 Py_INCREF(result) in vector_pop
# 从: /* Py_INCREF(result); */  /* BUG_LOCATION_3: INCREF removed - BUG */
# 到: Py_INCREF(result);  /* BUG_LOCATION_3: This INCREF will be removed by inject_bug.py */
sed -i 's|/\* Py_INCREF(result); \*/  /\* BUG_LOCATION_3: INCREF removed - BUG \*/|Py_INCREF(result);  /* BUG_LOCATION_3: This INCREF will be removed by inject_bug.py */|' "$TARGET"

if grep -q "BUG_LOCATION_3: This INCREF will be removed by inject_bug.py" "$TARGET"; then
    echo "  ✅ Bug 3 修复成功"
else
    echo "  ❌ Bug 3 修复失败"
    exit 1
fi

echo ""
echo ">>> 重新编译扩展..."
cd /build/cpython-ext && pip install -e .

echo ""
echo ">>> 验证修复..."
cd /build/cpython-ext && python test_loop.py --iterations 10000 --seed 42

echo ""
echo "========================================="
echo "✅ 所有 bug 修复完成"
echo "========================================="
