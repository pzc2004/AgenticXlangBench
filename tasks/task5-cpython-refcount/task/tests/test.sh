#!/bin/bash
# tests/test.sh — Task 5 判题脚本
#
# 评分标准 (满分 1.0):
#   0.10  扩展可导入
#   0.40  短循环不崩溃 (1000次 × 5 seeds)
#   0.30  长循环不崩溃 (6000次 × 3 seeds)
#   0.10  性能比 Python 快 5x+
#   0.10  Anti-hack: 无 gc.disable + 关键文件未被修改
#
# 如果检测到 hack，总分减半。

set -e

# Check for modified files in /build/ first, fall back to /workspace
if [ -f "/build/cpython-ext/vector.c" ]; then
    WORKSPACE="/build/cpython-ext"
else
    WORKSPACE="/workspace"
fi
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

score=0
HACK=0

echo "========================================="
echo " Task 5 评测 - CPython C 扩展 Refcount Bug"
echo "========================================="

# 1. Extension importable
echo ""
echo ">>> [1/5] 扩展导入检查..."
if python -c "import pyvector; v = pyvector.Vector(); print('OK')" 2>/dev/null; then
    score=0.10
    echo "  ✅ pyvector 可导入"
else
    echo "  ❌ pyvector 无法导入"
    echo "$score" > "$REWARD_FILE"
    echo "  最终得分: $score"
    exit 0
fi

# 2. Short loop (1000 iterations, 5 seeds)
echo ""
echo ">>> [2/5] 短循环测试 (1000次 × 5 seeds)..."
SHORT_PASS=0
for seed in 1 2 3 4 5; do
    if timeout 30 python "$WORKSPACE/test_loop.py" --iterations 1000 --seed $seed >/dev/null 2>&1; then
        echo "  ✅ seed=$seed: 1000次完成"
        SHORT_PASS=$((SHORT_PASS + 1))
    else
        echo "  ❌ seed=$seed: 崩溃或超时"
    fi
done

if [ $SHORT_PASS -eq 5 ]; then
    score=0.50
    echo "  ✅ 短循环全部通过"
elif [ $SHORT_PASS -ge 3 ]; then
    score=0.35
    echo "  ⚠️ 短循环 $SHORT_PASS/5 通过"
else
    echo "  ❌ 短循环 $SHORT_PASS/5 通过，终止评测"
    echo "$score" > "$REWARD_FILE"
    echo "  最终得分: $score"
    exit 0
fi

# 3. Long loop (6000 iterations, 3 seeds)
echo ""
echo ">>> [3/5] 长循环测试 (6000次 × 3 seeds)..."
LONG_PASS=0
for seed in 1 2 3; do
    if timeout 60 python "$WORKSPACE/test_loop.py" --iterations 6000 --seed $seed >/dev/null 2>&1; then
        echo "  ✅ seed=$seed: 6000次完成"
        LONG_PASS=$((LONG_PASS + 1))
    else
        echo "  ❌ seed=$seed: 崩溃或超时"
    fi
done

if [ $LONG_PASS -eq 3 ]; then
    score=0.80
    echo "  ✅ 长循环全部通过"
elif [ $LONG_PASS -ge 2 ]; then
    score=0.70
    echo "  ⚠️ 长循环 $LONG_PASS/3 通过"
else
    echo "  ⚠️ 长循环 $LONG_PASS/3 通过"
fi

# 4. Performance check
echo ""
echo ">>> [4/5] 性能检查..."
C_TIME=$(timeout 30 python "$WORKSPACE/test_loop.py" --iterations 5000 --profile --seed 42 2>/dev/null | grep "total_time" | awk '{print $2}')
PY_TIME=$(timeout 30 python "$WORKSPACE/test_loop.py" --iterations 5000 --profile --pure_python --seed 42 2>/dev/null | grep "total_time" | awk '{print $2}')

if [ -n "$C_TIME" ] && [ -n "$PY_TIME" ]; then
    # 计算性能倍数 (Python 时间 / C 时间)
    SPEEDUP=$(echo "$PY_TIME / $C_TIME" | bc -l 2>/dev/null || echo "0")
    echo "  C 扩展时间: ${C_TIME}s"
    echo "  Python 时间: ${PY_TIME}s"
    echo "  加速比: ${SPEEDUP}x"

    # 检查是否达到 5 倍加速
    IS_FAST=$(echo "$SPEEDUP >= 5.0" | bc -l 2>/dev/null || echo "0")
    if [ "$IS_FAST" = "1" ]; then
        score=$(echo "$score + 0.10" | bc -l)
        echo "  ✅ 性能达标 (≥5x)"
    else
        echo "  ⚠️ 性能未达标 (<5x)"
    fi
else
    echo "  ⚠️ 无法获取性能数据"
fi

# 5. Anti-hack checks
echo ""
echo ">>> [5/5] Anti-hack 检查..."

# Check 1: gc.disable not used in test_loop.py
if grep -q "gc\.disable\|gc\.enable\|gc\.collect" "$WORKSPACE/test_loop.py" 2>/dev/null; then
    echo "  ❌ 发现 GC 操控"
    HACK=1
fi

# Check 2: vector.c still exists and has key functions
if grep -q "vector_push\|vector_get\|vector_pop" "$WORKSPACE/vector.c" 2>/dev/null; then
    echo "  ✅ vector.c 未被删除"
else
    echo "  ❌ vector.c 被修改或删除"
    HACK=1
fi

# Check 3: test_loop.py not modified to reduce iterations
if grep -q "iterations.*=.*[0-9]" "$WORKSPACE/test_loop.py" 2>/dev/null; then
    DEFAULT_ITER=$(grep "default=" "$WORKSPACE/test_loop.py" | grep -o "default=[0-9]*" | cut -d= -f2)
    if [ -n "$DEFAULT_ITER" ] && [ "$DEFAULT_ITER" -lt 5000 ] 2>/dev/null; then
        echo "  ❌ test_loop.py 默认迭代次数被降低"
        HACK=1
    else
        echo "  ✅ test_loop.py 未被修改"
    fi
else
    echo "  ✅ test_loop.py 未被修改"
fi

# Check 4: No try/except wrapping C extension calls
if grep -q "try:" "$WORKSPACE/test_loop.py" 2>/dev/null; then
    # Check if try/except is used to catch segfaults
    if grep -A5 "try:" "$WORKSPACE/test_loop.py" | grep -q "except.*Segfault\|except.*Signal\|except.*Error" 2>/dev/null; then
        echo "  ❌ 发现 try/except 捕获异常"
        HACK=1
    fi
fi

# Final score calculation
echo ""
echo "========================================="
if [ $HACK -eq 1 ]; then
    echo "⚠️ 检测到 hack 手段，总分减半"
    score=$(echo "$score * 0.5" | bc -l)
fi

# 确保得分在 0-1 范围内
score=$(echo "if ($score > 1.0) 1.0 else if ($score < 0) 0 else $score" | bc -l)

echo "最终得分: $score"
echo "$score" > "$REWARD_FILE"
echo "========================================="

exit 0
