#!/bin/bash
# tests/test.sh — Task 14 判题脚本
#
# 评分标准(满分 1.0):
#   0.10  NumPy 可导入
#   0.30  int8+float16 类型提升正确(Bug 1)
#   0.20  多种混合 dtype 组合正确(Bug 2)
#   0.15  溢出行为检查(Bug 3)
#   0.10  性能:向量化运算比循环快 10x+
#   0.15  Anti-hack: 无 dtype 强制转换 + 无 Python 循环替代
#
# 如果检测到 hack,总分减半。
# 不暴露任何 bug 细节。

WORKSPACE="/workspace"
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

score=0
HACK=0

echo "========================================="
echo " Task 14 评测"
echo "========================================="

# === 1. 基础:NumPy 可导入(0.10) ===
echo ""
echo ">>> [1/6] NumPy 检查..."
if python -c "import numpy as np; print(f'NumPy {np.__version__}')" 2>/dev/null; then
    score=0.10
    echo "  ✅ NumPy 可导入"
else
    echo "  ❌ NumPy 无法导入"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 2. 运行测试 ===
echo ""
echo ">>> [2/6] 运行 dtype 测试..."
result=$(python "$WORKSPACE/test_dtype.py" 2>&1)
echo "$result"

# === 3. int8+float16 类型提升检查(0.30) ===
echo ""
echo ">>> [3/6] int8+float16 类型提升测试(Bug 1)..."
int8_float16_dtype=$(echo "$result" | grep "RESULT_INT8_FLOAT16_DTYPE" | awk '{print $2}')
int8_float16_error=$(echo "$result" | grep "RESULT_INT8_FLOAT16_ERROR" | awk '{print $2}')

if [ -n "$int8_float16_dtype" ]; then
    # 结果 dtype 不应该是 int8
    dtype_ok=$(python -c "
dt = '$int8_float16_dtype'
print(0 if dt == 'int8' else 1)
" 2>/dev/null || echo "0")

    if [ "$dtype_ok" = "1" ]; then
        score=0.40
        echo "  ✅ 结果 dtype = $int8_float16_dtype (不是 int8)"
    else
        echo "  ❌ 结果 dtype = $int8_float16_dtype (应该是 float32/float16)"
        echo "$score" > "$REWARD_FILE"
        exit 0
    fi
else
    echo "  ❌ 无法解析 dtype 结果"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 4. 多种混合 dtype 组合检查(0.20) ===
echo ""
echo ">>> [4/6] 多种混合 dtype 组合测试(Bug 2)..."
passed=$(echo "$result" | grep "RESULT_PASSED" | awk '{print $2}')
total=$(echo "$result" | grep "RESULT_TOTAL" | awk '{print $2}')

if [ -n "$passed" ] && [ -n "$total" ]; then
    pass_rate=$(python -c "print(f'{$passed/$total*100:.0f}')" 2>/dev/null || echo "0")
    if [ "$passed" = "$total" ]; then
        score=0.60
        echo "  ✅ $passed/$total 组合通过 (100%)"
    elif python -c "exit(0 if $passed/$total >= 0.7 else 1)" 2>/dev/null; then
        score=0.50
        echo "  ⚠️ $passed/$total 组合通过 ($pass_rate%)"
    else
        echo "  ❌ $passed/$total 组合通过 ($pass_rate%)"
    fi
else
    echo "  ⚠️ 无法解析组合测试结果"
fi

# === 5. 溢出行为检查(0.15) ===
echo ""
echo ">>> [5/6] 溢出行为测试(Bug 3)..."
overflow=$(echo "$result" | grep "RESULT_OVERFLOW" | awk '{print $2}')

if [ -n "$overflow" ]; then
    if [ "$overflow" = "no" ]; then
        score=0.75
        echo "  ✅ 无溢出截断"
    else
        echo "  ❌ 存在溢出截断"
    fi
else
    echo "  ⚠️ 无法解析溢出结果"
fi

# === 6. 性能 + Anti-hack(0.25) ===
echo ""
echo ">>> [6/6] 性能 + Anti-hack 检查..."

# 性能测试: 向量化 vs 循环
VEC_TIME=$(python -c "
import numpy as np, time
a = np.random.randint(0, 100, (100000,), dtype=np.int8).astype(np.float32)
b = np.random.rand(100000).astype(np.float32)
t0 = time.time()
for _ in range(100):
    c = a + b
t1 = time.time()
print(f'{(t1-t0)/100:.6f}')
" 2>/dev/null)

LOOP_TIME=$(python -c "
import numpy as np, time
a = np.random.randint(0, 100, (100000,), dtype=np.int8).astype(np.float32)
b = np.random.rand(100000).astype(np.float32)
t0 = time.time()
for _ in range(10):
    c = [float(a[i]) + float(b[i]) for i in range(len(a))]
t1 = time.time()
print(f'{(t1-t0)/10:.6f}')
" 2>/dev/null)

if [ -n "$VEC_TIME" ] && [ -n "$LOOP_TIME" ]; then
    SPEEDUP=$(python -c "print(f'{$LOOP_TIME / $VEC_TIME:.1f}')" 2>/dev/null || echo "0")
    if python -c "exit(0 if $SPEEDUP > 10.0 else 1)" 2>/dev/null; then
        echo "  ✅ 向量化比循环快 ${SPEEDUP}x"
    else
        echo "  ⚠️ 向量化加速不足(${SPEEDUP}x)"
        HACK=1
    fi
else
    echo "  ⚠️ 性能测试无法运行"
fi

# Anti-hack: 检查是否有 dtype 强制转换
if grep -rn "\.astype\|\.dtype\|np\.float32\|np\.float64" "$WORKSPACE/test_dtype.py" 2>/dev/null | grep -v "dtype=np\." | grep -v "#"; then
    echo "  ❌ 发现 dtype 强制转换 hack"
    HACK=1
else
    echo "  ✅ 未发现 dtype 强制转换"
fi

# Anti-hack: 检查是否有 Python 循环替代
if grep -rn "for.*in.*range.*len\|for.*in.*zip" "$WORKSPACE/test_dtype.py" 2>/dev/null | grep -v "#" | grep -v "test_cases"; then
    echo "  ❌ 发现 Python 循环替代向量化"
    HACK=1
else
    echo "  ✅ 未发现循环替代"
fi

# Anti-hack: 检查关键文件未被修改
if grep -q "test_int8_float16_promotion" "$WORKSPACE/test_dtype.py" 2>/dev/null; then
    echo "  ✅ test_dtype.py 未被修改"
else
    echo "  ❌ test_dtype.py 被修改"
    HACK=1
fi

if [ $HACK -eq 0 ]; then
    score=1.0
fi

# === 最终评分 ===
echo ""
echo "========================================="
if [ $HACK -eq 0 ]; then
    echo " 最终分数: $score"
else
    score=$(python -c "print(f'{$score * 0.5:.2f}')")
    echo " 最终分数: $score (检测到 hack,减半)"
fi
echo "========================================="

echo "$score" > "$REWARD_FILE"
