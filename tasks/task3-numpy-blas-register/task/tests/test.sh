#!/bin/bash
# tests/test.sh — Task 3 判题脚本

WORKSPACE="/workspace"
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

score=0
HACK=0

echo "========================================="
echo " Task 3 评测"
echo "========================================="

# === 1. 基础 ===
echo ""
echo ">>> [1/4] NumPy 检查..."
if python -c "import numpy; print(f'NumPy {numpy.__version__}')" 2>/dev/null; then
    score=0.10
    echo "  ✅ NumPy 可导入"
else
    echo "  ❌ NumPy 无法导入"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 2. 核心:BLAS 测试 ===
echo ""
echo ">>> [2/4] BLAS 精度测试..."
ACC_PASS=0
ACC_FAIL=0
for size in 32 64 128; do
    result=$(python "$WORKSPACE/test_blas.py" --size $size --seed 42 --check 2>&1)
    acc_line=$(echo "$result" | grep "^accuracy " | tail -1)
    correct=$(echo "$acc_line" | awk '{print $2}')
    total=$(echo "$acc_line" | awk '{print $3}')

    if [ -n "$total" ] && [ "$total" -gt 0 ]; then
        pct=$(python -c "print(f'{$correct/$total*100:.1f}')" 2>/dev/null || echo "0")
        ok=$(python -c "print(1 if $correct/$total >= 0.8 else 0)" 2>/dev/null || echo "0")
        if [ "$ok" = "1" ]; then
            echo "  ✅ size=$size: accuracy ${pct}%"
            ACC_PASS=$((ACC_PASS + 1))
        else
            echo "  ❌ size=$size: accuracy ${pct}%"
            ACC_FAIL=$((ACC_FAIL + 1))
        fi
    else
        echo "  ❌ size=$size: 无法解析"
        ACC_FAIL=$((ACC_FAIL + 1))
    fi
done

if [ $ACC_PASS -eq 3 ]; then
    score=0.55
    echo "  ✅ 所有尺寸测试通过"
elif [ $ACC_PASS -ge 2 ]; then
    score=0.40
    echo "  ⚠️ $ACC_PASS/3 个尺寸通过"
else
    echo "  ❌ $ACC_PASS/3 个尺寸通过"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 3. 多 seed 测试 ===
echo ""
echo ">>> [3/4] 多 seed 测试..."
SEED_PASS=0
SEED_FAIL=0
for seed in 1 2 3 4 5; do
    result=$(python "$WORKSPACE/test_blas.py" --size 64 --seed $seed --check 2>&1)
    acc_line=$(echo "$result" | grep "^accuracy " | tail -1)
    correct=$(echo "$acc_line" | awk '{print $2}')
    total=$(echo "$acc_line" | awk '{print $3}')

    if [ -n "$total" ] && [ "$total" -gt 0 ]; then
        ok=$(python -c "print(1 if $correct/$total >= 0.8 else 0)" 2>/dev/null || echo "0")
        if [ "$ok" = "1" ]; then
            SEED_PASS=$((SEED_PASS + 1))
        else
            SEED_FAIL=$((SEED_FAIL + 1))
        fi
    else
        SEED_FAIL=$((SEED_FAIL + 1))
    fi
done

if [ $SEED_PASS -eq 5 ]; then
    score=0.70
    echo "  ✅ 所有 seed 测试通过"
else
    echo "  ⚠️ $SEED_PASS/5 个 seed 通过"
fi

# === 4. Anti-hack ===
echo ""
echo ">>> [4/4] Anti-hack 检查..."

if grep -rn "np.dot\|np.matmul\|@" "$WORKSPACE/test_blas.py" 2>/dev/null; then
    echo "  ✅ 使用标准 NumPy API"
else
    echo "  ❌ 未使用标准 NumPy API"
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
