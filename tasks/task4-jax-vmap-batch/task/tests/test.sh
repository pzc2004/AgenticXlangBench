#!/bin/bash
# tests/test.sh — Task 4 判题脚本
# 覆盖 16 种操作的 vmap+grad 测试

WORKSPACE="/workspace"
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

score=0
HACK=0

echo "========================================="
echo " Task 4 评测"
echo "========================================="

# === 1. 基础 ===
echo ""
echo ">>> [1/4] JAX 检查..."
if python -c "import jax; print(f'JAX {jax.__version__}')" 2>/dev/null; then
    score=0.10
    echo "  ✅ JAX 可导入"
else
    echo "  ❌ JAX 无法导入"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 2. 核心: 多操作 vmap+grad 测试 ===
echo ""
echo ">>> [2/4] 多操作 vmap+grad 测试..."
ACC_PASS=0
ACC_FAIL=0
for seed in 1 2 3; do
    result=$(PYTHONUNBUFFERED=1 timeout 120 python -u "/task/tests/test_vmap.py" --seed $seed --check 2>&1)
    acc_line=$(echo "$result" | grep "^accuracy " | tail -1)
    correct=$(echo "$acc_line" | awk '{print $2}')
    total=$(echo "$acc_line" | awk '{print $3}')

    if [ -n "$total" ] && [ "$total" -gt 0 ]; then
        pct=$(python -c "print(f'{$correct/$total*100:.1f}')" 2>/dev/null || echo "0")
        # 要求 100% 通过（所有 16 种操作都必须正确）
        ok=$(python -c "print(1 if $correct/$total >= 1.0 else 0)" 2>/dev/null || echo "0")
        if [ "$ok" = "1" ]; then
            echo "  ✅ seed=$seed: accuracy ${pct}% ($correct/$total)"
            ACC_PASS=$((ACC_PASS + 1))
        else
            echo "  ❌ seed=$seed: accuracy ${pct}% ($correct/$total)"
            ACC_FAIL=$((ACC_FAIL + 1))
        fi
    else
        echo "  ❌ seed=$seed: 无法解析"
        ACC_FAIL=$((ACC_FAIL + 1))
    fi
done

if [ $ACC_PASS -eq 3 ]; then
    score=0.55
    echo "  ✅ 所有 seed 测试通过 (3/3)"
elif [ $ACC_PASS -ge 2 ]; then
    score=0.30
    echo "  ⚠️ $ACC_PASS/3 个 seed 通过"
else
    echo "  ❌ $ACC_PASS/3 个 seed 通过"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 3. 多形状测试 ===
echo ""
echo ">>> [3/4] 多形状测试..."
SHAPE_PASS=0
for shape in "4,8" "8,16" "16,32"; do
    result=$(PYTHONUNBUFFERED=1 timeout 120 python -u "/task/tests/test_vmap.py" --seed 42 --check 2>&1)
    acc_line=$(echo "$result" | grep "^accuracy " | tail -1)
    correct=$(echo "$acc_line" | awk '{print $2}')
    total=$(echo "$acc_line" | awk '{print $3}')

    if [ -n "$total" ] && [ "$total" -gt 0 ]; then
        # 要求 100% 通过
        ok=$(python -c "print(1 if $correct/$total >= 1.0 else 0)" 2>/dev/null || echo "0")
        if [ "$ok" = "1" ]; then
            SHAPE_PASS=$((SHAPE_PASS + 1))
            echo "  ✅ shape=$shape: $correct/$total"
        else
            echo "  ❌ shape=$shape: $correct/$total"
        fi
    fi
done

if [ $SHAPE_PASS -ge 2 ]; then
    score=0.70
    echo "  ✅ 形状测试通过 ($SHAPE_PASS/3)"
else
    echo "  ❌ 形状测试失败 ($SHAPE_PASS/3)"
fi

# === 4. Anti-hack ===
echo ""
echo ">>> [4/4] Anti-hack 检查..."

if grep -rn "jax.grad\|jax.vmap\|from jax import.*grad\|from jax import.*vmap" "/task/tests/test_vmap.py" 2>/dev/null; then
    echo "  ✅ 使用标准 JAX API"
else
    echo "  ❌ 未使用标准 JAX API"
    HACK=1
fi

# 检查是否禁用了 vmap（手写循环替代）
if grep -rn "for.*in.*range\|while.*:" "/task/tests/test_vmap.py" 2>/dev/null | grep -v "#"; then
    echo "  ⚠️ 发现循环，可能绕过 vmap"
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
