#!/bin/bash
# tests/test.sh — Task 1 判题脚本
#
# 评分标准(满分 1.0):
#   0.10  PyTorch 可导入
#   0.05  CUDA 可用
#   0.30  小 batch 训练 accuracy 检查(Bug 1+2)
#   0.20  大 batch 训练无 NaN 检查(Bug 3)
#   0.15  多 batch_size 训练检查
#   0.10  GPU 比 CPU 快 1.5x+
#   0.10  Anti-hack: 无 NaN 掩盖 + 无 CPU 回退 + 关键文件未被修改
#
# 如果检测到 hack,总分减半。
# 不暴露任何 bug 细节。

WORKSPACE="/workspace"
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

score=0
HACK=0

echo "========================================="
echo " Task 1 评测"
echo "========================================="

# === 1. 基础:PyTorch 可导入(0.10) ===
echo ""
echo ">>> [1/6] PyTorch 检查..."
if python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    score=0.10
    echo "  ✅ PyTorch 可导入"
else
    echo "  ❌ PyTorch 无法导入"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 2. 基础:GPU 可用(0.05) ===
echo ""
echo ">>> [2/6] GPU 检查..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    score=0.15
    echo "  ✅ CUDA 可用"
else
    echo "  ❌ CUDA 不可用"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 3. 小 batch 训练 accuracy 检查(0.30) ===
echo ""
echo ">>> [3/6] 小 batch 训练测试(Bug 1+2)..."
SMALL_PASS=0
SMALL_FAIL=0
for seed in 1 2 3 4 5; do
    result=$(python "$WORKSPACE/train.py" --steps 50 --seed $seed --batch_size 16 --device cuda --eval_fixed_data 2>&1)
    if echo "$result" | grep -q "nan_detected True"; then
        echo "  ⚠️ seed=$seed: NaN detected"
        SMALL_FAIL=$((SMALL_FAIL + 1))
        continue
    fi
    acc_line=$(echo "$result" | grep "^accuracy " | tail -1)
    correct=$(echo "$acc_line" | awk '{print $2}')
    total=$(echo "$acc_line" | awk '{print $3}')
    if [ -n "$total" ] && [ "$total" -gt 0 ]; then
        pct=$(python -c "print(f'{$correct/$total*100:.1f}')" 2>/dev/null || echo "0")
        ok=$(python -c "print(1 if $correct/$total >= 0.30 else 0)" 2>/dev/null || echo "0")
        if [ "$ok" = "1" ]; then
            echo "  ✅ seed=$seed: accuracy ${pct}%"
            SMALL_PASS=$((SMALL_PASS + 1))
        else
            echo "  ❌ seed=$seed: accuracy ${pct}% (< 30%)"
            SMALL_FAIL=$((SMALL_FAIL + 1))
        fi
    else
        echo "  ❌ seed=$seed: 无法解析 accuracy"
        SMALL_FAIL=$((SMALL_FAIL + 1))
    fi
done

if [ $SMALL_PASS -eq 5 ]; then
    score=0.45
    echo "  ✅ 5 个 seed 全部通过"
elif [ $SMALL_PASS -ge 3 ]; then
    score=0.35
    echo "  ⚠️ $SMALL_PASS/5 个 seed 通过"
else
    echo "  ❌ $SMALL_PASS/5 个 seed 通过"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 4. 大 batch 训练无 NaN 检查(0.20) ===
echo ""
echo ">>> [4/6] 大 batch 训练测试(Bug 3)..."
LARGE_PASS=0
LARGE_FAIL=0
for bs in 64 128; do
    result=$(python "$WORKSPACE/train.py" --steps 30 --seed 42 --batch_size $bs --device cuda --eval_fixed_data 2>&1)
    if echo "$result" | grep -q "nan_detected True"; then
        echo "  ❌ batch_size=$bs: NaN detected"
        LARGE_FAIL=$((LARGE_FAIL + 1))
    else
        acc_line=$(echo "$result" | grep "^accuracy " | tail -1)
        correct=$(echo "$acc_line" | awk '{print $2}')
        total=$(echo "$acc_line" | awk '{print $3}')
        if [ -n "$total" ] && [ "$total" -gt 0 ]; then
            pct=$(python -c "print(f'{$correct/$total*100:.1f}')" 2>/dev/null || echo "0")
            ok=$(python -c "print(1 if $correct/$total >= 0.30 else 0)" 2>/dev/null || echo "0")
            if [ "$ok" = "1" ]; then
                echo "  ✅ batch_size=$bs: accuracy ${pct}%"
                LARGE_PASS=$((LARGE_PASS + 1))
            else
                echo "  ❌ batch_size=$bs: accuracy ${pct}% (< 30%)"
                LARGE_FAIL=$((LARGE_FAIL + 1))
            fi
        else
            echo "  ❌ batch_size=$bs: 无法解析 accuracy"
            LARGE_FAIL=$((LARGE_FAIL + 1))
        fi
    fi
done

if [ $LARGE_FAIL -eq 0 ]; then
    score=0.65
    echo "  ✅ 大 batch 测试全部通过"
else
    echo "  ⚠️ $LARGE_FAIL/2 个大 batch 测试失败"
fi

# === 5. 多 batch_size 测试(0.15) ===
echo ""
echo ">>> [5/6] 多 batch_size 测试..."
BATCH_PASS=0
BATCH_FAIL=0
for bs in 16 32 64 128; do
    result=$(python "$WORKSPACE/train.py" --steps 30 --seed 42 --batch_size $bs --device cuda --eval_fixed_data 2>&1)
    if echo "$result" | grep -q "nan_detected True"; then
        echo "  ❌ batch_size=$bs: NaN detected"
        BATCH_FAIL=$((BATCH_FAIL + 1))
        continue
    fi
    acc_line=$(echo "$result" | grep "^accuracy " | tail -1)
    correct=$(echo "$acc_line" | awk '{print $2}')
    total=$(echo "$acc_line" | awk '{print $3}')
    if [ -n "$total" ] && [ "$total" -gt 0 ]; then
        pct=$(python -c "print(f'{$correct/$total*100:.1f}')" 2>/dev/null || echo "0")
        ok=$(python -c "print(1 if $correct/$total >= 0.30 else 0)" 2>/dev/null || echo "0")
        if [ "$ok" = "1" ]; then
            echo "  ✅ batch_size=$bs: accuracy ${pct}%"
            BATCH_PASS=$((BATCH_PASS + 1))
        else
            echo "  ❌ batch_size=$bs: accuracy ${pct}% (< 30%)"
            BATCH_FAIL=$((BATCH_FAIL + 1))
        fi
    else
        echo "  ❌ batch_size=$bs: 无法解析 accuracy"
        BATCH_FAIL=$((BATCH_FAIL + 1))
    fi
done

if [ $BATCH_FAIL -eq 0 ]; then
    score=0.80
    echo "  ✅ 4 种 batch_size 全部通过"
else
    echo "  ⚠️ $BATCH_FAIL/4 种 batch_size 失败"
fi

# === 6. 性能 + Anti-hack(0.20) ===
echo ""
echo ">>> [6/6] 性能 + Anti-hack 检查..."

GPU_TIME=$(python "$WORKSPACE/train.py" --steps 10 --seed 42 --device cuda --profile 2>&1 | grep "avg_step_time" | awk '{print $NF}')
CPU_TIME=$(python "$WORKSPACE/train.py" --steps 10 --seed 42 --device cpu --profile 2>&1 | grep "avg_step_time" | awk '{print $NF}')

if [ -n "$GPU_TIME" ] && [ -n "$CPU_TIME" ]; then
    SPEEDUP=$(python -c "print(f'{$CPU_TIME / $GPU_TIME:.1f}')" 2>/dev/null || echo "0")
    if python -c "exit(0 if $SPEEDUP > 1.5 else 1)" 2>/dev/null; then
        echo "  ✅ GPU 比 CPU 快 ${SPEEDUP}x"
    else
        echo "  ⚠️ GPU 加速不足(${SPEEDUP}x)"
        HACK=1
    fi
else
    echo "  ⚠️ 性能测试无法运行"
fi

if grep -rn "nan_to_num\|clip_grad\|torch.where.*nan\|replace.*nan\|torch.clamp" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ❌ 发现 NaN 掩盖 / 梯度操控 hack"
    HACK=1
else
    echo "  ✅ 未发现 NaN 掩盖"
fi

if grep -rn "device.*=.*'cpu'\|\.cpu()" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ❌ 发现 CPU 回退"
    HACK=1
else
    echo "  ✅ 未发现 CPU 回退"
fi

if grep -q "ImageClassifier" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ✅ train.py 未被修改"
else
    echo "  ❌ train.py 被修改"
    HACK=1
fi

if grep -q "FeatureNorm\|F.layer_norm" "$WORKSPACE/model.py" 2>/dev/null && \
   grep -q "BatchNorm2d\|GroupNorm" "$WORKSPACE/model.py" 2>/dev/null; then
    echo "  ✅ model.py 未被修改"
else
    echo "  ❌ model.py 被修改"
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
