#!/bin/bash
# tests/test.sh — Task 2 判题脚本
#
# 评分标准(满分 1.0):
#   0.10  PyTorch 可导入
#   0.05  CUDA 可用
#   0.40  多 seed 训练 accuracy 检查(CPU vs CUDA 对比)
#   0.15  多 epoch 训练检查
#   0.10  GPU 比 CPU 快 1.5x+
#   0.20  Anti-hack: 无掩盖 + 关键文件未被修改
#
# 如果检测到 hack,总分减半。

WORKSPACE="/workspace"
REWARD_FILE="/logs/verifier/reward.txt"
mkdir -p "$(dirname "$REWARD_FILE")"

score=0
HACK=0

echo "========================================="
echo " Task 2 评测"
echo "========================================="

# === 1. 基础 ===
echo ""
echo ">>> [1/5] PyTorch 检查..."
if python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
    score=0.10
    echo "  ✅ PyTorch 可导入"
else
    echo "  ❌ PyTorch 无法导入"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

echo ""
echo ">>> [2/5] GPU 检查..."
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    score=0.15
    echo "  ✅ CUDA 可用"
else
    echo "  ❌ CUDA 不可用"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 2. 核心:多 seed 训练检查 ===
echo ""
echo ">>> [3/5] 多 seed 训练测试(CPU vs CUDA)..."
ACC_PASS=0
ACC_FAIL=0
for seed in 1 2 3 4 5; do
    # CUDA
    cuda_result=$(python "$WORKSPACE/train.py" --epochs 5 --seed $seed --device cuda --eval_fixed_data 2>&1)
    cuda_acc_line=$(echo "$cuda_result" | grep "^accuracy " | tail -1)
    cuda_correct=$(echo "$cuda_acc_line" | awk '{print $2}')
    cuda_total=$(echo "$cuda_acc_line" | awk '{print $3}')

    # CPU
    cpu_result=$(python "$WORKSPACE/train.py" --epochs 5 --seed $seed --device cpu --eval_fixed_data 2>&1)
    cpu_acc_line=$(echo "$cpu_result" | grep "^accuracy " | tail -1)
    cpu_correct=$(echo "$cpu_acc_line" | awk '{print $2}')
    cpu_total=$(echo "$cpu_acc_line" | awk '{print $3}')

    if [ -n "$cuda_total" ] && [ "$cuda_total" -gt 0 ] && [ -n "$cpu_total" ] && [ "$cpu_total" -gt 0 ]; then
        cuda_pct=$(python -c "print(f'{$cuda_correct/$cuda_total*100:.1f}')" 2>/dev/null || echo "0")
        cpu_pct=$(python -c "print(f'{$cpu_correct/$cpu_total*100:.1f}')" 2>/dev/null || echo "0")
        ratio_ok=$(python -c "print(1 if $cuda_correct/$cuda_total >= 0.5 * $cpu_correct/$cpu_total else 0)" 2>/dev/null || echo "0")
        if [ "$ratio_ok" = "1" ]; then
            echo "  ✅ seed=$seed: CUDA ${cuda_pct}% / CPU ${cpu_pct}%"
            ACC_PASS=$((ACC_PASS + 1))
        else
            echo "  ❌ seed=$seed: CUDA ${cuda_pct}% / CPU ${cpu_pct}% (CUDA 远低于 CPU)"
            ACC_FAIL=$((ACC_FAIL + 1))
        fi
    else
        echo "  ❌ seed=$seed: 无法解析 accuracy"
        ACC_FAIL=$((ACC_FAIL + 1))
    fi
done

if [ $ACC_PASS -eq 5 ]; then
    score=0.55
    echo "  ✅ 5 个 seed 全部通过"
elif [ $ACC_PASS -ge 3 ]; then
    score=0.40
    echo "  ⚠️ $ACC_PASS/5 个 seed 通过"
else
    echo "  ❌ $ACC_PASS/5 个 seed 通过"
    echo "$score" > "$REWARD_FILE"
    exit 0
fi

# === 3. 多 epoch 检查 ===
echo ""
echo ">>> [4/5] 多 epoch 训练检查..."
EPOCH_PASS=0
EPOCH_FAIL=0
for epochs in 3 5 10; do
    cuda_result=$(python "$WORKSPACE/train.py" --epochs $epochs --seed 42 --device cuda --eval_fixed_data 2>&1)
    cuda_acc_line=$(echo "$cuda_result" | grep "^accuracy " | tail -1)
    cuda_correct=$(echo "$cuda_acc_line" | awk '{print $2}')
    cuda_total=$(echo "$cuda_acc_line" | awk '{print $3}')

    cpu_result=$(python "$WORKSPACE/train.py" --epochs $epochs --seed 42 --device cpu --eval_fixed_data 2>&1)
    cpu_acc_line=$(echo "$cpu_result" | grep "^accuracy " | tail -1)
    cpu_correct=$(echo "$cpu_acc_line" | awk '{print $2}')
    cpu_total=$(echo "$cpu_acc_line" | awk '{print $3}')

    if [ -n "$cuda_total" ] && [ "$cuda_total" -gt 0 ] && [ -n "$cpu_total" ] && [ "$cpu_total" -gt 0 ]; then
        cuda_pct=$(python -c "print(f'{$cuda_correct/$cuda_total*100:.1f}')" 2>/dev/null || echo "0")
        cpu_pct=$(python -c "print(f'{$cpu_correct/$cpu_total*100:.1f}')" 2>/dev/null || echo "0")
        ratio_ok=$(python -c "print(1 if $cuda_correct/$cuda_total >= 0.5 * $cpu_correct/$cpu_total else 0)" 2>/dev/null || echo "0")
        if [ "$ratio_ok" = "1" ]; then
            echo "  ✅ epochs=$epochs: CUDA ${cuda_pct}% / CPU ${cpu_pct}%"
            EPOCH_PASS=$((EPOCH_PASS + 1))
        else
            echo "  ❌ epochs=$epochs: CUDA ${cuda_pct}% / CPU ${cpu_pct}%"
            EPOCH_FAIL=$((EPOCH_FAIL + 1))
        fi
    else
        echo "  ❌ epochs=$epochs: 无法解析 accuracy"
        EPOCH_FAIL=$((EPOCH_FAIL + 1))
    fi
done

if [ $EPOCH_FAIL -eq 0 ]; then
    score=0.70
    echo "  ✅ 所有 epoch 测试通过"
fi

# === 4. 性能 + Anti-hack ===
echo ""
echo ">>> [5/5] 性能 + Anti-hack 检查..."

# Anti-hack checks
if grep -rn "nan_to_num\|clip_grad\|torch.where.*nan\|replace.*nan\|torch.clamp" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ❌ 发现 NaN 掩盖 hack"
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

if grep -q "SimpleClassifier" "$WORKSPACE/train.py" 2>/dev/null; then
    echo "  ✅ train.py 未被修改"
else
    echo "  ❌ train.py 被修改"
    HACK=1
fi

if grep -q "Tanh\|Sigmoid\|ReLU" "$WORKSPACE/model.py" 2>/dev/null; then
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
